"""
agent.py — LangGraph RAG Agent for Knowledge Transfer (KT).

LLM:        Groq  (FREE tier — llama3-8b-8192, no credit card)
Embeddings: HuggingFace sentence-transformers (FREE, local CPU)
Vector DB:  ChromaDB (FREE, local)

Graph flow:
  user question
       |
  [retrieve]     -> fetch top-k chunks from ChromaDB
       |
  [grade_docs]   -> filter irrelevant chunks (LLM grader)
       |
  [generate] or [rewrite_query]  <- if no relevant docs, rewrite & retry (max 2x)
       |
  [hallucination_check]  -> verify answer is grounded
       |
     answer
"""

from __future__ import annotations

import json
import os
import logging
from typing import Annotated, List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

from ingest import get_vectorstore

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Model config ───────────────────────────────────────────────────────────────
# Groq free models: llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768, gemma2-9b-it
LLM_MODEL   = os.getenv("LLM_MODEL", "llama3-8b-8192")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
TOP_K       = int(os.getenv("RETRIEVER_TOP_K", "6"))


def _get_llm(temperature: float = TEMPERATURE) -> ChatGroq:
    return ChatGroq(model=LLM_MODEL, temperature=temperature)


# ── State schema ───────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages:      Annotated[List[BaseMessage], add_messages]
    question:      str
    documents:     List[Document]
    generation:    str
    rewrite_count: int


# ── Prompts ────────────────────────────────────────────────────────────────────
_GENERATE_SYSTEM = """You are a helpful Knowledge Transfer (KT) assistant.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information, clearly say so.
Always cite the source document name(s) when available.

Context:
{context}
"""

_GRADE_SYSTEM = """You are a relevance grader.
Given a document chunk and a user question, decide if the chunk is relevant.
Reply with ONLY a JSON object on one line: {{"score": "yes"}} or {{"score": "no"}}
Be lenient — partial relevance counts as "yes".
"""

_HALLUCINATION_SYSTEM = """You are a grounding checker.
Given an answer and supporting context, decide if the answer is fully grounded.
Reply with ONLY a JSON object on one line: {{"score": "yes"}} if grounded, {{"score": "no"}} if hallucinated.
"""

_REWRITE_SYSTEM = """You are a query optimizer.
Rewrite the user's question to improve vector-store retrieval.
Make it more specific. Return ONLY the rewritten question, nothing else.
"""

_generate_prompt      = ChatPromptTemplate.from_messages([("system", _GENERATE_SYSTEM), ("human", "{question}")])
_grade_prompt         = ChatPromptTemplate.from_messages([("system", _GRADE_SYSTEM), ("human", "Document:\n{document}\n\nQuestion: {question}")])
_hallucination_prompt = ChatPromptTemplate.from_messages([("system", _HALLUCINATION_SYSTEM), ("human", "Answer:\n{generation}\n\nContext:\n{context}")])
_rewrite_prompt       = ChatPromptTemplate.from_messages([("system", _REWRITE_SYSTEM), ("human", "{question}")])


# ── Helpers ────────────────────────────────────────────────────────────────────
def _docs_to_context(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "?")
        parts.append(f"[{i}] (source: {source}, page: {page})\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _parse_json_score(content: str, default: str = "yes") -> str:
    """Safely parse a yes/no score from LLM JSON output."""
    try:
        return json.loads(content.strip()).get("score", default)
    except Exception:
        return "yes" if "yes" in content.lower() else "no"


# ── Graph nodes ────────────────────────────────────────────────────────────────
def retrieve(state: AgentState) -> AgentState:
    logger.info("NODE: retrieve | question=%s", state["question"][:80])
    vectorstore = get_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    docs        = retriever.invoke(state["question"])
    logger.info("  -> %d chunks retrieved", len(docs))
    return {**state, "documents": docs}


def grade_documents(state: AgentState) -> AgentState:
    logger.info("NODE: grade_documents")
    chain    = _grade_prompt | _get_llm()
    relevant = []
    for doc in state["documents"]:
        result = chain.invoke({"document": doc.page_content, "question": state["question"]})
        score  = _parse_json_score(result.content, default="no")
        if score == "yes":
            relevant.append(doc)
    logger.info("  -> %d/%d chunks kept", len(relevant), len(state["documents"]))
    return {**state, "documents": relevant}


def generate(state: AgentState) -> AgentState:
    logger.info("NODE: generate")
    context = _docs_to_context(state["documents"])
    chain   = _generate_prompt | _get_llm()
    result  = chain.invoke({"context": context, "question": state["question"]})
    logger.info("  -> answer generated (%d chars)", len(result.content))
    return {
        **state,
        "generation": result.content,
        "messages":   state["messages"] + [AIMessage(content=result.content)],
    }


def rewrite_query(state: AgentState) -> AgentState:
    count = state.get("rewrite_count", 0) + 1
    logger.info("NODE: rewrite_query (attempt %d)", count)
    chain    = _rewrite_prompt | _get_llm()
    result   = chain.invoke({"question": state["question"]})
    new_q    = result.content.strip()
    logger.info("  -> rewritten: %s", new_q[:80])
    return {**state, "question": new_q, "rewrite_count": count}


# ── Edge conditions ────────────────────────────────────────────────────────────
def decide_after_grading(state: AgentState) -> Literal["generate", "rewrite_query"]:
    if state["documents"]:
        return "generate"
    if state.get("rewrite_count", 0) >= 2:
        logger.info("Max rewrites reached; generating with empty context")
        return "generate"
    return "rewrite_query"


def decide_after_hallucination_check(state: AgentState) -> Literal["end", "generate"]:
    logger.info("NODE: hallucination_check")
    context = _docs_to_context(state["documents"])
    chain   = _hallucination_prompt | _get_llm()
    result  = chain.invoke({"generation": state["generation"], "context": context})
    score   = _parse_json_score(result.content, default="yes")
    if score == "yes":
        logger.info("  -> answer grounded")
        return "end"
    logger.info("  -> hallucination detected, regenerating")
    return "generate"


# ── Build graph ────────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("retrieve",       retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("generate",        generate)
    g.add_node("rewrite_query",   rewrite_query)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges("grade_documents", decide_after_grading,
                            {"generate": "generate", "rewrite_query": "rewrite_query"})
    g.add_edge("rewrite_query", "retrieve")
    g.add_conditional_edges("generate", decide_after_hallucination_check,
                            {"end": END, "generate": "generate"})
    return g.compile()


# Compile once at import time
_graph = build_graph()


# ── Public API ─────────────────────────────────────────────────────────────────
def ask(question: str, chat_history: List[BaseMessage] | None = None) -> dict:
    """
    Run the KT RAG agent.
    Returns: {"answer": str, "sources": list[str], "question": str}
    """
    history = chat_history or []
    initial_state: AgentState = {
        "messages":      history + [HumanMessage(content=question)],
        "question":      question,
        "documents":     [],
        "generation":    "",
        "rewrite_count": 0,
    }
    final = _graph.invoke(initial_state)
    sources = list({doc.metadata.get("source", "unknown") for doc in final.get("documents", [])})
    return {
        "answer":   final.get("generation", ""),
        "sources":  sources,
        "question": final.get("question", question),
    }


if __name__ == "__main__":
    q = input("Ask a question: ").strip()
    r = ask(q)
    print("\nAnswer:\n", r["answer"])
    print("\nSources:", r["sources"])
