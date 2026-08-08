"""
agent.py — LangGraph RAG Agent for Knowledge Transfer (KT).

Graph flow:
  user question
       │
  [retrieve]  ──► fetch top-k chunks from ChromaDB
       │
  [grade_docs] ─► filter out irrelevant chunks
       │
  ┌────┴────┐
  │         │
[generate] [rewrite_query]  ◄── if no relevant docs found, rewrite and re-retrieve
  │
[check_hallucination]  ──► verify answer is grounded in context
  │
  ▼
 answer
"""

from __future__ import annotations

import os
import logging
from typing import Annotated, List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

from ingest import get_vectorstore

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Model config ───────────────────────────────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
TOP_K = int(os.getenv("RETRIEVER_TOP_K", "6"))

# ── Shared LLM instances ───────────────────────────────────────────────────────
_llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)
_llm_json = ChatOpenAI(model=LLM_MODEL, temperature=0)  # used for structured grading


# ── State schema ───────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    """Shared state passed between graph nodes."""
    messages: Annotated[List[BaseMessage], add_messages]  # full conversation history
    question: str                                          # current user question
    documents: List[Document]                             # retrieved / filtered chunks
    generation: str                                       # LLM answer
    rewrite_count: int                                    # guard against infinite rewrite loops


# ── Prompts ────────────────────────────────────────────────────────────────────
_GENERATE_SYSTEM = """You are a helpful Knowledge Transfer (KT) assistant.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information, clearly say so — do not make things up.
Always cite the source document name(s) in your answer when available.

Context:
{context}
"""

_GRADE_SYSTEM = """You are a relevance grader.
Given a retrieved document chunk and a user question, decide if the chunk is relevant.
Reply with a single JSON object: {{"score": "yes"}} or {{"score": "no"}}.
Be lenient — if the chunk is even partially related, score it "yes".
"""

_HALLUCINATION_SYSTEM = """You are a grounding checker.
Given an LLM-generated answer and the supporting context chunks, decide if the answer
is fully grounded in the provided context (no hallucinations).
Reply with a single JSON object: {{"score": "yes"}} if grounded, {{"score": "no"}} otherwise.
"""

_REWRITE_SYSTEM = """You are a query optimizer.
Rewrite the user's question to improve vector-store retrieval.
Make it more specific, using domain terminology from knowledge transfer / software documentation.
Return ONLY the rewritten question, nothing else.
"""

_generate_prompt = ChatPromptTemplate.from_messages([
    ("system", _GENERATE_SYSTEM),
    ("human", "{question}"),
])

_grade_prompt = ChatPromptTemplate.from_messages([
    ("system", _GRADE_SYSTEM),
    ("human", "Document:\n{document}\n\nQuestion: {question}"),
])

_hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", _HALLUCINATION_SYSTEM),
    ("human", "Answer:\n{generation}\n\nContext:\n{context}"),
])

_rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", _REWRITE_SYSTEM),
    ("human", "{question}"),
])


# ── Helper ─────────────────────────────────────────────────────────────────────
def _docs_to_context(docs: List[Document]) -> str:
    """Format a list of Document chunks into a single context string."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[{i}] (source: {source}, page: {page})\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ── Node functions ─────────────────────────────────────────────────────────────
def retrieve(state: AgentState) -> AgentState:
    """Retrieve top-k relevant chunks from ChromaDB."""
    logger.info("NODE: retrieve | question=%s", state["question"][:80])
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(state["question"])
    logger.info("  → %d chunks retrieved", len(docs))
    return {**state, "documents": docs}


def grade_documents(state: AgentState) -> AgentState:
    """Filter retrieved chunks to keep only relevant ones."""
    logger.info("NODE: grade_documents")
    question = state["question"]
    docs = state["documents"]
    relevant: List[Document] = []

    chain = _grade_prompt | _llm_json

    for doc in docs:
        result = chain.invoke({"document": doc.page_content, "question": question})
        try:
            import json
            score_obj = json.loads(result.content)
            score = score_obj.get("score", "no")
        except Exception:
            # Fallback: check raw content
            score = "yes" if "yes" in result.content.lower() else "no"

        if score == "yes":
            relevant.append(doc)

    logger.info("  → %d/%d chunks kept after grading", len(relevant), len(docs))
    return {**state, "documents": relevant}


def generate(state: AgentState) -> AgentState:
    """Generate an answer grounded in the retrieved context."""
    logger.info("NODE: generate")
    context = _docs_to_context(state["documents"])
    chain = _generate_prompt | _llm
    result = chain.invoke({"context": context, "question": state["question"]})
    generation = result.content
    logger.info("  → answer generated (%d chars)", len(generation))
    return {
        **state,
        "generation": generation,
        "messages": state["messages"] + [AIMessage(content=generation)],
    }


def rewrite_query(state: AgentState) -> AgentState:
    """Rewrite the question to improve retrieval on next attempt."""
    logger.info("NODE: rewrite_query (attempt %d)", state.get("rewrite_count", 0) + 1)
    chain = _rewrite_prompt | _llm
    result = chain.invoke({"question": state["question"]})
    new_question = result.content.strip()
    logger.info("  → rewritten: %s", new_question[:80])
    return {
        **state,
        "question": new_question,
        "rewrite_count": state.get("rewrite_count", 0) + 1,
    }


# ── Edge condition functions ───────────────────────────────────────────────────
def decide_after_grading(state: AgentState) -> Literal["generate", "rewrite_query"]:
    """
    After grading, decide whether to generate or rewrite.
    If no relevant docs remain and we haven't retried too many times → rewrite.
    """
    if state["documents"]:
        return "generate"
    if state.get("rewrite_count", 0) >= 2:
        # Prevent infinite loops — generate a "not found" style answer
        logger.info("Max rewrites reached; forcing generate with empty context")
        return "generate"
    return "rewrite_query"


def decide_after_hallucination_check(state: AgentState) -> Literal["end", "generate"]:
    """
    After hallucination check, decide whether to accept or regenerate.
    If grounded → end. If hallucinated → try generating again once.
    """
    logger.info("NODE: check_hallucination")
    context = _docs_to_context(state["documents"])
    chain = _hallucination_prompt | _llm_json
    result = chain.invoke({"generation": state["generation"], "context": context})

    try:
        import json
        score = json.loads(result.content).get("score", "yes")
    except Exception:
        score = "yes" if "yes" in result.content.lower() else "no"

    if score == "yes":
        logger.info("  → answer is grounded ✓")
        return "end"

    logger.info("  → hallucination detected, regenerating")
    return "generate"


# ── Build the LangGraph ────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate", generate)
    graph.add_node("rewrite_query", rewrite_query)

    # Edges
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
        },
    )

    # After rewrite → go back to retrieve
    graph.add_edge("rewrite_query", "retrieve")

    # After generate → check for hallucinations
    graph.add_conditional_edges(
        "generate",
        decide_after_hallucination_check,
        {
            "end": END,
            "generate": "generate",
        },
    )

    return graph.compile()


# ── Public API ─────────────────────────────────────────────────────────────────

# Compile once at import time (reused across requests)
_graph = build_graph()


def ask(question: str, chat_history: List[BaseMessage] | None = None) -> dict:
    """
    Run the KT RAG agent for a given question.

    Args:
        question:     The user's natural-language question.
        chat_history: Optional prior conversation messages.

    Returns:
        {
            "answer":    str,
            "sources":   list[str],   # unique source filenames cited
            "question":  str,         # possibly rewritten question
        }
    """
    history = chat_history or []
    initial_state: AgentState = {
        "messages": history + [HumanMessage(content=question)],
        "question": question,
        "documents": [],
        "generation": "",
        "rewrite_count": 0,
    }

    final_state = _graph.invoke(initial_state)

    sources = list({
        doc.metadata.get("source", "unknown")
        for doc in final_state.get("documents", [])
    })

    return {
        "answer": final_state.get("generation", ""),
        "sources": sources,
        "question": final_state.get("question", question),
    }


# ── Quick local test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_q = input("Ask a question: ").strip()
    result = ask(test_q)
    print("\nAnswer:\n", result["answer"])
    print("\nSources:", result["sources"])
