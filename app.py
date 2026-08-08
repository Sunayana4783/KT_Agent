"""
app.py — Streamlit chat UI for the KT Agent.

Talks to the FastAPI backend (main.py) via HTTP.
Set BACKEND_URL env var to point at your deployed Render service.
"""

from __future__ import annotations

import os
import requests
import streamlit as st
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="KT Agent — Knowledge Transfer",
    page_icon="📚",
    layout="wide",
)

# ── Session state init ─────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict] = []

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files: list[str] = []


# ── Helper functions ───────────────────────────────────────────────────────────
def _backend_healthy() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _ingest_pdf(uploaded_file) -> dict | None:
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        r = requests.post(f"{BACKEND_URL}/ingest", files=files, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Ingestion failed: {e}")
        return None


def _ask_agent(question: str, history: list[dict]) -> dict | None:
    try:
        payload = {"question": question, "chat_history": history}
        r = requests.post(f"{BACKEND_URL}/ask", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Agent request failed: {e}")
        return None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 KT Agent")
    st.caption("Knowledge Transfer via PDF + RAG")

    st.divider()

    # Backend status
    status_color = "🟢" if _backend_healthy() else "🔴"
    st.markdown(f"**Backend:** {status_color} `{BACKEND_URL}`")

    st.divider()

    # PDF upload
    st.subheader("Upload Knowledge Base")
    uploaded = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDFs to add to the knowledge base.",
    )

    if st.button("📥 Ingest PDFs", use_container_width=True, type="primary"):
        if not uploaded:
            st.warning("Please select at least one PDF file first.")
        else:
            for f in uploaded:
                with st.spinner(f"Ingesting {f.name} ..."):
                    result = _ingest_pdf(f)
                if result:
                    st.success(f"✅ {result['filename']} — {result['chunks_stored']} chunks stored")
                    if f.name not in st.session_state.ingested_files:
                        st.session_state.ingested_files.append(f.name)

    if st.session_state.ingested_files:
        st.divider()
        st.subheader("Ingested Files")
        for fname in st.session_state.ingested_files:
            st.markdown(f"- 📄 {fname}")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.caption("Built with LangGraph 1.0 · LangChain · ChromaDB · FastAPI")


# ── Main chat area ─────────────────────────────────────────────────────────────
st.title("💬 Ask your Knowledge Base")
st.caption("Upload PDFs in the sidebar, then ask questions here.")

# Render conversation history
for turn in st.session_state.chat_history:
    role = turn["role"]
    content = turn["content"]
    with st.chat_message(role):
        st.markdown(content)
        if role == "assistant" and turn.get("sources"):
            with st.expander("📎 Sources"):
                for src in turn["sources"]:
                    st.markdown(f"- `{src}`")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Show user message immediately
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Only pass role/content pairs to backend (no sources key)
            history_for_api = [
                {"role": t["role"], "content": t["content"]}
                for t in st.session_state.chat_history[:-1]  # exclude current question
            ]
            response = _ask_agent(prompt, history_for_api)

        if response:
            answer = response.get("answer", "No answer returned.")
            sources = response.get("sources", [])
            rewritten = response.get("question_used", prompt)

            st.markdown(answer)

            if sources:
                with st.expander("📎 Sources"):
                    for src in sources:
                        st.markdown(f"- `{src}`")

            if rewritten != prompt:
                st.caption(f"🔍 Query rewritten to: *{rewritten}*")

            # Persist assistant turn
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
        else:
            st.error("No response from the agent. Check that the backend is running and a knowledge base has been ingested.")
