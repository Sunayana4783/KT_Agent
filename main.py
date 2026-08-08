"""
main.py — FastAPI backend for the KT Agent.

Endpoints:
  POST /ingest          — upload & ingest a PDF file
  POST /ask             — ask a question against the knowledge base
  GET  /health          — health check (used by Render)
  GET  /docs            — auto-generated Swagger UI (built-in FastAPI)
"""

from __future__ import annotations

import os
import logging
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

load_dotenv()

# Set before any ML imports to prevent TensorFlow conflicts
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", "uploads"))
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="KT Agent API",
    description=(
        "Knowledge Transfer agent powered by LangGraph 1.0 + RAG.\n\n"
        "Upload PDF documents and ask questions grounded in their content."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup: pre-load embedding model so first request is fast ─────────────────
@app.on_event("startup")
async def preload_models():
    """
    Download and cache the embedding model at startup.
    This prevents timeout on first /ingest request.
    """
    try:
        logger.info("Pre-loading embedding model at startup...")
        from ingest import _get_embeddings
        _get_embeddings()
        logger.info("Embedding model loaded and cached.")
    except Exception as e:
        logger.warning("Could not pre-load embedding model: %s", e)


# ── Pydantic schemas ───────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, example="What is the deployment process?")
    chat_history: Optional[List[dict]] = Field(
        default=None,
        description="Optional prior turns: [{role: 'user'|'assistant', content: '...'}]",
    )


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    question_used: str   # may differ from input if query was rewritten


class IngestResponse(BaseModel):
    message: str
    filename: str
    chunks_stored: int


class HealthResponse(BaseModel):
    status: str
    version: str


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Render uses this endpoint to verify the service is alive."""
    return {"status": "ok", "version": app.version}


@app.post("/ingest", response_model=IngestResponse, tags=["Knowledge Base"])
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and ingest it into the vector store.

    - Accepts: `multipart/form-data` with a `file` field.
    - The PDF is saved to disk, parsed page-by-page, chunked, embedded, and
      stored in ChromaDB.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    # Save uploaded file to a temp location
    dest_path = UPLOADS_DIR / file.filename
    try:
        with dest_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info("Saved upload: %s (%d bytes)", dest_path, dest_path.stat().st_size)
    except Exception as exc:
        logger.exception("Failed to save upload")
        raise HTTPException(status_code=500, detail=f"Could not save file: {exc}")

    # Ingest into ChromaDB
    try:
        from ingest import ingest_pdf as _ingest_pdf
        vectorstore = _ingest_pdf(dest_path)
        collection = vectorstore.get()
        chunk_count = len(collection.get("ids", []))
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {exc}")

    return IngestResponse(
        message="PDF ingested successfully.",
        filename=file.filename,
        chunks_stored=chunk_count,
    )


@app.post("/ask", response_model=AskResponse, tags=["Agent"])
def ask_question(request: AskRequest):
    """
    Ask the KT agent a question.

    The agent will:
    1. Retrieve relevant chunks from the knowledge base.
    2. Grade and filter chunks for relevance.
    3. Generate a grounded answer (with hallucination check).
    4. Return the answer along with source document names.
    """
    from langchain_core.messages import HumanMessage, AIMessage
    from agent import ask as agent_ask

    # Convert optional chat history to LangChain message objects
    history = []
    if request.chat_history:
        for turn in request.chat_history:
            role = turn.get("role", "").lower()
            content = turn.get("content", "")
            if role == "user":
                history.append(HumanMessage(content=content))
            elif role == "assistant":
                history.append(AIMessage(content=content))

    try:
        result = agent_ask(question=request.question, chat_history=history)
    except Exception as exc:
        logger.exception("Agent error")
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        question_used=result["question"],
    )


@app.get("/", tags=["System"])
def root():
    """Redirect hint — visit /docs for the Swagger UI."""
    return JSONResponse({"message": "KT Agent API is running. Visit /docs for usage."})


# ── Entry point for local dev ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )
