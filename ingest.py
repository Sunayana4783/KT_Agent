"""
ingest.py — PDF ingestion pipeline for the KT Agent.

Embeddings: ChromaDB built-in ONNX model (uses onnxruntime, ~50MB, no torch)
Vector store: ChromaDB (FREE, local persistent)

Fits in Render free tier 512MB RAM — no torch, no sentence-transformers needed.
"""

import os
import logging
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", "vectorstore"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "kt_knowledge_base")
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "200"))


class _ONNXEmbeddings:
    """
    Wraps ChromaDB's built-in ONNX MiniLM embedding function into
    a LangChain-compatible embeddings interface.
    Uses onnxruntime only (~50MB) — no torch, no sentence-transformers.
    """
    def __init__(self):
        self._fn = ONNXMiniLM_L6_V2()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._fn(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._fn([text])[0]


def _get_embeddings() -> _ONNXEmbeddings:
    logger.info("Loading ONNX MiniLM embedding model (lightweight, no torch)")
    return _ONNXEmbeddings()


def load_pdf(file_path: str | Path) -> List[Document]:
    """Load a single PDF and return a list of LangChain Documents (one per page)."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    logger.info("Loading PDF: %s", file_path.name)
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = file_path.name
    logger.info("  -> %d pages loaded", len(docs))
    return docs


def load_pdfs_from_dir(directory: str | Path) -> List[Document]:
    """Load all PDFs found inside directory recursively."""
    directory = Path(directory)
    pdf_files = list(directory.rglob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", directory)
        return []
    all_docs: List[Document] = []
    for pdf in pdf_files:
        all_docs.extend(load_pdf(pdf))
    logger.info("Total pages loaded: %d", len(all_docs))
    return all_docs


def split_documents(docs: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split into %d chunks (size=%d, overlap=%d)", len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)
    return chunks


def get_vectorstore(embeddings=None) -> Chroma:
    """Return the persistent ChromaDB vector store."""
    if embeddings is None:
        embeddings = _get_embeddings()
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )


def ingest_documents(docs: List[Document]) -> Chroma:
    """Chunk, embed, and upsert documents into the persistent vector store."""
    if not docs:
        raise ValueError("No documents to ingest.")
    embeddings = _get_embeddings()
    chunks = split_documents(docs)
    logger.info("Upserting %d chunks into ChromaDB ...", len(chunks))
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR),
    )
    logger.info("Ingestion complete. Persisted at: %s", VECTORSTORE_DIR)
    return vectorstore


def ingest_pdf(file_path: str | Path) -> Chroma:
    """Convenience: load a single PDF and ingest it."""
    docs = load_pdf(file_path)
    return ingest_documents(docs)


def ingest_directory(directory: str | Path = "data") -> Chroma:
    """Convenience: load all PDFs in a directory and ingest them."""
    docs = load_pdfs_from_dir(directory)
    return ingest_documents(docs)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <pdf_file_or_directory>")
        sys.exit(1)
    target = Path(sys.argv[1])
    if target.is_dir():
        ingest_directory(target)
    else:
        ingest_pdf(target)
