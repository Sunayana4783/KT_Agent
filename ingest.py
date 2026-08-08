"""
ingest.py — PDF ingestion pipeline for the KT Agent.

Embeddings: HuggingFace sentence-transformers (FREE, runs locally on CPU)
Vector store: ChromaDB (FREE, local persistent)

No API key required for ingestion.
"""

import os
import logging

# Must be set BEFORE importing sentence_transformers / transformers
# to prevent TensorFlow from being imported (protobuf version conflict)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", "vectorstore"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "kt_knowledge_base")
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "200"))

# Free local embedding model — downloads once (~90 MB), then cached
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return a local HuggingFace sentence-transformer embeddings instance."""
    logger.info("Loading embedding model: %s (local, free)", EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


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
    """Load all PDFs found inside *directory* recursively."""
    directory = Path(directory)
    pdf_files = list(directory.rglob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", directory)
        return []

    all_docs: List[Document] = []
    for pdf in pdf_files:
        all_docs.extend(load_pdf(pdf))

    logger.info("Total pages loaded from directory: %d", len(all_docs))
    return all_docs


def split_documents(docs: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split into %d chunks (size=%d, overlap=%d)", len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)
    return chunks


def get_vectorstore(embeddings: HuggingFaceEmbeddings | None = None) -> Chroma:
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

    logger.info("Upserting %d chunks into ChromaDB collection '%s' ...", len(chunks), COLLECTION_NAME)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR),
    )
    logger.info("Ingestion complete. Vector store persisted at: %s", VECTORSTORE_DIR)
    return vectorstore


def ingest_pdf(file_path: str | Path) -> Chroma:
    """Convenience: load a single PDF and ingest it."""
    docs = load_pdf(file_path)
    return ingest_documents(docs)


def ingest_directory(directory: str | Path = "data") -> Chroma:
    """Convenience: load all PDFs in a directory and ingest them."""
    docs = load_pdfs_from_dir(directory)
    return ingest_documents(docs)


# ── CLI entry-point ────────────────────────────────────────────────────────────
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
