"""
RAG Service — Retrieval-Augmented Generation with ChromaDB

This is the core of what makes Xwave an actual knowledge-aware agent vs a generic chatbot.
When a user asks a question, this service retrieves relevant chunks from the company's
uploaded knowledge base and injects them into the LLM's context.

Architecture:
    Upload docs → Chunk → Embed (sentence-transformers) → Store in ChromaDB
    Query → Embed question → Similarity search → Return top-k chunks

Everything runs locally. Zero API calls. Zero cost.

What you learn here:
- Vector embeddings and similarity search (core RAG concept)
- Document chunking strategies (why chunk size matters)
- ChromaDB as a lightweight vector store
- How production AI agents like Mira ground their responses in company data
"""

import os
import logging
import hashlib
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ── ChromaDB Setup ───────────────────────────────────────────────────────────

# Persist to disk so knowledge survives restarts
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

# Main knowledge collection — one per "tenant" in a multi-tenant system
COLLECTION_NAME = "xwave_knowledge"

def _get_collection():
    """Get or create the main knowledge collection."""
    return _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for text
    )


# ── Text Chunking ────────────────────────────────────────────────────────────

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,          # ~500 chars per chunk (good for embedding models)
    chunk_overlap=80,        # overlap so context doesn't get cut mid-sentence
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)


def _compute_doc_id(text: str, source: str) -> str:
    """Deterministic ID so re-uploading the same doc doesn't duplicate chunks."""
    return hashlib.md5(f"{source}:{text[:100]}".encode()).hexdigest()


# ── Document Ingestion ───────────────────────────────────────────────────────

async def ingest_text(text: str, source: str = "manual", metadata: dict = None) -> dict:
    """
    Ingest raw text into the knowledge base.
    Chunks it, embeds it (ChromaDB handles embedding via its default model),
    and stores it for retrieval.
    """
    collection = _get_collection()
    chunks = _splitter.split_text(text)

    if not chunks:
        return {"status": "empty", "chunks": 0}

    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        doc_id = _compute_doc_id(chunk, source) + f"_{i}"
        ids.append(doc_id)
        documents.append(chunk)
        metadatas.append({
            "source": source,
            "chunk_index": i,
            "total_chunks": len(chunks),
            **(metadata or {}),
        })

    # ChromaDB uses its built-in embedding function (all-MiniLM-L6-v2 by default)
    # No need to manually embed — it's handled automatically
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    logger.info(f"📚 Ingested {len(chunks)} chunks from '{source}'")
    return {"status": "ingested", "source": source, "chunks": len(chunks)}


async def ingest_file(file_path: str) -> dict:
    """
    Ingest a file (txt, md, or pdf) into the knowledge base.
    """
    path = Path(file_path)
    if not path.exists():
        return {"status": "error", "message": f"File not found: {file_path}"}

    ext = path.suffix.lower()

    if ext in (".txt", ".md"):
        text = path.read_text(encoding="utf-8")
    elif ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            return {"status": "error", "message": "pypdf not installed. Run: pip install pypdf"}
    else:
        return {"status": "error", "message": f"Unsupported file type: {ext}"}

    return await ingest_text(text, source=path.name)


async def ingest_directory(dir_path: str) -> list[dict]:
    """Ingest all supported files in a directory."""
    results = []
    path = Path(dir_path)
    for file in sorted(path.iterdir()):
        if file.suffix.lower() in (".txt", ".md", ".pdf"):
            result = await ingest_file(str(file))
            results.append(result)
    return results


# ── Retrieval ────────────────────────────────────────────────────────────────

async def retrieve(query: str, top_k: int = 4) -> list[dict]:
    """
    Retrieve the most relevant knowledge chunks for a given query.
    This is the R in RAG — what gets injected into the LLM context.

    Returns list of {text, source, score} dicts, ranked by relevance.
    """
    collection = _get_collection()

    if collection.count() == 0:
        logger.info("📭 Knowledge base is empty — no retrieval possible")
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
    )

    retrieved = []
    for i, doc in enumerate(results["documents"][0]):
        distance = results["distances"][0][i] if results.get("distances") else 0
        meta = results["metadatas"][0][i] if results.get("metadatas") else {}
        retrieved.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "relevance": round(1 - distance, 4),  # cosine distance → similarity
            "chunk_index": meta.get("chunk_index", 0),
        })

    logger.info(
        f"🔍 Retrieved {len(retrieved)} chunks for query: '{query[:50]}...' "
        f"(best relevance: {retrieved[0]['relevance']:.3f})" if retrieved else ""
    )
    return retrieved


def format_context_for_llm(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a context block the LLM can use.
    This gets injected into the system prompt.
    """
    if not chunks:
        return ""

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source: {chunk['source']}]\n{chunk['text']}"
        )

    return (
        "KNOWLEDGE BASE CONTEXT (use this to answer the user's question):\n"
        "─────────────────────────────────────────────────────────────────\n"
        + "\n\n".join(context_parts)
        + "\n─────────────────────────────────────────────────────────────────\n"
        "If the answer is in the context above, use it. "
        "If not, say so honestly — do NOT make up information."
    )


# ── Management ───────────────────────────────────────────────────────────────

def get_knowledge_stats() -> dict:
    """Return stats about the current knowledge base."""
    collection = _get_collection()
    count = collection.count()

    # Get unique sources
    sources = set()
    if count > 0:
        all_meta = collection.get(include=["metadatas"])
        for meta in all_meta["metadatas"]:
            sources.add(meta.get("source", "unknown"))

    return {
        "total_chunks": count,
        "sources": sorted(sources),
        "collection": COLLECTION_NAME,
        "status": "ready" if count > 0 else "empty",
    }


def clear_knowledge_base():
    """Wipe the entire knowledge base. Use with caution."""
    _client.delete_collection(COLLECTION_NAME)
    logger.warning("🗑️ Knowledge base cleared")
    return {"status": "cleared"}
