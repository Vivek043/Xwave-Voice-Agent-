"""
Knowledge Base Router — Upload, manage, and query the RAG knowledge base.

Endpoints:
  POST /knowledge/upload     → Upload a file to the knowledge base
  POST /knowledge/ingest     → Ingest raw text into the knowledge base
  POST /knowledge/load-defaults → Load the sample knowledge base docs
  GET  /knowledge/stats      → View knowledge base statistics
  POST /knowledge/search     → Test retrieval (search without agent)
  DELETE /knowledge/clear    → Wipe the knowledge base

Postman setup:
  1. First: POST /knowledge/load-defaults (loads the sample NovaCRM docs)
  2. Then: POST /agent/chat with {"message": "What pricing plans do you offer?"}
  3. Watch the agent use the knowledge base to answer accurately!
"""

import os
import logging
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

from services.rag_service import (
    ingest_text,
    ingest_file,
    ingest_directory,
    retrieve,
    format_context_for_llm,
    get_knowledge_stats,
    clear_knowledge_base,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Path to the sample knowledge base
KB_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge_base")


class IngestTextRequest(BaseModel):
    text: str
    source: str = "manual"


class SearchRequest(BaseModel):
    query: str
    top_k: int = 4


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a file (txt, md, pdf) to the knowledge base.

    Postman: POST /knowledge/upload
    Body → form-data → Key: "file" (File) → attach your document
    """
    # Save uploaded file temporarily
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    result = await ingest_file(temp_path)

    # Clean up
    os.remove(temp_path)

    return result


@router.post("/ingest")
async def ingest_raw_text(body: IngestTextRequest):
    """
    Ingest raw text directly into the knowledge base.

    Postman: POST /knowledge/ingest
    Body → JSON: {"text": "Your company information here...", "source": "company-info"}
    """
    return await ingest_text(body.text, source=body.source)


@router.post("/load-defaults")
async def load_default_knowledge_base():
    """
    Load the sample NovaCRM knowledge base documents.
    Call this once to populate the knowledge base with demo data.

    Postman: POST /knowledge/load-defaults
    """
    if not os.path.exists(KB_DIR):
        return {"status": "error", "message": f"Knowledge base directory not found: {KB_DIR}"}

    results = await ingest_directory(KB_DIR)

    total_chunks = sum(r.get("chunks", 0) for r in results)
    return {
        "status": "loaded",
        "files_processed": len(results),
        "total_chunks": total_chunks,
        "details": results,
    }


@router.get("/stats")
async def knowledge_stats():
    """View current knowledge base statistics."""
    return get_knowledge_stats()


@router.post("/search")
async def search_knowledge(body: SearchRequest):
    """
    Test retrieval directly (without going through the agent).
    Useful for debugging RAG quality.

    Postman: POST /knowledge/search
    Body → JSON: {"query": "pricing plans", "top_k": 3}
    """
    chunks = await retrieve(body.query, top_k=body.top_k)
    formatted = format_context_for_llm(chunks)

    return {
        "query": body.query,
        "results": chunks,
        "formatted_context": formatted,
        "total_results": len(chunks),
    }


@router.delete("/clear")
async def clear_kb():
    """Wipe the entire knowledge base. Use with caution."""
    return clear_knowledge_base()
