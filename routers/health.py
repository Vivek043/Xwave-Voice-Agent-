"""
Health check endpoint — verifies all services are reachable.
Test this first in Postman: GET http://localhost:8000/health
"""

from fastapi import APIRouter
import httpx
import os
from dotenv import load_dotenv

load_dotenv()
router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Check connectivity to all external services.
    Great for Postman — run this to verify your setup before testing other endpoints.
    """
    results = {}

    # Check Ollama (local)
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/tags",
                timeout=3.0
            )
        results["ollama"] = "✅ connected" if r.status_code == 200 else f"❌ status {r.status_code}"
    except Exception as e:
        results["ollama"] = f"❌ {str(e)} — Is Ollama running? Run: ollama serve"

    # Check Groq
    groq_key = os.getenv("GROQ_API_KEY", "")
    results["groq"] = "✅ key present" if groq_key.startswith("gsk_") else "❌ missing GROQ_API_KEY in .env"

    # Check HuggingFace
    hf_token = os.getenv("HF_API_TOKEN", "")
    results["huggingface"] = "✅ key present" if hf_token.startswith("hf_") else "❌ missing HF_API_TOKEN in .env"

    # Check LangSmith
    ls_key = os.getenv("LANGCHAIN_API_KEY", "")
    results["langsmith"] = "✅ key present" if ls_key else "⚠️ missing LANGCHAIN_API_KEY (tracing disabled)"

    all_ok = "❌" not in str(results.values())

    return {
        "status": "healthy" if all_ok else "degraded",
        "services": results,
        "tip": "Open /docs for interactive API docs"
    }
