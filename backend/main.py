"""
Voice Agent — FastAPI Backend
Xwave AI voice agent (free stack)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from routers import voice, sentiment, agent, health, knowledge
from services.db_service import init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialize the database."""
    logger.info("🚀 Starting Voice Agent backend...")
    init_db()
    logger.info("✅ Database initialized")
    yield
    logger.info("👋 Shutting down...")


app = FastAPI(
    title="Voice Agent API",
    description="""
    Xwave AI Voice Agent v2 with:
    - 🎤 Speech-to-Text via Groq Whisper
    - 🧠 LLM via Ollama (local) + Groq fallback
    - 💬 Sentiment Analysis via VADER (local)
    - 🔊 Text-to-Speech via Edge-TTS
    - 📚 RAG Knowledge Base via ChromaDB
    - 🔧 Agentic Tool Calling (CRM, tickets, password reset)
    - 📊 Full Tracing via LangSmith
    - ⚠️ Smart Escalation based on sentiment
    """,
    version="2.0.0",
    lifespan=lifespan,
)

# Allow browser to call the API.
# expose_headers is required so the browser JS can READ custom response headers
# like X-Transcript, X-Sentiment etc. Without this they appear as null in the UI.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Session-Id",
        "X-Turn-Number",
        "X-Sentiment",
        "X-Sentiment-Score",
        "X-Action",
        "X-Transcript",
        "X-RAG-Sources",
        "X-Tool-Used",
    ],
)

# Mount routers
app.include_router(health.router, tags=["Health"])
app.include_router(voice.router, prefix="/voice", tags=["Voice (STT + TTS)"])
app.include_router(sentiment.router, prefix="/sentiment", tags=["Sentiment Analysis"])
app.include_router(agent.router, prefix="/agent", tags=["Agent (LangGraph)"])
app.include_router(knowledge.router, prefix="/knowledge", tags=["Knowledge Base (RAG)"])


@app.get("/")
async def root():
    return {
        "message": "Xwave AI Agent v2 — RAG + Tools + Voice 🎙️",
        "docs": "/docs",
        "health": "/health",
        "knowledge": "/knowledge/stats",
    }
