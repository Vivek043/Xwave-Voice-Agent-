"""
Agent Router — The Full Pipeline (v2: with RAG + Tool Calling)

Endpoints:
  POST /agent/chat → Full turn: sentiment → RAG → LangGraph → tools → response
  GET  /agent/conversation/{session_id} → View conversation history
  POST /agent/chat/voice → Audio in, audio out (full voice pipeline)
"""

import uuid
import logging
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import Response
from pydantic import BaseModel

from services.sentiment_service import analyze_sentiment
from services.tts_service import synthesize_speech, get_voice_for_sentiment
from services.db_service import (
    create_conversation,
    save_turn,
    get_conversation_turns_raw,
    count_consecutive_negative_turns,
)
from services.stt_service import transcribe_audio
from agent.graph import agent_graph

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str
    session_id: str = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "I need help resetting my password",
                    "session_id": None
                },
                {
                    "message": "What's the status of ticket TK-1001?",
                    "session_id": "existing-session-id"
                }
            ]
        }
    }


@router.post("/chat")
async def chat(body: ChatRequest):
    """
    Full agent turn: sentiment → RAG retrieval → LangGraph → tool calling → response.

    New in v2:
    - RAG: Agent searches knowledge base for relevant info before responding
    - Tools: Agent can call tools (check_ticket, reset_password, etc.)
    - The response includes retrieval sources and tool actions taken

    Test in Postman:
    1. POST http://localhost:8000/agent/chat
    2. {"message": "What are your pricing plans?"}  → tests RAG
    3. {"message": "Check status of TK-1001"}       → tests tool calling
    4. {"message": "Reset my password for vivek@example.com"} → tests tool calling
    """

    # Create or resume session
    session_id = body.session_id or str(uuid.uuid4())
    if not body.session_id:
        create_conversation(session_id)
        logger.info(f"New session: {session_id}")

    # Get conversation context from DB
    past_turns = get_conversation_turns_raw(session_id)
    turn_number = len(past_turns) + 1
    negative_turns = count_consecutive_negative_turns(session_id)

    # Build message history for LLM context
    history = []
    for turn in past_turns[-6:]:
        history.append({"role": "user", "content": f"[Sentiment: {turn.sentiment_label}] {turn.user_transcript}"})
        history.append({"role": "assistant", "content": turn.agent_response})

    # Step 1: Sentiment analysis
    sentiment = await analyze_sentiment(body.message)
    logger.info(f"Turn {turn_number} | Sentiment: {sentiment['label']} ({sentiment['score']:+.2f})")

    # Step 2: Run LangGraph agent (with RAG + tools — LangSmith traces everything)
    state = await agent_graph.ainvoke({
        "session_id": session_id,
        "user_message": body.message,
        "sentiment": sentiment,
        "turn_number": turn_number,
        "conversation_history": history,
        "should_escalate": False,
        "negative_turn_count": negative_turns,
        "system_prompt": "",
        # RAG fields
        "retrieved_context": "",
        "retrieval_sources": [],
        # Tool fields
        "tool_call_raw": "",
        "tool_name": "",
        "tool_params": {},
        "tool_result": {},
        "needs_tool": False,
        # Output fields
        "agent_response": "",
        "model_used": "",
        "action": "",
    })

    agent_response = state["agent_response"]
    model_used = state["model_used"]
    action = state["action"]
    was_escalated = action == "escalate"

    # Step 3: Persist to SQLite
    save_turn(
        session_id=session_id,
        turn_number=turn_number,
        user_transcript=body.message,
        sentiment=sentiment,
        agent_response=agent_response,
        llm_model=model_used,
        was_escalated=was_escalated,
    )

    return {
        "session_id": session_id,
        "turn": turn_number,
        "user_message": body.message,
        "sentiment": sentiment,
        "agent_response": agent_response,
        "model_used": model_used,
        "action": action,
        "negative_turns": negative_turns,
        # New v2 fields
        "rag_sources": state.get("retrieval_sources", []),
        "tool_used": state.get("tool_name", ""),
        "tool_result": state.get("tool_result", {}),
        "tip": "Visit smith.langchain.com to see the full trace with RAG + tools"
    }


@router.get("/conversation/{session_id}")
async def get_conversation(session_id: str):
    """View the full conversation history for a session."""
    turns = get_conversation_turns_raw(session_id)
    if not turns:
        return {"session_id": session_id, "turns": [], "note": "No turns found for this session"}

    return {
        "session_id": session_id,
        "total_turns": len(turns),
        "turns": [
            {
                "turn": t.turn_number,
                "timestamp": t.timestamp.isoformat(),
                "user": t.user_transcript,
                "sentiment": f"{t.sentiment_label} ({t.sentiment_score:+.2f}) {t.sentiment_score < -0.5 and '⚠️' or ''}",
                "agent": t.agent_response,
                "model": t.llm_model_used,
                "escalated": t.was_escalated,
            }
            for t in turns
        ]
    }


@router.post("/chat/voice")
async def chat_voice(
    audio: UploadFile = File(...),
    session_id: str = Form(None),
):
    """
    Full voice pipeline: Audio in → Transcript → Sentiment → RAG → Agent → Tools → Audio out.
    """
    # Step 1: Transcribe audio
    audio_bytes = await audio.read()
    transcription = await transcribe_audio(audio_bytes, audio.filename or "audio.webm")
    user_text = transcription["text"]

    # Step 2: Run the text chat pipeline
    class _Req(BaseModel):
        message: str
        session_id: str | None

    result = await chat(_Req(message=user_text, session_id=session_id))

    # Step 3: Convert agent response to speech
    voice_key = get_voice_for_sentiment(result["sentiment"]["score"])
    audio_response = await synthesize_speech(result["agent_response"], voice_key=voice_key)

    # Return audio + metadata in headers
    headers = {
        "X-Session-Id": result["session_id"],
        "X-Turn-Number": str(result["turn"]),
        "X-Sentiment": result["sentiment"]["label"],
        "X-Sentiment-Score": str(result["sentiment"]["score"]),
        "X-Action": result["action"],
        "X-Transcript": user_text[:200],
        "X-RAG-Sources": ",".join(result.get("rag_sources", [])),
        "X-Tool-Used": result.get("tool_used", ""),
    }

    return Response(
        content=audio_response,
        media_type="audio/mpeg",
        headers=headers,
    )
