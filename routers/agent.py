"""
Agent Router — The Full Pipeline

Endpoint:
  POST /agent/chat → Full turn: sentiment analysis + LangGraph + response
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
    session_id: str = None  # Optional — creates new session if not provided

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "I need help resetting my password",
                    "session_id": None
                },
                {
                    "message": "This is the third time I'm calling! Nothing gets fixed!",
                    "session_id": "existing-session-id"
                }
            ]
        }
    }


@router.post("/chat")
async def chat(body: ChatRequest):
    """
    Full agent turn: sentiment → LangGraph → response.

    **Postman setup:**
    1. Method: POST
    2. URL: http://localhost:8000/agent/chat
    3. Body → raw → JSON
    4. First call (no session_id): {"message": "Hi, I need help with my account"}
    5. Copy the session_id from the response
    6. Second call (keep session going): {"message": "...", "session_id": "copy-here"}
    7. Send multiple negative messages to watch escalation trigger

    **Then go to LangSmith:** smith.langchain.com → your project
    → See the full agent trace with nodes, prompts, and token counts
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
    for turn in past_turns[-6:]:  # Last 6 turns
        history.append({"role": "user", "content": f"[Sentiment: {turn.sentiment_label}] {turn.user_transcript}"})
        history.append({"role": "assistant", "content": turn.agent_response})

    # Step 1: Sentiment analysis
    sentiment = await analyze_sentiment(body.message)
    logger.info(f"Turn {turn_number} | Sentiment: {sentiment['label']} ({sentiment['score']:+.2f})")

    # Step 2: Run LangGraph agent (LangSmith traces this automatically)
    state = await agent_graph.ainvoke({
        "session_id": session_id,
        "user_message": body.message,
        "sentiment": sentiment,
        "turn_number": turn_number,
        "conversation_history": history,
        "should_escalate": False,
        "negative_turn_count": negative_turns,
        "system_prompt": "",
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
        "tip": "Visit smith.langchain.com to see the full trace of this conversation"
    }


@router.get("/conversation/{session_id}")
async def get_conversation(session_id: str):
    """
    View the full conversation history for a session.

    Postman: GET http://localhost:8000/agent/conversation/{session_id}
    """
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
    Full voice pipeline: Audio in → Transcript → Sentiment → Agent → Audio out.

    This is the complete Xwave voice loop. Test via Postman:
    1. POST /agent/chat/voice
    2. Body → form-data
    3. Key: "audio" (File) → attach a recorded .mp3/.wav
    4. Key: "session_id" (Text) → optional, leave blank for new session
    5. Response → save as .mp3 and play the agent's voice response!
    """
    # Step 1: Transcribe audio
    audio_bytes = await audio.read()
    transcription = await transcribe_audio(audio_bytes, audio.filename or "audio.webm")
    user_text = transcription["text"]

    # Step 2: Run the text chat pipeline
    from pydantic import BaseModel as BM
    class _Req(BM):
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
    }

    return Response(
        content=audio_response,
        media_type="audio/mpeg",
        headers=headers,
    )
