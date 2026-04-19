"""
Database Service — SQLite via SQLAlchemy
No setup needed. SQLite is built into Python. Zero cost.

What you learn here:
- ORM-based database design (SQLAlchemy)
- Conversation state persistence (critical for multi-turn AI agents)
- Session management pattern
"""

import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Float, Integer, Text, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = "sqlite:///./voice_agent.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ──────────────────────────────────────────────────────────────────────────────
# Database Models
# ──────────────────────────────────────────────────────────────────────────────

class Conversation(Base):
    """Tracks a conversation session (one phone call = one session)."""
    __tablename__ = "conversations"

    id = Column(String, primary_key=True)           # UUID session ID
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = Column(String, default="active")        # active | escalated | closed
    escalation_count = Column(Integer, default=0)
    total_turns = Column(Integer, default=0)


class Turn(Base):
    """Records each exchange within a conversation."""
    __tablename__ = "turns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False)
    turn_number = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # User's side
    user_transcript = Column(Text)
    sentiment_label = Column(String)        # POSITIVE | NEGATIVE | NEUTRAL
    sentiment_score = Column(Float)         # -1.0 to +1.0
    escalate_flag = Column(Boolean, default=False)

    # Agent's side
    agent_response = Column(Text)
    llm_model_used = Column(String)         # e.g. "llama3.1:8b" or "llama-3.1-70b-versatile"
    was_escalated = Column(Boolean, default=False)


# ──────────────────────────────────────────────────────────────────────────────
# CRUD Functions
# ──────────────────────────────────────────────────────────────────────────────

def init_db():
    """Create all tables. Called on app startup."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for FastAPI endpoints to get a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_conversation(session_id: str) -> Conversation:
    db = SessionLocal()
    conv = Conversation(id=session_id)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    db.close()
    return conv


def save_turn(
    session_id: str,
    turn_number: int,
    user_transcript: str,
    sentiment: dict,
    agent_response: str,
    llm_model: str,
    was_escalated: bool = False,
) -> Turn:
    db = SessionLocal()

    turn = Turn(
        session_id=session_id,
        turn_number=turn_number,
        user_transcript=user_transcript,
        sentiment_label=sentiment.get("label", "NEUTRAL"),
        sentiment_score=sentiment.get("score", 0.0),
        escalate_flag=sentiment.get("escalate_flag", False),
        agent_response=agent_response,
        llm_model_used=llm_model,
        was_escalated=was_escalated,
    )
    db.add(turn)

    # Update conversation metadata
    conv = db.query(Conversation).filter(Conversation.id == session_id).first()
    if conv:
        conv.total_turns = turn_number
        if was_escalated:
            conv.escalation_count += 1
            conv.status = "escalated"

    db.commit()
    db.refresh(turn)
    db.close()
    return turn


def get_conversation_history(session_id: str) -> list[dict]:
    """Get all turns for a session — used to build LLM context."""
    db = SessionLocal()
    turns = (
        db.query(Turn)
        .filter(Turn.session_id == session_id)
        .order_by(Turn.turn_number)
        .all()
    )
    db.close()

    return [
        {
            "role": "user",
            "content": f"[Sentiment: {t.sentiment_label} {t.sentiment_score:+.2f}] {t.user_transcript}",
        }
        if i % 2 == 0
        else {
            "role": "assistant",
            "content": t.agent_response,
        }
        for i, t in enumerate(
            [msg for t in turns for msg in [
                {"type": "user", "transcript": t.user_transcript, "sentiment": t.sentiment_label, "score": t.sentiment_score},
                {"type": "agent", "response": t.agent_response},
            ]]
        )
    ]


def get_conversation_turns_raw(session_id: str) -> list[Turn]:
    """Get raw Turn objects for building LLM messages."""
    db = SessionLocal()
    turns = (
        db.query(Turn)
        .filter(Turn.session_id == session_id)
        .order_by(Turn.turn_number)
        .all()
    )
    db.close()
    return turns


def count_consecutive_negative_turns(session_id: str) -> int:
    """Count how many of the last N turns had negative sentiment. Used for escalation."""
    turns = get_conversation_turns_raw(session_id)
    if not turns:
        return 0

    count = 0
    for turn in reversed(turns):
        if turn.sentiment_score is not None and turn.sentiment_score < -0.3:
            count += 1
        else:
            break

    return count
