"""
Sentiment Analysis Service — VADER (local, zero cold-starts, no API key)

WHY VADER instead of HuggingFace:
- HuggingFace free inference API returns 503 (model unloaded) on almost every
  request, causing the "always NEUTRAL / score 0" bug you were seeing.
- VADER runs 100% locally in Python — no network call, no cold-start, instant.
- VADER is purpose-built for short conversational text (exactly what speech produces).

What sentiment detects vs what it doesn't:
  ✅ The MEANING of your words  ("I'm frustrated" → NEGATIVE)
  ✅ Intensity markers          ("very angry", "absolutely terrible")
  ✅ Negations                  ("not happy at all" → NEGATIVE)
  ❌ Tone of voice / pitch      (that needs a separate audio emotion model)

So speak naturally but use expressive words to test it:
  "I am so frustrated, nothing ever works"  → FRUSTRATED / UPSET
  "Everything is perfect, thanks!"          → HAPPY
"""

import os
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Initialised once at import time — pure dictionary lookup, no network
_analyzer = SentimentIntensityAnalyzer()


def _score_to_emotion(compound: float) -> dict:
    """Map VADER compound score (-1 to +1) to a descriptive emotion label."""
    if compound >= 0.7:
        label, emoji = "HAPPY", "😄"
    elif compound >= 0.4:
        label, emoji = "POSITIVE", "😊"
    elif compound >= 0.1:
        label, emoji = "CALM", "🙂"
    elif compound >= -0.1:
        label, emoji = "NEUTRAL", "😐"
    elif compound >= -0.4:
        label, emoji = "CONCERNED", "😕"
    elif compound >= -0.65:
        label, emoji = "FRUSTRATED", "😟"
    else:
        label, emoji = "UPSET", "😤"

    threshold = float(os.getenv("SENTIMENT_ESCALATION_THRESHOLD", "-0.5"))

    return {
        "label": label,
        "emoji": emoji,
        "score": round(compound, 4),
        "confidence": round(abs(compound), 4),
        "escalate_flag": compound < threshold,
        "raw": {},
    }


async def analyze_sentiment(text: str) -> dict:
    """
    Analyze sentiment of transcribed speech using VADER.
    Completely local — no network call, no API key, works offline.

    Postman test:
        POST /sentiment/analyze
        {"text": "I am so frustrated, this is broken for the third day!"}
        → should return FRUSTRATED with score ~ -0.65
    """
    if not text or not text.strip():
        return {
            "label": "NEUTRAL", "emoji": "😐", "score": 0.0,
            "confidence": 0.0, "escalate_flag": False, "raw": {}
        }

    scores = _analyzer.polarity_scores(text)
    result = _score_to_emotion(scores["compound"])
    result["raw"] = {
        "positive": round(scores["pos"], 3),
        "neutral":  round(scores["neu"], 3),
        "negative": round(scores["neg"], 3),
        "compound": round(scores["compound"], 4),
    }

    logger.info(
        f"✅ Sentiment: {result['label']} {result['emoji']} "
        f"(score={result['score']:+.3f}, escalate={result['escalate_flag']}) "
        f"| text='{text[:60]}'"
    )
    return result
