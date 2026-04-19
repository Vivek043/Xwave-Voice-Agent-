"""
Sentiment Analysis Router

Endpoint:
  POST /sentiment/analyze → Text in, sentiment score out
"""

from fastapi import APIRouter
from pydantic import BaseModel
from services.sentiment_service import analyze_sentiment

router = APIRouter()


class TextInput(BaseModel):
    text: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "I am extremely frustrated! This has been broken for 3 days!"},
                {"text": "Thanks for the quick help, everything is working now!"},
                {"text": "I need to reset my password"},
            ]
        }
    }


@router.post("/analyze")
async def analyze(body: TextInput):
    """
    Analyze sentiment of text using HuggingFace DistilBERT (free inference).

    **Postman setup:**
    1. Method: POST
    2. URL: http://localhost:8000/sentiment/analyze
    3. Body → raw → JSON
    4. Body:
       ```json
       {"text": "I am really frustrated with this service!"}
       ```
    5. Send → see NEGATIVE score around -0.95

    Try these to see how the score changes:
    - "Everything is working perfectly, thanks!" → POSITIVE ~+0.99
    - "I need to reset my password" → POSITIVE/mild ~+0.5
    - "This is the third time I'm calling, nothing gets fixed!" → NEGATIVE ~-0.97

    Note: First call may take 20s (model warming up on HuggingFace servers).
    """
    result = await analyze_sentiment(body.text)
    return {
        "input": body.text[:100] + "..." if len(body.text) > 100 else body.text,
        **result,
        "interpretation": (
            "😊 User is happy" if result["score"] > 0.3
            else "😟 User is slightly negative" if result["score"] > -0.3
            else "😤 User is frustrated — agent switches to empathetic mode" if result["score"] > -0.7
            else "🚨 User is very upset — escalation may be triggered"
        )
    }
