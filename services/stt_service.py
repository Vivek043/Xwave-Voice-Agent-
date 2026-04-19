"""
Speech-to-Text Service — Groq Whisper Large v3
Free tier: 2,000 requests/day, 7,200 audio seconds/hour

What you learn here:
- Calling a third-party REST API in Python (async with httpx)
- Handling binary audio file uploads (multipart form data)
- Groq's OpenAI-compatible API format
"""

import os
import io
import logging
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Groq client — uses GROQ_API_KEY from .env automatically
client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))


async def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> dict:
    """
    Transcribe audio bytes to text using Groq Whisper Large v3.

    Args:
        audio_bytes: Raw audio data (webm, mp4, wav, mp3, etc.)
        filename: Name with extension — Groq uses this to detect format

    Returns:
        dict with 'text' (transcript) and 'language' (detected language)

    Postman test:
        POST /voice/transcribe
        Body: form-data, key="audio", value=<audio file>
    """
    logger.info(f"Transcribing {len(audio_bytes)} bytes of audio via Groq Whisper...")

    try:
        # Groq's transcription API is OpenAI-compatible
        # We wrap bytes in a tuple: (filename, bytes, content_type)
        transcription = await client.audio.transcriptions.create(
            file=(filename, audio_bytes, "audio/webm"),
            model="whisper-large-v3",       # Best free model on Groq
            response_format="verbose_json", # Returns language + segments too
            language=None,                  # Auto-detect language
        )

        result = {
            "text": transcription.text,
            "language": getattr(transcription, "language", "unknown"),
            "duration_seconds": getattr(transcription, "duration", None),
        }

        logger.info(f"✅ Transcription: '{result['text'][:80]}...' (lang: {result['language']})")
        return result

    except Exception as e:
        logger.error(f"❌ Groq Whisper error: {e}")
        raise RuntimeError(f"Transcription failed: {e}")
