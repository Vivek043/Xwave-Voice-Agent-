"""
Text-to-Speech Service — Microsoft Edge TTS
No API key. No account. Zero cost. Uses Edge's neural voices.

What you learn here:
- Using async generators for streaming audio
- Python async file I/O (aiofiles)
- Audio format handling (mp3)
"""

import os
import io
import logging
import asyncio
import tempfile
import edge_tts
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Neural voice options (free, no account needed)
# Run `edge-tts --list-voices` in terminal to see all 400+ voices
VOICES = {
    "default": "en-US-JennyNeural",     # Warm, professional female voice
    "male": "en-US-GuyNeural",           # Natural male voice
    "empathetic": "en-US-AriaNeural",    # More expressive, good for escalation
    "formal": "en-US-ChristopherNeural", # Formal/authoritative tone
}


async def synthesize_speech(
    text: str,
    voice_key: str = "default",
    rate: str = "+0%",   # Speed: -50% to +100%
    pitch: str = "+0Hz", # Pitch adjustment
) -> bytes:
    """
    Convert text to speech audio using Edge-TTS neural voices.

    Args:
        text: The text to speak
        voice_key: One of "default", "male", "empathetic", "formal"
        rate: Speech rate adjustment e.g. "+20%", "-10%"
        pitch: Pitch adjustment e.g. "+5Hz", "-10Hz"

    Returns:
        MP3 audio as bytes — send directly to browser

    Postman test:
        GET /voice/speak?text=Hello+I+am+your+voice+agent&voice=empathetic
        → Save response as .mp3 and play it
    """
    voice_name = VOICES.get(voice_key, VOICES["default"])
    logger.info(f"Synthesizing TTS: voice={voice_name}, text='{text[:60]}...'")

    try:
        # Write to temp file (edge-tts doesn't support in-memory directly)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        communicate = edge_tts.Communicate(text, voice=voice_name, rate=rate, pitch=pitch)
        await communicate.save(tmp_path)

        # Read the file back to bytes
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()

        # Clean up temp file
        os.unlink(tmp_path)

        logger.info(f"✅ TTS generated: {len(audio_bytes)} bytes of MP3 audio")
        return audio_bytes

    except Exception as e:
        logger.error(f"❌ TTS error: {e}")
        raise RuntimeError(f"TTS synthesis failed: {e}")


def get_voice_for_sentiment(sentiment_score: float) -> str:
    """
    Pick an appropriate voice based on user's sentiment.
    Negative sentiment → use empathetic voice.
    Positive sentiment → use warm default voice.

    This is the kind of adaptive behavior Xwave uses.
    """
    if sentiment_score < -0.5:
        return "empathetic"  # User is frustrated — be more gentle
    elif sentiment_score < -0.2:
        return "default"     # Slightly negative — stay warm
    else:
        return "default"     # Positive — keep it professional
