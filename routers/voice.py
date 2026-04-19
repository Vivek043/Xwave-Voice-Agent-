"""
Voice Router — STT and TTS endpoints

Endpoints:
  POST /voice/transcribe  → Upload audio, get transcript
  GET  /voice/speak       → Text in, MP3 audio out

Test both in Postman first before wiring to the frontend.
"""

import io
import logging
from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import Response

from services.stt_service import transcribe_audio
from services.tts_service import synthesize_speech

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(..., description="Audio file (webm, mp3, wav, mp4)")
):
    """
    Transcribe an audio file to text using Groq Whisper Large v3.

    **Postman setup:**
    1. Method: POST
    2. URL: http://localhost:8000/voice/transcribe
    3. Body → form-data
    4. Key: "audio" (type: File)
    5. Value: choose any .mp3 or .wav file
    6. Send → see the transcript in response

    Returns:
        text: The transcribed text
        language: Detected language
        duration_seconds: Audio duration
    """
    audio_bytes = await audio.read()
    filename = audio.filename or "audio.webm"

    logger.info(f"Received audio: {filename} ({len(audio_bytes)} bytes)")

    result = await transcribe_audio(audio_bytes, filename)
    return {
        "status": "success",
        "filename": filename,
        **result
    }


@router.get("/speak")
async def speak(
    text: str = Query(..., description="Text to convert to speech"),
    voice: str = Query("default", description="Voice: default, male, empathetic, formal"),
    rate: str = Query("+0%", description="Speed: -50% to +100%"),
):
    """
    Convert text to speech using Microsoft Edge-TTS (no API key needed).

    **Postman setup:**
    1. Method: GET
    2. URL: http://localhost:8000/voice/speak?text=Hello+I+am+your+voice+agent&voice=empathetic
    3. Send
    4. In response → Save Response → save as .mp3 → play it!

    Try different voices:
    - default: en-US-JennyNeural (warm female)
    - male: en-US-GuyNeural
    - empathetic: en-US-AriaNeural (more expressive)
    - formal: en-US-ChristopherNeural
    """
    audio_bytes = await synthesize_speech(text, voice_key=voice, rate=rate)

    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": f'inline; filename="response.mp3"',
            "Content-Length": str(len(audio_bytes)),
        }
    )
