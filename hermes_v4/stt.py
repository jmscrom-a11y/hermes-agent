"""Local speech-to-text for Hermes V4 voice messages via faster-whisper.

Runs on CPU with no API key, consistent with this project's local-first
LLM setup (Ollama). The model is loaded lazily on first use and reused
for subsequent transcriptions.
"""

from __future__ import annotations

import asyncio
import threading

from hermes_v4.config.settings import get_settings

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from faster_whisper import WhisperModel

                settings = get_settings()
                _model = WhisperModel(settings.STT_MODEL, device="cpu", compute_type="int8")
    return _model


def _transcribe_sync(path: str) -> str:
    settings = get_settings()
    model = _get_model()
    segments, _info = model.transcribe(path, language=settings.STT_LANGUAGE or None)
    return "".join(segment.text for segment in segments).strip()


async def transcribe(path: str) -> str:
    """Transcribe an audio file to text.

    Model load + inference are both blocking/CPU-bound, so they run in a
    worker thread to avoid stalling the bot's event loop.
    """
    return await asyncio.to_thread(_transcribe_sync, path)
