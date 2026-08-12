"""Local text-to-speech for Hermes V4 voice replies.

Uses macOS's built-in `say` command plus ffmpeg to produce OGG/Opus audio
(the format Telegram's send_voice expects) — no API key, no extra model
download, consistent with this project's local-first setup. Only wired
up for voice-in -> voice-out replies (see hermes_v4/telegram/bot.py);
text messages stay text-only.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile

from hermes_v4.config.settings import get_settings


def _synthesize_sync(text: str, voice: str | None) -> bytes:
    if not text.strip():
        raise ValueError("Cannot synthesize empty text")
    if shutil.which("say") is None:
        raise RuntimeError("macOS 'say' command not found — TTS is only supported on macOS")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found — required to convert 'say' output to OGG/Opus")

    settings = get_settings()
    voice = voice or settings.TTS_VOICE

    with tempfile.NamedTemporaryFile(suffix=".aiff") as aiff, tempfile.NamedTemporaryFile(suffix=".ogg") as ogg:
        say_cmd = ["say", "-o", aiff.name]
        if voice:
            say_cmd += ["-v", voice]
        say_cmd.append(text)
        subprocess.run(say_cmd, check=True, capture_output=True)

        subprocess.run(
            ["ffmpeg", "-y", "-i", aiff.name, "-c:a", "libopus", "-ar", "48000", ogg.name],
            check=True, capture_output=True,
        )
        return open(ogg.name, "rb").read()


async def synthesize(text: str, voice: str | None = None) -> bytes:
    """Synthesize speech from text, returning OGG/Opus-encoded audio bytes.

    Both `say` and ffmpeg are blocking subprocess calls, so this runs in a
    worker thread to avoid stalling the bot's event loop.
    """
    return await asyncio.to_thread(_synthesize_sync, text, voice)
