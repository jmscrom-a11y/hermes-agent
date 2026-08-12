"""Hermes V4 configuration settings.

Extends V3 settings with V4-specific configuration.
Uses pydantic-settings with .env file loading.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env from project root
_project_root = Path(__file__).resolve().parent.parent.parent
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(str(_env_path), override=True)


class HermesSettings(BaseSettings):
    """Hermes V4 configuration.

    All settings are read from environment variables with sensible defaults.
    V3 settings are preserved for backward compatibility.
    """

    model_config = SettingsConfigDict(
        env_file=None,  # We load .env manually above
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── V3 Compatibility ──────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434/v1"
    MODEL: str = "ornith:9b"
    BOT_TOKEN: str = ""
    BACKUP_DIR: str = "data/backups"
    LOG_DIR: str = "logs"
    AUTO_RESTART: bool = False
    RAG_INDEX_DIR: str = "data/faiss_index"
    RAG_TOP_K: int = 4
    TELEGRAM_ALLOWED_USER_IDS: List[str] = Field(default_factory=list)
    ALLOWED_FILE_DIRS: List[str] = Field(default_factory=lambda: [str(Path.cwd())])
    ALLOWED_SHELL_COMMANDS: List[str] = Field(
        default_factory=lambda: ["python3", "venv/bin/python", "ls", "pwd", "cat", "rg", "git"]
    )

    # ── LLM ───────────────────────────────────────────────────────
    # "ollama" (local, free), "openai", or "gemini" (both hosted — far
    # more reliable at following the planner's JSON schema than a local
    # 9B model, at the cost of needing an API key and network access).
    LLM_PROVIDER: str = "ollama"
    LLM_BASE_URL: str = "http://localhost:11434/v1"
    LLM_MODEL: str = "ornith:9b"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 4096
    # ornith:9b (and other thinking-capable Ollama models) run a hidden
    # reasoning pass before answering unless this is off — measured 22.9s
    # vs 0.7s for the same prompt. Off by default for chat/tool-call
    # latency; flip on only if you need it for genuinely hard reasoning.
    LLM_THINKING_ENABLED: bool = False
    # How long Ollama keeps the interactive chat/planning model resident in
    # memory after a request. -1 = never unload — a finite value still lets
    # macOS's disk cache for the weights file get evicted under memory
    # pressure, which turned a warm ~0.2s reload into a ~20s one in testing;
    # staying resident in VRAM sidesteps that entirely. Only applied to
    # calls using the default chat model — REPORT_LLM_MODEL (much bigger,
    # used for latency-tolerant one-off report generation) intentionally
    # keeps Ollama's normal default so it doesn't stay resident unused.
    LLM_CHAT_KEEP_ALIVE: str = "-1"

    # Vision-capable model for the Telegram photo handler (Ollama only —
    # already pulled locally, unlike a hosted vision model that'd need
    # its own API key/route).
    LLM_VISION_MODEL: str = "llava:latest"

    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"

    # ── Planner ───────────────────────────────────────────────────
    PLANNER_STRATEGY: str = "default"
    PLANNER_MAX_STEPS: int = 20
    PLANNER_CONTEXT_WINDOW: int = 8000
    # Reuse a previously-generated plan for an identical (normalized) request
    # instead of paying for another planning LLM call. Keyed on exact request
    # text only — doesn't help paraphrases, but is safe since the cached plan
    # was itself validated for that exact text. TTL bounds staleness (e.g. a
    # cached "generate_report" plan outliving new documents in the RAG index).
    PLANNER_CACHE_TTL_SECONDS: float = 300.0
    PLANNER_CACHE_SIZE: int = 128

    # ── Executor ──────────────────────────────────────────────────
    EXECUTOR_MAX_PARALLEL: int = 4
    # Must exceed CLAUDE_CODE_TIMEOUT (900s) — this wraps every step
    # including claude_code/generate_report calls, so a shorter value here
    # makes that tool's own timeout unreachable (this wrapper always fires
    # first) and asyncio.TimeoutError has no message, so a step killed by
    # it fails with a blank error.
    EXECUTOR_DEFAULT_TIMEOUT: int = 1200
    EXECUTOR_RETRY_MAX_ATTEMPTS: int = 3
    EXECUTOR_RETRY_BACKOFF_FACTOR: float = 2.0

    # ── Workflow ──────────────────────────────────────────────────
    WORKFLOW_MAX_NODES: int = 50
    WORKFLOW_STATE_PERSIST: bool = True

    # ── Memory ────────────────────────────────────────────────────
    MEMORY_BACKEND: str = "sqlite"
    MEMORY_SQLITE_PATH: str = "data/hermes_v4.db"
    MEMORY_REDIS_URL: str = "redis://localhost:6379/0"
    MEMORY_TASK_RETENTION_DAYS: int = 90
    # How many recent user/assistant turns to feed back to the planner so
    # follow-up questions ("그거 좀 더 설명해줘") work without repeating context.
    CONVERSATION_HISTORY_TURNS: int = 10

    # ── Tools ─────────────────────────────────────────────────────
    # Names here map to Tool implementations registered in the
    # ToolRegistry (see app/agent.py). "ollama" (LLM provider) and
    # "telegram" (messaging interface) are wired separately, not as
    # Tools, so they don't belong in this list.
    TOOLS_ENABLED: List[str] = Field(
        default_factory=lambda: [
            "rag", "claude_code", "web_search", "git", "generate_report", "schedule_reminder",
            "list_reminders", "cancel_reminder", "edit_reminder", "usage_report", "remember",
        ]
    )

    # ── Report Tool ───────────────────────────────────────────────
    REPORTS_DIR: str = "data/reports"
    REPORT_SECTIONS: int = 5
    REPORT_BULLETS_PER_SECTION: int = 4
    # Report generation is a one-off, latency-tolerant request unlike
    # interactive chat — worth spending a bigger/slower local model here
    # for richer output. Empty string = use LLM_MODEL (the chat default).
    REPORT_LLM_MODEL: str = "qwen3.6:27b"
    # Number of distinct search angles the local model fans out into before
    # synthesizing a research brief (background, latest news, stats,
    # counterpoints, examples, etc.) instead of one flat search. Searches
    # run in parallel (asyncio.gather), so this mainly costs synthesis-call
    # tokens/time, not wall-clock search time.
    REPORT_RESEARCH_QUERIES: int = 6

    # ── Telegram ──────────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_ALLOWED_USER_IDS_V4: List[str] = Field(default_factory=list)

    # ── RAG ───────────────────────────────────────────────────────
    # bge-m3: measured against this corpus, cosine similarity for a truly
    # relevant chunk (0.59) clears an unrelated "hub" chunk (0.32-0.37) by a
    # wide margin. mxbai-embed-large and nomic-embed-text were both measured
    # to invert this — an unrelated chunk scored *higher* than the relevant
    # one (e.g. mxbai: 0.67 relevant vs 0.79 unrelated) — because short,
    # generic-sounding chunks sit near the embedding space's centroid and
    # spuriously look similar to almost any query. bge-m3 is also a verified
    # top performer on public multilingual/Korean retrieval benchmarks
    # (MIRACL etc.), not just this one corpus.
    RAG_EMBEDDING_MODEL: str = "bge-m3"
    # Local Ollama OCR model used to ingest image files (.png/.jpg/.jpeg)
    # and to fall back on scanned/image-only PDF pages where PyMuPDF's
    # text-layer extraction comes back empty (see app/rag/loader.py).
    RAG_OCR_ENABLED: bool = True
    RAG_OCR_MODEL: str = "frob/unlimited-ocr:q8_0"
    # A PDF page with fewer extracted characters than this is treated as
    # scanned/image-only and re-OCR'd from a rendered page image.
    RAG_OCR_MIN_TEXT_CHARS: int = 20

    # ── Voice (Speech-to-Text) ───────────────────────────────────────
    # faster-whisper model size — "base" (~150MB, fast) is the default;
    # "small"/"medium" trade speed for accuracy. Runs locally on CPU, no
    # API key, consistent with this project's local-first LLM setup.
    STT_MODEL: str = "base"
    STT_LANGUAGE: str = "ko"

    # ── Voice (Text-to-Speech) ───────────────────────────────────────
    # macOS `say` voice name (empty = system default). Only used for
    # voice-in -> voice-out replies; text messages stay text-only.
    TTS_VOICE: str = ""
    TTS_MAX_CHARS: int = 500

    # ── Reminders ─────────────────────────────────────────────────
    # JobQueue itself is in-memory only (forgotten on restart); reminders
    # are also written here so main() can re-register them on startup.
    REMINDERS_STORE_PATH: str = "data/reminders.json"
    # JobQueue.run_daily() treats a tzinfo-naive time as UTC, not local
    # time — without this, a "매일 아침 6시" reminder silently fires at
    # 06:00 UTC (15:00 KST) instead of 6am local time.
    REMINDER_TIMEZONE: str = "Asia/Seoul"
    # A reminder whose prompt mentions "날씨" gets real web-search-grounded
    # weather for each of these locations instead of an LLM guess with no
    # location context (see ReminderTool._fire) — every other reminder
    # topic is unaffected.
    WEATHER_LOCATIONS: List[str] = Field(default_factory=lambda: ["서울", "광명"])
    # A reminder whose prompt is just a bare "브리핑"/"뉴스" (no specific
    # subject) fans out into these topics for web-search grounding instead
    # of searching that uninformative literal text (see
    # ReminderTool._generate_briefing).
    BRIEFING_TOPICS: List[str] = Field(default_factory=lambda: ["주요 뉴스", "경제 뉴스", "IT/테크 뉴스"])

    # ── Claude Code Tool ──────────────────────────────────────────
    HERMES_WORKSPACE: str = "."
    CLAUDE_CODE_BIN: str = "claude"
    CLAUDE_CODE_PERMISSION_MODE: str = "acceptEdits"
    # PPTX design calls (purpose="pptx_design") routinely ran right up
    # against the old 600s ceiling and got killed mid-QA/revision pass —
    # usage logs show pptx_design averaging ~64 turns (up to 136), and the
    # two most recent design calls both hit exactly 600.0s and were killed
    # before Claude Code could act on its own QA findings (font/chart-size
    # fixes). 900s gives it room to actually finish that loop.
    CLAUDE_CODE_TIMEOUT: int = 900
    # Ambient shells on this machine export a placeholder ANTHROPIC_API_KEY
    # (used for OpenAI-compatible Ollama endpoints) that shadows normal
    # `claude` CLI login and breaks headless calls with 401s. Strip it by
    # default; set to False if you intentionally want API-key auth.
    CLAUDE_CODE_UNSET_API_KEY: bool = True
    # Every successful claude_code call's cost/turn count is appended here
    # — claude -p reports total_cost_usd per call but nothing otherwise
    # tracks it, so spend is invisible without this.
    CLAUDE_CODE_USAGE_LOG_PATH: str = "data/claude_code_usage.jsonl"

    # ── Git Tool ──────────────────────────────────────────────────
    # push/reset/checkout/rebase/clean are deliberately excluded — those
    # can discard local work or affect the shared remote, which an
    # LLM-triggered tool call shouldn't be able to do unsupervised.
    GIT_TOOL_ALLOWED_COMMANDS: List[str] = Field(
        default_factory=lambda: ["status", "diff", "log", "branch", "show", "add", "commit", "rev-parse"]
    )

    # ── Safety ────────────────────────────────────────────────────
    SAFETY_ENABLED: bool = True
    SAFETY_ALLOWED_FILE_DIRS: List[str] = Field(
        default_factory=lambda: [str(Path.cwd())]
    )
    SAFETY_ALLOWED_SHELL_COMMANDS: List[str] = Field(
        default_factory=lambda: ["python3", "ls", "pwd", "cat", "rg", "git"]
    )

    # ── Tavily (V3 reuse) ─────────────────────────────────────────
    TAVILY_API_KEY: str = ""

    # ── Logging ───────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"

    # ── Server (optional API) ─────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # ── Post-init CSV parsing ─────────────────────────────────────
    def model_post_init(self, __context) -> None:
        """Parse CSV strings from environment into lists."""
        self.TELEGRAM_ALLOWED_USER_IDS = self._split_csv(
            os.environ.get("TELEGRAM_ALLOWED_USER_IDS", "")
        ) or self.TELEGRAM_ALLOWED_USER_IDS
        self.ALLOWED_FILE_DIRS = self._split_csv(
            os.environ.get("ALLOWED_FILE_DIRS", "")
        ) or self.ALLOWED_FILE_DIRS
        self.ALLOWED_SHELL_COMMANDS = self._split_csv(
            os.environ.get("ALLOWED_SHELL_COMMANDS", "")
        ) or self.ALLOWED_SHELL_COMMANDS
        self.SAFETY_ALLOWED_FILE_DIRS = self._split_csv(
            os.environ.get("SAFETY_ALLOWED_FILE_DIRS", "")
        ) or self.SAFETY_ALLOWED_FILE_DIRS
        self.SAFETY_ALLOWED_SHELL_COMMANDS = self._split_csv(
            os.environ.get("SAFETY_ALLOWED_SHELL_COMMANDS", "")
        ) or self.SAFETY_ALLOWED_SHELL_COMMANDS
        self.TOOLS_ENABLED = self._split_csv(
            os.environ.get("TOOLS_ENABLED", "")
        ) or self.TOOLS_ENABLED

    @staticmethod
    def _split_csv(value: str) -> list[str] | None:
        """Split a comma-separated string into a list, returning None if empty."""
        if not value or not value.strip():
            return None
        return [v.strip() for v in value.split(",") if v.strip()]


# Module-level singleton (V3 compatibility)
_settings: HermesSettings | None = None


def get_settings() -> HermesSettings:
    """Get or create the singleton settings instance."""
    global _settings
    if _settings is None:
        _settings = HermesSettings()
    return _settings


def __getattr__(name: str):
    """Lazy delegation to settings singleton for V3 compatibility."""
    settings = get_settings()
    if hasattr(settings, name):
        return getattr(settings, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
