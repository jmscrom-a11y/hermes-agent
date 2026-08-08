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
    LLM_PROVIDER: str = "ollama"
    LLM_BASE_URL: str = "http://localhost:11434/v1"
    LLM_MODEL: str = "ornith:9b"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 4096

    # ── Planner ───────────────────────────────────────────────────
    PLANNER_STRATEGY: str = "default"
    PLANNER_MAX_STEPS: int = 20
    PLANNER_CONTEXT_WINDOW: int = 8000

    # ── Executor ──────────────────────────────────────────────────
    EXECUTOR_MAX_PARALLEL: int = 4
    EXECUTOR_DEFAULT_TIMEOUT: int = 300
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

    # ── Tools ─────────────────────────────────────────────────────
    TOOLS_ENABLED: List[str] = Field(
        default_factory=lambda: [
            "ollama", "rag", "tavily", "telegram", "git", "claude_code",
        ]
    )

    # ── Telegram ──────────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_ALLOWED_USER_IDS_V4: List[str] = Field(default_factory=list)

    # ── RAG ───────────────────────────────────────────────────────
    RAG_EMBEDDING_MODEL: str = "mxbai-embed-large"

    # ── Claude Code Tool ──────────────────────────────────────────
    HERMES_WORKSPACE: str = "."
    CLAUDE_CODE_BIN: str = "claude"
    CLAUDE_CODE_PERMISSION_MODE: str = "acceptEdits"
    CLAUDE_CODE_TIMEOUT: int = 600
    # Ambient shells on this machine export a placeholder ANTHROPIC_API_KEY
    # (used for OpenAI-compatible Ollama endpoints) that shadows normal
    # `claude` CLI login and breaks headless calls with 401s. Strip it by
    # default; set to False if you intentionally want API-key auth.
    CLAUDE_CODE_UNSET_API_KEY: bool = True

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
