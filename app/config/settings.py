import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _split_csv(value: str | None) -> list[str]:
    """쉼표 구분 환경 변수 값을 list[str]로 파싱."""
    if not value or not value.strip():
        return []
    return [p.strip() for p in value.split(",") if p.strip()]


class _AppSettings(BaseSettings):
    """Hermes Agent 설정 — 환경 변수 + .env 파일 자동 로드."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Ollama ──────────────────────────────────────────────
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434/v1")
    MODEL: str = Field(default="ornith:9b")

    # ── Telegram ────────────────────────────────────────────
    BOT_TOKEN: str = Field(default="")

    # ── Agent ───────────────────────────────────────────────
    BACKUP_DIR: str = Field(default="backup")
    LOG_DIR: str = Field(default="logs")
    AUTO_RESTART: bool = Field(default=True)

    # ── RAG ─────────────────────────────────────────────────
    RAG_INDEX_DIR: str = Field(default="data/faiss_index")
    RAG_TOP_K: int = Field(default=4)

    # ── 후처리: CSV-parse 대상은 os.getenv()에서 직접 읽음 ──
    def model_post_init(self, _context) -> None:
        # TELEGRAM_ALLOWED_USER_IDS
        raw = os.getenv("TELEGRAM_ALLOWED_USER_IDS", "")
        self._telegram_allowed_user_ids: list[str] = _split_csv(raw)

        # ALLOWED_FILE_DIRS — 기본값: 현재 디렉토리
        raw_dirs = os.getenv("ALLOWED_FILE_DIRS", str(Path.cwd()))
        self._allowed_file_dirs: list[str] = _split_csv(raw_dirs) if raw_dirs else [str(Path.cwd())]

        # ALLOWED_SHELL_COMMANDS — 기본값
        raw_cmds = os.getenv(
            "ALLOWED_SHELL_COMMANDS",
            "python3,venv/bin/python,ls,pwd,cat,rg,claude,aider",
        )
        self._allowed_shell_commands: list[str] = _split_csv(raw_cmds) if raw_cmds else [
            "python3", "venv/bin/python", "ls", "pwd", "cat", "rg", "claude", "aider"
        ]

    @property
    def TELEGRAM_ALLOWED_USER_IDS(self) -> list[str]:
        return self._telegram_allowed_user_ids

    @property
    def ALLOWED_FILE_DIRS(self) -> list[str]:
        return self._allowed_file_dirs

    @property
    def ALLOWED_SHELL_COMMANDS(self) -> list[str]:
        return self._allowed_shell_commands


# ── 모듈 레벨 싱글톤 (기존 import 호환) ─────────────────────
_settings = _AppSettings()

OLLAMA_BASE_URL: str = _settings.OLLAMA_BASE_URL
MODEL: str = _settings.MODEL
BOT_TOKEN: str = _settings.BOT_TOKEN
BACKUP_DIR: str = _settings.BACKUP_DIR
LOG_DIR: str = _settings.LOG_DIR
AUTO_RESTART: bool = _settings.AUTO_RESTART
RAG_INDEX_DIR: str = _settings.RAG_INDEX_DIR
RAG_TOP_K: int = _settings.RAG_TOP_K

def __getattr__(name: str):
    """모듈 레벨에서 _settings.property 에 접근 (lazy delegation)."""
    return getattr(_settings, name)
