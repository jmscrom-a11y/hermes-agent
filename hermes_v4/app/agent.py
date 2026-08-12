"""Main entry point for Hermes V4.

Wires together the LLM provider, tool registry, planner, workflow
engine, and Telegram interface, then starts polling.
"""

from __future__ import annotations

import logging

from hermes_v4.config.settings import get_settings
from hermes_v4.core.base import ToolRegistry
from hermes_v4.llm.ollama_provider import OllamaProvider
from hermes_v4.llm.provider import LLMProvider
from hermes_v4.memory.store import SqliteMemoryStore
from hermes_v4.planner.planner import Planner
from hermes_v4.telegram.bot import build_application
from hermes_v4.tools.claude_code_tool import ClaudeCodeTool
from hermes_v4.tools.git_tool import GitTool
from hermes_v4.tools.memory_tool import RememberTool
from hermes_v4.tools.rag_tool import RAGTool
from hermes_v4.tools.reminder_tool import (
    CancelReminderTool,
    EditReminderTool,
    ListRemindersTool,
    ReminderTool,
)
from hermes_v4.tools.report_tool import ReportTool
from hermes_v4.tools.usage_tool import UsageTool
from hermes_v4.tools.web_search_tool import WebSearchTool
from hermes_v4.workflow.engine import WorkflowEngine

logger = logging.getLogger(__name__)

# Maps a TOOLS_ENABLED entry to its Tool class. Entries with no
# implementation yet are skipped with a warning instead of crashing
# startup. "generate_report" and "schedule_reminder" need the shared LLM
# instance (and both optionally WebSearchTool — generate_report for
# research, schedule_reminder for grounding "날씨"/"뉴스" reminders in real
# search instead of an LLM guess); "list_reminders"/"cancel_reminder"/
# "edit_reminder" wrap the already-registered "schedule_reminder" instance;
# "remember" needs
# the shared memory store. All are special-cased in build_tool_registry()
# instead of living here.
_AVAILABLE_TOOLS = {
    "rag": RAGTool,
    "claude_code": ClaudeCodeTool,
    "web_search": WebSearchTool,
    "git": GitTool,
    "usage_report": UsageTool,
}


def build_llm_provider(settings) -> LLMProvider:
    if settings.LLM_PROVIDER == "ollama":
        return OllamaProvider()
    if settings.LLM_PROVIDER == "openai":
        from hermes_v4.llm.openai_provider import OpenAIProvider

        return OpenAIProvider()
    if settings.LLM_PROVIDER == "gemini":
        from hermes_v4.llm.gemini_provider import GeminiProvider

        return GeminiProvider()
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.LLM_PROVIDER!r}")


def build_tool_registry(settings, llm: LLMProvider, memory=None) -> ToolRegistry:
    registry = ToolRegistry()
    for name in settings.TOOLS_ENABLED:
        if name == "generate_report":
            # Self-researches via WebSearchTool and hands PPTX slide design
            # to ClaudeCodeTool if either is already registered — list
            # TOOLS_ENABLED with "web_search"/"claude_code" before
            # "generate_report" (the default order) for these to be available.
            registry.register(
                ReportTool(
                    llm,
                    web_search_tool=registry.get_tool("web_search"),
                    claude_code_tool=registry.get_tool("claude_code"),
                )
            )
            continue
        if name == "schedule_reminder":
            registry.register(ReminderTool(llm, web_search_tool=registry.get_tool("web_search")))
            continue
        if name in ("list_reminders", "cancel_reminder", "edit_reminder"):
            # All three wrap the same ReminderTool instance's persisted
            # store/JobQueue, so "schedule_reminder" must already be
            # registered (the default TOOLS_ENABLED order lists it first).
            base = registry.get_tool("schedule_reminder")
            if base is None:
                logger.warning(
                    "'%s' is listed in TOOLS_ENABLED but 'schedule_reminder' isn't enabled; skipping", name
                )
                continue
            if name == "list_reminders":
                registry.register(ListRemindersTool(base))
            elif name == "cancel_reminder":
                registry.register(CancelReminderTool(base))
            else:
                registry.register(EditReminderTool(base))
            continue
        if name == "remember":
            if memory is None:
                logger.warning(
                    "'remember' is listed in TOOLS_ENABLED but MEMORY_BACKEND is disabled; skipping"
                )
                continue
            registry.register(RememberTool(memory))
            continue
        tool_cls = _AVAILABLE_TOOLS.get(name)
        if tool_cls is None:
            logger.warning(
                "Tool '%s' is listed in TOOLS_ENABLED but has no implementation yet; skipping", name
            )
            continue
        registry.register(tool_cls())
    return registry


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.LOG_LEVEL)

    llm = build_llm_provider(settings)
    memory = SqliteMemoryStore() if settings.MEMORY_BACKEND == "sqlite" else None
    registry = build_tool_registry(settings, llm, memory)
    planner = Planner(
        llm_provider=llm,
        tool_registry=registry,
        max_steps=settings.PLANNER_MAX_STEPS,
        context_window=settings.PLANNER_CONTEXT_WINDOW,
        cache_ttl_seconds=settings.PLANNER_CACHE_TTL_SECONDS,
        cache_size=settings.PLANNER_CACHE_SIZE,
    )
    engine = WorkflowEngine(
        registry,
        max_parallel=settings.EXECUTOR_MAX_PARALLEL,
        default_step_timeout=settings.EXECUTOR_DEFAULT_TIMEOUT,
        memory=memory,
    )

    logger.info("Hermes V4 starting with tools: %s", registry.list_tools())
    application = build_application(registry, llm, planner=planner, engine=engine, memory=memory)

    reminder_tool = registry.get_tool("schedule_reminder")
    if reminder_tool is not None:
        if application.job_queue is None:
            logger.warning(
                "JobQueue unavailable (install python-telegram-bot[job-queue]) — "
                "schedule_reminder tool will fail at call time"
            )
        else:
            reminder_tool.bind_job_queue(application.job_queue)

            async def _post_init(_app) -> None:
                await reminder_tool.restore_all()

            application.post_init = _post_init

    application.run_polling()


if __name__ == "__main__":
    main()
