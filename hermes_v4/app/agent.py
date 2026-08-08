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
from hermes_v4.tools.rag_tool import RAGTool
from hermes_v4.tools.report_tool import ReportTool
from hermes_v4.tools.web_search_tool import WebSearchTool
from hermes_v4.workflow.engine import WorkflowEngine

logger = logging.getLogger(__name__)

# Maps a TOOLS_ENABLED entry to its Tool class. Entries with no
# implementation yet are skipped with a warning instead of crashing
# startup. "generate_report" needs the shared LLM instance (and
# optionally WebSearchTool) so it's special-cased in
# build_tool_registry() instead of living here.
_AVAILABLE_TOOLS = {
    "rag": RAGTool,
    "claude_code": ClaudeCodeTool,
    "web_search": WebSearchTool,
    "git": GitTool,
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


def build_tool_registry(settings, llm: LLMProvider) -> ToolRegistry:
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
    registry = build_tool_registry(settings, llm)
    planner = Planner(
        llm_provider=llm,
        tool_registry=registry,
        max_steps=settings.PLANNER_MAX_STEPS,
        context_window=settings.PLANNER_CONTEXT_WINDOW,
    )
    memory = SqliteMemoryStore() if settings.MEMORY_BACKEND == "sqlite" else None
    engine = WorkflowEngine(
        registry,
        max_parallel=settings.EXECUTOR_MAX_PARALLEL,
        default_step_timeout=settings.EXECUTOR_DEFAULT_TIMEOUT,
        memory=memory,
    )

    logger.info("Hermes V4 starting with tools: %s", registry.list_tools())
    application = build_application(registry, llm, planner=planner, engine=engine, memory=memory)
    application.run_polling()


if __name__ == "__main__":
    main()
