"""Telegram interface for Hermes V4.

Every text message is handed to the Planner, which decides (via LLM)
which registered tools to use; the WorkflowEngine then executes the
resulting plan. If planning or execution fails outright, falls back
to a direct conversational LLM reply so the bot stays responsive.
"""

from __future__ import annotations

import logging

from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from hermes_v4.config.settings import get_settings
from hermes_v4.core.base import ToolRegistry
from hermes_v4.core.context import ExecutionContext
from hermes_v4.llm.provider import LLMProvider
from hermes_v4.planner.planner import Planner
from hermes_v4.workflow.engine import WorkflowEngine

logger = logging.getLogger(__name__)


def _is_authorized(update: Update, allowed_user_ids: list[str]) -> bool:
    if not allowed_user_ids:
        return True
    user = update.effective_user
    return bool(user and str(user.id) in allowed_user_ids)


def _final_answer(plan) -> str | None:
    """Best-effort extraction of the plan's final answer from its last step."""
    for step in reversed(plan.steps):
        for action in reversed(step.actions):
            value = plan.context.get(action.output_key)
            if value:
                return str(value)
    return None


async def _fallback_reply(llm: LLMProvider, question: str) -> str:
    completion = await llm.generate([{"role": "user", "content": question}])
    return completion.content


def make_message_handler(planner: Planner, engine: WorkflowEngine, llm: LLMProvider, allowed_user_ids: list[str]):
    async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.text:
            return
        if not _is_authorized(update, allowed_user_ids):
            await update.message.reply_text("unauthorized")
            return

        question = update.message.text
        user = update.effective_user
        exec_context = ExecutionContext(
            request_id=str(update.update_id),
            user_id=str(user.id) if user else None,
            request=question,
        )

        try:
            plan = await planner.plan(question, exec_context)
            plan = await engine.run(plan, exec_context)
            # Use whatever the plan produced even if a later step failed —
            # e.g. a search step can succeed and a redundant "report" step
            # can fail; discarding the search result in that case would
            # throw away a real answer in favor of a tool-less chat reply.
            answer = _final_answer(plan)
            if answer is None:
                raise RuntimeError(plan.error or "plan produced no answer")
            if not plan.is_complete:
                logger.warning("Plan '%s' partially failed (%s); replying with partial result", plan.id, plan.error)
        except Exception as exc:
            logger.warning("Planned execution failed (%s), falling back to direct reply", exc)
            answer = await _fallback_reply(llm, question)

        await update.message.reply_text(answer)

    return handle_message


def build_application(
    tool_registry: ToolRegistry,
    llm: LLMProvider,
    planner: Planner | None = None,
    engine: WorkflowEngine | None = None,
) -> Application:
    settings = get_settings()
    bot_token = settings.TELEGRAM_BOT_TOKEN or settings.BOT_TOKEN
    if not bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN (or BOT_TOKEN) is not set.")

    planner = planner or Planner(llm_provider=llm, tool_registry=tool_registry)
    engine = engine or WorkflowEngine(tool_registry)
    allowed_user_ids = settings.TELEGRAM_ALLOWED_USER_IDS_V4 or settings.TELEGRAM_ALLOWED_USER_IDS

    application = Application.builder().token(bot_token).build()
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, make_message_handler(planner, engine, llm, allowed_user_ids))
    )
    return application
