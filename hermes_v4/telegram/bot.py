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

# Telegram's hard limit is 4096 chars per message; leave margin for
# multi-byte counting differences.
_TELEGRAM_MESSAGE_LIMIT = 4000


def _split_message(text: str, limit: int = _TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Split text into <=limit chunks, preferring paragraph/line breaks."""
    if len(text) <= limit:
        return [text]

    chunks = []
    remaining = text
    while len(remaining) > limit:
        split_at = remaining.rfind("\n\n", 0, limit)
        if split_at == -1:
            split_at = remaining.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


async def _reply(update: Update, text: str) -> None:
    chunks = _split_message(text)
    for i, chunk in enumerate(chunks, start=1):
        prefix = f"[{i}/{len(chunks)}]\n" if len(chunks) > 1 else ""
        await update.message.reply_text(prefix + chunk)


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


def make_message_handler(
    planner: Planner,
    engine: WorkflowEngine,
    llm: LLMProvider,
    allowed_user_ids: list[str],
    memory=None,
    history_turns: int = 10,
):
    async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.text:
            return
        if not _is_authorized(update, allowed_user_ids):
            await update.message.reply_text("unauthorized")
            return

        question = update.message.text
        user = update.effective_user
        user_id = str(user.id) if user else "anonymous"

        history = await memory.get_recent_messages(user_id, limit=history_turns) if memory else []
        exec_context = ExecutionContext(
            request_id=str(update.update_id),
            user_id=user_id,
            request=question,
            metadata={"previous_context": history},
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

        await _reply(update, answer)

        if memory is not None:
            await memory.save_message(user_id, "user", question)
            await memory.save_message(user_id, "assistant", answer)

    return handle_message


def build_application(
    tool_registry: ToolRegistry,
    llm: LLMProvider,
    planner: Planner | None = None,
    engine: WorkflowEngine | None = None,
    memory=None,
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
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            make_message_handler(
                planner, engine, llm, allowed_user_ids, memory=memory,
                history_turns=settings.CONVERSATION_HISTORY_TURNS,
            ),
        )
    )
    return application
