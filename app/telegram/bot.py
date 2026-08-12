import os, json, pathlib, requests, pptx
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from duckduckgo_search import DDGS
from app.config.settings import BOT_TOKEN, TELEGRAM_ALLOWED_USER_IDS
from app.rag.pipeline import load_pipeline, RAGPipeline
from app.services.safety import (
    SafetyError,
    cancel_action,
    confirm_action,
    dry_run_delete_file,
    dry_run_shell,
)
from app.telegram.handlers.chat import handle_message, answer_question
from app.telegram.handlers.write import handle_write as _write_handler

URL = "http://localhost:11434/v1/chat/completions"


def make_ppt(t, c):
    p = pptx.Presentation()
    s = p.slides.add_slide(p.slide_layouts[1])
    s.shapes.title.text = str(t)
    if isinstance(c, list): c = chr(10).join(c)
    s.placeholders[1].text = str(c)
    p.save("report.pptx")
def search_web(q):
    try:
        res = DDGS().news(q, max_results=5)
        if not res: res = DDGS().text(q, max_results=5)
    except Exception:
        pass

def _is_authorized(update):
    if not TELEGRAM_ALLOWED_USER_IDS:
        return True
    user = update.effective_user
    return bool(user and str(user.id) in TELEGRAM_ALLOWED_USER_IDS)

async def _reject_unauthorized(update):
    if update.message:
        await update.message.reply_text("unauthorized")

def _confirmation_message(action):
    return f"{action.preview}\n\nconfirm: /confirm {action.token}\ncancel: /cancel {action.token}"

async def handle_delete(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        await _reject_unauthorized(update)
        return
    if not context.args:
        await update.message.reply_text("usage: /delete path")
        return

    try:
        action = dry_run_delete_file(context.args[0])
        await update.message.reply_text(_confirmation_message(action))
    except SafetyError as exc:
        await update.message.reply_text(str(exc))

async def handle_shell(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        await _reject_unauthorized(update)
        return
    if not update.message or not update.message.text:
        return

    _, _, command = update.message.text.partition(" ")
    if not command.strip():
        await update.message.reply_text("usage: /shell command")
        return

    try:
        action = dry_run_shell(command.strip())
        await update.message.reply_text(_confirmation_message(action))
    except SafetyError as exc:
        await update.message.reply_text(str(exc))

async def handle_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        await _reject_unauthorized(update)
        return
    if not context.args:
        await update.message.reply_text("usage: /confirm token")
        return

    try:
        result = confirm_action(context.args[0])
        await update.message.reply_text(result)
    except SafetyError as exc:
        await update.message.reply_text(str(exc))

async def handle_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        await _reject_unauthorized(update)
        return
    if not context.args:
        await update.message.reply_text("usage: /cancel token")
        return

    try:
        result = cancel_action(context.args[0])
        await update.message.reply_text(result)
    except SafetyError as exc:
        await update.message.reply_text(str(exc))

def build_application():
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is not set.")

    from app.config.settings import RAG_INDEX_DIR, RAG_TOP_K
    from app.llm.agent import ask_llm

    # RAG 파이프라인 인라인 생성
    pipeline: RAGPipeline | None = None
    try:
        pipeline = load_pipeline(
            RAG_INDEX_DIR,
            k=RAG_TOP_K,
            llm=ask_llm,
            hybrid=True,
        )
    except Exception as e:
        print("LOAD_PIPELINE ERROR:", repr(e))
        pipeline = None

    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("write", lambda u, c: _write_handler(u, c, _is_authorized, _reject_unauthorized)))
    application.add_handler(CommandHandler("delete", handle_delete))
    application.add_handler(CommandHandler("shell", handle_shell))
    application.add_handler(CommandHandler("confirm", handle_confirm))
    application.add_handler(CommandHandler("cancel", handle_cancel))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, lambda u, c: handle_message(u, c, pipeline))
    )
    return application

def main():
    build_application().run_polling()

if __name__ == "__main__":
    main()
