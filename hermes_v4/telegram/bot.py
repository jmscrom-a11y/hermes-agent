"""Telegram interface for Hermes V4.

Every text message is handed to the Planner, which decides (via LLM)
which registered tools to use; the WorkflowEngine then executes the
resulting plan. If planning or execution fails outright, falls back
to a direct conversational LLM reply so the bot stays responsive.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import pathlib
import tempfile
import time
import uuid
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from hermes_v4.config.settings import get_settings
from hermes_v4.core.base import ToolRegistry
from hermes_v4.core.context import ExecutionContext
from hermes_v4.llm.provider import LLMProvider
from hermes_v4.planner.planner import Planner, PlanValidationError
from hermes_v4.prompts import GENERAL_ASSISTANT_SYSTEM_PROMPT
from hermes_v4.stt import transcribe
from hermes_v4.tools.rag_tool import RAGTool
from hermes_v4.tools.report_tool import CONVERTIBLE_DOC_SUFFIXES
from hermes_v4.tts import synthesize
from hermes_v4.workflow.engine import WorkflowEngine

logger = logging.getLogger(__name__)

_SUPPORTED_DOC_SUFFIXES = {
    ".pdf", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".py",
    ".png", ".jpg", ".jpeg", ".docx", ".pptx",
}

# Caption/message keywords that turn a document upload (or a later text
# message referencing one) into a "convert this to a PPTX" request instead
# of the default (add to the RAG index).
_PPTX_CONVERT_KEYWORDS = [
    "pptx", "ppt", "피피티", "슬라이드", "발표자료", "프레젠테이션",
]


def _is_pptx_conversion_request(text: str) -> bool:
    lower = text.lower()
    return any(k in lower for k in _PPTX_CONVERT_KEYWORDS)


# Same pattern as _PPTX_CONVERT_KEYWORDS — a document upload's caption (or a
# later text message referencing the most recent upload) that mentions one
# of these skips the RAG-save/PPTX-convert prompt and goes straight to a
# summary/translation reply instead.
_SUMMARIZE_KEYWORDS = ["요약"]
_TRANSLATE_KEYWORDS = ["번역", "translate"]

_TARGET_LANGUAGE_KEYWORDS = {
    "한국어": ["한국어", "한글", "korean"],
    "일본어": ["일본어", "japanese"],
    "중국어": ["중국어", "chinese"],
    "영어": ["영어", "english"],
}


def _is_summarize_request(text: str) -> bool:
    lower = text.lower()
    return any(k in lower for k in _SUMMARIZE_KEYWORDS)


def _is_translate_request(text: str) -> bool:
    lower = text.lower()
    return any(k in lower for k in _TRANSLATE_KEYWORDS)


def _detect_target_language(text: str) -> str:
    lower = text.lower()
    for language, keywords in _TARGET_LANGUAGE_KEYWORDS.items():
        if any(k in lower for k in keywords):
            return language
    return "영어"


# Tracks the most recently uploaded document per chat (chat_id -> (path,
# uploaded_at)) so a *separate* follow-up text message like "이거 pptx로
# 만들어줘" can still be routed to the document-conversion path instead of
# falling through to the Planner's generate_report, which has no idea a
# file was ever uploaded and would hallucinate an unrelated topic.
_recent_uploads: dict[int, tuple[pathlib.Path, float]] = {}
_RECENT_UPLOAD_TTL_SECONDS = 1800

# Tracks the most recently sent photo per chat (chat_id -> (raw_bytes,
# received_at)) so a follow-up text message like "사진에서 글자 좀 더
# 자세히 읽어줘" can be answered against that image instead of falling
# through to the Planner with no idea an image was ever involved. Only
# kicks in when the follow-up text actually references an image (see
# _is_photo_followup) — a random unrelated question after a photo
# shouldn't be forced through the vision model.
_recent_photos: dict[int, tuple[bytes, float]] = {}
_RECENT_PHOTO_TTL_SECONDS = 900

_PHOTO_FOLLOWUP_KEYWORDS = ["사진", "이미지", "그림"]


def _is_photo_followup(text: str) -> bool:
    lower = text.lower()
    return any(k in lower for k in _PHOTO_FOLLOWUP_KEYWORDS)

# Pending "what should I do with this file?" choices, keyed by a one-shot
# token embedded in the inline keyboard's callback_data (Telegram callback
# data can't carry a full file path safely/reliably, so it's indirected
# through this dict instead). Consumed (popped) as soon as a button is
# pressed; entries older than the TTL are treated as expired.
_pending_document_uploads: dict[str, dict[str, Any]] = {}
_PENDING_UPLOAD_TTL_SECONDS = 1800

# Same indirection as _pending_document_uploads, for the photo flow's
# "save this to the knowledge base?" follow-up button — holds the raw
# image bytes in memory (nothing written to data/docs) until the user
# opts in, so casual "what's in this photo?" questions don't clutter the
# RAG index.
_pending_photo_uploads: dict[str, dict[str, Any]] = {}

# Telegram's hard limit is 4096 chars per message; leave margin for
# multi-byte counting differences.
_TELEGRAM_MESSAGE_LIMIT = 4000

# Every message otherwise goes through Planner (1 LLM call) and, if no
# tool matches (the common case for greetings/small talk), falls back to
# a second LLM call — doubling latency for requests that never needed a
# tool. Messages matching these keywords skip planning entirely and go
# straight to a single direct LLM reply, mirroring the v3 bot's intent
# classifier (app/telegram/handlers/chat.py).
_CHITCHAT_KEYWORDS = [
    "안녕", "하이", "헬로", "hello", "hi",
    "안녕하세요", "반갑습니다", "만나서 반가워",
    "감사합니다", "고마워", "수고하세요", "수고해",
    "잘 가", "안녕히 가세요", "안녕히 계세요",
    "미안해", "죄송합니다", "sorry", "thanks",
    "어떻게 지내", "오늘 뭐해", "배경이 뭐야", "누구야",
    "자기소개", "너는 누구", "무슨 로봇", "로봇이니",
    "how are you", "what's up", "nice to meet", "thank you",
    "tell me about yourself", "who are you", "you are a bot",
]


def _is_chitchat(question: str) -> bool:
    lower = question.lower().strip()
    return any(k in lower for k in _CHITCHAT_KEYWORDS)


# Shown on /start and /help — kept as a single static message (rather than
# folding into the LLM chitchat fallback) so the feature list stays exact
# and doesn't drift or get abbreviated by the model on a given call.
_WELCOME_MESSAGE = """\
안녕하세요, Hermes입니다 🙂 이런 걸 할 수 있어요:

💬 자유 대화
코딩/개발부터 일상 질문까지 편하게 물어보세요. 음성 메시지로 말하면 인식해서 답하고, 필요하면 음성으로도 답장해요.

📚 지식베이스 (RAG)
문서를 보내면(PDF/DOCX/PPTX/MD/TXT/JSON/YAML/TOML/INI/CFG/PY, 이미지 PNG/JPG) 자동으로 지식베이스에 저장돼요. 사진으로 보내면 답변 후 "지식베이스에 저장" 버튼으로 선택 저장할 수 있고, 스캔된 PDF나 이미지는 OCR로 텍스트를 추출해 인덱싱해요. 저장된 문서 내용은 이후 질문에 자동으로 활용됩니다.

📊 보고서 · PPTX 생성
주제를 주면 웹 검색 기반으로 리서치해서 보고서나 PPTX를 만들어줘요. PDF/DOCX/TXT/MD 문서를 보내고 "PPTX로 변환해줘"라고 하면 그 문서 내용으로도 만들 수 있어요.

📝 문서 요약 · 번역
PDF/DOCX/PPTX/MD/TXT 문서를 보내면서 "요약해줘"나 "영어로 번역해줘"라고 하면 문서 전체 내용을 바탕으로 답해요. 캡션 없이 먼저 보냈다면 뒤이어 "요약해줘"라고 텍스트만 보내도 방금 보낸 문서를 대상으로 처리해요.

⏰ 리마인더
"매일 아침 8시에 뉴스 알려줘"처럼 반복 알림, "내일 3시에 회의 알려줘"처럼 1회성 알림을 만들 수 있어요. 목록 확인/수정/취소도 대화로 요청하면 돼요.

🔍 웹 검색 · Git 조회
최신 정보가 필요한 질문은 웹 검색으로, 이 프로젝트 코드 관련 질문은 git 이력 조회로 답해요.

🖼 이미지 질문
사진을 보내면 무엇인지 설명하거나 궁금한 점에 답해요. "이 사진 좀 더 자세히 봐줘"처럼 이어서 물어보면 방금 보낸 사진을 계속 참고해서 답해요.

🧠 기억하기
"내 생일은 O월 O일이야"처럼 기억해달라고 하면 이후 대화에서 참고해요.

📈 사용량 조회
"이번 달 사용량 알려줘"처럼 물어보면 확인할 수 있어요.

궁금한 거 편하게 물어보세요!\
"""


def make_start_handler(allowed_user_ids: list[str]):
    async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        if not _is_authorized(update, allowed_user_ids):
            await update.message.reply_text("unauthorized")
            return
        await update.message.reply_text(_WELCOME_MESSAGE)

    return handle_start


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


def _final_answer(plan) -> tuple[str | None, list[str]]:
    """Best-effort extraction of the plan's final answer (and any file
    artifacts, e.g. from ReportTool) from its last successful step."""
    for step in reversed(plan.steps):
        for action in reversed(step.actions):
            value = plan.context.get(action.output_key)
            if value:
                meta = plan.context.get(f"{action.output_key}__metadata") or {}
                return str(value), list(meta.get("file_paths") or [])
    return None, []


async def _fallback_reply(llm: LLMProvider, question: str, facts: list[dict] | None = None) -> str:
    system = GENERAL_ASSISTANT_SYSTEM_PROMPT
    if facts:
        lines = "\n".join(f"- {f.get('content', '')}" for f in facts)
        system = f"{system}\n\n사용자에 대해 기억하고 있는 정보 (참고만 하세요):\n{lines}"
    completion = await llm.generate(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]
    )
    return completion.content


# A "Plan has no steps" PlanValidationError means the planner LLM looked at
# the request and couldn't find enough to act on (e.g. a bare "찾아줘" with
# no topic) — not a system failure. Falling back to _fallback_reply for this
# case answers from the model's own knowledge with zero document grounding,
# which looks like a normal (if occasionally wrong) reply and gives no sign
# that RAG/tools were never actually consulted. Ask what they meant instead.
_EMPTY_PLAN_MESSAGE = "Plan has no steps"

_CLARIFY_SYSTEM_PROMPT = (
    "사용자의 요청이 너무 모호해서 어떤 작업을 해야 할지 판단할 수 없습니다. "
    "절대 스스로 답을 지어내거나 추측하지 말고, 무엇을 원하는지 되묻는 "
    "짧은 한국어 질문 한 문장만 답하세요."
)


async def _clarify_reply(llm: LLMProvider, question: str) -> str:
    completion = await llm.generate(
        [
            {"role": "system", "content": _CLARIFY_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
    )
    return completion.content


@contextlib.asynccontextmanager
async def _typing_indicator(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    """Keep Telegram's "typing..." indicator up for the duration of the
    block. Telegram clears it after ~5s, so it has to be re-sent on a
    loop rather than fired once — planning + tool execution can easily
    take longer than that with no other feedback to the user otherwise.
    """
    async def _loop() -> None:
        while True:
            try:
                await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            except Exception:
                pass
            await asyncio.sleep(4)

    task = asyncio.create_task(_loop())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


# Tools that need the caller's chat id — a small local LLM has no reliable
# way to transcribe a real numeric chat id, so it's injected deterministically
# after planning instead (see planner/strategy.py rule 7).
_CHAT_ID_TOOLS = {"schedule_reminder", "list_reminders", "cancel_reminder", "edit_reminder"}


def _inject_chat_id(plan, chat_id: int) -> None:
    """Fill in "chat_id" for any reminder-related action, whether the plan
    just came from a fresh LLM call or the planner's cache."""
    for step in plan.steps:
        for action in step.actions:
            if action.tool_name in _CHAT_ID_TOOLS:
                action.input["chat_id"] = chat_id


def _inject_user_id(plan, user_id: str) -> None:
    """Fill in "user_id" for any remember action — same rationale as
    _inject_chat_id (see planner/strategy.py rule 8)."""
    for step in plan.steps:
        for action in step.actions:
            if action.tool_name == "remember":
                action.input["user_id"] = user_id


async def _answer_and_reply(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    question: str,
    planner: Planner,
    engine: WorkflowEngine,
    llm: LLMProvider,
    memory,
    history_turns: int,
) -> str:
    """Plan + execute (or fast-path) an already-extracted question and reply.

    Shared by the text and voice handlers — voice just transcribes to text
    first and hands the result in here, since telegram.Update objects are
    frozen and can't have update.message.text set after the fact. Returns
    the final answer text so callers (e.g. the voice handler, for TTS) can
    reuse it without re-parsing the reply.
    """
    user = update.effective_user
    user_id = str(user.id) if user else "anonymous"

    history = await memory.get_recent_messages(user_id, limit=history_turns) if memory else []
    facts = await memory.get_facts(user_id) if memory else []
    exec_context = ExecutionContext(
        request_id=str(update.update_id),
        user_id=user_id,
        request=question,
        metadata={"previous_context": history, "user_facts": facts},
    )

    file_paths: list[str] = []
    async with _typing_indicator(context, update.effective_chat.id):
        if _is_chitchat(question):
            answer = await _fallback_reply(llm, question, facts)
        else:
            try:
                plan = await planner.plan(question, exec_context)
                _inject_chat_id(plan, update.effective_chat.id)
                _inject_user_id(plan, user_id)
                plan = await engine.run(plan, exec_context)
                # Use whatever the plan produced even if a later step failed —
                # e.g. a search step can succeed and a redundant "report" step
                # can fail; discarding the search result in that case would
                # throw away a real answer in favor of a tool-less chat reply.
                answer, file_paths = _final_answer(plan)
                if answer is None:
                    raise RuntimeError(plan.error or "plan produced no answer")
                if not plan.is_complete:
                    logger.warning("Plan '%s' partially failed (%s); replying with partial result", plan.id, plan.error)
            except PlanValidationError as exc:
                if str(exc) == _EMPTY_PLAN_MESSAGE:
                    logger.info("Request too vague to plan (%r); asking for clarification", question)
                    answer = await _clarify_reply(llm, question)
                else:
                    logger.warning("Plan validation failed (%s), falling back to direct reply", exc)
                    answer = await _fallback_reply(llm, question, facts)
            except Exception as exc:
                logger.warning("Planned execution failed (%s), falling back to direct reply", exc)
                answer = await _fallback_reply(llm, question, facts)

    await _reply(update, answer)
    for path in file_paths:
        try:
            with open(path, "rb") as f:
                await update.message.reply_document(f, filename=pathlib.Path(path).name)
        except OSError as exc:
            logger.warning("Failed to send generated file '%s': %s", path, exc)

    if memory is not None:
        await memory.save_message(user_id, "user", question)
        await memory.save_message(user_id, "assistant", answer)

    return answer


def make_message_handler(
    planner: Planner,
    engine: WorkflowEngine,
    llm: LLMProvider,
    allowed_user_ids: list[str],
    memory=None,
    history_turns: int = 10,
    report_tool=None,
):
    async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.text:
            return
        if not _is_authorized(update, allowed_user_ids):
            await update.message.reply_text("unauthorized")
            return

        # A document upload with no conversion/summary/translate keyword in
        # its caption doesn't go through the Planner at all (see
        # make_document_handler) — so a follow-up text like "이거 pptx로
        # 만들어줘" has to be caught here instead. Without this, it falls
        # through to the Planner's generate_report, which has no idea a
        # file was uploaded and hallucinates an unrelated topic from the
        # bare request text.
        if report_tool is not None and update.effective_chat is not None:
            upload = _recent_uploads.get(update.effective_chat.id)
            if upload is not None:
                path, uploaded_at = upload
                if time.time() - uploaded_at <= _RECENT_UPLOAD_TTL_SECONDS and path.exists():
                    text = update.message.text
                    if _is_pptx_conversion_request(text):
                        await _convert_document_to_pptx(update, context, path, path.name, report_tool)
                        return
                    if _is_summarize_request(text):
                        await _summarize_document(update, context, path, path.name, report_tool)
                        return
                    if _is_translate_request(text):
                        await _translate_document(
                            update, context, path, path.name, report_tool, _detect_target_language(text)
                        )
                        return

        # Same indirection as the pptx-conversion follow-up above, but for
        # the most recent photo (see _recent_photos) — a plain text message
        # has no image attached to it at all, so the only way "그 사진에서
        # 글자 좀 확대해서 다시 읽어줘" can be answered is by remembering
        # the last photo this chat sent and re-running it through the
        # vision model with the new question as the prompt.
        if update.effective_chat is not None and _is_photo_followup(update.message.text):
            recent_photo = _recent_photos.get(update.effective_chat.id)
            if recent_photo is not None:
                raw, received_at = recent_photo
                if time.time() - received_at <= _RECENT_PHOTO_TTL_SECONDS:
                    vision_fn = getattr(llm, "generate_vision", None)
                    if vision_fn is not None:
                        image_b64 = base64.b64encode(raw).decode("ascii")
                        completion = None
                        async with _typing_indicator(context, update.effective_chat.id):
                            try:
                                completion = await vision_fn(update.message.text, [image_b64])
                            except Exception as exc:
                                logger.warning("Photo follow-up vision request failed: %s", exc)
                        if completion is not None:
                            _recent_photos[update.effective_chat.id] = (raw, time.time())
                            await _reply(update, completion.content)
                            return

        await _answer_and_reply(
            update, context, update.message.text, planner, engine, llm, memory, history_turns
        )

    return handle_message


def make_voice_handler(
    planner: Planner,
    engine: WorkflowEngine,
    llm: LLMProvider,
    allowed_user_ids: list[str],
    memory=None,
    history_turns: int = 10,
):
    async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.voice:
            return
        if not _is_authorized(update, allowed_user_ids):
            await update.message.reply_text("unauthorized")
            return

        tg_file = await update.message.voice.get_file()
        with tempfile.NamedTemporaryFile(suffix=".ogg") as tmp:
            await tg_file.download_to_drive(tmp.name)
            async with _typing_indicator(context, update.effective_chat.id):
                try:
                    question = await transcribe(tmp.name)
                except Exception as exc:
                    logger.warning("Voice transcription failed: %s", exc)
                    await update.message.reply_text(f"음성 인식 실패: {exc}")
                    return

        if not question:
            await update.message.reply_text("음성에서 텍스트를 인식하지 못했습니다.")
            return

        await update.message.reply_text(f"🎤 {question}")
        answer = await _answer_and_reply(update, context, question, planner, engine, llm, memory, history_turns)

        # Voice-in -> voice-out: reply with audio too, best-effort (TTS
        # failure shouldn't hide the text answer already sent above).
        settings = get_settings()
        try:
            audio = await synthesize(answer[: settings.TTS_MAX_CHARS])
            await update.message.reply_voice(voice=audio)
        except Exception as exc:
            logger.warning("TTS synthesis failed: %s", exc)

    return handle_voice


def make_photo_handler(
    llm: LLMProvider, allowed_user_ids: list[str],
    docs_dir=None, index_dir: str | None = None, embedding_model: str | None = None, rag_tool=None,
):
    docs_dir = pathlib.Path(docs_dir) if docs_dir is not None else None

    async def _ingest_photo(message, file_name: str, raw: bytes) -> None:
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / file_name).write_bytes(raw)
        await message.reply_text(f"'{file_name}' 인덱스를 다시 만드는 중 (OCR로 텍스트 추출)...")
        try:
            await asyncio.to_thread(_ingest_document_sync, docs_dir / file_name, index_dir, embedding_model)
        except Exception as exc:
            logger.exception("RAG index rebuild failed after photo upload of '%s'", file_name)
            await message.reply_text(f"인덱스 재생성 실패: {exc}")
            return
        if rag_tool is not None:
            rag_tool.invalidate_cache()
        await message.reply_text(f"'{file_name}' 인덱스 반영 완료. 이제 이 이미지 내용에 대해 질문할 수 있어요.")

    async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.photo:
            return
        if not _is_authorized(update, allowed_user_ids):
            await update.message.reply_text("unauthorized")
            return

        # Not part of the LLMProvider interface (only Ollama has a local
        # vision model available) — feature-detect instead.
        vision_fn = getattr(llm, "generate_vision", None)
        if vision_fn is None:
            await update.message.reply_text("현재 LLM 프로바이더는 이미지 질문을 지원하지 않습니다.")
            return

        prompt = update.message.caption or "이 이미지에 대해 설명해줘."
        photo = update.message.photo[-1]  # last = highest resolution
        tg_file = await photo.get_file()
        raw = bytes(await tg_file.download_as_bytearray())
        image_b64 = base64.b64encode(raw).decode("ascii")

        if update.effective_chat is not None:
            _recent_photos[update.effective_chat.id] = (raw, time.time())

        async with _typing_indicator(context, update.effective_chat.id):
            try:
                completion = await vision_fn(prompt, [image_b64])
            except Exception as exc:
                logger.warning("Vision request failed: %s", exc)
                await update.message.reply_text(f"이미지 분석 실패: {exc}")
                return

        await _reply(update, completion.content)

        # Telegram compresses "photo" uploads (lossy, capped resolution),
        # which can hurt OCR accuracy on dense text — still offered since
        # most users share images this way rather than as an uncompressed
        # file, but docs_dir/rag_tool must have been wired in for it to work.
        if docs_dir is not None:
            token = uuid.uuid4().hex
            file_name = f"telegram_photo_{tg_file.file_unique_id}.jpg"
            _pending_photo_uploads[token] = {
                "bytes": raw, "file_name": file_name, "created_at": time.time(),
            }
            keyboard = InlineKeyboardMarkup([[
                InlineKeyboardButton("📚 지식베이스에 저장", callback_data=f"photosave|{token}"),
            ]])
            await update.message.reply_text(
                "이 이미지를 지식베이스에도 저장할까요? (OCR로 텍스트 추출)", reply_markup=keyboard,
            )

    async def handle_photo_save_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None or not query.data or not query.data.startswith("photosave|"):
            return
        if not _is_authorized(update, allowed_user_ids):
            await query.answer("unauthorized", show_alert=True)
            return

        await query.answer()
        _, token = query.data.split("|", 1)
        pending = _pending_photo_uploads.pop(token, None)
        await query.edit_message_reply_markup(reply_markup=None)
        if pending is None or time.time() - pending["created_at"] > _PENDING_UPLOAD_TTL_SECONDS:
            await query.message.reply_text("이미 처리되었거나 만료된 요청입니다. 사진을 다시 보내주세요.")
            return

        await _ingest_photo(query.message, pending["file_name"], pending["bytes"])

    return handle_photo, handle_photo_save_choice


def _ingest_document_sync(file_path: pathlib.Path, index_dir: str, embedding_model: str | None) -> None:
    """Embed and add *file_path* to the FAISS index in place.

    Uses add_to_index() rather than a full build_index() over all of
    data/docs — otherwise every single upload would re-embed the entire
    corpus, which gets slower (and more expensive) as it grows.
    """
    from app.rag.pipeline import add_to_index

    add_to_index(paths=[str(file_path)], index_dir=index_dir, embedding_model=embedding_model)


async def _convert_document_to_pptx(
    update: Update, context: ContextTypes.DEFAULT_TYPE, local_path: pathlib.Path, display_name: str, report_tool
) -> None:
    # effective_message (not .message) since this is also called from the
    # inline-button callback flow, where update.message is None and the
    # original message only reachable via update.callback_query.message.
    message = update.effective_message
    await message.reply_text(f"'{display_name}' 문서를 PPTX로 변환하는 중입니다 (몇 분 정도 걸릴 수 있어요)...")

    async with _typing_indicator(context, update.effective_chat.id):
        result = await report_tool.build_from_document(local_path, fmt="pptx")

    if not result.success:
        await message.reply_text(f"변환 실패: {result.error}")
        return

    for path in result.metadata.get("file_paths", []):
        try:
            with open(path, "rb") as f:
                await message.reply_document(f, filename=pathlib.Path(path).name)
        except OSError as exc:
            logger.warning("Failed to send converted file '%s': %s", path, exc)

    await message.reply_text(result.output)


async def _summarize_document(
    update: Update, context: ContextTypes.DEFAULT_TYPE, local_path: pathlib.Path, display_name: str, report_tool
) -> None:
    message = update.effective_message
    await message.reply_text(f"'{display_name}' 문서를 요약하는 중입니다...")

    async with _typing_indicator(context, update.effective_chat.id):
        result = await report_tool.summarize_document(local_path)

    if not result.success:
        await message.reply_text(f"요약 실패: {result.error}")
        return

    await _reply(update, result.output)


async def _translate_document(
    update: Update, context: ContextTypes.DEFAULT_TYPE, local_path: pathlib.Path, display_name: str, report_tool,
    target_language: str,
) -> None:
    message = update.effective_message
    await message.reply_text(f"'{display_name}' 문서를 {target_language}로 번역하는 중입니다...")

    async with _typing_indicator(context, update.effective_chat.id):
        result = await report_tool.translate_document(local_path, target_language=target_language)

    if not result.success:
        await message.reply_text(f"번역 실패: {result.error}")
        return

    await _reply(update, result.output)


def make_document_handler(
    docs_dir, index_dir: str, embedding_model: str | None, allowed_user_ids: list[str],
    rag_tool=None, report_tool=None,
):
    docs_dir = pathlib.Path(docs_dir)

    async def _ingest_rag(message, file_name: str) -> None:
        await message.reply_text(f"'{file_name}' 인덱스를 다시 만드는 중...")
        try:
            await asyncio.to_thread(_ingest_document_sync, docs_dir / file_name, index_dir, embedding_model)
        except Exception as exc:
            logger.exception("RAG index rebuild failed after upload of '%s'", file_name)
            await message.reply_text(f"인덱스 재생성 실패: {exc}")
            return
        if rag_tool is not None:
            rag_tool.invalidate_cache()
        await message.reply_text(f"'{file_name}' 인덱스 반영 완료. 이제 이 문서에 대해 질문할 수 있어요.")

    async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.document:
            return
        if not _is_authorized(update, allowed_user_ids):
            await update.message.reply_text("unauthorized")
            return

        doc = update.message.document
        file_name = doc.file_name or f"upload_{doc.file_unique_id}"
        suffix = pathlib.Path(file_name).suffix.lower()
        caption = update.message.caption or ""
        can_convert = report_tool is not None and suffix in CONVERTIBLE_DOC_SUFFIXES
        can_rag = suffix in _SUPPORTED_DOC_SUFFIXES

        if can_convert and _is_pptx_conversion_request(caption):
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_path = pathlib.Path(tmp_dir) / file_name
                tg_file = await doc.get_file()
                await tg_file.download_to_drive(str(local_path))
                await _convert_document_to_pptx(update, context, local_path, file_name, report_tool)
            return

        if can_convert and _is_summarize_request(caption):
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_path = pathlib.Path(tmp_dir) / file_name
                tg_file = await doc.get_file()
                await tg_file.download_to_drive(str(local_path))
                await _summarize_document(update, context, local_path, file_name, report_tool)
            return

        if can_convert and _is_translate_request(caption):
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_path = pathlib.Path(tmp_dir) / file_name
                tg_file = await doc.get_file()
                await tg_file.download_to_drive(str(local_path))
                await _translate_document(
                    update, context, local_path, file_name, report_tool, _detect_target_language(caption)
                )
            return

        if not can_rag and not can_convert:
            await update.message.reply_text(
                f"지원하지 않는 파일 형식입니다: {suffix or '(확장자 없음)'} "
                f"(지원: {', '.join(sorted(_SUPPORTED_DOC_SUFFIXES | CONVERTIBLE_DOC_SUFFIXES))})"
            )
            return

        docs_dir.mkdir(parents=True, exist_ok=True)
        dest = docs_dir / file_name
        tg_file = await doc.get_file()
        await tg_file.download_to_drive(str(dest))

        # Remembered so a *later* text message ("이거 pptx로 만들어줘") can
        # still trigger conversion even after a choice was already made
        # below — see _recent_uploads.
        if update.effective_chat is not None and can_convert:
            _recent_uploads[update.effective_chat.id] = (dest, time.time())

        if can_rag and can_convert:
            # Ambiguous format (pdf/md/txt) — ask instead of guessing which
            # one the user wants, or silently doing both.
            token = uuid.uuid4().hex
            _pending_document_uploads[token] = {
                "path": dest, "file_name": file_name, "created_at": time.time(),
            }
            keyboard = InlineKeyboardMarkup([[
                InlineKeyboardButton("📚 지식베이스에 저장", callback_data=f"docupload|rag|{token}"),
                InlineKeyboardButton("📊 PPTX로 변환", callback_data=f"docupload|pptx|{token}"),
            ]])
            await update.message.reply_text(f"'{file_name}' 저장 완료. 어떻게 할까요?", reply_markup=keyboard)
            return

        if can_convert:
            # Convert-only format (e.g. .docx/.pptx) — RAG can't index it,
            # so the only real choice is whether to spend the several
            # minutes converting it now.
            token = uuid.uuid4().hex
            _pending_document_uploads[token] = {
                "path": dest, "file_name": file_name, "created_at": time.time(),
            }
            keyboard = InlineKeyboardMarkup([[
                InlineKeyboardButton("📊 PPTX로 변환", callback_data=f"docupload|pptx|{token}"),
            ]])
            await update.message.reply_text(
                f"'{file_name}' 저장 완료. 이 형식은 지식베이스 인덱싱은 지원하지 않아요. PPTX로 변환할까요?",
                reply_markup=keyboard,
            )
            return

        # RAG-only format (.py/.json/...) — no ambiguity, index directly.
        await _ingest_rag(update.message, file_name)

    async def handle_document_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None or not query.data or not query.data.startswith("docupload|"):
            return
        if not _is_authorized(update, allowed_user_ids):
            await query.answer("unauthorized", show_alert=True)
            return

        await query.answer()
        _, action, token = query.data.split("|", 2)
        pending = _pending_document_uploads.pop(token, None)
        if pending is None or time.time() - pending["created_at"] > _PENDING_UPLOAD_TTL_SECONDS:
            await query.edit_message_text("이미 처리되었거나 만료된 요청입니다. 파일을 다시 보내주세요.")
            return

        path: pathlib.Path = pending["path"]
        file_name: str = pending["file_name"]
        await query.edit_message_reply_markup(reply_markup=None)

        if action == "rag":
            await _ingest_rag(query.message, file_name)
        elif action == "pptx":
            await _convert_document_to_pptx(update, context, path, file_name, report_tool)

    return handle_document, handle_document_choice


async def _on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global fallback for exceptions raised inside handlers.

    Without this, python-telegram-bot just logs "No error handlers are
    registered" and drops the update — the user sees no reply at all and
    has no way to tell a bug from the bot simply being slow (this is how a
    keep_alive regression that 400'd every planning call went unnoticed:
    silence, not an error message).
    """
    logger.error("Unhandled error while processing update: %s", update, exc_info=context.error)
    if isinstance(update, Update) and update.effective_message is not None:
        try:
            await update.effective_message.reply_text("처리 중 오류가 발생했어요. 잠시 후 다시 시도해주세요.")
        except Exception:
            logger.exception("Failed to notify user about the earlier error")


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
    start_handler = make_start_handler(allowed_user_ids)
    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(CommandHandler("help", start_handler))
    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            make_message_handler(
                planner, engine, llm, allowed_user_ids, memory=memory,
                history_turns=settings.CONVERSATION_HISTORY_TURNS,
                report_tool=tool_registry.get_tool("generate_report"),
            ),
        )
    )
    handle_photo, handle_photo_save_choice = make_photo_handler(
        llm, allowed_user_ids,
        docs_dir=RAGTool._default_docs_dir(),
        index_dir=settings.RAG_INDEX_DIR,
        embedding_model=settings.RAG_EMBEDDING_MODEL,
        rag_tool=tool_registry.get_tool("rag"),
    )
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(CallbackQueryHandler(handle_photo_save_choice, pattern=r"^photosave\|"))
    application.add_handler(
        MessageHandler(
            filters.VOICE,
            make_voice_handler(
                planner, engine, llm, allowed_user_ids, memory=memory,
                history_turns=settings.CONVERSATION_HISTORY_TURNS,
            ),
        )
    )
    handle_document, handle_document_choice = make_document_handler(
        RAGTool._default_docs_dir(),
        settings.RAG_INDEX_DIR,
        settings.RAG_EMBEDDING_MODEL,
        allowed_user_ids,
        rag_tool=tool_registry.get_tool("rag"),
        report_tool=tool_registry.get_tool("generate_report"),
    )
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(CallbackQueryHandler(handle_document_choice, pattern=r"^docupload\|"))
    application.add_error_handler(_on_error)
    return application
