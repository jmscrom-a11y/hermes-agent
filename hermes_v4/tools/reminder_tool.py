"""Reminder tool for Hermes V4.

Schedules a one-time or recurring daily message via Telegram's JobQueue
(python-telegram-bot's built-in APScheduler integration, sharing the bot's
own event loop — no separate scheduler process to wire up). The reminder's
content is generated fresh from the stored prompt each time it fires, so
e.g. "매일 아침 계란요리 레시피" gets a different recipe every day rather
than a cached one-off answer.

`chat_id` is never supplied by the planner — a small local LLM has no
reliable way to know the caller's numeric chat id. The Telegram layer
(hermes_v4/telegram/bot.py) injects it into the action input after
planning, before execution.
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
from datetime import datetime, time as dt_time
from typing import Any
from zoneinfo import ZoneInfo

from hermes_v4.config.settings import get_settings
from hermes_v4.core.base import Tool, ToolResult
from hermes_v4.llm.provider import LLMProvider
from hermes_v4.prompts import GENERAL_ASSISTANT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_TIME_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_WEATHER_KEYWORDS = ("날씨",)
# A reminder whose prompt mentions one of these gets grounded in a real web
# search instead of an LLM guess with no live data — same rationale as
# weather (see _generate_briefing).
_NEWS_KEYWORDS = ("뉴스", "브리핑", "헤드라인", "소식", "이슈")
# Prompts that are just the bare keyword have no specific subject to search
# for, so they fan out into BRIEFING_TOPICS instead of searching that
# uninformative literal text.
_GENERIC_NEWS_PROMPTS = {
    "뉴스", "오늘 뉴스", "뉴스 브리핑", "브리핑", "오늘 브리핑", "아침 브리핑",
    "일일 브리핑", "데일리 브리핑", "오늘의 브리핑",
}


class ReminderTool(Tool):
    """Schedules a one-time or recurring daily reminder message."""

    name = "schedule_reminder"
    description = (
        "사용자가 '매일 아침 8시에 ~ 보내줘'처럼 반복되는 알림이나, '내일 3시에 "
        "~ 알려줘', '이번 주 금요일에 ~ 알려줘'처럼 특정 날짜의 1회성 알림을 "
        "요청했을 때 사용하세요. 그 자리에서 바로 답하면 되는 질문에는 사용하지 "
        "마세요. 지정한 시간에 'prompt'를 바탕으로 매번 새 내용을 생성해서 "
        "보냅니다. 'prompt'에는 반복 주기나 시간·날짜 표현('매일', '아침', "
        "'8시에', '내일' 등)을 절대 넣지 마세요 — 그런 표현은 'time'/'date'에만 "
        "담고, 'prompt'에는 매번 생성할 내용의 주제만 짧게 넣으세요 "
        "(예: '계란요리 레시피')."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "time": {"type": "string", "description": "HH:MM, 24-hour, local time"},
            "date": {
                "type": "string",
                "description": (
                    "YYYY-MM-DD. 특정 날짜에 한 번만 보내는 1회성 알림일 때만 채우세요 "
                    "(현재 날짜를 기준으로 계산). 매일 반복되는 알림이면 비워두세요."
                ),
            },
            "prompt": {
                "type": "string",
                "description": (
                    "매번 생성할 내용의 주제만 (예: '계란요리 레시피'). "
                    "'매일'/'~시에'/'내일' 같은 시간·반복·날짜 표현은 넣지 말 것."
                ),
            },
        },
        "required": ["time", "prompt"],
    }
    output_schema = {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
    }

    def __init__(
        self,
        llm: LLMProvider,
        web_search_tool: Tool | None = None,
        store_path: str | pathlib.Path | None = None,
    ) -> None:
        settings = get_settings()
        self.llm = llm
        self.web_search_tool = web_search_tool
        self.job_queue = None
        self._store_path = pathlib.Path(store_path or settings.REMINDERS_STORE_PATH)
        self._tz = ZoneInfo(settings.REMINDER_TIMEZONE)
        self._weather_locations = settings.WEATHER_LOCATIONS
        self._briefing_topics = settings.BRIEFING_TOPICS

    def bind_job_queue(self, job_queue) -> None:
        """Attach the running Application's JobQueue.

        This tool is constructed in build_tool_registry(), before the
        Telegram Application (and its JobQueue) exists — main() calls this
        once both are built.
        """
        self.job_queue = job_queue

    async def validate_input(self, input: dict[str, Any]) -> list[str]:
        errors = []
        if not _TIME_RE.match(str(input.get("time", ""))):
            errors.append("'time' must be HH:MM (24-hour)")
        date_str = str(input.get("date") or "").strip()
        if date_str and not _DATE_RE.match(date_str):
            errors.append("'date' must be YYYY-MM-DD")
        if not str(input.get("prompt", "")).strip():
            errors.append("'prompt' is required and must be a non-empty string")
        if not input.get("chat_id"):
            errors.append("'chat_id' missing (internal error — the caller must inject it)")
        return errors

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        errors = await self.validate_input(input)
        if errors:
            return ToolResult.fail("; ".join(errors))
        if self.job_queue is None:
            return ToolResult.fail("스케줄러가 아직 준비되지 않았습니다. 잠시 후 다시 시도해주세요.")

        time_str = str(input["time"])
        date_str = str(input.get("date") or "").strip() or None
        prompt = str(input["prompt"]).strip()
        chat_id = input["chat_id"]
        hour, minute = (int(p) for p in time_str.split(":"))

        job_name = f"reminder:{chat_id}:{date_str or 'daily'}:{time_str}:{prompt}"
        for job in self.job_queue.get_jobs_by_name(job_name):
            job.schedule_removal()

        if date_str:
            year, month, day = (int(p) for p in date_str.split("-"))
            when = datetime(year, month, day, hour, minute, tzinfo=self._tz)
            if when <= datetime.now(self._tz):
                return ToolResult.fail(f"{date_str} {time_str}은(는) 이미 지난 시간입니다.")
            self.job_queue.run_once(
                self._fire, when=when, chat_id=chat_id, name=job_name,
                data={"prompt": prompt, "date": date_str},
            )
            self._persist(chat_id, time_str, prompt, date_str)
            return ToolResult.ok(output=f"{date_str} {time_str}에 '{prompt}' 내용으로 한 번 알려드릴게요.")

        self.job_queue.run_daily(
            self._fire,
            time=dt_time(hour=hour, minute=minute, tzinfo=self._tz),
            chat_id=chat_id,
            name=job_name,
            data={"prompt": prompt, "date": None},
        )
        self._persist(chat_id, time_str, prompt, None)
        return ToolResult.ok(output=f"매일 {time_str}에 '{prompt}' 내용으로 알려드릴게요.")

    async def _fire(self, context) -> None:
        job = context.job
        prompt = job.data["prompt"]
        date_str = job.data.get("date")
        try:
            text = await self._generate_content(prompt)
        except Exception as exc:
            logger.warning("Reminder content generation failed for '%s': %s", prompt, exc)
            text = f"(알림 내용 생성 실패: {exc})\n요청: {prompt}"
        await context.bot.send_message(chat_id=job.chat_id, text=text)

        # One-time reminders don't get re-registered on restart — remove
        # their persisted entry now that they've fired (a recurring
        # reminder's entry stays so restore_all() can re-register it).
        if date_str:
            self._remove_persisted(job.chat_id, prompt, date_str)

    async def _generate_content(self, prompt: str) -> str:
        if any(k in prompt for k in _WEATHER_KEYWORDS) and self.web_search_tool is not None:
            return await self._generate_weather(prompt)
        if any(k in prompt for k in _NEWS_KEYWORDS) and self.web_search_tool is not None:
            return await self._generate_briefing(prompt)
        completion = await self.llm.generate(
            [
                {"role": "system", "content": GENERAL_ASSISTANT_SYSTEM_PROMPT},
                {"role": "user", "content": f"다음 요청에 대한 내용을 간결하게 작성해줘: {prompt}"},
            ]
        )
        return completion.content

    async def _generate_weather(self, prompt: str) -> str:
        """Grounds a weather reminder in real search results per configured
        location — the plain LLM path has no live data or location context,
        so left alone it just says it doesn't know where the user means."""
        sections = []
        for city in self._weather_locations:
            try:
                result = await self.web_search_tool.execute({"query": f"{city} 오늘 날씨"})
            except Exception:
                logger.exception("Weather search failed for %s", city)
                continue
            if result.success and result.output:
                sections.append(f"### {city}\n{result.output}")
        if not sections:
            return "오늘 날씨 정보를 가져오지 못했어요."

        synth_prompt = (
            "다음은 지역별 오늘 날씨 웹 검색 원본 결과입니다. 지역마다 기온·강수확률·"
            "특이사항을 2~3줄로 간결하게 정리해서 알려주세요. 검색 결과에 없는 수치는 "
            "지어내지 말고, 없으면 없다고 하세요.\n\n" + "\n\n".join(sections)
        )
        completion = await self.llm.generate(
            [
                {"role": "system", "content": GENERAL_ASSISTANT_SYSTEM_PROMPT},
                {"role": "user", "content": synth_prompt},
            ]
        )
        return completion.content

    async def _generate_briefing(self, prompt: str) -> str:
        """Grounds a news/briefing reminder in real search results — same
        rationale as _generate_weather, generalized to any news topic. A
        bare "브리핑"/"뉴스" prompt has no specific subject to search for,
        so it fans out into BRIEFING_TOPICS instead of searching that
        uninformative literal text.
        """
        normalized = prompt.strip()
        queries = (
            [f"오늘 {topic}" for topic in self._briefing_topics]
            if normalized in _GENERIC_NEWS_PROMPTS
            else [f"{prompt} 오늘"]
        )

        sections = []
        for query in queries:
            try:
                result = await self.web_search_tool.execute({"query": query})
            except Exception:
                logger.exception("Briefing search failed for '%s'", query)
                continue
            if result.success and result.output:
                sections.append(f"### {query}\n{result.output}")
        if not sections:
            return "오늘 소식을 가져오지 못했어요."

        synth_prompt = (
            "다음은 웹 검색 원본 결과입니다. 항목별로 핵심만 2~3줄 불렛으로 간결하게 "
            "정리해서 브리핑해주세요. 검색 결과에 없는 내용은 지어내지 마세요.\n\n"
            + "\n\n".join(sections)
        )
        completion = await self.llm.generate(
            [
                {"role": "system", "content": GENERAL_ASSISTANT_SYSTEM_PROMPT},
                {"role": "user", "content": synth_prompt},
            ]
        )
        return completion.content

    def _persist(self, chat_id: Any, time_str: str, prompt: str, date_str: str | None) -> None:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        entries = self._load_all()
        entries = [
            e for e in entries
            if not (
                e["chat_id"] == chat_id and e["time"] == time_str and e["prompt"] == prompt
                and e.get("date") == date_str
            )
        ]
        entries.append({"chat_id": chat_id, "time": time_str, "prompt": prompt, "date": date_str})
        self._store_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2))

    def _remove_persisted(self, chat_id: Any, prompt: str, date_str: str | None) -> None:
        entries = self._load_all()
        entries = [
            e for e in entries
            if not (e["chat_id"] == chat_id and e["prompt"] == prompt and e.get("date") == date_str)
        ]
        self._store_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2))

    def _load_all(self) -> list[dict[str, Any]]:
        if not self._store_path.exists():
            return []
        try:
            return json.loads(self._store_path.read_text())
        except (json.JSONDecodeError, OSError):
            return []

    async def restore_all(self) -> None:
        """Re-register every previously scheduled reminder.

        JobQueue is in-memory only and forgets everything on restart —
        call once at startup, after bind_job_queue(). A one-time reminder
        whose date/time already passed while the bot was down is dropped
        (and its persisted entry removed) instead of firing late.
        """
        for entry in self._load_all():
            date_str = entry.get("date")
            if date_str:
                hour, minute = (int(p) for p in entry["time"].split(":"))
                year, month, day = (int(p) for p in date_str.split("-"))
                when = datetime(year, month, day, hour, minute, tzinfo=self._tz)
                if when <= datetime.now(self._tz):
                    logger.info("Dropping past one-time reminder %s (was due %s)", entry, when)
                    self._remove_persisted(entry["chat_id"], entry["prompt"], date_str)
                    continue
            result = await self.execute(entry)
            if not result.success:
                logger.warning("Failed to restore reminder %s: %s", entry, result.error)

    def list_for_chat(self, chat_id: Any) -> list[dict[str, Any]]:
        """Return every persisted reminder for one chat — the persisted
        store is the source of truth for "currently scheduled" since
        execute()/cancel() always keep it in sync with the JobQueue."""
        return [e for e in self._load_all() if e["chat_id"] == chat_id]

    def cancel(self, chat_id: Any, query: str) -> ToolResult:
        """Cancel a reminder identified by a free-text query (topic or
        time) — a small local LLM can't produce the exact internal job
        name, so this fuzzy-matches against the chat's own reminders
        instead (see ListRemindersTool/CancelReminderTool)."""
        entries = self.list_for_chat(chat_id)
        matches = [e for e in entries if self._matches(e, query)]
        if not matches:
            return ToolResult.fail(
                "해당하는 알림을 찾지 못했어요. list_reminders로 먼저 목록을 확인해보세요."
            )
        if len(matches) > 1:
            desc = "; ".join(self._describe(e) for e in matches)
            return ToolResult.fail(f"여러 개가 일치해서 하나를 특정할 수 없어요: {desc}")

        entry = matches[0]
        date_str = entry.get("date")
        job_name = f"reminder:{chat_id}:{date_str or 'daily'}:{entry['time']}:{entry['prompt']}"
        if self.job_queue is not None:
            for job in self.job_queue.get_jobs_by_name(job_name):
                job.schedule_removal()
        self._remove_persisted(chat_id, entry["prompt"], date_str)
        return ToolResult.ok(output=f"'{self._describe(entry)}' 알림을 취소했어요.")

    async def edit(
        self,
        chat_id: Any,
        query: str,
        new_time: str | None = None,
        new_date: str | None = None,
        new_prompt: str | None = None,
    ) -> ToolResult:
        """Reschedule a reminder identified by a free-text query, keeping
        whichever fields the caller didn't ask to change. Implemented as
        cancel-then-recreate rather than mutating the JobQueue job in place
        — run_once/run_daily jobs don't support rescheduling, and this way
        the persisted store and JobQueue can't drift out of sync."""
        entries = self.list_for_chat(chat_id)
        matches = [e for e in entries if self._matches(e, query)]
        if not matches:
            return ToolResult.fail(
                "해당하는 알림을 찾지 못했어요. list_reminders로 먼저 목록을 확인해보세요."
            )
        if len(matches) > 1:
            desc = "; ".join(self._describe(e) for e in matches)
            return ToolResult.fail(f"여러 개가 일치해서 하나를 특정할 수 없어요: {desc}")

        entry = matches[0]
        time_str = str(new_time).strip() if new_time else entry["time"]
        if not _TIME_RE.match(time_str):
            return ToolResult.fail("'time' must be HH:MM (24-hour)")
        date_str = str(new_date).strip() if new_date is not None else entry.get("date")
        date_str = date_str or None
        if date_str and not _DATE_RE.match(date_str):
            return ToolResult.fail("'date' must be YYYY-MM-DD")
        prompt = str(new_prompt).strip() if new_prompt else entry["prompt"]

        old_date_str = entry.get("date")
        old_job_name = f"reminder:{chat_id}:{old_date_str or 'daily'}:{entry['time']}:{entry['prompt']}"
        if self.job_queue is not None:
            for job in self.job_queue.get_jobs_by_name(old_job_name):
                job.schedule_removal()
        self._remove_persisted(chat_id, entry["prompt"], old_date_str)

        result = await self.execute(
            {"chat_id": chat_id, "time": time_str, "date": date_str, "prompt": prompt}
        )
        if not result.success:
            return result
        return ToolResult.ok(output=f"알림을 '{self._describe({'time': time_str, 'date': date_str, 'prompt': prompt})}'(으)로 수정했어요.")

    @staticmethod
    def _matches(entry: dict[str, Any], query: str) -> bool:
        q = query.strip().lower()
        if not q:
            return False
        prompt = str(entry.get("prompt", "")).lower()
        time_str = str(entry.get("time", ""))
        return q in prompt or prompt in q or q == time_str or q.replace(" ", "") == time_str

    @staticmethod
    def _describe(entry: dict[str, Any]) -> str:
        date_str = entry.get("date")
        schedule = f"{date_str} {entry['time']} (1회성)" if date_str else f"매일 {entry['time']} (반복)"
        return f"{schedule} - {entry['prompt']}"


class ListRemindersTool(Tool):
    """Lists every reminder currently scheduled for the requesting chat."""

    name = "list_reminders"
    description = (
        "사용자가 '내 알림 목록 보여줘', '예약된 리마인더 뭐 있어?'처럼 현재 "
        "예약된 리마인더를 확인하고 싶어할 때 사용하세요."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}
    output_schema = {"type": "object", "properties": {"summary": {"type": "string"}}}

    def __init__(self, reminder_tool: ReminderTool) -> None:
        self.reminder_tool = reminder_tool

    async def validate_input(self, input: dict[str, Any]) -> list[str]:
        errors = []
        if not input.get("chat_id"):
            errors.append("'chat_id' missing (internal error — the caller must inject it)")
        return errors

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        errors = await self.validate_input(input)
        if errors:
            return ToolResult.fail("; ".join(errors))

        entries = self.reminder_tool.list_for_chat(input["chat_id"])
        if not entries:
            return ToolResult.ok(output="예약된 리마인더가 없어요.")
        lines = [f"- {self.reminder_tool._describe(e)}" for e in entries]
        return ToolResult.ok(output="예약된 리마인더:\n" + "\n".join(lines))


class CancelReminderTool(Tool):
    """Cancels a previously scheduled reminder identified by a free-text query."""

    name = "cancel_reminder"
    description = (
        "사용자가 '~ 알림 취소해줘', '~ 리마인더 지워줘'처럼 이미 예약된 리마인더를 "
        "취소하고 싶어할 때 사용하세요. 'query'에는 취소할 리마인더를 식별할 수 있는 "
        "주제나 시간을 넣으세요 (예: '계란요리', '07:00')."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "취소할 리마인더의 주제나 시간 (예: '계란요리', '07:00')",
            },
        },
        "required": ["query"],
    }
    output_schema = {"type": "object", "properties": {"summary": {"type": "string"}}}

    def __init__(self, reminder_tool: ReminderTool) -> None:
        self.reminder_tool = reminder_tool

    async def validate_input(self, input: dict[str, Any]) -> list[str]:
        errors = []
        if not str(input.get("query", "")).strip():
            errors.append("'query' is required and must be a non-empty string")
        if not input.get("chat_id"):
            errors.append("'chat_id' missing (internal error — the caller must inject it)")
        return errors

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        errors = await self.validate_input(input)
        if errors:
            return ToolResult.fail("; ".join(errors))
        return self.reminder_tool.cancel(input["chat_id"], str(input["query"]))


class EditReminderTool(Tool):
    """Reschedules an already-scheduled reminder identified by a free-text query."""

    name = "edit_reminder"
    description = (
        "사용자가 '알림 시간 8시로 바꿔줘', '계란요리 알림 9시로 변경해줘', "
        "'그 리마인더 내용 뉴스로 바꿔줘'처럼 이미 예약된 리마인더의 시간·날짜·내용을 "
        "수정하고 싶어할 때 사용하세요. 새로운 알림을 만드는 게 아니라 기존 알림을 "
        "찾아서 바꾸는 것입니다. 'query'에는 수정할 리마인더를 식별할 수 있는 기존 "
        "주제나 시간을 넣고, 'new_time'/'new_date'/'new_prompt' 중 실제로 바뀌는 "
        "값만 채우세요 (바뀌지 않는 값은 비워두면 기존 값이 유지됩니다)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "수정할 리마인더를 식별할 기존 주제나 시간 (예: '계란요리', '07:00')",
            },
            "new_time": {"type": "string", "description": "HH:MM, 24-hour. 시간을 바꿀 때만 채우세요."},
            "new_date": {
                "type": "string",
                "description": "YYYY-MM-DD. 1회성 알림의 날짜를 바꿀 때만 채우세요.",
            },
            "new_prompt": {
                "type": "string",
                "description": "매번 생성할 내용의 새 주제. 내용을 바꿀 때만 채우세요.",
            },
        },
        "required": ["query"],
    }
    output_schema = {"type": "object", "properties": {"summary": {"type": "string"}}}

    def __init__(self, reminder_tool: ReminderTool) -> None:
        self.reminder_tool = reminder_tool

    async def validate_input(self, input: dict[str, Any]) -> list[str]:
        errors = []
        if not str(input.get("query", "")).strip():
            errors.append("'query' is required and must be a non-empty string")
        if not input.get("chat_id"):
            errors.append("'chat_id' missing (internal error — the caller must inject it)")
        return errors

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        errors = await self.validate_input(input)
        if errors:
            return ToolResult.fail("; ".join(errors))
        return await self.reminder_tool.edit(
            input["chat_id"],
            str(input["query"]),
            new_time=input.get("new_time") or None,
            new_date=input.get("new_date") or None,
            new_prompt=input.get("new_prompt") or None,
        )
