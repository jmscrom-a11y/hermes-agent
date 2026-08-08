"""Report generation tool for Hermes V4.

Given a topic, asks the LLM for a structured outline (title + sections
+ bullets) and renders it as a PPTX and/or PDF file. Files are handed
back via ToolResult.metadata["file_paths"]; the Telegram layer sends
them as document attachments.
"""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import re
import time
from typing import Any

from hermes_v4.config.settings import get_settings
from hermes_v4.core.base import Tool, ToolResult
from hermes_v4.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

_VALID_FORMATS = {"pptx", "pdf", "both"}


def _slugify(text: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip()
    slug = re.sub(r"\s+", "_", slug)
    return slug[:60] or "report"


class ReportTool(Tool):
    """Generates a PPTX/PDF report for a given topic."""

    name = "generate_report"
    description = (
        "주제를 주면 최신 웹 검색 결과로 리서치한 뒤 발표자료(PPTX) 또는 문서(PDF) 보고서를 생성합니다. "
        "리서치를 직접 수행하므로 별도의 web_search 단계를 먼저 호출할 필요는 없습니다. "
        "'보고서 만들어줘', 'PPT로 정리해줘', 'PDF로 만들어줘' 같은 요청에 사용하세요."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "format": {"type": "string", "enum": sorted(_VALID_FORMATS)},
        },
        "required": ["topic"],
    }
    output_schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "file_paths": {"type": "array", "items": {"type": "string"}},
        },
    }

    def __init__(
        self,
        llm: LLMProvider,
        web_search_tool: Tool | None = None,
        claude_code_tool: Tool | None = None,
        output_dir: str | pathlib.Path | None = None,
        model: str | None = None,
    ) -> None:
        settings = get_settings()
        self.llm = llm
        self.web_search_tool = web_search_tool
        self.claude_code_tool = claude_code_tool
        self.output_dir = pathlib.Path(output_dir or settings.REPORTS_DIR)
        self.num_sections = settings.REPORT_SECTIONS
        self.bullets_per_section = settings.REPORT_BULLETS_PER_SECTION
        # Report generation is a one-off, latency-tolerant request (unlike
        # interactive chat), so it's worth spending a bigger/slower local
        # model here for noticeably richer output. Empty = use the LLM
        # provider's own default model.
        self.model = model if model is not None else settings.REPORT_LLM_MODEL or None

    async def validate_input(self, input: dict[str, Any]) -> list[str]:
        errors = []
        topic = input.get("topic")
        if not topic or not str(topic).strip():
            errors.append("'topic' is required and must be a non-empty string")
        fmt = input.get("format", "both")
        if fmt not in _VALID_FORMATS:
            errors.append(f"'format' must be one of {sorted(_VALID_FORMATS)}")
        return errors

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        errors = await self.validate_input(input)
        if errors:
            return ToolResult.fail("; ".join(errors))

        topic = str(input["topic"]).strip()
        fmt = input.get("format", "both")

        start = time.monotonic()
        research = await self._research(topic)
        try:
            outline = await self._generate_outline(topic, research)
        except Exception as exc:
            return ToolResult.fail(f"Failed to generate report outline: {exc}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        slug = _slugify(outline.get("title") or topic)
        file_paths = []
        try:
            if fmt in ("pptx", "both"):
                if self.claude_code_tool is not None:
                    file_paths.append(await self._build_pptx_via_claude_code(outline, slug))
                else:
                    file_paths.append(self._build_pptx(outline, slug))
            if fmt in ("pdf", "both"):
                file_paths.append(self._build_pdf(outline, slug))
        except Exception as exc:
            return ToolResult.fail(f"Failed to render report file: {exc}")

        duration_ms = int((time.monotonic() - start) * 1000)
        summary = (
            f"'{outline.get('title', topic)}' 보고서를 생성했습니다 "
            f"({len(outline.get('sections', []))}개 섹션, {len(file_paths)}개 파일)."
        )
        return ToolResult(
            success=True,
            output=summary,
            metadata={"file_paths": [str(p) for p in file_paths], "title": outline.get("title", topic)},
            duration_ms=duration_ms,
        )

    async def _research(self, topic: str) -> str:
        """Best-effort web research to ground the outline. Never fails the
        whole report — if search is unavailable or errors, falls back to
        the LLM's own knowledge (logged, not raised)."""
        if self.web_search_tool is None:
            return ""
        try:
            result = await self.web_search_tool.execute({"query": topic})
        except Exception:
            return ""
        return str(result.output) if result.success else ""

    async def _generate_outline(self, topic: str, research: str = "") -> dict[str, Any]:
        research_block = (
            "\n\n다음은 이 주제로 검색한 웹 검색 결과입니다. 주제와 실제로 관련된 내용만 "
            "사실 관계·최신 동향 파악에 참고하고, 주제와 무관하거나 엉뚱한 내용(예: 검색어의 "
            "일부 단어만 우연히 겹치는 뉴스)은 완전히 무시하세요. 관련 있는 내용이 없다면 "
            f"검색 결과를 참고하지 말고 당신의 기존 지식으로만 작성하세요:\n{research}"
            if research
            else ""
        )
        prompt = f"""다음 주제로 보고서/발표자료 개요를 작성하세요: "{topic}"{research_block}

{self.num_sections}개의 섹션으로 구성하고, 각 섹션마다 {self.bullets_per_section}개의 핵심 bullet point를 작성하세요.
반드시 한국어로 작성하고, 아래 JSON 형식으로만 응답하세요:

{{"title": "보고서 제목", "sections": [{{"heading": "섹션 제목", "bullets": ["내용1", "내용2"]}}]}}"""

        completion = await self.llm.generate(
            [
                {"role": "system", "content": "You are a report/presentation outline generator. Respond with JSON only."},
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            response_format="json",
        )
        data = json.loads(completion.content)
        if not isinstance(data, dict) or "sections" not in data:
            raise ValueError(f"LLM returned malformed outline: {completion.content[:200]}")
        return data

    async def _build_pptx_via_claude_code(self, outline: dict[str, Any], slug: str) -> pathlib.Path:
        """Delegates slide design to Claude Code — it writes and runs its
        own python-pptx script, so it can apply real design judgement
        (color theme, varied layouts, typographic hierarchy) instead of
        the fixed bullet-list template in _build_pptx(). Falls back to
        that template if Claude Code fails for any reason, so a network
        hiccup or auth issue never fails the whole report.
        """
        dest = self.output_dir / f"{slug}.pptx"
        outline_json = json.dumps(outline, ensure_ascii=False, indent=2)
        task = f"""python-pptx를 사용해서 아래 개요를 바탕으로 전문적이고 시각적으로 매력적인 한국어
PowerPoint 발표자료를 만들어 '{dest}' 경로에 저장해주세요. 스크립트를 작성해서 실행하는 방식으로
진행하세요.

디자인 요구사항:
- 타이틀 슬라이드 + 섹션별 슬라이드로 구성 (섹션당 1슬라이드)
- 통일된 색상 테마 적용 (배경/포인트 컬러, 무채색 기본 텍스트 등 일관성 있게)
- 단순 글머리 기호 나열이 아니라 슬라이드마다 레이아웃에 변화를 주세요 (강조 박스, 2단 구성 등 자유롭게 판단)
- 제목/본문 폰트 크기·굵기로 정보 위계를 명확히 표현
- 맑은 고딕 등 한글이 잘 표시되는 폰트를 지정
- 완료 후 '{dest}'에 파일이 정상 생성됐는지 확인하세요

개요:
{outline_json}
"""
        call_start = time.monotonic()
        try:
            result = await self.claude_code_tool.execute(
                {
                    "task": task,
                    "cwd": str(self.output_dir),
                    "permission_mode": "acceptEdits",
                    # acceptEdits only auto-approves Write/Edit — running the
                    # script it writes needs Bash, which still prompts (and
                    # silently hangs/denies since headless mode has no TTY
                    # to approve it). Scope the pre-approval to python3 only.
                    "allowed_tools": ["Bash(python3 *)"],
                }
            )
        except Exception:
            logger.exception("Claude Code PPTX design raised an exception; falling back to template")
            result = None
        logger.info(
            "Claude Code PPTX call returned after %.1fs, dest.exists()=%s, dir=%s",
            time.monotonic() - call_start, dest.exists(), sorted(p.name for p in self.output_dir.iterdir()),
        )

        if result is not None and result.success:
            # The claude subprocess reporting done doesn't guarantee the
            # file it wrote is visible to us yet — give it a window rather
            # than immediately declaring failure and overwriting a file
            # that's about to land.
            for _ in range(40):
                if dest.exists():
                    return dest
                await asyncio.sleep(1)
            logger.warning(
                "Still not visible after waiting; dir now=%s",
                sorted(p.name for p in self.output_dir.iterdir()),
            )

        if result is None:
            logger.warning("Claude Code PPTX design unavailable; falling back to template")
        elif not result.success:
            logger.warning("Claude Code PPTX design failed (%s); falling back to template", result.error)
        else:
            logger.warning(
                "Claude Code reported success but '%s' was not created; falling back to template", dest
            )
        return self._build_pptx(outline, slug)

    def _build_pptx(self, outline: dict[str, Any], slug: str) -> pathlib.Path:
        from pptx import Presentation

        prs = Presentation()
        title_slide = prs.slides.add_slide(prs.slide_layouts[0])
        title_slide.shapes.title.text = outline.get("title", "")
        if len(title_slide.placeholders) > 1:
            title_slide.placeholders[1].text = "Hermes Agent 자동 생성 보고서"

        bullet_layout = prs.slide_layouts[1]
        for section in outline.get("sections", []):
            slide = prs.slides.add_slide(bullet_layout)
            slide.shapes.title.text = section.get("heading", "")
            body = slide.placeholders[1].text_frame
            body.clear()
            bullets = section.get("bullets", [])
            for i, bullet in enumerate(bullets):
                p = body.paragraphs[0] if i == 0 else body.add_paragraph()
                p.text = str(bullet)

        dest = self.output_dir / f"{slug}.pptx"
        prs.save(str(dest))
        return dest

    def _build_pdf(self, outline: dict[str, Any], slug: str) -> pathlib.Path:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer

        registered = set(pdfmetrics.getRegisteredFontNames())
        for font_name in ("HYGothic-Medium", "HYSMyeongJo-Medium"):
            if font_name not in registered:
                pdfmetrics.registerFont(UnicodeCIDFont(font_name))

        styles = getSampleStyleSheet()
        styles["Title"].fontName = "HYGothic-Medium"
        styles["Heading2"].fontName = "HYGothic-Medium"
        styles["Normal"].fontName = "HYSMyeongJo-Medium"

        dest = self.output_dir / f"{slug}.pdf"
        doc = SimpleDocTemplate(str(dest), pagesize=A4)
        story = [Paragraph(outline.get("title", ""), styles["Title"]), Spacer(1, 20)]
        for section in outline.get("sections", []):
            story.append(Paragraph(section.get("heading", ""), styles["Heading2"]))
            bullets = section.get("bullets", [])
            story.append(ListFlowable([ListItem(Paragraph(str(b), styles["Normal"])) for b in bullets]))
            story.append(Spacer(1, 12))
        doc.build(story)
        return dest
