"""Report generation tool for Hermes V4.

Given a topic, asks the LLM for a structured outline (title + sections
+ bullets) and renders it as a PPTX and/or PDF file. Files are handed
back via ToolResult.metadata["file_paths"]; the Telegram layer sends
them as document attachments.
"""

from __future__ import annotations

import json
import pathlib
import re
import time
from typing import Any

from hermes_v4.config.settings import get_settings
from hermes_v4.core.base import Tool, ToolResult
from hermes_v4.llm.provider import LLMProvider

_VALID_FORMATS = {"pptx", "pdf", "both"}


def _slugify(text: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip()
    slug = re.sub(r"\s+", "_", slug)
    return slug[:60] or "report"


class ReportTool(Tool):
    """Generates a PPTX/PDF report for a given topic."""

    name = "generate_report"
    description = (
        "주제를 주면 LLM 지식을 바탕으로 발표자료(PPTX) 또는 문서(PDF) 보고서를 생성합니다. "
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

    def __init__(self, llm: LLMProvider, output_dir: str | pathlib.Path | None = None) -> None:
        settings = get_settings()
        self.llm = llm
        self.output_dir = pathlib.Path(output_dir or settings.REPORTS_DIR)
        self.num_sections = settings.REPORT_SECTIONS
        self.bullets_per_section = settings.REPORT_BULLETS_PER_SECTION

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
        try:
            outline = await self._generate_outline(topic)
        except Exception as exc:
            return ToolResult.fail(f"Failed to generate report outline: {exc}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        slug = _slugify(outline.get("title") or topic)
        file_paths = []
        try:
            if fmt in ("pptx", "both"):
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

    async def _generate_outline(self, topic: str) -> dict[str, Any]:
        prompt = f"""다음 주제로 보고서/발표자료 개요를 작성하세요: "{topic}"

{self.num_sections}개의 섹션으로 구성하고, 각 섹션마다 {self.bullets_per_section}개의 핵심 bullet point를 작성하세요.
반드시 한국어로 작성하고, 아래 JSON 형식으로만 응답하세요:

{{"title": "보고서 제목", "sections": [{{"heading": "섹션 제목", "bullets": ["내용1", "내용2"]}}]}}"""

        completion = await self.llm.generate(
            [
                {"role": "system", "content": "You are a report/presentation outline generator. Respond with JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format="json",
        )
        data = json.loads(completion.content)
        if not isinstance(data, dict) or "sections" not in data:
            raise ValueError(f"LLM returned malformed outline: {completion.content[:200]}")
        return data

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
