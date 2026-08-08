"""RAG search tool for Hermes V4.

Wraps the existing (v3) hybrid RAG pipeline — FAISS + BM25 with RRF fusion
and cosine-similarity reranking — behind the Tool interface so the planner
can invoke it like any other tool.
"""

from __future__ import annotations

import asyncio
import pathlib
import time
from typing import Any

from hermes_v4.config.settings import get_settings
from hermes_v4.core.base import Tool, ToolResult


class RAGTool(Tool):
    """Answers questions by retrieving from the local document index."""

    name = "rag"
    description = (
        "저장된 문서(PDF, 마크다운 등)에서 관련 내용을 검색해 질문에 답변합니다. "
        "문서 내용, 요약, 특정 자료에 대한 질문에 사용하세요."
    )
    input_schema = {
        "type": "object",
        "properties": {"question": {"type": "string"}},
        "required": ["question"],
    }
    output_schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "sources": {"type": "array", "items": {"type": "string"}},
        },
    }

    def __init__(self, docs_dir: str | pathlib.Path | None = None) -> None:
        settings = get_settings()
        self._index_dir = settings.RAG_INDEX_DIR
        self._top_k = settings.RAG_TOP_K
        self._docs_dir = pathlib.Path(docs_dir) if docs_dir else self._default_docs_dir()
        self._pipeline = None
        self._load_lock = asyncio.Lock()

    @staticmethod
    def _default_docs_dir() -> pathlib.Path:
        return pathlib.Path(__file__).resolve().parents[2] / "data" / "docs"

    async def validate_input(self, input: dict[str, Any]) -> list[str]:
        errors = []
        question = input.get("question")
        if not question or not str(question).strip():
            errors.append("'question' is required and must be a non-empty string")
        return errors

    async def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        async with self._load_lock:
            if self._pipeline is None:
                self._pipeline = await asyncio.to_thread(self._load_pipeline_sync)
        return self._pipeline

    def _load_pipeline_sync(self):
        from app.rag.loader import collect_files
        from app.rag.pipeline import load_pipeline

        doc_paths = [str(p) for p in collect_files([str(self._docs_dir)])]
        return load_pipeline(
            self._index_dir,
            k=self._top_k,
            hybrid=True,
            documents=doc_paths,
        )

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        errors = await self.validate_input(input)
        if errors:
            return ToolResult.fail("; ".join(errors))

        question = str(input["question"]).strip()
        start = time.monotonic()
        try:
            pipeline = await self._get_pipeline()
        except Exception as exc:
            return ToolResult.fail(f"Failed to load RAG pipeline: {exc}")

        try:
            documents = await asyncio.to_thread(pipeline.retrieve, question)
            answer = await asyncio.to_thread(
                lambda: pipeline.llm(pipeline.prompt_builder.build(question, documents))
            )
            formatted = pipeline._format_answer(answer, documents)
        except Exception as exc:
            return ToolResult.fail(f"RAG execution failed: {exc}")

        sources = sorted(
            {
                (getattr(doc, "metadata", {}) or {}).get("source", "unknown")
                for doc in documents
            }
        )
        duration_ms = int((time.monotonic() - start) * 1000)
        return ToolResult(
            success=True,
            output=formatted,
            metadata={"sources": sources, "num_documents": len(documents)},
            duration_ms=duration_ms,
        )
