import uuid
from dataclasses import dataclass, field

from app.rag.loader import collect_files


@dataclass
class ChunkMeta:
    """청크 메타데이터.

    Attributes:
        chunk_id: 청크 고유 ID.
        document_id: 원본 문서 고유 ID.
        source: 파일 경로.
        line_range: 청크의 라인 범위 (start, end).
    """

    chunk_id: str = field(default_factory=lambda: f"chunk_{uuid.uuid4().hex[:8]}")
    document_id: str = field(default_factory=lambda: f"doc_{uuid.uuid4().hex[:8]}")
    source: str = ""
    line_range: tuple[int, int] = field(default_factory=lambda: (0, 0))


def _require_splitter():
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise ImportError(
            "RAG chunking requires langchain-text-splitters. "
            "Install: pip install langchain-text-splitters"
        ) from exc
    return RecursiveCharacterTextSplitter


def create_text_splitter(chunk_size=500, chunk_overlap=150):
    RecursiveCharacterTextSplitter = _require_splitter()
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def split_documents(documents, chunk_size=500, chunk_overlap=150):
    """문서를 청킹하고 메타데이터(chunk_id, document_id, source, line_range)를 추가.

    Args:
        documents: langchain Document 목록.
        chunk_size: 청크 크기.
        chunk_overlap: 청크 간 오버랩.

    Returns:
        메타데이터가 포함된 langchain Document 목록.
    """
    splitter = create_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    enriched = []
    for doc in documents:
        source = getattr(doc, "metadata", {}).get("source", "unknown")
        document_id = f"doc_{uuid.uuid4().hex[:8]}"
        line_start = 0

        # metadata에 source가 없으면 page_content의 일부로 기록
        if "source" not in getattr(doc, "metadata", {}):
            doc.metadata["source"] = source

        sub_docs = splitter.split_documents([doc])
        for sub_doc in sub_docs:
            # 라인 범위 계산 (대략적: chunk_size 기준으로 추정)
            content = sub_doc.page_content
            lines = content.split("\n")
            line_end = line_start + len(lines) - 1

            sub_doc.metadata["chunk_id"] = f"chunk_{uuid.uuid4().hex[:8]}"
            sub_doc.metadata["document_id"] = document_id
            sub_doc.metadata["source"] = source
            sub_doc.metadata["line_range"] = (line_start, line_end)

            enriched.append(sub_doc)
            line_start = max(0, line_end - chunk_overlap + 1)  # overlap 반영

    return enriched
