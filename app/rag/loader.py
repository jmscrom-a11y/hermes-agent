import base64
import re
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

# frob/unlimited-ocr tags every detected element with a layout label and
# bounding box, e.g. "text [38, 75, 231, 110]Hermes Agent RAG Pipeline" —
# that's layout metadata, not document content, so it's stripped before the
# text is used as a RAG document (see _ocr_image_bytes).
_OCR_LAYOUT_PREFIX_RE = re.compile(r"(?m)^\s*\w+\s*\[\d+(?:,\s*\d+){3}\]\s*")

# PDFs with a broken/subset font cmap (observed in data/docs/NotebookLM_ai_agent.pdf,
# likely exported from a slide tool) decode fine structurally but individual
# glyphs — usually punctuation like curly quotes — land on the wrong Unicode
# codepoint, e.g. a quote mark decoding as U+3400 "㑀". These blocks are
# essentially never used in real Korean/English business documents, so any
# occurrence is a strong signal the page's text layer is corrupted rather
# than just short. A pure length check (RAG_OCR_MIN_TEXT_CHARS) misses this
# because the page can have plenty of characters — they're just wrong ones.
_MOJIBAKE_RE = re.compile(
    r"[㌀-㏿"  # CJK Compatibility (e.g. ㏖ ㏗)
    r"㐀-䶿"  # CJK Unified Ideographs Extension A (e.g. 㑀 㑁)
    r"豈-﫿]"  # CJK Compatibility Ideographs
)


def _is_garbled(text: str) -> bool:
    return bool(_MOJIBAKE_RE.search(text))


# See the comment where this is used (PDF page filtering below) — measured
# against this corpus, 7/102 pages fall under this and all were single
# headings/fragments with no standalone informational value.
_MIN_DOC_CHARS = 20


SUPPORTED_TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
}

# OCR'd through a local vision model (see _ocr_image_bytes) rather than
# decoded directly — there's no text layer to extract.
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

# Extracted via markitdown (see _extract_text_via_markitdown) — the same
# library hermes_v4/tools/report_tool.py already uses for PPTX conversion,
# reused here instead of parsing OOXML by hand with python-docx/python-pptx.
OFFICE_SUFFIXES = {".docx", ".pptx"}

# Default allowed base directory for file discovery.
# All resolved paths MUST remain inside this directory to prevent external access.
DEFAULT_ROOT_DIR: Path = Path(__file__).resolve().parents[2] / "data" / "docs"


def _normalize_ollama_host(url: str) -> str:
    """Strip any OpenAI-compat path (e.g. /v1) so Ollama's native /api/... works."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "localhost"
    port = parsed.port or 11434
    return f"{scheme}://{host}:{port}"


def _upright(image_bytes: bytes) -> bytes:
    """Bake in the image's EXIF orientation before OCR.

    Phone cameras (e.g. this corpus's Galaxy S25+ photo uploads) store
    rotation as EXIF metadata rather than rotating the pixel data itself.
    Fed the raw (sideways) pixels, frob/unlimited-ocr doesn't just read the
    text at an angle — it hallucinates unrelated table/text content in a
    different language entirely. Measured on a real upload: raw bytes
    produced 31s of garbled Chinese exam-table text; EXIF-corrected bytes
    produced the correct Korean recipe text in 8s.
    """
    import io

    from PIL import Image, ImageOps

    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=92)
        return buf.getvalue()
    except Exception:
        return image_bytes


def _ocr_image_bytes(image_bytes: bytes) -> str:
    """Extract text from image bytes via a local Ollama OCR/vision model.

    Uses the same model as data/docs image ingestion and the scanned-PDF
    fallback below — settings.RAG_OCR_MODEL (default: frob/unlimited-ocr:q8_0).
    """
    import httpx

    from hermes_v4.config.settings import get_settings

    settings = get_settings()
    host = _normalize_ollama_host(settings.LLM_BASE_URL)
    payload = {
        "model": settings.RAG_OCR_MODEL,
        "messages": [
            {
                "role": "user",
                "content": "Extract all text from this image verbatim. Output only the extracted text, no commentary.",
                "images": [base64.b64encode(_upright(image_bytes)).decode("ascii")],
            }
        ],
        "stream": False,
        "think": False,
    }
    with httpx.Client(timeout=120.0) as client:
        response = client.post(f"{host}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
    text = _OCR_LAYOUT_PREFIX_RE.sub("", data["message"]["content"])
    return _dedupe_ocr_lines(text.strip())


def _dedupe_ocr_lines(text: str) -> str:
    """Drop exact-duplicate lines from an OCR result.

    frob/unlimited-ocr occasionally re-emits the same content under two
    different layout labels (e.g. both a "text" and a "footer" element
    with identical text) — a straight hallucinated repeat, not a document
    that actually repeats itself. Keeps first occurrence and order.
    """
    seen: set[str] = set()
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        lines.append(line)
    return "\n".join(lines)


def _extract_text_via_markitdown(path: Path) -> str:
    from markitdown import MarkItDown

    return (MarkItDown().convert(str(path)).text_content or "").strip()


def _ocr_scanned_pdf_pages(path: Path, docs: List, min_text_chars: int) -> List:
    """OCR pages whose extracted text layer is too sparse, or garbled, to be usable.

    PyMuPDFLoader only pulls the PDF's text layer, which is either empty for
    scanned/image-only pages, or — for PDFs with a broken font cmap — full
    length but full of mojibake (see _is_garbled). This renders just those
    specific pages to images and OCRs them instead, leaving genuinely fine
    pages untouched.
    """
    import fitz  # pymupdf

    needs_ocr = [
        i
        for i, d in enumerate(docs)
        if len(d.page_content.strip()) < min_text_chars or _is_garbled(d.page_content)
    ]
    if not needs_ocr:
        return docs

    try:
        pdf = fitz.open(str(path))
    except Exception as exc:  # noqa: BLE001
        print(f"[loader] {path.name!r}: could not open for OCR fallback — {exc}")
        return docs

    for i in needs_ocr:
        if i >= pdf.page_count:
            continue
        image_bytes = pdf[i].get_pixmap(dpi=200).tobytes("png")
        try:
            text = _ocr_image_bytes(image_bytes)
        except Exception as exc:  # noqa: BLE001
            print(f"[loader] {path.name!r} page {i + 1}: OCR failed — {exc}")
            continue
        if text:
            docs[i].page_content = text
    pdf.close()
    return docs


def _require_loaders():
    try:
        from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
    except ImportError as exc:
        raise ImportError(
            "RAG loaders require langchain-community and pymupdf. "
            "Install: pip install langchain langchain-community pymupdf"
        ) from exc
    return PyMuPDFLoader, TextLoader


def _is_within_root(file_path: Path, root: Path) -> bool:
    """Return True if *file_path* is strictly inside *root*.

    Uses ``resolve()`` to canonicalise symlinks and ``..`` segments so that
    a path like ``/data/docs/../etc/passwd`` cannot escape the allowed zone.
    """
    try:
        return root in file_path.resolve().parents or file_path.resolve() == root
    except OSError:
        return False


def collect_files(
    paths: List[str | Path],
    suffixes: Optional[set[str]] = None,
    root_dir: Optional[Path] = None,
) -> List[Path]:
    """Return an ordered list of supported files found under *root_dir*.

    Parameters
    ----------
    paths : iterable of str or Path
        Directories (or files) to scan.  When *paths* contains directories
        their contents are discovered recursively, but every resolved file
        must still lie inside *root_dir*.
    suffixes : set[str], optional
        File extensions to include.  Defaults to ``SUPPORTED_TEXT_SUFFIXES``
        plus ``".pdf"``.
    root_dir : Path, optional
        Absolute directory that acts as the allowed boundary.  When omitted
        the default is ``data/docs`` relative to this package.

    Raises
    ------
    ValueError
        If any resolved path escapes *root_dir* (suspicious / malicious input).
    """
    suffixes = suffixes or (SUPPORTED_TEXT_SUFFIXES | {".pdf"} | IMAGE_SUFFIXES | OFFICE_SUFFIXES)
    root = Path(root_dir).resolve() if root_dir else DEFAULT_ROOT_DIR.resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    files: List[Path] = []

    for raw_path in paths:
        path = Path(raw_path)

        # --- Single file case --------------------------------------------------
        if path.is_file():
            resolved = path.resolve()
            if not _is_within_root(resolved, root):
                raise ValueError(
                    f"Refusing to load file outside allowed root: {raw_path!r} "
                    f"(resolved to {resolved})"
                )
            if resolved.suffix.lower() in suffixes:
                files.append(resolved)

        # --- Directory case ----------------------------------------------------
        elif path.is_dir():
            resolved_root = path.resolve()
            if not _is_within_root(resolved_root, root):
                raise ValueError(
                    f"Refusing to scan directory outside allowed root: {raw_path!r} "
                    f"(resolved to {resolved_root})"
                )
            for child in resolved_root.rglob("*"):
                if child.is_file():
                    if not _is_within_root(child, root):
                        # Defensive: rglob should never escape, but guard anyway.
                        raise ValueError(
                            f"Refusing to load file outside allowed root: {child}"
                        )
                    if child.suffix.lower() in suffixes:
                        files.append(child)

        else:
            # Path doesn't exist — skip silently (matches previous behaviour).
            continue

    return sorted(files)


def load_documents(
    paths: List[str | Path],
    root_dir: Optional[Path] = None,
) -> List:
    """Load documents from *paths* into LangChain Document objects.

    Files that fail to decode (e.g. corrupt PDFs, binary blobs, malformed
    text files) are skipped with a warning instead of crashing the pipeline.
    """
    PyMuPDFLoader, TextLoader = _require_loaders()
    from hermes_v4.config.settings import get_settings

    settings = get_settings()
    documents: List = []

    for path in collect_files(paths, root_dir=root_dir):
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            try:
                loader = PyMuPDFLoader(str(path))
                docs = loader.load()
            except Exception as exc:  # noqa: BLE001 — surface all PDF errors
                print(f"[loader] Skipping {path.name!r}: failed to load PDF — {exc}")
                continue
            if settings.RAG_OCR_ENABLED:
                docs = _ocr_scanned_pdf_pages(path, docs, settings.RAG_OCR_MIN_TEXT_CHARS)
            # A page whose entire content is a short heading/fragment (e.g.
            # "MCP", "하지만 한계가 명확") can't stand alone as useful RAG
            # context, and empirically dominates vector search results it
            # has no business winning: very short chunks sit unusually close
            # to the embedding space's centroid, so they surface as a
            # spuriously high-cosine-similarity "hub" match for nearly any
            # query, crowding out the genuinely relevant (but longer, more
            # specific) chunk. Drop them after the OCR pass above has had a
            # chance to recover real content for pages that were merely
            # scanned/empty rather than genuinely this short.
            docs = [d for d in docs if len(d.page_content.strip()) >= _MIN_DOC_CHARS]
        elif suffix in IMAGE_SUFFIXES:
            if not settings.RAG_OCR_ENABLED:
                print(f"[loader] Skipping {path.name!r}: OCR disabled (RAG_OCR_ENABLED=False)")
                continue
            try:
                text = _ocr_image_bytes(path.read_bytes())
            except Exception as exc:  # noqa: BLE001
                print(f"[loader] Skipping {path.name!r}: OCR failed — {exc}")
                continue
            if not text:
                print(f"[loader] Skipping {path.name!r}: OCR produced no text")
                continue
            docs = [Document(page_content=text, metadata={"source": str(path)})]
        elif suffix in OFFICE_SUFFIXES:
            try:
                text = _extract_text_via_markitdown(path)
            except Exception as exc:  # noqa: BLE001
                print(f"[loader] Skipping {path.name!r}: text extraction failed — {exc}")
                continue
            if not text:
                print(f"[loader] Skipping {path.name!r}: no extractable text")
                continue
            docs = [Document(page_content=text, metadata={"source": str(path)})]
        else:
            try:
                loader = TextLoader(str(path), encoding="utf-8")
                docs = loader.load()
            except UnicodeDecodeError as exc:
                print(
                    f"[loader] Skipping {path.name!r}: "
                    f"UnicodeDecodeError ({exc}) — file is likely not UTF-8"
                )
                continue
            except Exception as exc:  # noqa: BLE001 — surface other load errors
                print(f"[loader] Skipping {path.name!r}: {exc}")
                continue

        documents.extend(docs)

    return documents
