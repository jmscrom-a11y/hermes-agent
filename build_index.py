"""Build and persist a FAISS index from documents in data/docs."""

import os
import pathlib

from app.rag.pipeline import build_index
from hermes_v4.config.settings import get_settings

ROOT = pathlib.Path(__file__).resolve().parent
DOCS_DIR = ROOT / "data" / "docs"
INDEX_DIR = ROOT / "data" / "faiss_index"


def main() -> None:
    # Resolve embedding model from .env (via pydantic-settings / os.environ).
    # Must match hermes_v4's RAG_EMBEDDING_MODEL — RAGTool loads the saved
    # index with that setting, so building with a different model here
    # silently produces a dimension mismatch at query time (FAISS asserts
    # on vector dim, RAGTool swallows it as a generic "RAG execution failed").
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", get_settings().RAG_EMBEDDING_MODEL)

    # base_url is now resolved inside create_embeddings() for consistency.
    # To override, set OLLAMA_EMBED_BASE_URL in .env or pass it explicitly.

    # Load all files under data/docs
    paths = [str(p) for p in DOCS_DIR.rglob("*") if p.is_file()]
    if not paths:
        print(f"No documents found under {DOCS_DIR}")
        return

    # Build and persist the index
    build_index(
        paths=paths,
        index_dir=str(INDEX_DIR),
        embedding_model=embed_model,
    )

    print("Index build completed.")


if __name__ == "__main__":
    main()
