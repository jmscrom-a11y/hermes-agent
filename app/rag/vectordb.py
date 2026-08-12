from pathlib import Path


DEFAULT_INDEX_DIR = "data/faiss_index"

# Ollama's embed endpoint drops the connection when a single request embeds
# too many texts at once (reproduced: 150 texts succeed, 200 fail with
# "read: connection reset by peer" from the underlying llama-server). Batch
# the embedding calls to stay well under that threshold.
EMBED_BATCH_SIZE = 64


def _require_faiss():
    try:
        from langchain_community.vectorstores import FAISS
    except ImportError as exc:
        raise ImportError(
            "FAISS vector store requires langchain-community and faiss-cpu. "
            "Install: pip install langchain-community faiss-cpu"
        ) from exc
    return FAISS


def create_vector_db(documents, embeddings, batch_size=EMBED_BATCH_SIZE):
    FAISS = _require_faiss()
    if not documents:
        raise ValueError("Cannot create a FAISS index from an empty document list.")

    vector_db = None
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        if vector_db is None:
            vector_db = FAISS.from_documents(batch, embeddings)
        else:
            vector_db.add_documents(batch)
    return vector_db


def add_documents_to_vector_db(vector_db, documents, batch_size=EMBED_BATCH_SIZE):
    """Embed and add *documents* to an already-loaded FAISS index in place.

    Batched the same way as create_vector_db, for the same reason (Ollama's
    embed endpoint drops the connection above ~150-200 texts/request).
    """
    for i in range(0, len(documents), batch_size):
        vector_db.add_documents(documents[i : i + batch_size])
    return vector_db


def save_vector_db(vector_db, index_dir=DEFAULT_INDEX_DIR):
    path = Path(index_dir)
    path.mkdir(parents=True, exist_ok=True)
    vector_db.save_local(str(path))
    return str(path)


def load_vector_db(embeddings, index_dir=DEFAULT_INDEX_DIR):
    FAISS = _require_faiss()
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def documents_from_vector_db(vector_db) -> list:
    """Return every chunk already stored in *vector_db*'s docstore.

    The FAISS index's docstore holds the exact same Document objects (chunk
    text + metadata) that were embedded when the index was built — reading
    them back is a pure in-memory lookup. This exists so BM25's corpus can
    be sourced from here instead of re-parsing every file in data/docs (PDF
    extraction + OCR fallback) from scratch on every pipeline load, which
    measured at ~145s for this corpus vs. a few seconds reading the already-
    loaded index. It also incidentally fixes a granularity mismatch: the
    previous re-load produced page-level documents while FAISS holds the
    smaller split_documents() chunks, so BM25 and vector search were never
    scoring the same units of text.
    """
    return [
        vector_db.docstore.search(doc_id)
        for doc_id in vector_db.index_to_docstore_id.values()
    ]
