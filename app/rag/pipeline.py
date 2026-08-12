import os
import pathlib
from dataclasses import dataclass

from app.llm.agent import ask_llm
from app.rag.chunker import split_documents
from app.rag.embeddings import DEFAULT_EMBEDDING_MODEL, create_embeddings
from app.rag.loader import load_documents
from app.rag.retriever import (
    BM25Retriever,
    DEFAULT_SCORE_THRESHOLD,
    HybridRetriever,
    candidate_pool_size,
    create_hybrid_retriever,
    create_retriever,
    retrieve_documents,
)
from app.rag.vectordb import (
    add_documents_to_vector_db,
    create_vector_db,
    documents_from_vector_db,
    load_vector_db,
    save_vector_db,
)


DEFAULT_RAG_TEMPLATE = """당신은 Hermes Agent의 RAG 응답 엔진입니다.
아래 컨텍스트만 근거로 사용해 한국어로 답변하세요.
컨텍스트에 답이 없으면 모른다고 말하세요.

질문:
{question}

컨텍스트:
{context}

답변:"""


def _require_prompt_template():
    try:
        from langchain_core.prompts import PromptTemplate
    except ImportError:
        class PromptTemplate:
            def __init__(self, template, input_variables):
                self.template = template
                self.input_variables = input_variables

            def format(self, **kwargs):
                return self.template.format(**kwargs)

    return PromptTemplate


@dataclass
class PromptBuilder:
    template: str = DEFAULT_RAG_TEMPLATE

    def __post_init__(self):
        PromptTemplate = _require_prompt_template()
        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=["question", "context"],
        )

    # 문서당 최대 내용 길이 (context 과다 방지). 실제 청크가 1000자를
    # 훌쩍 넘는 경우가 흔해서(예: 목록형 항목), 300자는 답변에 필요한
    # 핵심 내용이 나오기도 전에 잘라버려 LLM이 "모른다"고 답하게 만들었다.
    MAX_CONTENT_PER_DOC = 1200

    def build_context(self, documents):
        parts = []
        for index, document in enumerate(documents, start=1):
            if isinstance(document, dict):
                metadata = document.get("metadata", {}) or {}
                content = document.get("page_content", str(document))
            else:
                metadata = getattr(document, "metadata", {}) or {}
                content = getattr(document, "page_content", str(document))
            source = metadata.get("source", "unknown")
            page = metadata.get("page")
            location = f"{source}:{page}" if page is not None else source

            # 내용 길이 제한
            if len(content) > self.MAX_CONTENT_PER_DOC:
                content = content[: self.MAX_CONTENT_PER_DOC] + "..."

            parts.append(f"[{index}] {location}\n{content}")
        print("=" * 80)
        print("CONTEXT SENT TO LLM")
        print("\n\n".join(parts))
        print("=" * 80)
        return "\n\n".join(parts)

    def build(self, question, documents):
        return self.prompt.format(
            question=question,
            context=self.build_context(documents),
        )


class RAGPipeline:
    def __init__(self, retriever, llm=ask_llm, prompt_builder=None, hybrid=False, documents=None, embedding_model=None):
        self.llm = llm
        self.prompt_builder = prompt_builder or PromptBuilder()

        if hybrid and documents is not None:
            self.retriever = create_hybrid_retriever(retriever, documents, embedding_model=embedding_model)
        else:
            self.retriever = retriever

    def retrieve(self, question):
        return retrieve_documents(self.retriever, question)

    def build_prompt(self, question, documents=None):
        documents = documents if documents is not None else self.retrieve(question)
        return self.prompt_builder.build(question, documents)

    def answer(self, question):
        documents = self.retrieve(question)
        prompt = self.prompt_builder.build(question, documents)
        answer = self.llm(prompt)
        return self._format_answer(answer, documents)

    def _format_answer(self, answer, documents, include_sources=True):
        """답변에 Sources 섹션을 붙여 반환.

        Args:
            answer: LLM 응답 텍스트
            documents: 검색된 문서 목록
            include_sources: True 면 Sources 섹션 붙임 (기본값).
                            False 면 pure 답변만 반환 (인사/잡담 등).
        """
        if not include_sources or not documents:
            return answer

        sources = []
        for doc in documents:
            metadata = getattr(doc, "metadata", {}) or {}
            source = metadata.get("source", "unknown")
            sources.append(source)

        return f"{answer}\n\nSources: " + ", ".join(dict.fromkeys(sources))


def build_index(paths, index_dir=None, embedding_model=None, chunk_size=500, chunk_overlap=150):
    documents = load_documents(paths)
    chunks = split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    embeddings = create_embeddings(model=embedding_model)
    vector_db = create_vector_db(chunks, embeddings)
    if index_dir:
        save_vector_db(vector_db, index_dir)
    return vector_db


def add_to_index(paths, index_dir, embedding_model=None, chunk_size=500, chunk_overlap=150):
    """Embed and add *paths* into the FAISS index at *index_dir* in place.

    Unlike build_index(), this doesn't re-embed the whole data/docs corpus
    — only the new documents — so a single-file upload stays cheap even as
    the corpus grows. Falls back to creating a fresh index if none exists
    yet at *index_dir*.
    """
    documents = load_documents(paths)
    if not documents:
        raise ValueError(f"No extractable content in: {paths}")
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embeddings = create_embeddings(model=embedding_model)

    index_path = pathlib.Path(index_dir)
    if (index_path / "index.faiss").exists():
        vector_db = load_vector_db(embeddings, index_dir)
        add_documents_to_vector_db(vector_db, chunks)
    else:
        vector_db = create_vector_db(chunks, embeddings)
    save_vector_db(vector_db, index_dir)
    return vector_db


def load_pipeline(index_dir, embedding_model=None, k=4, llm=ask_llm, hybrid=False, score_threshold=DEFAULT_SCORE_THRESHOLD):
    # Resolve the actual model name (mirrors create_embeddings' own fallback)
    # so the reranker below uses the *same* model that built the vector
    # index, instead of silently defaulting to a different one.
    resolved_model = embedding_model or os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_EMBEDDING_MODEL)
    embeddings = create_embeddings(model=resolved_model)
    vector_db = load_vector_db(embeddings, index_dir)
    # Fetch a wider candidate pool than the final k for RRF fusion/reranking
    # to actually work with — capping the vector retriever at the final k
    # (e.g. 4) starves the fusion pool and drops correct-but-not-top-4
    # documents before they ever get a chance to be reranked.
    retriever = create_retriever(vector_db, k=candidate_pool_size(k))
    if hybrid:
        # BM25's corpus comes straight from the already-loaded FAISS
        # docstore (see documents_from_vector_db) instead of re-parsing
        # `documents` from disk — that used to re-run PDF/OCR extraction on
        # every pipeline load (~145s for this corpus) even though the exact
        # same chunk text was already sitting in the index we just loaded.
        bm25_documents = documents_from_vector_db(vector_db)
        pipeline_retriever = create_hybrid_retriever(
            retriever, bm25_documents, k=k, score_threshold=score_threshold, embedding_model=resolved_model
        )
    else:
        pipeline_retriever = retriever
    return RAGPipeline(retriever=pipeline_retriever, llm=llm)


def build_pipeline(paths, index_dir=None, embedding_model=None, k=4, llm=ask_llm, hybrid=False, score_threshold=DEFAULT_SCORE_THRESHOLD):
    resolved_model = embedding_model or os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_EMBEDDING_MODEL)
    docs = load_documents(paths)
    chunks = split_documents(
        docs,
        chunk_size=500,
        chunk_overlap=150,
    )
    embeddings = create_embeddings(model=resolved_model)
    vector_db = create_vector_db(chunks, embeddings)
    retriever = create_retriever(vector_db, k=candidate_pool_size(k))
    if hybrid:
        pipeline_retriever = create_hybrid_retriever(
            retriever, docs, k=k, score_threshold=score_threshold, embedding_model=resolved_model
        )
    else:
        pipeline_retriever = retriever
    return RAGPipeline(retriever=pipeline_retriever, llm=llm)
