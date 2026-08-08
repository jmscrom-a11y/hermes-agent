import math
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.rag.vectordb import create_vector_db, load_vector_db, save_vector_db


DEFAULT_TOP_K = 10
DEFAULT_SCORE_THRESHOLD = 0.5


def create_retriever(vector_db, k=DEFAULT_TOP_K, search_type="similarity"):
    return vector_db.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )


def retrieve_documents(retriever, query):
    print(type(retriever), repr(retriever), hasattr(retriever, "invoke"))
    if hasattr(retriever, "invoke"):
        docs = retriever.invoke(query)
    else:
        docs = retriever.get_relevant_documents(query)
    # TEMP debug: print first 500 chars of each retrieved doc
    for i, doc in enumerate(docs):
        content = getattr(doc, "page_content", str(doc))[:500]
        print(f"[DEBUG retrieve_documents] doc[{i}]: {content}")
    return docs


# BM25 Retriever


def _tokenize(text: str) -> list[str]:
    """문서를 토큰으로 분해합니다. 영문/숫자/한글/일본어 등 Unicode 문자 클래스를 사용합니다."""
    # 영문 + 숫자 + 한글 + 일본어 + 중국어 + 기타 Unicode 문자
    return re.findall(r"\w+|[^\s]", text, re.UNICODE)


@dataclass(frozen=True)
class BM25Retriever:
    """BM25Okapi 기반 문서 검색기."""

    documents: list  # langchain Document objects

    def __post_init__(self):
        tokenized_docs = [_tokenize(doc.page_content) for doc in self.documents]
        object.__setattr__(self, "_tokenized", tokenized_docs)
        object.__setattr__(self, "_bm25", BM25Okapi(tokenized_docs))

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> list:
        """query에 대한 BM25 스코어 상위 k개 문서를 반환합니다."""
        if not self.documents:
            return []
        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        top_k_idx = scores.argsort()[::-1][:k]
        return [self.documents[i] for i in top_k_idx]


# Reranker (query-document cosine similarity 기반)


@dataclass(frozen=True)
class Reranker:
    """Ollama 임베딩 기반 query-document cosine similarity reranker.

    hybrid 검색에서 re-scoring 후 threshold 미달 문서를 필터링합니다.
    """

    threshold: float = DEFAULT_SCORE_THRESHOLD

    def rerank(self, query: str, documents: list, k: int = DEFAULT_TOP_K) -> list:
        """query에 대한 문서별 cosine similarity를 재계산하고,
        threshold 미달 문서를 필터링한 후 상위 k개를 반환합니다.

        Args:
            query: 검색 쿼리.
            documents: rerank 대상 Document 목록 (RRF fusion 결과).
            k: 반환 최대 문서 수.

        Returns:
            threshold 이상인 문서 중 similarity 기준 상위 k개.
        """
        if not documents:
            return []

        from app.rag.embeddings import create_embeddings

        embedder = create_embeddings()
        query_vec = embedder.embed_query(query)

        # RRF fusion scores reflect BM25/vector rank agreement — a strong,
        # precise relevance signal (e.g. exact keyword matches) that pure
        # cosine similarity can miss or be outscored by a generically
        # similar-sounding but wrong document. Sorting by cosine similarity
        # alone (as this used to) discarded that signal entirely once it
        # reached reranking. Normalize RRF score to 0-1 and blend it in
        # instead; cosine similarity still acts as the relevance gate via
        # `threshold`.
        rrf_scores = [item[1] for item in documents if isinstance(item, tuple)]
        max_rrf = max(rrf_scores) if rrf_scores else 0.0

        scored = []
        for item in documents:
            if isinstance(item, tuple):
                doc, rrf_score = item
                content = doc.page_content
            else:
                doc, rrf_score = item, 0.0
                content = item.page_content
            doc_vec = embedder.embed_documents([content])[0]
            sim = _cosine_similarity(query_vec, doc_vec)
            if sim >= self.threshold:
                normalized_rrf = (rrf_score / max_rrf) if max_rrf else 0.0
                combined = sim + normalized_rrf
                scored.append((doc, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, _score in scored[:k]]
        print(f"[RERANK] count={len(reranked_docs)}, first_doc[:100]={repr(reranked_docs[0].page_content[:100]) if reranked_docs else '[]'}")
        return reranked_docs


def _cosine_similarity(a: list, b: list) -> float:
    """두 벡터 간 cosine similarity를 계산합니다."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# Hybrid Retriever (BM25 + FAISS, RRF fusion + reranking)


@dataclass(frozen=True)
class HybridRetriever:
    """FAISS cosine similarity + BM25 기반 hybrid 검색기.

    RRF(Rank Reciprocal Fusion)으로 두 소스의 결과를 결합한 후,
    최종 RRF 스코어와 query-document cosine similarity 기준 이중 필터링을 수행합니다.

    Args:
        vector_retriever: FAISS retriever (as_retriever() 결과).
        bm25_retriever: BM25Retriever 인스턴스.
        k: 반환 최대 문서 수.
        reranker: Reranker 인스턴스 (optional, default: threshold=0.5).
        score_threshold: 최종 RRF 스코어 최소 임계값 (default: 0.5).
    """

    vector_retriever: object
    bm25_retriever: BM25Retriever
    k: int = DEFAULT_TOP_K
    reranker: Reranker | None = None
    score_threshold: float = DEFAULT_SCORE_THRESHOLD

    def retrieve(self, query: str) -> list:
        """RRF fusion + reranking 후 score_threshold 미만의 낮은 관련성 문서를 제거합니다."""
        # Fetch a wider BM25 candidate pool than the final k — narrowing to
        # k here starves RRF fusion the same way capping the vector
        # retriever at k did (see load_pipeline/build_pipeline).
        vector_docs = self._get_vector_results(query)
        bm25_docs = self.bm25_retriever.retrieve(query, k=max(self.k * 3, 10))

        if not vector_docs and not bm25_docs:
            print("[WARN] retrieve: both vector and BM25 returned empty results")
            return []

        def _get_doc_key(doc) -> str:
            """문서에 안정적인 키 생성 (metadata[source] + page_content 길이)."""
            meta = getattr(doc, 'metadata', {}) or {}
            source = meta.get('source', 'unknown')
            content_len = len(getattr(doc, 'page_content', ''))
            return f"{source}:{content_len}"

        # RRF fusion: 문서 키 기반 rank 매핑
        vector_ranks: dict[str, float] = {}
        for rank, doc in enumerate(vector_docs, start=1):
            key = _get_doc_key(doc)
            vector_ranks[key] = 1.0 / rank

        bm25_ranks: dict[str, float] = {}
        for rank, doc in enumerate(bm25_docs, start=1):
            key = _get_doc_key(doc)
            bm25_ranks[key] = 1.0 / rank

        # fusion: 모든 candidate 에 대한 RRF score 계산
        all_keys = set(vector_ranks.keys()) | set(bm25_ranks.keys())
        rrf_scores: dict[str, float] = {}
        for key in all_keys:
            score = vector_ranks.get(key, 0.0) + bm25_ranks.get(key, 0.0)
            rrf_scores[key] = score

        # doc 매핑 (key -> doc)
        doc_map: dict[str, object] = {}
        for doc in vector_docs:
            key = _get_doc_key(doc)
            if key not in doc_map:
                doc_map[key] = doc
        for doc in bm25_docs:
            key = _get_doc_key(doc)
            if key not in doc_map:
                doc_map[key] = doc

        # candidates: (doc, rrf_score)
        candidates = [(doc_map[key], score) for key, score in rrf_scores.items()]

        # NOTE: score_threshold is a cosine-similarity cutoff (0-1) for the
        # Reranker below. RRF scores live on a totally different scale (sum
        # of reciprocal ranks, rarely > 1.0 unless a doc is rank-1 in both
        # vector and BM25 search), so applying the same threshold here as a
        # pre-filter silently dropped correct-but-not-top-ranked documents
        # before the reranker ever got to judge them semantically. All fused
        # candidates go to reranking; RRF score only decides fallback order.

        # reranking 수행 (전체 fusion 후보 대상)
        if self.reranker is not None and candidates:
            reranked_docs = self.reranker.rerank(query, candidates, k=self.k)
            if not reranked_docs and candidates:
                print(f"[WARN] reranker filtered all {len(candidates)} candidates; falling back to RRF top-{self.k}")
                candidates.sort(key=lambda x: x[1], reverse=True)
                return [doc for doc, _score in candidates[: self.k]]
            return reranked_docs

        # reranker 없음: RRF score 기준 정렬 후 반환
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _score in candidates[: self.k]]

    def _get_vector_results(self, query: str) -> list:
        """FAISS retriever 결과 반환 (invoke / get_relevant_documents 지원)."""
        if hasattr(self.vector_retriever, "invoke"):
            return self.vector_retriever.invoke(query)
        return self.vector_retriever.get_relevant_documents(query)

    def get_relevant_documents(self, query: str) -> list:
        """기존 retriever API 호환 alias."""
        return self.retrieve(query)


def create_hybrid_retriever(vector_retriever, documents, k=DEFAULT_TOP_K, score_threshold=DEFAULT_SCORE_THRESHOLD):
    """FAISS retriever + BM25 retriever를 RRF fusion + reranking으로 결합합니다.

    Args:
        vector_retriever: FAISS.as_retriever() 결과.
        documents: 원본 Document 목록 (BM25 학습용).
        k: 반환 최대 문서 수.
        score_threshold: RRF 스코어 최소 임계값 (default: 0.5).

    Returns:
        HybridRetriever 인스턴스.
    """
    bm25_retriever = BM25Retriever(documents)
    reranker = Reranker(threshold=score_threshold)
    return HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        k=k,
        reranker=reranker,
        score_threshold=score_threshold,
    )
