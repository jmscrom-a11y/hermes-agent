import math
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.rag.vectordb import EMBED_BATCH_SIZE, create_vector_db, load_vector_db, save_vector_db


DEFAULT_TOP_K = 10
# Normalized-similarity threshold used by Reranker (see its docstring) — not
# an absolute cosine value anymore.
DEFAULT_SCORE_THRESHOLD = 0.3
# Absolute cosine-similarity floor (see Reranker.rerank docstring for why
# min-max normalization alone isn't enough). Measured on bge-m3 against this
# corpus: off-topic queries ("손흥민 골 몇 개", "김치찌개 레시피") topped out
# at raw sim 0.34-0.44 across their best-matching chunk; genuinely answerable
# queries ("영업권은 언제 기록하나요", "클로드 스킬 기능이 뭐야") started at
# 0.59+. 0.5 sits in that gap with margin on both sides.
DEFAULT_ABSOLUTE_SIM_FLOOR = 0.5

# How much wider than the final k the vector/BM25 candidate pools should be
# before RRF fusion. A correct-but-not-lexically-obvious document can rank
# well outside a naive top-(k*3) pool (observed: rank ~50 among 250 chunks
# for a legitimate query before the BM25 particle fix) — a materially wider
# pool costs little (BM25 already scores every doc; FAISS just returns more
# of an already-computed search) but gives such documents a real chance to
# reach reranking instead of being dropped before fusion ever sees them.
CANDIDATE_POOL_MULTIPLIER = 8
CANDIDATE_POOL_MIN = 30


def candidate_pool_size(k: int) -> int:
    return max(k * CANDIDATE_POOL_MULTIPLIER, CANDIDATE_POOL_MIN)


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


# Matches a digit followed by whitespace then a single, standalone Korean
# character (not the start of a longer word) — the shape of PDF-extraction
# artifacts like "6 대" or "5 명", not real word boundaries like "통제활동 6".
_NUMERAL_SPACING_RE = re.compile(r"(?<=[0-9])\s+(?=[가-힣](?:[^가-힣0-9]|$))")


def _normalize_numeral_spacing(text: str) -> str:
    """숫자와 뒤따르는 단일 한글 글자(단위/조사) 사이의 공백을 제거합니다
    (예: "6 대" -> "6대").

    PDF 텍스트 추출 시 자간 조정으로 숫자와 뒤따르는 한글 사이에 없어야 할
    공백이 종종 들어간다. 이 공백이 남아있으면 "6대"처럼 자연스럽게 붙여
    입력한 질의가 "6 대"로 추출된 문서 텍스트와 토큰 단위로 전혀 매칭되지
    않아 BM25가 정답 문서를 완전히 놓친다.
    """
    return _NUMERAL_SPACING_RE.sub("", text)


_HANGUL_TOKEN_RE = re.compile(r"^[가-힣]+$")


def _get_kiwi():
    """Lazily construct a process-wide Kiwi instance (dictionary load is
    ~0.5s, not worth paying on every _tokenize() call). Returns None if
    kiwipiepy isn't installed, so callers can fall back to the plain regex
    splitter — kiwipiepy is a soft dependency like faiss/langchain loaders
    elsewhere in this module.
    """
    global _KIWI
    if _KIWI is _KIWI_UNSET:
        try:
            from kiwipiepy import Kiwi

            _KIWI = Kiwi()
        except ImportError:
            _KIWI = None
    return _KIWI


_KIWI_UNSET = object()
_KIWI = _KIWI_UNSET

# Morpheme tags to keep as BM25 tokens: nouns (N*), verb/adjective stems
# (V*), foreign words (SL, e.g. English mixed into Korean text), hanja (SH),
# numbers (SN). Drops particles (J*: 이/가/을/를/에서/...), endings (E*),
# and punctuation (S* other than SL/SH/SN) — exactly the class of suffix
# that made a query like "감가상각이" (noun + 이 particle) fail to match a
# document's bare "감가상각" under naive whitespace/regex tokenization,
# but resolved properly instead of via a hardcoded particle-suffix list.
_KIWI_KEEP_PREFIXES = ("N", "V", "SL", "SH", "SN")


def _tokenize(text: str) -> list[str]:
    """문서를 토큰으로 분해합니다.

    kiwipiepy가 설치되어 있으면 형태소 분석으로 조사/어미를 제거한 의미
    단위(명사, 용언 어간, 외래어, 한자, 숫자)만 토큰으로 사용합니다.
    설치되어 있지 않으면 영문/숫자/한글/기타 Unicode 문자 클래스 기반의
    단순 정규식 분해로 대체합니다.
    """
    text = _normalize_numeral_spacing(text)
    kiwi = _get_kiwi()
    if kiwi is not None:
        tokens = [t.form for t in kiwi.tokenize(text) if t.tag.startswith(_KIWI_KEEP_PREFIXES)]
    else:
        tokens = re.findall(r"\w+|[^\s]", text, re.UNICODE)
    # 한글 복합어는 띄어쓰기가 있을 수도 없을 수도 있다 (예: 질문의 "베샤멜소스"
    # vs 문서의 "베샤멜 소스" — 형태소 분석기도 사전에 없는 복합어는 붙여 쓴
    # 쪽을 하나의 토큰으로, 띄어 쓴 쪽을 별개 토큰들로 나눠버릴 수 있다).
    # BM25는 토큰이 정확히 일치해야 매칭되므로, 인접한 순수 한글 토큰 쌍을
    # 이어붙인 토큰을 추가로 넣어 두 표기 모두 매칭되게 한다.
    merged = [
        a + b
        for a, b in zip(tokens, tokens[1:])
        if _HANGUL_TOKEN_RE.match(a) and _HANGUL_TOKEN_RE.match(b)
    ]
    return tokens + merged


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


# mxbai-embed-large is an asymmetric retrieval model — it expects queries
# (not documents) prefixed with this instruction. Without it, query and
# document vectors aren't in a consistently comparable space, which was
# measured to actively invert rankings on this corpus (an unrelated chunk
# scored a *higher* raw cosine similarity than the actually-relevant one on
# a real query). Harmless no-op for other embedding models.
_MXBAI_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _embed_query_for_rerank(embedder, query: str, model_name: str | None) -> list:
    if model_name and "mxbai" in model_name.lower():
        query = _MXBAI_QUERY_PREFIX + query
    return embedder.embed_query(query)


@dataclass(frozen=True)
class Reranker:
    """Ollama 임베딩 기반 query-document cosine similarity reranker.

    hybrid 검색에서 re-scoring 후 relative-threshold 미달 문서를 필터링합니다.
    """

    threshold: float = DEFAULT_SCORE_THRESHOLD
    embedding_model: str | None = None
    absolute_floor: float = DEFAULT_ABSOLUTE_SIM_FLOOR

    def rerank(self, query: str, documents: list, k: int = DEFAULT_TOP_K) -> list:
        """query에 대한 문서별 cosine similarity를 재계산하고,
        상위 k개를 반환합니다.

        cosine similarity는 이 query의 candidate 집합 내에서 min-max로
        정규화한 뒤 threshold를 적용합니다 — 임베딩 모델에 따라 raw cosine
        값의 유효 범위가 좁게 압축되거나(예: 실측 0.55~0.85 사이에 무관한
        문서와 관련 문서가 뒤섞여 나옴) query마다 달라, 고정된 절대값
        threshold는 신뢰할 수 있는 필터가 되지 못한다. Query 단위 상대
        정규화는 이 문제와 무관하게 "이 후보군 안에서 상대적으로 얼마나
        유사한가"를 일관되게 측정한다.

        하지만 relative normalization만으로는 후보군 전체가 질문과 무관해도
        그 중 "가장 덜 무관한" 문서가 항상 normalized_sim=1.0을 받아 threshold를
        통과한다 — 완전히 동떨어진 질문(예: 회계 문서 코퍼스에 축구 질문)에도
        매번 근거 없는 Sources가 붙는 원인이었다. absolute_floor는 그 후보가
        애초에 최소한의 절대적 관련성을 갖는지 별도로 검증하는 이중 게이트다.

        Args:
            query: 검색 쿼리.
            documents: rerank 대상 Document 목록 (RRF fusion 결과).
            k: 반환 최대 문서 수.

        Returns:
            정규화된 유사도 기준 threshold 이상 *이면서* raw cosine similarity가
            absolute_floor 이상인 문서 중 combined score 상위 k개.
        """
        if not documents:
            return []

        from app.rag.embeddings import create_embeddings

        embedder = create_embeddings(model=self.embedding_model)
        query_vec = _embed_query_for_rerank(embedder, query, self.embedding_model)

        # RRF fusion scores reflect BM25/vector rank agreement — a strong,
        # precise relevance signal (e.g. exact keyword matches) that pure
        # cosine similarity can miss or be outscored by a generically
        # similar-sounding but wrong document. Sorting by cosine similarity
        # alone (as this used to) discarded that signal entirely once it
        # reached reranking. Normalize RRF score to 0-1 and blend it in
        # instead.
        rrf_scores = [item[1] for item in documents if isinstance(item, tuple)]
        max_rrf = max(rrf_scores) if rrf_scores else 0.0

        # Batched embed_documents() calls instead of one HTTP round-trip per
        # candidate — with a ~30-doc candidate pool (see candidate_pool_size)
        # the per-doc version meant 30 sequential Ollama calls per query,
        # dominating end-to-end RAG latency. Chunked at the same batch size
        # as vectordb.py's index-build path, for the same reason (Ollama's
        # embed endpoint drops the connection on overly large batches).
        docs_and_scores = []
        contents = []
        for item in documents:
            if isinstance(item, tuple):
                doc, rrf_score = item
                content = doc.page_content
            else:
                doc, rrf_score = item, 0.0
                content = item.page_content
            docs_and_scores.append((doc, rrf_score))
            contents.append(content)

        doc_vecs = []
        for i in range(0, len(contents), EMBED_BATCH_SIZE):
            doc_vecs.extend(embedder.embed_documents(contents[i : i + EMBED_BATCH_SIZE]))

        candidates = [
            (doc, rrf_score, _cosine_similarity(query_vec, doc_vec))
            for (doc, rrf_score), doc_vec in zip(docs_and_scores, doc_vecs)
        ]

        sims = [sim for _doc, _rrf, sim in candidates]
        sim_min, sim_max = min(sims), max(sims)
        sim_span = sim_max - sim_min

        scored = []
        for doc, rrf_score, sim in candidates:
            normalized_sim = ((sim - sim_min) / sim_span) if sim_span else 1.0
            if normalized_sim >= self.threshold and sim >= self.absolute_floor:
                normalized_rrf = (rrf_score / max_rrf) if max_rrf else 0.0
                combined = normalized_sim + normalized_rrf
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
        bm25_docs = self.bm25_retriever.retrieve(query, k=candidate_pool_size(self.k))

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

        # reranking 수행 (전체 fusion 후보 대상). 후보 전체가 필터링돼 빈
        # 리스트가 나오는 것은 버그가 아니라 "이 질문에 답할 근거 문서가
        # 없다"는 유효한 결과다 — 예전엔 이 경우 RRF top-k로 되돌아가
        # absolute_floor/threshold가 걸러낸 무관한 문서를 그대로 반환했다.
        if self.reranker is not None and candidates:
            return self.reranker.rerank(query, candidates, k=self.k)

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


def create_hybrid_retriever(
    vector_retriever,
    documents,
    k=DEFAULT_TOP_K,
    score_threshold=DEFAULT_SCORE_THRESHOLD,
    embedding_model=None,
):
    """FAISS retriever + BM25 retriever를 RRF fusion + reranking으로 결합합니다.

    Args:
        vector_retriever: FAISS.as_retriever() 결과.
        documents: 원본 Document 목록 (BM25 학습용).
        k: 반환 최대 문서 수.
        score_threshold: 정규화된 유사도 최소 임계값 (default: 0.3, 0-1 스케일).
        embedding_model: reranker가 cosine similarity 계산에 사용할 임베딩
            모델. vector_retriever를 만든 것과 같은 모델이어야 query/document
            벡터가 같은 공간에 있다는 게 보장된다 — 생략하면 embeddings.py의
            기본값(nomic-embed-text)으로 조용히 갈아타 버려서, FAISS 인덱스가
            다른 모델(예: mxbai-embed-large)로 만들어졌을 때 reranker의
            유사도 계산이 의미 없는 값이 된다.

    Returns:
        HybridRetriever 인스턴스.
    """
    bm25_retriever = BM25Retriever(documents)
    reranker = Reranker(threshold=score_threshold, embedding_model=embedding_model)
    return HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        k=k,
        reranker=reranker,
        score_threshold=score_threshold,
    )
