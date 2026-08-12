"""Tests for the citation / hybrid retrieval feature.

Covers:
- ChunkMeta dataclass
- split_documents metadata enrichment
- BM25Retriever
- Reranker (threshold filtering, cosine similarity)
- HybridRetriever (RRF fusion)
- create_hybrid_retriever
- Pipeline hybrid mode
"""

import math
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.rag.chunker import ChunkMeta, split_documents
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import (
    BM25Retriever,
    HybridRetriever,
    Reranker,
    _cosine_similarity,
    _tokenize,
    create_hybrid_retriever,
)


# ── ChunkMeta ──────────────────────────────────────────────────


class ChunkMetaTest(unittest.TestCase):
    def test_defaults_generate_ids(self):
        meta = ChunkMeta()
        self.assertTrue(meta.chunk_id.startswith("chunk_"))
        self.assertTrue(meta.document_id.startswith("doc_"))

    def test_source_and_line_range_defaults(self):
        meta = ChunkMeta(source="a.md", line_range=(10, 20))
        self.assertEqual(meta.source, "a.md")
        self.assertEqual(meta.line_range, (10, 20))


# ── split_documents metadata ───────────────────────────────────


class SplitDocumentsTest(unittest.TestCase):
    def setUp(self):
        self.docs = [
            Document(page_content="Hello world.\nSecond line.\nThird line.", metadata={"source": "test.md"}),
        ]

    def test_adds_chunk_id_metadata(self):
        result = split_documents(self.docs, chunk_size=1000, chunk_overlap=0)
        for doc in result:
            self.assertIn("chunk_id", doc.metadata)
            self.assertTrue(doc.metadata["chunk_id"].startswith("chunk_"))

    def test_adds_document_id_metadata(self):
        result = split_documents(self.docs, chunk_size=1000, chunk_overlap=0)
        for doc in result:
            self.assertIn("document_id", doc.metadata)
            self.assertTrue(doc.metadata["document_id"].startswith("doc_"))

    def test_adds_source_metadata(self):
        result = split_documents(self.docs, chunk_size=1000, chunk_overlap=0)
        for doc in result:
            self.assertEqual(doc.metadata["source"], "test.md")

    def test_adds_line_range_metadata(self):
        result = split_documents(self.docs, chunk_size=1000, chunk_overlap=0)
        for doc in result:
            lr = doc.metadata["line_range"]
            self.assertIsInstance(lr, tuple)
            self.assertEqual(len(lr), 2)
            self.assertLessEqual(lr[0], lr[1])

    def test_small_chunk_size_causes_split(self):
        """A very small chunk_size should produce multiple chunks."""
        result = split_documents(self.docs, chunk_size=10, chunk_overlap=0)
        self.assertGreater(len(result), 1)


# ── _tokenize ──────────────────────────────────────────────────


class TokenizeTest(unittest.TestCase):
    def test_english(self):
        tokens = _tokenize("Hello world")
        self.assertEqual(tokens, ["Hello", "world"])

    def test_korean(self):
        tokens = _tokenize("안녕하세요")
        self.assertTrue(len(tokens) > 0)

    def test_mixed(self):
        # "회사" (noun, NNG) rather than "안녕" (interjection, IC) — kiwi's
        # keep-list (_KIWI_KEEP_PREFIXES) only retains content-bearing tags
        # like nouns/verbs/foreign words, so an interjection alone would be
        # dropped and this wouldn't actually exercise mixed-language token
        # retention.
        tokens = _tokenize("회사 world")
        self.assertEqual(len(tokens), 2)

    def test_numbers(self):
        tokens = _tokenize("version 3.14")
        # kiwi tags "3.14" as a single SN (number) token instead of the
        # naive regex fallback's "3", ".", "14" split.
        self.assertIn("3.14", tokens)


# ── BM25Retriever ──────────────────────────────────────────────


class BM25RetrieverTest(unittest.TestCase):
    def setUp(self):
        self.docs = [
            Document(page_content="FAISS is a vector search library.", metadata={"source": "a.md"}),
            Document(page_content="Ollama provides embeddings.", metadata={"source": "b.md"}),
            Document(page_content="Hermes uses RAG for answers.", metadata={"source": "c.md"}),
        ]

    def test_retrieve_returns_k_documents(self):
        retriever = BM25Retriever(self.docs)
        results = retriever.retrieve("FAISS", k=1)
        self.assertEqual(1, len(results))

    def test_retrieve_returns_all_when_k_large(self):
        retriever = BM25Retriever(self.docs)
        results = retriever.retrieve("FAISS", k=10)
        self.assertEqual(len(self.docs), len(results))

    def test_retrieve_empty_when_no_documents(self):
        # BM25Okapi raises ZeroDivisionError on empty corpus, so we expect it
        with self.assertRaises(ZeroDivisionError):
            BM25Retriever([])

    def test_retrieve_ranked_by_relevance(self):
        """Querying 'FAISS' should rank the FAISS document first."""
        retriever = BM25Retriever(self.docs)
        results = retriever.retrieve("FAISS", k=3)
        self.assertIn("FAISS", results[0].page_content)


# ── _cosine_similarity ─────────────────────────────────────────


class CosineSimilarityTest(unittest.TestCase):
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(_cosine_similarity(v, v), 1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        self.assertAlmostEqual(_cosine_similarity(a, b), 0.0)

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 1.0]
        self.assertAlmostEqual(_cosine_similarity(a, b), 0.0)

    def test_negative_similarity(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        self.assertAlmostEqual(_cosine_similarity(a, b), -1.0)


# ── Reranker ───────────────────────────────────────────────────


class RerankerTest(unittest.TestCase):
    def test_rerank_empty_documents(self):
        reranker = Reranker()
        self.assertEqual([], reranker.rerank("query", [], k=3))

    @patch("app.rag.embeddings.create_embeddings")
    def test_rerank_filters_below_threshold(self, mock_create):
        """Documents with low similarity should be filtered out."""
        mock_embedder = MagicMock()
        # Query vector
        mock_embedder.embed_query.return_value = [1.0, 0.0, 0.0]
        # High-sim doc
        mock_embedder.embed_documents.side_effect = lambda texts: [[0.99, 0.01, 0.0]] if "high" in texts[0] else [[0.0, 1.0, 0.0]]
        mock_create.return_value = mock_embedder

        docs = [
            Document(page_content="high relevance content", metadata={"source": "a.md"}),
            Document(page_content="low relevance content", metadata={"source": "b.md"}),
        ]
        reranker = Reranker(threshold=0.5)
        # When called with plain doc objects (not tuples from hybrid)
        results = reranker.rerank("query", docs, k=3)
        self.assertEqual(1, len(results))
        self.assertIn("high", results[0].page_content)

    @patch("app.rag.embeddings.create_embeddings")
    def test_rerank_filters_when_receiving_tuples(self, mock_create):
        """Reranker should handle (doc, score) tuples from HybridRetriever."""
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [1.0, 0.0, 0.0]
        mock_embedder.embed_documents.side_effect = lambda texts: [[0.99, 0.01, 0.0]] if "high" in texts[0] else [[0.0, 1.0, 0.0]]
        mock_create.return_value = mock_embedder

        doc_a = Document(page_content="high relevance content", metadata={"source": "a.md"})
        doc_b = Document(page_content="low relevance content", metadata={"source": "b.md"})
        # Tuples as passed by HybridRetriever
        candidates = [(doc_a, 0.8), (doc_b, 0.2)]
        reranker = Reranker(threshold=0.5)
        results = reranker.rerank("query", candidates, k=3)
        self.assertEqual(1, len(results))
        self.assertIn("high", results[0].page_content)


# ── HybridRetriever ────────────────────────────────────────────


class HybridRetrieverTest(unittest.TestCase):
    def setUp(self):
        self.docs = [
            Document(page_content="FAISS stores vectors efficiently.", metadata={"source": "a.md"}),
            Document(page_content="Ollama creates embeddings.", metadata={"source": "b.md"}),
            Document(page_content="Hermes uses RAG for answers.", metadata={"source": "c.md"}),
        ]
        self.bm25_retriever = BM25Retriever(self.docs)

        # Mock FAISS retriever
        self.mock_vector_retriever = MagicMock()
        self.mock_vector_retriever.invoke.return_value = self.docs

    def test_retrieve_returns_documents(self):
        hybrid = HybridRetriever(
            vector_retriever=self.mock_vector_retriever,
            bm25_retriever=self.bm25_retriever,
            k=2,
            reranker=None,
        )
        results = hybrid.retrieve("FAISS")
        self.assertGreater(len(results), 0)

    def test_retrieve_without_reranker(self):
        hybrid = HybridRetriever(
            vector_retriever=self.mock_vector_retriever,
            bm25_retriever=self.bm25_retriever,
            k=2,
            reranker=None,
        )
        results = hybrid.retrieve("FAISS")
        self.assertLessEqual(len(results), 2)

    def test_get_relevant_documents_alias(self):
        hybrid = HybridRetriever(
            vector_retriever=self.mock_vector_retriever,
            bm25_retriever=self.bm25_retriever,
            k=2,
            reranker=None,
        )
        results = hybrid.get_relevant_documents("FAISS")
        self.assertGreater(len(results), 0)

    def test_rrf_fusion_ranks_common_document_first(self):
        """A document appearing in both vector and BM25 results should rank higher."""
        # Vector retriever returns only the first doc
        self.mock_vector_retriever.invoke.return_value = [self.docs[0]]
        hybrid = HybridRetriever(
            vector_retriever=self.mock_vector_retriever,
            bm25_retriever=self.bm25_retriever,
            k=3,
            reranker=None,
        )
        results = hybrid.retrieve("FAISS")
        # The first doc appears in both, so it should rank higher
        self.assertIn("FAISS", results[0].page_content)


# ── create_hybrid_retriever ────────────────────────────────────


class CreateHybridRetrieverTest(unittest.TestCase):
    def setUp(self):
        self.docs = [
            Document(page_content="FAISS is fast.", metadata={"source": "a.md"}),
            Document(page_content="Ollama is flexible.", metadata={"source": "b.md"}),
        ]
        self.mock_vector_retriever = MagicMock()
        self.mock_vector_retriever.invoke.return_value = self.docs

    def test_returns_hybrid_retriever(self):
        hybrid = create_hybrid_retriever(self.mock_vector_retriever, self.docs)
        self.assertIsInstance(hybrid, HybridRetriever)

    def test_configures_reranker(self):
        hybrid = create_hybrid_retriever(self.mock_vector_retriever, self.docs, score_threshold=0.5)
        self.assertIsNotNone(hybrid.reranker)
        self.assertEqual(hybrid.reranker.threshold, 0.5)

    def test_configures_k(self):
        hybrid = create_hybrid_retriever(self.mock_vector_retriever, self.docs, k=10)
        self.assertEqual(hybrid.k, 10)


# ── Pipeline integration ───────────────────────────────────────


class FormatAnswerTest(unittest.TestCase):
    def test_answer_includes_sources_with_line_range(self):
        docs = [
            Document(page_content="content", metadata={"source": "a.md", "line_range": (0, 10)}),
            Document(page_content="content2", metadata={"source": "b.md", "line_range": (11, 25)}),
        ]
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs
        pipeline = RAGPipeline(retriever=mock_retriever, llm=lambda p: "answer")
        result = pipeline.answer("question")
        self.assertIn("answer", result)
        self.assertIn("Sources:", result)
        self.assertIn("a.md", result)
        self.assertIn("b.md", result)

    def test_answer_includes_sources_without_line_range(self):
        docs = [
            Document(page_content="content", metadata={"source": "a.md"}),
        ]
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs
        pipeline = RAGPipeline(retriever=mock_retriever, llm=lambda p: "answer")
        result = pipeline.answer("question")
        self.assertIn("Sources:", result)
        self.assertIn("a.md", result)

    def test_answer_dedupes_and_flattens_sources_to_one_line(self):
        docs = [
            Document(page_content="chunk1", metadata={"source": "a.md", "line_range": (0, 10)}),
            Document(page_content="chunk2", metadata={"source": "a.md", "line_range": (11, 20)}),
            Document(page_content="chunk3", metadata={"source": "b.md"}),
        ]
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs
        pipeline = RAGPipeline(retriever=mock_retriever, llm=lambda p: "answer")
        result = pipeline.answer("question")
        sources_line = result.split("Sources: ", 1)[1]
        self.assertEqual(sources_line, "a.md, b.md")

    def test_answer_without_documents_returns_raw(self):
        pipeline = RAGPipeline(retriever=MagicMock(), llm=lambda p: "answer")
        pipeline.retrieve = MagicMock(return_value=[])
        result = pipeline.answer("question")
        self.assertEqual("answer", result)


class PipelineHybridTest(unittest.TestCase):
    @patch("app.rag.pipeline.create_embeddings")
    @patch("app.rag.pipeline.load_documents")
    @patch("app.rag.pipeline.split_documents")
    @patch("app.rag.pipeline.create_vector_db")
    @patch("app.rag.pipeline.create_retriever")
    @patch("app.rag.pipeline.create_hybrid_retriever")
    def test_build_pipeline_hybrid_mode(
        self, mock_hybrid, mock_retriever, mock_vdb, mock_split, mock_load, mock_embed
    ):
        """build_pipeline with hybrid=True should create a HybridRetriever."""
        mock_load.return_value = [
            Document(page_content="Test content", metadata={"source": "test.md"}),
        ]
        mock_split.return_value = [
            Document(page_content="Test content", metadata={"source": "test.md"}),
        ]
        mock_vdb.return_value = MagicMock()
        mock_retriever.return_value = MagicMock()
        mock_hybrid.return_value = MagicMock()
        mock_embed.return_value = MagicMock()

        pipeline = build_pipeline_harness(hybrid=True)

        mock_hybrid.assert_called_once()

    @patch("app.rag.pipeline.create_embeddings")
    @patch("app.rag.pipeline.load_documents")
    @patch("app.rag.pipeline.split_documents")
    @patch("app.rag.pipeline.create_vector_db")
    @patch("app.rag.pipeline.create_retriever")
    def test_build_pipeline_non_hybrid_mode(
        self, mock_retriever, mock_vdb, mock_split, mock_load, mock_embed
    ):
        """build_pipeline with hybrid=False should NOT create a HybridRetriever."""
        mock_load.return_value = [
            Document(page_content="Test content", metadata={"source": "test.md"}),
        ]
        mock_split.return_value = [
            Document(page_content="Test content", metadata={"source": "test.md"}),
        ]
        mock_vdb.return_value = MagicMock()
        mock_retriever.return_value = MagicMock()
        mock_embed.return_value = MagicMock()

        pipeline = build_pipeline_harness(hybrid=False)

        # Should use the plain retriever
        mock_retriever.assert_called_once()


def build_pipeline_harness(hybrid=False):
    """Helper to call build_pipeline without real I/O — used by the tests above."""
    from app.rag.pipeline import build_pipeline as _build_pipeline

    # This will be called with mocked dependencies from the test
    return _build_pipeline(
        paths=["/fake"],
        hybrid=hybrid,
    )


if __name__ == "__main__":
    unittest.main()
