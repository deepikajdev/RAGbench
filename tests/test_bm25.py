"""Unit tests for BM25 keyword search retriever."""

import unittest
from ragbench.core.models import Document
from ragbench.datasets.loader import load_sample_corpus
from ragbench.retrievers.bm25 import BM25Retriever


class TestBM25Retriever(unittest.TestCase):
    """Test suite covering BM25 ranking mechanics and edge cases."""

    def setUp(self) -> None:
        """Create a tiny, controlled document corpus for testing."""
        self.docs = [
            Document(
                doc_id="doc_alpha",
                title="Python Programming Basics",
                content="Python is an interpreted, high-level programming language with dynamic semantics.",
            ),
            Document(
                doc_id="doc_beta",
                title="Database Systems and SQL",
                content="Relational databases use SQL to query structured data tables with indexes.",
            ),
            Document(
                doc_id="doc_gamma",
                title="Advanced Python Generators",
                content="Python generators and iterators allow lazy evaluation of sequences in Python.",
            ),
        ]
        self.retriever = BM25Retriever.from_documents(self.docs)

    def test_ranking_relevance_and_scores(self) -> None:
        """Query 'Python generators' must rank doc_gamma highest due to exact matches and frequency."""
        results = self.retriever.search("Python generators", top_k=3)

        self.assertGreater(len(results), 0)
        top_match = results[0]
        self.assertEqual(top_match.doc_id, "doc_gamma")
        self.assertEqual(top_match.rank, 1)
        self.assertGreater(top_match.score, 0.0)

        # doc_alpha also mentions Python, so it should rank #2 with a lower score
        if len(results) > 1:
            self.assertEqual(results[1].doc_id, "doc_alpha")
            self.assertGreater(top_match.score, results[1].score)

    def test_empty_query_returns_empty_list(self) -> None:
        """Empty string or whitespace-only queries must return an empty list."""
        self.assertEqual(self.retriever.search(""), [])
        self.assertEqual(self.retriever.search("   "), [])
        self.assertEqual(self.retriever.search("!@#$%^&*()"), [])
        self.assertEqual(self.retriever.retrieve_ids(""), [])

    def test_missing_matches_returns_empty_list(self) -> None:
        """Query with terms not present anywhere in the corpus returns an empty list."""
        results = self.retriever.search("quantum superconductivity astrophysics")
        self.assertEqual(results, [])

    def test_case_insensitivity_and_punctuation(self) -> None:
        """Search should be case-insensitive and handle punctuation cleanly."""
        results_lower = self.retriever.search("python")
        results_upper = self.retriever.search("PYTHON!")
        self.assertEqual(len(results_lower), len(results_upper))
        self.assertEqual(results_lower[0].doc_id, results_upper[0].doc_id)

    def test_retrieve_ids_matches_search_order(self) -> None:
        """retrieve_ids must return the exact sequence of IDs produced by search."""
        search_results = self.retriever.search("databases SQL", top_k=2)
        id_results = self.retriever.retrieve_ids("databases SQL", top_k=2)

        expected_ids = [doc.doc_id for doc in search_results]
        self.assertEqual(id_results, expected_ids)
        self.assertEqual(id_results[0], "doc_beta")

    def test_invalid_parameters_raise(self) -> None:
        """Invalid k1, b, or top_k values must raise ValueError."""
        with self.assertRaises(ValueError):
            BM25Retriever(k1=0.0)

        with self.assertRaises(ValueError):
            BM25Retriever(k1=-1.0)

        with self.assertRaises(ValueError):
            BM25Retriever(b=-0.1)

        with self.assertRaises(ValueError):
            BM25Retriever(b=1.2)

        with self.assertRaises(ValueError):
            self.retriever.search("test", top_k=0)

    def test_empty_corpus_handling(self) -> None:
        """Indexing an empty corpus should not raise division by zero and return empty results."""
        empty_retriever = BM25Retriever.from_documents([])
        self.assertEqual(empty_retriever.search("anything"), [])
        self.assertEqual(empty_retriever.retrieve_ids("anything"), [])

    def test_sample_corpus_real_search(self) -> None:
        """BM25 indexing on the bundled sample corpus retrieves relevant RAG documents."""
        corpus = load_sample_corpus()
        self.assertGreater(len(corpus), 0)

        sample_retriever = BM25Retriever.from_documents(corpus)
        results = sample_retriever.search("chunk overlap text splitting", top_k=3)

        self.assertGreater(len(results), 0)
        # doc_chunking_overlap specifically addresses chunk overlap in text splitting
        self.assertEqual(results[0].doc_id, "doc_chunking_overlap")
    def test_tie_breaking_deterministic(self) -> None:
        """Documents with identical BM25 scores must be ordered deterministically by doc_id."""
        docs = [
            Document(doc_id="doc_zulu", content="machine learning models algorithms"),
            Document(doc_id="doc_alpha", content="machine learning models algorithms"),
        ]
        retriever = BM25Retriever.from_documents(docs)
        results = retriever.search("machine learning", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].score, results[1].score)
        # Alphabetical tie-breaking ensures doc_alpha precedes doc_zulu
        self.assertEqual(results[0].doc_id, "doc_alpha")
        self.assertEqual(results[0].rank, 1)
        self.assertEqual(results[1].doc_id, "doc_zulu")
        self.assertEqual(results[1].rank, 2)

    def test_empty_document_in_corpus(self) -> None:
        """Documents with empty content should be safely indexed and never match queries."""
        docs = [
            Document(doc_id="doc_empty", content="", title=""),
            Document(doc_id="doc_valid", content="information retrieval indexing"),
        ]
        retriever = BM25Retriever.from_documents(docs)
        results = retriever.search("retrieval", top_k=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].doc_id, "doc_valid")

    def test_top_k_limits(self) -> None:
        """Results must strictly respect the requested top_k cutoff."""
        results_top_1 = self.retriever.search("python", top_k=1)
        self.assertEqual(len(results_top_1), 1)

        results_top_2 = self.retriever.search("python", top_k=2)
        self.assertEqual(len(results_top_2), 2)

    def test_query_term_repetition(self) -> None:
        """Repeating query terms should execute safely without crashing."""
        results = self.retriever.search("python python generators", top_k=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].doc_id, "doc_gamma")

    def test_pipeline_integration_with_evaluator(self) -> None:
        """Verify real BM25 retriever integrates with RetrievalEvaluator and achieves valid metrics."""
        from ragbench.datasets.loader import load_sample_dataset
        from ragbench.evaluators.retrieval_evaluator import RetrievalEvaluator

        corpus = load_sample_corpus()
        queries, rankings = load_sample_dataset()
        bm25 = BM25Retriever.from_documents(corpus)

        real_rankings = {
            q.query_id: bm25.retrieve_ids(q.query_text, top_k=5)
            for q in queries
        }
        rankings["Real_BM25_Retriever"] = real_rankings

        evaluator = RetrievalEvaluator(k_values=[1, 3, 5])
        report = evaluator.evaluate(queries, rankings)

        self.assertIn("Real_BM25_Retriever", report.summaries)
        summary = report.summaries["Real_BM25_Retriever"]
        self.assertEqual(summary.num_queries, 5)
        # Check that MRR and Recall@5 are in valid range
        self.assertGreater(summary.mrr, 0.0)
        self.assertLessEqual(summary.mrr, 1.0)
        self.assertEqual(summary.mean_recall_at_k[5], 1.0)

        # Baseline metrics must remain unchanged (regression check)
        self.assertAlmostEqual(report.summaries["Lexical_BM25_Baseline"].mrr, 0.8667, places=3)
        self.assertAlmostEqual(report.summaries["Dense_Semantic_Baseline"].mrr, 0.8000, places=3)


if __name__ == "__main__":
    unittest.main()
