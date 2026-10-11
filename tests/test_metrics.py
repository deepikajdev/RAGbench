"""Unit tests for RAG retrieval metrics and edge cases."""

import unittest
from ragbench.metrics.retrieval import (
    recall_at_k,
    mean_recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
)


class TestRecallAtK(unittest.TestCase):
    """Test suite covering Recall@K metric behavior and boundary edge cases."""

    def test_recall_at_k_perfect_retrieval(self) -> None:
        """When all ground truth documents are present in top K, recall is 1.0."""
        retrieved = ["doc_A", "doc_B", "doc_C"]
        relevant = {"doc_A", "doc_B"}
        self.assertEqual(recall_at_k(retrieved, relevant, k=2), 1.0)
        self.assertEqual(recall_at_k(retrieved, relevant, k=3), 1.0)

    def test_recall_at_k_partial_retrieval(self) -> None:
        """When only a fraction of relevant docs are found, recall is proportional."""
        retrieved = ["doc_A", "doc_X", "doc_Y"]
        relevant = {"doc_A", "doc_B"}
        # Only doc_A is found among 2 relevant documents
        self.assertEqual(recall_at_k(retrieved, relevant, k=2), 0.5)

    def test_recall_at_k_zero_hits(self) -> None:
        """When no relevant documents are in top K, recall is 0.0."""
        retrieved = ["doc_X", "doc_Y", "doc_Z"]
        relevant = {"doc_A", "doc_B"}
        self.assertEqual(recall_at_k(retrieved, relevant, k=3), 0.0)

    def test_recall_at_k_cutoff_boundary(self) -> None:
        """Relevant document past the cutoff depth K must not count toward Recall@K."""
        # Relevant doc_C is at rank 3 (0-indexed 2)
        retrieved = ["doc_X", "doc_Y", "doc_C"]
        relevant = {"doc_C"}

        self.assertEqual(recall_at_k(retrieved, relevant, k=2), 0.0)
        self.assertEqual(recall_at_k(retrieved, relevant, k=3), 1.0)

    def test_recall_at_k_empty_retrieved_list(self) -> None:
        """Empty retrieved results list should return 0.0."""
        retrieved: list[str] = []
        relevant = {"doc_A", "doc_B"}
        self.assertEqual(recall_at_k(retrieved, relevant, k=5), 0.0)

    def test_recall_at_k_invalid_k_zero_raises(self) -> None:
        """Cutoff K=0 must raise ValueError."""
        with self.assertRaises(ValueError):
            recall_at_k(["doc_A"], {"doc_A"}, k=0)

    def test_recall_at_k_invalid_k_negative_raises(self) -> None:
        """Cutoff K < 0 must raise ValueError."""
        with self.assertRaises(ValueError):
            recall_at_k(["doc_A"], {"doc_A"}, k=-3)

    def test_recall_at_k_invalid_k_non_int_raises(self) -> None:
        """Non-integer K must raise ValueError."""
        with self.assertRaises(ValueError):
            recall_at_k(["doc_A"], {"doc_A"}, k=1.5)  # type: ignore

        with self.assertRaises(ValueError):
            recall_at_k(["doc_A"], {"doc_A"}, k=True)  # type: ignore

    def test_recall_at_k_empty_relevant_docs_raises(self) -> None:
        """Evaluating with no ground truth documents must raise ValueError."""
        with self.assertRaises(ValueError):
            recall_at_k(["doc_A"], set(), k=1)

    def test_recall_at_k_duplicate_retrieved_documents(self) -> None:
        """Duplicate document IDs in retrieved list do not inflate recall."""
        retrieved = ["doc_A", "doc_A", "doc_B"]
        relevant = {"doc_A", "doc_B", "doc_C"}
        # Top 2 has only doc_A (1 unique relevant hit out of 3 total)
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=2), 1 / 3)
        # Top 3 has doc_A and doc_B (2 unique relevant hits out of 3 total)
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=3), 2 / 3)

    def test_recall_at_k_k_larger_than_retrieved(self) -> None:
        """When K exceeds the number of retrieved items, evaluate all available."""
        retrieved = ["doc_A"]
        relevant = {"doc_A", "doc_B"}
        self.assertEqual(recall_at_k(retrieved, relevant, k=10), 0.5)

    def test_mean_recall_at_k(self) -> None:
        """Mean recall calculates the average correctly, and handles empty input."""
        self.assertEqual(mean_recall_at_k([]), 0.0)
        self.assertAlmostEqual(mean_recall_at_k([0.5, 1.0, 0.0]), 0.5)


class TestReciprocalRank(unittest.TestCase):
    """Test suite covering Reciprocal Rank (RR) and Mean Reciprocal Rank (MRR)."""

    def test_relevant_document_at_rank_1(self) -> None:
        """First relevant document at rank 1 yields RR = 1.0."""
        retrieved = ["doc_A", "doc_B", "doc_C"]
        relevant = {"doc_A"}
        self.assertEqual(reciprocal_rank(retrieved, relevant), 1.0)

    def test_relevant_document_at_rank_2(self) -> None:
        """First relevant document at rank 2 yields RR = 0.5 (1/2)."""
        retrieved = ["doc_X", "doc_A", "doc_B"]
        relevant = {"doc_A", "doc_B"}
        self.assertEqual(reciprocal_rank(retrieved, relevant), 0.5)

    def test_relevant_document_at_rank_3(self) -> None:
        """First relevant document at rank 3 yields RR = 1/3 (~0.3333)."""
        retrieved = ["doc_X", "doc_Y", "doc_A"]
        relevant = {"doc_A"}
        self.assertAlmostEqual(reciprocal_rank(retrieved, relevant), 1 / 3)

    def test_no_relevant_documents_retrieved(self) -> None:
        """When no relevant documents appear in the retrieved list, RR is 0.0."""
        retrieved = ["doc_X", "doc_Y", "doc_Z"]
        relevant = {"doc_A", "doc_B"}
        self.assertEqual(reciprocal_rank(retrieved, relevant), 0.0)

    def test_empty_retrieved_list(self) -> None:
        """Empty retrieved list returns 0.0."""
        retrieved: list[str] = []
        relevant = {"doc_A"}
        self.assertEqual(reciprocal_rank(retrieved, relevant), 0.0)

    def test_reciprocal_rank_with_cutoff_k(self) -> None:
        """Cutoff k restricts the search depth: rank 3 excluded when k=2."""
        retrieved = ["doc_X", "doc_Y", "doc_A"]
        relevant = {"doc_A"}
        self.assertEqual(reciprocal_rank(retrieved, relevant, k=2), 0.0)
        self.assertAlmostEqual(reciprocal_rank(retrieved, relevant, k=3), 1 / 3)

    def test_invalid_k_raises(self) -> None:
        """Cutoff k <= 0 or non-integer must raise ValueError."""
        with self.assertRaises(ValueError):
            reciprocal_rank(["doc_A"], {"doc_A"}, k=0)

        with self.assertRaises(ValueError):
            reciprocal_rank(["doc_A"], {"doc_A"}, k=-2)

        with self.assertRaises(ValueError):
            reciprocal_rank(["doc_A"], {"doc_A"}, k=1.5)  # type: ignore

        with self.assertRaises(ValueError):
            reciprocal_rank(["doc_A"], {"doc_A"}, k=True)  # type: ignore

    def test_empty_relevant_docs_raises(self) -> None:
        """Empty relevant documents set must raise ValueError."""
        with self.assertRaises(ValueError):
            reciprocal_rank(["doc_A"], set())

    def test_mean_reciprocal_rank(self) -> None:
        """Mean reciprocal rank computes arithmetic average, and handles empty list."""
        self.assertEqual(mean_reciprocal_rank([]), 0.0)
        # Average of 1.0 (rank 1), 0.5 (rank 2), and 0.0 (no hit) = 1.5 / 3 = 0.5
        self.assertAlmostEqual(mean_reciprocal_rank([1.0, 0.5, 0.0]), 0.5)


if __name__ == "__main__":
    unittest.main()

