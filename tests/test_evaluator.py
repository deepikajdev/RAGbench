"""Unit tests for dataset loading, evaluator pipeline, and report generation."""

import unittest
from ragbench.core.models import EvaluationQuery
from ragbench.datasets.loader import load_sample_dataset
from ragbench.evaluators.retrieval_evaluator import RetrievalEvaluator
from ragbench.formatters.table import format_evaluation_report


class TestEvaluatorPipeline(unittest.TestCase):
    """Test suite covering the end-to-end evaluation pipeline."""

    def test_load_sample_dataset(self) -> None:
        """Sample dataset must load 5 valid queries and 2 candidate rankings."""
        queries, rankings = load_sample_dataset()
        self.assertEqual(len(queries), 5)
        self.assertIn("Lexical_BM25_Baseline", rankings)
        self.assertIn("Dense_Semantic_Baseline", rankings)

        # Verify query structure
        first_q = queries[0]
        self.assertIsInstance(first_q, EvaluationQuery)
        self.assertEqual(first_q.query_id, "q_01")
        self.assertGreater(len(first_q.relevant_doc_ids), 0)

    def test_evaluator_runs_and_produces_valid_summaries(self) -> None:
        """RetrievalEvaluator should compute mean Recall@K for each system."""
        queries, rankings = load_sample_dataset()
        evaluator = RetrievalEvaluator(k_values=[1, 3, 5])
        report = evaluator.evaluate(queries, rankings)

        self.assertEqual(report.k_values, [1, 3, 5])
        self.assertEqual(len(report.summaries), 2)

        for system_name, summary in report.summaries.items():
            self.assertEqual(summary.num_queries, 5)
            self.assertIn(1, summary.mean_recall_at_k)
            self.assertIn(3, summary.mean_recall_at_k)
            self.assertIn(5, summary.mean_recall_at_k)
            # Scores must be bounded within [0.0, 1.0]
            for k in [1, 3, 5]:
                score = summary.mean_recall_at_k[k]
                self.assertTrue(0.0 <= score <= 1.0, f"Score {score} out of bounds for K={k}")

    def test_evaluator_invalid_k_raises(self) -> None:
        """Evaluator initialization with invalid k values must raise ValueError."""
        with self.assertRaises(ValueError):
            RetrievalEvaluator(k_values=[0, 3])

        with self.assertRaises(ValueError):
            RetrievalEvaluator(k_values=[-1, 5])

    def test_format_evaluation_report_generates_text(self) -> None:
        """format_evaluation_report must output a readable report string containing system names."""
        queries, rankings = load_sample_dataset()
        evaluator = RetrievalEvaluator(k_values=[1, 3, 5])
        report = evaluator.evaluate(queries, rankings)

        text_report = format_evaluation_report(report)
        self.assertIn("RAGBENCH RETRIEVAL EVALUATION REPORT", text_report)
        self.assertIn("Lexical_BM25_Baseline", text_report)
        self.assertIn("Dense_Semantic_Baseline", text_report)
        self.assertIn("Mean R@1", text_report)
        self.assertIn("Mean R@3", text_report)
        self.assertIn("Mean R@5", text_report)


if __name__ == "__main__":
    unittest.main()
