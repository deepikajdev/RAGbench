"""Unit tests for the RAGBench FastAPI backend."""

import unittest
from fastapi.testclient import TestClient

from ragbench.api.app import create_app


class TestFastAPIBackend(unittest.TestCase):
    """Test suite for RAGBench FastAPI REST endpoints."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the FastAPI TestClient once for the suite."""
        cls.app = create_app()
        cls.client = TestClient(cls.app)

    def test_health_endpoint(self) -> None:
        """GET /health must return 200 OK with corpus and version metadata."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["version"], "0.1.0")
        self.assertEqual(data["service"], "RAGBench Evaluation API")
        self.assertTrue(data["corpus_loaded"])
        self.assertEqual(data["corpus_doc_count"], 19)
        self.assertIn("Real_BM25_Retriever", data["available_retrievers"])

    def test_list_datasets_endpoint(self) -> None:
        """GET /api/datasets must list the bundled sample benchmark dataset."""
        response = self.client.get("/api/datasets")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("datasets", data)
        datasets = data["datasets"]
        self.assertGreaterEqual(len(datasets), 1)

        sample = next((d for d in datasets if d["dataset_id"] == "sample"), None)
        self.assertIsNotNone(sample)
        self.assertEqual(sample["name"], "RAG Core Mechanics Benchmark")
        self.assertEqual(sample["num_queries"], 5)
        self.assertEqual(sample["corpus_doc_count"], 19)
        self.assertEqual(sample["default_k_values"], [1, 3, 5])

    def test_list_experiments_returns_empty_list(self) -> None:
        """GET /api/experiments must return an empty list when persistence is not configured."""
        response = self.client.get("/api/experiments")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    def test_evaluate_default_live_bm25(self) -> None:
        """POST /api/evaluate executes real BM25 and returns verified metrics."""
        payload = {
            "dataset_id": "sample",
            "k_values": [1, 3, 5],
            "include_baselines": False,
        }
        response = self.client.post("/api/evaluate", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["dataset_id"], "sample")
        self.assertEqual(data["queries_evaluated"], 5)
        self.assertEqual(data["corpus_size"], 19)
        self.assertEqual(data["k_values"], [1, 3, 5])

        # Only the live BM25 retriever should be present when baselines are excluded
        self.assertEqual(len(data["summaries"]), 1)
        bm25_summary = data["summaries"][0]
        self.assertEqual(bm25_summary["system_name"], "Real_BM25_Retriever")
        self.assertEqual(bm25_summary["system_type"], "live")
        self.assertTrue(bm25_summary["is_live"])
        self.assertEqual(bm25_summary["num_queries"], 5)
        self.assertEqual(bm25_summary["mrr"], 1.0)
        self.assertIn("1", bm25_summary["mean_recall_at_k"])
        self.assertIn("3", bm25_summary["mean_recall_at_k"])
        self.assertIn("5", bm25_summary["mean_recall_at_k"])

        # Check per-query breakdown
        self.assertEqual(len(data["query_results"]), 5)
        first_q = data["query_results"][0]
        self.assertEqual(first_q["query_id"], "q_01")
        self.assertEqual(first_q["system_type"], "live")
        self.assertEqual(first_q["total_relevant"], 2)
        self.assertEqual(first_q["reciprocal_rank"], 1.0)
        self.assertGreater(len(first_q["retrieved_doc_ids"]), 0)

        # Disclaimer must be explicit
        self.assertIn("Real_BM25_Retriever", data["disclaimer"])

    def test_evaluate_with_illustrative_baselines(self) -> None:
        """POST /api/evaluate with include_baselines=True includes and labels baselines."""
        payload = {
            "dataset_id": "sample",
            "k_values": [1, 3, 5],
            "include_baselines": True,
        }
        response = self.client.post("/api/evaluate", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(len(data["summaries"]), 3)

        systems = {s["system_name"]: s for s in data["summaries"]}
        self.assertIn("Real_BM25_Retriever", systems)
        self.assertIn("Lexical_BM25_Baseline", systems)
        self.assertIn("Dense_Semantic_Baseline", systems)

        # Real retriever is labeled live
        self.assertTrue(systems["Real_BM25_Retriever"]["is_live"])
        self.assertEqual(systems["Real_BM25_Retriever"]["system_type"], "live")

        # Baselines are labeled illustrative
        self.assertFalse(systems["Lexical_BM25_Baseline"]["is_live"])
        self.assertEqual(systems["Lexical_BM25_Baseline"]["system_type"], "illustrative")
        self.assertFalse(systems["Dense_Semantic_Baseline"]["is_live"])
        self.assertEqual(systems["Dense_Semantic_Baseline"]["system_type"], "illustrative")

    def test_evaluate_custom_k_values(self) -> None:
        """POST /api/evaluate respects customized k_values list."""
        payload = {
            "dataset_id": "sample",
            "k_values": [2, 4],
            "include_baselines": False,
        }
        response = self.client.post("/api/evaluate", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["k_values"], [2, 4])
        summary = data["summaries"][0]
        self.assertIn("2", summary["mean_recall_at_k"])
        self.assertIn("4", summary["mean_recall_at_k"])
        self.assertNotIn("1", summary["mean_recall_at_k"])

    def test_evaluate_invalid_dataset_id_raises_400(self) -> None:
        """POST /api/evaluate with non-existent dataset must return HTTP 400."""
        payload = {
            "dataset_id": "non_existent_corpus_99",
            "k_values": [1, 3],
        }
        response = self.client.post("/api/evaluate", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("not found", response.json()["detail"])

    def test_evaluate_invalid_k_values_raises_422(self) -> None:
        """POST /api/evaluate with empty or negative k_values must return HTTP 422."""
        # Empty list
        resp1 = self.client.post("/api/evaluate", json={"k_values": []})
        self.assertEqual(resp1.status_code, 422)

        # Negative k
        resp2 = self.client.post("/api/evaluate", json={"k_values": [-1, 3]})
        self.assertEqual(resp2.status_code, 422)

        # Zero k
        resp3 = self.client.post("/api/evaluate", json={"k_values": [0, 5]})
        self.assertEqual(resp3.status_code, 422)

    def test_cors_headers(self) -> None:
        """API must return permissive CORS headers for configured origins."""
        response = self.client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:3000")


if __name__ == "__main__":
    unittest.main()
