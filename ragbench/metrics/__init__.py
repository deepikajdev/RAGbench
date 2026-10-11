"""Evaluation metrics package for RAG pipelines."""

from ragbench.metrics.retrieval import recall_at_k, mean_recall_at_k

__all__ = ["recall_at_k", "mean_recall_at_k"]
