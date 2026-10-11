"""RAGBench: Production-oriented benchmark and evaluation suite for RAG pipelines."""

from ragbench.core.models import (
    EvaluationQuery,
    RankedResult,
    QueryResultMetric,
    EvaluationSummary,
    EvaluationReport,
)
from ragbench.metrics.retrieval import recall_at_k, mean_recall_at_k
from ragbench.evaluators.retrieval_evaluator import RetrievalEvaluator
from ragbench.datasets.loader import (
    load_sample_dataset,
    load_dataset_from_json,
    convert_rankings_to_results,
    get_default_dataset_path,
)
from ragbench.formatters.table import format_evaluation_report

__version__ = "0.1.0"

__all__ = [
    "EvaluationQuery",
    "RankedResult",
    "QueryResultMetric",
    "EvaluationSummary",
    "EvaluationReport",
    "recall_at_k",
    "mean_recall_at_k",
    "RetrievalEvaluator",
    "load_sample_dataset",
    "load_dataset_from_json",
    "convert_rankings_to_results",
    "get_default_dataset_path",
    "format_evaluation_report",
]
