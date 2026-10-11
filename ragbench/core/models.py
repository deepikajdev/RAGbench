"""Core data structures and domain models for RAGBench."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Optional


@dataclass(frozen=True)
class EvaluationQuery:
    """Represents a benchmark query with ground-truth relevant document IDs."""
    query_id: str
    query_text: str
    relevant_doc_ids: Set[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.query_id:
            raise ValueError("query_id cannot be empty")
        if not self.relevant_doc_ids:
            raise ValueError(f"relevant_doc_ids cannot be empty for query {self.query_id!r}")


@dataclass
class RankedResult:
    """Represents a ranked list of retrieved document IDs returned by a retrieval system."""
    query_id: str
    system_name: str
    retrieved_doc_ids: List[str]
    scores: Optional[List[float]] = None


@dataclass
class QueryResultMetric:
    """Per-query evaluation breakdown for a specific retrieval system."""
    query_id: str
    query_text: str
    system_name: str
    total_relevant: int
    retrieved_count: int
    recall_at_k: Dict[int, float]


@dataclass
class EvaluationSummary:
    """Aggregated evaluation metrics for a retrieval system across all evaluated queries."""
    system_name: str
    num_queries: int
    mean_recall_at_k: Dict[int, float]


@dataclass
class EvaluationReport:
    """Full evaluation report holding both query-level details and system-level summaries."""
    k_values: List[int]
    query_results: List[QueryResultMetric]
    summaries: Dict[str, EvaluationSummary]
