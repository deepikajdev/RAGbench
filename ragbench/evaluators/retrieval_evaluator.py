"""Retrieval evaluator orchestrating metrics computation across datasets and systems."""

from typing import Dict, List, Sequence, Union

from ragbench.core.models import (
    EvaluationQuery,
    EvaluationReport,
    EvaluationSummary,
    QueryResultMetric,
    RankedResult,
)
from ragbench.metrics.retrieval import (
    recall_at_k,
    mean_recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
)


class RetrievalEvaluator:
    """Evaluates ranked retrieval outputs against benchmark queries using Recall@K and MRR."""

    def __init__(self, k_values: Sequence[int] = (1, 3, 5)) -> None:
        """Initialize the evaluator with target cutoff depths.

        Args:
            k_values: Sequence of positive integer cutoffs (e.g. [1, 3, 5]).
        """
        sorted_k = sorted(list(set(k_values)))
        for k in sorted_k:
            if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
                raise ValueError(f"All k values must be positive integers >= 1, got {k!r}")
        self.k_values: List[int] = sorted_k

    def evaluate_system(
        self,
        system_name: str,
        queries: Sequence[EvaluationQuery],
        rankings: Dict[str, List[str]],
    ) -> tuple[List[QueryResultMetric], EvaluationSummary]:
        """Evaluate a single retrieval system against a collection of queries.

        Args:
            system_name: Name/identifier of the retrieval system.
            queries: Sequence of ground-truth EvaluationQuery objects.
            rankings: Map of query_id -> list of retrieved doc IDs.

        Returns:
            Tuple of (list of per-query metrics, aggregate EvaluationSummary).
        """
        query_metrics: List[QueryResultMetric] = []
        k_scores: Dict[int, List[float]] = {k: [] for k in self.k_values}
        rr_scores: List[float] = []

        for q in queries:
            retrieved = rankings.get(q.query_id, [])
            recalls: Dict[int, float] = {}

            for k in self.k_values:
                score = recall_at_k(retrieved, q.relevant_doc_ids, k=k)
                recalls[k] = score
                k_scores[k].append(score)

            rr = reciprocal_rank(retrieved, q.relevant_doc_ids)
            rr_scores.append(rr)

            metric = QueryResultMetric(
                query_id=q.query_id,
                query_text=q.query_text,
                system_name=system_name,
                total_relevant=len(q.relevant_doc_ids),
                retrieved_count=len(retrieved),
                reciprocal_rank=rr,
                recall_at_k=recalls,
            )
            query_metrics.append(metric)

        mean_recalls: Dict[int, float] = {
            k: mean_recall_at_k(scores) for k, scores in k_scores.items()
        }
        mrr = mean_reciprocal_rank(rr_scores)

        summary = EvaluationSummary(
            system_name=system_name,
            num_queries=len(queries),
            mrr=mrr,
            mean_recall_at_k=mean_recalls,
        )

        return query_metrics, summary

    def evaluate(
        self,
        queries: Sequence[EvaluationQuery],
        systems_rankings: Dict[str, Dict[str, List[str]]],
    ) -> EvaluationReport:
        """Run full evaluation comparing all supplied retrieval systems.

        Args:
            queries: Sequence of ground-truth EvaluationQuery instances.
            systems_rankings: Map of system_name -> (query_id -> list of retrieved doc IDs).

        Returns:
            EvaluationReport containing per-query results and system summaries.
        """
        if not queries:
            raise ValueError("queries collection cannot be empty")
        if not systems_rankings:
            raise ValueError("systems_rankings cannot be empty")

        all_query_metrics: List[QueryResultMetric] = []
        summaries: Dict[str, EvaluationSummary] = {}

        for system_name, rankings in systems_rankings.items():
            metrics, summary = self.evaluate_system(system_name, queries, rankings)
            all_query_metrics.extend(metrics)
            summaries[system_name] = summary

        return EvaluationReport(
            k_values=list(self.k_values),
            query_results=all_query_metrics,
            summaries=summaries,
        )
