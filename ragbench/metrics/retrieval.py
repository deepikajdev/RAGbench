"""Retrieval evaluation metrics for ranking quality."""

from typing import Iterable, Sequence, Set, Union


def recall_at_k(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Union[Set[str], Sequence[str]],
    k: int,
) -> float:
    """Calculate Recall@K for a single ranked list of retrieved documents.

    Recall@K measures the proportion of ground-truth relevant documents
    that appear in the top-K retrieved results:

        Recall@K = |TopK(Retrieved) ∩ Relevant| / |Relevant|

    Args:
        retrieved_doc_ids: Ordered sequence of document IDs returned by the retriever
            (rank 1 is index 0).
        relevant_doc_ids: Set or sequence of ground-truth relevant document IDs.
        k: Positive integer cutoff threshold (rank depth to inspect).

    Returns:
        Float value between 0.0 and 1.0 representing the proportion of relevant documents found.

    Raises:
        ValueError: If k <= 0 or if relevant_doc_ids is empty.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"k must be a positive integer >= 1, got {k!r}")

    relevant_set = set(relevant_doc_ids)
    if not relevant_set:
        raise ValueError("relevant_doc_ids cannot be empty; ground truth must contain at least 1 document ID")

    if not retrieved_doc_ids:
        return 0.0

    # Inspect the top-K retrieved items
    top_k_candidates = retrieved_doc_ids[:k]
    matched_hits = set(top_k_candidates) & relevant_set

    return len(matched_hits) / len(relevant_set)


def mean_recall_at_k(scores: Sequence[float]) -> float:
    """Calculate the arithmetic mean of Recall@K scores across queries.

    Args:
        scores: Sequence of float recall scores.

    Returns:
        Mean recall score, or 0.0 if scores sequence is empty.
    """
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
