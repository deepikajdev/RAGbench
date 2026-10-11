"""Retrieval evaluation metrics for ranking quality."""

from typing import Iterable, Optional, Sequence, Set, Union


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


def reciprocal_rank(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Union[Set[str], Sequence[str]],
    k: Optional[int] = None,
) -> float:
    """Calculate Reciprocal Rank (RR) for a single ranked list of retrieved documents.

    Reciprocal Rank measures the inverse rank position of the FIRST relevant document:

        RR = 1 / rank_of_first_relevant_document

    If no relevant document is retrieved (or none appears within cutoff k), RR is 0.0:
        - Rank 1 -> 1.0 (1/1)
        - Rank 2 -> 0.5 (1/2)
        - Rank 3 -> 0.3333 (1/3)
        - Not found -> 0.0

    Args:
        retrieved_doc_ids: Ordered sequence of document IDs returned by the retriever
            (rank 1 is index 0).
        relevant_doc_ids: Set or sequence of ground-truth relevant document IDs.
        k: Optional positive integer cutoff threshold. If provided, only ranks <= k
            are inspected.

    Returns:
        Float value between 0.0 and 1.0 representing the reciprocal rank.

    Raises:
        ValueError: If k is provided and <= 0, or if relevant_doc_ids is empty.
    """
    if k is not None:
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError(f"k must be a positive integer >= 1, got {k!r}")

    relevant_set = set(relevant_doc_ids)
    if not relevant_set:
        raise ValueError("relevant_doc_ids cannot be empty; ground truth must contain at least 1 document ID")

    candidates = retrieved_doc_ids[:k] if k is not None else retrieved_doc_ids
    for rank, doc_id in enumerate(candidates, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(scores: Sequence[float]) -> float:
    """Calculate Mean Reciprocal Rank (MRR) across a collection of queries.

    MRR is the arithmetic mean of the Reciprocal Rank scores across all evaluated queries:

        MRR = (1 / |Q|) * sum_{q in Q} RR(q)

    Args:
        scores: Sequence of float reciprocal rank scores.

    Returns:
        Float value between 0.0 and 1.0, or 0.0 if scores is empty.
    """
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

