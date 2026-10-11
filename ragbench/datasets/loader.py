"""Dataset loader utilities for benchmark evaluation queries and rankings."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

from ragbench.core.models import EvaluationQuery, RankedResult


def get_default_dataset_path() -> Path:
    """Return the absolute path to the bundled sample dataset JSON."""
    return Path(__file__).parent / "sample_dataset.json"


def load_dataset_from_json(
    file_path: Path | str,
) -> Tuple[List[EvaluationQuery], Dict[str, Dict[str, List[str]]]]:
    """Load evaluation queries and candidate system rankings from a JSON file.

    Args:
        file_path: Path to the JSON benchmark file.

    Returns:
        A tuple of:
          - List of EvaluationQuery instances.
          - Dict mapping system_name -> (dict mapping query_id -> list of retrieved doc IDs).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is invalid.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark file not found at: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    queries_data = data.get("queries", [])
    if not queries_data:
        raise ValueError(f"No queries found in benchmark file: {path}")

    queries: List[EvaluationQuery] = []
    for item in queries_data:
        query = EvaluationQuery(
            query_id=item["query_id"],
            query_text=item["query_text"],
            relevant_doc_ids=set(item["relevant_doc_ids"]),
            metadata=item.get("metadata", {}),
        )
        queries.append(query)

    sample_rankings: Dict[str, Dict[str, List[str]]] = data.get("sample_rankings", {})
    return queries, sample_rankings


def load_sample_dataset() -> Tuple[List[EvaluationQuery], Dict[str, Dict[str, List[str]]]]:
    """Load the bundled default sample dataset."""
    return load_dataset_from_json(get_default_dataset_path())


def convert_rankings_to_results(
    rankings_by_system: Dict[str, Dict[str, List[str]]],
) -> List[RankedResult]:
    """Convert a dictionary of system rankings into a list of RankedResult models."""
    results: List[RankedResult] = []
    for system_name, query_map in rankings_by_system.items():
        for query_id, doc_ids in query_map.items():
            results.append(
                RankedResult(
                    query_id=query_id,
                    system_name=system_name,
                    retrieved_doc_ids=doc_ids,
                )
            )
    return results
