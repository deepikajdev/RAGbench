#!/usr/bin/env python3
"""RAGBench Retrieval Evaluator CLI.

Evaluates and compares candidate retrieval rankings on benchmark datasets using Recall@K.
Runs out of the box using only Python's standard library.
"""

import argparse
import json
import sys
from pathlib import Path

# Add current workspace directory to sys.path so ragbench is importable directly
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from ragbench.evaluators.retrieval_evaluator import RetrievalEvaluator
from ragbench.datasets.loader import load_sample_dataset, load_dataset_from_json
from ragbench.formatters.table import format_evaluation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAGBench: Evaluate and compare retrieval systems using Recall@K."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to custom JSON dataset file. Defaults to bundled sample dataset.",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Rank cutoff depths to evaluate (default: 1 3 5).",
    )
    parser.add_argument(
        "--no-details",
        action="store_true",
        help="Hide query-level breakdown table and display only summary metrics.",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Output raw evaluation results as JSON instead of formatted text.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Load dataset & candidate rankings
    if args.dataset:
        dataset_path = Path(args.dataset)
        print(f"[RAGBench] Loading custom benchmark dataset: {dataset_path}")
        queries, rankings = load_dataset_from_json(dataset_path)
    else:
        print("[RAGBench] Loading bundled sample benchmark dataset (5 RAG queries)...")
        queries, rankings = load_sample_dataset()

    print(f"[RAGBench] Evaluating {len(rankings)} systems across {len(queries)} benchmark queries...")
    print(f"[RAGBench] Target cutoffs: K = {args.k_values}")
    print()

    # Run evaluation
    evaluator = RetrievalEvaluator(k_values=args.k_values)
    report = evaluator.evaluate(queries, rankings)

    if args.json_output:
        # Structured JSON serializable output
        json_summary = {
            "k_values": report.k_values,
            "summaries": {
                name: {
                    "num_queries": s.num_queries,
                    "mrr": s.mrr,
                    "mean_recall_at_k": s.mean_recall_at_k,
                }
                for name, s in report.summaries.items()
            },
            "queries": [
                {
                    "query_id": q.query_id,
                    "system_name": q.system_name,
                    "reciprocal_rank": q.reciprocal_rank,
                    "recall_at_k": q.recall_at_k,
                }
                for q in report.query_results
            ],
        }
        print(json.dumps(json_summary, indent=2))
    else:
        output = format_evaluation_report(report, show_query_details=not args.no_details)
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
