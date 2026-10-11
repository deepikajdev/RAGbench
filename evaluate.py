#!/usr/bin/env python3
"""RAGBench Retrieval Evaluator CLI.

Evaluates and compares candidate retrieval rankings on benchmark datasets using Recall@K and MRR.
Includes real BM25 keyword search execution against a document corpus.
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
from ragbench.datasets.loader import (
    load_sample_dataset,
    load_dataset_from_json,
    load_sample_corpus,
    load_corpus_from_json,
)
from ragbench.retrievers.bm25 import BM25Retriever
from ragbench.formatters.table import format_evaluation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAGBench: Evaluate and compare retrieval systems using Recall@K and MRR."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to custom JSON dataset file. Defaults to bundled sample dataset.",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path to custom JSON corpus file. Defaults to bundled sample corpus.",
    )
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help="Execute an ad-hoc BM25 search query against the corpus and print ranked matches.",
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

    # 1. Load document corpus and build real BM25 retriever
    if args.corpus:
        corpus = load_corpus_from_json(Path(args.corpus))
    else:
        corpus = load_sample_corpus()

    bm25 = BM25Retriever.from_documents(corpus)

    # 2. If user requested an ad-hoc search, execute it and display results
    if args.search:
        print(f"[RAGBench] Executing BM25 search query: {args.search!r}")
        print(f"[RAGBench] Indexed corpus size: {len(corpus)} documents")
        print()
        results = bm25.search(args.search, top_k=max(args.k_values))
        if not results:
            print("  No matching documents found.")
        else:
            for item in results:
                doc = next((d for d in corpus if d.doc_id == item.doc_id), None)
                title = f" ({doc.title})" if doc and doc.title else ""
                snippet = f" | {doc.content[:90]}..." if doc else ""
                print(f"  Rank {item.rank}: [{item.doc_id}]{title} - Score: {item.score:.4f}{snippet}")
        return 0

    # 3. Load dataset & reference candidate rankings
    if args.dataset:
        dataset_path = Path(args.dataset)
        print(f"[RAGBench] Loading custom benchmark dataset: {dataset_path}")
        queries, rankings = load_dataset_from_json(dataset_path)
    else:
        print(f"[RAGBench] Loading bundled sample benchmark (5 queries) & corpus ({len(corpus)} docs)...")
        queries, rankings = load_sample_dataset()

    # 4. Dynamically execute real BM25 retriever on all benchmark queries
    max_k = max(args.k_values)
    real_bm25_rankings = {}
    for q in queries:
        retrieved_ids = bm25.retrieve_ids(q.query_text, top_k=max_k)
        real_bm25_rankings[q.query_id] = retrieved_ids

    # Add real executed retriever to rankings for comparative evaluation
    rankings["Real_BM25_Retriever"] = real_bm25_rankings

    if not args.json_output:
        # Demonstrate live execution on the first benchmark query
        demo_query = queries[0]
        demo_search = bm25.search(demo_query.query_text, top_k=3)
        print(f"[RAGBench] Live BM25 Execution Demo on query '{demo_query.query_id}': \"{demo_query.query_text}\"")
        for item in demo_search:
            doc = next((d for d in corpus if d.doc_id == item.doc_id), None)
            title = f" ({doc.title})" if doc and doc.title else ""
            print(f"   -> Rank {item.rank}: [{item.doc_id}]{title} (Score: {item.score:.4f})")
        print()

        print(f"[RAGBench] Evaluating {len(rankings)} systems across {len(queries)} benchmark queries...")
        print(f"[RAGBench] Target cutoffs: K = {args.k_values}")
        print()

        # ---------------------------------------------------------------
        # SYSTEM LEGEND: clarify which systems execute real retrieval
        # ---------------------------------------------------------------
        print("[RAGBench] System Legend:")
        predefined_systems = [s for s in rankings if s != "Real_BM25_Retriever"]
        for ps in predefined_systems:
            print(f"   [ILLUSTRATIVE] {ps} "
                  f"-- predefined static rankings (not a live retrieval execution)")
        print("   [LIVE RETRIEVAL] Real_BM25_Retriever "
              "-- dynamically executed BM25 search against the sample corpus")
        print()

    # 5. Run evaluation
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
