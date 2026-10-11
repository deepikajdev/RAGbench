"""Readable terminal formatting utilities for evaluation reports."""

from typing import List
from ragbench.core.models import EvaluationReport


def format_evaluation_report(report: EvaluationReport, show_query_details: bool = True) -> str:
    """Format an EvaluationReport into a readable ASCII text report.

    Args:
        report: The EvaluationReport instance to render.
        show_query_details: Whether to render per-query score breakdown.

    Returns:
        Formatted multi-line string.
    """
    lines: List[str] = []
    separator_width = 80
    lines.append("=" * separator_width)
    lines.append("                     RAGBENCH RETRIEVAL EVALUATION REPORT")
    lines.append("=" * separator_width)
    lines.append("")

    # 1. Summary Comparison Table
    lines.append("--- SUMMARY METRICS (Mean Recall@K across all queries) ---")
    headers = ["Retrieval System", "Queries"] + [f"Mean R@{k}" for k in report.k_values]
    col_widths = [max(len(h), 26 if i == 0 else 12) for i, h in enumerate(headers)]

    header_row = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    divider_row = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    lines.append(header_row)
    lines.append(divider_row)

    for system_name, summary in report.summaries.items():
        row_vals = [
            system_name.ljust(col_widths[0]),
            str(summary.num_queries).center(col_widths[1]),
        ]
        for idx, k in enumerate(report.k_values):
            val = summary.mean_recall_at_k.get(k, 0.0)
            percentage_str = f"{val:.4f} ({val * 100:.1f}%)"
            row_vals.append(percentage_str.rjust(col_widths[idx + 2]))
        lines.append(" | ".join(row_vals))

    lines.append("")

    # 2. System Comparison Insight
    if len(report.summaries) >= 2:
        lines.append("--- HEAD-TO-HEAD COMPARISON ---")
        for k in report.k_values:
            scores = [
                (name, summary.mean_recall_at_k.get(k, 0.0))
                for name, summary in report.summaries.items()
            ]
            scores.sort(key=lambda x: x[1], reverse=True)
            leader, leader_score = scores[0]
            runner_up, runner_score = scores[1]
            diff = leader_score - runner_score
            if diff > 0.0001:
                lines.append(
                    f"  * At K={k}: '{leader}' leads '{runner_up}' by +{diff:.4f} (+{diff * 100:.1f}%)"
                )
            else:
                lines.append(f"  * At K={k}: Tied at {leader_score:.4f} ({leader_score * 100:.1f}%)")
        lines.append("")

    # 3. Query-level Breakdown (if requested)
    if show_query_details and report.query_results:
        lines.append("--- PER-QUERY BREAKDOWN ---")
        q_headers = ["Query ID", "System", "Rel Docs", "Retrieved"] + [f"R@{k}" for k in report.k_values]
        q_widths = [10, 26, 10, 10] + [8 for _ in report.k_values]

        q_header_row = " | ".join(h.ljust(q_widths[i]) for i, h in enumerate(q_headers))
        q_divider_row = "-+-".join("-" * q_widths[i] for i in range(len(q_headers)))
        lines.append(q_header_row)
        lines.append(q_divider_row)

        for res in report.query_results:
            row_vals = [
                res.query_id.ljust(q_widths[0]),
                res.system_name.ljust(q_widths[1]),
                str(res.total_relevant).center(q_widths[2]),
                str(res.retrieved_count).center(q_widths[3]),
            ]
            for idx, k in enumerate(report.k_values):
                score = res.recall_at_k.get(k, 0.0)
                row_vals.append(f"{score:.2f}".rjust(q_widths[idx + 4]))
            lines.append(" | ".join(row_vals))

        lines.append("")

    lines.append("=" * separator_width)
    return "\n".join(lines)
