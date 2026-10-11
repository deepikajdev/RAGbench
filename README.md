# RAGBench 🎯

> **A production-oriented benchmark and evaluation platform for Retrieval-Augmented Generation (RAG) pipelines.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-25%20passed-brightgreen.svg)]()

---

## 📖 Vision & Purpose

In production AI systems, a RAG pipeline is only as good as its retrieval stage. When a retriever fails to surface relevant context or places critical documents too deep in the ranking, downstream language models either hallucinate, produce incomplete answers, or suffer from attention degradation.

**RAGBench** is being built as an extensible, production-grade benchmarking platform to systematically evaluate, compare, and stress-test RAG retrieval and generation architectures. Instead of ad-hoc manual testing or proprietary black-box evaluations, RAGBench provides reproducible, transparent evaluation metrics grounded in standard Information Retrieval (IR) science.

---

## 🚀 Current Milestone: Retrieval Evaluation Core

This milestone provides the core retrieval evaluation framework featuring **Recall@K** and **Mean Reciprocal Rank (MRR)**:

* **Zero-Dependency Core**: Runs out of the box using only Python's standard library. No API keys, external databases, or third-party services required.
* **Recall@K Metric Engine**: Implements the standard mathematical IR formula with strict validation, boundary cutoff enforcement, and edge-case handling.
* **MRR (Mean Reciprocal Rank) Engine**: Evaluates how high the retriever places the *first* relevant document, rewarding systems that surface key information immediately.
* **Sample Benchmark Dataset**: Includes a 5-query ground-truth benchmark covering essential RAG engineering topics (chunking, embeddings, context dilution, reranking, hybrid search).
* **Baseline Comparative Evaluation**: Compares two simulated retrieval systems (**Lexical BM25 Baseline** vs. **Dense Semantic Baseline**) side-by-side across cutoffs $K \in \{1, 3, 5\}$.
* **Comprehensive Test Suite**: 25 automated unit tests covering mathematical accuracy, rank cutoffs, boundary edge cases, input validation, and pipeline reporting.
* **Extensible Architecture**: Clean domain model separation designed for straightforward growth into embeddings, vector stores, API services, and dashboards.

---

## 📐 Evaluation Metrics

### 1. Recall@K

**Recall@K** measures what percentage of all known relevant documents were successfully retrieved within the top-$K$ ranked results:

$$\text{Recall}@K = \frac{|\text{Top}_K(\text{Retrieved}) \cap \text{Relevant}|}{|\text{Relevant}|}$$

- **Why it matters for RAG**: If $\text{Recall}@K = 1.0$, the downstream LLM generator received **all necessary reference context** to produce an accurate answer. If $\text{Recall}@K < 1.0$, missing context directly induces model hallucinations.
- **Tuning $K$**: Evaluating multiple values of $K$ (e.g. $K=1, 3, 5$) helps optimize context-window budget vs. information coverage.

---

### 2. Mean Reciprocal Rank (MRR)

**Reciprocal Rank (RR)** measures how quickly (at what rank position) the retrieval system surfaced the **first relevant document**:

$$\text{RR} = \frac{1}{\text{rank}_{\text{first relevant}}}$$

If no relevant document is retrieved, $\text{RR} = 0.0$.

| Rank of 1st Relevant Document | Reciprocal Rank (RR) | Score Interpretation |
|---|---|---|
| **Rank 1** | $1 / 1 = \mathbf{1.00}$ | Best possible: top result is relevant |
| **Rank 2** | $1 / 2 = \mathbf{0.50}$ | Very good: second item is relevant |
| **Rank 3** | $1 / 3 \approx \mathbf{0.33}$ | Moderate: third item is relevant |
| **Rank 4** | $1 / 4 = \mathbf{0.25}$ | Fair: fourth item is relevant |
| **Not Found** | $\mathbf{0.00}$ | Failure: no relevant document retrieved |

**Mean Reciprocal Rank (MRR)** is the arithmetic mean of Reciprocal Rank scores across all queries $Q$:

$$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \text{RR}(q)$$

- **Why it matters for RAG**: LLMs suffer from "lost-in-the-middle" attention degradation—they attend most strongly to documents at the very top of their prompt context. A high MRR guarantees that the most pertinent knowledge is placed at the head of the prompt where the model's attention is sharpest.

---

## 📂 Project Structure

```text
RAGbench/
├── .gitignore                      # Git ignore patterns for Python
├── pyproject.toml                  # Modern PEP 518/621 packaging metadata
├── README.md                       # Project overview, quickstart, and roadmap
├── evaluate.py                     # Standalone CLI evaluation runner
├── ragbench/
│   ├── __init__.py                 # Public package exports
│   ├── core/                       # Core domain models
│   │   ├── __init__.py
│   │   └── models.py               # EvaluationQuery, RankedResult, EvaluationReport
│   ├── metrics/                    # Information retrieval metrics
│   │   ├── __init__.py
│   │   └── retrieval.py            # recall_at_k, mean_recall_at_k, reciprocal_rank, mean_reciprocal_rank
│   ├── datasets/                   # Benchmark datasets & loader
│   │   ├── __init__.py
│   │   ├── loader.py               # JSON loading & parsing utilities
│   │   └── sample_dataset.json     # 5-query benchmark with candidate rankings
│   ├── evaluators/                 # Multi-system evaluation orchestrator
│   │   ├── __init__.py
│   │   └── retrieval_evaluator.py  # RetrievalEvaluator engine (Recall@K + MRR)
│   ├── formatters/                 # Report rendering & presentation
│   │   ├── __init__.py
│   │   └── table.py                # ASCII terminal table formatter
│   └── future/                     # Architectural design & roadmap
│       └── README.md               # Production architecture blueprints
└── tests/                          # Automated unit test suite (25 tests)
    ├── __init__.py
    ├── test_metrics.py             # 21 unit tests for Recall@K and MRR
    └── test_evaluator.py           # 4 pipeline & dataset loader tests
```

---

## ⚡ Quickstart

### Prerequisites
- Python 3.10+ (Standard library only; no pip dependencies required for this milestone)

### 1. Run the Evaluator
Run the evaluation runner directly from your terminal:

```bash
python evaluate.py
```

#### CLI Options:
```bash
# Evaluate custom cutoff depths (e.g., K = 1, 2, 4)
python evaluate.py --k-values 1 2 4

# View summary table only (hide per-query breakdown)
python evaluate.py --no-details

# Output machine-readable JSON (includes MRR and per-query RR)
python evaluate.py --json-output

# Run against a custom JSON benchmark file
python evaluate.py --dataset path/to/dataset.json
```

### 2. Sample Output Preview

```text
================================================================================
                     RAGBENCH RETRIEVAL EVALUATION REPORT
================================================================================

--- SUMMARY METRICS (MRR and Mean Recall@K across all queries) ---
Retrieval System           | Queries      | MRR          | Mean R@1     | Mean R@3     | Mean R@5    
---------------------------+--------------+--------------+--------------+--------------+-------------
Lexical_BM25_Baseline      |      5       | 0.8667 (86.7%) | 0.5000 (50.0%) | 0.7000 (70.0%) | 1.0000 (100.0%)
Dense_Semantic_Baseline    |      5       | 0.8000 (80.0%) | 0.3000 (30.0%) | 1.0000 (100.0%) | 1.0000 (100.0%)

--- HEAD-TO-HEAD COMPARISON ---
  * MRR: 'Lexical_BM25_Baseline' leads 'Dense_Semantic_Baseline' by +0.0667 (+6.7%)
  * At K=1: 'Lexical_BM25_Baseline' leads 'Dense_Semantic_Baseline' by +0.2000 (+20.0%)
  * At K=3: 'Dense_Semantic_Baseline' leads 'Lexical_BM25_Baseline' by +0.3000 (+30.0%)
  * At K=5: Tied at 1.0000 (100.0%)

--- PER-QUERY BREAKDOWN ---
Query ID   | System                     | Rel Docs   | Retrieved  | RR       | R@1      | R@3      | R@5     
-----------+----------------------------+------------+------------+----------+----------+----------+---------
q_01       | Lexical_BM25_Baseline      |     2      |     5      |     1.00 |     0.50 |     1.00 |     1.00
q_02       | Lexical_BM25_Baseline      |     2      |     5      |     1.00 |     0.50 |     0.50 |     1.00
q_03       | Lexical_BM25_Baseline      |     2      |     5      |     0.33 |     0.00 |     0.50 |     1.00
q_04       | Lexical_BM25_Baseline      |     1      |     5      |     1.00 |     1.00 |     1.00 |     1.00
q_05       | Lexical_BM25_Baseline      |     2      |     5      |     1.00 |     0.50 |     0.50 |     1.00
q_01       | Dense_Semantic_Baseline    |     2      |     5      |     0.50 |     0.00 |     1.00 |     1.00
q_02       | Dense_Semantic_Baseline    |     2      |     5      |     1.00 |     0.50 |     1.00 |     1.00
q_03       | Dense_Semantic_Baseline    |     2      |     5      |     1.00 |     0.50 |     1.00 |     1.00
q_04       | Dense_Semantic_Baseline    |     1      |     5      |     0.50 |     0.00 |     1.00 |     1.00
q_05       | Dense_Semantic_Baseline    |     2      |     5      |     1.00 |     0.50 |     1.00 |     1.00

================================================================================
```

---

## 🧪 Running Unit Tests

Run the complete test suite using Python's standard `unittest` framework:

```bash
python -m unittest discover -s tests -v
```

All 25 tests validate:
- **Recall@K**: Exact mathematical ratios, rank cutoff boundaries ($K+1$ excluded), duplicate candidate handling without inflation, and empty result lists.
- **MRR & Reciprocal Rank**: Rank 1 (1.0), rank 2 (0.5), rank 3 (0.3333), zero hits (0.0), empty retrieved list (0.0), and arithmetic dataset averaging.
- **Validation & Edge Cases**: Empty ground truth sets, invalid/negative $K$ values, non-integer types.
- **End-to-End Pipeline**: Dataset parsing, multi-system summary computation, and report formatting.

---

## 🗺️ Architectural Roadmap

RAGBench is architected to scale into an end-to-end evaluation platform. See [`ragbench/future/README.md`](ragbench/future/README.md) for detailed blueprints.

| Phase | Milestone | Focus Areas |
|---|---|---|
| **Phase 1 (Current)** | **Core Retrieval Metrics** | Recall@K, Reciprocal Rank (RR), Mean Reciprocal Rank (MRR), standard library models, sample benchmark dataset, CLI runner, automated unit tests. |
| **Phase 2** | **Retrievers & Fusion** | BM25 lexical indexing, dense embedding models (`sentence-transformers`), Hybrid Search (RRF), cross-encoder rerankers. |
| **Phase 3** | **Extended Metric Suite** | Precision@K, NDCG@K, Hit Rate, and RAG Triad generation metrics (Faithfulness, Answer Relevance, Context Relevance). |
| **Phase 4** | **Benchmark Corpus** | 50–100+ annotated domain questions (Technical Docs, Healthcare, Legal) with golden context chunks. |
| **Phase 5** | **Production Platform** | FastAPI backend, PostgreSQL + pgvector persistence, telemetry (latency, token costs), Docker containerization, Next.js web dashboard. |

---

## 🔒 Git Commands for Staging and Committing

When you are ready to review and commit this milestone:

```bash
# 1. Review modified files
git status
git diff

# 2. Stage the changes
git add .

# 3. Commit with a conventional commit message
git commit -m "feat: add Mean Reciprocal Rank (MRR) evaluation"

# 4. Push to GitHub
git push
```

---

## 📄 License
This project is licensed under the MIT License.
