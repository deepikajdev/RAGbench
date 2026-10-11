# RAGBench 🎯

> **A production-oriented benchmark and evaluation platform for Retrieval-Augmented Generation (RAG) pipelines.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-16%20passed-brightgreen.svg)]()

---

## 📖 Vision & Purpose

In production AI systems, a RAG pipeline is only as good as its retrieval stage. When a retriever fails to surface relevant context, language models either hallucinate or decline to answer. 

**RAGBench** is being built as an extensible, production-grade benchmarking platform to systematically evaluate, compare, and stress-test RAG retrieval and generation architectures. Instead of ad-hoc manual testing or proprietary black-box evaluations, RAGBench provides reproducible, transparent evaluation metrics grounded in standard Information Retrieval (IR) science.

---

## 🚀 Current Milestone: Retrieval Foundation

This foundation release establishes the core architecture and implements **Recall@K** retrieval evaluation:

* **Zero-Dependency Core**: Runs out of the box using only Python's standard library. No API keys, external databases, or third-party services required.
* **Recall@K Metric Engine**: Implements the standard mathematical IR formula with strict validation, boundary cutoff enforcement, and edge-case handling (empty sets, cutoffs, duplicates).
* **Sample Benchmark Dataset**: Includes a 5-query ground-truth benchmark covering essential RAG engineering topics (chunking, embeddings, context dilution, reranking, hybrid search).
* **Baseline Comparative Evaluation**: Compares two simulated retrieval systems (**Lexical BM25 Baseline** vs. **Dense Semantic Baseline**) across rank cutoffs $K \in \{1, 3, 5\}$.
* **Comprehensive Test Suite**: 16 unit tests covering mathematical accuracy, edge cases, dataset parsing, and end-to-end evaluation pipelines.
* **Extensible Architecture**: Clean domain model separation designed for straightforward growth into embeddings, vector stores, API services, and dashboards.

---

## 📐 Evaluation Metric: Recall@K

### What is Recall@K?

In RAG retrieval, **Recall@K** measures what percentage of all known relevant documents were successfully retrieved within the top-$K$ ranked results:

$$\text{Recall}@K = \frac{|\text{Top}_K(\text{Retrieved}) \cap \text{Relevant}|}{|\text{Relevant}|}$$

### Why it matters for RAG:
- If $\text{Recall}@K = 1.0$, the downstream LLM generator received **all necessary reference context** to produce an accurate answer.
- If $\text{Recall}@K < 1.0$, context is missing from the prompt, which directly induces model hallucination or omissions.
- Evaluating multiple values of $K$ (e.g. $K=1, 3, 5$) helps optimize context-window budget vs. information coverage.

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
│   ├── metrics/                    # Metric implementations
│   │   ├── __init__.py
│   │   └── retrieval.py            # recall_at_k, mean_recall_at_k
│   ├── datasets/                   # Benchmark datasets & loader
│   │   ├── __init__.py
│   │   ├── loader.py               # JSON loading & parsing utilities
│   │   └── sample_dataset.json     # 5-query benchmark with candidate rankings
│   ├── evaluators/                 # Multi-system evaluation orchestrator
│   │   ├── __init__.py
│   │   └── retrieval_evaluator.py  # RetrievalEvaluator engine
│   ├── formatters/                 # Report rendering & presentation
│   │   ├── __init__.py
│   │   └── table.py                # ASCII terminal table formatter
│   └── future/                     # Architectural design & roadmap
│       └── README.md               # Production architecture blueprints
└── tests/                          # Automated unit test suite
    ├── __init__.py
    ├── test_metrics.py             # 12 metric tests & boundary edge cases
    └── test_evaluator.py           # 4 pipeline & dataset tests
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

# Output machine-readable JSON
python evaluate.py --json-output

# Run against a custom JSON benchmark file
python evaluate.py --dataset path/to/dataset.json
```

### 2. Sample Output Preview

```text
================================================================================
                     RAGBENCH RETRIEVAL EVALUATION REPORT
================================================================================

--- SUMMARY METRICS (Mean Recall@K across all queries) ---
Retrieval System           | Queries      | Mean R@1     | Mean R@3     | Mean R@5    
---------------------------+--------------+--------------+--------------+-------------
Lexical_BM25_Baseline      |      5       | 0.5000 (50.0%) | 0.7000 (70.0%) | 1.0000 (100.0%)
Dense_Semantic_Baseline    |      5       | 0.3000 (30.0%) | 1.0000 (100.0%) | 1.0000 (100.0%)

--- HEAD-TO-HEAD COMPARISON ---
  * At K=1: 'Lexical_BM25_Baseline' leads 'Dense_Semantic_Baseline' by +0.2000 (+20.0%)
  * At K=3: 'Dense_Semantic_Baseline' leads 'Lexical_BM25_Baseline' by +0.3000 (+30.0%)
  * At K=5: Tied at 1.0000 (100.0%)
```

---

## 🧪 Running Unit Tests

Run the complete test suite using Python's standard `unittest` framework:

```bash
python -m unittest discover -s tests -v
```

All 16 tests will execute, validating:
- Exact mathematical Recall@K calculations
- Rank cutoff boundaries (items at rank $K+1$ excluded)
- Handling of duplicate retrieved IDs without score inflation
- Edge case handling (empty candidate results, empty ground truth, invalid $K$ types/values)
- Dataset loading and report generation

---

## 🗺️ Architectural Roadmap

RAGBench is architected to scale into an end-to-end evaluation platform. See [`ragbench/future/README.md`](ragbench/future/README.md) for full blueprints.

| Phase | Milestone | Focus Areas |
|---|---|---|
| **Phase 1 (Current)** | **Foundation** | Recall@K, standard library models, sample dataset, comparative CLI runner, unit tests. |
| **Phase 2** | **Retrievers & Fusion** | BM25 lexical indexing, dense embedding models (`sentence-transformers`), Hybrid Search (RRF), cross-encoder rerankers. |
| **Phase 3** | **Extended Metrics** | Precision@K, MRR (Mean Reciprocal Rank), NDCG@K, Hit Rate, and RAG Triad generation metrics (Faithfulness, Answer Relevance). |
| **Phase 4** | **Benchmark Corpus** | 50–100+ annotated domain questions (Technical Docs, Healthcare, Legal) with golden context chunks. |
| **Phase 5** | **Production Platform** | FastAPI backend, PostgreSQL + pgvector persistence, telemetry (latency, token costs), Docker containerization, Next.js web dashboard. |

---

## 🔒 Git Connection & Commit Instructions

This workspace was initialized on the `main` branch. To connect it to your GitHub repository and create your first commit safely:

```bash
# 1. Connect remote repository (if not already added)
git remote add origin https://github.com/deepikajdev/RAGbench.git

# 2. Verify the remote URL
git remote -v

# 3. Stage the foundation files
git add .

# 4. Commit with the designated conventional commit message
git commit -m "feat: add RAG retrieval evaluation MVP"

# 5. Push to GitHub
git push -u origin main
```

---

## 📄 License
This project is licensed under the MIT License.
