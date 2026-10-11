# RAGBench 🎯

> **A production-oriented benchmark and evaluation platform for Retrieval-Augmented Generation (RAG) pipelines.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-48%20passed-brightgreen.svg)]()

---

## 📖 Vision & Purpose

In production AI systems, a RAG pipeline is only as good as its retrieval stage. When a retriever fails to surface relevant context or places critical documents too deep in the ranking, downstream language models either hallucinate, produce incomplete answers, or suffer from attention degradation.

**RAGBench** is being built as an extensible, production-grade benchmarking platform to systematically evaluate, compare, and stress-test RAG retrieval and generation architectures. Instead of ad-hoc manual testing or proprietary black-box evaluations, RAGBench provides reproducible, transparent evaluation metrics grounded in standard Information Retrieval (IR) science.

---

## 🚀 Current Milestone: Real BM25 Keyword Retriever & Evaluation

This milestone delivers **Phase 3**: a real, fully functional BM25 lexical retriever integrated directly with our **Recall@K** and **MRR** evaluation pipeline:

* **Zero-Dependency BM25 Engine**: Pure-Python implementation of BM25Okapi using only the standard library (`math`, `re`, `collections`). No external heavy dependencies or C-extensions required.
* **Corpus Indexing & Search**: Inverted index with document length normalization and term frequency saturation, returning scored and ranked document candidates (`ScoredDocument`).
* **Robust Edge-Case Handling**: Cleanly handles empty queries, out-of-vocabulary terms, punctuation, case insensitivity, empty documents, top-$K$ limits, and deterministic tie-breaking.
* **Sample Document Corpus**: Bundles a 19-document technical corpus covering chunking, embeddings, hallucinations, reranking, hybrid search, and distractor topics.
* **End-to-End Evaluation Pipeline**: The real BM25 retriever is dynamically queried across all benchmark questions, and its real rankings are evaluated alongside simulated baselines for Recall@K and MRR.
* **Interactive CLI & Search Demo**: Search the indexed corpus on the fly with `--search "<query>"` or run the comparative benchmark with full tabular breakdown.
* **Comprehensive Test Suite**: 39 automated unit tests verifying mathematical correctness, ranking order, tie-breaking, edge cases, and regression safety.

---

## 🔍 Understanding BM25 (Best Matching 25)

### What is BM25?
**BM25** (specifically BM25Okapi) is a ranking function used by search engines and information retrieval systems to score the relevance of a document $D$ to a search query $Q$. It improves upon classic TF-IDF by addressing two fundamental challenges:

1. **Term Frequency Saturation**: Mentioning a keyword twice is much better than once, but mentioning it 20 times instead of 10 times gives diminishing returns.
2. **Document Length Normalization**: Longer documents naturally contain more words; BM25 penalizes long, rambling texts to ensure shorter, focused passages with matching terms are not unfairly outranked.

### The BM25Okapi Scoring Formula

$$\text{Score}(D, Q) = \sum_{q_i \in Q} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where:
* $f(q_i, D)$: Frequency of query term $q_i$ in document $D$.
* $|D|$: Length of document $D$ in tokens.
* $\text{avgdl}$: Average document length across all documents in the corpus.
* $k_1$ (default $1.5$): **Term frequency saturation parameter**. Governs how quickly term frequency saturates. Higher values allow higher term frequency to have more impact.
* $b$ (default $0.75$): **Length normalization parameter**. When $b=1$, full length normalization is applied; when $b=0$, document length normalization is disabled entirely.

### Smooth Non-Negative Inverse Document Frequency (IDF)
To prevent common terms that appear in more than half the corpus from receiving negative scores, RAGBench utilizes the Robertson / Lucene non-negative smooth IDF formulation:

$$\text{IDF}(q_i) = \ln\left(1 + \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5}\right)$$

Where $N$ is the total number of documents and $n(q_i)$ is the number of documents containing term $q_i$.

### How to Interpret BM25 Output Scores
* **Positive float score ($\ge 0.0$)**: Higher scores indicate higher predicted relevance.
* **Score of $0.0$ / Empty Results**: If no terms from the query appear in the inverted index, the document receives no score and is omitted from the top-$K$ candidates.
* **Score magnitude**: BM25 scores are relative within a specific corpus and query; they are not normalized probabilities between $0.0$ and $1.0$.

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
├── evaluate.py                     # Standalone CLI evaluation & BM25 search runner
├── ragbench/
│   ├── __init__.py                 # Public package exports
│   ├── core/                       # Core domain models
│   │   ├── __init__.py
│   │   └── models.py               # Document, ScoredDocument, EvaluationQuery, RankedResult
│   ├── retrievers/                 # Retrieval engines
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseRetriever abstract base class
│   │   └── bm25.py                 # Pure-Python BM25Okapi retriever implementation
│   ├── metrics/                    # Information retrieval metrics
│   │   ├── __init__.py
│   │   └── retrieval.py            # recall_at_k, mean_recall_at_k, reciprocal_rank, mean_reciprocal_rank
│   ├── datasets/                   # Benchmark datasets & corpus loaders
│   │   ├── __init__.py
│   │   ├── loader.py               # JSON loading & parsing utilities for corpus & datasets
│   │   ├── sample_corpus.json      # 19-document technical corpus with distractors
│   │   └── sample_dataset.json     # 5-query benchmark with reference rankings
│   ├── evaluators/                 # Multi-system evaluation orchestrator
│   │   ├── __init__.py
│   │   └── retrieval_evaluator.py  # RetrievalEvaluator engine (Recall@K + MRR)
│   ├── formatters/                 # Report rendering & presentation
│   │   ├── __init__.py
│   │   └── table.py                # ASCII terminal table formatter
│   └── future/                     # Architectural design & roadmap
│       └── README.md               # Production architecture blueprints
└── tests/                          # Automated unit test suite (39 tests)
    ├── __init__.py
    ├── test_metrics.py             # 21 unit tests for Recall@K and MRR
    ├── test_evaluator.py           # 7 pipeline, dataset, and corpus loader tests
    └── test_bm25.py                # 11 unit tests for BM25 ranking, tie-breaking, and edge cases
```

---

## ⚡ Quickstart

### Prerequisites
- Python 3.10+ (Standard library only; zero pip dependencies required)

### 1. Ad-Hoc BM25 Corpus Search Demo
Search the indexed corpus directly for any technical query and inspect ranked matches with scores:

```bash
python evaluate.py --search "chunk overlap text splitting"
```

**Demonstration Output**:
```text
[RAGBench] Executing BM25 search query: 'chunk overlap text splitting'
[RAGBench] Indexed corpus size: 19 documents

  Rank 1: [doc_chunking_overlap] (Text Splitting and Chunk Overlap Mechanics) - Score: 13.0386 | Chunk overlap is a vital technique used during document text splitting. When chunking long...
  Rank 2: [doc_chunking_strategies] (Document Chunking Strategies for RAG) - Score: 5.1177 | Various chunking strategies exist for RAG systems, including fixed character windows, recu...
  Rank 3: [doc_vector_intro] (Vector Databases and Indexing Mechanics) - Score: 1.7496 | Vector databases store embedding representations and perform approximate nearest neighbor ...
  Rank 4: [doc_dense_embeddings] (Dense Vector Embeddings and Semantic Search) - Score: 1.6498 | Dense vector embeddings represent text passages as high-dimensional continuous vectors. Un...
```

---

### 2. Run the Full Comparative Evaluation
Execute the real BM25 retriever dynamically on all benchmark queries and compare it head-to-head against reference baselines:

```bash
python evaluate.py
```

#### CLI Options:
```bash
# Evaluate custom cutoff depths (e.g., K = 1, 2, 4)
python evaluate.py --k-values 1 2 4

# Ad-hoc BM25 query against corpus
python evaluate.py --search "Reciprocal Rank Fusion hybrid"

# View summary table only (hide per-query breakdown)
python evaluate.py --no-details

# Output machine-readable JSON (includes MRR and per-query RR)
python evaluate.py --json-output

# Run against a custom JSON benchmark file and corpus
python evaluate.py --dataset path/to/dataset.json --corpus path/to/corpus.json
```

### 3. Evaluation Report Preview

```text
[RAGBench] Loading bundled sample benchmark (5 queries) & corpus (19 docs)...
[RAGBench] Live BM25 Execution Demo on query 'q_01': "What is chunk overlap and why is it used in text splitting?"
   -> Rank 1: [doc_chunking_overlap] (Text Splitting and Chunk Overlap Mechanics) (Score: 18.0054)
   -> Rank 2: [doc_chunking_strategies] (Document Chunking Strategies for RAG) (Score: 5.2533)
   -> Rank 3: [doc_token_limits] (Context Budget and Token Limit Considerations) (Score: 3.8132)

[RAGBench] Evaluating 3 systems across 5 benchmark queries...
[RAGBench] Target cutoffs: K = [1, 3, 5]

================================================================================
                     RAGBENCH RETRIEVAL EVALUATION REPORT
================================================================================

--- SUMMARY METRICS (MRR and Mean Recall@K across all queries) ---
Retrieval System           | Queries      | MRR          | Mean R@1     | Mean R@3     | Mean R@5    
---------------------------+--------------+--------------+--------------+--------------+-------------
Lexical_BM25_Baseline      |      5       | 0.8667 (86.7%) | 0.5000 (50.0%) | 0.7000 (70.0%) | 1.0000 (100.0%)
Dense_Semantic_Baseline    |      5       | 0.8000 (80.0%) | 0.3000 (30.0%) | 1.0000 (100.0%) | 1.0000 (100.0%)
Real_BM25_Retriever        |      5       | 1.0000 (100.0%) | 0.6000 (60.0%) | 1.0000 (100.0%) | 1.0000 (100.0%)

--- HEAD-TO-HEAD COMPARISON ---
  * MRR: 'Real_BM25_Retriever' leads 'Lexical_BM25_Baseline' by +0.1333 (+13.3%)
  * At K=1: 'Real_BM25_Retriever' leads 'Lexical_BM25_Baseline' by +0.1000 (+10.0%)
  * At K=3: Tied at 1.0000 (100.0%)
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
q_01       | Real_BM25_Retriever        |     2      |     5      |     1.00 |     0.50 |     1.00 |     1.00
q_02       | Real_BM25_Retriever        |     2      |     5      |     1.00 |     0.50 |     1.00 |     1.00
q_03       | Real_BM25_Retriever        |     2      |     5      |     1.00 |     0.50 |     1.00 |     1.00
q_04       | Real_BM25_Retriever        |     1      |     5      |     1.00 |     1.00 |     1.00 |     1.00
q_05       | Real_BM25_Retriever        |     2      |     5      |     1.00 |     0.50 |     1.00 |     1.00

================================================================================
```

---

## 🌐 FastAPI REST Backend (Frontend Integration Ready)

RAGBench includes a typed **FastAPI** backend for serving evaluation metrics and benchmark datasets to web frontends (such as Next.js or v0 dashboards).

### Installation & Server Startup (Windows PowerShell)

1. **Install dependencies**:
   ```powershell
   python -m pip install -r requirements.txt
   ```

2. **Start the local development server**:
   ```powershell
   python -m uvicorn ragbench.api.app:app --reload --host 127.0.0.1 --port 8000
   ```

3. **Explore Interactive OpenAPI / Swagger UI**:
   Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser.

### Configuration

* **`RAGBENCH_CORS_ORIGINS`**: Comma-separated list of permitted frontend origins (default: `http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173`).
  ```powershell
  $env:RAGBENCH_CORS_ORIGINS = "http://localhost:3000,http://localhost:3001"
  python -m uvicorn ragbench.api.app:app --reload
  ```

### API Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health status, version, and corpus indexing state. |
| `GET` | `/api/datasets` | Metadata on supported benchmark datasets and document corpora. |
| `POST` | `/api/evaluate` | Executes live BM25 retrieval across benchmark queries and computes MRR + Recall@K. |
| `GET` | `/api/experiments` | Returns empty array `[]` (persistent history store is reserved for a future milestone). |

### Example POST /api/evaluate Request & Response

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/api/evaluate" \
     -H "Content-Type: application/json" \
     -d '{
       "dataset_id": "sample",
       "k_values": [1, 3, 5],
       "include_baselines": false
     }'
```

**Response:**
```json
{
  "dataset_id": "sample",
  "k_values": [1, 3, 5],
  "queries_evaluated": 5,
  "corpus_size": 19,
  "summaries": [
    {
      "system_name": "Real_BM25_Retriever",
      "system_type": "live",
      "is_live": true,
      "description": "Dynamically executed BM25 ranking against corpus text.",
      "num_queries": 5,
      "mrr": 1.0,
      "mean_recall_at_k": {
        "1": 0.6,
        "3": 1.0,
        "5": 1.0
      }
    }
  ],
  "query_results": [
    {
      "query_id": "q_01",
      "query_text": "What is chunk overlap and why is it used in text splitting?",
      "system_name": "Real_BM25_Retriever",
      "system_type": "live",
      "total_relevant": 2,
      "retrieved_count": 5,
      "reciprocal_rank": 1.0,
      "recall_at_k": {
        "1": 0.5,
        "3": 1.0,
        "5": 1.0
      },
      "retrieved_doc_ids": [
        "doc_chunking_overlap",
        "doc_chunking_strategies",
        "doc_token_limits",
        "doc_intro_rag",
        "doc_vector_intro"
      ]
    }
  ],
  "disclaimer": "NOTICE: 'Real_BM25_Retriever' was executed live against the indexed text corpus. Systems labeled 'illustrative' are predefined static benchmark rankings intended for comparative reference only."
}
```

---

## 💻 Next.js Frontend Dashboard (`frontend/`)

RAGBench includes an integrated Next.js (React 19 / Tailwind v4) frontend that connects to the live FastAPI backend.

### Running Backend and Frontend Locally (Windows PowerShell)

Open two PowerShell terminal tabs:

**Terminal 1 — Start the Python FastAPI Backend:**
```powershell
# From repository root
python -m uvicorn ragbench.api.app:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 — Start the Next.js Frontend:**
```powershell
# Navigate to frontend/ and run
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Frontend Environment Configuration

* **`NEXT_PUBLIC_API_URL`**: Backend base URL (default: `http://127.0.0.1:8000`).
* Configured in `frontend/.env.local` (and documented in `frontend/.env.local.example`).
* Backend CORS permits origins `http://localhost:3000` and `http://127.0.0.1:3000` by default.

### Key Frontend Features

1. **Live Backend Health & Metadata**: Checks `/health` on load to display live indexed document counts and service version.
2. **Real Benchmark Datasets Screen**: Connects to `GET /api/datasets` to load actual corpus and query statistics.
3. **Interactive Evaluation Runner**: Form submits to `POST /api/evaluate` to dynamically run the Python BM25 retriever against the corpus, displaying real-time loading feedback and duplicate submission protection.
4. **Transparent Results Display**:
   - Live BM25 results are clearly labeled `[LIVE]`.
   - Illustrative baselines are tagged `[ILLUSTRATIVE]`.
   - Displays real MRR, Recall@1, Recall@3, Recall@5, and ranked retrieved document IDs per query.
5. **Empty History Representation**: Connects to `GET /api/experiments` and honestly displays that persistent storage is not yet implemented, while tracking active session runs in memory.

---

## 🧪 Running Unit Tests

Run the complete test suite using Python's standard `unittest` framework:

```powershell
python -m unittest discover -s tests -v
```

All 48 tests validate:
- **FastAPI REST API (9 tests)**: Health check, datasets listing, empty experiments list, live BM25 evaluation execution, illustrative baseline handling, custom $K$ rank cutoffs, 400 bad request handling, 422 validation errors, and CORS header injection.
- **BM25 Retrieval & Scoring (11 tests)**: Exact token matches, term frequency saturation, length normalization, empty queries, out-of-vocabulary terms, punctuation & case insensitivity, deterministic tie-breaking, empty document handling, top-$K$ limits, query term repetition, and integration with the evaluation pipeline.
- **Recall@K & MRR Metrics (21 tests)**: Exact mathematical ratios, rank cutoff boundaries ($K+1$ excluded), duplicate candidate handling without inflation, empty result lists, reciprocal ranks for ranks 1–4, and zero-hit edge cases.
- **Pipeline & Dataset Loaders (7 tests)**: Parsing sample queries, loading 19-document corpus, file-not-found error handling, multi-system summary evaluation, and ASCII table generation.

---

## 🗺️ Architectural Roadmap

RAGBench is architected to scale into an end-to-end evaluation platform. See [`ragbench/future/README.md`](ragbench/future/README.md) for detailed blueprints.

| Phase | Milestone | Focus Areas | Status |
|---|---|---|---|
| **Phase 1** | **Core Retrieval Metrics** | Recall@K, Reciprocal Rank (RR), Mean Reciprocal Rank (MRR), standard library models, sample benchmark dataset. | Completed |
| **Phase 2** | **Evaluation Foundation** | Multi-system evaluation pipeline, ASCII report formatters, JSON output, CLI runner. | Completed |
| **Phase 3 (Current)** | **BM25 Keyword Retriever** | Inverted index BM25Okapi retriever, 19-document sample corpus, live search CLI, end-to-end evaluation integration. | Completed |
| **Phase 4** | **Dense Embeddings & Hybrid Search** | Vector embeddings (`sentence-transformers`), Reciprocal Rank Fusion (RRF), Cross-Encoder rerankers. | Planned |
| **Phase 5** | **Generation Quality (RAG Triad)** | Faithfulness, Context Relevance, Answer Relevance evaluation using lightweight evaluators. | Planned |
| **Phase 6** | **Platform & UI** | FastAPI REST endpoints, PostgreSQL + pgvector persistence, telemetry, React dashboard. | Planned |

---

## 📄 License
This project is licensed under the MIT License.

