# RAGBench: Platform Architecture & Future Evolution

This document outlines the architectural roadmap for scaling RAGBench from its initial foundation milestone (Recall@K evaluation) into a comprehensive, production-grade RAG evaluation and benchmarking platform.

---

## 1. System Architecture Overview

```
                                +---------------------------+
                                |  Web Dashboard / UI       |
                                |  (Next.js / React)        |
                                +-------------+-------------+
                                              |
                                              v
                                +---------------------------+
                                |  FastAPI Backend Service  |
                                |  - /api/v1/evaluations    |
                                |  - /api/v1/benchmarks     |
                                |  - /api/v1/runs           |
                                +-------------+-------------+
                                              |
                     +------------------------+------------------------+
                     |                                                 |
                     v                                                 v
         +-----------------------+                         +-----------------------+
         | Retrieval Pipelines   |                         | Evaluation Engine     |
         | - BM25 (Rank-BM25)    |                         | - Retrieval Metrics   |
         | - Dense Embeddings    |                         |   (Recall, MRR, NDCG) |
         | - Hybrid Search (RRF) |                         | - Generation Metrics  |
         | - Cross-Encoder Rerank|                         |   (Faithfulness, Rel) |
         +-----------+-----------+                         +-----------+-----------+
                     |                                                 |
                     v                                                 v
         +-----------------------+                         +-----------------------+
         | Vector / Relational DB|                         | Experiment Telemetry  |
         | PostgreSQL + pgvector |                         | Latency, Cost, Token  |
         +-----------------------+                         +-----------------------+
```

---

## 2. Planned Subsystem Milestones

### Phase 2: First-Stage Retrievers & Hybrid Search
- **BM25 Lexical Retriever**: Inverted index retrieval using BM25Okapi for exact token and keyword matching.
- **Dense Vector Retriever**: Embedding model integration (e.g. `sentence-transformers`, text-embedding-3) with cosine / dot-product similarity.
- **Hybrid Fusion (RRF)**: Reciprocal Rank Fusion combining sparse and dense rankings with configurable weights.
- **Cross-Encoder Reranker**: Two-stage retrieval pipeline passing top-$N$ candidates to a cross-encoder scoring model.

### Phase 3: Comprehensive Metric Suite
- **Retrieval Metrics**:
  - Precision@K
  - Mean Reciprocal Rank (MRR)
  - Normalized Discounted Cumulative Gain (NDCG@K)
  - Hit Rate@K
- **Generation Quality Metrics (RAG Triad)**:
  - Context Relevance (is the retrieved text relevant to the prompt?)
  - Groundedness / Faithfulness (is the generated answer supported by context?)
  - Answer Relevance (does the answer directly address the user query?)

### Phase 4: Golden Benchmark Dataset Expansion
- Expand sample benchmark to 50–100+ domain-specific question-answer-context triples across multiple domains (Technical Docs, Healthcare, Legal, Finance).
- Support for chunk-level ground-truth annotations and synthetic dataset generation.

### Phase 5: Storage, API, and Deployment
- **FastAPI Core**: RESTful API for submitting benchmark runs, querying historical metrics, and triggering CI/CD evals.
- **PostgreSQL + pgvector**: Persistent storage for documents, embeddings, runs, and query histories.
- **Experiment Tracking**: Latency (p50, p95, p99), token consumption, API cost estimation per query.
- **Containerization**: Standard Docker and Docker Compose definitions for reproducible local and cloud deployment.
- **Web Dashboard**: Interactive visual leaderboard comparing retrieval strategies and ablation studies.
