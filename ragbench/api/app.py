"""FastAPI application factory and route definitions for RAGBench."""

import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from ragbench.api.schemas import (
    DatasetInfo,
    DatasetListResponse,
    EvaluationRequest,
    EvaluationResponse,
    ExperimentItem,
    HealthResponse,
    QueryResultDetail,
    SystemSummary,
)
from ragbench.core.models import Document, EvaluationQuery
from ragbench.datasets.loader import (
    load_sample_corpus,
    load_sample_dataset,
)
from ragbench.evaluators.retrieval_evaluator import RetrievalEvaluator
from ragbench.retrievers.bm25 import BM25Retriever

# Package version
__version__ = "0.1.0"


def get_allowed_origins() -> List[str]:
    """Parse CORS permitted origins from environment or default to common dev ports."""
    env_origins = os.getenv("RAGBENCH_CORS_ORIGINS", "")
    if env_origins.strip():
        return [origin.strip() for origin in env_origins.split(",") if origin.strip()]
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="RAGBench Evaluation API",
        description=(
            "Production-oriented benchmark and evaluation backend for "
            "Retrieval-Augmented Generation (RAG) pipelines."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Configure CORS for frontend integration
    origins = get_allowed_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -------------------------------------------------------------------------
    # 1. GET /health
    # -------------------------------------------------------------------------
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["System"],
        summary="Service Health Check",
    )
    def health_check() -> HealthResponse:
        """Return application health, version, and indexed corpus status."""
        try:
            corpus = load_sample_corpus()
            corpus_loaded = True
            corpus_doc_count = len(corpus)
        except Exception:
            corpus_loaded = False
            corpus_doc_count = 0

        return HealthResponse(
            status="ok",
            version=__version__,
            service="RAGBench Evaluation API",
            corpus_loaded=corpus_loaded,
            corpus_doc_count=corpus_doc_count,
            available_retrievers=["Real_BM25_Retriever"],
        )

    # -------------------------------------------------------------------------
    # 2. GET /api/datasets
    # -------------------------------------------------------------------------
    @app.get(
        "/api/datasets",
        response_model=DatasetListResponse,
        tags=["Datasets"],
        summary="List Available Benchmark Datasets",
    )
    def list_datasets() -> DatasetListResponse:
        """Return metadata for all available benchmark datasets supported by the loader."""
        try:
            queries, _ = load_sample_dataset()
            num_queries = len(queries)
        except Exception:
            num_queries = 0

        try:
            corpus = load_sample_corpus()
            corpus_count = len(corpus)
        except Exception:
            corpus_count = 0

        sample_info = DatasetInfo(
            dataset_id="sample",
            name="RAG Core Mechanics Benchmark",
            description=(
                "Curated benchmark covering document chunking strategies, text splitting, "
                "vector embeddings, approximate nearest neighbors, and context budgeting."
            ),
            num_queries=num_queries,
            corpus_doc_count=corpus_count,
            default_k_values=[1, 3, 5],
        )

        return DatasetListResponse(datasets=[sample_info])

    # -------------------------------------------------------------------------
    # 3. POST /api/evaluate
    # -------------------------------------------------------------------------
    @app.post(
        "/api/evaluate",
        response_model=EvaluationResponse,
        tags=["Evaluation"],
        summary="Run Retrieval Benchmark Evaluation",
    )
    def run_evaluation(request: EvaluationRequest) -> EvaluationResponse:
        """Execute BM25 retrieval against the actual corpus and calculate metrics.

        - Executes live BM25 retrieval for all benchmark queries.
        - Calculates Reciprocal Rank, MRR, and Recall@K at requested cutoffs.
        - Optionally includes illustrative static baseline rankings for comparative inspection.
        """
        # Validate dataset existence
        if request.dataset_id != "sample":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dataset '{request.dataset_id}' not found. Supported datasets: ['sample']",
            )

        # 1. Load ground truth benchmark and text corpus
        try:
            queries, reference_rankings = load_sample_dataset()
            corpus = load_sample_corpus()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed loading benchmark dataset or corpus: {e}",
            )

        # 2. Build live BM25 retriever index from real corpus documents
        try:
            bm25 = BM25Retriever.from_documents(corpus)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed building BM25 index: {e}",
            )

        # 3. Execute real BM25 search on every query
        max_k = max(request.k_values)
        real_bm25_rankings: Dict[str, List[str]] = {}
        for q in queries:
            retrieved_ids = bm25.retrieve_ids(q.query_text, top_k=max_k)
            real_bm25_rankings[q.query_id] = retrieved_ids

        # 4. Assemble system rankings to evaluate
        systems_rankings: Dict[str, Dict[str, List[str]]] = {
            "Real_BM25_Retriever": real_bm25_rankings,
        }

        # Include illustrative baselines only if explicitly requested
        if request.include_baselines:
            for baseline_name, query_map in reference_rankings.items():
                systems_rankings[baseline_name] = query_map

        # 5. Execute retrieval evaluation engine
        evaluator = RetrievalEvaluator(k_values=request.k_values)
        report = evaluator.evaluate(queries, systems_rankings)

        # 6. Format system summaries
        summaries: List[SystemSummary] = []
        for sys_name, summary in report.summaries.items():
            if sys_name == "Real_BM25_Retriever":
                sys_type = "live"
                is_live = True
                desc = "Dynamically executed BM25 ranking against corpus text."
            else:
                sys_type = "illustrative"
                is_live = False
                desc = "Predefined static rankings (illustrative baseline, not a live execution)."

            # Convert int keys to string for JSON compatibility
            mean_recall_str_keys = {
                str(k): round(val, 4) for k, val in summary.mean_recall_at_k.items()
            }

            summaries.append(
                SystemSummary(
                    system_name=sys_name,
                    system_type=sys_type,
                    is_live=is_live,
                    description=desc,
                    num_queries=summary.num_queries,
                    mrr=round(summary.mrr, 4),
                    mean_recall_at_k=mean_recall_str_keys,
                )
            )

        # 7. Format per-query detailed results
        query_details: List[QueryResultDetail] = []
        for q_metric in report.query_results:
            sys_name = q_metric.system_name
            is_live_sys = (sys_name == "Real_BM25_Retriever")
            retrieved_doc_ids = systems_rankings.get(sys_name, {}).get(q_metric.query_id, [])

            recall_str_keys = {
                str(k): round(val, 4) for k, val in q_metric.recall_at_k.items()
            }

            query_details.append(
                QueryResultDetail(
                    query_id=q_metric.query_id,
                    query_text=q_metric.query_text,
                    system_name=sys_name,
                    system_type="live" if is_live_sys else "illustrative",
                    total_relevant=q_metric.total_relevant,
                    retrieved_count=q_metric.retrieved_count,
                    reciprocal_rank=round(q_metric.reciprocal_rank, 4),
                    recall_at_k=recall_str_keys,
                    retrieved_doc_ids=retrieved_doc_ids,
                )
            )

        disclaimer = (
            "NOTICE: 'Real_BM25_Retriever' was executed live against the indexed text corpus. "
            "Systems labeled 'illustrative' are predefined static benchmark rankings intended for "
            "comparative reference only."
        )

        return EvaluationResponse(
            dataset_id=request.dataset_id,
            k_values=request.k_values,
            queries_evaluated=len(queries),
            corpus_size=len(corpus),
            summaries=summaries,
            query_results=query_details,
            disclaimer=disclaimer,
        )

    # -------------------------------------------------------------------------
    # 4. GET /api/experiments
    # -------------------------------------------------------------------------
    @app.get(
        "/api/experiments",
        response_model=List[ExperimentItem],
        tags=["Experiments"],
        summary="List Saved Experiment History",
    )
    def list_experiments() -> List[ExperimentItem]:
        """Return experiment history.

        Note: Persistent experiment storage is not yet implemented.
        Returns an empty list rather than simulated data.
        """
        return []

    return app


# Default application instance for uvicorn
app = create_app()
