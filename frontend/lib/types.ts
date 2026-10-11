// Backend API contract schemas
export interface HealthResponse {
  status: string
  version: string
  service: string
  corpus_loaded: boolean
  corpus_doc_count: number
  available_retrievers: string[]
}

export interface DatasetInfo {
  dataset_id: string
  name: string
  description: string
  num_queries: number
  corpus_doc_count: number
  default_k_values: number[]
}

export interface DatasetListResponse {
  datasets: DatasetInfo[]
}

export interface EvaluationRequest {
  dataset_id: string
  k_values: number[]
  include_baselines: boolean
}

export interface SystemSummary {
  system_name: string
  system_type: 'live' | 'illustrative'
  is_live: boolean
  description: string
  num_queries: number
  mrr: number
  mean_recall_at_k: Record<string, number>
}

export interface QueryResultDetail {
  query_id: string
  query_text: string
  system_name: string
  system_type: 'live' | 'illustrative'
  total_relevant: number
  retrieved_count: number
  reciprocal_rank: number
  recall_at_k: Record<string, number>
  retrieved_doc_ids: string[]
}

export interface EvaluationResponse {
  dataset_id: string
  k_values: number[]
  queries_evaluated: number
  corpus_size: number
  summaries: SystemSummary[]
  query_results: QueryResultDetail[]
  disclaimer: string
}

export interface ExperimentItem {
  id: string
  created_at: string
  dataset_id: string
  system_name: string
  mrr: number
}

// UI Domain models
export type RetrievalMethod = 'BM25' | 'Dense' | 'Hybrid'
export type ValidationStatus = 'valid' | 'invalid' | 'pending'
export type RunStatus = 'pending' | 'running' | 'completed' | 'failed'

export interface Metric {
  recall1: number
  recall3: number
  recall5: number
  mrr: number
  latency?: number
}

export interface Dataset {
  id: string
  name: string
  description?: string
  documentCount: number
  questionCount: number
  validationStatus: ValidationStatus
  lastUpdated: Date
}

export interface Experiment {
  id: string
  name: string
  datasetId: string
  method: RetrievalMethod
  topK: number
  config: Record<string, unknown>
  metric: Metric
  status: RunStatus
  isLive?: boolean
  createdAt: Date
  completedAt?: Date
}

export interface QueryResult {
  query: string
  documentId: string
  isRelevant: boolean
  rank: number
  score: number
}

export interface ExperimentDetail extends Experiment {
  datasetVersion: string
  queryResults: QueryResult[]
}
