import type { Dataset, Experiment, ExperimentDetail, QueryResult } from './types'

export const demoDatasets: Dataset[] = [
  {
    id: 'ds_001',
    name: 'TREC DL 2019',
    documentCount: 3213456,
    questionCount: 43,
    validationStatus: 'valid',
    lastUpdated: new Date('2024-10-15'),
  },
  {
    id: 'ds_002',
    name: 'Natural Questions',
    documentCount: 2189457,
    questionCount: 307373,
    validationStatus: 'valid',
    lastUpdated: new Date('2024-10-10'),
  },
  {
    id: 'ds_003',
    name: 'SQuAD v2.0',
    documentCount: 4895,
    questionCount: 130319,
    validationStatus: 'pending',
    lastUpdated: new Date('2024-10-20'),
  },
]

export const demoExperiments: Experiment[] = [
  {
    id: 'exp_001',
    name: 'BM25 Baseline',
    datasetId: 'ds_001',
    method: 'BM25',
    topK: 10,
    config: { algorithm: 'okapi_bm25', k1: 1.5, b: 0.75 },
    metric: { recall1: 0.42, recall3: 0.58, recall5: 0.68, mrr: 0.51, latency: 12 },
    status: 'completed',
    createdAt: new Date('2024-10-18T10:30:00'),
    completedAt: new Date('2024-10-18T10:45:00'),
  },
  {
    id: 'exp_002',
    name: 'Dense Retrieval (ColBERT)',
    datasetId: 'ds_001',
    method: 'Dense',
    topK: 10,
    config: { model: 'colbert-v2', nprobe: 100 },
    metric: { recall1: 0.61, recall3: 0.74, recall5: 0.82, mrr: 0.68, latency: 58 },
    status: 'completed',
    createdAt: new Date('2024-10-18T09:00:00'),
    completedAt: new Date('2024-10-18T09:22:00'),
  },
  {
    id: 'exp_003',
    name: 'Hybrid (BM25 + Dense)',
    datasetId: 'ds_001',
    method: 'Hybrid',
    topK: 10,
    config: { bm25_weight: 0.4, dense_weight: 0.6, ensemble: 'rrf' },
    metric: { recall1: 0.67, recall3: 0.79, recall5: 0.86, mrr: 0.72, latency: 72 },
    status: 'completed',
    createdAt: new Date('2024-10-18T08:15:00'),
    completedAt: new Date('2024-10-18T08:42:00'),
  },
  {
    id: 'exp_004',
    name: 'BM25 NQ Baseline',
    datasetId: 'ds_002',
    method: 'BM25',
    topK: 5,
    config: { algorithm: 'okapi_bm25', k1: 1.5, b: 0.75 },
    metric: { recall1: 0.38, recall3: 0.51, recall5: 0.61, mrr: 0.44, latency: 10 },
    status: 'completed',
    createdAt: new Date('2024-10-17T14:20:00'),
    completedAt: new Date('2024-10-17T14:35:00'),
  },
  {
    id: 'exp_005',
    name: 'Dense Reranking Experiment',
    datasetId: 'ds_002',
    method: 'Dense',
    topK: 5,
    config: { model: 'colbert-v2', nprobe: 128, use_reranker: true },
    metric: { recall1: 0.52, recall3: 0.68, recall5: 0.76, mrr: 0.61, latency: 145 },
    status: 'completed',
    createdAt: new Date('2024-10-16T11:00:00'),
    completedAt: new Date('2024-10-16T11:35:00'),
  },
]

const demoQueryResults: QueryResult[] = [
  {
    query: 'What is the capital of France?',
    documentId: 'doc_4521',
    isRelevant: true,
    rank: 1,
    score: 0.92,
  },
  {
    query: 'What is the capital of France?',
    documentId: 'doc_1234',
    isRelevant: true,
    rank: 2,
    score: 0.88,
  },
  {
    query: 'What is the capital of France?',
    documentId: 'doc_7890',
    isRelevant: false,
    rank: 3,
    score: 0.72,
  },
  {
    query: 'What is the capital of France?',
    documentId: 'doc_2156',
    isRelevant: true,
    rank: 4,
    score: 0.68,
  },
  {
    query: 'What is the capital of France?',
    documentId: 'doc_9823',
    isRelevant: true,
    rank: 5,
    score: 0.64,
  },
]

export const demoExperimentDetail: ExperimentDetail = {
  ...demoExperiments[2],
  datasetVersion: 'v2.1.0',
  queryResults: demoQueryResults,
}

export function getDatasetById(id: string): Dataset | undefined {
  return demoDatasets.find((d) => d.id === id)
}

export function getExperimentById(id: string): Experiment | undefined {
  return demoExperiments.find((e) => e.id === id)
}

export function getExperimentsByDataset(datasetId: string): Experiment[] {
  return demoExperiments.filter((e) => e.datasetId === datasetId)
}
