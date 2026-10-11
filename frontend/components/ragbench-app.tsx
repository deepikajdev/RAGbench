'use client'

import { useEffect, useState } from 'react'
import {
  AlertTriangle,
  ArrowUpRight,
  BarChart3,
  Check,
  ChevronRight,
  Database,
  FileCheck2,
  FlaskConical,
  Gauge,
  Info,
  LayoutDashboard,
  Menu,
  MoreHorizontal,
  Play,
  Plus,
  RefreshCw,
  Search,
  Settings2,
  SlidersHorizontal,
  Sparkles,
} from 'lucide-react'
import {
  executeEvaluation,
  fetchDatasets,
  fetchExperiments,
  fetchHealth,
  getApiBaseUrl,
  setApiBaseUrl,
} from '@/lib/api-client'
import type {
  DatasetInfo,
  EvaluationResponse,
  ExperimentItem,
  HealthResponse,
  QueryResultDetail,
  RetrievalMethod,
  SystemSummary,
} from '@/lib/types'
import { Button } from '@/components/ui/button'

const navItems = [
  { label: 'Overview', icon: LayoutDashboard },
  { label: 'Datasets', icon: Database },
  { label: 'Experiments', icon: FlaskConical },
  { label: 'Compare', icon: BarChart3 },
]

const formatPercent = (value: number) => `${(value * 100).toFixed(1)}%`

function MetricCard({
  label,
  value,
  note,
  isLive,
}: {
  label: string
  value: string
  note: string
  isLive?: boolean
}) {
  return (
    <div className="metric-block">
      <div className="eyebrow" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span>{label}</span>
        {isLive && <span className="tag-live">LIVE</span>}
      </div>
      <div className="metric-value">{value}</div>
      <div className="metric-note">{note}</div>
    </div>
  )
}

export default function RagbenchApp() {
  const [page, setPage] = useState('Overview')
  const [mobileOpen, setMobileOpen] = useState(false)

  // Backend state
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthLoading, setHealthLoading] = useState(false)
  const [datasets, setDatasets] = useState<DatasetInfo[]>([])
  const [datasetsLoading, setDatasetsLoading] = useState(false)
  const [experiments, setExperiments] = useState<ExperimentItem[]>([])
  const [experimentsLoading, setExperimentsLoading] = useState(false)

  // Evaluation runs state (session storage in memory)
  const [evaluationRuns, setEvaluationRuns] = useState<EvaluationResponse[]>([])
  const [activeRun, setActiveRun] = useState<EvaluationResponse | null>(null)
  const [evaluating, setEvaluating] = useState(false)
  const [evalError, setEvalError] = useState<string | null>(null)
  const [apiError, setApiError] = useState<string | null>(null)

  // Evaluation form configuration state (persisted across page navigation)
  const [datasetId, setDatasetId] = useState('sample')
  const [k1, setK1] = useState(true)
  const [k3, setK3] = useState(true)
  const [k5, setK5] = useState(true)
  const [includeBaselines, setIncludeBaselines] = useState(true)
  const [method, setMethod] = useState<RetrievalMethod>('BM25')

  // Settings configuration state (persisted across page navigation)
  const [endpoint, setEndpoint] = useState(getApiBaseUrl())
  const [statusMsg, setStatusMsg] = useState<string | null>(null)
  const [testing, setTesting] = useState(false)

  const navigate = (next: string) => {
    setPage(next)
    setMobileOpen(false)
  }

  // Load initial backend status and datasets
  const loadBackendData = async () => {
    setApiError(null)
    setHealthLoading(true)
    setDatasetsLoading(true)
    setExperimentsLoading(true)

    try {
      const h = await fetchHealth()
      setHealth(h)
    } catch (err: unknown) {
      setHealth(null)
      const msg = err instanceof Error ? err.message : 'Failed to reach API'
      setApiError(`Could not connect to FastAPI backend at ${getApiBaseUrl()}: ${msg}`)
    } finally {
      setHealthLoading(false)
    }

    try {
      const d = await fetchDatasets()
      setDatasets(d.datasets)
    } catch {
      // Ignored if health already captured error
    } finally {
      setDatasetsLoading(false)
    }

    try {
      const exp = await fetchExperiments()
      setExperiments(exp)
    } catch {
      // Experiments endpoint returns []
    } finally {
      setExperimentsLoading(false)
    }
  }

  useEffect(() => {
    loadBackendData()
  }, [])

  // Handler for running evaluation
  const handleRunEvaluation = async (
    datasetId: string,
    kValues: number[],
    includeBaselines: boolean,
  ) => {
    setEvaluating(true)
    setEvalError(null)

    try {
      const result = await executeEvaluation({
        dataset_id: datasetId,
        k_values: kValues,
        include_baselines: includeBaselines,
      })

      setEvaluationRuns((prev) => [result, ...prev])
      setActiveRun(result)
      navigate('detail:latest')
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Evaluation request failed'
      setEvalError(msg)
    } finally {
      setEvaluating(false)
    }
  }

  const latestLiveRun = evaluationRuns[0] || null
  const realBm25Summary = latestLiveRun?.summaries.find((s) => s.is_live) || null

  // -------------------------------------------------------------------------
  // PAGE RENDERERS
  // -------------------------------------------------------------------------

  function renderOverview() {
    return (
      <div className="page-content">
        <header className="page-header">
          <div>
            <div className="breadcrumb">
              Workspace <ChevronRight /> Overview
            </div>
            <h1>Retrieval, measured.</h1>
            <p>Evaluate information retrieval systems against reproducible benchmarks with verified metrics.</p>
          </div>
          <Button onClick={() => navigate('Experiments')}>
            <Plus data-icon="inline-start" /> New evaluation
          </Button>
        </header>

        {apiError && (
          <div className="alert-box">
            <AlertTriangle />
            <span>{apiError}</span>
            <button onClick={() => navigate('Settings')}>Configure API &rarr;</button>
          </div>
        )}

        <div className="demo-banner">
          <Sparkles />
          <span>
            {health?.status === 'ok' ? (
              <>
                <strong>BACKEND LIVE</strong> Connected to FastAPI ({health.service} v{health.version}).{' '}
                {health.corpus_doc_count} documents indexed in BM25 corpus.
              </>
            ) : (
              <>
                <strong>BACKEND OFFLINE</strong> No active connection to {getApiBaseUrl()}. Check Settings to configure
                and start the server.
              </>
            )}
          </span>
          <button onClick={() => navigate('Settings')}>
            Backend Settings <ArrowUpRight />
          </button>
        </div>

        <section className="metrics-grid">
          <MetricCard
            label="Evaluation runs"
            value={evaluationRuns.length > 0 ? String(evaluationRuns.length) : '0'}
            note={evaluationRuns.length > 0 ? 'Live in-memory runs this session' : 'None executed yet'}
          />
          <MetricCard
            label="Available datasets"
            value={datasets.length > 0 ? String(datasets.length) : '0'}
            note={datasets[0] ? `${datasets[0].name} (${datasets[0].corpus_doc_count} docs)` : 'Loading...'}
          />
          <MetricCard
            label="Live BM25 Recall@5"
            value={realBm25Summary ? formatPercent(realBm25Summary.mean_recall_at_k['5'] || 0) : '--'}
            note={realBm25Summary ? 'Real executed BM25 on corpus' : 'Execute an evaluation to measure'}
            isLive={Boolean(realBm25Summary)}
          />
          <MetricCard
            label="Live BM25 MRR"
            value={realBm25Summary ? realBm25Summary.mrr.toFixed(4) : '--'}
            note={realBm25Summary ? 'Mean Reciprocal Rank across 5 queries' : 'Execute an evaluation to measure'}
            isLive={Boolean(realBm25Summary)}
          />
        </section>

        {evaluationRuns.length > 0 ? (
          <section className="section-block">
            <div className="section-heading">
              <div>
                <div className="eyebrow">Session run history</div>
                <h2>Live Evaluations ({evaluationRuns.length})</h2>
              </div>
              <button className="text-button" onClick={() => navigate('Experiments')}>
                Run another <ArrowUpRight />
              </button>
            </div>
            <div className="table-shell">
              <table>
                <thead>
                  <tr>
                    <th>System / Engine</th>
                    <th>Type</th>
                    <th>Dataset</th>
                    <th className="numeric">Queries</th>
                    <th className="numeric">MRR</th>
                    <th className="numeric">Recall@1</th>
                    <th className="numeric">Recall@5</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {evaluationRuns.map((run, runIdx) => {
                    const liveSys = run.summaries.find((s) => s.is_live)
                    return (
                      <tr
                        key={runIdx}
                        className="row-action"
                        onClick={() => {
                          setActiveRun(run)
                          navigate('detail:latest')
                        }}
                      >
                        <td>
                          <strong>{liveSys?.system_name || 'BM25 Evaluation'}</strong>
                        </td>
                        <td>
                          <span className="badge-live">[LIVE]</span>
                        </td>
                        <td className="mono">{run.dataset_id}</td>
                        <td className="numeric">{run.queries_evaluated}</td>
                        <td className="numeric strong-number">{liveSys?.mrr.toFixed(4)}</td>
                        <td className="numeric">{formatPercent(liveSys?.mean_recall_at_k['1'] || 0)}</td>
                        <td className="numeric">{formatPercent(liveSys?.mean_recall_at_k['5'] || 0)}</td>
                        <td>
                          <ChevronRight className="row-chevron" />
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </section>
        ) : (
          <div className="empty-card">
            <h3>No live evaluations run yet</h3>
            <p>
              Click <strong>&quot;New evaluation&quot;</strong> to execute the real Python BM25 retriever against the
              bundled 19-document corpus and benchmark queries.
            </p>
            <div style={{ marginTop: '16px' }}>
              <Button onClick={() => navigate('Experiments')}>
                <Play data-icon="inline-start" /> Launch BM25 Benchmark Run
              </Button>
            </div>
          </div>
        )}
      </div>
    )
  }

  function renderDatasets() {
    return (
      <div className="page-content">
        <header className="page-header">
          <div>
            <div className="breadcrumb">
              Workspace <ChevronRight /> Datasets
            </div>
            <h1>Benchmark Datasets.</h1>
            <p>Curated corpora and ground-truth query relevance collections supported by RAGBench.</p>
          </div>
          <Button onClick={loadBackendData} disabled={datasetsLoading}>
            <RefreshCw data-icon="inline-start" /> {datasetsLoading ? 'Refreshing...' : 'Refresh'}
          </Button>
        </header>

        {datasetsLoading ? (
          <div className="empty-card">
            <h3>Loading datasets from backend...</h3>
            <p>Connecting to {getApiBaseUrl()}/api/datasets</p>
          </div>
        ) : datasets.length === 0 ? (
          <div className="empty-card">
            <h3>No datasets returned</h3>
            <p>The backend did not return any datasets. Verify that the FastAPI server is running.</p>
          </div>
        ) : (
          <div className="dataset-grid">
            {datasets.map((ds) => (
              <div
                key={ds.dataset_id}
                className="dataset-card"
                onClick={() => navigate(`dataset:${ds.dataset_id}`)}
                style={{ cursor: 'pointer' }}
              >
                <div className="dataset-card-top">
                  <span className="mono id-code">{ds.dataset_id}</span>
                  <span className="validation validation-valid">
                    <Check /> VALIDATED
                  </span>
                </div>
                <h2>{ds.name}</h2>
                <p style={{ color: '#666', fontSize: '11px', margin: '0 0 16px', lineHeight: 1.4 }}>
                  {ds.description}
                </p>
                <div className="dataset-stats">
                  <span>
                    <strong>{ds.corpus_doc_count}</strong> corpus docs
                  </span>
                  <span>
                    <strong>{ds.num_queries}</strong> eval queries
                  </span>
                </div>
                <div className="dataset-footer">
                  Cutoffs K = [{ds.default_k_values.join(', ')}] <ChevronRight />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    )
  }

  function renderDatasetDetail(datasetId: string) {
    const ds = datasets.find((d) => d.dataset_id === datasetId) || datasets[0]
    return (
      <div className="page-content">
        <header className="page-header">
          <div>
            <div className="breadcrumb">
              <button onClick={() => navigate('Datasets')}>Datasets</button> <ChevronRight /> Detail
            </div>
            <h1>{ds ? ds.name : 'Dataset Detail'}</h1>
            <p>
              <span className="mono">{ds?.dataset_id || datasetId}</span> · Bundled technical benchmark for RAG retrieval
            </p>
          </div>
          <span className="validation validation-valid">
            <Check /> READY FOR RETRIEVAL
          </span>
        </header>

        <div className="detail-meta dataset-detail-meta">
          <div>
            <span className="eyebrow">Corpus Documents</span>
            <strong>{ds?.corpus_doc_count || 19}</strong>
            <span className="mono">Text passages indexed</span>
          </div>
          <div>
            <span className="eyebrow">Benchmark Queries</span>
            <strong>{ds?.num_queries || 5}</strong>
            <span className="mono">Ground-truth pairs</span>
          </div>
          <div>
            <span className="eyebrow">Cutoff Depths</span>
            <strong>K = [{ds?.default_k_values.join(', ') || '1, 3, 5'}]</strong>
            <span className="mono">Evaluated ranks</span>
          </div>
        </div>

        <section className="section-block">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Corpus Topics & Coverage</div>
              <h2>Document Corpus Topics</h2>
            </div>
          </div>
          <p style={{ color: '#666', fontSize: '13px', lineHeight: 1.6 }}>
            Covers text splitting and chunk overlap mechanics, fixed vs. semantic chunking, dense vector embeddings,
            cosine similarity vs. Euclidean distance, lost-in-the-middle context dilution, hallucination root causes,
            cross-encoder reranking, and Reciprocal Rank Fusion (RRF).
          </p>
        </section>

        <section className="section-block">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Benchmark Query Samples</div>
              <h2>Sample Questions</h2>
            </div>
          </div>
          <div className="question-list">
            <div>
              <span className="mono">q_01</span>
              <strong>What is chunk overlap and why is it used in text splitting?</strong>
              <span className="mono">rel: doc_chunking_overlap, doc_chunking_strategies</span>
            </div>
            <div>
              <span className="mono">q_02</span>
              <strong>How do dense vector embeddings differ from sparse BM25 lexical search?</strong>
              <span className="mono">rel: doc_bm25_sparse_retrieval, doc_dense_embeddings</span>
            </div>
            <div>
              <span className="mono">q_03</span>
              <strong>What causes context dilution and &apos;lost in the middle&apos; phenomena in RAG generation?</strong>
              <span className="mono">rel: doc_lost_in_the_middle, doc_hallucination_causes</span>
            </div>
          </div>
        </section>

        <div style={{ marginTop: '28px' }}>
          <Button onClick={() => navigate('Experiments')}>
            <Play data-icon="inline-start" /> Run BM25 Evaluation On This Dataset
          </Button>
        </div>
      </div>
    )
  }

  function renderExperiments() {
    const handleSubmit = (e: React.FormEvent) => {
      e.preventDefault()
      if (evaluating) return

      const selectedK: number[] = []
      if (k1) selectedK.push(1)
      if (k3) selectedK.push(3)
      if (k5) selectedK.push(5)

      if (selectedK.length === 0) {
        setEvalError('Please select at least one K cutoff value (e.g. 1, 3, or 5).')
        return
      }

      handleRunEvaluation(datasetId, selectedK, includeBaselines)
    }

    return (
      <div className="page-content">
        <header className="page-header">
          <div>
            <div className="breadcrumb">
              Workspace <ChevronRight /> Experiments
            </div>
            <h1>New evaluation.</h1>
            <p>Execute real BM25 retrieval against the corpus and compute verified Recall@K and MRR metrics.</p>
          </div>
          <span className="badge-live">[REAL EXECUTION]</span>
        </header>

        {evalError && (
          <div className="alert-box">
            <AlertTriangle />
            <span>{evalError}</span>
            <button onClick={() => setEvalError(null)}>Dismiss</button>
          </div>
        )}

        <div className="experiment-form-layout">
          <form className="config-form" onSubmit={handleSubmit}>
            <div className="form-section">
              <div className="eyebrow">01 / Target Dataset</div>
              <label>
                Dataset
                <select value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
                  {datasets.length > 0 ? (
                    datasets.map((d) => (
                      <option key={d.dataset_id} value={d.dataset_id}>
                        {d.name} ({d.dataset_id}) · {d.corpus_doc_count} docs
                      </option>
                    ))
                  ) : (
                    <option value="sample">RAG Core Mechanics Benchmark (sample)</option>
                  )}
                </select>
              </label>
            </div>

            <div className="form-section">
              <div className="eyebrow">02 / Retrieval Strategy</div>
              <div className="strategy-options">
                <button
                  type="button"
                  className={method === 'BM25' ? 'strategy selected' : 'strategy'}
                  onClick={() => setMethod('BM25')}
                >
                  <span className="strategy-radio" />
                  <strong>BM25</strong>
                  <small>Real Python Okapi BM25 engine running against corpus text</small>
                </button>
                <button
                  type="button"
                  className={method === 'Dense' ? 'strategy selected' : 'strategy'}
                  onClick={() => setMethod('Dense')}
                  title="Phase 4 milestone"
                >
                  <span className="strategy-radio" />
                  <strong>Dense</strong>
                  <small>Phase 4 milestone (Dense embeddings planned)</small>
                </button>
                <button
                  type="button"
                  className={method === 'Hybrid' ? 'strategy selected' : 'strategy'}
                  onClick={() => setMethod('Hybrid')}
                  title="Phase 4 milestone"
                >
                  <span className="strategy-radio" />
                  <strong>Hybrid</strong>
                  <small>Phase 4 milestone (RRF Fusion planned)</small>
                </button>
              </div>
            </div>

            <div className="form-section">
              <div className="eyebrow">03 / Metric Cutoffs (Recall@K)</div>
              <div style={{ display: 'flex', gap: '20px', marginTop: '6px' }}>
                <label style={{ flexDirection: 'row', alignItems: 'center', gap: '8px' }}>
                  <input type="checkbox" checked={k1} onChange={(e) => setK1(e.target.checked)} />
                  <span>K = 1</span>
                </label>
                <label style={{ flexDirection: 'row', alignItems: 'center', gap: '8px' }}>
                  <input type="checkbox" checked={k3} onChange={(e) => setK3(e.target.checked)} />
                  <span>K = 3</span>
                </label>
                <label style={{ flexDirection: 'row', alignItems: 'center', gap: '8px' }}>
                  <input type="checkbox" checked={k5} onChange={(e) => setK5(e.target.checked)} />
                  <span>K = 5</span>
                </label>
              </div>
            </div>

            <div className="form-section">
              <div className="eyebrow">04 / Baseline Comparison</div>
              <label style={{ flexDirection: 'row', alignItems: 'flex-start', gap: '10px' }}>
                <input
                  type="checkbox"
                  checked={includeBaselines}
                  onChange={(e) => setIncludeBaselines(e.target.checked)}
                  style={{ marginTop: '3px' }}
                />
                <div>
                  <strong>Include illustrative reference baselines</strong>
                  <small style={{ display: 'block', color: '#777', marginTop: '2px' }}>
                    Includes static benchmark baselines (Lexical_BM25_Baseline, Dense_Semantic_Baseline) clearly labeled
                    as [ILLUSTRATIVE] for side-by-side comparison.
                  </small>
                </div>
              </label>
            </div>

            <div className="form-actions">
              <Button type="submit" disabled={evaluating}>
                <Play data-icon="inline-start" />
                {evaluating ? 'Executing BM25 against corpus...' : 'Run evaluation'}
              </Button>
              {evaluating && (
                <span className="form-message">
                  Sending POST /api/evaluate to backend...
                </span>
              )}
            </div>
          </form>

          <aside className="config-aside">
            <div className="eyebrow">Live Backend Execution</div>
            <p>
              When you submit this form, RAGBench issues a real request to <code>POST /api/evaluate</code>. The Python
              backend indexes the 19 documents, tokenizes all benchmark queries, computes IDF scores, and generates
              genuine rankings and metrics.
            </p>
            <div className="aside-note">
              <Gauge />
              <span>
                <strong>No fake data:</strong> Live results are stamped with <code>is_live: true</code>, while reference
                baselines are labeled <code>[ILLUSTRATIVE]</code>.
              </span>
            </div>
          </aside>
        </div>

        {/* Experiment History Section */}
        <section className="section-block" style={{ marginTop: '54px' }}>
          <div className="section-heading">
            <div>
              <div className="eyebrow">Persistent Experiment History</div>
              <h2>Backend Saved Runs</h2>
            </div>
          </div>
          {experimentsLoading ? (
            <p style={{ color: '#888', font: '11px var(--font-mono)' }}>Checking /api/experiments...</p>
          ) : experiments.length === 0 ? (
            <div className="empty-card">
              <h3>Empty Experiment History</h3>
              <p>
                <code>GET /api/experiments</code> returned an empty list. RAGBench Phase 3 executes evaluations
                on-demand in memory. Persistent experiment storage (database) is reserved for a future milestone.
              </p>
            </div>
          ) : null}
        </section>
      </div>
    )
  }

  function renderDetail(run: EvaluationResponse | null) {
    if (!run) {
      return (
        <div className="page-content">
          <div className="empty-card">
            <h3>No run selected</h3>
            <p>Please launch an evaluation first.</p>
            <Button onClick={() => navigate('Experiments')} style={{ marginTop: '12px' }}>
              Go to Experiments
            </Button>
          </div>
        </div>
      )
    }

    const liveSummary = run.summaries.find((s) => s.is_live)
    const otherSummaries = run.summaries.filter((s) => !s.is_live)

    return (
      <div className="page-content">
        <header className="page-header">
          <div>
            <div className="breadcrumb">
              <button onClick={() => navigate('Overview')}>Overview</button> <ChevronRight /> Evaluation Run Details
            </div>
            <div className="detail-title">
              <span className="badge-live">[LIVE BM25 EXECUTION]</span>
              <h1>Real BM25 Evaluation Results</h1>
            </div>
            <p>
              Dataset: <span className="mono">{run.dataset_id}</span> · Evaluated {run.queries_evaluated} queries across{' '}
              {run.corpus_size} corpus documents
            </p>
          </div>
          <Button onClick={() => navigate('Experiments')}>
            <Play data-icon="inline-start" /> Run New Evaluation
          </Button>
        </header>

        {/* Disclaimer Callout */}
        <div className="demo-banner">
          <Info />
          <span>{run.disclaimer}</span>
        </div>

        {/* Top Summary Metrics */}
        <div className="detail-meta">
          <div>
            <span className="eyebrow">MRR (Mean Reciprocal Rank)</span>
            <strong>{liveSummary ? liveSummary.mrr.toFixed(4) : '--'}</strong>
            <span className="mono">{liveSummary ? formatPercent(liveSummary.mrr) : ''} live score</span>
          </div>
          <div>
            <span className="eyebrow">Mean Recall@1 / Recall@3</span>
            <strong>
              {liveSummary?.mean_recall_at_k['1'] ? formatPercent(liveSummary.mean_recall_at_k['1']) : '--'} /{' '}
              {liveSummary?.mean_recall_at_k['3'] ? formatPercent(liveSummary.mean_recall_at_k['3']) : '--'}
            </strong>
            <span className="mono">Rank 1 and 3 cutoffs</span>
          </div>
          <div>
            <span className="eyebrow">Mean Recall@5</span>
            <strong>
              {liveSummary?.mean_recall_at_k['5'] ? formatPercent(liveSummary.mean_recall_at_k['5']) : '--'}
            </strong>
            <span className="mono">Top-5 cutoff completeness</span>
          </div>
        </div>

        {/* Systems Comparison Table */}
        <section className="section-block">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Evaluated Systems</div>
              <h2>System Summaries</h2>
            </div>
          </div>
          <div className="table-shell">
            <table>
              <thead>
                <tr>
                  <th>System Name</th>
                  <th>System Type</th>
                  <th>Description</th>
                  <th className="numeric">MRR</th>
                  {run.k_values.map((k) => (
                    <th key={k} className="numeric">
                      Recall@{k}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {run.summaries.map((s: SystemSummary) => (
                  <tr key={s.system_name}>
                    <td>
                      <strong>{s.system_name}</strong>
                    </td>
                    <td>
                      {s.is_live ? (
                        <span className="badge-live">[LIVE]</span>
                      ) : (
                        <span className="badge-illustrative">[ILLUSTRATIVE]</span>
                      )}
                    </td>
                    <td style={{ fontSize: '11px', color: '#666' }}>{s.description}</td>
                    <td className="numeric strong-number">{s.mrr.toFixed(4)}</td>
                    {run.k_values.map((k) => (
                      <td key={k} className="numeric">
                        {s.mean_recall_at_k[String(k)] !== undefined
                          ? formatPercent(s.mean_recall_at_k[String(k)])
                          : '--'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* Per-Query Breakdown */}
        <section className="section-block">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Per-Query Inspection</div>
              <h2>Query Retrieval Rankings & Retrieved Document IDs</h2>
            </div>
            <span className="badge-live">LIVE BM25 RANKINGS</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {run.query_results
              .filter((q: QueryResultDetail) => q.system_name === 'Real_BM25_Retriever')
              .map((q: QueryResultDetail) => (
                <div key={q.query_id} className="query-card">
                  <div className="query-header">
                    <span className="mono">{q.query_id}</span>
                    <strong style={{ fontSize: '13px' }}>&quot;{q.query_text}&quot;</strong>
                    <div style={{ marginLeft: 'auto', display: 'flex', gap: '16px', alignItems: 'center' }}>
                      <span className="mono" style={{ fontSize: '11px', color: '#444' }}>
                        RR: <b>{q.reciprocal_rank.toFixed(2)}</b>
                      </span>
                      <span className="mono" style={{ fontSize: '11px', color: '#444' }}>
                        Relevant: <b>{q.total_relevant}</b>
                      </span>
                    </div>
                  </div>
                  <div style={{ padding: '14px 16px' }}>
                    <div style={{ marginBottom: '8px', fontSize: '11px', color: '#777' }}>
                      Retrieved Document IDs in rank order:
                    </div>
                    <div>
                      {q.retrieved_doc_ids.map((docId: string, idx: number) => (
                        <span key={docId} className="doc-pill">
                          #{idx + 1} {docId}
                        </span>
                      ))}
                    </div>
                    <div
                      style={{
                        marginTop: '10px',
                        display: 'flex',
                        gap: '12px',
                        fontSize: '11px',
                        color: '#666',
                        fontFamily: 'var(--font-mono)',
                      }}
                    >
                      {run.k_values.map((k) => (
                        <span key={k}>
                          Recall@{k}: {q.recall_at_k[String(k)] !== undefined ? formatPercent(q.recall_at_k[String(k)]) : '--'}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
          </div>
        </section>

        {otherSummaries.length > 0 && (
          <div className="aside-note" style={{ marginTop: '24px' }}>
            <FileCheck2 />
            <span>
              Reference baseline query rankings are available on the backend and documented in{' '}
              <code>ragbench/datasets/sample_dataset.json</code>.
            </span>
          </div>
        )}
      </div>
    )
  }

  function renderCompare() {
    const run = latestLiveRun

    return (
      <div className="page-content">
        <header className="page-header">
          <div>
            <div className="breadcrumb">
              Workspace <ChevronRight /> Compare
            </div>
            <h1>Compare retrieval systems.</h1>
            <p>Side-by-side comparison of real BM25 performance against benchmark baselines.</p>
          </div>
          <Button onClick={() => navigate('Experiments')}>
            <Play data-icon="inline-start" /> New run
          </Button>
        </header>

        {run ? (
          <div>
            <div className="demo-banner">
              <Sparkles />
              <span>
                <strong>COMPARISON VIEW</strong> Comparing systems evaluated in latest run against{' '}
                <code>{run.dataset_id}</code>.
              </span>
            </div>

            <div className="comparison-table table-shell">
              <table>
                <thead>
                  <tr>
                    <th>Metric</th>
                    {run.summaries.map((s) => (
                      <th key={s.system_name}>
                        <div>{s.system_name}</div>
                        <small className="mono">
                          {s.is_live ? (
                            <span className="badge-live">[LIVE]</span>
                          ) : (
                            <span className="badge-illustrative">[ILLUSTRATIVE]</span>
                          )}
                        </small>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>
                      <strong>Mean Reciprocal Rank (MRR)</strong>
                    </td>
                    {run.summaries.map((s) => (
                      <td key={s.system_name} className="compare-number">
                        {s.mrr.toFixed(4)}
                      </td>
                    ))}
                  </tr>
                  {run.k_values.map((k) => (
                    <tr key={k}>
                      <td>
                        <strong>Mean Recall@{k}</strong>
                      </td>
                      {run.summaries.map((s) => (
                        <td key={s.system_name} className="compare-number">
                          {s.mean_recall_at_k[String(k)] !== undefined
                            ? formatPercent(s.mean_recall_at_k[String(k)])
                            : '--'}
                        </td>
                      ))}
                    </tr>
                  ))}
                  <tr>
                    <td>
                      <strong>Execution Engine</strong>
                    </td>
                    {run.summaries.map((s) => (
                      <td key={s.system_name} style={{ fontSize: '11px', color: '#666' }}>
                        {s.description}
                      </td>
                    ))}
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className="empty-card">
            <h3>No evaluation runs to compare yet</h3>
            <p>
              Launch an evaluation with the &quot;Include illustrative reference baselines&quot; option checked to
              generate a head-to-head comparison table.
            </p>
            <div style={{ marginTop: '16px' }}>
              <Button onClick={() => navigate('Experiments')}>
                <Play data-icon="inline-start" /> Run Benchmark Evaluation
              </Button>
            </div>
          </div>
        )}
      </div>
    )
  }

  function renderSettings() {
    const handleSave = () => {
      setApiBaseUrl(endpoint)
      setStatusMsg('API endpoint saved to local storage.')
    }

    const handleTestConnection = async () => {
      setTesting(true)
      setStatusMsg(null)
      try {
        setApiBaseUrl(endpoint)
        const h = await fetchHealth()
        setHealth(h)
        setStatusMsg(`Connected successfully! ${h.service} v${h.version} (${h.corpus_doc_count} docs indexed).`)
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Connection failed'
        setStatusMsg(`Connection test failed: ${msg}`)
      } finally {
        setTesting(false)
      }
    }

    return (
      <div className="page-content">
        <header className="page-header">
          <div>
            <div className="breadcrumb">
              Workspace <ChevronRight /> Settings
            </div>
            <h1>Backend Settings.</h1>
            <p>Configure and verify connectivity to the Python FastAPI evaluation service.</p>
          </div>
        </header>

        <div className="settings-card">
          <div className="settings-heading">
            <div>
              <div className="eyebrow">Backend connection</div>
              <h2>API Base URL</h2>
              <p>RAGBench expects a running FastAPI service with CORS enabled for this frontend origin.</p>
            </div>
            <span
              className={`connection-status ${health?.status === 'ok' ? 'status-connected' : 'status-disconnected'}`}
            >
              <span className="status-dot" />
              {health?.status === 'ok' ? 'Connected' : 'Not connected'}
            </span>
          </div>

          <label>
            Base URL
            <input value={endpoint} onChange={(e) => setEndpoint(e.target.value)} />
          </label>

          <div className="settings-actions">
            <Button onClick={handleTestConnection} disabled={testing}>
              {testing ? 'Testing...' : 'Test Connection'}
            </Button>
            <button className="text-button" onClick={handleSave}>
              Save endpoint
            </button>
            <span className="mono">Default: http://127.0.0.1:8000</span>
          </div>

          {statusMsg && (
            <div style={{ marginTop: '14px', fontSize: '11px', color: '#2457d6', font: '11px var(--font-mono)' }}>
              {statusMsg}
            </div>
          )}
        </div>

        <div className="settings-note">
          <FileCheck2 />
          <div>
            <strong>How to start the backend on Windows PowerShell</strong>
            <p style={{ marginTop: '6px' }}>
              Run the following command from the repository root:
              <br />
              <code>python -m uvicorn ragbench.api.app:app --reload --host 127.0.0.1 --port 8000</code>
            </p>
          </div>
        </div>
      </div>
    )
  }

  // Router switch
  const renderCurrentPage = () => {
    if (page === 'detail:latest' || page.startsWith('detail:')) {
      return renderDetail(activeRun || latestLiveRun)
    }
    if (page.startsWith('dataset:')) {
      return renderDatasetDetail(page.split(':')[1])
    }
    switch (page) {
      case 'Overview':
        return renderOverview()
      case 'Datasets':
        return renderDatasets()
      case 'Experiments':
        return renderExperiments()
      case 'Compare':
        return renderCompare()
      case 'Settings':
        return renderSettings()
      default:
        return renderOverview()
    }
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileOpen ? 'open' : ''}`}>
        <div className="brand">
          <span className="brand-mark">
            <span />
            <span />
            <span />
          </span>
          <span>RAGBench</span>
          <span className="brand-version">v0.1</span>
        </div>

        <div className="workspace-switcher">
          <span className="workspace-avatar">R</span>
          <div>
            <strong>Evaluation Lab</strong>
            <small>BM25 Milestone</small>
          </div>
          <MoreHorizontal />
        </div>

        <nav>
          <div className="nav-label">Workspace</div>
          {navItems.map(({ label, icon: Icon }) => (
            <button
              key={label}
              className={page === label ? 'active' : ''}
              onClick={() => navigate(label)}
            >
              <Icon />
              {label}
              {label === 'Compare' && <span className="nav-index">⌘2</span>}
            </button>
          ))}
          <div className="nav-divider" />
          <button
            className={page === 'Settings' ? 'active' : ''}
            onClick={() => navigate('Settings')}
          >
            <Settings2 />
            Settings
          </button>
        </nav>

        <div className="sidebar-footer">
          <div
            className="status-dot"
            style={{
              background: health?.status === 'ok' ? '#34a853' : '#d39834',
            }}
          />
          <span>{health?.status === 'ok' ? 'FastAPI Live' : 'Offline'}</span>
          <span className="mono">LOCAL</span>
        </div>
      </aside>

      <main className="main-area">
        <button
          className="mobile-menu"
          onClick={() => setMobileOpen(!mobileOpen)}
          aria-label="Toggle navigation"
        >
          <Menu />
        </button>
        {renderCurrentPage()}
        <footer className="app-footer">
          <span>RAGBench / information retrieval evaluation workbench</span>
          <span className="mono">build 0.1.0-fastapi-nextjs</span>
        </footer>
      </main>
    </div>
  )
}
