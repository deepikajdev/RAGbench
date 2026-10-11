import type {
  DatasetListResponse,
  EvaluationRequest,
  EvaluationResponse,
  ExperimentItem,
  HealthResponse,
} from './types'

export function getApiBaseUrl(): string {
  // Check if user set a custom URL in localStorage (client-side only)
  if (typeof window !== 'undefined') {
    const custom = localStorage.getItem('ragbench_custom_api_url')
    if (custom && custom.trim()) {
      return custom.trim().replace(/\/$/, '')
    }
  }
  return (process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')
}

export function setApiBaseUrl(url: string): void {
  if (typeof window !== 'undefined') {
    if (url && url.trim()) {
      localStorage.setItem('ragbench_custom_api_url', url.trim())
    } else {
      localStorage.removeItem('ragbench_custom_api_url')
    }
  }
}

export async function apiCall<T>(
  endpoint: string,
  options?: RequestInit,
): Promise<T> {
  const baseUrl = getApiBaseUrl()
  const url = `${baseUrl}${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`

  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    })

    if (!response.ok) {
      let errorMessage = `API error ${response.status}: ${response.statusText}`
      try {
        const errorBody = await response.json()
        if (errorBody && errorBody.detail) {
          if (typeof errorBody.detail === 'string') {
            errorMessage = errorBody.detail
          } else if (Array.isArray(errorBody.detail)) {
            errorMessage = errorBody.detail.map((e: { msg?: string }) => e.msg || JSON.stringify(e)).join(', ')
          }
        }
      } catch {
        // Fallback to HTTP status text
      }
      throw new Error(errorMessage)
    }

    return await response.json()
  } catch (error: unknown) {
    if (error instanceof Error) {
      throw error
    }
    throw new Error('Unknown network or server error occurred')
  }
}

export async function fetchHealth(): Promise<HealthResponse> {
  return apiCall<HealthResponse>('/health')
}

export async function fetchDatasets(): Promise<DatasetListResponse> {
  return apiCall<DatasetListResponse>('/api/datasets')
}

export async function fetchExperiments(): Promise<ExperimentItem[]> {
  return apiCall<ExperimentItem[]>('/api/experiments')
}

export async function executeEvaluation(
  request: EvaluationRequest,
): Promise<EvaluationResponse> {
  return apiCall<EvaluationResponse>('/api/evaluate', {
    method: 'POST',
    body: JSON.stringify(request),
  })
}
