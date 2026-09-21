// detecto API client
// ---------------------------------------------------------------------------
// The single place that knows the backend URL and the error envelope shape.
// Components call these functions and catch ApiError; they never touch fetch.
//
// Base URL is configurable at build time via VITE_API_BASE_URL (see Vite docs).
// Default targets the local FastAPI dev server.

export const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

/** Error carrying the backend's structured {code, message, detail}. */
export class ApiError extends Error {
  constructor(code, message, detail = null) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.detail = detail
  }
}

/**
 * Perform a request and normalise both success and failure.
 *
 * Reads the body as text first: a non-FastAPI failure (proxy error, 502 HTML)
 * would otherwise throw an opaque JSON parse error instead of a clear message.
 */
async function request(url, options = {}) {
  let response
  try {
    response = await fetch(url, options)
  } catch {
    // fetch only rejects on network-level failure.
    throw new ApiError(
      'network_error',
      'Could not reach the server. Is the backend running?',
      `Tried ${url}`
    )
  }

  const text = await response.text()
  let data = null
  if (text) {
    try {
      data = JSON.parse(text)
    } catch {
      data = null
    }
  }

  if (!response.ok) {
    const envelope = data && data.error ? data.error : null
    throw new ApiError(
      envelope?.code || `http_${response.status}`,
      envelope?.message || `Request failed with status ${response.status}`,
      envelope?.detail || (data ? null : text.slice(0, 300))
    )
  }

  return data
}

/**
 * Run person detection on an image file.
 * @param {File} file - JPEG or PNG selected by the user.
 * @param {{annotated?: boolean}} [opts]
 * @returns {Promise<object>} DetectionResponse from the backend.
 */
export function detect(file, { annotated = false } = {}) {
  const form = new FormData()
  form.append('file', file)
  return request(`${API_BASE}/detect?annotated=${annotated}`, {
    method: 'POST',
    body: form,
  })
}

/**
 * Fetch stored detections.
 * Empty/undefined values are omitted so the backend applies its defaults.
 * @param {{start?, end?, time_start?, time_end?, min_confidence?, limit?, offset?}} filters
 */
export function getHistory(filters = {}) {
  const params = new URLSearchParams()
  Object.entries(filters).forEach(([key, value]) => {
    if (value !== '' && value !== null && value !== undefined) {
      params.append(key, value)
    }
  })
  const query = params.toString()
  return request(`${API_BASE}/history${query ? `?${query}` : ''}`)
}

/** Delete all stored detections. Destructive. */
export function resetHistory() {
  return request(`${API_BASE}/reset`, { method: 'DELETE' })
}

/** Liveness probe; also reports whether the model is loaded. */
export function getHealth() {
  return request(`${API_BASE}/health`)
}
