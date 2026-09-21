// DetectionPanel presents the result of a detection, or the state leading up
// to it. It handles all four states so DetectionPage stays a thin orchestrator:
//   idle    -> no result yet
//   loading -> inference in flight
//   error   -> ApiError from the client (rendered as UI, never console-only)
//   result  -> counts, confidence, timings (including the zero-person case)

function Stat({ label, value, hint }) {
  return (
    <div className="stat">
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
      {hint && <div className="stat-hint">{hint}</div>}
    </div>
  )
}

export default function DetectionPanel({ result, loading = false, error = null }) {
  if (loading) {
    return (
      <div className="card panel">
        <div className="panel-loading">
          <span className="spinner" aria-hidden="true" />
          Running detection…
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="card panel">
        <div className="error-banner" role="alert">
          <div>
            <strong>{error.code || 'error'}</strong>
            {error.message ? `: ${error.message}` : ''}
          </div>
          {error.detail && <div className="error-detail mono">{error.detail}</div>}
        </div>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="card panel">
        <div className="empty-state">No detection yet. Upload an image to begin.</div>
      </div>
    )
  }

  const hasPeople = result.person_count > 0
  // avg_confidence is null (not 0) when nobody is detected -- show a dash.
  const avgConfidence =
    result.avg_confidence === null || result.avg_confidence === undefined
      ? '—'
      : result.avg_confidence.toFixed(3)

  return (
    <div className="card panel">
      <h2 className="panel-title">Result</h2>

      <div className="stats-grid">
        <Stat label="People detected" value={result.person_count} />
        <Stat label="Avg confidence" value={avgConfidence} hint="Mean of returned boxes" />
        <Stat
          label="Inference"
          value={`${result.inference_time_ms.toFixed(0)} ms`}
          hint="Model only"
        />
        <Stat
          label="Total"
          value={`${result.total_time_ms.toFixed(0)} ms`}
          hint="Decode → response"
        />
      </div>

      <div className="panel-meta">
        Image {result.image_width} × {result.image_height} px ·{' '}
        {result.boxes.length} bounding box{result.boxes.length === 1 ? '' : 'es'}
      </div>

      {/* Zero-person results are a valid outcome, not an error. */}
      {!hasPeople && (
        <div className="zero-state">No people detected in this image.</div>
      )}
    </div>
  )
}
