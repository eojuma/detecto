import { useState } from 'react'

// HistoryFilters holds DRAFT filter state locally and only notifies the parent
// on submit. That keeps typing free of network requests -- the parent fetches
// once per Apply, driving both the table and the chart from one result set.

const EMPTY = { start: '', end: '', time_start: '', time_end: '' }

/**
 * @param {{
 *   onApply: (filters: object) => void,
 *   disabled?: boolean,
 * }} props
 */
export default function HistoryFilters({ onApply, disabled = false }) {
  const [draft, setDraft] = useState(EMPTY)
  // The confidence filter is opt-in. It MUST be omitted rather than sent as 0,
  // because the backend uses `avg_confidence >= ?` and NULL (zero-person rows)
  // never satisfies that comparison -- sending 0 would hide those rows.
  const [useConfidence, setUseConfidence] = useState(false)
  const [minConfidence, setMinConfidence] = useState(0.5)
  const [localError, setLocalError] = useState(null)

  function update(field, value) {
    setDraft((current) => ({ ...current, [field]: value }))
  }

  function buildFilters() {
    const filters = {}
    if (draft.start) filters.start = draft.start
    if (draft.end) filters.end = draft.end
    if (draft.time_start) filters.time_start = draft.time_start
    if (draft.time_end) filters.time_end = draft.time_end
    if (useConfidence) filters.min_confidence = minConfidence
    return filters
  }

  function handleSubmit(event) {
    event.preventDefault()
    // Mirror the server's range validation for instant feedback.
    if (draft.start && draft.end && draft.start > draft.end) {
      setLocalError('Start date must not be after end date.')
      return
    }
    if (draft.time_start && draft.time_end && draft.time_start > draft.time_end) {
      setLocalError('Start time must not be after end time.')
      return
    }
    setLocalError(null)
    onApply(buildFilters())
  }

  function handleReset() {
    setDraft(EMPTY)
    setUseConfidence(false)
    setMinConfidence(0.5)
    setLocalError(null)
    onApply({}) // clear all filters
  }

  return (
    <form className="card filters-form" onSubmit={handleSubmit}>
      <div className="filter-grid">
        <label className="filter-field">
          <span>From date</span>
          <input
            type="date"
            value={draft.start}
            onChange={(e) => update('start', e.target.value)}
            disabled={disabled}
          />
        </label>

        <label className="filter-field">
          <span>To date</span>
          <input
            type="date"
            value={draft.end}
            onChange={(e) => update('end', e.target.value)}
            disabled={disabled}
          />
        </label>

        <label className="filter-field">
          <span>From time (UTC)</span>
          <input
            type="time"
            value={draft.time_start}
            onChange={(e) => update('time_start', e.target.value)}
            disabled={disabled}
          />
        </label>

        <label className="filter-field">
          <span>To time (UTC)</span>
          <input
            type="time"
            value={draft.time_end}
            onChange={(e) => update('time_end', e.target.value)}
            disabled={disabled}
          />
        </label>

        <div className="filter-field filter-confidence">
          <label className="filter-checkbox">
            <input
              type="checkbox"
              checked={useConfidence}
              onChange={(e) => setUseConfidence(e.target.checked)}
              disabled={disabled}
            />
            <span>Minimum confidence</span>
          </label>
          <div className="confidence-control">
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={minConfidence}
              onChange={(e) => setMinConfidence(Number(e.target.value))}
              disabled={disabled || !useConfidence}
            />
            <span className="mono confidence-value">{minConfidence.toFixed(2)}</span>
          </div>
        </div>
      </div>

      {localError && (
        <div className="field-error" role="alert">
          {localError}
        </div>
      )}

      <div className="filter-actions">
        <button type="submit" disabled={disabled}>
          Apply filters
        </button>
        <button type="button" onClick={handleReset} disabled={disabled}>
          Reset
        </button>
      </div>
    </form>
  )
}
