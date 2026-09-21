import { useCallback, useEffect, useState } from 'react'
import { getHistory, resetHistory } from '../api.js'
import HistoryFilters from '../components/HistoryFilters.jsx'
import HistoryTable from '../components/HistoryTable.jsx'
import CountChart from '../components/CountChart.jsx'

// HistoryPage owns the APPLIED filters and the fetched result set.
//
// Both the chart and the table render the same `records` array, so they can
// never disagree -- this is what "filters drive the chart and the table
// together" means in practice.
export default function HistoryPage() {
  const [filters, setFilters] = useState({})
  const [records, setRecords] = useState([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const load = useCallback(async (activeFilters) => {
    setLoading(true)
    setError(null)
    try {
      const data = await getHistory(activeFilters)
      setRecords(data.records)
      setTotal(data.total)
    } catch (err) {
      setError(err)
      setRecords([])
      setTotal(0)
    } finally {
      setLoading(false)
    }
  }, [])

  // Re-fetch whenever the applied filters change (including initial mount).
  useEffect(() => {
    load(filters)
  }, [filters, load])

  const handleApply = useCallback((nextFilters) => {
    setFilters(nextFilters)
  }, [])

  const handleClear = useCallback(async () => {
    const confirmed = window.confirm(
      'Delete ALL detection history? This cannot be undone.'
    )
    if (!confirmed) return
    try {
      await resetHistory()
      // Setting a fresh object re-triggers the effect, reloading the view.
      setFilters({})
    } catch (err) {
      setError(err)
    }
  }, [])

  return (
    <div>
      <div className="page-head">
        <div>
          <h1 className="page-title">History View</h1>
          <p className="page-subtitle">
            Past detections over time. Date/time filters are interpreted in UTC.
          </p>
        </div>
        <button
          type="button"
          className="danger"
          onClick={handleClear}
          disabled={loading}
        >
          Clear history
        </button>
      </div>

      <HistoryFilters onApply={handleApply} disabled={loading} />

      {error && (
        <div className="error-banner" role="alert">
          <strong>{error.code || 'error'}</strong>: {error.message}
        </div>
      )}

      {loading && (
        <div className="panel-loading">
          <span className="spinner" aria-hidden="true" />
          Loading history…
        </div>
      )}

      {!loading && !error && (
        <>
          <CountChart records={records} />
          <div className="history-meta">
            {total} record{total === 1 ? '' : 's'} match
            {records.length < total ? ` · showing ${records.length}` : ''}
          </div>
          <HistoryTable records={records} />
        </>
      )}
    </div>
  )
}
