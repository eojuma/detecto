import { useMemo, useState } from 'react'

// HistoryTable renders fetched records and sorts them client-side.
//
// Sorting vs filtering: filtering is a server concern (it changes which rows
// exist and drives the chart too), while sorting is pure presentation over
// rows we already hold -- cheap and instant in memory.

const COLUMNS = [
  { key: 'timestamp', label: 'Timestamp' },
  { key: 'person_count', label: 'People' },
  { key: 'avg_confidence', label: 'Avg confidence' },
  { key: 'inference_time_ms', label: 'Inference' },
]

function formatTimestamp(value) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString()
}

function SortHeader({ label, sortKey, sort, onSort }) {
  const active = sort.key === sortKey
  return (
    <th scope="col">
      <button
        type="button"
        className={`sort-btn${active ? ' active' : ''}`}
        onClick={() => onSort(sortKey)}
        aria-label={`Sort by ${label}`}
      >
        {label}
        <span className="sort-arrow" aria-hidden="true">
          {active ? (sort.dir === 'asc' ? '▲' : '▼') : '↕'}
        </span>
      </button>
    </th>
  )
}

/**
 * @param {{ records: Array<object> }} props
 */
export default function HistoryTable({ records = [] }) {
  const [sort, setSort] = useState({ key: 'timestamp', dir: 'desc' })

  const sorted = useMemo(() => {
    const copy = [...records]
    copy.sort((a, b) => {
      let av = a[sort.key]
      let bv = b[sort.key]

      if (sort.key === 'timestamp') {
        av = new Date(av).getTime()
        bv = new Date(bv).getTime()
      }
      // NULL avg_confidence (zero-person rows) sorts last in either direction
      // rather than coercing to 0 and masquerading as the worst confidence.
      if (av === null || av === undefined) av = -Infinity
      if (bv === null || bv === undefined) bv = -Infinity

      if (av < bv) return sort.dir === 'asc' ? -1 : 1
      if (av > bv) return sort.dir === 'asc' ? 1 : -1
      return 0
    })
    return copy
  }, [records, sort])

  function toggleSort(key) {
    setSort((current) =>
      current.key === key
        ? { key, dir: current.dir === 'asc' ? 'desc' : 'asc' }
        : { key, dir: 'desc' }
    )
  }

  if (records.length === 0) {
    return <div className="empty-state">No detections match these filters.</div>
  }

  return (
    <div className="table-wrap">
      <table className="history-table">
        <thead>
          <tr>
            {COLUMNS.map((column) => (
              <SortHeader
                key={column.key}
                label={column.label}
                sortKey={column.key}
                sort={sort}
                onSort={toggleSort}
              />
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((record) => (
            <tr key={record.id}>
              <td title={record.timestamp}>{formatTimestamp(record.timestamp)}</td>
              <td className="mono">{record.person_count}</td>
              <td className="mono">
                {record.avg_confidence === null || record.avg_confidence === undefined
                  ? '—'
                  : record.avg_confidence.toFixed(3)}
              </td>
              <td className="mono">{record.inference_time_ms.toFixed(0)} ms</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
