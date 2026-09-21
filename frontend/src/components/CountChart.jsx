import { useMemo } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

// CountChart plots person count over time.
//
// Why a numeric time axis (scale="time") rather than a category axis?
//   Detections are discrete events, not evenly-spaced samples. A time scale
//   positions each point by real elapsed time, so genuine gaps look like gaps
//   instead of being compressed to uniform spacing.

const DAY_MS = 24 * 60 * 60 * 1000

function formatTick(ms, showDate) {
  const date = new Date(ms)
  if (showDate) {
    return date.toLocaleDateString([], { month: 'short', day: 'numeric' })
  }
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function ChartTooltip({ active, payload }) {
  if (!active || !payload || payload.length === 0) return null
  const point = payload[0].payload
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-time">{new Date(point.t).toLocaleString()}</div>
      <div>
        <strong>{point.count}</strong> {point.count === 1 ? 'person' : 'people'}
      </div>
      <div className="chart-tooltip-muted">
        avg confidence:{' '}
        {point.confidence === null || point.confidence === undefined
          ? '—'
          : point.confidence.toFixed(3)}
      </div>
    </div>
  )
}

/**
 * @param {{ records: Array<object> }} props
 */
export default function CountChart({ records = [] }) {
  const data = useMemo(
    () =>
      records
        .map((record) => ({
          t: new Date(record.timestamp).getTime(),
          count: record.person_count,
          confidence: record.avg_confidence,
        }))
        .sort((a, b) => a.t - b.t), // ascending time for a left-to-right series
    [records]
  )

  if (data.length === 0) {
    return <div className="empty-state">No data to chart yet.</div>
  }

  const spanMs = data[data.length - 1].t - data[0].t
  const showDate = spanMs > DAY_MS

  return (
    <div className="card chart-card">
      <h2 className="panel-title">People over time</h2>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: -16 }}>
          <CartesianGrid stroke="#2e3846" strokeDasharray="3 3" />
          <XAxis
            dataKey="t"
            type="number"
            scale="time"
            domain={['dataMin', 'dataMax']}
            tickFormatter={(ms) => formatTick(ms, showDate)}
            stroke="#9aa7b4"
            fontSize={12}
          />
          <YAxis
            allowDecimals={false}
            stroke="#9aa7b4"
            fontSize={12}
            width={44}
            label={{
              value: 'people',
              angle: -90,
              position: 'insideLeft',
              fill: '#9aa7b4',
              fontSize: 12,
            }}
          />
          <Tooltip content={<ChartTooltip />} />
          <Line
            type="monotone"
            dataKey="count"
            stroke="#3fb950"
            strokeWidth={2}
            dot={{ r: 3, fill: '#3fb950' }}
            activeDot={{ r: 5 }}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
