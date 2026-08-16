/**
 * Shared presentation primitives.
 *
 * Kept in one place so every operator screen renders the same way: a number
 * always carries its unit, a status always maps to the same colour, and an empty
 * or failed state always explains what to do about it rather than showing a
 * blank panel.
 */
import { AlertCircle, Inbox, Loader2 } from 'lucide-react'

// ── Layout ─────────────────────────────────────────────────────────────────
export function Card({ title, subtitle, actions, children, className = '', ...rest }) {
  return (
    <section className={`ui-card ${className}`} {...rest}>
      {(title || actions) && (
        <header className="ui-card-head">
          <div>
            {title && <h3 className="ui-card-title">{title}</h3>}
            {subtitle && <p className="ui-card-sub">{subtitle}</p>}
          </div>
          {actions && <div className="ui-card-actions">{actions}</div>}
        </header>
      )}
      <div className="ui-card-body">{children}</div>
    </section>
  )
}

export function Grid({ min = 260, gap = 16, children, style, ...rest }) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: `repeat(auto-fit, minmax(${min}px, 1fr))`,
        gap,
        ...style,
      }}
      {...rest}
    >
      {children}
    </div>
  )
}

export function PageHeader({ title, description, actions }) {
  return (
    <div className="ui-page-head">
      <div>
        <h1 className="ui-page-title">{title}</h1>
        {description && <p className="ui-page-desc">{description}</p>}
      </div>
      {actions && <div className="ui-page-actions">{actions}</div>}
    </div>
  )
}

// ── Data display ───────────────────────────────────────────────────────────
export function Metric({ label, value, unit, hint, tone = 'default', icon: Icon }) {
  return (
    <div className={`ui-metric tone-${tone}`}>
      <div className="ui-metric-label">
        {Icon && <Icon size={13} />}
        {label}
      </div>
      <div className="ui-metric-value">
        {value}
        {unit && <span className="ui-metric-unit">{unit}</span>}
      </div>
      {hint && <div className="ui-metric-hint">{hint}</div>}
    </div>
  )
}

const STATUS_TONE = {
  ok: 'good', normal: 'good', valid: 'good', true: 'good', active: 'good',
  suspect: 'warn', warning: 'warn', warn: 'warn',
  degraded: 'bad', critical: 'bad', failed: 'bad', false: 'bad', error: 'bad',
}

export function Badge({ children, tone, status }) {
  const resolved = tone || STATUS_TONE[String(status ?? children).toLowerCase()] || 'muted'
  return <span className={`ui-badge tone-${resolved}`}>{children}</span>
}

export function Bar({ value, max = 1, tone = 'accent', height = 6, label }) {
  const pct = Math.max(0, Math.min(100, (value / (max || 1)) * 100))
  return (
    <div className="ui-bar-wrap">
      <div className="ui-bar" style={{ height }}>
        <div className={`ui-bar-fill tone-${tone}`} style={{ width: `${pct}%` }} />
      </div>
      {label && <span className="ui-bar-label">{label}</span>}
    </div>
  )
}

/**
 * Signed contribution bar for attribution displays.
 * Zero sits in the middle so the direction of an effect is readable at a glance.
 */
export function DivergingBar({ value, scale }) {
  const magnitude = Math.min(1, Math.abs(value) / (scale || 1))
  const positive = value >= 0
  return (
    <div className="ui-diverge">
      <div className="ui-diverge-half left">
        {!positive && (
          <div className="ui-diverge-fill neg" style={{ width: `${magnitude * 100}%` }} />
        )}
      </div>
      <div className="ui-diverge-axis" />
      <div className="ui-diverge-half right">
        {positive && (
          <div className="ui-diverge-fill pos" style={{ width: `${magnitude * 100}%` }} />
        )}
      </div>
    </div>
  )
}

export function Table({ columns, rows, empty = 'Nothing to show yet.', dense = false }) {
  if (!rows?.length) return <EmptyState message={empty} />
  return (
    <div className="ui-table-wrap">
      <table className={`ui-table${dense ? ' dense' : ''}`}>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} style={{ textAlign: c.align || 'left', width: c.width }}>
                {c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={row.__key ?? i}>
              {columns.map((c) => (
                <td key={c.key} style={{ textAlign: c.align || 'left' }}>
                  {c.render ? c.render(row, i) : row[c.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function KeyValue({ items }) {
  return (
    <dl className="ui-kv">
      {items.map(({ label, value, hint }) => (
        <div key={label} className="ui-kv-row">
          <dt>
            {label}
            {hint && <span className="ui-kv-hint">{hint}</span>}
          </dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  )
}

// ── States ─────────────────────────────────────────────────────────────────
export function Loading({ label = 'Loading…', inline = false }) {
  return (
    <div className={`ui-state${inline ? ' inline' : ''}`}>
      <Loader2 size={inline ? 14 : 20} className="ui-spin" />
      <span>{label}</span>
    </div>
  )
}

export function EmptyState({ message, action }) {
  return (
    <div className="ui-state">
      <Inbox size={20} />
      <span>{message}</span>
      {action}
    </div>
  )
}

export function ErrorState({ message, action }) {
  return (
    <div className="ui-state error">
      <AlertCircle size={20} />
      <span>{message}</span>
      {action}
    </div>
  )
}

/**
 * Short explanatory note.
 *
 * Used to state a method or a caveat next to the number it applies to, so a
 * reader never has to guess what a figure means or how far to trust it.
 */
export function Note({ children, tone = 'muted' }) {
  // A <div>, not a <p>: notes routinely wrap lists and multi-line blocks, and a
  // block element nested inside a paragraph is invalid HTML that React reports
  // as a hydration error.
  return <div className={`ui-note tone-${tone}`}>{children}</div>
}

// ── Formatting ─────────────────────────────────────────────────────────────
export const fmt = {
  num: (v, digits = 2) =>
    v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(digits),
  int: (v) => (v === null || v === undefined ? '—' : Math.round(Number(v)).toLocaleString()),
  pct: (v, digits = 1) =>
    v === null || v === undefined ? '—' : `${Number(v).toFixed(digits)}%`,
  frac: (v, digits = 0) =>
    v === null || v === undefined ? '—' : `${(Number(v) * 100).toFixed(digits)}%`,
  hours: (v) => {
    if (v === null || v === undefined) return '—'
    const n = Number(v)
    if (!Number.isFinite(n)) return 'unbounded'
    if (n < 1) return `${Math.round(n * 60)} min`
    return `${n.toFixed(1)} h`
  },
  minutes: (v) => {
    if (v === null || v === undefined) return '—'
    const n = Number(v)
    const h = Math.floor(n / 60)
    const m = Math.round(n % 60)
    return h > 0 ? `${h}h ${String(m).padStart(2, '0')}m` : `${m}m`
  },
  clock: (minutes, startMinute = 360) => {
    if (minutes === null || minutes === undefined) return '—'
    const total = (startMinute + Number(minutes)) % 1440
    const h = Math.floor(total / 60)
    const m = Math.round(total % 60)
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`
  },
  time: (iso) => (iso ? new Date(iso).toLocaleTimeString() : '—'),
  datetime: (iso) => (iso ? new Date(iso).toLocaleString() : '—'),
  hash: (h, n = 10) => (h ? `${h.slice(0, n)}…` : '—'),
}

export function fillTone(fill) {
  if (fill === null || fill === undefined) return 'muted'
  if (fill >= 0.85) return 'bad'
  if (fill >= 0.65) return 'warn'
  return 'good'
}

export function trustTone(trust) {
  if (trust >= 0.75) return 'good'
  if (trust >= 0.45) return 'warn'
  return 'bad'
}
