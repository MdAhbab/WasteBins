export default function RingGauge({ value = 0, size = 90, strokeWidth = 8, color }) {
  const pct = Math.max(0, Math.min(1, value))
  const r = (size - strokeWidth) / 2
  const circumference = 2 * Math.PI * r
  const dash = pct * circumference
  const gap = circumference - dash

  const resolvedColor =
    color ||
    (pct >= 0.85 ? 'var(--red)'
      : pct >= 0.65 ? 'var(--amber)'
        : 'var(--emerald)')

  return (
    <div className="ring-container" style={{ width: size, height: size }}>
      <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
        {/* Track */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth={strokeWidth}
        />
        {/* Fill */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={resolvedColor}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={`${dash} ${gap}`}
          style={{ transition: 'stroke-dasharray 0.6s cubic-bezier(0.4,0,0.2,1)' }}
        />
      </svg>
      <div className="ring-label">
        <span style={{ fontSize: size * 0.18, fontWeight: 700, color: resolvedColor }}>
          {Math.round(pct * 100)}%
        </span>
      </div>
    </div>
  )
}
