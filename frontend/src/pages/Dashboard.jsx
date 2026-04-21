import { useEffect, useState, useRef } from 'react'
import { RefreshCw, Cpu, TrendingUp, AlertTriangle, CheckCircle, Activity, Navigation } from 'lucide-react'
import { fetchDashboard, fetchSettings } from '../api/endpoints'
import { toast } from '../components/Toast'
import RingGauge from '../components/RingGauge'
import RoutePanel from '../components/RoutePanel'

// ── Stat Card ──────────────────────────────────────────────────────────────
function StatCard({ label, value, sub, icon: Icon, color }) {
  return (
    <div className="stat-card">
      <div className="stat-icon" style={{ background: `${color}20` }}>
        <Icon size={18} color={color} />
      </div>
      <div className="stat-label">{label}</div>
      <div className="stat-value" style={{ color }}>{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  )
}

// ── Sensor Card ────────────────────────────────────────────────────────────
function SensorCard({ reading }) {
  const wl = parseFloat(reading.waste_level ?? 0)
  const pct = Math.round(wl * 100)
  const status = wl >= 0.85 ? 'critical' : wl >= 0.65 ? 'warning' : 'normal'
  const badgeLabel = status.charAt(0).toUpperCase() + status.slice(1)
  const badgeClass = `level-badge badge-${status}`

  return (
    <div className={`sensor-card ${status}`}>
      <div className="sensor-card-header">
        <span className="sensor-name">{reading.node?.name ?? 'Unknown'}</span>
        <span className={badgeClass}>{badgeLabel}</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        <RingGauge value={wl} size={76} strokeWidth={7} />
        <div style={{ flex: 1 }}>
          <div className="fill-bar">
            <div className="fill-bar-inner" style={{
              width: `${pct}%`,
              background: wl >= 0.85 ? 'var(--red)' : wl >= 0.65 ? 'var(--amber)' : 'var(--emerald)',
            }} />
          </div>
          <div className="sensor-metrics">
            <div className="metric">
              <span className="metric-label">Temperature</span>
              <span className="metric-value">{reading.temperature?.toFixed(1)}°C</span>
            </div>
            <div className="metric">
              <span className="metric-label">Humidity</span>
              <span className="metric-value">{reading.humidity?.toFixed(1)}%</span>
            </div>
            <div className="metric">
              <span className="metric-label">Gas Level</span>
              <span className="metric-value">{((reading.gas_level ?? 0) * 100).toFixed(0)}%</span>
            </div>
            <div className="metric">
              <span className="metric-label">Updated</span>
              <span className="metric-value" style={{ fontSize: 11 }}>
                {new Date(reading.timestamp).toLocaleTimeString()}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Dashboard ──────────────────────────────────────────────────────────────
export default function Dashboard() {
  const [data, setData] = useState(null)
  const [route, setRoute] = useState(null)
  const [loading, setLoading] = useState(true)
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [userLocation, setUserLocation] = useState({ lat: null, lng: null })
  const intervalRef = useRef(null)

  // Load settings to get user location for route computation
  useEffect(() => {
    fetchSettings()
      .then(res => {
        if (res.data.latitude && res.data.longitude) {
          setUserLocation({ lat: res.data.latitude, lng: res.data.longitude })
        }
      })
      .catch(() => {})
  }, [])

  const load = async () => {
    try {
      const res = await fetchDashboard()
      setData(res.data)
      if (res.data.latest_route) setRoute(res.data.latest_route)
    } catch {
      toast('Failed to load dashboard data', 'error')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  useEffect(() => {
    if (autoRefresh) {
      intervalRef.current = setInterval(load, 10000)
    } else {
      clearInterval(intervalRef.current)
    }
    return () => clearInterval(intervalRef.current)
  }, [autoRefresh])

  const stats = data?.stats ?? {}

  return (
    <div>
      {/* ── Toolbar ── */}
      <div className="section-header mb-6" style={{ marginBottom: 20 }}>
        <div className="text-muted text-sm">
          {new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, color: 'var(--text-secondary)', cursor: 'pointer' }}>
            <div className="toggle" style={{ position: 'relative', display: 'inline-block' }}>
              <input id="auto-refresh-toggle" type="checkbox" checked={autoRefresh}
                onChange={e => setAutoRefresh(e.target.checked)} />
              <span className="toggle-track" />
              <span className="toggle-thumb" />
            </div>
            Auto-refresh (10s)
          </label>
          <button id="dashboard-refresh-btn" className="btn btn-ghost btn-sm" onClick={load}>
            <RefreshCw size={14} className={loading ? 'spin' : ''} />
            Refresh
          </button>
        </div>
      </div>

      {/* ── Stat Grid ── */}
      <div className="stat-grid" style={{ marginBottom: 24 }}>
        <StatCard label="Total Bins" value={loading ? '—' : stats.total_bins ?? 0} icon={Activity} color="var(--accent)" />
        <StatCard label="Critical" value={loading ? '—' : stats.critical_bins ?? 0} sub="≥85% full" icon={AlertTriangle} color="var(--red)" />
        <StatCard label="Warning" value={loading ? '—' : stats.warning_bins ?? 0} sub="65–84%" icon={TrendingUp} color="var(--amber)" />
        <StatCard label="Normal" value={loading ? '—' : stats.normal_bins ?? 0} sub="<65% full" icon={CheckCircle} color="var(--emerald)" />
        <StatCard label="Avg Fill" value={loading ? '—' : `${stats.avg_fill_pct ?? 0}%`} icon={Cpu} color="var(--accent)" />
        {data?.priority_info && (
          <StatCard label="Priority Bins" value={data.priority_info.top_nodes?.length ?? 0} sub="selected by AI" icon={Navigation} color="var(--emerald)" />
        )}
      </div>

      {/* ── AI model banner ── */}
      {data?.model_version && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10, padding: '10px 16px',
          background: 'var(--accent-dim)', border: '1px solid var(--accent)',
          borderRadius: 10, marginBottom: 24, fontSize: 13,
        }}>
          <Cpu size={15} color="var(--accent)" />
          <span style={{ fontWeight: 600, color: 'var(--accent)' }}>{data.model_version}</span>
          <span style={{ color: 'var(--text-secondary)' }}>· Random Forest routing model active</span>
        </div>
      )}

      {/* ── Route Visualizer (full width) ── */}
      <div style={{ marginBottom: 28 }}>
        <RoutePanel
          readings={data?.readings ?? []}
          route={route}
          userLat={userLocation.lat}
          userLng={userLocation.lng}
          onRouteComputed={(r) => {
            setRoute(r.route ? { route_data: r.route, total_cost: r.total_cost, timestamp: new Date().toISOString() } : null)
            toast('Route updated', 'success')
            load()
          }}
        />
      </div>

      {/* ── Sensor Cards ── */}
      <div>
        <div className="section-header" style={{ marginBottom: 14 }}>
          <h2 className="section-title">Live Sensor Readings</h2>
          <span className="text-muted text-xs">{data?.readings?.length ?? 0} active nodes</span>
        </div>

        {loading ? (
          <div className="sensor-grid">
            {[1, 2, 3, 4, 5, 6].map(i => (
              <div key={i} className="skeleton" style={{ height: 190, borderRadius: 14 }} />
            ))}
          </div>
        ) : !data?.readings?.length ? (
          <div className="card" style={{ textAlign: 'center', padding: 48, color: 'var(--text-muted)' }}>
            No sensor data yet. Run <code style={{ color: 'var(--accent)' }}>python send_dummy_data.py</code> to populate.
          </div>
        ) : (
          <div className="sensor-grid">
            {data.readings.map(r => <SensorCard key={r.id} reading={r} />)}
          </div>
        )}
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        .spin { animation: spin 1s linear infinite; }
      `}</style>
    </div>
  )
}
