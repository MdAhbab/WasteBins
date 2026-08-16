/**
 * Operations dashboard.
 *
 * Answers the supervisor's first question — what needs attention right now —
 * and shows the reasoning behind the answer rather than only the ranking. Each
 * priority carries the confidence behind it, so a score built on distrusted
 * sensors is visibly different from one built on healthy ones.
 *
 * Deliberately cheap: scoring runs on every load, but planning does not. The
 * solver lives behind an explicit action on the Fleet screen.
 */
import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Activity, AlertTriangle, CheckCircle, Cpu, Navigation, RefreshCw,
  TrendingUp, WifiOff,
} from 'lucide-react'
import { fetchDashboard, fetchLatestPlan, fetchSettings } from '../api/endpoints'
import { toast } from '../components/Toast'
import { IS_DEMO, DEMO_DASHBOARD } from '../demo'
import BinMap from '../components/BinMap'
import RingGauge from '../components/RingGauge'
import {
  Badge, Bar, Card, EmptyState, Grid, Loading, Metric, Note, Table,
  fillTone, fmt, trustTone,
} from '../components/ui'

export default function Dashboard() {
  const [data, setData] = useState(null)
  const [routes, setRoutes] = useState([])
  const [loading, setLoading] = useState(true)
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [userLocation, setUserLocation] = useState({ lat: null, lng: null })
  const intervalRef = useRef(null)

  useEffect(() => {
    if (IS_DEMO) {
      setUserLocation({ lat: 23.8069, lng: 90.3687 })
      return
    }
    fetchSettings()
      .then((res) => {
        if (res.data.latitude != null && res.data.longitude != null) {
          setUserLocation({ lat: res.data.latitude, lng: res.data.longitude })
        }
      })
      .catch(() => {})
  }, [])

  const load = async () => {
    if (IS_DEMO) {
      setData(DEMO_DASHBOARD)
      setLoading(false)
      return
    }
    try {
      const res = await fetchDashboard()
      setData(res.data)
      // The stored plan is fetched separately so a slow plan lookup cannot
      // delay the readings the operator is actually watching.
      fetchLatestPlan()
        .then((planRes) => {
          const stops = planRes.data?.plan?.stops || []
          const byVehicle = new Map()
          for (const stop of stops) {
            if (!byVehicle.has(stop.vehicle)) byVehicle.set(stop.vehicle, [])
            byVehicle.get(stop.vehicle).push(stop)
          }
          setRoutes([...byVehicle.entries()].map(([vehicleId, list]) => ({
            vehicle_id: vehicleId,
            stops: list.sort((a, b) => a.sequence - b.sequence)
              .map((s) => ({ node_id: s.node })),
          })))
        })
        .catch(() => setRoutes([]))
    } catch {
      toast('Failed to load dashboard data', 'error')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  useEffect(() => {
    if (autoRefresh) {
      intervalRef.current = setInterval(load, 15000)
    } else {
      clearInterval(intervalRef.current)
    }
    return () => clearInterval(intervalRef.current)
  }, [autoRefresh])

  const stats = data?.stats ?? {}
  const priority = data?.priority_info
  const health = data?.health_summary

  return (
    <div>
      {/* ── Toolbar ── */}
      <div className="section-header" style={{ marginBottom: 20 }}>
        <div className="text-muted text-sm">
          {new Date().toLocaleDateString('en-US', {
            weekday: 'long', year: 'numeric', month: 'long', day: 'numeric',
          })}
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13,
                          color: 'var(--text-secondary)', cursor: 'pointer' }}>
            <div className="toggle" style={{ position: 'relative', display: 'inline-block' }}>
              <input id="auto-refresh-toggle" type="checkbox" checked={autoRefresh}
                     onChange={(e) => setAutoRefresh(e.target.checked)} />
              <span className="toggle-track" />
              <span className="toggle-thumb" />
            </div>
            Auto-refresh (15s)
          </label>
          <button id="dashboard-refresh-btn" className="btn btn-ghost btn-sm" onClick={load}>
            <RefreshCw size={14} className={loading ? 'ui-spin' : ''} />
            Refresh
          </button>
        </div>
      </div>

      {/* ── Stats ── */}
      <Grid min={165} style={{ marginBottom: 20 }}>
        <Metric label="Total bins" value={loading ? '—' : stats.total_bins ?? 0} icon={Activity} />
        <Metric label="Critical" value={loading ? '—' : stats.critical_bins ?? 0}
                hint="≥85% full" tone="bad" icon={AlertTriangle} />
        <Metric label="Warning" value={loading ? '—' : stats.warning_bins ?? 0}
                hint="65–84%" tone="warn" icon={TrendingUp} />
        <Metric label="Normal" value={loading ? '—' : stats.normal_bins ?? 0}
                hint="<65% full" tone="good" icon={CheckCircle} />
        <Metric label="Average fill" value={loading ? '—' : fmt.pct(stats.avg_fill_pct, 0)}
                icon={Cpu} />
        {stats.offline_bins > 0 && (
          <Metric label="No reading" value={stats.offline_bins} tone="warn"
                  hint="excluded from the average" icon={WifiOff} />
        )}
      </Grid>

      {/* ── Model / health banner ── */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 22 }}>
        {data?.model_version && (
          <div className="banner">
            <Cpu size={15} color="var(--accent)" />
            <span style={{ fontWeight: 600, color: 'var(--accent)' }}>{data.model_version}</span>
            <span style={{ color: 'var(--text-secondary)' }}>
              {data.model_available
                ? '· forward overflow + hazard model active'
                : '· not trained yet — priorities use the rule only'}
            </span>
          </div>
        )}
        {health && (
          <Link to="/app/sensors" className="banner" style={{ textDecoration: 'none' }}>
            <Activity size={15} color={health.channels_degraded > 0 ? 'var(--amber)' : 'var(--emerald)'} />
            <span style={{ fontWeight: 600,
                           color: health.channels_degraded > 0 ? 'var(--amber)' : 'var(--emerald)' }}>
              {health.channels_degraded} channel{health.channels_degraded === 1 ? '' : 's'} degraded
            </span>
            <span style={{ color: 'var(--text-secondary)' }}>
              · mean trust {fmt.num(health.mean_trust, 2)}
            </span>
          </Link>
        )}
      </div>

      {/* ── Map + priorities ── */}
      <Grid min={360} style={{ marginBottom: 24 }}>
        <Card title="Network" subtitle="Marker size and colour follow fill level; each planned
                                       route is drawn in its vehicle's colour.">
          <BinMap readings={data?.readings ?? []} routes={routes}
                  userLat={userLocation.lat} userLng={userLocation.lng} />
        </Card>

        <Card
          title="Collection priority"
          subtitle="Hazardous bins occupy a strictly higher tier, so they always rank first
                    regardless of the equity weighting."
          actions={<Link to="/app/fleet" className="btn btn-primary btn-sm">
            <Navigation size={13} /> Dispatch
          </Link>}
        >
          {loading ? (
            <Loading />
          ) : !priority?.top_nodes?.length ? (
            <EmptyState message="No priorities computed yet." />
          ) : (
            <>
              <Table
                dense
                columns={[
                  { key: 'name', header: 'Bin',
                    render: (r) => (
                      <span>
                        <strong>{r.name}</strong>
                        {r.tier === 0 && <Badge tone="bad" >hazard</Badge>}
                      </span>
                    ) },
                  { key: 'score', header: 'Urgency', width: 120,
                    render: (r) => (
                      <Bar value={r.effective_score ?? r.score}
                           tone={r.tier === 0 ? 'bad' : 'accent'}
                           label={fmt.num(r.effective_score ?? r.score, 2)} />
                    ) },
                  { key: 'predicted_overflow_h', header: 'Overflow in', align: 'right',
                    render: (r) => (
                      <span className="ui-num">
                        {r.predicted_overflow_h == null ? '—' : fmt.hours(r.predicted_overflow_h)}
                      </span>
                    ) },
                  { key: 'confidence', header: 'Confidence', align: 'right',
                    render: (r) => (
                      <Badge tone={trustTone(r.confidence)}>{fmt.frac(r.confidence, 0)}</Badge>
                    ) },
                ]}
                rows={priority.top_nodes.map((n) => ({ ...n, __key: n.id }))}
              />
              {priority.low_confidence_bins?.length > 0 && (
                <Note tone="warn">
                  {priority.low_confidence_bins.length} bin
                  {priority.low_confidence_bins.length === 1 ? '' : 's'} scored on too little
                  trustworthy sensor evidence to dispatch on. The right response is a technician,
                  not a truck.
                </Note>
              )}
            </>
          )}
        </Card>
      </Grid>

      {/* ── Sensor cards ── */}
      <div>
        <div className="section-header" style={{ marginBottom: 14 }}>
          <h2 className="section-title">Live sensor readings</h2>
          <span className="text-muted text-xs">{data?.readings?.length ?? 0} reporting bins</span>
        </div>

        {loading ? (
          <div className="sensor-grid">
            {[1, 2, 3, 4, 5, 6].map((i) => (
              <div key={i} className="skeleton" style={{ height: 190, borderRadius: 14 }} />
            ))}
          </div>
        ) : !data?.readings?.length ? (
          <Card>
            <EmptyState message="No sensor data yet. Run `python manage.py seed_demo` to populate
                                 a demonstration network." />
          </Card>
        ) : (
          <div className="sensor-grid">
            {data.readings.map((r) => <SensorCard key={r.id} reading={r} />)}
          </div>
        )}
      </div>
    </div>
  )
}

function SensorCard({ reading }) {
  const fill = reading.waste_level
  const missing = fill === null || fill === undefined
  const pct = missing ? 0 : Math.round(fill * 100)
  const status = missing ? 'offline'
    : fill >= 0.85 ? 'critical' : fill >= 0.65 ? 'warning' : 'normal'

  return (
    <div className={`sensor-card ${status === 'offline' ? '' : status}`}>
      <div className="sensor-card-header">
        <span className="sensor-name">{reading.node?.name ?? 'Unknown'}</span>
        <Badge tone={missing ? 'muted' : fillTone(fill)}>
          {missing ? 'no reading' : status}
        </Badge>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        <RingGauge value={missing ? 0 : fill} size={76} strokeWidth={7} />
        <div style={{ flex: 1 }}>
          <div className="fill-bar">
            <div className="fill-bar-inner" style={{
              width: `${pct}%`,
              background: missing ? 'var(--text-muted)'
                : fill >= 0.85 ? 'var(--red)' : fill >= 0.65 ? 'var(--amber)' : 'var(--emerald)',
            }} />
          </div>
          <div className="sensor-metrics">
            <Reading label="Temperature" value={reading.temperature} unit="°C" digits={1} />
            <Reading label="Humidity" value={reading.humidity} unit="%" digits={1} />
            <Reading label="Gas level" value={reading.gas_level} scale={100} unit="%" digits={0} />
            <div className="metric">
              <span className="metric-label">Updated</span>
              <span className="metric-value" style={{ fontSize: 11 }}>
                {fmt.time(reading.timestamp)}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

/**
 * A single reading.  A missing channel renders as "—", never as 0: the whole
 * point of the sensing work is that absent and empty are different states.
 */
function Reading({ label, value, unit = '', digits = 1, scale = 1 }) {
  const absent = value === null || value === undefined
  return (
    <div className="metric">
      <span className="metric-label">{label}</span>
      <span className="metric-value" style={{ color: absent ? 'var(--text-muted)' : undefined }}>
        {absent ? '—' : `${(value * scale).toFixed(digits)}${unit}`}
      </span>
    </div>
  )
}
