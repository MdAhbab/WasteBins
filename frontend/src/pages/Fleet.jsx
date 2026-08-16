/**
 * Fleet dispatch.
 *
 * The screen where a supervisor actually decides what the trucks do today.  It
 * exposes the constraints the planner enforces (capacity, shift, time windows,
 * stream licensing), lets the equity/urgency trade-off be tuned before planning,
 * and shows the resulting routes stop by stop with the arrival clock and the
 * running load — the two things a crew supervisor checks first.
 *
 * Planning is an explicit action, never a side effect of opening the page: the
 * solver is the only expensive operation in the system.
 */
import { useEffect, useMemo, useState } from 'react'
import {
  Truck, Play, GitCompare, Route as RouteIcon, AlertTriangle,
  Leaf, Timer, Package, RefreshCw,
} from 'lucide-react'
import {
  fetchFleetConfig, fetchLatestPlan, generatePlan, comparePlanners,
} from '../api/endpoints'
import {
  Badge, Bar, Card, EmptyState, ErrorState, Grid, KeyValue, Loading,
  Metric, Note, PageHeader, Table, fmt,
} from '../components/ui'
import { toast } from '../components/Toast'

const SOLVER_LABELS = {
  proposed: 'Proposed (regret-2 + local search)',
  genetic: 'Genetic algorithm',
  aco: 'Ant colony optimisation',
  risk_graph: 'Risk-penalised graph',
  risk_graph_ls: 'Risk-penalised graph + local search',
  ortools: 'OR-Tools (guided local search)',
}

export default function Fleet() {
  const [config, setConfig] = useState(null)
  const [plan, setPlan] = useState(null)
  const [loading, setLoading] = useState(true)
  const [planning, setPlanning] = useState(false)
  const [comparing, setComparing] = useState(false)
  const [comparison, setComparison] = useState(null)
  const [error, setError] = useState(null)

  const [algorithm, setAlgorithm] = useState('proposed')
  const [gamma, setGamma] = useState(0.55)
  const [tauH, setTauH] = useState(48)
  const [budget, setBudget] = useState(4)
  const [useTraffic, setUseTraffic] = useState(true)

  useEffect(() => {
    let alive = true
    Promise.all([fetchFleetConfig(), fetchLatestPlan()])
      .then(([cfg, latest]) => {
        if (!alive) return
        setConfig(cfg.data)
        if (latest.data?.plan) setPlan(normaliseStored(latest.data.plan))
      })
      .catch(() => alive && setError('Could not load the fleet configuration.'))
      .finally(() => alive && setLoading(false))
    return () => { alive = false }
  }, [])

  const run = async () => {
    setPlanning(true)
    setComparison(null)
    try {
      const res = await generatePlan({
        algorithm, gamma, tau_h: tauH,
        time_budget_s: budget, use_traffic: useTraffic,
      })
      setPlan(res.data)
      const m = res.data.metrics
      toast(`Plan ready: ${m.bins_served} bins, ${fmt.num(m.distance_km, 1)} km, ` +
            `${fmt.num(m.co2_kg, 1)} kg CO₂`, 'success')
    } catch (e) {
      toast(e?.response?.data?.error || 'Planning failed.', 'error')
    } finally {
      setPlanning(false)
    }
  }

  const compare = async () => {
    setComparing(true)
    try {
      const available = Object.entries(config?.solvers || {})
        .filter(([, ok]) => ok).map(([name]) => name)
      const res = await comparePlanners({ algorithms: available, time_budget_s: 2 })
      setComparison(res.data)
    } catch {
      toast('Comparison failed.', 'error')
    } finally {
      setComparing(false)
    }
  }

  if (loading) return <Loading label="Loading fleet…" />
  if (error) return <ErrorState message={error} />

  const metrics = plan?.metrics || {}
  const solvers = Object.entries(config?.solvers || {})

  return (
    <div>
      <PageHeader
        title="Fleet dispatch"
        description="Plans a prize-collecting capacitated route with time windows: every truck
                     starts and ends at its depot, tips mid-shift when full, respects its shift
                     limit and only collects streams it is licensed for. Bins may be skipped —
                     the objective trades collected urgency against travel cost and CO₂."
        actions={
          <>
            <button className="btn btn-ghost btn-sm" onClick={compare} disabled={comparing || planning}>
              <GitCompare size={14} className={comparing ? 'ui-spin' : ''} />
              {comparing ? 'Comparing…' : 'Compare planners'}
            </button>
            <button className="btn btn-primary btn-sm" onClick={run} disabled={planning}>
              <Play size={14} className={planning ? 'ui-spin' : ''} />
              {planning ? 'Planning…' : 'Generate plan'}
            </button>
          </>
        }
      />

      {/* ── Controls ── */}
      <Card
        title="Planner settings"
        subtitle="γ sets how much accumulated waiting time counts against raw urgency. Hazardous
                  bins sit in a strictly higher tier, so raising γ improves equity without
                  delaying a hazard response."
        style={{ marginBottom: 18 }}
      >
        <Grid min={210}>
          <div className="ui-field">
            <label htmlFor="fleet-algorithm">Solver</label>
            <select id="fleet-algorithm" className="ui-select" value={algorithm}
                    onChange={(e) => setAlgorithm(e.target.value)}>
              {solvers.map(([name, ok]) => (
                <option key={name} value={name} disabled={!ok}>
                  {SOLVER_LABELS[name] || name}{ok ? '' : ' — not installed'}
                </option>
              ))}
            </select>
          </div>
          <div className="ui-field">
            <label htmlFor="fleet-gamma">Equity weight γ = {gamma.toFixed(2)}</label>
            <input id="fleet-gamma" className="ui-range" type="range" min="0" max="0.95" step="0.05"
                   value={gamma} onChange={(e) => setGamma(Number(e.target.value))} />
          </div>
          <div className="ui-field">
            <label htmlFor="fleet-tau">Target max wait τ = {tauH} h</label>
            <input id="fleet-tau" className="ui-range" type="range" min="12" max="120" step="6"
                   value={tauH} onChange={(e) => setTauH(Number(e.target.value))} />
          </div>
          <div className="ui-field">
            <label htmlFor="fleet-budget">Search budget = {budget}s</label>
            <input id="fleet-budget" className="ui-range" type="range" min="1" max="15" step="1"
                   value={budget} onChange={(e) => setBudget(Number(e.target.value))} />
          </div>
          <div className="ui-field">
            <label htmlFor="fleet-traffic">Traffic</label>
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13,
                            textTransform: 'none', letterSpacing: 0, color: 'var(--text-secondary)' }}>
              <input id="fleet-traffic" type="checkbox" checked={useTraffic}
                     onChange={(e) => setUseTraffic(e.target.checked)} />
              Use the congestion surface
            </label>
          </div>
        </Grid>
      </Card>

      {/* ── Plan outcome ── */}
      {!plan ? (
        <Card title="No plan yet">
          <EmptyState message="Generate a plan to see the routes, load profile and emissions." />
        </Card>
      ) : (
        <>
          <Grid min={175} style={{ marginBottom: 18 }}>
            <Metric label="Vehicles used" value={metrics.vehicles_used ?? '—'}
                    hint={`of ${config?.vehicles?.length ?? 0} available`} icon={Truck} />
            <Metric label="Bins served" value={metrics.bins_served ?? '—'}
                    hint={`${metrics.bins_unserved ?? 0} deferred`} icon={Package} />
            <Metric label="Distance" value={fmt.num(metrics.distance_km, 1)} unit="km"
                    icon={RouteIcon} />
            <Metric label="CO₂" value={fmt.num(metrics.co2_kg, 1)} unit="kg"
                    hint={`${fmt.num(metrics.co2_kg_per_km, 2)} kg/km realised`}
                    tone="accent" icon={Leaf} />
            <Metric label="Hazard coverage" value={fmt.pct(metrics.hazard_coverage_pct, 0)}
                    hint={`${metrics.hazard_served ?? 0}/${metrics.hazard_total ?? 0} bins`}
                    tone={metrics.hazard_coverage_pct >= 100 ? 'good' : 'bad'}
                    icon={AlertTriangle} />
            <Metric label="Missed overflows" value={metrics.missed_overflow ?? '—'}
                    tone={metrics.missed_overflow > 0 ? 'bad' : 'good'}
                    hint="served after the predicted deadline" icon={Timer} />
          </Grid>

          <Grid min={320} style={{ marginBottom: 18 }}>
            <Card title="Routes" subtitle="Arrival clock assumes the shift starts at 06:00.">
              {plan.routes?.length
                ? plan.routes.map((route, i) => (
                    <RouteBlock key={route.vehicle_id} route={route} index={i} />
                  ))
                : <EmptyState message="The planner produced no feasible route." />}
            </Card>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <Card title="Plan quality">
                <KeyValue items={[
                  { label: 'Objective', value: fmt.num(plan.objective, 1),
                    hint: 'travel + CO₂ + time + skipped urgency' },
                  { label: 'Solve time', value: `${fmt.num(plan.compute_ms, 0)} ms` },
                  { label: 'Capacity used', value: fmt.pct(metrics.capacity_utilisation_pct, 1) },
                  { label: 'Depot trips', value: metrics.trips ?? '—',
                    hint: 'includes mid-shift tipping' },
                  { label: 'Collected load', value: `${fmt.int(metrics.load_kg)} kg` },
                  { label: 'Urgency collected', value: fmt.pct(metrics.prize_collected_pct, 1) },
                  { label: 'Mean hazard response',
                    value: fmt.hours(metrics.mean_hazard_response_h) },
                  { label: 'Worst hazard response',
                    value: fmt.hours(metrics.worst_hazard_response_h) },
                ]} />
                {plan.params?.aging && (
                  <Note tone="accent">
                    With γ = {plan.params.aging.gamma} and τ = {plan.params.aging.tau_h} h, the
                    certified worst-case wait for the least urgent bin is{' '}
                    <strong>{plan.params.aging.worst_case_wait_bound_h === null
                      ? 'unbounded — γ must exceed 0.5 for a hard guarantee'
                      : `${fmt.num(plan.params.aging.worst_case_wait_bound_h, 1)} h`}</strong>.
                  </Note>
                )}
              </Card>

              {plan.unserved?.length > 0 && (
                <Card title={`Deferred bins (${plan.unserved.length})`}
                      subtitle="Skipping is a decision, not a failure: these cost less urgency
                                than the travel needed to reach them this cycle.">
                  <Table
                    dense
                    columns={[
                      { key: 'node_id', header: 'Bin', render: (r) => <strong>#{r.node_id}</strong> },
                      { key: 'prize', header: 'Urgency', align: 'right',
                        render: (r) => <span className="ui-num">{fmt.num(r.prize, 3)}</span> },
                      { key: 'hazard', header: 'Hazard', align: 'right',
                        render: (r) => r.hazard ? <Badge tone="bad">yes</Badge> : '—' },
                    ]}
                    rows={plan.unserved}
                  />
                </Card>
              )}
            </div>
          </Grid>
        </>
      )}

      {/* ── Solver comparison ── */}
      {comparison && (
        <Card
          title="Planner comparison"
          subtitle="Every solver received the identical bins, fleet, traffic surface and objective
                    weights, and every plan was re-scored with the same feasibility simulator."
          style={{ marginBottom: 18 }}
        >
          <Table
            columns={[
              { key: 'algorithm', header: 'Planner',
                render: (r) => <strong>{SOLVER_LABELS[r.algorithm] || r.algorithm}</strong> },
              { key: 'objective', header: 'Objective', align: 'right',
                render: (r) => <span className="ui-num">{fmt.num(r.objective, 1)}</span> },
              { key: 'km', header: 'Distance', align: 'right',
                render: (r) => <span className="ui-num">{fmt.num(r.metrics?.distance_km, 2)} km</span> },
              { key: 'co2', header: 'CO₂', align: 'right',
                render: (r) => <span className="ui-num">{fmt.num(r.metrics?.co2_kg, 1)} kg</span> },
              { key: 'served', header: 'Served', align: 'right',
                render: (r) => r.metrics?.bins_served ?? '—' },
              { key: 'missed', header: 'Missed', align: 'right',
                render: (r) => (r.metrics?.missed_overflow ?? 0) > 0
                  ? <Badge tone="bad">{r.metrics.missed_overflow}</Badge> : '0' },
              { key: 'ms', header: 'Compute', align: 'right',
                render: (r) => <span className="ui-num">{fmt.num(r.compute_ms, 0)} ms</span> },
            ]}
            rows={(comparison.ranking || []).map((row) => ({
              ...row, ...comparison.results[row.algorithm], __key: row.algorithm,
            }))}
          />
          <Note>{comparison.note}</Note>
        </Card>
      )}

      {/* ── Fleet configuration ── */}
      <Card title="Fleet" subtitle="Constraints the planner treats as hard.">
        <Table
          columns={[
            { key: 'name', header: 'Vehicle', render: (v) => <strong>{v.name}</strong> },
            { key: 'capacity_kg', header: 'Capacity', align: 'right',
              render: (v) => <span className="ui-num">{fmt.int(v.capacity_kg)} kg</span> },
            { key: 'shift_minutes', header: 'Shift', align: 'right',
              render: (v) => <span className="ui-num">{fmt.minutes(v.shift_minutes)}</span> },
            { key: 'compaction_ratio', header: 'Compaction', align: 'right',
              render: (v) => <span className="ui-num">{fmt.num(v.compaction_ratio, 1)}×</span> },
            { key: 'euro_class', header: 'Emissions class',
              render: (v) => <Badge tone="muted">{v.euro_class}</Badge> },
            { key: 'accepts_streams', header: 'Licensed streams',
              render: (v) => (v.accepts_streams?.length ? v.accepts_streams.join(', ') : 'all') },
          ]}
          rows={config?.vehicles || []}
        />
        <Note>
          Enforced: {config?.constraints?.join(' · ')}
        </Note>
      </Card>
    </div>
  )
}

// ── Route rendering ────────────────────────────────────────────────────────
function RouteBlock({ route, index }) {
  const capacityPct = route.capacity_kg ? route.load_kg / route.capacity_kg : 0
  return (
    <div className={`veh-${index % 5}`} style={{ marginBottom: 22 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 9, marginBottom: 4 }}>
        <span className="veh-swatch" />
        <strong style={{ fontSize: 13.5 }}>{route.vehicle_name}</strong>
        <Badge tone="muted">{route.stops.length} stops</Badge>
        {route.trips > 1 && <Badge tone="accent">{route.trips} trips</Badge>}
      </div>
      <div className="ui-stop-meta" style={{ marginBottom: 8 }}>
        <span>{fmt.num(route.distance_km, 2)} km</span>
        <span>{fmt.num(route.co2_kg, 1)} kg CO₂</span>
        <span>{fmt.minutes(route.duration_min)}</span>
      </div>
      <Bar value={capacityPct} tone={capacityPct > 0.9 ? 'warn' : 'accent'}
           label={`${fmt.int(route.load_kg)}/${fmt.int(route.capacity_kg)} kg`} />

      <div className="ui-route" style={{ marginTop: 10 }}>
        {route.stops.map((stop) => (
          <div key={`${stop.node_id}-${stop.sequence}`}
               className={`ui-stop${stop.hazard ? ' hazard' : ''}`}>
            <div className="ui-stop-head">
              <span className="ui-stop-name">
                Bin #{stop.node_id}
                {stop.hazard && <Badge tone="bad" >hazard</Badge>}
              </span>
              <span className="ui-num" style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                {fmt.clock(stop.start_service_min)}
              </span>
            </div>
            <div className="ui-stop-meta">
              <span>{fmt.num(stop.leg_distance_m / 1000, 2)} km leg</span>
              <span>{fmt.num(stop.leg_co2_kg, 2)} kg CO₂</span>
              <span>load {fmt.int(stop.load_after_kg)} kg</span>
              <span>urgency {fmt.num(stop.prize, 2)}</span>
              {stop.wait_min > 0.5 && <span>waited {fmt.num(stop.wait_min, 0)} min for window</span>}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

/** Reshape a stored plan (flat stops) into the grouped shape the UI renders. */
function normaliseStored(stored) {
  const byVehicle = new Map()
  for (const stop of stored.stops || []) {
    if (!byVehicle.has(stop.vehicle)) byVehicle.set(stop.vehicle, [])
    byVehicle.get(stop.vehicle).push(stop)
  }
  return {
    plan_id: stored.id,
    algorithm: stored.algorithm,
    metrics: stored.metrics || {},
    params: stored.params || {},
    objective: stored.metrics?.objective ?? null,
    compute_ms: stored.compute_ms,
    unserved: [],
    routes: [...byVehicle.entries()].map(([vehicleId, stops]) => ({
      vehicle_id: vehicleId,
      vehicle_name: `Vehicle ${vehicleId}`,
      trips: 1,
      distance_km: stops.reduce((s, x) => s + (x.leg_distance_m || 0), 0) / 1000,
      co2_kg: stops.reduce((s, x) => s + (x.leg_co2_kg || 0), 0),
      duration_min: stops.length ? stops[stops.length - 1].departure_min : 0,
      load_kg: stops.length ? stops[stops.length - 1].load_after_kg : 0,
      capacity_kg: null,
      stops: stops
        .slice()
        .sort((a, b) => a.sequence - b.sequence)
        .map((s) => ({
          node_id: s.node, sequence: s.sequence,
          arrival_min: s.arrival_min, start_service_min: s.arrival_min,
          departure_min: s.departure_min, wait_min: 0,
          leg_distance_m: s.leg_distance_m, leg_co2_kg: s.leg_co2_kg,
          load_after_kg: s.load_after_kg, prize: s.priority, hazard: false,
        })),
    })),
  }
}
