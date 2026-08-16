/**
 * Sensing integrity.
 *
 * Two jobs. First, show the operator which channels are believed and by how
 * much — trust is continuous, not a red/green light, because the failure modes
 * that matter in the field are partial. Second, let that claim be *tested*: the
 * fault console injects any mode from the taxonomy into a live bin and shows
 * whether the detectors catch it.
 *
 * Injected readings are labelled and removable, so a demonstration never
 * contaminates the record.
 */
import { useEffect, useMemo, useState } from 'react'
import {
  Activity, RefreshCw, ShieldAlert, Syringe, Trash2, TrendingDown,
} from 'lucide-react'
import {
  clearInjectedFaults, fetchFaultTaxonomy, fetchNodes, fetchSensorHealth, injectFault,
} from '../api/endpoints'
import {
  Badge, Card, EmptyState, ErrorState, Grid, KeyValue, Loading, Metric,
  Note, PageHeader, Table, fmt, trustTone,
} from '../components/ui'
import { toast } from '../components/Toast'

const CHANNELS = ['waste_level', 'gas_level', 'temperature', 'humidity']
const CHANNEL_LABEL = {
  waste_level: 'Fill', gas_level: 'Gas', temperature: 'Temp', humidity: 'Humidity',
}

export default function SensorHealth() {
  const [health, setHealth] = useState(null)
  const [taxonomy, setTaxonomy] = useState(null)
  const [nodes, setNodes] = useState([])
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const [nodeId, setNodeId] = useState('')
  const [mode, setMode] = useState('drift')
  const [channel, setChannel] = useState('gas_level')
  const [magnitude, setMagnitude] = useState(0.35)
  const [samples, setSamples] = useState(30)
  const [lastResult, setLastResult] = useState(null)

  const load = async (refresh = false) => {
    try {
      const res = await fetchSensorHealth(refresh ? { refresh: 'true' } : undefined)
      setHealth(res.data)
    } catch {
      setError('Could not load sensor health.')
    }
  }

  useEffect(() => {
    let alive = true
    Promise.all([fetchSensorHealth({ refresh: 'true' }), fetchFaultTaxonomy(), fetchNodes()])
      .then(([h, t, n]) => {
        if (!alive) return
        setHealth(h.data)
        setTaxonomy(t.data)
        setNodes(n.data)
        if (n.data.length) setNodeId(String(n.data[0].id))
      })
      .catch(() => alive && setError('Could not load sensor health.'))
      .finally(() => alive && setLoading(false))
    return () => { alive = false }
  }, [])

  const byNode = useMemo(() => {
    const map = new Map()
    for (const row of health?.channels || []) {
      if (!map.has(row.node)) map.set(row.node, { name: row.node_name, channels: {} })
      map.get(row.node).channels[row.channel] = row
    }
    return map
  }, [health])

  const inject = async () => {
    setBusy(true)
    try {
      const res = await injectFault({
        node_id: Number(nodeId), mode, channel,
        magnitude, rate: 0.9, stealth: 0.45, n_samples: samples,
      })
      setLastResult(res.data)
      const detected = res.data.assessment?.[channel]?.status
      toast(
        detected && detected !== 'ok'
          ? `Detected: ${channel} is now "${detected}"`
          : `Injected, but ${channel} still reads healthy`,
        detected && detected !== 'ok' ? 'success' : 'error',
      )
      await load(true)
    } catch (e) {
      toast(e?.response?.data?.error || 'Injection failed.', 'error')
    } finally {
      setBusy(false)
    }
  }

  const clear = async () => {
    setBusy(true)
    try {
      const res = await clearInjectedFaults()
      setLastResult(null)
      toast(`Removed ${res.data.removed} injected readings.`, 'success')
      await load(true)
    } catch {
      toast('Could not clear injected readings.', 'error')
    } finally {
      setBusy(false)
    }
  }

  if (loading) return <Loading label="Assessing sensors…" />
  if (error) return <ErrorState message={error} />

  const summary = health?.summary || {}
  const counts = summary.status_counts || {}

  return (
    <div>
      <PageHeader
        title="Sensing integrity"
        description="Every channel carries a continuous trust weight, and that weight — not a
                     binary available/unavailable flag — is what scales its contribution to a
                     bin's priority. A degraded channel fades out smoothly instead of either
                     being fully believed or silently replaced by a zero."
        actions={
          <button className="btn btn-ghost btn-sm" onClick={() => load(true)} disabled={busy}>
            <RefreshCw size={14} /> Re-assess
          </button>
        }
      />

      <Grid min={175} style={{ marginBottom: 18 }}>
        <Metric label="Mean trust" value={fmt.num(summary.mean_trust, 3)}
                tone={trustTone(summary.mean_trust)} icon={Activity} />
        <Metric label="Lowest trust" value={fmt.num(summary.min_trust, 3)}
                tone={trustTone(summary.min_trust)} icon={TrendingDown} />
        <Metric label="Healthy" value={counts.ok ?? 0} unit="ch" tone="good" />
        <Metric label="Suspect" value={counts.suspect ?? 0} unit="ch" tone="warn" />
        <Metric label="Degraded" value={counts.degraded ?? 0} unit="ch" tone="bad" />
        <Metric label="Failed" value={counts.failed ?? 0} unit="ch" tone="bad"
                hint="dropped from scoring" icon={ShieldAlert} />
      </Grid>

      {/* ── Trust matrix ── */}
      <Card
        title="Trust by bin and channel"
        subtitle="1.00 is fully believed; below 0.15 the channel is dropped and the remaining
                  weights are renormalised so the score stays on a comparable scale."
        style={{ marginBottom: 18 }}
      >
        {byNode.size === 0 ? (
          <EmptyState message="No sensor health recorded yet." />
        ) : (
          <div className="trust-grid"
               style={{ gridTemplateColumns: `minmax(110px, 1.4fr) repeat(${CHANNELS.length}, 1fr)` }}>
            <div className="trust-head" style={{ textAlign: 'left' }}>Bin</div>
            {CHANNELS.map((c) => <div key={c} className="trust-head">{CHANNEL_LABEL[c]}</div>)}
            {[...byNode.entries()].map(([id, node]) => (
              <Row key={id} id={id} node={node} />
            ))}
          </div>
        )}
        <Note>
          Detectors: {Object.keys(health?.detectors || {}).join(' · ')}. Drift and spike are
          measured against the fleet, not in absolute terms — the whole network's temperature
          rises together every afternoon, and a bin being emptied is a large legitimate step.
        </Note>
      </Card>

      <Grid min={330}>
        {/* ── Fault console ── */}
        <Card
          title="Fault console"
          subtitle="Inject a fault into a live bin and see whether the detectors catch it.
                    Injected readings are labelled and can be removed again."
          actions={
            <button className="btn btn-ghost btn-sm" onClick={clear} disabled={busy}>
              <Trash2 size={13} /> Clear injected
            </button>
          }
        >
          <Grid min={150} gap={12}>
            <div className="ui-field">
              <label htmlFor="fault-node">Bin</label>
              <select id="fault-node" className="ui-select" value={nodeId}
                      onChange={(e) => setNodeId(e.target.value)}>
                {nodes.map((n) => <option key={n.id} value={n.id}>{n.name}</option>)}
              </select>
            </div>
            <div className="ui-field">
              <label htmlFor="fault-channel">Channel</label>
              <select id="fault-channel" className="ui-select" value={channel}
                      onChange={(e) => setChannel(e.target.value)}>
                {CHANNELS.map((c) => <option key={c} value={c}>{CHANNEL_LABEL[c]}</option>)}
              </select>
            </div>
            <div className="ui-field">
              <label htmlFor="fault-mode">Fault mode</label>
              <select id="fault-mode" className="ui-select" value={mode}
                      onChange={(e) => setMode(e.target.value)}>
                {Object.keys(taxonomy?.modes || {}).map((m) => (
                  <option key={m} value={m}>{m.replace(/_/g, ' ')}</option>
                ))}
              </select>
            </div>
            <div className="ui-field">
              <label htmlFor="fault-mag">Magnitude {magnitude.toFixed(2)}</label>
              <input id="fault-mag" className="ui-range" type="range" min="0.05" max="0.8" step="0.05"
                     value={magnitude} onChange={(e) => setMagnitude(Number(e.target.value))} />
            </div>
            <div className="ui-field">
              <label htmlFor="fault-n">Samples {samples}</label>
              <input id="fault-n" className="ui-range" type="range" min="10" max="80" step="5"
                     value={samples} onChange={(e) => setSamples(Number(e.target.value))} />
            </div>
          </Grid>

          {taxonomy?.modes?.[mode] && <Note tone="accent">{taxonomy.modes[mode]}</Note>}

          <button className="btn btn-primary btn-sm" onClick={inject} disabled={busy}
                  style={{ marginTop: 12 }}>
            <Syringe size={14} className={busy ? 'ui-spin' : ''} />
            {busy ? 'Working…' : 'Inject fault'}
          </button>

          {lastResult && (
            <div style={{ marginTop: 14 }}>
              <KeyValue items={[
                { label: 'Samples written', value: lastResult.samples_written },
                { label: 'Samples corrupted', value: lastResult.samples_affected },
                { label: 'Resulting status',
                  value: <Badge status={lastResult.assessment?.[channel]?.status}>
                    {lastResult.assessment?.[channel]?.status || 'unknown'}
                  </Badge> },
                { label: 'Resulting trust',
                  value: fmt.num(lastResult.assessment?.[channel]?.trust, 3) },
                { label: 'Detectors that fired',
                  value: (lastResult.assessment?.[channel]?.flags || []).join(', ') || 'none' },
              ]} />
            </div>
          )}
        </Card>

        {/* ── Taxonomy ── */}
        <Card
          title="Fault taxonomy"
          subtitle="The original evaluation tested exactly one of these — a channel reporting
                    zero, which is the easiest case to survive because the anomaly is obvious."
        >
          <Table
            dense
            columns={[
              { key: 'mode', header: 'Mode',
                render: (r) => (
                  <strong style={{ whiteSpace: 'nowrap' }}>{r.mode.replace(/_/g, ' ')}</strong>
                ) },
              { key: 'desc', header: 'Physical origin' },
            ]}
            rows={Object.entries(taxonomy?.modes || {}).map(([m, d]) => ({
              __key: m, mode: m, desc: d,
            }))}
          />
        </Card>
      </Grid>
    </div>
  )
}

function Row({ id, node }) {
  return (
    <>
      <div className="trust-row-label" title={node.name}>{node.name}</div>
      {CHANNELS.map((channel) => {
        const cell = node.channels[channel]
        if (!cell) return <div key={channel} className="trust-cell tone-muted">—</div>
        const flags = cell.detail?.flags || []
        return (
          <div
            key={channel}
            className={`trust-cell tone-${trustTone(cell.trust)}`}
            title={`${cell.status}${flags.length ? ` — ${flags.join(', ')}` : ''}`}
          >
            {fmt.num(cell.trust, 2)}
          </div>
        )
      })}
    </>
  )
}
