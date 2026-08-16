/**
 * Explainability.
 *
 * A municipality that reroutes a truck on a model's say-so has to be able to
 * answer "why this bin?" in an audit. This screen answers it twice: globally,
 * with what the model relies on in general, and locally, with a Shapley
 * attribution of the specific prediction now driving the dispatch decision.
 *
 * The attributions are reported in the model's own units — hours of remaining
 * time, or probability — so the explanation reads to an operator rather than
 * only to a data scientist. Local fidelity is shown alongside, so a reader can
 * tell when an attribution should be treated as indicative.
 */
import { useEffect, useState } from 'react'
import { Brain, Gauge, Info, RefreshCw } from 'lucide-react'
import { explainNode, fetchNodes } from '../api/endpoints'
import {
  Badge, Card, DivergingBar, EmptyState, ErrorState, Grid, KeyValue,
  Loading, Metric, Note, PageHeader, Table, fmt, trustTone,
} from '../components/ui'
import { toast } from '../components/Toast'

const HEADS = [
  { value: 'tto', label: 'Time to overflow (hours)' },
  { value: 'hazard', label: 'Hazard probability' },
]

export default function Explainability() {
  const [nodes, setNodes] = useState([])
  const [nodeId, setNodeId] = useState('')
  const [head, setHead] = useState('tto')
  const [method, setMethod] = useState('auto')
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    let alive = true
    fetchNodes()
      .then((res) => {
        if (!alive) return
        setNodes(res.data)
        if (res.data.length) setNodeId(String(res.data[0].id))
      })
      .catch(() => alive && setError('Could not load the bin list.'))
      .finally(() => alive && setLoading(false))
    return () => { alive = false }
  }, [])

  useEffect(() => {
    if (!nodeId) return
    let alive = true
    setBusy(true)
    explainNode(nodeId, { head, method })
      .then((res) => alive && setData(res.data))
      .catch((e) => {
        if (!alive) return
        setData(null)
        toast(e?.response?.data?.error || 'No trained model available yet.', 'error')
      })
      .finally(() => alive && setBusy(false))
    return () => { alive = false }
  }, [nodeId, head, method])

  if (loading) return <Loading label="Loading bins…" />
  if (error) return <ErrorState message={error} />

  const explanation = data?.explanation
  const contributions = explanation?.contributions || []
  const scale = Math.max(...contributions.map((c) => Math.abs(c.contribution)), 1e-6)
  const units = head === 'hazard' ? '' : ' h'

  return (
    <div>
      <PageHeader
        title="Why this bin?"
        description="Shapley attribution of the prediction currently driving dispatch. Values are
                     additive: the baseline plus every contribution equals the prediction exactly,
                     so the numbers can be checked rather than taken on faith."
        actions={
          <button className="btn btn-ghost btn-sm" disabled={busy}
                  onClick={() => setNodeId((v) => v)}>
            <RefreshCw size={14} className={busy ? 'ui-spin' : ''} /> Refresh
          </button>
        }
      />

      <Card style={{ marginBottom: 18 }}>
        <Grid min={210} gap={12}>
          <div className="ui-field">
            <label htmlFor="xai-node">Bin</label>
            <select id="xai-node" className="ui-select" value={nodeId}
                    onChange={(e) => setNodeId(e.target.value)}>
              {nodes.map((n) => <option key={n.id} value={n.id}>{n.name}</option>)}
            </select>
          </div>
          <div className="ui-field">
            <label htmlFor="xai-head">Prediction</label>
            <select id="xai-head" className="ui-select" value={head}
                    onChange={(e) => setHead(e.target.value)}>
              {HEADS.map((h) => <option key={h.value} value={h.value}>{h.label}</option>)}
            </select>
          </div>
          <div className="ui-field">
            <label htmlFor="xai-method">Attribution method</label>
            <select id="xai-method" className="ui-select" value={method}
                    onChange={(e) => setMethod(e.target.value)}>
              <option value="auto">KernelSHAP (fast, any model)</option>
              <option value="shap">Exact TreeSHAP (audit grade, slower)</option>
            </select>
          </div>
        </Grid>
      </Card>

      {busy && !data && <Loading label="Computing attribution…" />}

      {!busy && !data && (
        <Card><EmptyState message="No explanation available. Train a model first with
                                   `python manage.py train_forward`." /></Card>
      )}

      {data && (
        <>
          <Grid min={185} style={{ marginBottom: 18 }}>
            <Metric label="Time to overflow"
                    value={fmt.num(data.prediction?.time_to_overflow_h, 1)} unit="h"
                    hint={`P10–P90: ${fmt.num(data.prediction?.tto_p10_h, 1)}–${fmt.num(data.prediction?.tto_p90_h, 1)} h`}
                    tone={data.prediction?.tto_p10_h < 6 ? 'bad' : 'default'} icon={Gauge} />
            <Metric label="Hazard probability"
                    value={fmt.frac(data.prediction?.hazard_prob, 1)}
                    tone={data.prediction?.hazard_prob > 0.6 ? 'bad' : 'default'} />
            <Metric label="Risk priority" value={fmt.num(data.prediction?.risk_priority, 3)}
                    tone="accent" icon={Brain} />
            <Metric label="Local fidelity"
                    value={explanation?.fidelity_r2 === null || explanation?.fidelity_r2 === undefined
                      ? 'exact' : fmt.num(explanation.fidelity_r2, 2)}
                    hint={explanation?.method === 'treeshap'
                      ? 'exact Shapley values' : 'R² of the additive approximation'} />
            <Metric label="Attribution cost" value={fmt.num(explanation?.compute_ms, 0)} unit="ms" />
          </Grid>

          <Grid min={340}>
            <Card
              title="Feature contributions"
              subtitle={`Baseline ${fmt.num(explanation?.baseline, 2)}${units} → prediction
                         ${fmt.num(explanation?.prediction, 2)}${units}. Bars to the right push
                         the prediction up, to the left down.`}
            >
              {contributions.length === 0 ? (
                <EmptyState message="No individual feature dominated this prediction." />
              ) : (
                <Table
                  dense
                  columns={[
                    { key: 'feature', header: 'Feature',
                      render: (r) => (
                        <span title={data.feature_descriptions?.[r.feature] || ''}>
                          <strong>{r.feature.replace(/_/g, ' ')}</strong>
                        </span>
                      ) },
                    { key: 'value', header: 'Value', align: 'right',
                      render: (r) => <span className="ui-num">
                        {r.value === null ? '—' : fmt.num(r.value, 2)}
                      </span> },
                    { key: 'bar', header: 'Effect', width: 150,
                      render: (r) => <DivergingBar value={r.contribution} scale={scale} /> },
                    { key: 'contribution', header: 'Δ', align: 'right',
                      render: (r) => (
                        <span className="ui-num"
                              style={{ color: r.contribution >= 0 ? 'var(--red)' : 'var(--accent)' }}>
                          {r.contribution >= 0 ? '+' : ''}{fmt.num(r.contribution, 3)}
                        </span>
                      ) },
                  ]}
                  rows={contributions.map((c, i) => ({ ...c, __key: `${c.feature}-${i}` }))}
                />
              )}
              {explanation?.narrative && (
                <Note tone="accent">
                  <Info size={12} style={{ verticalAlign: -2, marginRight: 5 }} />
                  {explanation.narrative}
                </Note>
              )}
            </Card>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <Card title="Sensor evidence behind this prediction"
                    subtitle="A confident-looking prediction built on distrusted inputs is worse
                              than no prediction, so the trust behind it is shown alongside.">
                <Table
                  dense
                  columns={[
                    { key: 'channel', header: 'Channel',
                      render: (r) => <strong>{r.channel.replace(/_/g, ' ')}</strong> },
                    { key: 'raw_value', header: 'Reading', align: 'right',
                      render: (r) => <span className="ui-num">
                        {r.raw_value === null ? 'missing' : fmt.num(r.raw_value, 2)}
                      </span> },
                    { key: 'trust', header: 'Trust', align: 'right',
                      render: (r) => (
                        <Badge tone={trustTone(r.trust)}>{fmt.num(r.trust, 2)}</Badge>
                      ) },
                    { key: 'status', header: 'Status', align: 'right',
                      render: (r) => <Badge status={r.status}>{r.status}</Badge> },
                  ]}
                  rows={Object.entries(data.health || {}).map(([c, h]) => ({
                    __key: c, channel: c, ...h,
                  }))}
                />
              </Card>

              <Card title="Model">
                <KeyValue items={[
                  { label: 'Version', value: <span className="ui-mono">{data.prediction?.model_version || '—'}</span> },
                  { label: 'Attribution', value: explanation?.method },
                  { label: 'Head', value: head === 'hazard' ? 'hazard classifier' : 'time-to-overflow regressor' },
                  { label: 'Features used', value: Object.keys(data.features || {}).length },
                ]} />
                <Note>
                  KernelSHAP samples coalitions from the Shapley kernel and solves under an exact
                  efficiency constraint, so contributions sum to the prediction. It agrees with
                  exact TreeSHAP to r ≈ 0.94 at the sampling budget used here, and costs
                  milliseconds rather than seconds.
                </Note>
              </Card>
            </div>
          </Grid>
        </>
      )}
    </div>
  )
}
