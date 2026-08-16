/**
 * Model governance.
 *
 * Shows what is actually deployed, how it was validated, and what the online
 * layer is doing — the three questions a reviewer or an auditor asks about a
 * learned component.
 *
 * The serving policy is stated explicitly because it is a design commitment,
 * not an implementation detail: batch training never runs on the request path,
 * so an operator using the application never pays for a training job. The
 * request path performs at most one bounded gradient step on a small linear
 * residual corrector, and that corrector only takes effect while it is
 * measurably beating the frozen base model.
 */
import { useEffect, useState } from 'react'
import {
  Activity, Boxes, Brain, GitBranch, RotateCcw, ShieldCheck, TrendingUp,
} from 'lucide-react'
import { fetchModelStatus, resetContinualLearner } from '../api/endpoints'
import {
  Badge, Bar, Card, EmptyState, ErrorState, Grid, KeyValue, Loading,
  Metric, Note, PageHeader, Table, fmt,
} from '../components/ui'
import { toast } from '../components/Toast'

export default function Models() {
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const load = () =>
    fetchModelStatus()
      .then((res) => setStatus(res.data))
      .catch(() => setError('Could not load model status.'))

  useEffect(() => {
    let alive = true
    load().finally(() => alive && setLoading(false))
    return () => { alive = false }
  }, [])

  const reset = async () => {
    setBusy(true)
    try {
      await resetContinualLearner()
      await load()
      toast('Online corrector discarded; serving the frozen base model.', 'success')
    } catch {
      toast('Could not reset the corrector.', 'error')
    } finally {
      setBusy(false)
    }
  }

  if (loading) return <Loading label="Loading model status…" />
  if (error) return <ErrorState message={error} />

  const model = status?.forward_model || {}
  const metrics = model.metrics || {}
  const continual = status?.continual || {}
  const importance = status?.global_importance || []
  const protocol = status?.hpo_protocol || {}

  if (!model.available) {
    return (
      <div>
        <PageHeader title="Models" />
        <Card>
          <EmptyState message="No model has been trained yet. Run
                               `python manage.py train_forward` to fit the forward bundle." />
        </Card>
      </div>
    )
  }

  return (
    <div>
      <PageHeader
        title="Models"
        description="A forward-looking bundle: a regressor for hours until overflow, quantile
                     heads giving a prediction interval, and a calibrated hazard classifier. The
                     labels come from the strictly future trajectory, which the features cannot
                     see, so the scores are forecasting skill rather than self-consistency."
        actions={
          <button className="btn btn-ghost btn-sm" onClick={reset} disabled={busy}>
            <RotateCcw size={14} className={busy ? 'ui-spin' : ''} /> Reset corrector
          </button>
        }
      />

      <Grid min={185} style={{ marginBottom: 18 }}>
        <Metric label="Test R²" value={fmt.num(metrics.reg_r2, 3)}
                hint={`mean-predictor baseline ${fmt.num(metrics.baseline_mean_r2, 3)}`}
                tone="accent" icon={TrendingUp} />
        <Metric label="Test MAE" value={fmt.num(metrics.reg_mae_h, 2)} unit="h"
                hint={`baseline ${fmt.num(metrics.baseline_mean_mae_h, 2)} h`} />
        <Metric label="Grouped CV R²" value={fmt.num(metrics.reg_cv_r2_mean, 3)}
                hint={`± ${fmt.num(metrics.reg_cv_r2_std, 3)} across bins`} />
        <Metric label="Hazard AUC" value={fmt.num(metrics.hazard_roc_auc, 3)}
                hint={`Brier ${fmt.num(metrics.hazard_brier, 3)}`} icon={ShieldCheck} />
        <Metric label="Interval coverage" value={fmt.frac(metrics.interval_coverage_p10_p90, 1)}
                hint={`nominal ${fmt.frac(metrics.interval_nominal_coverage, 0)}`}
                tone={Math.abs((metrics.interval_coverage_p10_p90 ?? 0) - 0.8) < 0.05
                  ? 'good' : 'warn'} />
        <Metric label="Training rows" value={fmt.int(model.training_rows)}
                hint={`${metrics.n_bins ?? '—'} bins`} icon={Boxes} />
      </Grid>

      <Grid min={340}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <Card title="Continual learning"
                subtitle="A small linear model learns the residual of the frozen ensemble by
                          stochastic gradient descent, bounded so it can never slow a request.">
            <Grid min={140} gap={12} style={{ marginBottom: 12 }}>
              <Metric label="Corrector" value={continual.active ? 'Active' : 'Dormant'}
                      tone={continual.active ? 'good' : 'default'}
                      hint={continual.active ? 'beating the base model'
                        : 'predictions pass through untouched'} icon={Activity} />
              <Metric label="Error reduction" value={fmt.pct(continual.improvement_pct, 1)}
                      tone={continual.improvement_pct > 0 ? 'good' : 'default'} />
              <Metric label="Drift events" value={continual.drift_events ?? 0}
                      tone={continual.needs_retrain ? 'warn' : 'default'} />
            </Grid>
            <KeyValue items={[
              { label: 'Samples seen', value: fmt.int(continual.samples_seen) },
              { label: 'Replay buffer', value: fmt.int(continual.buffer_size) },
              { label: 'Updates applied', value: fmt.int(continual.updates_applied) },
              { label: 'Updates skipped', value: fmt.int(continual.updates_skipped),
                hint: 'rate limited' },
              { label: 'Prequential MAE (base)',
                value: fmt.num(continual.prequential_mae_base, 3) },
              { label: 'Prequential MAE (corrected)',
                value: fmt.num(continual.prequential_mae_corrected, 3) },
              { label: 'Budget per update',
                value: `${continual.budget?.max_samples_per_update ?? '—'} samples / ` +
                       `${continual.budget?.max_ms_per_update ?? '—'} ms` },
            ]} />
            {continual.needs_retrain && (
              <Note tone="warn">
                The corrector is removing a large share of the error, which means the frozen
                ensemble has gone stale. Schedule an offline retrain —
                <span className="ui-mono"> manage.py train_forward</span>.
              </Note>
            )}
            <Note>{status?.serving_policy?.note}</Note>
          </Card>

          <Card title="Validation protocol"
                subtitle="How the model was selected, stated so it can be checked.">
            <KeyValue items={[
              { label: 'Outer split', value: protocol.outer_split || model.validation },
              { label: 'Inner CV', value: protocol.inner_cv || '—' },
              { label: 'Search', value: protocol.search || '—' },
              { label: 'Refit', value: protocol.refit || '—' },
            ]} />
            {protocol.rationale && (
              <Note>
                <strong>Why these choices.</strong>
                <ul style={{ margin: '6px 0 0', paddingLeft: 16 }}>
                  {Object.entries(protocol.rationale).map(([key, reason]) => (
                    <li key={key} style={{ marginBottom: 3 }}>{reason}.</li>
                  ))}
                </ul>
              </Note>
            )}
          </Card>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <Card title="What the model relies on"
                subtitle="Permutation importance on the held-out set: the loss increase when each
                          feature is shuffled.">
            {importance.length === 0 ? (
              <EmptyState message="Importance was not computed for this artefact." />
            ) : (
              <Table
                dense
                columns={[
                  { key: 'feature', header: 'Feature',
                    render: (r) => <strong>{r.feature.replace(/_/g, ' ')}</strong> },
                  { key: 'bar', header: 'Share', width: 130,
                    render: (r) => <Bar value={r.share} tone="accent" /> },
                  { key: 'share', header: '', align: 'right',
                    render: (r) => <span className="ui-num">{fmt.frac(r.share, 1)}</span> },
                ]}
                rows={importance.slice(0, 12).map((r) => ({ ...r, __key: r.feature }))}
              />
            )}
          </Card>

          <Card title="Selected hyperparameters">
            <KeyValue items={Object.entries(model.hyperparameters?.regressor || {})
              .map(([k, v]) => ({ label: k.replace(/_/g, ' '), value: String(v) }))} />
          </Card>

          <Card title="Registry" subtitle="Every artefact is hashed, so a prediction stays
                                           attributable to the exact model that produced it.">
            <Table
              dense
              columns={[
                { key: 'version', header: 'Version',
                  render: (r) => (
                    <span className="ui-mono" style={{ fontSize: 11 }}>{r.version}</span>
                  ) },
                { key: 'kind', header: 'Kind',
                  render: (r) => <Badge tone="muted">{r.kind.replace(/_/g, ' ')}</Badge> },
                { key: 'training_rows', header: 'Rows', align: 'right',
                  render: (r) => <span className="ui-num">{fmt.int(r.training_rows)}</span> },
                { key: 'is_active', header: 'Active', align: 'right',
                  render: (r) => r.is_active
                    ? <Badge tone="good"><GitBranch size={10} /> live</Badge> : '—' },
              ]}
              rows={(status?.registry || []).map((r) => ({ ...r, __key: r.id }))}
            />
            {model.artifact_sha256 && (
              <Note>
                Active artefact digest:{' '}
                <span className="ui-mono">{fmt.hash(model.artifact_sha256, 24)}</span>
              </Note>
            )}
          </Card>

          <Card title="Explainability">
            <KeyValue items={[
              { label: 'Local default', value: status?.explainability?.local_default },
              { label: 'Exact TreeSHAP',
                value: status?.explainability?.treeshap_available ? 'available' : 'not installed' },
              { label: 'Global importance',
                value: status?.explainability?.global_importance_available ? 'stored' : 'not computed' },
            ]} />
          </Card>
        </div>
      </Grid>
    </div>
  )
}
