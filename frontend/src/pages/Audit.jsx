/**
 * Audit ledger.
 *
 * If a resident complains that their bin overflowed, the operator must be able
 * to show what the system knew, when it knew it, and that the record was not
 * edited afterwards. Every entry commits to its predecessor, so altering any
 * historical record invalidates every entry after it — and the verifier says so
 * explicitly, naming the first sequence number that fails.
 *
 * This is a permissioned hash chain, not a blockchain, and the UI says so: a
 * single operator running its own database has a tamper-evidence problem, not a
 * Byzantine-agreement problem.
 */
import { useEffect, useState } from 'react'
import { CheckCircle2, FileSearch, Link2, ShieldCheck, ShieldX } from 'lucide-react'
import { fetchAuditLedger, verifyLedger } from '../api/endpoints'
import {
  Badge, Card, EmptyState, ErrorState, Grid, KeyValue, Loading, Metric,
  Note, PageHeader, Table, fmt,
} from '../components/ui'
import { toast } from '../components/Toast'

const EVENT_TONE = {
  reading: 'muted', plan: 'accent', service: 'good',
  model: 'accent', fault: 'warn', config: 'muted', genesis: 'muted',
}

export default function Audit() {
  const [ledger, setLedger] = useState(null)
  const [verification, setVerification] = useState(null)
  const [proof, setProof] = useState(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [expanded, setExpanded] = useState(null)

  useEffect(() => {
    let alive = true
    Promise.all([fetchAuditLedger({ limit: 60 }), verifyLedger()])
      .then(([l, v]) => {
        if (!alive) return
        setLedger(l.data)
        setVerification(v.data)
      })
      .catch(() => alive && setError('Could not load the audit ledger.'))
      .finally(() => alive && setLoading(false))
    return () => { alive = false }
  }, [])

  const runVerify = async () => {
    setBusy(true)
    try {
      const res = await verifyLedger()
      setVerification(res.data)
      toast(res.data.valid ? 'Ledger verified intact.' : 'Tampering detected.',
            res.data.valid ? 'success' : 'error')
    } catch {
      toast('Verification failed.', 'error')
    } finally {
      setBusy(false)
    }
  }

  const showProof = async (sequence) => {
    try {
      const res = await verifyLedger({ sequence })
      setProof(res.data)
    } catch {
      toast('Could not build an inclusion proof.', 'error')
    }
  }

  if (loading) return <Loading label="Loading ledger…" />
  if (error) return <ErrorState message={error} />

  const valid = verification?.valid

  return (
    <div>
      <PageHeader
        title="Audit ledger"
        description="Append-only, hash-chained record of telemetry, dispatch decisions,
                     collections, model activations and detected faults. Each entry commits to
                     its predecessor, so any retroactive edit invalidates the whole suffix."
        actions={
          <button className="btn btn-primary btn-sm" onClick={runVerify} disabled={busy}>
            <ShieldCheck size={14} className={busy ? 'ui-spin' : ''} />
            {busy ? 'Verifying…' : 'Verify chain'}
          </button>
        }
      />

      <Grid min={185} style={{ marginBottom: 18 }}>
        <Metric label="Chain status" value={valid ? 'Intact' : 'Broken'}
                tone={valid ? 'good' : 'bad'} icon={valid ? ShieldCheck : ShieldX}
                hint={valid ? 'every link verified'
                  : `first failure at #${verification?.first_invalid_sequence}`} />
        <Metric label="Entries" value={fmt.int(verification?.total_entries)} icon={FileSearch} />
        <Metric label="Blocks anchored" value={verification?.blocks_verified ?? 0}
                hint={`${verification?.block_size ?? 64} entries per Merkle root`} icon={Link2} />
        <Metric label="Signed" value={verification?.signed ? 'HMAC' : 'Chain only'}
                tone={verification?.signed ? 'good' : 'default'}
                hint={verification?.signed ? 'entries additionally signed'
                  : 'set AUDIT_HMAC_KEY to sign'} />
      </Grid>

      {!valid && verification?.errors?.length > 0 && (
        <Card title="Integrity failures" style={{ marginBottom: 18 }}>
          <Table
            dense
            columns={[
              { key: 'sequence', header: '#', render: (r) => <strong>{r.sequence}</strong> },
              { key: 'error', header: 'Failure',
                render: (r) => <Badge tone="bad">{r.error.replace(/_/g, ' ')}</Badge> },
              { key: 'detail', header: 'Detail' },
            ]}
            rows={verification.errors.map((e, i) => ({ ...e, __key: i }))}
          />
        </Card>
      )}

      <Grid min={340}>
        <Card
          title="Recent entries"
          subtitle="Click an entry to inspect its payload, or request a Merkle inclusion proof."
        >
          {!ledger?.entries?.length ? (
            <EmptyState message="The ledger is empty." />
          ) : (
            <Table
              dense
              columns={[
                { key: 'sequence', header: '#', width: 52,
                  render: (r) => <strong className="ui-num">{r.sequence}</strong> },
                { key: 'event_type', header: 'Event',
                  render: (r) => (
                    <Badge tone={EVENT_TONE[r.event_type] || 'muted'}>{r.event_type}</Badge>
                  ) },
                { key: 'created_at', header: 'When',
                  render: (r) => (
                    <span style={{ fontSize: 11.5 }}>{fmt.datetime(r.created_at)}</span>
                  ) },
                { key: 'actor', header: 'Actor' },
                { key: 'entry_hash', header: 'Hash',
                  render: (r) => (
                    <span className="ui-mono" title={r.entry_hash}>{fmt.hash(r.entry_hash, 12)}</span>
                  ) },
                { key: 'actions', header: '', align: 'right',
                  render: (r) => (
                    <div style={{ display: 'flex', gap: 6, justifyContent: 'flex-end' }}>
                      <button className="btn btn-ghost btn-sm"
                              onClick={() => setExpanded(expanded === r.sequence ? null : r)}>
                        {expanded?.sequence === r.sequence ? 'Hide' : 'Payload'}
                      </button>
                      <button className="btn btn-ghost btn-sm"
                              onClick={() => showProof(r.sequence)}>Proof</button>
                    </div>
                  ) },
              ]}
              rows={ledger.entries.map((e) => ({ ...e, __key: e.sequence }))}
            />
          )}
          {expanded && (
            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '.05em',
                            color: 'var(--text-muted)', marginBottom: 6, fontWeight: 600 }}>
                Entry #{expanded.sequence} payload
              </div>
              <pre className="ui-mono" style={{
                background: 'var(--bg-surface)', border: '1px solid var(--border)',
                borderRadius: 8, padding: 12, overflowX: 'auto', fontSize: 11,
                maxHeight: 260, lineHeight: 1.55, color: 'var(--text-secondary)',
              }}>{JSON.stringify(expanded.payload, null, 2)}</pre>
              <KeyValue items={[
                { label: 'Payload digest',
                  value: <span className="ui-mono">{fmt.hash(expanded.payload_sha256, 20)}</span> },
                { label: 'Previous hash',
                  value: <span className="ui-mono">{fmt.hash(expanded.prev_hash, 20)}</span> },
              ]} />
            </div>
          )}
        </Card>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {proof && (
            <Card title={`Inclusion proof for #${proof.sequence}`}
                  subtitle="Proves this entry belongs to its block without disclosing the rest of
                            the block — which matters when the block concerns other residents.">
              {proof.error ? (
                <ErrorState message={proof.error} />
              ) : (
                <>
                  <KeyValue items={[
                    { label: 'Verified',
                      value: proof.verified
                        ? <Badge tone="good"><CheckCircle2 size={11} /> yes</Badge>
                        : <Badge tone="bad">no</Badge> },
                    { label: 'Block', value: `#${proof.block_index} (${proof.block_entries} entries)` },
                    { label: 'Block complete', value: proof.block_complete ? 'yes' : 'still filling' },
                    { label: 'Proof length', value: `${proof.proof.length} steps` },
                    { label: 'Merkle root',
                      value: <span className="ui-mono">{fmt.hash(proof.merkle_root, 18)}</span> },
                  ]} />
                  <Note>
                    A proof of {proof.proof.length} hashes is enough to verify membership of a
                    block containing {proof.block_entries} entries — logarithmic, not linear.
                  </Note>
                </>
              )}
            </Card>
          )}

          <Card title="Event mix">
            <Table
              dense
              columns={[
                { key: 'event', header: 'Event',
                  render: (r) => <Badge tone={EVENT_TONE[r.event] || 'muted'}>{r.event}</Badge> },
                { key: 'count', header: 'Entries', align: 'right',
                  render: (r) => <span className="ui-num">{fmt.int(r.count)}</span> },
              ]}
              rows={Object.entries(ledger?.counts_by_event || {})
                .sort((a, b) => b[1] - a[1])
                .map(([event, count]) => ({ __key: event, event, count }))}
            />
          </Card>

          <Card title="How this works">
            <KeyValue items={[
              { label: 'Mechanism', value: 'SHA-256 hash chain' },
              { label: 'Anchoring', value: `Merkle root every ${ledger?.design?.block_size ?? 64}` },
              { label: 'Signature', value: ledger?.design?.signature || 'HMAC-SHA256 (optional)' },
            ]} />
            {ledger?.design?.not_a_blockchain && (
              <Note tone="accent">{ledger.design.not_a_blockchain}</Note>
            )}
            {ledger?.design?.limits?.length > 0 && (
              <Note tone="warn">
                <strong>Limits.</strong> {ledger.design.limits.join(' ')}
              </Note>
            )}
          </Card>
        </div>
      </Grid>
    </div>
  )
}
