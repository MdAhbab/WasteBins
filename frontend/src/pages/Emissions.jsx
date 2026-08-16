/**
 * Sustainability.
 *
 * The point of this screen is to show *where* the carbon actually goes. A single
 * flat kg-per-km factor cannot distinguish a line-haul kilometre from a
 * collection kilometre, and in refuse collection those differ by a factor of
 * four — most of it burned standing still at the kerb. Abatement decisions
 * depend entirely on that split, so the modal breakdown is the headline and the
 * flat-factor equivalent is shown next to it for comparison.
 */
import { useEffect, useState } from 'react'
import { Fuel, Gauge, Leaf, TrafficCone, Wind } from 'lucide-react'
import { fetchEmissions, fetchTraffic } from '../api/endpoints'
import {
  Bar, Card, EmptyState, ErrorState, Grid, KeyValue, Loading, Metric,
  Note, PageHeader, Table, fmt,
} from '../components/ui'

const MODE_LABEL = {
  cruise: 'Cruise', stop_go: 'Stop-and-go', idle: 'Idle', compaction: 'Compaction',
}
const MODE_TONE = {
  cruise: 'accent', stop_go: 'warn', idle: 'bad', compaction: 'good',
}

export default function Emissions() {
  const [data, setData] = useState(null)
  const [traffic, setTraffic] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    let alive = true
    Promise.all([fetchEmissions(), fetchTraffic()])
      .then(([e, t]) => {
        if (!alive) return
        setData(e.data)
        setTraffic(t.data)
      })
      .catch(() => alive && setError('Could not load emissions data.'))
      .finally(() => alive && setLoading(false))
    return () => { alive = false }
  }, [])

  if (loading) return <Loading label="Loading emissions…" />
  if (error) return <ErrorState message={error} />

  const hasPlan = data && !data.error
  const scenarios = data?.duty_cycle_comparison || []
  const peakHour = (traffic?.hourly_profile || [])
    .reduce((a, b) => (b.multiplier > (a?.multiplier ?? -1) ? b : a), null)

  return (
    <div>
      <PageHeader
        title="Sustainability"
        description="Fuel and CO₂ from a modal physics model: tractive power against rolling and
                     aerodynamic resistance for the current laden mass, kinetic energy repeatedly
                     bought back in stop-and-go traffic, engine-on idle at the kerb, and
                     hydraulic compaction per lift."
      />

      {hasPlan ? (
        <>
          <Grid min={190} style={{ marginBottom: 18 }}>
            <Metric label="Plan distance" value={fmt.num(data.distance_km, 1)} unit="km"
                    icon={Gauge} />
            <Metric label="Modal CO₂" value={fmt.num(data.modal_co2_kg, 1)} unit="kg"
                    tone="accent" icon={Leaf} />
            <Metric label="Realised intensity" value={fmt.num(data.modal_kg_per_km, 2)}
                    unit="kg/km" icon={Fuel} />
            <Metric label="Flat-factor estimate" value={fmt.num(data.flat_factor_co2_kg, 1)}
                    unit="kg" hint={`at a constant ${data.flat_factor_kg_per_km} kg/km`} />
            <Metric label="Difference" value={fmt.pct(data.difference_pct, 1)}
                    tone={Math.abs(data.difference_pct) > 15 ? 'warn' : 'default'}
                    hint="modal vs flat factor" />
          </Grid>
          <Note tone="accent" >{data.note}</Note>
        </>
      ) : (
        <Card style={{ marginBottom: 18 }}>
          <EmptyState message="Generate a fleet plan to see its measured emissions." />
        </Card>
      )}

      <Grid min={340} style={{ marginTop: 18 }}>
        <Card
          title="Where the carbon goes"
          subtitle="Same vehicle, same model, five duty cycles. The spread is the whole argument
                    against a single emission factor."
        >
          <Table
            columns={[
              { key: 'scenario', header: 'Duty cycle',
                render: (r) => <strong>{r.scenario}</strong> },
              { key: 'co2_kg_per_km', header: 'kg CO₂/km', align: 'right',
                render: (r) => (
                  <span className="ui-num">
                    {r.distance_km > 0 ? fmt.num(r.co2_kg_per_km, 2) : '—'}
                  </span>
                ) },
              { key: 'co2_kg', header: 'kg CO₂', align: 'right',
                render: (r) => <span className="ui-num">{fmt.num(r.co2_kg, 2)}</span> },
              { key: 'split', header: 'Split', width: 190,
                render: (r) => <ModeSplit share={r.breakdown_share} /> },
            ]}
            rows={scenarios.map((s) => ({ ...s, __key: s.scenario }))}
          />
          <Note>
            A stationary truck emits real CO₂ at no distance at all, which is why the
            per-kilometre column is undefined for the last row — and why a per-kilometre factor
            cannot represent collection work.
          </Note>
        </Card>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <Card
            title="Traffic context"
            subtitle="Congestion drives both travel time and fuel: speed enters the emissions
                      model, and saturation raises the kinetic energy bought back per kilometre."
          >
            <KeyValue items={[
              { label: 'Provider', value: traffic?.provider?.provider || '—' },
              { label: 'Speed model', value: 'BPR volume-delay' },
              { label: 'Free-flow speed',
                value: `${fmt.num(traffic?.provider?.bpr?.freeflow_kmh, 0)} km/h` },
              { label: 'Busiest hour today',
                value: peakHour ? `${String(peakHour.hour).padStart(2, '0')}:00 — ${fmt.num(peakHour.speed_kmh, 1)} km/h` : '—' },
              { label: 'Active incidents', value: traffic?.incidents?.length ?? 0 },
            ]} />
            <div style={{ marginTop: 14 }}>
              <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '.05em',
                            color: 'var(--text-muted)', marginBottom: 8, fontWeight: 600 }}>
                Operating speed across the day
              </div>
              <HourlyProfile hours={traffic?.hourly_profile || []} />
            </div>
          </Card>

          {traffic?.incidents?.length > 0 && (
            <Card title="Active incidents" subtitle="Localised, time-limited congestion events.">
              <Table
                dense
                columns={[
                  { key: 'label', header: 'Type',
                    render: (r) => <strong>{r.label}</strong> },
                  { key: 'severity', header: 'Severity', align: 'right',
                    render: (r) => <Bar value={r.severity} tone="warn"
                                        label={fmt.num(r.severity, 2)} /> },
                  { key: 'ends_in_h', header: 'Clears in', align: 'right',
                    render: (r) => fmt.hours(r.ends_in_h) },
                ]}
                rows={traffic.incidents.map((i, k) => ({ ...i, __key: k }))}
              />
            </Card>
          )}

          <Card title="Model" subtitle="Coefficients and their sources.">
            <KeyValue items={[
              { label: 'Formulation', value: data?.model?.model || 'modal physics-based' },
              { label: 'Diesel CO₂', value: `${data?.model?.co2_per_litre_diesel_kg ?? '—'} kg/L` },
              { label: 'Diesel LHV', value: `${data?.model?.diesel_lhv_mj_per_l ?? '—'} MJ/L` },
              { label: 'Kerb mass',
                value: `${fmt.int(data?.model?.profile?.kerb_mass_kg)} kg` },
              { label: 'Idle rate',
                value: `${data?.model?.profile?.idle_fuel_l_per_h ?? '—'} L/h` },
              { label: 'Compaction', value: `${data?.model?.profile?.pto_fuel_l_per_lift ?? '—'} L/lift` },
            ]} />
            {data?.model?.references && (
              <Note>
                {data.model.references.map((r, i) => <div key={i}>{r}</div>)}
              </Note>
            )}
          </Card>
        </div>
      </Grid>
    </div>
  )
}

function ModeSplit({ share }) {
  if (!share) return '—'
  const parts = Object.entries(share).filter(([, v]) => v > 0.001)
  return (
    <div>
      <div style={{ display: 'flex', height: 8, borderRadius: 99, overflow: 'hidden',
                    background: 'rgba(255,255,255,.07)' }}>
        {parts.map(([mode, value]) => (
          <div key={mode}
               title={`${MODE_LABEL[mode]}: ${(value * 100).toFixed(0)}%`}
               style={{
                 width: `${value * 100}%`,
                 background: `var(--${MODE_TONE[mode] === 'accent' ? 'accent'
                   : MODE_TONE[mode] === 'warn' ? 'amber'
                   : MODE_TONE[mode] === 'bad' ? 'red' : 'emerald'})`,
               }} />
        ))}
      </div>
      <div style={{ display: 'flex', gap: 9, marginTop: 5, fontSize: 10,
                    color: 'var(--text-muted)', flexWrap: 'wrap' }}>
        {parts.map(([mode, value]) => (
          <span key={mode}>{MODE_LABEL[mode]} {(value * 100).toFixed(0)}%</span>
        ))}
      </div>
    </div>
  )
}

function HourlyProfile({ hours }) {
  if (!hours.length) return <EmptyState message="No traffic profile available." />
  const max = Math.max(...hours.map((h) => h.speed_kmh), 1)
  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', gap: 2, height: 78 }}>
      {hours.map((h) => {
        const ratio = h.speed_kmh / max
        return (
          <div key={h.hour} style={{ flex: 1, display: 'flex', flexDirection: 'column',
                                     alignItems: 'center', gap: 3 }}
               title={`${String(h.hour).padStart(2, '0')}:00 — ${h.speed_kmh.toFixed(1)} km/h`}>
            <div style={{
              width: '100%',
              height: `${Math.max(4, ratio * 62)}px`,
              borderRadius: '2px 2px 0 0',
              background: ratio < 0.45 ? 'var(--red)' : ratio < 0.75 ? 'var(--amber)' : 'var(--emerald)',
              opacity: 0.85,
            }} />
            {h.hour % 6 === 0 && (
              <span style={{ fontSize: 8.5, color: 'var(--text-muted)' }}>{h.hour}</span>
            )}
          </div>
        )
      })}
    </div>
  )
}
