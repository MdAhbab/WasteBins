import { useState, useCallback } from 'react'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  LineChart, Line, CartesianGrid, Legend,
} from 'recharts'
import { Navigation, Route, Zap, TrendingUp } from 'lucide-react'
import { GoogleMap, useJsApiLoader, Marker, Polyline } from '@react-google-maps/api'
import api from '../api/axios'

// ── Colour helpers ──────────────────────────────────────────────────────────
const scoreColor = (s) =>
  s >= 0.7 ? '#ff4444' : s >= 0.4 ? '#ffb800' : '#00ff88'

const fillColor = (pct) =>
  pct >= 85 ? '#ff4444' : pct >= 65 ? '#ffb800' : '#00ff88'

// ── Google Map of Bins and Route ─────────────────────────────────────────────
const mapContainerStyle = { width: '100%', height: '350px', borderRadius: '10px' }
const defaultCenter = { lat: 23.7330, lng: 90.3965 } // Dhaka Uni bounds

function BinMap({ readings, route, userLat, userLng }) {
  const { isLoaded } = useJsApiLoader({
    id: 'google-map-script',
    googleMapsApiKey: import.meta.env.VITE_GOOGLE_MAPS_API_KEY || ""
  })

  if (!isLoaded) return <div style={{height: 350, display: 'flex', alignItems: 'center', justifyContent: 'center'}}>Loading Map...</div>

  const path = route?.route_data?.path ?? []
  const nodeMap = Object.fromEntries((readings || []).map(r => [r.node?.id, r]))

  const routeCoordinates = []
  
  if (userLat && userLng) {
    routeCoordinates.push({ lat: userLat, lng: userLng })
  }
  
  path.forEach(id => {
    const node = nodeMap[id]?.node
    if (node?.latitude && node?.longitude) {
      routeCoordinates.push({ lat: node.latitude, lng: node.longitude })
    }
  })

  return (
    <div style={{ borderRadius: 10, border: '1px solid var(--border)', overflow: 'hidden' }}>
      <GoogleMap mapContainerStyle={mapContainerStyle} center={defaultCenter} zoom={15}>
        {/* Draw Route Polyline */}
        {routeCoordinates.length > 1 && (
          <Polyline
            path={routeCoordinates}
            options={{ strokeColor: '#00d4ff', strokeOpacity: 0.8, strokeWeight: 4, strokeDasharray: '5,5' }}
          />
        )}
        
        {/* User Location Marker */}
        {userLat && userLng && (
          <Marker position={{ lat: userLat, lng: userLng }} label="User" icon={{ path: 1, scale: 5, strokeColor: 'white' }} />
        )}

        {/* Bin Node Markers */}
        {(readings || []).map((r, i) => {
          const node = r.node
          if (!node?.latitude) return null
          
          let color = '#00ff88' // OK
          if ((r.waste_level * 100) >= 85) color = '#ff4444' // CRIT
          else if ((r.waste_level * 100) >= 65) color = '#ffb800' // WARN

          const routeIdx = path.indexOf(node.id)
          const labelText = routeIdx >= 0 ? `${routeIdx + 1}` : ''

          return (
            <Marker
              key={node.id}
              position={{ lat: node.latitude, lng: node.longitude }}
              label={{ text: labelText, color: '#060d1a', fontWeight: 'bold' }}
              icon={{
                path: 'M 0,0 C -2,-20 -10,-22 -10,-30 A 10,10 0 1,1 10,-30 C 10,-22 2,-20 0,0 z',
                fillColor: color,
                fillOpacity: 1,
                strokeColor: '#000',
                strokeWeight: 2,
                scale: 1.2
              }}
              title={node.name}
            />
          )
        })}
      </GoogleMap>
    </div>
  )
}

// ── Route step cards ────────────────────────────────────────────────────────
function RouteSteps({ route, readings }) {
  if (!route?.route_data?.path?.length) {
    return <div className="text-muted text-sm" style={{ textAlign: 'center', padding: 24 }}>No route computed yet. Run the dummy data sender and trigger a route.</div>
  }
  const path = route.route_data.path
  const nodeMap = Object.fromEntries((readings ?? []).map(r => [r.node?.id, r]))
  const priorities = route.route_data?.priority_scores ?? {}

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {path.map((id, idx) => {
        const r = nodeMap[id]
        const name = r?.node?.name ?? `Node ${id}`
        const wl = r?.waste_level ?? 0
        const score = priorities[id] ?? 0
        const color = fillColor(wl * 100)
        const isLast = idx === path.length - 1
        return (
          <div key={id} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            {/* Step connector */}
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0, flexShrink: 0, width: 28 }}>
              <div style={{
                width: 28, height: 28, borderRadius: '50%',
                background: idx === 0 ? 'var(--accent)' : isLast ? 'var(--emerald)' : `${color}30`,
                border: `2px solid ${idx === 0 ? 'var(--accent)' : isLast ? 'var(--emerald)' : color}`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 11, fontWeight: 700,
                color: idx === 0 ? '#060d1a' : isLast ? '#060d1a' : color,
              }}>{idx + 1}</div>
              {!isLast && <div style={{ width: 2, height: 12, background: 'var(--border)', marginTop: 2 }} />}
            </div>

            {/* Info */}
            <div style={{
              flex: 1, background: 'var(--bg-card)', border: '1px solid var(--border)',
              borderLeft: `3px solid ${color}`, borderRadius: 8, padding: '8px 12px',
              display: 'flex', alignItems: 'center', gap: 10,
            }}>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 13, fontWeight: 600 }}>{name}</div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
                  Fill: {(wl * 100).toFixed(0)}% · Priority: {(score * 100).toFixed(0)}%
                </div>
              </div>
              {/* Mini fill bar */}
              <div style={{ width: 60 }}>
                <div style={{ height: 4, background: 'var(--border)', borderRadius: 99, overflow: 'hidden' }}>
                  <div style={{ width: `${wl * 100}%`, height: '100%', background: color, borderRadius: 99, transition: 'width 0.6s ease' }} />
                </div>
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}

// ── Priority radar chart ────────────────────────────────────────────────────
function PriorityRadar({ readings }) {
  if (!readings?.length) return null
  const data = readings.slice(0, 6).map(r => ({
    name: r.node?.name?.match(/Bin-[A-F]/)?.[0] ?? `B${r.node?.id}`,
    fill: Math.round((r.waste_level ?? 0) * 100),
    gas: Math.round((r.gas_level ?? 0) * 100),
    temp: Math.round(Math.abs((r.temperature ?? 25) - 25) / 15 * 100),
    humidity: Math.round(Math.max(0, ((r.humidity ?? 50) - 50) / 50 * 100)),
  }))

  return (
    <ResponsiveContainer width="100%" height={220}>
      <RadarChart data={data} cx="50%" cy="50%" outerRadius={80}>
        <PolarGrid stroke="rgba(255,255,255,0.08)" />
        <PolarAngleAxis dataKey="name" tick={{ fontSize: 11, fill: 'rgba(240,244,255,0.6)' }} />
        <Radar name="Fill%" dataKey="fill" stroke="var(--accent)" fill="var(--accent)" fillOpacity={0.18} />
        <Radar name="Gas%" dataKey="gas" stroke="var(--amber)" fill="var(--amber)" fillOpacity={0.12} />
        <Tooltip contentStyle={{ background: '#0d1526', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, fontSize: 12 }} />
      </RadarChart>
    </ResponsiveContainer>
  )
}

// ── Fill level bar chart ────────────────────────────────────────────────────
function FillChart({ readings }) {
  if (!readings?.length) return null
  const data = readings.map(r => ({
    name: r.node?.name?.match(/Bin-[A-F]/)?.[0] ?? `B${r.node?.id}`,
    fill: Math.round((r.waste_level ?? 0) * 100),
    gas: Math.round((r.gas_level ?? 0) * 100),
  }))

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} margin={{ top: 4, right: 10, bottom: 0, left: -20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
        <XAxis dataKey="name" tick={{ fontSize: 11, fill: 'rgba(240,244,255,0.55)' }} />
        <YAxis tick={{ fontSize: 11, fill: 'rgba(240,244,255,0.55)' }} domain={[0, 100]} unit="%" />
        <Tooltip contentStyle={{ background: '#0d1526', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, fontSize: 12 }}
          formatter={(v, n) => [`${v}%`, n === 'fill' ? 'Fill Level' : 'Gas Level']} />
        <Bar dataKey="fill" radius={[4, 4, 0, 0]}>
          {data.map((d, i) => <Cell key={i} fill={fillColor(d.fill)} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

// ── Main Route Panel ────────────────────────────────────────────────────────
export default function RoutePanel({ readings, route, userLat, userLng, onRouteComputed }) {
  const [computing, setComputing] = useState(false)
  const [tab, setTab] = useState('steps') // 'steps' | 'map' | 'chart' | 'radar'

  const computeRoute = useCallback(async () => {
    if (!userLat || !userLng) {
      alert('Please set your location in Settings first.')
      return
    }
    setComputing(true)
    try {
      const res = await api.post('/api/v1/../compute-route/', {
        user_lat: userLat, user_lng: userLng, alpha: 0.5, top_n: 6,
      }, { baseURL: '' })
      if (onRouteComputed) onRouteComputed(res.data)
    } catch (e) {
      console.error('Route compute failed:', e)
    } finally {
      setComputing(false)
    }
  }, [userLat, userLng, onRouteComputed])

  const tabs = [
    { id: 'steps', label: 'Route Steps' },
    { id: 'map', label: 'Bin Map' },
    { id: 'chart', label: 'Fill Chart' },
    { id: 'radar', label: 'Sensor Radar' },
  ]

  return (
    <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
      {/* Header */}
      <div style={{
        padding: '16px 20px', borderBottom: '1px solid var(--border)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ background: 'var(--accent-dim)', borderRadius: 8, padding: 7 }}>
            <Navigation size={16} color="var(--accent)" />
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 15 }}>Optimal Collection Route</div>
            {route && (
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
                {route.route_data?.path?.length ?? 0} stops · cost {route.total_cost?.toFixed(1)} ·{' '}
                {new Date(route.timestamp).toLocaleTimeString()}
              </div>
            )}
          </div>
        </div>
        <button
          id="compute-route-btn"
          className="btn btn-primary btn-sm"
          onClick={computeRoute}
          disabled={computing}
        >
          <Zap size={14} />
          {computing ? 'Computing…' : 'Compute Route'}
        </button>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', borderBottom: '1px solid var(--border)', padding: '0 12px' }}>
        {tabs.map(t => (
          <button key={t.id}
            onClick={() => setTab(t.id)}
            style={{
              padding: '10px 14px', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
              color: tab === t.id ? 'var(--accent)' : 'var(--text-muted)',
              background: 'none', border: 'none', cursor: 'pointer',
              borderBottom: tab === t.id ? '2px solid var(--accent)' : '2px solid transparent',
              transition: 'color 0.2s',
            }}>
            {t.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ padding: '18px 20px' }}>
        {tab === 'steps' && <RouteSteps route={route} readings={readings} />}
        {tab === 'map'   && <BinMap readings={readings} route={route} userLat={userLat} userLng={userLng} />}
        {tab === 'chart' && (
          <div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8 }}>Waste fill level per bin</div>
            <FillChart readings={readings} />
          </div>
        )}
        {tab === 'radar' && (
          <div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8 }}>Fill vs Gas sensor comparison</div>
            <PriorityRadar readings={readings} />
          </div>
        )}
      </div>
    </div>
  )
}
