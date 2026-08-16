/**
 * Google Maps view of the bin network and the planned routes.
 *
 * Two failure modes are handled explicitly rather than left to hang, because
 * both are what a reviewer or a new deployment will hit first:
 *
 *   - No API key configured. `useJsApiLoader` never resolves without one, so a
 *     naive implementation shows "Loading map…" forever with no explanation.
 *     Here the key is checked up front and the panel says exactly which
 *     variable to set.
 *   - The Maps script fails to load (offline, blocked, quota). `loadError` is
 *     surfaced instead of being ignored.
 *
 * Everything else on the page works without a key; only this panel degrades.
 */
import { useMemo } from 'react'
import { GoogleMap, Marker, Polyline, useJsApiLoader } from '@react-google-maps/api'
import { MapPin } from 'lucide-react'

const API_KEY = import.meta.env.VITE_GOOGLE_MAPS_API_KEY || ''

const VEHICLE_COLOURS = ['#3b82f6', '#10b981', '#f59e0b', '#a855f7', '#ec4899']
const DEPOT = { lat: 23.8069, lng: 90.3687 }

function fillColour(fill) {
  if (fill === null || fill === undefined) return '#6b7280'
  if (fill >= 0.85) return '#ff4444'
  if (fill >= 0.65) return '#ffb800'
  return '#10b981'
}

/** Mean position of the bins, so the map opens on the actual network. */
function centreOf(readings) {
  const points = (readings || [])
    .map((r) => r.node)
    .filter((n) => n?.latitude != null && n?.longitude != null)
  if (!points.length) return DEPOT
  return {
    lat: points.reduce((s, n) => s + n.latitude, 0) / points.length,
    lng: points.reduce((s, n) => s + n.longitude, 0) / points.length,
  }
}

export default function BinMap({ readings = [], routes = [], userLat, userLng, height = 380 }) {
  const centre = useMemo(() => centreOf(readings), [readings])
  const nodeById = useMemo(
    () => Object.fromEntries((readings || []).map((r) => [r.node?.id, r.node])),
    [readings],
  )

  const { isLoaded, loadError } = useJsApiLoader({
    id: 'google-map-script',
    googleMapsApiKey: API_KEY,
  })

  if (!API_KEY) {
    return (
      <div className="map-placeholder">
        <MapPin size={22} />
        <div>
          <strong>Map disabled — no Google Maps API key.</strong>
          <div style={{ marginTop: 7 }}>
            Create <code>frontend/.env.local</code> containing{' '}
            <code>VITE_GOOGLE_MAPS_API_KEY=your-browser-key</code> and restart the dev
            server. Every other panel works without it.
          </div>
          <div style={{ marginTop: 9, color: 'var(--text-secondary)' }}>
            {readings.length} bins · {routes.length} planned route{routes.length === 1 ? '' : 's'}
          </div>
        </div>
      </div>
    )
  }

  if (loadError) {
    return (
      <div className="map-placeholder">
        <MapPin size={22} />
        <div>
          <strong>Google Maps failed to load.</strong>
          <div style={{ marginTop: 7 }}>
            Check the API key, its referrer restrictions, and network access.
          </div>
        </div>
      </div>
    )
  }

  if (!isLoaded) {
    return <div className="map-placeholder">Loading map…</div>
  }

  return (
    <div style={{ borderRadius: 10, border: '1px solid var(--border)', overflow: 'hidden' }}>
      <GoogleMap
        mapContainerStyle={{ width: '100%', height: `${height}px` }}
        center={centre}
        zoom={13}
        options={{
          streetViewControl: false,
          mapTypeControl: false,
          fullscreenControl: false,
          // Dark styling so the map does not fight the rest of the interface.
          styles: [
            { elementType: 'geometry', stylers: [{ color: '#0d1526' }] },
            { elementType: 'labels.text.stroke', stylers: [{ color: '#0d1526' }] },
            { elementType: 'labels.text.fill', stylers: [{ color: '#7a8ba6' }] },
            { featureType: 'road', elementType: 'geometry', stylers: [{ color: '#1c2740' }] },
            { featureType: 'water', elementType: 'geometry', stylers: [{ color: '#0a1020' }] },
            { featureType: 'poi', stylers: [{ visibility: 'off' }] },
          ],
        }}
      >
        {/* One polyline per vehicle, each in its own colour. */}
        {routes.map((route, index) => {
          const path = [{ lat: DEPOT.lat, lng: DEPOT.lng }]
          for (const stop of route.stops || []) {
            const node = nodeById[stop.node_id]
            if (node?.latitude != null) path.push({ lat: node.latitude, lng: node.longitude })
          }
          path.push({ lat: DEPOT.lat, lng: DEPOT.lng })
          if (path.length < 3) return null
          return (
            <Polyline
              key={route.vehicle_id ?? index}
              path={path}
              options={{
                strokeColor: VEHICLE_COLOURS[index % VEHICLE_COLOURS.length],
                strokeOpacity: 0.85,
                strokeWeight: 3.5,
              }}
            />
          )
        })}

        {/* Depot */}
        <Marker
          position={DEPOT}
          title="Depot"
          icon={{
            path: window.google.maps.SymbolPath.BACKWARD_CLOSED_ARROW,
            scale: 6,
            fillColor: '#f0f4ff',
            fillOpacity: 1,
            strokeColor: '#0d1526',
            strokeWeight: 2,
          }}
        />

        {/* Operator location, when known */}
        {userLat != null && userLng != null && (
          <Marker
            position={{ lat: userLat, lng: userLng }}
            title="Your location"
            icon={{
              path: window.google.maps.SymbolPath.CIRCLE,
              scale: 6,
              fillColor: '#3b82f6',
              fillOpacity: 1,
              strokeColor: '#ffffff',
              strokeWeight: 2,
            }}
          />
        )}

        {/* Bins, sized and coloured by fill */}
        {(readings || []).map((reading) => {
          const node = reading.node
          if (node?.latitude == null) return null
          const fill = reading.waste_level
          return (
            <Marker
              key={node.id}
              position={{ lat: node.latitude, lng: node.longitude }}
              title={`${node.name} — ${fill == null ? 'no reading' : `${Math.round(fill * 100)}% full`}`}
              icon={{
                path: window.google.maps.SymbolPath.CIRCLE,
                scale: fill == null ? 5 : 5 + fill * 5,
                fillColor: fillColour(fill),
                fillOpacity: 0.9,
                strokeColor: '#0d1526',
                strokeWeight: 1.5,
              }}
            />
          )
        })}
      </GoogleMap>
    </div>
  )
}
