import { useEffect, useState } from 'react'
import { Bell, MapPin, Clock, Save, Navigation, Wifi } from 'lucide-react'
import { fetchSettings, updateSettings } from '../api/endpoints'
import { toast } from '../components/Toast'

function Toggle({ id, checked, onChange, label, desc }) {
  return (
    <div className="toggle-row">
      <div className="toggle-info">
        <span className="toggle-name">{label}</span>
        {desc && <span className="toggle-desc">{desc}</span>}
      </div>
      <label className="toggle" htmlFor={id}>
        <input id={id} type="checkbox" checked={checked} onChange={onChange} />
        <span className="toggle-track" />
        <span className="toggle-thumb" />
      </label>
    </div>
  )
}

export default function Settings() {
  const [settings, setSettings] = useState(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [detecting, setDetecting] = useState(false)

  const set = (key, val) => setSettings(s => ({ ...s, [key]: val }))

  useEffect(() => {
    fetchSettings()
      .then(res => setSettings(res.data))
      .catch(() => toast('Failed to load settings', 'error'))
      .finally(() => setLoading(false))
  }, [])

  const handleSave = async (e) => {
    e.preventDefault()
    setSaving(true)
    try {
      await updateSettings({
        notify_email: settings.notify_email,
        polling_interval_sec: settings.polling_interval_sec,
        latitude: settings.latitude || null,
        longitude: settings.longitude || null,
        location_name: settings.location_name || '',
        auto_update_location: settings.auto_update_location,
      })
      toast('Settings saved', 'success')
    } catch (err) {
      const data = err.response?.data
      const msg = typeof data === 'object'
        ? Object.values(data).flat()[0] || 'Failed to save settings'
        : 'Failed to save settings'
      toast(msg, 'error')
    } finally {
      setSaving(false)
    }
  }

  const detectLocation = () => {
    if (!navigator.geolocation) {
      toast('Geolocation not supported by your browser', 'error')
      return
    }
    setDetecting(true)
    navigator.geolocation.getCurrentPosition(
      pos => {
        set('latitude', pos.coords.latitude)
        set('longitude', pos.coords.longitude)
        toast('Location detected', 'success')
        setDetecting(false)
      },
      () => {
        toast('Could not detect location', 'error')
        setDetecting(false)
      }
    )
  }

  if (loading || !settings) {
    return (
      <div style={{ maxWidth: 680, display: 'grid', gap: 20 }}>
        {[1, 2, 3].map(i => (
          <div key={i} className="skeleton" style={{ height: 160, borderRadius: 14 }} />
        ))}
      </div>
    )
  }

  return (
    <div style={{ maxWidth: 680 }}>
      <form id="settings-form" onSubmit={handleSave} style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

        {/* Notifications */}
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <Bell size={16} color="var(--accent)" />
            <h2 className="section-title">Notifications</h2>
          </div>
          <p className="text-muted text-sm" style={{ marginBottom: 16 }}>
            Configure how you receive alerts about waste bins.
          </p>

          <Toggle
            id="notify-email-toggle"
            checked={!!settings.notify_email}
            onChange={e => set('notify_email', e.target.checked)}
            label="Email Notifications"
            desc="Receive critical alerts via email"
          />
          <Toggle
            id="auto-location-toggle"
            checked={!!settings.auto_update_location}
            onChange={e => set('auto_update_location', e.target.checked)}
            label="Auto-Update Location"
            desc="Automatically use GPS for route calculation"
          />
        </div>

        {/* Polling interval */}
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <Wifi size={16} color="var(--accent)" />
            <h2 className="section-title">Data Polling</h2>
          </div>
          <p className="text-muted text-sm" style={{ marginBottom: 20 }}>
            How often the dashboard checks for new sensor readings.
          </p>

          <div className="form-group">
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
              <label className="form-label">
                <Clock size={13} style={{ display: 'inline', marginRight: 5, verticalAlign: 'middle' }} />
                Polling Interval
              </label>
              <span style={{ fontSize: 14, fontWeight: 700, color: 'var(--accent)' }}>
                {settings.polling_interval_sec}s
              </span>
            </div>
            <input
              id="polling-slider"
              type="range"
              className="slider"
              min={5}
              max={300}
              step={5}
              value={settings.polling_interval_sec}
              onChange={e => set('polling_interval_sec', parseInt(e.target.value))}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6 }}>
              <span className="text-muted text-xs">5s (fast)</span>
              <span className="text-muted text-xs">300s (5 min)</span>
            </div>
          </div>
        </div>

        {/* Location */}
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <MapPin size={16} color="var(--accent)" />
            <h2 className="section-title">Your Location</h2>
          </div>
          <p className="text-muted text-sm" style={{ marginBottom: 20 }}>
            Used to calculate optimal waste collection routes from your position.
          </p>

          <div className="form-group" style={{ marginBottom: 14 }}>
            <label className="form-label" htmlFor="location-name-input">Location Name</label>
            <input
              id="location-name-input"
              className="form-input"
              value={settings.location_name || ''}
              onChange={e => set('location_name', e.target.value)}
              placeholder="e.g. Dhaka University, Dhaka"
            />
          </div>

          <div className="form-row" style={{ marginBottom: 16 }}>
            <div className="form-group">
              <label className="form-label" htmlFor="latitude-input">Latitude</label>
              <input
                id="latitude-input"
                type="number"
                step="any"
                className="form-input"
                value={settings.latitude ?? ''}
                onChange={e => set('latitude', e.target.value ? parseFloat(e.target.value) : null)}
                placeholder="e.g. 23.7806"
              />
            </div>
            <div className="form-group">
              <label className="form-label" htmlFor="longitude-input">Longitude</label>
              <input
                id="longitude-input"
                type="number"
                step="any"
                className="form-input"
                value={settings.longitude ?? ''}
                onChange={e => set('longitude', e.target.value ? parseFloat(e.target.value) : null)}
                placeholder="e.g. 90.2794"
              />
            </div>
          </div>

          <button
            id="detect-location-btn"
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={detectLocation}
            disabled={detecting}
          >
            <Navigation size={14} />
            {detecting ? 'Detecting...' : 'Detect My Location'}
          </button>

          {settings.latitude && settings.longitude && (
            <div style={{
              marginTop: 14, padding: '10px 14px', borderRadius: 8,
              background: 'var(--emerald-dim)', border: '1px solid var(--emerald)',
              fontSize: 12, color: 'var(--emerald)', display: 'flex', gap: 8, alignItems: 'center',
            }}>
              <MapPin size={12} />
              Location set: {parseFloat(settings.latitude).toFixed(4)}, {parseFloat(settings.longitude).toFixed(4)}
            </div>
          )}
        </div>

        {/* Save */}
        <button
          id="settings-save-btn"
          type="submit"
          className="btn btn-primary"
          disabled={saving}
          style={{ alignSelf: 'flex-start' }}
        >
          <Save size={15} />
          {saving ? 'Saving...' : 'Save Settings'}
        </button>
      </form>
    </div>
  )
}
