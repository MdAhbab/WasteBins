import { useEffect, useState } from 'react'
import { User, Mail, Calendar, Shield, Save, Key } from 'lucide-react'
import { fetchProfile, updateProfile } from '../api/endpoints'
import { toast } from '../components/Toast'

export default function Profile({ onUserChange }) {
  const [user, setUser] = useState(null)
  const [form, setForm] = useState({ first_name: '', last_name: '', email: '' })
  const [saving, setSaving] = useState(false)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchProfile()
      .then(res => {
        setUser(res.data)
        setForm({
          first_name: res.data.first_name || '',
          last_name: res.data.last_name || '',
          email: res.data.email || '',
        })
      })
      .catch(() => toast('Failed to load profile', 'error'))
      .finally(() => setLoading(false))
  }, [])

  const handleSave = async (e) => {
    e.preventDefault()
    setSaving(true)
    try {
      const res = await updateProfile(form)
      setUser(res.data)
      toast('Profile updated successfully', 'success')
      if (onUserChange) onUserChange(res.data)
    } catch (err) {
      const msg = err.response?.data?.email?.[0] || 'Failed to update profile'
      toast(msg, 'error')
    } finally {
      setSaving(false)
    }
  }

  const initials = user
    ? (user.first_name?.[0] || user.username?.[0] || '?').toUpperCase() +
      (user.last_name?.[0] || '').toUpperCase()
    : '?'

  if (loading) {
    return (
      <div style={{ display: 'grid', gap: 20 }}>
        {[1, 2].map(i => (
          <div key={i} className="skeleton" style={{ height: 220, borderRadius: 14 }} />
        ))}
      </div>
    )
  }

  return (
    <div style={{ maxWidth: 720 }}>

      {/* Avatar + identity card */}
      <div className="card mb-6" style={{ marginBottom: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
          <div className="profile-avatar">{initials}</div>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 22, fontWeight: 800, letterSpacing: '-0.5px' }}>
              {user?.first_name || user?.username}
              {user?.last_name ? ` ${user.last_name}` : ''}
            </div>
            <div className="text-muted text-sm" style={{ marginTop: 4 }}>@{user?.username}</div>
            <div style={{ display: 'flex', gap: 12, marginTop: 10, flexWrap: 'wrap' }}>
              {user?.is_staff && (
                <span style={{
                  display: 'flex', alignItems: 'center', gap: 5, fontSize: 12, fontWeight: 600,
                  padding: '3px 10px', borderRadius: 99,
                  background: 'var(--accent-dim)', color: 'var(--accent)',
                }}>
                  <Shield size={11} /> Staff
                </span>
              )}
              <span style={{
                display: 'flex', alignItems: 'center', gap: 5, fontSize: 12,
                color: 'var(--text-muted)',
              }}>
                <Calendar size={11} />
                Joined {new Date(user?.date_joined).toLocaleDateString()}
              </span>
              {user?.last_login && (
                <span style={{
                  display: 'flex', alignItems: 'center', gap: 5, fontSize: 12,
                  color: 'var(--text-muted)',
                }}>
                  Last login: {new Date(user.last_login).toLocaleString()}
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Edit form */}
      <div className="card mb-6" style={{ marginBottom: 20 }}>
        <div className="section-header" style={{ marginBottom: 20 }}>
          <h2 className="section-title">
            <User size={16} style={{ display: 'inline', marginRight: 8, verticalAlign: 'middle' }} />
            Edit Profile
          </h2>
        </div>

        <form id="profile-form" onSubmit={handleSave}>
          <div className="form-row" style={{ marginBottom: 16 }}>
            <div className="form-group">
              <label className="form-label" htmlFor="profile-first-name">First Name</label>
              <input
                id="profile-first-name"
                className="form-input"
                value={form.first_name}
                onChange={e => setForm(f => ({ ...f, first_name: e.target.value }))}
                placeholder="Enter first name"
              />
            </div>
            <div className="form-group">
              <label className="form-label" htmlFor="profile-last-name">Last Name</label>
              <input
                id="profile-last-name"
                className="form-input"
                value={form.last_name}
                onChange={e => setForm(f => ({ ...f, last_name: e.target.value }))}
                placeholder="Enter last name"
              />
            </div>
          </div>

          <div className="form-group" style={{ marginBottom: 20 }}>
            <label className="form-label" htmlFor="profile-email">
              <Mail size={13} style={{ display: 'inline', marginRight: 5, verticalAlign: 'middle' }} />
              Email Address
            </label>
            <input
              id="profile-email"
              type="email"
              className="form-input"
              value={form.email}
              onChange={e => setForm(f => ({ ...f, email: e.target.value }))}
              placeholder="Enter email address"
            />
          </div>

          <button
            id="profile-save-btn"
            type="submit"
            className="btn btn-primary"
            disabled={saving}
          >
            <Save size={15} />
            {saving ? 'Saving...' : 'Save Changes'}
          </button>
        </form>
      </div>

      {/* Password note */}
      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div className="stat-icon" style={{ background: 'var(--amber-dim)' }}>
            <Key size={16} color="var(--amber)" />
          </div>
          <div>
            <div style={{ fontSize: 14, fontWeight: 600 }}>Password</div>
            <div className="text-muted text-sm">
              To change your password, use the Django admin panel or the password reset flow.
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
