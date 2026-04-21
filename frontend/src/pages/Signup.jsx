import { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Eye, EyeOff, UserPlus, Leaf, Shield, Activity } from 'lucide-react'
import { signup } from '../api/endpoints'
import { ensureCsrf } from '../api/axios'
import '../auth.css'

const BG = '/ImageAssets/img2.png'

const FEATURES = [
  { icon: Shield,   color: '#10b981', text: 'Secure session-based authentication' },
  { icon: Activity, color: '#3b82f6', text: 'Real-time sensor data from 6 live bins' },
  { icon: Leaf,     color: '#f59e0b', text: 'Reduce waste overflow with AI prioritisation' },
]

export default function Signup() {
  const navigate = useNavigate()
  const [form, setForm] = useState({ username: '', email: '', password: '', password2: '' })
  const [showPw, setShowPw] = useState(false)
  const [loading, setLoading] = useState(false)
  const [errors, setErrors] = useState({})

  useEffect(() => { ensureCsrf() }, [])

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))

  const handleSubmit = async (e) => {
    e.preventDefault()
    setErrors({})
    setLoading(true)
    try {
      await signup(form)
      navigate('/app/dashboard', { replace: true })
    } catch (err) {
      const d = err.response?.data
      if (d && typeof d === 'object') setErrors(d)
      else setErrors({ non_field: 'Signup failed. Please try again.' })
    } finally {
      setLoading(false)
    }
  }

  const FieldErr = ({ f }) => errors[f]
    ? <div className="auth-field-error">{errors[f]}</div>
    : null

  return (
    <div className="auth-page">
      {/* Full-bleed background — different image from login */}
      <div className="auth-bg auth-bg-active" style={{ backgroundImage: `url(${BG})` }} />
      <div className="auth-bg-overlay" />

      <div className="auth-content">
        {/* ── LEFT: Form card (order first) ── */}
        <div className="auth-form-side">
          <div className="auth-card">
            <div className="auth-card-title">Create account</div>
            <div className="auth-card-sub">Join WasteBins in seconds</div>

            {errors.non_field && <div className="auth-card-error">{errors.non_field}</div>}

            <form id="signup-form" onSubmit={handleSubmit}>
              <div className="auth-field">
                <label className="auth-label" htmlFor="su-username">Username</label>
                <input
                  id="su-username"
                  className={`auth-input${errors.username ? ' error' : ''}`}
                  placeholder="Choose a username"
                  autoComplete="username"
                  value={form.username}
                  onChange={e => set('username', e.target.value)}
                  required
                />
                <FieldErr f="username" />
              </div>

              <div className="auth-field">
                <label className="auth-label" htmlFor="su-email">Email Address</label>
                <input
                  id="su-email"
                  type="email"
                  className={`auth-input${errors.email ? ' error' : ''}`}
                  placeholder="Enter your email"
                  autoComplete="email"
                  value={form.email}
                  onChange={e => set('email', e.target.value)}
                  required
                />
                <FieldErr f="email" />
              </div>

              <div className="auth-field">
                <label className="auth-label" htmlFor="su-password">Password</label>
                <div className="auth-pw-wrap">
                  <input
                    id="su-password"
                    type={showPw ? 'text' : 'password'}
                    className={`auth-input${errors.password ? ' error' : ''}`}
                    placeholder="Create a password"
                    autoComplete="new-password"
                    value={form.password}
                    onChange={e => set('password', e.target.value)}
                    required
                  />
                  <button type="button" className="auth-pw-toggle" id="su-pw-toggle"
                    onClick={() => setShowPw(p => !p)}>
                    {showPw ? <EyeOff size={17} /> : <Eye size={17} />}
                  </button>
                </div>
                <FieldErr f="password" />
              </div>

              <div className="auth-field">
                <label className="auth-label" htmlFor="su-password2">Confirm Password</label>
                <input
                  id="su-password2"
                  type={showPw ? 'text' : 'password'}
                  className={`auth-input${errors.password2 ? ' error' : ''}`}
                  placeholder="Repeat your password"
                  autoComplete="new-password"
                  value={form.password2}
                  onChange={e => set('password2', e.target.value)}
                  required
                />
                <FieldErr f="password2" />
              </div>

              <button id="signup-submit" type="submit" className="auth-submit" disabled={loading}>
                <UserPlus size={16} />
                {loading ? 'Creating account…' : 'Create Account'}
              </button>
            </form>

            <div className="auth-switch">
              Already have an account?
              <Link to="/login">Sign in</Link>
            </div>
          </div>
        </div>

        {/* ── RIGHT: Info panel (order second) ── */}
        <div className="auth-info">
          <div className="auth-brand">
            <span className="auth-brand-name">WasteBins</span>
          </div>

          <div className="auth-info-body">
            <div className="auth-info-tag">
              <UserPlus size={11} />
              Join the Platform
            </div>
            <h1 className="auth-info-headline">
              Join the smart<br />
              <span>waste revolution.</span>
            </h1>
            <p className="auth-info-sub">
              Create your account to access live bin monitoring,
              AI-powered priority scoring, and optimised collection routes
              across your city.
            </p>
            <div className="auth-features">
              {FEATURES.map(({ icon: Icon, color, text }) => (
                <div className="auth-feature" key={text}>
                  <div className="auth-feature-dot" style={{ background: color }} />
                  <Icon size={14} color={color} style={{ flexShrink: 0 }} />
                  <span className="auth-feature-text">{text}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="auth-info-footer">
            &copy; {new Date().getFullYear()} WasteBins &middot; Smart Waste Management System
          </div>
        </div>
      </div>
    </div>
  )
}
