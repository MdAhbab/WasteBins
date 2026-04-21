import { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Eye, EyeOff, LogIn, Zap, MapPin, BarChart3, Cpu } from 'lucide-react'
import { login } from '../api/endpoints'
import { ensureCsrf } from '../api/axios'
import '../auth.css'

const BG = '/ImageAssets/img1.jpg'

const FEATURES = [
  { icon: BarChart3, color: '#3b82f6', text: 'Live fill-level monitoring across all 6 bins' },
  { icon: Cpu,       color: '#10b981', text: 'Random Forest AI predicts priority in real time' },
  { icon: MapPin,    color: '#6366f1', text: 'Dijkstra routing optimises every collection run' },
  { icon: Zap,       color: '#f59e0b', text: 'Gas, temp & humidity sensors — all in one view' },
]

export default function Login() {
  const navigate = useNavigate()
  const [form, setForm]     = useState({ username: '', password: '' })
  const [showPw, setShowPw] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError]   = useState('')

  useEffect(() => { ensureCsrf() }, [])

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    if (!form.username.trim() || !form.password) {
      setError('Please enter your username and password.')
      return
    }
    setLoading(true)
    try {
      await login({ username: form.username.trim(), password: form.password })
      navigate('/app/dashboard', { replace: true })
    } catch (err) {
      setError(err.response?.data?.error || 'Invalid credentials. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-page">
      {/* Full-bleed background */}
      <div className="auth-bg auth-bg-active" style={{ backgroundImage: `url(${BG})` }} />
      <div className="auth-bg-overlay" />

      <div className="auth-content">
        {/* ── LEFT: Form card (order first) ── */}
        <div className="auth-form-side">
          <div className="auth-card">
            <div className="auth-card-title">Welcome back</div>
            <div className="auth-card-sub">Sign in to your WasteBins account</div>

            {error && <div className="auth-card-error">{error}</div>}

            <form id="login-form" onSubmit={handleSubmit}>
              <div className="auth-field">
                <label className="auth-label" htmlFor="login-username">Username or Email</label>
                <input
                  id="login-username"
                  className="auth-input"
                  placeholder="Enter username or email"
                  autoComplete="username"
                  value={form.username}
                  onChange={e => set('username', e.target.value)}
                  required
                />
              </div>

              <div className="auth-field">
                <label className="auth-label" htmlFor="login-password">Password</label>
                <div className="auth-pw-wrap">
                  <input
                    id="login-password"
                    type={showPw ? 'text' : 'password'}
                    className="auth-input"
                    placeholder="Enter your password"
                    autoComplete="current-password"
                    value={form.password}
                    onChange={e => set('password', e.target.value)}
                    required
                  />
                  <button
                    type="button"
                    className="auth-pw-toggle"
                    id="login-pw-toggle"
                    onClick={() => setShowPw(p => !p)}
                    aria-label={showPw ? 'Hide password' : 'Show password'}
                  >
                    {showPw ? <EyeOff size={17} /> : <Eye size={17} />}
                  </button>
                </div>
              </div>

              <button
                id="login-submit"
                type="submit"
                className="auth-submit"
                disabled={loading}
              >
                <LogIn size={16} />
                {loading ? 'Signing in…' : 'Sign In'}
              </button>
            </form>

            <div className="auth-switch">
              Don't have an account?
              <Link to="/signup">Create one</Link>
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
              <Zap size={11} />
              Smart Waste Platform
            </div>
            <h1 className="auth-info-headline">
              Cleaner routes.<br />
              <span>Smarter cities.</span>
            </h1>
            <p className="auth-info-sub">
              Monitor waste bins in real time, predict collection urgency with AI,
              and dispatch optimised routes — all from a single dashboard.
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
