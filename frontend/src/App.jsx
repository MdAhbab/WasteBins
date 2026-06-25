import { BrowserRouter, Routes, Route, Navigate, useNavigate } from 'react-router-dom'
import { useState, useEffect } from 'react'
import Sidebar, { SidebarProvider, useSidebar } from './components/Sidebar'
import TopBar from './components/TopBar'
import { ToastContainer } from './components/Toast'
import Dashboard from './pages/Dashboard'
import Profile from './pages/Profile'
import Settings from './pages/Settings'
import Notifications from './pages/Notifications'
import Login from './pages/Login'
import Signup from './pages/Signup'
import { fetchMe, fetchNotifications } from './api/endpoints'
import { IS_DEMO, DEMO_USER } from './demo'

const PAGE_TITLES = {
  '/app/dashboard': 'Dashboard',
  '/app/profile': 'Profile',
  '/app/settings': 'Settings',
  '/app/notifications': 'Notifications',
}

function Shell() {
  const { collapsed } = useSidebar()
  const navigate = useNavigate()
  const [user, setUser] = useState(null)
  const [notifCount, setNotifCount] = useState(0)

  useEffect(() => {
    // Preview/demo: skip the real auth check and show a demo viewer so visitors
    // can explore the dashboard without a backend. Production is unaffected.
    if (IS_DEMO) {
      setUser(DEMO_USER)
    } else {
      fetchMe()
        .then(res => setUser(res.data))
        .catch(() => navigate('/login', { replace: true }))
    }

    loadNotifCount()
    const iv = setInterval(loadNotifCount, 30000)

    const onAuthRequired = () => { if (!IS_DEMO) navigate('/login', { replace: true }) }
    window.addEventListener('auth:required', onAuthRequired)

    return () => {
      clearInterval(iv)
      window.removeEventListener('auth:required', onAuthRequired)
    }
  }, [])

  const loadNotifCount = () => {
    fetchNotifications({ unread: 'true' })
      .then(res => setNotifCount(Array.isArray(res.data) ? res.data.length : 0))
      .catch(() => {})
  }

  const handleNotifCountChange = (delta, reset = false) => {
    if (reset) setNotifCount(0)
    else setNotifCount(c => Math.max(0, c + delta))
  }

  const title = PAGE_TITLES[window.location.pathname] || 'WasteBins'

  return (
    <div className="app-shell">
      <Sidebar notifCount={notifCount} />
      <div className={`main-area${collapsed ? ' sidebar-collapsed' : ''}`}>
        <TopBar title={title} notifCount={notifCount} user={user} />
        <main className="page-content">
          <Routes>
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="profile" element={<Profile onUserChange={setUser} />} />
            <Route path="settings" element={<Settings />} />
            <Route path="notifications" element={<Notifications onNotifCountChange={handleNotifCountChange} />} />
            <Route path="*" element={<Navigate to="dashboard" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <SidebarProvider>
        <ToastContainer />
        <Routes>
          {/* Auth pages – handle both /login and /login/ */}
          <Route path="/login" element={<Login />} />
          <Route path="/login/" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/signup/" element={<Signup />} />
          {/* App shell */}
          <Route path="/app/*" element={<Shell />} />
          {/* Root & catch-all → dashboard in demo/preview, otherwise login */}
          <Route path="/" element={<Navigate to={IS_DEMO ? '/app/dashboard' : '/login'} replace />} />
          <Route path="*" element={<Navigate to={IS_DEMO ? '/app/dashboard' : '/login'} replace />} />
        </Routes>
      </SidebarProvider>
    </BrowserRouter>
  )
}
