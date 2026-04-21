import { useState, createContext, useContext } from 'react'
import { useNavigate, useLocation, NavLink } from 'react-router-dom'
import {
  LayoutDashboard, User, Settings, Bell,
  ChevronLeft, ChevronRight, LogOut
} from 'lucide-react'

const SidebarCtx = createContext({ collapsed: false })

export function useSidebar() { return useContext(SidebarCtx) }

export function SidebarProvider({ children }) {
  const [collapsed, setCollapsed] = useState(false)
  return (
    <SidebarCtx.Provider value={{ collapsed, setCollapsed }}>
      {children}
    </SidebarCtx.Provider>
  )
}

const NAV_ITEMS = [
  { to: '/app/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/app/notifications', label: 'Notifications', icon: Bell },
  { to: '/app/profile', label: 'Profile', icon: User },
  { to: '/app/settings', label: 'Settings', icon: Settings },
]

export default function Sidebar({ notifCount = 0 }) {
  const { collapsed, setCollapsed } = useSidebar()

  const handleLogout = async () => {
    try {
      // Get CSRF and post to Django logout
      const resp = await fetch('/api/csrf/')
      const { csrfToken } = await resp.json()
      await fetch('/logout/', {
        method: 'GET',
        credentials: 'include',
        headers: { 'X-CSRFToken': csrfToken },
      })
    } finally {
      window.location.href = '/login/'
    }
  }

  return (
    <nav className={`sidebar${collapsed ? ' collapsed' : ''}`}>
      {/* Toggle button */}
      <button
        className="sidebar-toggle"
        onClick={() => setCollapsed(!collapsed)}
        aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
      >
        {collapsed
          ? <ChevronRight size={12} />
          : <ChevronLeft size={12} />}
      </button>

      {/* Logo */}
      <div className="sidebar-logo">
        <div className="sidebar-logo-icon">
          W
        </div>
        <span className="sidebar-logo-text">WasteBins</span>
      </div>

      {/* Nav */}
      <div className="sidebar-nav">
        {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
            title={collapsed ? label : undefined}
          >
            <div style={{ position: 'relative' }}>
              <Icon className="nav-icon" />
              {label === 'Notifications' && notifCount > 0 && (
                <span style={{
                  position: 'absolute', top: -6, right: -6,
                  background: 'var(--red)', color: '#fff',
                  borderRadius: '99px', fontSize: 9, fontWeight: 700,
                  padding: '0 3px', minWidth: 14, height: 14,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  {notifCount > 9 ? '9+' : notifCount}
                </span>
              )}
            </div>
            <span className="nav-label">{label}</span>
          </NavLink>
        ))}
      </div>

      {/* Footer */}
      <div className="sidebar-footer">
        <button className="nav-item" onClick={handleLogout} title={collapsed ? 'Logout' : undefined}>
          <LogOut className="nav-icon" />
          <span className="nav-label">Logout</span>
        </button>
      </div>
    </nav>
  )
}
