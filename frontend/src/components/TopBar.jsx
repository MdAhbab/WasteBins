import { Bell } from 'lucide-react'
import { useSidebar } from './Sidebar'
import { useNavigate } from 'react-router-dom'

export default function TopBar({ title, notifCount = 0, user = null }) {
  const { collapsed } = useSidebar()
  const navigate = useNavigate()

  const initials = user
    ? (user.first_name?.[0] || user.username?.[0] || '?').toUpperCase() +
      (user.last_name?.[0] || '').toUpperCase()
    : '?'

  return (
    <header className={`topbar${collapsed ? ' sidebar-collapsed' : ''}`}>
      <h1 className="topbar-title">{title}</h1>

      <div className="topbar-actions">
        {/* Notification Bell */}
        <button
          id="topbar-notif-btn"
          className="icon-btn"
          onClick={() => navigate('/app/notifications')}
          aria-label={`Notifications (${notifCount} unread)`}
        >
          <Bell size={18} />
          {notifCount > 0 && (
            <span className="notif-badge">
              {notifCount > 9 ? '9+' : notifCount}
            </span>
          )}
        </button>

        {/* User Avatar */}
        <div
          className="user-avatar"
          title={user?.username || ''}
          style={{ cursor: 'pointer' }}
          onClick={() => navigate('/app/profile')}
          role="button"
          id="topbar-user-avatar"
        >
          {initials}
        </div>
      </div>
    </header>
  )
}
