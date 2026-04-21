import { useEffect, useState } from 'react'
import { Check, CheckCheck, AlertCircle, AlertTriangle, Info, RefreshCw } from 'lucide-react'
import { fetchNotifications, markNotificationRead, markAllNotificationsRead } from '../api/endpoints'
import { toast } from '../components/Toast'

const FILTERS = ['All', 'Unread', 'INFO', 'WARN', 'CRITICAL']

const LEVEL_META = {
  CRITICAL: { color: 'var(--red)', dot: 'notif-dot-critical', label: 'Critical', icon: AlertCircle },
  WARN:     { color: 'var(--amber)', dot: 'notif-dot-warn', label: 'Warning', icon: AlertTriangle },
  INFO:     { color: 'var(--emerald)', dot: 'notif-dot-info', label: 'Info', icon: Info },
}

function NotifItem({ notif, onRead }) {
  const meta = LEVEL_META[notif.level] || LEVEL_META.INFO
  const Icon = meta.icon
  const isUnread = !notif.is_read

  const timeStr = (() => {
    const d = new Date(notif.created_at)
    const diff = Date.now() - d.getTime()
    const mins = Math.floor(diff / 60000)
    if (mins < 1) return 'Just now'
    if (mins < 60) return `${mins}m ago`
    const hrs = Math.floor(mins / 60)
    if (hrs < 24) return `${hrs}h ago`
    return d.toLocaleDateString()
  })()

  return (
    <div className={`notif-item${isUnread ? ' unread' : ''} ${notif.level?.toLowerCase()}`}>
      <span className={`notif-dot ${meta.dot}`} />
      <div className="notif-body">
        <div className="notif-msg">{notif.message}</div>
        <div className="notif-meta">
          <span style={{
            padding: '1px 6px', borderRadius: 99, fontSize: 10, fontWeight: 700,
            background: `${meta.color}20`, color: meta.color,
          }}>
            {meta.label}
          </span>
          <span>{timeStr}</span>
          {isUnread && (
            <span style={{
              padding: '1px 6px', borderRadius: 99, fontSize: 10, fontWeight: 600,
              background: 'var(--accent-dim)', color: 'var(--accent)',
            }}>
              Unread
            </span>
          )}
        </div>
      </div>
      {isUnread && (
        <button
          className="btn btn-ghost btn-sm"
          style={{ flexShrink: 0 }}
          onClick={() => onRead(notif.id)}
          title="Mark as read"
          id={`mark-read-btn-${notif.id}`}
        >
          <Check size={13} />
        </button>
      )}
    </div>
  )
}

export default function Notifications({ onNotifCountChange }) {
  const [notifs, setNotifs] = useState([])
  const [loading, setLoading] = useState(true)
  const [activeFilter, setActiveFilter] = useState('All')
  const [markingAll, setMarkingAll] = useState(false)

  const load = async (filter = activeFilter) => {
    setLoading(true)
    try {
      const params = {}
      if (filter === 'Unread') params.unread = 'true'
      else if (['INFO', 'WARN', 'CRITICAL'].includes(filter)) params.level = filter
      const res = await fetchNotifications(params)
      setNotifs(res.data)
    } catch {
      toast('Failed to load notifications', 'error')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load('All') }, [])

  const handleFilterChange = (f) => {
    setActiveFilter(f)
    load(f)
  }

  const handleRead = async (id) => {
    try {
      await markNotificationRead(id)
      setNotifs(prev => prev.map(n => n.id === id ? { ...n, is_read: true } : n))
      if (onNotifCountChange) onNotifCountChange(-1)
    } catch {
      toast('Failed to mark as read', 'error')
    }
  }

  const handleMarkAll = async () => {
    setMarkingAll(true)
    try {
      await markAllNotificationsRead()
      setNotifs(prev => prev.map(n => ({ ...n, is_read: true })))
      if (onNotifCountChange) onNotifCountChange(0, true)
      toast('All notifications marked as read', 'success')
    } catch {
      toast('Failed to mark all as read', 'error')
    } finally {
      setMarkingAll(false)
    }
  }

  const unreadCount = notifs.filter(n => !n.is_read).length

  return (
    <div style={{ maxWidth: 760 }}>

      {/* Toolbar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20, flexWrap: 'wrap' }}>
        <div className="filter-tabs" style={{ flex: 1 }}>
          {FILTERS.map(f => (
            <button
              key={f}
              id={`notif-filter-${f.toLowerCase()}`}
              className={`filter-tab${activeFilter === f ? ' active' : ''}`}
              onClick={() => handleFilterChange(f)}
            >
              {f}
              {f === 'Unread' && unreadCount > 0 && (
                <span style={{
                  marginLeft: 6, background: 'var(--accent)', color: '#060d1a',
                  borderRadius: 99, padding: '0 5px', fontSize: 10, fontWeight: 700,
                  display: 'inline-flex', alignItems: 'center',
                }}>
                  {unreadCount}
                </span>
              )}
            </button>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button id="notif-refresh-btn" className="btn btn-ghost btn-sm" onClick={() => load(activeFilter)}>
            <RefreshCw size={13} />
            Refresh
          </button>
          {unreadCount > 0 && (
            <button
              id="notif-mark-all-btn"
              className="btn btn-ghost btn-sm"
              onClick={handleMarkAll}
              disabled={markingAll}
            >
              <CheckCheck size={13} />
              {markingAll ? 'Marking...' : 'Mark All Read'}
            </button>
          )}
        </div>
      </div>

      {/* Summary row */}
      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 20,
      }}>
        {[
          { label: 'Critical', count: notifs.filter(n => n.level === 'CRITICAL').length, color: 'var(--red)', icon: AlertCircle },
          { label: 'Warning', count: notifs.filter(n => n.level === 'WARN').length, color: 'var(--amber)', icon: AlertTriangle },
          { label: 'Info', count: notifs.filter(n => n.level === 'INFO').length, color: 'var(--emerald)', icon: Info },
        ].map(({ label, count, color, icon: Icon }) => (
          <div key={label} className="card" style={{ textAlign: 'center', padding: 16 }}>
            <div style={{ fontSize: 22, fontWeight: 800, color }}>{count}</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 5, justifyContent: 'center', marginTop: 4 }}>
              <Icon size={13} color={color} />
              <span className="text-muted text-sm">{label}</span>
            </div>
          </div>
        ))}
      </div>

      {/* List */}
      {loading ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {[1,2,3,4,5].map(i => (
            <div key={i} className="skeleton" style={{ height: 76, borderRadius: 8 }} />
          ))}
        </div>
      ) : notifs.length === 0 ? (
        <div className="card" style={{ textAlign: 'center', padding: 60, color: 'var(--text-muted)' }}>
          <Info size={32} style={{ marginBottom: 12, opacity: 0.4 }} />
          <div>No notifications</div>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {notifs.map(n => (
            <NotifItem key={n.id} notif={n} onRead={handleRead} />
          ))}
        </div>
      )}
    </div>
  )
}
