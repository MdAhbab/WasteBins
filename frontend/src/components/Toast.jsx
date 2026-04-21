import { useEffect, useState } from 'react'
import { CheckCircle, XCircle, Info } from 'lucide-react'

let toastId = 0
const listeners = new Set()

export function toast(message, type = 'info') {
  const id = ++toastId
  listeners.forEach(fn => fn({ id, message, type }))
  setTimeout(() => {
    listeners.forEach(fn => fn({ id, remove: true }))
  }, 3600)
}

export function ToastContainer() {
  const [toasts, setToasts] = useState([])

  useEffect(() => {
    const fn = (evt) => {
      if (evt.remove) {
        setToasts(prev => prev.filter(t => t.id !== evt.id))
      } else {
        setToasts(prev => [...prev, evt])
      }
    }
    listeners.add(fn)
    return () => listeners.delete(fn)
  }, [])

  const Icon = { success: CheckCircle, error: XCircle, info: Info }

  return (
    <div className="toast-container">
      {toasts.map(t => {
        const Ic = Icon[t.type] || Info
        return (
          <div key={t.id} className={`toast toast-${t.type}`}>
            <Ic size={16} />
            {t.message}
          </div>
        )
      })}
    </div>
  )
}
