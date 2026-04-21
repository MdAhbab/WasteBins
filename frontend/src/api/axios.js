import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  withCredentials: true,
  headers: { 'Content-Type': 'application/json' },
})

// Automatically attach CSRF token from cookie on mutating requests
api.interceptors.request.use((config) => {
  const method = config.method?.toUpperCase()
  if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(method)) {
    const csrfToken = getCookie('csrftoken')
    if (csrfToken) {
      config.headers['X-CSRFToken'] = csrfToken
    }
  }
  return config
})

function getCookie(name) {
  const value = `; ${document.cookie}`
  const parts = value.split(`; ${name}=`)
  if (parts.length === 2) return parts.pop().split(';').shift()
  return null
}

// On 401/403, let the app handle it via React Router (not a hard redirect)
api.interceptors.response.use(
  (res) => res,
  (err) => {
    // Only redirect if not already on an auth route to avoid redirect loops
    if (
      (err.response?.status === 401 || err.response?.status === 403) &&
      !window.location.pathname.startsWith('/login') &&
      !window.location.pathname.startsWith('/signup')
    ) {
      // Dispatch a custom event so React Router can navigate cleanly
      window.dispatchEvent(new CustomEvent('auth:required'))
    }
    return Promise.reject(err)
  }
)

// Ensure the CSRF cookie is set before the first mutating request
export async function ensureCsrf() {
  await fetch('/api/csrf/', { credentials: 'include' })
}

export default api
