import api from './axios'

// ── Auth ───────────────────────────────────────────────────────────────────
export const login = (data) => api.post('/auth/login/', data)
export const signup = (data) => api.post('/auth/signup/', data)
export const logoutApi = () => api.post('/auth/logout/')
export const fetchMe = () => api.get('/me/')

// ── Dashboard & bins ───────────────────────────────────────────────────────
export const fetchDashboard = (params) => api.get('/dashboard/', { params })
export const fetchNodes = () => api.get('/nodes/')
export const fetchNodeHistory = (id, params) => api.get(`/nodes/${id}/history/`, { params })

// ── Profile & settings ─────────────────────────────────────────────────────
export const fetchProfile = () => api.get('/profile/')
export const updateProfile = (data) => api.put('/profile/', data)
export const fetchSettings = () => api.get('/settings/')
export const updateSettings = (data) => api.put('/settings/', data)

// ── Notifications ──────────────────────────────────────────────────────────
export const fetchNotifications = (params) => api.get('/notifications/', { params })
export const markNotificationRead = (id) => api.post(`/notifications/${id}/read/`)
export const markAllNotificationsRead = () => api.post('/notifications/mark-all-read/')

// ── Fleet dispatch ─────────────────────────────────────────────────────────
export const fetchFleetConfig = () => api.get('/fleet/config/')
export const updateVehicle = (id, data) => api.put(`/fleet/vehicles/${id}/`, data)
export const fetchLatestPlan = (params) => api.get('/fleet/plan/', { params })
export const generatePlan = (data) => api.post('/fleet/plan/', data)
export const listPlans = (params) => api.get('/fleet/plans/', { params })
export const comparePlanners = (data) => api.post('/fleet/compare/', data)
export const fetchServiceEvents = (params) => api.get('/fleet/service/', { params })
export const recordService = (data) => api.post('/fleet/service/', data)
export const fetchEquity = (params) => api.get('/fleet/equity/', { params })

// ── Sensing integrity ──────────────────────────────────────────────────────
export const fetchSensorHealth = (params) => api.get('/sensors/health/', { params })
export const fetchFaultTaxonomy = () => api.get('/sensors/faults/')
export const injectFault = (data) => api.post('/sensors/inject/', data)
export const clearInjectedFaults = (params) => api.delete('/sensors/inject/', { params })

// ── Models & explainability ────────────────────────────────────────────────
export const fetchModelStatus = () => api.get('/models/status/')
export const resetContinualLearner = () => api.post('/models/continual/reset/')
export const explainNode = (id, params) => api.get(`/explain/${id}/`, { params })

// ── Sustainability & context ───────────────────────────────────────────────
export const fetchEmissions = (params) => api.get('/emissions/', { params })
export const fetchTraffic = () => api.get('/traffic/')

// ── Audit ledger ───────────────────────────────────────────────────────────
export const fetchAuditLedger = (params) => api.get('/audit/', { params })
export const verifyLedger = (params) => api.get('/audit/verify/', { params })
