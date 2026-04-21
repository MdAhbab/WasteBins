import api from './axios'

// Auth
export const login = (data) => api.post('/auth/login/', data)
export const signup = (data) => api.post('/auth/signup/', data)
export const logoutApi = () => api.post('/auth/logout/')

// User
export const fetchMe = () => api.get('/me/')

// Dashboard
export const fetchDashboard = (params) => api.get('/dashboard/', { params })

// Profile
export const fetchProfile = () => api.get('/profile/')
export const updateProfile = (data) => api.put('/profile/', data)

// Settings
export const fetchSettings = () => api.get('/settings/')
export const updateSettings = (data) => api.put('/settings/', data)

// Notifications
export const fetchNotifications = (params) => api.get('/notifications/', { params })
export const markNotificationRead = (id) => api.post(`/notifications/${id}/read/`)
export const markAllNotificationsRead = () => api.post('/notifications/mark-all-read/')
