// Preview-only demo mode.
// Lets portfolio visitors explore the WasteBins dashboard on the public Vercel
// deployment without a real backend or login. Active ONLY on *.vercel.app, so
// it never affects local development or a real backend-connected deployment.
export const IS_DEMO =
  typeof window !== 'undefined' && /\.vercel\.app$/i.test(window.location.hostname)

export const DEMO_USER = {
  id: 0,
  username: 'demo',
  first_name: 'Demo',
  last_name: 'Viewer',
  email: 'demo@wastebins.app',
  role: 'admin',
}

// Sample data shaped exactly like the /dashboard/ API response. Mirpur, Dhaka.
export const DEMO_DASHBOARD = {
  model_version: 'RandomForest v2.1 (R²≈0.93)',
  stats: { total_bins: 24, critical_bins: 4, warning_bins: 7, normal_bins: 13, avg_fill_pct: 58 },
  priority_info: { top_nodes: [1, 4, 2, 6] },
  readings: [
    { id: 1, waste_level: 0.93, node: { id: 1, name: 'Mirpur-10 Circle', latitude: 23.8069, longitude: 90.3687 } },
    { id: 2, waste_level: 0.74, node: { id: 2, name: 'Kazipara', latitude: 23.8001, longitude: 90.3653 } },
    { id: 3, waste_level: 0.41, node: { id: 3, name: 'Shewrapara', latitude: 23.7931, longitude: 90.3706 } },
    { id: 4, waste_level: 0.88, node: { id: 4, name: 'Mirpur-11', latitude: 23.8190, longitude: 90.3641 } },
    { id: 5, waste_level: 0.29, node: { id: 5, name: 'Pallabi', latitude: 23.8255, longitude: 90.3651 } },
    { id: 6, waste_level: 0.67, node: { id: 6, name: 'Mirpur-2', latitude: 23.8061, longitude: 90.3592 } },
  ],
  latest_route: null,
}
