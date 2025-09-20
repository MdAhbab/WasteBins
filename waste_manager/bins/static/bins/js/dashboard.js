async function pollReadings() {
  try {
    const res = await fetch('/api/readings/');
    if (!res.ok) return;
    const data = await res.json();
    const tbody = document.querySelector('#readings-table tbody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    (data.readings || []).forEach(r => {
      const tr = document.createElement('tr');
      
      // Create status badge
      let statusBadge = 'badge-success">Normal';
      if (r.gas_level > 80) statusBadge = 'badge-danger">Critical';
      else if (r.gas_level > 60) statusBadge = 'badge-warning">High';
      
      // Create progress bar width style
      const progressWidth = Math.min(100, Math.max(0, r.gas_level));
      
      tr.innerHTML = `
        <td class="node-name">${r.node.name}</td>
        <td class="temperature">${parseFloat(r.temperature).toFixed(1)}</td>
        <td class="humidity">${parseFloat(r.humidity).toFixed(1)}</td>
        <td class="gas-level">
          <div class="progress-bar">
            <div class="progress-fill" data-width="${progressWidth}"></div>
          </div>
          <span>${parseFloat(r.gas_level).toFixed(1)}%</span>
        </td>
        <td class="timestamp">${timeAgo(r.timestamp)}</td>
        <td class="status">
          <span class="badge ${statusBadge}</span>
        </td>
      `;
      tbody.appendChild(tr);
      
      // Set progress bar width after DOM insertion
      const progressFill = tr.querySelector('.progress-fill');
      if (progressFill) {
        progressFill.style.width = `${progressWidth}%`;
      }
    });
    
    // Update last updated time
    const lastUpdate = document.getElementById('last-update');
    if (lastUpdate) {
      lastUpdate.textContent = 'just now';
    }
  } catch (error) {
    console.error('Error polling readings:', error);
  }
}

async function refreshNotifs() {
  try {
    const res = await fetch('/api/notifications/');
    if (!res.ok) return;
    const data = await res.json();
    const cnt = (data.notifications || []).filter(n => !n.is_read).length;
    const countEl = document.getElementById('notif-count');
    if (countEl) {
      countEl.innerText = cnt;
      countEl.style.display = cnt > 0 ? 'flex' : 'none';
    }
    
    const contentEl = document.getElementById('notif-content');
    if (contentEl) {
      contentEl.innerHTML = (data.notifications || []).map(n => 
        `<div class="notif ${n.level}">${n.message}</div>`
      ).join('') || '<div class="notif">No notifications</div>';
    }
  } catch (error) {
    console.error('Error refreshing notifications:', error);
  }
}

function toggleNotif() {
  const panel = document.getElementById('notif-panel');
  if (panel) {
    panel.classList.toggle('show');
  }
}

async function computeRoute() {
  try {
    const alphaEl = document.getElementById('alpha');
    const alpha = alphaEl ? parseFloat(alphaEl.value) : 0.5;
    
    const routeCard = document.getElementById('route-card');
    if (routeCard) {
      routeCard.textContent = 'Computing optimal route...';
    }
    
    const res = await fetch('/api/compute-route/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCSRFToken()
      },
      body: JSON.stringify({alpha})
    });
    
    const data = await res.json();
    if (routeCard) {
      routeCard.textContent = JSON.stringify(data, null, 2);
    }
  } catch (error) {
    console.error('Error computing route:', error);
    const routeCard = document.getElementById('route-card');
    if (routeCard) {
      routeCard.textContent = 'Error computing route. Please try again.';
    }
  }
}

function getCSRFToken() {
  const name = 'csrftoken';
  const value = document.cookie.split('; ').find(row => row.startsWith(name+'='));
  return value ? decodeURIComponent(value.split('=')[1]) : '';
}

function timeAgo(timestamp) {
  const now = new Date();
  const time = new Date(timestamp);
  const diffInSeconds = Math.floor((now - time) / 1000);
  
  if (diffInSeconds < 60) return `${diffInSeconds}s ago`;
  if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}m ago`;
  if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}h ago`;
  return `${Math.floor(diffInSeconds / 86400)}d ago`;
}

// Close notification panel when clicking outside
document.addEventListener('click', function(event) {
  const panel = document.getElementById('notif-panel');
  const bell = document.getElementById('notif-bell');
  
  if (panel && bell && !bell.contains(event.target) && !panel.contains(event.target)) {
    panel.classList.remove('show');
  }
});

// Auto-refresh functionality
let refreshInterval;
let autoRefreshEnabled = true;

function startAutoRefresh() {
  if (refreshInterval) clearInterval(refreshInterval);
  refreshInterval = setInterval(() => {
    if (autoRefreshEnabled) {
      pollReadings();
      refreshNotifs();
    }
  }, 8000);
}

function stopAutoRefresh() {
  if (refreshInterval) {
    clearInterval(refreshInterval);
    refreshInterval = null;
  }
}

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
  pollReadings();
  refreshNotifs();
  startAutoRefresh();
  
  // Update server time every second
  const serverTimeEl = document.getElementById('server-time');
  if (serverTimeEl) {
    setInterval(() => {
      const now = new Date();
      serverTimeEl.textContent = now.toLocaleTimeString();
    }, 1000);
  }
});