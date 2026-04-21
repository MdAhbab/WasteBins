# Smart Waste Management System

A Django-based intelligent waste management system with AI-powered route optimization using Random Forest machine learning and priority-based Dijkstra's algorithm for efficient waste collection.

**Project Type:** Microprocessors and Microelectronics Lab Prototype  
**Last Updated:** April 21, 2026

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Algorithms & Mathematics](#algorithms--mathematics)
- [Technology Stack](#technology-stack)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)

---

## 🎯 Overview

This system optimizes waste collection routes by combining real-time sensor data, Google Maps tracking, and artificial intelligence. It uses a **Random Forest Regressor** to predict bin priorities and a modified **traffic-aware Dijkstra's algorithm** with dynamic edge weights to compute optimal collection routes.

### Core Innovation

The system implements a **priority-weighted routing algorithm** where high-priority bins appear "closer" in the graph through an inverse weight function:

```
weight(u → v) = base_distance(u, v) * (1 + (traffic_density[v] * 3.0)) / (1 + α × (priority[v] × 10))
```

This ensures the shortest paths favor urgent bins while dynamically routing collectors away from high-traffic congestion.

---

## ✨ Key Features

### 1. Real-Time Sensor Monitoring
- **Temperature** tracking (°C)
- **Humidity** monitoring (%)
- **Gas level** detection (odor/methane, 0.0-1.0)
- **Waste level** measurement (fill percentage, 0.0-1.0)
- **Traffic Density** tracking (0.0-1.0) to model real-world street congestion
- Automatic timestamp updates for each node

### 2. AI-Powered Priority Prediction
- **Random Forest** machine learning model (100 decision trees)
- Predicts bin urgency based on 15 features
- Statistical trend analysis (means, standard deviations)
- Temporal pattern recognition (hour, day of week)
- Model validation: R² score ~0.93

### 3. Smart Traffic-Aware Routing
- **Modified Dijkstra's algorithm** with priority and traffic-density weights.
- Routes favor high-priority bins and penalize congested traffic nodes.
- Virtual source node for user GPS location.
- Live **Google Maps Integration** displaying user location, route polylines, and dynamic bin statuses.
- Real-time simulation centered around **Mirpur, Dhaka** landmarks.

### 4. Multi-Criteria Priority System
Calculates bin urgency using weighted factors:
- **35%** - Waste level (most important)
- **25%** - Gas level
- **25%** - Distance from user
- **10%** - Temperature deviation
- **5%** - Humidity level

### 5. User Management
- Secure authentication system
- User location tracking (GPS coordinates)
- Customizable settings per user
- Role-based access control

### 6. Dashboard & Visualization
- Premium **Sapphire Blue** flat UI (No-Glow aesthetics).
- Live `@react-google-maps/api` map instance showing exact bin locations in Mirpur.
- Priority scores visualization.
- Interactive sidebars and system health monitoring.

---

## 🏗️ System Architecture

### Data Flow

```
┌─────────────────┐
│  IoT Sensors    │
│  (Temperature,  │
│   Humidity,     │
│   Gas, Waste)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Feature Engineering            │
│  • Statistical aggregation      │
│  • Temporal features            │
│  • Distance calculation         │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Priority Calculation           │
│  Option A: Rule-Based (Weights) │
│  Option B: Random Forest AI     │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Top-N Selection (1-5 bins)     │
│  Sort by priority score         │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Graph Construction             │
│  Dynamic edge weight =          │
│  distance * (1 + traffic*3.0)   │
│  / (1 + α × priority × 10)      │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Route Optimization             │
│  Dijkstra + Greedy TSP          │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Optimal Collection Route       │
│  Ordered bin sequence + cost    │
└─────────────────────────────────┘
```

---

## 🧮 Algorithms & Mathematics

### 1. Random Forest Regressor (Machine Learning)

**Purpose:** Predict priority scores for waste bins

**Configuration:**
- 100 decision trees (`n_estimators=100`)
- Maximum depth: 10
- Library: scikit-learn

**Input Features (15 total):**
1. `distance_from_user` - GPS distance in meters
2. `temperature` - Current temperature (°C)
3. `humidity` - Current humidity (%)
4. `gas_level` - Gas/odor level (0.0-1.0)
5. `waste_level` - Fill level (0.0-1.0)
6. `mean_temperature` - Average of last 10 readings
7. `std_temperature` - Standard deviation of last 10 readings
8. `mean_humidity` - Average of last 10 readings
9. `std_humidity` - Standard deviation of last 10 readings
10. `mean_gas` - Average of last 10 readings
11. `std_gas` - Standard deviation of last 10 readings
12. `mean_waste` - Average of last 10 readings
13. `std_waste` - Standard deviation of last 10 readings
14. `hour` - Hour of day (0-23)
15. `day_of_week` - Day of week (0-6)

**Output:** Priority score (0.0-1.0, higher = more urgent)

**Training:**
```bash
python manage.py shell -c "from bins.utils.ai.train_model import train_from_db; train_from_db()"
```

---

### 2. Haversine Distance Formula

**Purpose:** Calculate great-circle distance between GPS coordinates

**Formula:**
```
Given points: (lat1, lon1), (lat2, lon2)

R = 6,371,000 meters (Earth's radius)
φ1 = radians(lat1), φ2 = radians(lat2)
Δφ = radians(lat2 - lat1)
Δλ = radians(lon2 - lon1)

a = sin²(Δφ/2) + cos(φ1) × cos(φ2) × sin²(Δλ/2)
c = 2 × arcsin(√a)
distance = R × c
```

**Implementation:**
```python
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000.0  # meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c
```

---

### 3. Priority Score Calculation (Weighted Multi-Criteria)

**Purpose:** Calculate bin urgency using multiple weighted factors

**Formula:**
```
Priority = w₁×D + w₂×W + w₃×G + w₄×T + w₅×H

Where:
  D = Distance Priority (closer = higher, 0.0-1.0)
  W = Waste Level Priority (fuller = higher, 0.0-1.0)
  G = Gas Level Priority (smellier = higher, 0.0-1.0)
  T = Temperature Priority (deviation from 25°C)
  H = Humidity Priority (>70% = higher)

Weights:
  w₁ = 0.25 (25% - Distance)
  w₂ = 0.35 (35% - Waste Level) ← Most Important
  w₃ = 0.25 (25% - Gas Level)
  w₄ = 0.10 (10% - Temperature)
  w₅ = 0.05 (5%  - Humidity)
```

**Component Calculations:**

**Distance Priority:**
```python
distance_normalized = min(1.0, distance_m / 2000.0)  # Cap at 2km
distance_priority = 1.0 - distance_normalized  # Closer = higher
```

**Waste Level Priority:**
```python
waste_priority = max(0.0, min(1.0, waste_level))  # Direct mapping
```

**Gas Level Priority:**
```python
gas_priority = max(0.0, min(1.0, gas_level))  # Direct mapping
```

**Temperature Priority:**
```python
temp_deviation = abs(temperature - 25.0) / 15.0  # Normalize ±15°C
temp_priority = max(0.0, min(1.0, temp_deviation))
```

**Humidity Priority:**
```python
humidity_priority = max(0.0, min(1.0, (humidity - 50.0) / 50.0))
```

---

### 4. Dijkstra's Shortest Path Algorithm (Modified)

**Purpose:** Find optimal routes considering both distance and priority

**Classical Algorithm:**
```
1. Initialize all distances to infinity except source (0)
2. Use min-heap priority queue
3. While queue not empty:
   a. Extract node u with minimum distance
   b. For each neighbor v of u:
      - Calculate new_distance = distance[u] + weight(u, v)
      - If new_distance < distance[v]:
        * Update distance[v]
        * Update predecessor[v]
        * Add v to priority queue
4. Return distances and predecessors
```

**Complexity:**
- **Time:** O((V + E) log V) with binary heap
- **Space:** O(V)

**Implementation Location:** `waste_manager/bins/utils/dijkstra.py`

---

### 5. Dynamic Edge Weight Calculation (Key Innovation)

**Purpose:** Make high-priority bins appear "closer" in the routing graph

**Formula:**
```
weight(u → v) = base_distance(u, v) * (1 + (traffic_density[v] * 3.0)) / (1 + α × (priority[v] × 10))

Where:
  base_distance = Haversine distance in meters
  traffic_density = Live traffic multiplier (0.0 to 1.0) pushing the collector away from jams.
  priority[v] = Priority score of destination bin (0.0-1.0)
  α = Alpha parameter (default 0.5, configurable)
  × 10 = Normalization factor to amplify effect
```

**Weight Reduction Examples:**

For base_distance = 1000m, α = 0.5:

| Priority | Calculation | Final Weight | Reduction |
|----------|-------------|--------------|-----------|
| 0.0 | 1000 / (1 + 0.5×0×10) | 1000m | 0% |
| 0.3 | 1000 / (1 + 0.5×3) | 400m | 60% |
| 0.5 | 1000 / (1 + 0.5×5) | 286m | 71% |
| 0.7 | 1000 / (1 + 0.5×7) | 222m | 78% |
| 1.0 | 1000 / (1 + 0.5×10) | 167m | **83%** |

**Impact:** High-priority bins receive up to 83% weight reduction, making them 6× more likely to be visited first.

---

### 6. Greedy TSP Approximation (Route Construction)

**Purpose:** Visit multiple bins in optimal order

**Algorithm:**
```
1. Start from user location (virtual source node)
2. Build graph with priority-weighted edges
3. Run Dijkstra from current position
4. Select nearest unvisited node (by weighted distance)
5. Move to that node
6. Repeat steps 3-5 until all nodes visited
7. Return ordered route path
```

**Complexity:** O(N × (V + E) log V) where N = number of bins to visit

**Quality:** Provides 2-approximation for metric TSP in practice

---

### 7. Statistical Feature Engineering

**Purpose:** Extract meaningful patterns from sensor data

**Mean Calculation:**
```python
mean_value = np.mean([reading1, reading2, ..., reading10])
```
- Smooths noise in sensor data
- Captures recent trends

**Standard Deviation:**
```python
std_value = np.std([reading1, reading2, ..., reading10])
```
- Measures variability
- Identifies unstable/erratic bins
- Higher std = potential issues

---

## 🛠️ Technology Stack

### Backend
- **Django 4.2** - Web framework
- **Python 3.13** - Programming language
- **MySQL 8+** - Database

### Machine Learning
- **scikit-learn 1.5.1** - Random Forest model
- **pandas 2.2.2** - Data manipulation
- **numpy 1.26.4** - Numerical computations
- **joblib 1.4.2** - Model persistence

### Frontend
- **React 18** (Vite)
- **Google Maps API** (`@react-google-maps/api`)
- **Lucide Icons**
- **Vanilla CSS** (Premium Sapphire Blue theme)

### Database Schema

**Key Models:**
- `Node` - Waste bin locations with GPS coordinates
- `SensorReading` - Sensor data (temperature, humidity, gas, waste level)
- `CollectionRoute` - Computed optimal routes
- `AICost` - AI model predictions
- `UserSetting` - User preferences and locations
- `BinGroup` - Logical grouping of bins
- `Notification` - System alerts

---

## 🔌 API Endpoints

### Sensor Data
```
POST /api/readings/submit/
Content-Type: application/json

{
  "node_id": 1,
  "temperature": 28.5,
  "humidity": 65.0,
  "gas_level": 0.45,
  "waste_level": 0.78
}

Response: 201 Created
{
  "status": "ok",
  "reading": {...}
}
```

### Get Latest Readings
```
GET /api/readings/?limit=10

Response: 200 OK
{
  "readings": [...]
}
```

### Compute Optimal Route
```
POST /api/compute-route/
Content-Type: application/json

{
  "user_lat": 23.7800,
  "user_lng": 90.3000,
  "top_n": 5,
  "alpha": 0.5,
  "group": "downtown"  // optional
}

Response: 200 OK
{
  "route": {
    "path": [12, 7, 23, 15, 8],
    "edges": [
      {"u": "user_location", "v": 12, "w": 156.3},
      {"u": 12, "v": 7, "w": 243.7}
    ],
    "priority_scores": {12: 0.87, 7: 0.82, ...},
    "algorithm_version": "priority_based_v2"
  },
  "total_cost": 1234.5,
  "selected_nodes": [12, 7, 23, 15, 8]
}
```

### Train AI Model (Admin Only)
```
POST /api/train-model/

Response: 200 OK
{
  "status": "trained",
  "meta": {
    "version": "rf_priority_100_42",
    "n_samples": 45,
    "validation_r2": 0.93,
    "validation_mse": 0.0025
  }
}
```

### AI Predictions
```
POST /api/predict-cost/

Response: 200 OK
{
  "predictions": [
    {
      "node_id": 1,
      "predicted_cost": 0.87,
      "model_version": "rf_priority_100_42"
    }
  ]
}
```

### Update User Location
```
POST /api/update-location/
Content-Type: application/json

{
  "latitude": 23.7800,
  "longitude": 90.3000,
  "location_name": "Office"
}

Response: 200 OK
{
  "success": true,
  "message": "Location updated successfully"
}
```

### Get Notifications
```
GET /api/notifications/

Response: 200 OK
{
  "notifications": [
    {
      "id": 1,
      "message": "High priority bin detected",
      "level": "warning",
      "is_read": false,
      "created_at": "2025-10-14T10:30:00Z"
    }
  ]
}
```

---

## 📁 Project Structure

```
Wastebins/
├── README.md                          # This file
├── SETUP.md                           # Installation guide
├── waste_manager/                     # Django project root
│   ├── manage.py                      # Django CLI
│   ├── requirements.txt               # Python dependencies
│   ├── waste_manager/                 # Project settings
│   │   ├── settings.py               # Main configuration
│   │   ├── urls.py                   # URL routing
│   │   └── wsgi.py                   # WSGI config
│   ├── bins/                         # Main application
│   │   ├── models.py                 # Database models
│   │   ├── views.py                  # Views & API endpoints
│   │   ├── urls.py                   # App URL patterns
│   │   ├── forms.py                  # User forms
│   │   ├── admin.py                  # Admin interface
│   │   ├── serializers.py            # Data serialization
│   │   ├── utils/                    # Algorithm implementations
│   │   │   ├── dijkstra.py          # Modified Dijkstra algorithm
│   │   │   ├── priority_calculator.py # Priority scoring
│   │   │   └── ai/
│   │   │       ├── train_model.py   # Random Forest training
│   │   │       └── model_store.py   # Model persistence
│   │   ├── management/               # Custom Django commands
│   │   │   └── commands/
│   │   │       ├── load_sample_data.py
│   │   │       └── check_system.py
│   │   ├── migrations/               # Database migrations
│   │   ├── templates/                # HTML templates
│   │   │   └── bins/
│   │   │       ├── base.html
│   │   │       ├── auth/            # Login/signup pages
│   │   │       └── bins/            # Dashboard pages
│   │   └── static/                   # CSS/JS assets
│   │       └── bins/
│   │           ├── css/
│   │           └── js/
│   ├── fixtures/                     # Sample data
│   │   ├── sample_nodes.json
│   │   ├── sample_readings.json
│   │   └── sample_ai_costs.json
│   └── model_store/                  # Trained ML models
│       ├── rf_cost_model.joblib
│       └── rf_cost_model_meta.json
└── venv/                             # Virtual environment (not in git)
```

---

## ⚙️ Configuration

### Database Settings (`settings.py`)

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'waste_manager_db',
        'USER': 'root',
        'PASSWORD': 'root',
        'HOST': 'localhost',
        'PORT': '3306',
    }
}
```

### Routing Parameters

```python
# Default alpha for priority influence (0.0-1.0)
ROUTING_ALPHA = 0.5

# Model storage directory
MODEL_STORE_DIR = BASE_DIR / 'model_store'
MODEL_FILENAME = MODEL_STORE_DIR / 'rf_cost_model.joblib'
MODEL_META_FILENAME = MODEL_STORE_DIR / 'rf_cost_model_meta.json'
```

### Priority Weights (Customizable)

Edit `bins/utils/priority_calculator.py`:

```python
def __init__(self, 
             distance_weight: float = 0.25,
             waste_weight: float = 0.35,
             gas_weight: float = 0.25,
             temperature_weight: float = 0.10,
             humidity_weight: float = 0.05,
             max_distance_m: float = 2000.0):
```

---

## 📖 Usage Guide

### 1. Zero-Config Launch (Recommended)

To launch the full system (Backend, Frontend, and Dummy Sensor Data) in a single command, run the setup orchestrator from the project root:

```bash
python run_setup.py
```
This handles dependencies, virtual environments, migrations, and process scaling automatically.

### 2. Access Dashboard
```
http://localhost:5173/login
```
*(The React Vite dev-server runs on port 5173 globally).*

### 2. Submit Sensor Data

**Using Postman:**
```
POST http://localhost:8000/api/readings/submit/
Headers: Content-Type: application/json
Body:
{
  "node_id": 1,
  "temperature": 28.5,
  "humidity": 65.0,
  "gas_level": 0.45,
  "waste_level": 0.78,
  "traffic_density": 0.8
}
```

### 3. Compute Route from Your Location

**Dashboard:** Click "Compute Route" and allow location access

**API:**
```bash
curl -X POST http://localhost:8000/api/compute-route/ \
  -H "Content-Type: application/json" \
  -d '{
    "user_lat": 23.7800,
    "user_lng": 90.3000,
    "top_n": 5,
    "alpha": 0.5
  }'
```

### 4. Train AI Model

```bash
python manage.py shell
>>> from bins.utils.ai.train_model import train_from_db
>>> result = train_from_db()
>>> print(f"Model trained with R² = {result['validation_metrics']['r2']:.3f}")
```

### 5. Launch Mirpur Live Simulation

```bash
python send_dummy_data.py --username admin --password yourpassword
```
*(This starts broadcasting traffic_density and bin levels directly to the local server, shifting coordinates to the Mirpur operational area).*

### 6. System Health Check

```bash
python manage.py check_system
```

Verifies database connectivity, model existence, and data integrity.

---

## 📊 Performance Characteristics

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Haversine Distance | O(1) | Simple trigonometry |
| Priority Calculation | O(1) | Weighted sum |
| Random Forest Prediction | O(T×D×log(N)) | T=100 trees, D=10 depth |
| Dijkstra's Algorithm | O((V+E) log V) | V nodes, E edges |
| Route Optimization | O(N×(V+E) log V) | N iterations |
| Top-N Selection | O(V log V) | Sorting nodes |

### Scalability

- **Small Scale (< 50 bins):** All operations near-instantaneous
- **Medium Scale (50-500 bins):** Route optimization < 1 second
- **Large Scale (500+ bins):** Consider geographical clustering

---

## 🎓 Academic Context

This project demonstrates:

1. **Machine Learning:** Supervised learning with Random Forest
2. **Graph Theory:** Modified shortest path algorithms
3. **Computational Geometry:** Haversine distance on spherical surfaces
4. **Multi-Criteria Optimization:** Weighted decision making
5. **Statistical Analysis:** Trend detection and feature engineering
6. **Real-Time Systems:** IoT sensor data processing
7. **Full-Stack Development:** Django web application

**Suitable for:**
- Microprocessors & Microelectronics lab projects
- IoT system design courses
- Machine learning applications
- Algorithm optimization studies

---

## 📄 License & 👥 Contributors

This project is developed by Ahbab and team for educational and research purposes as part of a Microprocessors and Microelectronics laboratory prototype.

**For setup instructions, see [SETUP.md](SETUP.md)**

