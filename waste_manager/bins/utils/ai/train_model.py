import math
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from django.utils import timezone
from django.db.models import Avg
from bins.models import SensorReading, Node, AICost
from .model_store import save_model

def _haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between two points in meters"""
    R = 6371000.0  # Earth's radius in meters
    phi1, phi2 = math.radians(lat1 or 0.0), math.radians(lat2 or 0.0)
    dphi = math.radians((lat2 or 0.0) - (lat1 or 0.0))
    dlambda = math.radians((lon2 or 0.0) - (lon1 or 0.0))
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def _calculate_priority_score(distance_m, waste_level, gas_level, temperature, humidity):
    """
    Calculate priority score based on requirements:
    - Higher distance → lower priority
    - Higher waste level → higher priority
    - Higher gas level → higher priority
    - Higher temperature and humidity → slightly higher priority
    
    Returns priority score from 0.0 to 1.0 (higher = more urgent)
    """
    # Normalize distance (0-2000m range, inverted for priority)
    distance_norm = max(0.0, min(1.0, distance_m / 2000.0))
    distance_priority = 1.0 - distance_norm  # Closer = higher priority
    
    # Waste level priority (0.0-1.0, direct mapping)
    waste_priority = max(0.0, min(1.0, waste_level))
    
    # Gas level priority (0.0-1.0, direct mapping)
    gas_priority = max(0.0, min(1.0, gas_level))
    
    # Temperature priority (normalized around 25°C, higher deviation = higher priority)
    temp_deviation = abs(temperature - 25.0) / 15.0  # ±15°C range
    temp_priority = max(0.0, min(1.0, temp_deviation))
    
    # Humidity priority (normalized, >70% = higher priority)
    humidity_priority = max(0.0, min(1.0, (humidity - 50.0) / 50.0))
    
    # Combined priority with weights
    priority = (
        0.25 * distance_priority +  # Distance weight: 25%
        0.35 * waste_priority +     # Waste level weight: 35%
        0.25 * gas_priority +       # Gas level weight: 25%
        0.10 * temp_priority +      # Temperature weight: 10%
        0.05 * humidity_priority    # Humidity weight: 5%
    )
    
    return max(0.0, min(1.0, priority))

def _extract_features_df(user_lat=23.7806, user_lng=90.2794):
    """
    Build feature dataset for Random Forest training
    Uses actual priority calculation based on specifications
    """
    readings = SensorReading.objects.select_related('node').order_by('node_id', '-timestamp')
    rows = []
    
    # Group readings by node
    node_groups = {}
    for r in readings:
        node_groups.setdefault(r.node_id, []).append(r)
    
    for node_id, items in node_groups.items():
        node = items[0].node
        latest_reading = items[0]
        
        # Calculate distance from user location
        distance_m = _haversine_distance(
            user_lat, user_lng, 
            node.latitude or 0.0, node.longitude or 0.0
        )
        
        # Calculate priority score as target variable
        priority_score = _calculate_priority_score(
            distance_m=distance_m,
            waste_level=latest_reading.waste_level,
            gas_level=latest_reading.gas_level,
            temperature=latest_reading.temperature,
            humidity=latest_reading.humidity
        )
        
        # Statistical features for better prediction
        temperatures = [r.temperature for r in items[:10]]  # Last 10 readings
        humidities = [r.humidity for r in items[:10]]
        gas_levels = [r.gas_level for r in items[:10]]
        waste_levels = [r.waste_level for r in items[:10]]
        
        # Time-based features
        hour = latest_reading.timestamp.hour
        dow = latest_reading.timestamp.weekday()
        
        row = {
            'node_id': node_id,
            'latitude': node.latitude or 0.0,
            'longitude': node.longitude or 0.0,
            'distance_from_user': distance_m,
            'temperature': latest_reading.temperature,
            'humidity': latest_reading.humidity,
            'gas_level': latest_reading.gas_level,
            'waste_level': latest_reading.waste_level,
            'mean_temperature': np.mean(temperatures),
            'std_temperature': np.std(temperatures) if len(temperatures) > 1 else 0.0,
            'mean_humidity': np.mean(humidities),
            'std_humidity': np.std(humidities) if len(humidities) > 1 else 0.0,
            'mean_gas': np.mean(gas_levels),
            'std_gas': np.std(gas_levels) if len(gas_levels) > 1 else 0.0,
            'mean_waste': np.mean(waste_levels),
            'std_waste': np.std(waste_levels) if len(waste_levels) > 1 else 0.0,
            'hour': hour,
            'day_of_week': dow,
            'priority_score': priority_score  # Target variable
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

def train_from_db(n_estimators=100, random_state=42, test_size=0.2):
    """
    Train Random Forest model for priority prediction
    """
    df = _extract_features_df()
    if df.empty:
        raise ValueError("No sensor data available to train.")
    
    if len(df) < 5:
        raise ValueError("Need at least 5 data points to train the model.")
    
    # Define feature columns (exclude node_id and target)
    feature_cols = [
        'distance_from_user', 'temperature', 'humidity', 'gas_level', 'waste_level',
        'mean_temperature', 'std_temperature', 'mean_humidity', 'std_humidity',
        'mean_gas', 'std_gas', 'mean_waste', 'std_waste', 'hour', 'day_of_week'
    ]
    
    X = df[feature_cols].values
    y = df['priority_score'].values
    
    # Split data for training and validation
    if len(df) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    else:
        # Use all data for training if dataset is small
        X_train, X_test, y_train, y_test = X, X, y, y
    
    # Train Random Forest model
    model = RandomForestRegressor(
        n_estimators=n_estimators, 
        random_state=random_state,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1
    )
    model.fit(X_train, y_train)
    
    # Validate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Create metadata
    meta = {
        'version': f'rf_priority_{n_estimators}_{random_state}',
        'trained_at': timezone.now().isoformat(),
        'features': feature_cols,
        'n_samples': int(df.shape[0]),
        'n_estimators': n_estimators,
        'validation_mse': float(mse),
        'validation_r2': float(r2),
        'target': 'priority_score',
        'feature_importance': dict(zip(feature_cols, model.feature_importances_.tolist()))
    }
    
    # Save model
    save_model(model, meta)
    
    return {
        'meta': meta, 
        'feature_importances': model.feature_importances_.tolist(),
        'validation_metrics': {
            'mse': float(mse),
            'r2': float(r2),
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
    }