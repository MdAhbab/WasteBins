import math
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from django.utils import timezone
from django.db.models import Avg
from bins.models import SensorReading, Node, AICost
from .model_store import save_model

def _extract_features_df():
    # Build a per-node feature table from latest readings and aggregates
    readings = SensorReading.objects.select_related('node').order_by('node_id', '-timestamp')
    rows = []
    # Precompute per-node rolling aggregates (simple last and mean)
    node_groups = {}
    for r in readings:
        node_groups.setdefault(r.node_id, []).append(r)
    for node_id, items in node_groups.items():
        node = items[0].node
        last = items[0]
        mean_temp = np.mean([it.temperature for it in items])
        mean_hum = np.mean([it.humidity for it in items])
        mean_gas = np.mean([it.gas_level for it in items])
        hour = last.timestamp.hour
        dow = last.timestamp.weekday()
        rows.append({
            'node_id': node_id,
            'temperature': last.temperature,
            'humidity': last.humidity,
            'gas_level': last.gas_level,
            'mean_temperature': mean_temp,
            'mean_humidity': mean_hum,
            'mean_gas': mean_gas,
            'hour': hour,
            'dow': dow,
        })
    df = pd.DataFrame(rows)
    return df

def train_from_db(n_estimators=100, random_state=42):
    df = _extract_features_df()
    if df.empty:
        raise ValueError("No sensor data available to train.")
    # Synthetic target: optionally from last AICost if present, else proxy using gas_level
    # In production, target should be historical ground-truth cost/fuel/time labels.
    if AICost.objects.exists():
        latest_costs = (AICost.objects
                        .order_by('node_id', '-timestamp')
                        .distinct('node_id'))
        cost_map = {c.node_id: c.predicted_cost for c in latest_costs}
        df['target_cost'] = df['node_id'].map(cost_map).fillna(df['gas_level'])
    else:
        df['target_cost'] = df['gas_level']
    feature_cols = [c for c in df.columns if c not in ('node_id', 'target_cost')]
    X = df[feature_cols].values
    y = df['target_cost'].values
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X, y)
    meta = {
        'version': f'rf_{n_estimators}_{random_state}',
        'trained_at': timezone.now().isoformat(),
        'features': feature_cols,
        'n_samples': int(df.shape[0]),
    }
    save_model(model, meta)
    return {'meta': meta, 'feature_importances_': getattr(model, 'feature_importances_', []).tolist()}