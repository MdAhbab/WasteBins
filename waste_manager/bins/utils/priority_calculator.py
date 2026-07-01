"""
Priority calculation system for smart waste bin routing
Implements the specified priority rules for dynamic node prioritization
"""
import math
from typing import Dict, List, Tuple, Optional
from django.db.models import QuerySet
from django.conf import settings
from bins.models import Node, SensorReading
from .ai.model_store import load_model, load_forward_bundle

# Default features in case settings is missing them
DEFAULT_DYNAMIC_FEATURES = {
    'distance_m': {'type': 'priority', 'weight': 0.25, 'min_val': 0.0, 'max_val': 2000.0, 'impact': 'negative'},
    'waste_level': {'type': 'priority', 'weight': 0.35, 'min_val': 0.0, 'max_val': 1.0, 'impact': 'positive'},
    'gas_level': {'type': 'priority', 'weight': 0.25, 'min_val': 0.0, 'max_val': 1.0, 'impact': 'positive'},
    'temperature': {'type': 'priority', 'weight': 0.10, 'min_val': 10.0, 'max_val': 40.0, 'optimal': 25.0, 'impact': 'deviation'},
    'humidity': {'type': 'priority', 'weight': 0.05, 'min_val': 50.0, 'max_val': 100.0, 'impact': 'positive'},
    'traffic_density': {'type': 'cost_multiplier', 'weight': 0.10, 'min_val': 0.0, 'max_val': 1.0, 'impact': 'negative'}
}

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance between two points in meters"""
    R = 6371000.0  # Earth's radius in meters
    phi1, phi2 = math.radians(lat1 or 0.0), math.radians(lat2 or 0.0)
    dphi = math.radians((lat2 or 0.0) - (lat1 or 0.0))
    dlambda = math.radians((lon2 or 0.0) - (lon1 or 0.0))
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

class PriorityCalculator:
    """
    Calculates node priorities based on the specified rules:
    - Higher distance → lower priority
    - Higher waste level → higher priority  
    - Higher gas level → higher priority
    - Higher temperature and humidity → slightly higher priority
    """
    
    def __init__(self, 
                 distance_weight: float = 0.25,
                 waste_weight: float = 0.35,
                 gas_weight: float = 0.25,
                 temperature_weight: float = 0.10,
                 humidity_weight: float = 0.05,
                 max_distance_m: float = 2000.0):
        """
        Initialize priority calculator with weights
        
        Args:
            distance_weight: Weight for distance component (default 25%)
            waste_weight: Weight for waste level component (default 35%)
            gas_weight: Weight for gas level component (default 25%)
            temperature_weight: Weight for temperature component (default 10%)
            humidity_weight: Weight for humidity component (default 5%)
            max_distance_m: Maximum distance for normalization (default 2000m)
        """
        self.distance_weight = distance_weight
        self.waste_weight = waste_weight
        self.gas_weight = gas_weight
        self.temperature_weight = temperature_weight
        self.humidity_weight = humidity_weight
        self.max_distance_m = max_distance_m
    
    def calculate_single_priority(self, **kwargs) -> float:
        """
        Calculate priority score dynamically based on available features and settings.
        If a feature is missing from kwargs, it skips it and normalizes the weights
        so the final score remains balanced between 0.0 and 1.0.
        """
        feature_config = getattr(settings, 'DYNAMIC_FEATURES', DEFAULT_DYNAMIC_FEATURES)
        
        active_weight_sum = 0.0
        priority_score = 0.0
        
        for feature, config in feature_config.items():
            # Only process features that are present in our input and are meant to affect priority
            if feature not in kwargs or kwargs[feature] is None:
                continue
                
            val = float(kwargs[feature])
            weight = config.get('weight', 0.0)
            
            # Allow cost_multipliers to also have a priority weight if requested
            if weight <= 0.0:
                continue
                
            min_v = config.get('min_val', 0.0)
            max_v = config.get('max_val', 1.0)
            impact = config.get('impact', 'positive')
            
            # Normalization Algebra
            if impact == 'deviation' and 'optimal' in config:
                opt = config['optimal']
                # Max possible deviation is max(abs(max_v - opt), abs(min_v - opt))
                max_dev = max(abs(max_v - opt), abs(min_v - opt))
                norm = abs(val - opt) / max_dev if max_dev > 0 else 0.0
            else:
                norm = (val - min_v) / (max_v - min_v) if max_v > min_v else 0.0
                
            # Bound [0, 1]
            norm = max(0.0, min(1.0, norm))
            
            # Invert for negative impact (e.g. higher distance = lower priority)
            if impact == 'negative':
                norm = 1.0 - norm
                
            priority_score += norm * weight
            active_weight_sum += weight
            
        # Rescale based on active features
        if active_weight_sum > 0:
            final_priority = priority_score / active_weight_sum
        else:
            final_priority = 0.0
            
        return max(0.0, min(1.0, final_priority))
    
    def calculate_node_priorities(self, 
                                nodes: List[Node],
                                user_lat: float,
                                user_lng: float,
                                use_ai_model: bool = True) -> Dict[int, float]:
        """
        Calculate priorities for all nodes
        
        Args:
            nodes: List of Node objects
            user_lat: User latitude
            user_lng: User longitude
            use_ai_model: Whether to use trained AI model for prediction
        
        Returns:
            Tuple of (priorities_dict, traffic_scores_dict)
        """
        priorities = {}
        traffic_scores = {}

        # Get latest readings for all nodes
        latest_readings = self._get_latest_readings(nodes)

        # Prefer the forward-looking (non-circular) bundle when available; it
        # predicts hazard/time-to-overflow from the recent history rather than
        # re-deriving a formula from the current reading.
        forward_bundle = load_forward_bundle() if use_ai_model else None
        ai_model = load_model() if (use_ai_model and forward_bundle is None) else None

        for node in nodes:
            reading = latest_readings.get(node.id)
            if not reading:
                priorities[node.id] = 0.0
                continue

            # Calculate distance from user
            distance_m = haversine_distance(
                user_lat, user_lng,
                node.latitude or 0.0, node.longitude or 0.0
            )

            if forward_bundle is not None:
                priority = self._predict_priority_forward(forward_bundle, node)
            elif ai_model:
                # Use AI model for prediction
                priority = self._predict_priority_with_ai(
                    ai_model, node, reading, distance_m
                )
            else:
                # Use rule-based calculation with dynamic kwargs extraction
                kwargs = {
                    'distance_m': distance_m,
                    'waste_level': reading.waste_level,
                    'gas_level': reading.gas_level,
                    'temperature': reading.temperature,
                    'humidity': reading.humidity
                }
                if hasattr(reading, 'traffic_density') and reading.traffic_density is not None:
                    kwargs['traffic_density'] = reading.traffic_density
                
                priority = self.calculate_single_priority(**kwargs)
            
            priorities[node.id] = priority
            traffic_scores[node.id] = getattr(reading, 'traffic_density', 0.0)
        
        return priorities, traffic_scores
    
    def _get_latest_readings(self, nodes: List[Node]) -> Dict[int, SensorReading]:
        """Get the latest sensor reading for each node"""
        node_ids = [n.id for n in nodes]
        readings = SensorReading.objects.filter(
            node_id__in=node_ids
        ).order_by('node_id', '-timestamp')
        
        latest_by_node = {}
        for reading in readings:
            if reading.node_id not in latest_by_node:
                latest_by_node[reading.node_id] = reading
        
        return latest_by_node
    
    def _predict_priority_with_ai(self, 
                                model,
                                node: Node,
                                reading: SensorReading,
                                distance_m: float) -> float:
        """
        Use trained Random Forest model to predict priority
        
        Args:
            model: Trained sklearn model
            node: Node object
            reading: Latest sensor reading
            distance_m: Distance from user in meters
        
        Returns:
            Predicted priority score (0.0-1.0)
        """
        try:
            # Prepare features matching training data format
            traffic = getattr(reading, 'traffic_density', 0.0)
            features = [
                distance_m,                    # distance_from_user
                reading.temperature,           # temperature
                reading.humidity,              # humidity
                reading.gas_level,             # gas_level
                reading.waste_level,           # waste_level
                traffic,                       # traffic_density
                reading.temperature,           # mean_temperature (simplified)
                0.0,                          # std_temperature (simplified)
                reading.humidity,              # mean_humidity (simplified)
                0.0,                          # std_humidity (simplified)
                reading.gas_level,             # mean_gas (simplified)
                0.0,                          # std_gas (simplified)
                reading.waste_level,           # mean_waste (simplified)
                0.0,                          # std_waste (simplified)
                traffic,                       # mean_traffic (simplified)
                0.0,                          # std_traffic (simplified)
                reading.timestamp.hour,        # hour
                reading.timestamp.weekday()    # day_of_week
            ]
            
            # Predict priority
            prediction = model.predict([features])[0]
            
            # Ensure output is in valid range
            return max(0.0, min(1.0, float(prediction)))
            
        except Exception as e:
            # Fallback to rule-based calculation if AI prediction fails
            kwargs = {
                'distance_m': distance_m,
                'waste_level': reading.waste_level,
                'gas_level': reading.gas_level,
                'temperature': reading.temperature,
                'humidity': reading.humidity
            }
            if hasattr(reading, 'traffic_density') and reading.traffic_density is not None:
                kwargs['traffic_density'] = reading.traffic_density
            return self.calculate_single_priority(**kwargs)
    
    def _predict_priority_forward(self, bundle, node: Node) -> float:
        """
        Risk-aware priority from the forward-looking bundle, using the node's
        recent reading history to build the engineered feature row.
        Falls back to the renormalised rule if anything is missing.
        """
        try:
            from .ai.train_forward import FEATURE_COLS, ROLL, predict_forward
            import numpy as np
            recent = list(
                SensorReading.objects.filter(node_id=node.id)
                .order_by('-timestamp')[:ROLL]
            )[::-1]
            if not recent:
                raise ValueError("no readings")
            latest = recent[-1]

            def series(attr):
                return [float(getattr(r, attr)) for r in recent]

            row = {}
            for src, name in [('waste_level', 'waste'), ('gas_level', 'gas'),
                              ('temperature', 'temp'), ('humidity', 'humidity')]:
                vals = series(src)
                row[name] = vals[-1]
                row[f'mean_{name}'] = float(np.mean(vals))
                row[f'std_{name}'] = float(np.std(vals)) if len(vals) > 1 else 0.0
                row[f'trend_{name}'] = (vals[-1] - vals[0]) / ROLL if len(vals) > 1 else 0.0
            row['hour'] = latest.timestamp.hour
            row['dow'] = latest.timestamp.weekday()
            row = {c: row.get(c, 0.0) for c in FEATURE_COLS}
            return predict_forward(bundle, row)['risk_priority']
        except Exception:
            reading = SensorReading.objects.filter(node_id=node.id).order_by('-timestamp').first()
            if not reading:
                return 0.0
            kwargs = {'waste_level': reading.waste_level, 'gas_level': reading.gas_level,
                      'temperature': reading.temperature, 'humidity': reading.humidity}
            return self.calculate_single_priority(**kwargs)

    def select_top_priority_nodes(self,
                                nodes: List[Node],
                                user_lat: float,
                                user_lng: float,
                                max_nodes: int = 5) -> List[Tuple[Node, float]]:
        """
        Select top priority nodes based on distance and other factors
        
        Args:
            nodes: List of all available nodes
            user_lat: User latitude
            user_lng: User longitude
            max_nodes: Maximum number of nodes to select (1-5)
        
        Returns:
            List of (Node, priority_score) tuples sorted by priority (highest first)
        """
        # Calculate priorities for all nodes
        priorities, _ = self.calculate_node_priorities(nodes, user_lat, user_lng)
        
        # Sort nodes by priority (highest first)
        node_priorities = [
            (node, priorities.get(node.id, 0.0)) 
            for node in nodes
        ]
        node_priorities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N nodes
        return node_priorities[:max_nodes]

# Global instance for easy access
priority_calculator = PriorityCalculator()