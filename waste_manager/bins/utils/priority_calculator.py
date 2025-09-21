"""
Priority calculation system for smart waste bin routing
Implements the specified priority rules for dynamic node prioritization
"""
import math
from typing import Dict, List, Tuple, Optional
from django.db.models import QuerySet
from bins.models import Node, SensorReading
from .ai.model_store import load_model

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
    
    def calculate_single_priority(self, 
                                distance_m: float,
                                waste_level: float,
                                gas_level: float,
                                temperature: float,
                                humidity: float) -> float:
        """
        Calculate priority score for a single node
        
        Args:
            distance_m: Distance from user location in meters
            waste_level: Waste level (0.0-1.0)
            gas_level: Gas level (0.0-1.0)
            temperature: Temperature in Celsius
            humidity: Humidity percentage (0-100)
        
        Returns:
            Priority score from 0.0 to 1.0 (higher = more urgent)
        """
        # Distance priority (closer = higher priority)
        distance_norm = min(1.0, max(0.0, distance_m / self.max_distance_m))
        distance_priority = 1.0 - distance_norm
        
        # Waste level priority (higher = higher priority)
        waste_priority = max(0.0, min(1.0, waste_level))
        
        # Gas level priority (higher = higher priority) 
        gas_priority = max(0.0, min(1.0, gas_level))
        
        # Temperature priority (deviation from 25°C = higher priority)
        temp_deviation = abs(temperature - 25.0) / 15.0  # ±15°C range
        temp_priority = max(0.0, min(1.0, temp_deviation))
        
        # Humidity priority (>70% = higher priority)
        humidity_priority = max(0.0, min(1.0, (humidity - 50.0) / 50.0))
        
        # Calculate weighted priority score
        priority = (
            self.distance_weight * distance_priority +
            self.waste_weight * waste_priority +
            self.gas_weight * gas_priority +
            self.temperature_weight * temp_priority +
            self.humidity_weight * humidity_priority
        )
        
        return max(0.0, min(1.0, priority))
    
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
            Dict mapping node_id to priority score (0.0-1.0)
        """
        priorities = {}
        
        # Get latest readings for all nodes
        latest_readings = self._get_latest_readings(nodes)
        
        # Load AI model if requested and available
        ai_model = load_model() if use_ai_model else None
        
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
            
            if ai_model:
                # Use AI model for prediction
                priority = self._predict_priority_with_ai(
                    ai_model, node, reading, distance_m
                )
            else:
                # Use rule-based calculation
                priority = self.calculate_single_priority(
                    distance_m=distance_m,
                    waste_level=reading.waste_level,
                    gas_level=reading.gas_level,
                    temperature=reading.temperature,
                    humidity=reading.humidity
                )
            
            priorities[node.id] = priority
        
        return priorities
    
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
            # This should match the feature order from train_model.py
            features = [
                distance_m,                    # distance_from_user
                reading.temperature,           # temperature
                reading.humidity,              # humidity
                reading.gas_level,             # gas_level
                reading.waste_level,           # waste_level
                reading.temperature,           # mean_temperature (simplified)
                0.0,                          # std_temperature (simplified)
                reading.humidity,              # mean_humidity (simplified)
                0.0,                          # std_humidity (simplified)
                reading.gas_level,             # mean_gas (simplified)
                0.0,                          # std_gas (simplified)
                reading.waste_level,           # mean_waste (simplified)
                0.0,                          # std_waste (simplified)
                reading.timestamp.hour,        # hour
                reading.timestamp.weekday()    # day_of_week
            ]
            
            # Predict priority
            prediction = model.predict([features])[0]
            
            # Ensure output is in valid range
            return max(0.0, min(1.0, float(prediction)))
            
        except Exception as e:
            # Fallback to rule-based calculation if AI prediction fails
            return self.calculate_single_priority(
                distance_m=distance_m,
                waste_level=reading.waste_level,
                gas_level=reading.gas_level,
                temperature=reading.temperature,
                humidity=reading.humidity
            )
    
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
        priorities = self.calculate_node_priorities(nodes, user_lat, user_lng)
        
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