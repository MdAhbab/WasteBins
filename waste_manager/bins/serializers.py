def serialize_node(node):
    return {
        'id': node.id,
        'name': node.name,
        'lat': node.latitude,
        'lng': node.longitude,
        'group': node.group.name if node.group else None,
    }

def serialize_reading(r):
    return {
        'node': serialize_node(r.node),
        'temperature': r.temperature,
        'humidity': r.humidity,
        'gas_level': r.gas_level,
        'distance_to_next_bin': r.distance_to_next_bin,
        'timestamp': r.timestamp.isoformat(),
    }