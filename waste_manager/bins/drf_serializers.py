"""
DRF ModelSerializers for the WasteBins React frontend.
These are separate from the legacy serialize_reading / serialize_node helpers
so that existing template-based views are not disrupted.
"""
from django.contrib.auth.models import User
from rest_framework import serializers
from .models import Node, SensorReading, Notification, UserSetting, CollectionRoute, BinGroup


class BinGroupSerializer(serializers.ModelSerializer):
    class Meta:
        model = BinGroup
        fields = ['id', 'name', 'created_at']


class NodeSerializer(serializers.ModelSerializer):
    group = BinGroupSerializer(read_only=True)
    last_reading_at = serializers.SerializerMethodField()

    class Meta:
        model = Node
        fields = ['id', 'name', 'group', 'latitude', 'longitude',
                  'last_update', 'created_at', 'last_reading_at']

    def get_last_reading_at(self, obj):
        reading = obj.readings.order_by('-timestamp').first()
        return reading.timestamp.isoformat() if reading else None


class SensorReadingSerializer(serializers.ModelSerializer):
    node = NodeSerializer(read_only=True)
    waste_percentage = serializers.SerializerMethodField()

    class Meta:
        model = SensorReading
        fields = ['id', 'node', 'temperature', 'humidity', 'gas_level',
                  'waste_level', 'waste_percentage', 'distance_to_next_bin', 'timestamp']

    def get_waste_percentage(self, obj):
        return round(float(obj.waste_level) * 100, 1) if obj.waste_level is not None else 0.0


class NotificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Notification
        fields = ['id', 'message', 'level', 'is_read', 'created_at']


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'first_name', 'last_name', 'email',
                  'date_joined', 'last_login', 'is_staff']
        read_only_fields = ['id', 'username', 'date_joined', 'last_login', 'is_staff']


class UserSettingSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserSetting
        fields = [
            'notify_email', 'polling_interval_sec',
            'latitude', 'longitude', 'location_name', 'auto_update_location',
            'created_at', 'updated_at',
        ]
        read_only_fields = ['created_at', 'updated_at']

    def validate(self, data):
        lat = data.get('latitude')
        lng = data.get('longitude')
        if lat is not None and not (-90 <= lat <= 90):
            raise serializers.ValidationError({'latitude': 'Must be between -90 and 90.'})
        if lng is not None and not (-180 <= lng <= 180):
            raise serializers.ValidationError({'longitude': 'Must be between -180 and 180.'})
        if (lat is None) != (lng is None):
            raise serializers.ValidationError('Provide both latitude and longitude, or neither.')
        return data


class CollectionRouteSerializer(serializers.ModelSerializer):
    generated_by_username = serializers.SerializerMethodField()

    class Meta:
        model = CollectionRoute
        fields = ['id', 'route_data', 'total_cost', 'generated_by_username', 'timestamp']

    def get_generated_by_username(self, obj):
        return obj.generated_by.username if obj.generated_by else None
