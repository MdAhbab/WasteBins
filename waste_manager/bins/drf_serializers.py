"""DRF serializers for the WasteBins React frontend."""
from django.contrib.auth.models import User
from rest_framework import serializers

from .models import (AuditEntry, BinGroup, CollectionRoute, Depot, ModelVersion,
                     Node, Notification, RoutePlan, RouteStop, SensorHealth,
                     SensorReading, ServiceEvent, UserSetting, Vehicle)


class BinGroupSerializer(serializers.ModelSerializer):
    class Meta:
        model = BinGroup
        fields = ["id", "name", "created_at"]


class NodeSerializer(serializers.ModelSerializer):
    group = BinGroupSerializer(read_only=True)
    last_reading_at = serializers.SerializerMethodField()
    full_load_kg = serializers.SerializerMethodField()

    class Meta:
        model = Node
        fields = ["id", "name", "group", "latitude", "longitude", "waste_stream",
                  "capacity_liters", "waste_density_kg_per_m3", "service_minutes",
                  "window_start_min", "window_end_min", "is_active",
                  "last_collected_at", "last_update", "created_at",
                  "last_reading_at", "full_load_kg"]

    def get_last_reading_at(self, obj):
        # Prefer the denormalised column so a list of nodes does not trigger one
        # query per row.
        return obj.last_update.isoformat() if obj.last_update else None

    def get_full_load_kg(self, obj):
        return round(obj.full_load_kg(), 1)


class SensorReadingSerializer(serializers.ModelSerializer):
    node = NodeSerializer(read_only=True)
    waste_percentage = serializers.SerializerMethodField()

    class Meta:
        model = SensorReading
        fields = ["id", "node", "temperature", "humidity", "gas_level",
                  "waste_level", "waste_percentage", "traffic_density",
                  "distance_to_next_bin", "battery_v", "rssi_dbm",
                  "is_synthetic", "fault_label", "timestamp"]

    def get_waste_percentage(self, obj):
        return round(float(obj.waste_level) * 100, 1) if obj.waste_level is not None else None


class SensorHealthSerializer(serializers.ModelSerializer):
    node_name = serializers.CharField(source="node.name", read_only=True)

    class Meta:
        model = SensorHealth
        fields = ["id", "node", "node_name", "channel", "status", "trust",
                  "drift_estimate", "stuck_streak", "missing_streak",
                  "last_ok_at", "detail", "updated_at"]


class DepotSerializer(serializers.ModelSerializer):
    class Meta:
        model = Depot
        fields = ["id", "name", "latitude", "longitude", "open_minute",
                  "close_minute", "is_active"]


class VehicleSerializer(serializers.ModelSerializer):
    depot_name = serializers.CharField(source="depot.name", read_only=True)

    class Meta:
        model = Vehicle
        fields = ["id", "name", "depot", "depot_name", "capacity_kg",
                  "shift_minutes", "shift_start_minute", "avg_speed_kmh",
                  "kerb_mass_kg", "euro_class", "compaction_ratio",
                  "accepts_streams", "is_active"]


class RouteStopSerializer(serializers.ModelSerializer):
    node_name = serializers.CharField(source="node.name", read_only=True)
    latitude = serializers.FloatField(source="node.latitude", read_only=True)
    longitude = serializers.FloatField(source="node.longitude", read_only=True)

    class Meta:
        model = RouteStop
        fields = ["id", "vehicle", "node", "node_name", "latitude", "longitude",
                  "sequence", "arrival_min", "departure_min", "leg_distance_m",
                  "leg_co2_kg", "load_after_kg", "priority"]


class RoutePlanSerializer(serializers.ModelSerializer):
    stops = RouteStopSerializer(many=True, read_only=True)
    generated_by_username = serializers.SerializerMethodField()

    class Meta:
        model = RoutePlan
        fields = ["id", "algorithm", "params", "metrics", "horizon_start",
                  "compute_ms", "timestamp", "generated_by_username", "stops"]

    def get_generated_by_username(self, obj):
        return obj.generated_by.username if obj.generated_by else None


class RoutePlanSummarySerializer(serializers.ModelSerializer):
    """Plan without its stops, for list endpoints."""

    class Meta:
        model = RoutePlan
        fields = ["id", "algorithm", "metrics", "compute_ms", "timestamp"]


class ServiceEventSerializer(serializers.ModelSerializer):
    node_name = serializers.CharField(source="node.name", read_only=True)
    vehicle_name = serializers.CharField(source="vehicle.name", read_only=True)

    class Meta:
        model = ServiceEvent
        fields = ["id", "node", "node_name", "vehicle", "vehicle_name", "plan",
                  "collected_at", "fill_at_collection", "load_kg", "wait_hours",
                  "was_overflowing"]


class AuditEntrySerializer(serializers.ModelSerializer):
    class Meta:
        model = AuditEntry
        fields = ["sequence", "event_type", "payload", "payload_sha256",
                  "prev_hash", "entry_hash", "merkle_root", "signature",
                  "actor", "created_at"]


class ModelVersionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelVersion
        fields = ["id", "name", "kind", "version", "artifact_sha256",
                  "hyperparameters", "metrics", "training_rows",
                  "validation_scheme", "is_active", "trained_at"]


class NotificationSerializer(serializers.ModelSerializer):
    node_name = serializers.CharField(source="node.name", read_only=True, default=None)

    class Meta:
        model = Notification
        fields = ["id", "message", "level", "category", "node", "node_name",
                  "is_read", "created_at"]


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["id", "username", "first_name", "last_name", "email",
                  "date_joined", "last_login", "is_staff"]
        read_only_fields = ["id", "username", "date_joined", "last_login", "is_staff"]


class UserSettingSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserSetting
        fields = ["notify_email", "polling_interval_sec", "latitude", "longitude",
                  "location_name", "auto_update_location", "routing_alpha",
                  "aging_gamma", "aging_tau_h", "created_at", "updated_at"]
        read_only_fields = ["created_at", "updated_at"]

    def validate(self, data):
        lat = data.get("latitude", getattr(self.instance, "latitude", None))
        lng = data.get("longitude", getattr(self.instance, "longitude", None))
        if lat is not None and not (-90 <= lat <= 90):
            raise serializers.ValidationError({"latitude": "Must be between -90 and 90."})
        if lng is not None and not (-180 <= lng <= 180):
            raise serializers.ValidationError({"longitude": "Must be between -180 and 180."})
        if ("latitude" in data or "longitude" in data) and ((lat is None) != (lng is None)):
            raise serializers.ValidationError(
                "Provide both latitude and longitude, or neither.")

        gamma = data.get("aging_gamma")
        if gamma is not None and not (0.0 <= gamma < 1.0):
            raise serializers.ValidationError(
                {"aging_gamma": "Must be in [0, 1); at 1 the score ignores urgency entirely."})
        tau = data.get("aging_tau_h")
        if tau is not None and tau <= 0:
            raise serializers.ValidationError({"aging_tau_h": "Must be positive."})
        alpha = data.get("routing_alpha")
        if alpha is not None and alpha < 0:
            raise serializers.ValidationError({"routing_alpha": "Must be non-negative."})
        return data


class CollectionRouteSerializer(serializers.ModelSerializer):
    generated_by_username = serializers.SerializerMethodField()

    class Meta:
        model = CollectionRoute
        fields = ["id", "route_data", "total_cost", "generated_by_username", "timestamp"]

    def get_generated_by_username(self, obj):
        return obj.generated_by.username if obj.generated_by else None
