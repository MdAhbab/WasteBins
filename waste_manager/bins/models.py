"""
Domain model for the WasteBins intelligent collection platform.

The schema covers four layers:

  1. Sensing      -- Node (a physical bin), SensorReading, SensorHealth.
  2. Fleet        -- Depot, Vehicle, and the deployment constraints a real
                     municipal operation imposes (capacity, shift, time window).
  3. Dispatch     -- RoutePlan / RouteStop / ServiceEvent, the auditable record
                     of what was planned and what was actually collected.
  4. Governance   -- ModelVersion (model registry) and AuditEntry (a hash-chained,
                     tamper-evident log of telemetry and dispatch decisions).
"""
from django.conf import settings
from django.db import models
from django.utils import timezone


# ---------------------------------------------------------------------------
# 1. Sensing
# ---------------------------------------------------------------------------
class BinGroup(models.Model):
    name = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class Node(models.Model):
    """A physical waste bin instrumented with a sensor node."""

    WASTE_STREAMS = (
        ("general", "General"),
        ("organic", "Organic"),
        ("recyclable", "Recyclable"),
        ("hazardous", "Hazardous"),
    )

    name = models.CharField(max_length=100, unique=True)
    group = models.ForeignKey(BinGroup, on_delete=models.CASCADE, null=True, blank=True,
                              related_name="nodes")
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)

    # --- physical + operational attributes used by the fleet planner ---
    waste_stream = models.CharField(max_length=16, choices=WASTE_STREAMS, default="general")
    capacity_liters = models.FloatField(
        default=1100.0, help_text="Nominal bin volume in litres (EN 840 1100 L default)."
    )
    waste_density_kg_per_m3 = models.FloatField(
        default=220.0, help_text="Uncompacted municipal solid waste bulk density."
    )
    service_minutes = models.FloatField(
        default=4.0, help_text="Time a crew needs on site to empty this bin."
    )
    window_start_min = models.IntegerField(
        default=0, help_text="Earliest servicing time, minutes after shift start."
    )
    window_end_min = models.IntegerField(
        default=1440, help_text="Latest servicing time, minutes after shift start."
    )
    is_active = models.BooleanField(default=True)
    last_collected_at = models.DateTimeField(null=True, blank=True)

    last_update = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]

    def __str__(self):
        return self.name

    def get_latest_reading(self):
        """Most recent sensor reading for this node."""
        return self.readings.order_by("-timestamp").first()

    def full_load_kg(self) -> float:
        """Mass of waste in this bin when 100% full."""
        return (self.capacity_liters / 1000.0) * self.waste_density_kg_per_m3

    def load_kg(self, fill_fraction: float) -> float:
        """Mass of waste currently in the bin for a given fill fraction."""
        return max(0.0, float(fill_fraction)) * self.full_load_kg()

    def hours_since_collection(self, now=None) -> float:
        now = now or timezone.now()
        if not self.last_collected_at:
            return float(settings.ROUTING_AGING_TAU_H)
        return max(0.0, (now - self.last_collected_at).total_seconds() / 3600.0)


class SensorReading(models.Model):
    node = models.ForeignKey(Node, on_delete=models.CASCADE, related_name="readings")
    temperature = models.FloatField(null=True, blank=True)
    humidity = models.FloatField(null=True, blank=True)
    gas_level = models.FloatField(null=True, blank=True)
    waste_level = models.FloatField(
        null=True, blank=True, default=0.0,
        help_text="Fill fraction 0.00-1.00. NULL marks a genuinely missing channel.",
    )
    traffic_density = models.FloatField(default=0.0, help_text="Observed congestion 0.0-1.0.")
    distance_to_next_bin = models.FloatField(null=True, blank=True)

    # --- link-layer / device telemetry used by the fault detectors ---
    battery_v = models.FloatField(null=True, blank=True)
    rssi_dbm = models.FloatField(null=True, blank=True)
    is_synthetic = models.BooleanField(
        default=False, help_text="True when produced by the simulator or a fault-injection run."
    )
    fault_label = models.CharField(
        max_length=32, blank=True, default="",
        help_text="Ground-truth fault mode when injected (evaluation only).",
    )

    timestamp = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["-timestamp"]
        indexes = [
            models.Index(fields=["node", "-timestamp"]),
            models.Index(fields=["-timestamp"]),
        ]

    def __str__(self):
        return f"{self.node.name} @ {self.timestamp.isoformat()}"

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        # Keep the denormalised node timestamp in step for dashboard freshness.
        Node.objects.filter(pk=self.node_id).update(last_update=self.timestamp)


class SensorHealth(models.Model):
    """Per-channel health state maintained by the runtime validation layer."""

    STATUS = (
        ("ok", "OK"),
        ("suspect", "Suspect"),
        ("degraded", "Degraded"),
        ("failed", "Failed"),
    )
    CHANNELS = (
        ("waste_level", "Waste level"),
        ("gas_level", "Gas level"),
        ("temperature", "Temperature"),
        ("humidity", "Humidity"),
    )

    node = models.ForeignKey(Node, on_delete=models.CASCADE, related_name="health")
    channel = models.CharField(max_length=24, choices=CHANNELS)
    status = models.CharField(max_length=12, choices=STATUS, default="ok")
    trust = models.FloatField(default=1.0, help_text="Channel trust weight in [0,1].")
    drift_estimate = models.FloatField(default=0.0, help_text="Estimated additive bias.")
    stuck_streak = models.IntegerField(default=0)
    missing_streak = models.IntegerField(default=0)
    last_ok_at = models.DateTimeField(null=True, blank=True)
    detail = models.JSONField(default=dict, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("node", "channel")
        indexes = [models.Index(fields=["node", "channel"])]

    def __str__(self):
        return f"{self.node.name}/{self.channel}={self.status} (trust {self.trust:.2f})"


# ---------------------------------------------------------------------------
# 2. Fleet
# ---------------------------------------------------------------------------
class Depot(models.Model):
    name = models.CharField(max_length=100, unique=True)
    latitude = models.FloatField()
    longitude = models.FloatField()
    open_minute = models.IntegerField(default=0, help_text="Depot opening, minutes from midnight.")
    close_minute = models.IntegerField(default=1440, help_text="Depot closing, minutes from midnight.")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class Vehicle(models.Model):
    """A collection truck with the constraints that bound a real shift."""

    EURO_CLASSES = (
        ("euro3", "Euro III"),
        ("euro4", "Euro IV"),
        ("euro5", "Euro V"),
        ("euro6", "Euro VI"),
        ("ev", "Battery electric"),
    )

    name = models.CharField(max_length=64, unique=True)
    depot = models.ForeignKey(Depot, on_delete=models.CASCADE, related_name="vehicles")
    capacity_kg = models.FloatField(
        default=6000.0,
        help_text="Payload mass limit in kg. Mass is conserved: compaction does not reduce it.",
    )
    body_volume_m3 = models.FloatField(
        default=16.0,
        help_text=("Usable body volume in cubic metres. Compaction reduces the volume a load "
                   "occupies, so this is the limit compaction acts on. Whether volume or mass "
                   "binds first depends on density: they meet at capacity_kg / (volume * "
                   "compaction), which is 150 kg/m3 at the defaults, so mass binds for the "
                   "densities configured here. Zero disables the volume constraint."),
    )
    shift_minutes = models.FloatField(default=480.0, help_text="Hard shift duration limit.")
    shift_start_minute = models.IntegerField(default=360, help_text="Shift start, minutes from midnight.")
    avg_speed_kmh = models.FloatField(default=20.0, help_text="Free-flow planning speed.")
    kerb_mass_kg = models.FloatField(default=12000.0, help_text="Empty vehicle mass (payload model).")
    euro_class = models.CharField(max_length=8, choices=EURO_CLASSES, default="euro4")
    compaction_ratio = models.FloatField(default=2.5, help_text="Volume reduction achieved on board.")
    accepts_streams = models.JSONField(
        default=list, blank=True,
        help_text="Waste streams this vehicle may collect; empty means all.",
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]

    def __str__(self):
        return self.name


# ---------------------------------------------------------------------------
# 3. Dispatch
# ---------------------------------------------------------------------------
class RoutePlan(models.Model):
    """One dispatch decision: which vehicles serve which bins, in what order."""

    algorithm = models.CharField(max_length=48, default="cvrptw_savings_ls")
    params = models.JSONField(default=dict, blank=True)
    metrics = models.JSONField(default=dict, blank=True)
    horizon_start = models.DateTimeField(default=timezone.now)
    generated_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
                                     null=True, blank=True)
    compute_ms = models.FloatField(default=0.0)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-timestamp"]

    def __str__(self):
        return f"RoutePlan #{self.pk} ({self.algorithm})"


class RouteStop(models.Model):
    plan = models.ForeignKey(RoutePlan, on_delete=models.CASCADE, related_name="stops")
    vehicle = models.ForeignKey(Vehicle, on_delete=models.CASCADE, related_name="stops")
    node = models.ForeignKey(Node, on_delete=models.CASCADE, related_name="stops")
    sequence = models.IntegerField()
    arrival_min = models.FloatField(default=0.0, help_text="Minutes after shift start.")
    departure_min = models.FloatField(default=0.0)
    leg_distance_m = models.FloatField(default=0.0)
    leg_co2_kg = models.FloatField(default=0.0)
    load_after_kg = models.FloatField(default=0.0)
    priority = models.FloatField(default=0.0)

    class Meta:
        ordering = ["vehicle_id", "sequence"]
        indexes = [models.Index(fields=["plan", "vehicle", "sequence"])]


class CollectionRoute(models.Model):
    """Legacy single-vehicle route record, retained for API backward compatibility."""

    route_data = models.JSONField()
    total_cost = models.FloatField()
    generated_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
                                     null=True, blank=True)
    plan = models.ForeignKey(RoutePlan, on_delete=models.SET_NULL, null=True, blank=True,
                             related_name="legacy_routes")
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-timestamp"]


class ServiceEvent(models.Model):
    """A bin that was actually emptied -- ground truth for equity and aging."""

    node = models.ForeignKey(Node, on_delete=models.CASCADE, related_name="service_events")
    vehicle = models.ForeignKey(Vehicle, on_delete=models.SET_NULL, null=True, blank=True,
                                related_name="service_events")
    plan = models.ForeignKey(RoutePlan, on_delete=models.SET_NULL, null=True, blank=True,
                             related_name="service_events")
    collected_at = models.DateTimeField(default=timezone.now)
    fill_at_collection = models.FloatField(default=0.0)
    load_kg = models.FloatField(default=0.0)
    wait_hours = models.FloatField(default=0.0, help_text="Hours since the previous collection.")
    was_overflowing = models.BooleanField(default=False)

    class Meta:
        ordering = ["-collected_at"]
        indexes = [models.Index(fields=["node", "-collected_at"])]


# ---------------------------------------------------------------------------
# 4. Governance
# ---------------------------------------------------------------------------
class ModelVersion(models.Model):
    """Registry entry for a trained artefact, so predictions stay attributable."""

    KINDS = (
        ("forward_bundle", "Forward-looking bundle"),
        ("legacy_rf", "Legacy random forest"),
        ("continual", "Continual residual corrector"),
    )

    name = models.CharField(max_length=64)
    kind = models.CharField(max_length=32, choices=KINDS, default="forward_bundle")
    version = models.CharField(max_length=64)
    artifact_path = models.CharField(max_length=512, blank=True, default="")
    artifact_sha256 = models.CharField(max_length=64, blank=True, default="")
    hyperparameters = models.JSONField(default=dict, blank=True)
    search_space = models.JSONField(default=dict, blank=True)
    metrics = models.JSONField(default=dict, blank=True)
    training_rows = models.IntegerField(default=0)
    validation_scheme = models.CharField(max_length=128, blank=True, default="")
    is_active = models.BooleanField(default=False)
    trained_at = models.DateTimeField(default=timezone.now)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-trained_at"]
        unique_together = ("kind", "version")

    def __str__(self):
        return f"{self.kind}:{self.version}"


class AuditEntry(models.Model):
    """
    Append-only, hash-chained audit record.

    entry_hash = H(sequence || prev_hash || event_type || canonical_json(payload)
                   || timestamp), so any retroactive edit invalidates every later
    entry.  Block boundaries additionally store a Merkle root over the block's
    entry hashes for compact third-party verification.
    """

    EVENTS = (
        ("reading", "Sensor reading ingested"),
        ("plan", "Route plan generated"),
        ("service", "Bin serviced"),
        ("model", "Model version activated"),
        ("fault", "Sensor fault detected"),
        ("config", "Configuration changed"),
        ("genesis", "Ledger genesis"),
    )

    sequence = models.BigIntegerField(unique=True)
    event_type = models.CharField(max_length=16, choices=EVENTS)
    payload = models.JSONField(default=dict, blank=True)
    payload_sha256 = models.CharField(max_length=64)
    prev_hash = models.CharField(max_length=64)
    entry_hash = models.CharField(max_length=64, unique=True)
    merkle_root = models.CharField(max_length=64, blank=True, default="",
                                   help_text="Set on the last entry of each block.")
    signature = models.CharField(max_length=64, blank=True, default="",
                                 help_text="HMAC-SHA256 when AUDIT_HMAC_KEY is configured.")
    actor = models.CharField(max_length=64, blank=True, default="system")
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["sequence"]
        indexes = [
            models.Index(fields=["sequence"]),
            models.Index(fields=["event_type", "-created_at"]),
        ]

    def __str__(self):
        return f"#{self.sequence} {self.event_type} {self.entry_hash[:12]}"


# ---------------------------------------------------------------------------
# Misc application models
# ---------------------------------------------------------------------------
class AICost(models.Model):
    node = models.ForeignKey(Node, on_delete=models.CASCADE, related_name="ai_costs")
    features = models.JSONField()
    predicted_cost = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)
    model_version = models.CharField(max_length=50, blank=True, default="")

    def __str__(self):
        return f"{self.node.name} cost={self.predicted_cost:.3f} ({self.model_version})"


class Notification(models.Model):
    LEVEL_CHOICES = (
        ("INFO", "INFO"),
        ("WARN", "WARN"),
        ("CRITICAL", "CRITICAL"),
    )
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
                             null=True, blank=True)
    node = models.ForeignKey(Node, on_delete=models.SET_NULL, null=True, blank=True,
                             related_name="notifications")
    message = models.TextField()
    level = models.CharField(max_length=10, choices=LEVEL_CHOICES, default="INFO")
    category = models.CharField(max_length=24, blank=True, default="general")
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]


class UserSetting(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                                related_name="settings")
    notify_email = models.BooleanField(default=False)
    polling_interval_sec = models.PositiveIntegerField(default=10)

    latitude = models.FloatField(null=True, blank=True,
                                 help_text="User's latitude for route calculation")
    longitude = models.FloatField(null=True, blank=True,
                                  help_text="User's longitude for route calculation")
    location_name = models.CharField(max_length=255, blank=True, default="")
    auto_update_location = models.BooleanField(default=False)

    # Planner preferences surfaced in the UI
    routing_alpha = models.FloatField(default=0.5)
    aging_gamma = models.FloatField(default=0.5)
    aging_tau_h = models.FloatField(default=48.0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        location_info = ""
        if self.latitude is not None and self.longitude is not None:
            location_info = f" @ ({self.latitude:.4f}, {self.longitude:.4f})"
        return f"Settings({self.user.username}){location_info}"

    def has_location(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    def get_location_dict(self):
        if self.has_location():
            return {"lat": float(self.latitude), "lng": float(self.longitude),
                    "name": self.location_name}
        return None
