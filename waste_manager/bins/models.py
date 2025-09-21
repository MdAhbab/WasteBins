from django.conf import settings
from django.db import models
from django.utils import timezone

class BinGroup(models.Model):
    name = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class Node(models.Model):
    name = models.CharField(max_length=100, unique=True)
    group = models.ForeignKey(BinGroup, on_delete=models.CASCADE, null=True, blank=True, related_name='nodes')
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    last_update = models.DateTimeField(auto_now=True)  # Auto-update timestamp for dashboard
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
    
    def get_latest_reading(self):
        """Get the most recent sensor reading for this node"""
        return self.readings.order_by('-timestamp').first()

class SensorReading(models.Model):
    node = models.ForeignKey(Node, on_delete=models.CASCADE, related_name='readings')
    temperature = models.FloatField()
    humidity = models.FloatField()
    gas_level = models.FloatField()
    # Waste level: 0.00 - 1.00 (two decimals suggested), required for priority calculation
    waste_level = models.FloatField(
        null=False, 
        default=0.0,
        help_text="Waste level from 0.00 to 1.00 (0% to 100% full)"
    )
    distance_to_next_bin = models.FloatField(null=True, blank=True)
    timestamp = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['node', '-timestamp']),
        ]

    def __str__(self):
        return f"{self.node.name} @ {self.timestamp.isoformat()}"
    
    def save(self, *args, **kwargs):
        """Update node's last_update timestamp when a new reading is saved"""
        super().save(*args, **kwargs)
        # Update the node's last_update timestamp
        self.node.last_update = self.timestamp
        self.node.save(update_fields=['last_update'])

class AICost(models.Model):
    node = models.ForeignKey(Node, on_delete=models.CASCADE, related_name='ai_costs')
    features = models.JSONField()
    predicted_cost = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)
    model_version = models.CharField(max_length=50, blank=True, default='')

    def __str__(self):
        return f"{self.node.name} cost={self.predicted_cost:.3f} ({self.model_version})"

class CollectionRoute(models.Model):
    route_data = models.JSONField()  # e.g., {"path": [id1, id2, ...], "edges": [{"u": id1, "v": id2, "w": 123.4}, ...]}
    total_cost = models.FloatField()
    generated_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

class Notification(models.Model):
    LEVEL_CHOICES = (
        ('INFO', 'INFO'),
        ('WARN', 'WARN'),
        ('CRITICAL', 'CRITICAL'),
    )
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True)
    message = models.TextField()
    level = models.CharField(max_length=10, choices=LEVEL_CHOICES, default='INFO')
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

class UserSetting(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='settings')
    notify_email = models.BooleanField(default=False)
    polling_interval_sec = models.PositiveIntegerField(default=10)
    
    # User location fields for route calculation
    latitude = models.FloatField(
        null=True, 
        blank=True,
        help_text="User's latitude coordinate for route calculation"
    )
    longitude = models.FloatField(
        null=True, 
        blank=True,
        help_text="User's longitude coordinate for route calculation"
    )
    location_name = models.CharField(
        max_length=255, 
        blank=True, 
        default="",
        help_text="Optional location name/address for reference"
    )
    auto_update_location = models.BooleanField(
        default=False,
        help_text="Automatically use current GPS location when available"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        location_info = ""
        if self.latitude and self.longitude:
            location_info = f" @ ({self.latitude:.4f}, {self.longitude:.4f})"
        return f"Settings({self.user.username}){location_info}"
    
    def has_location(self):
        """Check if user has valid location coordinates"""
        return self.latitude is not None and self.longitude is not None
    
    def get_location_dict(self):
        """Get location as dictionary for API usage"""
        if self.has_location():
            return {
                'lat': float(self.latitude),
                'lng': float(self.longitude),
                'name': self.location_name
            }
        return None