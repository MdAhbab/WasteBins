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
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class SensorReading(models.Model):
    node = models.ForeignKey(Node, on_delete=models.CASCADE, related_name='readings')
    temperature = models.FloatField()
    humidity = models.FloatField()
    gas_level = models.FloatField()
    distance_to_next_bin = models.FloatField(null=True, blank=True)
    timestamp = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.node.name} @ {self.timestamp.isoformat()}"

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
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Settings({self.user.username})"