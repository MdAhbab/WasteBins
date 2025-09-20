from django.contrib import admin
from .models import BinGroup, Node, SensorReading, AICost, CollectionRoute, Notification, UserSetting

@admin.register(BinGroup)
class BinGroupAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'created_at')

@admin.register(Node)
class NodeAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'group', 'latitude', 'longitude', 'created_at')
    list_filter = ('group',)

@admin.register(SensorReading)
class SensorReadingAdmin(admin.ModelAdmin):
    list_display = ('id', 'node', 'temperature', 'humidity', 'gas_level', 'distance_to_next_bin', 'timestamp')
    list_filter = ('node',)

@admin.register(AICost)
class AICostAdmin(admin.ModelAdmin):
    list_display = ('id', 'node', 'predicted_cost', 'model_version', 'timestamp')
    list_filter = ('model_version',)

@admin.register(CollectionRoute)
class CollectionRouteAdmin(admin.ModelAdmin):
    list_display = ('id', 'total_cost', 'generated_by', 'timestamp')

@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'level', 'is_read', 'created_at')
    list_filter = ('level', 'is_read')

@admin.register(UserSetting)
class UserSettingAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'notify_email', 'polling_interval_sec', 'created_at')