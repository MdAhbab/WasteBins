from django.db import migrations, models
import django.db.models.deletion
from django.conf import settings
import django.utils.timezone

class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='BinGroup',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=100, unique=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='Node',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=100, unique=True)),
                ('latitude', models.FloatField(blank=True, null=True)),
                ('longitude', models.FloatField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('group', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='nodes', to='bins.bingroup')),
            ],
        ),
        migrations.CreateModel(
            name='SensorReading',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('temperature', models.FloatField()),
                ('humidity', models.FloatField()),
                ('gas_level', models.FloatField()),
                ('distance_to_next_bin', models.FloatField(blank=True, null=True)),
                ('timestamp', models.DateTimeField(default=django.utils.timezone.now)),
                ('node', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='readings', to='bins.node')),
            ],
        ),
        migrations.CreateModel(
            name='AICost',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('features', models.JSONField()),
                ('predicted_cost', models.FloatField()),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('model_version', models.CharField(blank=True, default='', max_length=50)),
                ('node', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='ai_costs', to='bins.node')),
            ],
        ),
        migrations.CreateModel(
            name='CollectionRoute',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('route_data', models.JSONField()),
                ('total_cost', models.FloatField()),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('generated_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='Notification',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('message', models.TextField()),
                ('level', models.CharField(choices=[('INFO', 'INFO'), ('WARN', 'WARN'), ('CRITICAL', 'CRITICAL')], default='INFO', max_length=10)),
                ('is_read', models.BooleanField(default=False)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='UserSetting',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('notify_email', models.BooleanField(default=False)),
                ('polling_interval_sec', models.PositiveIntegerField(default=10)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='settings', to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]