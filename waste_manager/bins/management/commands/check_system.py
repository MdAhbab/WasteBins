from django.core.management.base import BaseCommand
from django.db import connection
from bins.models import BinGroup, Node, SensorReading, AICost
from bins.utils.ai.model_store import get_model_version
import os

class Command(BaseCommand):
    help = "Check system health and configuration"

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('=== System Health Check ==='))
        
        # Database connectivity
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            self.stdout.write(self.style.SUCCESS('✓ Database connection: OK'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'✗ Database connection: {e}'))
        
        # Model counts
        try:
            bin_groups = BinGroup.objects.count()
            nodes = Node.objects.count()
            readings = SensorReading.objects.count()
            ai_costs = AICost.objects.count()
            
            self.stdout.write(self.style.SUCCESS(f'✓ Data counts:'))
            self.stdout.write(f'  - Bin Groups: {bin_groups}')
            self.stdout.write(f'  - Nodes: {nodes}')
            self.stdout.write(f'  - Sensor Readings: {readings}')
            self.stdout.write(f'  - AI Costs: {ai_costs}')
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'✗ Model access: {e}'))
        
        # AI Model
        try:
            version = get_model_version()
            self.stdout.write(self.style.SUCCESS(f'✓ AI Model version: {version}'))
        except Exception as e:
            self.stdout.write(self.style.WARNING(f'? AI Model: {e}'))
        
        # Static files
        from django.conf import settings
        static_dir = settings.STATICFILES_DIRS[0] if settings.STATICFILES_DIRS else None
        if static_dir and os.path.exists(static_dir):
            self.stdout.write(self.style.SUCCESS(f'✓ Static files directory: {static_dir}'))
        else:
            self.stdout.write(self.style.WARNING(f'? Static files directory not found'))
        
        # Template files
        template_dir = settings.TEMPLATES[0]['DIRS'][0] if settings.TEMPLATES[0]['DIRS'] else None
        if template_dir and os.path.exists(template_dir):
            self.stdout.write(self.style.SUCCESS(f'✓ Template files directory: {template_dir}'))
        else:
            self.stdout.write(self.style.WARNING(f'? Template files directory not found'))
        
        self.stdout.write(self.style.SUCCESS('=== Health Check Complete ==='))