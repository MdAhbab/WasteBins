from django.core.management.base import BaseCommand
from django.core.management import call_command
from pathlib import Path
from django.conf import settings

class Command(BaseCommand):
    help = "Load sample fixtures into the database."

    def handle(self, *args, **options):
        base = settings.BASE_DIR / 'fixtures'
        files = ['sample_nodes.json', 'sample_readings.json', 'sample_ai_costs.json']
        for f in files:
            self.stdout.write(self.style.NOTICE(f'Loading {f}'))
            call_command('loaddata', str(base / f))
        self.stdout.write(self.style.SUCCESS('Loaded sample fixtures.'))