"""
Test settings: run the suite without a MySQL server by using an in-memory
SQLite database.  Usage:

    python manage.py test bins --settings=waste_manager.test_settings
"""
from .settings import *  # noqa: F401,F403

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': ':memory:',
    }
}
