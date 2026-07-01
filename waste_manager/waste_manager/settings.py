import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'change-me-in-production'
DEBUG = True

# Allow access from any IP address on the LAN
ALLOWED_HOSTS = ['*']

# Configure CSRF to trust the React dev server (Vite on port 5173)
CSRF_TRUSTED_ORIGINS = [
    'http://localhost:8000',
    'http://127.0.0.1:8000',
    'http://localhost:5173',
    'http://127.0.0.1:5173',
    'http://localhost:5174',
    'http://127.0.0.1:5174',
]

# Allow cookies to be sent cross-origin so the Vite dev proxy works
CSRF_COOKIE_SECURE = False
CSRF_COOKIE_HTTPONLY = False
CSRF_COOKIE_SAMESITE = 'Lax'
SESSION_COOKIE_SAMESITE = 'Lax'
SESSION_COOKIE_SECURE = False

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'bins',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'waste_manager.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [
            BASE_DIR / 'bins' / 'templates',
        ],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'waste_manager.wsgi.application'
ASGI_APPLICATION = 'waste_manager.asgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'waste_manager_db',
        'USER': 'root',
        'PASSWORD': '12345678',
        'HOST': '127.0.0.1',
        'PORT': '3306',
        'OPTIONS': {
            'charset': 'utf8mb4',
            'init_command': "SET sql_mode='STRICT_TRANS_TABLES'",
        },
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',},
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Asia/Dhaka'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'bins' / 'static']

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

LOGIN_URL = '/login/'
LOGIN_REDIRECT_URL = '/dashboard/'   # kept for Django admin / fallback
LOGOUT_REDIRECT_URL = '/login/'

# ---------------------------------------------------------------------------
# CORS – allow the Vite dev server to call the Django API
# ---------------------------------------------------------------------------
CORS_ALLOWED_ORIGINS = [
    'http://localhost:5173',
    'http://127.0.0.1:5173',
    'http://localhost:5174',
    'http://127.0.0.1:5174',
]

CSRF_TRUSTED_ORIGINS = [
    'http://localhost:5173',
    'http://127.0.0.1:5173',
    'http://localhost:5174',
    'http://127.0.0.1:5174',
]
CORS_ALLOW_CREDENTIALS = True
CORS_EXPOSE_HEADERS = ['Content-Type', 'X-CSRFToken']
CORS_ALLOW_HEADERS = [
    'accept', 'accept-encoding', 'authorization', 'content-type',
    'dnt', 'origin', 'user-agent', 'x-csrftoken', 'x-requested-with',
]

# ---------------------------------------------------------------------------
# Django REST Framework
# ---------------------------------------------------------------------------
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.BasicAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
}

# Routing and AI defaults
ROUTING_ALPHA = 0.5
MODEL_STORE_DIR = BASE_DIR / 'model_store'
MODEL_STORE_DIR.mkdir(exist_ok=True)
MODEL_FILENAME = MODEL_STORE_DIR / 'rf_cost_model.joblib'
MODEL_META_FILENAME = MODEL_STORE_DIR / 'rf_cost_model_meta.json'

# Forward-looking (non-circular) prediction bundle: a histogram gradient-boosting
# regressor for time-to-overflow, P10/P50/P90 quantile regressors for uncertainty,
# and a calibrated hazard classifier. See bins/utils/ai/train_forward.py.
FORWARD_MODEL_FILENAME = MODEL_STORE_DIR / 'forward_bundle.joblib'
FORWARD_MODEL_META_FILENAME = MODEL_STORE_DIR / 'forward_bundle_meta.json'

# Routing refinements toggle (2-opt/Or-opt local search + orienteering budget).
ROUTING_REFINE = True
ROUTING_AGING_GAMMA = 0.5      # anti-starvation weight (0 disables)
ROUTING_AGING_TAU_H = 48.0     # hours at which the aging boost saturates

# Local settings override
try:
    from .local_settings import *
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Dynamic AI & Routing Feature Configuration
# ---------------------------------------------------------------------------
# Dynamic feature toggles and their weights. 
# The algorithm math dynamically balances and normalizes if a feature is missing or added.
DYNAMIC_FEATURES = {
    'distance_m': {
        'type': 'priority',
        'weight': 0.25,
        'min_val': 0.0,
        'max_val': 2000.0,
        'impact': 'negative'     # Closer = higher priority
    },
    'waste_level': {
        'type': 'priority',
        'weight': 0.35,
        'min_val': 0.0,
        'max_val': 1.0,
        'impact': 'positive'     # Fuller = higher priority
    },
    'gas_level': {
        'type': 'priority',
        'weight': 0.25,
        'min_val': 0.0,
        'max_val': 1.0,
        'impact': 'positive'     # Smelly = higher priority
    },
    'temperature': {
        'type': 'priority',
        'weight': 0.10,
        'min_val': 10.0,
        'max_val': 40.0,
        'optimal': 25.0,         # Deviations from 25C increase priority (spoilage risk)
        'impact': 'deviation'
    },
    'humidity': {
        'type': 'priority',
        'weight': 0.05,
        'min_val': 50.0,
        'max_val': 100.0,
        'impact': 'positive'     # Higher humidity = higher priority
    },
    'traffic_density': {
        'type': 'cost_multiplier', # Used in Dijkstra optimization
        'routing_weight': 3.0,     # Multiplies physical distance (1 + traffic * weight)
        'min_val': 0.0,
        'max_val': 1.0,
        # It also carries a small negative priority weight (optional, but requested to mirror previous behavior)
        'weight': 0.10,
        'impact': 'negative'       # High traffic slightly reduces a bin's raw priority
    }
}