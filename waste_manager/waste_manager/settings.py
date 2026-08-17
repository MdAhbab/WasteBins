"""
Django settings for the WasteBins intelligent collection platform.

Everything environment-sensitive is read from the process environment with a
safe local default, so the project boots out of the box (SQLite) and can be
pointed at a production database without editing source.  A `.env` file in the
repository root is loaded automatically when present.
"""
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent

# The algorithm implementations live in `wastebins_core` at the repository root
# so that the service, the experiment suite and the tests all run the identical
# code.  Put it on the path before any app module imports it.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Minimal .env loader (no third-party dependency)
# ---------------------------------------------------------------------------
def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


for _candidate in (REPO_ROOT / ".env", BASE_DIR / ".env"):
    _load_dotenv(_candidate)


def env(name, default=None):
    return os.environ.get(name, default)


def env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def env_float(name, default):
    try:
        return float(os.environ[name])
    except (KeyError, TypeError, ValueError):
        return default


def env_int(name, default):
    try:
        return int(os.environ[name])
    except (KeyError, TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
SECRET_KEY = env("DJANGO_SECRET_KEY", "dev-only-insecure-key-change-in-production")
DEBUG = env_bool("DJANGO_DEBUG", True)

ALLOWED_HOSTS = [h.strip() for h in env("DJANGO_ALLOWED_HOSTS", "*").split(",") if h.strip()]

_DEV_ORIGINS = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:5174", "http://127.0.0.1:5174",
    "http://localhost:8000", "http://127.0.0.1:8000",
]
_extra_origins = [o.strip() for o in env("DJANGO_EXTRA_ORIGINS", "").split(",") if o.strip()]

CSRF_TRUSTED_ORIGINS = _DEV_ORIGINS + _extra_origins

CSRF_COOKIE_SECURE = env_bool("DJANGO_COOKIE_SECURE", False)
CSRF_COOKIE_HTTPONLY = False
CSRF_COOKIE_SAMESITE = "Lax"
SESSION_COOKIE_SAMESITE = "Lax"
SESSION_COOKIE_SECURE = env_bool("DJANGO_COOKIE_SECURE", False)

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "corsheaders",
    "bins",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "waste_manager.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "bins" / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "waste_manager.wsgi.application"
ASGI_APPLICATION = "waste_manager.asgi.application"


# ---------------------------------------------------------------------------
# Database -- SQLite by default so the project runs with zero setup.
# Set DB_ENGINE=mysql (plus DB_NAME/DB_USER/DB_PASSWORD/DB_HOST/DB_PORT) to use
# the MySQL deployment configuration described in the manuscript.
# ---------------------------------------------------------------------------
_DB_ENGINE = env("DB_ENGINE", "sqlite").lower()

if _DB_ENGINE in ("mysql", "django.db.backends.mysql"):
    try:  # PyMySQL shim so no compiled mysqlclient is required
        import pymysql  # noqa: F401

        pymysql.install_as_MySQLdb()
    except ImportError:  # pragma: no cover - optional dependency
        pass
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.mysql",
            "NAME": env("DB_NAME", "waste_manager_db"),
            "USER": env("DB_USER", "root"),
            "PASSWORD": env("DB_PASSWORD", ""),
            "HOST": env("DB_HOST", "127.0.0.1"),
            "PORT": env("DB_PORT", "3306"),
            "OPTIONS": {
                "charset": "utf8mb4",
                "init_command": "SET sql_mode='STRICT_TRANS_TABLES'",
            },
        }
    }
elif _DB_ENGINE in ("postgres", "postgresql", "django.db.backends.postgresql"):
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": env("DB_NAME", "waste_manager_db"),
            "USER": env("DB_USER", "postgres"),
            "PASSWORD": env("DB_PASSWORD", ""),
            "HOST": env("DB_HOST", "127.0.0.1"),
            "PORT": env("DB_PORT", "5432"),
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": env("DB_NAME", str(BASE_DIR / "db.sqlite3")),
            "OPTIONS": {"timeout": 20},
        }
    }

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = env("DJANGO_TIME_ZONE", "Asia/Dhaka")
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "bins" / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LOGIN_URL = "/login/"
LOGIN_REDIRECT_URL = "/dashboard/"
LOGOUT_REDIRECT_URL = "/login/"


# ---------------------------------------------------------------------------
# CORS / DRF
# ---------------------------------------------------------------------------
CORS_ALLOWED_ORIGINS = [o for o in _DEV_ORIGINS if ":517" in o] + _extra_origins
CORS_ALLOW_CREDENTIALS = True
CORS_EXPOSE_HEADERS = ["Content-Type", "X-CSRFToken"]
CORS_ALLOW_HEADERS = [
    "accept", "accept-encoding", "authorization", "content-type",
    "dnt", "origin", "user-agent", "x-csrftoken", "x-requested-with",
]

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework.authentication.BasicAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
}


# ---------------------------------------------------------------------------
# Model store
# ---------------------------------------------------------------------------
MODEL_STORE_DIR = Path(env("MODEL_STORE_DIR", str(BASE_DIR / "model_store")))
MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FILENAME = MODEL_STORE_DIR / "rf_cost_model.joblib"
MODEL_META_FILENAME = MODEL_STORE_DIR / "rf_cost_model_meta.json"

# Forward-looking (non-circular) prediction bundle: a histogram gradient-boosting
# regressor for time-to-overflow, P10/P50/P90 quantile regressors for uncertainty,
# and a calibrated hazard classifier.  See bins/utils/ai/train_forward.py.
FORWARD_MODEL_FILENAME = MODEL_STORE_DIR / "forward_bundle.joblib"
FORWARD_MODEL_META_FILENAME = MODEL_STORE_DIR / "forward_bundle_meta.json"

# Continual-learning residual corrector (small, incrementally updated online).
CONTINUAL_MODEL_FILENAME = MODEL_STORE_DIR / "continual_state.joblib"
CONTINUAL_META_FILENAME = MODEL_STORE_DIR / "continual_state_meta.json"


# ---------------------------------------------------------------------------
# Serving budget -- the deployed app must never run a heavy training job on the
# request path.  Batch training is an explicit offline command; online updates
# are bounded by these limits and are skipped when the budget is exceeded.
# ---------------------------------------------------------------------------
ML_SERVING = {
    "ALLOW_INLINE_BATCH_TRAINING": env_bool("ML_ALLOW_INLINE_TRAINING", False),
    "CONTINUAL_ENABLED": env_bool("ML_CONTINUAL_ENABLED", True),
    "CONTINUAL_MAX_SAMPLES_PER_UPDATE": env_int("ML_CONTINUAL_MAX_SAMPLES", 256),
    "CONTINUAL_MAX_MS_PER_UPDATE": env_float("ML_CONTINUAL_MAX_MS", 40.0),
    "CONTINUAL_MIN_SECONDS_BETWEEN_UPDATES": env_float("ML_CONTINUAL_MIN_INTERVAL_S", 15.0),
    "PREDICTION_CACHE_SECONDS": env_float("ML_PREDICTION_CACHE_S", 5.0),
    "EXPLAIN_MAX_PERTURBATIONS": env_int("ML_EXPLAIN_PERTURBATIONS", 240),
}


# ---------------------------------------------------------------------------
# Routing configuration
# ---------------------------------------------------------------------------
ROUTING_ALPHA = env_float("ROUTING_ALPHA", 0.5)
ROUTING_REFINE = env_bool("ROUTING_REFINE", True)
ROUTING_AGING_GAMMA = env_float("ROUTING_AGING_GAMMA", 0.5)   # anti-starvation weight (0 disables)
ROUTING_AGING_TAU_H = env_float("ROUTING_AGING_TAU_H", 48.0)  # hours at which the aging boost saturates

FLEET_DEFAULTS = {
    "VEHICLE_CAPACITY_KG": env_float("FLEET_CAPACITY_KG", 6000.0),
    "SHIFT_MINUTES": env_float("FLEET_SHIFT_MINUTES", 480.0),
    "SERVICE_MINUTES_PER_BIN": env_float("FLEET_SERVICE_MINUTES", 4.0),
    "BIN_VOLUME_KG": env_float("FLEET_BIN_FULL_KG", 240.0),  # mass of a 100%-full 1100 L bin
    "AVG_SPEED_KMH": env_float("FLEET_AVG_SPEED_KMH", 20.0),
    "DEPOT_LAT": env_float("FLEET_DEPOT_LAT", 23.8069),
    "DEPOT_LNG": env_float("FLEET_DEPOT_LNG", 90.3687),
    # Search budget for a plan requested from the UI.  The solver treats this as
    # an upper bound and only starts an improvement round it has time to finish,
    # so raising it buys quality and lowers responsiveness with no risk of an
    # overrun.  The interactive default is deliberately well below the budget the
    # experiments use: an operator waiting on a page has a different tolerance
    # than a benchmark run, and the published figures come from the experiment
    # scripts calling the solver directly, not from this endpoint.
    "PLAN_TIME_BUDGET_S": env_float("FLEET_PLAN_BUDGET_S", 2.0),
    # Comparison runs every installed solver in sequence, so its per-solver
    # budget is smaller again -- the page cost is this multiplied by five.
    "COMPARE_TIME_BUDGET_S": env_float("FLEET_COMPARE_BUDGET_S", 1.0),
}


# ---------------------------------------------------------------------------
# Traffic provider.  "synthetic" is a deterministic, reproducible congestion
# surface; "live" activates the HTTP adapter (requires TRAFFIC_API_URL/KEY) and
# transparently falls back to synthetic when the feed is unreachable.
# ---------------------------------------------------------------------------
TRAFFIC = {
    "PROVIDER": env("TRAFFIC_PROVIDER", "synthetic"),
    "API_URL": env("TRAFFIC_API_URL", ""),
    "API_KEY": env("TRAFFIC_API_KEY", ""),
    "CACHE_SECONDS": env_float("TRAFFIC_CACHE_S", 120.0),
    "TIMEOUT_SECONDS": env_float("TRAFFIC_TIMEOUT_S", 3.0),
    "FREEFLOW_SPEED_KMH": env_float("TRAFFIC_FREEFLOW_KMH", 34.0),
    "MIN_SPEED_KMH": env_float("TRAFFIC_MIN_KMH", 5.0),
}


# ---------------------------------------------------------------------------
# Audit ledger
# ---------------------------------------------------------------------------
AUDIT_LEDGER = {
    "ENABLED": env_bool("AUDIT_LEDGER_ENABLED", True),
    "BLOCK_SIZE": env_int("AUDIT_BLOCK_SIZE", 64),
    "HMAC_KEY": env("AUDIT_HMAC_KEY", ""),  # optional shared secret for signing
}


# ---------------------------------------------------------------------------
# Dynamic AI & routing feature configuration.  The priority algebra renormalises
# automatically when a feature is unavailable (sensor failure) or newly added.
# ---------------------------------------------------------------------------
DYNAMIC_FEATURES = {
    "distance_m": {
        "type": "priority", "weight": 0.25,
        "min_val": 0.0, "max_val": 2000.0,
        "impact": "negative",          # closer = higher priority
    },
    "waste_level": {
        "type": "priority", "weight": 0.35,
        "min_val": 0.0, "max_val": 1.0,
        "impact": "positive",          # fuller = higher priority
    },
    "gas_level": {
        "type": "priority", "weight": 0.25,
        "min_val": 0.0, "max_val": 1.0,
        "impact": "positive",          # higher odour/methane = higher priority
    },
    "temperature": {
        "type": "priority", "weight": 0.10,
        "min_val": 10.0, "max_val": 40.0, "optimal": 25.0,
        "impact": "deviation",         # deviation from 25 C raises spoilage risk
    },
    "humidity": {
        "type": "priority", "weight": 0.05,
        "min_val": 50.0, "max_val": 100.0,
        "impact": "positive",
    },
    "traffic_density": {
        "type": "cost_multiplier", "routing_weight": 3.0,
        "min_val": 0.0, "max_val": 1.0,
        "weight": 0.10, "impact": "negative",
    },
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "[{levelname}] {name}: {message}", "style": "{"},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "simple"},
    },
    "loggers": {
        "bins": {"handlers": ["console"], "level": env("BINS_LOG_LEVEL", "INFO")},
    },
}


# ---------------------------------------------------------------------------
# Local settings override (never committed)
# ---------------------------------------------------------------------------
try:
    from .local_settings import *  # noqa: F401,F403
except ImportError:
    pass
