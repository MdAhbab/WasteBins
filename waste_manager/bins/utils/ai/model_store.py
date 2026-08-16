"""
Backward-compatible facade over :mod:`bins.services.models_registry`.

Existing views and management commands import these names; the implementation
now lives in the service layer, which adds caching, artefact hashing and the
continual-learning state that this module never had.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from django.conf import settings
from joblib import dump, load

from bins.services.models_registry import (get_model_version, load_forward_bundle,
                                           load_forward_meta, save_forward_bundle)

__all__ = [
    "model_path", "meta_path", "save_model", "load_model", "get_model_version",
    "load_meta", "forward_path", "forward_meta_path", "save_forward_bundle",
    "load_forward_bundle", "load_forward_meta",
]


def model_path() -> Path:
    return Path(settings.MODEL_FILENAME)


def meta_path() -> Path:
    return Path(settings.MODEL_META_FILENAME)


def forward_path() -> Path:
    return Path(settings.FORWARD_MODEL_FILENAME)


def forward_meta_path() -> Path:
    return Path(settings.FORWARD_MODEL_META_FILENAME)


def save_model(model, meta: Dict) -> None:
    """Persist the legacy random-forest artefact (retained for comparison runs)."""
    path = model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, path)
    meta_path().write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_model():
    path = model_path()
    if not path.exists():
        return None
    try:
        return load(path)
    except Exception:
        return None


def load_meta() -> Dict:
    path = meta_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
