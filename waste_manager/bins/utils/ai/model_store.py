import json
from pathlib import Path
from django.conf import settings
from joblib import dump, load

def model_path():
    return settings.MODEL_FILENAME

def meta_path():
    return settings.MODEL_META_FILENAME

def save_model(model, meta: dict):
    dump(model, model_path())
    with open(meta_path(), 'w') as f:
        json.dump(meta, f)

def load_model():
    p = model_path()
    if Path(p).exists():
        return load(p)
    return None

def get_model_version():
    mp = meta_path()
    if Path(mp).exists():
        with open(mp, 'r') as f:
            data = json.load(f)
            return data.get('version', 'unknown')
    return 'unknown'

def load_meta():
    mp = meta_path()
    if Path(mp).exists():
        with open(mp, 'r') as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Forward-looking bundle (regressor + quantiles + calibrated hazard classifier)
# ---------------------------------------------------------------------------
def forward_path():
    return settings.FORWARD_MODEL_FILENAME


def forward_meta_path():
    return settings.FORWARD_MODEL_META_FILENAME


def save_forward_bundle(bundle: dict, meta: dict):
    """bundle keys: 'regressor', 'q10', 'q50', 'q90', 'classifier', 'features'."""
    dump(bundle, forward_path())
    with open(forward_meta_path(), 'w') as f:
        json.dump(meta, f)


def load_forward_bundle():
    p = forward_path()
    if Path(p).exists():
        return load(p)
    return None


def load_forward_meta():
    mp = forward_meta_path()
    if Path(mp).exists():
        with open(mp, 'r') as f:
            return json.load(f)
    return {}