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