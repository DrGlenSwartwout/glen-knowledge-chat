"""Reader for the curated organ/meridian -> characteristic STRESS-FACTOR layer
(data/organ_stress_factors.json). Factors span four categories (toxin, microbe,
emotional, physical); each carries an evidence tier and a citation. Meridian
entries inherit their organ's factors via the file's `entry_keys` map. Pure;
empty on any load failure."""
import json
import os

_REPO_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_FILENAME = "organ_stress_factors.json"


def _path(path=None):
    if path:
        return path
    d = os.environ.get("DATA_DIR")
    if d and os.path.exists(os.path.join(d, _FILENAME)):
        return os.path.join(d, _FILENAME)
    p = os.path.join(_REPO_DATA, _FILENAME)
    return p if os.path.exists(p) else None


def load(path=None):
    p = _path(path)
    if not p:
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def stressors_for(entry_name, data=None):
    """Stress-factor records ([{category, factor, tier, note, source, url}]) for an
    organ/meridian entry name, or [] when the entry has no curated association."""
    data = data if data is not None else load()
    key = (data.get("entry_keys") or {}).get((entry_name or "").strip())
    if not key:
        return []
    return (data.get("factors") or {}).get(key, [])
