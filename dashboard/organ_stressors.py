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


def for_pattern_organs(organ_names, clinical_catalog, data=None):
    """Aggregate + dedupe the stress factors of the organs an E4L pattern involves.

    `organ_names` are the E4L organ structure names (e.g. "Heart", "Adrenal Gland").
    Each is canon-matched to a clinical Organ entry (same canon/alias logic as the
    cross-links) and its factors are collected. Returns records in first-seen order,
    deduped by (category, factor), each annotated with the `organs` it came from.
    These are the organs' characteristic stressors — reference context, not a claim
    that the pattern is caused by them."""
    from dashboard import glossary_crosslinks as _gx
    data = data if data is not None else load()
    canon_to_name = {}
    for d in (clinical_catalog or {}).get("dimensions", []):
        if d.get("key") == "organs":
            for e in d.get("entries", []):
                canon_to_name.setdefault(_gx.canon(e.get("name", "")), e.get("name"))
    agg, order = {}, []
    for on in (organ_names or []):
        cn = canon_to_name.get(_gx.canon(on))
        if not cn:
            continue
        for f in stressors_for(cn, data):
            k = (f.get("category"), f.get("factor"))
            if k not in agg:
                rec = dict(f)
                rec["organs"] = []
                agg[k] = rec
                order.append(k)
            if on not in agg[k]["organs"]:
                agg[k]["organs"].append(on)
    return [agg[k] for k in order]
