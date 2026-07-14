"""Reader over the curated clinical-theory catalogue
(data/clinical_theory_catalog.json) — a manual snapshot of clinicaltheory.com's
dimension pages (Organs, Meridians, Miasms, Chemistry). Pure; returns empty
structures on any load failure. Remedy-link -> product mapping is deferred."""
import json
import os

_REPO_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_FILENAME = "clinical_theory_catalog.json"


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
        return {"dimensions": []}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f) or {"dimensions": []}
    except Exception:
        return {"dimensions": []}


def dimensions(catalog=None):
    """Lightweight list for the hub — no entries."""
    cat = catalog if catalog is not None else load()
    out = []
    for d in cat.get("dimensions", []):
        out.append({
            "key": d.get("key", ""), "title": d.get("title", ""),
            "blurb": d.get("blurb", ""),
            "entry_count": d.get("entry_count", len(d.get("entries", []))),
        })
    return out


def get_dimension(key, catalog=None):
    """Full dimension record (with entries) or None."""
    cat = catalog if catalog is not None else load()
    for d in cat.get("dimensions", []):
        if d.get("key") == key:
            return d
    return None
