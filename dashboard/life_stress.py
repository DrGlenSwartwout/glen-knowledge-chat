"""Life Stress essence recommender — shared by the prod portal (app.py) and the
local biofield report app (biofield_local_app.py). Matches a client's E4L scan
emotion patterns to supportive Terrain Restore essences. Never raises."""
import json
import os

from dashboard import biofield_e4l
from dashboard import order_destination

_PRODUCTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "products.json")
_EMOTION_MAP_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data",
                                 "life_stress_emotion_map.json")


def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def slug_for_essence(name, products=None):
    """Lowercased exact-or-substring match of an essence NAME to a products.json
    slug. '' if unresolved or blank. Never raises. Mirrors app.py:_resolve_buy_slug."""
    q = (name or "").strip().lower()
    if not q:
        return ""
    if products is None:
        products = _load_json(_PRODUCTS_PATH)
    for slug, entry in (products.get("products") or {}).items():
        pn = str((entry or {}).get("name", "")).strip().lower()
        if not pn:
            continue
        if pn == q or (len(q) > 4 and (q in pn or pn in q)):
            return slug
    return ""
