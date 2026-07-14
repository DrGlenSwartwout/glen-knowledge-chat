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


def _weight(rank):
    """Higher rank number = lower priority. Rank 1 dominates. None ranks get a small weight."""
    try:
        r = int(rank)
        return 1.0 / r if r > 0 else 0.5
    except (TypeError, ValueError):
        return 0.5


def recommend(email, today, *, db_path=None, products=None, emotion_map=None, max_emotions=2):
    """Match a client's scan emotion patterns to supportive Terrain Restore essences.
    Returns {"label","patterns","items"} or None. Never raises."""
    try:
        scan = biofield_e4l.scan_context(email, today, db_path=db_path)
        if not scan or not scan.get("found"):
            return None
        findings = scan.get("findings") or []
        codes = [f.get("code") for f in findings if f.get("code")]
        emo_by_code = biofield_e4l.emotions_for_codes(codes, db_path=db_path)
        if not emo_by_code:
            return None
        # aggregate emotion score, weighted by each finding's rank
        scores = {}
        for f in findings:
            for emo in emo_by_code.get(f.get("code"), []):
                scores[emo] = scores.get(emo, 0.0) + _weight(f.get("rank"))
        if not scores:
            return None
        top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:max_emotions]
        if emotion_map is None:
            emotion_map = _load_json(_EMOTION_MAP_PATH)
        items, seen = [], set()
        for emo, _score in top:
            for name in (emotion_map.get(emo) or []):
                slug = slug_for_essence(name, products)
                if not slug or slug in seen:
                    continue
                seen.add(slug)
                items.append({"name": name,
                              "url": order_destination.destination_for(slug),
                              "note": f"for the {emo.lower()} pattern in your scan"})
        if not items:
            return None
        return {"label": "Life Stress",
                "patterns": [{"emotion": e, "score": round(s, 4)} for e, s in top],
                "items": items}
    except Exception:
        return None
