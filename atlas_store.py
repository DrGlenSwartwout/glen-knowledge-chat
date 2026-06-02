"""Knowledge Atlas store: load/validate/approve concept data. No Flask/Pinecone deps."""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
CONCEPTS_PATH = DATA_DIR / "atlas-concepts.json"
PENDING_PATH = DATA_DIR / "atlas-pending.json"
VIDEOS_PATH = DATA_DIR / "atlas-videos.json"

_REQUIRED = ("id", "label", "summary", "cluster", "coords", "links", "status")


def validate_concept(c):
    """Return (ok, error_message_or_None)."""
    if not isinstance(c, dict):
        return False, "concept must be an object"
    for key in _REQUIRED:
        if key not in c:
            return False, f"missing required field: {key}"
    coords = c.get("coords") or {}
    for axis in ("x", "y"):
        v = coords.get(axis)
        if not isinstance(v, (int, float)) or not (0.0 <= float(v) <= 1.0):
            return False, f"coords.{axis} must be a number in [0,1]"
    if not isinstance(c.get("links"), list):
        return False, "links must be a list"
    return True, None


def _read(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_concepts():
    return _read(CONCEPTS_PATH, {"version": "", "concepts": []})


def load_pending():
    return _read(PENDING_PATH, {"version": "", "concepts": []})


def load_videos():
    return _read(VIDEOS_PATH, {"version": "", "videos": []}).get("videos", [])


def build_graph():
    """Public payload for /atlas/data: validated live concepts + hierarchy index."""
    concepts = [c for c in load_concepts().get("concepts", [])
                if c.get("status") == "live" and validate_concept(c)[0]]
    hierarchy = {}
    for c in concepts:
        hierarchy.setdefault(c.get("parent") or "ungrouped", []).append(c["id"])
    return {"concepts": concepts, "hierarchy": hierarchy}


def approve_concept(concept_id):
    pending = load_pending()
    live = load_concepts()
    keep, moved = [], None
    for c in pending.get("concepts", []):
        if c.get("id") == concept_id:
            moved = {**c, "status": "live"}
        else:
            keep.append(c)
    if moved is None:
        raise KeyError(concept_id)
    live_concepts = [c for c in live.get("concepts", []) if c.get("id") != concept_id]
    live_concepts.append(moved)
    live["concepts"] = live_concepts
    pending["concepts"] = keep
    _write(CONCEPTS_PATH, live)
    _write(PENDING_PATH, pending)


def reject_concept(concept_id):
    pending = load_pending()
    pending["concepts"] = [c for c in pending.get("concepts", []) if c.get("id") != concept_id]
    _write(PENDING_PATH, pending)
