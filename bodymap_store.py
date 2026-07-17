"""Body Map store: load/validate iris & sclera zone data. No Flask/Pinecone deps."""
import json
import os
from pathlib import Path

REPO_DATA = Path(__file__).resolve().parent / "data"


def _persist_dir():
    d = Path(os.environ.get("DATA_DIR") or "/data")
    if d.is_dir() and os.access(d, os.W_OK):
        return d
    return REPO_DATA


DATA_DIR = _persist_dir()

SYSTEMS = {
    "iridology": DATA_DIR / "bodymap-iridology.json",
    "sclerology": DATA_DIR / "bodymap-sclerology.json",
}

_SEED_NAMES = ("bodymap-iridology.json", "bodymap-sclerology.json")
_REQUIRED = ("id", "eye", "germ_layer", "radial", "sector", "anatomy", "meaning_standard")


def reseed_from_repo(force=False):
    """Copy git-committed seeds onto the persistent dir on first boot; never clobber curation."""
    if DATA_DIR == REPO_DATA:
        return False
    seeded = False
    for fname in _SEED_NAMES:
        dst, src = DATA_DIR / fname, REPO_DATA / fname
        if src.exists() and (force or not dst.exists()):
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            seeded = True
    return seeded


def validate_zone(z):
    """Return (ok, error_message_or_None)."""
    if not isinstance(z, dict):
        return False, "zone must be an object"
    for key in _REQUIRED:
        if key not in z:
            return False, f"missing required field: {key}"
    if z.get("eye") not in ("right", "left"):
        return False, "eye must be 'right' or 'left'"
    radial = z.get("radial") or {}
    ri, ro = radial.get("r_inner"), radial.get("r_outer")
    if not all(isinstance(v, (int, float)) for v in (ri, ro)):
        return False, "radial.r_inner/r_outer must be numbers"
    if not (0.0 <= float(ri) < float(ro) <= 3.0):
        return False, "radial must satisfy 0 <= r_inner < r_outer <= 3"
    sector = z.get("sector") or {}
    s, e = sector.get("start_deg"), sector.get("end_deg")
    if not all(isinstance(v, (int, float)) for v in (s, e)):
        return False, "sector.start_deg/end_deg must be numbers"
    if not (0.0 <= float(s) < float(e) <= 360.0):
        return False, "sector must satisfy 0 <= start_deg < end_deg <= 360"
    return True, None


def _read(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_map(system):
    path = SYSTEMS.get(system)
    if path is None:
        raise KeyError(system)
    return _read(path, {"system": system, "reference_frame": "unit_circle",
                        "germ_layers": [], "zones": []})
