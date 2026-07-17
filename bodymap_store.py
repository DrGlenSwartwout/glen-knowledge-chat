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


def build_payload(system):
    """Public payload for /body-map/data: valid zones only, each with meaning_display."""
    data = load_map(system)
    zones = []
    for z in data.get("zones", []):
        if not validate_zone(z)[0]:
            continue
        display = (z.get("meaning_glen") or "").strip() or z.get("meaning_standard", "")
        zones.append({**z, "meaning_display": display})
    return {
        "system": data.get("system", system),
        "reference_frame": data.get("reference_frame", "unit_circle"),
        "germ_layers": data.get("germ_layers", []),
        "zones": zones,
    }


def set_zone_overlay(system, zone_id, text):
    """Persist Glen's meaning_glen for one zone. Raises KeyError if the zone is unknown."""
    path = SYSTEMS[system]
    data = load_map(system)
    hit = False
    for z in data.get("zones", []):
        if z.get("id") == zone_id:
            z["meaning_glen"] = text
            hit = True
            break
    if not hit:
        raise KeyError(zone_id)
    _write(path, data)


ATLAS_CLUSTER_MAP = {
    "gut-digestive": {"system": "iridology", "zone": "iris-R-intestines"},
    "brain-nervous": {"system": "iridology", "zone": "iris-R-brain"},
    "circulation-cardio": {"system": "iridology", "zone": "iris-L-heart"},
    "structural-musculoskeletal": {"system": "iridology", "layer": "mesoderm"},
    "detox-drainage": {"system": "iridology", "zone": "iris-R-liver"},
    "metabolic-bloodsugar": {"system": "iridology", "zone": "iris-R-liver"},
    "immune": {"system": "iridology", "zone": "iris-R-intestines"},
    "eye-health": {"system": "iridology"},
}


def resolve_atlas_target(concept):
    """Body Map deep-link target for an Atlas concept, or None.
    Order: per-concept override (concept['body_map'] with a 'system') -> cluster map -> None."""
    if not isinstance(concept, dict):
        return None
    override = concept.get("body_map")
    if isinstance(override, dict) and override.get("system"):
        return override
    return ATLAS_CLUSTER_MAP.get(concept.get("cluster"))


def atlas_target_url(target):
    """Build the /body-map deep-link URL for a target dict. Returns '' when falsy/invalid."""
    if not isinstance(target, dict) or not target.get("system"):
        return ""
    from urllib.parse import urlencode
    params = {"system": target["system"]}
    if target.get("eye"):
        params["eye"] = target["eye"]
    if target.get("zone"):
        params["zone"] = target["zone"]
    elif target.get("layer"):
        params["layer"] = target["layer"]
    return "/body-map?" + urlencode(params)
