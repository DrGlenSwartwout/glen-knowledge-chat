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
    "ear": DATA_DIR / "bodymap-ear.json",
    "foot": DATA_DIR / "bodymap-foot.json",
    "hand": DATA_DIR / "bodymap-hand.json",
    "meridian": DATA_DIR / "bodymap-meridian.json",
    "eav": DATA_DIR / "bodymap-eav.json",
    "neurotome": DATA_DIR / "bodymap-neurotome.json",
    "lymph": DATA_DIR / "bodymap-lymph.json",
    "face": DATA_DIR / "bodymap-face.json",
}

_SEED_NAMES = ("bodymap-iridology.json", "bodymap-sclerology.json", "bodymap-ear.json",
               "bodymap-foot.json", "bodymap-hand.json", "bodymap-meridian.json", "bodymap-eav.json", "bodymap-neurotome.json", "bodymap-lymph.json", "bodymap-face.json")
_REQUIRED_COMMON = ("id", "anatomy", "meaning_standard")


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
    """Return (ok, error_message_or_None). Accepts sector zones (iris) and point zones (ear)."""
    if not isinstance(z, dict):
        return False, "zone must be an object"
    for key in _REQUIRED_COMMON:
        if key not in z or z.get(key) is None:
            return False, f"missing required field: {key}"
    # `side`/`eye` is the laterality OR view/layer selector (left/right, front/back/side,
    # hand/foot, or a named map layer like diagnosis/acu/lymph). Any non-empty string.
    _side = z.get("side") or z.get("eye")
    if not isinstance(_side, str) or not _side.strip():
        return False, "side/eye (laterality or view) must be a non-empty string"
    if not (z.get("group") or z.get("germ_layer")):
        return False, "missing grouping (group or germ_layer)"
    geo = z.get("geometry") or {}
    gtype = geo.get("type") or ("sector" if ("radial" in z and "sector" in z) else None)
    if gtype == "ellipse":
        cx, cy, rx, ry = geo.get("cx"), geo.get("cy"), geo.get("rx"), geo.get("ry")
        if not all(isinstance(v, (int, float)) for v in (cx, cy, rx, ry)):
            return False, "ellipse cx/cy/rx/ry must be numbers"
        if not (0.0 <= float(cx) <= 1.0 and 0.0 <= float(cy) <= 1.0):
            return False, "ellipse cx/cy must be in [0,1]"
        if not (0.0 < float(rx) <= 1.0 and 0.0 < float(ry) <= 1.0):
            return False, "ellipse rx/ry must be in (0,1]"
        return True, None
    if gtype == "polygon":
        pts = geo.get("points")
        if not isinstance(pts, list) or len(pts) < 3:
            return False, "geometry polygon needs >= 3 points"
        for p in pts:
            if not (isinstance(p, (list, tuple)) and len(p) == 2
                    and all(isinstance(v, (int, float)) for v in p)
                    and 0.0 <= float(p[0]) <= 1.0 and 0.0 <= float(p[1]) <= 1.0):
                return False, "polygon points must be [x,y] pairs in [0,1]"
        return True, None
    if gtype == "point":
        x, y = geo.get("x"), geo.get("y")
        if not all(isinstance(v, (int, float)) for v in (x, y)):
            return False, "geometry point x/y must be numbers"
        if not (0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0):
            return False, "geometry point x/y must be in [0,1]"
        return True, None
    if gtype == "path":
        d = geo.get("d")
        if not isinstance(d, str) or not d.strip():
            return False, "geometry path needs a non-empty 'd' string"
        return True, None
    if gtype == "sector":
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
    return False, "unknown geometry type"


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
        "groups": data.get("groups", []),
        "outline": data.get("outline", ""),
        "outlines": data.get("outlines", {}),
        "outline_side": data.get("outline_side", ""),
        "anchors": data.get("anchors", []),
        "side_noun": data.get("side_noun", ""),
        "group_noun": data.get("group_noun", ""),
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


def upsert_zone(system, zone):
    """Add or replace a zone (matched by id) in the system's seed. Raises ValueError if invalid."""
    ok, err = validate_zone(zone)
    if not ok:
        raise ValueError(err)
    path = SYSTEMS[system]
    data = load_map(system)
    zones = data.setdefault("zones", [])
    for i, z in enumerate(zones):
        if z.get("id") == zone.get("id"):
            zones[i] = zone
            break
    else:
        zones.append(zone)
    _write(path, data)


def delete_zone(system, zone_id):
    """Remove a zone by id. Raises KeyError if not present."""
    path = SYSTEMS[system]
    data = load_map(system)
    kept = [z for z in data.get("zones", []) if z.get("id") != zone_id]
    if len(kept) == len(data.get("zones", [])):
        raise KeyError(zone_id)
    data["zones"] = kept
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
