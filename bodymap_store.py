"""Body Map store: load/validate iris & sclera zone data. No Flask/Pinecone deps."""
import json
import os
import re
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
    "organs": DATA_DIR / "bodymap-organs.json",
    "skeleton": DATA_DIR / "bodymap-skeleton.json",
    "muscle": DATA_DIR / "bodymap-muscle.json",
    "dental": DATA_DIR / "bodymap-dental.json",
    "organclock": DATA_DIR / "bodymap-organclock.json",
    "nervous": DATA_DIR / "bodymap-nervous.json",
    "endocrine": DATA_DIR / "bodymap-endocrine.json",
    "respiratory": DATA_DIR / "bodymap-respiratory.json",
    "digestive": DATA_DIR / "bodymap-digestive.json",
    "cardiovascular": DATA_DIR / "bodymap-cardiovascular.json",
    "urogenital": DATA_DIR / "bodymap-urogenital.json",
}

_SEED_NAMES = ("bodymap-iridology.json", "bodymap-sclerology.json", "bodymap-ear.json",
               "bodymap-foot.json", "bodymap-hand.json", "bodymap-meridian.json", "bodymap-eav.json", "bodymap-neurotome.json", "bodymap-lymph.json", "bodymap-face.json", "bodymap-organs.json", "bodymap-skeleton.json", "bodymap-muscle.json", "bodymap-dental.json", "bodymap-organclock.json", "bodymap-nervous.json", "bodymap-endocrine.json", "bodymap-respiratory.json", "bodymap-digestive.json", "bodymap-cardiovascular.json", "bodymap-urogenital.json")
# NOTE: bodymap-tissue-layers.json is deliberately NOT in _SEED_NAMES. It is
# Glen-curated (the tissue-layer editor), so the boot-time force-reseed must never
# clobber his saved assignments. load_tissue_organs() falls back to the repo seed
# when no persisted copy exists yet (fresh disk), so a new deploy still ships the
# defaults; once Glen edits, his DATA_DIR copy persists across deploys.
# Systems overview catalog: display metadata for the landing page, grouped by
# category (order preserved). Every SYSTEMS key must appear exactly once here —
# test_system_catalog_covers_all_systems enforces it so the two can't drift.
SYSTEM_CATALOG = [
    # Eyes & reflex microsystems
    ("iridology", "Iridology (iris)", "Eyes & reflex microsystems", "The iris read as a map of organs and body systems, in radial germ-layer rings."),
    ("sclerology", "Sclerology (sclera)", "Eyes & reflex microsystems", "The red vessels in the white of the eye and what they signal."),
    ("ear", "Ear (auricular)", "Eyes & reflex microsystems", "The ear as an inverted homunculus — the whole body on the auricle."),
    ("foot", "Foot reflexology", "Eyes & reflex microsystems", "Reflex zones across the soles that correspond to the organs."),
    ("hand", "Hand reflexology", "Eyes & reflex microsystems", "Reflex zones across the palms that correspond to the organs."),
    ("face", "Facial diagnosis", "Eyes & reflex microsystems", "Five layers: TCM organ zones, acupoints, lymph, nerves, and EAV points."),
    # Meridian & energetic
    ("meridian", "Meridians (acupuncture)", "Meridian & energetic", "The fourteen acupuncture channels and their key points on the body."),
    ("eav", "Acupuncture & EAV points", "Meridian & energetic", "Jing-well terminals and Voll (EAV) points on the hands and feet."),
    ("dental", "Dental / meridian chart", "Meridian & energetic", "Each tooth and the meridian and organs it relates to."),
    ("organclock", "Chinese organ clock", "Meridian & energetic", "The 24-hour cycle of when each organ meridian peaks."),
    # Body systems
    ("organs", "Organ atlas", "Body systems", "The major internal organs at their positions on the whole body."),
    ("skeleton", "Skeleton", "Body systems", "The bones and joints, front and back."),
    ("muscle", "Muscles", "Body systems", "The major muscle groups, front and back."),
    ("nervous", "Nervous system", "Body systems", "Brain, spinal cord, peripheral nerves and autonomic ganglia."),
    ("endocrine", "Endocrine system", "Body systems", "The hormone-producing glands, head to pelvis."),
    ("respiratory", "Respiratory system", "Body systems", "The airway, lungs by lobe, pleura and diaphragm."),
    ("digestive", "Digestive system", "Body systems", "The GI tract from mouth to rectum plus the accessory organs."),
    ("cardiovascular", "Cardiovascular system", "Body systems", "The heart, great vessels and the major arteries and veins."),
    ("urogenital", "Urogenital system", "Body systems", "The urinary tract and reproductive organs, male and female."),
    ("lymph", "Immune, lymphatic & connective tissue", "Body systems", "Lymph nodes and ducts, the immune organs, and connective tissue."),
    # Nerve maps
    ("neurotome", "Dermatomes (neurotomes)", "Nerve maps", "The skin regions each mapped to their spinal nerve segment."),
]


def system_catalog():
    """The overview catalog as a flat ordered list of dicts (id, name, category,
    description). The landing page groups by category, preserving order."""
    return [{"id": s, "name": n, "category": c, "description": d}
            for (s, n, c, d) in SYSTEM_CATALOG]


# Dr. Glen Swartwout's 5 Embryological Tissue Layers (the tissue 5 C's), deepest ->
# most superficial, each with two sub-layers. FIXED taxonomy; organ assignments are
# editable (data/bodymap-tissue-layers.json). Drives the embryological depth-peel.
TISSUE_LAYERS = [
    {"id": "compression", "name": "Compression", "depth": 1,
     "sublayers": [{"id": "urogenital", "name": "Urogenital"}, {"id": "muscle", "name": "Muscle"}]},
    {"id": "connection", "name": "Connection", "depth": 2,
     "sublayers": [{"id": "bone", "name": "Bone/Connective Tissue"}, {"id": "cardiovascular", "name": "Cardiovascular/Immune"}]},
    {"id": "conversion", "name": "Conversion", "depth": 3,
     "sublayers": [{"id": "digestive", "name": "Digestive"}, {"id": "respiratory", "name": "Respiratory"}]},
    {"id": "communication", "name": "Communication", "depth": 4,
     "sublayers": [{"id": "nerve", "name": "Nerve"}, {"id": "endocrine", "name": "Endocrine"}]},
    {"id": "containment", "name": "Containment", "depth": 5,
     "sublayers": [{"id": "oroderm", "name": "Oroderm"}, {"id": "integument", "name": "Integument"}]},
]
_SUBLAYER_TO_LAYER = {sl["id"]: L["id"] for L in TISSUE_LAYERS for sl in L["sublayers"]}
_VALID_SUBLAYERS = set(_SUBLAYER_TO_LAYER)
_TISSUE_FILE = "bodymap-tissue-layers.json"


def sublayer_to_layer(sublayer_id):
    """The layer id a sub-layer belongs to (None if unknown)."""
    return _SUBLAYER_TO_LAYER.get(sublayer_id)


def _tissue_path():
    """Persisted organ-assignment file (DATA_DIR when writable, else the repo seed)."""
    return DATA_DIR / _TISSUE_FILE


def load_tissue_organs():
    """The organ -> sub-layer assignments (list of {id, name, sublayer, keywords}).
    Reads the persisted copy, falling back to the repo seed. [] on any failure."""
    for path in (_tissue_path(), REPO_DATA / _TISSUE_FILE):
        try:
            if path.exists():
                return (json.loads(path.read_text(encoding="utf-8")) or {}).get("organs", [])
        except (OSError, ValueError):
            continue
    return []


def tissue_catalog():
    """{layers: TISSUE_LAYERS, organs: [...]} — everything the editor and the
    depth-peel need in one call."""
    return {"layers": TISSUE_LAYERS, "organs": load_tissue_organs()}


def set_organ_sublayer(organ_id, sublayer_id):
    """Reassign one organ to a sub-layer and persist. Returns the updated organ dict.
    Raises KeyError (unknown organ) / ValueError (unknown sub-layer)."""
    if sublayer_id not in _VALID_SUBLAYERS:
        raise ValueError(f"unknown sub-layer: {sublayer_id}")
    organs = load_tissue_organs()
    for o in organs:
        if o.get("id") == organ_id:
            o["sublayer"] = sublayer_id
            _tissue_path().write_text(json.dumps({"organs": organs}, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
            return o
    raise KeyError(organ_id)


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


# Extra organ terms that a finding name may use but a zone's anatomy spells
# differently. Keyed by the stemmed term (see _stem); values are stemmed phrases
# to ALSO try. One-directional: the zone's own spelling matches on its own.
_ZONE_SYNONYMS = {
    "colon": ["large intestine", "colon"],
    "large intestine": ["colon", "large intestine"],
    "large bowel": ["large intestine", "colon"],
    "bowel": ["large intestine", "small intestine", "colon"],
    "gall bladder": ["gallbladder"],
    "suprarenal": ["adrenal"],
}

# E4L finding names carry non-anatomical noise words ("Liver Driver", "Heart
# Imprinter"). Stripping them lets the remaining organ phrase match, without
# splitting genuine two-word organ names ("large intestine") into a shared word
# ("intestine") that would cross-match the wrong organ.
_NOISE_WORDS = {"driver", "imprinter", "the", "and", "region", "of", "a", "an"}


def _stem(s):
    """Lowercase and singularize each word so an organ name matches whether the
    finding or the zone spells it singular or plural: '-ies'->'-y' (ovaries->ovary,
    arteries->artery) then a trailing '-s' ('Lungs'->'lung'). Matching-only; not display."""
    s = re.sub(r"(\w)ies\b", r"\1y", (s or "").strip().lower())
    return re.sub(r"(\w)s\b", r"\1", s)


def zone_ids(system, side=None):
    """All zone ids in `system` (optionally restricted to one view/side). Used to
    light a whole system for a system-level finding (e.g. a 'Bone' finding lights
    the entire skeleton). Never raises on a missing system -> []."""
    try:
        zones = load_map(system).get("zones", [])
    except (KeyError, FileNotFoundError, ValueError):
        return []
    if side:
        zones = [z for z in zones if (z.get("side") or z.get("eye")) == side]
    return [z.get("id") for z in zones if z.get("id")]


def resolve_finding_zones(system, names, side=None):
    """Map a client's finding organ/system names to Body Map zone ids in `system`.

    Pure (no DB): matches each name against every zone's `anatomy` (and its
    optional `meridian_organs` list, so a tooth lights for its associated organs)
    on a word-boundary, plural-insensitive basis, so a 'Liver' finding lights every
    zone whose anatomy names the liver. `side` (e.g. 'diagnosis') restricts to
    one view/layer. Returns {"zones": [ordered unique ids], "by_name": {name: [ids]}}.
    A name that matches nothing is simply absent from by_name. Never raises on a
    missing system -> empty result."""
    try:
        zones = load_map(system).get("zones", [])
    except (KeyError, FileNotFoundError, ValueError):
        return {"zones": [], "by_name": {}}
    if side:
        zones = [z for z in zones if (z.get("side") or z.get("eye")) == side]
    zdata = [(z.get("id"), _stem(z.get("anatomy", "") + " " + " ".join(z.get("meridian_organs") or [])))
             for z in zones]
    by_name, ordered, seen = {}, [], set()
    for raw in (names or []):
        base = _stem(raw)
        if not base:
            continue
        # Candidates: the whole stemmed phrase, the phrase with E4L noise words
        # removed ("Liver Driver" -> "liver"), and synonyms of both. A genuine
        # two-word organ name ("large intestine") is matched as a phrase, never
        # split into a shared word ("intestine") that would cross-match a sibling
        # organ ("small intestine"). Single-word organs still work via `base`.
        words = [w for w in base.split() if w not in _NOISE_WORDS]
        phrase = " ".join(words)
        cands = {base, phrase}
        if len(words) == 1:
            cands.add(words[0])
        for t in list(cands):
            cands.update(_ZONE_SYNONYMS.get(t, []))
        pats = [re.compile(r"\b" + re.escape(t) + r"\b") for t in cands if t]
        hits = [zid for zid, anat in zdata if zid and any(p.search(anat) for p in pats)]
        if hits:
            by_name[str(raw)] = hits
            for zid in hits:
                if zid not in seen:
                    seen.add(zid)
                    ordered.append(zid)
    return {"zones": ordered, "by_name": by_name}
