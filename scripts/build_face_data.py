#!/usr/bin/env python3
"""Build data/bodymap-face.json — the face as a multi-layer microsystem.

One front-view face outline (oval + ears + eyes/brows/nose/mouth features), with
FIVE toggleable map layers via the Side control:
  * diagnosis  — Chinese facial-diagnosis organ zones (5 TCM elements), ellipses
  * acu        — facial acupuncture points
  * lymph      — facial lymph nodes + drainage pathways
  * nerve      — trigeminal dermatomes V1/V2/V3, polygons
  * eav        — facial EAV / Voll measurement points

Paired items are emitted on both sides; midline items single. Photo-overlay
anchors (hairline + chin) are view-agnostic. Positions are a standards-based
rendering for Glen's clinical refinement, not gospel.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def catmull_rom_closed(points):
    n = len(points)
    d = f"M{points[0][0]:.4f} {points[0][1]:.4f} "
    for i in range(n):
        p0, p1, p2, p3 = points[(i - 1) % n], points[i], points[(i + 1) % n], points[(i + 2) % n]
        d += (f"C{p1[0] + (p2[0]-p0[0])/6:.4f} {p1[1] + (p2[1]-p0[1])/6:.4f} "
              f"{p2[0] - (p3[0]-p1[0])/6:.4f} {p2[1] - (p3[1]-p1[1])/6:.4f} "
              f"{p2[0]:.4f} {p2[1]:.4f} ")
    return d.strip() + " Z"


FACE = [
    (0.500, 0.060), (0.590, 0.075), (0.665, 0.115), (0.710, 0.210), (0.725, 0.340),
    (0.755, 0.430), (0.750, 0.560), (0.695, 0.660), (0.625, 0.800), (0.545, 0.895),
    (0.500, 0.920), (0.455, 0.895), (0.375, 0.800), (0.305, 0.660), (0.250, 0.560),
    (0.245, 0.430), (0.275, 0.340), (0.290, 0.210), (0.335, 0.115), (0.410, 0.075),
]
FEATURES = " ".join([
    "M0.330 0.420 C0.345 0.408 0.415 0.408 0.430 0.420 C0.415 0.432 0.345 0.432 0.330 0.420 Z",
    "M0.570 0.420 C0.585 0.408 0.655 0.408 0.670 0.420 C0.655 0.432 0.585 0.432 0.570 0.420 Z",
    "M0.325 0.370 C0.350 0.352 0.410 0.352 0.435 0.368",
    "M0.565 0.368 C0.590 0.352 0.650 0.352 0.675 0.370",
    "M0.500 0.400 L0.500 0.585 M0.460 0.605 C0.478 0.622 0.522 0.622 0.540 0.605",
    "M0.435 0.725 C0.470 0.748 0.530 0.748 0.565 0.725",
])

GROUPS = [
    {"id": "wood", "label": "Wood — Liver / Gallbladder"},
    {"id": "fire", "label": "Fire — Heart / Small Intestine"},
    {"id": "earth", "label": "Earth — Spleen / Stomach"},
    {"id": "metal", "label": "Metal — Lung / Large Intestine"},
    {"id": "water", "label": "Water — Kidney / Bladder"},
    {"id": "acu", "label": "Acupuncture points"},
    {"id": "lymph-node", "label": "Lymph nodes"},
    {"id": "lymph-vessel", "label": "Lymph drainage"},
    {"id": "trigeminal", "label": "Trigeminal dermatomes"},
    {"id": "eav", "label": "EAV / Voll points"},
]

# ---- diagnosis (ellipse): (slug, name, group, bilateral, cx, cy, rx, ry) ----
DIAGNOSIS = [
    ("glabella", "Liver (glabella)", "wood", False, 0.500, 0.335, 0.030, 0.030),
    ("nose-bridge", "Liver / gallbladder (nose bridge)", "wood", False, 0.500, 0.425, 0.024, 0.045),
    ("temple", "Gallbladder (temple)", "wood", True, 0.315, 0.280, 0.035, 0.055),
    ("eyes", "Liver (eyes)", "wood", True, 0.380, 0.420, 0.050, 0.026),
    ("forehead-mind", "Heart / mind (upper forehead)", "fire", False, 0.500, 0.130, 0.100, 0.035),
    ("nose-tip", "Heart (nose tip)", "fire", False, 0.500, 0.600, 0.036, 0.030),
    ("nose-mid", "Spleen / pancreas (mid nose)", "earth", False, 0.500, 0.520, 0.030, 0.035),
    ("upper-lip", "Spleen (upper lip)", "earth", False, 0.500, 0.680, 0.036, 0.020),
    ("mouth", "Stomach (mouth)", "earth", False, 0.500, 0.728, 0.052, 0.024),
    ("forehead-li", "Large intestine (mid forehead)", "metal", False, 0.500, 0.200, 0.090, 0.028),
    ("cheek", "Lungs (cheek)", "metal", True, 0.345, 0.575, 0.060, 0.060),
    ("nostril", "Bronchi (nostril wing)", "metal", True, 0.445, 0.610, 0.026, 0.022),
    ("forehead-bladder", "Bladder (forehead hairline)", "water", False, 0.500, 0.255, 0.080, 0.026),
    ("under-eye", "Kidney (under-eye)", "water", True, 0.400, 0.470, 0.045, 0.026),
    ("jaw", "Kidney (jaw)", "water", True, 0.360, 0.780, 0.042, 0.050),
    ("ear", "Kidney (ear)", "water", True, 0.748, 0.510, 0.022, 0.045),
    ("chin", "Kidney / bladder / hormones (chin)", "water", False, 0.500, 0.850, 0.062, 0.040),
]

# ---- acupuncture (point): (slug, name, bilateral, x, y) ----
ACU = [
    ("yintang", "Yintang (EX-HN3)", False, 0.500, 0.335),
    ("gb14", "GB14 Yangbai", True, 0.420, 0.285),
    ("bl2", "BL2 Zanzhu", True, 0.445, 0.365),
    ("yuyao", "Yuyao (EX-HN4)", True, 0.380, 0.360),
    ("taiyang", "Taiyang (EX-HN5)", True, 0.300, 0.420),
    ("st1", "ST1 Chengqi", True, 0.400, 0.462),
    ("st2", "ST2 Sibai", True, 0.400, 0.498),
    ("si18", "SI18 Quanliao", True, 0.365, 0.535),
    ("li20", "LI20 Yingxiang", True, 0.455, 0.605),
    ("st3", "ST3 Juliao", True, 0.430, 0.615),
    ("gv26", "GV26 Renzhong", False, 0.500, 0.665),
    ("st4", "ST4 Dicang", True, 0.425, 0.725),
    ("cv24", "CV24 Chengjiang", False, 0.500, 0.775),
    ("st6", "ST6 Jiache", True, 0.345, 0.720),
    ("st7", "ST7 Xiaguan", True, 0.335, 0.545),
]

# ---- lymph nodes (point) ----
LYMPH_NODES = [
    ("preauricular", "Preauricular nodes", True, 0.712, 0.440),
    ("parotid", "Parotid nodes", True, 0.685, 0.500),
    ("buccal", "Buccal / facial nodes", True, 0.400, 0.660),
    ("submandibular", "Submandibular nodes", True, 0.420, 0.820),
    ("submental", "Submental nodes", False, 0.500, 0.862),
]
# ---- lymph drainage (path): (slug, name, bilateral, [waypoints]) ----
LYMPH_PATHS = [
    ("drain-pre", "Drainage to preauricular", True, [(0.500, 0.520), (0.610, 0.480), (0.700, 0.450)]),
    ("drain-sub", "Drainage to submandibular", True, [(0.470, 0.680), (0.445, 0.780), (0.425, 0.815)]),
]

# ---- trigeminal dermatomes (polygon): (slug, name, [points]) — midline, single ----
NERVE = [
    ("v1", "V1 (ophthalmic)", [[0.280, 0.055], [0.720, 0.055], [0.700, 0.300], [0.660, 0.400],
                               [0.340, 0.400], [0.300, 0.300]]),
    ("v2", "V2 (maxillary)", [[0.340, 0.400], [0.660, 0.400], [0.640, 0.550], [0.600, 0.670],
                              [0.400, 0.670], [0.360, 0.550]]),
    ("v3", "V3 (mandibular)", [[0.400, 0.670], [0.600, 0.670], [0.620, 0.800], [0.550, 0.900],
                               [0.450, 0.900], [0.380, 0.800]]),
]

# ---- EAV / Voll facial points (point) ----
EAV = [
    ("eav-st", "Stomach vessel (EAV)", True, 0.400, 0.500),
    ("eav-li", "Large-intestine vessel (EAV)", True, 0.455, 0.600),
    ("eav-gb", "Gallbladder vessel (EAV)", True, 0.300, 0.420),
    ("eav-bl", "Bladder vessel (EAV)", True, 0.445, 0.370),
    ("eav-si", "Small-intestine vessel (EAV)", True, 0.360, 0.530),
    ("eav-te", "Triple-energizer vessel (EAV)", True, 0.700, 0.430),
    ("eav-cv", "Conception vessel (EAV)", False, 0.500, 0.670),
    ("eav-gv", "Governing vessel (EAV)", False, 0.500, 0.860),
    ("eav-lymph", "Lymph vessel (EAV)", True, 0.360, 0.660),
    ("eav-nerve", "Nerve-degeneration vessel (EAV)", True, 0.420, 0.285),
]


def _base(zid, view, group, geometry, name):
    return {
        "id": zid, "side": view, "bilateral": False, "group": group,
        "geometry": geometry, "anatomy": name,
        "meaning_standard": f"{name} ({view} layer).",
        "meaning_glen": "",
        "layers": {"embryological_depth": None, "stress_affirmation": None, "touch_for_health": None},
    }


def _pair(zones, zid, view, group, geo_fn, x, y, name, bilateral):
    zones.append(_base(f"{zid}-R" if bilateral else zid, view, group, geo_fn(x), name))
    if bilateral:
        zones.append(_base(f"{zid}-L", view, group, geo_fn(round(1 - x, 4)), name))


def main():
    zones = []
    for slug, name, group, bil, cx, cy, rx, ry in DIAGNOSIS:
        _pair(zones, f"face-{slug}", "diagnosis", group,
              lambda xx: {"type": "ellipse", "cx": xx, "cy": cy, "rx": rx, "ry": ry}, cx, cy, name, bil)
    for slug, name, bil, x, y in ACU:
        _pair(zones, f"face-acu-{slug}", "acu", "acu",
              lambda xx: {"type": "point", "x": xx, "y": y}, x, y, name, bil)
    for slug, name, bil, x, y in LYMPH_NODES:
        _pair(zones, f"face-ln-{slug}", "lymph", "lymph-node",
              lambda xx: {"type": "point", "x": xx, "y": y}, x, y, name, bil)
    for slug, name, bil, wps in LYMPH_PATHS:
        def dpath(w):
            return {"type": "path", "d": "M" + " L".join(f"{a:.4f} {b:.4f}" for a, b in w)}
        zones.append(_base(f"face-lp-{slug}-R" if bil else f"face-lp-{slug}", "lymph", "lymph-vessel", dpath(wps), name))
        if bil:
            zones.append(_base(f"face-lp-{slug}-L", "lymph", "lymph-vessel",
                               dpath([(round(1 - a, 4), b) for a, b in wps]), name))
    for slug, name, poly in NERVE:
        zones.append(_base(f"face-{slug}", "nerve", "trigeminal", {"type": "polygon", "points": poly}, name))
    for slug, name, bil, x, y in EAV:
        _pair(zones, f"face-{slug}", "eav", "eav",
              lambda xx: {"type": "point", "x": xx, "y": y}, x, y, name, bil)

    data = {
        "system": "face",
        "reference_frame": "face_outline",
        "side_noun": "layer", "group_noun": "element",
        "outline": catmull_rom_closed(FACE) + " " + FEATURES,
        "groups": GROUPS,
        "anchors": [
            {"key": "hairline", "template": {"x": 0.500, "y": 0.070}, "hint": "Tap the centre of the hairline."},
            {"key": "chin", "template": {"x": 0.500, "y": 0.915}, "hint": "Tap the bottom of the chin."},
        ],
        "zones": zones,
    }
    out = ROOT / "data" / "bodymap-face.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    from collections import Counter
    print(f"wrote {out}: {len(zones)} zones", dict(Counter(z["side"] for z in zones)))


if __name__ == "__main__":
    main()
