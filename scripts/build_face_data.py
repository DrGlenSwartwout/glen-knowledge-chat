#!/usr/bin/env python3
"""Build data/bodymap-face.json — facial diagnosis / face reflexology map.

Front-view face outline (oval + ears + eyes/brows/nose/mouth reference features)
with the organ-correspondence zones of Chinese facial diagnosis (mien shiang),
grouped by the five TCM elements. Single front view; paired zones are emitted on
both sides. Ellipse geometry (areas), like the foot/hand reflexology charts.

Anchors let the map warp onto a face selfie (hairline + chin). Positions are a
standards-based rendering for Glen's clinical refinement, not gospel.
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


# face oval + ear bumps, traced clockwise from the forehead top
FACE = [
    (0.500, 0.060), (0.590, 0.075), (0.665, 0.115), (0.710, 0.210), (0.725, 0.340),
    (0.755, 0.430), (0.750, 0.560), (0.695, 0.660), (0.625, 0.800), (0.545, 0.895),
    (0.500, 0.920), (0.455, 0.895), (0.375, 0.800), (0.305, 0.660), (0.250, 0.560),
    (0.245, 0.430), (0.275, 0.340), (0.290, 0.210), (0.335, 0.115), (0.410, 0.075),
]

# internal reference features (stroked open sub-paths)
FEATURES = " ".join([
    # eyes
    "M0.330 0.420 C0.345 0.408 0.415 0.408 0.430 0.420 C0.415 0.432 0.345 0.432 0.330 0.420 Z",
    "M0.570 0.420 C0.585 0.408 0.655 0.408 0.670 0.420 C0.655 0.432 0.585 0.432 0.570 0.420 Z",
    # brows
    "M0.325 0.370 C0.350 0.352 0.410 0.352 0.435 0.368",
    "M0.565 0.368 C0.590 0.352 0.650 0.352 0.675 0.370",
    # nose (bridge + nostrils)
    "M0.500 0.400 L0.500 0.585 M0.460 0.605 C0.478 0.622 0.522 0.622 0.540 0.605",
    # mouth
    "M0.435 0.725 C0.470 0.748 0.530 0.748 0.565 0.725",
])

GROUPS = [
    {"id": "wood", "label": "Wood — Liver / Gallbladder"},
    {"id": "fire", "label": "Fire — Heart / Small Intestine"},
    {"id": "earth", "label": "Earth — Spleen / Stomach"},
    {"id": "metal", "label": "Metal — Lung / Large Intestine"},
    {"id": "water", "label": "Water — Kidney / Bladder"},
]

# (slug, name, element, bilateral, cx, cy, rx, ry)
ZONES = [
    # ---- Wood (liver / gallbladder) ----
    ("glabella", "Liver (glabella)", "wood", False, 0.500, 0.335, 0.030, 0.030),
    ("nose-bridge", "Liver / gallbladder (nose bridge)", "wood", False, 0.500, 0.425, 0.024, 0.045),
    ("temple", "Gallbladder (temple)", "wood", True, 0.315, 0.280, 0.035, 0.055),
    ("eyes", "Liver (eyes)", "wood", True, 0.380, 0.420, 0.050, 0.026),
    # ---- Fire (heart / small intestine) ----
    ("forehead-mind", "Heart / mind (upper forehead)", "fire", False, 0.500, 0.130, 0.100, 0.035),
    ("nose-tip", "Heart (nose tip)", "fire", False, 0.500, 0.600, 0.036, 0.030),
    # ---- Earth (spleen / stomach) ----
    ("nose-mid", "Spleen / pancreas (mid nose)", "earth", False, 0.500, 0.520, 0.030, 0.035),
    ("upper-lip", "Spleen (upper lip)", "earth", False, 0.500, 0.680, 0.036, 0.020),
    ("mouth", "Stomach (mouth)", "earth", False, 0.500, 0.728, 0.052, 0.024),
    # ---- Metal (lung / large intestine) ----
    ("forehead-li", "Large intestine (mid forehead)", "metal", False, 0.500, 0.200, 0.090, 0.028),
    ("cheek", "Lungs (cheek)", "metal", True, 0.345, 0.575, 0.060, 0.060),
    ("nostril", "Bronchi (nostril wing)", "metal", True, 0.445, 0.610, 0.026, 0.022),
    # ---- Water (kidney / bladder) ----
    ("forehead-bladder", "Bladder (forehead hairline)", "water", False, 0.500, 0.255, 0.080, 0.026),
    ("under-eye", "Kidney (under-eye)", "water", True, 0.400, 0.470, 0.045, 0.026),
    ("jaw", "Kidney (jaw)", "water", True, 0.360, 0.780, 0.042, 0.050),
    ("ear", "Kidney (ear)", "water", True, 0.748, 0.510, 0.022, 0.045),
    ("chin", "Kidney / bladder / hormones (chin)", "water", False, 0.500, 0.850, 0.062, 0.040),
]


def _mk(zid, group, cx, cy, rx, ry, name):
    return {
        "id": zid, "side": "front", "bilateral": False, "group": group,
        "geometry": {"type": "ellipse", "cx": cx, "cy": cy, "rx": rx, "ry": ry},
        "anatomy": name,
        "meaning_standard": f"{name} — facial-diagnosis reflex area.",
        "meaning_glen": "",
        "layers": {"embryological_depth": None, "stress_affirmation": None, "touch_for_health": None},
    }


def main():
    zones = []
    for slug, name, group, bilateral, cx, cy, rx, ry in ZONES:
        if bilateral:
            zones.append(_mk(f"face-{slug}-R", group, cx, cy, rx, ry, name))
            zones.append(_mk(f"face-{slug}-L", group, round(1 - cx, 4), cy, rx, ry, name))
        else:
            zones.append(_mk(f"face-{slug}", group, cx, cy, rx, ry, name))
    data = {
        "system": "face",
        "reference_frame": "face_outline",
        "side_noun": "view", "group_noun": "element",
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
    print(f"wrote {out}: {len(zones)} zones", dict(Counter(z["group"] for z in zones)))


if __name__ == "__main__":
    main()
