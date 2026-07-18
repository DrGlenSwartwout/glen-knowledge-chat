#!/usr/bin/env python3
"""Build data/bodymap-neurotome.json — the FULL dermatome / neurotome map.

Every dermatome level (trigeminal V1-V3, and every spinal root C2-C8, T1-T12,
L1-L5, S1-S5) is a `polygon` region on the body figure, coloured by segment
group. Front + back views. Trunk bands are generated parametrically between the
torso edges at each level (evenly distributed by height, refine positions later);
head and limb dermatomes are explicit strips. Limb dermatomes are bilateral
(mirrored x -> 1-x).

Positions are a standards-based rendering for Glen's clinical refinement, not
gospel.
"""
import json
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("bbo", ROOT / "scripts" / "build_body_outline.py")
bbo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bbo)

GROUPS = [
    {"id": "cranial", "label": "Cranial (trigeminal V1-V3)"},
    {"id": "cervical", "label": "Cervical (C2-C8)"},
    {"id": "thoracic", "label": "Thoracic (T1-T12)"},
    {"id": "lumbar", "label": "Lumbar (L1-L5)"},
    {"id": "sacral", "label": "Sacral (S1-S5)"},
]

# torso half-contour: (y, x_left, x_right) — full-width trunk bands span these
_EDGES = [(0.174, 0.345, 0.655), (0.210, 0.365, 0.635), (0.250, 0.385, 0.615),
          (0.300, 0.405, 0.595), (0.350, 0.415, 0.585), (0.400, 0.425, 0.575),
          (0.450, 0.435, 0.565), (0.520, 0.400, 0.600), (0.560, 0.395, 0.605)]


def _edge(y):
    pts = _EDGES
    if y <= pts[0][0]:
        return pts[0][1], pts[0][2]
    if y >= pts[-1][0]:
        return pts[-1][1], pts[-1][2]
    for i in range(len(pts) - 1):
        y0, l0, r0 = pts[i]
        y1, l1, r1 = pts[i + 1]
        if y0 <= y <= y1:
            f = (y - y0) / (y1 - y0)
            return round(l0 + (l1 - l0) * f, 4), round(r0 + (r1 - r0) * f, 4)


def _band(y0, y1):
    l0, r0 = _edge(y0)
    l1, r1 = _edge(y1)
    return [[l0, y0], [r0, y0], [r1, y1], [l1, y1]]


def _trunk_levels(labels, y0, y1, view, group_of):
    """Even horizontal bands, one per label, from y0 (top) to y1."""
    out = []
    n = len(labels)
    h = (y1 - y0) / n
    for i, lab in enumerate(labels):
        a, b = y0 + i * h, y0 + (i + 1) * h
        out.append((lab, lab, view, group_of(lab), False, _band(a, b)))
    return out


def _seg_group(lab):
    c = lab[0]
    return {"C": "cervical", "T": "thoracic", "L": "lumbar", "S": "sacral"}[c]


# explicit head + limb dermatomes (slug/label, view, group, bilateral, polygon)
EXPLICIT = [
    # ---- head, front ----
    ("V1", "V1 (ophthalmic)", "front", "cranial", False, [[0.44, 0.020], [0.56, 0.020], [0.558, 0.058], [0.442, 0.058]]),
    ("V2", "V2 (maxillary)", "front", "cranial", False, [[0.448, 0.058], [0.552, 0.058], [0.548, 0.090], [0.452, 0.090]]),
    ("V3", "V3 (mandibular)", "front", "cranial", False, [[0.456, 0.090], [0.544, 0.090], [0.536, 0.126], [0.464, 0.126]]),
    ("C2f", "C2", "front", "cervical", False, [[0.462, 0.126], [0.538, 0.126], [0.542, 0.150], [0.458, 0.150]]),
    ("C3f", "C3", "front", "cervical", False, [[0.458, 0.150], [0.542, 0.150], [0.552, 0.174], [0.448, 0.174]]),
    # ---- head/neck, back ----
    ("C2b", "C2 (occiput)", "back", "cervical", False, [[0.452, 0.040], [0.548, 0.040], [0.552, 0.100], [0.448, 0.100]]),
    ("C3b", "C3", "back", "cervical", False, [[0.458, 0.100], [0.542, 0.100], [0.548, 0.140], [0.452, 0.140]]),
    ("C4b", "C4", "back", "cervical", False, [[0.452, 0.140], [0.548, 0.140], [0.552, 0.174], [0.448, 0.174]]),
    # ---- arm (right; bilateral) front ----
    ("C5f", "C5 (lateral arm)", "front", "cervical", True, [[0.660, 0.190], [0.720, 0.216], [0.718, 0.320], [0.665, 0.300]]),
    ("C6f", "C6 (forearm/thumb)", "front", "cervical", True, [[0.700, 0.360], [0.752, 0.450], [0.788, 0.600], [0.742, 0.606], [0.700, 0.462]]),
    ("C7f", "C7 (middle finger)", "front", "cervical", True, [[0.730, 0.582], [0.775, 0.610], [0.760, 0.646], [0.720, 0.622]]),
    ("C8f", "C8 (little finger)", "front", "cervical", True, [[0.718, 0.602], [0.756, 0.628], [0.742, 0.650], [0.710, 0.628]]),
    ("T1f", "T1 (medial arm)", "front", "thoracic", True, [[0.664, 0.342], [0.696, 0.442], [0.686, 0.552], [0.656, 0.442]]),
    # ---- arm (right; bilateral) back ----
    ("C5b", "C5 (posterior arm)", "back", "cervical", True, [[0.660, 0.196], [0.716, 0.220], [0.716, 0.330], [0.666, 0.310]]),
    ("C6b", "C6 (posterior forearm)", "back", "cervical", True, [[0.700, 0.360], [0.752, 0.450], [0.786, 0.600], [0.742, 0.606], [0.702, 0.462]]),
    ("C7b", "C7 (middle finger)", "back", "cervical", True, [[0.730, 0.582], [0.772, 0.610], [0.758, 0.646], [0.720, 0.622]]),
    ("C8b", "C8 (little finger)", "back", "cervical", True, [[0.716, 0.600], [0.756, 0.628], [0.742, 0.652], [0.708, 0.628]]),
    ("T1b", "T1 (medial arm)", "back", "thoracic", True, [[0.664, 0.342], [0.696, 0.442], [0.686, 0.552], [0.656, 0.442]]),
    # ---- leg (right; bilateral) front ----
    ("L2f", "L2 (upper thigh)", "front", "lumbar", True, [[0.520, 0.530], [0.628, 0.545], [0.622, 0.645], [0.528, 0.640]]),
    ("L3f", "L3 (thigh/knee)", "front", "lumbar", True, [[0.512, 0.645], [0.604, 0.650], [0.596, 0.772], [0.522, 0.770]]),
    ("L4f", "L4 (medial leg)", "front", "lumbar", True, [[0.514, 0.772], [0.560, 0.774], [0.556, 0.905], [0.516, 0.905]]),
    ("L5f", "L5 (lateral leg/dorsum)", "front", "lumbar", True, [[0.570, 0.774], [0.608, 0.784], [0.618, 0.960], [0.580, 0.960]]),
    ("S1f", "S1 (lateral foot)", "front", "sacral", True, [[0.578, 0.958], [0.620, 0.965], [0.610, 0.988], [0.568, 0.988]]),
    # ---- leg (right; bilateral) back ----
    ("S2Lb", "S2 (posterior thigh)", "back", "sacral", True, [[0.518, 0.578], [0.612, 0.585], [0.604, 0.700], [0.526, 0.696]]),
    ("S1Lb", "S1/L5 (posterior calf)", "back", "sacral", True, [[0.526, 0.700], [0.604, 0.704], [0.596, 0.870], [0.534, 0.868]]),
    ("S1Hb", "S1 (sole/heel)", "back", "sacral", True, [[0.534, 0.868], [0.596, 0.872], [0.618, 0.984], [0.556, 0.988]]),
]


def _mk(zid, view, group, poly, label):
    return {
        "id": zid, "side": view, "bilateral": False, "group": group,
        "geometry": {"type": "polygon", "points": poly},
        "anatomy": label,
        "meaning_standard": f"{label} dermatome / nerve territory.",
        "meaning_glen": "",
        "layers": {"embryological_depth": None, "stress_affirmation": None, "touch_for_health": None},
    }


def main():
    rows = list(EXPLICIT)
    # trunk bands: FRONT C4 + T2..T12 + L1 ; BACK C4 + T1..T12 + L1..L5 + S1..S5
    rows += [("C4f", "C4", "front", "cervical", False, _band(0.174, 0.205))]
    rows += _trunk_levels([f"T{i}" for i in range(2, 13)], 0.205, 0.450, "front", _seg_group)
    rows += [("L1f", "L1", "front", "lumbar", False, _band(0.450, 0.520))]
    rows += [("C4bt", "C4", "back", "cervical", False, _band(0.174, 0.205))]
    rows += _trunk_levels([f"T{i}" for i in range(1, 13)], 0.205, 0.440, "back", _seg_group)
    rows += _trunk_levels([f"L{i}" for i in range(1, 6)], 0.440, 0.550, "back", _seg_group)
    rows += _trunk_levels([f"S{i}" for i in range(1, 6)], 0.550, 0.615, "back", _seg_group)

    zones = []
    for slug, label, view, group, bilateral, poly in rows:
        if bilateral:
            zones.append(_mk(f"neuro-{slug}-R", view, group, poly, label))
            zones.append(_mk(f"neuro-{slug}-L", view, group, [[round(1 - x, 4), y] for x, y in poly], label))
        else:
            zones.append(_mk(f"neuro-{slug}", view, group, poly, label))
    front = bbo.build_outline()
    data = {
        "system": "neurotome",
        "reference_frame": "body_outline",
        "side_noun": "view", "group_noun": "segment",
        "outline": front,
        "outlines": {"front": front, "back": front},
        "groups": GROUPS, "zones": zones,
        "anchors": [
            {"key": "head-f", "view": "front", "template": {"x": 0.50, "y": 0.02}, "hint": "Tap the top of the head."},
            {"key": "feet-f", "view": "front", "template": {"x": 0.50, "y": 0.985}, "hint": "Tap the point between the ankles."},
            {"key": "head-b", "view": "back", "template": {"x": 0.50, "y": 0.02}, "hint": "Tap the top of the head."},
            {"key": "feet-b", "view": "back", "template": {"x": 0.50, "y": 0.985}, "hint": "Tap the point between the ankles."},
        ],
    }
    out = ROOT / "data" / "bodymap-neurotome.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    from collections import Counter
    print(f"wrote {out}: {len(zones)} regions", dict(Counter(z["side"] for z in zones)))


if __name__ == "__main__":
    main()
