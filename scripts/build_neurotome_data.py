#!/usr/bin/env python3
"""Build data/bodymap-neurotome.json — dermatome / neurotome map on the body.

Each dermatome (the skin territory of one spinal nerve root, plus the cranial
trigeminal divisions) is a `polygon` region on the body figure, coloured by
segment group (cranial / cervical / thoracic / lumbar / sacral). Front and back
views via the Side control, reusing the meridian body silhouette. Trunk bands
span the midline (single polygons); limb dermatomes are bilateral (emitted on
both sides, mirrored x -> 1-x).

Positions are a standards-based rendering of a dermatome chart for Glen's
clinical refinement via the admin drawing tool, not gospel. Starter set of the
major dermatomes; expandable.
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

# (slug, label, view, group, bilateral, [[x,y], ...])
DERMATOMES = [
    # ===================== FRONT =====================
    ("V1", "V1 (ophthalmic)", "front", "cranial", False, [[0.44, 0.020], [0.56, 0.020], [0.558, 0.058], [0.442, 0.058]]),
    ("V2", "V2 (maxillary)", "front", "cranial", False, [[0.448, 0.058], [0.552, 0.058], [0.548, 0.090], [0.452, 0.090]]),
    ("V3", "V3 (mandibular)", "front", "cranial", False, [[0.456, 0.090], [0.544, 0.090], [0.536, 0.126], [0.464, 0.126]]),
    ("C3f", "C3", "front", "cervical", False, [[0.462, 0.128], [0.538, 0.128], [0.552, 0.172], [0.448, 0.172]]),
    ("C4f", "C4", "front", "cervical", False, [[0.345, 0.174], [0.655, 0.174], [0.635, 0.208], [0.365, 0.208]]),
    ("T2f", "T2", "front", "thoracic", False, [[0.366, 0.208], [0.634, 0.208], [0.622, 0.246], [0.378, 0.246]]),
    ("T4f", "T4 (nipple)", "front", "thoracic", False, [[0.378, 0.246], [0.622, 0.246], [0.608, 0.286], [0.392, 0.286]]),
    ("T6f", "T6", "front", "thoracic", False, [[0.392, 0.286], [0.608, 0.286], [0.596, 0.326], [0.404, 0.326]]),
    ("T8f", "T8", "front", "thoracic", False, [[0.404, 0.326], [0.596, 0.326], [0.586, 0.366], [0.414, 0.366]]),
    ("T10f", "T10 (navel)", "front", "thoracic", False, [[0.414, 0.366], [0.586, 0.366], [0.576, 0.410], [0.424, 0.410]]),
    ("T12f", "T12", "front", "thoracic", False, [[0.424, 0.410], [0.576, 0.410], [0.566, 0.450], [0.434, 0.450]]),
    ("L1f", "L1", "front", "lumbar", False, [[0.434, 0.450], [0.566, 0.450], [0.605, 0.520], [0.395, 0.520]]),
    # arm (right; bilateral)
    ("C5f", "C5 (lateral arm)", "front", "cervical", True, [[0.660, 0.190], [0.720, 0.216], [0.718, 0.320], [0.665, 0.300]]),
    ("C6f", "C6 (forearm/thumb)", "front", "cervical", True, [[0.700, 0.360], [0.752, 0.450], [0.788, 0.600], [0.742, 0.606], [0.700, 0.462]]),
    ("C7f", "C7 (middle finger)", "front", "cervical", True, [[0.730, 0.582], [0.775, 0.610], [0.760, 0.646], [0.720, 0.622]]),
    ("C8f", "C8 (little finger)", "front", "cervical", True, [[0.718, 0.602], [0.756, 0.628], [0.742, 0.650], [0.710, 0.628]]),
    ("T1f", "T1 (medial arm)", "front", "thoracic", True, [[0.664, 0.342], [0.696, 0.442], [0.686, 0.552], [0.656, 0.442]]),
    # leg (right; bilateral)
    ("L2f", "L2 (upper thigh)", "front", "lumbar", True, [[0.520, 0.530], [0.628, 0.545], [0.622, 0.645], [0.528, 0.640]]),
    ("L3f", "L3 (thigh/knee)", "front", "lumbar", True, [[0.512, 0.645], [0.604, 0.650], [0.596, 0.772], [0.522, 0.770]]),
    ("L4f", "L4 (medial leg)", "front", "lumbar", True, [[0.514, 0.772], [0.560, 0.774], [0.556, 0.905], [0.516, 0.905]]),
    ("L5f", "L5 (lateral leg/dorsum)", "front", "lumbar", True, [[0.570, 0.774], [0.608, 0.784], [0.618, 0.960], [0.580, 0.960]]),
    ("S1f", "S1 (lateral foot)", "front", "sacral", True, [[0.578, 0.958], [0.620, 0.965], [0.610, 0.988], [0.568, 0.988]]),
    # ===================== BACK =====================
    ("C2b", "C2 (occiput)", "back", "cervical", False, [[0.452, 0.040], [0.548, 0.040], [0.552, 0.100], [0.448, 0.100]]),
    ("C4b", "C4 (upper back)", "back", "cervical", False, [[0.345, 0.174], [0.655, 0.174], [0.635, 0.214], [0.365, 0.214]]),
    ("T3b", "T3", "back", "thoracic", False, [[0.366, 0.214], [0.634, 0.214], [0.620, 0.264], [0.380, 0.264]]),
    ("T6b", "T6", "back", "thoracic", False, [[0.380, 0.264], [0.620, 0.264], [0.600, 0.320], [0.400, 0.320]]),
    ("T9b", "T9", "back", "thoracic", False, [[0.400, 0.320], [0.600, 0.320], [0.584, 0.376], [0.416, 0.376]]),
    ("T12b", "T12", "back", "thoracic", False, [[0.416, 0.376], [0.584, 0.376], [0.572, 0.430], [0.428, 0.430]]),
    ("L2b", "L2 (low back)", "back", "lumbar", False, [[0.428, 0.430], [0.572, 0.430], [0.600, 0.500], [0.400, 0.500]]),
    ("S2b", "S2-S4 (buttock)", "back", "sacral", False, [[0.400, 0.500], [0.600, 0.500], [0.610, 0.575], [0.390, 0.575]]),
    # arm (right; bilateral)
    ("C5b", "C5 (posterior arm)", "back", "cervical", True, [[0.660, 0.196], [0.716, 0.220], [0.716, 0.330], [0.666, 0.310]]),
    ("C6b", "C6 (posterior forearm)", "back", "cervical", True, [[0.700, 0.360], [0.752, 0.450], [0.786, 0.600], [0.742, 0.606], [0.702, 0.462]]),
    ("C8b", "C8 (little finger)", "back", "cervical", True, [[0.716, 0.600], [0.756, 0.628], [0.742, 0.652], [0.708, 0.628]]),
    ("T1b", "T1 (medial arm)", "back", "thoracic", True, [[0.664, 0.342], [0.696, 0.442], [0.686, 0.552], [0.656, 0.442]]),
    # leg (right; bilateral)
    ("S2L", "S2 (posterior thigh)", "back", "sacral", True, [[0.518, 0.578], [0.612, 0.585], [0.604, 0.700], [0.526, 0.696]]),
    ("L5b", "L5 (posterior calf)", "back", "lumbar", True, [[0.526, 0.700], [0.604, 0.704], [0.596, 0.870], [0.534, 0.868]]),
    ("S1b", "S1 (sole/heel)", "back", "sacral", True, [[0.534, 0.868], [0.596, 0.872], [0.618, 0.984], [0.556, 0.988]]),
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
    zones = []
    for slug, label, view, group, bilateral, poly in DERMATOMES:
        if bilateral:
            zones.append(_mk(f"neuro-{slug}-R", view, group, poly, label))
            mirrored = [[round(1 - x, 4), y] for x, y in poly]
            zones.append(_mk(f"neuro-{slug}-L", view, group, mirrored, label))
        else:
            zones.append(_mk(f"neuro-{slug}", view, group, poly, label))
    front = bbo.build_outline()
    data = {
        "system": "neurotome",
        "reference_frame": "body_outline",
        "side_noun": "view",
        "group_noun": "segment",
        "outline": front,
        "outlines": {"front": front, "back": front},
        "groups": GROUPS,
        "anchors": [],
        "zones": zones,
    }
    out = ROOT / "data" / "bodymap-neurotome.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    from collections import Counter
    print(f"wrote {out}: {len(zones)} regions", dict(Counter(z["side"] for z in zones)))


if __name__ == "__main__":
    main()
