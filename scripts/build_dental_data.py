#!/usr/bin/env python3
"""Build data/bodymap-dental.json — the dental / meridian tooth chart.

Holistic-dentistry chart: each tooth relates to a meridian and its organs (the
Voll / Kramer tooth-meridian relationships). Both arches are shown at once (upper
arch on top, lower arch below), 32 teeth as `ellipse` zones arranged along each
arch. A tooth carries its associated organs in a `meridian_organs` list, so a
client's organ finding lights the teeth on that organ's meridian (e.g. a Kidney
finding lights the incisors; Liver lights the canines).

The upper/lower reciprocal relationship is standard: upper premolars & lower
molars = Lung / Large Intestine; upper molars & lower premolars = Stomach /
Spleen-Pancreas. Positions are a standards-based rendering for Glen's refinement.
Laterality is anatomical-as-viewed (patient's right = viewer's left, x < 0.5).
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GROUPS = [
    {"id": "incisor", "label": "Incisors — Kidney / Bladder"},
    {"id": "canine", "label": "Canines — Liver / Gallbladder"},
    {"id": "premolar", "label": "Premolars"},
    {"id": "molar", "label": "Molars"},
    {"id": "wisdom", "label": "Wisdom — Heart / Small Intestine"},
]

# tooth type -> (display name, group). Meridian organs differ upper vs lower.
TYPES = {
    "CI": ("central incisor", "incisor"),
    "LI": ("lateral incisor", "incisor"),
    "C": ("canine", "canine"),
    "1PM": ("first premolar", "premolar"),
    "2PM": ("second premolar", "premolar"),
    "1M": ("first molar", "molar"),
    "2M": ("second molar", "molar"),
    "3M": ("third molar (wisdom)", "wisdom"),
}
UPPER_ORGANS = {
    "CI": ["Kidney", "Bladder"], "LI": ["Kidney", "Bladder"],
    "C": ["Liver", "Gallbladder"],
    "1PM": ["Lung", "Large Intestine"], "2PM": ["Lung", "Large Intestine"],
    "1M": ["Stomach", "Spleen", "Pancreas"], "2M": ["Stomach", "Spleen", "Pancreas"],
    "3M": ["Heart", "Small Intestine"],
}
LOWER_ORGANS = {
    "CI": ["Kidney", "Bladder"], "LI": ["Kidney", "Bladder"],
    "C": ["Liver", "Gallbladder"],
    "1PM": ["Stomach", "Spleen", "Pancreas"], "2PM": ["Stomach", "Spleen", "Pancreas"],
    "1M": ["Lung", "Large Intestine"], "2M": ["Lung", "Large Intestine"],
    "3M": ["Heart", "Small Intestine"],
}
# right back -> midline -> left back (16 teeth per arch)
ORDER = ["3M", "2M", "1M", "2PM", "1PM", "C", "LI", "CI",
         "CI", "LI", "C", "1PM", "2PM", "1M", "2M", "3M"]
GROUP_OVERRIDE = {"wisdom": "wisdom"}  # keep wisdom in its own colour group


def _arch(n, cx, cy, rx, ry, a0, a1):
    """n points along an ellipse arc from a0->a1 degrees (0=east, CCW), y-down."""
    pts = []
    for i in range(n):
        t = math.radians(a0 + (a1 - a0) * i / (n - 1))
        pts.append((cx + rx * math.cos(t), cy - ry * math.sin(t)))
    return pts


def _arc_path(cx, cy, rx, ry, a0, a1, steps=48):
    pts = _arch(steps, cx, cy, rx, ry, a0, a1)
    return "M " + " L ".join(f"{x:.4f} {y:.4f}" for x, y in pts)


def _tooth(arch, idx, ttype, x, y, organs):
    name, base_group = TYPES[ttype]
    group = "wisdom" if base_group == "wisdom" else base_group
    side_lbl = "right" if idx < 8 else "left"
    num = idx + 1 if arch == "upper" else idx + 17  # simple 1-16 / 17-32
    anatomy = f"{arch.capitalize()} {side_lbl} {name}"
    meaning = "Related meridian organs: " + ", ".join(organs) + "."
    return {
        "id": f"tooth-{arch}-{num}", "side": "chart", "bilateral": False,
        "group": group, "geometry": {"type": "ellipse", "cx": round(x, 4),
        "cy": round(y, 4), "rx": 0.017, "ry": 0.017},
        "anatomy": anatomy, "meridian_organs": organs,
        "meaning_standard": meaning, "meaning_glen": "",
        "layers": {"embryological_depth": "ectoderm", "stress_affirmation": None, "touch_for_health": None},
    }


def main():
    zones = []
    # upper arch: ∩ opening down; patient-right (idx 0) on viewer-left (west)
    upper = _arch(16, 0.500, 0.360, 0.340, 0.230, 210, -30)
    for i, ttype in enumerate(ORDER):
        x, y = upper[i]
        zones.append(_tooth("upper", i, ttype, x, y, UPPER_ORGANS[ttype]))
    # lower arch: ∪ opening up
    lower = _arch(16, 0.500, 0.640, 0.340, 0.230, 150, 390)
    for i, ttype in enumerate(ORDER):
        x, y = lower[i]
        zones.append(_tooth("lower", i, ttype, x, y, LOWER_ORGANS[ttype]))
    outline = (_arc_path(0.500, 0.360, 0.360, 0.250, 210, -30)
               + " " + _arc_path(0.500, 0.640, 0.360, 0.250, 150, 390))
    data = {
        "system": "dental", "reference_frame": "dental_outline",
        "side_noun": "chart", "group_noun": "tooth type",
        "outline": outline, "outlines": {"chart": outline},
        "groups": GROUPS,
        "anchors": [
            {"key": "arch-top", "view": "chart", "template": {"x": 0.50, "y": 0.13}, "hint": "Tap between your two top front teeth."},
            {"key": "arch-bottom", "view": "chart", "template": {"x": 0.50, "y": 0.87}, "hint": "Tap between your two bottom front teeth."},
        ],
        "zones": zones,
    }
    out = ROOT / "data" / "bodymap-dental.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    from collections import Counter
    print(f"wrote {out}: {len(zones)} teeth", dict(Counter(z["group"] for z in zones)))


if __name__ == "__main__":
    main()
