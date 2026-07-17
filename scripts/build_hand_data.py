#!/usr/bin/env python3
"""Build data/bodymap-hand.json — the hand (palm) reflexology system.

Mirrors the foot's model: the same 34-organ set and 8 groups, bilateral zones
authored once on the canonical right hand (renderer mirrors for the left), and
the same visceral asymmetries placed per side. Hand mapping conventions:
  * right hand, palm view, fingers up, thumb to the LEFT (radial = medial =
    low x), so `bilateral` mirroring and `outline_side:"right"` behave exactly
    as they do for the foot;
  * fingers/thumb carry the head & sinus reflexes; the thenar (thumb ball) the
    neck/thyroid; the upper palm the chest/lung; the mid-palm the digestive and
    urinary organs; the heel of the palm / wrist the pelvis and elimination;
  * spine runs down the radial (thumb-side) edge; shoulder/arm/leg down the
    ulnar (little-finger) edge.

Left-only viscera (heart, spleen, stomach, pancreas, descending & sigmoid colon)
are stored in the LEFT-hand display frame (already mirrored), matching how the
renderer draws non-bilateral zones; right-only viscera (liver, gallbladder,
ascending colon, ileocecal) are in the right-hand frame.

Positions are standard-chart approximate and render-verified to sit inside the
outline; refine clinically via the admin drawing tool.
"""
import json
from pathlib import Path

import importlib.util

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("bho", ROOT / "scripts" / "build_hand_outline.py")
bho = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bho)

GROUPS = [
    {"id": "head-sinus", "label": "Head and sinuses (fingers)"},
    {"id": "neck-thyroid", "label": "Neck and thyroid"},
    {"id": "chest-lung", "label": "Chest, lungs and heart"},
    {"id": "digestive", "label": "Digestive organs"},
    {"id": "urinary", "label": "Urinary and adrenal"},
    {"id": "spine", "label": "Spine (radial/thumb edge)"},
    {"id": "pelvis-elimination", "label": "Pelvis and elimination (wrist)"},
    {"id": "limb", "label": "Shoulder, arm and leg (ulnar edge)"},
]

ANCHORS = [
    {"key": "thumb-tip", "template": {"x": 0.13, "y": 0.50}, "hint": "Tap the tip of your thumb."},
    {"key": "middle-finger-tip", "template": {"x": 0.50, "y": 0.05}, "hint": "Tap the tip of your middle finger."},
    {"key": "wrist-center", "template": {"x": 0.50, "y": 0.90}, "hint": "Tap the centre of your wrist."},
]

# (slug, anatomy, group, kind, cx, cy, rx, ry, meaning_standard)
# kind: bi = bilateral; L = left hand only; R = right hand only.
ZONES = [
    # ---- head & sinuses: thumb + fingers ----
    ("brain", "Brain and head", "head-sinus", "bi", 0.180, 0.500, 0.045, 0.040, "Reflex for the brain and head, on the thumb."),
    ("pituitary", "Pituitary", "head-sinus", "bi", 0.210, 0.560, 0.028, 0.024, "Pituitary reflex, centre of the thumb."),
    ("sinuses", "Sinuses", "head-sinus", "bi", 0.500, 0.200, 0.026, 0.022, "Sinus reflexes, across the finger tips."),
    ("eyes", "Eyes", "head-sinus", "bi", 0.448, 0.400, 0.035, 0.028, "Eye reflex, base of the index and middle fingers."),
    ("ears", "Ears", "head-sinus", "bi", 0.640, 0.410, 0.038, 0.028, "Ear reflex, base of the ring and little fingers."),
    # ---- neck & thyroid: thumb base / thenar ----
    ("neck-throat", "Neck and throat", "neck-thyroid", "bi", 0.300, 0.605, 0.030, 0.025, "Neck and throat reflex, base of the thumb."),
    ("thyroid", "Thyroid", "neck-thyroid", "bi", 0.370, 0.535, 0.040, 0.030, "Thyroid reflex, on the thenar (thumb ball)."),
    # ---- chest / lung (heart is left-only) ----
    ("lung", "Lungs and bronchi", "chest-lung", "bi", 0.520, 0.475, 0.090, 0.040, "Lung and bronchial reflex, upper palm below the fingers."),
    ("diaphragm", "Diaphragm and solar plexus", "chest-lung", "bi", 0.500, 0.550, 0.080, 0.030, "Diaphragm and solar-plexus reflex, across the palm."),
    ("heart", "Heart", "chest-lung", "L", 0.560, 0.480, 0.045, 0.040, "Heart reflex, upper palm — left hand only."),
    # ---- limb (lateral / ulnar edge) ----
    ("shoulder-arm", "Shoulder and arm", "limb", "bi", 0.660, 0.480, 0.038, 0.032, "Shoulder and arm reflex, ulnar (little-finger) edge."),
    ("knee-leg", "Knee and leg", "limb", "bi", 0.640, 0.710, 0.032, 0.045, "Knee and leg reflex, lower ulnar edge."),
    # ---- digestive ----
    ("R-liver", "Liver", "digestive", "R", 0.600, 0.620, 0.070, 0.050, "Liver reflex, right hand, mid palm."),
    ("R-gallbladder", "Gallbladder", "digestive", "R", 0.620, 0.665, 0.028, 0.022, "Gallbladder reflex, right hand."),
    ("L-spleen", "Spleen", "digestive", "L", 0.400, 0.625, 0.045, 0.038, "Spleen reflex, left hand, mid palm."),
    ("L-stomach", "Stomach", "digestive", "L", 0.535, 0.600, 0.050, 0.038, "Stomach reflex, left hand, upper mid palm."),
    ("L-pancreas", "Pancreas", "digestive", "L", 0.510, 0.655, 0.045, 0.030, "Pancreas reflex, left hand."),
    ("transverse-colon", "Transverse colon", "digestive", "bi", 0.500, 0.710, 0.095, 0.030, "Transverse colon reflex, across the mid palm."),
    ("small-intestine", "Small intestine", "digestive", "bi", 0.500, 0.785, 0.080, 0.045, "Small-intestine reflex, lower central palm."),
    ("R-ascending-colon", "Ascending colon", "digestive", "R", 0.615, 0.730, 0.028, 0.050, "Ascending colon reflex, right hand, ulnar side."),
    ("R-ileocecal", "Ileocecal valve", "digestive", "R", 0.615, 0.800, 0.028, 0.020, "Ileocecal-valve reflex, right hand, lower palm."),
    ("L-descending-colon", "Descending colon", "digestive", "L", 0.405, 0.730, 0.028, 0.050, "Descending colon reflex, left hand, ulnar side."),
    ("L-sigmoid-colon", "Sigmoid colon", "digestive", "L", 0.470, 0.815, 0.045, 0.028, "Sigmoid colon reflex, left hand, lower palm."),
    # ---- urinary & adrenal ----
    ("adrenal", "Adrenal gland", "urinary", "bi", 0.445, 0.620, 0.028, 0.020, "Adrenal reflex, atop the kidney, mid palm."),
    ("kidney", "Kidney", "urinary", "bi", 0.450, 0.675, 0.035, 0.030, "Kidney reflex, mid palm."),
    ("ureter", "Ureter", "urinary", "bi", 0.440, 0.745, 0.028, 0.040, "Ureter reflex, running toward the bladder."),
    ("bladder", "Bladder", "urinary", "bi", 0.420, 0.820, 0.035, 0.028, "Bladder reflex, lower radial palm."),
    # ---- spine: radial (thumb-side) edge ----
    ("cervical-spine", "Cervical spine", "spine", "bi", 0.345, 0.560, 0.020, 0.030, "Cervical spine reflex, upper radial edge."),
    ("thoracic-spine", "Thoracic spine", "spine", "bi", 0.355, 0.660, 0.020, 0.050, "Thoracic spine reflex, radial edge."),
    ("lumbar-spine", "Lumbar spine", "spine", "bi", 0.375, 0.760, 0.020, 0.045, "Lumbar spine reflex, lower radial edge."),
    ("sacral-coccyx", "Sacrum and coccyx", "spine", "bi", 0.405, 0.840, 0.025, 0.035, "Sacrum and coccyx reflex, radial edge near the wrist."),
    # ---- pelvis & elimination: heel of palm / wrist ----
    ("hip-pelvis", "Hip and pelvis", "pelvis-elimination", "bi", 0.520, 0.860, 0.050, 0.030, "Hip and pelvis reflex, heel of the palm."),
    ("sciatic", "Sciatic nerve", "pelvis-elimination", "bi", 0.560, 0.835, 0.070, 0.025, "Sciatic-nerve reflex, across the wrist."),
]


def build_zone(slug, anatomy, group, kind, cx, cy, rx, ry, meaning):
    bilateral = kind == "bi"
    if kind == "L":
        zid, side = f"hand-L-{slug[2:] if slug.startswith('L-') else slug}", "left"
    elif kind == "R":
        zid, side = f"hand-R-{slug[2:] if slug.startswith('R-') else slug}", "right"
    else:
        zid, side = f"hand-{slug}", "right"
    return {
        "id": zid,
        "side": side,
        "bilateral": bilateral,
        "group": group,
        "geometry": {"type": "ellipse", "cx": cx, "cy": cy, "rx": rx, "ry": ry},
        "anatomy": anatomy,
        "meaning_standard": meaning,
        "meaning_glen": "",
        "layers": {"embryological_depth": None, "stress_affirmation": None, "touch_for_health": None},
    }


def main():
    data = {
        "system": "hand",
        "reference_frame": "hand_outline",
        "side_noun": "palm",
        "group_noun": "region",
        "groups": GROUPS,
        "anchors": ANCHORS,
        "outline_side": "right",
        "outline_source": "procedural open-hand outline (right hand, palm view); scripts/build_hand_outline.py Catmull-Rom spline",
        "outline": bho.catmull_rom_closed(bho.HAND_POINTS),
        "zones": [build_zone(*z) for z in ZONES],
    }
    out = ROOT / "data" / "bodymap-hand.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"wrote {out} with {len(data['zones'])} zones")


if __name__ == "__main__":
    main()
