#!/usr/bin/env python3
"""Build data/bodymap-iridology.json — a full iris (iridology) chart.

Renders the classic Jensen/Bernard iris map onto the engine's `sector` geometry:
each zone is an annular sector defined by a radial band (0 = pupil, 1 = iris rim)
and a clock range. Clock convention (matches static/body-map.js clockToNormalized):
deg = clock_hour * 30, with 0deg = 12 o'clock, 90 = 3:00, 180 = 6:00, 270 = 9:00.

Radial organisation follows the existing germ-layer rings:
  endoderm  0.00-0.33  digestive & glandular  (stomach/intestine rings inner)
  mesoderm  0.33-0.66  organs, muscle, skeletal, circulatory
  ectoderm  0.66-1.00  brain/sensory (upper) and the lymph & skin rings (outer)

Right iris = right side of the body; left iris = left side. Midline / paired
structures appear (mirrored) in both eyes; side organs (liver, gallbladder,
ascending colon on the RIGHT; heart, spleen, pancreas tail, descending & sigmoid
colon on the LEFT) appear only in their eye — mirroring the foot/hand asymmetry.

Positions are a faithful rendering of the published Jensen chart and are meant
for Glen's clinical refinement via the admin drawing tool, not gospel.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GERM_LAYERS = [
    {"id": "endoderm", "label": "Endoderm (inner ring - digestive & glandular)", "r_inner": 0.0, "r_outer": 0.33},
    {"id": "mesoderm", "label": "Mesoderm (middle ring - muscle, skeletal, circulatory)", "r_inner": 0.33, "r_outer": 0.66},
    {"id": "ectoderm", "label": "Ectoderm (outer ring - skin & nervous system)", "r_inner": 0.66, "r_outer": 1.0},
]

# Each entry: (slug, anatomy, eye, germ_layer, r_inner, r_outer, deg_start, deg_end, meaning)
# eye: "right" | "left" | "both" (both = emitted in each eye, clock mirrored for left)
# For "both" zones the degrees are given for the RIGHT iris; the left copy is
# reflected across the vertical (12-6) axis: (360-end, 360-start), 0-360 kept.
ZONES = [
    # ---- digestive rings (annular, both eyes) ----
    ("stomach", "Stomach", "both", "endoderm", 0.06, 0.18, 0, 360, "Stomach ring, immediately around the pupil."),
    ("intestines", "Intestinal tract", "both", "endoderm", 0.18, 0.30, 0, 360, "Bowel / intestinal ring, just outside the stomach."),
    # ---- brain & sensory (top, both eyes) ----
    # slug "brain" keeps the id iris-R-brain that Atlas cluster deep-links target
    ("brain", "Cerebrum", "both", "ectoderm", 0.66, 0.92, 334, 358, "Cerebrum, upper iris toward 12 o'clock."),
    ("pituitary", "Pituitary", "both", "ectoderm", 0.56, 0.74, 0, 14, "Pituitary, at 12 o'clock."),
    ("cerebellum", "Cerebellum", "both", "ectoderm", 0.66, 0.92, 16, 40, "Cerebellum, just past 12 o'clock."),
    ("face-sinus", "Face and sinuses", "both", "ectoderm", 0.60, 0.82, 42, 62, "Face and sinus area, upper iris."),
    # ---- neck / thyroid / chest ----
    ("throat-neck", "Throat and neck", "both", "mesoderm", 0.40, 0.58, 54, 74, "Throat and neck (~2 o'clock)."),
    ("thyroid", "Thyroid", "both", "mesoderm", 0.36, 0.54, 70, 88, "Thyroid gland (~2:30)."),
    ("bronchi-lung", "Lung and bronchi", "both", "mesoderm", 0.40, 0.62, 88, 112, "Lung and bronchials (~3 o'clock)."),
    ("shoulder-chest", "Shoulder and chest", "both", "ectoderm", 0.62, 0.85, 106, 128, "Shoulder / upper chest (~4 o'clock)."),
    # ---- lower body (both eyes) ----
    ("kidney", "Kidney", "both", "mesoderm", 0.40, 0.58, 162, 192, "Kidney (~6 o'clock)."),
    ("adrenal", "Adrenal gland", "both", "mesoderm", 0.42, 0.56, 148, 166, "Adrenal gland, above the kidney (~5:30)."),
    ("bladder", "Bladder", "both", "endoderm", 0.22, 0.34, 170, 190, "Bladder, lower iris near the pupil (~6 o'clock)."),
    ("reproductive", "Reproductive / pelvic", "both", "mesoderm", 0.34, 0.50, 192, 210, "Reproductive and pelvic organs (~6:30)."),
    ("leg-limb", "Leg and lower limb", "both", "ectoderm", 0.62, 0.86, 196, 220, "Lower limb / leg (~7 o'clock)."),
    ("lumbar-spine", "Lumbar spine", "both", "mesoderm", 0.54, 0.66, 205, 224, "Lumbar spine (~7 o'clock)."),
    # ---- outer rings (annular, both eyes) ----
    ("lymphatic", "Lymphatic / circulation", "both", "ectoderm", 0.80, 0.90, 0, 360, "Lymphatic and circulation ring."),
    ("skin", "Skin / elimination", "both", "ectoderm", 0.90, 1.0, 0, 360, "Skin and elimination, the outermost ring."),
    # ---- RIGHT-iris organs ----
    ("liver", "Liver", "right", "mesoderm", 0.38, 0.60, 228, 258, "Liver, lower nasal right iris (~8 o'clock)."),
    ("gallbladder", "Gallbladder", "right", "mesoderm", 0.40, 0.55, 246, 262, "Gallbladder, beside the liver (~8:20)."),
    ("pancreas-head", "Pancreas (head)", "right", "mesoderm", 0.36, 0.52, 262, 280, "Head of the pancreas (~9 o'clock)."),
    ("ascending-colon", "Ascending colon", "right", "endoderm", 0.24, 0.36, 268, 296, "Ascending colon, right iris (~9:30)."),
    ("appendix", "Appendix / ileocecal", "right", "endoderm", 0.22, 0.32, 250, 264, "Appendix and ileocecal valve (~8:30)."),
    # ---- LEFT-iris organs ----
    ("heart", "Heart", "left", "mesoderm", 0.42, 0.62, 88, 116, "Heart, left iris (~3 o'clock)."),
    ("spleen", "Spleen", "left", "mesoderm", 0.40, 0.56, 118, 140, "Spleen, left iris (~4 o'clock)."),
    ("pancreas-tail", "Pancreas (tail)", "left", "mesoderm", 0.36, 0.52, 100, 118, "Tail of the pancreas, left iris."),
    ("descending-colon", "Descending colon", "left", "endoderm", 0.24, 0.36, 64, 92, "Descending colon, left iris (~2:30)."),
    ("sigmoid-colon", "Sigmoid colon", "left", "endoderm", 0.22, 0.34, 150, 172, "Sigmoid colon, left iris (~5:00)."),
]


def _mirror(a, b):
    """Reflect a clock range across the vertical axis for the left iris."""
    if a == 0 and b == 360:
        return 0, 360
    return 360 - b, 360 - a


def build_zone(slug, anatomy, eye, layer, ri, ro, ds, de, meaning):
    tag = {"right": "R", "left": "L", "both": "R"}[eye]
    return {
        "id": f"iris-{tag}-{slug}",
        "eye": eye if eye != "both" else "right",
        "germ_layer": layer,
        "radial": {"r_inner": ri, "r_outer": ro},
        "sector": {"start_deg": ds, "end_deg": de},
        "anatomy": anatomy,
        "meaning_standard": meaning,
        "meaning_glen": "",
        "layers": {"embryological_depth": None, "stress_affirmation": None, "touch_for_health": None},
    }


def main():
    zones = []
    for slug, anatomy, eye, layer, ri, ro, ds, de, meaning in ZONES:
        if eye in ("right", "both"):
            zones.append(build_zone(slug, anatomy, "right", layer, ri, ro, ds, de, meaning))
        if eye == "left":
            zones.append(build_zone(slug, anatomy, "left", layer, ri, ro, ds, de, meaning))
        elif eye == "both":
            ms, me = _mirror(ds, de)
            z = build_zone(slug, anatomy, "left", layer, ri, ro, ms, me, meaning)
            z["id"] = f"iris-L-{slug}"
            zones.append(z)
    data = {
        "system": "iridology",
        "reference_frame": "unit_circle",
        "germ_layers": GERM_LAYERS,
        "zones": zones,
    }
    out = ROOT / "data" / "bodymap-iridology.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    n_r = sum(1 for z in zones if z["eye"] == "right")
    n_l = sum(1 for z in zones if z["eye"] == "left")
    print(f"wrote {out}: {len(zones)} zones ({n_r} right, {n_l} left)")


if __name__ == "__main__":
    main()
