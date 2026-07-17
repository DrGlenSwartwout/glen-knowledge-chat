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
    # ---- brain & brain-stem (top, both eyes) ----
    # slug "brain" keeps the id iris-R-brain that Atlas cluster deep-links target
    ("brain", "Cerebrum", "both", "ectoderm", 0.66, 0.92, 334, 356, "Cerebrum, upper iris toward 12 o'clock."),
    ("pineal", "Pineal", "both", "ectoderm", 0.72, 0.88, 350, 360, "Pineal, top of the iris."),
    ("pituitary", "Pituitary", "both", "ectoderm", 0.54, 0.72, 0, 12, "Pituitary, at 12 o'clock."),
    ("medulla", "Medulla / brain-stem", "both", "ectoderm", 0.62, 0.76, 6, 18, "Medulla oblongata / brain-stem."),
    ("cerebellum", "Cerebellum", "both", "ectoderm", 0.66, 0.92, 16, 34, "Cerebellum, just past 12 o'clock."),
    # ---- head & sensory (upper, both eyes) ----
    ("nose", "Nose", "both", "ectoderm", 0.54, 0.70, 30, 42, "Nose, upper iris."),
    ("eye-vision", "Eye / vision", "both", "ectoderm", 0.56, 0.74, 40, 52, "Eye and vision centre."),
    ("face-sinus", "Face and sinuses", "both", "ectoderm", 0.60, 0.80, 44, 58, "Face and sinus area."),
    ("ear-hearing", "Ear / hearing", "both", "ectoderm", 0.56, 0.74, 58, 70, "Ear and hearing centre."),
    ("teeth-jaw", "Teeth and jaw", "both", "mesoderm", 0.28, 0.42, 40, 52, "Teeth and jaw."),
    ("mouth-tongue", "Mouth and tongue", "both", "mesoderm", 0.30, 0.44, 50, 62, "Mouth and tongue."),
    ("tonsils", "Tonsils", "both", "mesoderm", 0.34, 0.48, 60, 72, "Tonsils."),
    ("throat-neck", "Throat and neck", "both", "mesoderm", 0.42, 0.58, 62, 76, "Throat and neck (~2 o'clock)."),
    # ---- neck / chest (both eyes) ----
    ("trachea-esophagus", "Trachea / oesophagus", "both", "endoderm", 0.14, 0.28, 74, 90, "Trachea and oesophagus."),
    ("thyroid", "Thyroid", "both", "mesoderm", 0.34, 0.50, 76, 90, "Thyroid gland (~2:30)."),
    ("parathyroid", "Parathyroid", "both", "mesoderm", 0.32, 0.44, 88, 98, "Parathyroid, beside the thyroid."),
    ("bronchi-lung", "Lung and bronchi", "both", "mesoderm", 0.42, 0.62, 92, 116, "Lung and bronchials (~3 o'clock)."),
    ("thymus", "Thymus", "both", "mesoderm", 0.32, 0.46, 108, 120, "Thymus, upper chest."),
    ("shoulder-chest", "Shoulder and chest", "both", "ectoderm", 0.62, 0.85, 116, 132, "Shoulder / upper chest (~4 o'clock)."),
    ("breast-mammary", "Breast / mammary", "both", "ectoderm", 0.58, 0.78, 120, 136, "Breast / mammary area."),
    ("diaphragm", "Diaphragm", "both", "mesoderm", 0.34, 0.48, 128, 142, "Diaphragm."),
    ("solar-plexus", "Solar plexus", "both", "mesoderm", 0.30, 0.42, 142, 156, "Solar plexus."),
    # ---- lower body (both eyes) ----
    ("adrenal", "Adrenal gland", "both", "mesoderm", 0.42, 0.56, 150, 166, "Adrenal gland, above the kidney (~5:30)."),
    ("kidney", "Kidney", "both", "mesoderm", 0.40, 0.58, 162, 192, "Kidney (~6 o'clock)."),
    ("ureter", "Ureter", "both", "mesoderm", 0.32, 0.46, 176, 192, "Ureter, kidney to bladder."),
    ("bladder", "Bladder", "both", "endoderm", 0.22, 0.34, 172, 190, "Bladder, lower iris near the pupil (~6 o'clock)."),
    ("urethra", "Urethra", "both", "endoderm", 0.14, 0.26, 184, 196, "Urethra."),
    ("rectum", "Rectum", "both", "endoderm", 0.20, 0.32, 188, 202, "Rectum."),
    ("reproductive", "Reproductive / pelvic", "both", "mesoderm", 0.34, 0.50, 194, 210, "Reproductive and pelvic organs (~6:30)."),
    # ---- spine, pelvis, limb (both eyes, ~7 o'clock) ----
    ("cervical-spine", "Cervical spine", "both", "mesoderm", 0.50, 0.62, 196, 206, "Cervical spine."),
    ("thoracic-spine", "Thoracic spine", "both", "mesoderm", 0.52, 0.64, 206, 218, "Thoracic / dorsal spine."),
    ("lumbar-spine", "Lumbar spine", "both", "mesoderm", 0.52, 0.64, 218, 228, "Lumbar spine (~7 o'clock)."),
    ("sacrum-coccyx", "Sacrum and coccyx", "both", "mesoderm", 0.50, 0.62, 228, 240, "Sacrum and coccyx."),
    ("pelvis-hip", "Pelvis and hip", "both", "ectoderm", 0.60, 0.80, 208, 226, "Pelvis and hip."),
    ("leg-limb", "Leg and lower limb", "both", "ectoderm", 0.66, 0.90, 200, 224, "Lower limb / leg (~7 o'clock)."),
    # ---- outer rings (annular, both eyes) ----
    ("lymphatic", "Lymphatic / circulation", "both", "ectoderm", 0.80, 0.90, 0, 360, "Lymphatic and circulation ring."),
    ("skin", "Skin / elimination", "both", "ectoderm", 0.90, 1.0, 0, 360, "Skin and elimination, the outermost ring."),
    # ---- RIGHT-iris organs ----
    ("liver", "Liver", "right", "mesoderm", 0.38, 0.60, 228, 258, "Liver, lower nasal right iris (~8 o'clock)."),
    ("gallbladder", "Gallbladder", "right", "mesoderm", 0.40, 0.55, 248, 262, "Gallbladder, beside the liver (~8:20)."),
    ("duodenum", "Duodenum", "right", "endoderm", 0.20, 0.32, 256, 270, "Duodenum, right iris."),
    ("appendix", "Appendix / ileocecal", "right", "endoderm", 0.22, 0.32, 250, 262, "Appendix and ileocecal valve (~8:30)."),
    ("pancreas-head", "Pancreas (head)", "right", "mesoderm", 0.36, 0.52, 262, 280, "Head of the pancreas (~9 o'clock)."),
    ("ascending-colon", "Ascending colon", "right", "endoderm", 0.24, 0.36, 270, 298, "Ascending colon, right iris (~9:30)."),
    ("transverse-colon", "Transverse colon", "right", "endoderm", 0.24, 0.36, 300, 330, "Transverse colon, upper iris."),
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
