#!/usr/bin/env python3
"""Build data/bodymap-sclerology.json — a full sclera (sclerology) chart.

Sclerology reads the markings/vessels in the white of the eye. Like the iris it
maps the body around a clock face, but in the scleral ring OUTSIDE the iris
(r 1.0 = limbus -> 3.0 toward the canthus). Same `sector` geometry and clock
convention as the iris (deg = hour*30; 0 = 12 o'clock, 90 = 3:00, 180 = 6:00).

Radial bands (germ_layers, reused from the existing seed):
  perilimbal    1.0-1.6  nearest the iris
  mid-sclera    1.6-2.3
  outer-sclera  2.3-3.0  toward the canthus

Right sclera = right side of the body; left = left. Side organs (liver,
gallbladder, ascending colon, appendix RIGHT; heart, spleen, pancreas,
descending & sigmoid colon LEFT) appear only in their eye; midline/paired
structures appear mirrored in both. Organ clock positions follow the same
body-on-the-eye layout as the iris chart.

Positions are a standards-based rendering for Glen's clinical refinement via the
admin drawing tool, not gospel.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BANDS = [
    {"id": "perilimbal", "label": "Perilimbal (nearest the iris)", "r_inner": 1.0, "r_outer": 1.6},
    {"id": "mid-sclera", "label": "Mid sclera", "r_inner": 1.6, "r_outer": 2.3},
    {"id": "outer-sclera", "label": "Outer sclera (toward the canthus)", "r_inner": 2.3, "r_outer": 3.0},
]
BAND_R = {b["id"]: (b["r_inner"], b["r_outer"]) for b in BANDS}

# (slug, anatomy, eye, band, deg_start, deg_end, meaning)  eye: right|left|both
ZONES = [
    # ---- head & brain (outer band, top) ----
    ("brain", "Brain / cerebrum", "both", "outer-sclera", 334, 356, "Cerebral / head markings, upper sclera."),
    ("pituitary", "Pituitary / pineal", "both", "outer-sclera", 0, 12, "Pituitary / pineal, top of the sclera."),
    ("sinus-face", "Face and sinuses", "both", "outer-sclera", 40, 60, "Face and sinus region."),
    ("eye-ear", "Eye and ear", "both", "outer-sclera", 58, 74, "Eye and ear region."),
    # ---- neck / chest ----
    ("throat-neck", "Throat and neck", "both", "mid-sclera", 60, 78, "Throat and neck (~2 o'clock)."),
    ("thyroid", "Thyroid", "both", "mid-sclera", 74, 90, "Thyroid (~2:30)."),
    ("lung-bronchi", "Lung and bronchi", "both", "mid-sclera", 90, 114, "Lung and bronchials (~3 o'clock)."),
    ("shoulder-arm", "Shoulder and arm", "both", "outer-sclera", 112, 130, "Shoulder / arm (~4 o'clock)."),
    ("breast-chest", "Breast and chest", "both", "outer-sclera", 118, 136, "Breast / chest wall."),
    ("diaphragm", "Diaphragm", "both", "mid-sclera", 128, 144, "Diaphragm."),
    ("solar-plexus", "Solar plexus", "both", "perilimbal", 142, 156, "Solar plexus."),
    # ---- lower trunk ----
    ("adrenal", "Adrenal gland", "both", "mid-sclera", 150, 166, "Adrenal gland (~5:30)."),
    ("kidney", "Kidney", "both", "mid-sclera", 162, 192, "Kidney (~6 o'clock)."),
    ("ureter", "Ureter", "both", "perilimbal", 176, 192, "Ureter."),
    ("bladder", "Bladder", "both", "perilimbal", 170, 190, "Bladder (~6 o'clock)."),
    ("prostate-uterus", "Prostate / uterus", "both", "mid-sclera", 192, 202, "Prostate / uterus."),
    ("reproductive", "Ovaries / testes", "both", "mid-sclera", 202, 212, "Ovaries / testes."),
    # ---- spine, pelvis, limb ----
    ("cervical-spine", "Cervical spine", "both", "mid-sclera", 196, 206, "Cervical spine."),
    ("thoracic-spine", "Thoracic spine", "both", "mid-sclera", 206, 218, "Thoracic spine."),
    ("lumbar-spine", "Lumbar spine", "both", "mid-sclera", 218, 228, "Lumbar spine."),
    ("sacrum-coccyx", "Sacrum and coccyx", "both", "mid-sclera", 228, 238, "Sacrum and coccyx."),
    ("pelvis-hip", "Pelvis and hip", "both", "outer-sclera", 208, 226, "Pelvis and hip."),
    ("leg-limb", "Leg and lower limb", "both", "outer-sclera", 196, 214, "Lower limb / leg (~7 o'clock)."),
    # ---- digestive (perilimbal) ----
    ("stomach", "Stomach", "both", "perilimbal", 130, 156, "Stomach region."),
    ("colon-transverse", "Transverse colon", "both", "perilimbal", 300, 330, "Transverse colon, upper sclera."),
    ("lymphatic", "Lymphatic / immune", "both", "outer-sclera", 156, 176, "Lymphatic / immune congestion."),
    # ---- RIGHT sclera organs ----
    ("liver", "Liver / gallbladder line", "right", "perilimbal", 235, 265, "Liver and gallbladder, lower nasal right sclera (~8 o'clock)."),
    ("gallbladder", "Gallbladder", "right", "perilimbal", 250, 266, "Gallbladder."),
    ("pancreas-head", "Pancreas (head)", "right", "mid-sclera", 262, 280, "Head of the pancreas (~9 o'clock)."),
    ("ascending-colon", "Ascending colon", "right", "perilimbal", 268, 296, "Ascending colon, right sclera."),
    ("appendix", "Appendix / ileocecal", "right", "perilimbal", 250, 262, "Appendix / ileocecal valve."),
    # ---- LEFT sclera organs ----
    ("heart", "Heart line", "left", "mid-sclera", 92, 120, "Heart, left sclera (~3 o'clock)."),
    ("spleen", "Spleen", "left", "mid-sclera", 118, 140, "Spleen, left sclera (~4 o'clock)."),
    ("pancreas-tail", "Pancreas (tail)", "left", "mid-sclera", 100, 118, "Tail of the pancreas, left sclera."),
    ("descending-colon", "Descending colon", "left", "perilimbal", 64, 92, "Descending colon, left sclera."),
    ("sigmoid-colon", "Sigmoid colon", "left", "perilimbal", 150, 172, "Sigmoid colon, left sclera."),
]


def _mirror(a, b):
    if a == 0 and b == 360:
        return 0, 360
    return 360 - b, 360 - a


def build_zone(slug, anatomy, eye, band, ds, de, meaning):
    tag = {"right": "R", "left": "L", "both": "R"}[eye]
    ri, ro = BAND_R[band]
    return {
        "id": f"sclera-{tag}-{slug}",
        "eye": eye if eye != "both" else "right",
        "germ_layer": band,
        "radial": {"r_inner": ri, "r_outer": ro},
        "sector": {"start_deg": ds, "end_deg": de},
        "anatomy": anatomy,
        "meaning_standard": meaning,
        "meaning_glen": "",
        "layers": {"embryological_depth": None, "stress_affirmation": None, "touch_for_health": None},
    }


def main():
    zones = []
    for slug, anatomy, eye, band, ds, de, meaning in ZONES:
        if eye in ("right", "both"):
            zones.append(build_zone(slug, anatomy, "right", band, ds, de, meaning))
        if eye == "left":
            zones.append(build_zone(slug, anatomy, "left", band, ds, de, meaning))
        elif eye == "both":
            ms, me = _mirror(ds, de)
            z = build_zone(slug, anatomy, "left", band, ms, me, meaning)
            z["id"] = f"sclera-L-{slug}"
            zones.append(z)
    data = {
        "system": "sclerology",
        "reference_frame": "unit_circle",
        "germ_layers": BANDS,
        "zones": zones,
    }
    out = ROOT / "data" / "bodymap-sclerology.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    n_r = sum(1 for z in zones if z["eye"] == "right")
    n_l = sum(1 for z in zones if z["eye"] == "left")
    print(f"wrote {out}: {len(zones)} zones ({n_r} right, {n_l} left)")


if __name__ == "__main__":
    main()
