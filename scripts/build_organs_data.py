#!/usr/bin/env python3
"""Build data/bodymap-organs.json — a whole-body organ atlas.

The major internal organs as labelled `ellipse` zones at their anatomical
positions on the body silhouette (front + back views), each named for its organ
so a client's findings light the real organ location. This is the whole-body
counterpart to the face-diagnosis map: a Liver finding lights the liver here, a
Kidney finding the kidneys, a Brain finding the head — including organs that have
no facial zone.

Positions are a standards-based rendering for Glen's clinical refinement, not
gospel. Laterality is anatomical-as-viewed (anterior view: the patient's right
organ sits on the viewer's left, x < 0.5). Front + back reuse the meridian
silhouette + per-view outlines.
"""
import json
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("bbo", ROOT / "scripts" / "build_body_outline.py")
bbo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bbo)

GROUPS = [
    {"id": "neuro-endocrine", "label": "Brain & glands"},
    {"id": "cardio-respiratory", "label": "Heart & lungs"},
    {"id": "digestive", "label": "Digestive organs"},
    {"id": "urinary", "label": "Urinary & adrenal"},
    {"id": "reproductive", "label": "Reproductive"},
    {"id": "structural", "label": "Spine & structure"},
]

# ORGANS: (slug, name, view, group, cx, cy, rx, ry, meaning)
ORGANS = [
    # ---- FRONT: neuro-endocrine ----
    ("brain", "Brain", "front", "neuro-endocrine", 0.500, 0.055, 0.046, 0.050, "The brain, within the cranium."),
    ("pituitary", "Pituitary gland", "front", "neuro-endocrine", 0.500, 0.083, 0.013, 0.013, "Pituitary, the master endocrine gland at the skull base."),
    ("pineal", "Pineal gland", "front", "neuro-endocrine", 0.517, 0.073, 0.011, 0.011, "Pineal gland, deep in the midbrain."),
    ("thyroid", "Thyroid & parathyroid", "front", "neuro-endocrine", 0.500, 0.156, 0.030, 0.018, "Thyroid and parathyroid glands, front of the neck."),
    ("thymus", "Thymus", "front", "neuro-endocrine", 0.500, 0.238, 0.030, 0.030, "Thymus, upper anterior mediastinum."),
    # ---- FRONT: cardio-respiratory ----
    ("heart", "Heart", "front", "cardio-respiratory", 0.474, 0.286, 0.036, 0.042, "The heart, centre-left of the chest."),
    ("lung-r", "Lung (right)", "front", "cardio-respiratory", 0.436, 0.268, 0.042, 0.058, "Right lung."),
    ("lung-l", "Lung (left)", "front", "cardio-respiratory", 0.564, 0.268, 0.042, 0.058, "Left lung."),
    ("bronchi", "Bronchi & trachea", "front", "cardio-respiratory", 0.500, 0.210, 0.016, 0.036, "Trachea and main bronchi."),
    # ---- FRONT: digestive ----
    ("liver", "Liver", "front", "digestive", 0.438, 0.366, 0.056, 0.036, "Liver, right upper abdomen."),
    ("gallbladder", "Gallbladder", "front", "digestive", 0.462, 0.388, 0.015, 0.015, "Gallbladder, under the liver."),
    ("stomach", "Stomach", "front", "digestive", 0.552, 0.366, 0.042, 0.032, "Stomach, left upper abdomen."),
    ("spleen", "Spleen", "front", "digestive", 0.588, 0.360, 0.022, 0.024, "Spleen, far left upper abdomen."),
    ("pancreas", "Pancreas", "front", "digestive", 0.520, 0.388, 0.038, 0.015, "Pancreas, behind the stomach."),
    ("duodenum", "Duodenum", "front", "digestive", 0.520, 0.412, 0.020, 0.014, "Duodenum, first part of the small intestine."),
    ("small-intestine", "Small intestine", "front", "digestive", 0.500, 0.462, 0.056, 0.046, "Small intestine, coiled in the central abdomen."),
    ("colon-ascending", "Ascending colon", "front", "digestive", 0.432, 0.452, 0.016, 0.040, "Ascending colon (large intestine), right side."),
    ("colon-transverse", "Transverse colon", "front", "digestive", 0.500, 0.424, 0.062, 0.014, "Transverse colon (large intestine), across the upper abdomen."),
    ("colon-descending", "Descending colon", "front", "digestive", 0.568, 0.452, 0.016, 0.040, "Descending colon (large intestine), left side."),
    ("colon-sigmoid", "Sigmoid colon & rectum", "front", "digestive", 0.470, 0.505, 0.026, 0.018, "Sigmoid colon and rectum, lower left."),
    ("appendix", "Appendix & ileocecal", "front", "digestive", 0.442, 0.492, 0.013, 0.013, "Appendix and ileocecal valve, lower right."),
    # ---- FRONT: urinary & adrenal ----
    ("adrenal-r", "Adrenal gland (right)", "front", "urinary", 0.456, 0.402, 0.013, 0.011, "Right adrenal (suprarenal) gland, atop the kidney."),
    ("adrenal-l", "Adrenal gland (left)", "front", "urinary", 0.544, 0.402, 0.013, 0.011, "Left adrenal (suprarenal) gland, atop the kidney."),
    ("kidney-r", "Kidney (right)", "front", "urinary", 0.450, 0.424, 0.022, 0.032, "Right kidney."),
    ("kidney-l", "Kidney (left)", "front", "urinary", 0.550, 0.424, 0.022, 0.032, "Left kidney."),
    ("bladder", "Bladder & urethra", "front", "urinary", 0.500, 0.538, 0.026, 0.024, "Urinary bladder, in the pelvis."),
    # ---- FRONT: reproductive ----
    ("reproductive", "Reproductive organs", "front", "reproductive", 0.500, 0.560, 0.032, 0.020, "Reproductive organs (uterus / ovaries or prostate), pelvis."),
    ("ovary-r", "Ovary / testis (right)", "front", "reproductive", 0.462, 0.552, 0.013, 0.013, "Right ovary or testis."),
    ("ovary-l", "Ovary / testis (left)", "front", "reproductive", 0.538, 0.552, 0.013, 0.013, "Left ovary or testis."),
    # ---- BACK: structural spine + kidneys/brain ----
    ("cerebellum", "Cerebellum & brainstem", "back", "neuro-endocrine", 0.500, 0.070, 0.038, 0.036, "Cerebellum and brainstem, back of the head."),
    ("cervical-spine", "Cervical spine", "back", "structural", 0.500, 0.170, 0.016, 0.030, "Cervical vertebrae (C1-C7), the neck."),
    ("thoracic-spine", "Thoracic spine", "back", "structural", 0.500, 0.300, 0.016, 0.090, "Thoracic vertebrae (T1-T12), the mid back."),
    ("lumbar-spine", "Lumbar spine", "back", "structural", 0.500, 0.430, 0.018, 0.045, "Lumbar vertebrae (L1-L5), the low back."),
    ("sacrum", "Sacrum & coccyx", "back", "structural", 0.500, 0.490, 0.020, 0.028, "Sacrum and coccyx, base of the spine."),
    ("kidney-r-b", "Kidney (right, posterior)", "back", "urinary", 0.452, 0.410, 0.022, 0.032, "Right kidney, seen from behind."),
    ("kidney-l-b", "Kidney (left, posterior)", "back", "urinary", 0.548, 0.410, 0.022, 0.032, "Left kidney, seen from behind."),
    ("adrenal-r-b", "Adrenal gland (right, posterior)", "back", "urinary", 0.458, 0.388, 0.013, 0.011, "Right adrenal gland, posterior."),
    ("adrenal-l-b", "Adrenal gland (left, posterior)", "back", "urinary", 0.542, 0.388, 0.013, 0.011, "Left adrenal gland, posterior."),
    ("lungs-back", "Lungs (posterior)", "back", "cardio-respiratory", 0.500, 0.270, 0.070, 0.058, "Lungs, seen from behind."),
]


def _mk(slug, name, view, group, cx, cy, rx, ry, meaning):
    return {
        "id": f"organ-{slug}", "side": view, "bilateral": False, "group": group,
        "geometry": {"type": "ellipse", "cx": round(cx, 4), "cy": round(cy, 4),
                     "rx": round(rx, 4), "ry": round(ry, 4)},
        "anatomy": name, "meaning_standard": meaning, "meaning_glen": "",
        "layers": {"embryological_depth": None, "stress_affirmation": None, "touch_for_health": None},
    }


def main():
    zones = [_mk(*o) for o in ORGANS]
    front = bbo.build_outline()
    data = {
        "system": "organs",
        "reference_frame": "body_outline",
        "side_noun": "view",
        "group_noun": "system",
        "outline": front,
        "outlines": {"front": front, "back": front},
        "groups": GROUPS,
        "anchors": [
            {"key": "head-f", "view": "front", "template": {"x": 0.50, "y": 0.02}, "hint": "Tap the top of the head."},
            {"key": "feet-f", "view": "front", "template": {"x": 0.50, "y": 0.985}, "hint": "Tap the point between the ankles."},
            {"key": "head-b", "view": "back", "template": {"x": 0.50, "y": 0.02}, "hint": "Tap the top of the head."},
            {"key": "feet-b", "view": "back", "template": {"x": 0.50, "y": 0.985}, "hint": "Tap the point between the ankles."},
        ],
        "zones": zones,
    }
    out = ROOT / "data" / "bodymap-organs.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    from collections import Counter
    print(f"wrote {out}: {len(zones)} zones", dict(Counter(z["side"] for z in zones)))


if __name__ == "__main__":
    main()
