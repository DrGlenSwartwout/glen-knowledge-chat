#!/usr/bin/env python3
"""Build data/bodymap-endocrine.json — the endocrine system atlas (front + back).

The hormone-producing glands, each named so a client's gland finding (Thyroid,
Adrenal, Pituitary, Pineal, Pancreas...) lights it.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bodymap_atlas_lib as lib

GROUPS = [
    {"id": "cranial", "label": "Cranial glands"},
    {"id": "neck", "label": "Neck glands"},
    {"id": "adrenal-pancreas", "label": "Adrenal & pancreas"},
    {"id": "gonad", "label": "Gonads"},
]
e, z = lib.ellipse, lib.zone
Z = [
    z("endo-pineal", "Pineal gland", "front", "cranial", e(0.512, 0.058, 0.011, 0.011), "Pineal gland — melatonin, circadian rhythm."),
    z("endo-pituitary", "Pituitary gland", "front", "cranial", e(0.500, 0.074, 0.012, 0.012), "Pituitary — the master gland."),
    z("endo-hypothalamus", "Hypothalamus", "front", "cranial", e(0.488, 0.066, 0.012, 0.010), "Hypothalamus — links nervous and endocrine systems."),
    z("endo-thyroid", "Thyroid gland", "front", "neck", e(0.500, 0.156, 0.028, 0.016), "Thyroid — metabolism."),
    z("endo-parathyroid", "Parathyroid glands", "front", "neck", e(0.500, 0.166, 0.012, 0.008), "Parathyroids — calcium balance."),
    z("endo-thymus", "Thymus", "front", "neck", e(0.500, 0.238, 0.028, 0.026), "Thymus — T-cell maturation, immune."),
    z("endo-adrenal-r", "Adrenal gland (right)", "front", "adrenal-pancreas", e(0.456, 0.402, 0.014, 0.012), "Right adrenal — stress hormones, cortisol, adrenaline."),
    z("endo-adrenal-l", "Adrenal gland (left)", "front", "adrenal-pancreas", e(0.544, 0.402, 0.014, 0.012), "Left adrenal — stress hormones, cortisol, adrenaline."),
    z("endo-pancreas", "Pancreas (islets)", "front", "adrenal-pancreas", e(0.520, 0.388, 0.038, 0.015), "Pancreatic islets — insulin and glucagon."),
    z("endo-ovary-r", "Ovary / testis (right)", "front", "gonad", e(0.462, 0.552, 0.014, 0.014), "Right gonad — sex hormones."),
    z("endo-ovary-l", "Ovary / testis (left)", "front", "gonad", e(0.538, 0.552, 0.014, 0.014), "Left gonad — sex hormones."),
]

lib.write_system("endocrine", GROUPS, Z, group_noun="gland group", views=("front",))
