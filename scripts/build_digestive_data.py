#!/usr/bin/env python3
"""Build data/bodymap-digestive.json — the digestive system atlas with accessory
organs (front + back).

The GI tract from mouth to rectum plus the accessory organs (salivary glands,
liver, gallbladder, pancreas), each named so a client's digestive finding
(Stomach, Liver, Gallbladder, Colon, Small Intestine, Pancreas...) lights it.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bodymap_atlas_lib as lib

GROUPS = [
    {"id": "upper-gi", "label": "Mouth to stomach"},
    {"id": "small-intestine", "label": "Small intestine"},
    {"id": "large-intestine", "label": "Large intestine"},
    {"id": "accessory", "label": "Accessory organs"},
]
e, p, z, cr = lib.ellipse, lib.path, lib.zone, lib.catmull
Z = [
    z("dig-salivary", "Salivary glands", "front", "accessory", e(0.470, 0.116, 0.014, 0.010), "Salivary glands — begin carbohydrate digestion."),
    z("dig-mouth", "Mouth & tongue", "front", "upper-gi", e(0.500, 0.116, 0.016, 0.010), "Mouth, tongue and teeth — mechanical digestion begins."),
    z("dig-esophagus", "Esophagus", "front", "upper-gi", p(cr([(0.500, 0.150), (0.502, 0.220), (0.510, 0.320)])), "Esophagus — carries food to the stomach."),
    z("dig-stomach", "Stomach", "front", "upper-gi", e(0.552, 0.366, 0.042, 0.032), "Stomach — acid and enzyme digestion."),
    z("dig-liver", "Liver", "front", "accessory", e(0.438, 0.366, 0.056, 0.036), "Liver — bile, detoxification, metabolism."),
    z("dig-gallbladder", "Gallbladder", "front", "accessory", e(0.462, 0.388, 0.015, 0.015), "Gallbladder — stores and concentrates bile."),
    z("dig-pancreas", "Pancreas", "front", "accessory", e(0.520, 0.388, 0.038, 0.015), "Pancreas — digestive enzymes and bicarbonate."),
    z("dig-duodenum", "Duodenum", "front", "small-intestine", e(0.522, 0.412, 0.020, 0.014), "Duodenum — first part of the small intestine."),
    z("dig-jejunum-ileum", "Jejunum & ileum", "front", "small-intestine", e(0.500, 0.462, 0.054, 0.044), "Jejunum and ileum — nutrient absorption."),
    z("dig-colon-asc", "Ascending colon", "front", "large-intestine", e(0.432, 0.452, 0.016, 0.040), "Ascending colon (large intestine)."),
    z("dig-colon-trans", "Transverse colon", "front", "large-intestine", e(0.500, 0.424, 0.062, 0.014), "Transverse colon (large intestine)."),
    z("dig-colon-desc", "Descending colon", "front", "large-intestine", e(0.568, 0.452, 0.016, 0.040), "Descending colon (large intestine)."),
    z("dig-sigmoid", "Sigmoid colon & rectum", "front", "large-intestine", e(0.470, 0.505, 0.026, 0.018), "Sigmoid colon and rectum."),
    z("dig-appendix", "Appendix & ileocecal", "front", "large-intestine", e(0.442, 0.492, 0.013, 0.013), "Appendix and ileocecal valve, lower right."),
]

lib.write_system("digestive", GROUPS, Z, group_noun="region", views=("front",))
