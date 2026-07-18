#!/usr/bin/env python3
"""Build data/bodymap-respiratory.json — the respiratory system atlas (front + back).

Airway from nose to alveoli, the lungs by lobe, pleura and diaphragm — each named
so a client's respiratory finding (Lung, Bronchi, Trachea, Larynx, Diaphragm...)
lights the structure.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bodymap_atlas_lib as lib

GROUPS = [
    {"id": "upper", "label": "Upper airway"},
    {"id": "lower", "label": "Lower airway"},
    {"id": "lungs", "label": "Lungs"},
    {"id": "mechanics", "label": "Pleura & diaphragm"},
]
e, p, z, cr = lib.ellipse, lib.path, lib.zone, lib.catmull
Z = [
    z("resp-nose", "Nose & nasal sinuses", "front", "upper", e(0.500, 0.088, 0.016, 0.014), "Nasal passages and paranasal sinuses."),
    z("resp-pharynx", "Pharynx", "front", "upper", e(0.500, 0.128, 0.012, 0.016), "Pharynx — the throat."),
    z("resp-larynx", "Larynx", "front", "upper", e(0.500, 0.150, 0.012, 0.012), "Larynx — the voice box."),
    z("resp-trachea", "Trachea", "front", "lower", p(cr([(0.500, 0.166), (0.500, 0.200), (0.500, 0.232)])), "Trachea — the windpipe."),
    z("resp-bronchi", "Bronchi", "front", "lower", e(0.500, 0.250, 0.026, 0.014), "Main bronchi branching to each lung."),
    z("resp-lung-r-upper", "Right lung (upper lobe)", "front", "lungs", e(0.436, 0.246, 0.040, 0.036), "Right lung, upper lobe."),
    z("resp-lung-r-lower", "Right lung (lower lobe)", "front", "lungs", e(0.442, 0.300, 0.040, 0.030), "Right lung, middle and lower lobes."),
    z("resp-lung-l-upper", "Left lung (upper lobe)", "front", "lungs", e(0.564, 0.246, 0.040, 0.036), "Left lung, upper lobe."),
    z("resp-lung-l-lower", "Left lung (lower lobe)", "front", "lungs", e(0.558, 0.300, 0.040, 0.030), "Left lung, lower lobe."),
    z("resp-pleura", "Pleura", "front", "mechanics", e(0.500, 0.276, 0.078, 0.066, ), "Pleural membranes lining the lungs."),
    z("resp-diaphragm", "Diaphragm", "front", "mechanics", e(0.500, 0.352, 0.080, 0.012), "Diaphragm — the main muscle of breathing."),
    # back
    z("resp-lung-rb", "Right lung (posterior)", "back", "lungs", e(0.446, 0.270, 0.046, 0.056), "Right lung, seen from behind."),
    z("resp-lung-lb", "Left lung (posterior)", "back", "lungs", e(0.554, 0.270, 0.046, 0.056), "Left lung, seen from behind."),
    z("resp-diaphragm-b", "Diaphragm (posterior)", "back", "mechanics", e(0.500, 0.352, 0.080, 0.012), "Diaphragm attachment at the lower ribs and spine."),
]

lib.write_system("respiratory", GROUPS, Z, group_noun="region")
