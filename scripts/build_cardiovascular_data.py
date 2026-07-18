#!/usr/bin/env python3
"""Build data/bodymap-cardiovascular.json — the cardiovascular system atlas
(front + back).

The heart, the great vessels and major arteries/veins as stroked paths, each named
so a client's cardiovascular finding (Heart, Aorta, Blood Vessels, Circulation...)
lights the structure.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bodymap_atlas_lib as lib

GROUPS = [
    {"id": "heart", "label": "Heart"},
    {"id": "great-vessels", "label": "Great vessels"},
    {"id": "arteries", "label": "Arteries"},
    {"id": "veins", "label": "Veins"},
]
e, p, z, cr = lib.ellipse, lib.path, lib.zone, lib.catmull
Z = [
    z("cardio-heart", "Heart", "front", "heart", e(0.474, 0.286, 0.036, 0.042), "The heart — four chambers, the central pump."),
    z("cardio-aorta", "Aorta", "front", "great-vessels", p(cr([(0.492, 0.250), (0.500, 0.300), (0.500, 0.400), (0.500, 0.480)])), "Aorta — the main artery from the heart."),
    z("cardio-vena-cava", "Vena cava", "front", "great-vessels", p(cr([(0.516, 0.250), (0.520, 0.320), (0.518, 0.420), (0.516, 0.490)])), "Superior and inferior vena cava — return blood to the heart."),
    z("cardio-pulmonary", "Pulmonary vessels", "front", "great-vessels", e(0.500, 0.256, 0.028, 0.014), "Pulmonary artery and veins — to and from the lungs."),
    z("cardio-carotid-r", "Carotid artery (right)", "front", "arteries", p(cr([(0.478, 0.170), (0.480, 0.130), (0.484, 0.100)])), "Right carotid artery — to the head."),
    z("cardio-carotid-l", "Carotid artery (left)", "front", "arteries", p(cr([(0.522, 0.170), (0.520, 0.130), (0.516, 0.100)])), "Left carotid artery — to the head."),
    z("cardio-artery-arm-r", "Arm artery (right)", "front", "arteries", p(cr([(0.400, 0.240), (0.360, 0.360), (0.320, 0.480)])), "Right brachial artery."),
    z("cardio-artery-arm-l", "Arm artery (left)", "front", "arteries", p(cr([(0.600, 0.240), (0.640, 0.360), (0.680, 0.480)])), "Left brachial artery."),
    z("cardio-iliac-r", "Iliac & femoral artery (right)", "front", "arteries", p(cr([(0.470, 0.520), (0.460, 0.640), (0.458, 0.800)])), "Right iliac and femoral artery — to the leg."),
    z("cardio-iliac-l", "Iliac & femoral artery (left)", "front", "arteries", p(cr([(0.530, 0.520), (0.540, 0.640), (0.542, 0.800)])), "Left iliac and femoral artery — to the leg."),
    z("cardio-portal", "Hepatic portal vein", "front", "veins", e(0.500, 0.380, 0.020, 0.012), "Portal vein — carries gut blood to the liver."),
    # back
    z("cardio-heart-b", "Heart (posterior)", "back", "heart", e(0.500, 0.286, 0.034, 0.040), "The heart, seen from behind."),
    z("cardio-aorta-b", "Descending aorta", "back", "great-vessels", p(cr([(0.500, 0.300), (0.500, 0.400), (0.500, 0.480)])), "Descending aorta along the spine."),
    z("cardio-leg-vein-r", "Leg veins (right)", "back", "veins", p(cr([(0.456, 0.560), (0.454, 0.700), (0.458, 0.880)])), "Right deep leg veins."),
    z("cardio-leg-vein-l", "Leg veins (left)", "back", "veins", p(cr([(0.544, 0.560), (0.546, 0.700), (0.542, 0.880)])), "Left deep leg veins."),
]

lib.write_system("cardiovascular", GROUPS, Z, group_noun="region")
