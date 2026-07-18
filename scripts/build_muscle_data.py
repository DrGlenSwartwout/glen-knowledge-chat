#!/usr/bin/env python3
"""Build data/bodymap-muscle.json — a whole-body muscular atlas.

Major muscle groups as labelled `ellipse` zones on the body silhouette (front +
back), each named for its muscle so a client's Muscle finding lights the
musculature (whole-system theme in the personalization endpoint) and a specific
structure lights that muscle. Front + back reuse the meridian silhouette.

Positions are a standards-based rendering for Glen's clinical refinement, not
gospel. Laterality is anatomical-as-viewed (patient's right = viewer's left, x<0.5).
"""
import json
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("bbo", ROOT / "scripts" / "build_body_outline.py")
bbo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bbo)

GROUPS = [
    {"id": "head-neck", "label": "Head & neck"},
    {"id": "shoulder-arm", "label": "Shoulder & arm"},
    {"id": "chest-back", "label": "Chest & back"},
    {"id": "core", "label": "Core & abdomen"},
    {"id": "hip-thigh", "label": "Hip & thigh"},
    {"id": "lower-leg", "label": "Lower leg"},
]

# (slug, name, view, group, cx, cy, rx, ry, meaning)
MUSCLES = [
    # ---- FRONT ----
    ("scm-r", "Sternocleidomastoid (right)", "front", "head-neck", 0.470, 0.150, 0.012, 0.024, "Right SCM, side of the neck."),
    ("scm-l", "Sternocleidomastoid (left)", "front", "head-neck", 0.530, 0.150, 0.012, 0.024, "Left SCM, side of the neck."),
    ("trapezius-f", "Trapezius (upper)", "front", "chest-back", 0.500, 0.180, 0.060, 0.014, "Upper trapezius, across the shoulders."),
    ("deltoid-rf", "Deltoid (right)", "front", "shoulder-arm", 0.372, 0.244, 0.026, 0.030, "Right deltoid, the shoulder cap."),
    ("deltoid-lf", "Deltoid (left)", "front", "shoulder-arm", 0.628, 0.244, 0.026, 0.030, "Left deltoid, the shoulder cap."),
    ("pectoralis-r", "Pectoralis (right)", "front", "chest-back", 0.452, 0.262, 0.036, 0.028, "Right pectoral (chest) muscle."),
    ("pectoralis-l", "Pectoralis (left)", "front", "chest-back", 0.548, 0.262, 0.036, 0.028, "Left pectoral (chest) muscle."),
    ("biceps-r", "Biceps (right)", "front", "shoulder-arm", 0.352, 0.316, 0.020, 0.044, "Right biceps brachii."),
    ("biceps-l", "Biceps (left)", "front", "shoulder-arm", 0.648, 0.316, 0.020, 0.044, "Left biceps brachii."),
    ("forearm-rf", "Forearm flexors (right)", "front", "shoulder-arm", 0.322, 0.420, 0.018, 0.052, "Right forearm flexors."),
    ("forearm-lf", "Forearm flexors (left)", "front", "shoulder-arm", 0.678, 0.420, 0.018, 0.052, "Left forearm flexors."),
    ("rectus", "Rectus abdominis", "front", "core", 0.500, 0.420, 0.030, 0.060, "The abdominal 'six-pack' muscle."),
    ("obliques-r", "Obliques (right)", "front", "core", 0.446, 0.430, 0.018, 0.050, "Right external oblique."),
    ("obliques-l", "Obliques (left)", "front", "core", 0.554, 0.430, 0.018, 0.050, "Left external oblique."),
    ("quadriceps-r", "Quadriceps (right)", "front", "hip-thigh", 0.452, 0.640, 0.028, 0.080, "Right quadriceps, front thigh."),
    ("quadriceps-l", "Quadriceps (left)", "front", "hip-thigh", 0.548, 0.640, 0.028, 0.080, "Left quadriceps, front thigh."),
    ("tibialis-r", "Tibialis (right shin)", "front", "lower-leg", 0.456, 0.820, 0.018, 0.070, "Right tibialis anterior, shin."),
    ("tibialis-l", "Tibialis (left shin)", "front", "lower-leg", 0.544, 0.820, 0.018, 0.070, "Left tibialis anterior, shin."),
    # ---- BACK ----
    ("trapezius-b", "Trapezius", "back", "chest-back", 0.500, 0.220, 0.060, 0.050, "Trapezius, upper back and neck."),
    ("deltoid-rb", "Deltoid (right, posterior)", "back", "shoulder-arm", 0.372, 0.244, 0.026, 0.030, "Right posterior deltoid."),
    ("deltoid-lb", "Deltoid (left, posterior)", "back", "shoulder-arm", 0.628, 0.244, 0.026, 0.030, "Left posterior deltoid."),
    ("triceps-r", "Triceps (right)", "back", "shoulder-arm", 0.352, 0.320, 0.020, 0.046, "Right triceps brachii."),
    ("triceps-l", "Triceps (left)", "back", "shoulder-arm", 0.648, 0.320, 0.020, 0.046, "Left triceps brachii."),
    ("lats-r", "Latissimus dorsi (right)", "back", "chest-back", 0.452, 0.340, 0.034, 0.050, "Right lat, the mid-back wing."),
    ("lats-l", "Latissimus dorsi (left)", "back", "chest-back", 0.548, 0.340, 0.034, 0.050, "Left lat, the mid-back wing."),
    ("erector", "Erector spinae", "back", "core", 0.500, 0.420, 0.022, 0.070, "The spinal erector muscles, low back."),
    ("glutes-r", "Gluteus (right)", "back", "hip-thigh", 0.456, 0.548, 0.030, 0.030, "Right gluteal muscles."),
    ("glutes-l", "Gluteus (left)", "back", "hip-thigh", 0.544, 0.548, 0.030, 0.030, "Left gluteal muscles."),
    ("hamstring-r", "Hamstrings (right)", "back", "hip-thigh", 0.452, 0.650, 0.028, 0.080, "Right hamstrings, back thigh."),
    ("hamstring-l", "Hamstrings (left)", "back", "hip-thigh", 0.548, 0.650, 0.028, 0.080, "Left hamstrings, back thigh."),
    ("calf-r", "Gastrocnemius (right calf)", "back", "lower-leg", 0.456, 0.820, 0.020, 0.066, "Right calf muscle."),
    ("calf-l", "Gastrocnemius (left calf)", "back", "lower-leg", 0.544, 0.820, 0.020, 0.066, "Left calf muscle."),
]


def _mk(slug, name, view, group, cx, cy, rx, ry, meaning):
    return {
        "id": f"muscle-{slug}", "side": view, "bilateral": False, "group": group,
        "geometry": {"type": "ellipse", "cx": round(cx, 4), "cy": round(cy, 4),
                     "rx": round(rx, 4), "ry": round(ry, 4)},
        "anatomy": name, "meaning_standard": meaning, "meaning_glen": "",
        "layers": {"embryological_depth": "mesoderm", "stress_affirmation": None, "touch_for_health": None},
    }


def main():
    zones = [_mk(*m) for m in MUSCLES]
    front = bbo.build_outline()
    data = {
        "system": "muscle", "reference_frame": "body_outline",
        "side_noun": "view", "group_noun": "region",
        "outline": front, "outlines": {"front": front, "back": front},
        "groups": GROUPS,
        "anchors": [
            {"key": "head-f", "view": "front", "template": {"x": 0.50, "y": 0.02}, "hint": "Tap the top of the head."},
            {"key": "feet-f", "view": "front", "template": {"x": 0.50, "y": 0.985}, "hint": "Tap the point between the ankles."},
            {"key": "head-b", "view": "back", "template": {"x": 0.50, "y": 0.02}, "hint": "Tap the top of the head."},
            {"key": "feet-b", "view": "back", "template": {"x": 0.50, "y": 0.985}, "hint": "Tap the point between the ankles."},
        ],
        "zones": zones,
    }
    out = ROOT / "data" / "bodymap-muscle.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    from collections import Counter
    print(f"wrote {out}: {len(zones)} zones", dict(Counter(z["side"] for z in zones)))


if __name__ == "__main__":
    main()
