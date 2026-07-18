#!/usr/bin/env python3
"""Build data/bodymap-skeleton.json — a whole-body skeletal atlas.

Major bones and joints as labelled `ellipse` zones on the body silhouette
(front + back), each named for its bone/joint so a client's Bone finding lights
the skeleton (whole-system theme in the personalization endpoint) and a specific
structure (Femur, Hip Joint, Knees) lights that bone/joint. Front + back reuse
the meridian silhouette + per-view outlines.

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
    {"id": "skull", "label": "Skull & jaw"},
    {"id": "spine", "label": "Spine"},
    {"id": "thorax", "label": "Ribs, sternum & shoulder girdle"},
    {"id": "upper-limb", "label": "Arm & hand"},
    {"id": "pelvis", "label": "Pelvis"},
    {"id": "lower-limb", "label": "Leg & foot"},
    {"id": "joints", "label": "Joints"},
]

# (slug, name, view, group, cx, cy, rx, ry, meaning)
BONES = [
    # ---- FRONT ----
    ("skull", "Skull", "front", "skull", 0.500, 0.055, 0.050, 0.060, "The cranial bones."),
    ("mandible", "Mandible (jaw bone)", "front", "skull", 0.500, 0.116, 0.032, 0.020, "The lower jaw bone."),
    ("cervical", "Cervical vertebrae", "front", "spine", 0.500, 0.160, 0.014, 0.026, "Cervical spine (C1-C7)."),
    ("clavicle-r", "Clavicle (right)", "front", "thorax", 0.442, 0.186, 0.040, 0.009, "Right collarbone."),
    ("clavicle-l", "Clavicle (left)", "front", "thorax", 0.558, 0.186, 0.040, 0.009, "Left collarbone."),
    ("sternum", "Sternum", "front", "thorax", 0.500, 0.252, 0.016, 0.052, "The breastbone."),
    ("ribs-r", "Ribs (right)", "front", "thorax", 0.442, 0.278, 0.052, 0.062, "Right rib cage."),
    ("ribs-l", "Ribs (left)", "front", "thorax", 0.558, 0.278, 0.052, 0.062, "Left rib cage."),
    ("humerus-r", "Humerus (right)", "front", "upper-limb", 0.352, 0.300, 0.020, 0.072, "Right upper-arm bone."),
    ("humerus-l", "Humerus (left)", "front", "upper-limb", 0.648, 0.300, 0.020, 0.072, "Left upper-arm bone."),
    ("forearm-r", "Radius & ulna (right)", "front", "upper-limb", 0.322, 0.420, 0.018, 0.064, "Right forearm bones."),
    ("forearm-l", "Radius & ulna (left)", "front", "upper-limb", 0.678, 0.420, 0.018, 0.064, "Left forearm bones."),
    ("hand-r", "Hand bones (right)", "front", "upper-limb", 0.302, 0.508, 0.022, 0.030, "Right carpals, metacarpals, phalanges."),
    ("hand-l", "Hand bones (left)", "front", "upper-limb", 0.698, 0.508, 0.022, 0.030, "Left carpals, metacarpals, phalanges."),
    ("pelvis", "Pelvis", "front", "pelvis", 0.500, 0.532, 0.062, 0.032, "The pelvic bones (ilium, ischium, pubis)."),
    ("femur-r", "Femur (right)", "front", "lower-limb", 0.452, 0.632, 0.020, 0.090, "Right thigh bone."),
    ("femur-l", "Femur (left)", "front", "lower-limb", 0.548, 0.632, 0.020, 0.090, "Left thigh bone."),
    ("patella-r", "Patella (right)", "front", "lower-limb", 0.458, 0.726, 0.012, 0.012, "Right kneecap."),
    ("patella-l", "Patella (left)", "front", "lower-limb", 0.542, 0.726, 0.012, 0.012, "Left kneecap."),
    ("shin-r", "Tibia & fibula (right)", "front", "lower-limb", 0.456, 0.822, 0.018, 0.082, "Right shin bones."),
    ("shin-l", "Tibia & fibula (left)", "front", "lower-limb", 0.544, 0.822, 0.018, 0.082, "Left shin bones."),
    ("foot-r", "Foot bones (right)", "front", "lower-limb", 0.462, 0.952, 0.030, 0.020, "Right tarsals, metatarsals, phalanges."),
    ("foot-l", "Foot bones (left)", "front", "lower-limb", 0.538, 0.952, 0.030, 0.020, "Left tarsals, metatarsals, phalanges."),
    # front joints
    ("shoulder-r", "Shoulder joint (right)", "front", "joints", 0.372, 0.238, 0.017, 0.017, "Right shoulder (glenohumeral) joint."),
    ("shoulder-l", "Shoulder joint (left)", "front", "joints", 0.628, 0.238, 0.017, 0.017, "Left shoulder (glenohumeral) joint."),
    ("elbow-r", "Elbow joint (right)", "front", "joints", 0.334, 0.382, 0.015, 0.015, "Right elbow joint."),
    ("elbow-l", "Elbow joint (left)", "front", "joints", 0.666, 0.382, 0.015, 0.015, "Left elbow joint."),
    ("wrist-r", "Wrist joint (right)", "front", "joints", 0.312, 0.486, 0.013, 0.013, "Right wrist joint."),
    ("wrist-l", "Wrist joint (left)", "front", "joints", 0.688, 0.486, 0.013, 0.013, "Left wrist joint."),
    ("hip-r", "Hip joint (right)", "front", "joints", 0.446, 0.560, 0.016, 0.016, "Right hip joint."),
    ("hip-l", "Hip joint (left)", "front", "joints", 0.554, 0.560, 0.016, 0.016, "Left hip joint."),
    ("knee-r", "Knee joint (right)", "front", "joints", 0.456, 0.740, 0.015, 0.015, "Right knee joint."),
    ("knee-l", "Knee joint (left)", "front", "joints", 0.544, 0.740, 0.015, 0.015, "Left knee joint."),
    ("ankle-r", "Ankle joint (right)", "front", "joints", 0.460, 0.912, 0.014, 0.014, "Right ankle joint."),
    ("ankle-l", "Ankle joint (left)", "front", "joints", 0.540, 0.912, 0.014, 0.014, "Left ankle joint."),
    # ---- BACK ----
    ("occiput", "Occiput (skull)", "back", "skull", 0.500, 0.058, 0.048, 0.052, "Back of the skull."),
    ("thoracic-b", "Thoracic vertebrae", "back", "spine", 0.500, 0.300, 0.016, 0.090, "Thoracic spine (T1-T12)."),
    ("lumbar-b", "Lumbar vertebrae", "back", "spine", 0.500, 0.430, 0.018, 0.045, "Lumbar spine (L1-L5)."),
    ("sacrum-b", "Sacrum & coccyx", "back", "spine", 0.500, 0.492, 0.020, 0.028, "Sacrum and coccyx (tailbone)."),
    ("scapula-r", "Scapula (right)", "back", "thorax", 0.440, 0.262, 0.030, 0.036, "Right shoulder blade."),
    ("scapula-l", "Scapula (left)", "back", "thorax", 0.560, 0.262, 0.030, 0.036, "Left shoulder blade."),
    ("ilium-r", "Ilium (right)", "back", "pelvis", 0.456, 0.520, 0.026, 0.028, "Right iliac crest of the pelvis."),
    ("ilium-l", "Ilium (left)", "back", "pelvis", 0.544, 0.520, 0.026, 0.028, "Left iliac crest of the pelvis."),
    ("femur-rb", "Femur (right, posterior)", "back", "lower-limb", 0.452, 0.632, 0.020, 0.090, "Right thigh bone, posterior."),
    ("femur-lb", "Femur (left, posterior)", "back", "lower-limb", 0.548, 0.632, 0.020, 0.090, "Left thigh bone, posterior."),
]


def _mk(slug, name, view, group, cx, cy, rx, ry, meaning):
    return {
        "id": f"bone-{slug}", "side": view, "bilateral": False, "group": group,
        "geometry": {"type": "ellipse", "cx": round(cx, 4), "cy": round(cy, 4),
                     "rx": round(rx, 4), "ry": round(ry, 4)},
        "anatomy": name, "meaning_standard": meaning, "meaning_glen": "",
        "layers": {"embryological_depth": "mesoderm", "stress_affirmation": None, "touch_for_health": None},
    }


def main():
    zones = [_mk(*b) for b in BONES]
    front = bbo.build_outline()
    data = {
        "system": "skeleton", "reference_frame": "body_outline",
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
    out = ROOT / "data" / "bodymap-skeleton.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    from collections import Counter
    print(f"wrote {out}: {len(zones)} zones", dict(Counter(z["side"] for z in zones)))


if __name__ == "__main__":
    main()
