#!/usr/bin/env python3
"""Build data/bodymap-ear.json — a full auricular (ear) acupuncture chart.

Expands the ear from a 10-point left-only stub to the standard auricular point
set on BOTH ears. The ear is an inverted-foetus homunculus: head at the lobe
(bottom), viscera in the concha (centre), spine along the antihelix ridge, limbs
on the crura and scapha, pelvic organs in the triangular fossa.

Outline is upgraded from a featureless blob to an anatomical ear: the outer helix
rim + lobe, plus internal subpaths for the antihelix Y-ridge, the crus of helix,
the concha bowl, the tragus and the antitragus — so the regions the points sit in
are actually visible (multi-subpath stroke, same idea as the foot's sole+toes).

Points are `bilateral:true` on the canonical LEFT ear; the renderer mirrors them
(and the outline via outline_side="left") for the right ear, since auricular
points are the same on both ears.

Positions are a standards-based rendering (Chinese/WHO auricular map) for Glen's
clinical refinement via the admin drawing tool, not gospel.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- anatomical ear outline: outer rim + internal landmark ridges ----
OUTLINE = " ".join([
    # outer helix rim + lobe
    "M 0.55 0.05 C 0.75 0.06 0.84 0.24 0.80 0.42 C 0.78 0.56 0.70 0.64 0.64 0.72",
    "C 0.58 0.82 0.57 0.93 0.47 0.95 C 0.37 0.97 0.28 0.90 0.30 0.79",
    "C 0.23 0.76 0.18 0.67 0.24 0.55 C 0.18 0.44 0.21 0.26 0.32 0.15",
    "C 0.40 0.07 0.47 0.05 0.55 0.05 Z",
    # inner fold of the helix rim
    "M 0.53 0.11 C 0.67 0.13 0.74 0.28 0.71 0.42 C 0.70 0.52 0.64 0.59 0.59 0.66",
    # antihelix ridge body
    "M 0.40 0.70 C 0.41 0.60 0.42 0.50 0.43 0.41",
    # superior antihelix crus
    "M 0.43 0.41 C 0.43 0.31 0.45 0.23 0.47 0.16",
    # inferior antihelix crus
    "M 0.43 0.41 C 0.39 0.35 0.35 0.31 0.31 0.29",
    # crus of helix (crosses into the concha)
    "M 0.35 0.45 C 0.43 0.46 0.49 0.47 0.55 0.47",
    # concha bowl
    "M 0.46 0.49 C 0.56 0.49 0.61 0.56 0.57 0.64 C 0.53 0.71 0.44 0.69 0.42 0.61 C 0.40 0.54 0.42 0.49 0.46 0.49 Z",
    # tragus
    "M 0.30 0.51 C 0.25 0.53 0.24 0.60 0.27 0.65 C 0.30 0.67 0.34 0.64 0.34 0.58",
    # antitragus
    "M 0.39 0.72 C 0.43 0.71 0.48 0.73 0.48 0.77 C 0.47 0.80 0.42 0.80 0.40 0.77",
])

GROUPS = [
    {"id": "helix", "label": "Helix (outer rim)"},
    {"id": "scapha", "label": "Scapha (upper limb)"},
    {"id": "triangular-fossa", "label": "Triangular fossa (pelvis / shen men)"},
    {"id": "antihelix", "label": "Antihelix (spine & trunk)"},
    {"id": "crus-helix", "label": "Crus of helix (digestive line)"},
    {"id": "concha", "label": "Concha (internal organs)"},
    {"id": "tragus", "label": "Tragus (nose / adrenal / throat)"},
    {"id": "antitragus", "label": "Antitragus (head / brain)"},
    {"id": "lobe", "label": "Lobe (head & face)"},
]

ANCHORS = [
    {"key": "helix-top", "template": {"x": 0.55, "y": 0.07}, "hint": "Tap the top rim of your ear."},
    {"key": "tragus", "template": {"x": 0.29, "y": 0.58}, "hint": "Tap the tragus (flap in front of the canal)."},
    {"key": "lobe-bottom", "template": {"x": 0.47, "y": 0.93}, "hint": "Tap the bottom of your earlobe."},
]

# (slug, anatomy, group, x, y, meaning)
POINTS = [
    # ---- helix (outer rim): apex, tubercle, genitals, elimination, diaphragm ----
    ("ear-apex", "Ear apex", "helix", 0.52, 0.09, "Ear apex — fever, allergy, calming."),
    ("helix-tubercle", "Helix tubercle (Darwin)", "helix", 0.77, 0.23, "Darwin's tubercle, upper helix."),
    ("liver-yang", "Liver yang", "helix", 0.78, 0.34, "Liver-yang point, helix."),
    ("external-genital", "External genitals", "helix", 0.30, 0.20, "External genital reflex, helix."),
    ("urethra", "Urethra", "helix", 0.28, 0.26, "Urethra reflex, helix."),
    ("anus", "Anus / rectum", "helix", 0.27, 0.32, "Anus / lower rectum, helix."),
    ("diaphragm-helix", "Diaphragm (helix root)", "crus-helix", 0.34, 0.45, "Diaphragm / ear-centre, helix root."),
    # ---- scapha (upper limb) ----
    ("clavicle", "Clavicle", "scapha", 0.64, 0.66, "Clavicle, lower scapha."),
    ("shoulder", "Shoulder", "scapha", 0.67, 0.58, "Shoulder, scapha."),
    ("shoulder-joint", "Shoulder joint", "scapha", 0.70, 0.50, "Shoulder joint, scapha."),
    ("elbow", "Elbow", "scapha", 0.72, 0.42, "Elbow, scapha."),
    ("wrist", "Wrist", "scapha", 0.73, 0.33, "Wrist, scapha."),
    ("fingers", "Fingers", "scapha", 0.71, 0.25, "Fingers, upper scapha."),
    ("wind-stream", "Wind stream (allergy)", "scapha", 0.68, 0.29, "Wind-stream / allergy point, scapha."),
    # ---- triangular fossa (pelvis, shen men) ----
    ("shenmen", "Shen men", "triangular-fossa", 0.45, 0.24, "Shen Men — master calming / pain point."),
    ("uterus-genital", "Uterus / internal genital", "triangular-fossa", 0.41, 0.21, "Uterus / internal genital, triangular fossa."),
    ("pelvis", "Pelvis", "triangular-fossa", 0.47, 0.30, "Pelvis, triangular fossa."),
    ("hypertension", "Hypertension groove", "triangular-fossa", 0.43, 0.30, "Blood-pressure-lowering point."),
    # ---- superior antihelix crus (lower limb) ----
    ("toe", "Toe", "antihelix", 0.44, 0.15, "Toe, superior crus."),
    ("heel", "Heel", "antihelix", 0.49, 0.15, "Heel, superior crus."),
    ("ankle", "Ankle", "antihelix", 0.46, 0.20, "Ankle, superior crus."),
    ("knee", "Knee", "antihelix", 0.42, 0.26, "Knee, superior crus."),
    ("hip", "Hip", "antihelix", 0.38, 0.31, "Hip, superior crus."),
    # ---- inferior antihelix crus (buttock, sciatic, sympathetic) ----
    ("buttock", "Buttock", "antihelix", 0.34, 0.33, "Buttock, inferior crus."),
    ("sciatic", "Sciatic nerve", "antihelix", 0.32, 0.30, "Sciatic nerve, inferior crus."),
    ("sympathetic", "Sympathetic", "antihelix", 0.30, 0.35, "Autonomic / sympathetic point."),
    # ---- antihelix body (spine + trunk) ----
    ("cervical-vertebrae", "Cervical vertebrae", "antihelix", 0.39, 0.68, "Cervical spine, lower antihelix."),
    ("thoracic-vertebrae", "Thoracic vertebrae", "antihelix", 0.41, 0.58, "Thoracic spine, mid antihelix."),
    ("lumbosacral-vertebrae", "Lumbosacral vertebrae", "antihelix", 0.43, 0.47, "Lumbosacral spine, upper antihelix."),
    ("neck", "Neck", "antihelix", 0.36, 0.69, "Neck, antihelix."),
    ("chest", "Chest", "antihelix", 0.38, 0.60, "Chest, antihelix."),
    ("abdomen", "Abdomen", "antihelix", 0.40, 0.50, "Abdomen, antihelix."),
    ("mammary", "Breast / mammary", "antihelix", 0.36, 0.62, "Breast / mammary, antihelix."),
    # ---- crus of helix (digestive line) ----
    ("mouth", "Mouth", "crus-helix", 0.38, 0.47, "Mouth, crus of helix."),
    ("esophagus", "Oesophagus", "crus-helix", 0.42, 0.47, "Oesophagus, crus of helix."),
    ("cardia", "Cardia", "crus-helix", 0.45, 0.47, "Cardia, crus of helix."),
    ("duodenum", "Duodenum", "crus-helix", 0.52, 0.46, "Duodenum, end of crus of helix."),
    # ---- concha: cymba (upper, urinary/hepatobiliary/bowel) ----
    ("point-zero", "Point zero", "concha", 0.50, 0.50, "Point Zero — the reference / balance point."),
    ("stomach", "Stomach", "concha", 0.47, 0.50, "Stomach, around point zero."),
    ("small-intestine", "Small intestine", "concha", 0.50, 0.46, "Small intestine, cymba concha."),
    ("large-intestine", "Large intestine", "concha", 0.53, 0.45, "Large intestine, cymba concha."),
    ("appendix", "Appendix", "concha", 0.55, 0.46, "Appendix, cymba concha."),
    ("kidney", "Kidney", "concha", 0.56, 0.43, "Kidney, cymba concha."),
    ("ureter", "Ureter", "concha", 0.58, 0.45, "Ureter, cymba concha."),
    ("bladder", "Bladder", "concha", 0.59, 0.41, "Bladder, cymba concha."),
    ("pancreas-gallbladder", "Pancreas / gallbladder", "concha", 0.60, 0.47, "Pancreas / gallbladder, cymba concha."),
    ("liver", "Liver", "concha", 0.61, 0.53, "Liver, cymba concha."),
    ("prostate", "Prostate", "concha", 0.61, 0.40, "Prostate, cymba concha."),
    # ---- concha: cavum (lower, thoracic viscera) ----
    ("heart", "Heart", "concha", 0.49, 0.55, "Heart, centre of the cavum concha."),
    ("lung", "Lung", "concha", 0.52, 0.60, "Lung, surrounding the heart."),
    ("trachea", "Trachea / bronchi", "concha", 0.45, 0.58, "Trachea / bronchi, cavum concha."),
    ("spleen", "Spleen", "concha", 0.57, 0.61, "Spleen, cavum concha (left ear emphasis)."),
    ("san-jiao", "San jiao (triple burner)", "concha", 0.47, 0.63, "San jiao / triple burner, cavum concha."),
    ("endocrine", "Endocrine", "concha", 0.43, 0.63, "Endocrine point, concha notch."),
    # ---- tragus (nose / adrenal / throat) ----
    ("external-nose", "External nose", "tragus", 0.31, 0.56, "External nose, tragus."),
    ("adrenal", "Adrenal", "tragus", 0.28, 0.61, "Adrenal, tip of the tragus."),
    ("internal-nose", "Internal nose", "tragus", 0.30, 0.63, "Internal nose, inner tragus."),
    ("throat-pharynx", "Throat / pharynx", "tragus", 0.32, 0.53, "Throat / pharynx, tragus."),
    ("external-ear", "External ear", "tragus", 0.33, 0.50, "External ear, upper tragus."),
    # ---- antitragus (head / brain) ----
    ("subcortex", "Subcortex", "antitragus", 0.43, 0.76, "Subcortex — pain, inflammation, brain."),
    ("brainstem", "Brainstem", "antitragus", 0.40, 0.73, "Brainstem, antitragus."),
    ("forehead", "Forehead", "antitragus", 0.46, 0.74, "Forehead, antitragus."),
    ("temple", "Temple (taiyang)", "antitragus", 0.48, 0.72, "Temple / taiyang, antitragus."),
    ("occiput", "Occiput", "antitragus", 0.44, 0.72, "Occiput, antitragus."),
    ("asthma", "Asthma (ping chuan)", "antitragus", 0.42, 0.78, "Asthma / ping-chuan, antitragus apex."),
    # ---- lobe (head & face) ----
    ("eye", "Eye", "lobe", 0.48, 0.86, "Eye, centre of the lobe."),
    ("tongue", "Tongue", "lobe", 0.44, 0.84, "Tongue, lobe."),
    ("teeth-jaw", "Teeth / jaw", "lobe", 0.41, 0.85, "Teeth / jaw, lobe."),
    ("tonsil", "Tonsil", "lobe", 0.49, 0.92, "Tonsil, lower lobe."),
    ("internal-ear", "Internal ear", "lobe", 0.53, 0.85, "Internal ear, lobe."),
    ("cheek", "Cheek", "lobe", 0.51, 0.83, "Cheek, lobe."),
    ("face", "Face / upper jaw", "lobe", 0.45, 0.81, "Face / upper jaw, upper lobe."),
]


def build_point(slug, anatomy, group, x, y, meaning):
    return {
        "id": f"ear-{slug}",
        "side": "left",
        "bilateral": True,
        "group": group,
        "geometry": {"type": "point", "x": x, "y": y},
        "anatomy": anatomy,
        "meaning_standard": meaning,
        "meaning_glen": "",
        "layers": {"embryological_depth": None, "stress_affirmation": None, "touch_for_health": None},
    }


def main():
    data = {
        "system": "ear",
        "reference_frame": "ear_outline",
        "side_noun": "ear",
        "group_noun": "region",
        "outline": OUTLINE,
        "outline_side": "left",
        "groups": GROUPS,
        "anchors": ANCHORS,
        "zones": [build_point(*p) for p in POINTS],
    }
    out = ROOT / "data" / "bodymap-ear.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"wrote {out}: {len(data['zones'])} points (bilateral -> both ears)")


if __name__ == "__main__":
    main()
