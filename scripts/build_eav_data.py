#!/usr/bin/env python3
"""Build data/bodymap-eav.json — dorsal hand & foot acupuncture / EAV point chart.

The acupuncture terminal (jing-well) points and the EAV / Voll (Ryodoraku)
measurement points cluster at the finger- and toe-tips / nail corners, where
they are far too cramped on the whole-body meridian figure. This system gives
them their own zoomed **dorsal hand** and **dorsal foot** views (reusing the
existing hand/foot silhouettes) via the per-view-outline mechanism; the Side
control switches Hand <-> Foot.

Two point groups: the classical jing-well terminal acupoints, and the Voll EAV
vessel measurement points on the other nail corners. Positions are a
standards-based rendering (TCM jing-well + Voll CMPs) for Glen's EAV-clinical
refinement via the admin drawing tool, not gospel.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _outline(system):
    return json.loads((ROOT / "data" / f"bodymap-{system}.json").read_text())["outline"]


GROUPS = [
    {"id": "acupoint", "label": "Acupuncture terminal (jing-well)"},
    {"id": "eav", "label": "EAV / Voll vessel points"},
]

# (slug, anatomy, view, group, x, y, meaning)
POINTS = [
    # ===== HAND (dorsal, right) — finger tips point up, thumb to the left =====
    # classical jing-well terminal points at the nail corners
    ("LU11", "LU11 Shaoshang (Lung)", "hand", "acupoint", 0.150, 0.470, "Lung jing-well, radial thumb-nail corner."),
    ("LI1", "LI1 Shangyang (Large Intestine)", "hand", "acupoint", 0.365, 0.120, "Large-intestine jing-well, radial index-nail corner."),
    ("PC9", "PC9 Zhongchong (Pericardium)", "hand", "acupoint", 0.500, 0.058, "Pericardium jing-well, tip of the middle finger."),
    ("TE1", "TE1 Guanchong (Triple Energizer)", "hand", "acupoint", 0.635, 0.128, "Triple-energizer jing-well, ulnar ring-nail corner."),
    ("HT9", "HT9 Shaochong (Heart)", "hand", "acupoint", 0.665, 0.205, "Heart jing-well, radial little-finger nail corner."),
    ("SI1", "SI1 Shaoze (Small Intestine)", "hand", "acupoint", 0.715, 0.208, "Small-intestine jing-well, ulnar little-finger nail corner."),
    # Voll EAV vessel measurement points on the opposite nail corners
    ("HLy", "Lymph vessel (EAV)", "hand", "eav", 0.115, 0.548, "Voll lymph-vessel CMP, ulnar thumb corner."),
    ("HNd", "Nerve degeneration (EAV)", "hand", "eav", 0.415, 0.120, "Voll nerve-degeneration CMP, ulnar index corner."),
    ("HAl", "Allergy vessel (EAV)", "hand", "eav", 0.472, 0.078, "Voll allergy CMP, radial middle corner."),
    ("HOd", "Organ degeneration (EAV)", "hand", "eav", 0.528, 0.078, "Voll organ-degeneration CMP, ulnar middle corner."),
    ("HFd", "Fatty degeneration (EAV)", "hand", "eav", 0.585, 0.128, "Voll fatty-degeneration CMP, radial ring corner."),
    # ===== FOOT (dorsal, right) — toes point up, big toe to the left =====
    ("SP1", "SP1 Yinbai (Spleen)", "foot", "acupoint", 0.205, 0.070, "Spleen jing-well, medial big-toe nail corner."),
    ("LR1", "LR1 Dadun (Liver)", "foot", "acupoint", 0.292, 0.060, "Liver jing-well, lateral big-toe nail corner."),
    ("ST45", "ST45 Lidui (Stomach)", "foot", "acupoint", 0.490, 0.088, "Stomach jing-well, lateral 2nd-toe nail corner."),
    ("GB44", "GB44 Zuqiaoyin (Gallbladder)", "foot", "acupoint", 0.752, 0.150, "Gallbladder jing-well, lateral 4th-toe nail corner."),
    ("BL67", "BL67 Zhiyin (Bladder)", "foot", "acupoint", 0.852, 0.212, "Bladder jing-well, lateral little-toe nail corner."),
    # Voll EAV foot vessel points
    ("FLy", "Lymph vessel (EAV)", "foot", "eav", 0.232, 0.098, "Voll lymph-vessel CMP, big toe."),
    ("FPa", "Pancreas (EAV)", "foot", "eav", 0.258, 0.092, "Voll pancreas CMP, big toe."),
    ("FJd", "Joint degeneration (EAV)", "foot", "eav", 0.600, 0.100, "Voll joint-degeneration CMP, 3rd toe."),
    ("FKi", "Kidney (EAV)", "foot", "eav", 0.808, 0.220, "Voll kidney CMP, little toe."),
]


def build_point(slug, anatomy, view, group, x, y, meaning):
    return {
        "id": f"eav-{slug}",
        "side": view,
        "bilateral": False,
        "group": group,
        "geometry": {"type": "point", "x": x, "y": y},
        "anatomy": anatomy,
        "meaning_standard": meaning,
        "meaning_glen": "",
        "layers": {"embryological_depth": None, "stress_affirmation": None, "touch_for_health": None},
    }


def main():
    data = {
        "system": "eav",
        "reference_frame": "extremity_outline",
        "side_noun": "view",
        "group_noun": "point type",
        "outlines": {"hand": _outline("hand"), "foot": _outline("foot")},
        "groups": GROUPS,
        "anchors": [],
        "zones": [build_point(*p) for p in POINTS],
    }
    out = ROOT / "data" / "bodymap-eav.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    from collections import Counter
    print(f"wrote {out}: {len(data['zones'])} points",
          dict(Counter(z["side"] for z in data["zones"])))


if __name__ == "__main__":
    main()
