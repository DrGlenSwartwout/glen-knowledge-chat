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

# Full Voll digit-terminal / control-measurement-point (CMP) set: both nail
# corners of every finger and toe carry a vessel, plus the degeneration vessels
# on the finger/toe dorsum. Classical meridian jing-wells (group "acupoint")
# double as the Voll CMP for their meridian; the extra Voll vessels are "eav".
# (slug, anatomy, view, group, x, y, meaning)
POINTS = [
    # ===== HAND (dorsal, right) — finger tips point up, thumb to the left =====
    # -- classical meridian jing-well terminals (= their Voll CMP) --
    ("LU11", "LU11 Shaoshang (Lung)", "hand", "acupoint", 0.150, 0.470, "Lung jing-well / Voll lung CMP, radial thumb corner."),
    ("LI1", "LI1 Shangyang (Large Intestine)", "hand", "acupoint", 0.360, 0.125, "Large-intestine jing-well / Voll CMP, radial index corner."),
    ("PC9", "PC9 Zhongchong (Pericardium)", "hand", "acupoint", 0.500, 0.055, "Pericardium jing-well, tip of the middle finger."),
    ("TE1", "TE1 Guanchong (Triple Energizer)", "hand", "acupoint", 0.640, 0.130, "Triple-energizer jing-well / Voll endocrine CMP, ulnar ring corner."),
    ("HT9", "HT9 Shaochong (Heart)", "hand", "acupoint", 0.665, 0.205, "Heart jing-well / Voll CMP, radial little-finger corner."),
    ("SI1", "SI1 Shaoze (Small Intestine)", "hand", "acupoint", 0.718, 0.210, "Small-intestine jing-well / Voll CMP, ulnar little-finger corner."),
    # -- Voll extra vessels on the free nail corners --
    ("HLy", "Lymph vessel (EAV)", "hand", "eav", 0.108, 0.550, "Voll lymph-vessel CMP, ulnar thumb corner."),
    ("HNd", "Nerve degeneration (EAV)", "hand", "eav", 0.418, 0.125, "Voll nerve-degeneration CMP, ulnar index corner."),
    ("HCi", "Circulation vessel (EAV)", "hand", "eav", 0.470, 0.078, "Voll circulation CMP, radial middle corner."),
    ("HOd", "Organ degeneration (EAV)", "hand", "eav", 0.530, 0.078, "Voll organ/parenchymal-degeneration CMP, ulnar middle corner."),
    ("HAl", "Allergy / immune vessel (EAV)", "hand", "eav", 0.582, 0.130, "Voll allergy CMP, radial ring corner."),
    # -- Voll degeneration vessels on the finger dorsum --
    ("HSk", "Skin degeneration (EAV)", "hand", "eav", 0.398, 0.188, "Voll skin-degeneration vessel, index dorsum."),
    ("HFd", "Fatty degeneration (EAV)", "hand", "eav", 0.500, 0.150, "Voll fatty-degeneration vessel, middle dorsum."),
    ("HJd", "Joint degeneration (EAV)", "hand", "eav", 0.618, 0.192, "Voll joint-degeneration vessel, ring dorsum."),
    # ===== FOOT (dorsal, right) — toes point up, big toe to the left =====
    # -- classical meridian jing-well terminals --
    ("SP1", "SP1 Yinbai (Spleen)", "foot", "acupoint", 0.200, 0.072, "Spleen jing-well, medial big-toe corner."),
    ("LR1", "LR1 Dadun (Liver)", "foot", "acupoint", 0.295, 0.060, "Liver jing-well, lateral big-toe corner."),
    ("ST45", "ST45 Lidui (Stomach)", "foot", "acupoint", 0.492, 0.088, "Stomach jing-well, lateral 2nd-toe corner."),
    ("GB44", "GB44 Zuqiaoyin (Gallbladder)", "foot", "acupoint", 0.755, 0.150, "Gallbladder jing-well, lateral 4th-toe corner."),
    ("BL67", "BL67 Zhiyin (Bladder)", "foot", "acupoint", 0.855, 0.213, "Bladder jing-well, lateral little-toe corner."),
    # -- Voll extra vessels on the toes --
    ("FLy", "Lymph vessel (EAV)", "foot", "eav", 0.232, 0.100, "Voll lymph-vessel CMP, big toe (medial dorsum)."),
    ("FPa", "Pancreas (EAV)", "foot", "eav", 0.258, 0.088, "Voll pancreas CMP, big toe."),
    ("FSt2", "Stomach branch (EAV)", "foot", "eav", 0.435, 0.100, "Voll stomach-branch CMP, medial 2nd-toe corner."),
    ("FSk", "Skin degeneration (EAV)", "foot", "eav", 0.575, 0.112, "Voll skin-degeneration vessel, medial 3rd-toe corner."),
    ("FJd", "Joint degeneration (EAV)", "foot", "eav", 0.620, 0.098, "Voll joint-degeneration vessel, lateral 3rd-toe corner."),
    ("FAr", "Articular degeneration (EAV)", "foot", "eav", 0.700, 0.166, "Voll articular-degeneration vessel, medial 4th-toe corner."),
    ("FKi", "Kidney (EAV)", "foot", "eav", 0.812, 0.222, "Voll kidney CMP, medial little-toe corner."),
    ("FFd", "Fatty degeneration (EAV)", "foot", "eav", 0.468, 0.138, "Voll fatty-degeneration vessel, 2nd-toe dorsum."),
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
