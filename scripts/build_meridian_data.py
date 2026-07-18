#!/usr/bin/env python3
"""Build data/bodymap-meridian.json — acupuncture meridians on a body figure.

New whole-body system: reference_frame "body_outline" over a front-facing human
silhouette (shared front/back — see build_body_outline). Each channel is a
`path` geometry (a stroked line) coloured by meridian, with its acupoints as
`point` zones along it. The Side control is Front/Back (zone side = "front"/
"back"). Bilateral channels are emitted on BOTH body halves (right copy as
authored + left copy mirrored x -> 1-x); the midline vessels (Ren/Du) are single.

STARTER SET (5 of 14 channels) to validate the architecture end to end before
mass-authoring the rest: Conception Vessel (front midline), Governing Vessel
(back midline), Stomach (front, bilateral), Bladder (back, bilateral), Lung
(front arm, bilateral). Positions are a standards-based rendering for Glen's
clinical refinement, not gospel.
"""
import json
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("bbo", ROOT / "scripts" / "build_body_outline.py")
bbo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bbo)


def catmull_rom_open(pts):
    """Open Catmull-Rom spline through pts -> smooth path 'd'."""
    if len(pts) < 2:
        return ""
    p = [pts[0]] + list(pts) + [pts[-1]]
    d = f"M{pts[0][0]:.4f} {pts[0][1]:.4f} "
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        c1x = p1[0] + (p2[0] - p0[0]) / 6.0
        c1y = p1[1] + (p2[1] - p0[1]) / 6.0
        c2x = p2[0] - (p3[0] - p1[0]) / 6.0
        c2y = p2[1] - (p3[1] - p1[1]) / 6.0
        d += f"C{c1x:.4f} {c1y:.4f} {c2x:.4f} {c2y:.4f} {p2[0]:.4f} {p2[1]:.4f} "
    return d.strip()


# key, name, side/view, bilateral, waypoints (right copy / midline), [(slug, name, x, y), ...]
CHANNELS = [
    # ===== FRONT view =====
    ("CV", "Conception Vessel (Ren)", "front", False,
     [(0.50, 0.565), (0.50, 0.47), (0.50, 0.42), (0.50, 0.34), (0.50, 0.27), (0.50, 0.19), (0.50, 0.13)],
     [("CV1", "CV1 Huiyin", 0.50, 0.565), ("CV4", "CV4 Guanyuan", 0.50, 0.48),
      ("CV6", "CV6 Qihai", 0.50, 0.455), ("CV8", "CV8 Shenque (navel)", 0.50, 0.42),
      ("CV12", "CV12 Zhongwan", 0.50, 0.37), ("CV17", "CV17 Shanzhong", 0.50, 0.27),
      ("CV22", "CV22 Tiantu", 0.50, 0.175), ("CV24", "CV24 Chengjiang", 0.50, 0.13)]),
    ("ST", "Stomach", "front", True,
     [(0.56, 0.11), (0.57, 0.135), (0.555, 0.165), (0.575, 0.235), (0.585, 0.285),
      (0.575, 0.42), (0.60, 0.53), (0.585, 0.66), (0.575, 0.78), (0.57, 0.885), (0.60, 0.985)],
     [("ST1", "ST1 Chengqi", 0.56, 0.11), ("ST9", "ST9 Renying", 0.555, 0.165),
      ("ST12", "ST12 Quepen", 0.585, 0.20), ("ST25", "ST25 Tianshu", 0.575, 0.42),
      ("ST36", "ST36 Zusanli", 0.585, 0.72), ("ST40", "ST40 Fenglong", 0.575, 0.83),
      ("ST44", "ST44 Neiting", 0.595, 0.975)]),
    ("SP", "Spleen", "front", True,
     [(0.535, 0.985), (0.548, 0.955), (0.552, 0.90), (0.558, 0.82), (0.562, 0.77),
      (0.575, 0.66), (0.60, 0.54), (0.622, 0.42), (0.645, 0.30), (0.66, 0.255)],
     [("SP3", "SP3 Taibai", 0.548, 0.955), ("SP6", "SP6 Sanyinjiao", 0.554, 0.895),
      ("SP9", "SP9 Yinlingquan", 0.562, 0.775), ("SP10", "SP10 Xuehai", 0.578, 0.72),
      ("SP21", "SP21 Dabao", 0.66, 0.255)]),
    ("KI", "Kidney", "front", True,
     [(0.548, 0.99), (0.552, 0.945), (0.556, 0.88), (0.562, 0.79), (0.565, 0.66),
      (0.55, 0.52), (0.535, 0.42), (0.54, 0.30), (0.545, 0.245)],
     [("KI1", "KI1 Yongquan", 0.55, 0.985), ("KI3", "KI3 Taixi", 0.554, 0.94),
      ("KI7", "KI7 Fuliu", 0.558, 0.90), ("KI27", "KI27 Shufu", 0.545, 0.25)]),
    ("LU", "Lung", "front", True,
     [(0.63, 0.245), (0.675, 0.27), (0.71, 0.36), (0.725, 0.44), (0.755, 0.52),
      (0.783, 0.585), (0.795, 0.625)],
     [("LU1", "LU1 Zhongfu", 0.63, 0.245), ("LU5", "LU5 Chize", 0.72, 0.44),
      ("LU7", "LU7 Lieque", 0.775, 0.57), ("LU9", "LU9 Taiyuan", 0.785, 0.59),
      ("LU11", "LU11 Shaoshang", 0.80, 0.628)]),
    ("PC", "Pericardium", "front", True,
     [(0.635, 0.27), (0.68, 0.36), (0.71, 0.44), (0.745, 0.53), (0.775, 0.585), (0.78, 0.628)],
     [("PC3", "PC3 Quze", 0.71, 0.44), ("PC6", "PC6 Neiguan", 0.77, 0.57),
      ("PC8", "PC8 Laogong", 0.778, 0.615)]),
    ("HT", "Heart", "front", True,
     [(0.655, 0.26), (0.685, 0.35), (0.70, 0.44), (0.735, 0.53), (0.755, 0.585), (0.745, 0.63)],
     [("HT3", "HT3 Shaohai", 0.70, 0.44), ("HT7", "HT7 Shenmen", 0.755, 0.585),
      ("HT9", "HT9 Shaochong", 0.745, 0.63)]),
    ("LI", "Large Intestine", "front", True,
     [(0.74, 0.635), (0.755, 0.60), (0.735, 0.50), (0.715, 0.44), (0.695, 0.34),
      (0.68, 0.26), (0.56, 0.155), (0.525, 0.10)],
     [("LI4", "LI4 Hegu", 0.748, 0.605), ("LI11", "LI11 Quchi", 0.715, 0.44),
      ("LI15", "LI15 Jianyu", 0.685, 0.26), ("LI20", "LI20 Yingxiang", 0.525, 0.10)]),
    ("LR", "Liver", "front", True,
     [(0.53, 0.985), (0.548, 0.955), (0.556, 0.87), (0.56, 0.78), (0.575, 0.66),
      (0.585, 0.54), (0.60, 0.37)],
     [("LR1", "LR1 Dadun", 0.53, 0.985), ("LR3", "LR3 Taichong", 0.548, 0.95),
      ("LR8", "LR8 Ququan", 0.56, 0.78), ("LR14", "LR14 Qimen", 0.60, 0.37)]),
    # ===== BACK view =====
    ("GV", "Governing Vessel (Du)", "back", False,
     [(0.50, 0.565), (0.50, 0.50), (0.50, 0.45), (0.50, 0.35), (0.50, 0.25), (0.50, 0.17),
      (0.50, 0.09), (0.50, 0.05), (0.50, 0.11)],
     [("GV1", "GV1 Changqiang", 0.50, 0.56), ("GV4", "GV4 Mingmen", 0.50, 0.45),
      ("GV14", "GV14 Dazhui", 0.50, 0.19), ("GV20", "GV20 Baihui (vertex)", 0.50, 0.045),
      ("GV26", "GV26 Renzhong", 0.50, 0.105)]),
    ("BL", "Bladder", "back", True,
     [(0.53, 0.095), (0.55, 0.05), (0.545, 0.17), (0.55, 0.27), (0.55, 0.40),
      (0.545, 0.53), (0.565, 0.60), (0.575, 0.70), (0.57, 0.80), (0.565, 0.90), (0.60, 0.985)],
     [("BL1", "BL1 Jingming", 0.53, 0.095), ("BL10", "BL10 Tianzhu", 0.545, 0.17),
      ("BL13", "BL13 Feishu", 0.55, 0.27), ("BL23", "BL23 Shenshu", 0.55, 0.47),
      ("BL40", "BL40 Weizhong", 0.57, 0.78), ("BL60", "BL60 Kunlun", 0.565, 0.945),
      ("BL67", "BL67 Zhiyin", 0.595, 0.98)]),
    ("SI", "Small Intestine", "back", True,
     [(0.745, 0.635), (0.758, 0.60), (0.74, 0.52), (0.72, 0.44), (0.70, 0.35),
      (0.655, 0.27), (0.575, 0.185), (0.55, 0.12)],
     [("SI3", "SI3 Houxi", 0.752, 0.61), ("SI8", "SI8 Xiaohai", 0.72, 0.44),
      ("SI19", "SI19 Tinggong", 0.55, 0.12)]),
    ("TE", "Triple Energizer", "back", True,
     [(0.755, 0.635), (0.76, 0.60), (0.74, 0.52), (0.72, 0.44), (0.70, 0.35),
      (0.66, 0.27), (0.58, 0.18), (0.555, 0.11)],
     [("TE5", "TE5 Waiguan", 0.74, 0.52), ("TE14", "TE14 Jianliao", 0.66, 0.27),
      ("TE23", "TE23 Sizhukong", 0.555, 0.11)]),
    # ===== SIDE (lateral) view — single, down the profile =====
    ("GB", "Gallbladder", "side", False,
     [(0.56, 0.085), (0.50, 0.062), (0.55, 0.105), (0.49, 0.10), (0.505, 0.165),
      (0.50, 0.245), (0.505, 0.36), (0.495, 0.47), (0.49, 0.54), (0.505, 0.65),
      (0.515, 0.77), (0.525, 0.865), (0.545, 0.945), (0.605, 0.985)],
     [("GB20", "GB20 Fengchi", 0.50, 0.19), ("GB21", "GB21 Jianjing", 0.505, 0.245),
      ("GB30", "GB30 Huantiao", 0.49, 0.54), ("GB34", "GB34 Yanglingquan", 0.52, 0.80),
      ("GB40", "GB40 Qiuxu", 0.545, 0.945), ("GB44", "GB44 Zuqiaoyin", 0.60, 0.98)]),
]

GROUP_LABELS = {
    "LU": "Lung", "LI": "Large Intestine", "ST": "Stomach", "SP": "Spleen",
    "HT": "Heart", "SI": "Small Intestine", "BL": "Bladder", "KI": "Kidney",
    "PC": "Pericardium", "TE": "Triple Energizer", "GB": "Gallbladder", "LR": "Liver",
    "CV": "Conception Vessel (Ren)", "GV": "Governing Vessel (Du)",
}


# Full classical point count per channel (sums to 361).
POINT_COUNTS = {
    "LU": 11, "LI": 20, "ST": 45, "SP": 21, "HT": 9, "SI": 19, "BL": 67, "KI": 27,
    "PC": 9, "TE": 23, "GB": 44, "LR": 14, "CV": 24, "GV": 28,
}
# Well-known pinyin names for select points (point number -> name).
NAMED = {
    "LU": {1: "Zhongfu", 5: "Chize", 7: "Lieque", 9: "Taiyuan", 11: "Shaoshang"},
    "LI": {1: "Shangyang", 4: "Hegu", 11: "Quchi", 15: "Jianyu", 20: "Yingxiang"},
    "ST": {1: "Chengqi", 9: "Renying", 12: "Quepen", 25: "Tianshu", 36: "Zusanli", 40: "Fenglong", 44: "Neiting"},
    "SP": {1: "Yinbai", 3: "Taibai", 6: "Sanyinjiao", 9: "Yinlingquan", 10: "Xuehai", 21: "Dabao"},
    "HT": {3: "Shaohai", 7: "Shenmen", 9: "Shaochong"},
    "SI": {1: "Shaoze", 3: "Houxi", 8: "Xiaohai", 19: "Tinggong"},
    "BL": {1: "Jingming", 10: "Tianzhu", 13: "Feishu", 23: "Shenshu", 40: "Weizhong", 60: "Kunlun", 67: "Zhiyin"},
    "KI": {1: "Yongquan", 3: "Taixi", 7: "Fuliu", 27: "Shufu"},
    "PC": {3: "Quze", 6: "Neiguan", 8: "Laogong", 9: "Zhongchong"},
    "TE": {1: "Guanchong", 5: "Waiguan", 14: "Jianliao", 23: "Sizhukong"},
    "GB": {20: "Fengchi", 21: "Jianjing", 30: "Huantiao", 34: "Yanglingquan", 40: "Qiuxu", 44: "Zuqiaoyin"},
    "LR": {1: "Dadun", 3: "Taichong", 8: "Ququan", 14: "Qimen"},
    "CV": {1: "Huiyin", 4: "Guanyuan", 6: "Qihai", 8: "Shenque", 12: "Zhongwan", 17: "Shanzhong", 22: "Tiantu", 24: "Chengjiang"},
    "GV": {1: "Changqiang", 4: "Mingmen", 14: "Dazhui", 20: "Baihui", 26: "Renzhong"},
}


def sample_polyline(pts, n):
    """n points evenly spaced by arc length from pts[0] (point 1) to pts[-1]."""
    seg = [((pts[i + 1][0] - pts[i][0]) ** 2 + (pts[i + 1][1] - pts[i][1]) ** 2) ** 0.5
           for i in range(len(pts) - 1)]
    total = sum(seg) or 1.0
    out = []
    for i in range(n):
        target = (0.0 if n == 1 else i / (n - 1)) * total
        acc = 0.0
        for j in range(len(seg)):
            if acc + seg[j] >= target or j == len(seg) - 1:
                f = (target - acc) / seg[j] if seg[j] > 0 else 0.0
                out.append((round(pts[j][0] + (pts[j + 1][0] - pts[j][0]) * f, 4),
                            round(pts[j][1] + (pts[j + 1][1] - pts[j][1]) * f, 4)))
                break
            acc += seg[j]
    return out


def _mk(zid, side, group, geometry, anatomy, meaning):
    return {
        "id": zid, "side": side, "bilateral": False, "group": group,
        "geometry": geometry, "anatomy": anatomy,
        "meaning_standard": meaning, "meaning_glen": "",
        "layers": {"embryological_depth": None, "stress_affirmation": None, "touch_for_health": None},
    }


def main():
    zones = []
    groups = []
    for key, name, side, bilateral, waypoints, _curated in CHANNELS:
        groups.append({"id": key, "label": GROUP_LABELS[key]})
        # full classical point set, distributed along the channel path
        n = POINT_COUNTS[key]
        sampled = sample_polyline(waypoints, n)
        pts = []
        for k in range(1, n + 1):
            x, y = sampled[k - 1]
            pin = NAMED.get(key, {}).get(k)
            pname = f"{key}{k} {pin}" if pin else f"{key}{k}"
            pts.append((f"{key}{k}", pname, x, y))
        copies = [("R", waypoints, pts)]
        if bilateral:
            wl = [(round(1 - x, 4), y) for (x, y) in waypoints]
            pl = [(s, nm, round(1 - x, 4), y) for (s, nm, x, y) in pts]
            copies.append(("L", wl, pl))
        for tag, wps, ppts in copies:
            suffix = f"-{tag}" if bilateral else ""
            zones.append(_mk(
                f"mer-{key}{suffix}-line", side, key,
                {"type": "path", "d": catmull_rom_open(wps)},
                f"{name} channel", f"The {name} channel."))
            for slug, pname, x, y in ppts:
                zones.append(_mk(
                    f"mer-{slug}{suffix}", side, key,
                    {"type": "point", "x": x, "y": y},
                    pname, f"{pname}, on the {name} channel."))
    front = bbo.build_outline()
    data = {
        "system": "meridian",
        "reference_frame": "body_outline",
        "side_noun": "view",
        "group_noun": "channel",
        "outline": front,                       # fallback
        "outlines": {"front": front, "back": front, "side": bbo.build_side_outline()},
        "groups": groups,
        "anchors": [
            {"key": "head-f", "view": "front", "template": {"x": 0.50, "y": 0.02}, "hint": "Tap the top of the head."},
            {"key": "feet-f", "view": "front", "template": {"x": 0.50, "y": 0.985}, "hint": "Tap the point between the ankles."},
            {"key": "head-b", "view": "back", "template": {"x": 0.50, "y": 0.02}, "hint": "Tap the top of the head."},
            {"key": "feet-b", "view": "back", "template": {"x": 0.50, "y": 0.985}, "hint": "Tap the point between the ankles."},
            {"key": "head-s", "view": "side", "template": {"x": 0.475, "y": 0.02}, "hint": "Tap the top of the head."},
            {"key": "foot-s", "view": "side", "template": {"x": 0.55, "y": 0.985}, "hint": "Tap the front of the foot."},
        ],
        "zones": zones,
    }
    out = ROOT / "data" / "bodymap-meridian.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    lines = sum(1 for z in zones if z["geometry"]["type"] == "path")
    pts = sum(1 for z in zones if z["geometry"]["type"] == "point")
    print(f"wrote {out}: {len(zones)} zones ({lines} channel lines, {pts} points), {len(groups)} channels")


if __name__ == "__main__":
    main()
