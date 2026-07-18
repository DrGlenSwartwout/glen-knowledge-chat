#!/usr/bin/env python3
"""Shared helpers for whole-body system atlases (nervous, endocrine, respiratory,
digestive, cardiovascular, urogenital, immune/lymphatic).

Keeps every system map on the SAME body silhouette and a consistent zone shape, so
a given structure (heart, kidney) sits in the same place across maps. Each system
generator imports this, defines its groups + zones, and calls write_system().

Positions are standards-based renderings for Glen's clinical refinement, not gospel.
Laterality is anatomical-as-viewed (patient's right = viewer's left, x < 0.5).
"""
import json
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_FRONT = None


def front_outline():
    """The shared front body silhouette (cached)."""
    global _FRONT
    if _FRONT is None:
        spec = importlib.util.spec_from_file_location("bbo", ROOT / "scripts" / "build_body_outline.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _FRONT = m.build_outline()
    return _FRONT


def ellipse(cx, cy, rx, ry):
    return {"type": "ellipse", "cx": round(cx, 4), "cy": round(cy, 4),
            "rx": round(rx, 4), "ry": round(ry, 4)}


def path(d):
    return {"type": "path", "d": d}


def catmull(points):
    """Smooth open catmull-rom path 'd' through normalized (x,y) points."""
    p = list(points)
    if len(p) < 2:
        x, y = p[0]
        return f"M {x:.4f} {y:.4f}"
    pts = [p[0]] + p + [p[-1]]
    d = f"M {p[0][0]:.4f} {p[0][1]:.4f}"
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        c1x = p1[0] + (p2[0] - p0[0]) / 6.0
        c1y = p1[1] + (p2[1] - p0[1]) / 6.0
        c2x = p2[0] - (p3[0] - p1[0]) / 6.0
        c2y = p2[1] - (p3[1] - p1[1]) / 6.0
        d += f" C {c1x:.4f} {c1y:.4f} {c2x:.4f} {c2y:.4f} {p2[0]:.4f} {p2[1]:.4f}"
    return d


def zone(zid, name, view, group, geom, meaning, **extra):
    z = {"id": zid, "side": view, "bilateral": False, "group": group,
         "geometry": geom, "anatomy": name, "meaning_standard": meaning,
         "meaning_glen": "",
         "layers": {"embryological_depth": None, "stress_affirmation": None, "touch_for_health": None}}
    z.update(extra)
    return z


def _anchors(views):
    out = []
    for v in views:
        out.append({"key": f"head-{v}", "view": v, "template": {"x": 0.50, "y": 0.02},
                    "hint": "Tap the top of the head."})
        out.append({"key": f"feet-{v}", "view": v, "template": {"x": 0.50, "y": 0.985},
                    "hint": "Tap the point between the ankles."})
    return out


def write_system(system, groups, zones, *, side_noun="view", group_noun="region",
                 views=("front", "back")):
    """Write data/bodymap-<system>.json on the shared front silhouette. Every view
    (front/back, or male/female) shares the silhouette via the outlines map."""
    front = front_outline()
    data = {
        "system": system, "reference_frame": "body_outline",
        "side_noun": side_noun, "group_noun": group_noun,
        "outline": front, "outlines": {v: front for v in views},
        "groups": groups, "anchors": _anchors(views), "zones": zones,
    }
    out = ROOT / "data" / f"bodymap-{system}.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    from collections import Counter
    print(f"wrote {out}: {len(zones)} zones", dict(Counter(z['side'] for z in zones)))
    return data
