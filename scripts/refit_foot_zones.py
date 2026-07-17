#!/usr/bin/env python3
"""Reproject the foot reflexology zones onto the realistic 5-toe outline.

The 34-zone reflexology layout was authored for a fuller, wider foot. When the
outline was replaced with a realistic anatomical silhouette (CC0 svgsilh #42674,
a right foot, sole view, with a deep medial arch), ~19 ovals fell outside the
new shape's arch cavity and toe gaps.

Rather than re-derive the anatomy, this script *reprojects* each zone onto the
new outline while preserving its clinical role:

  * medial<->lateral position is preserved as a *fraction across the sole width*
    at the zone's vertical band, so a zone stays on the arch side / lateral side
    it belongs to, but now hugs the realistic edges;
  * the six pure-toe reflexes (brain, sinuses, pituitary, eyes, ears, neck) are
    placed explicitly on the actual toe pads of the outline.

Frames: side="right" zones (bilateral + right-only) live in the right-foot
frame and are fit against the right outline edges. side="left" zones live in the
mirrored (left-foot) display frame and are fit against the mirrored edges. This
matches the renderer, which mirrors the outline and bilateral zones for the left
foot but draws non-bilateral zones at their stored (already-mirrored) coords.

Idempotent-ish: reads data/bodymap-foot.json, rewrites zone geometry only.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOOT = ROOT / "data" / "bodymap-foot.json"

# --- old authoring envelope: the fuller foot the zones were laid out on ---
OLD_L, OLD_R = 0.27, 0.73          # medial/lateral edges of the old sole field
MARGIN = 0.012                      # keep ovals this far inside the outline edge
TOE_CY_CUTOFF = 0.24                # zones above this are pure-toe reflexes


def _tokens(d):
    return re.findall(r"[MLCZ]|-?\d*\.?\d+", d)


def _flatten(d):
    """Flatten an SVG path (M/L/C/Z, absolute) into polygon subpaths."""
    subs, cur = [], None
    cx = cy = sx = sy = 0.0
    toks = _tokens(d)
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == "M":
            if cur:
                subs.append(cur)
            cx, cy = float(toks[i + 1]), float(toks[i + 2])
            sx, sy = cx, cy
            cur = [(cx, cy)]
            i += 3
        elif t == "L":
            cx, cy = float(toks[i + 1]), float(toks[i + 2])
            cur.append((cx, cy))
            i += 3
        elif t == "C":
            x1, y1, x2, y2, x, y = (float(toks[i + j]) for j in range(1, 7))
            for s in range(1, 17):
                u = s / 16.0
                mt = 1 - u
                bx = mt**3 * cx + 3 * mt * mt * u * x1 + 3 * mt * u * u * x2 + u**3 * x
                by = mt**3 * cy + 3 * mt * mt * u * y1 + 3 * mt * u * u * y2 + u**3 * y
                cur.append((bx, by))
            cx, cy = x, y
            i += 7
        elif t == "Z":
            cur.append((sx, sy))
            i += 1
        else:
            i += 1
    if cur:
        subs.append(cur)
    return subs


def _bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _edges_at(poly, y):
    """Leftmost/rightmost x where the polygon spans horizontal line y."""
    xs = []
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 <= y <= y2) or (y2 <= y <= y1):
            if abs(y2 - y1) < 1e-9:
                xs += [x1, x2]
            else:
                xs.append(x1 + (x2 - x1) * (y - y1) / (y2 - y1))
    return (min(xs), max(xs)) if xs else None


def build_geometry(outline):
    subs = sorted(_flatten(outline), key=len, reverse=True)
    sole = subs[0]
    toes = subs[1:]
    tb = _bbox(sole)                       # sole bbox
    toe_centers = sorted(((_bbox(t)[0] + _bbox(t)[2]) / 2,
                          (_bbox(t)[1] + _bbox(t)[3]) / 2, _bbox(t)) for t in toes)
    return {"sole": sole, "sole_bbox": tb, "toes": toe_centers}


def reproject_sole(cx, cy, rx, ry, geo, mirror):
    """Place an oval at its fraction-across-width, hugging the realistic edges."""
    frac = (cx - OLD_L) / (OLD_R - OLD_L)
    frac = min(1.0, max(0.0, frac))
    e = _edges_at(geo["sole"], cy)
    if e is None:                          # below/above sole: clamp to nearest
        _, y0, _, y1 = geo["sole_bbox"]
        e = _edges_at(geo["sole"], min(max(cy, y0 + 1e-3), y1 - 1e-3)) or (OLD_L, OLD_R)
    L, R = e
    if mirror:
        L, R = 1 - R, 1 - L
        frac = 1 - frac                    # medial stays medial in mirrored frame
    width = R - L
    rx_fit = min(rx, max(0.02, width / 2 - MARGIN))
    lo = L + MARGIN + rx_fit
    hi = R - MARGIN - rx_fit
    if hi < lo:
        lo = hi = (L + R) / 2
    ncx = lo + frac * (hi - lo)
    # vertical clamp within a sane sole band; the deep arch is narrow, so keep
    # ovals from being tall enough to poke past its curving edges.
    ry_fit = min(ry, 0.045)
    ncy = min(0.95 - ry_fit, max(0.205 + ry_fit, cy))
    return round(ncx, 3), round(ncy, 3), round(rx_fit, 3), round(ry_fit, 3)


def place_toe_zones(geo):
    """Explicit toe-pad targets keyed by zone id (right frame; bilateral)."""
    toes = geo["toes"]                      # left->right: big, 2, 3, 4, pinky
    big = toes[0][0]
    t2, t3, t4, t5 = (toes[1][0], toes[2][0], toes[3][0], toes[4][0])
    # Splayed toes leave gaps that are OUTSIDE the outline, so each toe reflex is
    # centred on an actual toe pad (not in a gap), sized to sit within one toe.
    return {
        "foot-brain":       (round(big, 3), 0.085, 0.05, 0.045),   # big-toe pad, head/brain
        "foot-pituitary":   (round(big, 3), 0.135, 0.03, 0.02),    # centre of big toe
        "foot-neck-throat": (round(big + 0.03, 3), 0.235, 0.03, 0.02),  # base of big toe
        "foot-sinuses":     (round(t3, 3), 0.095, 0.04, 0.025),    # 3rd-toe tip (sinuses)
        "foot-eyes":        (round(t2, 3), 0.115, 0.04, 0.03),     # 2nd-toe pad (eyes)
        "foot-ears":        (round(t4, 3), 0.160, 0.045, 0.03),    # 4th-toe pad (ears)
    }


def main():
    data = json.loads(FOOT.read_text())
    geo = build_geometry(data["outline"])
    toe_targets = place_toe_zones(geo)
    changed = 0
    for z in data["zones"]:
        g = z.get("geometry", {})
        if g.get("type") != "ellipse":
            continue
        if z["id"] in toe_targets:
            cx, cy, rx, ry = toe_targets[z["id"]]
        else:
            mirror = z.get("side") == "left"
            cx, cy, rx, ry = reproject_sole(g["cx"], g["cy"], g["rx"], g["ry"], geo, mirror)
        g.update(cx=cx, cy=cy, rx=rx, ry=ry)
        changed += 1
    FOOT.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"reprojected {changed} zones onto realistic outline")


if __name__ == "__main__":
    main()
