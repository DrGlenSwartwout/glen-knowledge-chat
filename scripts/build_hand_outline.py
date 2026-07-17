#!/usr/bin/env python3
"""Generate a smooth, realistic open-hand outline (right hand, palm view).

svgsilh has no clean open-hand silhouette (only gestures / inky handprints), so
the hand outline is drawn parametrically: a set of boundary points tracing the
palm, four fingers, and thumb, smoothed into one closed cubic-bezier path via a
Catmull-Rom spline. Building it this way (vs. tracing a photo) lets us place the
digits exactly where the reflexology zone grid needs them, so zones fit by
construction. Coordinates are normalized to the unit box; right hand, palm up,
fingers pointing up, thumb to the left (radial/medial = left, matching the
foot's medial edge convention so `bilateral` mirroring works the same way).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Boundary points, traced clockwise from the wrist's ulnar (little-finger) side,
# up the outside, around each fingertip and down into each web, around the thumb,
# and back down the radial side to the wrist. Fingertip apex + two shoulder
# points give a rounded tip; web bottoms dip between the finger bases.
HAND_POINTS = [
    # Each fingertip is a rounded cap: two apex points sit close together across
    # the top (same y, small x gap) so the spline arcs over a rounded tip instead
    # of pulling to a single spike. The thumb is a slim, tapered digit (not a wide
    # lobe), kept wide enough at the base to enclose the brain/pituitary zones.
    # ---- wrist / ulnar (right) side of palm ----
    (0.620, 0.905),   # wrist, ulnar corner
    (0.700, 0.720),   # palm ulnar mid
    (0.728, 0.560),   # palm ulnar upper (widest)
    (0.732, 0.455),   # base of little finger, outer
    # ---- little (pinky) finger ----
    (0.722, 0.330),
    (0.714, 0.248),   # pinky outer shoulder
    (0.702, 0.214),   # pinky tip, outer apex
    (0.676, 0.214),   # pinky tip, inner apex (rounded cap)
    (0.664, 0.248),   # pinky inner shoulder
    (0.652, 0.360),   # web pinky-ring
    # ---- ring finger ----
    (0.636, 0.300),
    (0.622, 0.168),   # ring outer shoulder
    (0.610, 0.132),   # ring tip, outer apex
    (0.588, 0.132),   # ring tip, inner apex
    (0.576, 0.168),   # ring inner shoulder
    (0.560, 0.382),   # web ring-middle
    # ---- middle finger (longest) ----
    (0.544, 0.250),
    (0.524, 0.112),   # middle outer shoulder
    (0.511, 0.074),   # middle tip, outer apex
    (0.489, 0.074),   # middle tip, inner apex
    (0.476, 0.112),   # middle inner shoulder
    (0.456, 0.382),   # web middle-index
    # ---- index finger ----
    (0.438, 0.270),
    (0.418, 0.176),   # index outer shoulder
    (0.405, 0.140),   # index tip, outer apex
    (0.383, 0.140),   # index tip, inner apex
    (0.370, 0.178),   # index inner shoulder
    (0.348, 0.450),   # deep web index-thumb
    # ---- thumb (radial side, slim taper angled up-left, rounded tip) ----
    (0.300, 0.500),   # thenar web into thumb (upper)
    (0.230, 0.455),   # thumb outer shoulder
    (0.150, 0.468),   # thumb outer tip-shoulder
    (0.113, 0.500),   # thumb tip, outer apex
    (0.113, 0.540),   # thumb tip, inner apex (rounded cap)
    (0.152, 0.572),   # thumb inner tip-shoulder
    (0.238, 0.602),   # thumb base into palm (lower)
    # ---- radial (left) side of palm down to wrist ----
    (0.300, 0.720),
    (0.330, 0.850),
    (0.395, 0.905),   # wrist, radial corner
]


def catmull_rom_closed(points):
    """Closed Catmull-Rom spline -> one cubic-bezier 'd' string through points."""
    n = len(points)
    d = f"M{points[0][0]:.4f} {points[0][1]:.4f} "
    for i in range(n):
        p0 = points[(i - 1) % n]
        p1 = points[i]
        p2 = points[(i + 1) % n]
        p3 = points[(i + 2) % n]
        c1x = p1[0] + (p2[0] - p0[0]) / 6.0
        c1y = p1[1] + (p2[1] - p0[1]) / 6.0
        c2x = p2[0] - (p3[0] - p1[0]) / 6.0
        c2y = p2[1] - (p3[1] - p1[1]) / 6.0
        d += f"C{c1x:.4f} {c1y:.4f} {c2x:.4f} {c2y:.4f} {p2[0]:.4f} {p2[1]:.4f} "
    return d.strip() + " Z"


def main():
    path = catmull_rom_closed(HAND_POINTS)
    print(path)


if __name__ == "__main__":
    main()
