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
    # ---- wrist / ulnar (right) side of palm ----
    (0.620, 0.905),   # wrist, ulnar corner
    (0.700, 0.720),   # palm ulnar mid
    (0.735, 0.560),   # palm ulnar upper (widest)
    (0.740, 0.455),   # base of little finger, outer
    # ---- little (pinky) finger ----
    (0.742, 0.360),
    (0.720, 0.240),   # pinky outer shoulder
    (0.690, 0.205),   # pinky tip
    (0.660, 0.245),   # pinky inner shoulder
    (0.648, 0.380),   # web pinky-ring
    # ---- ring finger ----
    (0.632, 0.300),
    (0.610, 0.115),   # ring tip
    (0.582, 0.290),
    (0.560, 0.395),   # web ring-middle
    # ---- middle finger (longest) ----
    (0.535, 0.250),
    (0.500, 0.045),   # middle tip
    (0.465, 0.250),
    (0.440, 0.395),   # web middle-index
    # ---- index finger ----
    (0.420, 0.270),
    (0.388, 0.110),   # index tip
    (0.356, 0.300),
    (0.338, 0.455),   # deep web index-thumb
    # ---- thumb (radial side, angled up-left) ----
    (0.278, 0.470),
    (0.190, 0.455),   # thumb outer shoulder
    (0.128, 0.505),   # thumb tip
    (0.170, 0.585),   # thumb inner shoulder
    (0.268, 0.630),   # thumb base into palm
    # ---- radial (left) side of palm down to wrist ----
    (0.300, 0.740),
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
