#!/usr/bin/env python3
"""Generate a front-facing human-body silhouette outline for the meridian system.

A silhouette is ~identical front and back, so one outline serves both views; the
meridian paths and points are what distinguish front from back. Standard
anatomical-ish stance: arms slightly abducted, legs slightly apart, so the
limb channels have room. Normalized unit box, head at top. The right half
(viewer's right) is traced head->crotch and mirrored (x -> 1-x) for the left,
then closed with a Catmull-Rom spline (same smoothing as the hand outline).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Right half of the body boundary, head-top down to the crotch (clockwise).
RIGHT_HALF = [
    (0.500, 0.012),   # top of head (shared apex)
    (0.560, 0.035),
    (0.575, 0.075),   # side of head
    (0.560, 0.110),   # jaw
    (0.532, 0.140),   # neck
    (0.560, 0.150),
    (0.700, 0.178),   # shoulder / deltoid
    (0.735, 0.235),
    (0.748, 0.320),   # upper arm
    (0.752, 0.410),   # elbow
    (0.770, 0.500),   # forearm
    (0.788, 0.585),   # wrist
    (0.800, 0.628),   # hand
    (0.775, 0.650),   # fingertips
    (0.735, 0.628),   # hand, inner
    (0.700, 0.545),   # forearm, inner
    (0.686, 0.435),   # elbow, inner
    (0.676, 0.300),   # upper arm, inner
    (0.652, 0.248),   # armpit
    (0.618, 0.360),   # waist
    (0.640, 0.470),   # hip crest
    (0.668, 0.532),   # hip
    (0.636, 0.645),   # thigh
    (0.610, 0.762),   # knee
    (0.590, 0.880),   # calf
    (0.576, 0.955),   # ankle
    (0.622, 0.986),   # foot / toe
    (0.524, 0.988),   # foot, inner / heel
    (0.514, 0.955),   # ankle, inner
    (0.512, 0.800),   # calf, inner
    (0.506, 0.600),   # inner thigh
    (0.500, 0.560),   # crotch (shared)
]


def catmull_rom_closed(points):
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


def body_points():
    # right half head->crotch, then mirrored left half crotch->head (exclude the
    # two shared endpoints so they aren't duplicated in the closed loop)
    left_half = [(1 - x, y) for (x, y) in reversed(RIGHT_HALF[1:-1])]
    return RIGHT_HALF + left_half


def build_outline():
    return catmull_rom_closed(body_points())


if __name__ == "__main__":
    print(build_outline())
