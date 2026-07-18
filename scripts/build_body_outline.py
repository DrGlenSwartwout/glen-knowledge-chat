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
# ~7.5-head proportions with anatomical contours: deltoid, tapered arm, narrow
# waist, hip flare, thigh/calf taper.
RIGHT_HALF = [
    (0.500, 0.012),   # crown (shared apex)
    (0.552, 0.028),   # upper head
    (0.563, 0.070),   # temporal / side of head
    (0.548, 0.108),   # jaw
    (0.519, 0.136),   # jaw -> neck
    (0.523, 0.160),   # neck
    (0.578, 0.172),   # trapezius slope
    (0.662, 0.186),   # shoulder / acromion
    (0.708, 0.216),   # deltoid
    (0.707, 0.288),   # upper arm
    (0.700, 0.368),   # arm above elbow
    (0.713, 0.448),   # elbow / forearm
    (0.736, 0.548),   # forearm
    (0.759, 0.612),   # wrist / hand
    (0.749, 0.652),   # fingertips
    (0.719, 0.634),   # hand, inner
    (0.695, 0.562),   # forearm, inner
    (0.681, 0.456),   # elbow, inner
    (0.665, 0.342),   # upper arm, inner
    (0.643, 0.290),   # armpit
    (0.611, 0.346),   # ribcage side
    (0.567, 0.400),   # waist (narrowest)
    (0.627, 0.506),   # hip / pelvis (widest)
    (0.629, 0.576),   # upper thigh
    (0.609, 0.690),   # thigh
    (0.591, 0.776),   # knee
    (0.599, 0.856),   # calf
    (0.567, 0.945),   # ankle
    (0.613, 0.986),   # foot / toe
    (0.523, 0.988),   # foot, inner / heel
    (0.517, 0.946),   # ankle, inner
    (0.523, 0.800),   # calf, inner
    (0.521, 0.640),   # inner thigh
    (0.506, 0.546),   # upper inner thigh
    (0.500, 0.520),   # crotch (shared)
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


# Lateral (side) profile, facing right: front of the body = +x, back = -x.
# Traced clockwise from the crown, down the FRONT, around the foot, up the BACK.
SIDE_PROFILE = [
    (0.475, 0.014),   # crown
    (0.540, 0.028),   # top of head
    (0.562, 0.062),   # forehead
    (0.580, 0.088),   # brow / nose bridge
    (0.602, 0.100),   # nose tip
    (0.576, 0.116),   # upper lip
    (0.560, 0.134),   # chin
    (0.540, 0.160),   # under chin / throat
    (0.560, 0.206),   # clavicle
    (0.586, 0.256),   # chest (pectoral)
    (0.578, 0.332),   # lower chest
    (0.560, 0.402),   # belly
    (0.548, 0.472),   # lower belly
    (0.540, 0.522),   # groin (front)
    (0.562, 0.586),   # front thigh
    (0.556, 0.682),   # thigh
    (0.548, 0.770),   # front knee
    (0.558, 0.842),   # shin
    (0.548, 0.922),   # lower shin
    (0.545, 0.956),   # front ankle
    (0.602, 0.988),   # toes
    (0.484, 0.990),   # heel
    (0.485, 0.956),   # back ankle
    (0.454, 0.906),   # achilles
    (0.438, 0.850),   # calf (bulge)
    (0.470, 0.774),   # back of knee
    (0.454, 0.690),   # hamstring
    (0.432, 0.576),   # buttock (bulge)
    (0.450, 0.506),   # gluteal fold
    (0.460, 0.454),   # sacrum
    (0.448, 0.390),   # lumbar (lordosis)
    (0.434, 0.312),   # mid back
    (0.442, 0.240),   # upper back (thoracic)
    (0.470, 0.186),   # nape
    (0.434, 0.120),   # occiput
    (0.432, 0.070),   # back of head
    (0.456, 0.028),   # back of crown
]


def build_outline():
    return catmull_rom_closed(body_points())


def build_side_outline():
    return catmull_rom_closed(SIDE_PROFILE)


if __name__ == "__main__":
    print(build_outline())
