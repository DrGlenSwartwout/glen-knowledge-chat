"""Pure-geometry packer: how many cylindrical bottles fit in a USPS flat-rate
box. Each bottle is modeled as a square prism (footprint Ø×Ø, height H), packed
upright in horizontal shelves; all 3 box orientations are tried and the best is
kept. Conservative: no intra-layer stacking, no hex nesting. No DB, no I/O.

All dimensions in millimetres (integers).
"""
from __future__ import annotations
from typing import List, Set, Tuple

# Interior dims in mm (cm x 10): S 5x15x23, M 13x22x27, L 14x29x30
BOXES_MM = {"S": (50, 150, 230), "M": (130, 220, 270), "L": (140, 290, 300)}
BOX_ORDER = ("S", "M", "L")  # ascending volume / flat-rate cost


def _pack2d(squares: List[int], bw: int, bl: int) -> List[bool]:
    """Shelf-pack square footprints (side lengths) into a bw x bl base.
    Returns a placed/not-placed flag per square. Largest-first within a row."""
    placed = [False] * len(squares)
    y = 0
    while True:
        idxs = [i for i, p in enumerate(placed) if not p]
        if not idxs:
            break
        cand = [i for i in idxs if y + squares[i] <= bl and squares[i] <= bw]
        if not cand:
            break
        first = max(cand, key=lambda i: squares[i])
        row_h = squares[first]
        x = 0
        while True:
            opts = [i for i in idxs if not placed[i]
                    and squares[i] <= row_h and x + squares[i] <= bw]
            if not opts:
                break
            pick = max(opts, key=lambda i: squares[i])
            placed[pick] = True
            x += squares[pick]
        y += row_h
    return placed


def _pack_oriented(items: List[Tuple[int, int]], bw: int, bl: int, H: int) -> Set[int]:
    """Layer-pack items (list of (d,h)) into a base bw x bl, vertical room H.
    Returns the set of placed indices."""
    order = sorted(range(len(items)), key=lambda i: (-items[i][1], -items[i][0]))
    placed: Set[int] = set()
    used = 0
    while True:
        rem = [i for i in order if i not in placed]
        if not rem:
            break
        tallest = items[rem[0]][1]
        if used + tallest > H:
            break
        layer_h = tallest
        eligible = [i for i in rem if items[i][1] <= layer_h]
        flags = _pack2d([items[i][0] for i in eligible], bw, bl)
        layer = [eligible[k] for k, f in enumerate(flags) if f]
        if not layer:
            break
        placed.update(layer)
        used += layer_h
    return placed


def _effective(items, wrap_mm):
    return [(d + wrap_mm, h + wrap_mm) for (d, h) in items]


def fit_subset(items, box_mm, *, wrap_mm=0, box_margin_mm=0) -> Set[int]:
    """Indices of `items` that fit in one box, best of 3 orientations."""
    if not items:
        return set()
    eff = _effective(items, wrap_mm)
    a, b, c = (d - box_margin_mm for d in box_mm)
    best: Set[int] = set()
    for vax, (bw, bl) in ((a, (b, c)), (b, (a, c)), (c, (a, b))):
        if vax <= 0 or bw <= 0 or bl <= 0:
            continue
        placed = _pack_oriented(eff, bw, bl, vax)
        if len(placed) > len(best):
            best = placed
    return best


def fits_all(items, box_mm, *, wrap_mm=0, box_margin_mm=0) -> bool:
    return len(fit_subset(items, box_mm, wrap_mm=wrap_mm,
                          box_margin_mm=box_margin_mm)) == len(items)


def pack_count(items, box_mm, *, wrap_mm=0, box_margin_mm=0) -> int:
    return len(fit_subset(items, box_mm, wrap_mm=wrap_mm,
                          box_margin_mm=box_margin_mm))
