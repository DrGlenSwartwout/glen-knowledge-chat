#!/usr/bin/env python3
"""Build web-optimized journey-shell assets from the master scene art.

One-time / repeatable local build step. Outputs are committed under
static/journey/; nothing here runs at request time, so Pillow is a *build*
dependency only. It IS declared in requirements.txt as of #1042 -- a dependency
the tests import but nobody declares is the more fragile state, and the cost is
one small wheel prod never calls.

Usage:
    python3 scripts/build_journey_assets.py [SOURCE_PNG]

Default SOURCE_PNG: ~/Downloads/journey-ribbon-samples/v2-3-teal-dawn.png

Produces:
    static/journey/scene.webp              (full scene, < ~300 KB)
    static/journey/thumb-{scan,find,heal,give,home}.webp  (~96px ribbon chips)

The land/sign layout below was measured from the master art with a percentage
grid. If the master art changes, re-measure and update SIGN_CROPS / the printed
coordinates that feed static/shell-map.json's `scene` block.

If Pillow/webp is unavailable, fall back to macOS `sips` (see README note); this
script requires Pillow.
"""
import os
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "static" / "journey"
DEFAULT_SRC = Path.home() / "Downloads" / "journey-ribbon-samples" / "landmarks" / "5-journey-unified-v12-trumpet.png"

SCENE_MAX_W = 1328          # native master width; do not upscale
SCENE_WEBP_QUALITY = 82     # tuned to land scene.webp under ~300 KB
THUMB_PX = 96

# Square-ish crop boxes (fractions of W,H) centred on each waypoint, with enough
# surrounding meadow/path/water context to read as a little scene at chip size.
THUMB_CROPS = {
    "home": (0.03, 0.46, 0.25, 0.68),   # the green hobbit door
    "scan": (0.32, 0.24, 0.54, 0.46),   # Glendalf's ear & cupped hand
    "find": (0.53, 0.45, 0.75, 0.67),   # the remedy bottle & hands
    "heal": (0.51, 0.33, 0.73, 0.55),   # the path into the distance
    "give": (0.60, 0.07, 0.84, 0.31),   # the glowing cathedral
}

# Hotspot anchor points (% of scene box) where each waypoint sits.
# Printed for pasting into static/shell-map.json -> scene block.
HOTSPOTS = {
    "home": {"x": 14, "y": 57, "w": 13, "h": 23},
    "scan": {"x": 43, "y": 35, "w": 12, "h": 19},
    "find": {"x": 64, "y": 56, "w": 13, "h": 17},
    "heal": {"x": 62, "y": 44, "w": 14, "h": 13},
    "give": {"x": 72, "y": 18, "w": 16, "h": 27},
}


def _crop_box(im, frac):
    W, H = im.size
    l, t, r, b = frac
    return (int(W * l), int(H * t), int(W * r), int(H * b))


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    if not src.exists():
        sys.exit(f"source image not found: {src}")
    OUT.mkdir(parents=True, exist_ok=True)

    im = Image.open(src).convert("RGB")
    W, H = im.size
    print(f"source: {src}  ({W}x{H})")

    # Full scene (no upscale)
    scene = im
    if W > SCENE_MAX_W:
        scene = im.resize((SCENE_MAX_W, round(H * SCENE_MAX_W / W)), Image.LANCZOS)
    scene_path = OUT / "scene.webp"
    scene.save(scene_path, "WEBP", quality=SCENE_WEBP_QUALITY, method=6)
    kb = scene_path.stat().st_size / 1024
    print(f"scene.webp: {scene.size[0]}x{scene.size[1]}  {kb:.0f} KB")
    if kb > 300:
        print("  WARNING: scene.webp over 300 KB — lower SCENE_WEBP_QUALITY")

    # Thumbnails
    for key, frac in THUMB_CROPS.items():
        chip = im.crop(_crop_box(im, frac))
        # centre-square then resize
        cw, ch = chip.size
        s = min(cw, ch)
        chip = chip.crop(((cw - s) // 2, (ch - s) // 2, (cw + s) // 2, (ch + s) // 2))
        chip = chip.resize((THUMB_PX, THUMB_PX), Image.LANCZOS)
        tp = OUT / f"thumb-{key}.webp"
        chip.save(tp, "WEBP", quality=85, method=6)
        print(f"thumb-{key}.webp: {tp.stat().st_size/1024:.0f} KB")

    print("\nHotspots for static/shell-map.json -> scene block:")
    import json
    print(json.dumps(HOTSPOTS, indent=2))


if __name__ == "__main__":
    main()
