# tests/test_journey_assets.py
from pathlib import Path
from PIL import Image

OUT = Path(__file__).resolve().parent.parent / "static" / "journey"
KEYS = ["home", "scan", "find", "heal", "give"]

def test_scene_is_v12_aspect_and_small():
    p = OUT / "scene.webp"
    assert p.exists(), "scene.webp missing — run scripts/build_journey_assets.py"
    w, h = Image.open(p).size
    assert abs((w / h) - (1328 / 800)) < 0.02, f"scene aspect {w}x{h} is not the v12 1328x800 scene"
    assert p.stat().st_size < 300 * 1024, "scene.webp over 300 KB"

def test_all_five_thumbs_exist():
    for k in KEYS:
        t = OUT / f"thumb-{k}.webp"
        assert t.exists(), f"thumb-{k}.webp missing"
        assert t.stat().st_size < 30 * 1024
