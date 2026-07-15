from dashboard import life_stress

FINDINGS = {  # a fake scan_context payload
    "found": True,
    "findings": [
        {"code": "ED12", "rank": 1},   # Fear (top)
        {"code": "ED11", "rank": 2},   # Anger
        {"code": "ED3",  "rank": 3},   # no emotion
    ],
}
EMO = {"ED12": ["Fear"], "ED11": ["Anger"]}
MAP = {"Fear": ["Mimulus Flower Essence"], "Anger": ["Willow Flower Essence"]}
PRODUCTS = {"products": {
    "mimulus-flower-essence-in-terrain-restore": {"name": "Mimulus Flower Essence in Terrain Restore"},
    "willow-flower-essence-in-terrain-restore": {"name": "Willow Flower Essence in Terrain Restore"},
}}

def test_dominant_emotion_first_with_note(monkeypatch):
    monkeypatch.setattr(life_stress.biofield_e4l, "scan_context", lambda *a, **k: FINDINGS)
    monkeypatch.setattr(life_stress.biofield_e4l, "emotions_for_codes", lambda *a, **k: EMO)
    out = life_stress.recommend("a@b.com", "2026-07-14", products=PRODUCTS, emotion_map=MAP)
    assert out["label"] == "Life Stress"
    assert out["patterns"][0]["emotion"] == "Fear"          # rank 1 weighted highest
    names = [i["name"] for i in out["items"]]
    assert "Mimulus Flower Essence" in names
    fear_item = next(i for i in out["items"] if i["name"] == "Mimulus Flower Essence")
    assert fear_item["note"] == "for the fear pattern in your scan"
    assert fear_item["url"] == "/begin/product/mimulus-flower-essence-in-terrain-restore"

def test_no_scan_returns_none(monkeypatch):
    monkeypatch.setattr(life_stress.biofield_e4l, "scan_context", lambda *a, **k: {"found": False})
    assert life_stress.recommend("a@b.com", "2026-07-14", products=PRODUCTS, emotion_map=MAP) is None

def test_no_emotions_returns_none(monkeypatch):
    monkeypatch.setattr(life_stress.biofield_e4l, "scan_context", lambda *a, **k: FINDINGS)
    monkeypatch.setattr(life_stress.biofield_e4l, "emotions_for_codes", lambda *a, **k: {})
    assert life_stress.recommend("a@b.com", "2026-07-14", products=PRODUCTS, emotion_map=MAP) is None

def test_never_raises_on_bad_scan(monkeypatch):
    monkeypatch.setattr(life_stress.biofield_e4l, "scan_context", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert life_stress.recommend("a@b.com", "2026-07-14", products=PRODUCTS, emotion_map=MAP) is None
