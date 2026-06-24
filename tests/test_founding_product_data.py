import json
import app as appmod
import dashboard.founding as founding


def test_product_data_includes_founding_block(monkeypatch):
    monkeypatch.setattr(appmod, "_get_product", lambda s: {
        "slug": s, "name": "Neuro Magnesium", "price_cents": 8000,
        "description": "Foundational eye and brain support.", "qbo_item_id": ""} if s == "neuro-magnesium" else None)
    monkeypatch.setattr(founding, "_CONFIG", {
        "neuro-magnesium": {"cap": 2500, "batch_label": "Founding Batch No. 1",
                            "video_url": "/clip/neuro/promo.mp4", "closes_at": ""}})
    monkeypatch.setattr(founding, "count_reserved", lambda cx, slug: 653)
    c = appmod.app.test_client()
    r = c.get("/begin/product-data/neuro-magnesium")
    assert r.status_code == 200
    data = r.get_json()
    assert data["founding"]["batch_label"] == "Founding Batch No. 1"
    assert data["founding"]["remaining"] == 2500 - 653
    assert data["founding_video_url"] == "/clip/neuro/promo.mp4"
