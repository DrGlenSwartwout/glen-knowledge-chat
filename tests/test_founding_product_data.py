import importlib
import os
import sys
from pathlib import Path

import pytest

# Skip this entire module if Pinecone is not configured
if not os.environ.get("PINECONE_API_KEY"):
    pytest.skip("requires PINECONE_API_KEY in env (use doppler run)", allow_module_level=True)


def _load_app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return importlib.import_module("app")


def test_product_data_includes_founding_block(monkeypatch):
    appmod = _load_app()
    import dashboard.founding as founding
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


def test_product_page_data_includes_founding_block(monkeypatch):
    appmod = _load_app()
    import dashboard.founding as founding
    monkeypatch.setattr(appmod, "_get_product", lambda s: {
        "slug": s, "name": "Neuro Magnesium", "price_cents": 8000,
        "description": "Foundational eye and brain support.", "qbo_item_id": ""} if s == "neuro-magnesium" else None)
    monkeypatch.setattr(founding, "_CONFIG", {
        "neuro-magnesium": {"cap": 2500, "batch_label": "Founding Batch No. 1",
                            "video_url": "/clip/neuro/promo.mp4", "closes_at": ""}})
    monkeypatch.setattr(founding, "count_reserved", lambda cx, slug: 653)
    c = appmod.app.test_client()
    r = c.get("/begin/product-page-data/neuro-magnesium")
    assert r.status_code == 200
    data = r.get_json()
    assert data["founding"]["batch_label"] == "Founding Batch No. 1"
    assert data["founding"]["remaining"] == 2500 - 653
    assert data["founding_video_url"] == "/clip/neuro/promo.mp4"
