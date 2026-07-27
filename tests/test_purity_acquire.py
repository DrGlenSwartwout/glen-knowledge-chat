# tests/test_purity_acquire.py
from dashboard import purity_acquire as pa

CLEAN_PAGE = (
    "Magnesium Taurate by Jarrow Formulas SKU JAR-MAGTAU90 . "
    "Other Ingredients: Capsule (hydroxypropylmethylcellulose), magnesium "
    "stearate (vegetable source) and silicon dioxide. Keep out of reach."
)
JARROW_LINE = ("Capsule (hydroxypropylmethylcellulose), magnesium stearate "
               "(vegetable source) and silicon dioxide")
PROD = {"product_slug": "magnesium-taurate", "name": "Magnesium Taurate",
        "brand": "Jarrow Formulas", "sku": "JAR-MAGTAU90"}


def test_split_on_commas_and_and():
    items = pa.split_other_ingredients(JARROW_LINE)
    assert items == ["Capsule (hydroxypropylmethylcellulose)",
                     "magnesium stearate (vegetable source)", "silicon dioxide"]


def test_split_strips_label_and_period():
    items = pa.split_other_ingredients("Other Ingredients: silica, gelatin.")
    assert items == ["silica", "gelatin"]


def test_split_empty_line():
    assert pa.split_other_ingredients("") == []


def test_acquire_success_end_to_end():
    res = pa.acquire(PROD,
                     fetch=lambda url, headers: type("R", (), {"status_code": 200, "text": CLEAN_PAGE})(),
                     call_model=lambda s, n, b, k: {"other_ingredients_line": JARROW_LINE})
    assert res["ok"] is True
    assert res["source"] == "fullscript"
    assert res["raw"] == JARROW_LINE
    assert "magnesium stearate (vegetable source)" in res["parsed"]
    assert "silicon dioxide" in res["parsed"]


def test_acquire_fetch_fails_returns_unrated_shape():
    res = pa.acquire(PROD, fetch=lambda url, headers: type("R", (), {"status_code": 404, "text": ""})(),
                     call_model=lambda *a: {"other_ingredients_line": JARROW_LINE})
    assert res == {"raw": "", "parsed": None, "source": "fullscript", "ok": False}


def test_acquire_extract_finds_nothing_returns_unrated_shape():
    res = pa.acquire(PROD,
                     fetch=lambda url, headers: type("R", (), {"status_code": 200, "text": CLEAN_PAGE})(),
                     call_model=lambda s, n, b, k: {"other_ingredients_line": ""})
    assert res["ok"] is False and res["parsed"] is None


def test_acquire_never_raises_on_internal_error(monkeypatch):
    """Verify that acquire's own try/except guard catches exceptions that
    bypass callee exception handlers, returning the miss shape rather than
    propagating."""
    # Force an exception from inside acquire, bypassing the callees' own catches,
    # to prove acquire's OWN guard returns the miss shape rather than propagating.
    monkeypatch.setattr(pa._fi, "fetch_page_text",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    res = pa.acquire(PROD)
    assert res == {"raw": "", "parsed": None, "source": "fullscript", "ok": False}
