import os
# Dummy keys so `import app` (which constructs OpenAI + Pinecone clients at import)
# succeeds under a secretless CI without doppler.
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")
os.environ.setdefault("PINECONE_API_KEY", "pc-dummy")

import pytest

import app


@pytest.fixture()
def client():
    return app.app.test_client()


def _prep(monkeypatch):
    monkeypatch.setattr(
        app, "_get_product",
        lambda slug: {"slug": "brain-boost", "name": "Brain Boost",
                      "price_cents": 6997} if slug == "brain-boost" else None)


def test_product_page_has_no_cart_markup_when_flag_off(client, monkeypatch):
    _prep(monkeypatch)
    monkeypatch.setattr(app, "_PORTAL_CART_ENABLED", False)
    html = client.get("/begin/product/brain-boost").get_data(as_text=True)
    assert "data-cart-add" not in html
    assert "CART_ENABLED = false" in html


def test_product_page_exposes_the_cart_control_when_flag_on(client, monkeypatch):
    _prep(monkeypatch)
    monkeypatch.setattr(app, "_PORTAL_CART_ENABLED", True)
    html = client.get("/begin/product/brain-boost").get_data(as_text=True)
    assert "CART_ENABLED = true" in html
    # The control must actually survive in the served page, not just the flag
    # var: the function definition, its data-cart-add attribute, and its call
    # site all have to be present or the control is dead in production even
    # though CART_ENABLED reads true.
    assert "data-cart-add" in html
    assert "function renderCartControl" in html
    assert "renderCartControl(slug)" in html


def test_founding_close_out_still_targets_the_renamed_wrapper(client, monkeypatch):
    """Task 5 renamed the buy-control wrapper's id from sp-cta-block to
    buy-actions. No test in the repo exercises the founding-launch client JS
    (_applyFoundingStatus) that also looks this id up via getElementById, so
    assert directly against the served markup that the rename left no stale
    reference to the old id and that both the wrapper assignment and the
    founding lookup agree on the new one."""
    _prep(monkeypatch)
    monkeypatch.setattr(app, "_PORTAL_CART_ENABLED", False)
    html = client.get("/begin/product/brain-boost").get_data(as_text=True)
    assert "sp-cta-block" not in html
    assert "wrap.id = 'buy-actions'" in html
    assert "getElementById('buy-actions')" in html
