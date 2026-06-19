import sqlite3
from dashboard import sales_pages as sp
from dashboard import sales_copy as sc

def _cx():
    return sqlite3.connect(":memory:")

def test_upsert_then_get_section_roundtrip():
    cx = _cx()
    assert sp.get_section(cx, "longevity", "intro") is None
    sp.upsert_section(cx, "longevity", "intro", "Hello world.", model="m1")
    assert sp.get_section(cx, "longevity", "intro") == "Hello world."

def test_upsert_accretes_sections_in_one_row():
    cx = _cx()
    sp.upsert_section(cx, "energy", "intro", "A.")
    sp.upsert_section(cx, "energy", "description", "B.")
    page = sp.get_page(cx, "energy")
    assert page["content"] == {"intro": "A.", "description": "B."}
    assert page["state"] == "draft"

def test_get_page_missing_returns_none():
    assert sp.get_page(_cx(), "nope") is None

def test_prompt_includes_compliance_and_no_disease_claim():
    system, user = sc.build_section_prompt("intro", {"name": "Longevity", "ingredients": []})
    assert "treat" in system.lower() and "cure" in system.lower() and "prevent" in system.lower()
    assert "supports" in system.lower() or "structure/function" in system.lower()

def test_prompt_grounds_in_product_name_and_ingredients():
    prod = {"name": "Longevity", "ingredients": [{"name": "Resveratrol", "dose": "200 mg"}, "Quercetin"]}
    system, user = sc.build_section_prompt("research", prod)
    assert "Longevity" in user
    assert "Resveratrol" in user and "200 mg" in user and "Quercetin" in user

def test_narrative_sections_are_exactly_three():
    assert sc.NARRATIVE_SECTIONS == ("intro", "description", "research")


# ---------------------------------------------------------------------------
# Task 3: SALES_PAGES_AI_COPY flag + page-data ai markers
# ---------------------------------------------------------------------------

import importlib
import os
import pytest


def _reload_app(monkeypatch, tmp_path, ai="true"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SALES_PAGES_ENABLED", "true")
    monkeypatch.setenv("SALES_PAGES_AI_COPY", ai)
    import app as appmod
    importlib.reload(appmod)
    return appmod


def test_page_data_marks_narrative_pending_when_no_draft(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    data = appmod.app.test_client().get(f"/begin/product-page-data/{slug}").get_json()
    nar = {s["id"]: s for s in data["sections"] if s["id"] in ("intro", "description", "research")}
    assert all(s.get("ai") == "pending" for s in nar.values())


def test_page_data_serves_cached_draft(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    from dashboard import sales_pages as sp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        sp.upsert_section(cx, slug, "intro", "Cached intro copy.")
    data = appmod.app.test_client().get(f"/begin/product-page-data/{slug}").get_json()
    intro = next(s for s in data["sections"] if s["id"] == "intro")
    assert intro["ai"] == "cached" and intro["body"] == "Cached intro copy."


def test_page_data_no_ai_field_when_flag_off(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path, ai="false")
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    data = appmod.app.test_client().get(f"/begin/product-page-data/{slug}").get_json()
    assert all("ai" not in s for s in data["sections"])


# ---------------------------------------------------------------------------
# Task 4: /begin/product-page-gen SSE endpoint
# ---------------------------------------------------------------------------

class _FakeStream:
    def __init__(self, toks): self._toks = toks
    def __enter__(self): return self
    def __exit__(self, *a): return False
    @property
    def text_stream(self):
        for t in self._toks: yield t

class _FakeMessages:
    def __init__(self, toks, boom=False): self._toks=toks; self.boom=boom; self.calls=0
    def stream(self, **kw):
        self.calls += 1
        if self.boom: raise RuntimeError("claude down")
        return _FakeStream(self._toks)

class _FakeCl:
    def __init__(self, toks, boom=False): self.messages=_FakeMessages(toks, boom)

def _frames(resp):
    return resp.get_data(as_text=True)

def test_gen_streams_and_persists(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    monkeypatch.setattr(appmod, "_product_card", lambda p: {"ingredients": [{"name": "Resveratrol", "dose": "200 mg"}]})
    monkeypatch.setattr(appmod, "_cl", _FakeCl(["Live ", "intro ", "copy."]))
    body = _frames(appmod.app.test_client().get(f"/begin/product-page-gen/{slug}/intro"))
    assert "Live " in body and '"done": true' in body
    import sqlite3
    from dashboard import sales_pages as sp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert sp.get_section(cx, slug, "intro") == "Live intro copy."

def test_gen_returns_cached_without_calling_claude(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    import sqlite3
    from dashboard import sales_pages as sp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        sp.upsert_section(cx, slug, "intro", "Cached copy.")
    fake = _FakeCl([], boom=True)
    monkeypatch.setattr(appmod, "_cl", fake)
    body = _frames(appmod.app.test_client().get(f"/begin/product-page-gen/{slug}/intro"))
    assert "Cached copy." in body and fake.messages.calls == 0

def test_gen_404_when_flag_off(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path, ai="false")
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    assert appmod.app.test_client().get(f"/begin/product-page-gen/{slug}/intro").status_code == 404

def test_gen_error_frame_on_claude_failure(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    monkeypatch.setattr(appmod, "_product_card", lambda p: {"ingredients": []})
    monkeypatch.setattr(appmod, "_cl", _FakeCl([], boom=True))
    body = _frames(appmod.app.test_client().get(f"/begin/product-page-gen/{slug}/intro"))
    assert '"error": true' in body


# ---------------------------------------------------------------------------
# Task 6: integration sanity + flag default
# ---------------------------------------------------------------------------

def test_full_phase2_file_and_flag_default(monkeypatch, tmp_path):
    # flag unset → disabled
    monkeypatch.delenv("SALES_PAGES_AI_COPY", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path)); monkeypatch.setenv("SALES_PAGES_ENABLED", "true")
    import importlib, app as appmod; importlib.reload(appmod)
    assert appmod._SALES_AI_COPY_ENABLED is False
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    assert appmod.app.test_client().get(f"/begin/product-page-gen/{slug}/intro").status_code == 404


# ---------------------------------------------------------------------------
# Task 4 (Finding 1): em-dash strip applied to streamed + persisted text
# ---------------------------------------------------------------------------

def test_gen_strips_em_dash_from_streamed_and_persisted(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    monkeypatch.setattr(appmod, "_product_card", lambda p: {"ingredients": []})
    # Model emits a token with an em dash
    monkeypatch.setattr(appmod, "_cl", _FakeCl(["Calm — focused."]))
    body = _frames(appmod.app.test_client().get(f"/begin/product-page-gen/{slug}/intro"))
    # Streamed SSE body must not contain an em dash
    assert "—" not in body, "Em dash found in streamed SSE body"
    # Persisted text must not contain an em dash
    import sqlite3
    from dashboard import sales_pages as sp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        persisted = sp.get_section(cx, slug, "intro")
    assert persisted is not None, "Nothing was persisted"
    assert "—" not in persisted, "Em dash found in persisted section text"
