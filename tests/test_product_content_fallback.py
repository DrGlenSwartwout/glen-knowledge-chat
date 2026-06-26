"""Regression tests for the /begin/learn research-page content layer.

Root cause of the Longevity bug (2026-06-19): products whose panel is missing from
Pinecone `specific-formulations` (truncated scrape) yield empty page copy. The
learn_more generator then sent an EMPTY prompt to the model, which replied with a
refusal ("...PAGE COPY and RESEARCH SOURCES sections are empty"), and that refusal
was cached and served on the public page.

Fix: (1) ground generation in the manual data already in products.json (description
+ ingredients) as a fallback when Pinecone returns nothing; (2) never call the model
(and never cache a refusal) when there is no grounding material at all.
"""
import sqlite3
from dashboard import product_content as pc


class _FakeMessages:
    def __init__(self):
        self.calls = 0
        self.last_user = None

    def create(self, **kw):
        self.calls += 1
        self.last_user = kw["messages"][0]["content"]

        class _M:
            content = [type("T", (), {"text": "REAL grounded copy.\n\n## Sources\n"})()]
        return _M()


class _FakeCl:
    def __init__(self):
        self.messages = _FakeMessages()


# ── _page_text_from_product ──────────────────────────────────────────────────
def test_page_text_from_product_builds_from_manual_data():
    prod = {"name": "Longevity", "description": "Supports mitochondrial function.",
            "ingredients": [{"name": "NMN", "dose": "75 mg"}, "Quercetin"],
            "url": "http://example/longevity"}
    page = pc._page_text_from_product(prod)
    assert page is not None
    assert "Supports mitochondrial function." in page["text"]
    assert "NMN" in page["text"] and "75 mg" in page["text"]
    assert "Quercetin" in page["text"]
    assert page["url"] == "http://example/longevity"


def test_page_text_from_product_none_when_no_data():
    assert pc._page_text_from_product({"name": "X"}) is None


# ── _generate_learn_more guard ───────────────────────────────────────────────
def test_learn_more_blank_when_no_page_and_no_sources(monkeypatch):
    cl = _FakeCl()
    monkeypatch.setattr(pc, "_clients", lambda: (None, cl, None))
    out = pc._generate_learn_more({"name": "Longevity"}, None, [])
    assert out == {"markdown": ""}
    assert cl.messages.calls == 0  # model must NOT be asked -> no refusal possible


# ── get_or_generate end-to-end ───────────────────────────────────────────────
def test_learn_more_falls_back_to_products_json(monkeypatch, tmp_path):
    monkeypatch.setattr(pc, "LOG_DB", tmp_path / "chat_log.db")
    monkeypatch.setattr(pc, "_page_text", lambda p: None)          # Pinecone empty
    monkeypatch.setattr(pc, "_research_sources", lambda n, k=8, ingredients=None: [])
    cl = _FakeCl()
    monkeypatch.setattr(pc, "_clients", lambda: (None, cl, None))
    prod = {"slug": "longevity", "name": "Longevity",
            "description": "Supports mitochondrial function.",
            "ingredients": [{"name": "NMN", "dose": "75 mg"}], "url": "http://x"}
    res = pc.get_or_generate(prod, "learn_more", force=True)
    assert cl.messages.calls == 1                                  # grounded -> model called
    assert res["content"]["markdown"]                              # non-empty copy
    assert "NMN" in cl.messages.last_user                          # synthesized copy reached model
    assert "Supports mitochondrial function." in cl.messages.last_user


def test_learn_more_no_data_caches_no_refusal(monkeypatch, tmp_path):
    monkeypatch.setattr(pc, "LOG_DB", tmp_path / "chat_log.db")
    monkeypatch.setattr(pc, "_page_text", lambda p: None)
    monkeypatch.setattr(pc, "_research_sources", lambda n, k=8, ingredients=None: [])
    cl = _FakeCl()
    monkeypatch.setattr(pc, "_clients", lambda: (None, cl, None))
    prod = {"slug": "empty", "name": "Empty"}
    res = pc.get_or_generate(prod, "learn_more", force=True)
    assert res["content"]["markdown"] == ""
    assert cl.messages.calls == 0


def test_purge_refusal_cache_removes_only_refusals():
    cx = sqlite3.connect(":memory:")
    pc.init_product_content_table(cx)
    cx.execute("INSERT INTO generated_product_content (product_slug,content_type,content_json,generated_at) "
               "VALUES (?,?,?,?)", ("longevity", "learn_more",
               '{"markdown":"I am unable to proceed as written. The sections are empty."}', "t"))
    cx.execute("INSERT INTO generated_product_content (product_slug,content_type,content_json,generated_at) "
               "VALUES (?,?,?,?)", ("energy", "learn_more",
               '{"markdown":"Real grounded research copy about energy."}', "t"))
    n = pc.purge_refusal_cache(cx)
    assert n == 1
    rows = [r[0] for r in cx.execute("SELECT product_slug FROM generated_product_content").fetchall()]
    assert rows == ["energy"]


def test_card_falls_back_to_products_json(monkeypatch, tmp_path):
    monkeypatch.setattr(pc, "LOG_DB", tmp_path / "chat_log.db")
    monkeypatch.setattr(pc, "_page_text", lambda p: None)
    cl = _FakeCl()
    # card model returns valid JSON
    def _create(**kw):
        cl.messages.calls += 1
        cl.messages.last_user = kw["messages"][0]["content"]

        class _M:
            content = [type("T", (), {"text": '{"description":"d","ingredients":["NMN 75 mg"],"benefits":["b"]}'})()]
        return _M()
    cl.messages.create = _create
    monkeypatch.setattr(pc, "_clients", lambda: (None, cl, None))
    prod = {"slug": "longevity", "name": "Longevity", "description": "Supports mitochondria.",
            "ingredients": [{"name": "NMN", "dose": "75 mg"}], "url": "http://x"}
    res = pc.get_or_generate(prod, "card", force=True)
    assert cl.messages.calls == 1
    assert "NMN" in cl.messages.last_user
