"""Mentor-pages unit tests: store, seed build, and server render.

Pure-module tests: they do not import app.py, so they run under bare pytest
without booting Flask/gevent or sending any email.
"""
import sqlite3
import sys
from pathlib import Path

import pytest


def _mod(name):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return __import__(f"dashboard.{name}", fromlist=[name])
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"module not importable: {e}")


def _cx(tmp_path):
    mp = _mod("mentor_pages")
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    mp.init_table(cx)
    return cx


# ── store ────────────────────────────────────────────────────────────────────

def test_store_roundtrip_and_state(tmp_path):
    mp = _mod("mentor_pages"); cx = _cx(tmp_path)
    mp.upsert_section(cx, "jane-doe", "life_and_work", "A scientist.")
    mp.set_name(cx, "jane-doe", "Jane Doe")
    mp.set_field(cx, "jane-doe", "Biophysics")
    mp.set_lifespan(cx, "jane-doe", "1900-1980")
    mp.set_vital_status(cx, "jane-doe", "deceased")
    mp.set_lineage(cx, "jane-doe", ["Doe", "Roe"])
    mp.set_sources(cx, "jane-doe", ["Some Book (1970)"])
    mp.set_seo(cx, "jane-doe", {"title": "Jane Doe", "meta_description": "d"})
    p = mp.get_page(cx, "jane-doe")
    assert p["name"] == "Jane Doe"
    assert p["field"] == "Biophysics" and p["lifespan"] == "1900-1980"
    assert p["vital_status"] == "deceased"
    assert p["content"]["life_and_work"] == "A scientist."
    assert p["lineage"] == ["Doe", "Roe"]
    assert p["sources"] == ["Some Book (1970)"]
    assert p["seo"]["title"] == "Jane Doe"
    assert p["state"] == "draft"
    mp.set_state(cx, "jane-doe", "approved", by="glen")
    p = mp.get_page(cx, "jane-doe")
    assert p["state"] == "approved" and p["approved_by"] == "glen"
    # no row_factory leak after a getter
    assert isinstance(cx.execute("SELECT 1").fetchone(), tuple)


def test_list_public_only_approved_and_sorted(tmp_path):
    mp = _mod("mentor_pages"); cx = _cx(tmp_path)
    for slug, name in (("b-one", "Zeta"), ("a-two", "Alpha")):
        mp.upsert_section(cx, slug, "life_and_work", "x")
        mp.set_name(cx, slug, name)
    mp.set_state(cx, "b-one", "approved")
    mp.set_state(cx, "a-two", "approved")
    mp.upsert_section(cx, "draft-one", "life_and_work", "x")  # stays draft
    pub = mp.list_public(cx)
    slugs = [p["slug"] for p in pub]
    assert "draft-one" not in slugs
    assert slugs == ["a-two", "b-one"]  # alphabetical by name (Alpha, Zeta)


def test_request_notify_at_most_once(tmp_path):
    mp = _mod("mentor_pages"); cx = _cx(tmp_path)
    mp.record_request(cx, "jane-doe", "A@x.com")
    mp.record_request(cx, "jane-doe", "a@x.com")  # same, normalized -> one row
    assert len(mp.requesters_to_email(cx, "jane-doe")) == 1
    sent = []
    mp.notify_on_approve(cx, "jane-doe", "Jane Doe", "https://h.com",
                         send=lambda to, subj, body: sent.append((to, subj)))
    assert sent and sent[0][0] == "a@x.com"
    # second call finds nobody un-emailed
    mp.notify_on_approve(cx, "jane-doe", "Jane Doe", "https://h.com",
                         send=lambda *a: sent.append(a))
    assert len(sent) == 1


def test_notify_one_bad_send_does_not_stop_others(tmp_path):
    mp = _mod("mentor_pages"); cx = _cx(tmp_path)
    mp.record_request(cx, "s", "bad@x.com")
    mp.record_request(cx, "s", "good@x.com")
    ok = []

    def send(to, subj, body):
        if to == "bad@x.com":
            raise RuntimeError("smtp down")
        ok.append(to)

    mp.notify_on_approve(cx, "s", "S", "https://h.com", send=send)
    assert ok == ["good@x.com"]
    # the good one is marked emailed; the bad one remains for a retry
    remaining = [r["email"] for r in mp.requesters_to_email(cx, "s")]
    assert remaining == ["bad@x.com"]


# ── seed build (no LLM, no network) ──────────────────────────────────────────

def test_seed_build_produces_full_draft(tmp_path):
    mc = _mod("mentor_copy"); mp = _mod("mentor_pages"); cx = _cx(tmp_path)
    res = mc.build_page(cx, "harold-saxton-burr")
    assert res.get("source") == "seed" and res.get("state") == "draft"
    p = mp.get_page(cx, "harold-saxton-burr")
    for sec in mc.NARRATIVE_SECTIONS:
        assert p["content"].get(sec), f"missing seeded section {sec}"
    assert p["field"] and p["lifespan"] == "1889-1973"
    assert p["vital_status"] == "deceased"
    assert p["lineage"][0] == "Burr"
    assert any("Blueprint" in s for s in p["sources"])
    # the whole seeded page honors Glen's no-em-dash rule (body, field, title, sources)
    joined = " ".join([
        *p["content"].values(), p["field"], p["lifespan"],
        p["seo"].get("title", ""), p["seo"].get("meta_description", ""),
        *p["sources"], *p["lineage"],
    ])
    assert "—" not in joined


def test_grounded_build_without_source_stays_pending(tmp_path):
    """Unknown mentor + no retriever/client must NOT invent a biography."""
    mc = _mod("mentor_copy"); mp = _mod("mentor_pages"); cx = _cx(tmp_path)
    res = mc.build_page(cx, "unknown-person", "Unknown Person")
    assert res.get("state") == "pending"
    p = mp.get_page(cx, "unknown-person")
    assert not (p["content"] or {})  # nothing written


def test_grounded_build_with_fake_client_and_retriever(tmp_path):
    mc = _mod("mentor_copy"); mp = _mod("mentor_pages"); cx = _cx(tmp_path)

    class _Block:
        type = "text"
        text = "A grounded paragraph."

    class _Msg:
        content = [_Block()]

    class _Messages:
        def create(self, **kw):
            # ensure the source made it into the prompt
            assert "SOURCE-CONTEXT" in kw["messages"][0]["content"]
            return _Msg()

    class _Client:
        messages = _Messages()

    res = mc.build_page(cx, "grounded-one", "Grounded One",
                        client=_Client(), retriever=lambda q: "SOURCE-CONTEXT here")
    assert res.get("state") == "draft" and res.get("sections_built") == 4
    p = mp.get_page(cx, "grounded-one")
    assert p["content"]["life_and_work"] == "A grounded paragraph."


# ── render ───────────────────────────────────────────────────────────────────

def test_is_public_only_approved():
    mr = _mod("mentor_render")
    assert mr.is_public({"state": "approved"})
    assert not mr.is_public({"state": "draft"})
    assert not mr.is_public(None)


def test_render_page_html_contains_key_parts():
    mr = _mod("mentor_render")
    page = {
        "slug": "harold-saxton-burr", "name": "Harold Saxton Burr, PhD",
        "state": "approved", "field": "Bioelectrodynamics", "lifespan": "1889-1973",
        "lineage": ["Burr", "Becker", "Levin"],
        "sources": ["Blueprint for Immortality (1972)"],
        "seo": {"title": "Burr", "meta_description": "L-fields."},
        "content": {"life_and_work": "Para one.\n\nPara two.",
                    "key_contribution": "L-fields.",
                    "lineage": "Chain.", "why_it_matters": "It matters."},
    }
    html = mr.render_page_html(page, base_url="https://h.com")
    assert "<h1>Harold Saxton Burr, PhD</h1>" in html
    assert "Bioelectrodynamics" in html and "1889-1973" in html
    assert "Intellectual lineage" in html and "Becker" in html
    assert "Blueprint for Immortality" in html
    assert '"@type": "Person"' in html
    assert "<p>Para one.</p>" in html and "<p>Para two.</p>" in html  # paragraph split
    assert 'href="/begin"' in html  # CTA


def test_render_pending_is_noindex():
    mr = _mod("mentor_render")
    html = mr.render_pending_html("jane-doe", "Jane Doe")
    assert 'name="robots" content="noindex"' in html
    assert 'action="/mentors/jane-doe/request"' in html


def test_index_and_sitemap():
    mr = _mod("mentor_render")
    rows = [{"slug": "harold-saxton-burr", "name": "Harold Saxton Burr",
             "field": "Bioelectrodynamics", "lifespan": "1889-1973",
             "updated_at": "2026-07-14T00:00:00+00:00"}]
    idx = mr.render_index_html(rows)
    assert 'href="/mentors/harold-saxton-burr"' in idx
    sm = mr.render_sitemap_xml(rows, "https://h.com")
    assert "<loc>https://h.com/mentors/harold-saxton-burr</loc>" in sm
    assert "<lastmod>2026-07-14</lastmod>" in sm
