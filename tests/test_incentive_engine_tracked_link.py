"""Tests for routing the personal-email product link through the app-owned
tracked redirect (dashboard/email_click_tokens.py + /r/<token>/<source>/<slug>).

Approach: `_process_one_user` does topic selection via a locally-imported
`pinecone_content_pool.candidate_topics_for_audience` (imported inside the
function body, not a module attribute of `incentive_engine`), plus calls into
`_send_email` / `_record_send`. Short-circuiting all of that end-to-end is
brittle (the import binds inside the function call, not at `ie.<name>`), so
per the task brief's own guidance we instead unit-test the extracted
`_tracked_product_url` helper directly, and separately assert it is what
`_process_one_user` uses when building its `product` dict.
"""
import sqlite3

import incentive_engine as ie
from dashboard import email_click_tokens as ect


def test_tracked_product_url_builds_tracked_redirect_with_no_pii(tmp_path, monkeypatch):
    db = str(tmp_path / "log.db")
    monkeypatch.setattr(ie, "LOG_DB", db, raising=False)
    monkeypatch.setattr(ie, "_public_base", lambda: "https://illtowell.com", raising=False)

    url = ie._tracked_product_url("a@b.com", "terrain-restore", "email")

    assert url.startswith("https://illtowell.com/r/")
    assert url.endswith("/email/terrain-restore")
    assert "a@b.com" not in url  # no PII in the link

    token = url.split("/r/")[1].split("/email/")[0]
    cx = sqlite3.connect(db)
    assert ect.email_for(cx, token) == "a@b.com"
    cx.close()


def test_tracked_product_url_reuses_same_token_for_same_email(tmp_path, monkeypatch):
    db = str(tmp_path / "log.db")
    monkeypatch.setattr(ie, "LOG_DB", db, raising=False)
    monkeypatch.setattr(ie, "_public_base", lambda: "https://illtowell.com", raising=False)

    url1 = ie._tracked_product_url("a@b.com", "terrain-restore", "email")
    url2 = ie._tracked_product_url("a@b.com", "terrain-restore", "email")

    assert url1 == url2


def test_process_one_user_uses_tracked_product_url(monkeypatch):
    """`_process_one_user` should build product['url'] via `_tracked_product_url`
    rather than the hardcoded truly.vip shortlink."""
    calls = {}

    def fake_tracked_product_url(email, slug, source="email"):
        calls["args"] = (email, slug, source)
        return "https://illtowell.com/r/TOKEN/email/terrain-restore"

    monkeypatch.setattr(ie, "_tracked_product_url", fake_tracked_product_url, raising=False)
    monkeypatch.setattr(
        ie, "_load_user_state", lambda user_id: {}, raising=False
    )
    monkeypatch.setattr(ie, "should_send_today", lambda state, paused=False: True, raising=False)

    import pinecone_content_pool
    monkeypatch.setattr(
        pinecone_content_pool, "candidate_topics_for_audience",
        lambda audience: ["leaky-gut"], raising=False,
    )
    monkeypatch.setattr(
        ie, "select_topic_for_user", lambda state, topics, audience: "leaky-gut",
        raising=False,
    )
    monkeypatch.setattr(
        pinecone_content_pool, "fetch_source_text_for_topic",
        lambda topic, audience: "source text", raising=False,
    )

    captured = {}

    def fake_generate(user, topic, topic_source_text, product, is_beta, audience, **kw):
        captured["product"] = product
        return {"subject": "s", "body": "b"}

    monkeypatch.setattr(ie, "generate_personal_email", fake_generate, raising=False)
    monkeypatch.setattr(ie, "_send_email", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(ie, "_record_send", lambda *a, **k: None, raising=False)

    result = ie._process_one_user(
        {"id": 1, "email": "a@b.com", "name": "A"},
        {"beta_shared_code": "BETA5"},
        audience="client", is_beta=True,
    )

    assert result == "sent"
    assert calls["args"] == ("a@b.com", "terrain-restore", "email")
    assert captured["product"]["url"] == "https://illtowell.com/r/TOKEN/email/terrain-restore"
