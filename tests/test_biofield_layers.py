# tests/test_biofield_layers.py
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _load(mod):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module(mod)
    except Exception as e:
        pytest.skip(f"{mod} not importable: {e}")


def _LAYERS():
    return [
        {"n": 1, "title": "Mineral Mobilization", "summary": "Surface mineral stress.",
         "patterns": ["A1"], "remedy": {"name": "Nous Energy", "slug": "nous-energy", "meaning": "m"}},
        {"n": 2, "title": "Terrain Balance", "summary": "Deeper biofilm terrain.",
         "patterns": ["B2"], "remedy": {"name": "Cistus", "slug": "cistus-syntropy-immunitea", "meaning": "m2"}},
    ]


def test_layers_roundtrip(tmp_path):
    br = _load("dashboard.biofield_reveals")
    db = str(tmp_path / "r.db")
    with sqlite3.connect(db) as cx:
        br.init_table(cx); br.init_table(cx)  # idempotent ALTER
        rid, is_new = br.upsert(cx, "a@b.com", "2026-06-20", {"body": "x"}, [], "s", layers=_LAYERS())
        row = br.get(cx, rid)
    assert is_new and len(row["layers"]) == 2 and row["layers"][0]["title"] == "Mineral Mobilization"


def test_set_layers_and_reedit_when_approved(tmp_path):
    br = _load("dashboard.biofield_reveals")
    db = str(tmp_path / "r.db")
    with sqlite3.connect(db) as cx:
        br.init_table(cx)
        rid, _ = br.upsert(cx, "a@b.com", "2026-06-20", {"body": "x"}, [], "s", layers=_LAYERS())
        br.approve_first(cx, rid, "glen")  # now first_approved=1
        new = _LAYERS(); new[0]["title"] = "Renamed Layer"
        br.set_layers(cx, rid, new)
        br.set_interpretation(cx, rid, {"body": "edited after approval"})
        row = br.get(cx, rid)
    assert row["first_approved"] is True
    assert row["layers"][0]["title"] == "Renamed Layer"        # set_layers works post-approval
    assert row["interpretation"]["body"] == "edited after approval"  # set_interpretation too


def test_list_approved(tmp_path):
    br = _load("dashboard.biofield_reveals")
    db = str(tmp_path / "r.db")
    with sqlite3.connect(db) as cx:
        br.init_table(cx)
        r1, _ = br.upsert(cx, "p@x.com", "2026-06-20", {}, [], "s")
        r2, _ = br.upsert(cx, "a@x.com", "2026-06-20", {}, [], "s")
        br.approve_first(cx, r2, "glen")
        pend = [r["id"] for r in br.list_pending(cx)]
        appr = [r["id"] for r in br.list_approved(cx)]
    assert r1 in pend and r2 not in pend and r2 in appr and r1 not in appr


def _app_db(monkeypatch, tmp_path):
    app_module = _load("app")
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    from dashboard import biofield_reveals as br, biofield_meanings as bm
    with sqlite3.connect(db) as cx:
        br.init_table(cx); bm.init_table(cx)
        cx.execute("CREATE TABLE IF NOT EXISTS auth_tokens (token_hash TEXT, email TEXT, purpose TEXT, created_at TEXT, expires_at TEXT, consumed_at TEXT)")
        cx.commit()
    return app_module, db


def _push(app_module, body, key):
    return app_module.app.test_client().post("/api/e4l/reveal-draft", headers={"X-Console-Key": key}, json=body)


def _key(app_module):
    import os
    return os.environ.get("CRON_SECRET") or app_module.CONSOLE_SECRET or ""


def test_ingest_stores_layers_and_derives_remedies(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    key = _key(app_module)
    if not key: pytest.skip("no secret")
    prods = app_module._PRODUCTS.get("products") or {}
    real = next(iter(prods), None)
    if not real: pytest.skip("no catalog")
    rname = prods[real]["name"]
    layers = [{"n": 1, "title": "Layer One", "summary": "s1", "patterns": ["A"],
               "remedy": {"name": rname, "slug": real, "meaning": "pushed"}},
              {"n": 2, "title": "Layer Two", "summary": "s2", "patterns": ["B"],
               "remedy": {"name": "Totally Made Up", "slug": "nope-xyz", "meaning": "ghost"}}]
    r = _push(app_module, {"email": "a@b.com", "scan_date": "2026-06-20",
                           "interpretation": {"body": "x"}, "layers": layers}, key)
    assert r.get_json().get("ok") is True
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        row = br.list_pending(cx)[0]
    titles = [L["title"] for L in row["layers"]]
    assert titles == ["Layer One", "Layer Two"]                 # both layers kept (titles always)
    assert row["layers"][0]["remedy"]["slug"] == real           # catalog remedy survives
    assert row["layers"][1]["remedy"] is None                   # non-catalog remedy dropped from its layer
    assert "Totally Made Up" in row["dropped"]
    assert [rr["slug"] for rr in row["remedies"]] == [real]      # derived flat remedies = surviving layer remedies


def test_ingest_remedies_only_wraps_into_layers(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    key = _key(app_module)
    if not key: pytest.skip("no secret")
    prods = app_module._PRODUCTS.get("products") or {}
    real = next(iter(prods), None)
    if not real: pytest.skip("no catalog")
    _push(app_module, {"email": "c@b.com", "scan_date": "2026-06-20", "interpretation": {"body": "x"},
                       "remedies": [{"name": prods[real]["name"], "slug": real, "meaning": "m"}]}, key)
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        row = br.list_pending(cx)[0]
    assert len(row["layers"]) == 1 and row["layers"][0]["title"] == ""   # titleless wrap
    assert row["layers"][0]["remedy"]["slug"] == real
    assert [rr["slug"] for rr in row["remedies"]] == [real]


def _seed_approved_layers(app_module, db, email="t@x.com"):
    import secrets as _s
    from datetime import datetime, timezone, timedelta
    from dashboard import biofield_reveals as br
    token = "tk_" + _s.token_urlsafe(8)
    th = app_module._hash_token(token)
    with sqlite3.connect(db) as cx:
        rid, _ = br.upsert(cx, email, "2026-06-20", {"greeting": "Hi", "body": "b"},
                           [{"name": "Top", "slug": "rx-aaa", "meaning": "m"},
                            {"name": "Deep", "slug": "rx-bbb", "meaning": "m2"}], "s",
                           layers=[{"n": 1, "title": "Surface", "summary": "s1", "patterns": [],
                                    "remedy": {"name": "Top", "slug": "rx-aaa", "meaning": "m"}},
                                   {"n": 2, "title": "Root", "summary": "s2", "patterns": [],
                                    "remedy": {"name": "Deep", "slug": "rx-bbb", "meaning": "m2"}}])
        br.set_token(cx, rid, th); br.approve_first(cx, rid, "glen")
        cx.execute("INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) VALUES (?,?,?,?,?)",
                   (th, email, "biofield_reveal", datetime.now(timezone.utc).isoformat(),
                    (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()))
        cx.commit()
    return token


def _reveal_payload(app_module, token):
    import re, json as _j
    html = app_module.app.test_client().get(f"/begin/biofield/{token}").get_data(as_text=True)
    m = re.search(r"window.__REVEAL__ = (\{.*?\});", html)
    return _j.loads(m.group(1).replace("\\u003c", "<").replace("\\u003e", ">").replace("\\u0026", "&")) if m else None


def test_payload_titles_always_remedy_gated_nonpaid(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "is_member", lambda session_id="", email="": True)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: None)  # not paid
    token = _seed_approved_layers(app_module, db)  # approved but member has NOT claimed free unlock
    d = _reveal_payload(app_module, token)
    assert d["paid"] is False
    titles = [L["title"] for L in d["layers"]]
    assert titles == ["Surface", "Root"]                 # titles always shown
    assert all(L["summary"] for L in d["layers"])        # summaries always shown
    assert all(L["remedy"] is None and L["remedy_blurred"] for L in d["layers"])  # no remedy visible yet
    blob = __import__("json").dumps(d)
    assert "rx-aaa" not in blob and "rx-bbb" not in blob   # anti-bypass: withheld remedy slugs never emitted


def test_payload_paid_shows_all_layer_remedies(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "is_member", lambda session_id="", email="": True)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: {"ok": True})  # paid
    token = _seed_approved_layers(app_module, db)
    d = _reveal_payload(app_module, token)
    assert d["paid"] is True
    assert [L["title"] for L in d["layers"]] == ["Surface", "Root"]
    assert d["layers"][0]["remedy"]["slug"] == "rx-aaa" and d["layers"][1]["remedy"]["slug"] == "rx-bbb"
    assert all(not L["remedy_blurred"] for L in d["layers"])


def test_member_page_ships_layer_render(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "is_member", lambda session_id="", email="": True)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: {"ok": True})
    token = _seed_approved_layers(app_module, db)
    html = app_module.app.test_client().get(f"/begin/biofield/{token}").get_data(as_text=True)
    assert "renderLayers" in html and "layer-title" in html


def test_edit_action_layers_branch(monkeypatch, tmp_path):
    """_exec_edit with layers= strips remember, promotes meanings, re-derives flat remedies.
    Must also work on already-approved reveals (guards dropped in Task 1)."""
    app_module, db = _app_db(monkeypatch, tmp_path)
    from dashboard import biofield_reveals as br, biofield_meanings as bm, biofield_reveal_actions as bra
    promoted = {}

    def fake_upsert(cx, slug, meaning, by, source):
        promoted[slug] = meaning

    monkeypatch.setattr(bm, "upsert", fake_upsert)

    with sqlite3.connect(db) as cx:
        br.init_table(cx); bm.init_table(cx)
        rid, _ = br.upsert(cx, "edit@x.com", "2026-06-20", {"body": "orig"}, [], "s")
        br.approve_first(cx, rid, "glen")  # approved - edit must still work

        layers_in = [
            {"n": 1, "title": "Alpha", "summary": "s1", "patterns": [],
             "remedy": {"name": "Prod A", "slug": "prod-a", "meaning": "helps a", "remember": True}},
            {"n": 2, "title": "Beta", "summary": "s2", "patterns": [],
             "remedy": {"name": "Ghost", "slug": "", "meaning": "no slug", "remember": True}},
        ]
        ctx = {"cx": cx, "actor": None}
        result = bra._exec_edit({"id": rid, "layers": layers_in}, ctx)
        row = br.get(cx, rid)

    assert result == {"ok": True}
    # remember flag stripped from stored layers
    for layer in row["layers"]:
        rem = layer.get("remedy") or {}
        assert "remember" not in rem
    # meaning promoted for slugged remedy
    assert promoted.get("prod-a") == "helps a"
    # slug-less remedy yielded no layer remedy (None or no slug in derived)
    derived_slugs = [r["slug"] for r in row["remedies"] if r.get("slug")]
    assert derived_slugs == ["prod-a"]
    # both layers stored (title preserved)
    titles = [L["title"] for L in row["layers"]]
    assert titles == ["Alpha", "Beta"]


def test_console_page_ships_layer_editor(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "CONSOLE_SECRET", "sek", raising=False)
    html = app_module.app.test_client().get("/console/biofield-reveals?key=sek").get_data(as_text=True)
    assert "layer-title-field" in html and "Approved" in html and "collectLayers" in html


def test_nonpaid_layered_page_has_free_reveal_and_trial_cta(monkeypatch, tmp_path):
    """Regression for Critical finding: layered non-paid render must surface the
    CTA-driving payload data for free-reveal and $1 trial so the live-money path
    is reachable. Asserts on window.__REVEAL__ (runtime payload) not static HTML
    markers, which appear in every response regardless of state."""
    app_module, db = _app_db(monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "is_member", lambda session_id="", email="": True)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: None)  # not paid
    monkeypatch.setattr(app_module, "BIOFIELD_TRIAL_ENABLED", True, raising=False)
    token = _seed_approved_layers(app_module, db)
    d = _reveal_payload(app_module, token)
    assert d is not None, "window.__REVEAL__ payload must be present in the page"
    assert d["paid"] is False, "non-paid user must have paid=False"
    assert d["free_available"] is True, "free_available must be True (drives free-reveal CTA button)"
    assert d["trial_enabled"] is True, "trial_enabled must be True (drives $1 unlock CTA)"
    assert d["layers"] and len(d["layers"]) >= 1, "layered reveal must have at least one layer"
    assert all(L["remedy"] is None and L["remedy_blurred"] for L in d["layers"]), \
        "non-paid not-yet-unlocked reveal must gate all layer remedies"


def test_console_endpoint_returns_approved(monkeypatch, tmp_path):
    """GET /api/console/biofield-reveals must return both drafts and approved keys."""
    app_module, db = _app_db(monkeypatch, tmp_path)
    key = app_module.CONSOLE_SECRET or ""
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        br.init_table(cx)
        r1, _ = br.upsert(cx, "d@x.com", "2026-06-20", {}, [], "s")  # pending
        r2, _ = br.upsert(cx, "a@x.com", "2026-06-20", {}, [], "s")
        br.approve_first(cx, r2, "glen")                               # approved

    client = app_module.app.test_client()
    resp = client.get("/api/console/biofield-reveals", headers={"X-Console-Key": key})
    data = resp.get_json()
    assert "drafts" in data and "approved" in data
    draft_ids = [r["id"] for r in data["drafts"]]
    appr_ids = [r["id"] for r in data["approved"]]
    assert r1 in draft_ids and r2 not in draft_ids
    assert r2 in appr_ids and r1 not in appr_ids
