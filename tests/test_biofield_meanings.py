# tests/test_biofield_meanings.py
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


class _FakeClient:
    """Mimics anthropic client.messages.create(...).content[0].text."""
    def __init__(self, text="Supports the body's natural detox pathways.", raises=False):
        self._t, self._raises = text, raises
        outer = self
        class _M:
            def create(self, **kw):
                if outer._raises:
                    raise RuntimeError("llm down")
                return type("R", (), {"content": [type("C", (), {"text": outer._t})()]})()
        self.messages = _M()


def _db(tmp_path):
    bm = _load("dashboard.biofield_meanings")
    db = str(tmp_path / "m.db")
    with sqlite3.connect(db) as cx:
        bm.init_table(cx)
    return bm, db


def test_upsert_get_map_and_all(tmp_path):
    bm, db = _db(tmp_path)
    with sqlite3.connect(db) as cx:
        bm.upsert(cx, "nous-energy", "Guides healing and mobilizes metals.", "glen", "glen")
        assert bm.get_map(cx) == {"nous-energy": "Guides healing and mobilizes metals."}
        allrows = bm.get_all(cx)
    assert len(allrows) == 1 and allrows[0]["slug"] == "nous-energy" and allrows[0]["source"] == "glen"


def test_upsert_updates_single_row(tmp_path):
    bm, db = _db(tmp_path)
    with sqlite3.connect(db) as cx:
        bm.upsert(cx, "cistus", "First.", "ai", "ai")
        bm.upsert(cx, "cistus", "Second.", "glen", "glen")
        rows = bm.get_all(cx)
    assert len(rows) == 1 and rows[0]["meaning"] == "Second." and rows[0]["source"] == "glen"


def test_delete(tmp_path):
    bm, db = _db(tmp_path)
    with sqlite3.connect(db) as cx:
        bm.upsert(cx, "x", "m", "glen", "glen")
        bm.delete(cx, "x")
        assert bm.get_map(cx) == {}


def test_get_map_omits_empty(tmp_path):
    bm, db = _db(tmp_path)
    with sqlite3.connect(db) as cx:
        bm.upsert(cx, "x", "", "ai", "ai")
        assert bm.get_map(cx) == {}


def test_propose_meaning_builds_text():
    bm = _load("dashboard.biofield_meanings")
    product = {"name": "Nous Energy", "ingredients": [{"name": "spirit minerals"}],
               "benefits": ["mental energy"], "description": "Guides healing."}
    out = bm.propose_meaning(product, _FakeClient("Guides healing processes and mobilizes heavy metals."))
    assert out == "Guides healing processes and mobilizes heavy metals."


def test_propose_meaning_never_raises():
    bm = _load("dashboard.biofield_meanings")
    assert bm.propose_meaning({"name": "X"}, _FakeClient(raises=True)) == ""
    assert bm.propose_meaning({"name": "X"}, None) == ""


def test_reveal_dropped_column(tmp_path):
    br = _load("dashboard.biofield_reveals")
    db = str(tmp_path / "r.db")
    with sqlite3.connect(db) as cx:
        br.init_table(cx)
        br.init_table(cx)  # idempotent (no error on the ALTER second time)
        rid, is_new = br.upsert(cx, "a@b.com", "2026-06-20", {"body": "x"},
                                [{"name": "Top", "slug": "top"}], "s")
        assert is_new
        br.set_dropped(cx, rid, ["Mineral Binder", "Made Up"])
        row = br.get(cx, rid)
    assert row["dropped"] == ["Mineral Binder", "Made Up"]


def _app_db(monkeypatch, tmp_path):
    app_module = _load("app")
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    from dashboard import biofield_reveals as br, biofield_meanings as bm
    with sqlite3.connect(db) as cx:
        br.init_table(cx)
        bm.init_table(cx)
        cx.execute("CREATE TABLE IF NOT EXISTS auth_tokens (token_hash TEXT, email TEXT, purpose TEXT, created_at TEXT, expires_at TEXT, consumed_at TEXT)")
        cx.commit()
    return app_module, db


def _push(app_module, remedies, key):
    return app_module.app.test_client().post(
        "/api/e4l/reveal-draft",
        headers={"X-Console-Key": key},
        json={"email": "a@b.com", "scan_date": "2026-06-20",
              "interpretation": {"body": "x"}, "remedies": remedies})


def test_ingest_drops_non_catalog_and_records(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    key = app_module.os.environ.get("CRON_SECRET") or app_module.CONSOLE_SECRET or ""
    if not key:
        pytest.skip("no CRON_SECRET/CONSOLE_SECRET in env")
    # 'top' is a placeholder slug that won't resolve; force a known real slug via monkeypatch.
    real = next(iter((app_module._PRODUCTS.get("products") or {}).keys()), None)
    if not real:
        pytest.skip("no catalog products")
    r = _push(app_module, [{"name": app_module._PRODUCTS["products"][real]["name"], "slug": real, "meaning": "pushed"},
                           {"name": "Totally Made Up", "slug": "nope-xyz", "meaning": "ghost"}], key)
    assert r.get_json().get("ok") is True
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        rows = br.list_pending(cx)
    row = rows[0]
    slugs = [x.get("slug") for x in row["remedies"]]
    assert real in slugs and "nope-xyz" not in slugs
    assert "Totally Made Up" in row["dropped"]


def test_ingest_applies_canonical_override(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    key = app_module.os.environ.get("CRON_SECRET") or app_module.CONSOLE_SECRET or ""
    if not key:
        pytest.skip("no CRON_SECRET/CONSOLE_SECRET in env")
    real = next(iter((app_module._PRODUCTS.get("products") or {}).keys()), None)
    if not real:
        pytest.skip("no catalog products")
    from dashboard import biofield_meanings as bm
    with sqlite3.connect(db) as cx:
        bm.upsert(cx, real, "CANONICAL MEANING", "glen", "glen")
    _push(app_module, [{"name": app_module._PRODUCTS["products"][real]["name"], "slug": real, "meaning": "pushed text"}], key)
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        row = br.list_pending(cx)[0]
    assert row["remedies"][0]["meaning"] == "CANONICAL MEANING"


def _seed_reveal(br, db):
    with sqlite3.connect(db) as cx:
        rid, _ = br.upsert(cx, "a@b.com", "2026-06-20", {"body": "x"},
                           [{"name": "Top", "slug": "top", "meaning": "old"}], "s")
    return rid


def test_edit_remember_default_promotes(tmp_path):
    bra = _load("dashboard.biofield_reveal_actions")
    br = _load("dashboard.biofield_reveals")
    bm = _load("dashboard.biofield_meanings")
    db = str(tmp_path / "r.db")
    with sqlite3.connect(db) as cx:
        br.init_table(cx); bm.init_table(cx)
    rid = _seed_reveal(br, db)
    with sqlite3.connect(db) as cx:
        bra._exec_edit({"id": rid, "remedies": [{"name": "Top", "slug": "top", "meaning": "NEW MEANING"}]},
                       {"cx": cx, "actor": None})
        canon = bm.get_map(cx)
        row = br.get(cx, rid)
    assert canon.get("top") == "NEW MEANING"           # promoted (remember defaults on)
    assert row["remedies"][0]["meaning"] == "NEW MEANING"  # reveal row updated
    assert "remember" not in row["remedies"][0]         # remember stripped from stored remedy


def test_edit_remember_false_skips_canonical(tmp_path):
    bra = _load("dashboard.biofield_reveal_actions")
    br = _load("dashboard.biofield_reveals")
    bm = _load("dashboard.biofield_meanings")
    db = str(tmp_path / "r.db")
    with sqlite3.connect(db) as cx:
        br.init_table(cx); bm.init_table(cx)
    rid = _seed_reveal(br, db)
    with sqlite3.connect(db) as cx:
        bra._exec_edit({"id": rid, "remedies": [{"name": "Top", "slug": "top", "meaning": "ONE TIME", "remember": False}]},
                       {"cx": cx, "actor": None})
        canon = bm.get_map(cx)
        row = br.get(cx, rid)
    assert "top" not in canon                            # NOT promoted
    assert row["remedies"][0]["meaning"] == "ONE TIME"   # reveal row still updated


def test_actions_save_delete_propose(tmp_path):
    rma = _load("dashboard.remedy_meaning_actions")
    bm = _load("dashboard.biofield_meanings")
    db = str(tmp_path / "a.db")
    with sqlite3.connect(db) as cx:
        bm.init_table(cx)
    rma.configure(client=_FakeClient("AI MEANING."),
                  products={"nous-energy": {"name": "Nous Energy", "benefits": ["energy"]}})
    with sqlite3.connect(db) as cx:
        rma._exec_save({"slug": "nous-energy", "meaning": "Glen text."}, {"cx": cx, "actor": None})
        assert bm.get_map(cx)["nous-energy"] == "Glen text."
        rma._exec_propose({"slug": "nous-energy"}, {"cx": cx, "actor": None})
        assert bm.get_map(cx)["nous-energy"] == "AI MEANING."  # propose overwrites with ai text
        rma._exec_delete({"slug": "nous-energy"}, {"cx": cx, "actor": None})
        assert bm.get_map(cx) == {}


def test_action_propose_all_fills_missing(tmp_path):
    rma = _load("dashboard.remedy_meaning_actions")
    bm = _load("dashboard.biofield_meanings")
    db = str(tmp_path / "a.db")
    with sqlite3.connect(db) as cx:
        bm.init_table(cx)
        bm.upsert(cx, "has", "already", "glen", "glen")
    rma.configure(client=_FakeClient("AI."),
                  products={"has": {"name": "Has"}, "missing": {"name": "Missing"}})
    with sqlite3.connect(db) as cx:
        out = rma._exec_propose_all({}, {"cx": cx, "actor": None})
        mp = bm.get_map(cx)
    assert out["proposed"] == 1 and mp["missing"] == "AI." and mp["has"] == "already"
