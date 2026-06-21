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
