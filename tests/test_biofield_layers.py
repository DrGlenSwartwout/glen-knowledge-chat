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
