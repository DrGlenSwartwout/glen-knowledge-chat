import sqlite3, sys
from pathlib import Path
import pytest


def _mods():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from dashboard import biofield_reveals, biofield_reveal_actions
        return biofield_reveals, biofield_reveal_actions
    except Exception as e:
        pytest.skip(f"module not importable: {e}")


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    from dashboard import biofield_reveals
    biofield_reveals.init_table(cx)
    return cx


class _Actor:
    name = "glen"


def test_approve_flips_first_approved_no_email(tmp_path):
    br, acts = _mods(); cx = _cx(tmp_path)
    rid, _ = br.upsert(cx, "a@x.com", "2026-06-19", {"greeting": "Hi"}, [{"name": "C"}], "s")
    sent = []
    acts.configure(send=lambda *a, **k: sent.append(a))
    acts._exec_approve({"id": rid}, {"cx": cx, "actor": _Actor()})
    assert br.get(cx, rid)["first_approved"] is True
    assert sent == []  # no email on approve (it went out at ingest)


def test_edit_updates_interpretation_and_remedies(tmp_path):
    br, acts = _mods(); cx = _cx(tmp_path)
    rid, _ = br.upsert(cx, "a@x.com", "2026-06-19", {"greeting": "Old"}, [{"name": "A"}], "s")
    acts._exec_edit({"id": rid, "greeting": "New", "body": "b",
                     "remedies": [{"name": "B", "slug": "b", "meaning": "m"}]},
                    {"cx": cx, "actor": _Actor()})
    row = br.get(cx, rid)
    assert row["interpretation"]["greeting"] == "New" and row["remedies"][0]["name"] == "B"
    assert row["first_approved"] is False


def test_edit_layer_resolves_slug_and_meaning_from_typed_name(tmp_path):
    """Glen types only the remedy NAME on a layer that had no remedy; the edit
    resolves the catalog slug and fills the meaning from the canonical store."""
    br, acts = _mods(); cx = _cx(tmp_path)
    from dashboard import biofield_meanings as bm
    bm.init_table(cx)
    bm.upsert(cx, "nous-energy", "Supports steady cellular energy.", "glen", "glen")
    rid, _ = br.upsert(cx, "a@x.com", "2026-06-19", {"greeting": "Hi"}, [], "s",
                       layers=[{"n": 1, "title": "Energy", "summary": "s",
                                "patterns": ["ED1"], "remedy": None}])
    acts.configure(
        resolve_slug=lambda r: "nous-energy" if r.get("name") == "Nous Energy" else (r.get("slug") or None),
        products={"nous-energy": {"name": "Nous Energy"}}, client=None)
    acts._exec_edit({"id": rid, "layers": [
        {"n": 1, "title": "Energy", "summary": "s", "patterns": ["ED1"],
         "remedy": {"name": "Nous Energy"}}]},          # only a NAME, no slug/meaning
        {"cx": cx, "actor": _Actor()})
    lyr = br.get(cx, rid)["layers"][0]
    assert lyr["remedy"] is not None, "remedy must NOT be dropped when only a name is typed"
    assert lyr["remedy"]["slug"] == "nous-energy"            # slug resolved from name
    assert "steady cellular energy" in lyr["remedy"]["meaning"]  # meaning filled from canonical


def test_edit_layer_unresolvable_name_is_dropped(tmp_path):
    """A typed name that matches no catalog product still drops (anti-bypass)."""
    br, acts = _mods(); cx = _cx(tmp_path)
    rid, _ = br.upsert(cx, "a@x.com", "2026-06-19", {"greeting": "Hi"}, [], "s",
                       layers=[{"n": 1, "title": "X", "summary": "s",
                                "patterns": ["ED1"], "remedy": None}])
    acts.configure(resolve_slug=lambda r: None, products={}, client=None)
    acts._exec_edit({"id": rid, "layers": [
        {"n": 1, "title": "X", "summary": "s", "patterns": ["ED1"],
         "remedy": {"name": "Not A Real Product"}}]},
        {"cx": cx, "actor": _Actor()})
    assert br.get(cx, rid)["layers"][0]["remedy"] is None
