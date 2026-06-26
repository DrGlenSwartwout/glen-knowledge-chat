import sqlite3
from dashboard.biofield_stress import init_stress_tables, seed_from_scan, suggest_minimal_remedies

_FIND = [{"code": "ED1", "name": "Membrane"}, {"code": "ES3", "name": "Lymph"},
         {"code": "MR2", "name": "Calm Mind"}]            # MR2 not covered -> optional
_COV = {"neuro magnesium": {"ED1", "ES3"}}


def _seeded(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_stress_tables(cx)
    seed_from_scan(cx, "a5", _FIND, _COV)                 # ED1/ES3 required, MR2 optional
    return cx


def test_suggests_minimal_set_with_labels(tmp_path):
    cx = _seeded(tmp_path)
    res = suggest_minimal_remedies(cx, "a5", [])          # nothing on the chain yet
    assert res["picks"] == [{"remedy": "neuro magnesium", "covers": ["Membrane", "Lymph"]}]
    assert res["uncovered"] == []                         # MR2 is optional -> not targeted


def test_excludes_already_balanced(tmp_path):
    cx = _seeded(tmp_path)
    # a chain row whose remedy covers ED1+ES3 -> both balanced -> nothing left to suggest
    res = suggest_minimal_remedies(cx, "a5", [{"head": "x", "remedy": "Neuro Magnesium"}])
    assert res["picks"] == [] and res["uncovered"] == []


def test_uncovered_required_with_no_remedy(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_stress_tables(cx)
    # ED1 is required (in coverage union) but its only remedy is removed from the map:
    seed_from_scan(cx, "a5", [{"code": "ED1", "name": "Membrane"}], {"tonic": {"ED1"}})
    # break coverage so ED1 has no remedy (simulate a dropped remedy)
    cx.execute("DELETE FROM biofield_auth_remedy_coverage WHERE test_id=5")
    # re-mark ED1 required by re-seeding label only is unnecessary; ED1 stays required from first seed
    res = suggest_minimal_remedies(cx, "a5", [])
    assert res["picks"] == [] and res["uncovered"] == ["Membrane"]
