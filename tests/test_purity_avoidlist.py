"""Invariants of the COMMITTED data/excipient_avoidlist.json plus its loader.
Reads the real repo file (no DATA_DIR), the way the app will."""
from dashboard import purity_avoidlist as pa


def test_loads_committed_file():
    al = pa.load_avoidlist()
    assert al["version"], "must carry a version stamp"
    assert al["red"] and al["yellow"], "both lists present and non-empty"


def test_committed_file_is_valid():
    pa.validate(pa.load_avoidlist())  # must not raise


def test_every_entry_has_canonical_aliases_rationale():
    al = pa.load_avoidlist()
    for bucket in ("red", "yellow"):
        for e in al[bucket]:
            assert e["canonical"], bucket
            assert e["aliases"] and all(a.strip() for a in e["aliases"]), e["canonical"]
            assert e["rationale"].strip(), e["canonical"]


def test_red_and_yellow_are_disjoint():
    al = pa.load_avoidlist()
    reds = {e["canonical"] for e in al["red"]}
    yellows = {e["canonical"] for e in al["yellow"]}
    assert reds.isdisjoint(yellows), "a canonical can't be both red and yellow"


def test_stearate_is_red_and_silica_is_yellow():
    al = pa.load_avoidlist()
    assert any(a == "vegetable stearate" for e in al["red"] for a in e["aliases"])
    assert any(a == "silica" for e in al["yellow"] for a in e["aliases"])


def test_validate_rejects_missing_version():
    import pytest
    with pytest.raises(ValueError):
        pa.validate({"red": [], "yellow": []})


def test_validate_rejects_entry_without_aliases():
    import pytest
    bad = {"version": "x", "red": [{"canonical": "c", "aliases": [], "rationale": "r"}], "yellow": []}
    with pytest.raises(ValueError):
        pa.validate(bad)
