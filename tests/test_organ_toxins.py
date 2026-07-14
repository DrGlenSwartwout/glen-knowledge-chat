import json
from dashboard import organ_toxins as ot


def _data(tmp_path):
    p = tmp_path / "tox.json"
    p.write_text(json.dumps({
        "associations": {
            "stomach": [{"toxin": "Heavy metals", "tier": "moderate", "note": "n", "source": "S", "url": "u"}],
            "pancreas": [{"toxin": "Organochlorines", "tier": "strong", "source": "T", "url": "u2"}],
        },
        "entry_keys": {"Stomach": "stomach", "Stomach Meridian": "stomach", "Spleen Meridian": "pancreas"},
    }), encoding="utf-8")
    return ot.load(str(p))


def test_toxins_for_organ_and_meridian(tmp_path):
    d = _data(tmp_path)
    assert ot.toxins_for("Stomach", d)[0]["toxin"] == "Heavy metals"
    # meridian inherits its organ's toxins
    assert ot.toxins_for("Stomach Meridian", d)[0]["toxin"] == "Heavy metals"
    assert ot.toxins_for("Spleen Meridian", d)[0]["toxin"] == "Organochlorines"


def test_unmapped_entry_is_empty(tmp_path):
    d = _data(tmp_path)
    assert ot.toxins_for("Heart", d) == []
    assert ot.toxins_for("", d) == []


def test_missing_file_is_empty():
    assert ot.load("/no/such.json") == {}
    assert ot.toxins_for("Stomach", {}) == []


def test_real_data_has_glens_examples():
    # the shipped file: Stomach -> heavy metals, Spleen Meridian -> pancreas/pesticides
    d = ot.load()
    assert any("metal" in t["toxin"].lower() for t in ot.toxins_for("Stomach", d))
    assert any("metal" in t["toxin"].lower() for t in ot.toxins_for("Stomach Meridian", d))
    assert any("pesticid" in t["toxin"].lower() or "organochlorine" in t["toxin"].lower() or "organophosphate" in t["toxin"].lower()
               for t in ot.toxins_for("Spleen Meridian", d))
    # every source has a url and a tier
    for key, rows in d["associations"].items():
        for t in rows:
            assert t.get("url") and t.get("tier")
