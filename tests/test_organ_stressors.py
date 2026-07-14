import json
from dashboard import organ_stressors as os_


def _data(tmp_path):
    p = tmp_path / "sf.json"
    p.write_text(json.dumps({
        "factors": {
            "stomach": [
                {"category": "toxin", "factor": "Heavy metals", "tier": "moderate", "source": "S", "url": "u"},
                {"category": "microbe", "factor": "H. pylori", "tier": "strong", "source": "T", "url": "u2"},
                {"category": "emotional", "factor": "Worry", "tier": "traditional", "source": "TCM", "url": "u3"},
            ],
            "pancreas": [{"category": "toxin", "factor": "Organochlorines", "tier": "strong", "source": "P", "url": "u4"}],
        },
        "entry_keys": {"Stomach": "stomach", "Stomach Meridian": "stomach", "Spleen Meridian": "pancreas"},
    }), encoding="utf-8")
    return os_.load(str(p))


def test_stressors_span_categories(tmp_path):
    d = _data(tmp_path)
    cats = {f["category"] for f in os_.stressors_for("Stomach", d)}
    assert cats == {"toxin", "microbe", "emotional"}


def test_meridian_inherits_organ(tmp_path):
    d = _data(tmp_path)
    assert os_.stressors_for("Stomach Meridian", d)[0]["factor"] == "Heavy metals"
    assert os_.stressors_for("Spleen Meridian", d)[0]["factor"] == "Organochlorines"


def test_unmapped_and_missing(tmp_path):
    d = _data(tmp_path)
    assert os_.stressors_for("Heart", d) == []
    assert os_.load("/no/such.json") == {}
    assert os_.stressors_for("Stomach", {}) == []


def test_real_data_integrity_and_examples():
    d = os_.load()
    # Glen's examples still resolve
    assert any("metal" in f["factor"].lower() for f in os_.stressors_for("Stomach Meridian", d))
    assert any(f["category"] == "toxin" and ("organochlorine" in f["factor"].lower() or "organophosphate" in f["factor"].lower())
               for f in os_.stressors_for("Spleen Meridian", d))
    # emotional layer present (Heart Meridian -> joy), all 15 meridians resolve to something
    merids = ["Stomach Meridian","Spleen Meridian","Liver Meridian","Gall Bladder Meridian","Lung Meridian",
              "Large Intestine Meridian","Small Intestine Meridian","Kidney Meridian","Bladder Meridian",
              "Heart Meridian","Circulation Meridian","Triple Warmer Meridian","Conception Vessel",
              "Governing Vessel","Belt Vessel"]
    for m in merids:
        assert os_.stressors_for(m, d), f"{m} has no stress factors"
    # every row has category, tier, url
    valid_cat = {"toxin","microbe","emotional","physical"}
    for key, rows in d["factors"].items():
        for f in rows:
            assert f.get("category") in valid_cat and f.get("tier") and f.get("url")
