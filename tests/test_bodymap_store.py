import json
import bodymap_store


def _zone(**over):
    base = {
        "id": "iris-R-liver", "eye": "right", "germ_layer": "endoderm",
        "radial": {"r_inner": 0.10, "r_outer": 0.30},
        "sector": {"start_deg": 235, "end_deg": 260},
        "anatomy": "Liver", "meaning_standard": "Detox zone.",
        "meaning_glen": "", "layers": {},
    }
    base.update(over)
    return base


def test_validate_zone_accepts_complete():
    ok, err = bodymap_store.validate_zone(_zone())
    assert ok is True and err is None


def test_validate_zone_rejects_missing_field():
    z = _zone(); del z["anatomy"]
    ok, err = bodymap_store.validate_zone(z)
    assert ok is False and "anatomy" in err


def test_validate_zone_rejects_bad_radial_order():
    ok, err = bodymap_store.validate_zone(_zone(radial={"r_inner": 0.5, "r_outer": 0.2}))
    assert ok is False and "radial" in err


def test_validate_zone_rejects_bad_sector_range():
    ok, err = bodymap_store.validate_zone(_zone(sector={"start_deg": 10, "end_deg": 400}))
    assert ok is False and "sector" in err


def test_validate_zone_rejects_bad_eye():
    ok, err = bodymap_store.validate_zone(_zone(eye="middle"))
    assert ok is False and "eye" in err


def test_reseed_seeds_when_missing_and_respects_force(tmp_path, monkeypatch):
    repo, persist = tmp_path / "repo", tmp_path / "persist"
    repo.mkdir(); persist.mkdir()
    (repo / "bodymap-iridology.json").write_text('{"system":"iridology","germ_layers":[],"zones":[]}')
    (repo / "bodymap-sclerology.json").write_text('{"system":"sclerology","germ_layers":[],"zones":[]}')
    monkeypatch.setattr(bodymap_store, "REPO_DATA", repo)
    monkeypatch.setattr(bodymap_store, "DATA_DIR", persist)
    monkeypatch.setattr(bodymap_store, "SYSTEMS", {
        "iridology": persist / "bodymap-iridology.json",
        "sclerology": persist / "bodymap-sclerology.json",
    })
    assert bodymap_store.reseed_from_repo() is True
    assert (persist / "bodymap-iridology.json").exists()
    (persist / "bodymap-iridology.json").write_text('{"system":"iridology","germ_layers":[],"zones":[1]}')
    assert bodymap_store.reseed_from_repo() is False  # does not clobber curation
