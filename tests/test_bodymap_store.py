import json
import pathlib

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


def _seed_system(tmp_path, monkeypatch, zones):
    p = tmp_path / "bodymap-iridology.json"
    p.write_text(json.dumps({
        "system": "iridology", "reference_frame": "unit_circle",
        "germ_layers": [{"id": "endoderm", "label": "E", "r_inner": 0.0, "r_outer": 0.33}],
        "zones": zones,
    }))
    monkeypatch.setattr(bodymap_store, "SYSTEMS", dict(bodymap_store.SYSTEMS, iridology=p))
    return p


def test_build_payload_drops_invalid_and_sets_display(tmp_path, monkeypatch):
    good = _zone(id="iris-R-liver", meaning_glen="Glen's read.")
    bad = _zone(id="iris-R-bad", sector={"start_deg": 10, "end_deg": 999})
    _seed_system(tmp_path, monkeypatch, [good, bad])
    payload = bodymap_store.build_payload("iridology")
    ids = {z["id"] for z in payload["zones"]}
    assert ids == {"iris-R-liver"}
    assert payload["zones"][0]["meaning_display"] == "Glen's read."  # glen overrides standard


def test_build_payload_display_falls_back_to_standard(tmp_path, monkeypatch):
    _seed_system(tmp_path, monkeypatch, [_zone(id="iris-R-liver", meaning_glen="")])
    payload = bodymap_store.build_payload("iridology")
    assert payload["zones"][0]["meaning_display"] == "Detox zone."


def test_set_zone_overlay_persists(tmp_path, monkeypatch):
    p = _seed_system(tmp_path, monkeypatch, [_zone(id="iris-R-liver", meaning_glen="")])
    bodymap_store.set_zone_overlay("iridology", "iris-R-liver", "New clinical note.")
    reloaded = json.loads(p.read_text())
    assert reloaded["zones"][0]["meaning_glen"] == "New clinical note."


def test_set_zone_overlay_unknown_raises(tmp_path, monkeypatch):
    _seed_system(tmp_path, monkeypatch, [_zone(id="iris-R-liver")])
    try:
        bodymap_store.set_zone_overlay("iridology", "nope", "x")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_shipped_seeds_all_zones_valid():
    repo = pathlib.Path(bodymap_store.__file__).resolve().parent / "data"
    for fname in ("bodymap-iridology.json", "bodymap-sclerology.json"):
        data = json.loads((repo / fname).read_text())
        assert data.get("zones"), f"{fname} has no zones"
        for z in data["zones"]:
            ok, err = bodymap_store.validate_zone(z)
            assert ok, f"{fname} zone {z.get('id')}: {err}"
        # germ_layer ids referenced by zones must exist
        layer_ids = {g["id"] for g in data.get("germ_layers", [])}
        for z in data["zones"]:
            assert z["germ_layer"] in layer_ids, f"{fname} zone {z['id']} bad germ_layer"
