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


def test_resolve_atlas_target_override_wins():
    c = {"cluster": "brain-nervous", "body_map": {"system": "iridology", "zone": "iris-R-liver"}}
    assert bodymap_store.resolve_atlas_target(c) == {"system": "iridology", "zone": "iris-R-liver"}


def test_resolve_atlas_target_cluster_hit():
    c = {"cluster": "brain-nervous"}
    assert bodymap_store.resolve_atlas_target(c) == {"system": "iridology", "zone": "iris-R-brain"}


def test_resolve_atlas_target_unmapped_is_none():
    assert bodymap_store.resolve_atlas_target({"cluster": "antioxidants"}) is None
    assert bodymap_store.resolve_atlas_target({}) is None
    assert bodymap_store.resolve_atlas_target("nope") is None


def test_resolve_atlas_target_ignores_override_without_system():
    c = {"cluster": "brain-nervous", "body_map": {"zone": "iris-R-liver"}}
    assert bodymap_store.resolve_atlas_target(c) == {"system": "iridology", "zone": "iris-R-brain"}


def test_atlas_target_url_variants():
    assert bodymap_store.atlas_target_url({"system": "iridology", "zone": "iris-R-liver"}) == "/body-map?system=iridology&zone=iris-R-liver"
    assert bodymap_store.atlas_target_url({"system": "iridology", "layer": "mesoderm"}) == "/body-map?system=iridology&layer=mesoderm"
    assert bodymap_store.atlas_target_url({"system": "iridology"}) == "/body-map?system=iridology"
    assert bodymap_store.atlas_target_url(None) == ""
    # zone takes precedence over layer when both present
    assert bodymap_store.atlas_target_url({"system": "iridology", "zone": "iris-R-liver", "layer": "mesoderm"}) == "/body-map?system=iridology&zone=iris-R-liver"


def test_cluster_map_targets_exist_in_seed():
    repo = pathlib.Path(bodymap_store.__file__).resolve().parent / "data"
    seed = json.loads((repo / "bodymap-iridology.json").read_text())
    zones = {z["id"] for z in seed["zones"]}
    layers = {g["id"] for g in seed["germ_layers"]}
    for cluster, tgt in bodymap_store.ATLAS_CLUSTER_MAP.items():
        assert tgt.get("system") == "iridology", cluster
        if "zone" in tgt:
            assert tgt["zone"] in zones, f"{cluster} -> unknown zone {tgt['zone']}"
        if "layer" in tgt:
            assert tgt["layer"] in layers, f"{cluster} -> unknown layer {tgt['layer']}"


def _point_zone(**over):
    base = {
        "id": "ear-L-shenmen", "side": "left", "group": "triangular-fossa",
        "geometry": {"type": "point", "x": 0.44, "y": 0.28},
        "anatomy": "Shen Men", "meaning_standard": "Calming point.",
        "meaning_glen": "", "layers": {},
    }
    base.update(over)
    return base


def test_validate_point_zone_accepts_complete():
    ok, err = bodymap_store.validate_zone(_point_zone())
    assert ok is True and err is None


def test_validate_point_zone_rejects_out_of_range_xy():
    ok, err = bodymap_store.validate_zone(_point_zone(geometry={"type": "point", "x": 1.4, "y": 0.2}))
    assert ok is False and "point" in err


def test_validate_point_zone_requires_side_and_group():
    z = _point_zone(); del z["side"]
    ok, err = bodymap_store.validate_zone(z)
    assert ok is False and ("side" in err or "eye" in err)
    z2 = _point_zone(); del z2["group"]
    ok2, err2 = bodymap_store.validate_zone(z2)
    assert ok2 is False and "grouping" in err2


def test_validate_sector_zone_still_accepts_iris():
    iris = _zone()  # the iris fixture from earlier in this file (radial+sector+eye+germ_layer)
    ok, err = bodymap_store.validate_zone(iris)
    assert ok is True and err is None


def test_build_payload_passes_through_ear_fields(tmp_path, monkeypatch):
    p = tmp_path / "bodymap-ear.json"
    p.write_text(json.dumps({
        "system": "ear", "reference_frame": "ear_outline", "outline": "M 0 0 Z",
        "groups": [{"id": "lobe", "label": "Lobe"}],
        "anchors": [{"key": "helix-top", "template": {"x": 0.5, "y": 0.05}, "hint": "top"}],
        "zones": [_point_zone(group="lobe")],
    }))
    monkeypatch.setattr(bodymap_store, "SYSTEMS", dict(bodymap_store.SYSTEMS, ear=p))
    payload = bodymap_store.build_payload("ear")
    assert payload["reference_frame"] == "ear_outline"
    assert payload["outline"] == "M 0 0 Z"
    assert payload["groups"] == [{"id": "lobe", "label": "Lobe"}]
    assert payload["anchors"][0]["key"] == "helix-top"
    assert payload["zones"][0]["meaning_display"] == "Calming point."


def test_shipped_ear_seed_valid():
    import pathlib
    repo = pathlib.Path(bodymap_store.__file__).resolve().parent / "data"
    data = json.loads((repo / "bodymap-ear.json").read_text())
    assert data["reference_frame"] == "ear_outline"
    assert data.get("outline")
    assert data.get("zones")
    group_ids = {g["id"] for g in data.get("groups", [])}
    for z in data["zones"]:
        ok, err = bodymap_store.validate_zone(z)
        assert ok, f"ear zone {z.get('id')}: {err}"
        assert z["geometry"]["type"] == "point"
        assert z["group"] in group_ids, f"{z['id']} bad group {z['group']}"
    keys = {a["key"] for a in data.get("anchors", [])}
    assert {"helix-top", "lobe-bottom", "tragus"} <= keys


def test_foot_system_registered():
    assert "foot" in bodymap_store.SYSTEMS
    assert bodymap_store.SYSTEMS["foot"].name == "bodymap-foot.json"


def test_shipped_foot_seed_valid():
    import pathlib
    repo = pathlib.Path(bodymap_store.__file__).resolve().parent / "data"
    data = json.loads((repo / "bodymap-foot.json").read_text())
    assert data["reference_frame"] == "foot_outline"
    assert data.get("outline")
    group_ids = {g["id"] for g in data.get("groups", [])}
    sides = set()
    for z in data["zones"]:
        ok, err = bodymap_store.validate_zone(z)
        assert ok, f"foot zone {z.get('id')}: {err}"
        assert z["geometry"]["type"] == "polygon"   # reflexology zones are areas, not points
        assert z["group"] in group_ids, f"{z['id']} bad group {z['group']}"
        sides.add(z["side"])
    assert sides == {"left", "right"}, "both soles must be populated"
    keys = {a["key"] for a in data.get("anchors", [])}
    assert {"big-toe-tip", "heel-center", "little-toe-base"} <= keys
    # lateralized organs on the correct side
    ids = {z["id"] for z in data["zones"]}
    assert "foot-R-liver" in ids and "foot-L-heart" in ids


def _poly_zone(**over):
    base = {"id": "foot-liver", "side": "right", "bilateral": False, "group": "digestive",
            "geometry": {"type": "polygon", "points": [[0.6,0.44],[0.66,0.46],[0.64,0.53],[0.58,0.51]]},
            "anatomy": "Liver", "meaning_standard": "Liver reflex area.", "meaning_glen": "", "layers": {}}
    base.update(over); return base

def test_validate_polygon_accepts():
    ok, err = bodymap_store.validate_zone(_poly_zone()); assert ok is True and err is None

def test_validate_polygon_rejects_too_few_points():
    ok, err = bodymap_store.validate_zone(_poly_zone(geometry={"type":"polygon","points":[[0.1,0.1],[0.2,0.2]]}))
    assert ok is False and "polygon" in err

def test_validate_polygon_rejects_out_of_range():
    ok, err = bodymap_store.validate_zone(_poly_zone(geometry={"type":"polygon","points":[[0.1,0.1],[0.2,0.2],[1.4,0.3]]}))
    assert ok is False and "polygon" in err

def test_upsert_and_delete_zone(tmp_path, monkeypatch):
    p = tmp_path / "bodymap-foot.json"
    p.write_text(json.dumps({"system":"foot","reference_frame":"foot_outline","groups":[{"id":"digestive","label":"Digestive"}],"zones":[]}))
    monkeypatch.setattr(bodymap_store, "SYSTEMS", dict(bodymap_store.SYSTEMS, foot=p))
    bodymap_store.upsert_zone("foot", _poly_zone())
    assert len(json.loads(p.read_text())["zones"]) == 1
    bodymap_store.upsert_zone("foot", _poly_zone(anatomy="Liver v2"))   # update same id
    zs = json.loads(p.read_text())["zones"]; assert len(zs) == 1 and zs[0]["anatomy"] == "Liver v2"
    bodymap_store.delete_zone("foot", "foot-liver")
    assert json.loads(p.read_text())["zones"] == []
    try:
        bodymap_store.delete_zone("foot", "nope"); assert False
    except KeyError:
        pass

def test_upsert_zone_rejects_invalid(tmp_path, monkeypatch):
    p = tmp_path / "bodymap-foot.json"; p.write_text(json.dumps({"system":"foot","zones":[]}))
    monkeypatch.setattr(bodymap_store, "SYSTEMS", dict(bodymap_store.SYSTEMS, foot=p))
    try:
        bodymap_store.upsert_zone("foot", _poly_zone(anatomy=None)); assert False
    except ValueError:
        pass
