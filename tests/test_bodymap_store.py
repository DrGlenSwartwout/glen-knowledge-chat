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


def test_validate_zone_rejects_empty_eye():
    # side/eye is a free-form laterality/view selector now; only empty/non-string is rejected
    ok, err = bodymap_store.validate_zone(_zone(eye=""))
    assert ok is False and ("side" in err or "eye" in err)
    ok2, _ = bodymap_store.validate_zone(_zone(eye="diagnosis"))
    assert ok2 is True


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
    # auricular points are identical on both ears: bilateral on the canonical left,
    # mirrored by the renderer via outline_side for the right ear.
    assert data.get("outline_side") == "left"
    assert all(z.get("bilateral") for z in data["zones"]), "ear points must be bilateral"


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
        assert z["geometry"]["type"] == "ellipse"   # reflexology zones are ovals   # reflexology zones are areas, not points
        assert z["group"] in group_ids, f"{z['id']} bad group {z['group']}"
        sides.add(z["side"])
    assert sides == {"left", "right"}, "both soles must be populated"
    keys = {a["key"] for a in data.get("anchors", [])}
    assert {"big-toe-tip", "heel-center", "little-toe-base"} <= keys
    # lateralized organs on the correct side
    ids = {z["id"] for z in data["zones"]}
    assert "foot-R-liver" in ids and "foot-L-heart" in ids


def test_shipped_hand_seed_valid():
    import pathlib
    repo = pathlib.Path(bodymap_store.__file__).resolve().parent / "data"
    assert bodymap_store.SYSTEMS["hand"].name == "bodymap-hand.json"
    data = json.loads((repo / "bodymap-hand.json").read_text())
    assert data["reference_frame"] == "hand_outline"
    assert data.get("outline")
    group_ids = {g["id"] for g in data.get("groups", [])}
    sides = set()
    for z in data["zones"]:
        ok, err = bodymap_store.validate_zone(z)
        assert ok, f"hand zone {z.get('id')}: {err}"
        assert z["geometry"]["type"] == "ellipse"   # reflexology zones are ovals
        assert z["group"] in group_ids, f"{z['id']} bad group {z['group']}"
        sides.add(z["side"])
    assert sides == {"left", "right"}, "both palms must be populated"
    keys = {a["key"] for a in data.get("anchors", [])}
    assert {"thumb-tip", "middle-finger-tip", "wrist-center"} <= keys
    ids = {z["id"] for z in data["zones"]}
    assert "hand-R-liver" in ids and "hand-L-heart" in ids


def test_shipped_meridian_seed_valid():
    import pathlib
    repo = pathlib.Path(bodymap_store.__file__).resolve().parent / "data"
    assert bodymap_store.SYSTEMS["meridian"].name == "bodymap-meridian.json"
    data = json.loads((repo / "bodymap-meridian.json").read_text())
    assert data["reference_frame"] == "body_outline"
    assert set(data.get("outlines", {})) == {"front", "back", "side"}
    assert len(data.get("groups", [])) == 14, "12 primary channels + Ren + Du"
    gtypes, views = set(), set()
    for z in data["zones"]:
        ok, err = bodymap_store.validate_zone(z)
        assert ok, f"meridian zone {z.get('id')}: {err}"
        gtypes.add(z["geometry"]["type"])
        views.add(z["side"])
    assert {"path", "point"} <= gtypes, "channels are lines, acupoints are points"
    assert views <= {"front", "back", "side"} and "side" in views


def test_shipped_eav_seed_valid():
    import pathlib
    repo = pathlib.Path(bodymap_store.__file__).resolve().parent / "data"
    assert bodymap_store.SYSTEMS["eav"].name == "bodymap-eav.json"
    data = json.loads((repo / "bodymap-eav.json").read_text())
    assert set(data.get("outlines", {})) == {"hand", "foot"}
    views = set()
    for z in data["zones"]:
        ok, err = bodymap_store.validate_zone(z)
        assert ok, f"eav zone {z.get('id')}: {err}"
        assert z["geometry"]["type"] == "point"
        views.add(z["side"])
    assert views == {"hand", "foot"}
    ids = {z["id"] for z in data["zones"]}
    assert {"eav-LU11", "eav-BL67"} <= ids  # jing-well terminal points present


def test_shipped_neurotome_seed_valid():
    import pathlib
    repo = pathlib.Path(bodymap_store.__file__).resolve().parent / "data"
    assert bodymap_store.SYSTEMS["neurotome"].name == "bodymap-neurotome.json"
    data = json.loads((repo / "bodymap-neurotome.json").read_text())
    assert data["reference_frame"] == "body_outline"
    assert set(data.get("outlines", {})) == {"front", "back"}
    views = set()
    for z in data["zones"]:
        ok, err = bodymap_store.validate_zone(z)
        assert ok, f"neurotome zone {z.get('id')}: {err}"
        assert z["geometry"]["type"] == "polygon"
        views.add(z["side"])
    assert views == {"front", "back"}


def test_shipped_lymph_seed_valid():
    import pathlib
    repo = pathlib.Path(bodymap_store.__file__).resolve().parent / "data"
    assert bodymap_store.SYSTEMS["lymph"].name == "bodymap-lymph.json"
    data = json.loads((repo / "bodymap-lymph.json").read_text())
    assert set(data.get("outlines", {})) == {"front", "back"}
    gtypes, views = set(), set()
    for z in data["zones"]:
        ok, err = bodymap_store.validate_zone(z)
        assert ok, f"lymph zone {z.get('id')}: {err}"
        gtypes.add(z["geometry"]["type"])
        views.add(z["side"])
    assert {"point", "path"} <= gtypes  # nodes are points, ducts are lines
    assert views == {"front", "back"}


def test_shipped_face_seed_valid():
    import pathlib
    repo = pathlib.Path(bodymap_store.__file__).resolve().parent / "data"
    assert bodymap_store.SYSTEMS["face"].name == "bodymap-face.json"
    data = json.loads((repo / "bodymap-face.json").read_text())
    assert data["reference_frame"] == "face_outline"
    assert data.get("outline") and len(data.get("anchors", [])) == 2
    groups = {g["id"] for g in data["groups"]}
    assert {"wood", "fire", "earth", "metal", "water"} <= groups  # diagnosis elements
    views, gtypes = set(), set()
    for z in data["zones"]:
        ok, err = bodymap_store.validate_zone(z)
        assert ok, f"face zone {z.get('id')}: {err}"
        assert z["group"] in groups
        views.add(z["side"])
        gtypes.add(z["geometry"]["type"])
    # five map layers, multiple geometries
    assert {"diagnosis", "acu", "lymph", "nerve", "eav"} <= views
    assert {"ellipse", "point", "polygon", "path"} <= gtypes


def test_shipped_organs_seed_valid():
    import pathlib
    repo = pathlib.Path(bodymap_store.__file__).resolve().parent / "data"
    assert bodymap_store.SYSTEMS["organs"].name == "bodymap-organs.json"
    data = json.loads((repo / "bodymap-organs.json").read_text())
    assert data["reference_frame"] == "body_outline"
    assert data.get("outlines", {}).get("front") and data.get("outlines", {}).get("back")
    groups = {g["id"] for g in data["groups"]}
    views = set()
    for z in data["zones"]:
        ok, err = bodymap_store.validate_zone(z)
        assert ok, f"organs zone {z.get('id')}: {err}"
        assert z["group"] in groups
        assert z["geometry"]["type"] == "ellipse"
        views.add(z["side"])
    assert {"front", "back"} <= views


def test_resolve_finding_zones_organs_body_atlas():
    # whole-body atlas: findings light the real organ location, incl. off-face organs
    r = bodymap_store.resolve_finding_zones("organs", ["Brain", "Heart", "Kidney"])
    assert "organ-brain" in r["zones"] and "organ-heart" in r["zones"]
    assert "organ-kidney-r" in r["zones"] and "organ-kidney-l" in r["zones"]


def test_shipped_skeleton_and_muscle_seeds_valid():
    import pathlib
    repo = pathlib.Path(bodymap_store.__file__).resolve().parent / "data"
    for system, prefix in (("skeleton", "bone-"), ("muscle", "muscle-")):
        assert bodymap_store.SYSTEMS[system].name == f"bodymap-{system}.json"
        data = json.loads((repo / f"bodymap-{system}.json").read_text())
        assert data["reference_frame"] == "body_outline"
        assert data.get("outlines", {}).get("front") and data.get("outlines", {}).get("back")
        groups = {g["id"] for g in data["groups"]}
        views = set()
        for z in data["zones"]:
            ok, err = bodymap_store.validate_zone(z)
            assert ok, f"{system} zone {z.get('id')}: {err}"
            assert z["group"] in groups and z["id"].startswith(prefix)
            assert z["geometry"]["type"] == "ellipse"
            views.add(z["side"])
        assert {"front", "back"} <= views


def test_shipped_dental_seed_valid():
    import pathlib
    repo = pathlib.Path(bodymap_store.__file__).resolve().parent / "data"
    assert bodymap_store.SYSTEMS["dental"].name == "bodymap-dental.json"
    data = json.loads((repo / "bodymap-dental.json").read_text())
    assert data["reference_frame"] == "dental_outline"
    assert len(data["zones"]) == 32
    for z in data["zones"]:
        ok, err = bodymap_store.validate_zone(z)
        assert ok, f"dental zone {z.get('id')}: {err}"
        assert z["geometry"]["type"] == "ellipse"
        assert isinstance(z.get("meridian_organs"), list) and z["meridian_organs"]


def test_resolve_finding_zones_dental_meridian_organs():
    # a tooth lights for its associated meridian organ (not just its anatomy name)
    kidney = bodymap_store.resolve_finding_zones("dental", ["Kidney"])["zones"]
    assert len(kidney) == 8 and all("tooth-" in z for z in kidney)  # incisors, both arches
    liver = bodymap_store.resolve_finding_zones("dental", ["Liver"])["zones"]
    assert len(liver) == 4  # canines
    # upper/lower reciprocal: Lung lights upper premolars + lower molars
    lung = set(bodymap_store.resolve_finding_zones("dental", ["Lung"])["zones"])
    assert "tooth-upper-4" in lung and "tooth-lower-30" in lung


def test_shipped_organclock_seed_valid():
    import pathlib
    repo = pathlib.Path(bodymap_store.__file__).resolve().parent / "data"
    assert bodymap_store.SYSTEMS["organclock"].name == "bodymap-organclock.json"
    data = json.loads((repo / "bodymap-organclock.json").read_text())
    assert data["reference_frame"] == "unit_circle"
    assert len(data["zones"]) == 12  # 12 meridians
    secs = []
    for z in data["zones"]:
        ok, err = bodymap_store.validate_zone(z)
        assert ok, f"organclock zone {z.get('id')}: {err}"
        secs.append((z["sector"]["start_deg"], z["sector"]["end_deg"]))
    secs.sort()
    # 12 contiguous 30-degree sectors covering the whole circle, no wrap past 0
    assert secs[0][0] == 0 and secs[-1][1] == 360
    assert all(secs[i][1] == secs[i + 1][0] for i in range(len(secs) - 1))


def test_resolve_finding_zones_organclock_lights_its_window():
    assert bodymap_store.resolve_finding_zones("organclock", ["Liver"])["zones"] == ["clock-LR"]
    assert bodymap_store.resolve_finding_zones("organclock", ["Kidney"])["zones"] == ["clock-KI"]


def test_tissue_layers_taxonomy():
    layers = bodymap_store.TISSUE_LAYERS
    assert len(layers) == 5
    names = [L["name"] for L in layers]
    assert names == ["Compression", "Connection", "Conversion", "Communication", "Containment"]
    for L in layers:
        assert len(L["sublayers"]) == 2  # 5 layers x 2 sub-layers
    # sub-layer -> layer mapping is exhaustive and unique
    subs = [sl["id"] for L in layers for sl in L["sublayers"]]
    assert len(subs) == 10 and len(set(subs)) == 10
    assert bodymap_store.sublayer_to_layer("urogenital") == "compression"
    assert bodymap_store.sublayer_to_layer("integument") == "containment"
    assert bodymap_store.sublayer_to_layer("nope") is None


def test_tissue_catalog_seed_valid():
    cat = bodymap_store.tissue_catalog()
    assert cat["layers"] is bodymap_store.TISSUE_LAYERS
    organs = cat["organs"]
    assert len(organs) >= 40
    ids = [o["id"] for o in organs]
    assert len(ids) == len(set(ids)), "duplicate organ id"
    valid = {sl["id"] for L in bodymap_store.TISSUE_LAYERS for sl in L["sublayers"]}
    for o in organs:
        assert o["sublayer"] in valid, f"{o['id']} bad sublayer {o['sublayer']}"
        assert o.get("keywords"), f"{o['id']} has no keywords"


def test_set_organ_sublayer_persists(tmp_path, monkeypatch):
    seed = {"organs": [{"id": "liver", "name": "Liver", "sublayer": "digestive", "keywords": ["liver"]}]}
    p = tmp_path / "bodymap-tissue-layers.json"
    p.write_text(json.dumps(seed))
    monkeypatch.setattr(bodymap_store, "DATA_DIR", tmp_path)
    updated = bodymap_store.set_organ_sublayer("liver", "endocrine")
    assert updated["sublayer"] == "endocrine"
    assert json.loads(p.read_text())["organs"][0]["sublayer"] == "endocrine"
    # guards
    try:
        bodymap_store.set_organ_sublayer("liver", "bogus"); assert False
    except ValueError:
        pass
    try:
        bodymap_store.set_organ_sublayer("nope", "digestive"); assert False
    except KeyError:
        pass


def test_system_catalog_covers_all_systems():
    cat = bodymap_store.system_catalog()
    cat_ids = [c["id"] for c in cat]
    # every registered system has exactly one catalog entry, and vice versa
    assert len(cat_ids) == len(set(cat_ids)), "duplicate catalog ids"
    assert set(cat_ids) == set(bodymap_store.SYSTEMS), "catalog vs SYSTEMS drift"
    for c in cat:
        assert c["name"] and c["category"] and c["description"], f"incomplete catalog entry {c['id']}"


def _assert_body_atlas_valid(system, expected_views):
    """Shared check for a whole-body system atlas: on the body silhouette, every
    zone valid, groups referenced exist, and the expected views are present."""
    import pathlib
    repo = pathlib.Path(bodymap_store.__file__).resolve().parent / "data"
    assert bodymap_store.SYSTEMS[system].name == f"bodymap-{system}.json"
    data = json.loads((repo / f"bodymap-{system}.json").read_text())
    assert data["reference_frame"] == "body_outline"
    groups = {g["id"] for g in data["groups"]}
    views = set()
    for z in data["zones"]:
        ok, err = bodymap_store.validate_zone(z)
        assert ok, f"{system} zone {z.get('id')}: {err}"
        assert z["group"] in groups, f"{system} zone {z['id']} bad group"
        views.add(z["side"])
    assert set(expected_views) <= views


def test_shipped_nervous_seed_valid():
    _assert_body_atlas_valid("nervous", ("front", "back"))
    # a specific nerve finding lights its structure
    assert bodymap_store.resolve_finding_zones("nervous", ["Sciatic Nerve"])["zones"]
    assert bodymap_store.resolve_finding_zones("nervous", ["Optic Nerve"])["zones"]


def test_shipped_endocrine_seed_valid():
    _assert_body_atlas_valid("endocrine", ("front",))
    assert bodymap_store.resolve_finding_zones("endocrine", ["Thyroid"])["zones"]
    assert bodymap_store.resolve_finding_zones("endocrine", ["Adrenal Gland"])["zones"]


def test_shipped_respiratory_seed_valid():
    _assert_body_atlas_valid("respiratory", ("front", "back"))
    assert bodymap_store.resolve_finding_zones("respiratory", ["Lung"])["zones"]
    assert bodymap_store.resolve_finding_zones("respiratory", ["Bronchi"])["zones"]


def test_shipped_digestive_seed_valid():
    _assert_body_atlas_valid("digestive", ("front",))
    assert bodymap_store.resolve_finding_zones("digestive", ["Liver"])["zones"]
    # accessory organ + large/small intestine separation holds here too
    lg = bodymap_store.resolve_finding_zones("digestive", ["Large Intestine"])["zones"]
    assert "dig-jejunum-ileum" not in lg and any("colon" in z for z in lg)


def test_shipped_cardiovascular_seed_valid():
    _assert_body_atlas_valid("cardiovascular", ("front", "back"))
    assert bodymap_store.resolve_finding_zones("cardiovascular", ["Heart"])["zones"]
    assert bodymap_store.resolve_finding_zones("cardiovascular", ["Aorta"])["zones"]


def test_shipped_urogenital_seed_valid():
    _assert_body_atlas_valid("urogenital", ("female", "male"))
    # shared urinary organ appears in both views; sex-specific organs per view
    kidney = bodymap_store.resolve_finding_zones("urogenital", ["Kidney"])["zones"]
    assert any("female" in z for z in kidney) and any("male" in z for z in kidney)
    assert bodymap_store.resolve_finding_zones("urogenital", ["Prostate"])["zones"]
    assert bodymap_store.resolve_finding_zones("urogenital", ["Ovaries"])["zones"]


def test_lymph_immune_and_connective_extension():
    # extended lymph map carries immune organs + a connective-tissue group
    import pathlib
    repo = pathlib.Path(bodymap_store.__file__).resolve().parent / "data"
    data = json.loads((repo / "bodymap-lymph.json").read_text())
    assert "connective" in {g["id"] for g in data["groups"]}
    assert bodymap_store.resolve_finding_zones("lymph", ["Spleen"])["zones"]  # immune organ
    assert bodymap_store.resolve_finding_zones("lymph", ["Connective Tissue"])["zones"]


def test_zone_ids_whole_system_and_side():
    all_bones = bodymap_store.zone_ids("skeleton")
    front_bones = bodymap_store.zone_ids("skeleton", side="front")
    assert len(all_bones) > len(front_bones) > 0
    assert all(z.startswith("bone-") for z in all_bones)
    assert bodymap_store.zone_ids("nope") == []


def test_resolve_finding_zones_specific_bone_and_muscle():
    assert "bone-femur-r" in bodymap_store.resolve_finding_zones("skeleton", ["Femur"])["zones"]
    assert bodymap_store.resolve_finding_zones("skeleton", ["Hip Joint"])["zones"]  # joint match
    assert "muscle-biceps-r" in bodymap_store.resolve_finding_zones("muscle", ["Biceps"])["zones"]


def test_resolve_finding_zones_large_vs_small_intestine():
    # "Large Intestine" lights the colon, NOT the small intestine (shared word guard)
    lg = bodymap_store.resolve_finding_zones("organs", ["Large Intestine"])
    assert "organ-small-intestine" not in lg["zones"]
    assert any(z.startswith("organ-colon") for z in lg["zones"])
    sm = bodymap_store.resolve_finding_zones("organs", ["Small Intestine"])
    assert sm["zones"] == ["organ-small-intestine"]


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


# ---- resolve_finding_zones: light Body Map zones from a client's finding names ----

def test_resolve_finding_zones_matches_organ_on_face():
    r = bodymap_store.resolve_finding_zones("face", ["Liver"], side="diagnosis")
    assert "face-glabella" in r["zones"]
    assert r["by_name"]["Liver"]  # non-empty hit list

def test_resolve_finding_zones_driver_suffix_via_word():
    # a multi-word finding name ("Liver Driver") still lights via the word "liver"
    r = bodymap_store.resolve_finding_zones("face", ["Liver Driver"], side="diagnosis")
    assert "face-glabella" in r["zones"]

def test_resolve_finding_zones_bladder_not_gallbladder():
    # word-boundary match: a Bladder finding must NOT light gallbladder-only zones
    r = bodymap_store.resolve_finding_zones("face", ["Bladder"], side="diagnosis")
    assert "face-forehead-bladder" in r["zones"]
    assert "face-temple-R" not in r["zones"]  # "Gallbladder (temple)"

def test_resolve_finding_zones_colon_synonym():
    r = bodymap_store.resolve_finding_zones("face", ["Colon"], side="diagnosis")
    assert "face-forehead-li" in r["zones"]  # "Large intestine (mid forehead)"

def test_resolve_finding_zones_plural_insensitive():
    # E4L "Lung" (singular) matches the zone "Lungs (cheek)" (plural)
    r = bodymap_store.resolve_finding_zones("face", ["Lung"], side="diagnosis")
    assert "face-cheek-R" in r["zones"] and "face-cheek-L" in r["zones"]

def test_resolve_finding_zones_side_filter_and_noise():
    # noise words match nothing; side filter keeps it to the diagnosis layer
    r = bodymap_store.resolve_finding_zones("face", ["Source Driver", "Cell"], side="diagnosis")
    assert r["zones"] == [] and r["by_name"] == {}

def test_resolve_finding_zones_unknown_system_empty():
    r = bodymap_store.resolve_finding_zones("nope", ["Liver"])
    assert r == {"zones": [], "by_name": {}}

def test_resolve_finding_zones_dedupes_and_orders():
    # two findings hitting overlapping zones -> each zone id appears once
    r = bodymap_store.resolve_finding_zones("face", ["Kidney", "Bladder"], side="diagnosis")
    assert len(r["zones"]) == len(set(r["zones"]))
    assert "face-chin" in r["zones"]  # chin = "Kidney / bladder / hormones"
