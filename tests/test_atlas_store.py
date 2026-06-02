import json
import atlas_store


def _concept(**over):
    base = {
        "id": "biofield", "label": "Biofield", "aliases": ["energy field"],
        "summary": "The body's organizing energy field.",
        "namespaces": ["clinical-qa"], "cluster": "energetic-medicine",
        "parent": "energetic-medicine", "coords": {"x": 0.4, "y": 0.55},
        "neighbors": ["detox"], "links": [], "status": "live",
    }
    base.update(over)
    return base


def test_validate_concept_accepts_complete():
    ok, err = atlas_store.validate_concept(_concept())
    assert ok is True and err is None


def test_validate_concept_rejects_missing_id():
    c = _concept(); del c["id"]
    ok, err = atlas_store.validate_concept(c)
    assert ok is False and "id" in err


def test_validate_concept_rejects_bad_coords():
    ok, err = atlas_store.validate_concept(_concept(coords={"x": 2.0, "y": 0.5}))
    assert ok is False and "coords" in err


def test_build_graph_groups_hierarchy(tmp_path, monkeypatch):
    data = {"version": "t", "concepts": [
        _concept(id="biofield", parent="energetic-medicine"),
        _concept(id="detox", label="Detox", parent="foundations", cluster="foundations",
                 coords={"x": 0.1, "y": 0.2}, neighbors=["biofield"]),
    ]}
    p = tmp_path / "atlas-concepts.json"
    p.write_text(json.dumps(data))
    monkeypatch.setattr(atlas_store, "CONCEPTS_PATH", p)
    graph = atlas_store.build_graph()
    assert {c["id"] for c in graph["concepts"]} == {"biofield", "detox"}
    assert set(graph["hierarchy"].keys()) == {"energetic-medicine", "foundations"}
    assert "biofield" in graph["hierarchy"]["energetic-medicine"]


def test_approve_moves_pending_to_live(tmp_path, monkeypatch):
    monkeypatch.setattr(atlas_store, "PENDING_PATH", tmp_path / "atlas-pending.json")
    monkeypatch.setattr(atlas_store, "CONCEPTS_PATH", tmp_path / "atlas-concepts.json")
    (tmp_path / "atlas-pending.json").write_text(json.dumps(
        {"version": "t", "concepts": [_concept(status="pending")]}))
    (tmp_path / "atlas-concepts.json").write_text(json.dumps({"version": "t", "concepts": []}))
    atlas_store.approve_concept("biofield")
    live = json.loads((tmp_path / "atlas-concepts.json").read_text())
    pend = json.loads((tmp_path / "atlas-pending.json").read_text())
    assert [c["id"] for c in live["concepts"]] == ["biofield"]
    assert live["concepts"][0]["status"] == "live"
    assert pend["concepts"] == []
