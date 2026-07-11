from dashboard import related_products_actions as rpa, related_store as rs


def test_set_action_saves_manual(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    res = rpa._exec_set({"slug": "iop-syntropy", "related": ["immune-modulation"]}, {"cx": None})
    assert res["slug"] == "iop-syntropy"
    assert rs.load_manual("iop-syntropy") == ["immune-modulation"]
