from dashboard import related_products_actions as rpa, related_store as rs


def test_set_action_normalizes_bare_slug(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    res = rpa._exec_set({"slug": "iop-syntropy", "related": ["immune-modulation"]}, {"cx": None})
    assert res["slug"] == "iop-syntropy"
    # bare slug is normalized to {slug, reason:""}
    assert rs.load_manual("iop-syntropy") == [{"slug": "immune-modulation", "reason": ""}]


def test_set_action_saves_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    rpa._exec_set(
        {"slug": "iop-syntropy",
         "related": [{"slug": "immune-modulation", "reason": "  Pairs with pressure support "},
                     {"slug": "", "reason": "dropped"},
                     {"slug": "wholomega"}]},
        {"cx": None})
    assert rs.load_manual("iop-syntropy") == [
        {"slug": "immune-modulation", "reason": "Pairs with pressure support"},
        {"slug": "wholomega", "reason": ""},
    ]
