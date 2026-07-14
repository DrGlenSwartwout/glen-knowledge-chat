import dashboard.portal_view as pv


def test_assemble_attaches_remedy_ref_when_shown(monkeypatch):
    monkeypatch.setattr(pv, "entity_refs_remedy",
        lambda cx, name: {"name": name, "info": "Rebuilds terrain.", "href": "/begin/product/terrain-restore"})
    monkeypatch.setattr(pv, "entity_refs_function",
        lambda cx, title: {"name": title, "info": "", "href": None})
    content = {"layers": [{"n": 1, "title": "Liver", "meaning": "m", "remedy": "Terrain Restore", "dosing": "2/day"}]}
    out = pv._assemble_biofield(None, content, "confirmed",
                                scan_date=None, scan_dates=[], actionable=False, unlocked=True)
    L = out["layers"][0]
    assert L["remedy_info"] == "Rebuilds terrain."
    assert L["remedy_href"] == "/begin/product/terrain-restore"


def test_assemble_omits_remedy_ref_when_blurred(monkeypatch):
    called = {"n": 0}

    def _boom(cx, name):
        called["n"] += 1
        return {}

    monkeypatch.setattr(pv, "entity_refs_remedy", _boom)
    monkeypatch.setattr(pv, "entity_refs_function",
        lambda cx, title: {"name": title, "info": "", "href": None})
    content = {"layers": [{"n": 1, "title": "Liver", "remedy": "Terrain Restore"}]}
    out = pv._assemble_biofield(None, content, "confirmed",
                               scan_date=None, scan_dates=[], actionable=False, unlocked=False)
    assert out["blurred"] is True
    assert "remedy" not in out["layers"][0] and "remedy_info" not in out["layers"][0]
    assert called["n"] == 0  # never resolve a remedy for a blurred report
