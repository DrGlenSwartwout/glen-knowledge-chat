import datetime

from dashboard.biofield_reveal_import import synthesize_reveal_layers

_RAW = [
    {"n": 1, "title": "Oxidative load", "summary": "free-radical stress",
     "patterns": ["ED1", "ED2"], "pattern_labels": ["Cell membrane", "Mitochondria"],
     "remedy": {"name": "Neuro Magnesium"}},
    {"n": 2, "title": "Terrain", "summary": "", "patterns": ["ES3"],
     "pattern_labels": ["Lymphatics"], "remedy": None},
]


def _runner(found_date):
    def run(email, scan_id, e4l_db, catalog, today):
        if not found_date:
            return None, []
        return {"scan_id": 900, "scan_date": found_date}, _RAW
    return run


def test_maps_layers_and_marks_fresh_under_7_days():
    res = synthesize_reveal_layers("jane@x.com", today="2026-06-25",
                                   runner=_runner("2026-06-22"))
    assert res["found"] is True and res["fresh"] is True and res["days_ago"] == 3
    assert res["scan_id"] == 900 and res["scan_date"] == "2026-06-22"
    L0 = res["layers"][0]
    assert L0["n"] == 1 and L0["title"] == "Oxidative load"
    assert L0["most_affected"] == "Cell membrane, Mitochondria"
    assert L0["remedy_name"] == "Neuro Magnesium"
    # layer with remedy=None -> empty remedy_name, no crash
    assert res["layers"][1]["remedy_name"] == ""


def test_stale_scan_is_not_fresh_at_7_days():
    res = synthesize_reveal_layers("jane@x.com", today="2026-06-25",
                                   runner=_runner("2026-06-18"))  # 7 days
    assert res["found"] is True and res["days_ago"] == 7 and res["fresh"] is False


def test_no_scan_returns_not_found():
    res = synthesize_reveal_layers("nobody@x.com", today="2026-06-25",
                                   runner=_runner(None))
    assert res == {"found": False, "scan_id": None, "scan_date": None,
                   "days_ago": None, "fresh": False, "layers": []}
