from dashboard.biofield_reveal_import import synthesize_reveal_layers, build_coverage

_RAW = [
    {"n": 1, "title": "Ox", "summary": "", "patterns": ["ED1", "ED2"],
     "pattern_labels": ["Mem", "Mito"], "remedy": {"name": "Neuro Magnesium"}},
    {"n": 2, "title": "Terr", "summary": "", "patterns": ["ES3"],
     "pattern_labels": ["Lymph"], "remedy": {"name": "neuro magnesium"}},
    {"n": 3, "title": "X", "summary": "", "patterns": ["MB1"],
     "pattern_labels": ["B"], "remedy": None},
]


def test_layers_carry_codes():
    res = synthesize_reveal_layers("j@x.com", today="2026-06-25",
                                   runner=lambda *a, **k: ({"scan_id": 1, "scan_date": "2026-06-24"}, _RAW))
    assert res["layers"][0]["codes"] == ["ED1", "ED2"]
    assert res["layers"][2]["codes"] == ["MB1"]


def test_build_coverage_unions_and_lowercases():
    layers = [
        {"codes": ["ED1", "ED2"], "remedy_name": "Neuro Magnesium"},
        {"codes": ["ES3"], "remedy_name": "neuro magnesium"},
        {"codes": ["MB1"], "remedy_name": ""},
    ]
    cov = build_coverage(layers)
    assert cov == {"neuro magnesium": {"ED1", "ED2", "ES3"}}  # unioned, lowercased, empty-remedy skipped
