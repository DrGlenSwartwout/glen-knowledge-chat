from dashboard.biofield_setcover import minimal_remedies


def test_greedy_picks_max_then_alpha_tiebreak():
    cov = {"Zeta": {"a", "b"}, "Alpha": {"a", "b"}, "Beta": {"c"}}
    res = minimal_remedies({"a", "b", "c"}, cov)
    # Alpha and Zeta both cover 2 -> alphabetical tie-break picks Alpha first
    assert res["picks"][0] == {"remedy": "Alpha", "covers": ["a", "b"]}
    assert res["picks"][1] == {"remedy": "Beta", "covers": ["c"]}
    assert res["uncovered"] == []


def test_uncovered_codes_reported():
    res = minimal_remedies({"a", "x"}, {"R": {"a"}})
    assert res["picks"] == [{"remedy": "R", "covers": ["a"]}]
    assert res["uncovered"] == ["x"]


def test_subsumed_remedy_not_picked():
    res = minimal_remedies({"a", "b"}, {"Big": {"a", "b"}, "Small": {"a"}})
    assert [p["remedy"] for p in res["picks"]] == ["Big"]
    assert res["uncovered"] == []


def test_coverage_restricted_to_active():
    # remedy covers extra codes not in active -> covers only the active ones
    res = minimal_remedies({"a"}, {"R": {"a", "b", "c"}})
    assert res["picks"] == [{"remedy": "R", "covers": ["a"]}]


def test_empty_active():
    assert minimal_remedies(set(), {"R": {"a"}}) == {"picks": [], "uncovered": []}
