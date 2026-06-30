import dashboard.ash_map as am


def test_twelve_canonical_dimensions_in_order():
    assert am.DIM_KEYS == [
        "body", "mind", "spirit", "inheritance", "personal_history",
        "epigenetics", "symptoms", "terrain", "diagnosis", "treatment",
        "regulation", "prognosis",
    ]
    # ASH_DIMENSIONS carries key/name/meaning for each, in the same order
    assert [d["key"] for d in am.ASH_DIMENSIONS] == am.DIM_KEYS
    for d in am.ASH_DIMENSIONS:
        assert d["name"] and d["meaning"]


def test_state_order_ladder():
    assert am.STATE_ORDER == {"untouched": 0, "opened": 1, "explored": 2, "deep": 3}


def test_norm_email():
    assert am._norm_email("  Foo@Bar.COM ") == "foo@bar.com"


def test_blank_map_has_all_twelve_untouched_and_is_fresh():
    m = am._blank_map()
    assert set(m.keys()) == set(am.DIM_KEYS)
    for k in am.DIM_KEYS:
        assert m[k] == {
            "state": "untouched", "opened_excerpt": "",
            "notes": "", "last_touched_at": None,
        }
    # fresh dict each call — mutating one does not leak into the next
    m["body"]["notes"] = "x"
    assert am._blank_map()["body"]["notes"] == ""
