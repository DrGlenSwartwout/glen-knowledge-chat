from dashboard import health_profile

def test_editable_ids_include_dimensions_exclude_consent():
    ids = health_profile.EDITABLE_FIELD_IDS
    for included in ("terrain","penetration","tissue_layer","response","commitment","health_concerns"):
        assert included in ids        # dimensions are self-reported and change with healing -> editable
    assert "terms" not in ids         # consent/signature excluded

def test_build_block_off_when_disabled():
    assert health_profile.build_block(None, "a@b.com", False) == {"enabled": False}
