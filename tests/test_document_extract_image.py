from dashboard import document_extract as dx

LABEL = ("Magnesium Taurate. Supplement Facts. Magnesium 100mg. "
         "Other Ingredients: microcrystalline cellulose, magnesium stearate, silicon dioxide. "
         "Keep out of reach of children.")


def test_image_returns_verified_line():
    def fake(blob, ct):
        return {"label_text": LABEL,
                "other_ingredients_line": "microcrystalline cellulose, magnesium stearate, silicon dioxide"}
    got = dx.extract_other_ingredients_from_image(b"img", "image/jpeg", call_model=fake)
    assert got == "microcrystalline cellulose, magnesium stearate, silicon dioxide"


def test_image_fabricated_line_fails_closed():
    def fake(blob, ct):
        return {"label_text": LABEL, "other_ingredients_line": "titanium dioxide and pharmaceutical glaze"}
    assert dx.extract_other_ingredients_from_image(b"img", "image/jpeg", call_model=fake) is None


def test_image_explicit_none_returns_empty():
    def fake(blob, ct):
        return {"label_text": "Creatine. Other Ingredients: None. Store cool.",
                "none_source_quote": "Other Ingredients: None"}
    assert dx.extract_other_ingredients_from_image(b"img", "image/jpeg", call_model=fake) == ""


def test_image_no_transcription_fails_closed():
    for bad in [{"other_ingredients_line": "magnesium stearate"}, {"label_text": ""}, {"label_text": 123}, None, "x"]:
        assert dx.extract_other_ingredients_from_image(b"img", "image/jpeg",
                                                       call_model=lambda b, c, _b=bad: _b) is None


def test_image_model_error_fails_closed():
    def boom(blob, ct):
        raise RuntimeError("vision down")
    assert dx.extract_other_ingredients_from_image(b"img", "image/jpeg", call_model=boom) is None
