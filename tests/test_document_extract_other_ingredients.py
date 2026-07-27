from dashboard import document_extract as dx

SOURCE = (
    "Magnesium Taurate by Jarrow Formulas SKU JAR-MAGTAU90 . "
    "Other Ingredients: Capsule (hydroxypropylmethylcellulose), magnesium "
    "stearate (vegetable source) and silicon dioxide. Keep out of reach."
)


def test_returns_verified_line():
    line = "Capsule (hydroxypropylmethylcellulose), magnesium stearate (vegetable source) and silicon dioxide"

    def fake_model(source_text, name, brand, sku):
        return {"other_ingredients_line": line}

    got = dx.extract_other_ingredients(SOURCE, name="Magnesium Taurate",
                                       brand="Jarrow Formulas", sku="JAR-MAGTAU90",
                                       call_model=fake_model)
    assert got == line


def test_fabricated_line_fails_closed():
    # Model invents ingredients NOT present in the source -> verify_quotes drops
    # it -> None. This is the safety guard; it must bite.
    def fake_model(source_text, name, brand, sku):
        return {"other_ingredients_line": "pharmaceutical glaze and titanium dioxide"}

    assert dx.extract_other_ingredients(SOURCE, name="X", brand="Y",
                                        call_model=fake_model) is None


def test_empty_line_returns_none():
    def fake_model(source_text, name, brand, sku):
        return {"other_ingredients_line": ""}

    assert dx.extract_other_ingredients(SOURCE, name="X", brand="Y",
                                        call_model=fake_model) is None


def test_malformed_reply_returns_none():
    for bad in [None, [], "not a dict", {"wrong_key": "z"}, {"other_ingredients_line": 123}]:
        assert dx.extract_other_ingredients(SOURCE, name="X", brand="Y",
                                            call_model=lambda *a, _b=bad: _b) is None


def test_model_error_returns_none():
    def boom(source_text, name, brand, sku):
        raise RuntimeError("model down")
    assert dx.extract_other_ingredients(SOURCE, name="X", brand="Y",
                                        call_model=boom) is None
