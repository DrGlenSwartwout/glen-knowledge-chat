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


# --- explicit "Other Ingredients: None" -> verified pristine (returns "") ----
NONE_SOURCE = (
    "Creatine by Thorne SKU SF221 . "
    "Other Ingredients: None. Keep closed in a cool, dry place."
)


def test_explicit_none_verified_returns_empty_string():
    # Model reports the product lists none, backed by a verbatim page quote that
    # reads as an other-ingredients-none declaration -> "" (caller screens green).
    def fake_model(source_text, name, brand, sku):
        return {"none_source_quote": "Other Ingredients: None"}

    got = dx.extract_other_ingredients(NONE_SOURCE, name="Creatine",
                                       brand="Thorne", call_model=fake_model)
    assert got == ""


def test_none_claim_not_in_source_fails_closed():
    # A none-claim whose quote is NOT on the page must fail closed to None
    # (unrated), never "" -- this is what keeps 'not found' from turning green.
    def fake_model(source_text, name, brand, sku):
        return {"none_source_quote": "Other Ingredients: None"}

    # source that does NOT contain that declaration
    assert dx.extract_other_ingredients("Some page with real stearate content here.",
                                        name="X", brand="Y", call_model=fake_model) is None


def test_none_quote_without_ingredient_word_fails_closed():
    # A verbatim page quote that says "none" but is not an other-ingredients
    # declaration (no 'ingredient') must not trigger green.
    src = "This product contains none of the top nine allergens. Nice."
    def fake_model(source_text, name, brand, sku):
        return {"none_source_quote": "contains none of the top nine allergens"}

    assert dx.extract_other_ingredients(src, name="X", brand="Y",
                                        call_model=fake_model) is None


def test_line_takes_precedence_over_none_quote():
    # If the model returns BOTH a real line and a none quote, the line wins.
    def fake_model(source_text, name, brand, sku):
        return {"other_ingredients_line": "magnesium stearate (vegetable source) and silicon dioxide",
                "none_source_quote": "Other Ingredients: None"}

    got = dx.extract_other_ingredients(SOURCE, name="Magnesium Taurate",
                                       brand="Jarrow Formulas", call_model=fake_model)
    assert got == "magnesium stearate (vegetable source) and silicon dioxide"


# --- multi-product page: a NEIGHBOR's "None" must not green the target -------
# Target (Vitamin C) has a real red excipient; a DIFFERENT product (Creatine)
# far down the page declares None. The model borrows Creatine's none-quote and
# attributes it to Vitamin C. Anchoring must reject it (target name is far from
# the borrowed quote) -> None (unrated), never "".
def _two_product_page():
    filler = " lorem ipsum catalog padding. " * 60   # ~1800 chars of distance
    return ("Vitamin C 1000mg by Pure Encapsulations. "
            "Other Ingredients: hydrogenated palm oil, titanium dioxide."
            + filler +
            "Creatine by Thorne. Other Ingredients: None. Keep dry.")


def test_neighbor_none_quote_does_not_green_target():
    page = _two_product_page()
    def borrow_none(source_text, name, brand, sku):
        return {"none_source_quote": "Other Ingredients: None"}
    # asking about Vitamin C, model borrows Creatine's None -> must be rejected
    assert dx.extract_other_ingredients(page, name="Vitamin C 1000mg",
                                        brand="Pure Encapsulations",
                                        call_model=borrow_none) is None


def test_neighbor_line_quote_does_not_attach_to_target():
    page = _two_product_page()
    # model borrows the OTHER product's (Creatine's) none region text as a line
    def borrow_line(source_text, name, brand, sku):
        return {"other_ingredients_line": "None"}   # from Creatine's block, far from Vitamin C
    got = dx.extract_other_ingredients(page, name="Vitamin C 1000mg",
                                       brand="Pure Encapsulations", call_model=borrow_line)
    # "None" is 4 chars (< MIN_QUOTE_LEN) so verify drops it anyway; either way not the target's line
    assert got is None


def test_target_own_none_still_greens():
    # Sanity: the target's OWN none (near its name) still resolves to "".
    page = "Creatine by Thorne SKU SF221 . Other Ingredients: None. Keep dry."
    def own_none(source_text, name, brand, sku):
        return {"none_source_quote": "Other Ingredients: None"}
    assert dx.extract_other_ingredients(page, name="Creatine", brand="Thorne",
                                        call_model=own_none) == ""
