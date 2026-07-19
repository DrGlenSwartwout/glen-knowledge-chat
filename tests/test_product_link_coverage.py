"""Every sellable product must be linkable, and no product question may be
routed to Practice Better.

Regression: asked "what is the link for the product page for the Therapeutic
Nightlight", the bot answered that it had no link and told the customer to log
into healingoasis.practicebetter.io or email Dr. Glen. The product is real
($200, sells on invoices) and its sales page was live the whole time — it just
wasn't in the 230-entry curated alias table, so the prompt's (correct) "do NOT
invent URLs" rule left the bot with nothing, and it improvised a Practice Better
referral from the clinical-qa corpus. Practice Better cannot sell products and
is being deprecated.
"""
import app


def test_catalog_backstop_links_product_named_only_in_the_question():
    """The failing case: product named in the question, absent from retrieval."""
    directive = app.build_product_directive(
        snippets_text="",
        query_text="what is the link for the product page for the Therapeutic Nightlight",
    )
    assert "Therapeutic Nightlight → https://illtowell.com/begin/product/therapeutic-nightlight" in directive


def test_every_sellable_catalog_product_has_a_reachable_page_url():
    """No sellable product may be unlinkable — the root cause of the bug."""
    products = app._PRODUCTS["products"]
    sellable = {s: p for s, p in products.items()
                if not p.get("inactive") and not p.get("info_only")}
    assert len(sellable) > 900, "catalog unexpectedly small — check the fixture"
    for slug in sellable:
        assert app._catalog_page_url(slug) == f"https://illtowell.com/begin/product/{slug}"


def test_curated_alias_wins_over_catalog_backstop():
    """Glen's curated remedymatch/shortlink URLs must not be displaced."""
    directive = app.build_product_directive(query_text="tell me about Terrain Restore")
    assert "Terrain Restore → https://illtowell.com/begin/product/" not in directive


def test_fuzzy_match_tolerates_dropped_sku_qualifier():
    """Customers drop catalog prefixes: "MB5 Emotional Stress Release Hologram"
    gets asked for as "Emotional Stress Release Hologram"."""
    matches = app._catalog_link_matches(
        "do you have the Emotional Stress Release Hologram", {})
    assert "mb5-emotional-stress-release-hologram" in " ".join(matches.values())


def test_fuzzy_match_tolerates_compound_word_split():
    """"night light" must reach "Nightlight" — the exact customer phrasing."""
    for phrasing in ("I want the therapeutic night light",
                     "what's the link for the Therapeutic Nightlight"):
        matches = app._catalog_link_matches(phrasing, {})
        assert "begin/product/therapeutic-nightlight" in " ".join(matches.values()), phrasing


def test_ambiguous_phrase_links_nothing_rather_than_guessing():
    """A confidently wrong product link is worse than none."""
    # "Cerebral Cortex Hologram" maps to two distinct SKUs in the catalog.
    assert app._catalog_link_matches("tell me about the Cerebral Cortex Hologram", {}) == {}
    assert app._catalog_link_matches("what eye drops do you have", {}) == {}


def test_ordinary_prose_matches_nothing():
    """Symptom talk must not sprout product links."""
    assert app._catalog_link_matches(
        "I have trouble sleeping and feel tired all day", {}) == {}


def test_generic_short_names_do_not_false_positive():
    """Catalog holds short generic names ('Comfort', 'Relax') — never auto-link."""
    matches = app._catalog_link_matches(
        "I want comfort and relax time, and some rescue",
        app._PRODUCT_ALIASES.get("aliases", {}),
    )
    assert matches == {}


def test_superseded_product_links_to_its_successor():
    products = app._PRODUCTS["products"]
    superseded = {s: p for s, p in products.items()
                  if p.get("superseded_by") and len((p.get("name") or "")) >= 8}
    for slug, p in superseded.items():
        matches = app._catalog_link_matches(p["name"], {})
        if matches:
            assert p["superseded_by"] in list(matches.values())[0]


def test_system_prompt_forbids_sending_anyone_to_practice_better():
    """Blanket ban, every language level — PB is being retired."""
    for level in ("self-healing", "health-care", "science"):
        prompt = app.get_system_prompt(level)
        assert "NEVER SEND ANYONE TO PRACTICE BETTER" in prompt
        # the domains are named so the rule binds to the actual URLs
        for domain in ("healingoasis.practicebetter.io",
                       "my.practicebetter.io",
                       "app.practicebetter.io"):
            assert domain in prompt
        # must beat the stale clinical-qa corpus entries that still name PB
        assert "OVERRIDES any snippet" in prompt
        # course access has to survive the ban
        assert "https://truly.vip/Intro" in prompt
        assert "https://truly.vip/GetWell" in prompt
        # must not announce a retirement that hasn't happened — clients still
        # have active course access there
        assert "DO NOT ANNOUNCE PRACTICE BETTER'S STATUS" in prompt


def test_system_prompt_requires_a_direct_answer_to_a_direct_question():
    prompt = app.get_system_prompt("self-healing")
    assert "ANSWER PRODUCT QUESTIONS DIRECTLY" in prompt
