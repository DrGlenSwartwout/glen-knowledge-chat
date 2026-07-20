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


def test_curated_alias_is_not_duplicated_by_the_catalog_backstop():
    """A curated alias still owns its row — the fuzzy backstop must not add a
    second, competing row for the same product.

    (Until 2026-07-19 this also asserted the curated remedymatch.com URL won.
    That policy is reversed: the in-funnel page is now the destination, per
    dashboard/order_destination.py and because GrooveKart checkout was failing.)
    """
    directive = app.build_product_directive(query_text="tell me about Terrain Restore")
    rows = [l for l in directive.splitlines()
            if l.strip().startswith("• Terrain Restore ")]
    assert len(rows) == 1, rows
    assert "/begin/product/" in rows[0], rows[0]


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


def test_curated_aliases_point_in_funnel_not_at_groovekart():
    """Clients have been unable to check out on the GrooveKart storefront, and
    dashboard/order_destination.py already makes /begin/product/<slug> the one
    order destination. The chat link table has to obey the same rule."""
    directive = app.build_product_directive(query_text="what do you recommend")
    rows = [l for l in directive.splitlines() if l.strip().startswith("•")]
    in_funnel = [l for l in rows if "/begin/product/" in l]
    storefront = [l for l in rows if "remedymatch.com" in l]
    assert len(in_funnel) > 200, f"only {len(in_funnel)} of {len(rows)} rows in-funnel"
    # The few left are aliases with no catalog entry yet (the migration backlog).
    assert len(storefront) <= 10, [l.strip() for l in storefront]


def test_retired_alias_gets_no_purchase_link_at_all():
    """A retired SKU must not fall back to its old storefront URL — that hands
    out a buy link for a product the prompt forbids recommending."""
    directive = app.build_product_directive(query_text="what do you recommend")
    for name in ("Dental Regen Powder", "Endocrine Restore", "Electrolyte Mineral Manna"):
        row = [l for l in directive.splitlines() if l.strip().startswith(f"• {name} ")]
        assert row, f"{name} missing from table"
        assert "DESCRIBE-ONLY" in row[0], row[0]
        assert "http" not in row[0], f"{name} still carries a purchase URL: {row[0]}"


def test_alias_follows_superseded_sku_to_its_successor():
    slug, retired = app._alias_catalog_slug(
        "WholOmega 120 Capsules", {"catalog_name": "WholOmega 120 Capsules"})
    assert slug == "wholomega-120-gelcaps" and not retired


def test_no_alias_points_at_the_groovekart_storefront():
    """The storefront checkout fails clients — no row may route there."""
    directive = app.build_product_directive(query_text="what do you recommend")
    stragglers = [l.strip() for l in directive.splitlines()
                  if l.strip().startswith("•") and "remedymatch.com" in l]
    assert stragglers == [], stragglers


def test_pinned_slug_survives_naming_divergence():
    """Storefront and catalog are two naming universes; an explicit slug pins
    the mapping so a near-miss can't drop the product back to the storefront."""
    for alias, expected in (("Holy Grail Full Spectrum ORMUS", "holy-grail-ormus"),
                            ("Scar Soft", "scar-soft-drink"),
                            ("Living Water", "molecular-hydrogen-bottle")):
        info = app._PRODUCT_ALIASES["aliases"][alias]
        slug, retired = app._alias_catalog_slug(alias, info)
        assert slug == expected and not retired, (alias, slug, retired)


def test_pinned_slug_that_does_not_exist_never_falls_back_to_storefront():
    slug, retired = app._alias_catalog_slug("X", {"slug": "no-such-product",
                                                  "url": "https://remedymatch.com/x"})
    assert slug == "" and not retired


def test_molecular_hydrogen_bottle_is_sellable():
    p = app._get_product("molecular-hydrogen-bottle")
    assert p and p["price_cents"] == 24997
    assert p["bottle_type"] == "own-box", "device must not pack as a bottle"


def test_allerfree_is_buyable_but_never_volunteered():
    """Glen 2026-07-20: "AllerFree not retired, but not recommended by AI."

    Not-recommended is about what the bot proactively suggests. It must never
    become "you cannot buy this" — an earlier pass suppressed the link entirely
    and the bot started telling clients AllerFree had been retired, which is
    false and blocks a real sale.
    """
    directive = app.build_product_directive(query_text="where can I buy AllerFree")
    row = [l for l in directive.splitlines() if l.strip().startswith("• AllerFree ")]
    assert row, "AllerFree missing from the injection table"
    assert "/begin/product/allerfree-homeoenergetic-drops" in row[0], row[0]

    prompt = app.get_system_prompt("self-healing")
    assert "SELLABLE BUT NOT RECOMMENDED" in prompt
    # must not be taught as discontinued
    assert 'Do NOT recommend "AllerFree"' not in prompt


def test_allerfree_catalog_entry_stays_sellable():
    p = app._get_product("allerfree-homeoenergetic-drops")
    assert p and not p.get("inactive") and not p.get("info_only")


def test_injection_table_carries_authoritative_prices():
    """A client asking "how much?" must not leave the model guessing.

    Live regression 2026-07-20: the NIR Brain Frequency Helmet ($4,997 + $32
    shipping) was quoted to a client as "$754 (includes $132 shipping)" — and
    the $132 was the Healing Tools Package's shipping figure leaking across
    products. The table carried URLs but no prices, so the model invented one.
    """
    directive = app.build_product_directive(
        query_text="how much is the NIR Brain Frequency Helmet")
    row = [l for l in directive.splitlines()
           if l.strip().startswith("• NIR Brain Frequency Helmet ")]
    assert row, "helmet missing from table"
    assert "$4,997.00 list" in row[0], row[0]


def test_priced_rows_dominate_the_table():
    """Nearly every row should carry a price; a bare row is where invention starts."""
    directive = app.build_product_directive(query_text="what do you recommend")
    rows = [l for l in directive.splitlines()
            if l.strip().startswith("•") and "http" in l]
    priced = [l for l in rows if " list" in l]
    assert len(rows) > 100
    assert len(priced) / len(rows) > 0.9, f"only {len(priced)}/{len(rows)} rows priced"


def test_prompt_forbids_inventing_prices():
    prompt = app.get_system_prompt("self-healing")
    assert "NEVER INVENT A PRICE" in prompt
    # the two specific failure modes that produced the $754/$132 answer
    assert "retrieved snippet" in prompt
    assert "carry a price or shipping figure over from another product" in prompt


def test_inserted_word_does_not_break_the_match():
    """Live regression 2026-07-20: the storefront says "Harmony Soft Laser",
    the catalog says "Harmony Laser". The inserted "Soft" split [harmony,
    laser] into contiguous runs of 1, below the 2-token floor — so the product
    never entered the link table, and the model attached CLARITY's URL to it.
    A link that opens the wrong product page is worse than no link.
    """
    for phrasing in ("tell me about the Harmony Soft Laser 172 Hz",
                     "how much is the harmony soft laser"):
        d = app.build_product_directive(query_text=phrasing)
        row = [l for l in d.splitlines() if "harmony-laser" in l]
        assert row, phrasing
        assert "clarity" not in row[0], row[0]


def test_gap_tolerance_stays_tight():
    """Gaps must not let unrelated words bridge a match."""
    far = ("harmony is the goal of every protocol, and much later in a very "
           "different sentence we might mention a laser")
    assert "harmony-laser" not in str(app._catalog_link_matches(far, {}))


def test_prompt_forbids_borrowing_another_products_url():
    prompt = app.get_system_prompt("self-healing")
    assert "A TABLE URL BELONGS TO ITS OWN PRODUCT ONLY" in prompt


def test_membership_routes_to_in_app_not_a_competing_sku():
    """Glen 2026-07-20: the GrooveKart $497 membership must NOT become a catalog
    SKU competing with the in-app $99/mo membership. The bot routes to the
    in-app page; with no table price it says the page shows pricing rather than
    quoting a stale figure."""
    d = app.build_product_directive(query_text="how much is a monthly membership")
    row = [l for l in d.splitlines() if l.strip().startswith("• Monthly Membership ")]
    assert row and "/membership" in row[0], row
    assert "$" not in row[0], "membership row must carry no price"


def test_consult_only_high_ticket_never_offers_a_price_or_buy_link():
    """$28k-$100k Healing Oasis tiers + ASH Training: consult-only. The bot must
    route to a conversation, never quote a price or a buy link (the fabrication
    risk on a six-figure item is the worst case)."""
    d = app.build_product_directive(query_text="tell me about the Consultant Healing Oasis")
    for name in ("Consultant Healing Oasis", "Home Healing Oasis",
                 "Enterprise Healing Oasis", "Travel Healing Oasis"):
        row = [l for l in d.splitlines() if l.strip().startswith(f"• {name} ")]
        assert row, name
        assert "DESCRIBE-ONLY" in row[0] and "CONSULT ONLY" in row[0], row[0]
        assert "http" not in row[0].split("CONSULT ONLY")[0], f"{name} carries a link"


# ── services migration groups 2/3/7 (Glen 2026-07-20) ───────────────────────
# EVOX routes to the in-app booking; Remedy Match is redundant with membership;
# the high-ticket programs are consult-only; Infoceutical Sequence is retired.
def _directive(q):
    return app.build_product_directive(snippets_text="", query_text=q)

def test_evox_routes_to_the_in_app_booking():
    assert "https://illtowell.com/evox" in _directive("how do I book EVOX?")

def test_remedy_match_routes_to_membership_not_a_gk_subscription():
    d = _directive("what is the link for Remedy Match?")
    assert "https://illtowell.com/membership" in d
    assert "remedymatch.com/resources" not in d

def test_consult_only_services_offer_no_price_or_buy_link():
    for q in ("Formulation service",
              "Intensive Biofield Balancing Program",
              "Transgenerational Perception Reframing"):
        d = _directive(q)
        assert "CONSULT ONLY" in d, q
        assert "/begin/product/" not in d.split(q.split()[0])[-1][:120] or "CONSULT" in d

def test_infoceutical_sequence_is_retired_not_linkable():
    d = _directive("Infoceutical Sequence & Support")
    assert "DISCONTINUED" in d
    assert "resources/442" not in d


def test_core_concierge_is_not_yet_available_no_price_no_link():
    """Glen 2026-07-20: CoRe Concierge is coming but not available yet — not
    discontinued, not consult-arrangeable now. Bot must not quote a price or
    offer a buy link ($997/$9,997 tiers must never surface)."""
    d = app.build_product_directive(query_text="what is CoRe Concierge and how much")
    row = [l for l in d.splitlines() if l.strip().startswith("• CoRe Concierge ")]
    assert row, "CoRe Concierge missing from table"
    assert "NOT YET AVAILABLE" in row[0]
    assert "http" not in row[0].split("NOT YET AVAILABLE")[0], "carries a link"
    assert "997" not in row[0], "must not surface a price"
