# tests/test_portal_reorder_module.py
"""Task 5: portal reorder module payload — `reorder` (real portal-channel order
history, repertoire-priced), `membership_upsell` (personalized non-member
pitch), and `locked_rows` (forward-framed grayed rows). Mirrors the Flask
test-client harness in tests/test_client_portal_routes.py + the direct-order-
insert helper from tests/test_repertoire_wiring.py."""
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod._init_auth_tables()
    monkeypatch.setattr(appmod, "REPERTOIRE_ENABLED", True)
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def _seed_portal(appmod, email, name="Client", content=None):
    from dashboard import client_portal as cp
    content = content or {"greeting": "hi", "video": {}, "layers": []}
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.init_client_portal_table(cx)
    token, _ = cp.upsert_portal(cx, email, name, content)
    cx.close()
    return token


def _seed_order(appmod, *, source, email, slugs_qty, status="done", days_ago=1,
                 unit_cents=6997, external_ref=None):
    """Insert an orders row directly so tests control created_at/source
    precisely. slugs_qty: list of (slug, qty) tuples."""
    from dashboard.orders import init_orders_table
    created = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    items = [{"slug": s, "qty": q, "name": s, "unit_cents": unit_cents}
             for s, q in slugs_qty]
    ref = external_ref or f"o-{source}-{email}-{days_ago}-{slugs_qty[0][0]}"
    with sqlite3.connect(appmod.LOG_DB) as cx:
        init_orders_table(cx)
        cx.execute(
            "INSERT INTO orders (created_at, source, external_ref, channel, email, "
            "items_json, total_cents, status) VALUES (?,?,?,?,?,?,?,?)",
            (created, source, ref, "retail", email,
             json.dumps(items), unit_cents * sum(q for _, q in slugs_qty), status))
        cx.commit()


def _seed_purchase_history(appmod, email, rows):
    """Insert purchase_history slice rows directly. rows: list of
    (slug, days_ago, source, source_ref) tuples — the consolidated cross-channel
    record (fmp / groovekart slices) that _portal_reorder_module reads for the
    storefront display + the 'has this client bought before' (is_reorder) oracle."""
    from dashboard import purchase_history as ph
    with sqlite3.connect(appmod.LOG_DB) as cx:
        ph.init_purchase_history_table(cx)
        for slug, days_ago, source, source_ref in rows:
            at = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
            cx.execute(
                "INSERT OR IGNORE INTO purchase_history"
                "(email, slug, purchased_at, source, source_ref) VALUES (?,?,?,?,?)",
                (email.strip().lower(), slug, at, source, str(source_ref)))
        cx.commit()


def _seed_active_membership(appmod, email, *, source="founding"):
    expires = (datetime.utcnow() + timedelta(days=30)).isoformat() + "Z"
    with sqlite3.connect(appmod.LOG_DB) as cx:
        appmod.init_membership_tables(cx)
        cx.execute(
            "INSERT INTO memberships (id, email, granted_at, expires_at, granted_by, source) "
            "VALUES (?,?,?,?,?,?)",
            (f"mem_{email}", email, datetime.utcnow().isoformat() + "Z", expires,
             "test", source))
        cx.commit()


# ── (a) distinct SKUs from portal-channel history ────────────────────────────

def test_reorder_list_has_distinct_skus_from_portal_history(client):
    c, appmod = client
    email = "hist@example.com"
    tok = _seed_portal(appmod, email)
    _seed_order(appmod, source="portal-reorder", email=email,
                slugs_qty=[("nous-energy", 1)], days_ago=5)
    _seed_order(appmod, source="reorder", email=email,
                slugs_qty=[("neuro-magnesium", 2)], days_ago=10)
    # Storefront (GrooveKart) purchase now DOES appear (Glen 2026-07-11 reversal:
    # clients see ALL their purchases — portal + storefront — not portal-only).
    _seed_order(appmod, source="groovekart", email=email,
                slugs_qty=[("terrain-restore", 1)], days_ago=3)
    # Cancelled portal order must still be excluded.
    _seed_order(appmod, source="reorder", email=email,
                slugs_qty=[("nous-energy", 9)], days_ago=1, status="cancelled")

    j = c.get(f"/api/portal/{tok}").get_json()
    slugs = {r["slug"] for r in j["reorder"]}
    assert slugs == {"nous-energy", "neuro-magnesium", "terrain-restore"}
    row = next(r for r in j["reorder"] if r["slug"] == "neuro-magnesium")
    assert row["qty"] == 2
    assert row["name"]
    assert row["regular_cents"] == 6997
    # provenance channel is carried on every row (frontend maps it to a label).
    by_slug = {r["slug"]: r for r in j["reorder"]}
    assert by_slug["nous-energy"]["channel"] == "portal"
    assert by_slug["neuro-magnesium"]["channel"] == "portal"
    assert by_slug["terrain-restore"]["channel"] == "storefront"


def test_storefront_historical_purchase_from_purchase_history_appears(client):
    """A pre-webhook GrooveKart order survives only in the purchase_history
    'groovekart' slice (no orders-table row). It must still show in the portal,
    labeled storefront, and count as a true reorder (it IS in purchase_history)."""
    c, appmod = client
    email = "gkhist@example.com"
    tok = _seed_portal(appmod, email)
    _seed_purchase_history(appmod, email,
                           [("terrain-restore", 200, "groovekart", "gk-#1234")])

    j = c.get(f"/api/portal/{tok}").get_json()
    by_slug = {r["slug"]: r for r in j["reorder"]}
    assert "terrain-restore" in by_slug
    assert by_slug["terrain-restore"]["channel"] == "storefront"
    assert by_slug["terrain-restore"]["is_reorder"] is True


def test_clinical_fmp_purchase_appears_labeled_clinic(client):
    """Glen 2026-07-11 (2nd pass): ALL purchases show in the portal, every
    channel. A clinic/dispensary purchase surviving only in the purchase_history
    'fmp' slice appears, labeled 'clinic', and is a true reorder (in history)."""
    c, appmod = client
    email = "fmpclient@example.com"
    tok = _seed_portal(appmod, email)
    _seed_purchase_history(appmod, email, [("neuro-magnesium", 150, "fmp", "inv-77")])

    j = c.get(f"/api/portal/{tok}").get_json()
    by_slug = {r["slug"]: r for r in j["reorder"]}
    assert "neuro-magnesium" in by_slug
    assert by_slug["neuro-magnesium"]["channel"] == "clinic"
    assert by_slug["neuro-magnesium"]["is_reorder"] is True


def test_dispensary_order_appears_labeled_clinic(client):
    """A non-portal, non-storefront orders-table channel (dispensary) is folded
    into the display too, labeled 'clinic', with its real quantity."""
    c, appmod = client
    email = "dispclient@example.com"
    tok = _seed_portal(appmod, email)
    _seed_order(appmod, source="dispensary", email=email,
                slugs_qty=[("nous-energy", 3)], days_ago=8)

    j = c.get(f"/api/portal/{tok}").get_json()
    by_slug = {r["slug"]: r for r in j["reorder"]}
    assert by_slug["nous-energy"]["channel"] == "clinic"
    assert by_slug["nous-energy"]["qty"] == 3


def test_is_reorder_reflects_purchase_history_membership(client):
    """`is_reorder` (the 'Reorder' CTA gate) is true iff the SKU is in the
    client's purchase_history — a portal purchase absent from purchase_history is
    NOT framed as a reorder; one present in it (any slice, incl. fmp) is."""
    c, appmod = client
    email = "reorderflag@example.com"
    tok = _seed_portal(appmod, email)
    # nous-energy: portal purchase, and previously bought (fmp slice) -> reorder.
    _seed_order(appmod, source="reorder", email=email,
                slugs_qty=[("nous-energy", 1)], days_ago=5)
    _seed_purchase_history(appmod, email, [("nous-energy", 400, "fmp", "inv-1")])
    # neuro-magnesium: portal purchase only, never in purchase_history -> not a reorder.
    _seed_order(appmod, source="portal-reorder", email=email,
                slugs_qty=[("neuro-magnesium", 1)], days_ago=6)

    j = c.get(f"/api/portal/{tok}").get_json()
    by_slug = {r["slug"]: r for r in j["reorder"]}
    assert by_slug["nous-energy"]["is_reorder"] is True
    assert by_slug["neuro-magnesium"]["is_reorder"] is False


def test_portal_wins_when_slug_bought_on_both_channels(client):
    """A SKU bought both on the portal and the storefront shows exactly once,
    labeled as the portal purchase (portal takes precedence in the dedupe)."""
    c, appmod = client
    email = "bothchannels@example.com"
    tok = _seed_portal(appmod, email)
    _seed_order(appmod, source="reorder", email=email,
                slugs_qty=[("nous-energy", 2)], days_ago=5)
    _seed_order(appmod, source="groovekart", email=email,
                slugs_qty=[("nous-energy", 1)], days_ago=3)

    j = c.get(f"/api/portal/{tok}").get_json()
    rows = [r for r in j["reorder"] if r["slug"] == "nous-energy"]
    assert len(rows) == 1
    assert rows[0]["channel"] == "portal"
    assert rows[0]["qty"] == 2  # from the portal order


def test_storefront_display_equals_charge_for_member(client):
    """display == charge invariant holds for a NEWLY-included storefront slug:
    a member's your_cents in the payload matches what the real checkout engine
    (_price_cart) bills for that same slug."""
    c, appmod = client
    email = "gkmember@example.com"
    tok = _seed_portal(appmod, email)
    _seed_active_membership(appmod, email)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        appmod.repertoire.init_repertoire_table(cx)
        appmod.repertoire.add_skus(cx, email, ["terrain-restore"])
    _seed_order(appmod, source="groovekart", email=email,
                slugs_qty=[("terrain-restore", 1)], days_ago=4)

    j = c.get(f"/api/portal/{tok}").get_json()
    row = next(r for r in j["reorder"] if r["slug"] == "terrain-restore")
    assert row["channel"] == "storefront"
    assert row["is_member_price"] is True
    checkout_price = appmod._price_cart(
        [{"slug": "terrain-restore", "qty": 1}],
        ship={"country": "US", "state": "TX"}, email=email,
    )["priced"]["lines"][0]["line_total_cents"]
    assert row["your_cents"] == checkout_price


def test_reorder_dedupes_keeping_most_recent_qty(client):
    c, appmod = client
    email = "dedupe@example.com"
    tok = _seed_portal(appmod, email)
    _seed_order(appmod, source="portal-reorder", email=email,
                slugs_qty=[("nous-energy", 1)], days_ago=20)
    _seed_order(appmod, source="portal-reorder", email=email,
                slugs_qty=[("nous-energy", 3)], days_ago=2)  # most recent

    j = c.get(f"/api/portal/{tok}").get_json()
    rows = [r for r in j["reorder"] if r["slug"] == "nous-energy"]
    assert len(rows) == 1
    assert rows[0]["qty"] == 3


# ── (b) member repertoire pricing matches the real pricing engine ───────────

def test_member_repertoire_price_below_regular_and_matches_price_cart(client):
    c, appmod = client
    email = "member@example.com"
    tok = _seed_portal(appmod, email)
    _seed_active_membership(appmod, email)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        appmod.repertoire.init_repertoire_table(cx)
        appmod.repertoire.add_skus(cx, email, ["neuro-magnesium"])
    _seed_order(appmod, source="reorder", email=email,
                slugs_qty=[("neuro-magnesium", 1)], days_ago=5)

    j = c.get(f"/api/portal/{tok}").get_json()
    row = next(r for r in j["reorder"] if r["slug"] == "neuro-magnesium")
    assert row["is_member_price"] is True
    assert row["in_repertoire"] is True
    assert row["your_cents"] < row["regular_cents"]

    # Must equal the SAME pricing engine other checkout paths in this app use
    # for a repertoire member (Task 4's _price_cart), not a hand-rolled number.
    checkout_price = appmod._price_cart(
        [{"slug": "neuro-magnesium", "qty": 1}],
        ship={"country": "US", "state": "TX"}, email=email,
    )["priced"]["lines"][0]["line_total_cents"]
    assert row["your_cents"] == checkout_price


def test_non_member_pays_regular_price(client):
    c, appmod = client
    email = "nonmember@example.com"
    tok = _seed_portal(appmod, email)
    _seed_order(appmod, source="reorder", email=email,
                slugs_qty=[("neuro-magnesium", 1)], days_ago=5)

    j = c.get(f"/api/portal/{tok}").get_json()
    row = next(r for r in j["reorder"] if r["slug"] == "neuro-magnesium")
    assert row["your_cents"] == row["regular_cents"] == 6997
    assert row["is_member_price"] is False


# ── (c) membership_upsell for a non-member ───────────────────────────────────

def test_membership_upsell_savings_for_non_member(client):
    c, appmod = client
    email = "upsell@example.com"
    tok = _seed_portal(appmod, email)
    _seed_order(appmod, source="portal-reorder", email=email,
                slugs_qty=[("neuro-magnesium", 2)], days_ago=5, unit_cents=6997)
    _seed_order(appmod, source="reorder", email=email,
                slugs_qty=[("nous-energy", 1)], days_ago=10, unit_cents=6997)
    # Outside the 30d window — must not count.
    _seed_order(appmod, source="reorder", email=email,
                slugs_qty=[("nous-energy", 5)], days_ago=45, unit_cents=6997)

    j = c.get(f"/api/portal/{tok}").get_json()
    up = j["membership_upsell"]
    assert up["reorders_30d"] == 2
    assert up["spend_30d_cents"] == 6997 * 3  # 2 + 1 units in the 30d window
    assert up["member_would_pay_cents"] < up["spend_30d_cents"]
    assert up["savings_cents"] > 0
    assert up["savings_cents"] == up["spend_30d_cents"] - up["member_would_pay_cents"]
    assert up["net_after_fee_cents"] == appmod._prepay.MONTHLY_ANCHOR_CENTS - up["savings_cents"]
    assert up.get("already_member") is False


def test_membership_upsell_savings_only_counts_ff_products(client):
    """Glen 2026-07: the repertoire discount is FF-only (commit 899fba0). The
    membership_upsell projection must match — a non-FF product in the client's
    last-30-day history contributes ZERO to savings_cents (it stays at regular
    price even as a hypothetical member), only the FF line's real discount
    counts. vitamin-e-spectrum has no qty_pricing flag -> non-FF; nous-energy
    does -> FF."""
    c, appmod = client
    email = "ffupsell@example.com"
    tok = _seed_portal(appmod, email)
    _seed_order(appmod, source="portal-reorder", email=email,
                slugs_qty=[("nous-energy", 1)], days_ago=5, unit_cents=6997)  # FF
    _seed_order(appmod, source="reorder", email=email,
                slugs_qty=[("vitamin-e-spectrum", 1)], days_ago=6, unit_cents=3997)  # non-FF

    j = c.get(f"/api/portal/{tok}").get_json()
    up = j["membership_upsell"]

    from dashboard import pricing as _pricing
    settings = _pricing.load_settings(appmod._pricing_settings())
    ff_hyp_unit = appmod._rep_priced_unit_cents(
        appmod._get_product("nous-energy"), repertoire_slugs={"nous-energy"},
        settings=settings)
    expected_savings = 6997 - ff_hyp_unit  # non-FF line contributes 0 savings
    assert up["spend_30d_cents"] == 6997 + 3997
    assert up["member_would_pay_cents"] == ff_hyp_unit + 3997
    assert up["savings_cents"] == expected_savings


def test_membership_upsell_zeroed_for_already_member(client):
    c, appmod = client
    email = "already@example.com"
    tok = _seed_portal(appmod, email)
    _seed_active_membership(appmod, email)
    _seed_order(appmod, source="reorder", email=email,
                slugs_qty=[("neuro-magnesium", 1)], days_ago=5)

    j = c.get(f"/api/portal/{tok}").get_json()
    up = j["membership_upsell"]
    assert up["already_member"] is True
    assert up["savings_cents"] == 0


# ── locked_rows: 90-365d-old repertoire-eligible SKUs, forward-framed ────────

def test_locked_rows_tiered_by_age(client):
    c, appmod = client
    email = "locked@example.com"
    tok = _seed_portal(appmod, email)
    # 120 days ago -> reachable only by a 6mo (180d) commitment.
    _seed_order(appmod, source="reorder", email=email,
                slugs_qty=[("nous-energy", 1)], days_ago=120)
    # 300 days ago -> reachable only by a 12mo (365d) commitment.
    _seed_order(appmod, source="reorder", email=email,
                slugs_qty=[("neuro-magnesium", 1)], days_ago=300)
    # 40 days ago -> within reach of even the shortest term; not "locked".
    _seed_order(appmod, source="reorder", email=email,
                slugs_qty=[("terrain-restore", 1)], days_ago=40)
    # 500 days ago -> beyond any term's window; excluded entirely.
    _seed_order(appmod, source="reorder", email=email,
                slugs_qty=[("nous-energy", 1)], days_ago=500,
                external_ref="ancient")

    j = c.get(f"/api/portal/{tok}").get_json()
    by_slug = {r["slug"]: r for r in j["locked_rows"]}
    assert by_slug["nous-energy"]["tier"] == "6mo"
    assert by_slug["neuro-magnesium"]["tier"] == "12mo"
    assert "terrain-restore" not in by_slug


def test_locked_rows_excludes_non_ff_products(client):
    """Glen 2026-07: only FF products (_qty_eligible) can ever carry member
    pricing, so locked_rows (the "unlock member pricing at 6/12mo" pitch)
    must never list a non-FF SKU as unlockable. vitamin-e-spectrum has no
    qty_pricing flag -> non-FF, but was broadly eligible under the old rule
    (not a pure powder) -- the exact case the old broad gate got wrong."""
    c, appmod = client
    email = "lockedff@example.com"
    tok = _seed_portal(appmod, email)
    _seed_order(appmod, source="reorder", email=email,
                slugs_qty=[("nous-energy", 1)], days_ago=120)  # FF
    _seed_order(appmod, source="reorder", email=email,
                slugs_qty=[("vitamin-e-spectrum", 1)], days_ago=120)  # non-FF

    j = c.get(f"/api/portal/{tok}").get_json()
    by_slug = {r["slug"]: r for r in j["locked_rows"]}
    assert "nous-energy" in by_slug
    assert "vitamin-e-spectrum" not in by_slug


def test_locked_rows_excludes_slugs_already_in_repertoire(client):
    c, appmod = client
    email = "unlocked@example.com"
    tok = _seed_portal(appmod, email)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        appmod.repertoire.init_repertoire_table(cx)
        appmod.repertoire.add_skus(cx, email, ["nous-energy"])
    _seed_order(appmod, source="reorder", email=email,
                slugs_qty=[("nous-energy", 1)], days_ago=120)

    j = c.get(f"/api/portal/{tok}").get_json()
    assert all(r["slug"] != "nous-energy" for r in j["locked_rows"])


# ── (d) flag off => byte-identical to pre-Task-5 behavior ───────────────────

def test_flag_off_no_new_keys_and_reorder_items_unchanged(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "REPERTOIRE_ENABLED", False)
    email = "flagoff@example.com"
    tok = _seed_portal(appmod, email, content={
        "greeting": "hi", "video": {}, "layers": [],
        "reorder_items": [{"slug": "nous-energy", "qty": 1}],
    })
    _seed_order(appmod, source="reorder", email=email,
                slugs_qty=[("neuro-magnesium", 1)], days_ago=5)

    j = c.get(f"/api/portal/{tok}").get_json()
    assert "reorder" not in j
    assert "membership_upsell" not in j
    assert "locked_rows" not in j
    assert j["reorder_items"][0]["slug"] == "nous-energy"


def test_fresh_db_no_repertoire_table_does_not_crash(client):
    """Fresh-DB guard: even if the repertoire table isn't there yet (a DB
    created before this feature shipped), the payload build must not throw."""
    c, appmod = client
    email = "freshdb@example.com"
    tok = _seed_portal(appmod, email)
    _seed_order(appmod, source="reorder", email=email,
                slugs_qty=[("nous-energy", 1)], days_ago=5)
    # deliberately do NOT init the repertoire table before hitting the route

    r = c.get(f"/api/portal/{tok}")
    assert r.status_code == 200
    j = r.get_json()
    assert j["reorder"][0]["slug"] == "nous-energy"
