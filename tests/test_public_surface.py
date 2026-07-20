import inspect
from dashboard import public_surface as ps


def test_build_demo_view_takes_no_connection():
    """The structural guarantee: no cx parameter means no code path to real data."""
    sig = inspect.signature(ps.build_demo_view)
    assert list(sig.parameters) == []


def test_demo_view_is_marked_as_sample():
    view = ps.build_demo_view()
    assert view["sample"] is True


def test_demo_view_has_expected_shape():
    view = ps.build_demo_view()
    for key in ("greeting", "phase", "practitioner", "layers",
                "findings", "orders", "pricing", "body_map"):
        assert key in view, f"missing {key}"
    assert isinstance(view["layers"], list) and view["layers"]
    assert isinstance(view["findings"], list) and view["findings"]


def test_demo_view_is_a_copy_not_the_fixture():
    """Mutating a returned view must not poison the next caller."""
    first = ps.build_demo_view()
    first["findings"].append({"name": "injected", "note": "x"})
    second = ps.build_demo_view()
    assert all(f["name"] != "injected" for f in second["findings"])


def test_demo_module_does_not_import_sqlite():
    """public_surface builds payloads; it never opens a connection itself."""
    src = inspect.getsource(ps)
    assert "sqlite3.connect" not in src


import sqlite3


def _cx_with_affiliate(slug="prof-jane-doe", name="Jane Doe",
                       organization="Doe Wellness", status="approved"):
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    cx.executescript("""
      CREATE TABLE affiliate_signups (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, name TEXT,
        email TEXT, organization TEXT DEFAULT '', website TEXT DEFAULT '',
        promo_method TEXT DEFAULT '', slug TEXT, token TEXT,
        status TEXT DEFAULT 'approved', notes TEXT DEFAULT '',
        referred_by TEXT DEFAULT '', short_url TEXT DEFAULT '');
    """)
    cx.execute(
        "INSERT INTO affiliate_signups (created_at,name,email,organization,slug,token,status)"
        " VALUES ('2026-01-01',?,?,?,?,'tok',?)",
        (name, "jane@example.com", organization, slug, status))
    cx.commit()
    return cx


def test_storefront_returns_none_for_unknown_slug():
    cx = _cx_with_affiliate()
    assert ps.build_practitioner_storefront(cx, "nope-not-real") is None


def test_storefront_returns_public_identity():
    cx = _cx_with_affiliate()
    view = ps.build_practitioner_storefront(cx, "prof-jane-doe")
    assert view["practitioner_name"] == "Jane Doe"
    assert view["practice_name"] == "Doe Wellness"
    assert view["slug"] == "prof-jane-doe"


def test_storefront_keys_are_all_whitelisted():
    cx = _cx_with_affiliate()
    view = ps.build_practitioner_storefront(cx, "prof-jane-doe")
    assert set(view) <= ps.PRACTITIONER_PUBLIC_FIELDS


def test_storefront_whitelist_excludes_commercial_fields():
    forbidden = {"wallet_balance_cents", "margin", "markup", "wholesale_price",
                 "revenue", "order_volume", "dispensary_credit_total_cents",
                 "application_status", "resale_license_number", "email",
                 "token", "patients", "clients"}
    assert not (ps.PRACTITIONER_PUBLIC_FIELDS & forbidden)


def test_storefront_drops_an_injected_forbidden_field():
    """MUTATION TEST: prove the whitelist actually filters, not just that the
    happy path happens to omit the field. Inject a commercial column into the
    source table and assert it does not reach the output."""
    cx = _cx_with_affiliate()
    cx.execute("ALTER TABLE affiliate_signups ADD COLUMN wallet_balance_cents INTEGER DEFAULT 99999")
    cx.commit()
    view = ps.build_practitioner_storefront(cx, "prof-jane-doe")
    assert "wallet_balance_cents" not in view
    assert 99999 not in view.values()


def test_storefront_includes_profit_disclosure():
    """16 CFR 255.5(b) Ex.4 — disclosure duty is stronger on surfaces that do
    not look like ads. Default ON and inline, unlike the industry norm."""
    cx = _cx_with_affiliate()
    view = ps.build_practitioner_storefront(cx, "prof-jane-doe")
    assert view["profit_disclosure"]
    assert "%" not in view["profit_disclosure"], "disclose the fact, never a rate"
