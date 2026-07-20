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


def test_public_surface_module_never_opens_its_own_connection():
    """public_surface builds payloads; it never opens a connection itself --
    callers pass in `cx`. NOT an "avoids importing sqlite3" check: the module
    does `import sqlite3` (for the sqlite3.Error/sqlite3.OperationalError
    types used in its own except clauses). The real invariant is that it
    never calls sqlite3.connect(...) on its own."""
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


def test_public_only_drops_forbidden_keys():
    """MUTATION TEST: prove _public_only actually filters, not just that the
    happy-path view happens to omit forbidden keys. Feed it a dict containing
    both allowed and forbidden keys directly and assert the forbidden ones
    are dropped while the allowed ones survive."""
    view = {
        "slug": "prof-jane-doe",
        "practitioner_name": "Jane Doe",
        "wallet_balance_cents": 99999,
        "margin": 0.4,
        "email": "jane@example.com",
    }
    result = ps._public_only(view, ps.PRACTITIONER_PUBLIC_FIELDS)
    assert "wallet_balance_cents" not in result
    assert "margin" not in result
    assert "email" not in result
    assert result["slug"] == "prof-jane-doe"
    assert result["practitioner_name"] == "Jane Doe"


def test_public_only_is_the_guard_build_practitioner_storefront_relies_on():
    """Directly show _public_only is a real, general-purpose filter (drops a
    key that isn't in the whitelist for an arbitrary allowed-set), tying it
    to the guarantee build_practitioner_storefront's docstring claims."""
    result = ps._public_only(
        {"slug": "x", "not_whitelisted": "should be dropped"},
        ps.PRACTITIONER_PUBLIC_FIELDS,
    )
    assert result == {"slug": "x"}


def test_storefront_includes_profit_disclosure():
    """16 CFR 255.5, Example 4 — consumers don't expect a physician endorser to
    receive a percentage of gross product sales. (Verified against the CFR text;
    Sec. 255.5 has no lettered subsections, so the earlier "255.5(b)" was wrong.)
    Default ON and inline, unlike the industry norm."""
    cx = _cx_with_affiliate()
    view = ps.build_practitioner_storefront(cx, "prof-jane-doe")
    assert view["profit_disclosure"]
    assert "%" not in view["profit_disclosure"], "disclose the fact, never a rate"


def test_storefront_honors_a_narrowed_whitelist():
    """Wiring test: build_practitioner_storefront must apply
    PRACTITIONER_PUBLIC_FIELDS at call time via _public_only, not just
    happen to return an already-whitelist-shaped dict. Since `view` is
    hand-built from literal keys equal to the full whitelist, no fixture
    can inject a forbidden key through the DB row (that's the whole reason
    the original mutation test was vacuous). Instead, shrink the whitelist
    itself: if the guard is wired up, the output shrinks with it. If the
    guard were bypassed (bare `return view`), the output would ignore the
    narrowed whitelist entirely and still include the dropped key."""
    cx = _cx_with_affiliate()
    original = ps.PRACTITIONER_PUBLIC_FIELDS
    try:
        ps.PRACTITIONER_PUBLIC_FIELDS = frozenset(original - {"practitioner_name"})
        view = ps.build_practitioner_storefront(cx, "prof-jane-doe")
        assert "practitioner_name" not in view
    finally:
        ps.PRACTITIONER_PUBLIC_FIELDS = original


def test_storefront_returns_none_for_non_approved_affiliate():
    """A pending (not yet approved) affiliate must be treated as unknown —
    same as a slug that doesn't exist at all."""
    cx = _cx_with_affiliate(status="pending")
    assert ps.build_practitioner_storefront(cx, "prof-jane-doe") is None


from dashboard import share_header as sh


def test_share_header_none_for_unknown_slug():
    cx = _cx_with_affiliate()
    sh.init_share_headers_table(cx)
    assert ps.build_share_header(cx, "nope") is None


def test_share_header_none_when_not_approved():
    cx = _cx_with_affiliate()
    sh.init_share_headers_table(cx)
    sh.upsert_header(cx, "jane@example.com", "Jane", "Six months in.")
    assert ps.build_share_header(cx, "prof-jane-doe") is None


def test_share_header_returns_approved_header():
    cx = _cx_with_affiliate()
    sh.init_share_headers_table(cx)
    sh.upsert_header(cx, "jane@example.com", "Jane", "Six months in.")
    sh.approve(cx, "jane@example.com")
    hdr = ps.build_share_header(cx, "prof-jane-doe")
    assert hdr == {"display_name": "Jane", "body": "Six months in."}


def test_share_header_exposes_only_two_keys():
    cx = _cx_with_affiliate()
    sh.init_share_headers_table(cx)
    sh.upsert_header(cx, "jane@example.com", "Jane", "Six months in.")
    sh.approve(cx, "jane@example.com")
    assert set(ps.build_share_header(cx, "prof-jane-doe")) == {"display_name", "body"}


def test_share_header_honors_a_narrowed_whitelist():
    """Wiring test: build_share_header must apply SHARE_HEADER_PUBLIC_FIELDS
    at call time, not just happen to return an already-whitelist-shaped dict.
    get_approved()'s own SELECT already returns exactly the two whitelisted
    columns, so no fixture can inject a forbidden key through the DB row
    (that's why the original mutation test was vacuous). Instead, shrink the
    whitelist itself: if the guard is wired up, the output shrinks with it.
    If the guard were bypassed (bare `return dict(hdr)`), the output would
    ignore the narrowed whitelist entirely and still include 'body'.
    Mirrors test_storefront_honors_a_narrowed_whitelist."""
    cx = _cx_with_affiliate()
    sh.init_share_headers_table(cx)
    sh.upsert_header(cx, "jane@example.com", "Jane", "Six months in.")
    sh.approve(cx, "jane@example.com")
    original = ps.SHARE_HEADER_PUBLIC_FIELDS
    try:
        ps.SHARE_HEADER_PUBLIC_FIELDS = frozenset({"display_name"})
        hdr = ps.build_share_header(cx, "prof-jane-doe")
        assert "body" not in hdr
    finally:
        ps.SHARE_HEADER_PUBLIC_FIELDS = original


def test_share_header_missing_table_fails_closed():
    """A fresh deployment where init_share_headers_table has never been
    called must not 500 the public page — build_share_header should fail
    closed (return None) rather than let sqlite3.OperationalError propagate."""
    cx = _cx_with_affiliate()
    # Deliberately do NOT call sh.init_share_headers_table(cx).
    assert ps.build_share_header(cx, "prof-jane-doe") is None
