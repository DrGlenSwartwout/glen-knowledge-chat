"""Guard for the owner-only QBO read endpoint (/api/qbo/query).

The endpoint exists so reconciliation can SEE QBO rows (Payment/Deposit/etc)
from prod, where the refresh token rotates safely. These tests pin the rule that
keeps it read-only, so a later edit can't quietly widen it into a write path."""
from dashboard.qbo_query_guard import sanitize, DEFAULT_CAP


def test_select_passes_and_gets_a_result_cap():
    q, err = sanitize("SELECT * FROM Payment")
    assert err is None
    assert q == f"SELECT * FROM Payment MAXRESULTS {DEFAULT_CAP}"


def test_existing_maxresults_is_respected():
    q, err = sanitize("SELECT * FROM Payment MAXRESULTS 5")
    assert err is None
    assert q.count("MAXRESULTS") == 1
    assert q.endswith("MAXRESULTS 5")


def test_empty_query_rejected():
    assert sanitize("")[1] == "missing q"
    assert sanitize(None)[1] == "missing q"
    assert sanitize("   ")[1] == "missing q"


def test_non_select_rejected():
    for bad in ("UPDATE Customer SET Name='x'",
                "DELETE FROM Payment",
                "INSERT INTO Payment (Id) VALUES (1)",
                "select_something_else FROM X"):
        q, err = sanitize(bad)
        assert q is None
        assert err == "only SELECT queries are allowed", bad


def test_stacked_statements_rejected():
    q, err = sanitize("SELECT * FROM Payment; DELETE FROM Payment")
    assert q is None
    assert err == "multiple statements not allowed"


def test_trailing_semicolon_is_allowed_and_stripped():
    q, err = sanitize("SELECT * FROM Payment;")
    assert err is None
    assert ";" not in q


def test_leading_whitespace_and_case_insensitive():
    q, err = sanitize("   select * from Payment   ")
    assert err is None
    assert q.lower().startswith("select")


def test_cap_is_overridable():
    q, _err = sanitize("SELECT * FROM Payment", cap=7)
    assert q.endswith("MAXRESULTS 7")
