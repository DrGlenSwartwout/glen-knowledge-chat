import importlib, sqlite3, sys
from pathlib import Path
import pytest


def test_prepay_grant_columns_present():
    import app as appmod
    cx = sqlite3.connect(":memory:")
    cx.execute("CREATE TABLE IF NOT EXISTS prepay_term_grants (session_id TEXT PRIMARY KEY, email TEXT, tier_key TEXT, granted_at TEXT)")
    appmod._ensure_prepay_grant_columns(cx)
    cols = {r[1] for r in cx.execute("PRAGMA table_info(prepay_term_grants)")}
    assert {"attributed_practitioner_id", "practitioner_share_consent", "term_end"} <= cols


# ---------------------------------------------------------------------------
# Task 2: attributed prepay-term fulfilment stamps the grant + credits the
# doctor ONCE on the full prepaid lump (reusing the #565 care_share + wallet
# machinery). Public (non-dispensary) prepay is unchanged: no credit, grant
# still created.
# ---------------------------------------------------------------------------

def _load_app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def _fresh(app_module, monkeypatch, tmp_path):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    monkeypatch.setattr(app_module, "REPERTOIRE_ENABLED", False, raising=False)
    monkeypatch.setattr(app_module, "PUBLIC_BASE_URL", "", raising=False)
    monkeypatch.setattr(app_module, "_ingest_order", lambda **kw: None, raising=False)
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(app_module, "_notify_first_cc_signup", lambda *a, **k: None, raising=False)
    with sqlite3.connect(db) as cx:
        app_module.init_membership_tables(cx)
        cx.commit()
    return db


def _mock_prepay_session(app_module, monkeypatch, *, email, tier_key,
                         dispensary_pid=None, share_consent=None):
    """Feed a succeeded prepay_term session through the Stripe helpers, with
    optional dispensary attribution + consent in the metadata."""
    from dashboard import stripe_pay
    md = {"kind": "prepay_term", "email": email, "tier_key": tier_key}
    if dispensary_pid is not None:
        md["dispensary_pid"] = dispensary_pid
    if share_consent is not None:
        md["share_consent"] = share_consent
    monkeypatch.setattr(stripe_pay, "get_session",
        lambda s: {"metadata": dict(md), "payment_intent": "pi_1"})
    monkeypatch.setattr(stripe_pay, "get_payment_intent",
        lambda pi: {"customer": "cus_1", "payment_method": "pm_1", "status": "succeeded"})


def _patch_care_share(monkeypatch, *, modules=12):
    """Patch modules_for_practitioner to a fixed cert level + earn_care_share to
    a recorder (no Supabase / wallet DB). Returns the recorder list."""
    from dashboard import care_share, wallet
    monkeypatch.setattr(care_share, "modules_for_practitioner", lambda pid: modules)
    rec = []
    monkeypatch.setattr(wallet, "earn_care_share",
        lambda pid, cents, *, event_ref: rec.append((pid, cents, event_ref)) or cents)
    return rec


def _grant_row(db, session_id):
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        return cx.execute(
            "SELECT * FROM prepay_term_grants WHERE session_id=?", (session_id,)).fetchone()


def test_attributed_prepay_credits_doctor_on_lump(monkeypatch, tmp_path):
    """A doctor's attributed 12-month prepay ($990 lump) stamps the grant row
    with attribution + consent + term_end AND credits the doctor once, at the
    cert-scaled share of the FULL lump (share_cents(99000, 12) = $495)."""
    from dashboard import care_share
    A = _load_app()
    db = _fresh(A, monkeypatch, tmp_path)
    rec = _patch_care_share(monkeypatch, modules=12)

    _mock_prepay_session(A, monkeypatch, email="pat@x.com", tier_key="12mo",
                         dispensary_pid="prac-42", share_consent="1")
    res = A._fulfill_prepay_term("cs_1")
    assert res.get("ok") is True

    expected = care_share.share_cents(99000, 12)
    assert expected == 49500
    assert rec == [("prac-42", expected, "care_share:prepay:cs_1")]

    row = _grant_row(db, "cs_1")
    assert row["attributed_practitioner_id"] == "prac-42"
    assert row["practitioner_share_consent"] == 1
    assert row["term_end"]  # non-null calendar term end


def test_public_prepay_no_dispensary_no_credit(monkeypatch, tmp_path):
    """A public (front-door) prepay with NO dispensary_pid fires NO credit and
    still grants the term exactly as before."""
    A = _load_app()
    db = _fresh(A, monkeypatch, tmp_path)
    rec = _patch_care_share(monkeypatch, modules=12)

    _mock_prepay_session(A, monkeypatch, email="solo@x.com", tier_key="12mo")
    res = A._fulfill_prepay_term("cs_pub")
    assert res.get("ok") is True

    assert rec == []  # no doctor credited
    row = _grant_row(db, "cs_pub")
    assert row is not None  # term still granted
    assert row["attributed_practitioner_id"] in (None, "")


def test_attributed_prepay_idempotent(monkeypatch, tmp_path):
    """Redirect + webhook both fulfil the same session: the credit fires at most
    once (only the session-claim winner credits)."""
    A = _load_app()
    db = _fresh(A, monkeypatch, tmp_path)
    rec = _patch_care_share(monkeypatch, modules=12)

    _mock_prepay_session(A, monkeypatch, email="pat2@x.com", tier_key="12mo",
                         dispensary_pid="prac-7", share_consent="1")
    res1 = A._fulfill_prepay_term("cs_dup")
    res2 = A._fulfill_prepay_term("cs_dup")
    assert res1.get("ok") is True
    assert res2.get("ok") is True
    assert res2.get("already") is True

    assert len(rec) == 1
    assert rec[0][0] == "prac-7"
    assert rec[0][2] == "care_share:prepay:cs_dup"
