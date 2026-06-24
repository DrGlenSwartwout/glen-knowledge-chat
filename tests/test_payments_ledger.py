"""Unit tests for the Stripe payments ledger model (dashboard/payments.py).

A "payment" row is an order that captured card money via Stripe. Two shapes:
  - one-time checkout (trial / retail funnel): the `stripe_payment_intent`
    COLUMN is set via set_order_stripe_pi.
  - subscription renewal: the order is only ingested when the PaymentIntent
    SUCCEEDS, and the PI id lands in `external_ref` (NOT the column). These must
    NOT be missed by the ledger — they are the recurring revenue.
"""
import sqlite3

from dashboard import orders as O
from dashboard import payments as P
from dashboard import stripe_alerts as SA


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    O.init_orders_table(cx)
    SA.init_stripe_alerts_table(cx)
    return cx


def _mk_order(cx, **kw):
    src = kw.pop("source", "funnel")
    ref = kw.pop("external_ref")
    O.upsert_order(cx, source=src, external_ref=ref,
                   email=kw.pop("email", "a@x.com"), name=kw.pop("name", "A"),
                   items=[{"name": "X", "qty": 1, "desc": "X"}],
                   total_cents=kw.pop("total_cents", 0), address={}, channel="retail")
    row = cx.execute("SELECT id FROM orders WHERE source=? AND external_ref=?",
                     (src, ref)).fetchone()
    return row["id"]


def test_ledger_includes_one_time_and_subscription_excludes_unpaid():
    cx = _cx()
    # one-time card checkout: PI in the column
    oid = _mk_order(cx, source="funnel", external_ref="cs_1", total_cents=7000)
    O.set_order_stripe_pi(cx, oid, "pi_onetime")
    O.set_order_payment(cx, oid, method="card", amount_cents=7000)
    # subscription renewal: PI only in external_ref, column empty
    _mk_order(cx, source="subscription", external_ref="pi_sub", total_cents=9900)
    # a plain unpaid manual invoice: no Stripe money at all -> excluded
    _mk_order(cx, source="manual", external_ref="INV-unpaid", total_cents=5000)

    rows = P.list_payments(cx)
    refs = {r["external_ref"] for r in rows}
    assert refs == {"cs_1", "pi_sub"}  # unpaid manual order dropped


def test_ledger_includes_membership_recurring_charge():
    # The $99/mo live-group-coaching charge ingests source="membership" with the
    # PaymentIntent in external_ref (column unset). It must NOT be dropped.
    cx = _cx()
    _mk_order(cx, source="membership", external_ref="pi_mem99", total_cents=9900)
    # a $0 founding card-vault reservation (seti_ ref) is NOT a captured charge
    _mk_order(cx, source="founding", external_ref="seti_vault", total_cents=0)
    rows = P.list_payments(cx)
    refs = {r["external_ref"] for r in rows}
    assert refs == {"pi_mem99"}
    assert rows[0]["stripe_payment_intent"] == "pi_mem99"


def test_ledger_normalizes_subscription_pi_from_external_ref():
    cx = _cx()
    _mk_order(cx, source="subscription", external_ref="pi_renew99", total_cents=9900)
    [row] = P.list_payments(cx)
    assert row["stripe_payment_intent"] == "pi_renew99"
    assert row["source"] == "subscription"


def test_ledger_amount_falls_back_to_total_when_paid_cents_zero():
    cx = _cx()
    # subscription order: total set, paid_cents defaults to 0
    _mk_order(cx, source="subscription", external_ref="pi_a", total_cents=9900)
    # one-time order: paid_cents recorded explicitly
    oid = _mk_order(cx, source="funnel", external_ref="cs_b", total_cents=7000)
    O.set_order_stripe_pi(cx, oid, "pi_b")
    O.set_order_payment(cx, oid, method="card", amount_cents=6500)  # e.g. discount
    by_ref = {r["external_ref"]: r for r in P.list_payments(cx)}
    assert by_ref["pi_a"]["amount_cents"] == 9900   # fell back to total_cents
    assert by_ref["cs_b"]["amount_cents"] == 6500   # used paid_cents


def test_ledger_source_filter_and_limit():
    cx = _cx()
    _mk_order(cx, source="subscription", external_ref="pi_s1", total_cents=9900)
    oid = _mk_order(cx, source="funnel", external_ref="cs_f1", total_cents=7000)
    O.set_order_stripe_pi(cx, oid, "pi_f1")
    subs = P.list_payments(cx, source="subscription")
    assert [r["external_ref"] for r in subs] == ["pi_s1"]
    assert len(P.list_payments(cx, limit=1)) == 1


def test_ledger_newest_first():
    cx = _cx()
    a = _mk_order(cx, source="funnel", external_ref="cs_old", total_cents=1000)
    O.set_order_stripe_pi(cx, a, "pi_old")
    b = _mk_order(cx, source="funnel", external_ref="cs_new", total_cents=2000)
    O.set_order_stripe_pi(cx, b, "pi_new")
    refs = [r["external_ref"] for r in P.list_payments(cx)]
    assert refs.index("cs_new") < refs.index("cs_old")  # newest (higher id) first


def test_ledger_reports_paid_status_for_captured_rows():
    # Recurring orders keep the orders-table default pay_status='unpaid', yet they
    # only exist because a charge SUCCEEDED. The ledger must not mislabel them.
    cx = _cx()
    _mk_order(cx, source="subscription", external_ref="pi_s", total_cents=9900)
    [row] = P.list_payments(cx)
    assert row["pay_status"] == "paid"


def test_summary_counts_and_totals_captured_set():
    cx = _cx()
    oid = _mk_order(cx, source="funnel", external_ref="cs_1", total_cents=7000)
    O.set_order_stripe_pi(cx, oid, "pi_1")
    O.set_order_payment(cx, oid, method="card", amount_cents=7000)
    _mk_order(cx, source="subscription", external_ref="pi_2", total_cents=9900)
    _mk_order(cx, source="manual", external_ref="INV", total_cents=5000)  # excluded
    s = P.payments_summary(cx)
    assert s["count"] == 2
    assert s["total_cents"] == 7000 + 9900


def test_recent_failures_newest_first_with_fields():
    cx = _cx()
    SA.record_failure(cx, "checkout session create", "card declined",
                      now="2026-06-20T00:00:00+00:00", notify=False)
    SA.record_failure(cx, "subscription charge", "insufficient funds",
                      now="2026-06-24T00:00:00+00:00", notify=False)
    fails = P.recent_failures(cx)
    assert [f["context"] for f in fails] == ["subscription charge", "checkout session create"]
    assert fails[0]["error"] == "insufficient funds"
    assert "created_at" in fails[0] and "emailed_at" in fails[0]


def test_recent_failures_empty_when_no_table_rows():
    cx = _cx()
    assert P.recent_failures(cx) == []


# --- $1 trial backfill -------------------------------------------------------

def _grant(cx, session_id, email):
    cx.execute("CREATE TABLE IF NOT EXISTS biofield_trial_grants "
               "(session_id TEXT PRIMARY KEY, email TEXT, granted_at TEXT)")
    cx.execute("INSERT OR IGNORE INTO biofield_trial_grants (session_id, email, granted_at) "
               "VALUES (?,?,?)", (session_id, email, "2026-06-01T00:00:00Z"))
    cx.commit()


def test_backfill_creates_orders_for_paid_trials():
    cx = _cx()
    _grant(cx, "cs_a", "a@x.com")
    _grant(cx, "cs_b", "b@x.com")
    sessions = {"cs_a": {"payment_intent": "pi_a", "amount_total": 100},
                "cs_b": {"payment_intent": "pi_b", "amount_total": 100}}
    res = P.backfill_trial_orders(cx, lambda sid: sessions[sid])
    assert res["created"] == 2
    pays = P.list_payments(cx)
    assert {p["external_ref"] for p in pays} == {"pi_a", "pi_b"}
    assert all(p["source"] == "biofield_trial" and p["amount_cents"] == 100
               and p["pay_status"] == "paid" for p in pays)


def test_backfill_idempotent_skips_existing():
    cx = _cx()
    _grant(cx, "cs_a", "a@x.com")
    s = {"cs_a": {"payment_intent": "pi_a", "amount_total": 100}}
    P.backfill_trial_orders(cx, lambda sid: s[sid])
    res = P.backfill_trial_orders(cx, lambda sid: s[sid])
    assert res["created"] == 0 and res["skipped"] == 1
    assert len(P.list_payments(cx)) == 1


def test_backfill_skips_unpaid_and_survives_fetch_errors():
    cx = _cx()
    _grant(cx, "cs_unpaid", "u@x.com")
    _grant(cx, "cs_err", "e@x.com")
    _grant(cx, "cs_ok", "o@x.com")
    def fetch(sid):
        if sid == "cs_err":
            raise RuntimeError("stripe down")
        if sid == "cs_unpaid":
            return {"payment_intent": None}
        return {"payment_intent": "pi_ok", "amount_total": 100}
    res = P.backfill_trial_orders(cx, fetch)
    assert res["created"] == 1 and res["failed"] == 1 and res["unpaid"] == 1
    assert {p["external_ref"] for p in P.list_payments(cx)} == {"pi_ok"}


def test_backfill_falls_back_to_trial_amount_when_missing():
    cx = _cx()
    _grant(cx, "cs_a", "a@x.com")
    P.backfill_trial_orders(cx, lambda sid: {"payment_intent": "pi_a"})  # no amount
    [p] = P.list_payments(cx)
    assert p["amount_cents"] == P.TRIAL_AMOUNT_CENTS == 100
