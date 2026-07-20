import sqlite3
import pytest
from dashboard import order_payments as op
from dashboard import orders


@pytest.fixture
def cx():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    orders.init_orders_table(c)
    op.ensure_table(c)
    # a $412.82 order
    orders.upsert_order(c, source="qbo", external_ref="INV-1",
                        email="dana@example.com", total_cents=41282)
    return c


def _oid(cx):
    return cx.execute("SELECT id FROM orders").fetchone()[0]


def test_add_payment_reduces_balance(cx):
    oid = _oid(cx)
    op.add_payment(cx, oid, 22291, "Credit card (Stripe)", source="stripe",
                   external_ref="pi_1")
    op.add_payment(cx, oid, 13100, "Zelle")
    b = op.balance(cx, oid)
    assert b["paid_cents"] == 35391
    assert b["refunded_cents"] == 0
    assert b["invoice_cents"] == 41282
    assert b["balance_cents"] == 5891


def test_overpayment_is_negative_balance(cx):
    oid = _oid(cx)
    op.add_payment(cx, oid, 50000, "Zelle")
    assert op.balance(cx, oid)["balance_cents"] == -8718


def test_stripe_payment_idempotent_on_external_ref(cx):
    oid = _oid(cx)
    op.add_payment(cx, oid, 22291, "Credit card (Stripe)", source="stripe",
                   external_ref="pi_1")
    op.add_payment(cx, oid, 22291, "Credit card (Stripe)", source="stripe",
                   external_ref="pi_1")
    rows = op.list_payments(cx, oid)
    assert len([r for r in rows if r["kind"] == "payment"]) == 1


def test_add_payment_does_not_push_to_qbo(cx, monkeypatch):
    # Was test_add_payment_syncs_to_qbo, which asserted the opposite: that add_payment
    # CREATES a QBO payment. That behaviour is deliberately gone -- every payment already
    # reaches QuickBooks as a bank deposit (cards via eProcessing/PayPal/Authorize.net,
    # Zelle via the BofA feed), so pushing made a duplicate. Six such duplicates had to be
    # deleted by hand on 2026-07-19.
    from dashboard import qbo_billing
    monkeypatch.setattr(qbo_billing, "get_invoice",
                        lambda iid: {"CustomerRef": {"value": "42"}, "Balance": "412.82"})
    calls = {}
    monkeypatch.setattr(qbo_billing, "record_payment",
                        lambda *a, **k: calls.update(called=True) or {"Id": "P9"})
    oid = _oid(cx)
    row = op.add_payment(cx, oid, 13100, "Zelle")
    assert calls == {}                  # QBO was never asked to create anything
    assert row["qbo_txn_id"] is None    # nothing to mirror
    assert row["qbo_sync"] == "synced"  # terminal, so it cannot re-push later


def test_add_payment_links_an_existing_qbo_txn(cx, monkeypatch):
    # Linking survives and stays the preferred move when the QBO id is known.
    from dashboard import qbo_billing
    calls = {}
    monkeypatch.setattr(qbo_billing, "record_payment",
                        lambda *a, **k: calls.update(called=True) or {"Id": "P9"})
    oid = _oid(cx)
    row = op.add_payment(cx, oid, 13100, "Zelle", qbo_txn_id="P9")
    assert calls == {}
    assert row["qbo_txn_id"] == "P9"
    assert row["qbo_sync"] == "synced"


def test_void_excludes_from_balance_and_calls_qbo(cx, monkeypatch):
    from dashboard import qbo_billing
    voided = {}
    monkeypatch.setattr(qbo_billing, "void_payment",
                        lambda txn: voided.update(txn=txn))
    oid = _oid(cx)
    # The ledger no longer CREATES QBO payments (it would duplicate the bank deposit),
    # so a row acquires its txn id by LINKING an existing one. This test is about void
    # behaviour, which is unchanged.
    row = op.add_payment(cx, oid, 13100, "Zelle", qbo_txn_id="P9")
    op.void(cx, row["id"], "keyed wrong amount")
    assert op.balance(cx, oid)["paid_cents"] == 0
    assert voided == {"txn": "P9"}
    # voiding again is a no-op (no second QBO call)
    voided.clear()
    op.void(cx, row["id"], "again")
    assert voided == {}


def test_void_null_txn_skips_qbo(cx, monkeypatch):
    from dashboard import qbo_billing
    called = {"n": 0}
    monkeypatch.setattr(qbo_billing, "void_payment",
                        lambda txn: called.__setitem__("n", called["n"] + 1))
    oid = _oid(cx)
    # No push any more, so an unlinked row lands 'synced' with a NULL txn id (it used
    # to land 'error' when the push could not resolve a QBO invoice). Either way there
    # is no QBO txn, so void must not call QBO.
    row = op.add_payment(cx, oid, 100, "Cash")
    assert row["qbo_sync"] == "synced" and row["qbo_txn_id"] is None
    op.void(cx, row["id"], "typo")
    assert called["n"] == 0


def test_void_qbo_failure_is_flagged_not_swallowed(cx, monkeypatch):
    from dashboard import qbo_billing
    monkeypatch.setattr(qbo_billing, "void_payment",
                        lambda txn: (_ for _ in ()).throw(RuntimeError("QBO down")))
    oid = _oid(cx)
    # txn id comes from LINKING now, not from a push (the ledger no longer creates
    # QBO payments -- that duplicated the bank deposit).
    row = op.add_payment(cx, oid, 13100, "Zelle", qbo_txn_id="P9")
    voided = op.void(cx, row["id"], "keyed wrong amount")
    assert voided["status"] == "void"
    assert voided["qbo_sync"] == "void_error"
    # local status still flips even though QBO reversal failed — balance excludes it
    assert op.balance(cx, oid)["paid_cents"] == 0


def test_resync_repairs_a_flagged_void(cx, monkeypatch):
    from dashboard import qbo_billing
    monkeypatch.setattr(qbo_billing, "void_payment",
                        lambda txn: (_ for _ in ()).throw(RuntimeError("QBO down")))
    oid = _oid(cx)
    # txn id comes from LINKING now, not from a push (the ledger no longer creates
    # QBO payments -- that duplicated the bank deposit).
    row = op.add_payment(cx, oid, 13100, "Zelle", qbo_txn_id="P9")
    voided = op.void(cx, row["id"], "keyed wrong amount")
    assert voided["qbo_sync"] == "void_error"

    # QBO is back up — resync() should repair the flagged row without raising
    monkeypatch.setattr(qbo_billing, "void_payment", lambda txn: None)
    repaired = op.resync(cx, row["id"])
    assert repaired["status"] == "void"
    assert repaired["qbo_sync"] == "void_synced"


def test_refund_reduces_paid_and_guards_overrefund(cx, monkeypatch):
    from dashboard import qbo_billing
    monkeypatch.setattr(qbo_billing, "get_invoice",
                        lambda iid: {"CustomerRef": {"value": "42"}, "Balance": "1"})
    monkeypatch.setattr(qbo_billing, "record_payment", lambda *a, **k: {"Id": "P1"})
    monkeypatch.setattr(qbo_billing, "record_refund", lambda *a, **k: {"Id": "R1"})
    oid = _oid(cx)
    pay = op.add_payment(cx, oid, 13100, "Zelle")
    op.add_refund(cx, oid, 5000, "Zelle", refunds_payment_id=pay["id"])
    b = op.balance(cx, oid)
    assert b["refunded_cents"] == 5000
    assert b["paid_cents"] == 13100
    assert b["balance_cents"] == 41282 - (13100 - 5000)
    # cannot refund more than the payment's un-refunded remainder (8100 left)
    with pytest.raises(ValueError):
        op.add_refund(cx, oid, 9000, "Zelle", refunds_payment_id=pay["id"])


def test_standalone_refund_reduces_paid_and_caps_at_net_paid(cx, monkeypatch):
    from dashboard import qbo_billing
    monkeypatch.setattr(qbo_billing, "get_invoice",
                        lambda iid: {"CustomerRef": {"value": "42"}, "Balance": "1"})
    monkeypatch.setattr(qbo_billing, "record_payment", lambda *a, **k: {"Id": "P1"})
    monkeypatch.setattr(qbo_billing, "record_refund", lambda *a, **k: {"Id": "R1"})
    oid = _oid(cx)
    op.add_payment(cx, oid, 10000, "Zelle")
    # standalone refund — no refunds_payment_id — reduces net paid
    op.add_refund(cx, oid, 4000, "Zelle")
    b = op.balance(cx, oid)
    assert b["refunded_cents"] == 4000
    assert b["paid_cents"] == 10000
    # net paid is now 6000 — a further standalone refund is capped there
    with pytest.raises(ValueError):
        op.add_refund(cx, oid, 7000, "Zelle")


def test_mixed_standalone_and_payment_tied_refund_respects_net_paid(cx, monkeypatch):
    from dashboard import qbo_billing
    monkeypatch.setattr(qbo_billing, "get_invoice",
                        lambda iid: {"CustomerRef": {"value": "42"}, "Balance": "1"})
    monkeypatch.setattr(qbo_billing, "record_payment", lambda *a, **k: {"Id": "P1"})
    monkeypatch.setattr(qbo_billing, "record_refund", lambda *a, **k: {"Id": "R1"})
    oid = _oid(cx)
    # $100 payment A
    pay_a = op.add_payment(cx, oid, 10000, "Zelle")
    # a $50 standalone refund succeeds — net paid drops to $50
    op.add_refund(cx, oid, 5000, "Zelle")
    assert op.balance(cx, oid)["refunded_cents"] == 5000
    # a payment-tied refund of $80 against A would need $80 of net paid, but
    # only $50 remains — must be blocked even though A itself has $100 un-refunded
    with pytest.raises(ValueError):
        op.add_refund(cx, oid, 8000, "Zelle", refunds_payment_id=pay_a["id"])


def test_card_refund_calls_stripe_noncard_does_not(cx, monkeypatch):
    from dashboard import qbo_billing, stripe_pay
    monkeypatch.setattr(qbo_billing, "get_invoice",
                        lambda iid: {"CustomerRef": {"value": "42"}, "Balance": "1"})
    monkeypatch.setattr(qbo_billing, "record_payment", lambda *a, **k: {"Id": "P1"})
    monkeypatch.setattr(qbo_billing, "record_refund", lambda *a, **k: {"Id": "R1"})
    sr = {}
    monkeypatch.setattr(stripe_pay, "refund",
                        lambda pi, amount_cents=None: sr.update(pi=pi, amt=amount_cents)
                        or {"id": "re_1", "status": "succeeded", "amount": amount_cents})
    oid = _oid(cx)
    card = op.add_payment(cx, oid, 22291, "Credit card (Stripe)",
                          source="stripe", external_ref="pi_9")
    zelle = op.add_payment(cx, oid, 13100, "Zelle")
    r1 = op.add_refund(cx, oid, 10000, "Credit card (Stripe)",
                       refunds_payment_id=card["id"])
    assert sr == {"pi": "pi_9", "amt": 10000}
    assert r1["external_ref"] == "re_1"
    sr.clear()
    op.add_refund(cx, oid, 5000, "Zelle", refunds_payment_id=zelle["id"])
    assert sr == {}   # non-card refund did not touch Stripe


def test_void_refund_reapplies_paid(cx, monkeypatch):
    from dashboard import qbo_billing
    monkeypatch.setattr(qbo_billing, "get_invoice",
                        lambda iid: {"CustomerRef": {"value": "42"}, "Balance": "1"})
    monkeypatch.setattr(qbo_billing, "record_payment", lambda *a, **k: {"Id": "P1"})
    monkeypatch.setattr(qbo_billing, "record_refund", lambda *a, **k: {"Id": "R1"})
    voided_refunds = {}
    monkeypatch.setattr(qbo_billing, "void_refund",
                        lambda txn: voided_refunds.update(txn=txn))
    oid = _oid(cx)
    pay = op.add_payment(cx, oid, 13100, "Zelle")
    ref = op.add_refund(cx, oid, 5000, "Zelle", refunds_payment_id=pay["id"])
    # The ledger no longer pushes a RefundReceipt (the bank feed already carries the
    # money), so a refund row only has a txn id if one was linked. Set it directly —
    # add_refund has no qbo_txn_id parameter — so the dispatch assertion below still
    # has something to act on. The subject here is void ROUTING, which is unchanged.
    cx.execute("UPDATE order_payments SET qbo_txn_id='R1' WHERE id=?", (ref["id"],))
    cx.commit()
    op.void(cx, ref["id"], "issued in error")
    assert op.balance(cx, oid)["refunded_cents"] == 0
    # a refund row's qbo_txn_id is a RefundReceipt — must dispatch to void_refund,
    # never void_payment (wrong QBO endpoint)
    assert voided_refunds == {"txn": "R1"}


def test_payments_view_shows_card_refund_but_not_card_payment(cx, monkeypatch):
    """The money view (ledger_rows_for_payments_view) must surface a CARD refund
    (source='stripe', kind='refund' — it has no row in the Stripe charge ledger)
    while still excluding the card PAYMENT (source='stripe', kind='payment' — that
    charge already appears via dashboard.payments)."""
    from dashboard import qbo_billing, stripe_pay
    monkeypatch.setattr(qbo_billing, "get_invoice",
                        lambda iid: {"CustomerRef": {"value": "42"}, "Balance": "999"})
    monkeypatch.setattr(qbo_billing, "record_payment", lambda *a, **k: {"Id": "P1"})
    monkeypatch.setattr(qbo_billing, "record_refund", lambda *a, **k: {"Id": "R1"})
    monkeypatch.setattr(stripe_pay, "refund",
                        lambda pi, amount_cents=None: {"id": "re_1", "status": "succeeded",
                                                       "amount": amount_cents})
    oid = _oid(cx)
    card = op.add_payment(cx, oid, 22291, "Credit card (Stripe)",
                          source="stripe", external_ref="pi_9")
    op.add_payment(cx, oid, 13100, "Zelle")                       # manual payment
    op.add_refund(cx, oid, 5000, "Credit card (Stripe)",
                  refunds_payment_id=card["id"])                  # card refund (source='stripe')

    rows = op.ledger_rows_for_payments_view(cx)
    refs = [r["external_ref"] for r in rows]
    assert "pi_9" not in refs                    # card PAYMENT excluded (it's in the charge ledger)
    assert "re_1" in refs                        # card REFUND included (its only appearance)
    cr = [r for r in rows if r["source"] == "card:refund"][0]
    assert cr["amount_cents"] == -5000 and cr["external_ref"] == "re_1"
    assert any(r["source"].startswith("manual:") for r in rows)   # the Zelle still shows


def test_backfill_legacy_payments(cx):
    """Dry-run reports the plan without writing; apply creates source='legacy' rows
    (no QBO push); trials, already-ledgered, unpaid, and skip_order_ids are excluded;
    re-run is idempotent."""
    from dashboard import orders as O
    # order #1 (fixture, total 41282) -> a PAID non-trial legacy order (a candidate)
    cx.execute("UPDATE orders SET paid_cents=41282, pay_method='card', pay_status='paid' WHERE id=1")
    # #2 paid biofield_trial -> excluded (trial)
    O.upsert_order(cx, source="biofield_trial", external_ref="T-1", email="t@e.com", total_cents=100)
    cx.execute("UPDATE orders SET paid_cents=100, pay_method='card', pay_status='paid' WHERE external_ref='T-1'")
    # #3 paid non-trial that ALREADY has a ledger row -> excluded (idempotency guard)
    O.upsert_order(cx, source="qbo", external_ref="L-1", email="l@e.com", total_cents=5000)
    o3 = cx.execute("SELECT id FROM orders WHERE external_ref='L-1'").fetchone()[0]
    cx.execute("UPDATE orders SET paid_cents=5000, pay_method='Zelle', pay_status='paid' WHERE id=?", (o3,))
    cx.execute("INSERT INTO order_payments (order_id, kind, amount_cents, method, source, status, "
               "qbo_sync, created_at) VALUES (?, 'payment', 5000, 'Zelle', 'manual', 'active', "
               "'synced', '2026-01-01')", (o3,))
    # #4 unpaid -> excluded
    O.upsert_order(cx, source="qbo", external_ref="U-1", email="u@e.com", total_cents=9999)
    # #5 PAID but CANCELLED -> excluded (a cancelled order must never get a "paid" row)
    O.upsert_order(cx, source="qbo", external_ref="X-1", email="x@e.com", total_cents=8000)
    cx.execute("UPDATE orders SET paid_cents=8000, pay_method='card', pay_status='paid', "
               "status='cancelled' WHERE external_ref='X-1'")
    cx.commit()

    plan = op.backfill_legacy_payments(cx, dry_run=True)
    # only the paid, non-trial, non-cancelled, no-ledger order (#1) is a candidate
    assert [p["order_id"] for p in plan["candidates"]] == [1]
    assert plan["written"] == 0
    assert cx.execute("SELECT COUNT(*) FROM order_payments WHERE source='legacy'").fetchone()[0] == 0

    res = op.backfill_legacy_payments(cx, dry_run=False)
    assert res["written"] == 1
    row = cx.execute("SELECT * FROM order_payments WHERE order_id=1 AND source='legacy'").fetchone()
    assert row["amount_cents"] == 41282 and row["method"] == "card"
    assert row["kind"] == "payment" and row["status"] == "active"
    assert row["qbo_sync"] == "synced" and row["qbo_txn_id"] is None   # NEVER pushed to QBO

    assert op.backfill_legacy_payments(cx, dry_run=False)["written"] == 0   # idempotent re-run

    cx.execute("DELETE FROM order_payments WHERE source='legacy'"); cx.commit()
    skipped = op.backfill_legacy_payments(cx, dry_run=True, skip_order_ids=[1])
    assert [p["order_id"] for p in skipped["candidates"]] == []   # skip list honored


def test_resync_never_pushes_a_legacy_backfill_row(cx, monkeypatch):
    """A legacy-backfill row is qbo_sync='synced' with a NULL qbo_txn_id. resync()
    must NOT push it to QBO — its payment already exists there, so a push would be a
    duplicate. (The _push guard now honors qbo_sync, not just qbo_txn_id presence.)"""
    from dashboard import qbo_billing
    cx.execute("UPDATE orders SET paid_cents=41282, pay_method='card', pay_status='paid' WHERE id=1")
    cx.commit()
    op.backfill_legacy_payments(cx, dry_run=False)
    legacy = cx.execute("SELECT id FROM order_payments WHERE order_id=1 AND source='legacy'").fetchone()[0]
    calls = {"n": 0}
    monkeypatch.setattr(qbo_billing, "get_invoice",
                        lambda iid: {"CustomerRef": {"value": "42"}, "Balance": "999"})
    monkeypatch.setattr(qbo_billing, "record_payment",
                        lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or {"Id": "P9"})
    op.resync(cx, legacy)
    assert calls["n"] == 0   # NEVER pushed — no double-count
    row = cx.execute("SELECT qbo_sync, qbo_txn_id FROM order_payments WHERE id=?", (legacy,)).fetchone()
    assert row["qbo_sync"] == "synced" and row["qbo_txn_id"] is None
