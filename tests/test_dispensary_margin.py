"""settle_dispensary_margin delegates to the wallet credit (Supabase) + the dispensary
sale record. Both external effects are mocked; their invoice-idempotency is owned/tested
in wallet.earn_dropship_margin and practitioner_portal.record_dispensary_order."""
from dashboard import dispensary_rewards as dr
import dashboard.wallet as wallet_mod
import dashboard.practitioner_portal as pp_mod


def test_margin_delegates_to_wallet_and_record(monkeypatch):
    earn, rec = [], []
    monkeypatch.setattr(wallet_mod, "earn_dropship_margin",
                        lambda pid, margin, *, qbo_invoice_id, ref=None:
                        (earn.append((pid, margin, qbo_invoice_id)), margin)[1])
    monkeypatch.setattr(pp_mod, "record_dispensary_order",
                        lambda pid, *, invoice_id, credit_earned_cents, **k:
                        rec.append((pid, invoice_id, credit_earned_cents)))
    got = dr.settle_dispensary_margin({"practitioner_id": "prac-1", "margin_cents": 2000}, "INV1")
    assert got == 2000
    assert earn == [("prac-1", 2000, "INV1")]     # wallet credited, invoice-keyed
    assert rec == [("prac-1", "INV1", 2000)]      # dispensary sale recorded


def test_margin_noop_without_pid_or_inv(monkeypatch):
    earn, rec = [], []
    monkeypatch.setattr(wallet_mod, "earn_dropship_margin", lambda *a, **k: earn.append(1))
    monkeypatch.setattr(pp_mod, "record_dispensary_order", lambda *a, **k: rec.append(1))
    assert dr.settle_dispensary_margin({"margin_cents": 2000}, "INV1") == 0   # no pid
    assert dr.settle_dispensary_margin({"practitioner_id": "p"}, "") == 0      # no invoice
    assert earn == [] and rec == []               # neither effect fired
