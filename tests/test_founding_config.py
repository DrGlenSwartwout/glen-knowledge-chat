import json
import sqlite3
import dashboard.founding as founding
from dashboard import subscriptions as subs


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    subs.init_subscriptions_table(cx)
    subs.migrate_add_founding_columns(cx)
    return cx


def _patch_config(monkeypatch):
    monkeypatch.setattr(founding, "_CONFIG", {
        "neuro-magnesium": {"cap": 3, "batch_label": "Founding Batch No. 1",
                            "video_url": "/clip/neuro/promo.mp4", "closes_at": "2026-12-31"}})


def test_get_launch(monkeypatch):
    _patch_config(monkeypatch)
    assert founding.get_launch("neuro-magnesium")["cap"] == 3
    assert founding.get_launch("nope") is None


def test_remaining_and_is_open(monkeypatch):
    _patch_config(monkeypatch)
    cx = _cx()
    assert founding.remaining(cx, "neuro-magnesium") == 3
    assert founding.is_open(cx, "neuro-magnesium", now_iso="2026-07-01") is True
    for e in ("a@x.com", "b@x.com", "c@x.com"):
        subs.create_founding_reservation(cx, email=e, stripe_customer_id="c",
            stripe_payment_method_id="pm", items=[], ship_address={}, founding_slug="neuro-magnesium")
    assert founding.remaining(cx, "neuro-magnesium") == 0
    assert founding.is_open(cx, "neuro-magnesium", now_iso="2026-07-01") is False   # cap hit


def test_is_open_false_after_closes_at(monkeypatch):
    _patch_config(monkeypatch)
    cx = _cx()
    assert founding.is_open(cx, "neuro-magnesium", now_iso="2027-01-01") is False   # window closed
    assert founding.is_open(cx, "missing", now_iso="2026-07-01") is False
