"""Task 4: /api/cron/qbo-heal-pending cron route.

Runs the QBO PENDING-receipt heal sweep (dashboard.qbo_heal.heal_pending_receipts)
against the live LOG_DB, wired the same way the other /api/cron/* routes are:
X-Cron-Secret header matching CRON_SECRET (falls back to CONSOLE_SECRET).

Mirrors the harness in test_care_share_cron.py -- imports app directly (no
reload), resolves the real secret from the environment (doppler dev), and
posts with the X-Cron-Secret header.
"""
import os
from unittest import mock

import app as appmod


def _cron_secret():
    return os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "test-secret")


def _headers():
    return {"X-Cron-Secret": _cron_secret()}


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def test_requires_secret():
    resp = _client().post("/api/cron/qbo-heal-pending")
    assert resp.status_code == 401


def test_wrong_secret_rejected():
    resp = _client().post("/api/cron/qbo-heal-pending",
                          headers={"X-Cron-Secret": "definitely-wrong"})
    assert resp.status_code == 401


def test_correct_secret_invokes_heal_sweep():
    with mock.patch("dashboard.qbo_heal.heal_pending_receipts", return_value=[]) as heal:
        resp = _client().post("/api/cron/qbo-heal-pending", headers=_headers())
    assert resp.status_code == 200
    assert heal.called
    assert resp.get_json() == {"ok": True, "healed": [], "count": 0}
