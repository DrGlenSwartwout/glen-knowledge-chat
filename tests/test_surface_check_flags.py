"""A flag that must be on and is not on must alarm — deleted, set false, or set true
but never redeployed. REPERTOIRE_ENABLED and INVOICE_PAYLINK_ENABLED have no page that
404s, so the HTTP surface check from #736 structurally cannot see them.
"""
import urllib.error

import scripts.surface_check as S


def _payload(**flags):
    return {"ok": True, "data": {"flags": flags}}


def _on():
    return {"value": True, "env_present": True, "source": "import"}


def _fetch_ok(payload):
    def _f(url, key, timeout=0):
        return payload
    return _f


def _fetch_raises(exc):
    def _f(url, key, timeout=0):
        raise exc
    return _f


ALL_ON = _payload(**{name: _on() for name in S.REQUIRED_ON})


def test_required_on_is_the_four_flags_glen_named():
    assert set(S.REQUIRED_ON) == {"FIRESIDE_ENABLED", "REPERTOIRE_ENABLED",
                                  "INVOICE_PAYLINK_ENABLED", "SCAN_REQUEST_ENABLED"}


def test_all_on_reports_nothing():
    assert S.check_flags("https://x.test", "k", fetch=_fetch_ok(ALL_ON)) == []


def test_flag_set_false_is_a_failure():
    p = _payload(**{n: _on() for n in S.REQUIRED_ON})
    p["data"]["flags"]["REPERTOIRE_ENABLED"] = {"value": False, "env_present": True,
                                                "source": "import"}
    out = S.check_flags("https://x.test", "k", fetch=_fetch_ok(p))
    assert [f["flag"] for f in out] == ["REPERTOIRE_ENABLED"]
    assert "false" in out[0]["reason"].lower()


def test_deleted_flag_reason_differs_from_set_false():
    """The distinction that was unavailable during the incident."""
    p = _payload(**{n: _on() for n in S.REQUIRED_ON})
    p["data"]["flags"]["FIRESIDE_ENABLED"] = {"value": False, "env_present": False,
                                              "source": "import"}
    out = S.check_flags("https://x.test", "k", fetch=_fetch_ok(p))
    assert out[0]["flag"] == "FIRESIDE_ENABLED"
    assert "missing" in out[0]["reason"].lower()
    assert "false" not in out[0]["reason"].lower()


def test_stale_import_flag_names_the_missing_deploy():
    """env says on, process says off -> someone set the var and never redeployed."""
    p = _payload(**{n: _on() for n in S.REQUIRED_ON})
    p["data"]["flags"]["INVOICE_PAYLINK_ENABLED"] = {"value": False, "env_present": True,
                                                     "source": "import"}
    out = S.check_flags("https://x.test", "k", fetch=_fetch_ok(p))
    assert "redeploy" in out[0]["reason"].lower()


def test_absent_from_response_is_a_failure():
    """A deleted CALL-TIME flag has no global and no env key, so it vanishes."""
    p = _payload(**{n: _on() for n in S.REQUIRED_ON if n != "SCAN_REQUEST_ENABLED"})
    out = S.check_flags("https://x.test", "k", fetch=_fetch_ok(p))
    assert [f["flag"] for f in out] == ["SCAN_REQUEST_ENABLED"]
    assert "absent" in out[0]["reason"].lower()


def test_unwatched_flag_being_off_is_not_a_failure():
    """59 flags are deliberately unwatched; several are meant to be off. A watchdog that
    cries wolf gets ignored — which is how this incident stayed invisible."""
    p = _payload(**{n: _on() for n in S.REQUIRED_ON})
    p["data"]["flags"]["TWO_DOOR_ENABLED"] = {"value": False, "env_present": True,
                                              "source": "import"}
    assert S.check_flags("https://x.test", "k", fetch=_fetch_ok(p)) == []


def test_unreachable_endpoint_is_not_reported_as_drift():
    """The surfaces list already alarms when the app is down. One outage must not
    produce two contradictory stories."""
    out = S.check_flags("https://x.test", "k", fetch=_fetch_raises(OSError("refused")))
    assert len(out) == 1
    assert out[0]["flag"] == "*"
    assert "could not check" in out[0]["reason"].lower()


def test_unauthorized_is_not_reported_as_drift():
    """A real urllib HTTPError (what _fetch_json raises on 401) must report 'could not
    check', never four drift failures. The surfaces list already alarms when the app is
    down; one outage must not tell two contradictory stories."""
    err = urllib.error.HTTPError("https://x.test/api/console/flags", 401,
                                 "Unauthorized", {}, None)
    out = S.check_flags("https://x.test", "bad", fetch=_fetch_raises(err))
    assert len(out) == 1
    assert out[0]["flag"] == "*"
    assert "could not check" in out[0]["reason"].lower()


def test_missing_console_secret_skips_without_calling_fetch():
    """No key -> skip entirely, and prove it never reaches the network. With the early
    return removed, this fetch would raise and the test would fail."""
    def _explode(url, key, timeout=0):
        raise AssertionError("fetch must not be called without a console key")
    assert S.check_flags("https://x.test", "", fetch=_explode) == []


# ── wiring: the checker must actually run, and reach the alert ──
def test_format_alert_includes_flag_failures():
    _subject, body = S.format_alert("https://illtowell.com", [],
                                    [{"flag": "REPERTOIRE_ENABLED",
                                      "reason": "env var is MISSING (deleted)"}])
    assert "REPERTOIRE_ENABLED" in body
    assert "MISSING" in body


def test_format_alert_subject_counts_both_kinds():
    """One dead surface + one dead flag = 2 problems, not 1 of each in two emails."""
    subject, body = S.format_alert(
        "https://illtowell.com",
        [{"path": "/begin/fireside", "status": 404, "error": ""}],
        [{"flag": "REPERTOIRE_ENABLED", "reason": "env var is MISSING (deleted)"}])
    assert "2 problems" in subject
    assert "illtowell.com" in subject
    assert "/begin/fireside" in body and "REPERTOIRE_ENABLED" in body


def test_format_alert_singular_when_one_problem():
    subject, _ = S.format_alert("https://illtowell.com", [],
                                [{"flag": "FIRESIDE_ENABLED", "reason": "x"}])
    assert "1 problem on" in subject


def test_run_calls_check_flags_and_alerts(monkeypatch):
    """Guards against check_flags() existing while nothing invokes it."""
    sent = {}
    monkeypatch.setattr(S, "check_surfaces", lambda *a, **k: [])
    monkeypatch.setattr(S, "check_flags", lambda *a, **k: [
        {"flag": "REPERTOIRE_ENABLED", "reason": "env var is MISSING (deleted)"}])
    monkeypatch.setattr(S, "CONSOLE_SECRET", "k")
    monkeypatch.setattr(S, "send_alert", lambda subj, body, **k: sent.update(
        subject=subj, body=body) or True)
    out = S.run()
    assert [f["flag"] for f in out] == ["REPERTOIRE_ENABLED"]
    assert "REPERTOIRE_ENABLED" in sent["body"], "flag failure never reached the alert"


def test_run_is_quiet_when_everything_is_healthy(monkeypatch):
    called = []
    monkeypatch.setattr(S, "check_surfaces", lambda *a, **k: [])
    monkeypatch.setattr(S, "check_flags", lambda *a, **k: [])
    monkeypatch.setattr(S, "CONSOLE_SECRET", "k")
    monkeypatch.setattr(S, "send_alert", lambda *a, **k: called.append(True) or True)
    assert S.run() == []
    assert called == [], "no alert may be sent when nothing is wrong"
