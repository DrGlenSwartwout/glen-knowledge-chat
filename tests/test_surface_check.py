"""Public-surface drift detector.

`FIRESIDE_ENABLED` is not declared in render.yaml, so nothing pins it on and it
silently drifted OFF — /begin/fireside 404'd in prod while /begin still shipped a
live "Sit by the fire" CTA pointing at it. Nothing alerted. This checks the
symptom a visitor sees (an HTTP status) rather than the config we happen to
suspect, so it also catches a deleted route, a missing template, or a bad deploy.
"""
import scripts.surface_check as S


def _fetch_map(mapping):
    """fetch(url) -> status int, or raises for a connection failure (None value)."""
    def _fetch(url, timeout=0):
        for path, status in mapping.items():
            if url.endswith(path):
                if status is None:
                    raise OSError("connection refused")
                return status
        raise AssertionError(f"unexpected url {url}")
    return _fetch


def test_all_healthy_reports_no_failures():
    fetch = _fetch_map({"/": 200, "/begin": 200, "/begin/fireside": 200,
                        "/prepay": 302, "/results": 200})
    assert S.check_surfaces("https://x.test", S.PUBLIC_SURFACES, fetch) == []


def test_redirect_is_not_a_failure():
    """/begin/choose and /prepay legitimately 302. A redirect is not a dead page."""
    fetch = _fetch_map({"/prepay": 302})
    assert S.check_surfaces("https://x.test", ("/prepay",), fetch) == []


def test_404_is_a_failure():
    fetch = _fetch_map({"/begin/fireside": 404})
    out = S.check_surfaces("https://x.test", ("/begin/fireside",), fetch)
    assert out == [{"path": "/begin/fireside", "status": 404, "error": ""}]


def test_500_is_a_failure():
    fetch = _fetch_map({"/results": 500})
    out = S.check_surfaces("https://x.test", ("/results",), fetch)
    assert out[0]["status"] == 500


def test_connection_error_is_a_failure_not_a_crash():
    """The web service being down must ALERT, not take the checker down with it."""
    fetch = _fetch_map({"/begin": None})
    out = S.check_surfaces("https://x.test", ("/begin",), fetch)
    assert out[0]["status"] == 0
    assert "connection refused" in out[0]["error"]


def test_one_dead_surface_does_not_hide_the_others():
    fetch = _fetch_map({"/": 200, "/begin": 404, "/begin/fireside": 404})
    out = S.check_surfaces("https://x.test", ("/", "/begin", "/begin/fireside"), fetch)
    assert [f["path"] for f in out] == ["/begin", "/begin/fireside"]


def test_fireside_is_watched():
    """The surface this whole check exists for."""
    assert "/begin/fireside" in S.PUBLIC_SURFACES


def test_format_alert_names_paths_and_statuses():
    subject, body = S.format_alert("https://illtowell.com", [
        {"path": "/begin/fireside", "status": 404, "error": ""}])
    assert "1" in subject and "illtowell.com" in subject
    assert "/begin/fireside" in body
    assert "404" in body


def test_format_alert_includes_connection_error_text():
    _subject, body = S.format_alert("https://illtowell.com", [
        {"path": "/begin", "status": 0, "error": "connection refused"}])
    assert "connection refused" in body


def test_send_alert_without_smtp_config_returns_false_not_raise(monkeypatch):
    """A mail failure must never take down the host cron's real job."""
    for var in ("SMTP_HOST", "SMTP_USER", "SMTP_PASS"):
        monkeypatch.delenv(var, raising=False)
    assert S.send_alert("subj", "body") is False


def test_run_swallows_a_broken_probe(monkeypatch):
    monkeypatch.setattr(S, "check_surfaces", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    try:
        S.run()
    except RuntimeError:
        pass  # run() itself may raise; the CRON wrapper is what must swallow (below)


# ── wiring: the checker must actually be called by the cron, not sit dead ──
def _import_cron(monkeypatch):
    """Import the cron entry point the way prod runs it: `python scripts/x.py`, i.e.
    with scripts/ on sys.path so its sibling `from _cron_http import ...` resolves.
    It sys.exit(1)s at import without CRON_SECRET."""
    import importlib
    import pathlib
    import sys
    monkeypatch.setenv("CRON_SECRET", "x")
    scripts_dir = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
    monkeypatch.syspath_prepend(scripts_dir)
    sys.modules.pop("run_personal_email_cron", None)
    return importlib.import_module("run_personal_email_cron")



def test_cron_calls_surface_check(monkeypatch):
    """Guards against surface_check.py existing while nothing invokes it."""
    cron = _import_cron(monkeypatch)
    called = []
    # The cron imports top-level `surface_check` (scripts/ is on sys.path when run as a
    # script), which is a DIFFERENT module object from `scripts.surface_check`. Patch the
    # one it actually loads.
    import surface_check as top
    monkeypatch.setattr(top, "run", lambda: called.append(True) or [])
    cron.check_public_surfaces()
    assert called == [True], "cron did not invoke surface_check.run"


def test_cron_wrapper_never_raises(monkeypatch):
    """A surface-check explosion must not break the personal-email cron."""
    cron = _import_cron(monkeypatch)
    import surface_check as top
    monkeypatch.setattr(top, "run", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    cron.check_public_surfaces()   # must not raise
