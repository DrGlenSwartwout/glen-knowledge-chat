# Feature-Flag Drift Watch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Alert when a feature flag that must be on is not on — whether it was deleted, set false, or set true but never redeployed.

**Architecture:** A new owner-gated `GET /api/console/flags` reports, for every `*_ENABLED` flag, the value the *running process* holds plus whether the env var exists at all. `scripts/surface_check.py` (the daily cron watchdog from #736) gains `REQUIRED_ON` and `check_flags()`, and merges flag failures into its existing single alert email.

**Tech Stack:** Python 3, Flask, urllib (stdlib only in `scripts/`), pytest.

**Spec:** `docs/superpowers/specs/2026-07-09-flag-drift-watch-design.md`

**Branch:** `sess/5629cdf8-flagwatch`, worktree `/tmp/wt-deploy-chat-5629cdf8`, off `origin/main`.

## Global Constraints

- **Test command, pure modules:** `python3 -m pytest tests/<file> -q`
- **Test command, tests that `import app`:** `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) python3 -m pytest tests/<file> -q`
  Both halves required. `import app` validates a Pinecone key over the network at import; Doppler-prd also sets `DATA_DIR=/data`, unwritable locally. `sqlite3.OperationalError: unable to open database file` means you forgot the override — it is **not** an environmental failure.
- **Always read the skip count.** Several test harnesses call `pytest.skip("app not importable")`. `N passed, M skipped` with `M > 0` means the tests silently did not run.
- **`main` is red** (~96 pre-existing failures, flaky). Never judge by counts. Regression = diff `FAILED <nodeid>` **names** against an `origin/main` baseline.
- **`scripts/` is stdlib-only.** The cron's `buildCommand` is `true`; no third-party imports.
- **Booleans only, `*_ENABLED` keys only.** The endpoint must never return a non-flag env key and never echo a value as a string. `CONSOLE_SECRET` must not be able to leak through it.
- **No auto-remediation.** The check never re-sets a flag. It reports; Glen decides.
- **Best-effort by contract.** `check_public_surfaces()` in the cron never raises and never affects the personal-email send.
- Commit messages end with exactly:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

### Task 1: `GET /api/console/flags`

Reports every flag this process knows about. Discovery is the **union** of module globals and `os.environ` keys matching `^_?[A-Z][A-Z0-9_]*_ENABLED$` — no registry (there are 29 call-time flags; a hand-maintained list would rot) and no source scan.

**Files:**
- Modify: `app.py` — add `_flag_report()` + the route, immediately **before** `@app.route("/api/console/client-prefs", ...)` (currently `app.py:33384`). It must sit below `app.py:27708`, where `require_console_key, ok, fail` are imported.
- Test: `tests/test_console_flags.py` (create)

**Interfaces:**
- Consumes: `require_console_key`, `ok`, `fail` (already imported at `app.py:27708`); `re` and `os` (already imported).
- Produces: `app._flag_report() -> dict[str, dict]`, each value `{"value": bool, "env_present": bool, "source": "import" | "runtime"}`.
  Response envelope is `ok(...)`, i.e. `{"ok": true, "data": {"flags": {...}}}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_console_flags.py`:

```python
"""Report every *_ENABLED flag: the value the RUNNING PROCESS holds, and whether the
env var exists at all.

On 2026-07-09 three flags were DELETED from the prod service (a single-key GET returned
404 "not found"), not set false. A deleted var and a deliberate false make the app behave
identically but mean different things — REPERTOIRE_ENABLED missing silently charges paid
members more. That distinction is what `env_present` carries.

Two kinds of flag exist: 34 import-time module constants (can go stale — set the var,
skip the deploy, the app still behaves as if off) and 29 that exist only as call-time
os.environ reads. A globals scan finds none of the 29, so discovery is the UNION of
module globals and env keys.
"""
import importlib
import sys
from pathlib import Path

import pytest


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def test_import_flag_reports_global_not_env(monkeypatch):
    """An import-time flag's value is the GLOBAL, not os.environ. That is what makes
    'set true but never redeployed' detectable."""
    appmod = _app()
    monkeypatch.setattr(appmod, "FIRESIDE_ENABLED", False)
    monkeypatch.setenv("FIRESIDE_ENABLED", "true")      # env says on, process says off
    f = appmod._flag_report()["FIRESIDE_ENABLED"]
    assert f["value"] is False, "must report what the process holds"
    assert f["env_present"] is True
    assert f["source"] == "import"


def test_deleted_import_flag_reports_env_absent(monkeypatch):
    appmod = _app()
    monkeypatch.setattr(appmod, "FIRESIDE_ENABLED", False)
    monkeypatch.delenv("FIRESIDE_ENABLED", raising=False)
    f = appmod._flag_report()["FIRESIDE_ENABLED"]
    assert f["value"] is False
    assert f["env_present"] is False, "a deleted var must be distinguishable from a false one"


def test_runtime_flag_tracks_env_without_restart(monkeypatch):
    """SCAN_REQUEST_ENABLED has NO module global (app.py:14808 reads os.environ per
    call). It must still be reported, and its value follows the env immediately."""
    appmod = _app()
    assert not hasattr(appmod, "SCAN_REQUEST_ENABLED")
    monkeypatch.setenv("SCAN_REQUEST_ENABLED", "1")
    f = appmod._flag_report()["SCAN_REQUEST_ENABLED"]
    assert f["value"] is True
    assert f["source"] == "runtime"
    monkeypatch.setenv("SCAN_REQUEST_ENABLED", "off")
    assert appmod._flag_report()["SCAN_REQUEST_ENABLED"]["value"] is False


def test_deleted_runtime_flag_vanishes_from_the_report(monkeypatch):
    """It has no global and no env key, so it cannot appear. check_flags() then reports
    it via the absent-from-response rule — this is the SCAN_REQUEST_ENABLED deletion."""
    appmod = _app()
    monkeypatch.delenv("SCAN_REQUEST_ENABLED", raising=False)
    assert "SCAN_REQUEST_ENABLED" not in appmod._flag_report()


def test_only_enabled_keys_are_returned(monkeypatch):
    """CONSOLE_SECRET must not be able to leak through this endpoint."""
    appmod = _app()
    monkeypatch.setenv("CONSOLE_SECRET", "super-secret")
    monkeypatch.setenv("NOT_A_FLAG", "true")
    report = appmod._flag_report()
    assert "CONSOLE_SECRET" not in report
    assert "NOT_A_FLAG" not in report
    assert all(k.endswith("_ENABLED") for k in report)


def test_values_are_booleans_never_strings(monkeypatch):
    appmod = _app()
    monkeypatch.setenv("SCAN_REQUEST_ENABLED", "yes")
    for info in appmod._flag_report().values():
        assert isinstance(info["value"], bool)
        assert isinstance(info["env_present"], bool)
        assert info["source"] in ("import", "runtime")


def test_a_new_global_is_discovered_without_registration(monkeypatch):
    appmod = _app()
    monkeypatch.setattr(appmod, "BRAND_NEW_ENABLED", True, raising=False)
    assert appmod._flag_report()["BRAND_NEW_ENABLED"]["source"] == "import"


def test_route_requires_the_console_key(monkeypatch):
    appmod = _app()
    monkeypatch.setattr(appmod.dashboard, "CONSOLE_SECRET", "k")
    appmod.app.config["TESTING"] = True
    c = appmod.app.test_client()
    assert c.get("/api/console/flags").status_code == 401
    r = c.get("/api/console/flags", headers={"X-Console-Key": "k"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert "FIRESIDE_ENABLED" in body["data"]["flags"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) python3 -m pytest tests/test_console_flags.py -q`
Expected: FAIL — `AttributeError: module 'app' has no attribute '_flag_report'`.
Confirm the skip count is **0**. `N passed, M skipped` with `M > 0` means nothing ran.

- [ ] **Step 3: Write minimal implementation**

In `app.py`, immediately **before** `@app.route("/api/console/client-prefs", methods=["GET", "POST"])`, insert:

```python
# ── Feature-flag report ───────────────────────────────────────────────────────
_FLAG_NAME_RE = re.compile(r"^_?[A-Z][A-Z0-9_]*_ENABLED$")
_FLAG_TRUTHY = ("1", "true", "yes", "on")


def _flag_report():
    """Every *_ENABLED flag this process knows about.

    TWO kinds exist and they differ in a way that matters:
      - import-time constants (module globals, fixed when the process started). These
        can go STALE: set the env var, skip the deploy, and the app still behaves as if
        the flag were off. `value` therefore reads the GLOBAL, not os.environ.
      - call-time reads (no global at all; os.environ is read inside a function on each
        request, e.g. SCAN_REQUEST_ENABLED at app.py:14808). Never stale. A globals scan
        would never find them, so discovery is the UNION of globals and env keys.

    A deleted var and a deliberate false make the app behave identically but mean
    different things, so `env_present` is reported separately. A deleted CALL-TIME flag
    has neither a global nor an env key and so vanishes from this report entirely — the
    checker's absent-from-response rule is what catches that case.

    Only booleans, only *_ENABLED keys: no other env var can leak through here."""
    out = {}
    for name, val in list(globals().items()):
        if isinstance(val, bool) and _FLAG_NAME_RE.match(name):
            out[name] = {"value": bool(val),
                         "env_present": name in os.environ,
                         "source": "import"}
    for name in os.environ:
        if name in out or not _FLAG_NAME_RE.match(name):
            continue
        raw = (os.environ.get(name) or "").strip().lower()
        out[name] = {"value": raw in _FLAG_TRUTHY,
                     "env_present": True,
                     "source": "runtime"}
    return out


@app.route("/api/console/flags", methods=["GET"])
@require_console_key
def api_console_flags():
    """Owner: every feature flag as the RUNNING PROCESS sees it, plus whether each env
    var exists at all. Read-only. Consumed by scripts/surface_check.py's daily drift
    check — a flag with no HTTP surface (REPERTOIRE_ENABLED, INVOICE_PAYLINK_ENABLED)
    cannot be watched any other way."""
    try:
        return ok({"flags": _flag_report()})
    except Exception as e:
        return fail(e)
```

`app.py` already imports `os` (line 14) and `re` (line 15) at module scope. Use them directly — add no imports. (Note `app.py:23706` also does `import re as _re` inside a function; ignore it, it is function-local.)

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) python3 -m pytest tests/test_console_flags.py -q`
Expected: `8 passed`, **0 skipped**.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_console_flags.py
git commit -m "feat(console): GET /api/console/flags reports flag value + env presence

Two kinds of flag exist: 34 import-time module constants (can go stale after a
deploy) and 29 call-time os.environ reads (never stale, invisible to a globals
scan). Discovery is the union of globals and env keys — no registry to rot.

A deleted var and a deliberate false behave identically but mean different things,
so env_present is reported separately. Only booleans, only *_ENABLED keys.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `check_flags()` in the watchdog

**Files:**
- Modify: `scripts/surface_check.py` (add `REQUIRED_ON`, `CONSOLE_SECRET`, `_fetch_json`, `check_flags`)
- Test: `tests/test_surface_check_flags.py` (create)

**Interfaces:**
- Consumes: `GET /api/console/flags` → `{"ok": true, "data": {"flags": {NAME: {"value","env_present","source"}}}}` (Task 1).
- Produces:
  - `surface_check.REQUIRED_ON: tuple[str, ...]`
  - `surface_check.check_flags(base_url, console_key, required=REQUIRED_ON, fetch=_fetch_json) -> list[dict]`, each `{"flag": str, "reason": str}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_surface_check_flags.py`:

```python
"""A flag that must be on and is not on must alarm — deleted, set false, or set true
but never redeployed. REPERTOIRE_ENABLED and INVOICE_PAYLINK_ENABLED have no page that
404s, so the HTTP surface check from #736 structurally cannot see them.
"""
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
    out = S.check_flags("https://x.test", "bad", fetch=_fetch_raises(OSError("HTTP 401")))
    assert len(out) == 1 and out[0]["flag"] == "*"


def test_missing_console_secret_skips_rather_than_alarms():
    assert S.check_flags("https://x.test", "", fetch=_fetch_ok(ALL_ON)) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_surface_check_flags.py -q`
Expected: FAIL — `AttributeError: module 'scripts.surface_check' has no attribute 'REQUIRED_ON'`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/surface_check.py`, add `import json` beside the existing stdlib imports, then insert after the `OWNER_EMAIL` line:

```python
CONSOLE_SECRET = os.environ.get("CONSOLE_SECRET", "")

# Flags that must ALWAYS be true. Named by Glen 2026-07-09 after three vanished from the
# prod service. The other 59 *_ENABLED flags are deliberately unwatched — several are
# experiments where OFF is correct, and a watchdog that cries wolf gets ignored.
REQUIRED_ON = ("FIRESIDE_ENABLED", "REPERTOIRE_ENABLED",
               "INVOICE_PAYLINK_ENABLED", "SCAN_REQUEST_ENABLED")


def _fetch_json(url, key, timeout=20):
    """GET a console-gated JSON endpoint. Raises on any non-200."""
    req = urllib.request.Request(url, method="GET",
                                 headers={"X-Console-Key": key,
                                          "User-Agent": "surface-check/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")


def check_flags(base_url, console_key, required=REQUIRED_ON, fetch=_fetch_json):
    """One dict per flag that must be on and is not: {"flag", "reason"}.

    A deleted var, a deliberate false, and "set true but never redeployed" all reach the
    customer the same way, so all three alarm — but each names its own cause, because
    the fix differs. A CALL-TIME flag that was deleted has neither a module global nor an
    env key, so it vanishes from the report; the absent-from-response branch catches it.

    An unreachable or unauthorized endpoint is NOT drift: the surfaces list already
    alarms when the app is down, and one outage must not tell two contradictory stories.
    No console key -> skip entirely (the caller prints a notice)."""
    if not console_key:
        return []
    url = f"{base_url.rstrip('/')}/api/console/flags"
    try:
        payload = fetch(url, console_key)
    except Exception as e:  # noqa: BLE001 — a check failure is never drift
        return [{"flag": "*", "reason": f"could not check flags: {e}"}]
    flags = ((payload or {}).get("data") or {}).get("flags") or {}
    if not flags:
        return [{"flag": "*", "reason": "could not check flags: unexpected response"}]
    out = []
    for name in required:
        info = flags.get(name)
        if info is None:
            out.append({"flag": name,
                        "reason": "absent from /api/console/flags "
                                  "(env var deleted, or the constant was removed)"})
            continue
        if info.get("value"):
            continue
        if info.get("env_present"):
            reason = "set to false"
            if info.get("source") == "import":
                reason += " (or set true but never redeployed — flags read at import)"
        else:
            reason = "env var is MISSING (deleted)"
        out.append({"flag": name, "reason": reason})
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_surface_check_flags.py -q`
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/surface_check.py tests/test_surface_check_flags.py
git commit -m "feat(ops): check_flags() — a flag that must be on and is not alarms

REPERTOIRE_ENABLED and INVOICE_PAYLINK_ENABLED have no page that 404s, so the HTTP
surface check from #736 structurally cannot see them. REPERTOIRE off silently
charges paid members more.

Deleted / set-false / set-but-never-redeployed all alarm, each naming its own cause.
An unreachable or 401 endpoint reports 'could not check', never drift: the surfaces
list already alarms when the app is down.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Merge flag failures into the daily alert

A checker nothing calls is the dead-field pattern hit three times this week (`regular_cents`, `#738`'s `pickup_default`, `bundle_components`). This task wires it and proves the wiring.

**Files:**
- Modify: `scripts/surface_check.py` (`format_alert`, `run`)
- Test: `tests/test_surface_check_flags.py` (append)

**Interfaces:**
- Consumes: `check_surfaces()`, `check_flags()`, `format_alert()`, `send_alert()`.
- Produces: `format_alert(base_url, failures, flag_failures=())` — third parameter is new and defaults to empty, so existing callers are unchanged. `run()` returns `surface_failures + flag_failures`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_surface_check_flags.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_surface_check_flags.py -q`
Expected: FAIL — `TypeError: format_alert() takes 2 positional arguments but 3 were given`, and `test_run_calls_check_flags_and_alerts` fails because `run()` never calls `check_flags`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/surface_check.py`, replace `format_alert` and `run` with:

```python
def format_alert(base_url, failures, flag_failures=()):
    """(subject, body) naming each dead path and each flag that must be on and is not.
    Plain text; no HTML. `flag_failures` defaults to empty so existing callers are
    unchanged."""
    n = len(failures) + len(flag_failures)
    host = base_url.split("//", 1)[-1].rstrip("/")
    subject = f"[surface-check] {n} problem{'s' if n != 1 else ''} on {host}"
    lines = []
    if failures:
        lines += [f"{len(failures)} public surface"
                  f"{'s' if len(failures) != 1 else ''} failing on {base_url}:", ""]
        for f in failures:
            why = f["error"] or f"HTTP {f['status']}"
            lines.append(f"  {f['path']}  ->  {why}")
        lines.append("")
    if flag_failures:
        lines += [f"{len(flag_failures)} feature flag"
                  f"{'s' if len(flag_failures) != 1 else ''} not on:", ""]
        for f in flag_failures:
            lines.append(f"  {f['flag']}  ->  {f['reason']}")
        lines.append("")
    lines += [
        "A 404 on a flag-gated surface usually means its *_ENABLED env var drifted",
        "off on the Render web service. Flags absent from render.yaml have nothing",
        "pinning them on. Re-flip with a single-key PUT + an explicit POST /deploys.",
        "",
        "REPERTOIRE_ENABLED off silently charges paid members MORE (they lose",
        "repertoire reorder pricing). INVOICE_PAYLINK_ENABLED off means clients",
        "cannot pay an invoice online. Neither has a page that 404s.",
    ]
    return subject, "\n".join(lines)


def run():
    """Probe surfaces + flags, alert on failure, and always return the failure list.
    Best-effort by contract: the caller is the personal-email cron, which must never
    fail because a check did."""
    failures = check_surfaces(BASE_URL)
    if CONSOLE_SECRET:
        flag_failures = check_flags(BASE_URL, CONSOLE_SECRET)
    else:
        print("[surface-check] CONSOLE_SECRET not set — skipping flag check", flush=True)
        flag_failures = []
    if not failures and not flag_failures:
        print(f"[surface-check] {len(PUBLIC_SURFACES)} surfaces OK, "
              f"{len(REQUIRED_ON)} flags on, at {BASE_URL}", flush=True)
        return []
    subject, body = format_alert(BASE_URL, failures, flag_failures)
    print(f"[surface-check] {subject}", flush=True)
    for f in failures:
        print(f"[surface-check]   {f['path']} -> {f['error'] or f['status']}", flush=True)
    for f in flag_failures:
        print(f"[surface-check]   {f['flag']} -> {f['reason']}", flush=True)
    send_alert(subject, body)
    return failures + flag_failures
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_surface_check_flags.py tests/test_surface_check.py -q`
Expected: all pass. `tests/test_surface_check.py` (from #736) must still pass — `format_alert`'s third parameter defaults to empty.

- [ ] **Step 5: Commit**

```bash
git add scripts/surface_check.py tests/test_surface_check_flags.py
git commit -m "feat(ops): merge flag failures into the daily surface alert

One email, both kinds of problem. A checker nothing calls is the dead-field pattern
hit three times this week (regular_cents, #738's pickup_default, bundle_components),
so a wiring test asserts run() actually reaches check_flags and the failure reaches
the alert body.

No CONSOLE_SECRET -> skip the flag check with a notice, never alarm.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Prove it end-to-end against production, then gate and PR

**Files:** none modified.

**Interfaces:**
- Consumes: everything above.
- Produces: a clean branch-vs-baseline diff and an open PR.

- [ ] **Step 1: Run the endpoint against live prod (read-only)**

```bash
cd /tmp/wt-deploy-chat-5629cdf8
doppler run -p remedy-match -c prd -- python3 -c '
import os, json, urllib.request
key = os.environ["CONSOLE_SECRET"]
r = urllib.request.Request("https://illtowell.com/api/console/flags",
                           headers={"X-Console-Key": key})
with urllib.request.urlopen(r, timeout=30) as resp:
    flags = json.loads(resp.read().decode())["data"]["flags"]
for n in ("FIRESIDE_ENABLED","REPERTOIRE_ENABLED","INVOICE_PAYLINK_ENABLED","SCAN_REQUEST_ENABLED"):
    print(f"  {n:24} {flags.get(n)}")
print(f"  total flags reported: {len(flags)}")
print("  CONSOLE_SECRET leaked?", "CONSOLE_SECRET" in flags)
'
```
Expected: **after this branch deploys**, all four report `value: True`. Before it deploys the endpoint 404s — that is expected and is not a failure of this step. Record which you observed.
`CONSOLE_SECRET leaked?` must print `False`.

- [ ] **Step 2: Run the checker against live prod**

```bash
doppler run -p remedy-match -c prd -- python3 -c '
import os, sys; sys.path.insert(0, "scripts")
import surface_check as S
print("surface failures:", S.check_surfaces("https://illtowell.com") or "NONE")
print("flag failures   :", S.check_flags("https://illtowell.com", os.environ["CONSOLE_SECRET"]) or "NONE")
'
```
Expected (once deployed): both `NONE`. If the endpoint is not deployed yet, flag failures will be a single `{"flag": "*", "reason": "could not check flags: HTTP Error 404..."}` — which is the correct "not drift" behavior, and worth recording as evidence the carve-out works.

- [ ] **Step 3: Full suite on this branch, captured by name**

```bash
doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) python3 -m pytest tests/ -q -p no:cacheprovider 2>&1 \
  | grep -E "^FAILED" | sed -E 's/^(FAILED [^ ]+).*/\1/' | sort -u > /tmp/fw_b.set
```

- [ ] **Step 4: Full suite on the `origin/main` baseline**

```bash
git -C ~/deploy-chat worktree add -q --detach /tmp/wt-fw-base origin/main
cd /tmp/wt-fw-base && doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) \
  python3 -m pytest tests/ -q -p no:cacheprovider 2>&1 \
  | grep -E "^FAILED" | sed -E 's/^(FAILED [^ ]+).*/\1/' | sort -u > /tmp/fw_m.set
```

- [ ] **Step 5: Diff by NAME, not count**

```bash
comm -13 /tmp/fw_m.set /tmp/fw_b.set   # must be EMPTY
```
Expected: no output. `main` carries ~96 pre-existing failures and is flaky, so counts prove nothing. If a name appears, run it in isolation on **both** branches before calling it a regression — `test_biofield_suggest_remedies*` and `test_portal_concierge_eval` are known nondeterministic (main fails 2 of 4 runs).

- [ ] **Step 6: Open the PR (do not merge)**

```bash
git push -u origin sess/5629cdf8-flagwatch
gh pr create --title "feat(ops): watch the four feature flags that must always be on" \
  --body "See docs/superpowers/specs/2026-07-09-flag-drift-watch-design.md"
```

**Merging is a production deploy (auto-deploy on merge to `main`) and requires Glen's explicit authorization, naming this PR. Do not merge.**

---

## Manual verification (after deploy)

1. `GET /api/console/flags` without a key → **401**.
2. With the console key → all four watched flags report `value: true`.
3. `CONSOLE_SECRET` does **not** appear in the response.
4. Temporarily delete `FIRESIDE_ENABLED` on the Render service and redeploy → `/begin/fireside` 404s **and** `check_flags` reports `env var is MISSING (deleted)`, distinct from a `false`. Restore it afterwards. **This is a production write — do it only with Glen present, or skip it.**
5. The next daily cron run emails one message listing both surface and flag problems, not two.

## Follow-ups (not this plan)

- **Root-cause the deletions.** Two fireside drifts in one day; the second was a deletion, not a value change. Render's events API does not log env changes, so attribution is unsolved.
- Cadence is daily. Drift is visible within 24 hours, not minutes.
- `/admin/shipping` notes field, and the 727 unmapped products.
