#!/usr/bin/env python3
"""Public-surface drift detector — piggybacked on the one always-on Render cron.

Why this exists: `FIRESIDE_ENABLED` is not declared in render.yaml, so nothing
pins it on. It silently drifted OFF and /begin/fireside 404'd in production while
/begin still shipped a live "Sit by the fire" CTA pointing straight at it. Nothing
alerted; the only symptom was visitors landing on a dead page.

So this checks the SYMPTOM a visitor experiences (an HTTP status on the real URL)
rather than the config we happen to suspect today. That also catches a deleted
route, a missing template, a bad deploy, or the app being down.

It runs in the CRON container, which is a separate service from the web app. That
matters: a checker living inside the web app cannot alert when the web app is the
thing that's broken.

Failure rule: `status >= 400`, or the request raised. A 3xx redirect is NOT a
failure — /prepay and /begin/choose legitimately 302, and a redirect is not a
dead page.

No database. No throttle table: the host cron runs daily, so the schedule IS the
throttle — at most one alert per day while a surface is down.

Stdlib only (the cron's buildCommand is `true`).
"""
import json
import os
import sys
import urllib.error
import urllib.request

# Flag-gated surfaces + the core paths. Watching only fireside would have missed
# the outage class it belongs to.
PUBLIC_SURFACES = ("/", "/begin", "/begin/fireside", "/prepay", "/results")

BASE_URL = os.environ.get("PUBLIC_BASE_URL", "https://illtowell.com").rstrip("/")
OWNER_EMAIL = os.environ.get("GLEN_EMAIL", "drglenswartwout@gmail.com")

CONSOLE_SECRET = os.environ.get("CONSOLE_SECRET", "")

# Flags that must ALWAYS be true, read from a COMMITTED baseline rather than hardcoded.
#
# A hardcoded tuple rots on the first PR that adds a flag, and it makes deliberately
# turning a flag off look identical to a deletion. With a committed baseline, disabling a
# flag is a PULL REQUEST — git records who intended it. Render sells no audit log on this
# plan (`audit logs not available for this plan`) and its events API never records env
# changes, so git is the only attribution we can get. That is the point; the alerting is
# secondary.
#
# A flag absent from the baseline is SILENT: new flags default off, and a watchdog that
# cries wolf gets ignored — which is how the 2026-07-09 deletion stayed invisible.
_BASELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "flags_expected.json")
BASELINE_ERROR = ""


def _load_baseline(path=_BASELINE_PATH):
    """(flags, error). Never raises — a broken baseline must be REPORTED, not silently
    treated as 'nothing is expected to be on'."""
    try:
        with open(path) as fh:
            data = json.load(fh)
        flags = data.get("expected_on")
        if not isinstance(flags, list) or not flags:
            return (), f"expected_on is {type(flags).__name__}, expected a non-empty list"
        if not all(isinstance(f, str) and f for f in flags):
            return (), "expected_on contains a non-string entry"
        return tuple(flags), ""
    except Exception as e:  # noqa: BLE001
        return (), f"{type(e).__name__}: {e}"


REQUIRED_ON, BASELINE_ERROR = _load_baseline()


def _fetch_json(url, key, timeout=20):
    """GET a console-gated JSON endpoint. Raises on any non-200."""
    req = urllib.request.Request(url, method="GET",
                                 headers={"X-Console-Key": key,
                                          "User-Agent": "surface-check/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")


def check_flags(base_url, console_key, required=None, fetch=_fetch_json):
    """One dict per flag that must be on and is not: {"flag", "reason"}.

    A deleted var, a deliberate false, and "set true but never redeployed" all reach the
    customer the same way, so all three alarm — but each names its own cause, because
    the fix differs. A CALL-TIME flag that was deleted has neither a module global nor an
    env key, so it vanishes from the report; the absent-from-response branch catches it.

    An unreachable or unauthorized endpoint is NOT drift: the surfaces list already
    alarms when the app is down, and one outage must not tell two contradictory stories.
    No console key -> skip entirely (the caller prints a notice)."""
    # Resolve at CALL time, not def time: a default of `required=REQUIRED_ON` would bind
    # once at import, so patching the module attribute (in tests, or after a baseline
    # reload) would silently have no effect.
    if required is None:
        required = REQUIRED_ON
    if not console_key:
        return []
    if not required:
        # The baseline failed to load. Expecting nothing and passing silently is the one
        # outcome a watchdog must never have.
        return [{"flag": "*",
                 "reason": f"could not load baseline: {BASELINE_ERROR or 'empty'}"}]
    url = f"{base_url.rstrip('/')}/api/console/flags"
    try:
        payload = fetch(url, console_key)
        flags = ((payload or {}).get("data") or {}).get("flags") or {}
        if not isinstance(flags, dict):
            raise TypeError(f"flags is {type(flags).__name__}, expected dict")
    except Exception as e:  # noqa: BLE001 — a check failure is never drift
        return [{"flag": "*", "reason": f"could not check flags: {e}"}]
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
        if not isinstance(info, dict):
            out.append({"flag": name,
                        "reason": f"could not check flags: malformed entry "
                                  f"({type(info).__name__}, expected object)"})
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


def _fetch(url, timeout=20):
    """GET a URL, returning its HTTP status. A 4xx/5xx is a status, not an exception."""
    req = urllib.request.Request(url, method="GET",
                                 headers={"User-Agent": "surface-check/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def check_surfaces(base_url, paths=PUBLIC_SURFACES, fetch=_fetch):
    """Return one dict per FAILING surface: {path, status, error}.

    status >= 400 -> failure. A raised request -> failure with status 0 and the
    error text. Healthy surfaces are omitted. One dead surface never short-circuits
    the rest: every path is probed."""
    failures = []
    for path in paths:
        url = f"{base_url.rstrip('/')}{path}"
        try:
            status = int(fetch(url))
        except Exception as e:  # noqa: BLE001 — the app being down must ALERT, not crash
            failures.append({"path": path, "status": 0, "error": str(e)})
            continue
        if status >= 400:
            failures.append({"path": path, "status": status, "error": ""})
    return failures


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


def send_alert(subject, body, to_email=None):
    """SMTP straight from the cron container (it carries SMTP_* creds). Never raises:
    a mail failure must not fail the host cron's real job."""
    to_email = to_email or OWNER_EMAIL
    host = os.environ.get("SMTP_HOST")
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASS")
    sender = os.environ.get("SMTP_FROM", user)
    if not (host and user and password):
        print("[surface-check] SMTP not configured — alert not emailed", flush=True)
        return False
    try:
        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = to_email
        with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", "587")), timeout=15) as s:
            s.starttls()
            s.login(user, password)
            s.sendmail(sender, [to_email], msg.as_string())
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[surface-check] alert email failed: {e!r}", flush=True)
        return False


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
        _flags_note = (f"{len(REQUIRED_ON)} flags on" if CONSOLE_SECRET
                       else "flags NOT checked (no CONSOLE_SECRET)")
        print(f"[surface-check] {len(PUBLIC_SURFACES)} surfaces OK, "
              f"{_flags_note}, at {BASE_URL}", flush=True)
        return []
    subject, body = format_alert(BASE_URL, failures, flag_failures)
    print(f"[surface-check] {subject}", flush=True)
    for f in failures:
        print(f"[surface-check]   {f['path']} -> {f['error'] or f['status']}", flush=True)
    for f in flag_failures:
        print(f"[surface-check]   {f['flag']} -> {f['reason']}", flush=True)
    send_alert(subject, body)
    return failures + flag_failures


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
