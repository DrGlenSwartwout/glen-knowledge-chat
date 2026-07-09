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
import os
import sys
import urllib.error
import urllib.request

# Flag-gated surfaces + the core paths. Watching only fireside would have missed
# the outage class it belongs to.
PUBLIC_SURFACES = ("/", "/begin", "/begin/fireside", "/prepay", "/results")

BASE_URL = os.environ.get("PUBLIC_BASE_URL", "https://illtowell.com").rstrip("/")
OWNER_EMAIL = os.environ.get("GLEN_EMAIL", "drglenswartwout@gmail.com")


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


def format_alert(base_url, failures):
    """(subject, body) naming each dead path and why. Plain text; no HTML."""
    n = len(failures)
    host = base_url.split("//", 1)[-1].rstrip("/")
    subject = f"[surface-check] {n} dead surface{'s' if n != 1 else ''} on {host}"
    lines = [f"{n} public surface{'s' if n != 1 else ''} failing on {base_url}:", ""]
    for f in failures:
        why = f["error"] or f"HTTP {f['status']}"
        lines.append(f"  {f['path']}  ->  {why}")
    lines += [
        "",
        "A 404 on a flag-gated surface usually means its *_ENABLED env var drifted",
        "off on the Render web service. Flags absent from render.yaml have nothing",
        "pinning them on. Re-flip with a single-key PUT + an explicit POST /deploys.",
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
    """Probe, alert on failure, and always return the failure list. Best-effort by
    contract: the caller is the personal-email cron, which must never fail because
    a surface check did."""
    failures = check_surfaces(BASE_URL)
    if not failures:
        print(f"[surface-check] {len(PUBLIC_SURFACES)} surfaces OK on {BASE_URL}", flush=True)
        return failures
    subject, body = format_alert(BASE_URL, failures)
    print(f"[surface-check] {subject}", flush=True)
    for f in failures:
        print(f"[surface-check]   {f['path']} -> {f['error'] or f['status']}", flush=True)
    send_alert(subject, body)
    return failures


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
