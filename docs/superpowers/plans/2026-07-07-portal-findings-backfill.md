# Portal Findings Backfill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill `content.findings` on portals published before #661, surgically (findings-only, no email, never create a portal), so their stress-pattern chips light up.

**Architecture:** A small console-gated prod endpoint does a read-modify-write of only `content.findings` on an existing portal record OR one of its biofield-report rows. A local driver script computes each client's findings via `scan_context` and pushes them, guarded to matched-email clients with non-empty findings, dry-run by default.

**Tech Stack:** Python/Flask (endpoint in `app.py`); pytest. Endpoint tests use `app.test_client()` and need the Doppler import harness; the driver's pure planner is plain pytest.

## Global Constraints

- **No email:** the endpoint has NO email/send code path.
- **Never create:** unknown email → 404; never insert a `client_portals` row.
- **Findings-only:** mutate exactly `content["findings"]`; read every other field and write it back unchanged.
- **One target per call:** `scan_date` present → patch that ONE report row; `scan_date` absent → patch the portal record. (Multi-scan clients get per-date findings via one call each.)
- **Matched-email + has-findings only** (driver): only target emails already in the portal set; skip empties; log every skip.
- **Idempotent; dry-run by default** (driver `--apply` to execute).
- **Trim contract:** findings are `{code, name, description, rank}` dicts, identical to #661.

---

### Task 1: `POST /api/console/portal/backfill-findings` endpoint

**Files:**
- Modify: `app.py` (add the route among the other `/api/console/portal-*` routes, e.g. right after `api_console_portal_link_resend`)
- Test: `tests/test_portal_backfill_findings.py` (new)

**Interfaces:**
- Consumes: `_portal_console_ok()`, `_db_lock`, `LOG_DB`; `dashboard.client_portal.{init_client_portal_table, get_portal_content_by_email, upsert_portal}`; `dashboard.portal_biofield_reports.{init_table, list_report_dates, get_report, upsert_report}`.
- Produces: `POST /api/console/portal/backfill-findings` — body `{email, findings:[...], scan_date?}` → `{ok, found, patched_portal, patched_reports}`. 404 when the email has no portal.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_portal_backfill_findings.py`:

```python
import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod._init_auth_tables()
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


_LAYER = {"n": 1, "title": "Calm", "meaning": "settle", "remedy": "Terrain Restore",
          "dosing": "10 drops 3x/day"}
_F = [{"code": "ED3", "name": "Cell Driver", "description": "supports cells", "rank": 1}]


def _seed(c, email, scan_date=None):
    body = {"email": email, "name": "X",
            "content": {"greeting": "Aloha", "layers": [_LAYER], "findings": []}}
    if scan_date:
        body["scan_date"] = scan_date
    r = c.post("/api/console/biofield-portal?key=test-secret", json=body)
    assert r.status_code == 200
    return r.get_json()["token"]


def test_backfill_requires_console_key(client):
    c, _ = client
    r = c.post("/api/console/portal/backfill-findings", json={"email": "a@b.com", "findings": []})
    assert r.status_code == 401


def test_backfill_unknown_email_404_and_no_create(client):
    c, appmod = client
    import sqlite3
    r = c.post("/api/console/portal/backfill-findings?key=test-secret",
               json={"email": "ghost@none.com", "findings": _F})
    assert r.status_code == 404
    assert r.get_json()["found"] is False
    with sqlite3.connect(appmod.LOG_DB) as cx:
        n = cx.execute("SELECT COUNT(*) FROM client_portals WHERE email=?",
                       ("ghost@none.com",)).fetchone()[0]
    assert n == 0  # never created


def test_backfill_patches_portal_record_findings_only(client):
    c, _ = client
    tok = _seed(c, "rec@b.com")  # no scan_date -> no report row -> portal-record path
    r = c.post("/api/console/portal/backfill-findings?key=test-secret",
               json={"email": "rec@b.com", "findings": _F})
    assert r.status_code == 200
    j = r.get_json()
    assert j["patched_portal"] is True and j["patched_reports"] == 0
    d = c.get(f"/api/portal/{tok}").get_json()
    assert d["findings"] == _F
    assert d["layers"][0]["title"] == "Calm"  # every other field intact


def test_backfill_patches_report_by_scan_date(client):
    c, _ = client
    tok = _seed(c, "rep@b.com", scan_date="2026-06-25")
    f2 = [{"code": "ET1", "name": "Heart Driver", "description": "h", "rank": 1}]
    r = c.post("/api/console/portal/backfill-findings?key=test-secret",
               json={"email": "rep@b.com", "scan_date": "2026-06-25", "findings": f2})
    assert r.status_code == 200
    assert r.get_json()["patched_reports"] == 1
    d = c.get(f"/api/portal/{tok}?scan_date=2026-06-25").get_json()
    assert d["findings"] == f2


def test_backfill_idempotent(client):
    c, _ = client
    tok = _seed(c, "idem@b.com")
    for _ in range(2):
        r = c.post("/api/console/portal/backfill-findings?key=test-secret",
                   json={"email": "idem@b.com", "findings": _F})
        assert r.status_code == 200
    assert c.get(f"/api/portal/{tok}").get_json()["findings"] == _F
```

- [ ] **Step 2: Run the tests to verify they fail**

Run (mkdir the scratch dir first):
```bash
mkdir -p /tmp/bf-endpoint-test
doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/bf-endpoint-test python3 -m pytest tests/test_portal_backfill_findings.py -q
```
Expected: FAIL — the route doesn't exist yet (404 for the console-key test would pass by accident, but the patch tests fail because the endpoint returns Flask's 404 HTML, not JSON with `found`/`patched_*`).

- [ ] **Step 3: Implement the endpoint**

In `app.py`, add this route among the other `/api/console/portal-*` routes (e.g. right after `def api_console_portal_link_resend(...)`):

```python
@app.route("/api/console/portal/backfill-findings", methods=["POST"])
def api_console_portal_backfill_findings():
    """Surgically set content.findings on an EXISTING portal (or one of its
    biofield-report rows) without touching any other field, sending any email, or
    ever creating a portal. Backfills portals published before findings were baked
    in at publish time. scan_date present -> patch that report row; absent -> patch
    the portal record. Console-key gated."""
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    findings = body.get("findings")
    scan_date = (body.get("scan_date") or "").strip()
    if not email or not isinstance(findings, list):
        return jsonify({"ok": False, "error": "email and findings[] required"}), 400
    from dashboard import client_portal as _cp, portal_biofield_reports as _pbr
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        _pbr.init_table(cx)
        rec = _cp.get_portal_content_by_email(cx, email)
        if not rec:
            return jsonify({"ok": False, "found": False}), 404   # never create
        patched_portal = False
        patched_reports = 0
        if scan_date:
            rep = _pbr.get_report(cx, email, scan_date)
            if rep:
                rc = dict(rep.get("content") or {})
                rc["findings"] = findings
                _pbr.upsert_report(cx, email, scan_date, rep.get("scan_id") or "",
                                   rc, rep.get("status") or "confirmed")
                patched_reports = 1
        else:
            content = dict(rec.get("content") or {})
            content["findings"] = findings
            _cp.upsert_portal(cx, email, rec.get("name") or "", content)
            patched_portal = True
    return jsonify({"ok": True, "found": True,
                    "patched_portal": patched_portal, "patched_reports": patched_reports})
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/bf-endpoint-test python3 -m pytest tests/test_portal_backfill_findings.py -q
```
Expected: PASS — 5 passed. (A `[CRON]` line during the run is a harmless background thread.)

- [ ] **Step 5: Commit**

Stage ONLY these two paths (never `git add -A`/`.`):
```bash
git add app.py tests/test_portal_backfill_findings.py
git commit -m "feat(portal): console endpoint to backfill findings on existing portals"
```

---

### Task 2: local driver script `scripts/backfill_portal_findings.py`

**Files:**
- Create: `scripts/backfill_portal_findings.py`
- Test: `tests/test_backfill_driver.py` (new)

**Interfaces:**
- Consumes: `dashboard.biofield_e4l.scan_context` (findings), the console API (`/api/console/portal-links`, `/api/console/portal-link`, `/api/portal/<token>`), and the local intake DB (`biofield_auth_tests`).
- Produces: `plan_backfill(portal_emails, intake_emails, report_dates_of, findings_of) -> (patches, skips)` (pure), and a `main()` CLI (dry-run default, `--apply`).

- [ ] **Step 1: Write the failing tests for the pure planner**

Create `tests/test_backfill_driver.py`:

```python
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "backfill_driver",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "backfill_portal_findings.py"))
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


_F = [{"code": "ED3", "name": "Cell Driver", "description": "d", "rank": 1}]


def test_skips_email_without_portal():
    patches, skips = mod.plan_backfill(
        portal_emails={"has@p.com"},
        intake_emails=["missing@p.com"],
        report_dates_of=lambda e: [],
        findings_of=lambda e, d: _F)
    assert patches == []
    assert skips == [{"email": "missing@p.com", "reason": "no existing portal (would create/dup)"}]


def test_matched_with_report_dates_patches_each_date():
    patches, skips = mod.plan_backfill(
        portal_emails={"a@p.com"},
        intake_emails=["a@p.com"],
        report_dates_of=lambda e: ["2026-06-25", "2026-07-01"],
        findings_of=lambda e, d: _F if d == "2026-06-25" else [])
    # only the date that yields findings is patched; the empty one is dropped
    assert patches == [{"email": "a@p.com", "scan_date": "2026-06-25", "findings": _F}]
    assert skips == []


def test_matched_no_report_dates_patches_portal_record():
    patches, skips = mod.plan_backfill(
        portal_emails={"a@p.com"},
        intake_emails=["a@p.com"],
        report_dates_of=lambda e: [],
        findings_of=lambda e, d: _F)
    assert patches == [{"email": "a@p.com", "scan_date": None, "findings": _F}]


def test_matched_but_no_findings_is_skipped():
    patches, skips = mod.plan_backfill(
        portal_emails={"a@p.com"},
        intake_emails=["a@p.com"],
        report_dates_of=lambda e: [],
        findings_of=lambda e, d: [])
    assert patches == []
    assert skips == [{"email": "a@p.com", "reason": "no findings computed"}]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_backfill_driver.py -q`
Expected: FAIL — `scripts/backfill_portal_findings.py` doesn't exist (module load error).

- [ ] **Step 3: Write the driver script**

Create `scripts/backfill_portal_findings.py`:

```python
#!/usr/bin/env python3
"""Backfill content.findings on portals published before findings were baked in
at publish time (#661). Surgical + guarded: only emails that already have a portal
AND whose scan yields findings are patched, via the findings-only endpoint. No
email is ever sent; no portal is ever created. Dry-run by default; --apply executes.

Env: CONSOLE_SECRET (console key), PORTAL_PUBLISH_BASE_URL or --base (prod base),
E4L_DB (defaults to ~/AI-Training/e4l.db via dashboard.biofield_e4l).

Run:  python3 scripts/backfill_portal_findings.py            # dry-run
      python3 scripts/backfill_portal_findings.py --apply    # execute
"""
import argparse
import json
import os
import sqlite3
import sys
import urllib.parse
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # repo root, for `dashboard`
from dashboard.biofield_e4l import scan_context  # noqa: E402

INTAKE_DB = os.environ.get(
    "BIOFIELD_DB", os.path.join(os.path.dirname(_HERE), "chat_log.db"))


def _trim(raw):
    return [{"code": f.get("code", ""), "name": f.get("name", ""),
             "description": f.get("description", ""), "rank": f.get("rank")}
            for f in (raw or [])]


def plan_backfill(portal_emails, intake_emails, report_dates_of, findings_of):
    """Pure planner. Returns (patches, skips).
    patches: [{email, scan_date (None=portal record), findings}]. skips: [{email, reason}]."""
    patches, skips = [], []
    for email in intake_emails:
        e = (email or "").strip().lower()
        if e not in portal_emails:
            skips.append({"email": e, "reason": "no existing portal (would create/dup)"})
            continue
        dates = report_dates_of(e) or []
        entries = []
        if dates:
            for d in dates:
                f = findings_of(e, d) or []
                if f:
                    entries.append({"email": e, "scan_date": d, "findings": f})
        else:
            f = findings_of(e, None) or []
            if f:
                entries.append({"email": e, "scan_date": None, "findings": f})
        if not entries:
            skips.append({"email": e, "reason": "no findings computed"})
            continue
        patches.extend(entries)
    return patches, skips


def _get(url, key):
    req = urllib.request.Request(url, headers={"X-Console-Key": key})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def _post(url, key, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"X-Console-Key": key,
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="execute (default: dry-run)")
    ap.add_argument("--base", default=os.environ.get("PORTAL_PUBLISH_BASE_URL", ""))
    args = ap.parse_args(argv)
    key = os.environ.get("CONSOLE_SECRET", "")
    base = args.base.rstrip("/")
    if not key or not base:
        print("need CONSOLE_SECRET and --base/PORTAL_PUBLISH_BASE_URL", file=sys.stderr)
        return 2

    portals = _get(f"{base}/api/console/portal-links", key).get("portals", [])
    portal_emails = {(p.get("email") or "").strip().lower()
                     for p in portals if p.get("has_token")}

    cx = sqlite3.connect(f"file:{INTAKE_DB}?mode=ro", uri=True)
    intake_emails = sorted({(r[0] or "").strip().lower()
                            for r in cx.execute("SELECT email FROM biofield_auth_tests")
                            if (r[0] or "").strip()})
    cx.close()

    # token cache so report_dates_of can read /api/portal/<token> scan_dates
    _tok = {}
    def token_for(email):
        if email not in _tok:
            j = _get(f"{base}/api/console/portal-link?email={urllib.parse.quote(email)}", key)
            link = j.get("link") or ""
            _tok[email] = link.rsplit("/portal/", 1)[-1] if "/portal/" in link else ""
        return _tok[email]

    def report_dates_of(email):
        tok = token_for(email)
        if not tok:
            return []
        d = _get(f"{base}/api/portal/{tok}", key)
        return d.get("scan_dates") or []

    def findings_of(email, scan_date):
        # scan_date None -> latest scan for the portal-record patch
        from datetime import date
        today = scan_date or date.today().isoformat()
        try:
            return _trim(scan_context(email, today).get("findings") or [])
        except Exception:
            return []

    patches, skips = plan_backfill(portal_emails, intake_emails, report_dates_of, findings_of)

    for s in skips:
        print(f"SKIP {s['email']}: {s['reason']}")
    for p in patches:
        tgt = f"report {p['scan_date']}" if p["scan_date"] else "portal record"
        print(f"{'APPLY' if args.apply else 'DRY '} {p['email']} -> {tgt} "
              f"({len(p['findings'])} findings)")
        if args.apply:
            body = {"email": p["email"], "findings": p["findings"]}
            if p["scan_date"]:
                body["scan_date"] = p["scan_date"]
            res = _post(f"{base}/api/console/portal/backfill-findings", key, body)
            print(f"      -> {res}")
    print(f"\n{len(patches)} patch(es), {len(skips)} skip(s). "
          f"{'APPLIED' if args.apply else 'DRY-RUN (use --apply)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the planner tests to verify they pass**

Run: `python3 -m pytest tests/test_backfill_driver.py -q`
Expected: PASS — 4 passed. (The test loads the module and exercises only `plan_backfill`; no network/DB.)

- [ ] **Step 5: Commit**

```bash
git add scripts/backfill_portal_findings.py tests/test_backfill_driver.py
git commit -m "feat(portal): local driver to backfill findings (guarded, dry-run default)"
```

---

## Notes for the reviewer / executor

- The endpoint tests need the Doppler harness (`import app` validates Pinecone at import); the driver planner tests are plain pytest (module-load + pure function).
- Live run is a separate manual step (not in this plan): after both merge & deploy, run the driver dry-run, eyeball the 3 matched clients, then `--apply`, then render one portal. The driver never touches an email absent from the portal set, so it cannot email or duplicate.
- `findings_of(email, None)` uses today's date to fetch the latest scan for the portal-record patch; report patches use the exact report `scan_date`, keeping each report aligned to its own scan.
