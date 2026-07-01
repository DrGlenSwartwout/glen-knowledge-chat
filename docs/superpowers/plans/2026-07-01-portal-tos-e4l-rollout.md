# Personal Portals + Combined RM/E4L TOS Rollout — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a combined Remedy Match + Energy For Life (E4L) first-access Terms gate on the client portal, then provision + email personal portals to today's confirmed MasterClass attendees — with a staged rollout to the full roster afterward.

**Architecture:** The portal, its token auth, the TOS gate, and the `agree-tos` endpoint already exist. Phase 1 is a small copy/version change to the gate (Task 1) plus an idempotent ops script that loops the existing `/admin/portal/upsert` endpoint over a roster CSV with `send:true` (Task 2). The larger "roll out to everyone" work (full-roster ingest, GHL staged send, E4L onboarding-on-first-access) is deferred to follow-on plans that need external inputs before they can be specified without placeholders.

**Tech Stack:** Python 3 / Flask (deploy-chat `app.py`), vanilla-JS portal (`static/client-portal.html`), Postgres (`journey_state`, `client_portals`), pytest.

## Global Constraints

- Combined TOS version string (verbatim): `rm-e4l-tc-2026-07-01` (replaces `rm-tc-2026-05-28`).
- Do **not** re-prompt clients who already agreed under a prior version — `is_member()` treats any non-null `tos_agreed_at` as agreed; this is intended ("if they have not already").
- Nothing emails a client until Task 1 is deployed and render-verified (the gate must cover E4L before any link goes out).
- Admin endpoint auth: `X-Console-Key: <CONSOLE_SECRET>` (or `?key=`).
- All ops scripts idempotent and re-runnable; default to `--dry-run`.
- Portal mint for an email with no scan/order is valid (no-scan home).

---

### Task 1: Combined RM + E4L TOS gate (SP2)

**Files:**
- Modify: `app.py:3052` (`BEGIN_TOS_VERSION`)
- Modify: `static/client-portal.html:364-368` (gate copy + links)
- Test: `tests/test_portal_tos_e4l.py` (create)

**Interfaces:**
- Consumes: `begin_funnel.record_unlock(cx, session_id=..., tos=True, tos_version=BEGIN_TOS_VERSION)` (existing), `begin_funnel.is_member(email=...)` (existing).
- Produces: constant `BEGIN_TOS_VERSION == "rm-e4l-tc-2026-07-01"`; portal gate HTML that links BOTH `illtowell.com/terms` and the E4L terms URL.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_portal_tos_e4l.py
import re, pathlib

def test_tos_version_is_combined_rm_e4l():
    import app
    assert app.BEGIN_TOS_VERSION == "rm-e4l-tc-2026-07-01"

def test_portal_gate_links_both_rm_and_e4l_terms():
    html = pathlib.Path("static/client-portal.html").read_text()
    # gate must name E4L and link the E4L terms alongside the RM terms
    assert "illtowell.com/terms" in html
    assert re.search(r"E4L|Energy For Life", html)
    assert re.search(r"E4L_TOS_URL|portal\.e4l\.com|truly\.vip/E4L", html)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-0986591c && python -m pytest tests/test_portal_tos_e4l.py -v`
Expected: FAIL (version still `rm-tc-2026-05-28`; no E4L reference in gate).

- [ ] **Step 3: Bump the version constant**

In `app.py:3052` change:
```python
BEGIN_TOS_VERSION = "rm-tc-2026-05-28"
```
to:
```python
BEGIN_TOS_VERSION = "rm-e4l-tc-2026-07-01"
```

- [ ] **Step 4: Update the gate copy + links**

Replace `static/client-portal.html:364-368` gate block with (adds E4L line + link, keeps the RM link and the existing button/error div):
```javascript
    var g = '<div class="card"><h2>Welcome to your healing home</h2>'
      + '<p>Before we continue, please review and agree to our Terms of Service. '
      + 'These cover both Remedy Match and the Energy For Life (E4L) biofield scan.</p>'
      + '<p><a href="' + esc((window.TOS_URL || "https://illtowell.com/terms")) + '" target="_blank" rel="noopener">Read the Remedy Match Terms</a>'
      + ' &nbsp;·&nbsp; '
      + '<a href="' + esc((window.E4L_TOS_URL || "https://truly.vip/E4L")) + '" target="_blank" rel="noopener">Read the E4L Terms</a></p>'
      + '<button class="btn" id="tosAgreeBtn">I agree to both Terms</button>'
      + '<div class="err" id="tosErr" hidden></div></div>';
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-0986591c && python -m pytest tests/test_portal_tos_e4l.py -v`
Expected: PASS (both tests).

- [ ] **Step 6: Commit**

```bash
cd /tmp/wt-deploy-chat-0986591c
git add app.py static/client-portal.html tests/test_portal_tos_e4l.py
git commit -m "feat(portal): combined RM+E4L first-access TOS gate (v rm-e4l-tc-2026-07-01)"
```

- [ ] **Step 7: Deploy + render-verify (owner-run)**

Open PR, merge, let Render deploy, then load a freshly-minted test portal and confirm: gate appears on first visit, shows both the Remedy Match and E4L terms links, "I agree to both Terms" records agreement (portal opens on reload), zero console errors. The `E4L_TOS_URL` env var may be set on Render to override the default link.

---

### Task 2: Attendee cohort provisioning script (SP1)

**Files:**
- Create: `scripts/portal_provision_cohort.py`
- Test: `tests/test_portal_provision_cohort.py` (create)

**Interfaces:**
- Consumes: a roster CSV with header `name,email,match_source,confidence`; env `CONSOLE_SECRET`, `PUBLIC_BASE_URL`.
- Produces: for each row, a `POST {BASE}/admin/portal/upsert` with `{email, name, send: <bool>}` and header `X-Console-Key`; writes a manifest CSV `email,name,portal_url,emailed,status`; function `build_payloads(rows, send)` returning the list of request bodies (pure, unit-testable).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_portal_provision_cohort.py
from scripts.portal_provision_cohort import build_payloads

def test_build_payloads_normalizes_and_flags_send():
    rows = [{"name": "Maria Sutryn", "email": "Maria_Sutryn@Outlook.com ", "confidence": "high"}]
    out = build_payloads(rows, send=True)
    assert out == [{"email": "maria_sutryn@outlook.com", "name": "Maria Sutryn", "send": True}]

def test_build_payloads_skips_blank_or_low_confidence():
    rows = [
        {"name": "No Email", "email": "", "confidence": "high"},
        {"name": "Fuzzy Match", "email": "x@y.com", "confidence": "unresolved"},
    ]
    assert build_payloads(rows, send=True) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-0986591c && python -m pytest tests/test_portal_provision_cohort.py -v`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Write the script**

```python
# scripts/portal_provision_cohort.py
"""Provision + optionally email personal portals for a roster CSV.
Idempotent: /admin/portal/upsert keeps an existing portal's token.
Usage:
  python scripts/portal_provision_cohort.py roster.csv --dry-run
  python scripts/portal_provision_cohort.py roster.csv --send --limit 1   # smoke test one
  python scripts/portal_provision_cohort.py roster.csv --send             # full cohort
"""
import argparse, csv, json, os, sys, urllib.request

_SKIP_CONFIDENCE = {"unresolved", "low", ""}

def build_payloads(rows, send):
    out = []
    for r in rows:
        email = (r.get("email") or "").strip().lower()
        conf = (r.get("confidence") or "").strip().lower()
        if not email or "@" not in email or conf in _SKIP_CONFIDENCE:
            continue
        out.append({"email": email, "name": (r.get("name") or "").strip(), "send": bool(send)})
    return out

def _post(base, key, body):
    req = urllib.request.Request(
        base.rstrip("/") + "/admin/portal/upsert",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-Console-Key": key},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("roster")
    ap.add_argument("--send", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--manifest", default="portal-manifest.csv")
    a = ap.parse_args(argv)
    base = os.environ["PUBLIC_BASE_URL"]
    key = os.environ["CONSOLE_SECRET"]
    with open(a.roster) as f:
        rows = list(csv.DictReader(f))
    payloads = build_payloads(rows, send=a.send)
    if a.limit:
        payloads = payloads[: a.limit]
    print(f"{len(payloads)} portals to provision (send={a.send}, dry_run={a.dry_run})")
    with open(a.manifest, "w", newline="") as mf:
        w = csv.writer(mf); w.writerow(["email", "name", "portal_url", "emailed", "status"])
        for p in payloads:
            if a.dry_run:
                print("DRY", p["email"], "send=" + str(p["send"])); w.writerow([p["email"], p["name"], "", "", "dry-run"]); continue
            try:
                res = _post(base, key, p)
                url = base.rstrip("/") + (res.get("url") or "")
                w.writerow([p["email"], p["name"], url, res.get("emailed"), "ok"])
                print("OK ", p["email"], url, "emailed=" + str(res.get("emailed")))
            except Exception as e:  # log + continue; never abort the whole cohort
                w.writerow([p["email"], p["name"], "", "", f"error:{e}"]); print("ERR", p["email"], e)
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-0986591c && python -m pytest tests/test_portal_provision_cohort.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-0986591c
git add scripts/portal_provision_cohort.py tests/test_portal_provision_cohort.py
git commit -m "feat(ops): idempotent portal cohort provisioning + send script"
```

- [ ] **Step 6: Smoke test then run (owner-run, AFTER Task 1 is live)**

```bash
# dry-run to eyeball the cohort
python scripts/portal_provision_cohort.py <roster.csv> --dry-run
# smoke: provision+email ONE (Glen's own email) and verify the email + gate
python scripts/portal_provision_cohort.py <roster.csv> --send --limit 1
# full confirmed cohort
python scripts/portal_provision_cohort.py <roster.csv> --send
```

---

## Follow-on plans (Phase 2/3 — need inputs before they can be planned without placeholders)

Each becomes its own spec→plan once its inputs are in hand:

- **SP0 Full-roster ingest** — populate `people` from FMP CSV ∪ e4l.db ∪ inbound_leads (v1) and the GHL-tagged PB tranche (v2). *Blocked on:* confirming GHL contact-search availability + that PB accounts are tagged in GHL.
- **SP4 Staged GHL send** — portal URL → GHL custom field → throttled workflow, with `portal_send_log` de-dupe. *Blocked on:* the GHL workflow id that emails the link + the portal-URL custom-field id (or create them).
- **SP3 E4L onboarding on first access** — detect email ∉ `e4l_clients`, collect missing details, trigger GHL E4L onboarding + prefilled `truly.vip/E4L`. *Blocked on:* the exact field set E4L account creation requires; confirm prod can read `e4l.db`.

## Self-review notes

- Spec coverage: Task 1 ⇒ SP2; Task 2 ⇒ SP1 (attendee scope). SP0/SP3/SP4 explicitly carried to follow-on plans with their blocking inputs named — not dropped.
- No placeholders in Task 1/2 steps (full code + commands given). Follow-on section names blockers rather than faking tasks.
- Type consistency: `build_payloads(rows, send)` signature and the `{email,name,send}` body shape are used identically in test and script; manifest columns fixed.
