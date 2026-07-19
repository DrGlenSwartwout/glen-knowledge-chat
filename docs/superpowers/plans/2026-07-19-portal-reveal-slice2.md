# Portal-Reveal Slice 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** On a new biofield scan, auto-provision a bare client portal (no System B report) and add a portal link to the reveal-ready email alongside the funnel link — behind a flag, going-forward only.

**Architecture:** A pure email-body builder + a best-effort, flag-gated app helper that provisions via `client_portal.ensure_token` and returns the `/portal/<token>` URL. Both reveal-email send sites (`_send_reveal_link` and the `api_e4l_reveal_draft` inline block) call them. Flag off = byte-identical to today.

**Tech Stack:** Python 3.11 Flask app, sqlite (`chat_log.db` / `LOG_DB`), pytest.

## Global Constraints

- **Flag `PORTAL_LINK_IN_REVEAL_ENABLED` default OFF.** Flag off ⇒ no provisioning, no portal line, email byte-identical to today.
- **Never write `portal_biofield_reports`** — provision ONLY via `client_portal.ensure_token` (creates `client_portals` row + stable token, no report). Never call publish/`upsert_report`.
- **Idempotent:** `ensure_token` returns the same stable token on repeat calls — re-sends must not mint duplicate portals.
- **Best-effort:** any provisioning error ⇒ `portal_url = None` ⇒ email still sends with just the funnel link; reveal ingest never breaks.
- **Augment, not replace:** the funnel reveal link stays; the portal line is additive and only present when a portal URL was produced.
- Provision/link only for non-suppressed emails (both send sites already guard suppression before building the body).
- Portal URL = `portal_link(token)` = `{portal_base()}/portal/{token}`. Portal token ≠ reveal token.

---

### Task 1: Pure reveal-email body builder

**Files:**
- Modify: `app.py` (add `_reveal_email_body` near `_send_reveal_link`, ~line 762)
- Test: `tests/test_reveal_email_body.py`

**Interfaces:**
- Produces: `_reveal_email_body(reveal_url, portal_url=None) -> str`. With `portal_url` falsy, returns exactly today's body; with a portal_url, inserts one portal paragraph between the reveal line and the sign-off.

- [ ] **Step 1: Write the failing test**

```python
from app import _reveal_email_body

ORIGINAL = ("Aloha,\n\nYour Biofield Analysis is ready. View your reading here:\n"
            "https://x/begin/biofield/tok\n\nIn wellness,\nDr. Glen and Rae\n")

def test_body_without_portal_is_byte_identical_to_original():
    assert _reveal_email_body("https://x/begin/biofield/tok") == ORIGINAL
    assert _reveal_email_body("https://x/begin/biofield/tok", None) == ORIGINAL

def test_body_with_portal_adds_one_portal_paragraph_and_keeps_reveal():
    body = _reveal_email_body("https://x/begin/biofield/tok", "https://x/portal/ptok")
    assert "https://x/begin/biofield/tok" in body          # funnel link kept
    assert "https://x/portal/ptok" in body                 # portal link added
    assert body.endswith("In wellness,\nDr. Glen and Rae\n")
    assert body.count("https://x/portal/ptok") == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_reveal_email_body.py -q` (under `/tmp/wt-deploy-chat-16e52882`; importing `app` needs env — if it errors on import, run under `doppler run --project remedy-match --config prd -- ~/.venvs/deploy-chat311/bin/python -m pytest ...`)
Expected: FAIL — `cannot import name '_reveal_email_body'`

- [ ] **Step 3: Implement**

Add to `app.py` near `_send_reveal_link`:

```python
def _reveal_email_body(reveal_url, portal_url=None):
    """Reveal-ready email body. Adds the client-portal line only when a portal URL
    is provided (slice 2, flag-gated); with portal_url falsy it is byte-identical to
    the original wording."""
    body = ("Aloha,\n\nYour Biofield Analysis is ready. View your reading here:\n"
            f"{reveal_url}\n")
    if portal_url:
        body += ("\nYour personal client portal — where your scans and matches live — "
                 f"is here:\n{portal_url}\n")
    body += "\nIn wellness,\nDr. Glen and Rae\n"
    return body
```

- [ ] **Step 4: Run to verify it passes**

Run the same pytest command. Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_reveal_email_body.py
git commit -m "feat(reveal-email): pure body builder (portal line optional, flag-gated later)"
```

---

### Task 2: Characterize `ensure_token` — bare provision, no report, idempotent

**Files:**
- Test: `tests/test_ensure_token_bare_portal.py`

**Interfaces:**
- Consumes (existing): `client_portal.ensure_token(cx, email, name="") -> raw_token`; `client_portal.init_client_portal_table`; `portal_biofield_reports.init_table`.

This task locks slice 2's core assumption with tests (no new source). If a test reveals `ensure_token` does NOT satisfy these guarantees, STOP and report — the whole slice depends on it.

- [ ] **Step 1: Write the test**

```python
import sqlite3
from dashboard import client_portal as cp
from dashboard import portal_biofield_reports as pbr

def _db():
    cx = sqlite3.connect(":memory:")
    cp.init_client_portal_table(cx)
    pbr.init_table(cx)
    return cx

def test_ensure_token_creates_portal_without_a_system_b_report():
    cx = _db()
    tok = cp.ensure_token(cx, "a@x.com", "Ann")
    assert tok                                             # a raw token
    # a client_portals row now exists for the email
    assert cx.execute("SELECT COUNT(*) FROM client_portals WHERE email='a@x.com'").fetchone()[0] == 1
    # and NO portal_biofield_reports row was written
    assert cx.execute("SELECT COUNT(*) FROM portal_biofield_reports WHERE email='a@x.com'").fetchone()[0] == 0

def test_ensure_token_is_idempotent():
    cx = _db()
    t1 = cp.ensure_token(cx, "b@x.com", "Bee")
    t2 = cp.ensure_token(cx, "b@x.com", "Bee")
    assert t1 == t2                                        # same stable token, no duplicate portal
    assert cx.execute("SELECT COUNT(*) FROM client_portals WHERE email='b@x.com'").fetchone()[0] == 1
```

- [ ] **Step 2: Run to verify behavior**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_ensure_token_bare_portal.py -q`
Expected: PASS. If it FAILS (e.g. a report row appears, or tokens differ), STOP and report — do not proceed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ensure_token_bare_portal.py
git commit -m "test(portal): lock ensure_token bare-provision + idempotency (slice-2 assumption)"
```

---

### Task 3: Flag + `_ensure_portal_link` helper, wired into `_send_reveal_link`

**Files:**
- Modify: `app.py` — add `_portal_link_in_reveal_enabled()` + `_ensure_portal_link(cx, email, name)`; refactor `_send_reveal_link` (app.py:~763-801) to use them + the Task 1 builder.

**Interfaces:**
- Consumes: `_reveal_email_body` (Task 1), `client_portal.ensure_token`, existing `portal_link(token)`, existing flag-reading pattern.
- Produces: `_ensure_portal_link(cx, email, name="") -> str | None` (flag off or error ⇒ None).

- [ ] **Step 1: Add the flag reader + helper**

Follow the existing flag-reader style in `app.py` (e.g. how `_client_login_enabled()` reads env). Add:

```python
def _portal_link_in_reveal_enabled():
    return (os.environ.get("PORTAL_LINK_IN_REVEAL_ENABLED", "").strip().lower()
            in ("1", "true", "yes", "on"))

def _ensure_portal_link(cx, email, name=""):
    """Provision a BARE client portal (no System B report) and return its
    /portal/<token> URL. Flag-gated + best-effort: returns None when the flag is off
    or on any error, so the reveal email still sends with just the funnel link.
    Idempotent (ensure_token returns the same stable token on repeat calls)."""
    if not _portal_link_in_reveal_enabled():
        return None
    try:
        from dashboard import client_portal as _cp
        tok = _cp.ensure_token(cx, (email or "").strip().lower(), name or "")
        return portal_link(tok) if tok else None
    except Exception as e:
        print(f"[portal-provision] {e!r}", flush=True)
        return None
```

(Confirm `os` is imported at module scope in app.py — it is. Confirm `portal_link` exists — app.py:~221.)

- [ ] **Step 2: Refactor `_send_reveal_link` to provision + use the builder**

In `_send_reveal_link(rid)` (app.py:~763-801), inside the existing `with _db_lock, sqlite3.connect(LOG_DB) as cx:` block, after the suppression guard and after the reveal token is minted, compute the portal link; then build the body with the builder. Replace the current inline body construction (`body = ("Aloha,...")`) with:

```python
    portal_url = _ensure_portal_link(cx, email, "")
    ...
    url = f"{PUBLIC_BASE_URL}/begin/biofield/{tok}"
    body = _reveal_email_body(url, portal_url)
```

Keep everything else (token mint, `auth_tokens` insert, `_send_inquiry_email(email, "Your Biofield Analysis is ready", body)`, `set_notified`) unchanged. `_ensure_portal_link` must be called with the same `cx` (inside the db block) and AFTER the suppression check so a suppressed address is never provisioned/linked.

- [ ] **Step 3: Compile + run Task 1/2 tests**

Run: `~/.venvs/deploy-chat311/bin/python -m py_compile app.py`
Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_reveal_email_body.py tests/test_ensure_token_bare_portal.py -q`
Expected: compile clean; 4 passed.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(portal): flag-gated bare-portal provision + portal link in _send_reveal_link"
```

---

### Task 4: Wire into the `api_e4l_reveal_draft` inline email block

**Files:**
- Modify: `app.py` — the `is_new and notify` block in `api_e4l_reveal_draft` (~app.py:24950-24968).

**Interfaces:**
- Consumes: `_ensure_portal_link`, `_reveal_email_body` (Tasks 1/3).

- [ ] **Step 1: Refactor the inline block**

In the ingest path's send block (after the suppression guard, inside the existing `_db_lock` block that has `cx`, `email`, and the freshly minted reveal `token`), replace the inline body build:

Current (~24961-24964):
```python
    url  = f"{PUBLIC_BASE_URL}/begin/biofield/{token}"
    body = ("Aloha,\n\nYour Biofield Analysis is ready. View your reading here:\n"
            f"{url}\n\nIn wellness,\nDr. Glen and Rae\n")
    _send_inquiry_email(email, "Your Biofield Analysis is ready", body)
```
Replace with:
```python
    portal_url = _ensure_portal_link(cx, email, "")
    url  = f"{PUBLIC_BASE_URL}/begin/biofield/{token}"
    body = _reveal_email_body(url, portal_url)
    _send_inquiry_email(email, "Your Biofield Analysis is ready", body)
```
Confirm from the surrounding code that `cx` is the open connection in scope at this point and that this is after the suppression guard (`email_suppression.is_suppressed`). If `cx` is not in scope here, provision with the same connection the block uses; do NOT open a second connection under the lock. If placement is ambiguous, STOP and ask.

- [ ] **Step 2: Compile**

Run: `~/.venvs/deploy-chat311/bin/python -m py_compile app.py`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(portal): live reveal-draft ingest also provisions portal + adds link (flag-gated)"
```

---

## Notes for the reviewer / verification

- **Flag-off safety:** with `PORTAL_LINK_IN_REVEAL_ENABLED` unset, `_ensure_portal_link` returns None at both sites, so `_reveal_email_body(url, None)` reproduces today's exact email and no portal is provisioned. Task 1's byte-identity test is the guard.
- **No isolated test for the two app.py send sites** (they need app import + SMTP); they are covered by compile + the pure-unit tests for their two collaborators (builder + ensure_token) + the final review reading the wiring. End-to-end (flag on → provisions + email carries the portal link) is verified post-merge on the live app with the flag enabled for a test address.
- **Money path:** unchanged — the funnel link and conversion flow are untouched; the portal link is purely additive.
