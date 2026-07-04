# Attributed Prepay-Term Enrollment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a doctor enroll a patient in a 6- or 12-month **prepaid** Continuous Care term through their dispensary — earning a cert-scaled fee-share on the lump and making the prepay patient visible in the doctor's continuity tooling (A2: attribute the prepay grant, no shadow membership).

**Architecture:** The prepay-lump path (`_fulfill_prepay_term`) gains attribution: `prepay_term_grants` stores `attributed_practitioner_id` + `practitioner_share_consent` + `term_end`; fulfillment fires the fee-share on the lump (reusing `care_share`/`wallet`); `continuity_view`'s gate/roster UNION attributed, in-term prepay grants alongside the subscriptions gate; the dispensary card + endpoint offer the prepaid terms.

**Tech Stack:** Python 3, Flask, SQLite (`prepay_term_grants`, `subscriptions` in LOG_DB), Stripe, pytest, vanilla JS.

## Global Constraints

- Terms offered through the doctor: `1mo` (existing #565 monthly path, unchanged), `6mo` ($54600), `12mo` ($99000). Use `dashboard/prepay.py` `TIERS`/`get_tier`.
- Fee-share on a prepaid lump: `wallet.earn_care_share(str(pid), care_share.share_cents(tier["price_cents"], care_share.modules_for_practitioner(pid)), event_ref=f"care_share:prepay:{session_id}")` — fires ONCE, base = the **full lump**, rate read live, only when `dispensary_pid` is present AND `modules_for_practitioner(pid)` is not None (a real practitioner). Idempotent per `session_id` (the `care_share:prepay:` event_ref is distinct from the monthly cron's `care_share:<sub_id>:<order_count>`).
- The public (non-dispensary) prepay flow is UNCHANGED — no `dispensary_pid` → no attribution, no credit.
- C's gate stays the single access boundary: it now reads TWO sources (subscriptions membership OR in-term attributed prepay grant), with the SAME per-source predicate discipline. `authorized_patient` and `roster` must stay in lockstep.
- Money in integer cents. Best-effort credit (try/except) must never undo the committed grant.

---

## File Structure

- **Modify** `app.py` — `_fulfill_prepay_term` (attribution stamp + fee-share); `dispensary_continuous_care` (prepay-term branch); the `prepay_term_grants` schema (inline `CREATE TABLE` + a guarded column-add).
- **Modify** `dashboard/continuity_view.py` — `authorized_patient` + `roster` UNION prepay grants.
- **Modify** `static/practitioner-client.html` — term selector on the CC card.
- **Tests** — `tests/test_care_share_prepay.py` (fulfillment fee-share + attribution), `tests/test_continuity_prepay_gate.py` (gate UNION), extend a dispensary-enroll test for the endpoint branch.

---

### Task 1: `prepay_term_grants` — attribution columns

**Files:** Modify `app.py` (the `prepay_term_grants` `CREATE TABLE` in `_fulfill_prepay_term` ~L7458, + a guarded column-add helper); Test `tests/test_care_share_prepay.py`

**Interfaces:**
- Produces: `prepay_term_grants` has `attributed_practitioner_id TEXT`, `practitioner_share_consent INTEGER NOT NULL DEFAULT 0`, `term_end TEXT`; a helper `_ensure_prepay_grant_columns(cx)` that adds them idempotently.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_care_share_prepay.py
import sqlite3
def test_prepay_grant_columns_present():
    import app as appmod
    cx = sqlite3.connect(":memory:")
    cx.execute("CREATE TABLE IF NOT EXISTS prepay_term_grants (session_id TEXT PRIMARY KEY, email TEXT, tier_key TEXT, granted_at TEXT)")
    appmod._ensure_prepay_grant_columns(cx)
    cols = {r[1] for r in cx.execute("PRAGMA table_info(prepay_term_grants)")}
    assert {"attributed_practitioner_id", "practitioner_share_consent", "term_end"} <= cols
```

Note: importing `app` requires env — run tests via `doppler run -p remedy-match -c dev -- python -m pytest ...`.

- [ ] **Step 2: Run to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python -m pytest tests/test_care_share_prepay.py::test_prepay_grant_columns_present -q`
Expected: FAIL (`AttributeError: _ensure_prepay_grant_columns`)

- [ ] **Step 3: Add the helper + widen the CREATE TABLE**

Add near `_fulfill_prepay_term`:

```python
def _ensure_prepay_grant_columns(cx):
    """Idempotently add the attribution columns to prepay_term_grants."""
    have = {r[1] for r in cx.execute("PRAGMA table_info(prepay_term_grants)")}
    if "attributed_practitioner_id" not in have:
        cx.execute("ALTER TABLE prepay_term_grants ADD COLUMN attributed_practitioner_id TEXT")
    if "practitioner_share_consent" not in have:
        cx.execute("ALTER TABLE prepay_term_grants ADD COLUMN practitioner_share_consent INTEGER NOT NULL DEFAULT 0")
    if "term_end" not in have:
        cx.execute("ALTER TABLE prepay_term_grants ADD COLUMN term_end TEXT")
```

In `_fulfill_prepay_term`, immediately after the `CREATE TABLE IF NOT EXISTS prepay_term_grants (...)` statement, call `_ensure_prepay_grant_columns(cx)` (so existing prod tables gain the columns).

- [ ] **Step 4: Run to verify it passes** — GREEN.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_care_share_prepay.py
git commit -m "feat(prepay): attribution columns on prepay_term_grants"
```

---

### Task 2: `_fulfill_prepay_term` — stamp attribution + fire the fee-share on the lump

**Files:** Modify `app.py` (`_fulfill_prepay_term`, ~L7435); Test `tests/test_care_share_prepay.py`

**Interfaces:**
- Consumes: `_ensure_prepay_grant_columns` (Task 1); `care_share.share_cents`/`modules_for_practitioner`, `wallet.earn_care_share`, `prepay.term_end_date` / `get_tier`.
- Produces: an attributed prepay fulfillment stamps the grant row + credits the doctor once on the lump.

- [ ] **Step 1: Write the failing test**

Read the existing `_fulfill_prepay_term` tests (grep `_fulfill_prepay_term` in tests/) for how they stub `stripe_pay.get_session`/`get_payment_intent` to simulate a succeeded `prepay_term` session. Then:

```python
def test_attributed_prepay_credits_doctor_on_lump(monkeypatch, ...):
    # stub get_session to return metadata {kind:'prepay_term', email, tier_key:'12mo',
    #   dispensary_pid:'prac-42', share_consent:'1'} + payment_intent succeeded
    # patch care_share.modules_for_practitioner -> 12 (full cert) for determinism
    # patch wallet.earn_care_share to a recorder
    _fulfill_prepay_term(session_id)
    assert recorder.called_once_with pid='prac-42', cents=share_cents(99000, 12), event_ref='care_share:prepay:<sid>'
    # grant row stamped:
    row = <query prepay_term_grants for session_id>
    assert row.attributed_practitioner_id == 'prac-42' and row.practitioner_share_consent == 1 and row.term_end  # non-null

def test_public_prepay_no_dispensary_no_credit(monkeypatch, ...):
    # same but metadata has NO dispensary_pid -> earn_care_share NEVER called; grant still created
    ...

def test_attributed_prepay_idempotent(...):
    # two _fulfill_prepay_term(session_id) calls -> earn_care_share fires at most once (session-claim + event_ref dedup)
    ...
```

- [ ] **Step 2: Run to verify it fails** — the credit is never fired / grant not stamped.

- [ ] **Step 3: Implement** — inside `_fulfill_prepay_term`, in the `if claimed:` block (after `_grant_prepay_term(cx, email, tier_key)`), stamp the grant + fire the credit:

```python
                # Attributed dispensary prepay: stamp the grant + credit the doctor on the lump.
                disp_pid = (md.get("dispensary_pid") or "").strip() or None
                share_consent = 1 if (md.get("share_consent") or "").strip() == "1" else 0
                _term_end = _pp.term_end_date(datetime.utcnow().date().isoformat(), tier["months"])
                if disp_pid:
                    cx.execute(
                        "UPDATE prepay_term_grants SET attributed_practitioner_id=?, "
                        "practitioner_share_consent=?, term_end=? WHERE session_id=?",
                        (str(disp_pid), share_consent, _term_end, session_id))
                    cx.commit()
                    try:
                        from dashboard import care_share as _cshare, wallet as _wallet
                        m = _cshare.modules_for_practitioner(disp_pid)
                        if m is not None:
                            cents = _cshare.share_cents(int(tier["price_cents"]), m)
                            if cents > 0:
                                _wallet.earn_care_share(str(disp_pid), cents,
                                                        event_ref=f"care_share:prepay:{session_id}")
                    except Exception as _ce:
                        print(f"[prepay] care-share credit failed sid={session_id}: {_ce!r}", flush=True)
```

(The care-share block is best-effort and AFTER the committed grant — a credit failure must never undo the term. `session_id` is unique + `earn_care_share` dedups on the event_ref, so the redirect+webhook double-call credits once.)

- [ ] **Step 4: Run to verify it passes** — GREEN (all three tests).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_care_share_prepay.py
git commit -m "feat(prepay): attributed prepay fulfilment stamps grant + credits doctor on the lump"
```

---

### Task 3: Dispensary endpoint — prepay-term branch

**Files:** Modify `app.py` (`dispensary_continuous_care`); Test `tests/test_care_share_prepay.py` (or extend the dispensary-enroll test)

**Interfaces:**
- Consumes: `_pp.practitioner_id_by_dispensary_code`, `prepay.get_tier`, `stripe_pay.create_checkout_session`.
- Produces: `POST /dispensary/<code>/continuous-care` with `tier_key` in `6mo/12mo` builds a `prepay_term` Stripe session carrying `dispensary_pid` + `share_consent`; `1mo`/absent → the existing monthly path.

- [ ] **Step 1: Write the failing test** — post the endpoint with `tier_key='12mo'` (+ consent gate satisfied like the sibling dispensary tests) and assert `stripe_pay.create_checkout_session` was called with `metadata` containing `kind='prepay_term'`, `tier_key='12mo'`, `dispensary_pid=<pid>`, `share_consent`; and with the tier's `price_cents` (99000). Assert `tier_key='1mo'` still routes to the monthly `continuous_care_monthly` kind (unchanged).

- [ ] **Step 2: Run to verify it fails** — the endpoint ignores `tier_key` (always monthly).

- [ ] **Step 3: Implement** — in `dispensary_continuous_care`, read `tier_key = (body.get("tier_key") or "1mo").strip()`. If `tier_key in ("6mo","12mo")`: resolve `tier = _pp.get_tier(tier_key)`, build a `prepay_term` Stripe session mirroring the public prepay-ladder checkout's session build (grep the public prepay checkout route that emits `kind:"prepay_term"` — reuse its `create_checkout_session` shape, NO `save_card`), with `metadata={"email":email, "kind":"prepay_term", "tier_key":tier_key, "dispensary_pid":str(pid), "share_consent": ("1" if <consent flag from body> else "0")}` and the prepay return/cancel URLs; return `{ok, url}`. Otherwise keep the existing monthly branch untouched. Carry the `share_consent` value from the POST body the SAME way the monthly path already does (grep how `dispensary_continuous_care`/its monthly session sets `share_consent`).

- [ ] **Step 4: Run to verify it passes** — GREEN.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_care_share_prepay.py
git commit -m "feat(prepay): dispensary Start-Continuous-Care offers attributed 6mo/12mo prepay terms"
```

---

### Task 4: `continuity_view` gate/roster — UNION attributed in-term prepay grants

**Files:** Modify `dashboard/continuity_view.py`; Test `tests/test_continuity_prepay_gate.py`

**Interfaces:**
- Produces: `authorized_patient`/`roster` return True/include a patient who has EITHER a consented continuity subscriptions membership OR an attributed, consented, in-term (`term_end >= today`) prepay grant.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_continuity_prepay_gate.py
import sqlite3
from dashboard import continuity_view as cv

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    cx.execute("CREATE TABLE subscriptions (email TEXT, attributed_practitioner_id TEXT, practitioner_share_consent INT, kind TEXT, status TEXT)")
    cx.execute("CREATE TABLE prepay_term_grants (session_id TEXT PRIMARY KEY, email TEXT, tier_key TEXT, granted_at TEXT, attributed_practitioner_id TEXT, practitioner_share_consent INT DEFAULT 0, term_end TEXT)")
    return cx

def _grant(cx, email, pid, consent, term_end):
    cx.execute("INSERT INTO prepay_term_grants (session_id,email,tier_key,attributed_practitioner_id,practitioner_share_consent,term_end) VALUES (?,?,?,?,?,?)",
               ("s"+email, email, "12mo", pid, consent, term_end)); cx.commit()

def test_prepay_in_term_consented_is_authorized():
    cx=_cx(); _grant(cx,"pat@x.com","prac-42",1,"2999-01-01")
    assert cv.authorized_patient(cx,"prac-42","pat@x.com") is True
    assert "pat@x.com" in [r["email"] for r in cv.roster(cx,"prac-42")]

def test_prepay_expired_denied():
    cx=_cx(); _grant(cx,"pat@x.com","prac-42",1,"2000-01-01")
    assert cv.authorized_patient(cx,"prac-42","pat@x.com") is False

def test_prepay_unconsented_denied():
    cx=_cx(); _grant(cx,"pat@x.com","prac-42",0,"2999-01-01")
    assert cv.authorized_patient(cx,"prac-42","pat@x.com") is False

def test_prepay_other_doctor_denied():
    cx=_cx(); _grant(cx,"pat@x.com","prac-42",1,"2999-01-01")
    assert cv.authorized_patient(cx,"prac-99","pat@x.com") is False
```

Also keep an existing subscriptions-only test green (no regression).

- [ ] **Step 2: Run to verify it fails** — `authorized_patient` only checks subscriptions.

- [ ] **Step 3: Implement** — add a prepay check to both functions, guarded for a missing table (`try/except sqlite3.OperationalError`). Use a module-level `import datetime` and `datetime.date.today().isoformat()` for "today":

```python
def _prepay_authorized(cx, practitioner_id, patient_email):
    try:
        today = _dt.date.today().isoformat()
        r = cx.execute(
            "SELECT 1 FROM prepay_term_grants WHERE lower(email)=lower(?) "
            "AND attributed_practitioner_id=? AND practitioner_share_consent=1 "
            "AND term_end >= ? LIMIT 1",
            ((patient_email or "").strip(), str(practitioner_id), today)).fetchone()
        return r is not None
    except sqlite3.OperationalError:
        return False   # prepay_term_grants not present yet
```

Wire `authorized_patient` to `return (<existing subscriptions row check>) or _prepay_authorized(...)`. For `roster`, UNION the prepay emails: after the subscriptions `SELECT DISTINCT lower(email)`, add (guarded) `SELECT DISTINCT lower(email) FROM prepay_term_grants WHERE attributed_practitioner_id=? AND practitioner_share_consent=1 AND term_end >= ?`, dedupe the two email sets, and map through `_display_name`. Import `datetime as _dt` and `sqlite3` at the top of `continuity_view.py` if not already present.

- [ ] **Step 4: Run to verify it passes** — GREEN (prepay cases + existing subscriptions cases).

- [ ] **Step 5: Commit**

```bash
git add dashboard/continuity_view.py tests/test_continuity_prepay_gate.py
git commit -m "feat(prepay): continuity gate/roster include attributed in-term prepay patients"
```

---

### Task 5: Dispensary card — term selector

**Files:** Modify `static/practitioner-client.html` (the #565 "Start Continuous Care" card)

This is UI-only (no pytest cycle).

- [ ] **Step 1: Add the term selector** — on the CC card, add a Monthly / 6 months / 12 months choice (radio/segmented control) populated from the three tiers (Monthly $99 / 6 months $546 / 12 months $99/mo-equivalent — reuse the tier labels/prices; do NOT hardcode if a tiers payload is available). Default Monthly.

- [ ] **Step 2: Wire the POST** — the "Start Continuous Care" handler includes `tier_key` (the selected tier's key: `1mo`/`6mo`/`12mo`) in the POST body to `/dispensary/<code>/continuous-care`, alongside the existing `share_consent`. On success use the returned `url` exactly as the monthly path does.

- [ ] **Step 3: Verify** — extract the `<script>` and `node --check`; confirm the POST carries `tier_key` + `share_consent`. Report that live browser render is pending (controller will render-verify).

- [ ] **Step 4: Commit**

```bash
git add static/practitioner-client.html
git commit -m "feat(prepay): dispensary Start-Continuous-Care term selector (Monthly / 6mo / 12mo)"
```

---

## Self-Review

**Spec coverage:**
- Terms Monthly/6mo/12mo → Task 3 (endpoint) + Task 5 (UI). ✓
- Fee-share on the lump (full base, live rate, once, idempotent, attributed-only) → Task 2. ✓
- Attribution columns on the grant → Task 1; stamped → Task 2. ✓
- C-visibility via gate/roster UNION (in-term, consented) → Task 4. ✓
- Public prepay unaffected → Task 2 (no dispensary_pid → no credit), Task 3 (only 6mo/12mo branch changes). ✓

**Placeholder scan:** Tasks 2/3 reference the existing `_fulfill_prepay_term` stubs + the public prepay-checkout session build to reuse (rather than inventing the Stripe/session shape) — appropriate for edits into existing flows whose exact shape must be read. Task 4 has complete code (I have the exact current gate/roster queries). Tasks 1/4 code is complete.

**Type consistency:** `_ensure_prepay_grant_columns(cx)` (Task 1) used in Task 2. `care_share.share_cents(cents, modules)` / `modules_for_practitioner(pid)` / `wallet.earn_care_share(pid, cents, *, event_ref)` — same signatures as #565 (verified in that merged code). `term_end` column (Task 1) written in Task 2, read in Task 4. `prepay.term_end_date(start_yyyy_mm_dd, months)` / `get_tier(key)` per `dashboard/prepay.py`.

## Notes / open confirmations
- **Idempotency layering:** the grant claim (`prepay_term_grants(session_id)` PK) + `earn_care_share` event_ref dedup both guard against redirect+webhook double-fire; confirm the credit sits INSIDE the `if claimed:` block (so a second fulfilment that loses the claim doesn't re-credit).
- **Consent source on the endpoint (Task 3):** carry `share_consent` from the POST body exactly as the monthly `dispensary_continuous_care` path already does — read that first.
- Renewal attribution (a renewed prepay term staying attributed) is OUT of scope (spec Future).
