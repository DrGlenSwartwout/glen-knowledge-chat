# SP2b-4 — ASH ally on /member/scan-analysis/chat — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the ASH ally memory layer into `/member/scan-analysis/chat` — the last of the 8 client chat surfaces.

**Architecture:** Two additive touches in `member_scan_analysis_chat` (app.py:13545): prepend `ash_ally.ally_overlay(LOG_DB, email)` to the assembled `system` string, and fire `ash_ally.record_turn(...)` on a background daemon thread after the answer. Subject = the `rm_member_email` cookie. Same `ASH_ALLY_ENABLED` flag, dark.

**Tech Stack:** the existing `dashboard/ash_ally.py`; one Flask handler in `app.py`. No new tests (helper already covered) — gated by syntax-parse + grep, behavioral proof at go-live render-verify.

## Global Constraints

- Modify: `app.py` only (`member_scan_analysis_chat`, ~13545-13590).
- Subject email = `email` (the `rm_member_email` cookie, already resolved at line 13555). No new lookup.
- `from dashboard import ash_ally`, `LOG_DB`, `_db_lock` already imported (SP2b-1). Do not re-add.
- Overlay prepended to the local `system` var AFTER it's fully assembled (base + facts-or-educate-policy) and BEFORE `_cl.messages.create`. Only when non-empty.
- Record on a background daemon thread, try/except-wrapped, ONLY when `email` is non-empty; args `(LOG_DB, _db_lock, email, q, answer)`.
- Purely additive — no existing behavior changes when there's no member cookie (`email=""` → no overlay, no record).
- Edit ONLY `member_scan_analysis_chat` (identify by the `/member/scan-analysis/chat` route + `_SCAN_CHAT_SYSTEM` + `rm_member_email`). Do NOT touch the sibling `/member/scan-analysis` page handler or `/api/e4l/scan-analysis`.

---

### Task 1: Wire `/member/scan-analysis/chat` (overlay + record) + verify

Single additive wiring on one handler. No TDD (no new logic; the helper is unit-tested). Gate = ast.parse + grep. Behavioral proof = go-live render-verify.

**Files:**
- Modify: `app.py` (`member_scan_analysis_chat`)

**Interfaces:**
- Consumes: `ash_ally.ally_overlay(LOG_DB, email)`, `ash_ally.record_turn(LOG_DB, _db_lock, email, q, answer)`.

- [ ] **Step 1: Add the overlay touch**

The handler assembles `system` across lines ~13574-13578:

```python
    system = _SCAN_CHAT_SYSTEM
    if ctx["grounded"] and ctx["facts"]:
        system = system + "\n\nTHE MEMBER'S ANALYSIS FACTS:\n" + ctx["facts"]
    else:
        system = system + _EDUCATE_ONLY_POLICY
```

Immediately after that block (after the `else:` branch closes, before the `try:` that calls `_cl.messages.create`), insert:

```python
    _ally_ov = ash_ally.ally_overlay(LOG_DB, email)
    if _ally_ov:
        system = _ally_ov + "\n\n" + system
```

- [ ] **Step 2: Add the record touch**

After `answer` is produced and the response is built (`resp = jsonify({"answer": answer, ...})` at line ~13588), and before `return resp`, insert:

```python
    if email:
        try:
            import threading as _t
            _t.Thread(target=ash_ally.record_turn,
                      args=(LOG_DB, _db_lock, email, q, answer),
                      daemon=True).start()
        except Exception:
            pass
```

(`answer` is in scope here — the LLM-failure path returned early at the `except` above, so reaching this point guarantees `answer` exists. `q` is the user query.)

- [ ] **Step 3: Verify no syntax break**

Run: `python3 -c "import ast; ast.parse(open('app.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Verify the touches are wired**

Run: `grep -c "ash_ally.ally_overlay" app.py` and `grep -c "ash_ally.record_turn" app.py`
Expected: 8 and 8 (4 SSE from SP2b-1 + 2 scoped_reply from SP2b-2 + 1 practitioner from SP2b-3 + 1 here).

Run: `python3 - <<'PY'
import re
src = open("app.py").read()
# isolate the member_scan_analysis_chat function body
m = re.search(r"def member_scan_analysis_chat\(\):.*?(?=\n@app\.route|\ndef [a-z])", src, re.S)
body = m.group(0)
assert "ash_ally.ally_overlay(LOG_DB, email)" in body, "overlay touch missing"
assert "ash_ally.record_turn" in body and "(LOG_DB, _db_lock, email, q, answer)" in body, "record touch missing"
print("both touches present in member_scan_analysis_chat")
PY`
Expected: `both touches present in member_scan_analysis_chat`.

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat(scan-analysis): wire ASH ally into /member/scan-analysis/chat"
```

---

## Self-Review

**Spec coverage:**
- Overlay prepended to the assembled `system`, only when non-empty → Step 1 ✓
- Record on a bg daemon thread, only when `email` present, keyed on the cookie email → Step 2 ✓
- Subject = `rm_member_email` cookie (already resolved) → Global Constraints ✓
- Edit only this handler; purely additive; no-cookie = unchanged → Global Constraints + Steps ✓
- Counts become 8/8; both touches confirmed scoped to this handler → Step 4 ✓
- Same flag dark, fail-open helper, go-live render-verify → spec Verification ✓
- Out of scope (other handlers, Glendalf, whole-feature go-live) → not touched ✓

**Placeholder scan:** none — every step carries full content.

**Type consistency:** `ash_ally.ally_overlay(LOG_DB, email)` and `ash_ally.record_turn(LOG_DB, _db_lock, email, q, answer)` match the signatures used across SP2b-1/2/3.

**Note:** single task because there is no testable new logic and the change is one additive wiring on one handler; the per-task review is the gate (a separate whole-branch review is disproportionate for a single-task wiring branch).
