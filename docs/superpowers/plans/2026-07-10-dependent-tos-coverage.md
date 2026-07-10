# Dependent TOS coverage — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A dependent's own portal (pet, infant, minor) stops demanding a Terms agreement it can never give; the caregiver's agreement covers it. Behind a flag, dark until the covering copy is counsel-approved.

**Architecture:** `tos_agreed` in the portal payload is `is_member(email=primary_email)` — the token holder's own agreement. For a dependent's own token that is permanently false. We add a helper that also honours a linked, consented caregiver's agreement, plus a flag-gated copy clause so the accepted Terms actually cover dependents. Both the code and the copy are gated by one flag and are byte-identical to today when it is off.

**Tech Stack:** Python 3, Flask, sqlite3, vanilla JS, pytest, headless Chrome.

## Global Constraints

- Flag `DEPENDENT_TOS_ENABLED`, default OFF. Flag off → `tos_agreed` == `is_member(primary_email)` exactly, and the gate copy is unchanged. Byte-identical to today.
- The flag stays OFF at merge. Flipping it is a **Glen + counsel** decision, gated on the legal wording. This plan ships it dark.
- `primary_email` is the TOKEN HOLDER's email (captured at `app.py:15600`, before any `?member=` re-point). Never use `email_for_reports` for TOS — that is the #750 lesson and must not regress.
- The fix must NOT stamp `tos_agreed_at` on the dependent's own `journey_state`. Compliance (`_tos_agreed_emails`, `app.py:22399`) and the `/begin` funnel gates (`app.py:3626/3682`) must continue to exclude a dependent who did not personally agree.
- `share_consent` is required for coverage: a dependent who revoked it must agree themselves.
- Best-effort: any exception in the helper falls back to `is_member(primary_email)` and never breaks the portal load.

## Facts (verified 2026-07-10)

- `app.py:15767`: `"tos_agreed": is_member(email=primary_email) if primary_email else True,`
- `app.py:15600`: `primary_email = email_for_reports` (before the `?member=` re-point at ~15617).
- `is_member(session_id="", email="")` → `bool(journey_state.tos_agreed_at)`.
- `dashboard/household.caregivers_for(cx, member_email)` → `[{"primary_email": str, "share_consent": int}]`. Returns `[]` for an email with no caregiver linked above it (every standalone adult and every caregiver).
- The TOS gate copy is JS string concatenation in `static/client-portal.html:611-616`, rendered when `tos_agreed` is false.
- Live: `GET /api/portal/<sasha_token>` returns `tos_agreed: false`. Sasha is a cat.

## File Structure

| file | responsibility |
|---|---|
| `app.py` (modify) | `_dependent_tos_enabled()`; `_portal_tos_agreed(primary_email)`; use it at the `tos_agreed` computation; add `payload["tos_covers_dependents"]` for the frontend copy |
| `static/client-portal.html` (modify) | flag-gated clause in the TOS gate copy |
| `tests/test_dependent_tos.py` (create) | helper + endpoint tests |

---

### Task 1: the coverage helper and its wiring

**Files:**
- Modify: `app.py`
- Test: `tests/test_dependent_tos.py`

**Interfaces:**
- Produces:
  - `_dependent_tos_enabled() -> bool`
  - `_portal_tos_agreed(primary_email: str) -> bool` — `is_member(primary_email)`, OR (flag on) a linked consented caregiver has agreed. Falls back to `is_member(primary_email)` on any error.
  - `payload["tos_agreed"]` now uses it; `payload["tos_covers_dependents"]` carries the flag.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dependent_tos.py
"""A dependent's own portal must not gate on a Terms agreement it can never give.

A pet, an infant, a minor: each has its own portal token, handed to the caregiver. Opening
it asks "has this dependent agreed?" — permanently false. The caregiver's agreement covers
them. Derived at render time from the household link, so it never goes stale and a standalone
adult is byte-identical. NOT stamped on the dependent's own record: they did not agree.
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import household as hh

CARE = "caregiver@example.com"
PET = "pet@example.com"
ADULT = "standalone@example.com"


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


@pytest.fixture()
def app_db(tmp_db, monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    with sqlite3.connect(tmp_db) as cx:
        hh.init_household_tables(cx)
        hh.add_member(cx, CARE, PET, "Pet", "pet")   # a dependent under the caregiver
    return app


def _agree(app, email):
    """Give this email a real journey_state TOS acceptance."""
    import begin_funnel
    with sqlite3.connect(app.LOG_DB) as cx:
        begin_funnel.record_unlock(cx, session_id="s-" + email, trigger="tos",
                                   email=email, tos=True, tos_version=app.BEGIN_TOS_VERSION)


def test_flag_off_a_dependent_is_not_covered_by_the_caregiver(app_db, monkeypatch):
    monkeypatch.delenv("DEPENDENT_TOS_ENABLED", raising=False)
    _agree(app_db, CARE)
    assert app_db._portal_tos_agreed(PET) is False          # unchanged: the pet never agreed


def test_flag_on_a_dependent_is_covered_when_the_caregiver_agreed(app_db, monkeypatch):
    monkeypatch.setenv("DEPENDENT_TOS_ENABLED", "1")
    _agree(app_db, CARE)
    assert app_db._portal_tos_agreed(PET) is True


def test_flag_on_but_the_caregiver_has_not_agreed(app_db, monkeypatch):
    monkeypatch.setenv("DEPENDENT_TOS_ENABLED", "1")
    assert app_db._portal_tos_agreed(PET) is False


def test_a_dependent_who_revoked_consent_must_agree_themselves(app_db, monkeypatch):
    monkeypatch.setenv("DEPENDENT_TOS_ENABLED", "1")
    _agree(app_db, CARE)
    with sqlite3.connect(app_db.LOG_DB) as cx:
        hh.set_share_consent(cx, CARE, PET, 0)
    assert app_db._portal_tos_agreed(PET) is False


def test_the_dependent_agreeing_directly_still_works(app_db, monkeypatch):
    monkeypatch.setenv("DEPENDENT_TOS_ENABLED", "1")
    _agree(app_db, PET)
    assert app_db._portal_tos_agreed(PET) is True


def test_a_standalone_adult_is_unchanged(app_db, monkeypatch):
    monkeypatch.setenv("DEPENDENT_TOS_ENABLED", "1")
    assert app_db._portal_tos_agreed(ADULT) is False
    _agree(app_db, ADULT)
    assert app_db._portal_tos_agreed(ADULT) is True


def test_coverage_does_not_stamp_the_dependents_own_record(app_db, monkeypatch):
    """The dependent must NOT appear in the compliance set — they did not agree."""
    monkeypatch.setenv("DEPENDENT_TOS_ENABLED", "1")
    _agree(app_db, CARE)
    assert app_db._portal_tos_agreed(PET) is True
    with sqlite3.connect(app_db.LOG_DB) as cx:
        agreed = app_db._tos_agreed_emails(cx)
    assert CARE in agreed
    assert PET not in agreed


def test_a_blown_lookup_falls_back_to_the_dependents_own_agreement(app_db, monkeypatch):
    monkeypatch.setenv("DEPENDENT_TOS_ENABLED", "1")
    monkeypatch.setattr(hh, "caregivers_for",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db gone")))
    _agree(app_db, PET)
    assert app_db._portal_tos_agreed(PET) is True            # its own agreement still counts
    # and a pet that never agreed, with the lookup broken, stays gated (fail-closed)
    assert app_db._portal_tos_agreed("other-pet@example.com") is False
```

- [ ] **Step 2: Run and watch them fail**

Run: `cd ~/deploy-chat && doppler run -p remedy-match -c prd -- env DATA_DIR=$HOME/deploy-chat ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_dependent_tos.py -q -p no:cacheprovider`
Expected: `AttributeError: module 'app' has no attribute '_portal_tos_agreed'`

- [ ] **Step 3: Add the helper** (`app.py`, beside `_scan_recommendations_enabled` near line 14896)

```python
def _dependent_tos_enabled():
    """Flag: a caregiver's Terms acceptance covers the dependents in their care. Default
    OFF. When off, tos_agreed is is_member(primary_email) exactly — byte-identical to today.
    Flip only AFTER the gate copy says the caregiver agrees on behalf of those in their care
    (a counsel-approved wording)."""
    return (os.environ.get("DEPENDENT_TOS_ENABLED", "") or "").strip().lower() in (
        "1", "true", "yes", "on")


def _portal_tos_agreed(primary_email):
    """Has the Terms gate been satisfied for this token holder?

    The token holder's own agreement, OR (flag on) a linked, consented caregiver has agreed.
    A dependent — pet, infant, minor — has its own token but can never click 'I agree'; the
    caregiver agrees for it. Derived live from the household graph: caregivers_for() returns
    [] for any standalone adult, so this is byte-identical for non-dependents. It does NOT
    stamp the dependent's own tos_agreed_at, so compliance and funnel readers correctly still
    exclude a dependent who did not personally agree. Fail-closed to the token holder's own
    agreement on any error."""
    if is_member(email=primary_email):
        return True
    if not _dependent_tos_enabled() or not primary_email:
        return False
    try:
        from dashboard import household as _hh
        with sqlite3.connect(LOG_DB) as cx:
            caregivers = _hh.caregivers_for(cx, primary_email)   # a read; fetch then release
        for cg in caregivers:
            # is_member opens its own _db_lock connection, so do not nest it inside cx.
            if cg["share_consent"] and is_member(email=cg["primary_email"]):
                return True
    except Exception as _e:
        print(f"[dependent-tos] {_e!r}", flush=True)
    return False
```

- [ ] **Step 4: Wire it into the payload** (`app.py:15767`)

Replace:

```python
        "tos_agreed": is_member(email=primary_email) if primary_email else True,
```

with:

```python
        # Terms belong to the TOKEN HOLDER, not to whoever's tab is open (#750). AND a
        # dependent's own token is covered by its caregiver's agreement when DEPENDENT_TOS
        # is on — a pet/infant can't click "I agree". See _portal_tos_agreed.
        "tos_agreed": _portal_tos_agreed(primary_email) if primary_email else True,
        "tos_covers_dependents": _dependent_tos_enabled(),
```

- [ ] **Step 5: Green**

Run the same pytest command. Expected: `8 passed`.

- [ ] **Step 6: Commit**

```bash
cd ~/deploy-chat
git add app.py tests/test_dependent_tos.py
git commit -m "feat(portal): a caregiver's Terms acceptance covers dependents (flag-gated)"
```

---

### Task 2: the flag-gated gate copy

**Files:** Modify `static/client-portal.html`

Markup + inline JS. Render-verified, not unit-tested. The clause appears ONLY when
`d.tos_covers_dependents` is true, so with the flag off the gate reads exactly as today.

- [ ] **Step 1: Add the flag-gated clause**

In `static/client-portal.html`, the TOS gate block at ~611. Replace:

```javascript
      + '<p>Before we continue, please review and agree to our Terms of Service. '
      + 'These cover both Remedy Match and the Energy For Life (E4L) biofield scan.</p>'
```

with:

```javascript
      + '<p>Before we continue, please review and agree to our Terms of Service. '
      + 'These cover both Remedy Match and the Energy For Life (E4L) biofield scan.</p>'
      + (d.tos_covers_dependents
          ? '<p>By agreeing, you accept these Terms on your own behalf and on behalf of those '
            + 'in your care whose accounts you manage — including your minor children, '
            + 'dependents, and animals.</p>'
          : '')
```

> **Counsel note (do not skip):** this wording is a placeholder for Glen and counsel to
> finalise. The flag `DEPENDENT_TOS_ENABLED` must not be flipped until the wording is
> approved AND any COPPA obligation for under-13s is resolved. Until then the clause is dark.

Confirm `d` is the payload object in scope at this point (it is the same object whose
`tos_agreed` gates this block).

- [ ] **Step 2: Syntax-check the page's inline JS**

```bash
cd ~/deploy-chat
python3 - <<'PY' > /tmp/portal.js
import re
src = open("static/client-portal.html").read()
print('\n;\n'.join(re.findall(r'<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>', src, re.S)))
PY
node --check /tmp/portal.js && echo "JS PARSES OK"
```

`node --check` must pass — a syntax error here breaks the portal for every client.

- [ ] **Step 3: Commit**

```bash
cd ~/deploy-chat
git add static/client-portal.html
git commit -m "feat(portal): TOS gate says the caregiver agrees on behalf of dependents (flag-gated)"
```

---

### Task 3: regression, render-verify, PR (controller)

- [ ] **Step 1: Full-suite regression vs origin/main**

Capture baseline and branch failure sets, ANSI-stripped, `test_journey_assets.py` ignored,
`test_portal_concierge_eval.py::test_grounding_and_style_pass_rate` deselected on BOTH sides
(it is a nondeterministic LLM eval). `comm -23 branch base` must be empty.

- [ ] **Step 2: Render-verify BOTH flag states against a real dependent**

With the branch running locally and a proxy to prod's API:
- **Flag off** (default): the TOS gate reads exactly as today — no dependent clause; a
  dependent's own portal still shows the gate. Byte-identical.
- **Flag on** (inject `tos_covers_dependents: true` and `tos_agreed` per the helper): a pet's
  own portal whose caregiver has agreed renders the card instead of the Terms wall; the gate,
  when shown, carries the "on behalf of those in your care" clause.

Strip `<script>`/`<style>` before grepping the DOM (the `--dump-dom` trap: grepping raw output
matches the inline template, not the rendered page).

- [ ] **Step 3: Push and open the PR**

The PR body must state: flag default OFF, ships dark; the copy wording and the COPPA question
are Glen + counsel decisions that gate the flip; `_tos_agreed_emails` and the `/begin` gates
are deliberately unchanged so a dependent who did not personally agree stays out of compliance
counts.

---

## Non-goals

- Flipping the flag. That is Glen + counsel, gated on the legal wording and the COPPA question.
- Species sync and the animal greeting ("Give our Aloha to Sasha") — Slice 4. Note that once
  this ships and the flag flips, an animal's own portal renders and currently greets "Aloha
  Sasha"; Slice 4 fixes that.
- Auto-detecting dependents by age or species. The household `relationship` link governs.
- Any change to the `/begin` funnel TOS gates or the compliance collector.
