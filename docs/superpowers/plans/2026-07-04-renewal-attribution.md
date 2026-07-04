# Renewal Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Continuous Care attribution sticky — a renewed term (bought any way other than through the doctor's dispensary link) inherits the patient's most-recent attributed doctor + consent, so the doctor keeps the fee-share and continuity-tooling visibility across renewals.

**Architecture:** One `_last_attributed_practitioner(email)` lookup (most-recent attributed record across `prepay_term_grants` + `subscriptions`, sticky/most-recent-wins). Both fulfillment paths change their attribution resolution from "read `dispensary_pid` from metadata" to "explicit `dispensary_pid` OR inherited (pid + consent)", then feed the existing stamp/credit logic unchanged.

**Tech Stack:** Python 3, Flask, SQLite (`prepay_term_grants`, `subscriptions` in LOG_DB), pytest.

## Global Constraints

- Resolution: `explicit dispensary_pid + explicit share_consent` (session carries `dispensary_pid`) → else `_last_attributed_practitioner(email)` (inherited pid + inherited consent) → else unattributed (None, 0). Explicit ALWAYS wins.
- Sticky / most-recent-wins forever (no window). Inheritance only at a NEW term's fulfillment — never the sub-charge cron.
- No schema changes; reuse #565's `subscriptions.attributed_practitioner_id`/`practitioner_share_consent` + #575's `prepay_term_grants` attribution columns + `care_share`/`wallet`.
- The lookup only ever returns a record with a NON-NULL `attributed_practitioner_id`; it must not return the *current* session's just-created (not-yet-stamped) grant. It is table-guarded (`sqlite3.OperationalError` → skip that source) and returns None for a patient with no attributed history.
- Tests import `app` → run via `doppler run -p remedy-match -c dev -- python -m pytest ...`.

---

## File Structure

- **Modify** `app.py` — new `_last_attributed_practitioner(email, *, db_path=None)` helper; `_fulfill_prepay_term` (L~7490 resolution) + `_fulfill_continuous_care_monthly` (L~7590 resolution) inherit when no explicit `dispensary_pid`.
- **Test** `tests/test_renewal_attribution.py`.

---

### Task 1: `_last_attributed_practitioner` lookup

**Files:** Modify `app.py`; Test `tests/test_renewal_attribution.py`

**Interfaces:**
- Produces: `_last_attributed_practitioner(email, *, db_path=None) -> {"pid": str, "consent": int} | None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_renewal_attribution.py
import sqlite3, importlib, sys
from pathlib import Path

def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    return importlib.import_module("app")

def _seed(path):
    cx = sqlite3.connect(path)
    cx.execute("CREATE TABLE prepay_term_grants (session_id TEXT PRIMARY KEY, email TEXT, tier_key TEXT, granted_at TEXT, attributed_practitioner_id TEXT, practitioner_share_consent INTEGER DEFAULT 0, term_end TEXT)")
    cx.execute("CREATE TABLE subscriptions (email TEXT, attributed_practitioner_id TEXT, practitioner_share_consent INTEGER, kind TEXT, created_at TEXT)")
    cx.commit(); cx.close()

def test_none_when_no_history(tmp_path):
    app = _app(); p = str(tmp_path/"log.db"); _seed(p)
    assert app._last_attributed_practitioner("pat@x.com", db_path=p) is None

def test_prepay_grant_returned(tmp_path):
    app = _app(); p = str(tmp_path/"log.db"); _seed(p)
    cx = sqlite3.connect(p); cx.execute("INSERT INTO prepay_term_grants (session_id,email,granted_at,attributed_practitioner_id,practitioner_share_consent) VALUES ('s1','pat@x.com','2026-01-01T00:00:00Z','prac-42',1)"); cx.commit(); cx.close()
    assert app._last_attributed_practitioner("PAT@x.com", db_path=p) == {"pid": "prac-42", "consent": 1}

def test_most_recent_across_sources(tmp_path):
    app = _app(); p = str(tmp_path/"log.db"); _seed(p)
    cx = sqlite3.connect(p)
    cx.execute("INSERT INTO prepay_term_grants (session_id,email,granted_at,attributed_practitioner_id,practitioner_share_consent) VALUES ('s1','pat@x.com','2026-01-01T00:00:00Z','prac-A',1)")
    cx.execute("INSERT INTO subscriptions (email,attributed_practitioner_id,practitioner_share_consent,kind,created_at) VALUES ('pat@x.com','prac-B',0,'membership','2026-06-01T00:00:00Z')")
    cx.commit(); cx.close()
    assert app._last_attributed_practitioner("pat@x.com", db_path=p) == {"pid": "prac-B", "consent": 0}  # later wins

def test_ignores_unattributed_rows(tmp_path):
    app = _app(); p = str(tmp_path/"log.db"); _seed(p)
    cx = sqlite3.connect(p); cx.execute("INSERT INTO prepay_term_grants (session_id,email,granted_at,attributed_practitioner_id) VALUES ('s1','pat@x.com','2026-01-01T00:00:00Z',NULL)"); cx.commit(); cx.close()
    assert app._last_attributed_practitioner("pat@x.com", db_path=p) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python -m pytest tests/test_renewal_attribution.py -q`
Expected: FAIL (`AttributeError: _last_attributed_practitioner`)

- [ ] **Step 3: Write minimal implementation**

Add near the fulfillment helpers in `app.py`:

```python
def _last_attributed_practitioner(email, *, db_path=None):
    """The patient's most-recent attributed doctor across prepay grants + memberships.
    Returns {'pid': str, 'consent': int} or None. Sticky: most-recent-wins (no window).
    Only considers rows with a non-null attributed_practitioner_id; table-guarded."""
    e = (email or "").strip().lower()
    if not e:
        return None
    cands = []  # (timestamp, pid, consent)
    with sqlite3.connect(db_path or LOG_DB) as cx:
        try:
            r = cx.execute(
                "SELECT granted_at, attributed_practitioner_id, practitioner_share_consent "
                "FROM prepay_term_grants WHERE lower(email)=? AND attributed_practitioner_id IS NOT NULL "
                "ORDER BY granted_at DESC LIMIT 1", (e,)).fetchone()
            if r:
                cands.append((r[0] or "", str(r[1]), int(r[2] or 0)))
        except sqlite3.OperationalError:
            pass
        try:
            r = cx.execute(
                "SELECT created_at, attributed_practitioner_id, practitioner_share_consent "
                "FROM subscriptions WHERE lower(email)=? AND attributed_practitioner_id IS NOT NULL "
                "AND kind='membership' ORDER BY created_at DESC LIMIT 1", (e,)).fetchone()
            if r:
                cands.append((r[0] or "", str(r[1]), int(r[2] or 0)))
        except sqlite3.OperationalError:
            pass
    if not cands:
        return None
    best = max(cands, key=lambda c: c[0])   # ISO timestamps compare lexically by recency
    return {"pid": best[1], "consent": best[2]}
```

- [ ] **Step 4: Run to verify it passes** — GREEN.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_renewal_attribution.py
git commit -m "feat(renewal): _last_attributed_practitioner most-recent-across-sources lookup"
```

---

### Task 2: Prepay path inherits attribution

**Files:** Modify `app.py` (`_fulfill_prepay_term`, ~L7490); Test `tests/test_renewal_attribution.py`

**Interfaces:**
- Consumes: `_last_attributed_practitioner` (Task 1). The existing stamp + `earn_care_share` logic (unchanged) runs on the resolved `disp_pid`/`share_consent`.

- [ ] **Step 1: Write the failing test** — read the existing `_fulfill_prepay_term` tests (`tests/test_care_share_prepay.py`) for the Stripe-stub pattern. Add:
  - `test_prepay_renewal_inherits_prior_attribution`: seed the patient's prior attributed prepay grant (prac-42, consent 1); fulfil a NEW `prepay_term` session with metadata that has **no** `dispensary_pid` → assert the new grant row is stamped `attributed_practitioner_id='prac-42'`, `practitioner_share_consent=1`, and `earn_care_share` was called for `prac-42` on the lump.
  - `test_prepay_explicit_dispensary_pid_wins`: prior attribution is prac-42, but the new session carries `dispensary_pid='prac-99'` → the new grant is attributed to `prac-99` (explicit wins), not inherited.
  - `test_prepay_no_prior_no_explicit_unattributed`: no prior + no `dispensary_pid` → grant created, NOT stamped, `earn_care_share` NOT called (public behavior).

- [ ] **Step 2: Run to verify it fails** — inheritance not wired.

- [ ] **Step 3: Implement** — in `_fulfill_prepay_term`, replace the resolution at ~L7490-7491:

```python
                disp_pid = (md.get("dispensary_pid") or "").strip() or None
                share_consent = 1 if (md.get("share_consent") or "").strip() == "1" else 0
                if not disp_pid:
                    _inh = _last_attributed_practitioner(email)   # sticky renewal attribution
                    if _inh:
                        disp_pid = _inh["pid"]
                        share_consent = int(_inh["consent"])
```

(Everything below — the `if disp_pid:` stamp + care-share credit — is unchanged and now runs on the inherited pid/consent.)

- [ ] **Step 4: Run to verify it passes** — GREEN (+ the existing `tests/test_care_share_prepay.py` still passes: an attributed enrollment WITH `dispensary_pid` is unaffected, and the public-no-dispensary test now needs the patient to also have no prior attribution — confirm the existing public test seeds a fresh patient with no prior grant, else adjust it).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_renewal_attribution.py
git commit -m "feat(renewal): prepay fulfilment inherits prior attribution when no explicit dispensary_pid"
```

---

### Task 3: Monthly path inherits attribution

**Files:** Modify `app.py` (`_fulfill_continuous_care_monthly`, ~L7590); Test `tests/test_renewal_attribution.py`

**Interfaces:**
- Consumes: `_last_attributed_practitioner` (Task 1). The existing `create_membership(..., attributed_practitioner_id=disp_pid, practitioner_share_consent=share_consent)` + enrollment-charge credit (unchanged) run on the resolved values.

- [ ] **Step 1: Write the failing test** — read the existing `_fulfill_continuous_care_monthly` tests (`tests/test_care_share_enroll.py`) for the stub pattern. Add:
  - `test_monthly_reenroll_inherits_prior_attribution`: prior attributed record for the patient (prac-42, consent 1); a NEW `continuous_care_monthly` session with **no** `dispensary_pid` → the created membership has `attributed_practitioner_id='prac-42'`, `practitioner_share_consent=1`, and the enrollment charge credits `prac-42`.
  - `test_monthly_explicit_dispensary_pid_wins`: prior prac-42, session carries `dispensary_pid='prac-99'` → membership attributed to `prac-99`.

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** — in `_fulfill_continuous_care_monthly`, right after the resolution at ~L7590-7594:

```python
        disp_pid = (md.get("dispensary_pid") or "").strip() or None
        share_consent = 1 if (md.get("share_consent") or "").strip() == "1" else 0
        if not disp_pid:
            _inh = _last_attributed_practitioner(email)   # sticky renewal attribution
            if _inh:
                disp_pid = _inh["pid"]
                share_consent = int(_inh["consent"])
```

(The downstream `create_membership(..., attributed_practitioner_id=disp_pid, practitioner_share_consent=share_consent)` at ~L7689 + care-share credit are unchanged and now run on the inherited values. Confirm `email` is in scope at this point — it is resolved earlier in the function.)

- [ ] **Step 4: Run to verify it passes** — GREEN (+ the existing `tests/test_care_share_enroll.py` still passes: WITH-dispensary_pid unaffected; the direct-enrollment test must use a patient with no prior attribution — confirm/adjust).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_renewal_attribution.py
git commit -m "feat(renewal): monthly re-enrolment inherits prior attribution when no explicit dispensary_pid"
```

---

## Self-Review

**Spec coverage:**
- Sticky most-recent lookup across both sources → Task 1. ✓
- Explicit-wins / else-inherit / else-unattributed resolution → Tasks 2 (prepay) + 3 (monthly). ✓
- Consent inherited → Tasks 2/3 (share_consent set from the inherited record). ✓
- Not the cron → inheritance is only in the two new-term fulfillment functions; the cron is untouched. ✓
- No schema changes → confirmed (helper reads existing columns). ✓

**Placeholder scan:** Task 1 has complete helper + tests. Tasks 2/3 have the exact resolution insertion and instruct the implementer to read the existing per-path test stubs (`test_care_share_prepay.py` / `test_care_share_enroll.py`) to reuse their Stripe stubbing (rather than re-inventing it) and to confirm the existing public/direct tests seed patients with no prior attribution (else the inherited attribution would change their outcome — an intended behavior change to check).

**Type consistency:** `_last_attributed_practitioner(email, *, db_path=None) -> {"pid","consent"}|None` used identically in Tasks 2/3. The resolution swap is byte-identical in both paths.

## Notes / open confirmations
- **Existing public/direct tests:** the change means "a fulfilment with no `dispensary_pid` for a patient WITH prior attribution now inherits." Confirm the existing public-prepay / direct-monthly tests use patients with NO prior attributed record (so they stay unattributed as they assert). If one seeds a patient who also has a prior attributed grant, its expectation legitimately changes — adjust it to the new sticky behavior, not around it.
- **Lookup excludes the current session's grant:** in `_fulfill_prepay_term` the lookup runs BEFORE the current grant is stamped (it's NULL-attributed at that point), so it can't return itself. Confirm the call sits before the `if disp_pid:` stamp.
