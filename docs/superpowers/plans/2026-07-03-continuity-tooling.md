# Doctor Continuity Tooling (C) v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give a doctor a per-patient continuity view of their consented continuity patients (scan trajectory + what-changed + a suggested next step), and let them push a reviewed recommendation to the patient's portal as a one-tap member-priced reorder.

**Architecture:** A new `dashboard/continuity_view.py` holds the authorization gate + roster + view-assembly + recommend, all behind one gate. A `practitioner_recommendations` table backs the doctor→patient recommendation; a `subscriptions.practitioner_share_consent` flag gates access. The view assembles existing engines (`scan_analysis`, `biofield_narrative`/`biofield_profile`, the biofield recommended-remedy path); the patient portal renders the recommendation and reuses the existing member-priced checkout.

**Tech Stack:** Python 3, Flask, SQLite (`subscriptions`, `scan_analyses`, new `practitioner_recommendations`), Supabase (`practitioners`), pytest, vanilla JS portals.

## Global Constraints

- **BUILD ORDER: build after #565 (turnkey fee-share) is merged.** C queries `subscriptions.attributed_practitioner_id`, extends #565's dispensary Continuous Care enrollment (the consent checkbox), and reuses that enrollment card. Base this branch on a tree that contains #565.
- **Authorization gate (the keystone):** every per-patient read and every recommend-write MUST first pass `continuity_view.authorized_patient(practitioner_id, patient_email)` — True only when a subscription exists with `attributed_practitioner_id == practitioner_id AND practitioner_share_consent == 1 AND kind == "membership"` matching that patient's email. Any route returns 403 on False. No data leaves the gate for an unauthorized pair.
- Consent model A: `practitioner_share_consent` set at dispensary Continuous Care enrollment when the patient checks the authorization box.
- Recommendations never auto-send; the doctor reviews/edits first. The recommended remedy is a **product buy at member pricing** (add-to-cart, not a silent charge) — the $99/mo is the separate service.
- Integer cents. Follow existing module patterns (`cx`-first SQLite like `dashboard/scan_analysis.py`; `_practitioner_session_pid()` guards practitioner routes).

---

## File Structure

- **Create** `dashboard/continuity_view.py` — `authorized_patient`, `roster`, `patient_view`, `send_recommendation`, `latest_biofield_test_id`.
- **Create** `dashboard/practitioner_recommendations.py` — table + `create`/`active_for_patient`/`set_status`.
- **Modify** `dashboard/subscriptions.py` — `practitioner_share_consent` column (guarded ALTER) + `create_membership` kwarg.
- **Modify** `app.py` — the #565 dispensary CC enrollment (write the consent flag); new practitioner-portal routes (roster / patient view / recommend); patient-portal recommendation render + accept-to-cart.
- **Modify** `static/practitioner-client.html` (#565 card) — the consent checkbox; `static/practitioner-portal.html` — Clients-tab roster + per-patient view + recommend UI; the patient portal template — the recommendation card + one-tap reorder.
- **Tests** alongside each.

---

### Task 1: The authorization gate — `continuity_view.authorized_patient`

**Files:** Create `dashboard/continuity_view.py`; Test `tests/test_continuity_authz.py`

**Interfaces:**
- Produces: `authorized_patient(cx, practitioner_id, patient_email) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_continuity_authz.py
import sqlite3
from dashboard import continuity_view as cv, subscriptions as subs


def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    subs.init_tables(cx)   # use the real init entrypoint (confirm name in subscriptions.py)
    return cx


def _mk(cx, email, pid, consent):
    sid = subs.create_membership(cx, email=email, stripe_customer_id="c", stripe_payment_method_id="p",
                                 amount_cents=9900, next_charge_date="2026-08-01",
                                 attributed_practitioner_id=pid)
    if consent:
        cx.execute("UPDATE subscriptions SET practitioner_share_consent=1 WHERE id=?", (sid,)); cx.commit()
    return sid


def test_authorized_for_consented_continuity_patient():
    cx = _cx(); _mk(cx, "pat@x.com", "prac-42", consent=True)
    assert cv.authorized_patient(cx, "prac-42", "pat@x.com") is True


def test_denied_other_doctor():
    cx = _cx(); _mk(cx, "pat@x.com", "prac-42", consent=True)
    assert cv.authorized_patient(cx, "prac-99", "pat@x.com") is False


def test_denied_without_consent():
    cx = _cx(); _mk(cx, "pat@x.com", "prac-42", consent=False)
    assert cv.authorized_patient(cx, "prac-42", "pat@x.com") is False


def test_denied_unknown_patient():
    cx = _cx()
    assert cv.authorized_patient(cx, "prac-42", "nobody@x.com") is False
```

Note: confirm `subscriptions.init_tables` (or the real init name) and that `practitioner_share_consent` exists — Task 2 adds the column, so run this test AFTER Task 2's migration is present, OR add the column in this test's setup. Sequence Task 2 before Task 1 if simpler; the gate depends on the column existing.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_continuity_authz.py -q`
Expected: FAIL (`ModuleNotFoundError: dashboard.continuity_view`)

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/continuity_view.py
"""Doctor continuity tooling (C): per-patient continuity view + recommend loop.

Every per-patient read/write goes through authorized_patient() first — a doctor
may only ever touch a patient who has a CONSENTED CONTINUITY link to them.
"""


def authorized_patient(cx, practitioner_id, patient_email) -> bool:
    """True iff patient_email has an active-consented Continuous Care membership
    attributed to practitioner_id. The single access boundary for all of C."""
    if not practitioner_id or not patient_email:
        return False
    row = cx.execute(
        "SELECT 1 FROM subscriptions WHERE lower(email)=lower(?) "
        "AND attributed_practitioner_id=? AND practitioner_share_consent=1 "
        "AND kind='membership' LIMIT 1",
        ((patient_email or "").strip(), str(practitioner_id)),
    ).fetchone()
    return row is not None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_continuity_authz.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/continuity_view.py tests/test_continuity_authz.py
git commit -m "feat(continuity): authorization gate — consented continuity patient only"
```

---

### Task 2: Consent flag — `subscriptions.practitioner_share_consent`

**Files:** Modify `dashboard/subscriptions.py`; Test `tests/test_subscriptions_consent.py`

**Interfaces:**
- Produces: `create_membership(..., practitioner_share_consent=0)` persists the flag; column defaults 0.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_subscriptions_consent.py
import sqlite3
from dashboard import subscriptions as subs


def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    subs.init_tables(cx); return cx


def test_consent_defaults_zero():
    cx = _cx()
    sid = subs.create_membership(cx, email="p@x.com", stripe_customer_id="c",
        stripe_payment_method_id="p", amount_cents=9900, next_charge_date="2026-08-01")
    assert cx.execute("SELECT practitioner_share_consent FROM subscriptions WHERE id=?", (sid,)).fetchone()[0] == 0


def test_consent_persisted_when_set():
    cx = _cx()
    sid = subs.create_membership(cx, email="p@x.com", stripe_customer_id="c",
        stripe_payment_method_id="p", amount_cents=9900, next_charge_date="2026-08-01",
        practitioner_share_consent=1)
    assert cx.execute("SELECT practitioner_share_consent FROM subscriptions WHERE id=?", (sid,)).fetchone()[0] == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_subscriptions_consent.py -q`
Expected: FAIL (`TypeError: unexpected keyword 'practitioner_share_consent'`)

- [ ] **Step 3: Add the migration + kwarg**

In the subscriptions guarded ALTER block (same mechanism as `attributed_practitioner_id`, `dashboard/subscriptions.py:~305`), add:

```python
        "ALTER TABLE subscriptions ADD COLUMN practitioner_share_consent INTEGER NOT NULL DEFAULT 0",
```

and thread it through `create_membership` — add `practitioner_share_consent=0` to the signature, add the column to the INSERT column list + a placeholder, bind `int(bool(practitioner_share_consent))`. Thread the new migration call to the same membership-migration call sites the `attributed_practitioner_id` migration touched (per that precedent, else the INSERT hits "no such column").

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_subscriptions_consent.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/subscriptions.py tests/test_subscriptions_consent.py
git commit -m "feat(continuity): subscriptions.practitioner_share_consent flag"
```

---

### Task 3: Consent capture at enrollment

**Files:** Modify `static/practitioner-client.html` (#565 card) + `app.py` (the #565 dispensary CC enrollment → `_fulfill_continuous_care_monthly` / `create_membership` call); Test extend `tests/test_care_share_enroll.py` (from #565)

**Interfaces:**
- Consumes: Task 2's `create_membership(..., practitioner_share_consent=...)`; #565's dispensary enrollment endpoint + `dispensary_pid` metadata.
- Produces: enrollment through the dispensary with the consent box checked sets `practitioner_share_consent=1` on the membership.

- [ ] **Step 1: Write the failing test** — extend the #565 enrollment test: POST the dispensary Continuous Care enrollment with the consent field set, drive fulfilment, assert the created `subscriptions` row has `practitioner_share_consent=1` (and 0 when the box is unchecked).

```python
def test_enrollment_captures_practitioner_consent(...):
    # (mirror test_care_share_enroll.py setup) POST enrollment with share_consent=True
    # → fulfil → assert the membership row's practitioner_share_consent == 1
    ...
```

- [ ] **Step 2: Run to verify it fails** — `python -m pytest tests/test_care_share_enroll.py -q` (new case FAILs: flag stays 0).

- [ ] **Step 3: Implement** — add the authorization checkbox to the "Start Continuous Care" card (`static/practitioner-client.html`, near the existing agreement) with copy "I authorize sharing my wellness results with my enrolling practitioner."; carry its value in the enrollment POST body and into the Stripe session metadata (alongside `dispensary_pid`); in `_fulfill_continuous_care_monthly`, read it back and pass `practitioner_share_consent=1` into `create_membership` when set.

- [ ] **Step 4: Run to verify it passes** — GREEN.

- [ ] **Step 5: Commit** — `git commit -m "feat(continuity): capture practitioner-share consent at Continuous Care enrollment"`

---

### Task 4: `practitioner_recommendations` table

**Files:** Create `dashboard/practitioner_recommendations.py`; Test `tests/test_practitioner_recommendations.py`

**Interfaces:**
- Produces: `init_table(cx)`; `create(cx, *, practitioner_id, patient_email, items, note) -> int`; `active_for_patient(cx, patient_email) -> dict|None` (latest not-dismissed); `set_status(cx, rec_id, status)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_practitioner_recommendations.py
import sqlite3
from dashboard import practitioner_recommendations as pr


def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    pr.init_table(cx); return cx


def test_create_and_active():
    cx = _cx()
    rid = pr.create(cx, practitioner_id="prac-42", patient_email="P@x.com",
                    items=[{"slug": "nerve-repair", "qty": 1}], note="stay the course")
    a = pr.active_for_patient(cx, "p@x.com")   # case-insensitive
    assert a["id"] == rid and a["items"][0]["slug"] == "nerve-repair" and a["status"] == "sent"


def test_dismissed_not_active():
    cx = _cx()
    rid = pr.create(cx, practitioner_id="prac-42", patient_email="p@x.com", items=[], note="")
    pr.set_status(cx, rid, "dismissed")
    assert pr.active_for_patient(cx, "p@x.com") is None
```

- [ ] **Step 2: Run to verify it fails** — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/practitioner_recommendations.py
import json, sqlite3
from datetime import datetime, timezone

def _now(): return datetime.now(timezone.utc).isoformat()

def init_table(cx):
    cx.execute("CREATE TABLE IF NOT EXISTS practitioner_recommendations ("
               "id INTEGER PRIMARY KEY AUTOINCREMENT, practitioner_id TEXT NOT NULL, "
               "patient_email TEXT NOT NULL, items_json TEXT NOT NULL DEFAULT '[]', "
               "note TEXT DEFAULT '', status TEXT NOT NULL DEFAULT 'sent', created_at TEXT NOT NULL)")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_pr_patient ON practitioner_recommendations (lower(patient_email), status)")
    cx.commit()

def create(cx, *, practitioner_id, patient_email, items, note):
    init_table(cx)
    cur = cx.execute("INSERT INTO practitioner_recommendations "
        "(practitioner_id, patient_email, items_json, note, status, created_at) VALUES (?,?,?,?,'sent',?)",
        (str(practitioner_id), (patient_email or "").strip().lower(), json.dumps(items or []), note or "", _now()))
    cx.commit(); return cur.lastrowid

def active_for_patient(cx, patient_email):
    init_table(cx)
    r = cx.execute("SELECT * FROM practitioner_recommendations WHERE lower(patient_email)=lower(?) "
        "AND status!='dismissed' ORDER BY id DESC LIMIT 1", ((patient_email or "").strip(),)).fetchone()
    if not r: return None
    d = dict(r); d["items"] = json.loads(d.pop("items_json") or "[]"); return d

def set_status(cx, rec_id, status):
    init_table(cx)
    cx.execute("UPDATE practitioner_recommendations SET status=? WHERE id=?", (status, int(rec_id))); cx.commit()
```

- [ ] **Step 4: Run to verify it passes** — PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(continuity): practitioner_recommendations table"`

---

### Task 5: `continuity_view.roster` — the doctor's consented continuity patients

**Files:** Modify `dashboard/continuity_view.py`; Test `tests/test_continuity_authz.py`

**Interfaces:**
- Produces: `roster(cx, practitioner_id) -> [{"email","name"}]` — one row per consented continuity patient of this practitioner.

- [ ] **Step 1: Write the failing test** — seed two consented continuity members for `prac-42`, one for `prac-99`, one unconsented for `prac-42`; assert `roster(cx,"prac-42")` returns exactly the two consented `prac-42` patients (by email), never the other doctor's or the unconsented one.

- [ ] **Step 2: Run to verify it fails** — `AttributeError: roster`.

- [ ] **Step 3: Implement** — `SELECT DISTINCT lower(email) FROM subscriptions WHERE attributed_practitioner_id=? AND practitioner_share_consent=1 AND kind='membership'`; resolve display name from the customers/order record if available (else the email local-part). Keep it a straight query — the gate's predicate reused.

- [ ] **Step 4: Run to verify it passes** — PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(continuity): roster of consented continuity patients"`

---

### Task 6: `continuity_view.patient_view` — assemble trajectory + narrative + suggested step (gate-first)

**Files:** Modify `dashboard/continuity_view.py`; Test `tests/test_continuity_view.py`

**Interfaces:**
- Consumes: `authorized_patient` (Task 1); `dashboard/scan_analysis.py` `get(cx, email)`; `dashboard/biofield_narrative.py`/`biofield_profile.py`; the biofield recommended-remedy path (`dashboard/biofield_portal_publish.py` `build_portal_content(cx, test_id, special_price_cents=...)` → `reorder_items`).
- Produces:
  - `latest_biofield_test_id(cx, patient_email) -> str|None` — the patient's most recent biofield test (READ the biofield tables to find how a test maps to a patient email; this is the one integration lookup to resolve at build time).
  - `patient_view(cx, practitioner_id, patient_email) -> dict|None` — returns `None`/raises `PermissionError` when `authorized_patient` is False; otherwise `{"trajectory": <scan_analysis.get analysis>, "narrative": <what-changed read>, "suggested_step": <reorder_items>}`.

- [ ] **Step 1: Write the failing test** — the CRITICAL gate test: `patient_view(cx, "prac-99", "pat@x.com")` for a patient NOT theirs → raises `PermissionError` (or returns None) and reads NO scan data (patch `scan_analysis.get` to a spy asserting it is never called on the unauthorized path). For an authorized patient (seed a `scan_analyses` row via `scan_analysis.upsert` + a biofield test), assert the returned dict carries the trajectory, a narrative, and a suggested_step. Inject the biofield/narrative deps for determinism.

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** — `patient_view` calls `authorized_patient` FIRST and returns None / raises before any data read; then assembles: `scan_analysis.get(cx, email)` for the trajectory; the narrative via the biofield narrative module for the latest-vs-prior read; the suggested step via `latest_biofield_test_id` → `build_portal_content(...)["reorder_items"]`. READ `dashboard/biofield_narrative.py` and `biofield_portal_publish.py` for the exact call shapes and thread real args (catalog, special_price_cents = member price). Degrade gracefully (empty trajectory/suggested_step) when a patient has no scans yet — never crash the view.

- [ ] **Step 4: Run to verify it passes.**

- [ ] **Step 5: Commit** — `git commit -m "feat(continuity): gate-first per-patient view (trajectory + narrative + suggested step)"`

---

### Task 7: `continuity_view.send_recommendation` + practitioner routes

**Files:** Modify `dashboard/continuity_view.py`, `app.py`; Test `tests/test_continuity_routes.py`

**Interfaces:**
- Consumes: `authorized_patient`, `roster`, `patient_view`; `practitioner_recommendations.create` (Task 4); `_practitioner_session_pid()`.
- Produces:
  - `send_recommendation(cx, practitioner_id, patient_email, items, note) -> int` — gate-first, then `practitioner_recommendations.create(...)`, then notify the patient (reuse `dashboard/biofield_comms.py`/`recent_comms.py`).
  - Routes: `GET /api/practitioner/continuity/roster`; `GET /api/practitioner/continuity/patient/<patient_email>`; `POST /api/practitioner/continuity/recommend` — all `_practitioner_session_pid()`-guarded, and each calls the gate (403 on unauthorized patient).

- [ ] **Step 1: Write the failing test** — route tests (mirror `tests/test_practitioner_pricing_routes.py` for app import + `_practitioner_session_pid` monkeypatch): roster returns only the signed-in doctor's consented patients; patient route 403s for a patient not theirs (no data in body); recommend 403s for an unauthorized patient and, when authorized, writes a `practitioner_recommendations` row + records the notification.

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** the three routes + `send_recommendation`. Every route: resolve `pid = _practitioner_session_pid()` (401 if none); for the patient/recommend routes, `if not cv.authorized_patient(cx, pid, patient_email): return 403` BEFORE any read/write. Notification best-effort (try/except) so a comms failure doesn't fail the recommend.

- [ ] **Step 4: Run to verify it passes.**

- [ ] **Step 5: Commit** — `git commit -m "feat(continuity): recommend action + gate-checked practitioner routes"`

---

### Task 8: Patient-portal recommendation landing + Clients-tab UI

**Files:** Modify the patient portal template + `app.py` (accept-to-cart), `static/practitioner-portal.html`; Test `tests/test_continuity_landing.py`

**Interfaces:**
- Consumes: `practitioner_recommendations.active_for_patient`; the existing patient member-priced checkout/add-to-cart (READ the client-portal reorder path — the recommended items become a member-priced cart, NOT `dashboard/reorder.py` which is inventory).
- Produces: the patient portal renders the active recommendation ("Your practitioner recommends X") with one-tap add-to-cart at member pricing + a dismiss; a `POST` accept route adds the items to the patient's cart at member pricing and marks the recommendation `accepted`. The Clients tab renders the roster → per-patient view → review/edit suggested step + note → "Recommend to patient".

- [ ] **Step 1: Write the failing test** — assert the patient portal data/route surfaces `active_for_patient` for the signed-in patient; accepting adds the items to a member-priced cart (assert the line price = the patient's member price, reusing the existing pricing path) and flips status to `accepted`; a patient with no active recommendation sees none.

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** — patient-portal render + accept-to-cart (reuse the existing member-priced checkout entry; do NOT hand-roll pricing — thread through `pricing.compute`/the client checkout as the portal reorder already does). Clients-tab UI: roster list, per-patient view rendering trajectory/narrative/suggested-step, an editable suggested-step + note, and a "Recommend to patient" button POSTing to `/api/practitioner/continuity/recommend`. Match existing portal styling; vanilla JS.

- [ ] **Step 4: Verify** — run the landing test GREEN; browser render-verify both surfaces (practitioner per-patient view + patient recommendation card) before deploy.

- [ ] **Step 5: Commit** — `git commit -m "feat(continuity): patient-portal recommendation landing + Clients-tab continuity UI"`

---

## Self-Review

**Spec coverage:**
- Authorization gate → Task 1 (+ enforced in 5/6/7/8). ✓
- Consent flag + capture → Tasks 2, 3. ✓
- Roster → Task 5. ✓
- Per-patient view (trajectory/narrative/suggested step) → Task 6. ✓
- Action (i+ii) + routes → Task 7. ✓
- Landing (X) member-priced reorder + UI → Task 8. ✓
- `practitioner_recommendations` table → Task 4. ✓

**Placeholder scan:** Tasks 3, 5, 6, 8 are reuse-heavy and instruct the implementer to READ named modules (`biofield_narrative`, `biofield_portal_publish`, the client-portal member-priced checkout) and resolve two named integration lookups (patient-email → latest biofield test_id; the real member-priced add-to-cart entry) at build time, rather than inventing shapes for engines whose output must be read from source. New, self-contained code (Tasks 1, 2, 4, and the gate/route scaffolding) has complete code. This is deliberate for a reuse-assembly feature — the alternative (guessing the biofield/portal payload shapes now) would be fiction.

**Type consistency:** `authorized_patient(cx, pid, email)` is the first call in Tasks 1/5/6/7/8. `patient_view`/`roster`/`send_recommendation` signatures consistent across Tasks 5–8. `practitioner_recommendations.create/active_for_patient/set_status` consistent across Tasks 4/7/8. Consent column `practitioner_share_consent` consistent across Tasks 1/2/3/5.

## Notes / open confirmations

- **Build after #565 merges** (Global Constraints) — base the branch on a #565-containing tree.
- **Two integration lookups to resolve by reading at build time:** (1) patient-email → their latest biofield `test_id` (Task 6 `latest_biofield_test_id`); (2) the exact patient-facing member-priced add-to-cart entry (Task 8) — it is the client-portal checkout, NOT `dashboard/reorder.py` (inventory). Both are reads of existing code, not new design.
- **The authorization gate is the review focus** — it is the privacy keystone; the final whole-branch review should trace that no per-patient route can leak data across the consented-continuity boundary.
