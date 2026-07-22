# Portal — My Health Profile (Phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add "My Health Profile" to the client portal: a client-editable view of their intake record that writes back to the **same** `intake_responses` table the admin console reads, plus a persisted AI-suggestions queue fed by their Ask-Dr-Glen chat (confirm / edit / dismiss). Ships dark behind `PORTAL_HEALTH_PROFILE_ENABLED`.

**Architecture:** Reuse the existing intake store and portal write-back conventions. `intake_responses` (in `chat_log.db` = `LOG_DB`, keyed by email, answers as `answers_json`) is the single source of truth; `INTAKE_FORM` (`dashboard/intake.py`) defines the fields. A new `health_suggestions` table holds pending chat-extracted edits. The portal panel renders a curated, editable subset of the record + the suggestions queue; edits and confirmed suggestions both flow through one self-edit write path into `intake_responses`. Console read (`/api/console/intake/<email>`) reflects everything automatically.

**Tech Stack:** Python 3.9 + Flask (`app.py`, `dashboard/intake.py`, `dashboard/intake_public.py`, new `dashboard/health_suggestions.py`), SQLite, vanilla client JS/CSS in `static/client-portal.html`. Server changes get pytest; client changes get server-free Node checks + a pre-flip render pass.

## Global Constraints

- **Flag `PORTAL_HEALTH_PROFILE_ENABLED`, default OFF**, truthy set `("1","true","yes","on")` (mirror `PORTAL_HUB_ENABLED`).
- **One source of truth:** all writes land in `intake_responses` (email-keyed). No second health store.
- **Identity from the token, never the request body** — resolve via `_portal_record_for(cx, token)` (`app.py:19166`), like every `/api/portal/<token>/*` endpoint.
- **Never clobber a submitted intake:** a self-edit updates `answers_json` but must preserve `status='submitted'` and stamp an edit time; do not reset to draft or wipe unedited answers.
- **Curated editable scope (NOT the whole intake form):** only the client-owned health fields are editable here — top health goals/concerns and the personal-health-history section (sleep, dental, vaccinations, and the `supplements`/`diagnoses`/`medications`/`surgeries`/`allergies` tables). The 5 clinical-dimension scales (`terrain`, `penetration`, `tissue_layer`, `response`, `commitment`), personal-info identity fields, and consent/signature are **read-only or excluded** — they are clinician-assessed or legal.
- **Suggestions are pending until confirmed** — nothing a chat extractor proposes writes to `intake_responses` until the client confirms/edits.
- Copy: no em dashes, no ALL CAPS. Theme-aware CSS vars only. Render-verify (not just parse) before flag flip.

---

### Task 1: Flag + curated read into the portal payload

**Files:**
- Modify: `app.py` (define `_PORTAL_HEALTH_PROFILE_ENABLED` near `_PORTAL_HUB_ENABLED`; pass to `get_portal_view`)
- Modify: `dashboard/portal_view.py` (`get_portal_view` param + `"health_profile"` payload block via a new helper)
- Create: `dashboard/health_profile.py` (`curated_fields()` + `build_block(cx, email, enabled)`)
- Test: `tests/test_health_profile_block.py`

**Interfaces:**
- Produces: `dashboard/health_profile.py`:
  - `EDITABLE_FIELD_IDS: set[str]` — the curated allow-set (health goals + history fields, from `INTAKE_FORM`).
  - `build_block(cx, email, enabled) -> dict`: `{"enabled": bool, "status": "off"|"empty"|"has_record", "sections": [ {title, fields:[{id,label,type,value}]} ], "suggestion_count": int}`. Returns `{"enabled": False}` when flag off; degrades to empty on any error (mirror `_practitioner_finder_block`).

- [ ] **Step 1: Write the failing test**
```python
# tests/test_health_profile_block.py
from dashboard import health_profile

def test_editable_ids_exclude_clinical_and_consent():
    ids = health_profile.EDITABLE_FIELD_IDS
    for excluded in ("terrain","penetration","tissue_layer","response","commitment","terms"):
        assert excluded not in ids
    assert "health_concerns" in ids   # a curated health field IS editable

def test_build_block_off_when_disabled():
    assert health_profile.build_block(None, "a@b.com", False) == {"enabled": False}
```

- [ ] **Step 2: Run test to verify it fails** — `python3 -m pytest tests/test_health_profile_block.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement `dashboard/health_profile.py`.** Derive `EDITABLE_FIELD_IDS` from `INTAKE_FORM` (import it; include the health-goals + personal-health-history section field ids; exclude the 5 dimension scales, personal-info identity fields, and `terms`). `build_block` reads `intake.get_response(cx, email)`, projects the curated fields with their labels/types from `INTAKE_FORM`, and counts pending rows via `health_suggestions.count_pending(cx, email)` (guard with try/except so Task 5's table absence degrades to 0).

- [ ] **Step 4: Thread the flag + block.** In `app.py` add `_PORTAL_HEALTH_PROFILE_ENABLED` (same parse as `_PORTAL_HUB_ENABLED`) and pass `health_profile_enabled=...` into `get_portal_view`. In `dashboard/portal_view.py` add the param and `"health_profile": _hp.build_block(cx, email, health_profile_enabled)` in the view dict.

- [ ] **Step 5: Run tests** → PASS.

- [ ] **Step 6: Commit** — `feat: PORTAL_HEALTH_PROFILE_ENABLED flag + curated intake read block`

---

### Task 2: Render the My Health Profile tile + read-only panel

**Files:** Modify `static/client-portal.html` (add tile to `buildHubHtml` Understand group; add `data-panel="health"` in the wrap block; render `v.health_profile.sections` read-only).

**Interfaces:** Consumes `v.health_profile` (Task 1) and the existing `backToHub()` / panel-wrap conventions.

- [ ] **Step 1** — In `buildHubHtml`, add to the Understand group a tile `["health", "My Health Profile", "Your history, symptoms and intake"]` (only meaningful when `v.health_profile.enabled`; still safe if not, `showTab` falls back to hub). Show the pending-count as a small badge when `v.health_profile.suggestion_count > 0`.
- [ ] **Step 2** — Add a global `buildHealthProfileHtml(v)` that renders `v.health_profile.sections` as read-only field rows (label + value), plus a placeholder container `<div id="healthSuggestions"></div>` (populated in Task 7) and, per field, an "Edit" affordance (wired in Task 4). Escape every value with `esc()`.
- [ ] **Step 3** — In the wrap block, emit `${_hub && v.health_profile && v.health_profile.enabled ? \`<section data-panel="health" hidden>${back}${buildHealthProfileHtml(v)}</section>\` : ""}`.
- [ ] **Step 4** — Server-free Node check: `buildHealthProfileHtml` output contains the section titles + values for a sample `v`; both script blocks parse. Commit — `feat: My Health Profile tile + read-only record panel`

---

### Task 3: Self-edit write-back endpoint (no submitted-row clobber)

**Files:** Modify `app.py` (new `POST /api/portal/<token>/health-profile`); Modify `dashboard/intake.py` (add `save_self_edit(cx, email, partial_answers)` that merges into existing `answers_json`, preserves `status`, stamps `self_edited_at`); Test `tests/test_health_profile_write.py`.

**Interfaces:**
- Produces: `intake.save_self_edit(cx, email, partial_answers: dict) -> dict` — loads the row, merges only `EDITABLE_FIELD_IDS` keys (reuse `intake_public.merge_answers` for coercion/whitelisting), writes back `answers_json` with unchanged `status`, sets `self_edited_at`; creates the row as a draft if none exists. Returns the updated curated block.
- `POST /api/portal/<token>/health-profile`: body `{field_id, value}` or `{answers:{...}}`; resolves email from token; 404 when flag off; 400 on non-editable field id.

- [ ] **Step 1: Failing test**
```python
# tests/test_health_profile_write.py
import sqlite3
from dashboard import intake

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    intake.init_intake_table(cx); return cx

def test_self_edit_preserves_submitted_status():
    cx = _cx()
    intake.submit(cx, "a@b.com", {"sleep": "poor"})           # a submitted row
    intake.save_self_edit(cx, "a@b.com", {"sleep": "improving"})
    row = intake.get_response(cx, "a@b.com")
    assert row["status"] == "submitted"                        # not reset to draft
    assert row["answers"]["sleep"] == "improving"             # value updated

def test_self_edit_rejects_noneditable_field():
    cx = _cx(); intake.submit(cx, "a@b.com", {"sleep": "poor"})
    intake.save_self_edit(cx, "a@b.com", {"terrain": 5})       # clinical scale, excluded
    assert "terrain" not in intake.get_response(cx, "a@b.com")["answers"]
```
(Adjust `submit`/`get_response` calls to the real signatures in `dashboard/intake.py`.)

- [ ] **Step 2: Run → FAIL** (`save_self_edit` missing).
- [ ] **Step 3: Implement `save_self_edit`** in `dashboard/intake.py` per the interface (merge whitelisted keys via `health_profile.EDITABLE_FIELD_IDS` + `intake_public.merge_answers`, preserve status, stamp time, `_db_lock` at the endpoint layer).
- [ ] **Step 4: Implement the endpoint** in `app.py` mirroring `api_portal_client_fact` (`app.py:21523`): flag gate → `_portal_record_for` → email → validate field id in `EDITABLE_FIELD_IDS` → `intake.save_self_edit` under `_db_lock` → return refreshed `health_profile.build_block`.
- [ ] **Step 5: Run tests → PASS.**
- [ ] **Step 6: Commit** — `feat: portal health-profile self-edit write-back into intake_responses`

---

### Task 4: Wire the panel's inline editing

**Files:** Modify `static/client-portal.html` (edit affordance → editable input → POST to the Task 3 endpoint → live re-render of the field).

- [ ] **Step 1** — Per curated field, an "Edit" control swaps the value for an input (text / number / select per the field's `type` from the block). Save posts `{field_id, value}` to `/api/portal/<token>/health-profile` (reuse the existing portal fetch helper + token global), then updates the field in place from the JSON response. Table fields (supplements/diagnoses/etc.) edit as add/remove rows.
- [ ] **Step 2** — Server-free Node parse check + a logic check that the save handler builds the correct payload. Commit — `feat: inline editing for My Health Profile fields`

---

### Task 5: `health_suggestions` table + chat extraction

**Files:** Create `dashboard/health_suggestions.py`; Modify `app.py` portal-chat tail (`~21837`) to extract + persist; Test `tests/test_health_suggestions.py`.

**Interfaces:**
- Produces `dashboard/health_suggestions.py`:
  - `init_table(cx)` → `health_suggestions(id INTEGER PK, email TEXT, source_msg_id INTEGER, field_id TEXT, suggested_value TEXT, rationale TEXT, status TEXT DEFAULT 'pending', created_at TEXT, resolved_at TEXT)`, index `(email, status)`.
  - `add_pending(cx, email, field_id, value, rationale, source_msg_id)` — INSERT; dedupe on `(email, field_id, suggested_value, status='pending')` (UNIQUE + INSERT OR IGNORE) so re-mentions don't stack.
  - `list_pending(cx, email) -> list[dict]`; `count_pending(cx, email) -> int`; `resolve(cx, sug_id, email, status)` (`confirmed|edited|dismissed`, stamps `resolved_at`).
  - `extract_from_turn(client_msg, assistant_msg) -> list[{field_id,value,rationale}]` — parse a chat turn into candidate **editable** health facts, constrained to `health_profile.EDITABLE_FIELD_IDS` (model on the existing `_CONCIERGE_EXTRACT_SYSTEM` prompt at `app.py:11162`, but targeting health-record fields, not products).

- [ ] **Step 1: Failing test** — `init_table` then `add_pending` twice with identical `(email,field_id,value)` → `count_pending == 1` (dedupe); `resolve(...,'dismissed')` → `count_pending == 0`.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement `dashboard/health_suggestions.py`** (table + CRUD + dedupe). Leave `extract_from_turn` returning `[]` unless a model call is wired; keep it pure/testable (accept an injected extractor for the unit test).
- [ ] **Step 4: Hook extraction into the portal-chat tail** at `app.py:~21837` (beside `portal_chat.record_exchange`): flag-gated, call `extract_from_turn` on the exchange and `add_pending` each result keyed by the client email. Must never break the chat stream (wrap in try/except, mirror the existing suggestion SSE guard).
- [ ] **Step 5: Run tests → PASS.**
- [ ] **Step 6: Commit** — `feat: health_suggestions table + chat extraction of pending record edits`

---

### Task 6: Suggestions endpoints (list / confirm / edit / dismiss)

**Files:** Modify `app.py` (`GET /api/portal/<token>/health-suggestions`, `POST .../<sid>/resolve`); Test `tests/test_health_suggestions_api.py`.

**Interfaces:** `GET` → pending list for the token's email. `POST .../<sid>/resolve` body `{action: "confirm"|"edit"|"dismiss", value?}` — `confirm` writes `suggested_value` into `intake_responses` via `intake.save_self_edit`; `edit` writes the supplied `value`; both then `health_suggestions.resolve(...,'confirmed'|'edited')`; `dismiss` just resolves. All resolve the email from the token, guard the `sid` belongs to that email.

- [ ] **Step 1: Failing test** — seed a pending suggestion, POST confirm → the value appears in `intake_responses` (via `get_response`) AND the suggestion is no longer pending; POST dismiss on another → gone from pending, `intake_responses` unchanged; a sid belonging to a different email → 403/404.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement both endpoints** (flag gate, token identity, ownership check, `_db_lock`, reuse `save_self_edit`).
- [ ] **Step 4: Run tests → PASS.**
- [ ] **Step 5: Commit** — `feat: health-suggestions list + confirm/edit/dismiss endpoints`

---

### Task 7: Suggestions queue UI in the panel

**Files:** Modify `static/client-portal.html` (populate `#healthSuggestions` from the Task 6 GET; render each pending item with Confirm / Edit / Dismiss; on action, POST and update the record + count).

- [ ] **Step 1** — On opening the health panel (or at render when `suggestion_count>0`), fetch `GET /api/portal/<token>/health-suggestions` and render each as: "We heard this in your conversations: {rationale} → add to {field label}?" with Confirm / Edit / Dismiss. Confirm/Edit/Dismiss POST to `.../<sid>/resolve`; on success remove the item, update the tile badge, and refresh the affected field in the record.
- [ ] **Step 2** — Server-free Node parse + a logic check of the render/handlers. Commit — `feat: My Health Profile AI-suggestions confirm/edit/dismiss queue`

---

### Task 8: Rollout note + pre-flip verification checklist

- [ ] Write `docs/superpowers/plans/2026-07-22-portal-health-profile-rollout.md`: the flag (`PORTAL_HEALTH_PROFILE_ENABLED`, off), a render-verify checklist (tile + badge, record renders, inline edit saves and the console `/api/console/intake/<email>` reflects it, a chat mention produces a pending suggestion, confirm writes through, dismiss doesn't), and the note that edits/confirms write to the shared `intake_responses` so the console shows them. Commit.

## Self-Review

- **Spec coverage:** editable record (Tasks 1-4) + single-source-of-truth write-back to `intake_responses` (Task 3, verified against console read) + AI-suggestions confirm/edit/dismiss, pending-until-confirmed (Tasks 5-7). Flag-gated dark (Task 1). Curated scope excludes clinical scales/consent per Global Constraints.
- **Placeholder scan:** `extract_from_turn` ships testable with an injectable extractor; the live model wiring is explicit in Task 5 Step 4, not a TODO.
- **Type consistency:** `EDITABLE_FIELD_IDS`, `build_block`, `save_self_edit`, `add_pending/list_pending/count_pending/resolve/extract_from_turn` names are consistent across tasks; `health_profile` payload key and `data-panel="health"` match between server and client.

## Open decision for review
- **Curated editable scope** (Global Constraints) — confirm the client should edit health goals + personal-health-history but NOT the 5 clinical-dimension scales (my recommendation: yes, those are clinician-assessed). If you want clients to also self-report the dimension scales, that widens `EDITABLE_FIELD_IDS`.
