# Certification Biofield Bonus — Plan (Upgrade Ladder Mechanic 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A certification enrollee who has committed (pay-in-full or 12-month) earns **a bonus Biofield Analysis each month for 12 months**, plus **a bonus Biofield each time they complete a module** (1–12). Each earned bonus records an entitlement + drops an ops task for the team to run it concierge-style + notifies the practitioner. Ships dark behind `CERT_BONUS_ENABLED`.

**Architecture:** All sqlite (LOG_DB) + the existing `todos` spine — no Supabase migration. A new `dashboard/cert_bonus.py` holds the commitment store (admin-set flag, no in-app cert checkout exists), the grant ledger (idempotent), and a pure `due_bonuses()` that computes which monthly/level grants are owed. A daily cron sweeps active commitments, reads each practitioner's `modules_completed` (the merged `modules_completed_for_email`, Supabase), computes due grants, and for each new one records it + creates a `todos` task + best-effort notifies. An admin endpoint sets/clears the commitment.

**Why concierge (not self-serve):** the #114 `biofield_readiness` table is one-row-per-email (a single self-serve purchase), so it can't represent repeated monthly/level bonuses; and cert enrollees are high-touch (a $3,600+ program with Glen), so bonuses are delivered by the team. The bonus = entitlement + ops task + notify, NOT a self-serve gate seed. (Auto-seeding the gate is a later option.)

**Tech Stack:** Python 3.11, Flask, sqlite, pytest.

**Spec:** `docs/superpowers/specs/2026-06-15-upgrade-incentive-ladder-design.md` (Mechanic 3). Glen decisions (2026-06-15): commitment = admin/console flag (`cert_commitment` 'pif'|'monthly12', `cert_started_at`); level grant = one Biofield per module completed (1–12); monthly = 12 bonuses then stop.

**Reuse:** `practitioner_portal.modules_completed_for_email(email)` (merged #111, Supabase); the `todos` INSERT pattern (app.py ~2580/7100, `dedup_key` UNIQUE → idempotent); the cron-secret pattern (`X-Cron-Secret` vs `CRON_SECRET`/`CONSOLE_SECRET`, e.g. `cron_charge_subscriptions`); `add_months` (`dashboard/subscriptions.py`) for month math; the console-key gate for the admin endpoint.

**Test invocation:** pure → `~/.venvs/deploy-chat311/bin/python -m pytest <path> -q`. App → `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest <path> -q` (worktree; ignore the 2 known pre-existing failures).

---

### Task 1: `dashboard/cert_bonus.py` — commitment store + grant ledger + due_bonuses (pure)

**Files:** Create `dashboard/cert_bonus.py`; Test `tests/test_cert_bonus.py`

- [ ] **Step 1: Failing test**

```python
import sqlite3
from dashboard import cert_bonus as cb

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    cb.init_tables(cx); return cx

def test_commitment_set_get_list():
    cx = _cx()
    cb.set_commitment(cx, "doc@x.com", kind="pif", started_at="2026-01-15")
    r = cb.get_commitment(cx, "doc@x.com")
    assert r["kind"] == "pif" and r["started_at"] == "2026-01-15" and r["active"]
    assert [c["email"] for c in cb.list_active(cx)] == ["doc@x.com"]
    cb.clear_commitment(cx, "doc@x.com")
    assert cb.get_commitment(cx, "doc@x.com")["active"] == 0
    assert cb.list_active(cx) == []

def test_due_bonuses_monthly_and_level():
    # started 3 full months ago, 2 modules completed, nothing granted yet
    grants = cb.due_bonuses(started_at="2026-01-01", modules_completed=2,
                            granted=set(), today="2026-04-01")
    # monthly: months elapsed 1,2,3 (capped 12) ; level: 1,2
    assert ("monthly", 1) in grants and ("monthly", 3) in grants
    assert ("monthly", 4) not in grants            # only 3 months elapsed
    assert ("level", 1) in grants and ("level", 2) in grants
    assert ("level", 3) not in grants

def test_due_bonuses_excludes_already_granted_and_caps_12():
    grants = cb.due_bonuses(started_at="2024-01-01", modules_completed=12,
                            granted={("monthly", m) for m in range(1, 13)} | {("level", 1)},
                            today="2026-06-15")
    assert not any(k == "monthly" for k, _ in grants)   # all 12 monthly already granted
    assert ("level", 1) not in grants
    assert ("level", 12) in grants
    # monthly never exceeds 12 even though >12 months elapsed
    assert all(idx <= 12 for k, idx in grants if k == "monthly")

def test_record_grant_idempotent_and_granted_pairs():
    cx = _cx()
    cb.record_grant(cx, "doc@x.com", kind="monthly", idx=1, todo_id=10)
    cb.record_grant(cx, "doc@x.com", kind="monthly", idx=1, todo_id=11)  # dup ignored
    pairs = cb.granted_pairs(cx, "doc@x.com")
    assert pairs == {("monthly", 1)}
    n = cx.execute("SELECT COUNT(*) FROM cert_bonus_grants WHERE email=?", ("doc@x.com",)).fetchone()[0]
    assert n == 1
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** `dashboard/cert_bonus.py` (pure; cx + args). Tables:
```sql
CREATE TABLE IF NOT EXISTS cert_commitments (
  email TEXT PRIMARY KEY, kind TEXT, started_at TEXT,
  active INTEGER NOT NULL DEFAULT 1, created_at TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS cert_bonus_grants (
  email TEXT, kind TEXT, idx INTEGER, todo_id INTEGER, granted_at TEXT,
  PRIMARY KEY (email, kind, idx));
```
- `init_tables(cx)`, `_now()`.
- `set_commitment(cx, email, *, kind, started_at)` — upsert active=1.
- `get_commitment(cx, email)` → dict|None; `clear_commitment(cx, email)` — set active=0; `list_active(cx)` → list of dicts where active=1.
- `record_grant(cx, email, *, kind, idx, todo_id=None)` — `INSERT OR IGNORE` (PK dedup).
- `granted_pairs(cx, email)` → set of (kind, idx).
- **`due_bonuses(*, started_at, modules_completed, granted, today)`** (PURE, no cx): months_elapsed = whole months from started_at to today (use a simple year*12+month diff, or count via `subscriptions.add_months` comparisons); `monthly = {("monthly", m) for m in 1..min(months_elapsed, 12)}`; `level = {("level", n) for n in 1..max(0, min(int(modules_completed or 0), 12))}`; return sorted list of `(monthly ∪ level) − granted`. Keep month math dependency-free (parse YYYY-MM-DD, `(ty-sy)*12 + (tm-sm)`, minus 1 if today's day < start day... keep it simple: full elapsed months).

- [ ] **Step 4: Run → pass.** **Step 5: Commit** — `feat(cert-bonus): commitment store + ledger + due_bonuses`

---

### Task 2: cron sweep + admin commitment endpoint

**Files:** Modify `app.py`; Test `tests/test_cert_bonus_routes.py`

- [ ] **Step 1: Failing test** — `CERT_BONUS_ENABLED=1`, tmp LOG_DB.
  - `POST /api/cert/commitment {email, kind:"pif", started_at}` (console-gated; 401 without key) → sets the commitment (`cert_bonus.get_commitment` reflects it). A `{clear:true}` variant clears it.
  - `POST /api/cron/biofield-bonuses` (X-Cron-Secret) with one active commitment started ~3 months ago; monkeypatch `appmod._pp.modules_completed_for_email` → 2. → grants monthly 1..3 + level 1..2 = 5 bonuses: each creates a `todos` row (category "biofield-bonus") and a `cert_bonus_grants` row; response `{ok, granted: 5}`. Idempotent: a second run with the same modules/time → `granted: 0`, no new todos.
  - `?dry_run=1` → computes/â€‹returns the count, writes nothing.
  - Flag off → cron returns disabled, no grants.

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** in `app.py`:
  - `POST /api/cert/commitment` — console-key gated (mirror other console endpoints); body `{email, kind, started_at, clear?}`; `cert_bonus.init_tables`; `set_commitment` or `clear_commitment`; return the commitment.
  - `POST /api/cron/biofield-bonuses` — `X-Cron-Secret` (== `CRON_SECRET` or `CONSOLE_SECRET`), flag-gated (`CERT_BONUS_ENABLED`), `?dry_run`. For each `cert_bonus.list_active(cx)`:
    - `modules = _pp.modules_completed_for_email(email)` (best-effort; default 0 on error),
    - `granted = cert_bonus.granted_pairs(cx, email)`,
    - `due = cert_bonus.due_bonuses(started_at=c["started_at"], modules_completed=modules, granted=granted, today=<today ISO>)`,
    - for each `(kind, idx)` in due (skip writes if dry_run): insert a `todos` row (created_at, owner "glen", category "biofield-bonus", title f"Biofield bonus due — {email} ({kind} {idx})", body w/ context, priority "normal", source "cert-bonus", `dedup_key=f"cert-bonus-{email}-{kind}-{idx}"`), capture todo id, `cert_bonus.record_grant(cx, email, kind=kind, idx=idx, todo_id=tid)`; (best-effort: queue a practitioner notification — optional, a log line is fine for v1).
    - tally `granted`. Return `{ok, granted, dry_run}`.
  - Import `cert_bonus` + `_pp` (the module-level practitioner_portal alias) locally/at module scope as available.

- [ ] **Step 4: Run → pass.** **Step 5: Commit** — `feat(cert-bonus): cron sweep + admin commitment endpoint`

---

### Task 3: flag + doc + suite

**Files:** Create `docs/cert-bonus.md`

- [ ] **Step 1:** Confirm `CERT_BONUS_ENABLED` gates the cron (the admin endpoint is console-gated regardless). Add `CRON_SECRET`/scheduling note (a daily cron should hit `/api/cron/biofield-bonuses`).
- [ ] **Step 2:** `docs/cert-bonus.md`: the offer (committed cert enrollee → 12 monthly Biofield bonuses + one per module completed); commitment is an admin flag (`POST /api/cert/commitment`, no in-app cert checkout); concierge delivery (entitlement + `todos` task + notify, NOT the self-serve gate, and why); idempotent ledger; `CERT_BONUS_ENABLED` flag + the daily cron.
- [ ] **Step 3:** Suite green: `… -m pytest tests/test_cert_bonus.py tests/test_cert_bonus_routes.py -q`.
- [ ] **Step 4:** Commit — `docs(cert-bonus): certification Biofield bonus`

---

## Self-review
- **Spec coverage:** committed enrollee (admin flag, PIF/12-mo) → 12 monthly Biofield bonuses + one per module completed (Task 1 due_bonuses, Task 2 cron); concierge delivery via `todos` (Task 2); idempotent ledger (Task 1 PK + Task 2 dedup_key); `CERT_BONUS_ENABLED` dark flag.
- **Type consistency:** `cert_bonus` (init_tables/set_commitment/get_commitment/clear_commitment/list_active/record_grant/granted_pairs/due_bonuses); `(kind, idx)` pairs with kind in {"monthly","level"}; endpoints `/api/cert/commitment`, `/api/cron/biofield-bonuses`; todos category "biofield-bonus", dedup_key `cert-bonus-<email>-<kind>-<idx>`.
- **Deferred:** auto-seeding the self-serve #114 gate for a cert bonus (concierge for now); practitioner notification beyond a task/log; Supabase-native commitment field (sqlite is enough + testable); PIF vs 12-mo behavior differences (both → the same 12 monthly schedule for now).
- **Risk:** low — no money movement (grants are entitlements/tasks, not charges); idempotent (PK + dedup_key); dark flag; modules lookup best-effort (defaults 0). Worst case is a duplicate ops task, prevented by the dedup_key.

## Done
A committed certification enrollee earns 12 monthly Biofield bonuses + one per module completed, each delivered as an ops task to run concierge-style — shipped dark behind `CERT_BONUS_ENABLED`.
