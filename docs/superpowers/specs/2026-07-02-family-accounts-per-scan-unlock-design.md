# Family Accounts + Per-Scan Unlock — Design Spec

**Date:** 2026-07-02
**Status:** Approved design, pre-implementation
**Repo:** deploy-chat (illtowell.com portal)

---

## 1. Summary

Replace the portal's blur + $1-lifetime-unlock paywall with a **per-scan access model**, and add **family accounts** that render a tab per family member on the primary's portal.

Two independent-but-related changes, shipped together behind one flag:

1. **Per-scan unlock** (access-model change): no blur; free members unlock one scan per month by selecting it; paid members get all scans automatically.
2. **Family accounts** (new feature): link separate member accounts under a primary; the primary's portal shows a tab per member, each rendering that member's own reports.

**Billing is out of scope** (later pass): the $197/mo family membership subscription and how "paid member" status is *set*. This feature only *reads* paid/family status and enforces the free allowance.

---

## 2. Goals / Non-Goals

### Goals
- Retire the blur mechanic and the $1-lifetime-unlock entirely.
- Free members can permanently unlock **one scan per calendar month**, self-selected, **per member**.
- Paid members (and members of a paid $197/mo family) see **all** their scans unlocked automatically.
- A family primary's portal shows a **tab per member**; each tab shows that member's own reports (latest + date history).
- Each member keeps their **own account/email and reports** — nothing re-keyed. (E4L requires a unique email per account; family grouping is on our side.)
- Console-managed family linking (no auto-linking in v1).
- Ship behind a flag; flip to go live.

### Non-Goals (this pass)
- The $197/mo billing/subscription mechanism and sign-up flow.
- How "paid member" status is *set* (reuse existing paid-member detection).
- Auto-linking families by address/email heuristics.
- Member self-service family management (console-only for v1).
- Migrating historical blurred reports beyond flipping them to the new access rules.

---

## 3. Access Model

| Tier | Access |
|------|--------|
| **Free member** | Unlock **1 scan / calendar month**, self-selected. Unlock is **permanent** (the "1/month" is the rate of *new* unlocks, not a re-lock). Allowance is **per member**. |
| **Paid member** | **All** scans unlocked automatically. No selecting, no rows needed. |
| **$197/mo family** | Every member of the family is treated as paid → all members' scans auto-unlocked. |

**No blur anywhere.** A locked scan appears as a listed entry the client cannot open yet; unlocking grants permanent access to open that scan's full report.

---

## 4. Data Model

### 4.1 `family_members`
Links member accounts to a family primary.

```sql
CREATE TABLE IF NOT EXISTS family_members (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    primary_email TEXT NOT NULL,        -- the family's primary contact (lowercased)
    member_email  TEXT NOT NULL,        -- this member's own account email (lowercased)
    member_label  TEXT,                 -- display name, e.g. "Sasha (Karin's cat)"
    member_type   TEXT DEFAULT 'human', -- 'human' | 'pet'
    display_order INTEGER DEFAULT 0,
    created_at    TEXT,
    UNIQUE(primary_email, member_email)
);
CREATE INDEX IF NOT EXISTS ix_family_primary ON family_members(primary_email);
CREATE INDEX IF NOT EXISTS ix_family_member  ON family_members(member_email);
```

- The primary is themselves a member row (so the primary's own tab renders through the same path).
- A member belongs to at most one family in v1 (enforced by app logic; `member_email` may appear once).

### 4.2 `scan_unlocks`
One row per permanently-unlocked scan. Only free unlocks strictly need rows; paid/family are computed. Rows for paid/family may be written opportunistically for audit but are not required for access.

```sql
CREATE TABLE IF NOT EXISTS scan_unlocks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    member_email TEXT NOT NULL,         -- lowercased
    scan_id      TEXT NOT NULL,
    scan_date    TEXT,
    unlocked_at  TEXT NOT NULL,         -- ISO8601
    source       TEXT NOT NULL,         -- 'free_monthly' | 'paid' | 'family'
    UNIQUE(member_email, scan_id)
);
CREATE INDEX IF NOT EXISTS ix_unlock_member ON scan_unlocks(member_email);
```

**Monthly cap query** (free path): a member has spent their allowance for month `YYYY-MM` if any `scan_unlocks` row exists with `member_email = ? AND source='free_monthly' AND strftime('%Y-%m', unlocked_at) = ?`.

---

## 5. The Gate — one helper

All reveal/access decisions route through a single function so the logic lives in exactly one place.

```
def scan_accessible(cx, member_email, scan_id) -> bool:
    if is_paid_member(member_email):            # reuse existing paid detection
        return True
    if family_is_paid(cx, member_email):        # family's $197/mo active (reads billing signal)
        return True
    return has_unlock_row(cx, member_email, scan_id)   # explicit scan_unlocks row
```

- `is_paid_member` — reuse the existing paid-member detection already in the codebase.
- `family_is_paid` — resolve the member's `primary_email` via `family_members`, then check the family membership signal. In v1 this signal is a stub/flag (billing is a later pass) so it returns False unless a family is manually marked active for testing.
- `has_unlock_row` — explicit `scan_unlocks` lookup.

---

## 6. Free Monthly Unlock — action

Endpoint (member-authenticated via portal token): `POST /api/portal/<token>/unlock-scan` `{scan_id}`.

Logic:
1. Resolve the token → member_email.
2. If `scan_accessible` already true → no-op success (already unlocked / paid).
3. Else check the monthly cap for this member. If already spent this calendar month → refuse with a clear "1 free unlock per month — resets on [first of next month]" message.
4. Else insert `scan_unlocks(member_email, scan_id, scan_date, now, 'free_monthly')`. Permanent.

Idempotent on `(member_email, scan_id)` via the UNIQUE constraint. Cap enforced server-side (never trust the client).

---

## 7. Portal UI (no blur)

- **Token resolution:** on `/portal/<token>`, resolve the email. If that email is a family `primary_email`, load the family's members (ordered by `display_order`).
- **Family portal:** render a **tab bar**, one tab per member (label from `member_label`). Selecting a tab shows that member's report area.
- **Non-family portal:** no tab bar; render the single member's report area (unchanged layout).
- **Within a member's report area:**
  - List that member's scans (date history), newest first — reuse existing `list_report_dates`-style logic keyed to the member's email.
  - An **accessible** scan (per `scan_accessible`) opens its full report inline / on selection.
  - A **locked** scan shows a lock icon + **"Unlock this scan"** button. When the member's monthly free allowance is spent, the button is disabled with "1 free unlock used — resets [date]".
  - No blur overlay anywhere.

Member standalone tokens continue to work on their own (a member opening their own token sees just their report area, same per-scan rules).

---

## 8. Console — family management

- Console UI (owner-gated, `X-Console-Key`) to:
  - Create a family / set the primary.
  - Add a member: `member_email`, `member_label`, `member_type`, `display_order`.
  - Remove / reorder members.
- Backing endpoints: `POST /api/console/family` (upsert member), `DELETE /api/console/family/<member_email>`, `GET /api/console/family/<primary_email>`.
- No auto-linking in v1.

**Data cleanup for the current live case:** Sasha's report was published cross-keyed under Karin's real email as a stopgap. Under this design, Sasha's report moves back under her own account (fake E4L email `permanentlyyours777@hawaiiantel.net`), and a `family_members` row links her to primary `permanentlyyours@hawaii.rr.com` with label "Sasha (Karin Takahashi's cat)". The cross-keyed report row under Karin's email is removed.

---

## 9. Retiring blur + $1

- New model ships behind flag **`PORTAL_ACCESS_V2`** (default off).
- When **on**: blur rendering, the $1-lifetime-unlock flow, and the trial gate are bypassed; per-scan unlock + family tabs are active.
- Legacy flags `PORTAL_PAID_GATE_ENABLED` and `BIOFIELD_TRIAL_ENABLED` are retired once V2 is verified on prod.
- Ships dark; flip `PORTAL_ACCESS_V2=1` in Render to go live.
- Any code path that currently renders a blurred/partial report is gated: `if PORTAL_ACCESS_V2: use scan_accessible; else: legacy blur`.

---

## 10. Testing

- **Unit:** `scan_accessible` truth table (free/paid/family × unlocked/locked); monthly-cap enforcement across month boundaries (a `free_monthly` unlock in month N does not block month N+1); idempotent unlock insert.
- **Family resolution:** primary → members ordered; member with no family; member in a family; pet member.
- **Endpoint:** unlock-scan happy path, cap-exceeded refusal, already-paid no-op, unknown scan.
- **UI/integration:** family portal renders N tabs; locked vs accessible scan rendering; no blur present anywhere when V2 on; standalone member token unchanged.
- **Regression:** with `PORTAL_ACCESS_V2` off, legacy behavior is byte-for-byte unchanged.

---

## 11. Rollout

1. Land schema + gate + endpoints + UI behind `PORTAL_ACCESS_V2` (off).
2. Verify on prod dark (console family management works; test family with a stub-active membership; free unlock cap).
3. Migrate the Sasha/Karin live case (§8).
4. Flip `PORTAL_ACCESS_V2=1`; render-verify each surface.
5. Retire legacy blur/$1 flags after a soak.

---

## 12. Open Questions / Dependencies

- **Billing (separate sub-project):** the $197/mo family subscription and the mechanism that sets "family membership active" and "paid member." This spec reads those signals; `family_is_paid` is a stub until billing lands.
- **Paid-member detection:** confirm the exact existing helper to reuse (e.g. `_is_paid_member` / `biofield_readiness.paid_at`) during implementation.
- **Member in multiple families:** disallowed in v1 (one family per member).
