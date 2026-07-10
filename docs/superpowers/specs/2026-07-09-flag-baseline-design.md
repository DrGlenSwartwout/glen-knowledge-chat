# Feature-Flag Baseline — Design

**Date:** 2026-07-09
**Status:** Approved (design); implementation follows
**Extends:** PR #759 (`GET /api/console/flags` + `scripts/surface_check.py:check_flags`)

---

## Problem

PR #759 watches **four** flags via a hardcoded `REQUIRED_ON` tuple. The root-cause dig that same evening found the incident was bigger: **eight** flags had been deleted from the prod Render service, not three. Five were still off — `PREPAY_LADDER_ENABLED`, `CONTINUOUS_CARE_MONTHLY_ENABLED`, `TWO_DOOR_ENABLED`, `ANALYSIS_QUOTA_ENABLED`, `REVIEWS_ENABLED` — silently disabling a Stripe monthly checkout, the two-door reveal, the prepay ladder, analysis-quota enforcement, and the review flow. The #759 watchdog would not have caught any of them.

Attribution is impossible: Render's `GET /v1/owners/<id>/audit-logs` answers `audit logs not available for this plan`, and its events API records builds and deploys but never env changes.

## Goal

Watch **every flag that must be on**, and make turning one off an auditable act.

## Design

### 1. The baseline is a committed file, not a tuple

`scripts/flags_expected.json`:

```json
{"expected_on": ["ANALYSIS_QUOTA_ENABLED", "..."]}
```

38 names, generated from the live app on 2026-07-09 and reviewed by Glen.

`surface_check.REQUIRED_ON` is **derived** from that file at import.

**Why a file and not a longer tuple.** A hardcoded list rots on the first PR that adds a flag, and — worse — it makes *deliberately* turning a flag off look identical to a deletion. With a committed baseline, disabling a flag becomes a **pull request**: git records who intended it, when, and why. That is the attribution Render will not sell us. It is the whole point; the alerting is secondary.

### 2. Scope of the baseline

**Included:** the 38 public `*_ENABLED` flags reported `value: true` by `/api/console/flags` on 2026-07-09. All 38 are backed by a real env var (`env_present: true`).

**Excluded — the 7 leading-underscore aliases** (`_REVIEWS_ENABLED`, `_SALES_PAGES_ENABLED`, `_GMAIL_FEEDBACK_ENABLED`, `_SALES_AI_COPY_ENABLED`, `_SALES_IMAGE_PICK_ENABLED`, `_TESTIMONIALS_ENABLED`, `_TESTIMONIAL_INVITES_ENABLED`). They are module-internal names; each reports `env_present: false` because its environment variable is spelled without the underscore. Watching them would add noise, not coverage — their public counterparts are already in the baseline. A test pins this: **no baseline entry may start with `_`.**

**Silent on unknown flags.** A flag present in `app.py` but absent from the baseline produces **no** alert. New flags default off; alerting on them would punish every feature branch. The baseline answers *"what must be on"*, not *"what exists"*.

### 3. Failure semantics (unchanged from #759, now over 38 flags)

| observation | result |
|---|---|
| baseline flag `value: true` | pass |
| `value: false`, `env_present: true` | **fail** — "set to false" (+ redeploy hint when `source == "import"`) |
| `value: false`, `env_present: false` | **fail** — "env var is MISSING (deleted)" |
| baseline flag absent from the response | **fail** — the deleted-call-time-flag case |
| flag NOT in the baseline, any state | **silent** |
| endpoint 401 / unreachable / malformed | one `{"flag": "*"}` "could not check" — **never drift** |
| **baseline file missing or unparseable** | one `{"flag": "*"}` "could not load baseline" — **never drift, never silent pass** |

That last row is new and load-bearing. A watchdog whose expectations vanished must **say so**, not quietly pass because it now expects nothing.

### 4. Data flow

```
cron container (separate service; survives the web app being down)
  run_personal_email_cron.py -> surface_check.run()
        ├── check_surfaces(BASE_URL)                     # HTTP status >= 400
        └── check_flags(BASE_URL, CONSOLE_SECRET)
                 REQUIRED_ON  <- scripts/flags_expected.json   (committed baseline)
                 GET /api/console/flags  (X-Console-Key)
        └── one merged alert -> send_alert() -> SMTP
```

Still stdlib-only (`json` is stdlib). No new schedule, no new credential.

## Error handling

`check_flags()` never raises. The baseline load is guarded: a missing file, bad JSON, or a non-list `expected_on` all degrade to a single `"could not load baseline"` failure. `run()` remains best-effort and never affects the personal-email send.

## Testing

Each with a watched RED.

1. `flags_expected.json` parses; `expected_on` is a list of 38 unique, sorted, non-empty strings.
2. **No entry starts with `_`** (the alias exclusion).
3. Every entry ends with `_ENABLED`.
4. `REQUIRED_ON` equals the file's contents — not a hardcoded tuple. (Mutate the file in a tmp copy, reload, assert it changed.)
5. The four flags #759 watched are all still in the baseline (no coverage regression).
6. The five flags found deleted during the dig are in the baseline.
7. A baseline flag `value: false` → one failure naming it.
8. A **non**-baseline flag `value: false` → **no** failure.
9. Baseline file missing → one `{"flag": "*", "reason": "could not load baseline: ..."}`, never `[]`.
10. Baseline file with `expected_on` not a list → same single failure.
11. `run()` still merges flag failures into the alert body (wiring, re-asserted).

## Known limits

- The baseline is a **snapshot of intent on 2026-07-09**. If a flag was already wrongly off that day, this enshrines it. Reviewed by Glen; all 38 were `env_present: true`.
- Daily cadence. Drift is visible within 24 hours.
- Still no attribution *from Render*. Attribution now comes from **git**: turning a flag off requires editing the baseline.

## Follow-ups (not this spec)

- Render plan with audit logs is the only way to attribute an out-of-band deletion.
- Consider a boot-time assertion that every `expected_on` flag is true, so a bad deploy fails loudly rather than waiting for the cron.
