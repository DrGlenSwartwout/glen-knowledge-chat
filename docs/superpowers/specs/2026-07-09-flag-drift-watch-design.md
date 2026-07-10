# Feature-Flag Drift Watch — Design

**Date:** 2026-07-09
**Status:** Approved (design); implementation not started
**Extends:** PR #736 (`scripts/surface_check.py`)

---

## Problem

On 2026-07-09, three feature flags **vanished** from the prod Render service. A targeted single-key `GET /v1/services/<svc>/env-vars/FIRESIDE_ENABLED` returned `404 not found` — they were **deleted**, not set to a falsy value.

| flag | consequence while missing |
|---|---|
| `REPERTOIRE_ENABLED` | `_repertoire_slugs()` returns `None`; **paid members lose repertoire reorder pricing and are silently charged more** |
| `INVOICE_PAYLINK_ENABLED` | `pay_enabled` False; **clients cannot pay an invoice online** |
| `FIRESIDE_ENABLED` | `/begin/fireside` 404s behind a live "Sit by the fire" CTA on `/begin` |

Three other flags (`SCAN_REQUEST_ENABLED`, `PAY_IT_FORWARD_ENABLED`, `HOUSEHOLD_VIEW_ENABLED`) survived. **Cause unattributed.** The `render.yaml` Blueprint theory is *disproven*: none of the six are declared there, yet three survived. Render's events API records builds and deploys but **not** environment changes (`envUpdated: false` on every deploy that day).

**The existing watchdog structurally cannot see two of the three.** `scripts/surface_check.py` (PR #736) asks only "does this URL return `< 400`?" That catches `FIRESIDE_ENABLED`. `REPERTOIRE_ENABLED` and `INVOICE_PAYLINK_ENABLED` have no page that 404s — and they are the two that cost money.

## Goal

Alert when a flag that must be on is not on, whatever the cause: deleted, set false, or set correctly but never deployed.

## Scope

**Watched (must always be `true`)** — named by Glen, 2026-07-09:

```
FIRESIDE_ENABLED  REPERTOIRE_ENABLED  INVOICE_PAYLINK_ENABLED  SCAN_REQUEST_ENABLED
```

**Explicitly NOT watched:** the other **59** `*_ENABLED` flags in `app.py`. Several are experiments (`TWO_DOOR_ENABLED`, `JOURNEY_QUEST_ENABLED`, `PROGRAM_CARE_TASTER_ENABLED`) where *off* is the correct state. **Watching a flag that is deliberately off is worse than not watching it** — a watchdog that cries wolf gets ignored, which is how this incident stayed invisible.

**Out of scope: auto-remediation.** The check never re-sets a flag it finds off. Flags get turned off on purpose; a watchdog that silently reverses a human decision is worse than none. It reports; Glen decides.

---

## Current state (verified 2026-07-09)

- `app.py` holds **63 distinct `*_ENABLED` flags**: **34 module-level constants** evaluated at import, and **29 that exist only as call-time `os.environ.get(...)` reads inside functions** (verified by AST-free regex over `app.py`). `SCAN_REQUEST_ENABLED` (`app.py:14808`) is one of the 29 — not an exception but the common case. The two kinds need different handling — see Design §1.
- **No endpoint reports any flag.** The only one ever exposed is `_GMAIL_FEEDBACK_ENABLED`, inside an unrelated diagnostics payload (`app.py:7139`).
- `scripts/surface_check.py` exposes `PUBLIC_SURFACES`, `check_surfaces()`, `format_alert()`, `send_alert()`, `run()`. It uses **no** secret today.
- It runs from `scripts/run_personal_email_cron.py` (`check_public_surfaces()`), in the **cron container** — a separate service from the web app, which is why it still fires when the web app is down.
- That cron module already reads `CONSOLE_SECRET` (`scripts/run_personal_email_cron.py:33`) for its other piggybacks.
- Console-gated endpoints use the `@require_console_key` decorator (`dashboard/__init__.py:22`), as `/api/shipping/*` does.

---

## Design

### 1. `GET /api/console/flags` (new)

Gated with `@require_console_key`, matching `/api/shipping/*`.

**There are TWO kinds of flag in `app.py`, and they behave differently.** Found during spec self-review; it changed the design twice.

- **Import-time constants** (34) — `FIRESIDE_ENABLED = os.environ.get(...)` at module level (`app.py:5085`). Fixed when the process starts. Can go **stale**: set the env var, skip the deploy, and the app still behaves as if it's off.
- **Call-time reads** (29) — no module global at all; the environment is read inside a function on each request (`SCAN_REQUEST_ENABLED`, `app.py:14808`). Never stale. **A globals scan would never find any of them**, and the checker would report them permanently missing.

Call-time reads are the majority, so a hand-maintained registry of them would rot on the first PR that adds one. **Discovery therefore uses neither a registry nor a source scan.** The endpoint returns the union of:

1. every **module global** in `app.py` matching `^_?[A-Z][A-Z0-9_]*_ENABLED$`, and
2. every **`os.environ` key** matching the same pattern.

```json
{"ok": true, "flags": {
  "FIRESIDE_ENABLED":     {"value": true,  "env_present": true,  "source": "import"},
  "REPERTOIRE_ENABLED":   {"value": false, "env_present": false, "source": "import"},
  "SCAN_REQUEST_ENABLED": {"value": true,  "env_present": true,  "source": "runtime"}
}}
```

Per flag:

- **`value`** — the module global when one exists (what the running process holds); otherwise `os.environ` evaluated now with the same truthiness parse, `in ("1","true","yes","on")`. Either way: **what the customer experiences.**
- **`env_present`** — whether the env var exists at all.
- **`source`** — `"import"` if a module global backs it, else `"runtime"`. Only an `"import"` flag can be stale-after-deploy, so only there does `env_present: true, value: false` mean "you forgot to redeploy."

This union has exactly the right blind spot. A **deleted call-time flag** has no global and no env key, so it **vanishes from the response** — and the checker's "absent from response" rule (§3) reports it as a failure. That is precisely the case we need to catch: `SCAN_REQUEST_ENABLED` deleted would otherwise be invisible.

A **deleted import-time flag** still has its global (holding `False`), so it appears with `env_present: false` — the deletion is named explicitly.

**Values are booleans only.** No flag value is ever echoed as a string, and no non-`*_ENABLED` env key is read. `CONSOLE_SECRET` and friends cannot leak through this endpoint.

Policy (which flags *matter*) stays in the checker, not the app.

### 2. `scripts/surface_check.py` (extend)

```python
REQUIRED_ON = ("FIRESIDE_ENABLED", "REPERTOIRE_ENABLED",
               "INVOICE_PAYLINK_ENABLED", "SCAN_REQUEST_ENABLED")

def check_flags(base_url, console_key, fetch=_fetch_json) -> list[dict]
```

Returns one dict per failing flag: `{"flag", "reason"}`. `run()` merges surface failures and flag failures into **one** email.

Stdlib only (the cron's `buildCommand` is `true`). No new schedule, no new credential: the daily cron remains the throttle, exactly as in #736.

### 3. Failure semantics

| observation | result |
|---|---|
| `value` is `true` | pass |
| `value` is `false`, `env_present` true | **fail** — "set to false" |
| `value` is `false`, `env_present` false | **fail** — "env var missing (deleted)" |
| flag absent from the response | **fail** — "constant removed from app.py while still in REQUIRED_ON" |
| endpoint 401 / 404 / connection error | **not** flag drift → `"could not check flags: <reason>"` |
| `CONSOLE_SECRET` unset in the cron | skip flag check, print a notice, do not fail |

The 401/unreachable carve-out is deliberate. The surfaces list **already** alarms when the app is down. One outage must not generate two contradictory stories ("app is down" + "all four flags are off").

### 4. Data flow

```
cron container (separate service, survives the web app being down)
  run_personal_email_cron.py  ->  surface_check.run()
        ├── check_surfaces(BASE_URL)            # HTTP status >= 400
        └── check_flags(BASE_URL, CONSOLE_SECRET)
                 GET /api/console/flags  (X-Console-Key)
                        └── app.py globals + os.environ
        └── one merged alert -> send_alert() -> SMTP
```

## Error handling

Best-effort by contract, inherited from #736: `check_public_surfaces()` never raises and never affects the personal-email send. `send_alert()` returns `False` rather than raising when SMTP is unconfigured. `check_flags()` never raises — every failure path becomes a reported string.

## Testing

Each with a watched RED. `check_flags()` is pure with an injected fetch.

1. All four `true` → no failures.
2. One `value: false`, `env_present: true` → one failure, reason names "set to false".
3. One `value: false`, `env_present: false` → one failure, reason names "missing"/"deleted" — **distinct from case 2**.
4. A `REQUIRED_ON` flag absent from the response → failure naming the flag.
5. Endpoint 401 → exactly one "could not check" entry, **not** four drift failures.
6. Connection error → "could not check", no crash.
7. Unwatched flags being `false` (e.g. `TWO_DOOR_ENABLED`) → **no** failure.
8. Endpoint: `@require_console_key` gate — no key → 401.
9. Endpoint: `value` / `env_present` split — monkeypatch a global to `False` and delete its env var; assert `value is False` **and** `env_present is False`. Then set the env var without changing the global; assert `env_present` flips to `True` while `value` stays `False` (the "never redeployed" case).
10. Endpoint: discovery finds a newly-added `FOO_ENABLED` global without registration.
10b. Endpoint: a call-time flag set only in `os.environ` appears with `source: "runtime"`, and its `value` tracks the env **without** a restart — set it mid-test, assert the value flips. An import flag under the same treatment must NOT flip (its global is authoritative).
10c. Endpoint: a call-time flag that is **deleted** from `os.environ` vanishes from the response entirely — and `check_flags()` then reports it as a failure via the absent-from-response rule. This is the `SCAN_REQUEST_ENABLED` deletion case.
10d. Endpoint: no name appears twice, and no key outside `^_?[A-Z][A-Z0-9_]*_ENABLED$` is ever returned (a `CONSOLE_SECRET` in the environment must not appear).
11. **Wiring:** `run()` includes flag failures in the alert body. A checker nothing calls is the dead-field pattern hit three times this week (`regular_cents`, `#738`'s `pickup_default`, `bundle_components`).

## Known limits

- Reports what the **process** holds. An **import-time** flag set correctly but never redeployed still reads `false` — correctly, since that is what customers experience. A green check therefore requires **both** the variable and a deploy. This is the gotcha that bit us twice. **Call-time** flags (`source: "runtime"`) are immune to it.
- **Daily cadence.** Drift is visible within 24 hours, not minutes. The cron is the throttle; a faster loop is a separate decision.
- Does **not** identify *who* deleted a flag. Render's events API does not log env changes. Attribution remains unsolved.

## Follow-ups (not this spec)

- Root-cause the deletions themselves. Two fireside drifts in one day; the second was a deletion, not a value change.
- `/admin/shipping` notes field, and the 731 unmapped products.
