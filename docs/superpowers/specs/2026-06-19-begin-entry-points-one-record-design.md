# Begin Page #3 — Secondary Entry Points into One Record + Meaningful Card Progress

**Date:** 2026-06-19
**Status:** Approved (design); ready for implementation plan
**Repo:** deploy-chat (Flask, Render `glen-knowledge-chat`, illtowell.com)
**Parent:** Begin-page redesign (5 sub-projects). This is **#3**, building on #1 (hero + one-record identity, PR #191) and #2 (4-card journey map, PR #192).

---

## Problem

Two of Glen's strongest entry points - the **ScoreApp quiz** and the **E4L voice scan** - capture people, but their completions never reach `journey_state` (the one record from #1). The ScoreApp webhook writes `inbound_leads` / `referral_events` / GHL; E4L pushes scan dates to `scan_freshness`. Neither writes the journey. So the #2 journey map cannot reflect real progress, and an entry-point visitor is not unified with their `/begin` journey.

Separately, #2's cards are binary (done / next / available). Several cards actually **consolidate multiple meaningful activities**, so progress should read as a bar **filling partway across the card** as the visitor activates each activity it leads to.

## Goal

1. **One record:** quiz and scan completions (and a few other milestones) write to `journey_state` by email, so every entry point unifies into one record and the map reflects real activity.
2. **Meaningful progress:** each card carries an ordered list of sub-steps; the card fills a colored bar to `done / total` and its link points to the next undone step's destination.
3. **Smart Scan door:** the Scan link routes a returning E4L user to `portal.e4l.com` and a new user to `truly.vip/E4L`.
4. Rename the fourth card **Earn -> Give**.

## Scope (#3)

Capture entry-point completions into `journey_state` (by email, via `record_unlock`); evolve `begin_funnel.journey_map` and the `begin.html` strip from binary status to a **fractional fill** driven by a per-card sub-step list; compute the Scan step's href dynamically; wire the sub-step completion signals we can detect today and define the rest. Existing `inbound_leads` / `referral_events` / GHL / `scan_freshness` / the #2 contextual surfacing stay as-is.

**Out of scope (deferred):**
- The **Find -> Biofield Analysis interpretation with the $1-trial reveal** (top match free, others blurred, unblur on the $1 trial). The sub-step is *defined* here (gate `biofield`) but the reveal flow is built in **#4** (Biofield interpretation) and ties to the Ascension $1 tripwire. Until #4, that step is "defined, dark."
- Threading `amg_session` through the external round-trip (fragile for ScoreApp, impossible for E4L's date-only feed). Email-union is the join.

---

## Confirmed decisions (Glen, 2026-06-19)

- Capture **completion + identity** into the one record (not the result payload - quiz score / scan priorities stay in GHL / e4l.db for now).
- The **ScoreApp quiz** lands in the record (gate `quiz`) for unification + chat reference, but **lights no card** ("a glow, not a card").
- **Smart Scan routing:** scanned-or-has-account -> `portal.e4l.com`; otherwise -> `truly.vip/E4L`.
- **Partial-fill progress:** cards fill as the visitor activates the consolidated activities. Per-card 2-step model:

| Card | Step 1 | Step 2 |
|---|---|---|
| **Scan** (Your Biofield) | Complete 1st voice scan | Complete the DIY **Wellness Whispering** course (free) |
| **Find** (Your Remedy Match) | Find a match through the **chat** | Find a remedy via **automated Biofield Analysis** of a scan (reveal top match; rest blurred until the $1 trial) - built in #4 |
| **Heal** (the root causes) | Complete **Intake** form | Complete the **ASH MasterClass** |
| **Give** (was Earn) | Be an **Ambassador** | **Bring a friend** |

- **Practice Better holds** the Wellness Whispering course, Intake, and ASH MasterClass. PB internal automations will POST completion events to our existing `/webhook/practice-better`; we map those to journey gates. (Glen configures the PB automations; we build the code side + the event->gate map.)
- **Live page, no flag** (same as #1/#2); manual visual pass before launch. No emoji, no em dashes.

---

## Architecture

### New journey gates (cheap; `unlocked_gates` is a JSON list, no migration)

Add to `begin_funnel.VALID_TRIGGERS`: `course_ww`, `intake`, `masterclass`, `biofield`. (Reused, already present: `scan`, `quiz`, `question`, `paid_fork`, `share_video`.) Two sub-steps are **predicate-derived** rather than stored gates (always accurate, no backfill): **Ambassador** (an approved `affiliate_signups` row for the email) and **Bring a friend** (a `referral_redemptions` row with `owner_email` = the email).

### Part A - entry-point completions into the one record

A deterministic synthetic session keeps email-only entries to one row that `get_state` then unions with the visitor's real `/begin` row by email.

- `_entry_session_id(email)` (app.py): returns `"entry:" + sha1(lower(email))[:16]`.
- `_record_entry_unlock(trigger, email, first_name="", last_name="", ref_slug="")` (app.py): inits journey tables; resolves the synthetic session; **reads that row's `unlocked_gates` and returns early if `trigger` is already present** (idempotent - no duplicate `journey_events`, safe for the 5000-row E4L cron); else calls `begin_funnel.record_unlock(cx, session_id=synthetic, trigger=trigger, email=email, first_name=..., last_name=..., ref_slug=...)`. Wrapped in try/except; never raises into the caller.

Wire-ins:
- `/webhook/scoreapp` (`scoreapp_webhook`, app.py ~11267), after the existing GHL/inbound/referral writes: `_record_entry_unlock("quiz", email, first, last, utm_source)`.
- `/api/e4l/scan-freshness` (`api_e4l_scan_freshness`, app.py ~9177), after the `scan_freshness` upsert: for each ingested row with a non-empty scan date, `_record_entry_unlock("scan", email)`. ("Has ever scanned" = the `scan` gate; freshness remains the Biofield gate's separate concern.)

### Part B - sub-step model + fractional fill (`begin_funnel.journey_map`)

`JOURNEY_STEPS` evolves: each card gains an ordered `steps` list. Each step is `{ "key", "label", "satisfied_by", "href" }` where `satisfied_by` is either `("gate", "<gatename>")` or `("predicate", "<name>")` (predicate booleans are computed by the route and passed in). The fourth card's `label` becomes **"Give"** (paren e.g. "lift others"); its destinations unchanged.

```
JOURNEY_STEPS = [
  { key:"scan", label:"Scan", paren:"Your Biofield", steps:[
      { key:"voice_scan",  label:"Voice scan",            satisfied_by:("gate","scan") },        # href smart (Part C)
      { key:"ww_course",   label:"Wellness Whispering",   satisfied_by:("gate","course_ww"),
        href:"https://truly.vip/GetWell" } ] },
  { key:"find", label:"Find", paren:"Your Remedy Match", steps:[
      { key:"match_chat",  label:"Match via chat",        satisfied_by:("gate","question"),  href:"/begin/match" },
      { key:"biofield",    label:"Biofield interpretation",satisfied_by:("gate","biofield"),  href:"/begin/match", deferred:true } ] },  # reveal flow = #4
  { key:"heal", label:"Heal", paren:"the root causes", steps:[
      { key:"intake",      label:"Intake form",           satisfied_by:("gate","intake"),    href:"https://truly.vip/Join" },
      { key:"masterclass", label:"ASH MasterClass",       satisfied_by:("gate","masterclass"),href:"https://truly.vip/Intro" } ] },
  { key:"give", label:"Give", paren:"lift others", steps:[
      { key:"ambassador",  label:"Be an Ambassador",      satisfied_by:("predicate","ambassador"),    href:"/affiliate/apply" },
      { key:"bring_friend",label:"Bring a friend",        satisfied_by:("predicate","referred_friend"),href:"/begin/path" } ] },
]
```

`journey_map(state, ref, signals=None)` returns, per card:
- `steps`: each `{ key, label, done }` (for segment rendering / tooltip).
- `fill`: `done_count / len(steps)` (0.0 - 1.0).
- `status`: `"done"` when `fill == 1.0`; otherwise `"next"` for the FIRST card with `fill < 1.0`, `"available"` for the rest (exactly one "next" unless all done - same rule as #2, now keyed on fill).
- `href`: the destination of the **first undone step** (so the link advances them); for the Scan card's first step this is the **smart-routed** URL (Part C). A deferred-but-undone step (biofield) still contributes to "not done" but its href can fall back to its `href` field.

`satisfied_by ("gate", g)` checks `g in state["unlocked_gates"]`. `("predicate", p)` checks `signals[p]`. `signals` is a small dict the route computes: `{ "ambassador": bool, "referred_friend": bool, "has_e4l": bool }`. When `signals` is None (pure-unit-test calls), predicate steps are treated as not-done and `has_e4l` False.

### Part C - smart Scan routing

The Scan card's first-step (`voice_scan`) href is computed from `signals["has_e4l"]`:
- `has_e4l` true (the email has the `scan` gate OR a `scan_freshness` row OR - later - an E4L account feed) -> `https://portal.e4l.com`.
- else -> `https://truly.vip/E4L`.

External hrefs thread the ref-utm via the existing `_thread_href` (campaign `begin-journey-scan`); internal hrefs pass through. `has_e4l` is computed in the route via a `_has_e4l(cx, email)` helper (reads the `scan` gate from the unioned state and/or `scan_freshness`).

### Part D - signal wiring

- `scan` <- E4L ingest (Part A). `quiz` <- ScoreApp (Part A, no card).
- `question` <- already fired by the #2 Find-card click and the match flow.
- `course_ww` / `intake` / `masterclass` <- `/webhook/practice-better` (`pb_webhook`, app.py ~10590) extended with a configurable map `PB_EVENT_GATES = { "<pb event_type or course/form id>": "<gate>" }`. On a recognized completion event with an email, call `_record_entry_unlock("<gate>", pb_email, first, last)`. The exact PB event identifiers are filled in once Glen configures the PB automations; the map lives in one place and is easy to edit. The existing `pb_events` log + GHL signup path are untouched.
- `ambassador` <- predicate: `affiliate_signups` row for the email with `status='approved'`.
- `referred_friend` <- predicate: a `referral_redemptions` row with `owner_email` = the email.
- `biofield` <- deferred to #4 (the $1-trial reveal sets it).

### Part E - rendering (`static/begin.html`)

The #2 strip's `renderJourney()` evolves: each card draws a fill bar to `card.fill` (a colored inner element width = `fill * 100%`), keeps the done/next/available classes (now derived from fill), and still links to `card.href` and fires its click trigger. Build all card text via `textContent` (XSS-safe, as #2). The unfold, triggers, and the three state-refresh wirings from #2 are unchanged. Optional: a thin per-step segment underlay (2 segments) so the fill reads as "step 1 of 2 done." Earn -> Give label updates here and in `JOURNEY_STEPS`.

### Reuse / untouched
- One record: `record_unlock` / `get_state` (email-union) / `journey_state` - the backbone.
- Webhooks: `scoreapp_webhook`, `pb_webhook`, `api_e4l_scan_freshness` - extended additively, always return 200.
- Predicate sources: `affiliate_signups`, `dashboard/referrals.referral_redemptions`, `scan_freshness` - read-only.
- Untouched: `inbound_leads`, `referral_events`, GHL onboarding, `surface()` / `/begin/explore`, `/begin/match/chat`.

---

## Data flow

1. Cold visitor takes the E4L scan -> `/api/e4l/scan-freshness` ingest -> `_record_entry_unlock("scan", email)` -> synthetic row (email, `scan` gate).
2. They later land on `/begin`, chat, activate with the same email -> `get_state` unions the synthetic row -> Scan card shows step 1 done (50% fill); its link now routes to `portal.e4l.com` (has_e4l).
3. They finish the Wellness Whispering course in PB -> PB automation -> `/webhook/practice-better` -> `_record_entry_unlock("course_ww", email)` -> Scan card fills to 100% (done).
4. Their referred friend redeems a code -> `referral_redemptions(owner_email=them)` -> Give card's "Bring a friend" predicate true -> that step fills.
5. The map re-colors on each `/begin/state` refresh (the three sites #2 wired).

## Error handling

- Every webhook wire-in is wrapped and never changes the webhook's 200 response or existing behavior.
- `_record_entry_unlock` is idempotent (skips an already-present gate) - safe for the high-volume E4L cron and repeat webhooks.
- `journey_map` with `signals=None` degrades to gate-only (predicate steps not-done, Scan routes to signup) - keeps unit tests pure and the page functional if the route's signal computation fails.
- Unknown PB event types are ignored (logged to `pb_events` as today); only mapped events set a gate.
- A deferred step (`biofield`) renders as an undone segment until #4 ships; it never blocks the card from showing its earned fill.

## Testing

- **Part A:** `scoreapp_webhook` QUIZ_FINISHED with email -> `journey_state` synthetic row carries `quiz` (+ name); `get_state(email=)` shows it; union with a real session sharing the email merges. `api_e4l_scan_freshness` ingest -> `scan` gate by email; idempotent (re-ingest same email does not add a second event); webhook still returns 200 if `record_unlock` raises (monkeypatch it to raise).
- **`journey_map` fill/status:** no gates/signals -> Scan is "next", fill 0; `scan` set -> Scan fill 0.5; `scan`+`course_ww` -> Scan done (1.0), Find "next"; predicate `ambassador` true -> Give step 1 done; all steps across all cards done -> all fill 1.0, none "next". Non-contiguous (only `masterclass`) -> Heal fill 0.5, Scan still "next". `href` of a card = its first undone step's destination.
- **Smart routing:** `has_e4l` true -> Scan first-step href `portal.e4l.com` (ref-threaded); false -> `truly.vip/E4L`.
- **PB wiring:** a mapped `pb_webhook` event with email sets its gate via `_record_entry_unlock`; an unmapped event sets none; both return 200.
- **Predicates:** approved `affiliate_signups` -> ambassador true; `referral_redemptions.owner_email` present -> referred_friend true.
- **Serve:** `/begin/state` `journey_map` carries `fill` and `steps` per card; `/begin` renders the fill bar and the Give label.
- Front-end fill animation / segment visuals / smart-link visuals = manual visual pass (state it). Server logic has unit + Flask-test-client coverage. deploy-chat test isolation (tmp `LOG_DB`, `init_journey_tables`, mock GHL on free-tier transition). No emoji; no em dashes.

## Suggested build order (the plan may stage these)

1. **Increment 1 - one record + rename + smart door:** Part A (quiz/scan capture), Earn->Give label, smart Scan routing, and the fill model lighting the steps detectable today (`scan`, `question`, ambassador, referred_friend). All real signals; immediately useful.
2. **Increment 2 - PB completions:** extend `pb_webhook` + `PB_EVENT_GATES` for `course_ww` / `intake` / `masterclass`; Glen configures the PB automations. Lights the remaining Scan/Heal steps.
3. **Find step 2 (`biofield`)** lands with #4 (the $1-trial reveal).

## Notes

- **Live page, no flag.** `main` auto-deploys; the merge ships it. Visual pass required (the fill bars, the Give label, the smart Scan link).
- All copy provisional (labels, parens, captions) - BNSN site pass later. Keep strings in `JOURNEY_STEPS` + a few JS constants.
- New gates are additive to `VALID_TRIGGERS`; no `journey_state` schema change. Predicate steps avoid any backfill.
- This deliberately distinct fixed-journey map stays separate from the data-driven `surface()` contextual cards.
