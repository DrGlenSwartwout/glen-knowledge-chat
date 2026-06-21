# Begin #5 - Personalized Ascend (high-ticket escalation)

**Date:** 2026-06-20
**Status:** Approved (design); ready for implementation plan
**Repo:** deploy-chat (Flask, Render `glen-knowledge-chat`, illtowell.com)
**Parent:** Begin-page redesign, the 5th and final sub-project. #1 hero/identity, #2 journey map, #3 entry points -> one record, #4 Match/ordering + Biofield reveal/cart are all live. #5 brings the high-ticket Ascend ladder into the redesigned, one-record funnel.

---

## Problem

A static high-ticket Ascend ladder already exists: `/begin/ascend` ("Your Path, Your Depth") lists 6 rungs from `biofield-analysis` ($300) to `consultant-package` ($50K) from `begin_funnel.TIER_CATALOG`, each a card to a per-tier page with a "Book your consultation" CTA. It predates the redesign and is an orphan: the redesigned funnel (4-card journey map + one-record identity + the Biofield reveal/cart) never connects to it, it is not personalized to the member's record, and "Book your consultation" leads nowhere actionable. So a flat price list is shown to everyone, and high-ticket interest is not captured.

## Goal

Make `/begin/ascend` a personalized, consultative escalation: a short "where do you want to go?" question picks a track, the member's one-record state picks the rung within it, and that rung is presented as the hero (the full ladder still shown below). "Book your consultation" becomes a real, record-tied capture-and-notify hand-off. Behind `ASCEND_PERSONALIZED_ENABLED` (default off) so the live static ladder is untouched until flipped.

## Scope (#5)

The goal chooser, the `recommend_ascend` track->rung function, the recommend data endpoint, the personalized render of `/begin/ascend`, and the consultation-inquiry capture (`/begin/ascend/inquire`) with GHL tag + Glen notification + a one-record gate - all behind `ASCEND_PERSONALIZED_ENABLED`.

**Out of scope:** any self-serve checkout / charge for the high-ticket tiers (they sell via consultation, not a card - this is capture-and-notify only); a calendar/scheduling integration (the hand-off is an inquiry that Glen/Rae follow up on); changing `TIER_CATALOG` prices or copy (BNSN pass later); adding a 5th journey-map card (the 4-card map stays; Ascend is its own page reached by the existing want=ascend / paid_fork links plus a light member entry point); the conversational `/begin/match` chat.

---

## Confirmed decisions (Glen, 2026-06-20)

- **Core job: personalized escalation tied to the one record**, with a light capture/booking hand-off (option 1 + a light option 2).
- **Goal-driven (consultative):** a short "where do you want to go?" question picks the track; journey state picks the rung within it; that rung is the hero, the rest of the ladder below.
- **Track -> rung mapping (confirmed):**
  - **"Heal myself deeper"** -> `biofield-analysis` ($300). (Ongoing self-healing continuity is the $99/mo membership from #4b; Ascend's heal rung is this one-time deep causal consult.)
  - **"Learn the method"** -> `certification` (~$3,600), deepening to `one-to-one` (~$8,500).
  - **"Build my own practice"** -> `one-to-one` (~$8,500) -> `healing-oasis-tools` (~$14K) -> `hawaii-immersion` (~$25K) -> `consultant-package` (~$50K).
- Within a track, recommend the lowest rung the member's state shows they have NOT reached; default to the track entry rung when there is no signal. Full ladder always shown below the hero (nothing hidden).
- **Booking = capture-and-notify** (consultation inquiry), not a charge. Member-gated (ToS).
- Behind `ASCEND_PERSONALIZED_ENABLED` (default off). No emoji, no em dashes.

---

## Architecture

### 1. Flag
`ASCEND_PERSONALIZED_ENABLED = os.environ.get("ASCEND_PERSONALIZED_ENABLED", "").lower() in (...)`. When OFF, `/begin/ascend` serves the existing static ladder unchanged (no regression) and `/begin/ascend/inquire` returns `{ok:false}`. When ON, the page renders the goal chooser + personalized hero + capture.

### 2. The recommendation - `begin_funnel.recommend_ascend(state, goal, signals=None)` (pure)
Track ladders (slugs into `TIER_CATALOG`):
```
ASCEND_TRACKS = {
  "heal":  ["biofield-analysis"],
  "learn": ["certification", "one-to-one"],
  "build": ["one-to-one", "healing-oasis-tools", "hawaii-immersion", "consultant-package"],
}
```
`signals` (derived read-only, see endpoint) marks which rungs are already "reached": e.g. a paid member / biofield gate -> `biofield-analysis` reached; a `certified` role tag -> `certification` reached; a `practitioner` role tag -> `one-to-one` reached. `recommend_ascend` returns the first rung in the goal's track NOT in `reached`; if all are reached, the track's top rung; unknown/missing goal -> `heal` (the entry track). Pure and total - never raises, always returns a valid `TIER_CATALOG` slug.

### 3. Recommend data - `GET /begin/ascend/recommend?goal=<heal|learn|build>`
Resolve the member (session cookie + email via the journey one-record), build `signals` from read-only sources (membership via `_active_membership_for_email`, the `biofield`/`paid_fork` gates in `journey_state`, role tags from the People record), call `recommend_ascend`, and return `{ok, enabled, goal, recommended: <TIER_CATALOG entry>, ladder: [<all TIER_CATALOG entries, ordered by n>], is_member}`. When the flag is off -> `{ok:true, enabled:false}` (the page falls back to static). Never raises.

### 4. Personalized render - `static/begin-ascend.html`
When `enabled`: render the goal chooser (3 buttons: Heal myself deeper / Learn the method / Build my own practice; remembered in the cookie/localStorage), the **recommended rung as a hero card** (title, price, value, included, a "Book your consultation" button), and the **full ladder below** (the existing card list). Switching goal re-fetches `recommend` and re-renders the hero. When not enabled: the existing static ladder (current behavior). XSS-safe (textContent/setAttribute; no innerHTML of dynamic data).

### 5. Capture-and-notify - `POST /begin/ascend/inquire`
Body `{slug, goal, note?}`. Guard `ASCEND_PERSONALIZED_ENABLED` (else `{ok:false}`). Validate `slug in TIER_CATALOG`. Resolve email from the one-record/session. Membership-gate: `if not is_member(amg_session, email): return {ok:false, need_optin:true, error:"..."}, 403` (the existing OptinGate pattern). Then, best-effort (never 500):
- Record an `ascend_inquiries` row `(email, slug, goal, note, created_at)` (one record; idempotent upsert per `(email, slug)` so a repeat updates rather than duplicates).
- Tag in GHL: `ghl_onboard_contact(email, extra_tags=[f"ascend:inquiry:{slug}"], source_tag="ascend")` (best-effort, wrapped).
- Notify Glen: an SMTP email (reuse the existing SMTP send path) summarizing who + which rung + goal + note (best-effort, wrapped).
- One-record gate: `_record_entry_unlock("ascend", email)` so the journey reflects the escalation (idempotent, wrapped).
Return `{ok:true}`; the page shows a calm confirmation ("Dr. Glen and Rae will reach out to book your consultation"). A failure in any best-effort step logs and still returns `{ok:true}` if the inquiry row was written; if even the row write fails, `{ok:false}` with a friendly retry message.

### Reuse / untouched
- `TIER_CATALOG`, `/begin/ascend`, `/begin/ascend/<slug>`, `/begin/ascend-tier`, `begin-ascend-tier.html` (the per-tier page) - reused; the static ladder is the flag-off fallback.
- `is_member`, the OptinGate pattern, `_record_entry_unlock`, `ghl_onboard_contact`, the SMTP send path, `journey_state` / the one-record resolve, `_active_membership_for_email` - reused.
- Untouched: the 4-card journey map, the Biofield reveal/cart (#4), the pricing/billing engine, the $1 trial.

---

## Data flow
1. `ASCEND_PERSONALIZED_ENABLED` on. Member opens `/begin/ascend`; the page reads `recommend?goal=<remembered or heal>` -> renders the hero + ladder.
2. Member switches goal -> re-fetch `recommend` -> new hero.
3. Member clicks "Book your consultation" on the hero (or any ladder rung) -> `inquire {slug, goal, note?}` -> (member-gated) records the inquiry + GHL tag + Glen email + the `ascend` gate -> confirmation.
4. Glen/Rae follow up to book the consultation (outside the app).

## Error handling
- Flag off OR Stripe-irrelevant: the page is the static ladder; `recommend` -> `{enabled:false}`; `inquire` -> `{ok:false}`. No new behavior.
- Non-member -> 403 `{need_optin:true}` (OptinGate).
- Unknown `slug`/`goal` -> `recommend` falls back (goal -> heal); `inquire` with an unknown slug -> 400.
- All outward steps (GHL, email) are best-effort and wrapped; the inquiry succeeds as long as the row is written; the endpoint never 500s except a final catch.
- `recommend_ascend` is pure/total - any bad input yields the heal entry rung, never an exception.

## Testing
`tests/test_begin_ascend.py`:
- **recommend_ascend:** heal -> biofield-analysis; learn (no signals) -> certification; learn with `certified` -> one-to-one; build (no signals) -> one-to-one; build with `practitioner` -> healing-oasis-tools; unknown goal -> biofield-analysis; all-reached -> track top. Pure, no I/O.
- **recommend endpoint:** flag off -> `{enabled:false}`; flag on -> recommended entry + full ordered ladder + `is_member`.
- **inquire:** flag off -> `{ok:false}`; non-member -> 403 `need_optin`; member -> records an `ascend_inquiries` row + returns `{ok:true}` (mock `ghl_onboard_contact` + the SMTP send; assert they were called best-effort and that a GHL/email failure still returns `{ok:true}`); unknown slug -> 400; repeat inquire for the same `(email, slug)` -> single row (idempotent).
- Serve: `/begin/ascend` with the flag on ships the goal-chooser markers + the recommend/inquire endpoints; flag off ships the static ladder.
- GHL / SMTP / membership mocked; tmp `LOG_DB`; no emoji; no em dashes. Front-end (goal toggle, hero, inquiry form, confirmation) = manual visual pass.

## Notes
- **Behind `ASCEND_PERSONALIZED_ENABLED` (default off).** Merge is dark; go-live = flip the flag in Doppler. Not a money path (no charge); the only outward effects are a GHL tag + a Glen email, both best-effort and member-gated.
- All copy (the goal labels, the hero, the confirmation) is provisional - BNSN pass later.
- A light member entry point INTO `/begin/ascend` from the funnel (e.g., a CTA once a member / once Heal completes) can be a small follow-on; #5's deliverable is the personalized Ascend page + capture, reachable today via the existing `want=ascend` / `paid_fork` links.
- Ties to [[project_ascension_pricing_model]] (the high-ticket rungs above the $99/mo tier) and the one-record journey.
