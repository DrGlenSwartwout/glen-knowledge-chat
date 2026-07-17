# Unified Household + Caregiver Overlay — Address-Suggested Connections

**Date:** 2026-07-17
**Status:** Design — awaiting review
**Builds on:** `2026-05-26-household-tag-system-design.md` (CRM households), `2026-07-04-household-sharing-design.md` (caregiver links), `2026-07-10-dependent-tos-coverage-design.md` (dependent TOS), `2026-07-01-household-combined-shipment-design.md`

---

## Problem

Two clients at the same address should be easy to connect, and the connection should be suggested automatically. Today:

1. **Detection exists but is coarse.** `detect_household_candidates()` has a `shared-address-lastname` signal keyed on `city + state + last_name`. It misses a caregiver with a *different surname* (the whole point of a caregiver link) and matches unrelated same-city same-surname strangers.
2. **The two "household" systems are disjoint, and the connect flow only touches one.** The CRM household (`households`/`household_members` people-grouping + GHL tags) is symmetric grouping. The caregiver link (`dashboard/household.py` `household_members` email-pairs) is directional portal/TOS/CC. Confirming a candidate today creates only the CRM grouping.
3. **The real-world case breaks the either/or.** Two adult partners, different surnames, one is the caregiver, want **both** links at once: they are a household **and** one is caregiver for the other. A pet is also both. "Caregiver" is not a sibling of "household member" — it is a **directional overlay on top of** household membership.
4. **A capacitated adult owns their own consent.** For a dependent (pet/minor/incapacitated), the caregiver supplies consent. For a competent adult partner, the caregiver cannot consent on their behalf — routing their health reports to a partner requires *the adult's own* consent.

## Goal (this slice)

On the **CRM person card**, surface same-street-address people as suggested connections, with a control that lets an operator connect them as a household member and, optionally, layer a caregiver direction — with consent handling that is correct for both dependents and capacitated adults. Ship the CRM card surface first; other surfaces (intake, order review) come later.

---

## Core model

**Household membership is the base link. "Caregiver" is a directional overlay carrying a relationship word. The relationship word's membership in `DEPENDENT_RELATIONSHIPS` decides the consent behavior — no new subsystem.**

- **Connect as household member** → CRM household grouping (existing `create_household` / `add_household_member`). Symmetric.
- **Connect as caregiver** → *also* a household member (the overlay implies membership) **plus** a directional `household_members` row (`dashboard/household.py`), `primary_email = caregiver`, `member_email = cared-for`, with a relationship word.
- **Don't connect** → dismiss the pair so it stops surfacing.

Glen's original "triple" (member / caregiver / don't-connect) is preserved as the three visible intents; the correction is that choosing **caregiver** creates household membership too, then expands for direction + relationship + consent.

### Two consent classes, driven by the relationship word

`dashboard/household.py` already defines the gate:

```python
DEPENDENT_RELATIONSHIPS = {"child", "pet", "dependent", "charge", "caregiving-client"}
```

- **Dependent class** (word ∈ set): caregiver's own Terms agreement covers the dependent (the existing `DEPENDENT_TOS_ENABLED` path). `share_consent = 1`, `cc_enabled = 1` by default. No portal confirmation needed — the caregiver *is* the consenting authority.
- **Operational class** (word ∉ set — new words: `partner`, `spouse`, `manages-account`): caregiver gets view/CC and can act on behalf, but **never** substitutes consent. TOS coverage never applies. Sharing stays **dark until the cared-for adult's own consent is captured** (option C, below).

Adding an operational word requires **no logic change** to the consent path *provided the two guards below are in place* — the behavior falls out of set-membership.

---

## Two required guards (correctness-critical)

These close a latent hole: without them, adding an operational caregiver link would silently grant a competent adult TOS coverage and view access they never consented to.

### Guard 1 — TOS coverage must be dependent-relationship-only

`_portal_tos_agreed()` (app.py) loops `caregivers_for()` and grants coverage on `share_consent AND is_member(caregiver)` with **no relationship filter**. Today that is safe only because dependents are the sole link type. It must filter to dependent relationships:

- `dashboard.household.caregivers_for()` must return `relationship` alongside `primary_email` and `share_consent`.
- The coverage loop adds `and _hh.is_dependent(cg["relationship"])`.

Result: an operational caregiver link never grants TOS coverage, even with `share_consent = 1`. Mutation-test this: flip the filter off, watch a "capacitated-adult inherits coverage" test go red.

### Guard 2 — operational links do not default to consented

`add_member()` hard-codes `share_consent = 1`. View (`can_view`), CC (`cc_recipients_for`), and viewable lists all gate on `share_consent = 1`, so a default-1 operational link would expose the adult's portal the instant it is created.

- `add_member()` becomes consent-class-aware: dependent word → `share_consent = 1` (unchanged); operational word → `share_consent = 0` (pending) unless an explicit consent basis is recorded at creation.
- Existing dependent behavior is byte-identical.

---

## Option C — hybrid consent capture (operational links only)

New columns on `household_members` (`dashboard/household.py`), additive:

- `consent_basis TEXT` — one of `caregiver-authority` (dependents), `verbal`, `written`, `portal-confirmed`, or `''` (pending).
- `consent_recorded_by TEXT` — operator who recorded it.
- `consent_confirmed_at TEXT` — set only on a hard (portal) confirmation.

Creation paths for an operational caregiver overlay (operator chooses in the UI):

1. **Send for portal confirmation** (default, safest): row created `share_consent = 0`, `consent_basis = ''`. Nothing shares until the cared-for adult confirms in their own portal ("Share my reports with {name}?"). Confirmation sets `share_consent = 1`, `consent_basis = 'portal-confirmed'`, `consent_confirmed_at = now`.
2. **Record a basis now** (verbal / written): row created `share_consent = 1`, `consent_basis = 'verbal'|'written'`, `consent_confirmed_at = NULL`. CC/view active but the console flags the link **"consent unconfirmed"** until a later portal confirmation upgrades it.

Dependent overlays skip this entirely: `share_consent = 1`, `consent_basis = 'caregiver-authority'`, no confirmation step.

Portal confirmation upgrade is idempotent and never downgrades an already-confirmed link.

---

## Detection: new street-address signal

Add signal `shared-street-address` to `detect_household_candidates()`:

- Cluster people by normalized `(address1, zip)` — lowercase, trim, collapse internal whitespace, strip trailing punctuation on `address1`.
- Fire only when **both** `address1` and `zip` are non-empty and `address1` is substantive (≥ 4 chars). Sparse rows (no street on file) never match.
- **Does not require matching last name** — this is what catches the different-surname caregiver.
- Existing `_candidate_dedup_key` + `skipped_already_household` logic prevents overlap with the other three signals and with existing households.

The coarse `shared-address-lastname` signal stays (it still adds value for same-surname families with no street on file).

---

## API

- `GET /api/people/<id>/household-suggestions` — returns other people at the same street address, each annotated with current state: `already_in_household_together`, `existing_caregiver_link` (+ direction/relationship), `dismissed`. The card only offers genuine, un-actioned connections.
- `POST /api/people/<id>/connect` — routes the chosen intent:
  - `mode = "member"` → new CRM household or `add_household_member`.
  - `mode = "caregiver"` → CRM membership **plus** directional `add_member(caregiver_email, cared_for_email, relationship, label)`. Body carries `caregiver_person_id`, `cared_for_person_id`, `relationship`, and (operational only) `consent = {"method": "portal" | "verbal" | "written"}`.
  - `mode = "dismiss"` → record the pair dismissed.

Auth: `_check_console_or_scoped_auth` (matches sibling household routes). All writes under `_db_lock`.

### CRM household naming with differing surnames

Auto-name can no longer assume one surname. For a 2-person household with differing last names: `"{head_last} / {other_last} Household"`; identical surnames keep `"{last} Household"`. Operator-editable on the household page. When a caregiver overlay is set, the **household head defaults to the caregiver**.

---

## Frontend — CRM person card ("Possible connections")

One row per same-address suggestion. Top-level intent (honoring the triple):

- **○ Household member** — connect as peers.
- **○ Caregiver** — expands:
  - Direction: "This person cares for {other}" / "{other} cares for this person".
  - Relationship dropdown, grouped: *Dependent* (child, dependent, charge, caregiving-client, pet) vs *Operational* (partner, spouse, manages-account), with a one-line note on the consent difference.
  - If operational: consent sub-control — **Send portal confirmation** (default) or **Record basis** (verbal / written).
- **○ Don't connect** — dismiss.

Confirm → `POST /connect`, refresh the card. Existing links render read-only with their relationship, direction, and a **"consent unconfirmed"** badge where applicable.

---

## Testing

- **Signal:** two people, differing surnames, same `address1 + zip` → one `shared-street-address` candidate; empty street → none; already-in-household → skipped; dedup vs other signals.
- **Guard 1 (mutation-tested):** operational caregiver link + `DEPENDENT_TOS_ENABLED` on → cared-for adult `tos_agreed = false`; dependent link → `true`. Removing the relationship filter makes the operational case go red.
- **Guard 2:** operational link defaults `share_consent = 0`; `can_view`/`cc_recipients_for` return nothing until consent captured. Dependent link unchanged (`share_consent = 1`, cc on).
- **Option C:** portal-confirm path flips `share_consent`→1 + stamps `consent_confirmed_at`; verbal/written path is active-but-flagged; confirmation is idempotent and never downgrades.
- **Connect routing:** `member` creates grouping only; `caregiver` creates grouping **and** directional link with correct direction; `dismiss` stops the suggestion.
- **Naming:** differing surnames → "A / B Household"; head defaults to caregiver on overlay.

---

## Out of scope (later slices)

- Intake and order-review surfaces for the same control.
- The cared-for adult's portal-side "share my reports with {name}?" confirmation screen (this slice creates the pending state and the upgrade endpoint; the portal UI that calls it is a follow-up).
- Bulk review of the candidate queue (the existing `/api/household-candidates` queue remains as-is).

## Flags / safety

- Dependent TOS coverage remains dark behind `DEPENDENT_TOS_ENABLED` (Glen + counsel). This design must be *correct when it flips* — hence Guard 1.
- No auto-connect: address match only ever *suggests*; an operator confirms. (False-positive linking = privacy exposure; identity-merge review discipline applies.)
