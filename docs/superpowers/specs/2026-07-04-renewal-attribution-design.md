# Renewal Attribution — Sticky Continuity Across Term Renewals (v1) — Design

**Date:** 2026-07-04
**Status:** Approved (brainstormed with Glen 2026-07-04)
**Repo:** deploy-chat
**Fast-follow of:** #565 (monthly attributed enrollment + fee-share), #575 (attributed prepay-term enrollment), #572 (continuity tooling C). Touches the two Continuous Care fulfillment paths only.

## Summary

Today a Continuous Care term is attributed to a doctor only when the enrollment session carries an explicit `dispensary_pid` (the patient came through the doctor's dispensary link). But there is **no dedicated renewal flow** — a renewal is just the patient re-purchasing a term through the normal checkout (the loyalty rate is prompt copy only, `prepay.renewal_price_cents`). So when a patient renews any way *other* than clicking back through their doctor's link (a reminder email, the public checkout), the new term is **unattributed** — the doctor silently loses the patient and the fee-share on renewal. That breaks "retention is the job."

This makes attribution **sticky**: a renewed term inherits the patient's most-recent attributed doctor (and consent), so a doctor keeps their patient — and the fee-share + continuity-tooling visibility — across renewals, without the patient having to re-navigate the dispensary.

## The mechanic — inherit at fulfillment

Both Continuous Care fulfillment paths resolve the attributed doctor the same way, at the moment a **new term** is fulfilled:

```
resolved_pid, resolved_consent =
    (explicit dispensary_pid + explicit share_consent)   if the session carries dispensary_pid
    else (inherited pid + inherited consent)             if _last_attributed_practitioner(email) exists
    else (None, 0)                                        # truly new / direct / no history → unattributed
```

Once resolved, the **existing** stamp + fee-share logic fires unchanged (prepay: stamp the grant + `earn_care_share` on the lump; monthly: `create_membership(attributed_practitioner_id=...)` + the enrollment-charge credit). So a renewed term is attributed and credited to whichever doctor most recently held the patient.

- **Applies to both** `_fulfill_prepay_term` (prepay-lump renewals) and `_fulfill_continuous_care_monthly` (monthly re-enrollments).
- **NOT the cron:** inheritance happens only at a *new term's* fulfillment. An existing monthly membership the sub-charge cron keeps billing already carries its attribution — nothing to re-inherit.

## The lookup — one shared helper

`_last_attributed_practitioner(cx, email) -> {"pid": str, "consent": int} | None` (new, in `app.py`):
- The patient's **most-recent attributed record across BOTH sources**:
  - attributed `prepay_term_grants` (rows with a non-null `attributed_practitioner_id`), ordered by `granted_at`;
  - attributed `subscriptions` memberships (`attributed_practitioner_id` non-null, `kind='membership'`), ordered by `created_at`.
- Returns the single most-recent one's `attributed_practitioner_id` + its `practitioner_share_consent`. **Sticky / most-recent-wins forever** — no window. If a patient was attributed to A then later to B, a direct renewal inherits **B**.
- Table-guarded (a missing `prepay_term_grants`/column → treat as no prepay history) and returns `None` when the patient has no attributed history.

## Precedence + consent

- **Explicit `dispensary_pid` wins.** A patient who re-enrolls through a doctor's dispensary link (even a *different* doctor's) uses that explicit pid + its explicit consent — this is a deliberate (re)choice and matches "most-recent."
- **Else inherit.** The prior record's pid **and its `practitioner_share_consent`** carry onto the renewed term, so continuity-tooling access persists across renewals without re-consenting each term.
- **Else unattributed** (no prior + no explicit) → no credit; correct for a genuinely new direct patient.

## Components / files

- **`app.py`** — new `_last_attributed_practitioner(cx, email)` helper; both `_fulfill_prepay_term` and `_fulfill_continuous_care_monthly` change their attribution resolution from "read `dispensary_pid` from metadata" to "explicit `dispensary_pid` OR inherited" (pid + consent), then feed the existing stamp/credit logic.

No schema changes (reuses #575's `prepay_term_grants` attribution columns + #565's `subscriptions.attributed_practitioner_id`/`practitioner_share_consent`). No new tables. Reuses `care_share`/`wallet`.

## Testing

- **Prepay renewal inherits:** a patient with a prior attributed prepay grant, a NEW prepay-term fulfillment with **no** `dispensary_pid` → the new grant is stamped with the inherited pid + consent and the doctor is credited on the lump.
- **Monthly re-enrollment inherits:** a previously-attributed patient re-enrolls monthly with no `dispensary_pid` → the membership is attributed to the inherited doctor + the enrollment charge credits them.
- **Explicit wins:** a renewal WITH `dispensary_pid` (a different doctor) → uses the explicit pid, not the inherited one.
- **No prior + no explicit → unattributed:** a genuinely new direct term → no attribution, no credit (public flow unchanged).
- **Most-recent-wins:** patient attributed to A, then later to B → a direct renewal inherits B.
- **Cross-source recency:** an earlier attributed prepay grant vs a later attributed membership (or vice-versa) → inherits the chronologically latest.
- **Consent inherited:** prior `practitioner_share_consent=1` → renewed term is consented (C gate/roster include the renewed patient); prior `=0` → renewed term stays unconsented.
- **Helper isolation:** `_last_attributed_practitioner` returns None for a patient with no attributed history; table-guarded against a missing `prepay_term_grants`.

## Out of scope / Future

- **Windowed expiry** (inherit only within N days of the prior term end): deferred — v1 is sticky/most-recent-wins forever. The rare "lapsed a year, returned entirely on their own" case still credits the old doctor; low-stakes, can be layered on later if it matters.
- Any change to the public prepay/monthly checkout for genuinely-new direct buyers (they stay unattributed).
