# Dependent TOS coverage — design

**Date:** 2026-07-10
**Status:** design, ready for a plan
**Repo:** deploy-chat (Flask, Render service `glen-knowledge-chat`)
**Builds on:** household member-view (#584/#750), the portal TOS gate

---

## Problem, and it is live today

A portal is gated behind a Terms-of-Service acceptance: until the token holder has agreed,
the page renders a "please accept our Terms" wall and nothing else. `tos_agreed` in the
portal payload is computed as `is_member(email=primary_email)`, where `primary_email` is the
**token holder's own email** (captured at `app.py:15600`, before any `?member=` re-point).

A dependent — a pet, an infant, a minor, an incapacitated adult in someone's care — has their
own portal, their own login, and their own token. That token is handed to the caregiver: when
a Biofield report is published for a pet, the caregiver is emailed the **pet's own** portal
link (this is how Karin received Sasha's link).

When that link is opened, `primary_email` is the **dependent's** email, so `tos_agreed` asks
"has this cat agreed to our Terms?" — and the answer is permanently no, because the only thing
that would flip it is the cat clicking "I agree." The portal is a blank Terms wall forever.

**Verified live (2026-07-10):** `GET /api/portal/<sasha_token>` returns `tos_agreed: false`.
Sasha is a cat.

A TOS is a legal agreement between parties. An animal cannot be a party; neither can an
infant. Asking them to agree is not merely awkward — gating their portal on their own
impossible agreement is a bug.

## The principle

**The caregiver agrees, and that one agreement covers the dependents in their care.** This is
not animal-specific: a pet, a minor, and an infant are all *dependents*, and in every case the
guardian is the party who can agree. The codebase already names this set exactly —
`household.DEPENDENT_RELATIONSHIPS = {"child", "pet", "dependent", "charge", "caregiving-client"}`.

Doing it *properly* has two halves. Only one is code.

## Two halves

### 1. Code — the gate derives from the caregiver (render-time)

`tos_agreed` for a portal becomes:

> the token holder has agreed, **OR** the token belongs to a dependent whose linked, consented
> caregiver has agreed.

Expressed against helpers that already exist:

```python
def _portal_tos_agreed(cx, primary_email):
    if is_member(email=primary_email):
        return True
    # A dependent's own token: covered iff a consented caregiver has agreed.
    from dashboard import household as _hh
    for cg in _hh.caregivers_for(cx, primary_email):
        if cg["share_consent"] and is_member(email=cg["primary_email"]):
            return True
    return False
```

Why this shape:

- **No relationship check needed.** `caregivers_for(email)` returns the caregivers linked
  *above* this email. A dependent has some; a standalone adult or a caregiver has none, so the
  function returns `[]` and behaviour is byte-identical to today. The dependency is implicit
  in the household graph.
- **`share_consent` is the right condition.** It is the same flag that gates a caregiver's
  *view* (`can_view`). A pet's consent defaults to 1 and the pet cannot revoke it. A competent
  adult who revoked caregiver access has `share_consent=0` and is no longer covered — which is
  correct: they must then agree themselves.
- **Render-time, not write-time.** Deriving live means the day a caregiver accepts Terms,
  every dependent's portal unlocks; remove the link and it re-gates. No backfill, no
  staleness, no legally-odd record ("this cat agreed at 2026-07-10").
- **It composes with #750.** In the `?member=` path `primary_email` stays the caregiver, so
  `is_member` returns true on the first line — unchanged. This fix is specifically for the
  *dependent's-own-token* path (#750 did not cover it, because #750 is about whose report
  renders under `?member=`, not whose token is being opened).

**What render-time deliberately does NOT do:** it does not stamp `tos_agreed_at` on the
dependent's own `journey_state`. So the dependent's email stays out of `_tos_agreed_emails`
(`app.py:22399`, the compliance/tag collector) and out of the `/begin` funnel gates
(`app.py:3626/3682`). That is correct: the dependent did not agree. The caregiver did, and the
caregiver's email is already in those sets.

### 2. Legal — the TOS text must extend the agreement

For the caregiver's click to *legally* cover the dependent, the Terms the caregiver accepts
must say so. The portal's TOS gate copy (`static/client-portal.html`) needs a clause:

> By accepting, you agree to these Terms on your own behalf and on behalf of those in your care
> whose accounts you manage — including your minor children, dependents, and animals.

Without this sentence the code opens the gate but no one has legally agreed for the dependent.
**This is the actual "do it properly."** The code change and the copy change ship together and
flip together — the gate must not open for dependents until the accepted text covers them.

## Who is what — three different signals, do not conflate

| Question | Signal | Governs |
|---|---|---|
| Is this a **dependent**? | household `relationship` ∈ `DEPENDENT_RELATIONSHIPS` (implicit via `caregivers_for`) | **TOS coverage (this slice)** |
| Is this an **animal**? | E4L **Species** column (Slice 4, not yet synced) | the greeting ("Give our Aloha to Sasha") |
| Is this a **minor**? | age from DOB | see below |

Species is *not* the TOS signal: a human minor is a dependent too, and `Species=human` would
miss them. DOB is *not* reliable: Hershey's 2016 birthdate reads as pet-or-child. The
**household link** is the governing signal — set by the guardian when they create the
dependent's account.

## Minors, specifically — a flag for Glen and counsel, not a code decision

A child's account should be created and linked by the parent as `relationship: "child"`; the
parent's TOS then covers them, exactly as for a pet. But **minors under 13 may carry
COPPA-style obligations** — verifiable parental consent that goes beyond "the parent clicked
agree." That is a legal question about the Terms wording and the consent flow, for Glen and
counsel. This design does not resolve it; it flags it. The code path is identical (guardian
covers dependent); only the *sufficiency of the click* is the open legal question.

## Interaction with Slice 4 (Species / greeting)

Fixing the gate makes a dependent's own portal render. For an animal, it will then greet
"Aloha Sasha" — addressed to a cat. Glen's rule is "**Give our Aloha to Sasha**". So shipping
this slice makes Slice 4's greeting fix *visible* where it previously sat behind the gate.
Not a blocker; sequence Slice 4 soon after.

## Scope

**In:**
- `_portal_tos_agreed(cx, primary_email)` and its use at the `tos_agreed` computation in
  `api_client_portal` (`app.py:15767`).
- The TOS gate copy clause in `static/client-portal.html`.
- A flag, `DEPENDENT_TOS_ENABLED`, default off. Flag off → byte-identical to today. Flip only
  **after** the copy clause is live.

**Out:**
- Species sync and the animal greeting (Slice 4).
- Any change to the `/begin` funnel TOS gates or `_tos_agreed_emails`.
- The COPPA/minor legal determination (Glen + counsel).
- Auto-detecting dependents by age.

## Testing

- A caregiver who has agreed → their dependent's own-token portal returns `tos_agreed: true`;
  the dependent themselves never agreed.
- A dependent whose caregiver has **not** agreed → `tos_agreed: false`.
- A dependent with `share_consent=0` → `tos_agreed: false` (must agree themselves).
- A standalone adult (no caregiver) → unchanged: `tos_agreed` == `is_member(self)`.
- `?member=` path (caregiver's token) → unchanged (#750 still governs).
- Flag off → `tos_agreed` == `is_member(primary_email)`, byte-identical.
- The dependent's email does NOT appear in `_tos_agreed_emails` after coverage.
- Render-verify: Sasha's own portal, with a caregiver who agreed, renders her card instead of
  the Terms wall.

## Open question

None blocking the code. The legal wording of the "on behalf of those in your care" clause, and
the COPPA question for under-13s, are Glen-and-counsel decisions that gate the *flag flip*, not
the build.
