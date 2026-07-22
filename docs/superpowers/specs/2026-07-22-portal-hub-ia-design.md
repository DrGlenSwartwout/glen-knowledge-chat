# Client Portal — Hub Information Architecture — Design

**Date:** 2026-07-22
**Status:** Draft for review
**Owner:** Glen (illtowell.com / deploy-chat)
**Visual mockup:** https://claude.ai/code/artifact/b225762d-a68d-4eee-b989-a90f48d01871

## Goal

Reorganize the client portal from a single overloaded "Current Analysis" scroll
into a **hub of services**: a calm landing that orients the client and names the
next step, over a grid of focused, single-purpose tiles — each its own page. Give
the practitioner finder, an editable health record, and the client's owned-tools
inventory real homes instead of stacking them on one page.

This is an **information-architecture reorg plus two genuinely new pages** (My
Health Profile, My Healing Oasis). Most existing cards move; they are not rebuilt.

## Problem — today's portal

The portal is a single-page app: `GET /portal/<token>` (`app.py:19461`) serves
`static/client-portal.html`, whose entire UI is generated client-side by
`render(d, v)` (`client-portal.html:1131`) from a JSON payload. Server payload is
built by `get_portal_view()` (`dashboard/portal_view.py:291`).

- The **home page ("Current Analysis") carries ~30 cards** — welcome, membership,
  practitioner band, Ask Dr. Glen, finder, Body Map, MasterClasses, account,
  video/audio/report, invoices, scan analysis, matches, formulations, essences,
  PRL options, program card, a scan-history *teaser*, remedies/reorder/wishlist,
  orders, ambassador links, share page, consult, prefs, recommendations.
- Only **3 tabs** exist (Current Analysis, Scan History, Orders & Invoices),
  hard-coded at `client-portal.html:2120–2124`, and they render **only when the
  `scan_history_enabled` flag is on**; otherwise it's one long scroll with no tabs.
- **Scan history appears twice** — a teaser card on home *and* its own tab.
- The **practitioner finder is invisible to clients**: the full feature exists
  (`/practitioner-finder`, `app.py:37605`; embed card at `client-portal.html:1287`)
  but is gated behind `PORTAL_FINDER_ENABLED` (`app.py:6106` →
  `dashboard/portal_view.py:241`), which is off in prod.

## Non-goals

- Not rebuilding the scan analysis, orders, or finder engines — they are reused.
- Not changing the auth/token model — identity still derives from the portal token.
- Not a visual/brand redesign beyond what the hub layout requires.

## Design — the hub

### Landing

A status **banner** (where you are: current scan, terrain phase, match counts) with
a single **next-step** pill, over a **grid of service tiles** grouped by the healing
journey. The banner replaces the welcome/what's-next/preparing cards. **Account**
lives in the top bar, not the grid.

The grid is a launcher, not a tab bar. This is the deciding constraint: with 12
destinations today and more coming, a linear tab bar (which caps ~5) is not viable;
a grid scales by adding tiles without ever re-overloading the landing.

### The four journey groups and their tiles

**1 · Understand**
- **My Analysis** — current scan, matches, healing path, personal video/audio/
  written report, unlock gate, consult prompt.
- **My Remedies** — the *decision + audit* view (see below).
- **My Health Profile** *(NEW)* — editable history, symptoms, intake + an
  AI-suggestions queue (see below).

**2 · Act**
- **My Healing Oasis** *(NEW)* — unified owned-inventory + build-out (see below).
- **Find a Practitioner** — the existing finder, promoted to a first-class door.
- **Refer a Friend** — invite/share; Ambassador *doing* surface (grab link, share).

**3 · Learn & Ask**
- **Ask Dr. Glen** — chat about terrain & remedies.
- **Body Map** — `/portal/<token>/bodymap` (`app.py:19466`).
- **MasterClasses** — free courses.

**4 · Track**
- **Scan History** — `buildScanHistoryHtml(d)` (`client-portal.html:674`), reused.
  The duplicate home-page teaser card is **removed**.
- **Orders & Invoices** — `buildOrdersHtml(d)` (`client-portal.html:709`), reused.
  Unpaid invoices flagged here (pay-now); paid invoices are receipts/archive.
- **Referrals** — the Ambassador *dashboard*: who joined, rewards, payout status.

**Account** (top-bar menu, not a tile): profile summary, membership, notification
prefs, scan prefs, onboarding.

### New page — My Remedies

Two sections:

1. **Top recommended for you (ranked)** — the *decision* view: top remedies ranked
   by the analysis, each with its "why" and an **"add to my Oasis / order"** action
   that hands off into My Healing Oasis › Replenish. Absorbs today's "formulation
   matches," "practitioner recommends," "My Recommendations," PRL options, and the
   program card.

2. **What you take from other companies (client-maintained)** — a list the client
   builds of remedies/supplements they take elsewhere. Each entry carries:
   - **Brand** and **product name**
   - **Reason** (optional — why they take it)
   - **Importance 1–10** (client-set)
   - **Request review** button — asks Dr. Glen to review that product for their case
   - **Review access** — link(s) to the resulting review / any existing review Glen
     hosts for that product
   - **Suggested upgrade** — where it genuinely helps, a pointer to our equivalent/
     better formulation (e.g. reveal-style swaps). **Not every item gets a swap** —
     a well-chosen product is confirmed as such; this preserves clinical integrity
     and is not a blanket upsell.

**Why capture the full stack.** Knowing everything the client takes lets the engine
catch interactions and redundancy, feed the review-request queue, and target
upgrades only where warranted. This is the remedy-side mirror of Healing Oasis ›
Build Out's external-tools list — the client maintains it, Glen's engine reacts.

**Data:** the external-remedy list is client-maintained (brand/product/reason/
importance), the review-request produces a work item + a hosted review link, and
the suggested-upgrade mapping reuses the existing product/reveal swap logic.

> **Reuse found (2026-07-22):** the request-review + hosted-review half already
> exists — flag `REVIEWS_ENABLED` (`app.py:6102`) + `dashboard/supplement_reviews.py`
> + `_supplement_reviews_block` (`dashboard/portal_view.py:242`), keyed on
> `product_name`/`product_brand` with a `requested → ai_draft → confirmed` status
> and review text exposed only once Glen confirms. My Remedies' external list should
> build on this module, not a new one. Adds needed: client-set `reason`/`importance`
> and the suggested-upgrade mapping.

### New page — My Health Profile

Two halves:

1. **Editable record** — Current Symptoms, Health History & Intake, rendered as
   editable fields/chips (not a read-only dump). The client can correct or add
   anything.
2. **AI-suggestions queue** — a pending panel: facts the system extracted from the
   client's Ask-Dr.-Glen conversations ("headaches started after the move"),
   surfaced as *suggested, not yet saved*, each with **Confirm / Edit / Dismiss**.
   Nothing writes to the record until the client confirms. This mirrors the
   review-not-auto pattern used elsewhere in the app.

**Single source of truth (critical).** This tile is a client-facing, editable
**window onto the same intake record the console already holds** — not a second
copy. Every manual edit and every confirmed chat-suggestion **writes back to that
one intake store**, and the console client-360 view reflects it. There is no
parallel portal-only profile; the two must never diverge.

### New page — My Healing Oasis

The unified "your stuff" tile, two modes inside:

- **Replenish** — consumables (formulations, essences) the client owns, with
  reorder actions and running-low indicators. Absorbs today's "Order your
  remedies," reorder module, and wishlist cards.
- **Build Out** — the client's **tools inventory**, listing items bought from us
  *and* items they own from other sources (client-maintained, "+ Add a tool you
  own"), plus a **clinically-ordered roadmap of recommended additions** to
  complete their home/office Healing Oasis. The roadmap **leads with our
  highest-impact tools — Harmony, Water Ionizer, Kloud** — emphasized as the
  recommendations that most improve outcomes (recommended for nearly everyone),
  above the terrain-specific gap items that follow.

**Why the recommendations are generous and ordered.** Including externally-owned
tools is what makes the roadmap correct: if the client already owns a light device
from elsewhere, the engine won't recommend one and sequences around the gap. The
ordering is driven by **the client's analysis** (terrain phase first, low-cost
high-leverage next, larger investments once basics land) — not a generic catalog.
My Healing Oasis is therefore **downstream of My Analysis**.

**Hand-off with My Remedies.** My Remedies (Understand) is the ranked
recommendation; My Healing Oasis › Replenish (Act) is the owned/reorder inventory.
A ranked remedy in My Remedies carries an "add to my Oasis / order" action that
lands it in Replenish. Spec both as views over the same remedy data with an
explicit link between them.

### Ambassador — two surfaces, one engine

- **Refer a Friend (Act)** — the doing: personal link, invite, share page.
- **Referrals (Track)** — the dashboard: referred contacts, commissions/rewards,
  payout status.

Both read/write the same referral data (referral commission tiers already modeled
in the app). Absorbs today's "Your Ambassador links," "Become an Ambassador," and
"Your share page" cards.

## Card migration — where today's ~30 home cards go

| Today's home card | New home |
|---|---|
| Welcome / "healing home" | Landing banner |
| What's next for you | Landing banner (next-step pill) |
| Preparing your Biofield Analysis | Landing banner + My Analysis (status) |
| Your practitioner band | Landing banner / My Health Profile header |
| Your scan analysis + healing path | My Analysis |
| What your scan matched | My Analysis |
| Personal video / audio / written report | My Analysis |
| Share & Unlock | My Analysis (unlock gate) |
| Biofield Consult | My Analysis (+ Act prompt) |
| Your formulation matches | My Remedies (ranked) |
| Your practitioner recommends | My Remedies |
| My Recommendations | My Remedies |
| Premier Research Labs options | My Remedies → order into Oasis |
| Practitioner-composed program | My Remedies / My Analysis |
| Order your remedies | My Healing Oasis › Replenish |
| "Your Remedies" reorder module | My Healing Oasis › Replenish |
| Your Life Stress Essences | My Healing Oasis › Replenish |
| Your wishlist | My Healing Oasis › Build Out |
| Find a Practitioner (iframe) | Find a Practitioner tile |
| Your Body Map | Body Map tile |
| Free MasterClasses | MasterClasses tile |
| Ask Dr. Glen | Ask Dr. Glen tile |
| Scan history teaser | **Removed** (dup of Scan History tab) |
| Your orders | Orders & Invoices |
| Invoice / options & pricing | Orders & Invoices (unpaid) |
| History & receipts (legacy combined) | **Removed** (superseded by two tabs) |
| Your Ambassador links / Become Ambassador | Refer a Friend + Referrals |
| Your share page | Refer a Friend |
| Membership entry / "more savings" upsell | Account (+ landing accent) |
| Your account snapshot | Account |
| Notification prefs / Your preferences | Account |
| onboarding / coaches / peer (hidden) | Account / hidden |

## Data & payload changes

- Extend the payload builder `get_portal_view()` (`dashboard/portal_view.py:291`)
  and/or the `d` API (`api_client_portal`, `app.py:19474`) with the fields each new
  page needs: health-record read/write, chat-suggestion queue, ranked-remedy list,
  client-maintained **external-remedy** list (brand/product/reason/importance) +
  review-request queue + hosted-review links + upgrade mapping, owned-tools
  inventory (ours + external), referral dashboard data.
- **Health record read/write** binds to the existing intake store (same one the
  console client-360 reads); add a portal write path + the chat-extraction →
  pending-suggestion producer.
- **Finder** just needs the `PORTAL_FINDER_ENABLED` gate on (see Phase 0).

## Phasing

- **Phase 0 — finder on (independent, near-instant).** Flip `PORTAL_FINDER_ENABLED`
  so the finder is live for all portals now, decoupled from the reorg. (Flag-state
  check: confirm in prod, not dev.)
- **Phase 1 — the hub shell.** Landing banner + tile grid + `showTab()`-style
  routing to the existing surfaces (My Analysis, Scan History, Orders, finder,
  Body Map, MasterClasses). Move existing cards to their tiles; remove the two
  duplicate/legacy cards. No new data.
- **Phase 2 — My Health Profile.** Editable record over the shared intake store +
  the AI-suggestions confirm/edit/dismiss queue.
- **Phase 3 — My Healing Oasis.** Replenish (reorder inventory) + Build Out
  (owned tools incl. external + analysis-ordered roadmap), with the My Remedies ↔
  Replenish hand-off.
- **Phase 4 — Ambassador split.** Refer a Friend (Act) + Referrals dashboard
  (Track) over the existing referral engine.

Each phase ships independently and leaves the portal usable.

## Open questions

1. **My Analysis vs. My Remedies card split** — final line between the two for the
   overlapping cards (formulation matches, program, PRL). Draft above; confirm.
2. **Should the tile grid replace the tab bar for *all* portals now**, or stay
   behind a flag during rollout (mirroring how tabs are gated today)?
3. **Health-record write target** — confirm the exact existing intake store/table
   the portal edits write back to, so console and portal share one record.
4. **Membership/upsell placement** — Account only, or also a persistent landing
   accent?
