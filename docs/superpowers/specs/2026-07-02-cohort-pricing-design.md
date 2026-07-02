# Cohort-based pricing — design (DRAFT for discussion)

**Date:** 2026-07-02
**Status:** Draft — brainstorming, not yet approved to build.

## Why this exists
In one week we layered five pricing mechanisms — volume curve, per-client special price (#505), FF flat rate (#509), the proposed reorder-loyalty price, and a "lowest/explicit wins" resolver. Each is a one-off, and the combined precedence is getting hard to reason about, debug, and explain. This design **unifies all of them into one abstraction**: a client belongs to **cohort(s)**, each cohort carries a **pricing policy** (+ optional benefits). Every existing mechanism becomes a cohort. It also adds the thing that makes it strategic rather than just tidy: **clients choose how they earn discounts**, which tests the *selling* of the feature (preference), not just imposed outcomes.

## The model

**`cohorts`** — the catalog of pricing structures:
`id, key (unique), name, description, policy_json, benefits_json, active, is_default, created_at`

**`cohort_members`** — which clients are in which cohort:
`email, cohort_key, source ('chosen'|'earned'|'assigned'|'admin'), joined_at, expires_at (nullable)`

A client may belong to several cohorts (e.g. the default volume cohort + an earned reorder-loyalty cohort + a chosen plan). Resolution (below) picks the effective price per line.

## Pricing policy (per cohort) — declarative
A `policy_json` is a small rule set. Policy types (compose freely):
- **`volume`** — the standard order-wide volume curve. *(The default cohort.)*
- **`flat_ff`** `{cents}` — flat price for every FF. *(= today's FF-flat, #509.)*
- **`per_sku`** `{slug: cents}` — explicit per-product prices. *(= today's per-client special, #505.)*
- **`reorder_loyalty`** `{ff_cents, earn:'program'}` — a special price on SKUs the client purchased **on a program** (see "earning" below). *(= the new idea; $50 FFs.)*
- **`percent_off`** `{pct, scope:'ff'|'all'}` — N% off list.

Policies are data, so new structures (a test arm, a reward) are a row, not code.

## Resolution — one predictable rule
Per line item, the effective unit price is the **first** of:
1. **Explicit owner line edit** on this order (always wins).
2. **Best (lowest) applicable cohort policy** across all the client's cohorts.
3. **Standard volume/list.**

"Best applicable" = compute each cohort policy's price for the line, take the **lowest** — this is the "lowest wins" behavior, now scoped to *cohort policies the client actually holds* rather than an ad-hoc stack. (Alternative: strict cohort priority order. Recommend lowest-wins for client goodwill; revisit if it undercuts a negotiated rate — negotiated rates can be a `per_sku` cohort that we treat as a floor.)

## Earning a cohort (the reorder-loyalty case)
Per Glen: a SKU earns its reorder price when it was **purchased within a program** —
- a product **recommended and purchased in a one-off $99 program**, or
- **any product purchased during a prepaid 6- or 12-month program term**.

Implementation: on order fulfillment, if the order is program-linked, add the buyer to the `reorder_loyalty` cohort and record the earned SKUs (either per-SKU membership rows, or a policy that checks "has a prior program purchase of this slug"). Reorders of those SKUs then price at the loyalty rate via resolution.

## The wrinkle — choice-based assignment
At a natural moment (portal after first order / funnel post-scan), present:
> "We're building systems to make healing more accessible. How would you like to earn your savings?"

Two clearly-explained options, in **outcome** terms, e.g.:
- **Plan A — Stock up:** the more you keep on hand, the less each bottle costs (volume).
- **Plan B — Stay the course:** lock in a low per-bottle price on anything you've started (reorder-loyalty / flat).

Their pick sets their cohort (`source='chosen'`). Rules: **two options max**, a sensible default, plain language (no math), and **allow switching** (cooling-off) so a bad choice isn't punitive. This measures **what people prefer** (choice share) — the "selling" test Glen wants — on top of the outcome data.

## How it absorbs today's mechanisms (migration)
- Volume curve → the **default** cohort (everyone, unless in another).
- Per-client special (#505) → a **cohort-of-one** with a `per_sku` policy.
- FF flat (#509) → a `flat_ff` cohort.
- Reorder-loyalty (new) → a `reorder_loyalty` cohort, membership earned via program purchase.
- Split-test arms / individual rewards → named cohorts.

So the per-client-pricing UI we just built becomes "put this client in a cohort / give them a cohort-of-one."

## Measurement (the point of testing)
Stamp every order with the client's **effective cohort(s)** at purchase time. Report per cohort:
- new-client count & **choice share** (which plan people pick),
- **90-day reorder rate** and **6-month LTV** (NOT first-order conversion — a cheap first order that never reorders is a loss),
- avg discount given + **realized margin**.
Decide winners on LTV/retention, then **graduate** everyone to the winning structure (honoring anything promised as "lifetime").

## Guardrails (from the pushback)
- **Every policy independently margin-safe** — model adverse selection (assume each client picks their cheapest).
- Choice = **preference test**, self-selected, not a controlled causal test. Be explicit which question you're answering; optionally randomize the *default/framing* for a cleaner sub-experiment.
- Two options, outcome-framed, switchable.
- Choice-based assignment **removes** the fairness/ethics problem of random price-testing on sick clients.

## Rollout (phased — each shippable, flag-dark)
1. **Cohort layer**: tables + resolver; migrate the volume curve as the default cohort. *No behavior change* (regression-guarded).
2. **Migrate #505 + #509** to cohorts (per_sku, flat_ff); the existing UIs write cohort rows.
3. **`reorder_loyalty`** policy + program-purchase earning = first earned cohort.
4. **Choice step** (funnel/portal) + **measurement** report.

## Open decisions (need Glen)
- Multi-cohort resolution: **lowest-wins** (recommended) vs strict priority?
- Where/when the choice step appears, and the **two launch plans** + their exact rates.
- Precise "program purchase" definition to detect earning (the $99 one-off recommended-and-bought; prepaid 6/12-mo term) — needs orders tagged with a program id.
- Do negotiated per-client rates act as a **floor** (never undercut) or join lowest-wins?
