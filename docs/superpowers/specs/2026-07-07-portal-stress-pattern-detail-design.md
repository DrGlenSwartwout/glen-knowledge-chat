# Portal — click-to-reveal stress-pattern detail

**Date:** 2026-07-07
**Surface:** client-facing portal (`static/client-portal.html`)
**Scope:** one file, frontend only. No backend, no new endpoints, no schema change.

## Problem

The client portal shows a scan's stress patterns as a row of static chips (just the
pattern name). Each E4L finding already carries a plain-language `description`
(`e4l_items.e4l_description`), and that field already flows all the way to the
browser (`_findings` → stored `bf_content.findings` → `client_findings` in
`api_client_portal` → `payload.findings` → `d.findings`). The chips simply ignore
it. Clients can see *that* they have a pattern but not *what it means*.

Two content facts constrain the design:

1. **Descriptions are partial.** Of 223 `e4l_items`, 85 have a description — the
   energetic "Driver" categories (ET/ED/EI/ES/MB/BFA/MR). The other 136 (ER
   stresses, Nutrition, Environmental) are blank. On a real scan, a meaningful
   fraction of chips have nothing to reveal.
2. **Chips are first-view-only today.** The chip row (`patternsBlock`,
   ~line 713) is built only when `firstTime && findingsCount`, and `firstTime`
   is only ever true for non-confirmed reports. So chips appear once (the unfold
   animation) and never on return visits or on confirmed reports.

## Goal

Let a client click a stress-pattern chip to reveal its description inline, for the
chips that have one — and make the chips persist so the exploration is available
on every visit.

## Decisions (locked)

- **Blank chips:** only chips *with* a description are interactive. Blank ones stay
  exactly as today — a plain, non-clickable `<span>`. No fallback text, no
  AI-generated copy in this slice.
- **Interaction:** inline expand — a single detail panel directly under the chip
  row. One panel open at a time.
- **Coach / member cards:** out of scope. They already exist and are live
  (`initCoachesCard`, `initPeerCard`).

## Behavior change (intentional, called out)

Chips will render on **every** portal render, not just the first view — including
confirmed reports, which currently show no chips at all. The staggered `reveal`
animation stays a first-view-only flourish (gated on `firstTime`); the chips
themselves are no longer gated on `firstTime`. Returning clients and confirmed
reports will now show their pattern chips.

## Design

### Chip rendering (`patternsBlock`, ~line 713)

Rebuild the block so it is populated whenever `findingsCount > 0` (drop the
`firstTime` gate on the block; keep it on the `reveal` class + animation-delay):

- Finding with a non-empty `description`:
  `<button type="button" class="pat-chip pat-chip--detail{reveal?}"
   data-pname="{esc(name)}" data-pdesc="{esc(description)}"
   aria-expanded="false" aria-controls="patDetail">{esc(name)}</button>`
  A real `<button>` gives native focus + Enter/Space activation for free.
  `esc()` escapes `& < > "`, which is safe for the double-quoted `data-*`
  attributes.
- Finding with no description: unchanged — `<span class="pat-chip{reveal?}">…`.
- Intro line: append " — tap one to learn more" **only** when at least one
  finding in the set has a description.
- After the `.pat-wrap` div, emit one empty panel: `<div class="pat-detail"
  id="patDetail" hidden></div>`.

`patternsBlock` continues to be interpolated into the same three status branches
(`ai_draft`, `interested`/`requested`, `confirmed`) exactly where it is today —
no call-site changes beyond the block now being non-empty on return/confirmed.

### Detail panel + interaction

One delegated `click` listener (attached once, guarded by a module flag so the
portal's poll re-renders don't stack listeners) handles `.pat-chip--detail`:

1. Read `data-pname` / `data-pdesc` from the clicked chip.
2. If the clicked chip is already the open one → close: hide `#patDetail`, set its
   `aria-expanded="false"`, clear the "open" marker.
3. Otherwise → open/swap: reset every `.pat-chip--detail` to
   `aria-expanded="false"`, set the clicked chip to `"true"`, fill `#patDetail`
   with the name (small heading) + description, unhide it, and mark it open.

Keyboard: covered by native `<button>` (Enter/Space fire `click`). Focus ring
via existing focus styles.

### CSS

- `.pat-chip--detail{cursor:pointer}` + a subtle affordance (a small caret or dot,
  and a hover/focus background shift) so detail chips read as tappable while blank
  chips stay flat. Reuse existing tokens (`--brand-soft`, `--line`, `--brand`).
- `.pat-detail` — a quiet panel: small type, muted, left border accent
  (`box-shadow:inset 3px 0 0 var(--brand)` like the practitioner chat bubble),
  padding, margin under the wrap.
- Expand animation: gentle fade + slide on unhide; wrapped so
  `@media (prefers-reduced-motion: reduce)` disables it (mirror the existing
  `.reveal` reduced-motion rule).

## Out of scope

- Filling the 136 blank descriptions (ER / Nutrition / Environmental).
- Any change to coach/member/household cards, layers, remedies, or backend.
- Per-pattern deep links, sharing, or analytics on chip opens.

## Verification

Headless-render the real portal (`/portal/<token>`) against a token whose latest
scan has driver-category findings (ET/ED/EI/ES/MB — descriptions present) and at
least one blank-description finding (ER/Nutrition/Environmental). Assert:

1. Chips render on a **confirmed** report (persistence change works).
2. A described chip is a `<button.pat-chip--detail>`; clicking it reveals
   `#patDetail` containing the description text; `aria-expanded` flips to `true`.
3. A blank-description finding is a plain `<span.pat-chip>` with no detail button
   and no affordance.
4. Clicking a second described chip swaps the panel content and leaves only one
   panel open (first chip's `aria-expanded` back to `false`).
5. Clicking the open chip again closes the panel (`hidden`, `aria-expanded=false`).
