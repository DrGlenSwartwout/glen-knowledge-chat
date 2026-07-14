# Report Entity Hover + Link-Out — Design

**Date:** 2026-07-13
**Status:** Approved (design direction), pending spec review
**Surface set:** client portal report (`/portal/<token>`, `static/client-portal.html`) + public product pages (`static/begin-product.html`) + public ingredient pages (`static/begin-ingredient.html`)

## Problem

On the online report and product/ingredient pages, clinical entities (stress patterns, structures, functions, remedies, ingredients) render as plain text. A reader can't get a quick definition without leaving the page, and can't jump to the full detail page for an entity that has one. Today only stress-pattern chips are interactive, and that's a tap-to-expand inline panel — not a hover pop-up, and not a click-through to a full page.

## Goal

Give every supported entity a single, consistent affordance:

- **Hover (desktop)** → a small pop-up with basic info (name + 1–3 sentences).
- **Click** → the entity's full detail page, opened in a **new tab**, when one exists.
- **Tap (touch)** → the same pop-up, containing an "Open full page ↗" link when a destination exists (hover doesn't exist on touch, so tap must not be a dead end).

## Scope decision (confirmed)

**Option A.** Ship the full hover/click experience now for the entities that already have both basic-info text and a destination page. For entities with no page yet (stress patterns, structures), show pop-up only — no click-through. No new backend detail pages are built in this slice.

| Entity | Pop-up text source | Full-page destination (this slice) |
|---|---|---|
| Ingredient | `ingredients` DB (mechanism, dosing, safety) via existing ingredient-page data | `/begin/ingredient/<slug>` (new tab) |
| Remedy | `biofield_remedy_meanings.meaning` (+ `formulations.description` fallback) | product page `/begin/product/<slug>` when slug resolvable (new tab) |
| Function | `/learn` topic content | `/learn/<slug>` (new tab) |
| Stress pattern | `e4l_items.e4l_description` (already feeding the chip) | none → pop-up only |
| Structure | name only (no description text available) | none → **deferred entirely** this slice |

Structures carry no descriptive text and no page, so there is nothing to show; they are explicitly out of scope for this slice and remain plain text. A later slice can add structure text + pages.

## Architecture

### One reusable component: `entity-ref`

A single inline element type replaces the five ad-hoc renderings. It is data-driven, so the same markup + behavior works on the portal report and the product/ingredient pages.

**Markup contract** (rendered server-side or client-side depending on surface):

```html
<span class="entity-ref"
      data-etype="ingredient|remedy|function|pattern"
      data-name="Magnesium L-Threonate"
      data-info="Basic-info text, 1–3 sentences, pre-escaped."
      data-href="/begin/ingredient/magnesium-l-threonate">   <!-- omitted when no destination -->
  Magnesium L-Threonate
</span>
```

- `data-href` present → element is a link-out (new tab on click/tap-link). Absent → pop-up only.
- `data-info` always present → drives the pop-up body. `data-name` is the pop-up title.
- Escaping: `data-info`/`data-name` are HTML-attribute-escaped at build time; the popover re-escapes on inject (same discipline as the current `wirePatternDetails`).

### Behavior module: `wireEntityRefs()`

Extends the existing delegated-listener pattern (`wirePatternDetails` in `client-portal.html`, lines ~754–785) rather than adding a tooltip library.

- One delegated listener bound high in the tree (e.g. `#app` on the portal, page root on begin pages).
- **Desktop hover:** `mouseenter`/`focus` on an `.entity-ref` → show a single shared popover positioned near the element; `mouseleave`/`blur` → hide (with a short close delay so the pointer can travel into the popover if it becomes interactive). Only one popover instance exists at a time (reuse the `#patDetail` single-panel approach, generalized to `#entityPop`).
- **Click:** if `data-href` present → open in new tab (`window.open(href, '_blank', 'noopener')` or an `<a target="_blank" rel="noopener">`); else no-op (pop-up already shown on hover).
- **Touch:** no hover event fires; first tap shows the popover. If `data-href` present, the popover renders an "Open full page ↗" anchor (`target="_blank" rel="noopener"`). Tapping elsewhere dismisses.
- **Keyboard/a11y:** `.entity-ref` is focusable (`tabindex="0"` when non-link; native focus when it's an `<a>`); focus shows the popover, `Esc` closes it; `role="button"`/`aria-describedby` wiring on the popover. Link-out entities use a real anchor for correct semantics.

### Popover element: `#entityPop`

A single shared, absolutely-positioned container (title + info body + optional link row), reused for every entity — mirrors today's single `#patDetail` panel. Styled to the portal/brand palette; viewport-edge aware (flip above/below, clamp horizontally) so it never overflows on mobile.

## Data flow

No new tables. Each surface's existing data-assembly path is extended to emit the `entity-ref` fields:

- **Portal report** (`dashboard/portal_view.py` / `biofield_e4l.py` / `biofield_meanings.py`):
  - Stress-pattern chips: already carry `description` → re-render as `entity-ref` (`etype=pattern`, no href). This *replaces* the current `pat-chip--detail` button, preserving the existing text source.
  - Causal-chain layer **remedy**: look up `biofield_remedy_meanings.meaning` (fallback `formulations.description`) for `data-info`; resolve product slug for `data-href`. Where no slug resolves, pop-up only.
  - Causal-chain layer **function** (from `L.meaning`/function tagging): if the function maps to a `/learn/<slug>` topic page, emit `data-info` (topic summary) + `data-href`; else skip (leave plain) this slice.
- **Product page** (`begin-product.html` / `dashboard/product_page_sections.py`): ingredient list already renders `<a href="/begin/ingredient/<slug>">` same-tab. Upgrade those to `entity-ref` (`etype=ingredient`, `data-info` from ingredient summary, `data-href` → ingredient page, **new tab**).
- **Ingredient page** (`begin-ingredient.html`): related-forms / related-products links upgraded to `entity-ref` where a remedy/product destination + info exist.

### Slug / info resolution helper

A small server-side helper `entity_ref(etype, name, ...)` (in a new `dashboard/entity_refs.py`) returns `{name, info, href|None}` for a given entity, centralizing the per-type lookups (e4l_description, remedy meaning, ingredient summary, learn-topic). This keeps each surface's template code thin and gives one place to unit-test resolution.

## Error handling / edge cases

- **Missing info text:** if an entity has no `data-info`, it renders as plain text (no `entity-ref`) — never an empty pop-up.
- **Unresolvable destination:** `data-href` omitted → pop-up only, no broken link.
- **Remedy slug ambiguity:** if a remedy name maps to more than one product, prefer the canonical formulation slug; if none is unambiguous, omit `data-href` (pop-up only) rather than guess wrong.
- **Long info text:** popover body is capped (max-height + scroll) so it can't cover the whole viewport on mobile.
- **New-tab safety:** all link-outs use `rel="noopener"`.
- **No regression for patterns:** the existing `wirePatternDetails` behavior is superseded by `wireEntityRefs`; the pattern chip's current text source and "about half the catalog has no description" behavior are preserved (no description → plain chip, as today).

## Testing

- **Unit (`dashboard/entity_refs.py`):** resolution returns correct `{name, info, href}` per type; returns `href=None` for patterns/structures and for unresolvable remedy slugs; returns plain (no entity) when info is missing.
- **Render-verify (headless Chrome, per the render-verify house rule):**
  - Portal report: hover a pattern chip → pop-up shows description, no link. Hover a remedy with a resolvable product → pop-up + click opens product page in new tab. Hover an ingredient/function likewise.
  - Product page: ingredient `entity-ref` hover shows info; click opens ingredient page in a new tab.
  - Touch emulation: tap shows pop-up; "Open full page ↗" link present only when destination exists; tap-away dismisses.
  - Keyboard: focus shows pop-up, `Esc` closes.
- **No-description / no-destination cases** render as plain text (assert absence of `entity-ref`).

## Out of scope (this slice)

- Structure detail text and structure/stress-pattern glossary pages (future slice; would flip those from pop-up-only to click-through).
- Any new backend detail page.
- Chat surface (`/full-report`, `/results`) and print/PDF renders — pop-up/hover is a live-web affordance; print stays plain text. (Can be revisited if desired.)
