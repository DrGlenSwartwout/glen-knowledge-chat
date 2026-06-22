# Chat Creates-a-Page Offer — suggest building a guide for an uncovered topic

**Date:** 2026-06-21
**Status:** Approved design, ready for implementation plan
**Sub-project of:** the on-request content arc — the *creation* half of the chat sub-project
**Builds on:** [[project_topic_pages]] (the `/learn` pipeline + compliance gate) and the chat page-links
matcher (`dashboard/page_links.py`) shipped just before it.

---

## 1. Purpose

When a `/begin` chat conversation is about a health topic that has **no page yet**, offer to build
one: "Want a guide on *Magnesium Deficiency*? We'll create it and email you." The person's accept
records a **suggestion**; Glen later picks which suggestions to build, and the build flows through the
*existing* topic-page pipeline (haiku draft → compliance gate → Glen approval → ready-email).

This is the inverse of chat page-links (which surfaces *existing* pages). It creates no page directly
and charges no fee (page creation is free; the paid book/course formats are a separate later
sub-project).

## 2. Decisions (locked during brainstorming)

| Decision | Choice |
|---|---|
| Trigger model | **Suggestion queue → Glen picks.** Accept records a suggestion; nothing auto-builds. |
| Topic detection | **AI-extract a normalized topic** (`{name, kind, slug}`), so the queue aggregates demand across phrasings |
| Who sees the offer | **Everyone**; submitting requires an email (notify + funnel capture) |
| Fee | **Free** (page creation) |
| Storage | **Reuse `topic_pages` with a new `suggested` state — no new table**; askers go in the existing `topic_page_requests` |
| Chat surface | The `/begin` `/chat` endpoint only |
| Rollout | **DARK behind `CHAT_TOPIC_OFFER_ENABLED`** (default off) |

## 3. Flow

1. In the `/chat` SSE done handler, **after** page-links runs: if page-links found **no** existing
   page, call `topic_copy.extract_topic_candidate(query, answer, client)` → `{name, kind, slug}` or
   `None`.
2. If a candidate is returned **and** there is no `topic_pages` row for that slug in state
   `approved`/`draft`/`gated` (i.e. row absent or already `suggested`), surface an **offer card**:
   `{key: f"suggest:{slug}", title: "Want a guide on {name}?", sub: "We'll create it and email you",
   href: f"/learn/suggest/{slug}?kind={kind}&name=<url-encoded name>"}`.
3. The offer card is merged into `surfaced_cards` via the **existing `page_links.merge_cards`** (it
   counts as a link card, so the cap + proof-card protection already apply).
4. Click → `GET /learn/suggest/<slug>` renders a small styled page (site chrome, the topic
   prefilled) with an email field. Submit → `POST /learn/suggest/<slug>` records the suggestion and
   shows a confirmation ("We'll create this guide and email you when it's ready.").
5. A suggestion = a `topic_pages` row in state `suggested` + a `topic_page_requests` row for the
   email. Demand = count of requests for that slug.
6. Console `/console/topic-suggestions` lists `suggested` rows ranked by demand. **Build this**
   runs the existing build (drafts + links + compliance scan → `draft`/`gated`); then the normal
   `topic_page.approve` flow publishes it and emails everyone who asked (existing
   `notify_on_approve`). **Dismiss** drops a junk suggestion.

## 4. New / changed code

### `dashboard/topic_copy.py`
- `extract_topic_candidate(query, answer, client) -> {"name", "kind", "slug"} | None`
  - One synchronous haiku call. System prompt: "If this conversation is centrally about a single
    health topic a person would search for, return JSON `{name, kind}` where kind is one of
    symptom|condition|function; otherwise return `{}`." Parse, normalize: `kind` must be in
    `("symptom","condition","function")` else `None`; `slug = _slugify(name)` (reuse the same
    kebab-case slugify used elsewhere, e.g. `dashboard.ingredients.slugify`); return `None` on empty
    name / bad JSON / any error (never raises).

### `dashboard/topic_pages.py`
- `record_suggestion(cx, slug, name, kind, email) -> str` — returns the resulting state.
  - Read existing row. If its state is in `("approved","draft","gated")`: **do not downgrade** —
    only `record_request(cx, slug, email)` (demand still counts), return that state.
  - Else (row absent or state `suggested`): `set_name`/`set_kind`, `set_state(cx, slug, "suggested")`,
    `record_request(cx, slug, email)`, return `"suggested"`.
- `list_suggestions(cx) -> list[dict]` — rows where `state == "suggested"`, each with
  `{slug, name, kind, demand}` where `demand` = count of `topic_page_requests` rows for the slug,
  ordered by demand desc.

### `dashboard/topic_render.py`
- `render_suggest_html(slug, name, *, submitted=False) -> str` — styled (site chrome) page: when
  `submitted` is false, a short intro + an email form `POST`ing to `/learn/suggest/<slug>`; when true,
  a confirmation message. `noindex` (utility page, like the pending page). Pure, HTML-escaped.

### `dashboard/topic_page_actions.py`
- `topic_page.dismiss` — sets state `"dismissed"` (drops the suggestion off the queue; never public).
  `OWNER/OPS`, `LOW_WRITE`. (Building a suggestion reuses the existing `topic_page.regenerate`
  action — it drafts + scans regardless of starting state — so no separate build action is needed.)

### `app.py`
- Flag `CHAT_TOPIC_OFFER_ENABLED` (default off), by the other funnel flags.
- `/chat` done handler: behind the flag, after the page-links block, if **no link cards fired**, run
  `extract_topic_candidate`, gap-check the slug, and (when appropriate) build the offer card and pass
  it through `merge_cards` alongside the page-link/journey cards. Wrapped so it never breaks the
  stream; the AI call is the same `_cl` haiku client.
- `GET /learn/suggest/<slug>` (flag-gated; renders `render_suggest_html`) and
  `POST /learn/suggest/<slug>` (flag-gated; reads `email`/`kind`/`name`, calls `record_suggestion`,
  renders the submitted confirmation). Email required to record (no email → re-render the form with a
  gentle prompt).
- Console: `/console/topic-suggestions` (static page) + `GET /api/console/topic-suggestions`
  (`_sales_console_ok`-guarded, returns `list_suggestions`). The console page's **Build** button calls
  `topic_page.regenerate`; **Dismiss** calls `topic_page.dismiss`; both via the existing
  `/api/action/<key>` dispatch. Register `topic_page.dismiss` next to the other topic actions. Add
  `/console/topic-suggestions` to `static/console-search-index.json`.

## 5. Safety

- **Nothing publishes without Glen building *and* approving.** The compliance hard-gate is unchanged;
  a suggested or building page is never served publicly (the public `/learn/<slug>` path and the
  page-links index both already serve `approved` only, so `suggested`/`dismissed` rows never leak).
- A bad AI-extracted slug only ever creates a private `suggested` row, which Glen dismisses.
- The offer never auto-builds, so member whims cannot spawn AI drafts — the AI cost is paid only when
  Glen clicks Build.
- Flag off → no extraction call, no offer, no routes.
- All new chat-path work is wrapped in try/except so it can never break the chat stream; hrefs are
  built from the normalized slug (no model-generated URLs).

## 6. Testing

Unit tests (no `import app`; test `dashboard/*` helpers directly; verify `app.py` via `py_compile`):

- `extract_topic_candidate`: a clear single-topic conversation returns `{name, kind, slug}` with a
  valid kind and kebab slug (FakeClient returning known JSON); an off-topic / multi-topic
  conversation (FakeClient returning `{}`) returns `None`; a bad-kind or bad-JSON response returns
  `None`; a client error returns `None`.
- `record_suggestion`: a brand-new slug → row `suggested` + one request; a second asker → still
  `suggested`, demand 2; an existing `approved`/`draft`/`gated` slug → **state unchanged**, request
  still recorded (no downgrade).
- `list_suggestions`: returns only `suggested` rows, with correct `demand` counts, ordered desc.
- `render_suggest_html`: form variant posts to `/learn/suggest/<slug>`, is `noindex`, escapes the
  name; submitted variant shows the confirmation and no form.
- `topic_page.dismiss`: sets state `dismissed`; a dismissed row is excluded from `list_suggestions`.
- Wiring (`py_compile` + source asserts): flag present + default off; `extract_topic_candidate`,
  `/learn/suggest`, `topic_page.dismiss`, and `/api/console/topic-suggestions` referenced; the offer
  is gated behind `CHAT_TOPIC_OFFER_ENABLED` and only runs when no link cards fired.

## 7. Rollout

- Ships **DARK behind `CHAT_TOPIC_OFFER_ENABLED`** (default off).
- Go-live: flip the flag in Doppler `remedy-match/prd` after a live check that a chat message about an
  uncovered topic surfaces the offer card → the suggest form records a suggestion → it appears in
  `/console/topic-suggestions` → Build → approve → ready-email. Reversible by flipping back.

## 8. Out of scope (future)

- The paid book ($29.99) / course ($49.99) formats + Stripe fulfillment.
- Traffic-driven auto-creation (auto-build on demand thresholds).
- Demand-gated auto-build (the brainstorm's rejected alternative).
- Semantic topic extraction beyond the single haiku call; other chat surfaces.
