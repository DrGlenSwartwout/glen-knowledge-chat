# Record-Level Dashboard Deep-Links — Phase 1: Foundation + People

**Date:** 2026-06-26
**Status:** Design approved, ready for implementation plan
**Repo:** deploy-chat (`glen-knowledge-chat`)

## Problem

The dashboard's three briefings (`money-cash`, `clients-pipeline`,
`signals-patterns`) are pure LLM-generated markdown. Records are mentioned in
prose only — "Jane Doe (jane@example.com), $380 outstanding" — with no
machine-readable identifier. The console already has cross-link plumbing
(`actNavigate()` + `ACT_AREA`, anchors, `?email=` autoload — PR #341), but
**record-level** targeting was deferred precisely because the briefing markdown
carries no record ids.

Worse, the snapshot fed to Claude *does* contain some ids (invoice numbers,
sender emails), but Claude converts them to prose and they are discarded before
render.

**Goal (Phase 1):** when a briefing mentions a person/client, their name renders
as a link that lands on that person's record in the console, reusing existing
`?email=` destinations. No new console views in Phase 1.

## Scope

**In scope (Phase 1):**
- A shared citation-token mechanism (the foundation): registry → token → resolve
  → render. Built once here; reused by later phases.
- People/client links only, via email-based console destinations that already
  exist.

**Out of scope (later phases, reuse the mechanism):**
- Money: receivables / invoices (needs a `#receivables` row-highlight target).
- Orders (needs a single-order highlight target).
- Payments / Stripe (not in the briefing snapshot today).

Each later phase adds only its entity-emitters + its console destination; the
mechanism below is not re-touched.

## Approach: citation tokens, not prose-scraping

The model must never emit URLs or be trusted to round-trip JSON in free text
(the voice-journal bug lesson: *force structured refs, resolve server/registry
side*). Instead:

1. The snapshot carries a **registry** of linkable records, each with a short
   stable `ref` (`r1`, `r2`, …).
2. The prompt instructs the model: *when you mention a record that has a `ref`,
   write it as a markdown link using the ref as the URL — `[Jane Doe](ref:r3)`.
   Only use refs present in the data; never invent one; never write a real URL.*
3. At render, `ref:rN` is resolved against the registry to the real console URL.
   **Any unknown/missing ref is unwrapped to plain text** — graceful
   degradation, never a broken or hallucinated link.

Markdown-link syntax (`[display](ref:rN)`) is chosen over custom delimiters
because the model is reliably fluent in markdown links and unreliable at
balancing bespoke delimiters; `mdRender()` already turns them into anchors.

Prose-scraping (regex/name-matching the finished markdown) is rejected as the
primary mechanism — partial names, duplicate names, and ambiguity make it
fragile. Email-regex may be revisited only as a belt-and-suspenders fallback in
a later phase if compliance proves weak.

## Components

### 1. `dashboard/briefing_links.py` (shared foundation)

- `Linkable`: `{ref, type, display, url}`. `url` is a console **path + query
  with NO console key** (the key is appended client-side at click time so the
  secret never lands in stored files).
- `LinkRegistry`: snapshot modules append `Linkable`s to it. Assigns sequential
  refs (`r1`, `r2`, …). Dedups by `(type, url)` so the same person mentioned in
  two cards shares one ref.
- `person_url(email)`: returns the canonical console person destination. Single
  source of truth so later phases follow the same pattern.
  - **To verify during planning (do not assume):** the exact canonical person
    destination — `/console/crm?email=` vs portal vs the `/console` people tab —
    pinned by checking the live routes, per the "confirm the channel is live"
    rule. The helper isolates this to one place.

### 2. Snapshot enrichment (`dashboard/briefing_runner.py`)

While gathering the snapshot, register a `Linkable` for each person-bearing
record and stamp the matching `ref` next to that record in the JSON the model
sees. Phase-1 person sources:
- inbox oldest senders (have email)
- Practice Better invoice clients
- GHL contacts where an email is present

Attach the registry to the snapshot so refs are known to both the prompt and
the persistence step.

### 3. Prompt instruction

Add the ref-as-markdown-link rule (above) to each briefing prompt. Phrased so
the model only links records carrying a `ref` and never fabricates one.

### 4. Render + resolve

- **Persist** the registry as a sidecar `{slug}.links.json` alongside the
  existing `{slug}.md` (`dashboard/intelligence.py`). The briefing-serve
  endpoint returns `{markdown, links}`.
- **Client** (`static/dashboard.html`, in/after `mdRender()`): after
  markdown→HTML, find anchors with `href^="ref:"`, look each up in `links`, set
  the real `url` + a `.rec-link` class. Unknown/missing ref → replace the anchor
  with its text content (unwrap). A delegated click handler appends the console
  `key` at click time, reusing the existing `actNavigate` key logic, then
  navigates.

## Data flow

```
briefing_runner: gather snapshot
  → build LinkRegistry (person Linkables, refs r1..rN, deduped)
  → stamp ref onto each record in the snapshot JSON
  → LLM generates markdown containing [Jane Doe](ref:r3)
  → intelligence.py persists {slug}.md + {slug}.links.json
serve endpoint → {markdown, links}
dashboard.html mdRender → <a href="ref:r3">Jane Doe</a>
  → resolve ref:r3 via links → href=/console/...  + class=rec-link
  → (unknown ref → unwrap to text)
click → append console key → navigate to person record
```

## Backward compatibility

Additive. Existing briefings (no refs, no sidecar) render exactly as today —
the resolver only acts on `href^="ref:"` anchors and a missing `links` map is a
no-op. No flag needed because degradation is graceful; a flag is optional if a
safer rollout is wanted.

## Testing

Both hard-won rules baked in:

- **Unit:** registry ref-assignment + `(type,url)` dedup; `person_url`;
  resolver (known ref → anchor with real href; unknown ref → unwrapped to text;
  no-ref markdown untouched).
- **Real-shape test:** feed a realistic snapshot, assert the refs injected into
  the JSON match the registry — guards the *mock-masked-green-tests* trap (a
  mock that mirrors a wrong assumption passes while prod is broken).
- **Render-verify (not "script served"):** render `dashboard.html` headless,
  assert a ref-link briefing produces a clickable anchor with the right
  `/console/...` href **and zero console errors**. (The render-verify rule: a
  frontend change is confirmed by rendering the DOM, not by confirming the
  script is injected.)
- **LLM compliance** is not unit-testable → graceful degradation is the safety
  net; plus a manual check that a freshly generated briefing actually contains
  ref links.

## Open items resolved during planning

- Exact canonical `person_url` destination (verify against live routes).
- Whether the serve endpoint already returns a JSON envelope or raw markdown
  (adjust `{markdown, links}` shape to match existing contract).
