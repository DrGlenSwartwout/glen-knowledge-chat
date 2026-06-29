# Portal top chat — design

## Context
Each client/practitioner portal already has its own scoped chat, but it's buried mid-page.
Glen wants a **simple chat interface at the top of each portal page**, consistent across
all portals. This is a placement + consistency change — it reuses each portal's existing
chat backend and auth; it does **not** add new RAG plumbing or change what the assistants
know.

Decisions (confirmed with Glen):
- **Move the existing chat to the top** (not a slim expanding bar, not a duplicate). The
  chat becomes the first content block below the page header and scrolls normally (not a
  pinned/sticky bar).
- Scope = **all four portals**.

## Scope — the four portals (each keeps its own backend/auth/assistant)
| Portal | Page (static) | Existing chat backend | Change |
|---|---|---|---|
| Client | `client-portal.html` | `POST /api/portal/<token>/chat` (client scan + orders + Pinecone RAG, "Ask Dr. Glen", SSE stream) | Render the existing chat card **first** in `#app` |
| Practitioner | `practitioner-portal.html` | `POST /api/practitioner/chat` (catalog `scoped_reply`) | Move/ensure its chat at the top |
| Dropship | `practitioner-dropship.html` | practitioner scoped chat | Add a compact chat at the top wired to the practitioner backend |
| Patient dispensary | `practitioner-client.html` | `POST /api/client/<code>/chat` (catalog `scoped_reply`) | Move/ensure its chat at the top |

## Approach
- For each page, the chat block moves to be the **first** thing rendered after the page's
  header/greeting. For JS-injected pages (e.g. client-portal renders cards into `#app`),
  reorder so the chat card is appended first. For pages where the chat is static markup,
  move the markup up. Where a portal currently has **no** chat, add the same compact block
  wired to that portal's existing scoped endpoint.
- **Consistent markup** across all four: a card with a short title, a scrollable messages
  area, and an input row (text field + Send) — modeled on the client portal's existing
  chat. No restyling beyond what's needed for consistency.
- **No backend changes.** Each chat keeps its own endpoint, auth (token / practitioner
  session / dispensary code+consent), scoped assistant, and any consent gating.

## Out of scope
- No change to assistant knowledge, prompts, or models.
- No console widget change (Justus stays).
- Customer self-pay / invoice-page chat unchanged.
- No new sticky/pinned header behavior.

## Verification
For each of the four portals: render the page in a headless browser, confirm the chat is
the first block below the header, send one message and confirm a reply streams/returns,
and confirm **zero console errors**. Confirm no portal shows two chats (relocated, not
duplicated).
