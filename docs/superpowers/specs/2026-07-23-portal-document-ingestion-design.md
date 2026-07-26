# Portal Document Ingestion — clients and console upload medical records; AI drafts facts + narrative behind one review gate — Design

**Date:** 2026-07-23
**Status:** Approved in brainstorm with Glen 2026-07-23
**Repo:** deploy-chat

## Problem

Clients arrive with medical records — lab panels, imaging reports, specialist letters, discharge summaries. Today there is **no way to get them into the system**. They reach Glen by email, fax, or handoff and stay in his inbox. Nothing they contain reaches the clinical stores, the analysis, or remedy matching, and the client never sees anything back.

This spec adds the full path: **upload → AI extraction → Glen's review → live attributes + client-facing narrative**.

Two things happen to an uploaded record, both confirmed with Glen:

- **(b) Structured extraction** — diagnoses, conditions, boolean intake facts become *proposals* that, once approved, land in the canonical clinical stores.
- **(c) Client narrative** — a plain-language summary the client reads in their own portal, alongside their biofield reports.

**Both sit behind Glen's console review.** Nothing auto-publishes and nothing auto-writes to live clinical stores.

## Scope boundary — this is 1 of 3 sub-projects

The originating request decomposed into three independent projects. This spec is **#1** only.

- **#1 (this spec)** — Portal document ingestion.
- **#2 — Multi-photo Body Map + photo/map alignment editor.** `client_photos` becomes many-per-client with a saved per-photo transform, and the Body Map gains an alignment adjustment UI. Shares no code with #1. Nothing here touches `static/body-map.js`.
- **#3 — CTI-2: make `canonical_tags.person_attributes` authoritative.** Wires the in-house canonical attribute store into the readers that feed analysis and remedy matching. **This spec depends on #3 for part (b) to fully pay off** (see "What part (b) delivers, honestly").

## Boundary vs. adjacent systems

Each of these was read before being ruled in or out. The details matter, because the obvious targets turned out to be wrong.

- **`dashboard/client_conditions.py`** — `email TEXT PRIMARY KEY`, i.e. **one** condition per client. It is specifically the *operator override for the eye-condition support-program key*, not a diagnosis list. **Not a write target here.** Bulk-writing extracted diagnoses would clobber the support-program selection, each approval overwriting the last. Glen already has a dedicated console control for it (`/api/console/client-condition`); document approval leaves it alone.
- **`dashboard/client_facts.py`** — **boolean only** (`value INTEGER NOT NULL DEFAULT 0`; `set_fact` coerces `1 if value else 0`). Holds flags such as `on_areds2` that drive client-reported condition-program modifiers. **A valid write target, but only for boolean facts in the existing vocabulary.** It cannot hold a lab value or a medication name.
- **`dashboard/canonical_tags.py`** (`person_attributes`) — the correct final home for extracted clinical attributes: multi-valued `DISCRETE_FIELDS = (tags, conditions, terrain_concerns, body_systems)`, alias resolution through `canonical_vocab`, `UNIQUE(email, field, value_norm)` idempotency, and a `source=` parameter. **Currently dark** — its only importers are tests; the docstring states "no caller is wired yet (CTI-2 makes it authoritative)". This spec writes to it anyway (see below).
- **`client_360.client_tags_for_email`** — the clinical tags that inform Glen's work today are read from the synced, read-only **`e4l.db`**, computed from voice scans by the clinical tagger. **Never a write target for document extraction.**
- **`dashboard/analysis_requests.py`** — the `req → ai_draft → confirmed` ledger this spec's review gate deliberately mirrors. Not reused directly (different payload); the shape and the discipline are the precedent.
- **`/api/cert/upload`**, **`/api/biofield/photo`** — the existing upload routes. Both write bytes under `$DATA_DIR`. **That pattern is rejected here** (see finding 1).
- **`dashboard/portal_library.py`** — the My Library tile. Its `{enabled, items}` shape plus token-scoped asset streaming is the model the My Records tile copies.

## What part (b) delivers, honestly

Confirmed with Glen (option "3+1"): extraction writes to the **correct final home** even though one of those homes is not yet read.

| Extracted item | Write target on approval | Feeds matching today? |
|---|---|---|
| Boolean intake facts (e.g. `on_areds2`) | `client_facts.set_fact` | **Yes** — live consumers exist |
| Conditions, tags, terrain concerns, body systems | `canonical_tags.set_attr(..., source='document:<id>')` | **Not yet** — lights up when CTI-2 (#3) lands |
| Labs with numeric values, medications | **No structured store.** Shown in the narrative and on Glen's review screen | No |
| Eye-condition support-program key | **Not written.** Glen's existing console control owns it | n/a |

The deliberate calls:

- **Conditions are written dark.** They accumulate correctly in `person_attributes` from day one, stamped with document provenance, and become live for matching the moment CTI-2 wires the store. Zero blast radius today precisely because nothing reads it yet.
- **No numeric labs store is invented.** Nothing would read it. Labs reach Glen through the review screen and the client through the narrative. Building a store with no consumer is waste; if a consumer ever appears, the extraction records still hold the source data to backfill from.

This table is the spec's honesty contract: an extracted *diagnosis* does not influence remedy matching until #3 ships. Part (c) is unaffected and works on day one.

## Two verified findings that shape the design

Checked against running code, not assumed.

### 1. The disk is not an option

`$DATA_DIR` is a Render **persistent disk mounted on the web service only** (`render.yaml:90`). Prod cut over to multi-instance Postgres on 2026-07-22. An upload landing on instance A writes a file instance B cannot serve, so the cert/biofield "write bytes to `$DATA_DIR`" pattern would produce documents that intermittently 404. Bytes must live **in the database**, readable by every instance and inside the existing backup.

### 2. A runtime-created `BLOB` column would fail on Postgres

`client_photos` is the codebase's only `BLOB` column, and it survived the Postgres cutover only because the **one-time migration** path translates the type (`scripts/pgmig/schema_create.py:_translate_blob_type`, "Postgres has no BLOB type"). The **runtime** adapter does not:

```python
>>> pgcompat.translate_sql("CREATE TABLE ... (blob BLOB)")
'CREATE TABLE ... (blob BLOB)'          # BLOB passes through untranslated
```

A new table created at runtime by `init_table(cx)` with a `BLOB` column therefore **fails outright on Postgres** (`type "blob" does not exist`).

**Resolution: declare the column `BYTEA`.** Verified to round-trip Python `bytes` losslessly on SQLite — the full 0–255 byte range including NULs — while being the native binary type on Postgres. Zero changes to shared infra; blast radius confined to the new table. A test pins the round-trip so the subtlety cannot silently regress.

**Separately noted, not fixed here:** `client_photos.init_table` carries the same latent defect. On a *fresh* Postgres database it would fail to create; it works today only because the table already exists, migrated. Worth a one-line follow-up (`BLOB` → `BYTEA` there too, or porting the translation into runtime `pgcompat`), deliberately kept off this spec's critical path.

## Data model

Two new tables, created at runtime via the established `init_table(cx)` idiom.

### `client_documents` — the file

```
id             INTEGER PRIMARY KEY AUTOINCREMENT
email          TEXT NOT NULL            -- lowercased, the owning client
filename       TEXT
content_type   TEXT
byte_size      INTEGER
sha256         TEXT                     -- dedup: same bytes for same email = one row
blob           BYTEA                    -- the file itself (see finding 2)
source         TEXT                     -- 'portal-self' | 'console'
uploaded_at    TEXT
extract_status TEXT                     -- 'pending' | 'drafted' | 'failed' | 'skipped-unreadable'
```

Index on `(email)`, UNIQUE index on `(email, sha256)` so re-uploading an identical file is an idempotent no-op rather than a duplicate — the cross-process idempotency pattern (UNIQUE + `INSERT OR IGNORE`), which the adapter already translates to `ON CONFLICT DO NOTHING`.

### `client_document_extractions` — the AI draft

```
id              INTEGER PRIMARY KEY AUTOINCREMENT
document_id     INTEGER NOT NULL         -- -> client_documents.id
email           TEXT NOT NULL
status          TEXT                     -- 'ai_draft' | 'confirmed' | 'rejected'
narrative_md    TEXT                     -- part (c), client-facing
attributes_json TEXT                     -- [{field, value, source_quote}] -> person_attributes
facts_json      TEXT                     -- [{fact_key, value(bool), source_quote}] -> client_facts
unstructured_json TEXT                   -- [{label, value, source_quote}] labs/meds: display only
model           TEXT
created_at      TEXT
reviewed_at     TEXT
reviewed_by     TEXT
```

The three payload columns map exactly onto the three rows of the honesty table: attributes go to `person_attributes`, booleans to `client_facts`, and everything else is display-only. Keeping them separate at the schema level makes it impossible for a display-only lab value to silently acquire a write path later without a deliberate change.

All three are **proposals**, never live data, read only by the console review screen. The live stores are written exclusively at approval.

`field` in `attributes_json` is constrained to `canonical_tags.ALL_FIELDS`; `set_attr` rejects anything else by design, so an out-of-vocabulary field fails closed rather than writing junk.

### The fabrication guard

Every extracted item carries a **`source_quote`** — the verbatim span from the document it came from. This is the structural answer to the known failure mode where the model invents plausible clinical facts. Two enforcement layers:

1. **Prompt-level** — the model must return a `source_quote` per item; items without one are dropped.
2. **Verification-level** — each `source_quote` is checked against the document's extracted text. **An item whose quote does not appear in the source is dropped from the draft**, and the drop is logged.

Glen's review then sees the quote beside each proposal, so a survivor that is still wrong is visible at a glance rather than buried.

## Storage module — `dashboard/client_documents.py`

Persistence only. No HTTP, no rendering, no AI. Mirrors the shape of `client_photos.py`.

```
init_table(cx)
put(cx, email, blob, filename, content_type, source) -> {id, deduped: bool}
get(cx, doc_id)                  -> row or None (includes blob)
get_for_email(cx, doc_id, email) -> row or None   # scoped read; the isolation primitive
list_for_email(cx, email)        -> [row without blob, ...]
set_extract_status(cx, doc_id, status)
pending(cx, limit)               -> docs awaiting extraction
```

`get_for_email` is the **single scoping primitive**: every client-facing read goes through it, so a token can only ever reach its own owner's documents. Serving routes never call `get` directly.

Extraction drafts get a sibling module `dashboard/document_extractions.py` with the same discipline (`init_table`, `put_draft`, `get_for_document`, `confirm`, `reject`).

## Upload — two front-ends, one shared validator

A shared helper (`_accept_document_upload(cx, email, file_storage, source)`) does validation, hashing, storage, and enqueue, so both routes stay thin and cannot drift apart.

**Validation:**
- **Accepted for extraction:** `application/pdf` and `image/*` (JPG, PNG, WEBP, HEIC).
- **Stored but not extracted:** anything else, marked `extract_status='skipped-unreadable'`. Glen confirmed a potentially wide range of file types arrives; the rule is *store everything, extract what is readable*. Word docs are explicitly out of scope for extraction in v1; print-to-PDF handles the occasional one.
- **Size cap:** 30 MB (imaging PDFs are the large case). Over-cap is a 400, never a truncated store.

**Routes:**

- `POST /api/portal/<token>/documents` — client self-serve. Token-scoped: resolves the token to its owner's email via `_portal_record_for` and writes **only** that email, exactly as the existing `/api/portal/<token>/photo` upload does. `source='portal-self'`.
- `POST /api/console/client-document` — console-gated (the established console-key/owner-token check), takes an explicit `email`. `source='console'`. For records arriving by email or fax.

## Extraction pipeline — `dashboard/document_extract.py`

Mirrors the analysis-autoconfirm worker rather than inventing a new job system.

1. Select `client_documents` where `extract_status='pending'`.
2. Send the file to **Claude Opus (vision)** — PDFs and images both natively. One call returns structured JSON: `attributes[]`, `facts[]`, `unstructured[]`, `narrative_md`.
3. Apply the fabrication guard (drop items whose `source_quote` is absent from the source text).
4. Resolve each attribute's `value` through `canonical_tags.resolve(cx, field, value)` so the draft shows Glen the **canonical** form he will actually be approving, not the document's raw wording.
5. Write one `client_document_extractions` row at `status='ai_draft'`; set the document to `drafted`.
6. On failure → `extract_status='failed'`, no draft row, error logged. Re-runnable.

**This step writes nothing to `client_facts` or `person_attributes`.** That is the whole point of the gate.

## Console review — one screen, one approval

The `/console/client` page gains a **Documents** section. Per drafted document:

- the **raw file**, viewable inline (PDF embed / image),
- **proposed attributes** and **proposed boolean facts** as pre-checked checkboxes, each showing its `source_quote`,
- **labs and medications** listed read-only, clearly marked as not stored structurally,
- the **narrative** in an editable textarea.

**Approve** (`POST /api/console/client-document/<id>/approve`) does it all in one transaction:
- checked attributes → `canonical_tags.set_attr(cx, email, field, value, source='document:<id>')`
- checked booleans → `client_facts.set_fact(cx, email, fact_key, value)`
- edited narrative saved to the extraction row
- `status='confirmed'`, `reviewed_at` / `reviewed_by` stamped → the narrative becomes visible in the client's portal

Unchecked items are simply not written. **Reject** discards the draft and keeps the file: the document stays, only the AI's reading of it is thrown away. Re-extraction of a rejected document is possible by re-queuing.

`source='document:<id>'` makes provenance traceable: any attribute that came from a record traces to the exact file and the exact approval. It is also what lets CTI-2 later distinguish document-derived attributes from scan-derived or manually-entered ones.

## Client portal — the "My Records" tile

Gated behind `PORTAL_HUB_ENABLED`, the same flag as the other hub tiles, so it ships dark and lights up on a flag flip.

`GET /api/portal/<token>/documents` returns `{enabled, items: [...]}`, mirroring `/api/portal/<token>/library`. Each item carries `filename`, `uploaded_at`, `status`, a `file_url`, and — once approved — `narrative_md`.

`GET /api/portal/<token>/documents/<id>/file` streams the raw bytes, resolved through `get_for_email` so a token can only fetch its own owner's file. Headers `Cache-Control: private, no-store`, matching the existing photo-serving route.

**What the client sees:**
- Always: their uploaded file, with a **working download link to their own raw document** (confirmed with Glen).
- Before approval: *"Received — under review."*
- After approval: the **narrative**, rendered below the file.
- **Never:** the extracted attributes, facts, or labs. The client sees their own document and a reviewed plain-language summary, never a half-baked machine reading of their medical record.

## Error handling

- Upload with no file or empty bytes → 400.
- Over-cap → 400; unreadable type → stored unextracted. Never a partial write.
- Extraction failure → document marked `failed`; the client still sees "Received — under review" and can still download their file; Glen sees it in the console as needing attention. A failed extraction never blocks storage or download.
- Unknown or expired token → 404, consistent with the other portal routes.
- Approval on an already-confirmed draft → idempotent no-op rather than duplicate writes. (`set_attr`'s `UNIQUE(email, field, value_norm)` makes the attribute half naturally idempotent; the status check covers the rest.)

## Testing

**Store (`client_documents`)**
- put → get round-trip, **including a bytes round-trip through a `BYTEA` column** (pins finding 2).
- dedup: identical bytes for the same email insert once.
- `list_for_email` excludes blobs; `get_for_email` returns None for a non-owner.

**Upload routes**
- size cap and type rules; unreadable type stores with `skipped-unreadable`.
- **token scoping: token A cannot upload to or read B's documents** (the isolation test that matters most).
- console route rejects without a valid console key.

**Extraction**
- mocked model response → correct draft row shape; live stores untouched.
- **fabrication guard: an item whose `source_quote` is absent from the source is dropped.**
- attribute values are canonicalized via `resolve` before being drafted.
- failure path marks `failed` and writes no draft.

**Review**
- approve writes checked attributes to `person_attributes` with `source='document:<id>'`, checked booleans to `client_facts`, and the narrative;
- **approve does NOT write `client_conditions`** (guards the support-program override against regression);
- unchecked items are not written;
- reject discards the draft and keeps the document;
- re-approval is idempotent.

**Portal**
- pre-approval shows "under review" and no narrative; post-approval shows narrative;
- raw-file download works for the owner and 404s cross-token;
- tile respects `PORTAL_HUB_ENABLED`.

**CI conventions:** pin the product catalog per the `$DATA_DIR`-strips-catalog rule, and provide dummy `OPENAI`/`PINECONE` keys for any app-importing test module, per the established collection-error rule.

## Explicitly out of scope for v1

Confirmed with Glen:

- **CTI-2 wiring** (#3). Extracted attributes are written to their correct home but are not yet read by matching. Its own project.
- **No automatic remedy re-match** on approval. Glen runs matching as he does today.
- **No numeric labs / medication store.** Display-only until a consumer exists.
- **No OCR fallback** for files the vision path cannot read. Stored, marked unreadable; print-to-PDF is the workaround.
- **No Word/docx extraction.**
- **No external object storage.** Postgres `bytea` is the v1 store; S3/R2 is a clean migration later if volume ever demands it, and nothing here forecloses it — the storage module is the only thing that touches bytes.
