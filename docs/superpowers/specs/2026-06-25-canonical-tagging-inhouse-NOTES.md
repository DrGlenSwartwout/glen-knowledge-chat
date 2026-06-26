# Canonical Tagging In-House (CTI) — roadmap + decisions

**Date:** 2026-06-25
**Status:** Strategic direction captured; decomposed; CTI-1 to be specced next. NOT yet a buildable spec.

## Why

Today **GHL is the source-of-truth** for a person's clinical fields: the hourly `console_push_cron.py:sync_people_from_ghl()` (cron `app.py:19799`) maps GHL custom fields → `people.{conditions, terrain_concerns, body_systems, challenges, goals}` and POSTs to `/api/people?merge_tags=1` — overwriting the local copy each hour. Only `tags` are union-merged; the only local auto-tagging is rule-based operational tags (`type:`/`consent:` via `classify_people()`). No AI auto-tagging persists to people (reply_watcher's `extracted_conditions`/`extracted_topics` stay on the `personal_email_feedback` row only).

Glen's decision (2026-06-25): **bring canonical tagging in-house** — the app becomes the authority for the clinical fields; AI/rules/manual write to an in-house canonical store; GHL becomes a consumer (push-out), not the owner. Aligns with the broader FMP→app migration direction (make the app DB authoritative). Mirror the proven in-house-canonical pattern: `dashboard/biofield_meanings.py` (canonical remedy meanings) + `/console/remedy-meanings` curation.

## Confirmed strategic decisions (2026-06-25)

- **Field scope:** the FULL clinical set — `tags`, `conditions`, `terrain_concerns`, `body_systems`, `challenges`, `goals` — all become app-canonical.
- **GHL relationship:** app is source-of-truth and **pushes OUT to GHL**; GHL stops overwriting these fields.
- **Vocabulary:** a CONTROLLED canonical vocabulary with alias-normalization (so "adrenal fatigue"/"Adrenal Fatigue"/"adrenal exhaustion" collapse to one canonical term), mirroring canonical remedy-meanings — not free-form.

## Decomposition (each its own spec → plan → build)

- **CTI-1 — Canonical store + vocabulary (foundation, spec FIRST):** in-house tables = (a) a controlled canonical vocabulary (canonical term + aliases + field-type tag|condition|terrain_concern|body_system|challenge|goal) with normalization; (b) per-person canonical attributes (person, field, canonical_value, **source** = manual|ai|ghl|rule|scan, added_at). A **one-time import** of the current GHL-sourced `people.*` values into the store (so nothing is lost when ownership flips). `/api/people` READS canonical values. Non-breaking, behind the scenes — does NOT yet change the GHL sync or writers.
- **CTI-2 — GHL relationship flip:** stop the hourly pull from overwriting the 6 fields; add a push of app-canonical → GHL (one-way out). App becomes authoritative.
- **CTI-3 — Writers:** manual console edit + **AI auto-tag-from-comms** (the former "AutoTag": reply_watcher's `extracted_conditions`/`extracted_topics` → canonical person attributes, source=ai) + intake/scan derivations. (Supersedes the standalone AutoTag idea + its "`ai:*` tags to survive GHL" workaround — no longer needed once the app owns the data.)
- **CTI-4 — Console curation UI:** curate the vocabulary + per-person tags + alias mappings (mirror `/console/remedy-meanings`: edit, propose-AI, "remember"/promote-to-canonical).

## Interactions

- **B3a** already mines `people.{tags,conditions,...}` into the stress loop — once those come from the canonical store, B3a's mining gets cleaner/canonical values for free (read path in CTI-1).
- **CovExt** (non-scan stress → remedy coverage) is independent of CTI; proceed separately.
- **AutoTag** is folded into CTI-3 (no longer a standalone increment).

## Open questions for the CTI-1 spec

- Exact table shapes (vocab + per-person attributes); how aliases resolve (normalize + alias map, like remedy-meanings `_resolve_*`).
- Whether `/api/people` reads canonical-store values transparently (preferred) or the store shadows the existing `people` columns during transition (FMP-migration pattern: generate the file/columns one-way from the store).
- Import provenance: tag imported values `source='ghl'` so later GHL-flip logic knows which were externally sourced.
- How `tags` (operational `type:`/`consent:` + clinical) coexist in one store vs separate field-types.
