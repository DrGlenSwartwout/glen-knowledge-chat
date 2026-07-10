# Species + the animal greeting — design (Slice 4)

**Date:** 2026-07-10
**Status:** design, ready for a plan
**Repos:** vault `~/AI-Training` (scrape) + deploy-chat (portal)
**Builds on:** scan-recommendations Slices 1–2 (#761/#764), dependent-TOS (#767, dark)

---

## Problem

An animal has its own E4L account and its own portal. When its portal renders, it greets
"Aloha Sasha" — addressed to a cat. Glen's rule: an animal's page should read **"Give our
Aloha to Sasha"** — the human reading it is the caregiver, not the animal.

This becomes visible the moment `DEPENDENT_TOS_ENABLED` (#767) flips: that fix un-blanks an
animal's own portal, and the first thing it will render is the wrong greeting. So the two
slices pair — this one supplies the correct greeting for the portals #767 unblocks.

The blocker: **we do not know who is an animal.** Species is not in the scan PDF and not in
the current `e4l.db` (`e4l_clients` has no `species` column — Slice 0's reparse rebuilt that
table without it). The E4L web portal holds it, and `fetch-e4l-details.py` already knows how
to scrape it — but it has never populated it.

## Decisions (resolved with Glen, 2026-07-10)

1. **Scope: greeting + species plumbing only.** The infoceutical-only recommendation rule for
   animals is deferred to Slice 3, where the FF matches it acts on will exist.
2. **Scrape scope: the 162 clients with a portal + pushed recommendations** — exactly the set
   that can see a greeting today. Others get scraped as they gain portals.
3. **Species values:** E4L's dropdown is `Human`, `Cat`, `Dog`, `Horse`. For any other mammal,
   **an operator types a free-text species** in our system — E4L cannot represent it.
4. **`is_animal` = species is set AND `species.lower() != "human"`.** So `Cat`/`Dog`/`Horse`
   and any operator-typed value ("Rabbit", "Goat") are animals; blank or `Human` is not.

## What we verified (2026-07-10)

- `e4l_clients` columns: `client_id, name, email, phone, date_of_birth, notes, ghl_contact_id,
  archived_at`. **No `species`, `animal_name`, or `detail_scraped`.**
- Species is NOT in the scan PDF text (checked Sasha's; only boilerplate).
- `02 Skills/fetch-e4l-details.py` (Playwright, E4L login) scrapes the client View/Edit page
  and extracts `species: getSelectText('SpeciesID')` and `animal_name: get('AnimalName')`, and
  writes `species = ?` to `e4l_clients` — but that column does not exist in the live db, so it
  has never run against it. It scopes by `WHERE detail_scraped = 0`.
- The portal greeting is `static/client-portal.html:650`: `Aloha ${esc(first)}`, where `first`
  is the first token of `d.name`. `d.name` = `_member_name or portal.get("name")` — already
  **member-aware** (#750), so the greeting follows `?member=`.
- Sasha's `AnimalName` in E4L is "Sasha". (Her account was formerly registered under "Karin
  Takahashi"; Glen renamed it at energyforlife.com on 2026-07-10, so the durable source is now
  correct.) The greeting should still prefer `animal_name` — it is the field E4L designates for
  the animal, independent of whatever the account name happens to be.

## Architecture

Mirrors Slice 1's spine — local source → push → prod table → payload → render — because that
is the proven path for data that lives on Glen's Mac.

### Part A — vault: get species into `e4l.db`

1. **Schema:** add `species TEXT`, `animal_name TEXT`, `detail_scraped INTEGER DEFAULT 0` to
   `e4l_clients` (in `setup-e4l-db.py`, idempotent `ALTER`s guarded by try/except, matching the
   pattern Slice 0 used).
2. **Scope the scrape to the 162:** mark everyone `detail_scraped = 1` (skip), then set
   `detail_scraped = 0` for the clients who have a scan we pushed to prod. `fetch-e4l-details.py`
   then scrapes exactly those.
3. **Run** `fetch-e4l-details.py` (operator step — needs `E4L_USERNAME`/`E4L_PASSWORD`, ~162
   Playwright page loads). It populates `species` + `animal_name`.

### Part B — deploy-chat: get it to the portal

Mirror `client_scans` exactly.

1. **`dashboard/client_species.py`** — pure sqlite store. Table
   `client_species(email PK, species, animal_name, synced_at)`. `upsert(cx, email, species,
   animal_name)`; `get(cx, email) -> {species, animal_name, is_animal}` where
   `is_animal = bool(species) and species.strip().lower() != "human"`.
2. **`POST /api/console/client-species/sync`** — console-gated, `_db_lock`, batch upsert.
   Mirrors `client-scans/sync`. Sends no email.
3. **`GET /api/console/client-species`** — console read path (no email → corpus counts; with
   `?email=` → that client's row), so the scrape's result can be confirmed in prod.
4. **`POST /api/console/client-species` (operator override)** — the "Other" path. An operator
   sets `species` (free text) and `animal_name` for one email, for a mammal E4L cannot
   represent, or to correct a scrape. Idempotent upsert into the same table.
5. **Local pusher** `02 Skills/e4l-species-push.py` — reads `e4l_clients` read-only, POSTs
   species + animal_name for the clients that have them. Wired into `e4l-daily-watch.sh` beside
   the manifest and recommendations pushes.
6. **Payload:** `api_client_portal` gains `payload["is_animal"]` and `payload["animal_name"]`,
   computed from `client_species.get(email_for_reports)` — **member-aware**, so a caregiver on
   the pet's tab gets the animal greeting. Best-effort; a failure never breaks the load.
7. **Greeting** (`static/client-portal.html:650`), flag-gated:

   ```javascript
   <div class="hello">${d.is_animal
       ? 'Give our Aloha to ' + esc(d.animal_name || first)
       : 'Aloha ' + esc(first)}</div>
   ```

   Falls back to `first` if `animal_name` is somehow blank, so an animal never renders "Give
   our Aloha to ".

### Flag

`ANIMAL_GREETING_ENABLED`, default OFF. Off → `payload` never gains the keys and the greeting
is byte-identical to today. **Flip it alongside `DEPENDENT_TOS_ENABLED`** — that flag makes
animal portals render, and this one makes them greet correctly; together they complete the
animal-portal experience.

## The "Other" operator path — detail

E4L's four values cover most cases. For a rabbit, a goat, a ferret, the operator opens the
console and sets `species: "Rabbit", animal_name: "Thumper"` for that client's email via
`POST /api/console/client-species`. `is_animal` is true for any non-blank non-"Human" species,
so the greeting works with no code change per new mammal. The scrape and the override write the
same `client_species` row; the override wins (it is the deliberate operator signal).

## Testing

- `is_animal`: `Cat`/`Dog`/`Horse`/`Rabbit` → true; `Human`/blank → false; case-insensitive.
- The greeting uses `animal_name`, not the account name (Sasha, not "Karin Takahashi").
- `?member=`: a caregiver on the pet's tab gets "Give our Aloha to Sasha"; on their own tab,
  "Aloha <caregiver>".
- Flag off → payload has no `is_animal`/`animal_name`; greeting unchanged (byte-identical).
- The operator override upserts and wins over a scraped value.
- The console read path with no email returns corpus counts and no client data.
- The pusher opens `e4l.db` read-only (proven by capturing the connect args, not by chmod).
- Render-verify in a browser: Sasha's portal greets "Give our Aloha to Sasha"; a human's is
  unchanged.

## Non-goals

- The infoceutical-only recommendation rule (Slice 3).
- Scraping beyond the 162 portal clients.
- Auto-detecting species from DOB or name — the E4L dropdown and the operator override govern.
- A self-serve client-facing species picker — species is set by the scrape or the operator.

## Open question

None. (The E4L-account-name durability concern is resolved: Glen renamed Sasha at
energyforlife.com on 2026-07-10, so the scrape reads the corrected name and `AnimalName`.)
