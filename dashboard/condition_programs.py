"""Slice 1 of Condition Support Programs: sqlite store for Glen's 9
authored eye-condition support programs. Pure sqlite, no Flask.

`seed_if_empty` loads Glen-approved content from data/condition_programs_seed.json
exactly ONCE per store, tracked by a persisted `_seed_state` marker — never on
a row-count check. It never overwrites an operator's edit made via the console
editor (`upsert`), and it never resurrects rows an operator intentionally
cleared out. Ground truth once seeding has run is whatever the operator has
saved, not the seed file.
"""
import json
from datetime import datetime, timezone

from dashboard import db


def _now():
    return datetime.now(timezone.utc).isoformat()


_SEED_NAME = "condition_programs"

# Clinical (not alphabetical) display order Glen authored these programs in.
# A key not present here sorts after all listed keys, by key, so an
# operator-added program never disappears — it just lands at the end.
_CLINICAL_ORDER = [
    "glaucoma-elevated-iop", "glaucoma-normal-iop", "dry-amd", "wet-amd",
    "senile-cataract", "psc-cataract", "dry-eye", "retinitis-pigmentosa",
    "diabetic-retinopathy",
]


def _clinical_sort_key(condition_key):
    try:
        return (0, _CLINICAL_ORDER.index(condition_key))
    except ValueError:
        return (1, condition_key)


def init_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS condition_programs (
            condition_key TEXT PRIMARY KEY,
            label TEXT,
            consult_recommended INTEGER NOT NULL DEFAULT 0,
            items_json TEXT NOT NULL DEFAULT '[]',
            updated_at TEXT
        )""")
    if not db.column_exists(cx, "condition_programs", "modifiers_json"):
        cx.execute("ALTER TABLE condition_programs "
                   "ADD COLUMN modifiers_json TEXT NOT NULL DEFAULT '[]'")
    _ensure_seed_state_table(cx)


def _ensure_seed_state_table(cx):
    # Shared-shape marker table (also created by dashboard/broad_benefit.py)
    # tracking which stores have been seeded — ONCE — regardless of their
    # current row count. See seed_if_empty below.
    cx.execute("""
        CREATE TABLE IF NOT EXISTS _seed_state (
            name TEXT PRIMARY KEY,
            seeded_at TEXT
        )""")


def _row(r):
    if r is None:
        return None
    return {
        "condition_key": r["condition_key"],
        "label": r["label"],
        "consult_recommended": bool(r["consult_recommended"]),
        "items": json.loads(r["items_json"] or "[]"),
        "modifiers": json.loads((r["modifiers_json"] if "modifiers_json" in r.keys() else None) or "[]"),
        "updated_at": r["updated_at"],
    }


def get(cx, key):
    r = cx.execute("SELECT * FROM condition_programs WHERE condition_key=?",
                   (key,)).fetchone()
    return _row(r)


def all(cx):
    """All programs in clinical order (see _CLINICAL_ORDER), not alphabetical.
    Any program whose key isn't in that fixed list sorts last, by key."""
    rows = cx.execute("SELECT * FROM condition_programs").fetchall()
    parsed = [_row(r) for r in rows]
    parsed.sort(key=lambda p: _clinical_sort_key(p["condition_key"]))
    return parsed


def upsert(cx, key, label, consult_recommended, items, modifiers=None):
    now = _now()
    cx.execute("""
        INSERT INTO condition_programs
            (condition_key, label, consult_recommended, items_json, modifiers_json, updated_at)
        VALUES (?,?,?,?,?,?)
        ON CONFLICT(condition_key) DO UPDATE SET
            label=excluded.label,
            consult_recommended=excluded.consult_recommended,
            items_json=excluded.items_json,
            modifiers_json=excluded.modifiers_json,
            updated_at=excluded.updated_at
        """, (key, label, 1 if consult_recommended else 0,
              json.dumps(items or []), json.dumps(modifiers or []), now))
    cx.commit()


def seed_if_empty(cx, seed_dict):
    """Insert each program from seed_dict exactly ONCE ever, tracked by a
    persisted `_seed_state` marker (not the table's current row count).
    Idempotent (safe to call on every request) and never re-seeds after the
    first attempt — including after an operator edit via the console, or
    after the table was emptied out some other way.

    On that first-ever attempt, seeding still only inserts rows if the table
    is currently empty (preserves the original guard against clobbering rows
    already present some other way) — but the marker is recorded either way."""
    _ensure_seed_state_table(cx)
    already = cx.execute("SELECT 1 FROM _seed_state WHERE name=?",
                          (_SEED_NAME,)).fetchone()
    if already:
        return
    now = _now()
    (count,) = cx.execute("SELECT COUNT(*) FROM condition_programs").fetchone()
    if count == 0:
        for key, prog in (seed_dict or {}).items():
            cx.execute("""
                INSERT OR IGNORE INTO condition_programs
                    (condition_key, label, consult_recommended, items_json, modifiers_json, updated_at)
                VALUES (?,?,?,?,?,?)
                """, (key, prog.get("label") or "",
                      1 if prog.get("consult_recommended") else 0,
                      json.dumps(prog.get("items") or []),
                      json.dumps(prog.get("modifiers") or []), now))
    cx.execute("INSERT OR IGNORE INTO _seed_state (name, seeded_at) VALUES (?,?)",
               (_SEED_NAME, now))
    cx.commit()


def ensure_program(cx, key, label, items, consult_recommended=False):
    """Idempotent-ensure for a single program key, INDEPENDENT of the
    once-ever seed_if_empty marker. Inserts the program ONLY if that key is
    currently absent; if the row already exists (whether from the original
    seed file, a prior ensure_program call, or an operator's console edit),
    it is left completely untouched -- ground truth once a key exists is
    whatever is stored, not this call's arguments. Safe to call on every
    request/boot."""
    if get(cx, key) is not None:
        return
    now = _now()
    cx.execute("""
        INSERT INTO condition_programs
            (condition_key, label, consult_recommended, items_json, modifiers_json, updated_at)
        VALUES (?,?,?,?,?,?)
        ON CONFLICT(condition_key) DO NOTHING
        """, (key, label or "", 1 if consult_recommended else 0,
              json.dumps(items or []), json.dumps([]), now))
    cx.commit()


_DRY_EYE_MIGRATION_NAME = "dry_eye_modifiers_v1"


def _find_item(items, slug):
    for it in items or []:
        if (it.get("slug") or "") == slug:
            return dict(it)
    return None


def migrate_dry_eye_modifiers(cx):
    """ONE-TIME, marker-guarded restructure of the `dry-eye` program for
    stores seeded long ago (before seed_if_empty's marker fired) with the OLD
    flat shape -- aces-eye-drops, moisturize, wholomega, and
    moisture-eyes-night-oil all as unconditional base items. Rewrites it to
    base items (aces-eye-drops, wholomega) + two client-reported modifiers
    (aqueous_deficiency -> Moisturize, severe -> Moisture Eyes Night Oil),
    via the existing `upsert`.

    Tracked by its OWN `_seed_state` marker (distinct from seed_if_empty's),
    checked first -- so this runs AT MOST ONCE EVER, exactly like
    seed_if_empty. That is the whole point: if Glen later removes one of
    these items via the console editor, this migration must never resurrect
    it on a later call/boot. If the `dry-eye` program doesn't exist at all,
    there is nothing to migrate -- the marker is still recorded so we never
    look again.

    Whatever name/alts/dose the existing moisturize and
    moisture-eyes-night-oil items already carry (including any operator
    customization made before this migration ever ran) is preserved verbatim
    into the new modifier items; only the shape (base vs. modifier) changes."""
    _ensure_seed_state_table(cx)
    already = cx.execute("SELECT 1 FROM _seed_state WHERE name=?",
                          (_DRY_EYE_MIGRATION_NAME,)).fetchone()
    if already:
        return
    now = _now()
    prog = get(cx, "dry-eye")
    if prog is not None:
        old_items = prog.get("items") or []
        aces = _find_item(old_items, "aces-eye-drops") or {
            "slug": "aces-eye-drops", "name": "ACES Eye Drops",
            "alts": [{"slug": "ocuheal-eye-drops", "name": "OcuHeal Eye Drops"}]}
        wholomega = _find_item(old_items, "wholomega") or {
            "slug": "wholomega", "name": "WholOmega", "dose": "4 capsules/day"}
        moisturize = _find_item(old_items, "moisturize") or {
            "slug": "moisturize", "name": "Moisturize"}
        night_oil = _find_item(old_items, "moisture-eyes-night-oil") or {
            "slug": "moisture-eyes-night-oil", "name": "Moisture Eyes Night Oil",
            "alts": [{"slug": "moisture-eyes-night-drops",
                      "name": "Moisture Eyes Night Drops"}]}
        new_items = [aces, wholomega]
        new_modifiers = [
            {"when": "aqueous_deficiency", "action": "add",
             "source": "client-reported", "client_default": True,
             "items": [moisturize]},
            {"when": "severe", "action": "add",
             "source": "client-reported", "client_default": False,
             "items": [night_oil]},
        ]
        upsert(cx, "dry-eye", prog["label"], prog["consult_recommended"],
               new_items, new_modifiers)
    cx.execute("INSERT OR IGNORE INTO _seed_state (name, seeded_at) VALUES (?,?)",
               (_DRY_EYE_MIGRATION_NAME, now))
    cx.commit()


def resolve_program_items(program, audience="client", client_facts=None):
    """Apply a program's modifiers to its base items; return the resolved list.

    modifier = {when, action:"add"|"remove", items:[{slug,name?,dose?},...],
                source:"diagnosis-implied"|"clinician-measured"|"client-reported",
                client_default:bool}
    A modifier is ACTIVE when:
      - diagnosis-implied: client_default is True; for audience="practitioner",
        composer_default when the modifier sets it, else client_default
      - client-reported:   client_facts[when] is truthy
      - clinician-measured: never (client suppresses; the practitioner surface
        handles these as explicit toggles, not via this resolver)
    `audience` selects the default set: "client" uses client_default throughout
    (the money path); "practitioner" additionally honors a modifier's optional
    composer_default for diagnosis-implied modifiers, falling back to
    client_default when it is absent. add de-dupes against present slugs;
    remove drops by slug."""
    client_facts = client_facts or {}
    base = [dict(it) for it in (program.get("items") or [])]
    remove_slugs, additions = set(), []
    for mod in (program.get("modifiers") or []):
        source = mod.get("source")
        if source == "diagnosis-implied":
            if audience == "practitioner" and "composer_default" in mod:
                active = bool(mod.get("composer_default"))
            else:
                active = bool(mod.get("client_default"))
        elif source == "client-reported":
            active = bool(client_facts.get(mod.get("when")))
        else:
            active = False
        if not active:
            continue
        if mod.get("action") == "remove":
            for it in (mod.get("items") or []):
                s = (it.get("slug") or "").strip()
                if s:
                    remove_slugs.add(s)
        elif mod.get("action") == "add":
            additions.extend(dict(it) for it in (mod.get("items") or []))
    resolved = [it for it in base if (it.get("slug") or "").strip() not in remove_slugs]
    present = {(it.get("slug") or "").strip() for it in resolved}
    for it in additions:
        s = (it.get("slug") or "").strip()
        if s and s not in present:
            resolved.append(it); present.add(s)
    return resolved
