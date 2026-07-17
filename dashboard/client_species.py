"""Per-client species + animal name, mirrored from the local e4l.db into prod so the
portal can greet an animal correctly. Pure sqlite; no Flask, no network.

Source: the E4L scrape (Human/Cat/Dog/Horse + AnimalName), or an operator override for
any other mammal E4L cannot represent. is_animal = set and not Human — so a new species
needs no code change.
"""
import sqlite3
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def _norm(email):
    return (email or "").strip().lower()


def is_animal(species):
    return bool(species) and (species or "").strip().lower() != "human"


def init_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS client_species (
            email        TEXT PRIMARY KEY,
            species      TEXT,
            animal_name  TEXT,
            synced_at    TEXT
        )
    """)
    cx.commit()


def upsert(cx, email, species, animal_name):
    e = _norm(email)
    if not e:
        return
    cx.execute(
        "INSERT INTO client_species (email, species, animal_name, synced_at) "
        "VALUES (?,?,?,?) ON CONFLICT(email) DO UPDATE SET "
        "species=excluded.species, "
        # Never let a blank incoming name erase a name we already have: a re-sync
        # from E4L whose AnimalName field is empty must not wipe an operator-set
        # animal_name back to '' (which would drop the greeting to the account-name
        # fallback). A non-empty incoming name still overwrites (corrections apply).
        "animal_name=CASE WHEN excluded.animal_name <> '' THEN excluded.animal_name "
        "                 ELSE client_species.animal_name END, "
        "synced_at=excluded.synced_at",
        (e, (species or "").strip(), (animal_name or "").strip(), _now()))
    cx.commit()


def list_animals(cx):
    """Every animal record (species set and not Human), each with a needs_name flag
    for those still missing an animal_name. The console audit readout."""
    rows = cx.execute(
        "SELECT email, species, animal_name FROM client_species "
        "WHERE species IS NOT NULL AND lower(trim(species)) NOT IN ('', 'human') "
        "ORDER BY email").fetchall()
    return [{"email": r[0], "species": r[1], "animal_name": r[2] or "",
             "needs_name": not (r[2] or "").strip()} for r in rows]


def get(cx, email):
    row = cx.execute("SELECT species, animal_name FROM client_species WHERE email=?",
                     (_norm(email),)).fetchone()
    if not row:
        return None
    species, animal_name = row[0], row[1]
    return {"species": species, "animal_name": animal_name, "is_animal": is_animal(species)}
