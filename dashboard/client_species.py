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
        "species=excluded.species, animal_name=excluded.animal_name, synced_at=excluded.synced_at",
        (e, (species or "").strip(), (animal_name or "").strip(), _now()))
    cx.commit()


def get(cx, email):
    row = cx.execute("SELECT species, animal_name FROM client_species WHERE email=?",
                     (_norm(email),)).fetchone()
    if not row:
        return None
    species, animal_name = row[0], row[1]
    return {"species": species, "animal_name": animal_name, "is_animal": is_animal(species)}
