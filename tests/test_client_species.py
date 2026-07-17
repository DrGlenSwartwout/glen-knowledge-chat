"""Which portal clients are animals, and by what name to greet them.

Species comes from E4L (Human/Cat/Dog/Horse) or an operator-typed value for any other
mammal. is_animal = set and not Human. The greeting uses animal_name (Sasha), never the
account name.
"""
import sqlite3

import pytest

from dashboard import client_species as cs


@pytest.fixture()
def cx():
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    cs.init_table(con)
    yield con
    con.close()


@pytest.mark.parametrize("species,expected", [
    ("Cat", True), ("Dog", True), ("Horse", True), ("Rabbit", True),
    ("Human", False), ("human", False), ("  HUMAN ", False), ("", False), (None, False),
])
def test_is_animal(species, expected):
    assert cs.is_animal(species) is expected


def test_upsert_and_get(cx):
    cs.upsert(cx, "care@example.com", "Cat", "Sasha")
    r = cs.get(cx, "care@example.com")
    assert r == {"species": "Cat", "animal_name": "Sasha", "is_animal": True}


def test_get_absent_is_none(cx):
    assert cs.get(cx, "nobody@example.com") is None


def test_upsert_is_idempotent_and_updates(cx):
    cs.upsert(cx, "e@x.com", "Cat", "Sasha")
    cs.upsert(cx, "e@x.com", "Rabbit", "Thumper")     # operator override wins
    assert cx.execute("SELECT COUNT(*) FROM client_species").fetchone()[0] == 1
    assert cs.get(cx, "e@x.com")["species"] == "Rabbit"


def test_email_is_normalised(cx):
    cs.upsert(cx, "  Care@Example.COM ", "Dog", "Rex")
    assert cs.get(cx, "care@example.com")["animal_name"] == "Rex"


def test_a_human_row_is_stored_but_not_an_animal(cx):
    cs.upsert(cx, "person@example.com", "Human", "")
    assert cs.get(cx, "person@example.com")["is_animal"] is False


def test_blank_incoming_name_never_wipes_an_existing_name(cx):
    # An E4L re-sync with an empty AnimalName must NOT erase an operator-set name.
    cs.upsert(cx, "e@x.com", "Cat", "Sasha")
    cs.upsert(cx, "e@x.com", "Cat", "")            # re-sync, empty AnimalName
    assert cs.get(cx, "e@x.com")["animal_name"] == "Sasha"


def test_nonempty_incoming_name_still_overwrites(cx):
    cs.upsert(cx, "e@x.com", "Cat", "Sasha")
    cs.upsert(cx, "e@x.com", "Cat", "Mochi")       # a real correction still applies
    assert cs.get(cx, "e@x.com")["animal_name"] == "Mochi"


def test_list_animals_flags_missing_names(cx):
    cs.upsert(cx, "named@x.com", "Cat", "Sasha")
    cs.upsert(cx, "unnamed@x.com", "Dog", "")
    cs.upsert(cx, "human@x.com", "Human", "")       # excluded — not an animal
    animals = cs.list_animals(cx)
    emails = {a["email"]: a for a in animals}
    assert set(emails) == {"named@x.com", "unnamed@x.com"}
    assert emails["named@x.com"]["needs_name"] is False
    assert emails["unnamed@x.com"]["needs_name"] is True
