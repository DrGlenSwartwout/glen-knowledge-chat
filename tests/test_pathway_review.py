"""Canonical-pathway review: queue ordering, decision persistence, and the
non-destructive guarantee (no original pathway string is ever rewritten)."""
import sqlite3

import pytest

from dashboard import pathway_review as pr


@pytest.fixture
def cx(tmp_path):
    c = sqlite3.connect(str(tmp_path / "ingredients.db"))
    pr.init_tables(c)
    # Two ingredients whose free-text pathways both mean NF-kB, spelled apart,
    # plus a one-ingredient atom and a caution row that is not a pathway.
    rows = [
        (1, "NF-κB", "inhibits"),
        (2, "NF-κB inflammatory pathway", "modulates"),
        (3, "Nrf2/Keap1/ARE pathway", "activates"),
        (1, "Aromatase (CYP19)", "inhibits"),
    ]
    for ing, path, eff in rows:
        c.execute("INSERT INTO ingredient_pathways(ingredient_id,pathway,effect) "
                  "VALUES(?,?,?)", (ing, path, eff))
    c.commit()
    atoms = [
        (1, 1, 0, "NF-κB", "nf kappab", 0),
        (2, 2, 0, "NF-κB inflammatory", "nf kappab inflammatory", 0),
        (3, 3, 0, "Nrf2", "nrf2", 0),
        (4, 1, 0, "Aromatase", "aromatase", 0),
    ]
    c.executemany("INSERT INTO pathway_atoms(pathway_row_id,ingredient_id,position,"
                  "atom,atom_key,is_annotation) VALUES(?,?,?,?,?,?)", atoms)
    c.execute("INSERT INTO canonical_pathways(slug,label,family,status,created_at) "
              "VALUES('nf-kb','NF-κB','inflammation','proposed',datetime('now'))")
    c.executemany(
        "INSERT INTO pathway_atom_map(atom_key,canonical_id,decision,source) VALUES(?,?,?,?)",
        [("nf kappab", 1, "proposed", "ai"),
         ("nf kappab inflammatory", 1, "proposed", "ai"),
         ("nrf2", None, "proposed", "ai"),
         ("aromatase", None, "proposed-reject", "ai")])
    c.commit()
    yield c
    c.close()


def test_queue_holds_shared_atoms_and_hides_singletons(cx):
    keys = [r["atom_key"] for r in pr.queue(cx)]
    # nf kappab spans ingredients 1 and 2 via two different spellings? No --
    # each spelling is one ingredient here, so nothing is shared yet.
    assert keys == []
    assert pr.queue(cx, include_singletons=True)


def test_queue_orders_by_ingredient_reach(cx):
    # Give "nf kappab" a second ingredient so it outranks the singletons.
    cx.execute("INSERT INTO pathway_atoms(pathway_row_id,ingredient_id,position,"
               "atom,atom_key,is_annotation) VALUES(2,2,1,'NF-kB','nf kappab',0)")
    cx.commit()
    rows = pr.queue(cx)
    assert [r["atom_key"] for r in rows] == ["nf kappab"]
    assert rows[0]["ingredients"] == 2
    assert rows[0]["canonical_label"] == "NF-κB"


def test_lifted_atoms_add_reach_without_adding_review_work(cx):
    """The alias sweep lifts canonical names out of the discarded em-dash tail.
    Because a lifted atom reuses an atom_key that is already decided, it must
    raise the pathway's ingredient reach and never appear as a new queue item —
    that is the whole reason the sweep is free to run."""
    before = pr.queue(cx, include_singletons=True)
    cx.execute("INSERT INTO pathway_atoms(pathway_row_id,ingredient_id,position,"
               "atom,atom_key,is_annotation,source) "
               "VALUES(3,7,0,'nf kappab','nf kappab',0,'rest')")
    cx.commit()
    after = pr.queue(cx, include_singletons=True)
    assert len(after) == len(before)                     # no new card
    reach = {r["atom_key"]: r["ingredients"] for r in after}
    assert reach["nf kappab"] == 2                       # ingredient 1 plus 7
    # and a lifted atom whose key is already settled stays out entirely
    pr.decide(cx, "nf kappab", "confirmed", canonical_id=1)
    assert "nf kappab" not in [r["atom_key"] for r in pr.queue(cx, include_singletons=True)]


def test_annotation_atoms_never_enter_the_queue(cx):
    cx.execute("UPDATE pathway_atoms SET is_annotation=1")
    cx.commit()
    assert pr.queue(cx, include_singletons=True) == []


def test_confirm_records_the_link_and_confirms_the_canonical(cx):
    pr.decide(cx, "nf kappab", "confirmed", canonical_id=1)
    row = cx.execute("SELECT canonical_id, decision, source FROM pathway_atom_map "
                     "WHERE atom_key='nf kappab'").fetchone()
    assert row == (1, "confirmed", "glen")
    assert cx.execute("SELECT status FROM canonical_pathways WHERE id=1").fetchone()[0] \
        == "confirmed"


def test_orphan_and_reject_drop_any_proposed_link(cx):
    pr.decide(cx, "nf kappab", "orphan", canonical_id=1)
    assert cx.execute("SELECT canonical_id, decision FROM pathway_atom_map "
                      "WHERE atom_key='nf kappab'").fetchone() == (None, "orphan")
    pr.decide(cx, "nf kappab inflammatory", "rejected", canonical_id=1)
    assert cx.execute("SELECT canonical_id, decision FROM pathway_atom_map "
                      "WHERE atom_key='nf kappab inflammatory'").fetchone() == (None, "rejected")


def test_settled_atoms_leave_the_queue_and_undo_returns_them(cx):
    cx.execute("INSERT INTO pathway_atoms(pathway_row_id,ingredient_id,position,"
               "atom,atom_key,is_annotation) VALUES(2,2,1,'NF-kB','nf kappab',0)")
    cx.commit()
    assert [r["atom_key"] for r in pr.queue(cx)] == ["nf kappab"]
    pr.decide(cx, "nf kappab", "confirmed", canonical_id=1)
    assert pr.queue(cx) == []
    pr.undo(cx, "nf kappab")
    assert [r["atom_key"] for r in pr.queue(cx)] == ["nf kappab"]


def test_decide_never_touches_the_original_pathway_strings(cx):
    before = cx.execute("SELECT id, pathway, effect FROM ingredient_pathways "
                        "ORDER BY id").fetchall()
    pr.decide(cx, "nf kappab", "confirmed", canonical_id=1)
    pr.decide(cx, "nrf2", "orphan")
    pr.decide(cx, "aromatase", "rejected")
    after = cx.execute("SELECT id, pathway, effect FROM ingredient_pathways "
                       "ORDER BY id").fetchall()
    assert before == after


def test_bad_decision_verb_is_refused(cx):
    assert pr.decide(cx, "nf kappab", "merge-please", canonical_id=1) is None
    assert cx.execute("SELECT decision FROM pathway_atom_map "
                      "WHERE atom_key='nf kappab'").fetchone()[0] == "proposed"


def test_create_canonical_is_idempotent_on_slug(cx):
    a = pr.create_canonical(cx, "NLRP3 inflammasome", "inflammation")
    b = pr.create_canonical(cx, "NLRP3 inflammasome", "inflammation")
    assert a["created"] and not b["created"] and a["id"] == b["id"]
    assert pr.create_canonical(cx, "   ") is None


def test_examples_show_the_source_strings(cx):
    assert pr.examples(cx, "nf kappab") == ["NF-κB"]


def test_stats_track_settled_versus_pending(cx):
    s0 = pr.stats(cx)
    assert s0["pending"] == 4 and s0["settled"] == 0
    pr.decide(cx, "nrf2", "orphan")
    s1 = pr.stats(cx)
    assert s1["pending"] == 3 and s1["settled"] == 1


def test_direction_queue_surfaces_only_the_unconfident(cx):
    cx.executemany("INSERT INTO pathway_effect_direction(effect_key,direction,"
                   "confidence,decision,n_rows) VALUES(?,?,?,?,?)",
                   [("inhibits", "down", "high", "proposed", 114),
                    ("crosses", "unknown", "low", "proposed", 3),
                    ("feeds", "substrate", "low", "proposed", 4)])
    cx.commit()
    assert [d["effect_key"] for d in pr.direction_queue(cx)] == ["feeds", "crosses"]
    pr.set_direction(cx, "crosses", "substrate")
    assert [d["effect_key"] for d in pr.direction_queue(cx)] == ["feeds"]
    assert pr.set_direction(cx, "feeds", "sideways") is None
