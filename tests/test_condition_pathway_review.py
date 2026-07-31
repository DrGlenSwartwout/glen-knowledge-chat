"""Condition-pathway review CRUD. Same shape as pathway_review.

The load-bearing test is `test_confirm_with_a_changed_direction_marks_source_glen`.
The batch gate in the vault counts a FLIP as the signal that the generator is wrong,
and it reads that signal from source='glen'. If a plain confirm also set source='glen',
every confirmation would look like a flip and the gate would never open.
"""
import sqlite3

import pytest

from dashboard import condition_pathway_review as cpr


@pytest.fixture
def cx():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    cpr.init_tables(c)
    c.executescript("""
        INSERT INTO canonical_pathways(id,slug,label,family,status,created_at)
            VALUES(1,'nf-kb','NF-kB','inflammation','proposed','t'),
                  (2,'tear-film','Tear film stability','eye','proposed','t');
        INSERT INTO conditions(key,label,system,batch,created_at)
            VALUES('dry-eye','Dry Eye','eye','eye','t');
        INSERT INTO condition_pathway
            (id,condition_key,canonical_id,desired_direction,tier,rationale,decision,source)
            VALUES(1,'dry-eye',1,'down','core','surface inflammation','proposed','ai'),
                  (2,'dry-eye',2,'up','contributing','lipid layer','proposed','ai');
    """)
    c.commit()
    yield c
    c.close()


def test_queue_returns_only_proposed_edges(cx):
    cx.execute("UPDATE condition_pathway SET decision='confirmed' WHERE id=2")
    rows = cpr.queue(cx)
    assert [r["id"] for r in rows] == [1]


def test_queue_carries_the_labels_a_reviewer_needs(cx):
    r = cpr.queue(cx)[0]
    assert r["condition_label"] == "Dry Eye"
    assert r["pathway_label"] == "NF-kB"
    assert r["family"] == "inflammation"
    assert r["rationale"] == "surface inflammation"


def test_plain_confirm_keeps_source_ai(cx):
    cpr.decide(cx, 1, "confirmed")
    r = cx.execute("SELECT * FROM condition_pathway WHERE id=1").fetchone()
    assert (r["decision"], r["source"]) == ("confirmed", "ai")


def test_confirm_with_a_changed_direction_marks_source_glen(cx):
    """A flip is what the batch gate measures. It must be distinguishable from a
    plain confirmation or the gate never opens."""
    cpr.decide(cx, 1, "confirmed", desired_direction="up")
    r = cx.execute("SELECT * FROM condition_pathway WHERE id=1").fetchone()
    assert (r["desired_direction"], r["source"]) == ("up", "glen")


def test_confirm_with_a_changed_tier_marks_source_glen(cx):
    cpr.decide(cx, 1, "confirmed", tier="modifying")
    r = cx.execute("SELECT * FROM condition_pathway WHERE id=1").fetchone()
    assert (r["tier"], r["source"]) == ("modifying", "glen")


def test_confirm_restating_the_same_direction_is_not_a_flip(cx):
    cpr.decide(cx, 1, "confirmed", desired_direction="down", tier="core")
    r = cx.execute("SELECT * FROM condition_pathway WHERE id=1").fetchone()
    assert r["source"] == "ai"


def test_adverse_direction_is_rejected(cx):
    assert cpr.decide(cx, 1, "confirmed", desired_direction="adverse") is None
    r = cx.execute("SELECT * FROM condition_pathway WHERE id=1").fetchone()
    assert r["decision"] == "proposed"


def test_bad_decision_is_rejected(cx):
    assert cpr.decide(cx, 1, "maybe") is None


def test_undo_returns_an_edge_to_the_queue(cx):
    cpr.decide(cx, 1, "confirmed", desired_direction="up")
    cpr.undo(cx, 1)
    r = cx.execute("SELECT * FROM condition_pathway WHERE id=1").fetchone()
    assert (r["decision"], r["source"], r["decided_at"]) == ("proposed", "ai", None)


def test_needs_canonical_records_rather_than_creating(cx):
    """The vocabulary is closed to this work."""
    before = cx.execute("SELECT count(*) FROM canonical_pathways").fetchone()[0]
    cpr.needs_canonical(cx, "dry-eye", "Goblet cell mucin secretion", "no node exists")
    assert cx.execute("SELECT count(*) FROM canonical_pathways").fetchone()[0] == before
    assert cx.execute("SELECT count(*) FROM condition_needs_canonical").fetchone()[0] == 1


def test_needs_canonical_rejects_an_unknown_condition_key(cx):
    """No FK exists on condition_needs_canonical.condition_key, so a typo would
    otherwise silently create an orphaned row nothing ever surfaces. Guard it the
    same way the sibling's create_canonical() guards its input."""
    before = cx.execute("SELECT count(*) FROM condition_needs_canonical").fetchone()[0]
    assert cpr.needs_canonical(cx, "dryy-eye", "Goblet cell mucin secretion", "typo") is None
    assert cx.execute("SELECT count(*) FROM condition_needs_canonical").fetchone()[0] == before


def test_stats_counts_pending_and_decided(cx):
    cpr.decide(cx, 1, "confirmed")
    s = cpr.stats(cx)
    assert s["pending"] == 1 and s["confirmed"] == 1


def test_invalid_direction_raises_no_integrityerror_past_python_validation(cx):
    """The vault's real CHECK constraint would raise sqlite3.IntegrityError if a
    bad value ever reached the UPDATE. decide()'s own validation is the first
    line of defence and must reject cleanly with None -- never let the
    IntegrityError escape to a route."""
    assert cpr.decide(cx, 1, "confirmed", desired_direction="neutral") is None
    assert cpr.decide(cx, 1, "confirmed", tier="bogus-tier") is None


def test_unscored_conditions_are_filtered_from_the_queue_even_with_an_edge(cx):
    """Eight of the 42 conditions (astigmatism, strabismus, etc.) have no
    genuine biochemical mechanism and are marked scored=0. Today they carry zero
    edges by construction, but that invariant lives in the data, not the query --
    a future generator bug, backfill, or manual insert could still put an edge
    against one. This inserts an unscored condition WITH an edge, so a pass here
    can only mean the scored=1 filter is doing the work, not that the condition
    coincidentally has nothing to join against."""
    cx.execute("INSERT INTO conditions(key,label,system,batch,created_at,scored) "
               "VALUES('strabismus','Strabismus','eye','eye','t',0)")
    cx.execute("INSERT INTO condition_pathway"
               "(id,condition_key,canonical_id,desired_direction,tier,rationale,decision,source) "
               "VALUES(3,'strabismus',1,'up','core','spurious edge','proposed','ai')")
    cx.commit()
    row = cx.execute("SELECT scored FROM conditions WHERE key='strabismus'").fetchone()
    assert row["scored"] == 0
    # default is scored=1 for a condition inserted without the column
    default_row = cx.execute("SELECT scored FROM conditions WHERE key='dry-eye'").fetchone()
    assert default_row["scored"] == 1
    # the edge exists and is otherwise queue-eligible (decision='proposed'), so
    # its absence below can only be explained by the scored filter
    assert cx.execute("SELECT decision FROM condition_pathway WHERE id=3").fetchone()[0] \
        == "proposed"
    assert "strabismus" not in [r["condition_key"] for r in cpr.queue(cx)]


def test_corpus_absent_edges_outrank_tier_in_the_queue_order(cx):
    """The primary sort key is corpus-absence, not tier -- an edge with NO
    catalog signal must sort ahead of a core-tier edge that has one, or the
    ordering has silently degenerated to tier-only. condition_corpus_signal is
    empty in production today (the builder hasn't landed), so this is the only
    test that can prove the primary key actually outranks the tier key rather
    than merely agreeing with it."""
    # id 1: dry-eye/NF-kB, tier=core -- give it catalog signal (products exist)
    cx.execute("INSERT INTO condition_corpus_signal"
               "(condition_key,canonical_id,n_products,n_ingredients,computed_at) "
               "VALUES('dry-eye',1,5,3,'t')")
    # id 2: dry-eye/tear-film, tier=contributing (lower priority by tier alone) --
    # leave it with NO corpus signal row, so COALESCE(s.n_products,0)=0
    rows = cpr.queue(cx)
    assert [r["id"] for r in rows] == [2, 1]
