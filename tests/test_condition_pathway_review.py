"""Condition-pathway review CRUD. Same shape as pathway_review.

The load-bearing test is `test_confirm_with_a_changed_direction_marks_source_glen`.
The batch gate in the vault counts a FLIP as the signal that the generator is wrong,
and it reads that signal from source='glen'. If a plain confirm also set source='glen',
every confirmation would look like a flip and the gate would never open.
"""
import sqlite3

import pytest

from dashboard import condition_pathway_review as cpr
from dashboard import condition_pathway_review_html as html


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


def test_queue_must_never_filter_on_scored_or_the_batch_gate_deadlocks(cx):
    """DO NOT add a `c.scored = 1` filter to queue(). An earlier round of this
    module did exactly that, on the (reasonable-sounding) theory that an edge on
    an unscored condition -- one of the 8 of 42 marked scored=0 because they have
    no genuine biochemical mechanism -- should be hidden from Glen. It is wrong:
    the vault's `gate_metrics()` counts every edge belonging to a condition's
    batch with NO scored filter (`JOIN conditions c ... WHERE c.batch = ?`), and
    its gate requires `len(decided) == total`. `queue()` is the ONLY surface that
    can ever reach an edge -- if it hides one, that edge can never be decided,
    `decided` stays permanently below `total`, and the batch gate deadlocks with
    no visible cause. An adversarial review reproduced this exact deadlock through
    the real `load_batch` and `selfcheck` (which exempts an unscored condition
    from the MIN_EDGES check only -- an unscored condition that already has 2+
    edges falls to the `elif` and is never checked at all, so a stray edge can
    reach here uncaught by the writer's existing guard).

    The correct refusal lives at the WRITER (`02 Skills/condition_pathway.py`'s
    selfcheck gets a NEW edge-level check for this), not here. This queue must
    show the edge -- either it is fabricated or the `scored` flag is wrong, and
    Glen must see which, not have it silently stranded.

    If you are reading this because you just watched an unscored edge appear in
    the queue and it looked wrong: it is not a bug in this query. Do not re-add
    the filter. Fix it at the writer, or ask why the edge exists at all."""
    cx.execute("INSERT INTO conditions(key,label,system,batch,created_at,scored) "
               "VALUES('strabismus','Strabismus','eye','eye','t',0)")
    cx.execute("INSERT INTO condition_pathway"
               "(id,condition_key,canonical_id,desired_direction,tier,rationale,decision,source) "
               "VALUES(3,'strabismus',1,'up','core','spurious edge','proposed','ai')")
    cx.commit()
    assert "strabismus" in [r["condition_key"] for r in cpr.queue(cx)]
    # and the scored flag rides along on the row so the UI can flag it
    row = next(r for r in cpr.queue(cx) if r["condition_key"] == "strabismus")
    assert row["scored"] == 0


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


def test_card_shows_direction_and_tier_as_the_headline(cx):
    row = cpr.queue(cx)[0]
    out = html.render_edge_card(row)
    assert "Dry Eye" in out and "NF-kB" in out
    assert "down" in out and "core" in out
    assert "surface inflammation" in out


def test_card_shows_the_corpus_cross_check(cx):
    cx.execute("INSERT INTO condition_corpus_signal"
               "(condition_key,canonical_id,n_products,n_ingredients,example_skus,computed_at)"
               " VALUES('dry-eye',1,4,7,'OcuHeal, Clarity','t')")
    cx.commit()
    # Deviation from the brief's `cpr.queue(cx)[0]`: queue() sorts corpus-absent
    # edges first (test_corpus_absent_edges_outrank_tier_in_the_queue_order), so
    # once edge 1 gains a signal, edge 2 (still signal-less) sorts ahead of it and
    # index [0] is no longer the row this test means to inspect. Select the edge
    # by canonical_id instead of relying on queue position.
    row = next(r for r in cpr.queue(cx) if r["canonical_id"] == 1)
    out = html.render_edge_card(row)
    assert "4" in out and "OcuHeal" in out


def test_card_flags_an_edge_on_an_unscored_condition(cx):
    """queue() no longer filters on conditions.scored (filtering there deadlocks
    the batch gate -- see test_queue_must_never_filter_on_scored_or_the_batch_gate_deadlocks),
    so an edge on an unscored condition can reach this card. It must be flagged,
    not rendered as an ordinary edge, because either it is fabricated or the
    condition's scored flag is wrong."""
    cx.execute("INSERT INTO conditions(key,label,system,batch,created_at,scored) "
               "VALUES('strabismus','Strabismus','eye','eye','t',0)")
    cx.execute("INSERT INTO condition_pathway"
               "(id,condition_key,canonical_id,desired_direction,tier,rationale,decision,source) "
               "VALUES(3,'strabismus',1,'up','core','spurious edge','proposed','ai')")
    cx.commit()
    row = next(r for r in cpr.queue(cx) if r["condition_key"] == "strabismus")
    out = html.render_edge_card(row)
    assert "unscored" in out.lower()
    # an ordinary scored edge must NOT carry the flag
    scored_row = next(r for r in cpr.queue(cx) if r["condition_key"] == "dry-eye")
    assert "unscored" not in html.render_edge_card(scored_row).lower()


def test_card_flags_an_edge_no_product_covers(cx):
    """A pathway nothing in the catalog touches is either a product-line hole or a
    wrong edge. Either way the reviewer must see it."""
    out = html.render_edge_card(cpr.queue(cx)[0])
    assert "no product" in out.lower()


def test_html_is_escaped(cx):
    cx.execute("UPDATE condition_pathway SET rationale='<script>x</script>' WHERE id=1")
    cx.commit()
    out = html.render_edge_card(cpr.queue(cx)[0])
    assert "<script>" not in out and "&lt;script&gt;" in out


def test_page_renders_stats_and_nav(cx):
    """Both edges are still proposed, so the page must say 2 pending and render a card
    for each. A bare `'2' in out` would pass on almost any page and prove nothing."""
    out = html.render_condition_review_page(cpr.queue(cx), cpr.stats(cx), nav="<nav>N</nav>")
    assert "<nav>N</nav>" in out
    assert "2 pending" in out
    assert out.count('class="card"') == 2
