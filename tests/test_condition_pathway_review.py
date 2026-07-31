"""Condition-pathway review CRUD. Same shape as pathway_review.

The load-bearing tests are the ones about REFUSAL: `decide()` must refuse an edge that
is already decided (a stale second tab re-submitting the generator's original
direction used to win and silently reinstate a contraindicated one), and it must
refuse to erase a rationale that a blank form field submitted as "".

Note what is NOT load-bearing any more: `source`. The vault's batch gate used to read
its flip signal from source='glen'; it now compares desired_direction against
proposed_direction, an immutable record of what the generator first proposed. source
is provenance only, and `test_a_correction_survives_undo_and_redecision` is the reason
for the change -- undo() resets source while leaving the corrected value in place, so
the ordinary confirm -> undo -> reconfirm loop erased the old signal entirely.
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
            (id,condition_key,canonical_id,desired_direction,tier,
             proposed_direction,proposed_tier,rationale,decision,source)
            VALUES(1,'dry-eye',1,'down','core','down','core',
                   'surface inflammation','proposed','ai'),
                  (2,'dry-eye',2,'up','contributing','up','contributing',
                   'lipid layer','proposed','ai');
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
    got = cpr.decide(cx, 1, "confirmed", desired_direction="adverse")
    assert got["ok"] is False and got["error"] == "invalid-direction"
    r = cx.execute("SELECT * FROM condition_pathway WHERE id=1").fetchone()
    assert r["decision"] == "proposed"


def test_bad_decision_is_rejected(cx):
    got = cpr.decide(cx, 1, "maybe")
    assert got["ok"] is False and got["error"] == "invalid-decision"


def test_deciding_an_already_decided_edge_is_a_conflict_not_a_silent_overwrite(cx):
    """The Stargardt trace. Glen confirms the edge corrected to 'down' (vitamin A
    transport must go DOWN in ABCA4 disease -- vitamin A is contraindicated). A
    second tab, still showing the generator's original 'up', re-submits. Before the
    guard the second write won, stamped a fresh decided_at, and was
    indistinguishable from a real ruling: the safety interlock collapsed to the
    contraindicated direction with nothing anywhere recording that it had."""
    first = cpr.decide(cx, 1, "confirmed", desired_direction="down")
    assert first["ok"] is True
    stale = cpr.decide(cx, 1, "confirmed", desired_direction="up")
    assert stale["ok"] is False
    assert stale["error"] == "conflict"
    assert stale["decision"] == "confirmed"      # what it was already decided as
    r = cx.execute("SELECT * FROM condition_pathway WHERE id=1").fetchone()
    assert r["desired_direction"] == "down"      # the ruling stands


def test_a_conflict_is_distinguishable_from_a_bad_argument(cx):
    """The UI has to say 'another tab already decided this, reload' for one and
    'bad request' for the other. A shared falsy return cannot express that."""
    cpr.decide(cx, 1, "confirmed")
    conflict = cpr.decide(cx, 1, "confirmed")
    bad = cpr.decide(cx, 2, "confirmed", tier="urgent")
    missing = cpr.decide(cx, 999, "confirmed")
    assert conflict["error"] == "conflict"
    assert bad["error"] == "invalid-tier"
    assert missing["error"] == "unknown-edge"
    assert len({conflict["error"], bad["error"], missing["error"]}) == 3


def test_a_rejected_edge_is_equally_protected(cx):
    """The guard is on 'not proposed', not on 'confirmed' -- a rejection is a
    ruling too and a stale tab must not be able to confirm over it."""
    cpr.decide(cx, 1, "rejected", reject_reason="pathway-not-implicated")
    got = cpr.decide(cx, 1, "confirmed")
    assert got["error"] == "conflict" and got["decision"] == "rejected"
    assert cx.execute("SELECT decision FROM condition_pathway WHERE id=1"
                      ).fetchone()[0] == "rejected"


def test_undo_is_the_honest_way_to_re_decide(cx):
    """Re-deciding stays possible -- it just has to be deliberate."""
    cpr.decide(cx, 1, "confirmed", desired_direction="down")
    assert cpr.decide(cx, 1, "confirmed", desired_direction="up")["ok"] is False
    assert cpr.undo(cx, 1)["ok"] is True
    assert cpr.decide(cx, 1, "confirmed", desired_direction="up")["ok"] is True
    r = cx.execute("SELECT * FROM condition_pathway WHERE id=1").fetchone()
    assert (r["decision"], r["desired_direction"]) == ("confirmed", "up")


def test_undo_reports_an_unknown_edge_rather_than_claiming_success(cx):
    assert cpr.undo(cx, 999) == {"ok": False, "error": "unknown-edge"}


def test_a_correction_survives_undo_and_redecision(cx):
    """The reason the flip signal moved off `source`. undo() resets source to 'ai'
    and leaves the corrected direction in place, so a reconfirmation compares the
    correction against itself and records a corrected edge as AI-authored.
    proposed_direction is written at row creation and nothing here ever touches it,
    so the vault's gate still sees the correction."""
    cpr.decide(cx, 1, "confirmed", desired_direction="up")   # proposed was 'down'
    cpr.undo(cx, 1)
    cpr.decide(cx, 1, "confirmed", desired_direction="up")   # reconfirm what he sees
    r = cx.execute("SELECT * FROM condition_pathway WHERE id=1").fetchone()
    assert r["source"] == "ai"                    # the old signal is gone
    assert r["desired_direction"] == "up"
    assert r["proposed_direction"] == "down"      # the durable one is not


def test_an_empty_rationale_does_not_erase_the_recorded_one(cx):
    """A route reading an untouched form field submits "", not None. COALESCE falls
    through on NULL only, so "" used to overwrite the justification for a
    safety-relevant override, unrecoverably."""
    cpr.decide(cx, 1, "confirmed", rationale="")
    r = cx.execute("SELECT rationale FROM condition_pathway WHERE id=1").fetchone()
    assert r["rationale"] == "surface inflammation"


def test_a_whitespace_only_rationale_does_not_erase_the_recorded_one(cx):
    cpr.decide(cx, 2, "confirmed", rationale="   \n\t ")
    r = cx.execute("SELECT rationale FROM condition_pathway WHERE id=2").fetchone()
    assert r["rationale"] == "lipid layer"


def test_a_real_rationale_still_replaces_the_old_one(cx):
    cpr.decide(cx, 1, "confirmed", rationale="corrected: sign was inverted")
    r = cx.execute("SELECT rationale FROM condition_pathway WHERE id=1").fetchone()
    assert r["rationale"] == "corrected: sign was inverted"


def test_a_wrong_direction_rejection_records_its_reason(cx):
    """The vault's gate counts a 'wrong-direction' rejection as a direction defect
    alongside a flip -- rejecting an inverted edge is the same generator failure as
    correcting one, and used to count as nothing."""
    assert cpr.decide(cx, 1, "rejected", reject_reason="wrong-direction")["ok"] is True
    r = cx.execute("SELECT * FROM condition_pathway WHERE id=1").fetchone()
    assert (r["decision"], r["reject_reason"]) == ("rejected", "wrong-direction")


def test_an_unknown_reject_reason_is_refused_like_a_bad_direction(cx):
    got = cpr.decide(cx, 1, "rejected", reject_reason="i-just-dont-like-it")
    assert got["ok"] is False and got["error"] == "invalid-reject-reason"
    assert cx.execute("SELECT decision FROM condition_pathway WHERE id=1"
                      ).fetchone()[0] == "proposed"


def test_a_reject_reason_is_ignored_on_a_confirmation(cx):
    cpr.decide(cx, 1, "confirmed", reject_reason="wrong-direction")
    r = cx.execute("SELECT * FROM condition_pathway WHERE id=1").fetchone()
    assert r["decision"] == "confirmed" and r["reject_reason"] is None


def test_undo_clears_the_reject_reason(cx):
    cpr.decide(cx, 1, "rejected", reject_reason="wrong-direction")
    cpr.undo(cx, 1)
    r = cx.execute("SELECT * FROM condition_pathway WHERE id=1").fetchone()
    assert (r["decision"], r["reject_reason"], r["decided_at"]) == ("proposed", None, None)


def test_queue_takes_no_offset(cx):
    """Offset paging over a set that shrinks with every decision skips edges
    permanently: decide 20 of the first 40, ask for offset=40, and 20 edges you
    never saw are behind you while the session looks like forward progress. queue()
    always reads from the start; decided edges leave the set on their own. Do not
    re-add the parameter."""
    import inspect
    assert "offset" not in inspect.signature(cpr.queue).parameters


def test_the_queue_cannot_skip_an_edge_as_decisions_are_made(cx):
    """The property offset paging broke. With a limit of 1, deciding the edge that
    is shown must reveal the other one -- never advance past it."""
    seen = []
    for _ in range(2):
        row = cpr.queue(cx, limit=1)[0]
        seen.append(row["id"])
        cpr.decide(cx, row["id"], "confirmed")
    assert sorted(seen) == [1, 2]
    assert cpr.queue(cx, limit=1) == []


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
    assert cpr.decide(cx, 1, "confirmed", desired_direction="neutral")["ok"] is False
    assert cpr.decide(cx, 1, "confirmed", tier="bogus-tier")["ok"] is False
    assert cx.execute("SELECT decision FROM condition_pathway WHERE id=1"
                      ).fetchone()[0] == "proposed"


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


def test_high_signal_edges_outrank_tier_in_the_queue_order(cx):
    """Glen's ruling on the polarity: HIGHEST catalog signal first. An edge whose
    pathway nothing in the catalogue touches is inert -- no product can be
    misranked by it -- while a wrong direction on a pathway many products touch
    inverts live recommendations.

    The test is arranged so signal and tier DISAGREE, and it is only meaningful
    that way: the high-signal edge here is the LOWER-tier one (contributing), so it
    can only reach the front by the signal key outranking the tier key. If both
    keys favoured the same row this would pass on a tier-only ordering and prove
    nothing about precedence. (Same property the previous version of this test
    held under the opposite polarity.)"""
    # id 2: dry-eye/tear-film, tier=contributing -- LOWER by tier, but it is what
    # the catalog actually touches
    cx.execute("INSERT INTO condition_corpus_signal"
               "(condition_key,canonical_id,n_products,n_ingredients,computed_at) "
               "VALUES('dry-eye',2,5,3,'t')")
    # id 1: dry-eye/NF-kB, tier=core -- HIGHER by tier, no corpus signal row at all
    rows = cpr.queue(cx)
    assert [r["id"] for r in rows] == [2, 1]


def test_the_queue_orders_by_how_many_products_a_pathway_touches(cx):
    """Not merely present-vs-absent: more products touching a pathway means more
    recommendations a wrong direction there would invert."""
    cx.execute("INSERT INTO condition_corpus_signal"
               "(condition_key,canonical_id,n_products,n_ingredients,computed_at) "
               "VALUES('dry-eye',1,2,1,'t'),('dry-eye',2,9,4,'t')")
    assert [r["id"] for r in cpr.queue(cx)] == [2, 1]


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
    # Deviation from the brief's `cpr.queue(cx)[0]`: queue() orders on the corpus
    # signal (test_high_signal_edges_outrank_tier_in_the_queue_order), so which row
    # lands at index [0] depends on the signal this test just inserted. Select the
    # edge by canonical_id instead of relying on queue position -- the point here
    # is the card's rendering, not the ordering.
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


def test_card_offers_the_reject_reason_vocabulary_as_one_click_buttons(cx):
    """Task 10's highest-value fix: the page must let Glen state WHY he rejects an
    edge, in one click, not a modal -- he has 323 of these. Three buttons, one per
    REJECT_REASONS value, each carrying its reason as data-reason so the click
    handler needs no follow-up prompt."""
    out = html.render_edge_card(cpr.queue(cx)[0])
    for reason in cpr.REJECT_REASONS:
        assert ("data-act=\"rejected\" data-reason=\"%s\"" % reason) in out
    # and the page's script reads that attribute off the clicked button and
    # forwards it as reject_reason -- this is what makes a 'wrong-direction'
    # rejection visible to the vault's gate at all. The script is emitted once
    # per page (render_condition_review_page), not per card.
    assert "dataset.reason" in html._JS and "reject_reason" in html._JS


# --- init_tables() migration / backfill -----------------------------------------
# Zero tests existed on this side before Task 10 despite the vault's sibling
# `02 Skills/condition_pathway.py: migrate()` having three. All 323 live rows are
# currently 'proposed' (inert today), but the guard below closes the gap while it
# is cheap: a silent false negative here would read to gate_metrics() as "no
# direction defect" when a row was actually decided blind to the flip signal.

def test_init_tables_adds_the_proposal_columns_to_a_pre_existing_edge_table():
    """Simulates the live database: a condition_pathway table created before
    proposed_direction/proposed_tier/reject_reason existed. Additive ALTER,
    guarded by PRAGMA table_info, and idempotent -- mirrors the vault's
    `test_migrate_adds_the_proposal_columns_to_a_pre_existing_edge_table`."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript("""
        CREATE TABLE canonical_pathways(
            id INTEGER PRIMARY KEY AUTOINCREMENT, slug TEXT NOT NULL UNIQUE,
            label TEXT NOT NULL, family TEXT,
            status TEXT NOT NULL DEFAULT 'proposed', notes TEXT,
            created_at TEXT NOT NULL, decided_at TEXT);
        CREATE TABLE conditions(
            id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL UNIQUE,
            label TEXT NOT NULL, system TEXT, batch TEXT NOT NULL, aliases TEXT,
            status TEXT NOT NULL DEFAULT 'proposed', notes TEXT,
            created_at TEXT NOT NULL, decided_at TEXT,
            scored INTEGER NOT NULL DEFAULT 1);
        CREATE TABLE condition_pathway(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            condition_key TEXT NOT NULL,
            canonical_id INTEGER NOT NULL,
            desired_direction TEXT NOT NULL,
            tier TEXT NOT NULL,
            rationale TEXT NOT NULL,
            decision TEXT NOT NULL DEFAULT 'proposed',
            source TEXT NOT NULL,
            decided_at TEXT,
            UNIQUE(condition_key, canonical_id));
        INSERT INTO condition_pathway
            (condition_key,canonical_id,desired_direction,tier,rationale,source)
            VALUES('dry-eye',1,'down','core','r','ai');
    """)
    c.commit()
    cpr.init_tables(c)
    cols = {r[1] for r in c.execute("PRAGMA table_info(condition_pathway)")}
    assert {"proposed_direction", "proposed_tier", "reject_reason"} <= cols
    row = c.execute("SELECT * FROM condition_pathway").fetchone()
    assert (row["proposed_direction"], row["proposed_tier"]) == ("down", "core")
    cpr.init_tables(c)  # second pass must not raise "duplicate column"
    assert {r[1] for r in c.execute("PRAGMA table_info(condition_pathway)")} == cols


def test_init_tables_backfills_undecided_rows():
    """A row still 'proposed' has never been edited, so its current values ARE the
    generator's original proposal and can be copied in safely."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    cpr.init_tables(c)
    c.executescript("""
        INSERT INTO canonical_pathways(id,slug,label,family,status,created_at)
            VALUES(1,'nf-kb','NF-kB','inflammation','proposed','t');
        INSERT INTO conditions(key,label,system,batch,created_at)
            VALUES('dry-eye','Dry Eye','eye','eye','t');
        INSERT INTO condition_pathway
            (condition_key,canonical_id,desired_direction,tier,rationale,decision,source)
            VALUES('dry-eye',1,'up','modifying','r','proposed','ai');
    """)
    c.commit()
    cpr.init_tables(c)
    row = c.execute("SELECT * FROM condition_pathway").fetchone()
    assert (row["proposed_direction"], row["proposed_tier"]) == ("up", "modifying")


def test_init_tables_refuses_to_invent_a_baseline_for_an_already_decided_row():
    """A row decided before the column existed may have been corrected, so copying
    its current value in would manufacture a 'no change' baseline -- exactly the
    false negative these columns exist to remove, and exactly what gate_metrics()
    would read as 'no direction defect'. Stop loudly instead of guessing, same as
    the vault's `migrate()`."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript("""
        CREATE TABLE canonical_pathways(
            id INTEGER PRIMARY KEY AUTOINCREMENT, slug TEXT NOT NULL UNIQUE,
            label TEXT NOT NULL, family TEXT,
            status TEXT NOT NULL DEFAULT 'proposed', notes TEXT,
            created_at TEXT NOT NULL, decided_at TEXT);
        CREATE TABLE conditions(
            id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL UNIQUE,
            label TEXT NOT NULL, system TEXT, batch TEXT NOT NULL, aliases TEXT,
            status TEXT NOT NULL DEFAULT 'proposed', notes TEXT,
            created_at TEXT NOT NULL, decided_at TEXT,
            scored INTEGER NOT NULL DEFAULT 1);
        CREATE TABLE condition_pathway(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            condition_key TEXT NOT NULL,
            canonical_id INTEGER NOT NULL,
            desired_direction TEXT NOT NULL,
            tier TEXT NOT NULL,
            rationale TEXT NOT NULL,
            decision TEXT NOT NULL DEFAULT 'proposed',
            source TEXT NOT NULL,
            decided_at TEXT,
            UNIQUE(condition_key, canonical_id));
        INSERT INTO condition_pathway
            (condition_key,canonical_id,desired_direction,tier,rationale,decision,source)
            VALUES('dry-eye',1,'up','core','r','confirmed','glen');
    """)
    c.commit()
    try:
        cpr.init_tables(c)
        raise AssertionError("init_tables must refuse to backfill a decided row")
    except RuntimeError as e:
        assert "BLOCKED" in str(e)
