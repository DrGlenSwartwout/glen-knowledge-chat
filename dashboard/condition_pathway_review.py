"""CRUD for the condition -> canonical pathway review queue (vault ingredients.db).

The mirror of pathway_review, one axis over: there the reviewer maps an ingredient's
atom to a canonical pathway, here he confirms that a CONDITION implicates one, in a
stated direction and at a stated importance.

Direction is the field that matters. A missing edge makes a remedy invisible; an
inverted one makes the score recommend the opposite of what helps.

The vault's batch gate measures a direction FLIP by comparing an edge's live
`desired_direction` against `proposed_direction`, the generator's original proposal,
which is written once at row creation and which nothing in this module ever writes.
It deliberately does NOT read `source`: `source` says who last wrote the row, and
`undo()` resets it, so the ordinary confirm -> undo -> reconfirm loop used to erase a
recorded correction. `source` is kept here as provenance only -- no measurement
depends on it.

This module never inserts into canonical_pathways. Pure sqlite; the caller passes the
connection. Same shape as pathway_review.py.
"""
import os
import sqlite3

# Exactly the vault's DIRECTIONS vocabulary for THIS edge. "adverse" and
# "neutral" exist on the ingredient-effect side (pathway_review.DIRECTIONS) but
# are not valid here -- a condition's desired direction is a stance the
# generator takes, not an observed ingredient effect.
DIRECTIONS = ("up", "down", "balance", "substrate", "unknown")
TIERS = ("core", "contributing", "modifying")
DECISIONS = ("confirmed", "rejected")
# Mirrors the vault's REJECT_REASONS. 'wrong-direction' is not a bookkeeping detail:
# the vault's gate counts it as a direction defect alongside a flip, because a
# reviewer who throws a wrongly-directed edge away has found the same generator
# failure as one who corrects it.
REJECT_REASONS = ("wrong-direction", "pathway-not-implicated", "other")


def _db_path():
    return os.environ.get(
        "INGREDIENTS_DB",
        os.path.expanduser("~/AI-Training/ingredients.db"),
    )


def init_tables(cx):
    """Ensure the tables exist. The real vault db already has them; this lets tests
    and a fresh db work. Mirrors the vault migration in
    `02 Skills/condition_pathway.py` -- keep the two in step.

    The CHECK constraints here mirror the vault's, which are themselves generated
    from these same Python vocabulary tuples so the two cannot drift. Because the
    constraints exist, an invalid value reaching the UPDATE raises
    sqlite3.IntegrityError at the storage layer -- decide()'s own validation is
    the first line of defence and must reject cleanly before that ever fires.
    """
    cx.executescript(f"""
    CREATE TABLE IF NOT EXISTS canonical_pathways(
        id INTEGER PRIMARY KEY AUTOINCREMENT, slug TEXT NOT NULL UNIQUE,
        label TEXT NOT NULL, family TEXT, status TEXT NOT NULL DEFAULT 'proposed',
        notes TEXT, created_at TEXT NOT NULL, decided_at TEXT);
    CREATE TABLE IF NOT EXISTS conditions(
        id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL UNIQUE,
        label TEXT NOT NULL, system TEXT, batch TEXT NOT NULL, aliases TEXT,
        status TEXT NOT NULL DEFAULT 'proposed', notes TEXT,
        created_at TEXT NOT NULL, decided_at TEXT,
        scored INTEGER NOT NULL DEFAULT 1);
    CREATE TABLE IF NOT EXISTS condition_pathway(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        condition_key TEXT NOT NULL REFERENCES conditions(key) ON DELETE CASCADE,
        canonical_id INTEGER NOT NULL REFERENCES canonical_pathways(id) ON DELETE CASCADE,
        desired_direction TEXT NOT NULL CHECK(desired_direction IN {DIRECTIONS!r}),
        tier TEXT NOT NULL CHECK(tier IN {TIERS!r}),
        proposed_direction TEXT,
        proposed_tier TEXT,
        rationale TEXT NOT NULL,
        decision TEXT NOT NULL DEFAULT 'proposed' CHECK(decision IN ('proposed', 'confirmed', 'rejected')),
        reject_reason TEXT CHECK(reject_reason IS NULL OR reject_reason IN {REJECT_REASONS!r}),
        source TEXT NOT NULL, decided_at TEXT,
        UNIQUE(condition_key, canonical_id));
    CREATE INDEX IF NOT EXISTS idx_cp_condition ON condition_pathway(condition_key);
    CREATE INDEX IF NOT EXISTS idx_cp_canonical ON condition_pathway(canonical_id);
    CREATE TABLE IF NOT EXISTS condition_corpus_signal(
        condition_key TEXT NOT NULL, canonical_id INTEGER NOT NULL,
        n_products INTEGER NOT NULL, n_ingredients INTEGER NOT NULL,
        example_skus TEXT, computed_at TEXT NOT NULL,
        PRIMARY KEY (condition_key, canonical_id));
    CREATE TABLE IF NOT EXISTS condition_needs_canonical(
        id INTEGER PRIMARY KEY AUTOINCREMENT, condition_key TEXT NOT NULL,
        proposed_label TEXT NOT NULL, rationale TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'open', created_at TEXT NOT NULL);
    """)
    # CREATE TABLE IF NOT EXISTS is a no-op against the live vault table, which
    # predates these three columns, so they need their own guarded ALTER -- same
    # pattern as the vault's migrate(). The backfill here is the safe half only:
    # a row still 'proposed' has never been edited, so its current values ARE the
    # original proposal. A row already decided cannot have its original recovered,
    # and this module refuses to guess -- the vault's migrate() is the
    # authoritative migration and STOPS on that case rather than inventing a
    # baseline that would read as "nothing was corrected".
    cols = {r[1] for r in cx.execute("PRAGMA table_info(condition_pathway)")}
    if "proposed_direction" not in cols:
        cx.execute("ALTER TABLE condition_pathway ADD COLUMN proposed_direction TEXT")
    if "proposed_tier" not in cols:
        cx.execute("ALTER TABLE condition_pathway ADD COLUMN proposed_tier TEXT")
    if "reject_reason" not in cols:
        cx.execute("ALTER TABLE condition_pathway ADD COLUMN reject_reason TEXT "
                   "CHECK(reject_reason IS NULL OR reject_reason IN %r)" % (REJECT_REASONS,))
    cx.execute("UPDATE condition_pathway SET proposed_direction = desired_direction, "
               "proposed_tier = tier "
               "WHERE proposed_direction IS NULL AND decision = 'proposed'")
    cx.commit()


def _now(cx):
    return cx.execute("SELECT datetime('now')").fetchone()[0]


def stats(cx):
    cx.row_factory = sqlite3.Row
    rows = cx.execute("SELECT decision, count(*) n FROM condition_pathway "
                      "GROUP BY decision").fetchall()
    by = {r["decision"]: r["n"] for r in rows}
    return {"pending": by.get("proposed", 0),
            "confirmed": by.get("confirmed", 0),
            "rejected": by.get("rejected", 0),
            "conditions": cx.execute("SELECT count(*) FROM conditions").fetchone()[0]}


def queue(cx, limit=40):
    """The top `limit` undecided edges, HIGHEST-SIGNAL FIRST: most products touching
    the pathway, then core tier.

    Glen's ruling on the polarity. An edge whose pathway nothing in the catalogue
    touches is inert -- no product's rank can move on it, so a wrong direction there
    costs nothing today. A wrong direction on a pathway many products touch inverts
    live recommendations. Review effort goes where the damage is, not where the data
    is thinnest. (An earlier version sorted corpus-ABSENT edges first, on the theory
    that an uncovered pathway is the more suspicious row. It is more suspicious and
    less dangerous, and the queue orders by danger.)

    THERE IS NO `offset` PARAMETER, DELIBERATELY -- do not add one back. This query
    filters on `decision = 'proposed'`, so the set being paged SHRINKS with every
    decision. Offset paging over a mutating set skips rows permanently: decide 20 of
    the first 40, ask for offset=40, and you land past 40 edges you never saw, while
    the session looks like ordinary forward progress and nothing ever reports the
    hole. Because decided edges leave the set on their own, always reading from the
    start IS the paging -- progress is automatic and cannot skip. To see more, ask
    again.

    Deliberately NOT filtered on `conditions.scored`. The vault's `gate_metrics()`
    counts every edge belonging to a condition's batch regardless of `scored`, and
    its gate requires `len(decided) == total` -- this queue is the ONLY surface that
    can ever reach an edge, so hiding one here does not protect Glen, it stops that
    edge from ever being decided and deadlocks the batch gate permanently, with no
    visible cause. (An earlier version of this function filtered on `scored = 1`;
    an adversarial review reproduced exactly that deadlock through the real
    `load_batch` and this was reverted.) `c.scored` IS still selected below so an
    edge on an unscored condition can be flagged in the UI -- it should be visible,
    not hidden, because either the edge is fabricated or the `scored` flag is
    wrong, and Glen needs to see which. The refusal for this state belongs at the
    WRITER (`02 Skills/condition_pathway.py`'s `selfcheck`), not here.
    """
    cx.row_factory = sqlite3.Row
    return cx.execute("""
        SELECT cp.id, cp.condition_key, cp.canonical_id, cp.desired_direction,
               cp.tier, cp.rationale, cp.source,
               c.label AS condition_label, c.system, c.scored,
               p.label AS pathway_label, p.slug AS pathway_slug, p.family,
               COALESCE(s.n_products, 0)    AS corpus_products,
               COALESCE(s.n_ingredients, 0) AS corpus_ingredients,
               s.example_skus
          FROM condition_pathway cp
          JOIN conditions c        ON c.key = cp.condition_key
          JOIN canonical_pathways p ON p.id = cp.canonical_id
          LEFT JOIN condition_corpus_signal s
                 ON s.condition_key = cp.condition_key
                AND s.canonical_id  = cp.canonical_id
         WHERE cp.decision = 'proposed'
         ORDER BY COALESCE(s.n_products,0) DESC,
                  CASE cp.tier WHEN 'core' THEN 0 WHEN 'contributing' THEN 1 ELSE 2 END,
                  c.key, p.slug
         LIMIT ?""", (limit,)).fetchall()


def decide(cx, edge_id, decision, desired_direction=None, tier=None, rationale=None,
           reject_reason=None):
    """Record a decision on an edge that has not been decided yet.

    Returns a dict, never a bare id, because the caller has to tell three outcomes
    apart and two of them are failures that need opposite responses from the UI:

        {"ok": True,  "edge_id": N}
        {"ok": False, "error": "conflict", "decision": "confirmed"|"rejected"}
        {"ok": False, "error": "<invalid-*|unknown-edge>"}

    A CONFLICT is not a bad request. It means someone else already ruled on this
    edge -- a second browser tab left open on the queue, most likely -- and the
    honest thing for the UI to say is "another tab already decided this, reload",
    not "bad request". Telling the two apart is the whole reason for the dict.

    REFUSES to overwrite a decided edge. This read-then-write had no guard on what
    it read: two tabs showing the same card both submitted, the second write won,
    stamped a fresh decided_at, and was indistinguishable from a real ruling. On the
    retinitis-pigmentosa / Stargardt pair that is a safety interlock -- Stargardt
    wants vitamin A transport DOWN because vitamin A is contraindicated in ABCA4
    disease -- and the stale tab's 'up' silently reinstated the contraindicated
    direction. Re-deciding is still possible, the honest way: undo() first, which
    returns the edge to 'proposed'.

    `source` records provenance only ('glen' once a human has edited the values).
    NOTHING measures a flip from it -- the vault's gate compares desired_direction
    against proposed_direction, which this module never writes and undo() cannot
    reset. See the module docstring.
    """
    if decision not in DECISIONS:
        return {"ok": False, "error": "invalid-decision"}
    if desired_direction is not None and desired_direction not in DIRECTIONS:
        return {"ok": False, "error": "invalid-direction"}
    if tier is not None and tier not in TIERS:
        return {"ok": False, "error": "invalid-tier"}
    # Validated even on a confirmation, where the value is then ignored: an unknown
    # reason is a caller bug either way, and silently swallowing it on one branch
    # would let a typo'd 'wrong direction' reach the reject branch untested.
    if reject_reason is not None and reject_reason not in REJECT_REASONS:
        return {"ok": False, "error": "invalid-reject-reason"}
    cx.row_factory = sqlite3.Row
    row = cx.execute("SELECT * FROM condition_pathway WHERE id=?", (edge_id,)).fetchone()
    if row is None:
        return {"ok": False, "error": "unknown-edge"}
    if row["decision"] != "proposed":
        return {"ok": False, "error": "conflict", "decision": row["decision"]}

    new_dir = desired_direction if desired_direction is not None else row["desired_direction"]
    new_tier = tier if tier is not None else row["tier"]
    changed = (new_dir != row["desired_direction"]) or (new_tier != row["tier"])
    # A blank rationale means "the form field was empty", not "erase the recorded
    # reason". COALESCE alone falls through on NULL only, so any route that read an
    # untouched text input as "" wiped the justification for a safety-relevant
    # override, with no copy of it anywhere.
    new_rationale = rationale if (rationale or "").strip() else None
    cx.execute("UPDATE condition_pathway SET decision=?, desired_direction=?, tier=?, "
               "rationale=COALESCE(?, rationale), reject_reason=?, source=?, "
               "decided_at=? WHERE id=?",
               (decision, new_dir, new_tier, new_rationale,
                reject_reason if decision == "rejected" else None,
                "glen" if changed else row["source"], _now(cx), edge_id))
    cx.commit()
    return {"ok": True, "edge_id": edge_id}


def undo(cx, edge_id):
    """Return an edge to the queue so it can be decided again. Same dict shape as
    decide(), for the same reason.

    This is the ONLY way to re-decide an edge, since decide() refuses anything that
    is not 'proposed'. Clears the decision, the timestamp and the reject reason, and
    resets source to 'ai'.

    It leaves any corrected direction/tier in place -- the reviewer sees his own
    correction when the card comes back, which is what he wants. That used to erase
    the flip signal (source reset + corrected value left behind = a reconfirmation
    that compares the correction against itself); it no longer can, because the
    signal lives in proposed_direction, which nothing here writes.
    """
    cx.row_factory = sqlite3.Row
    row = cx.execute("SELECT decision FROM condition_pathway WHERE id=?",
                     (edge_id,)).fetchone()
    if row is None:
        return {"ok": False, "error": "unknown-edge"}
    cx.execute("UPDATE condition_pathway SET decision='proposed', source='ai', "
               "reject_reason=NULL, decided_at=NULL WHERE id=?", (edge_id,))
    cx.commit()
    return {"ok": True, "edge_id": edge_id}


def needs_canonical(cx, condition_key, label, rationale):
    """Record that a condition needs a pathway the vocabulary does not have. This
    NEVER creates the canonical: adding one is a separate decision made against the
    26 rules in `00 System/pathway-vocabulary-rules.md`.

    Validates condition_key against conditions the same way the sibling's
    create_canonical() guards its input -- there is no FK on this column, so a
    typo would otherwise silently create an orphaned row nothing ever surfaces.
    """
    label = (label or "").strip()
    if not condition_key or not label:
        return None
    known = cx.execute("SELECT 1 FROM conditions WHERE key=?", (condition_key,)).fetchone()
    if not known:
        return None
    cx.execute("INSERT INTO condition_needs_canonical"
               "(condition_key,proposed_label,rationale,created_at) VALUES(?,?,?,?)",
               (condition_key, label, rationale or "", _now(cx)))
    cx.commit()
    return label
