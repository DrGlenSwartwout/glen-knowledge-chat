"""CRUD for the condition -> canonical pathway review queue (vault ingredients.db).

The mirror of pathway_review, one axis over: there the reviewer maps an ingredient's
atom to a canonical pathway, here he confirms that a CONDITION implicates one, in a
stated direction and at a stated importance.

Direction is the field that matters. A missing edge makes a remedy invisible; an
inverted one makes the score recommend the opposite of what helps. So `decide` records
a changed direction or tier as source='glen', which is how the vault's batch gate
measures whether the generator is trustworthy enough to run the systemic batch. A
plain confirmation leaves source='ai' -- otherwise every confirmation would read as a
flip and the gate would never open.

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
        rationale TEXT NOT NULL,
        decision TEXT NOT NULL DEFAULT 'proposed' CHECK(decision IN ('proposed', 'confirmed', 'rejected')),
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


def queue(cx, limit=40, offset=0):
    """Undecided edges, worst-first: edges whose corpus signal is absent (nothing in
    the catalog touches that pathway) sort ahead of the rest, then core tier, because
    those are where a wrong direction costs the most.

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
         ORDER BY (COALESCE(s.n_products,0) = 0) DESC,
                  CASE cp.tier WHEN 'core' THEN 0 WHEN 'contributing' THEN 1 ELSE 2 END,
                  c.key, p.slug
         LIMIT ? OFFSET ?""", (limit, offset)).fetchall()


def decide(cx, edge_id, decision, desired_direction=None, tier=None, rationale=None):
    """Record a decision. Returns the edge id, or None on bad input.

    source becomes 'glen' ONLY when the direction or tier actually changed. That is
    what the vault's gate_metrics counts as a flip, and a plain confirmation must not
    be mistaken for one.
    """
    if decision not in DECISIONS:
        return None
    if desired_direction is not None and desired_direction not in DIRECTIONS:
        return None
    if tier is not None and tier not in TIERS:
        return None
    cx.row_factory = sqlite3.Row
    row = cx.execute("SELECT * FROM condition_pathway WHERE id=?", (edge_id,)).fetchone()
    if row is None:
        return None

    new_dir = desired_direction if desired_direction is not None else row["desired_direction"]
    new_tier = tier if tier is not None else row["tier"]
    changed = (new_dir != row["desired_direction"]) or (new_tier != row["tier"])
    cx.execute("UPDATE condition_pathway SET decision=?, desired_direction=?, tier=?, "
               "rationale=COALESCE(?, rationale), source=?, decided_at=? WHERE id=?",
               (decision, new_dir, new_tier, rationale,
                "glen" if changed else row["source"], _now(cx), edge_id))
    cx.commit()
    return edge_id


def undo(cx, edge_id):
    """Return an edge to the queue. Resets source to 'ai' so an undone flip stops
    counting toward the gate."""
    cx.execute("UPDATE condition_pathway SET decision='proposed', source='ai', "
               "decided_at=NULL WHERE id=?", (edge_id,))
    cx.commit()
    return edge_id


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
