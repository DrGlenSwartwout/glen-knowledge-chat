"""CRUD for the canonical functional-pathway vocabulary (vault ingredients.db).

`ingredient_pathways.pathway` is free text — 3,949 rows carrying 3,855 distinct
strings — so pathway coverage cannot rank remedies against a condition until the
vocabulary repeats across ingredients. The atomizer in the vault
(`02 Skills/pathway_atomize.py`) splits those strings into atoms and proposes
canonical merges; this module is the review side: Glen confirms, rejects, or
orphans each atom, and the decision sticks so a re-propose never asks twice.

Non-destructive by construction — no original pathway string is ever modified.
Decisions live in `pathway_atom_map`, keyed by the normalized atom, so
regenerating the derived `pathway_atoms` cache never loses a confirmation.

Pure sqlite; the caller passes the connection. Same shape as formulation_map.
"""
import os
import sqlite3

from dashboard import dbwrite

# Glen may set: confirmed (the merge stands), orphan (a real pathway but its
# own — stays unmapped and contributes no overlap, which is honest rather than
# lossy), rejected (not a functional pathway at all).
GLEN_DECISIONS = ("confirmed", "orphan", "rejected")
PENDING = ("proposed", "proposed-reject")
DIRECTIONS = ("up", "down", "balance", "substrate", "adverse", "neutral", "unknown")


def _db_path():
    return os.environ.get(
        "INGREDIENTS_DB",
        os.path.expanduser("~/AI-Training/ingredients.db"),
    )


def init_tables(cx):
    """Ensure the canonical-vocabulary tables exist (the real vault db already
    has them; this lets tests and a fresh db work). Mirrors the vault migration
    in `02 Skills/pathway_canonical.py` — keep the two in step."""
    cx.executescript("""
    CREATE TABLE IF NOT EXISTS ingredient_pathways(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ingredient_id INTEGER,
        pathway TEXT NOT NULL, effect TEXT, notes TEXT);
    CREATE TABLE IF NOT EXISTS canonical_pathways(
        id INTEGER PRIMARY KEY AUTOINCREMENT, slug TEXT NOT NULL UNIQUE,
        label TEXT NOT NULL, family TEXT, status TEXT NOT NULL DEFAULT 'proposed',
        notes TEXT, created_at TEXT NOT NULL, decided_at TEXT);
    CREATE TABLE IF NOT EXISTS pathway_atoms(
        id INTEGER PRIMARY KEY AUTOINCREMENT, pathway_row_id INTEGER NOT NULL,
        ingredient_id INTEGER, position INTEGER NOT NULL, atom TEXT NOT NULL,
        atom_key TEXT NOT NULL, is_annotation INTEGER NOT NULL DEFAULT 0);
    CREATE TABLE IF NOT EXISTS pathway_atom_map(
        atom_key TEXT PRIMARY KEY, canonical_id INTEGER, decision TEXT NOT NULL,
        source TEXT NOT NULL, rationale TEXT, decided_at TEXT);
    CREATE TABLE IF NOT EXISTS pathway_effect_direction(
        effect_key TEXT PRIMARY KEY, direction TEXT NOT NULL, confidence TEXT NOT NULL,
        decision TEXT NOT NULL DEFAULT 'proposed', n_rows INTEGER NOT NULL DEFAULT 0,
        decided_at TEXT);
    CREATE INDEX IF NOT EXISTS idx_pathway_atoms_key ON pathway_atoms(atom_key);
    CREATE INDEX IF NOT EXISTS idx_pathway_atoms_row ON pathway_atoms(pathway_row_id);
    """)
    cx.commit()


def _now(cx):
    return cx.execute("SELECT datetime('now')").fetchone()[0]


def _slugify(label):
    out = "".join(c if c.isalnum() else "-" for c in (label or "").lower())
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-") or "pathway"


# --------------------------------------------------------------------------- read

def stats(cx):
    """Progress header: how much of the vocabulary is settled."""
    cx.row_factory = sqlite3.Row
    q = lambda s, *a: cx.execute(s, a).fetchone()[0]  # noqa: E731
    settled = q("SELECT COUNT(*) FROM pathway_atom_map WHERE decision NOT IN (?,?)", *PENDING)
    pending = q("SELECT COUNT(*) FROM pathway_atom_map WHERE decision IN (?,?)", *PENDING)
    return {
        "settled": settled,
        "pending": pending,
        "canonical": q("SELECT COUNT(*) FROM canonical_pathways"),
        "canonical_confirmed": q("SELECT COUNT(*) FROM canonical_pathways WHERE status='confirmed'"),
        "atoms_total": q("SELECT COUNT(DISTINCT atom_key) FROM pathway_atoms WHERE is_annotation=0"),
        "directions_pending": q(
            "SELECT COUNT(*) FROM pathway_effect_direction WHERE decision='proposed'"),
    }


def canonicals(cx):
    """Every canonical pathway, for the reassign picker."""
    cx.row_factory = sqlite3.Row
    rows = cx.execute(
        "SELECT c.id, c.slug, c.label, c.family, c.status, c.notes, "
        "  (SELECT COUNT(*) FROM pathway_atom_map m WHERE m.canonical_id=c.id) atoms "
        "FROM canonical_pathways c ORDER BY c.family, c.label").fetchall()
    return [dict(r) for r in rows]


def examples(cx, atom_key, limit=3):
    """Raw pathway strings this atom came from — the evidence Glen judges against."""
    cx.row_factory = sqlite3.Row
    rows = cx.execute(
        "SELECT DISTINCT p.pathway FROM pathway_atoms a "
        "JOIN ingredient_pathways p ON p.id=a.pathway_row_id "
        "WHERE a.atom_key=? ORDER BY LENGTH(p.pathway) LIMIT ?",
        (atom_key, int(limit))).fetchall()
    return [r["pathway"] for r in rows]


def queue(cx, limit=40, offset=0, include_singletons=False):
    """Pending atoms, highest-impact first (most ingredients touched).

    Batch 1 is the atoms shared by >=2 ingredients — empirically what recurs in
    Glen's own corpus, and the batch that unblocks scoring. Singletons are a
    much longer tail and only load when asked for.
    """
    cx.row_factory = sqlite3.Row
    having = "" if include_singletons else "HAVING ingredients >= 2"
    rows = cx.execute(
        f"""
        SELECT a.atom_key,
               MIN(a.atom) AS atom,
               COUNT(DISTINCT a.ingredient_id) AS ingredients,
               COUNT(*) AS mentions,
               m.decision, m.canonical_id, m.rationale,
               c.label AS canonical_label, c.slug AS canonical_slug
        FROM pathway_atoms a
        JOIN pathway_atom_map m ON m.atom_key = a.atom_key
        LEFT JOIN canonical_pathways c ON c.id = m.canonical_id
        WHERE a.is_annotation = 0 AND m.decision IN (?,?)
        GROUP BY a.atom_key
        {having}
        ORDER BY ingredients DESC, mentions DESC, a.atom_key
        LIMIT ? OFFSET ?
        """, (*PENDING, int(limit), int(offset))).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["examples"] = examples(cx, r["atom_key"])
        out.append(d)
    return out


def pending_count(cx, include_singletons=False):
    having = "" if include_singletons else "HAVING COUNT(DISTINCT a.ingredient_id) >= 2"
    return cx.execute(
        f"""SELECT COUNT(*) FROM (
              SELECT a.atom_key FROM pathway_atoms a
              JOIN pathway_atom_map m ON m.atom_key=a.atom_key
              WHERE a.is_annotation=0 AND m.decision IN (?,?)
              GROUP BY a.atom_key {having})""", PENDING).fetchone()[0]


# -------------------------------------------------------------------------- write

def decide(cx, atom_key, decision, canonical_id=None, rationale=None):
    """Record Glen's call on one atom. Confirming a merge also confirms the
    canonical it merges into — a canonical nobody merges into stays proposed."""
    atom_key = (atom_key or "").strip()
    if not atom_key or decision not in GLEN_DECISIONS:
        return None
    # An orphan or a rejection carries no canonical link, whatever was proposed.
    cid = int(canonical_id) if (decision == "confirmed" and canonical_id) else None
    cx.execute(
        "INSERT INTO pathway_atom_map (atom_key, canonical_id, decision, source, "
        "rationale, decided_at) VALUES (?,?,?,'glen',?,?) "
        "ON CONFLICT(atom_key) DO UPDATE SET canonical_id=excluded.canonical_id, "
        "decision=excluded.decision, source='glen', rationale=excluded.rationale, "
        "decided_at=excluded.decided_at",
        (atom_key, cid, decision, rationale, _now(cx)))
    if cid:
        cx.execute("UPDATE canonical_pathways SET status='confirmed', decided_at=? "
                   "WHERE id=? AND status!='confirmed'", (_now(cx), cid))
    cx.commit()
    return {"atom_key": atom_key, "decision": decision, "canonical_id": cid}


def undo(cx, atom_key):
    """Put one atom back in the queue — a bad merge is reversible without
    re-ingesting anything, which is the point of keeping the originals."""
    cx.execute("UPDATE pathway_atom_map SET decision='proposed', source='glen', "
               "decided_at=? WHERE atom_key=?", (_now(cx), (atom_key or "").strip()))
    cx.commit()
    return {"atom_key": atom_key, "decision": "proposed"}


def create_canonical(cx, label, family=None, notes=None):
    """A new canonical pathway Glen authors during review.

    Only ever call this for a pathway the corpus actually needs — never to give
    an orphan a home. An orphan that stays unmapped costs nothing; a canonical
    invented to house one atom pollutes every score that reads the vocabulary.
    """
    label = (label or "").strip()
    if not label:
        return None
    slug = _slugify(label)
    row = cx.execute("SELECT id FROM canonical_pathways WHERE slug=?", (slug,)).fetchone()
    if row:
        return {"id": row[0], "slug": slug, "label": label, "created": False}
    new_id = dbwrite.insert_returning_id(
        cx, "INSERT INTO canonical_pathways (slug, label, family, status, notes, "
            "created_at, decided_at) VALUES (?,?,?,'confirmed',?,?,?)",
        (slug, label, family, notes, _now(cx), _now(cx)))
    cx.commit()
    return {"id": new_id, "slug": slug, "label": label, "created": True}


# ------------------------------------------------------------------- effect axis

def direction_queue(cx, limit=40):
    """Effect phrasings the classifier could not place confidently.

    Direction is a SEPARATE axis from pathway identity: NF-kB inhibited and
    NF-kB activated are one pathway, but an ingredient that drives a pathway the
    wrong way for a condition must not be credited as covering it, and an
    ADVERSE row must never add coverage at all.
    """
    cx.row_factory = sqlite3.Row
    rows = cx.execute(
        "SELECT effect_key, direction, confidence, n_rows FROM pathway_effect_direction "
        "WHERE decision='proposed' AND (direction='unknown' OR confidence='low') "
        "ORDER BY n_rows DESC, effect_key LIMIT ?", (int(limit),)).fetchall()
    return [dict(r) for r in rows]


def set_direction(cx, effect_key, direction):
    if direction not in DIRECTIONS:
        return None
    cx.execute("UPDATE pathway_effect_direction SET direction=?, confidence='high', "
               "decision='confirmed', decided_at=? WHERE effect_key=?",
               (direction, _now(cx), (effect_key or "").strip()))
    cx.commit()
    return {"effect_key": effect_key, "direction": direction}
