"""Increment 4b: the extensible Five-Fold Dimensions framework + the
depth-of-penetration reach match-check.

Seeds Glen's 15 five-fold dimensions (from 01 Clinical/five-fold-dimensions-of-healing.md)
as data, so adding bespoke logic later is a code change with no schema churn. Each
dimension declares which entities it can tag (`applies_to`). Depth of Penetration is the
one wired with a reach match-check: a remedy must reach at least as deep as the stress it
treats, or it is flagged. Tags attach to authored chain rows (stress side / remedy side).
"""
import json
import sqlite3

DEPTH_KEY = "depth_of_penetration"

# (key, name, ordered, applies_to, has_match_check, [(rank, value, code), ...])
DIMENSIONS = [
    (DEPTH_KEY, "Depth of Penetration", 1, ["stress", "remedy"], 1, [
        (1, "Gut / Environment", "gut"), (2, "Blood", "blood"),
        (3, "Extra-cellular fluid (incl. lymph)", "ecf"),
        (4, "Cytoplasm / cell metabolism", "cytoplasm"),
        (5, "Nucleoplasm / epigenetic", "nucleus")]),
    ("states_of_matter", "States of Matter", 1, ["stress"], 0, [
        (1, "Solid", "solid"), (2, "Liquid", "liquid"), (3, "Gas", "gas"),
        (4, "Plasma", "plasma"), (5, "Condensate", "condensate")]),
    ("cs_meta_model", "Meta-Model (5 C's)", 0, ["stress"], 0, [
        (1, "Context", "context"), (2, "Container", "container"),
        (3, "Content", "content"), (4, "Connection", "connection"),
        (5, "Communication", "communication")]),
    ("elements_tcm", "5 Elements (TCM)", 0, ["stress"], 0, [
        (1, "Wood", "wood"), (2, "Fire", "fire"), (3, "Earth", "earth"),
        (4, "Metal", "metal"), (5, "Water", "water")]),
    ("generations", "5 Generations of Inheritance", 1, ["stress"], 0, [
        (1, "Grandparents", "grandparents"), (2, "Parents", "parents"),
        (3, "Self", "self"), (4, "Children", "children"),
        (5, "Grandchildren", "grandchildren")]),
    ("infoceuticals", "Infoceuticals (5)", 0, ["remedy"], 0, [
        (1, "ET / Terrains", "et"), (2, "Source (Day/Night/BFA/Polarity)", "source"),
        (3, "ED / Strength-Organs", "ed"), (4, "EI / Flow-Meridians", "ei"),
        (5, "ES / Stars-Systems", "es")]),
    ("cardinal_signs", "5 Cardinal Signs", 0, ["stress"], 0, [
        (1, "Functio laesa", "functio_laesa"), (2, "Rubor", "rubor"),
        (3, "Dolor", "dolor"), (4, "Calor", "calor"), (5, "Tumor", "tumor")]),
    ("phases_of_terrain", "5 Phases of Terrain (5 R's)", 1, ["stress", "remedy"], 0, [
        (1, "Recharge", "recharge"), (2, "Rejuvenate", "rejuvenate"),
        (3, "Regenerate", "regenerate"), (4, "Reclaim", "reclaim"),
        (5, "Regulate", "regulate")]),
    ("pathology_types", "5 Pathology Types", 1, ["stress"], 0, [
        (1, "Hypertrophy", "hypertrophy"), (2, "Hyperplasia", "hyperplasia"),
        (3, "Metaplasia", "metaplasia"), (4, "Dysplasia", "dysplasia"),
        (5, "Neoplasia", "neoplasia")]),
    ("levels_of_therapy", "5 Levels of Therapy (5 S's)", 1, ["remedy"], 0, [
        (1, "Surgery", "surgery"), (2, "Suppression", "suppression"),
        (3, "Substitution", "substitution"), (4, "Support", "support"),
        (5, "Stimulation", "stimulation")]),
    ("levels_of_regulation", "5 Levels of Regulation", 1, ["stress"], 0, [
        (1, "Blocked", "blocked"), (2, "Negative", "negative"), (3, "Mixed", "mixed"),
        (4, "Positive", "positive"), (5, "Optimum", "optimum")]),
    ("stages_of_prognosis", "5 Stages of Prognosis", 1, ["stress"], 0, [
        (1, "Self-Limiting", "self_limiting"), (2, "Serious", "serious"),
        (3, "Degenerative", "degenerative"), (4, "Life-Threatening", "life_threatening"),
        (5, "Certain Death", "certain_death")]),
    ("tissue_layers", "5 Embryological Tissue Layers", 1, ["stress"], 0, [
        (1, "Compression", "compression"), (2, "Connection", "connection"),
        (3, "Conversion", "conversion"), (4, "Communication", "communication"),
        (5, "Containment", "containment")]),
    ("causal_cascade", "Causal Cascade (energy->matter)", 1, ["stress"], 0, [
        (1, "Biophysical Terrain", "terrain"), (2, "Bioenergetic Regulation", "regulation"),
        (3, "Epigenetics", "epigenetics"), (4, "Proteomics", "proteomics"),
        (5, "Damage", "damage")]),
    ("purpose_to_matter", "Transdimensional Causal Hierarchy", 1, ["stress"], 0, [
        (1, "Purpose", "purpose"), (2, "Meaning", "meaning"),
        (3, "Information", "information"), (4, "Energy", "energy"),
        (5, "Matter", "matter")]),
]


def init_dimension_tables(cx):
    cx.execute("CREATE TABLE IF NOT EXISTS biofield_dimensions("
               "key TEXT PRIMARY KEY, name TEXT, ordered INTEGER, applies_to TEXT, "
               "has_match_check INTEGER, sort_seq INTEGER)")
    cx.execute("CREATE TABLE IF NOT EXISTS biofield_dimension_values("
               "dim_key TEXT, rank INTEGER, value TEXT, code TEXT, "
               "PRIMARY KEY(dim_key, rank))")
    cx.execute("CREATE TABLE IF NOT EXISTS biofield_dimension_tags("
               "target_type TEXT, target_id TEXT, dim_key TEXT, value_rank INTEGER, "
               "PRIMARY KEY(target_type, target_id, dim_key))")
    cx.commit()


def seed_dimensions(cx):
    init_dimension_tables(cx)
    for seq, (key, name, ordered, applies, mc, vals) in enumerate(DIMENSIONS):
        cx.execute("INSERT OR REPLACE INTO biofield_dimensions"
                   "(key,name,ordered,applies_to,has_match_check,sort_seq) VALUES(?,?,?,?,?,?)",
                   (key, name, ordered, json.dumps(applies), mc, seq))
        for rank, value, code in vals:
            cx.execute("INSERT OR REPLACE INTO biofield_dimension_values"
                       "(dim_key,rank,value,code) VALUES(?,?,?,?)", (key, rank, value, code))
    cx.commit()


def list_dimensions(cx):
    init_dimension_tables(cx)
    cx.row_factory = sqlite3.Row
    rows = cx.execute("SELECT * FROM biofield_dimensions ORDER BY sort_seq").fetchall()
    return [{"key": r["key"], "name": r["name"], "ordered": r["ordered"],
             "applies_to": json.loads(r["applies_to"] or "[]"),
             "has_match_check": r["has_match_check"]} for r in rows]


def dimension_values(cx, dim_key):
    init_dimension_tables(cx)
    cx.row_factory = sqlite3.Row
    rows = cx.execute("SELECT rank, value, code FROM biofield_dimension_values "
                      "WHERE dim_key=? ORDER BY rank", (dim_key,)).fetchall()
    return [{"rank": r["rank"], "value": r["value"], "code": r["code"]} for r in rows]


def tag(cx, target_type, target_id, dim_key, rank):
    init_dimension_tables(cx)
    if rank in (None, "", 0, "0"):
        cx.execute("DELETE FROM biofield_dimension_tags WHERE target_type=? AND target_id=? AND dim_key=?",
                   (target_type, str(target_id), dim_key))
    else:
        cx.execute("INSERT OR REPLACE INTO biofield_dimension_tags"
                   "(target_type,target_id,dim_key,value_rank) VALUES(?,?,?,?)",
                   (target_type, str(target_id), dim_key, int(rank)))
    cx.commit()


def get_tag(cx, target_type, target_id, dim_key):
    init_dimension_tables(cx)
    r = cx.execute("SELECT value_rank FROM biofield_dimension_tags "
                   "WHERE target_type=? AND target_id=? AND dim_key=?",
                   (target_type, str(target_id), dim_key)).fetchone()
    return r[0] if r else None


def depth_match(stress_rank, remedy_rank):
    """A remedy must reach at least as deep as the stress it treats."""
    if stress_rank is None or remedy_rank is None:
        return "unknown"
    return "ok" if remedy_rank >= stress_rank else "shallow"


def depth_label(cx, rank):
    if rank is None:
        return ""
    r = cx.execute("SELECT value FROM biofield_dimension_values WHERE dim_key=? AND rank=?",
                   (DEPTH_KEY, int(rank))).fetchone()
    return r[0] if r else ""
