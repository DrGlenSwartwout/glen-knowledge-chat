"""Increment 2: verbal-notes + narrative for the local Biofield Analysis viewer.

Stores Glen's per-test verbal notes and the generated narrative locally, builds the
Glen-voice prompt (following the biofield-causal-chain-narrative skill rules), and
generates the narrative via an injected LLM callable `complete(system, user) -> str`
so the logic is testable without a live API call.
"""
import datetime
import sqlite3


def _now():
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def init_notes_tables(cx):
    cx.execute("CREATE TABLE IF NOT EXISTS biofield_notes "
               "(test_id TEXT PRIMARY KEY, notes TEXT, updated_at TEXT)")
    cx.execute("CREATE TABLE IF NOT EXISTS biofield_narratives "
               "(test_id TEXT PRIMARY KEY, narrative TEXT, updated_at TEXT)")
    cx.commit()


def _get(cx, table, col, test_id):
    init_notes_tables(cx)
    row = cx.execute(f"SELECT {col} FROM {table} WHERE test_id=?", (str(test_id),)).fetchone()
    return (row[0] if row and row[0] else "")


def _save(cx, table, col, test_id, val):
    init_notes_tables(cx)
    cx.execute(
        f"INSERT INTO {table} (test_id, {col}, updated_at) VALUES (?,?,?) "
        f"ON CONFLICT(test_id) DO UPDATE SET {col}=excluded.{col}, updated_at=excluded.updated_at",
        (str(test_id), val or "", _now()))
    cx.commit()


def get_notes(cx, test_id):
    return _get(cx, "biofield_notes", "notes", test_id)


def save_notes(cx, test_id, notes):
    _save(cx, "biofield_notes", "notes", test_id, notes)


def get_narrative(cx, test_id):
    return _get(cx, "biofield_narratives", "narrative", test_id)


def save_narrative(cx, test_id, narrative):
    _save(cx, "biofield_narratives", "narrative", test_id, narrative)


_SYSTEM = (
    "You write in Dr. Glen Swartwout's warm, calm clinical voice, as a letter to a "
    "patient about their Biofield Analysis (a Causal Chain Report). RULES:\n"
    "- Open with 'Aloha <first name>,' then 2-3 warm sentences framing the causal chain: "
    "the most recent layer sits on top, deeper and older roots beneath, and supporting them "
    "in order lets the chain unwind and the body self-correct.\n"
    "- One short plain-English paragraph per layer, top-down (Layer 1 = most recent/surface "
    "first, down to the deepest root). Name the remedy and its dosing for that layer.\n"
    "- DRAW THE RELATIONSHIPS: explain how each layer connects to the others -- how a surface "
    "layer sits on or is driven by a deeper root -- so the chain reads as one connected story, "
    "not a list.\n"
    "- OBSERVATION LANGUAGE ONLY: the body 'identified' / 'showed coherence with' / the remedy "
    "was 'detected as best suited'. NEVER 'probably', 'should', 'most likely', or any hedge.\n"
    "- Fold the clinician's verbal notes in naturally where they fit; do not quote them as a list.\n"
    "- Plain English; translate any technical codes. No jargon, no emojis, no AI-pleasantry "
    "filler ('I hope you're well'). Open with substance.\n"
    "- Close reassuringly: they need not absorb every detail; the body showed where to begin and "
    "in what order; start gently, watch, adjust; invite questions.\n"
    "- Sign off exactly: 'In wellness,' then 'Dr. Glen & Rae'.\n"
    "This is a DRAFT for Dr. Glen's review."
)


def build_narrative_prompt(report, notes):
    c = report.get("client") or {}
    lines = [f"PATIENT: {c.get('name') or ''}",
             f"DATE: {report.get('date') or ''}",
             "",
             "CAUSAL CHAIN (top-down, most recent layer first to deepest root):"]
    for l in report.get("layers") or []:
        ln = l.get("layer")
        lines.append(
            f"- Layer {ln if ln is not None else '?'}: {l.get('head') or ''}"
            f" (most affected: {l.get('most_affected') or ''})"
            f" -> remedy: {l.get('remedy') or ''}; dose: {l.get('dosage') or ''}"
            f" {l.get('frequency') or ''} {l.get('timing') or ''}".rstrip())
    lines += ["", "CLINICIAN VERBAL NOTES (weave in naturally):",
              (notes or "(none)")]
    return {"system": _SYSTEM, "user": "\n".join(lines)}


def generate_narrative(report, notes, complete):
    """complete(system, user) -> narrative text."""
    p = build_narrative_prompt(report, notes)
    return complete(p["system"], p["user"])
