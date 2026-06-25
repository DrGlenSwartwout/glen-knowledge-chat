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
    cx.execute("CREATE TABLE IF NOT EXISTS biofield_video_scripts "
               "(test_id TEXT PRIMARY KEY, script TEXT, updated_at TEXT)")
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


def get_video_script(cx, test_id):
    return _get(cx, "biofield_video_scripts", "script", test_id)


def save_video_script(cx, test_id, script):
    _save(cx, "biofield_video_scripts", "script", test_id, script)


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
    "- GROUNDED VOICE: write the way a calm clinician speaks to a patient -- concrete, warm, "
    "direct, plain. NO literary or poetic metaphors and NO ornamental flourish: do not call the "
    "analysis 'fascinating', do not use figures like 'a painting', 'weaving a story', 'tapestry', "
    "'cunning', 'a narrative of health', or 'journey'. Prefer short, plain sentences over flowery "
    "ones. Describe what was found and what to do, not how poetic it is.\n"
    "- Close reassuringly: they need not absorb every detail; the body showed where to begin and "
    "in what order; start gently, watch, adjust; invite questions.\n"
    "- Sign off exactly: 'In wellness,' then 'Dr. Glen & Rae'.\n"
    "This is a DRAFT for Dr. Glen's review."
)


_SCAN_GUIDANCE = (
    "\n- If a RECENT E4L VOICE SCAN block is present, you may reference what the scan "
    "showed as corroborating context for the causal chain. Use observation language; "
    "do not invent scan findings beyond those listed, and do not treat a scan marked "
    "stale as current.")

_PROFILE_GUIDANCE = (
    "\n- If a CLIENT-STATED CONCERNS block is present, acknowledge the client's own "
    "stated symptoms, challenges, and goals in plain, validating language and connect "
    "them to the causal chain where honest to do so. Do not invent concerns beyond those listed.")

_PROFILE_FIELDS = ("conditions", "challenges", "goals", "tags", "terrain_concerns", "body_systems")


def _profile_content(profile):
    return bool(profile) and any(str((profile or {}).get(f) or "").strip() for f in _PROFILE_FIELDS)


def _profile_block(profile):
    if not _profile_content(profile):
        return ""
    lines = ["CLIENT-STATED CONCERNS (acknowledge in the client's own terms):"]
    for f in _PROFILE_FIELDS:
        v = profile.get(f)
        if isinstance(v, (list, tuple)):
            v = ", ".join(str(x).strip() for x in v if str(x).strip())
        v = str(v or "").strip()
        if v:
            lines.append(f"- {f.replace('_', ' ')}: {v}")
    return "\n".join(lines)


def _narrative_findings(scan):
    """The scan findings fed to the patient narrative = INFOCEUTICALS only. ER/MR
    'stresses' are information Glen doesn't balance, so they stay off the patient
    message. Falls back to splitting `findings` by group for un-split callers."""
    if not (scan and scan.get("found")):
        return []
    fs = scan.get("infoceuticals")
    if fs is None:
        fs = [f for f in (scan.get("findings") or []) if f.get("group") != "stress"]
    return fs or []


def _scan_block(scan):
    """Optional context block from the client's most recent E4L voice scan. Empty
    string unless the scan has infoceutical findings (back-compatible)."""
    findings = _narrative_findings(scan)
    if not findings:
        return ""
    days = scan.get("days_ago")
    age = f"{days} day{'s' if days != 1 else ''} ago" if days is not None else "date unknown"
    fresh = "fresh" if scan.get("fresh") else "STALE — older than the 2-week window"
    lines = [f"RECENT E4L VOICE SCAN ({age}, {fresh}; scan {scan.get('scan_date') or ''}):"]
    for f in findings:
        rank = f.get("rank")
        desc = (f.get("description") or "").strip()
        lines.append(f"- {('#' + str(rank) + ' ') if rank is not None else ''}"
                     f"{f.get('code') or ''} {f.get('name') or ''}"
                     f"{(' — ' + desc) if desc else ''}".rstrip())
    return "\n".join(lines)


def _user_block(report, notes, scan=None, profile=None):
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
    sb = _scan_block(scan)
    if sb:
        lines += ["", sb]
    pb = _profile_block(profile)
    if pb:
        lines += ["", pb]
    lines += ["", "CLINICIAN VERBAL NOTES (weave in naturally):", (notes or "(none)")]
    return "\n".join(lines)


def _system_with_scan(base, scan):
    """Append scan guidance only when the narrative actually carries scan findings, so
    the no-scan prompt stays byte-identical to before (back-compat)."""
    return base + (_SCAN_GUIDANCE if _narrative_findings(scan) else "")


def build_narrative_prompt(report, notes, scan=None, profile=None):
    system = _system_with_scan(_SYSTEM, scan)
    if _profile_content(profile):
        system += _PROFILE_GUIDANCE
    return {"system": system, "user": _user_block(report, notes, scan, profile)}


def generate_narrative(report, notes, complete, scan=None, profile=None):
    """complete(system, user) -> narrative text. scan = E4L context; profile = People-hub context."""
    p = build_narrative_prompt(report, notes, scan, profile)
    return complete(p["system"], p["user"])


_VIDEO_SYSTEM = (
    "You are Dr. Glen Swartwout speaking ALOUD to a patient -- recording a short voice "
    "walkthrough of their Biofield Analysis. Output ONLY the words to be spoken: no stage "
    "directions, no headings, no markdown, no remedy bullet list. RULES:\n"
    "- SHORT: about 150 words, roughly 60-90 seconds spoken. Give an overview plus the 2-3 most "
    "important layers and their key remedy -- NOT every layer or every dose.\n"
    "- Open 'Aloha <first name>,' and speak warmly in the first person ('I', 'we'), "
    "conversational and plain, the way you'd talk to them across the table.\n"
    "- Frame the causal chain simply: the most recent layer sits on top, deeper roots beneath, "
    "and supporting them in order lets the body unwind and self-correct.\n"
    "- OBSERVATION LANGUAGE: the body 'identified' / 'showed' / 'pointed to'. NEVER 'probably', "
    "'should', 'most likely'.\n"
    "- Fold in the clinician's verbal notes naturally if they fit.\n"
    "- GROUNDED VOICE: plain, warm, direct. No literary or poetic metaphors, no AI filler.\n"
    "- Name where to begin and reassure them: start gently, watch, adjust, and you'll guide them. "
    "Close warmly. This is a DRAFT for Dr. Glen's review."
)


def build_video_script_prompt(report, notes, scan=None):
    return {"system": _system_with_scan(_VIDEO_SYSTEM, scan),
            "user": _user_block(report, notes, scan)}


def generate_video_script(report, notes, complete, scan=None):
    """complete(system, user) -> short spoken walkthrough script. `scan` optional."""
    p = build_video_script_prompt(report, notes, scan)
    return complete(p["system"], p["user"])
