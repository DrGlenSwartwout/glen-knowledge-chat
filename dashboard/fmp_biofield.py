"""FMP -> portal biofield importer.

Pulls a client's Causal Chain Report from the local FMP snapshot tables (fmp_snap_*
in chat_log.db), maps it to the portal layer schema, and (via the app's LLM) drafts
the warm prose in Glen's voice. Self-contained; the editor's "Import from FMP" button
calls import_content(). (Prior versions read the live Supabase fmp_newapp mirror,
which was decommissioned — see fetch_causal_chain.)

Layer 1 = most recent/surface; higher = deeper root (narrative-skill rule).
The FMP auto-"Consider" remedy lines and blank-remedy rows are skipped.
"""
import json
import os
import re
import sqlite3
from pathlib import Path

_MAP_PATH = os.path.join(os.path.dirname(__file__), "data", "infoceutical_codes.json")
try:
    with open(_MAP_PATH, encoding="utf-8") as _f:
        CODE_MAP = json.load(_f)
except Exception:
    CODE_MAP = {}

# Leading code token: a letter-cluster + number ("ED9", "ES13", "MB 6") OR a bare
# word code ("BFA", "Day"). The rest of the string is FMP's own plain-English name.
_CODE_RE = re.compile(r"^\s*([A-Za-z]{1,3}\s?\d+|[A-Za-z]{2,5})\s*(.*)$")


def translate_head_chain(s):
    """A clean layer title from an FMP head_chain like 'ED9 Muscle Driver'.
    Prefer FMP's trailing plain-English name; else the shipped code map; else raw."""
    s = (s or "").strip()
    if not s:
        return ""
    m = _CODE_RE.match(s)
    if not m:
        return s
    code = re.sub(r"\s+", "", m.group(1))   # "ED 9" / "ED9" -> "ED9"
    rest = (m.group(2) or "").strip()
    if rest:
        return rest
    return CODE_MAP.get(code, s)


def build_layers(rows):
    """FMP causal-chain rows -> portal layers (n/title/meaning/remedy/dosing).
    Drops Consider-auto-suggestions and blank-remedy rows; orders by FMP layer and
    renumbers 1..k. `meaning` is left blank for draft_prose() to fill."""
    keep = []
    for r in rows or []:
        remedy = (r.get("remedy") or "").strip()
        if not remedy or remedy.lower().startswith("consider"):
            continue
        try:
            layer = int(str(r.get("layer") or "").strip())
        except (TypeError, ValueError):
            layer = 9999
        keep.append((layer, {
            "title": translate_head_chain(r.get("head_chain")),
            "remedy": remedy,
            "dosing": (r.get("dosage") or "").strip(),
        }))
    keep.sort(key=lambda x: x[0])
    return [{"n": i, "title": d["title"], "meaning": "", "remedy": d["remedy"], "dosing": d["dosing"]}
            for i, (_, d) in enumerate(keep, start=1)]


# The live FMP mirror (Supabase schema fmp_newapp) was decommissioned; its data now
# lives in the local FMP snapshot tables (fmp_snap_*) in chat_log.db. The causal-chain
# remedies (with dosing) live in fmp_snap_client_remedy: `layer` is the chain layer,
# `data_2` is the Head-of-Chain label, `remedy`/`dosage` the protocol, and
# `zc_dosage_text` FMP's composed dosing line ("1 scoop - 2 times a day - in water").
_SNAP_SQL = (
    "SELECT r.layer AS layer, r.data_2 AS head_chain, r.remedy AS remedy, "
    "COALESCE(NULLIF(TRIM(r.zc_dosage_text), ''), r.dosage) AS dosage "
    "FROM fmp_snap_client_remedy r "
    "JOIN fmp_clients c ON c.id_pk = r.id_fk_client "
    "WHERE lower(c.email) = ? AND r.id_fk_test = ? "
    "ORDER BY CAST(r.layer AS INTEGER) ASC, CAST(r.id_pk AS INTEGER) ASC")

# Snapshot `active` is uniformly 'Yes' (an export artifact) so it can't isolate the
# current test the way live FMP's cbt.active=TRUE did. Scope to the client's newest
# test instead (highest id_fk_test — PKs increment over time = the report's subject).
_LATEST_TEST_SQL = (
    "SELECT r.id_fk_test FROM fmp_snap_client_remedy r "
    "JOIN fmp_clients c ON c.id_pk = r.id_fk_client "
    "WHERE lower(c.email) = ? "
    "ORDER BY CAST(r.id_fk_test AS INTEGER) DESC LIMIT 1")


def _snapshot_db_path():
    base = os.environ.get("DATA_DIR") or str(Path(__file__).resolve().parents[1])
    return str(Path(base) / "chat_log.db")


def fetch_causal_chain(email, conn=None):
    """Pull a client's most-recent Causal Chain remedies from the local FMP snapshot
    (fmp_snap_* in chat_log.db), mapped to the portal-layer shape
    {layer, head_chain, remedy, dosage}. Returns [] on no match / any error.

    `conn` is an optional open sqlite connection (tests inject a fixture DB);
    otherwise chat_log.db is opened read-only for the call. Dosing prefers FMP's
    composed zc_dosage_text over the bare dosage. Authoring is a LOCAL console
    operation — prod's chat_log.db may lack the snapshot tables, in which case this
    returns [] gracefully and the editor shows "not found"."""
    email = (email or "").strip().lower()
    if not email:
        return []
    own = conn is None
    cx = None
    try:
        cx = conn or sqlite3.connect(_snapshot_db_path())
        cx.row_factory = sqlite3.Row
        latest = cx.execute(_LATEST_TEST_SQL, (email,)).fetchone()
        if not latest:
            return []
        rows = cx.execute(_SNAP_SQL, (email, latest[0])).fetchall()
        return [{"layer": r["layer"], "head_chain": r["head_chain"],
                 "remedy": r["remedy"], "dosage": r["dosage"]} for r in rows]
    except Exception as e:
        print(f"[fmp-import] snapshot fetch failed: {e!r}", flush=True)
        return []
    finally:
        if own and cx is not None:
            cx.close()


_VOICE = (
    "You write in Dr. Glen Swartwout's warm clinical voice for a patient's biofield "
    "portal. RULES: plain English, no jargon codes. OBSERVATION voice — the body "
    "'identified' / 'showed coherence with' the remedies; NEVER 'probably', 'should', "
    "'most likely'. Layer 1 is the MOST RECENT/surface; higher layers are deeper, older "
    "roots. NO AI-pleasantry filler (no 'I hope you're doing well', no 'I wanted to "
    "write this for you'); open with substance. No emojis. "
    "Return STRICT JSON: {\"greeting\": str, \"layers\": [{\"title\": str, \"meaning\": str}]} "
    "with one layers entry per input layer IN ORDER. greeting: 2-3 warm sentences opening "
    "'Aloha <first name>.' framing the causal chain (recent layer on top, deeper roots "
    "beneath, supported in order so the chain unwinds). title: a short warm plain-English "
    "name for that layer. meaning: 1-2 sentences on what that layer addresses.")


def draft_prose(layers, name, tags=None):
    """Draft greeting + per-layer warm title/meaning in Glen's voice via the app's LLM.
    Degrades to empty (structural fields still usable) on any failure."""
    try:
        import openai
        payload = {
            "patient_name": name or "",
            "health_tags": list(tags or []),
            "layers": [{"meridian": l.get("title", ""), "remedy": l.get("remedy", ""),
                        "dosing": l.get("dosing", "")} for l in layers],
        }
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=os.environ.get("FMP_IMPORT_MODEL", "gpt-4o"),
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": _VOICE},
                      {"role": "user", "content": json.dumps(payload)}])
        data = json.loads(resp.choices[0].message.content)
        return {"greeting": data.get("greeting", ""), "layers": data.get("layers", []) or []}
    except Exception as e:
        print(f"[fmp-import] prose draft failed: {e!r}", flush=True)
        return {"greeting": "", "layers": []}


def import_content(email, name="", tags=None):
    """End-to-end: FMP pull -> layers -> AI prose -> portal content dict (or None
    when the client has no FMP causal chain). Ready to pre-fill the editor form."""
    rows = fetch_causal_chain(email)
    layers = build_layers(rows)
    if not layers:
        return None
    prose = draft_prose(layers, name, tags)
    pl = prose.get("layers") or []
    for i, layer in enumerate(layers):
        if i < len(pl):
            layer["title"] = (pl[i].get("title") or layer["title"])
            layer["meaning"] = (pl[i].get("meaning") or "")
    return {
        "greeting": prose.get("greeting", ""),
        "video": {"url": "", "label": "Watch your message from Dr. Glen"},
        "layers": layers,
        "reorder_items": [],
        "pricing_note": "",
    }
