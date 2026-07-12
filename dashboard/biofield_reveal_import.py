"""Import a client's E4L reveal (synthesized layers + remedies) into a local
Biofield Intake authoring test as needs-review causal-chain rows.

Runs the SAME synthesis pipeline as `02 Skills/e4l-reveal-push.py`, in-process on
Glen's Mac (PHI stays local). The vault pipeline is imported lazily through an
injectable `runner` so unit tests never touch the real e4l.db or the live matcher.
"""
import datetime
import os
import sqlite3

VAULT = os.path.expanduser("~/AI-Training")
SKILLS = os.path.join(VAULT, "02 Skills")
DEFAULT_E4L_DB = os.path.join(VAULT, "e4l.db")
DEFAULT_CATALOG = os.path.expanduser("~/deploy-chat/data/products.json")


def _days_ago(scan_date, today):
    try:
        s = datetime.date.fromisoformat((scan_date or "").strip())
        t = datetime.date.fromisoformat((today or "").strip())
    except ValueError:
        return None
    return max(0, (t - s).days)


def _run_synthesis(email, scan_id, e4l_db, catalog, today):
    """Real pipeline: resolve the scan, synthesize, normalize to reveal layers.
    Returns ({scan_id, scan_date} | None, raw_layers). Mirrors e4l-reveal-push.py."""
    import sys
    if SKILLS not in sys.path:
        sys.path.insert(0, SKILLS)
    import e4l_synthesis as E  # noqa: E402
    from e4l_reveal_lib import build_payload  # noqa: E402
    cx = sqlite3.connect(e4l_db)
    try:
        if scan_id:
            row = cx.execute("SELECT scan_id, scan_date FROM e4l_scans WHERE scan_id=?",
                             (scan_id,)).fetchone()
            scan = {"scan_id": row[0], "scan_date": row[1]} if row else None
        else:
            scan = E.latest_scan(cx, email)
        if not scan:
            return None, []
        patterns = E.pull_patterns(cx, scan["scan_id"], limit=12)
        label_map = {p["item_code"]: (p.get("full_name") or p.get("name") or p["item_code"])
                     for p in patterns if p.get("item_code")}
        cat = E.load_catalog(catalog)
        synth = E.synthesize(patterns, history="", rules=E.load_rules(),
                             ff_names=E.curated_ff_names(cat), layer_count=6)
        synth["layers"] = E.order_layers_by_pattern_count(synth.get("layers") or [])
        content = E.to_portal_content(
            synth, cat, formulation_map=E.load_formulation_map(cx),
            member_age=E.member_age_for_email(cx, email, today),
            age_rules=E.load_age_rules(cx))
        payload = build_payload(content, email, scan["scan_date"],
                                label_map=label_map, notify=False)
        return scan, ((payload or {}).get("layers") or [])
    finally:
        cx.close()


def synthesize_reveal_layers(email, scan_id=None, *, e4l_db=DEFAULT_E4L_DB,
                             catalog=DEFAULT_CATALOG, today, runner=None):
    runner = runner or _run_synthesis
    scan, raw = runner(email, scan_id, e4l_db, catalog, today)
    if not scan or not raw:
        return {"found": False, "scan_id": None, "scan_date": None,
                "days_ago": None, "fresh": False, "layers": []}
    days = _days_ago(scan["scan_date"], today)
    layers = []
    for L in raw:
        rem = L.get("remedy") or {}
        name = (rem.get("name") or "").strip() if isinstance(rem, dict) else ""
        layers.append({"n": L.get("n"),
                       "title": (L.get("title") or "").strip(),
                       "summary": (L.get("summary") or "").strip(),
                       "most_affected": ", ".join(L.get("pattern_labels") or []),
                       "remedy_name": name,
                       "codes": list(L.get("patterns") or [])})
    return {"found": True, "scan_id": scan["scan_id"], "scan_date": scan["scan_date"],
            "days_ago": days, "fresh": days is not None and days < 7, "layers": layers}


def build_coverage(layers):
    """Map each remedy (lowercased) to the set of scan stress codes it covers,
    derived from the synthesized layers. Empty-remedy layers are skipped."""
    cov = {}
    for L in layers or []:
        name = (L.get("remedy_name") or "").strip().lower()
        if not name:
            continue
        cov.setdefault(name, set()).update(L.get("codes") or [])
    return cov


def import_layers_to_test(cx, tid, layers):
    """Create one needs-review (confirmed=0) chain row per reveal layer. Dosing is
    auto-filled from the product catalog when the remedy name resolves. Returns the
    number of rows created."""
    from dashboard.biofield_authoring import add_chain_row, remedy_dosing
    n = 0
    for L in layers or []:
        name = (L.get("remedy_name") or "").strip()
        d = remedy_dosing(cx, name) if name else {"dosage": "", "frequency": "", "timing": ""}
        add_chain_row(cx, tid, L.get("n"), L.get("title") or "",
                      L.get("most_affected") or "", name,
                      dosage=d.get("dosage", ""), frequency=d.get("frequency", ""),
                      timing=d.get("timing", ""), confirmed=0, origin="scan",
                      codes=L.get("codes") or [])
        n += 1
    return n
