"""Causal Chain Report from the FMP snapshot (fmp_snap_* tables).

Layer-ordered: the layer number lives on the linked active-stress factor
(`client_causal_chain.id_fk_active_stress` -> `client_active_main_stress.layer`);
the chain row's own `layer` column exports blank (it was a FileMaker calc). Remedies
are joined from `client_remedy` (`id_fk_causal_chain`), not the chain row. Reuses
`build_schedule` for the times-of-day protocol. Read-only over the snapshot.
"""
import sqlite3

from dashboard.biofield_schedule import build_schedule


def _i(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def _has(cx, table):
    return cx.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                      (table,)).fetchone() is not None


def list_tests(cx, q=""):
    """Console list: one row per biofield test with client + remedy count."""
    cx.row_factory = sqlite3.Row
    if not _has(cx, "fmp_snap_client_biofield_test"):
        return []  # snapshot not loaded yet -> no FileMaker tests
    rows = [dict(r) for r in cx.execute("""
      SELECT b.id_pk AS test_id, b.date_test AS date,
             TRIM(COALESCE(cl.name_first,'')||' '||COALESCE(cl.name_last,'')) AS name,
             cl.email AS email,
             (SELECT COUNT(*) FROM fmp_snap_client_causal_chain cc
                JOIN fmp_snap_client_remedy r ON r.id_fk_causal_chain=cc.id_pk
                WHERE cc.id_fk_test=b.id_pk AND TRIM(COALESCE(r.remedy,''))<>'') AS layer_count
      FROM fmp_snap_client_biofield_test b
      LEFT JOIN fmp_snap_clients cl ON cl.id_pk=b.id_fk_client
    """).fetchall()]
    if q:
        ql = q.strip().lower()
        rows = [r for r in rows
                if ql in (r["name"] or "").lower() or ql in (r["email"] or "").lower()]
    rows.sort(key=lambda r: (r["layer_count"] or 0), reverse=True)
    return rows


def causal_chain_report(cx, test_id):
    """Full report for one test: client header, layer-ordered remedies, schedule."""
    cx.row_factory = sqlite3.Row
    if not _has(cx, "fmp_snap_client_biofield_test"):
        return {"test_id": str(test_id), "client": {"name": "", "email": ""},
                "date": "", "layers": [], "schedule": build_schedule([])}
    head = cx.execute("""
      SELECT b.date_test AS date,
             TRIM(COALESCE(cl.name_first,'')||' '||COALESCE(cl.name_last,'')) AS name,
             cl.email AS email
      FROM fmp_snap_client_biofield_test b
      LEFT JOIN fmp_snap_clients cl ON cl.id_pk=b.id_fk_client
      WHERE b.id_pk=?""", (test_id,)).fetchone()
    rows = cx.execute("""
      SELECT ams.layer AS layer, cc.head_chain AS head, cc.most_affected AS most_affected,
             r.remedy AS remedy, r.dosage AS dosage, r.frequency AS frequency,
             r.timing AS timing, cc.id_pk AS id_pk
      FROM fmp_snap_client_causal_chain cc
      LEFT JOIN fmp_snap_client_active_main_stress ams ON ams.id_pk=cc.id_fk_active_stress
      LEFT JOIN fmp_snap_client_remedy r ON r.id_fk_causal_chain=cc.id_pk
      WHERE cc.id_fk_test=? AND TRIM(COALESCE(r.remedy,''))<>''
    """, (test_id,)).fetchall()

    layers = []
    for r in rows:
        layers.append({
            "layer": _i(r["layer"]),
            "head": (r["head"] or "").strip(),
            "most_affected": (r["most_affected"] or "").strip(),
            "remedy": (r["remedy"] or "").strip(),
            "dosage": (r["dosage"] or "").strip(),
            "frequency": (r["frequency"] or "").strip(),
            "timing": (r["timing"] or "").strip(),
            "_id": _i(r["id_pk"]) or 0,
        })
    # layer ASC (most recent/surface -> deepest root); unlayered rows last, then by id
    layers.sort(key=lambda l: (l["layer"] is None, l["layer"] or 0, l["_id"]))
    for l in layers:
        l.pop("_id", None)

    schedule = build_schedule([
        {"name": l["remedy"], "dosage": l["dosage"],
         "frequency": l["frequency"], "timing": l["timing"]} for l in layers])

    return {
        "test_id": str(test_id),
        "client": {"name": (head["name"].strip() if head and head["name"] else ""),
                   "email": (head["email"] if head and head["email"] else "")},
        "date": (head["date"] if head and head["date"] else ""),
        "layers": layers,
        "schedule": schedule,
    }
