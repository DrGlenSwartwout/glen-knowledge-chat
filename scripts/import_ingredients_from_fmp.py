"""Import FileMaker raw-material data into chat_log.db (suppliers, ingredients,
ingredient_sources) + apply canonical clustering. Idempotent by fmp_id; preserves
console-edited curated columns. Dry-run default; --write to persist."""
from __future__ import annotations
import argparse, csv, json, os, re, sqlite3, sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

EXPORT_DIR = os.environ.get("FMP_EXPORT_DIR", "/tmp/fmp-export/newapp")
CANON_CSV = os.environ.get("FMP_CANONICAL_CSV",
    str(Path(__file__).resolve().parent.parent.parent / "AI-Training" / "02 Skills" / "fmp-loaders" / "mapping" / "canonical_clusters.csv"))

_AUDIT = {"PrimaryKey","CreationTimestamp","CreatedBy","ModificationTimestamp","ModifiedBy","id_pk"}


def _active(v):
    s = (v or "").strip().lower()
    if s in ("yes","1","true","y"): return 1
    if s in ("no","0","false","n"): return 0
    return None


def _num(v):
    if v is None: return None
    s = str(v).replace(",", "").strip()
    if not s: return None
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def _clean(v):
    if not v: return ""
    return re.sub(r"\s+", " ", str(v)).strip()


def _extras(row, mapped):
    out = {}
    for k, v in row.items():
        if k in mapped or k in _AUDIT or k.startswith("z") or not (v or "").strip():
            continue
        out[k] = v
    return json.dumps(out, ensure_ascii=False) if out else None


def _upsert(cx, table, fmp_cols, values, conflict_update_cols):
    # Two-statement upsert: INSERT OR IGNORE (partial-unique-index compatible) +
    # UPDATE only FMP cols (never touches curated cols).
    cols = ["fmp_id"] + fmp_cols
    ph = ",".join("?" for _ in cols)
    fmp_id = values[0]
    fmp_vals = values[1:]
    cx.execute(
        f"INSERT OR IGNORE INTO {table} ({','.join(cols)}) VALUES ({ph})",
        values,
    )
    setc = ", ".join(f"{c}=?" for c in fmp_cols) + ", updated_at=datetime('now')"
    cx.execute(
        f"UPDATE {table} SET {setc} WHERE fmp_id=?",
        (*fmp_vals, fmp_id),
    )


def import_suppliers(cx, rows):
    n = 0
    fmp_cols = ["company","address_street","address_city","address_province","address_postal_code",
                "email","phone_business","phone_cell","phone_fax","url","qb_id","active","extras"]
    mapped = set(fmp_cols) | {"id_pk"} | {"notes"}
    for r in rows:
        fid = (r.get("id_pk") or "").strip()
        if not fid: continue
        vals = [fid, _clean(r.get("company")) or "(unknown company)",
                r.get("address_street"), r.get("address_city"), r.get("address_province"), r.get("address_postal_code"),
                r.get("email"), r.get("phone_business"), r.get("phone_cell"), r.get("phone_fax"), r.get("url"),
                r.get("qb_id"), _active(r.get("active")), _extras(r, mapped)]
        _upsert(cx, "suppliers", fmp_cols, vals, fmp_cols)
        n += 1
    return n


_NAME_FIELDS = ["name_common","name_compound","name_scientific","name_species","name_favorite"]


def import_ingredients(cx, rows):
    n = 0
    fmp_cols = ["name","form","status","common_names","extras"]
    mapped = set(fmp_cols) | {"id_pk"} | set(_NAME_FIELDS) | {"active","form",
        "inci_name","cas_number","hygroscopic_rating","solubility","stability_notes","spec_notes","notes"}
    for r in rows:
        fid = (r.get("id_pk") or "").strip()
        if not fid: continue
        names = [_clean(r.get(f)) for f in _NAME_FIELDS if _clean(r.get(f))]
        name = names[0] if names else f"(unnamed FMP ingredient {fid})"
        commons = json.dumps([x for x in names[1:]], ensure_ascii=False) if len(names) > 1 else None
        status = "active" if _active(r.get("active")) == 1 else "inactive"
        vals = [fid, name, _clean(r.get("form")) or None, status, commons, _extras(r, mapped)]
        _upsert(cx, "ingredients", fmp_cols, vals, fmp_cols)
        n += 1
    return n


def import_sources(cx, rows):
    ing = {r[1]: r[0] for r in cx.execute("SELECT id, fmp_id FROM ingredients WHERE fmp_id IS NOT NULL")}
    sup = {r[1]: (r[0], r[2]) for r in cx.execute("SELECT id, fmp_id, company FROM suppliers WHERE fmp_id IS NOT NULL")}
    n = 0
    fmp_cols = ["ingredient_id","supplier_id","supplier_name","sku","price_per_unit","unit_size","unit_type","shipping_quote","extras"]
    mapped = set(fmp_cols) | {"id_pk","id_fk_raw","id_fk_supplier","product_id","price","purchase_size","purchase_size_unit","shipping",
        "preferred","lead_time_days","minimum_order","minimum_order_unit","notes"}
    for r in rows:
        fid = (r.get("id_pk") or "").strip()
        if not fid: continue
        iid = ing.get((r.get("id_fk_raw") or "").strip())
        s = sup.get((r.get("id_fk_supplier") or "").strip())
        sid, sname = (s[0], s[1]) if s else (None, None)
        vals = [fid, iid, sid, sname, _clean(r.get("product_id")) or None,
                _num(r.get("price")), _num(r.get("purchase_size")), _clean(r.get("purchase_size_unit")) or None,
                _num(r.get("shipping")), _extras(r, mapped)]
        _upsert(cx, "ingredient_sources", fmp_cols, vals, fmp_cols)
        n += 1
    return n


def apply_canonical(cx, cluster_rows):
    idmap = {r[1]: r[0] for r in cx.execute("SELECT id, fmp_id FROM ingredients WHERE fmp_id IS NOT NULL")}
    heads = {str(r["head_fmp_id"]).strip() for r in cluster_rows}
    cx.execute("UPDATE ingredients SET canonical_id=NULL")
    applied, skipped = 0, []
    for r in cluster_rows:
        h, m = str(r["head_fmp_id"]).strip(), str(r["member_fmp_id"]).strip()
        if h not in idmap or m not in idmap:
            skipped.append((h, m, "missing")); continue
        if m in heads:
            skipped.append((h, m, "member-is-head")); continue
        cx.execute("UPDATE ingredients SET canonical_id=? WHERE id=?", (idmap[h], idmap[m]))
        applied += 1
    return {"applied": applied, "skipped": len(skipped)}


def _read_csv(name):
    p = os.path.join(EXPORT_DIR, name)
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--db", default=None)
    args = ap.parse_args(argv)
    suppliers = _read_csv("suppliers.csv")
    ingredients = _read_csv("ingredients.csv")
    sources = _read_csv("ingredients_supplier.csv")
    with open(CANON_CSV, newline="") as f:
        clusters = list(csv.DictReader(f))
    print(f"suppliers={len(suppliers)} ingredients={len(ingredients)} sources={len(sources)} clusters={len(clusters)}")
    if not args.write:
        print("(dry run — pass --write to import)")
        return 0
    from dashboard.ingredient_catalog import _default_db_path, init_ingredients_schema
    cx = sqlite3.connect(args.db or _default_db_path())
    cx.row_factory = sqlite3.Row
    init_ingredients_schema(cx)
    ns = import_suppliers(cx, suppliers)
    ni = import_ingredients(cx, ingredients)
    nsrc = import_sources(cx, sources)
    canon = apply_canonical(cx, clusters)
    cx.commit(); cx.close()
    print(f"wrote suppliers={ns} ingredients={ni} sources={nsrc} canonical_applied={canon['applied']} skipped={canon['skipped']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
