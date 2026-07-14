"""PRL Supplement portal card data (pure sqlite; caller passes cx).
Sibling of dashboard/scan_recommendations.py. Owns schema + queries only.
"""
import json


def init_tables(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS prl_products (
        name TEXT PRIMARY KEY, external_id TEXT, url TEXT, focus_tags TEXT,
        product_type TEXT, best_ff TEXT, relation TEXT, ff_alts TEXT)""")
    cx.execute("""CREATE TABLE IF NOT EXISTS prl_focus_area_products (
        focus_area_id INTEGER, focus_area_name TEXT, prl_product_name TEXT, rank INTEGER)""")
    cx.execute("""CREATE TABLE IF NOT EXISTS prl_focus_area_items (
        focus_area_id INTEGER, item_code TEXT)""")
    cx.execute("""CREATE TABLE IF NOT EXISTS prl_scan_mirror (
        scan_id TEXT PRIMARY KEY, payload TEXT, captured_at TEXT)""")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_pfai_code ON prl_focus_area_items(item_code)")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_pfap_fa ON prl_focus_area_products(focus_area_id, rank)")
    cx.commit()


def sync_from_seed(cx, seed):
    """Idempotent full replace of the three reference tables (mirror untouched)."""
    cx.execute("DELETE FROM prl_products")
    cx.execute("DELETE FROM prl_focus_area_products")
    cx.execute("DELETE FROM prl_focus_area_items")
    for p in seed.get("products", []):
        cx.execute("""INSERT OR REPLACE INTO prl_products
            (name, external_id, url, focus_tags, product_type, best_ff, relation, ff_alts)
            VALUES (?,?,?,?,?,?,?,?)""",
            (p["name"], p.get("external_id"), p.get("url"),
             json.dumps(p.get("focus_tags") or []), p.get("product_type"),
             p.get("best_ff"), p.get("relation"), json.dumps(p.get("ff_alts") or [])))
    for fp in seed.get("focus_area_products", []):
        cx.execute("""INSERT INTO prl_focus_area_products
            (focus_area_id, focus_area_name, prl_product_name, rank) VALUES (?,?,?,?)""",
            (fp["focus_area_id"], fp.get("focus_area_name"), fp["prl_product_name"], fp.get("rank", 0)))
    for fi in seed.get("focus_area_items", []):
        cx.execute("INSERT INTO prl_focus_area_items (focus_area_id, item_code) VALUES (?,?)",
                   (fi["focus_area_id"], fi["item_code"]))
    cx.commit()
    return {"products": len(seed.get("products", [])),
            "focus_area_products": len(seed.get("focus_area_products", [])),
            "focus_area_items": len(seed.get("focus_area_items", []))}


def focus_areas_for_items(cx, item_codes):
    """Focus areas whose infoceuticals include any of item_codes, ranked by hit count."""
    codes = [c for c in (item_codes or []) if c]
    if not codes:
        return []
    q = ("SELECT i.focus_area_id, COALESCE(n.focus_area_name, '') AS focus_area_name, COUNT(*) AS hits "
         "FROM prl_focus_area_items i "
         "LEFT JOIN (SELECT DISTINCT focus_area_id, focus_area_name FROM prl_focus_area_products) n "
         "  ON n.focus_area_id = i.focus_area_id "
         f"WHERE i.item_code IN ({','.join('?' * len(codes))}) "
         "GROUP BY i.focus_area_id ORDER BY hits DESC, i.focus_area_id")
    return [dict(r) for r in cx.execute(q, codes).fetchall()]


def products_for_focus_area(cx, focus_area_id):
    rows = cx.execute("""
        SELECT fap.prl_product_name AS name, p.url AS url, p.best_ff AS best_ff,
               p.relation AS relation
        FROM prl_focus_area_products fap
        LEFT JOIN prl_products p ON p.name = fap.prl_product_name
        WHERE fap.focus_area_id = ? ORDER BY fap.rank""", (focus_area_id,)).fetchall()
    return [dict(r) for r in rows]


def mirror_for_scan(cx, scan_id):
    row = cx.execute("SELECT payload FROM prl_scan_mirror WHERE scan_id=?", (scan_id,)).fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None
