"""Fullscript dispensary channel data (pure sqlite; caller passes cx).
Sibling of dashboard/prl_supplement.py. Owns schema + queries only.

Fullscript is a SEPARATELY LISTED channel, like E4L and PRL. It deliberately
does NOT write to recommendation_events: that table's product_key is a
storefront slug, and both the portal recommendations block and the console 360
hub resolve keys against the storefront catalog. Fullscript products have no
storefront slug, so they would render broken there. Clicks live in
fullscript_clicks instead.
"""
import json


def init_tables(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS fullscript_products (
        name TEXT PRIMARY KEY, brand TEXT, external_id TEXT, product_slug TEXT,
        url TEXT, focus_tags TEXT, product_type TEXT, best_ff TEXT, relation TEXT,
        ff_alts TEXT, source TEXT, active INTEGER DEFAULT 1)""")
    cx.execute("""CREATE TABLE IF NOT EXISTS fullscript_focus_area_products (
        focus_area_id INTEGER, focus_area_name TEXT, fs_product_name TEXT, rank INTEGER)""")
    cx.execute("""CREATE TABLE IF NOT EXISTS fullscript_focus_area_items (
        focus_area_id INTEGER, item_code TEXT)""")
    cx.execute("""CREATE TABLE IF NOT EXISTS fullscript_condition_products (
        condition_key TEXT, fs_product_name TEXT, rank INTEGER)""")
    cx.execute("""CREATE TABLE IF NOT EXISTS fullscript_client_pins (
        email TEXT, fs_product_name TEXT, note TEXT, pinned_by TEXT, pinned_at TEXT)""")
    cx.execute("""CREATE TABLE IF NOT EXISTS fullscript_review_links (
        review_id INTEGER, fs_product_name TEXT, rank INTEGER, created_at TEXT)""")
    cx.execute("""CREATE TABLE IF NOT EXISTS fullscript_clicks (
        email TEXT, fs_product_name TEXT, origin TEXT, clicked_at TEXT)""")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_fsfai_code "
               "ON fullscript_focus_area_products(focus_area_id, rank)")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_fsfa_item "
               "ON fullscript_focus_area_items(item_code)")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_fspins_email "
               "ON fullscript_client_pins(email)")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_fsprod_slug "
               "ON fullscript_products(product_slug)")
    cx.commit()


def sync_from_seed(cx, seed):
    """Idempotent full replace of the three reference tables. Pins, review links
    and clicks are client data and are never touched."""
    cx.execute("DELETE FROM fullscript_products")
    cx.execute("DELETE FROM fullscript_focus_area_products")
    cx.execute("DELETE FROM fullscript_focus_area_items")
    cx.execute("DELETE FROM fullscript_condition_products")
    from dashboard import dbwrite
    for p in seed.get("products", []):
        dbwrite.insert_or_replace(
            cx, "fullscript_products",
            ("name", "brand", "external_id", "product_slug", "url", "focus_tags",
             "product_type", "best_ff", "relation", "ff_alts", "source", "active"),
            (p["name"], p.get("brand"), p.get("external_id"), p.get("product_slug"),
             p.get("url"), json.dumps(p.get("focus_tags") or []), p.get("product_type"),
             p.get("best_ff"), p.get("relation"), json.dumps(p.get("ff_alts") or []),
             p.get("source") or "seed", 1 if p.get("active", 1) else 0),
            conflict_cols=("name",))
    for fp in seed.get("focus_area_products", []):
        cx.execute("""INSERT INTO fullscript_focus_area_products
            (focus_area_id, focus_area_name, fs_product_name, rank) VALUES (?,?,?,?)""",
            (fp["focus_area_id"], fp.get("focus_area_name"), fp["fs_product_name"],
             fp.get("rank", 0)))
    for fi in seed.get("focus_area_items", []):
        cx.execute("INSERT INTO fullscript_focus_area_items "
                   "(focus_area_id, item_code) VALUES (?,?)",
                   (fi["focus_area_id"], fi["item_code"]))
    cx.commit()
    return {"products": len(seed.get("products", [])),
            "focus_area_products": len(seed.get("focus_area_products", [])),
            "focus_area_items": len(seed.get("focus_area_items", []))}
