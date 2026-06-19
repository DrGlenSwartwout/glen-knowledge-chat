import datetime

def _now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def init_table(cx):
    cx.execute("CREATE TABLE IF NOT EXISTS sales_image_pairs ("
               "product_slug TEXT, kind TEXT, champion_variant INTEGER, challenger_variant INTEGER, "
               "defenses INTEGER DEFAULT 0, converged INTEGER DEFAULT 0, "
               "last_render_at TEXT DEFAULT '', updated_at TEXT DEFAULT '', "
               "PRIMARY KEY(product_slug, kind))")
    cx.commit()

def get_pair(cx, slug, kind):
    init_table(cx)
    r = cx.execute("SELECT champion_variant, challenger_variant, defenses, converged, last_render_at "
                   "FROM sales_image_pairs WHERE product_slug=? AND kind=?", (slug, kind)).fetchone()
    if not r: return None
    return {"champion_variant": r[0], "challenger_variant": r[1], "defenses": r[2],
            "converged": bool(r[3]), "last_render_at": r[4] or ""}

def set_pair(cx, slug, kind, *, champion, challenger, defenses, converged, last_render_at):
    init_table(cx)
    cx.execute("INSERT INTO sales_image_pairs (product_slug, kind, champion_variant, challenger_variant, "
               "defenses, converged, last_render_at, updated_at) VALUES (?,?,?,?,?,?,?,?) "
               "ON CONFLICT(product_slug, kind) DO UPDATE SET champion_variant=excluded.champion_variant, "
               "challenger_variant=excluded.challenger_variant, defenses=excluded.defenses, "
               "converged=excluded.converged, last_render_at=excluded.last_render_at, updated_at=excluded.updated_at",
               (slug, kind, champion, challenger, int(defenses), 1 if converged else 0,
                last_render_at or "", _now()))
    cx.commit()

def ensure_pair(cx, slug, kind, ready_variants):
    p = get_pair(cx, slug, kind)
    if p: return p
    vs = sorted(set(v for v in ready_variants if v >= 1))
    if len(vs) < 2: return None
    set_pair(cx, slug, kind, champion=vs[0], challenger=vs[1], defenses=0, converged=False, last_render_at="")
    return get_pair(cx, slug, kind)
