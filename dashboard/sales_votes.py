import datetime

def _now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def init_table(cx):
    cx.execute("CREATE TABLE IF NOT EXISTS sales_page_votes ("
               "id INTEGER PRIMARY KEY AUTOINCREMENT, product_slug TEXT, kind TEXT, "
               "chosen_variant INTEGER, session_id TEXT, email TEXT DEFAULT '', "
               "created_at TEXT DEFAULT '', updated_at TEXT DEFAULT '', "
               "UNIQUE(session_id, product_slug, kind))")
    cx.commit()

def record_pick(cx, slug, kind, variant, session_id, email=""):
    init_table(cx); now = _now(); email = (email or "").strip().lower()
    cx.execute("INSERT INTO sales_page_votes (product_slug, kind, chosen_variant, session_id, email, created_at, updated_at) "
               "VALUES (?,?,?,?,?,?,?) ON CONFLICT(session_id, product_slug, kind) DO UPDATE SET "
               "chosen_variant=excluded.chosen_variant, "
               "email=CASE WHEN excluded.email!='' THEN excluded.email ELSE sales_page_votes.email END, "
               "updated_at=excluded.updated_at",
               (slug, kind, int(variant), session_id, email, now, now))
    if email and session_id:
        cx.execute("UPDATE sales_page_votes SET email=? WHERE session_id=? AND email=''", (email, session_id))
    cx.commit()

def _match_rows(cx, slug, session_id, email):
    init_table(cx)
    return cx.execute(
        "SELECT kind, chosen_variant FROM sales_page_votes WHERE product_slug=? AND "
        "(session_id=? OR (email!='' AND email=?))",
        (slug, session_id or "\x00", (email or "").strip().lower() or "\x00")).fetchall()

def get_picks(cx, slug, *, session_id="", email=""):
    out = {"botanical": None, "mechanism": None}
    for kind, var in _match_rows(cx, slug, session_id, email):
        if kind in out:
            out[kind] = var
    return out

def picked_both(cx, slug, *, session_id="", email=""):
    p = get_picks(cx, slug, session_id=session_id, email=email)
    return (p.get("botanical") or 0) >= 1 and (p.get("mechanism") or 0) >= 1

def tally(cx, slug):
    init_table(cx)
    rows = cx.execute("SELECT kind, chosen_variant, COUNT(*) FROM sales_page_votes "
                      "WHERE product_slug=? AND chosen_variant>=1 GROUP BY kind, chosen_variant", (slug,)).fetchall()
    out = {}
    for kind, var, n in rows:
        out.setdefault(kind, {})[var] = n
    return out

def pair_counts(cx, slug, kind, a, b, since=""):
    init_table(cx)
    if since:
        rows = cx.execute("SELECT chosen_variant, COUNT(*) FROM sales_page_votes WHERE product_slug=? "
                          "AND kind=? AND chosen_variant IN (?,?) AND updated_at>=? GROUP BY chosen_variant",
                          (slug, kind, a, b, since)).fetchall()
    else:
        rows = cx.execute("SELECT chosen_variant, COUNT(*) FROM sales_page_votes WHERE product_slug=? "
                          "AND kind=? AND chosen_variant IN (?,?) GROUP BY chosen_variant",
                          (slug, kind, a, b)).fetchall()
    d = {v: n for v, n in rows}
    return (d.get(a, 0), d.get(b, 0))
