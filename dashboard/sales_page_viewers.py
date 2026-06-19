import datetime


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _norm(email):
    return (email or "").strip().lower()


def init_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS sales_page_viewers ("
        "product_slug TEXT, email TEXT, name TEXT, first_seen_at TEXT, "
        "emailed_at TEXT DEFAULT '', PRIMARY KEY(product_slug, email))")
    cx.commit()


def record_viewer(cx, slug, email, name=""):
    init_table(cx)
    e = _norm(email)
    if not e:
        return
    cx.execute(
        "INSERT OR IGNORE INTO sales_page_viewers (product_slug, email, name, first_seen_at, emailed_at) "
        "VALUES (?,?,?,?,'')", (slug, e, name or "", _now()))
    cx.commit()


def viewers_to_email(cx, slug):
    init_table(cx)
    cur = cx.cursor()
    cur.row_factory = __import__("sqlite3").Row
    rows = cur.execute(
        "SELECT email, name FROM sales_page_viewers WHERE product_slug=? AND COALESCE(emailed_at,'')='' "
        "ORDER BY first_seen_at", (slug,)).fetchall()
    return [{"email": r["email"], "name": r["name"] or ""} for r in rows]


def mark_emailed(cx, slug, emails):
    init_table(cx)
    now = _now()
    for e in emails:
        cx.execute("UPDATE sales_page_viewers SET emailed_at=? WHERE product_slug=? AND email=?",
                   (now, slug, _norm(e)))
    cx.commit()


def notify_on_approve(cx, slug, product_name, base_url, *, send, strip=lambda s: s):
    """Email each un-emailed viewer of this slug once as Dr. Glen; mark them; return the count sent."""
    viewers = viewers_to_email(cx, slug)
    subject = f"Your {product_name} page is ready, reviewed by Dr. Glen"
    link = f"{base_url}/begin/product/{slug}"
    emailed = []
    for v in viewers:
        name = (v.get("name") or "").strip()
        greeting = f"Aloha {name}," if name else "Aloha,"
        body = strip(
            f"{greeting}\n\nThe {product_name} page you looked at has now been personally reviewed "
            f"by Dr. Glen and is ready:\n\n{link}\n\nIn wellness,\nDr. Glen & Rae")
        try:
            send(v["email"], subject, body, from_name="Dr. Glen Swartwout")
            emailed.append(v["email"])
        except Exception as e:  # noqa: BLE001 - one bad send must not stop the rest
            print(f"[sales-viewers] send failed for {v['email']}: {e}", flush=True)
    if emailed:
        mark_emailed(cx, slug, emailed)
    return len(emailed)
