import datetime

def _now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def init_tables(cx):
    cx.execute("CREATE TABLE IF NOT EXISTS sales_image_queue ("
               "product_slug TEXT PRIMARY KEY, state TEXT DEFAULT 'pending', "
               "requested_at TEXT DEFAULT '', updated_at TEXT DEFAULT '')")
    cx.execute("CREATE TABLE IF NOT EXISTS sales_page_images ("
               "id INTEGER PRIMARY KEY AUTOINCREMENT, product_slug TEXT, kind TEXT, "
               "variant INTEGER, filename TEXT, state TEXT DEFAULT 'ready', created_at TEXT DEFAULT '')")
    for _col, _decl in (("prompt_variant_id", "INTEGER"), ("model_id", "TEXT")):
        try:
            cx.execute(f"ALTER TABLE sales_page_images ADD COLUMN {_col} {_decl}")
        except Exception:
            pass
    cx.commit()

def enqueue(cx, slug):
    init_tables(cx); now = _now()
    cx.execute("INSERT INTO sales_image_queue (product_slug, state, requested_at, updated_at) "
               "VALUES (?, 'pending', ?, ?) ON CONFLICT(product_slug) DO UPDATE SET "
               "state='pending', requested_at=?, updated_at=?", (slug, now, now, now, now))
    cx.commit()

def list_pending(cx):
    init_tables(cx)
    return [r[0] for r in cx.execute(
        "SELECT product_slug FROM sales_image_queue WHERE state='pending' ORDER BY requested_at").fetchall()]

def _set_state(cx, slug, state):
    init_tables(cx)
    cx.execute("UPDATE sales_image_queue SET state=?, updated_at=? WHERE product_slug=?", (state, _now(), slug))
    cx.commit()

def mark_done(cx, slug):   _set_state(cx, slug, "done")
def mark_failed(cx, slug): _set_state(cx, slug, "failed")

def queue_state(cx, slug):
    init_tables(cx)
    row = cx.execute("SELECT state FROM sales_image_queue WHERE product_slug=?", (slug,)).fetchone()
    return row[0] if row else None

def record_image(cx, slug, kind, variant, filename, prompt_variant_id=None, model_id=None):
    init_tables(cx)
    cx.execute("INSERT INTO sales_page_images "
               "(product_slug, kind, variant, filename, state, created_at, prompt_variant_id, model_id) "
               "VALUES (?,?,?,?, 'ready', ?, ?, ?)",
               (slug, kind, int(variant), filename, _now(), prompt_variant_id, model_id))
    cx.commit()

def get_images(cx, slug):
    init_tables(cx)
    rows = cx.execute("SELECT kind, variant, filename, prompt_variant_id, model_id "
                      "FROM sales_page_images WHERE product_slug=? AND state='ready' "
                      "ORDER BY kind, variant", (slug,)).fetchall()
    return [{"kind": r[0], "variant": r[1], "filename": r[2],
             "prompt_variant_id": r[3], "model_id": r[4]} for r in rows]

def display_images(cx, slug):
    out = {"botanical": None, "mechanism": None}
    for img in get_images(cx, slug):
        if img["kind"] in out and out[img["kind"]] is None:
            out[img["kind"]] = img["filename"]
    return out

def list_image_slugs(cx):
    init_tables(cx)
    return [r[0] for r in cx.execute(
        "SELECT DISTINCT product_slug FROM sales_page_images WHERE state='ready'").fetchall()]

def next_variant(cx, slug, kind):
    init_tables(cx)
    r = cx.execute("SELECT MAX(variant) FROM sales_page_images WHERE product_slug=? AND kind=?",
                   (slug, kind)).fetchone()
    return (r[0] or 0) + 1

def tagged_count(cx, slug):
    init_tables(cx)
    r = cx.execute("SELECT COUNT(*) FROM sales_page_images WHERE product_slug=? AND state='ready' "
                   "AND prompt_variant_id IS NOT NULL", (slug,)).fetchone()
    return r[0] if r else 0

def needs_topup(cx, slug, target=8):
    return tagged_count(cx, slug) < target

def build_generation_jobs(cx, slug):
    """Missing (kind, slot) generation jobs for `slug`, up to 4/kind. Each job is tagged
    with its prompt variation and an assigned model. All 4 variations are covered per kind;
    models rotate by a per-product offset for balanced marginal coverage across products."""
    import zlib
    from dashboard import sales_prompt_variations as _pv
    from dashboard import sales_image_models as _mods
    from dashboard import sales_image_prompts as _sip
    init_tables(cx)
    present = {(im["kind"], im["variant"]) for im in get_images(cx, slug)}
    models = _mods.active_models(cx)
    if not models:
        return []
    offset = zlib.crc32(slug.encode("utf-8")) % len(models)
    jobs = []
    for kind in _sip.IMAGE_KINDS:
        variations = _pv.active_variations(cx, kind)[:4]
        for i, var in enumerate(variations):
            slot = i + 1
            if (kind, slot) in present:
                continue
            model = models[(i + offset) % len(models)]
            prompt_text = f"{var['prompt_template']} {_sip.NO_TEXT}"
            jobs.append({"kind": kind, "variant": slot,
                         "prompt_variant_id": var["id"], "model_id": model["id"],
                         "prompt_text": prompt_text})
    return jobs
