import datetime

def _now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

# id, label, engine_ref (Replicate model path) — ordered intentionally; baseline first.
_SEED = [
    ("flux-1.1-pro", "Flux 1.1 Pro", "black-forest-labs/flux-1.1-pro"),
    ("imagen-4",     "Imagen 4",     "google/imagen-4"),
    ("recraft-v3",   "Recraft V3",   "recraft-ai/recraft-v3"),
]

def init_table(cx):
    cx.execute("CREATE TABLE IF NOT EXISTS sales_image_models ("
               "id TEXT PRIMARY KEY, label TEXT, engine TEXT DEFAULT 'replicate', "
               "engine_ref TEXT, state TEXT DEFAULT 'active', created_at TEXT DEFAULT '')")
    cx.commit()

def seed(cx):
    init_table(cx)
    n = cx.execute("SELECT COUNT(*) FROM sales_image_models").fetchone()[0]
    if n:
        return
    now = _now()
    for mid, label, ref in _SEED:
        cx.execute("INSERT INTO sales_image_models (id, label, engine, engine_ref, state, created_at) "
                   "VALUES (?,?, 'replicate', ?, 'active', ?)", (mid, label, ref, now))
    cx.commit()

def active_models(cx):
    seed(cx)
    rows = cx.execute("SELECT id, label, engine, engine_ref FROM sales_image_models "
                      "WHERE state='active' ORDER BY rowid").fetchall()
    return [{"id": r[0], "label": r[1], "engine": r[2], "engine_ref": r[3]} for r in rows]
