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

_BASELINE_REF = "black-forest-labs/flux-1.1-pro"

def generate(cx, model_id, prompt, *, aspect="1:1"):
    """Return (image_bytes, used_model_id). Falls back to baseline Flux on engine error."""
    from dashboard import replicate_client as _rc
    by_id = {m["id"]: m for m in active_models(cx)}
    m = by_id.get(model_id)
    ref = m["engine_ref"] if m else _BASELINE_REF
    try:
        return _rc.generate_image(prompt, aspect_ratio=aspect, model_ref=ref), model_id
    except Exception as e:
        if ref == _BASELINE_REF:
            raise
        data = _rc.generate_image(prompt, aspect_ratio=aspect, model_ref=_BASELINE_REF)
        return data, "flux-1.1-pro"

_CANDIDATES = [
    ("ideogram-v3",  "Ideogram V3",          "ideogram-ai/ideogram-v3-quality"),
    ("flux-ultra",   "Flux 1.1 Pro Ultra",   "black-forest-labs/flux-1.1-pro-ultra"),
    ("sd-3.5-large", "Stable Diffusion 3.5 L","stability-ai/stable-diffusion-3.5-large"),
]

def seed_candidates(cx):
    init_table(cx); now = _now()
    for mid, label, ref in _CANDIDATES:
        cx.execute("INSERT OR IGNORE INTO sales_image_models (id, label, engine, engine_ref, state, created_at) "
                   "VALUES (?,?, 'replicate', ?, 'candidate', ?)", (mid, label, ref, now))
    cx.commit()

def candidate_models(cx):
    init_table(cx)
    rows = cx.execute("SELECT id, label, engine, engine_ref FROM sales_image_models "
                      "WHERE state='candidate' ORDER BY rowid").fetchall()
    return [{"id": r[0], "label": r[1], "engine": r[2], "engine_ref": r[3]} for r in rows]

def set_state(cx, id, state):
    init_table(cx)
    cx.execute("UPDATE sales_image_models SET state=? WHERE id=?", (state, id))
    cx.commit()
