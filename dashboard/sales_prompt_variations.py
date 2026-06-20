import datetime

def _now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

_SEED = {
    "botanical": [
        ("flat-lay", "Warm overhead flat-lay of an abundance of fresh whole herbs, green leaves, "
                     "roots, and colorful botanical ingredients arranged on a rustic wooden kitchen "
                     "counter, soft natural daylight."),
        ("kitchen-woman", "An attractive mature woman in a sunlit farmhouse kitchen gently preparing "
                          "fresh herbs at a wooden counter, a lush green herb garden visible through the "
                          "window behind her, golden-hour light."),
        ("still-life", "A close intimate still-life of fresh botanical ingredients — sprigs, flowers, and "
                       "sliced roots — on a weathered cutting board, shallow depth of field, soft morning light."),
        ("market-basket", "An abundant woven market basket overflowing with fresh colorful botanicals and "
                          "leafy greens on a garden table outdoors, dappled natural sunlight, lush plants behind."),
    ],
    "mechanism": [
        ("shielded-cell", "A single glowing living human cell surrounded by a radiant spherical protective "
                          "energy field, luminous particles flowing inward, deep teal studio background, "
                          "clean conceptual render."),
        ("dramatic-shield", "A luminous human cell with a shimmering protective shield, dramatic volumetric "
                            "light on a dark background, iridescent particles drifting toward it."),
        ("cross-section", "A cross-section of a vibrant human cell with a glowing luminous membrane and "
                          "energized interior, symmetrical centered composition, iridescent blue-violet palette."),
        ("repelling-field", "A radiant cellular energy field repelling dark chaotic stressor particles away "
                            "from a healthy glowing cell, warm amber glow on a black background, shallow depth of field."),
    ],
}

def init_table(cx):
    cx.execute("CREATE TABLE IF NOT EXISTS sales_prompt_variations ("
               "id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, label TEXT, "
               "prompt_template TEXT, state TEXT DEFAULT 'active', "
               "created_at TEXT DEFAULT '', retired_at TEXT DEFAULT '')")
    cx.commit()

def seed(cx):
    init_table(cx)
    n = cx.execute("SELECT COUNT(*) FROM sales_prompt_variations").fetchone()[0]
    if n:
        return
    now = _now()
    for kind, items in _SEED.items():
        for label, template in items:
            cx.execute("INSERT INTO sales_prompt_variations (kind, label, prompt_template, state, created_at) "
                       "VALUES (?,?,?, 'active', ?)", (kind, label, template, now))
    cx.commit()

def active_variations(cx, kind):
    seed(cx)
    rows = cx.execute("SELECT id, kind, label, prompt_template FROM sales_prompt_variations "
                      "WHERE kind=? AND state='active' ORDER BY id", (kind,)).fetchall()
    return [{"id": r[0], "kind": r[1], "label": r[2], "prompt_template": r[3]} for r in rows]
