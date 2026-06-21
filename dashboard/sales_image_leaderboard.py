def wilson_lower(pos, n, z=1.96):
    if n <= 0:
        return 0.0
    phat = pos / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * ((phat * (1 - phat) + z * z / (4 * n)) / n) ** 0.5
    return (centre - margin) / denom

def _labels(cx, table):
    try:
        return {r[0]: r[1] for r in cx.execute(f"SELECT id, label FROM {table}").fetchall()}
    except Exception:
        return {}

def _agg(cx, tag_col, label_map, min_volume):
    from dashboard import sales_image_exposures as _ex
    exp = _ex.per_product_counts(cx)
    votes = {tag: n for tag, n in cx.execute(
        f"SELECT {tag_col}, COUNT(*) FROM sales_page_votes "
        f"WHERE {tag_col} IS NOT NULL GROUP BY {tag_col}").fetchall()}
    prods = {}
    for slug, tag in cx.execute(
        f"SELECT DISTINCT product_slug, {tag_col} FROM sales_page_images "
        f"WHERE {tag_col} IS NOT NULL AND state='ready'").fetchall():
        prods.setdefault(tag, set()).add(slug)
    rows = []
    for tag in (set(votes) | set(prods)):
        impr = sum(exp.get(p, 0) for p in prods.get(tag, ()))
        v = votes.get(tag, 0)
        rows.append({"key": tag, "label": label_map.get(tag, str(tag)),
                     "votes": v, "impressions": impr,
                     "rate": (v / impr) if impr else 0.0,
                     "wilson": wilson_lower(v, impr),
                     "low_volume": impr < min_volume})
    rows.sort(key=lambda r: r["wilson"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows

def leaderboard(cx, min_volume=30):
    # tag_col values are fixed literals (not user input) -> safe to interpolate
    var_labels = _labels(cx, "sales_prompt_variations")
    model_labels = _labels(cx, "sales_image_models")
    return {"variations": _agg(cx, "prompt_variant_id", var_labels, min_volume),
            "models": _agg(cx, "model_id", model_labels, min_volume)}

def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

def render_html(data):
    def _rows(items):
        out = []
        for r in items:
            badge = ' <span style="color:#b00">low volume</span>' if r["low_volume"] else ""
            out.append(f"<tr><td>{r['rank']}</td><td>{_esc(r['label'])}</td>"
                       f"<td>{r['rate']*100:.1f}%</td><td>{r['votes']}</td>"
                       f"<td>{r['impressions']}</td><td>{badge}</td></tr>")
        return "".join(out)
    head = ("<tr><th>Rank</th><th>Label</th><th>Pick-rate</th>"
            "<th>Votes</th><th>Exposures</th><th></th></tr>")
    return ("<!doctype html><meta charset=utf-8><title>Image Leaderboard</title>"
            "<style>body{font-family:system-ui;margin:2rem}"
            "table{border-collapse:collapse;margin-bottom:2rem}"
            "td,th{border:1px solid #ccc;padding:4px 10px;text-align:left}</style>"
            "<h1>Image Split-Test Leaderboard</h1>"
            f"<h2>Prompt Variations</h2><table>{head}{_rows(data['variations'])}</table>"
            f"<h2>Models</h2><table>{head}{_rows(data['models'])}</table>")
