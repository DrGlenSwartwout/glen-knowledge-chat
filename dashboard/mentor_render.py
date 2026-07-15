"""Pure server-side renderer for public mentor pages (SEO-first).

No Flask import: takes a page dict, returns an HTML string. Keeps rendering
unit-testable without booting the app. Styled to match dashboard/topic_render.py
(dark-green/gold palette, Raleway/Open Sans, brandbar + disclaimer footer) so
mentor pages sit visually inside the same site.
"""
import html
import json

_SECTION_TITLES = {
    "life_and_work": "Life & Work",
    "key_contribution": "Key Contribution",
    "lineage": "The Lineage",
    "why_it_matters": "Why It Matters",
}
_SECTION_ORDER = ("life_and_work", "key_contribution", "lineage", "why_it_matters")

_FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600'
    '&family=Raleway:wght@600;700&display=swap" rel="stylesheet">'
)

_STYLE = (
    "<style>"
    ":root{--bg:#0a150d;--surface:#111f16;--border:#21472d;--cream:#fdf4d8;"
    "--muted:#a89870;--gold:#d4a843;--green:#3d8a52;--maxw:760px;"
    "--ease:cubic-bezier(0.22,1,0.36,1);"
    '--font:"Open Sans",system-ui,sans-serif;--heading:"Raleway",system-ui,sans-serif;}'
    "*{box-sizing:border-box;margin:0;padding:0;}"
    "body{background:var(--bg);color:var(--cream);font-family:var(--font);"
    "font-size:17px;line-height:1.7;-webkit-font-smoothing:antialiased;"
    "background-image:radial-gradient(120% 80% at 50% -10%,rgba(212,168,67,0.07),transparent 60%);"
    "background-attachment:fixed;}"
    "h1,h2,h3{font-family:var(--heading);font-weight:600;line-height:1.25;letter-spacing:-0.01em;}"
    "h1{font-size:2rem;color:var(--gold);margin:8px 0 2px;}"
    "h2{font-size:1.25rem;color:var(--cream);margin:30px 0 8px;}"
    "a{color:var(--gold);text-decoration:none;}"
    "a:hover{text-decoration:underline;}"
    "p{margin:0 0 14px;}"
    ".shell{max-width:var(--maxw);margin:0 auto;padding:8px 24px 0;}"
    ".brandbar{max-width:var(--maxw);margin:0 auto;padding:30px 24px 8px;"
    "display:flex;align-items:center;justify-content:center;gap:14px;}"
    ".brand-mark{width:7px;height:7px;border-radius:50%;background:var(--gold);"
    "box-shadow:0 0 0 4px rgba(212,168,67,0.12);}"
    ".brand-name{font-family:var(--heading);font-weight:600;font-size:13px;"
    "letter-spacing:0.32em;color:var(--muted);}"
    ".sub{color:var(--muted);font-size:14px;letter-spacing:0.04em;margin:0 0 6px;"
    "text-transform:uppercase;font-family:var(--heading);}"
    ".chain{margin:10px 0 4px;background:var(--surface);border:1px solid var(--border);"
    "border-radius:12px;padding:14px 18px;font-size:15px;line-height:1.9;}"
    ".chain .lbl{display:block;color:var(--muted);font-size:12px;letter-spacing:0.14em;"
    "text-transform:uppercase;margin-bottom:4px;}"
    ".chain b{color:var(--gold);font-weight:700;}"
    ".sources{margin-top:30px;border-top:1px solid var(--border);padding-top:16px;}"
    ".sources ul{list-style:none;margin-top:8px;}"
    ".sources li{font-size:14px;color:var(--muted);margin-bottom:6px;}"
    ".cta{margin-top:36px;text-align:center;}"
    ".cta-btn{display:inline-block;background:var(--gold);color:#0a150d;"
    "font-family:var(--heading);font-weight:700;font-size:15px;letter-spacing:0.03em;"
    "padding:14px 32px;border-radius:999px;border:none;cursor:pointer;"
    "transition:opacity .25s var(--ease),transform .25s var(--ease);}"
    ".cta-btn:hover{opacity:0.9;transform:translateY(-1px);text-decoration:none;}"
    "input[type=email]{font-family:var(--font);font-size:15px;padding:12px 16px;"
    "border-radius:10px;border:1px solid var(--border);background:var(--surface);"
    "color:var(--cream);width:100%;max-width:320px;margin-bottom:12px;}"
    ".index-list{list-style:none;}"
    ".index-list li{border-top:1px solid var(--border);padding:14px 0;}"
    ".index-list a{font-family:var(--heading);font-weight:600;font-size:1.05rem;}"
    ".index-list .meta{display:block;color:var(--muted);font-size:13px;margin-top:2px;}"
    "footer{margin-top:48px;padding:28px 24px 44px;text-align:center;"
    "border-top:1px solid rgba(33,71,45,0.5);}"
    ".foot-note{font-size:12.5px;color:var(--muted);opacity:0.7;max-width:540px;"
    "margin:0 auto;line-height:1.7;}"
    "@media(max-width:560px){body{font-size:16px;}.shell{padding:8px 18px 0;}h1{font-size:1.6rem;}}"
    "</style>"
)

_BRANDBAR = (
    '<div class="brandbar"><span class="brand-mark" aria-hidden="true"></span>'
    '<span class="brand-name">Remedy&nbsp;Match &nbsp;&middot;&nbsp; Healing&nbsp;Oasis</span></div>'
)

_FOOTER = (
    "<footer><p class=\"foot-note\">Shared in the spirit of education and to honor the "
    "lineage this work stands on. This space offers educational and historical context. "
    "It is not a substitute for diagnosis or medical care from your own provider.</p></footer>"
)


def is_public(page):
    return bool(page) and page.get("state") == "approved"


def _esc(s):
    return html.escape(str(s or ""), quote=True)


def _document(title, meta_desc, head_extra, body_inner, *, noindex=False):
    robots = '<meta name="robots" content="noindex">' if noindex else ""
    desc = f'<meta name="description" content="{_esc(meta_desc)}">' if meta_desc is not None else ""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{_esc(title)}</title>"
        f"{desc}{robots}"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<link rel="icon" type="image/png" href="/static/favicon.png">'
        f"{_FONTS}{_STYLE}{head_extra}</head><body>"
        f"{_BRANDBAR}"
        f'<section class="shell">{body_inner}</section>'
        f"{_FOOTER}</body></html>"
    )


def _subtitle(page):
    bits = [b for b in (page.get("field"), page.get("lifespan")) if b]
    if not bits:
        return ""
    return f'<p class="sub">{_esc(" · ".join(bits))}</p>'


def _lineage_block(lineage):
    lineage = [str(x) for x in (lineage or []) if str(x).strip()]
    if len(lineage) < 2:
        return ""
    chain = " &rarr; ".join(f"<b>{_esc(n)}</b>" for n in lineage)
    return ('<section class="chain"><span class="lbl">Intellectual lineage</span>'
            + chain + "</section>")


def _sources_block(sources):
    sources = [str(s) for s in (sources or []) if str(s).strip()]
    if not sources:
        return ""
    items = "".join(f"<li>{_esc(s)}</li>" for s in sources)
    return ('<section class="sources"><h2>Sources &amp; further reading</h2><ul>'
            + items + "</ul></section>")


def render_page_html(page, *, base_url=""):
    name = page.get("name") or page.get("slug")
    seo = page.get("seo") or {}
    title = seo.get("title") or f"{name}, mentor and lineage"
    meta = seo.get("meta_description") or ""
    content = page.get("content") or {}

    body_sections = []
    article_text_raw = []
    for sec in _SECTION_ORDER:
        text = content.get(sec)
        if not text:
            continue
        article_text_raw.append(str(text))
        # Render blank-line-separated paragraphs within a section.
        paras = "".join(f"<p>{_esc(p.strip())}</p>"
                        for p in str(text).split("\n\n") if p.strip())
        body_sections.append(
            f"<section><h2>{_esc(_SECTION_TITLES.get(sec, sec))}</h2>{paras}</section>"
        )

    jsonld = {
        "@context": "https://schema.org",
        "@type": "Person",
        "name": name,
        "description": meta or (article_text_raw[0][:200] if article_text_raw else ""),
    }
    if page.get("field"):
        jsonld["jobTitle"] = page["field"]
    jsonld_str = json.dumps(jsonld, ensure_ascii=False).replace("</", "<\\/")
    jsonld_tag = '<script type="application/ld+json">' + jsonld_str + "</script>"

    cta = ('<section class="cta"><a class="cta-btn" href="/begin">'
           "Explore your own healing path</a></section>")

    body_inner = (
        f"<main><h1>{_esc(name)}</h1>"
        + _subtitle(page)
        + _lineage_block(page.get("lineage"))
        + "".join(body_sections)
        + _sources_block(page.get("sources"))
        + cta
        + "</main>"
    )
    return _document(title, meta, jsonld_tag, body_inner)


def render_index_html(mentors):
    """Styled index of approved mentors. mentors: list of {slug, name, field, lifespan}."""
    items = []
    for m in (mentors or []):
        meta_bits = " · ".join(b for b in (m.get("field"), m.get("lifespan")) if b)
        meta = f'<span class="meta">{_esc(meta_bits)}</span>' if meta_bits else ""
        items.append(
            f'<li><a href="/mentors/{_esc(m["slug"])}">{_esc(m["name"])}</a>{meta}</li>'
        )
    inner = "".join(items) or "<li>These pages are on the way.</li>"
    body_inner = (
        "<main><h1>Mentors &amp; Lineage</h1>"
        "<p>The teachers, scientists, and pioneers whose work Dr. Glen Swartwout's "
        "clinical approach stands on.</p>"
        f'<ul class="index-list">{inner}</ul></main>'
    )
    return _document(
        "Mentors & Lineage",
        "The teachers and pioneers behind Dr. Glen Swartwout's clinical lineage.",
        "", body_inner)


def render_pending_html(slug, name):
    body_inner = (
        f"<main><h1>{_esc(name)}</h1>"
        "<p>This page is being prepared. Leave your email and we will send it when it is ready.</p>"
        f'<form method="post" action="/mentors/{_esc(slug)}/request">'
        '<input type="email" name="email" required placeholder="you@example.com">'
        '<button class="cta-btn" type="submit">Notify me</button></form>'
        "</main>"
    )
    return _document(name, None, "", body_inner, noindex=True)


def render_sitemap_xml(rows, base_url):
    """Build the /mentors sitemap from approved-page rows."""
    base = (base_url or "").rstrip("/")
    parts = []
    for r in rows:
        loc = html.escape(base + "/mentors/" + r["slug"], quote=True)
        stamp = (r.get("updated_at") or r.get("approved_at") or "")[:10]
        lastmod = f"<lastmod>{html.escape(stamp)}</lastmod>" if stamp else ""
        parts.append(f"<url><loc>{loc}</loc>{lastmod}</url>")
    return ('<?xml version="1.0" encoding="UTF-8"?>'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            + "".join(parts) + "</urlset>")
