"""Pure server-side renderer for public topic pages (SEO-first).

No Flask import: takes a page dict, returns an HTML string. Keeps rendering
unit-testable without booting the app. Styled to match the site's begin-* pages
(dark-green/gold palette, Raleway/Open Sans, brandbar + disclaimer footer).
"""
import html
import json

_SECTION_TITLES = {
    "overview": "Overview",
    "contributing_factors": "Commonly Associated Factors",
    "what_people_explore": "What People Often Explore",
}
_SECTION_ORDER = ("overview", "contributing_factors", "what_people_explore")

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
    "h1{font-size:2rem;color:var(--gold);margin:8px 0 6px;}"
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
    ".related{margin-top:34px;border-top:1px solid var(--border);padding-top:18px;}"
    ".related ul{list-style:none;display:flex;flex-wrap:wrap;gap:10px;margin-top:10px;}"
    ".related li a{display:inline-block;background:var(--surface);border:1px solid var(--border);"
    "border-radius:999px;padding:7px 16px;font-size:14px;}"
    ".cta{margin-top:36px;text-align:center;}"
    ".cta-btn{display:inline-block;background:var(--gold);color:#0a150d;"
    "font-family:var(--heading);font-weight:700;font-size:15px;letter-spacing:0.03em;"
    "padding:14px 32px;border-radius:999px;border:none;cursor:pointer;"
    "transition:opacity .25s var(--ease),transform .25s var(--ease);}"
    ".cta-btn:hover{opacity:0.9;transform:translateY(-1px);text-decoration:none;}"
    "input[type=email]{font-family:var(--font);font-size:15px;padding:12px 16px;"
    "border-radius:10px;border:1px solid var(--border);background:var(--surface);"
    "color:var(--cream);width:100%;max-width:320px;margin-bottom:12px;}"
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
    "<footer><p class=\"foot-note\">Shared in the spirit of education and self-understanding. "
    "This space offers wellness and educational guidance. It is not a substitute for diagnosis "
    "or medical care from your own provider. Individual results will vary.</p></footer>"
)


def is_public(page):
    return bool(page) and page.get("state") == "approved"


def _esc(s):
    return html.escape(str(s or ""), quote=True)


def _document(title, meta_desc, head_extra, body_inner, *, noindex=False):
    """Assemble a full styled HTML document. body_inner is placed inside the shell."""
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


def _related_block(links):
    links = links or {}
    rows = []
    for slug_name in (links.get("ingredients") or []):
        rows.append(f'<li><a href="/begin/ingredient/{_esc(slug_name["slug"])}">{_esc(slug_name["name"])}</a></li>')
    for slug_name in (links.get("products") or []):
        rows.append(f'<li><a href="/begin/product/{_esc(slug_name["slug"])}">{_esc(slug_name["name"])}</a></li>')
    for slug_name in (links.get("topics") or []):
        rows.append(f'<li><a href="/learn/{_esc(slug_name["slug"])}">{_esc(slug_name["name"])}</a></li>')
    if not rows:
        return ""
    return '<section class="related"><h2>Related</h2><ul>' + "".join(rows) + "</ul></section>"


def render_page_html(page, *, base_url=""):
    name = page.get("name") or page.get("slug")
    seo = page.get("seo") or {}
    title = seo.get("title") or f"{name} — wellness overview"
    meta = seo.get("meta_description") or ""
    content = page.get("content") or {}

    body_sections = []
    article_text_raw = []
    for sec in _SECTION_ORDER:
        text = content.get(sec)
        if not text:
            continue
        escaped_text = _esc(text)
        article_text_raw.append(str(text))
        body_sections.append(
            f"<section><h2>{_esc(_SECTION_TITLES.get(sec, sec))}</h2>"
            f"<p>{escaped_text}</p></section>"
        )

    jsonld = {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": title,
        "description": meta,
        "articleBody": " ".join(article_text_raw),
    }
    jsonld_str = json.dumps(jsonld, ensure_ascii=False).replace("</", "<\\/")
    jsonld_tag = '<script type="application/ld+json">' + jsonld_str + "</script>"

    cta = ('<section class="cta"><a class="cta-btn" href="/begin">'
           "Get guidance matched to you</a></section>")

    body_inner = (
        f"<main><h1>{_esc(name)}</h1>"
        + "".join(body_sections)
        + _related_block(page.get("links"))
        + cta
        + "</main>"
    )
    return _document(title, meta, jsonld_tag, body_inner)


def render_index_html(topics):
    """Styled index of approved topics. topics: list of {slug, name}."""
    items = "".join(
        f'<li><a href="/learn/{_esc(t["slug"])}">{_esc(t["name"])}</a></li>'
        for t in (topics or [])
    )
    inner = items or "<li>New guides are on the way.</li>"
    body_inner = (
        "<main><h1>Wellness Topics</h1>"
        '<section class="related"><ul>' + inner + "</ul></section></main>"
    )
    return _document("Wellness Topics", "Educational wellness guides from Dr. Glen Swartwout.",
                     "", body_inner)


def render_pending_html(slug, name):
    body_inner = (
        f"<main><h1>{_esc(name)}</h1>"
        "<p>This guide is being prepared. Leave your email and we will send it when it is ready.</p>"
        f'<form method="post" action="/learn/{_esc(slug)}/request">'
        '<input type="email" name="email" required placeholder="you@example.com">'
        '<button class="cta-btn" type="submit">Notify me</button></form>'
        "</main>"
    )
    return _document(name, None, "", body_inner, noindex=True)
