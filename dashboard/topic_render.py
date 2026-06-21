"""Pure server-side renderer for public topic pages (SEO-first).

No Flask import: takes a page dict, returns an HTML string. Keeps rendering
unit-testable without booting the app.
"""
import html
import json

_SECTION_TITLES = {
    "overview": "Overview",
    "contributing_factors": "Commonly Associated Factors",
    "what_people_explore": "What People Often Explore",
}
_SECTION_ORDER = ("overview", "contributing_factors", "what_people_explore")


def is_public(page):
    return bool(page) and page.get("state") == "approved"


def _esc(s):
    return html.escape(str(s or ""), quote=True)


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
    return "<section class=\"related\"><h2>Related</h2><ul>" + "".join(rows) + "</ul></section>"


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
    jsonld_tag = ('<script type="application/ld+json">'
                  + jsonld_str + "</script>")

    cta = ('<section class="cta"><p>Want guidance matched to you? '
           '<a href="/begin">Start your free assessment</a>.</p></section>')

    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>{_esc(title)}</title>"
        f'<meta name="description" content="{_esc(meta)}">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"{jsonld_tag}</head><body>"
        f"<main><h1>{_esc(name)}</h1>"
        + "".join(body_sections)
        + _related_block(page.get("links"))
        + cta
        + "</main></body></html>"
    )


def render_pending_html(slug, name):
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>{_esc(name)}</title>"
        '<meta name="robots" content="noindex">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "</head><body><main>"
        f"<h1>{_esc(name)}</h1>"
        "<p>This guide is being prepared. Leave your email and we will send it when it is ready.</p>"
        f'<form method="post" action="/learn/{_esc(slug)}/request">'
        '<input type="email" name="email" required placeholder="you@example.com">'
        '<button type="submit">Notify me</button></form>'
        "</main></body></html>"
    )
