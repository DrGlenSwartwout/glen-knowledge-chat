"""Task 7: Fullscript dispensary card — real headless render check.

Per the repo's render-verify convention (see feedback_portal_render_verify /
feedback_render_the_page_not_the_payload in the ops vault, and the sibling
manual harness at tests/manual/portal_pattern_harness.py): don't trust the
generated markup string, render the REAL static/client-portal.html in headless
Chromium against a synthetic payload and assert on the resulting DOM.

Pure stdlib http.server (no Flask/Doppler/Pinecone/full-app boot) serves the
real static file at /portal/<token> and stubs /api/portal/<token>(/view|
/recommendations) with a controlled payload, exactly like
tests/manual/portal_pattern_harness.py. Playwright drives a real browser
against it, matching tests/test_atlas_e2e.py's live-server + chromium idiom.

Mobile is checked via CSSOM at a viewport set on context creation (Playwright
true device-metrics emulation), not by resizing a window after load — per
feedback_portal_render_verify's "prove the media query from the CSSOM" rule.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
STATIC = STATIC_DIR / "client-portal.html"

_CTYPES = {".js": "text/javascript", ".css": "text/css", ".png": "image/png"}

XSS_NAME = "<script>alert(1)</script>"

FULLSCRIPT = {
    "dispensary_url": "https://us.fullscript.com/welcome/remedymatch/store-start",
    "groups": [
        {
            "origin": "scan",
            "heading": "Matched from your scan",
            "products": [
                {
                    "name": "Ortho Molecular B Complex",
                    "brand": "Ortho Molecular",
                    "product_slug": "ortho-b-complex",
                    "reason": "Your scan flagged B-vitamin support.",
                    "ff": {"name": "B-Complex Plus", "relation": "substitute", "slug": "b-complex-plus"},
                },
                {
                    "name": XSS_NAME,
                    "brand": "Pure Encapsulations",
                    "product_slug": "xss-product",
                    "reason": "Escaping probe.",
                    "ff": None,
                },
            ],
        },
        {
            "origin": "pinned",
            "heading": "Pinned by your practitioner",
            "products": [
                {
                    "name": "Thorne Magnesium Bisglycinate",
                    "brand": "Thorne",
                    "product_slug": "thorne-mag-bisglycinate",
                    "reason": "Standing pin.",
                    "ff": None,
                },
            ],
        },
    ],
}

BASE_D = {
    "name": "Test Client", "biofield_status": "confirmed", "blurred": False,
    "actionable": False, "scan_date": "2026-07-01", "scan_dates": ["2026-07-01"],
    "greeting": "Aloha Test,", "layers": [{"n": 1, "title": "Surface", "meaning": "x"}],
    "findings": [], "reorder_items": [], "messages": [],
    "membership_category": "none", "notify_on": True, "tos_agreed": True,
    "element_state": None, "element_backdrop_enabled": False,
}
V = {"biofield": {"visible": True, "status": "confirmed", "blurred": False,
     "scan_date": "2026-07-01", "scan_dates": ["2026-07-01"],
     "layers": [{"n": 1, "title": "Surface", "meaning": "x"}]}}

D_LIGHT = dict(BASE_D, token="lighttoken", fullscript_enabled=True, fullscript=FULLSCRIPT)
D_DARK = dict(BASE_D, token="darktoken", fullscript_enabled=False)
D_DARK_NO_KEY = dict(BASE_D, token="nokeytoken", fullscript_enabled=True)  # d.fullscript absent
# Flag OFF but the payload block IS present. Not reachable through today's
# server wiring (_fullscript_for returns None when dark, so the key is never
# set) -- which is precisely why it needs a test: without this case the
# `d.fullscript_enabled &&` half of the gate is unpinned, and deleting it
# leaves every other render test green. Verified: that mutation used to pass.
D_DARK_FLAG_OFF_WITH_PAYLOAD = dict(
    BASE_D, token="darkpayloadtoken", fullscript_enabled=False, fullscript=FULLSCRIPT)

DATA_BY_TOKEN = {
    "lighttoken": D_LIGHT,
    "darktoken": D_DARK,
    "nokeytoken": D_DARK_NO_KEY,
    "darkpayloadtoken": D_DARK_FLAG_OFF_WITH_PAYLOAD,
}


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = urlparse(self.path).path
        if p.startswith("/portal/"):
            self._send(200, STATIC.read_bytes(), "text/html; charset=utf-8")
            return
        # Serve the page's real static assets (entity-ref.js/css, favicon) so
        # unrelated wiring (e.g. wireEntityRefs()) doesn't throw and pollute the
        # pageerror signal we use to prove the Fullscript card itself is clean.
        if p.startswith("/static/"):
            f = STATIC_DIR / p[len("/static/"):]
            if f.is_file():
                ctype = _CTYPES.get(f.suffix, "application/octet-stream")
                self._send(200, f.read_bytes(), ctype)
                return
            self._send(404, b"", "text/plain")
            return
        if p.startswith("/api/portal/"):
            parts = p.split("/")
            token = parts[3] if len(parts) > 3 else ""
            suffix = parts[4] if len(parts) > 4 else ""
            if suffix == "view":
                self._send(200, json.dumps(V).encode(), "application/json")
                return
            if suffix == "recommendations":
                self._send(200, b"null", "application/json")
                return
            data = DATA_BY_TOKEN.get(token, D_DARK)
            self._send(200, json.dumps(data).encode(), "application/json")
            return
        if p.startswith("/api/community/coaches"):
            # Unrelated Task-4 feature; {} (not null) is the "no data yet" shape
            # initCoachesCard expects, so this stub doesn't trip an unrelated bug.
            self._send(200, b"{}", "application/json")
            return
        self._send(200, b"null", "application/json")

    def log_message(self, *a):
        pass


@pytest.fixture
def live_server():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    t.join()


def _open(pg, live_server, token):
    errors = []
    dialogs = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
    pg.goto(f"{live_server}/portal/{token}")
    pg.wait_for_selector("#app:not([hidden])")
    return errors, dialogs


def test_fullscript_card_renders_visible_with_groups_links_and_ff_chip(live_server):
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        errors, dialogs = _open(pg, live_server, "lighttoken")

        card = pg.wait_for_selector(".fullscript-card")
        assert card is not None
        # Visible = attached, non-zero box, not hidden by CSS -- not merely present.
        assert card.is_visible()
        box = card.bounding_box()
        assert box is not None and box["width"] > 0 and box["height"] > 0
        assert pg.eval_on_selector(".fullscript-card", "el => getComputedStyle(el).display") != "none"
        assert pg.eval_on_selector(".fullscript-card", "el => getComputedStyle(el).visibility") != "hidden"

        # Group headings render as text.
        headings = pg.eval_on_selector_all(".fullscript-card h3", "els => els.map(e => e.textContent)")
        assert "Matched from your scan" in headings
        assert "Pinned by your practitioner" in headings

        # At least one anchor routes through /fs/<token>/<slug>, never fullscript.com directly.
        fs_links = pg.eval_on_selector_all(
            '.fullscript-card a[href^="/fs/"]', "els => els.map(e => e.getAttribute('href'))"
        )
        assert len(fs_links) >= 1
        assert all(h.startswith("/fs/lighttoken/") for h in fs_links)

        # FF chip: present for the product with a non-null ff, absent for its sibling
        # and for the pinned product with ff: null.
        items = pg.eval_on_selector_all(".fullscript-card li", "els => els.map(e => e.textContent)")
        b_complex_li = next((t for t in items if "Ortho Molecular B Complex" in t), None)
        thorne_li = next((t for t in items if "Thorne Magnesium Bisglycinate" in t), None)
        assert b_complex_li is not None and "you also have" in b_complex_li and "B-Complex Plus" in b_complex_li
        assert thorne_li is not None and "you also have" not in thorne_li

        assert errors == []
        b.close()


def test_fullscript_card_absent_when_flag_off(live_server):
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        errors, dialogs = _open(pg, live_server, "darktoken")
        assert pg.query_selector(".fullscript-card") is None
        assert errors == []
        b.close()


def test_fullscript_card_absent_when_flag_on_but_no_payload(live_server):
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        errors, dialogs = _open(pg, live_server, "nokeytoken")
        assert pg.query_selector(".fullscript-card") is None
        assert errors == []
        b.close()


def test_fullscript_card_absent_when_flag_off_even_if_payload_leaks(live_server):
    """Pins the `d.fullscript_enabled &&` half of the render gate. The flag is the
    kill switch: if a caching bug or a future refactor ever ships the payload
    block while the channel is dark, the client must still render nothing.
    Without this case, deleting the flag check from the gate breaks no test."""
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        errors, dialogs = _open(pg, live_server, "darkpayloadtoken")
        assert pg.query_selector(".fullscript-card") is None, \
            "flag off must suppress the card even when the payload is present"
        assert "Matched from your scan" not in pg.content()
        assert errors == []
        b.close()


def test_fullscript_card_usable_at_narrow_viewport(live_server):
    # Viewport set at CONTEXT CREATION (Playwright's real device-metrics
    # emulation), never via a post-load window resize -- resizing after load
    # does not reliably reflow layout viewport in headless Chrome.
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport={"width": 375, "height": 812})
        pg = ctx.new_page()
        errors, dialogs = _open(pg, live_server, "lighttoken")

        pg.wait_for_selector(".fullscript-card")
        # Prove the narrow layout actually engaged, from the CSSOM -- not a
        # screenshot/breakpoint guess.
        assert pg.evaluate("window.matchMedia('(max-width:480px)').matches") is True
        assert pg.evaluate("window.innerWidth") == 375

        card = pg.query_selector(".fullscript-card")
        assert card.is_visible()
        box = card.bounding_box()
        assert box is not None and box["width"] > 0
        # Usable: the card fits the viewport width -- no horizontal overflow forcing it
        # off-screen or behind a scrollbar the member would have to fight.
        assert box["width"] <= 375
        assert pg.evaluate("document.documentElement.scrollWidth") <= 375 + 1  # +1px rounding slack

        # Buy links stay reachable (non-zero tap target) at this width.
        link_box = pg.eval_on_selector(
            '.fullscript-card a[href^="/fs/"]',
            "el => { const r = el.getBoundingClientRect(); return {w: r.width, h: r.height}; }",
        )
        assert link_box["w"] > 0 and link_box["h"] > 0

        assert errors == []
        b.close()


def test_fullscript_card_escapes_hostile_product_name_no_script_injection(live_server):
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        script_count_before = None
        errors, dialogs = _open(pg, live_server, "lighttoken")

        card = pg.wait_for_selector(".fullscript-card")
        # Rendered as visible TEXT, not markup: inner_text decodes entities back to
        # the literal string, but no <script> node was created for it.
        assert XSS_NAME in card.inner_text()

        script_texts = pg.eval_on_selector_all(
            "script", "els => els.map(e => e.textContent || '')"
        )
        assert not any("alert(1)" in t for t in script_texts)

        # The escaped form must appear in the raw HTML (proves esc() ran, not that
        # the string merely never reached the DOM).
        inner_html = pg.eval_on_selector(".fullscript-card", "el => el.innerHTML")
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in inner_html
        assert "<script>alert(1)</script>" not in inner_html

        # No alert() dialog fired -- if the string had executed as a script this
        # would have popped a JS dialog, which our handler above would have caught.
        assert dialogs == []
        assert errors == []
        b.close()
