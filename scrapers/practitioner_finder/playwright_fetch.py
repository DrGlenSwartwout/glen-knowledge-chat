"""Cloudflare-aware fetcher for adapters whose host sites serve a JS challenge.

Used by AANP / NANP / NCCAOM. Playwright runs the JS so the cf_clearance
cookie is granted in the browser context; subsequent requests within the
same session sail through without per-request challenges.

Design notes
------------
- A single Playwright browser context per session keeps the cf_clearance
  cookie warm across many requests (the whole point of the shim).
- The browser is launched headless. We deliberately use a static Mozilla
  UA matching the other adapters' static-UA convention — Cloudflare grants
  clearance based on the JS challenge solve, not the UA string, so this
  doesn't reduce the success rate but does make logs consistent.
- `get()` is a "navigate + return rendered HTML" call. `post()` injects a
  hidden form and submits it (the easiest way to drive an ASP.NET
  ``__doPostBack`` continuation, since we need both the cookie AND the
  full re-rendered page back).
- Selector waits silently swallow timeouts: callers consistently want
  "best-effort render, then parse whatever's there" — same contract as
  the static-UA `requests.get` it replaces (which doesn't know about
  rendering at all).
- No retries, no captcha-solver hook — those belong upstream of this
  module. If a host's challenge is harder than a stock Cloudflare JS
  check this shim won't help (see anticipated-failures in the runner
  report).
"""
from contextlib import contextmanager
from typing import Iterator, Optional

from playwright.sync_api import sync_playwright, Page
from playwright_stealth import Stealth

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/17.0 Safari/605.1.15"
)
DEFAULT_TIMEOUT_MS = 20_000
DEFAULT_SLEEP_S = 0.5


class PlaywrightFetcher:
    """Thin sync wrapper around a Playwright ``Page`` for GET + POST fetches.

    Instantiated by ``playwright_session()``. The same fetcher is re-used
    across the whole scrape so the Cloudflare cookie persists.
    """

    def __init__(self, page: Page):
        self._page = page

    # ------------------------------------------------------------------
    # GET
    # ------------------------------------------------------------------

    def get(
        self,
        url: str,
        wait_for_selector: Optional[str] = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        sleep_s: float = DEFAULT_SLEEP_S,
    ) -> str:
        """Navigate to ``url`` and return ``page.content()``.

        ``wait_for_selector`` — if given, waits for this CSS selector to
        appear before returning. Timeout is silently swallowed: the parser
        will get whatever rendered. If omitted, waits for ``networkidle``
        instead (also silently bounded by ``timeout_ms``).

        ``sleep_s`` mirrors the polite per-request sleep the static-UA
        adapters apply (0.5s default).
        """
        self._page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        self._post_wait(wait_for_selector, timeout_ms)
        if sleep_s > 0:
            self._page.wait_for_timeout(int(sleep_s * 1000))
        return self._page.content()

    # ------------------------------------------------------------------
    # POST
    # ------------------------------------------------------------------

    def post(
        self,
        url: str,
        form_data: dict,
        wait_for_selector: Optional[str] = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        sleep_s: float = DEFAULT_SLEEP_S,
    ) -> str:
        """Submit a POST as a form-encoded request and return rendered HTML.

        Implementation: build a hidden ``<form>`` in the current page via
        ``page.evaluate(...)`` with one ``<input>`` per ``form_data`` entry,
        then click submit. This works for two-step navigations (ASP.NET
        WebForms ``__doPostBack`` continuations are exactly this shape:
        the next page's URL is the same path, and the server uses
        ``__VIEWSTATE`` + ``__EVENTVALIDATION`` + ``__EVENTTARGET`` /
        ``__EVENTARGUMENT`` to drive the page transition).

        Caller must be on a page in the same origin as ``url`` already
        (the form action is the target URL; same-origin POSTs are the
        normal case for ASP.NET pagination). For the first POST in a
        session the runner navigates somewhere on the origin first.
        """
        # JSON-safe payload of inputs (escape attribute values as needed).
        # Playwright's page.evaluate JSON-encodes the second arg, so we
        # can pass the dict straight through.
        script = """
        ({url, fields}) => {
          const f = document.createElement('form');
          f.method = 'POST';
          f.action = url;
          f.style.display = 'none';
          for (const [k, v] of Object.entries(fields)) {
            const i = document.createElement('input');
            i.type = 'hidden';
            i.name = k;
            i.value = v == null ? '' : String(v);
            f.appendChild(i);
          }
          document.body.appendChild(f);
          f.submit();
        }
        """
        # Drive the navigation and wait for it to commit.
        with self._page.expect_navigation(
            wait_until="domcontentloaded", timeout=timeout_ms
        ):
            self._page.evaluate(script, {"url": url, "fields": form_data})
        self._post_wait(wait_for_selector, timeout_ms)
        if sleep_s > 0:
            self._page.wait_for_timeout(int(sleep_s * 1000))
        return self._page.content()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _post_wait(self, wait_for_selector: Optional[str], timeout_ms: int) -> None:
        """Wait for a selector or networkidle; never raises."""
        if wait_for_selector:
            try:
                self._page.wait_for_selector(
                    wait_for_selector, timeout=timeout_ms
                )
            except Exception:  # noqa: BLE001 - silently swallow per contract
                pass
            return
        try:
            self._page.wait_for_load_state("networkidle", timeout=timeout_ms)
        except Exception:  # noqa: BLE001 - silently swallow per contract
            pass


@contextmanager
def playwright_session(headless: bool = True) -> Iterator[PlaywrightFetcher]:
    """Context manager: launch chromium, yield a ``PlaywrightFetcher``.

    Cleanly tears down the browser + Playwright instance on exit (even
    on exception). The fetcher's underlying page persists for the
    duration of the ``with`` block, keeping the Cloudflare cookie hot
    across all requests inside it.
    """
    pw = sync_playwright().start()
    browser = None
    context = None
    try:
        browser = pw.chromium.launch(headless=headless)
        context = browser.new_context(user_agent=USER_AGENT)
        page = context.new_page()
        page.set_default_timeout(DEFAULT_TIMEOUT_MS)
        try:
            Stealth().apply_stealth_sync(page)
        except Exception:  # noqa: BLE001 - stealth is best-effort
            pass
        yield PlaywrightFetcher(page)
    finally:
        try:
            if context is not None:
                context.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if browser is not None:
                browser.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            pw.stop()
        except Exception:  # noqa: BLE001
            pass
