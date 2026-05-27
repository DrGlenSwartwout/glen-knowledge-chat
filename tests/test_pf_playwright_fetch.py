"""Tests for the shared Playwright fetcher used by AANP / NANP / NCCAOM.

The fetcher is thin glue over Playwright's sync API. We test:
  - The context manager calls playwright start, launches chromium,
    creates a context + page, and tears all three down on exit.
  - ``get()`` navigates, waits for the configured selector, sleeps, and
    returns ``page.content()``.
  - ``get()`` swallows selector-wait timeouts (consistent with the
    "render best-effort, parser handles the rest" contract).
  - ``get()`` falls back to ``networkidle`` when no selector is given.
  - ``post()`` runs the form-injection script via ``page.evaluate`` with
    the correct payload and uses ``expect_navigation``.

The mocks intercept ``playwright.sync_api.sync_playwright`` so no
chromium process is launched during the unit-test run.

One integration smoke test (``test_integration_example_com``) is skipped
by default — it actually launches headless chromium and fetches
``https://example.com`` to confirm the wiring works end-to-end. Run it
manually via:

    pytest tests/test_pf_playwright_fetch.py::test_integration_example_com -v --no-header -s
"""
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from scrapers.practitioner_finder.playwright_fetch import (
    DEFAULT_SLEEP_S,
    DEFAULT_TIMEOUT_MS,
    PlaywrightFetcher,
    USER_AGENT,
    playwright_session,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_page() -> MagicMock:
    """Mock Playwright Page with the methods the fetcher touches."""
    page = MagicMock(name="page")
    page.content.return_value = "<html><body>OK</body></html>"
    page.goto.return_value = None
    page.wait_for_selector.return_value = None
    page.wait_for_load_state.return_value = None
    page.wait_for_timeout.return_value = None
    page.evaluate.return_value = None
    page.set_default_timeout.return_value = None

    @contextmanager
    def _expect_nav(*args, **kwargs):
        yield MagicMock(name="navigation_response")

    page.expect_navigation = _expect_nav
    return page


# ---------------------------------------------------------------------------
# PlaywrightFetcher.get()
# ---------------------------------------------------------------------------


def test_get_navigates_and_returns_content():
    page = _make_mock_page()
    page.content.return_value = "<html>RESULT</html>"
    fetcher = PlaywrightFetcher(page)

    out = fetcher.get("https://example.com/foo")

    page.goto.assert_called_once_with(
        "https://example.com/foo",
        wait_until="domcontentloaded",
        timeout=DEFAULT_TIMEOUT_MS,
    )
    # No selector -> fall back to networkidle wait.
    page.wait_for_load_state.assert_called_once_with(
        "networkidle", timeout=DEFAULT_TIMEOUT_MS
    )
    page.wait_for_selector.assert_not_called()
    # Polite-sleep applied.
    page.wait_for_timeout.assert_called_once_with(int(DEFAULT_SLEEP_S * 1000))
    assert out == "<html>RESULT</html>"


def test_get_waits_for_selector_when_provided():
    page = _make_mock_page()
    fetcher = PlaywrightFetcher(page)

    fetcher.get("https://example.com/bar", wait_for_selector="#grid")

    page.wait_for_selector.assert_called_once_with(
        "#grid", timeout=DEFAULT_TIMEOUT_MS
    )
    page.wait_for_load_state.assert_not_called()


def test_get_swallows_selector_timeout():
    """A wait_for_selector timeout must not propagate — the parser still
    runs against whatever rendered."""
    page = _make_mock_page()
    page.wait_for_selector.side_effect = RuntimeError("Timeout 20000ms exceeded.")
    page.content.return_value = "<html>PARTIAL</html>"
    fetcher = PlaywrightFetcher(page)

    out = fetcher.get("https://example.com/slow", wait_for_selector=".rows")

    assert out == "<html>PARTIAL</html>"
    page.content.assert_called_once()


def test_get_swallows_networkidle_timeout():
    page = _make_mock_page()
    page.wait_for_load_state.side_effect = RuntimeError(
        "Timeout 20000ms exceeded."
    )
    fetcher = PlaywrightFetcher(page)

    # Must not raise.
    fetcher.get("https://example.com/")


def test_get_zero_sleep_skips_wait_for_timeout():
    page = _make_mock_page()
    fetcher = PlaywrightFetcher(page)

    fetcher.get("https://example.com/", sleep_s=0)

    page.wait_for_timeout.assert_not_called()


# ---------------------------------------------------------------------------
# PlaywrightFetcher.post()
# ---------------------------------------------------------------------------


def test_post_injects_form_with_payload():
    page = _make_mock_page()
    fetcher = PlaywrightFetcher(page)

    fetcher.post(
        "https://example.com/search",
        form_data={
            "__VIEWSTATE": "abc",
            "__EVENTTARGET": "ctl00$Page2",
        },
        wait_for_selector="#SearchResultsGrid",
    )

    # evaluate called exactly once with the (script, payload) pair.
    assert page.evaluate.call_count == 1
    args, _ = page.evaluate.call_args
    script, payload = args
    assert "document.createElement('form')" in script
    assert "f.submit()" in script
    assert payload == {
        "url": "https://example.com/search",
        "fields": {
            "__VIEWSTATE": "abc",
            "__EVENTTARGET": "ctl00$Page2",
        },
    }
    # Selector wait propagated.
    page.wait_for_selector.assert_called_once_with(
        "#SearchResultsGrid", timeout=DEFAULT_TIMEOUT_MS
    )


def test_post_swallows_selector_timeout():
    page = _make_mock_page()
    page.wait_for_selector.side_effect = RuntimeError("timeout")
    fetcher = PlaywrightFetcher(page)

    out = fetcher.post(
        "https://example.com/search",
        form_data={"x": "1"},
        wait_for_selector="#grid",
    )

    assert out == "<html><body>OK</body></html>"


# ---------------------------------------------------------------------------
# playwright_session() context manager
# ---------------------------------------------------------------------------


def test_session_opens_and_closes_cleanly():
    fake_page = _make_mock_page()
    fake_context = MagicMock(name="context")
    fake_context.new_page.return_value = fake_page
    fake_browser = MagicMock(name="browser")
    fake_browser.new_context.return_value = fake_context
    fake_pw = MagicMock(name="pw_instance")
    fake_pw.chromium.launch.return_value = fake_browser

    with patch(
        "scrapers.practitioner_finder.playwright_fetch.sync_playwright"
    ) as sync_pw:
        sync_pw.return_value.start.return_value = fake_pw

        with playwright_session() as fetcher:
            assert isinstance(fetcher, PlaywrightFetcher)
            # Static UA propagated to the context.
            fake_browser.new_context.assert_called_once_with(user_agent=USER_AGENT)
            fake_pw.chromium.launch.assert_called_once_with(headless=True)

        # On exit, context + browser + pw all torn down in reverse order.
        fake_context.close.assert_called_once()
        fake_browser.close.assert_called_once()
        fake_pw.stop.assert_called_once()


def test_session_tears_down_on_exception():
    fake_page = _make_mock_page()
    fake_context = MagicMock(name="context")
    fake_context.new_page.return_value = fake_page
    fake_browser = MagicMock(name="browser")
    fake_browser.new_context.return_value = fake_context
    fake_pw = MagicMock(name="pw_instance")
    fake_pw.chromium.launch.return_value = fake_browser

    with patch(
        "scrapers.practitioner_finder.playwright_fetch.sync_playwright"
    ) as sync_pw:
        sync_pw.return_value.start.return_value = fake_pw

        with pytest.raises(RuntimeError, match="boom"):
            with playwright_session():
                raise RuntimeError("boom")

        fake_context.close.assert_called_once()
        fake_browser.close.assert_called_once()
        fake_pw.stop.assert_called_once()


def test_session_passes_headless_false():
    fake_page = _make_mock_page()
    fake_context = MagicMock(name="context")
    fake_context.new_page.return_value = fake_page
    fake_browser = MagicMock(name="browser")
    fake_browser.new_context.return_value = fake_context
    fake_pw = MagicMock(name="pw_instance")
    fake_pw.chromium.launch.return_value = fake_browser

    with patch(
        "scrapers.practitioner_finder.playwright_fetch.sync_playwright"
    ) as sync_pw:
        sync_pw.return_value.start.return_value = fake_pw
        with playwright_session(headless=False):
            pass
        fake_pw.chromium.launch.assert_called_once_with(headless=False)


# ---------------------------------------------------------------------------
# Integration smoke (skipped by default)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="requires network + chromium binary")
def test_integration_example_com():
    """Real chromium launch + fetch of https://example.com. Skipped in
    the default test run; flip the decorator off to verify wiring."""
    with playwright_session() as fetcher:
        html = fetcher.get("https://example.com")
        assert "Example Domain" in html
