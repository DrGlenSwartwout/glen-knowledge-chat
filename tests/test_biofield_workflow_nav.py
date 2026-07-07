from dashboard.biofield_report_html import _workflow_nav, render_author_html


def test_strip_has_all_four_tabs_with_key(monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://illtowell.com")
    monkeypatch.setenv("CONSOLE_SECRET", "sekrit")
    html = _workflow_nav("intake")
    assert "https://illtowell.com/console/biofield-portal?key=sekrit" in html
    assert "https://illtowell.com/console/biofield-reveals?key=sekrit" in html
    assert "https://illtowell.com/console/biofield-intake?key=sekrit" in html
    assert "https://illtowell.com/console/clinical-tags?key=sekrit" in html


def test_active_tab_highlighted(monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://illtowell.com")
    monkeypatch.setenv("CONSOLE_SECRET", "")
    html = _workflow_nav("tags")
    # The Tags anchor carries the active class; the others don't.
    assert '<a class="active" href="https://illtowell.com/console/clinical-tags">Tags</a>' in html
    assert '<a class="" href="https://illtowell.com/console/biofield-portal">Biofield</a>' in html


def test_no_secret_omits_key_query(monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://illtowell.com")
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    html = _workflow_nav("biofield")
    assert "?key=" not in html
    assert "https://illtowell.com/console/biofield-portal\"" in html


def test_no_email_omits_consult_ready_button(monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://illtowell.com")
    monkeypatch.setenv("CONSOLE_SECRET", "sekrit")
    html = _workflow_nav("intake", client_email="")
    assert "Mark consult-ready" not in html


def test_email_adds_consult_ready_button_quoted(monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://illtowell.com")
    monkeypatch.setenv("CONSOLE_SECRET", "sekrit")
    html = _workflow_nav("intake", client_email="j+test@x.com")
    assert "Mark consult-ready" in html
    assert "email=j%2Btest%40x.com" in html  # quoted
    assert "&amp;key=sekrit" in html  # href is HTML-escaped, so & becomes &amp;


def test_render_author_html_includes_strip_and_button(monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://illtowell.com")
    monkeypatch.setenv("CONSOLE_SECRET", "sekrit")
    rep = {"test_id": "a1", "client": {"name": "Jane", "email": "j@x.com"},
           "date": "2026-06-23", "layers": []}
    html = render_author_html(rep)
    assert "console/biofield-intake?key=sekrit" in html
    assert "Mark consult-ready" in html
    assert "email=j%40x.com" in html


def test_render_author_html_no_button_without_email(monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://illtowell.com")
    monkeypatch.setenv("CONSOLE_SECRET", "sekrit")
    rep = {"test_id": "a1", "client": {"name": "Jane", "email": ""},
           "date": "2026-06-23", "layers": []}
    html = render_author_html(rep)
    assert "Mark consult-ready" not in html
