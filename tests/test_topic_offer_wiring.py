import json, subprocess, sys
from pathlib import Path


def _repo():
    return Path(__file__).resolve().parent.parent


def test_app_compiles():
    r = subprocess.run([sys.executable, "-m", "py_compile", str(_repo() / "app.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_flag_present_default_off():
    src = (_repo() / "app.py").read_text()
    assert 'os.environ.get("CHAT_TOPIC_OFFER_ENABLED"' in src


def test_wiring_references():
    src = (_repo() / "app.py").read_text()
    assert "extract_topic_candidate" in src
    assert "/learn/suggest/" in src
    assert "record_suggestion" in src
    assert "list_suggestions" in src


def test_console_search_index_lists_suggestions():
    blob = json.dumps(json.loads((_repo() / "static" / "console-search-index.json").read_text()))
    assert "/console/topic-suggestions" in blob


def test_console_suggestions_html_exists():
    assert (_repo() / "static" / "console-topic-suggestions.html").exists()
