import json
import subprocess
import sys
from pathlib import Path


def _repo():
    return Path(__file__).resolve().parent.parent


def test_app_compiles():
    r = subprocess.run([sys.executable, "-m", "py_compile", str(_repo() / "app.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_console_search_index_lists_topic_pages():
    data = json.loads((_repo() / "static" / "console-search-index.json").read_text())
    # accept either a list of entries or {"pages": [...]}
    blob = json.dumps(data)
    assert "/console/topic-pages" in blob


def test_topic_console_html_exists():
    assert (_repo() / "static" / "console-topic-pages.html").exists()
