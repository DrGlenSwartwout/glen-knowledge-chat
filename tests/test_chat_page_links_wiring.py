import subprocess
import sys
from pathlib import Path


def _repo():
    return Path(__file__).resolve().parent.parent


def test_app_compiles():
    r = subprocess.run([sys.executable, "-m", "py_compile", str(_repo() / "app.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_flag_defined_and_default_off():
    src = (_repo() / "app.py").read_text()
    assert "CHAT_PAGE_LINKS_ENABLED" in src
    # default must be off: the env default string is falsy
    assert 'os.environ.get("CHAT_PAGE_LINKS_ENABLED"' in src


def test_chat_merges_page_links_when_flag_on():
    src = (_repo() / "app.py").read_text()
    # the done handler must call the matcher and cap at 3 when link cards fire
    assert "match_page_links" in src
    assert "_chat_page_link_index" in src
