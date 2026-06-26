import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest


def _repo():
    return Path(__file__).resolve().parent.parent


def _extract_resolver():
    html = (_repo() / "static" / "dashboard.html").read_text()
    m = re.search(
        r"/\* === record-link resolver \(test-extracted\) === \*/(.*?)"
        r"/\* === end record-link resolver === \*/",
        html, re.S)
    assert m, "resolver marker block not found in dashboard.html"
    return m.group(1)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_resolver_behaviour():
    fn = _extract_resolver()
    script = fn + textwrap.dedent("""
      const links = {r1: {type:"person", display:"Jane",
                          url:"/console/crm?email=jane%40x.com"}};
      function assert(c, m){ if(!c){ console.error("FAIL: "+m); process.exit(1); } }

      // known ref -> anchor with rec-link class + url, no target=_blank
      let out = resolveRefLinks(
        '<a href="ref:r1" target="_blank" rel="noopener">Jane</a>', links);
      assert(out.indexOf('class="rec-link"') >= 0, "rec-link class");
      assert(out.indexOf('href="/console/crm?email=jane%40x.com"') >= 0, "resolved url");
      assert(out.indexOf("Jane") >= 0, "link text kept");
      assert(out.indexOf("target=") < 0, "no target=_blank on record link");

      // unknown ref -> unwrapped to plain text
      out = resolveRefLinks(
        '<a href="ref:r9" target="_blank" rel="noopener">Bob</a>', links);
      assert(out === "Bob", "unknown ref unwrapped, got: " + out);

      // real (non-ref) link untouched
      const ext = '<a href="https://x.com" target="_blank" rel="noopener">x</a>';
      assert(resolveRefLinks(ext, links) === ext, "external link untouched");

      // missing links map -> unwrap, no crash
      assert(resolveRefLinks('<a href="ref:r1">Jane</a>') === "Jane", "no map unwraps");
      console.log("OK");
    """)
    r = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_briefing_renderer_wires_resolver():
    html = (_repo() / "static" / "dashboard.html").read_text()
    assert "resolveRefLinks(mdRender(" in html
    assert "function recNavigate(" in html
    assert "encodeURIComponent(consoleKey)" in html  # key appended at click
