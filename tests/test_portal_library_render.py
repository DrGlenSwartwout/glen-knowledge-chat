import shutil, subprocess, textwrap
import pytest

@pytest.mark.skipif(not shutil.which("node"), reason="node not available")
def test_render_library_emits_read_and_listen():
    js = textwrap.dedent('''
      const fs = require('fs');
      const src = fs.readFileSync('static/js/portal-library.js','utf8');
      const mod = {exports:{}};
      new Function('module','exports','window', src)(mod, mod.exports, {});
      const html = (mod.exports.renderLibrary || global.renderLibrary)([
        {slug:'healing-glaucoma-starter', title:'Healing Glaucoma — Starter',
         pdf_url:'/api/portal/T/library/healing-glaucoma-starter/pdf',
         audio_url:'/api/portal/T/library/healing-glaucoma-starter/audio'}]);
      if (!/Healing Glaucoma/.test(html)) { console.error('missing title'); process.exit(1); }
      if (!/\\/pdf/.test(html) || !/<audio/.test(html)) { console.error('missing read/listen'); process.exit(1); }
      console.log('ok');
    ''')
    out = subprocess.run(["node", "-e", js], cwd=".", capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
