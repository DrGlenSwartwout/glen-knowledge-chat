import shutil, subprocess, textwrap
import pytest

@pytest.mark.skipif(not shutil.which("node"), reason="node not available")
def test_render_onboarding_emits_phases_done_and_link():
    js = textwrap.dedent('''
      const fs = require('fs');
      const src = fs.readFileSync('static/js/portal-onboarding.js','utf8');
      const mod = {exports:{}};
      new Function('module','exports','window', src)(mod, mod.exports, {});
      const status = {
        phases: [
          {key:'be_read', title:'Be read', steps:[
            {key:'voice', label:'Voice analysis', done:true, href:'https://truly.vip/E4L'},
            {key:'intake', label:'Intake', done:false, href:'https://truly.vip/Join'}
          ]},
          {key:'match', title:'Match remedies', steps:[
            {key:'scan_match', label:'Personalized match from your scan', done:false, href:'#recs'}
          ]},
          {key:'heal', title:'Accelerate healing', steps:[
            {key:'light', label:'Light', done:null, href:'https://clinicalpraxis.com'},
            {key:'pemf', label:'PEMF', done:null, href:'', soon:true}
          ]}
        ],
        member: false
      };
      const html = (mod.exports.renderOnboarding || global.renderOnboarding)(status);
      if (!/Be read/.test(html)) { console.error('missing be_read title'); process.exit(1); }
      if (!/Match remedies/.test(html)) { console.error('missing match title'); process.exit(1); }
      if (!/Accelerate healing/.test(html)) { console.error('missing heal title'); process.exit(1); }
      if (!/\\u2713/.test(html)) { console.error('missing done check'); process.exit(1); }
      if (!/<a[^>]*href="https:\\/\\/clinicalpraxis\\.com"[^>]*>Light<\\/a>/.test(html)) {
        console.error('missing heal link anchor'); process.exit(1);
      }
      if (!/coming soon/.test(html)) { console.error('missing soon badge'); process.exit(1); }
      console.log('ok');
    ''')
    out = subprocess.run(["node", "-e", js], cwd=".", capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
