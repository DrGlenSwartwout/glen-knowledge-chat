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
      if (!/<span class="ob-mark ob-mark-done">\\u2713<\\/span>/.test(html)) {
        console.error('done step missing ob-mark-done class'); process.exit(1);
      }
      if (!/<span class="ob-mark ob-mark-open">\\u25cb<\\/span>/.test(html)) {
        console.error('open step missing ob-mark-open class'); process.exit(1);
      }
      console.log('ok');
    ''')
    out = subprocess.run(["node", "-e", js], cwd=".", capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


@pytest.mark.skipif(not shutil.which("node"), reason="node not available")
def test_render_onboarding_triage_form_gated_on_history_done():
    js = textwrap.dedent('''
      const fs = require('fs');
      const src = fs.readFileSync('static/js/portal-onboarding.js','utf8');
      const mod = {exports:{}};
      new Function('module','exports','window', src)(mod, mod.exports, {});
      const render = mod.exports.renderOnboarding || global.renderOnboarding;

      function baseStatus(historyDone, member) {
        return {
          phases: [
            {key:'be_read', title:'Be read', steps:[
              {key:'voice', label:'Voice analysis', done:true, href:'https://truly.vip/E4L'}
            ]},
            {key:'match', title:'Match remedies', steps:[
              {key:'history', label:'Starter remedies from your history', done:historyDone, href:'#recs'},
              {key:'scan_match', label:'Personalized match from your scan', done:false, href:'#recs'}
            ]},
            {key:'heal', title:'Accelerate healing', steps:[
              {key:'light', label:'Light', done:null, href:'https://clinicalpraxis.com'}
            ]}
          ],
          member: member
        };
      }

      // (a) history NOT done -> the triage form is present.
      const htmlNotDone = render(baseStatus(false, false));
      if (!/ob-triage-form/.test(htmlNotDone)) { console.error('triage form missing when history.done===false'); process.exit(1); }
      if (!/name="iop_od"/.test(htmlNotDone)) { console.error('missing iop_od input'); process.exit(1); }
      if (!/name="iop_os"/.test(htmlNotDone)) { console.error('missing iop_os input'); process.exit(1); }
      if (!/name="on_meds"/.test(htmlNotDone)) { console.error('missing on_meds checkbox'); process.exit(1); }
      if (!/name="field_loss"/.test(htmlNotDone)) { console.error('missing field_loss checkbox'); process.exit(1); }
      if (!/data-category="normal"/.test(htmlNotDone) || !/data-category="elevated"/.test(htmlNotDone)) {
        console.error('missing category fallback buttons'); process.exit(1);
      }

      // (a) history IS done -> the triage form is absent.
      const htmlDone = render(baseStatus(true, false));
      if (/ob-triage-form/.test(htmlDone)) { console.error('triage form present when history.done===true'); process.exit(1); }

      console.log('ok');
    ''')
    out = subprocess.run(["node", "-e", js], cwd=".", capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


@pytest.mark.skipif(not shutil.which("node"), reason="node not available")
def test_render_onboarding_member_thread():
    js = textwrap.dedent('''
      const fs = require('fs');
      const src = fs.readFileSync('static/js/portal-onboarding.js','utf8');
      const mod = {exports:{}};
      new Function('module','exports','window', src)(mod, mod.exports, {});
      const render = mod.exports.renderOnboarding || global.renderOnboarding;

      function baseStatus(member) {
        return {
          phases: [
            {key:'match', title:'Match remedies', steps:[
              {key:'history', label:'Starter remedies from your history', done:true, href:'#recs'}
            ]},
            {key:'heal', title:'Accelerate healing', steps:[
              {key:'light', label:'Light', done:null, href:'https://clinicalpraxis.com'}
            ]}
          ],
          member: member
        };
      }

      const htmlNonMember = render(baseStatus(false));
      if (!/Upgrade/.test(htmlNonMember)) { console.error('missing Upgrade affordance for non-member'); process.exit(1); }
      if (/Member ✓/.test(htmlNonMember)) { console.error('non-member should not show member marker'); process.exit(1); }

      const htmlMember = render(baseStatus(true));
      if (!/Member/.test(htmlMember) || !/\\u2713/.test(htmlMember)) {
        console.error('missing member marker for member:true'); process.exit(1);
      }
      if (/Upgrade/.test(htmlMember)) { console.error('member should not see Upgrade affordance'); process.exit(1); }

      console.log('ok');
    ''')
    out = subprocess.run(["node", "-e", js], cwd=".", capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
