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
          {key:'be_read', title:'Discover What Your Body Is Saying', steps:[
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
      if (!/Discover What Your Body Is Saying/.test(html)) { console.error('missing be_read title'); process.exit(1); }
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
            {key:'be_read', title:'Discover What Your Body Is Saying', steps:[
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

      // (a) history NOT done -> the condition checklist is present.
      const htmlNotDone = render(baseStatus(false, false));
      if (!/ob-triage-form/.test(htmlNotDone)) { console.error('triage form missing when history.done===false'); process.exit(1); }
      if (!/name="iop_od"/.test(htmlNotDone)) { console.error('missing iop_od input'); process.exit(1); }
      if (!/name="iop_os"/.test(htmlNotDone)) { console.error('missing iop_os input'); process.exit(1); }
      const required = ['glaucoma','cataract','macular','dry-eye','retinitis-pigmentosa','diabetic-retinopathy','other'];
      for (const condition of required) {
        if (!htmlNotDone.includes('value="' + condition + '"')) {
          console.error('missing condition ' + condition); process.exit(1);
        }
      }
      if (!/name="other_condition"/.test(htmlNotDone)) {
        console.error('missing Other free-text field'); process.exit(1);
      }
      for (const kind of ['prescriptions','otc','supplements']) {
        if (!htmlNotDone.includes('name="' + kind + '_yes"') ||
            !htmlNotDone.includes('name="' + kind + '_text"')) {
          console.error('missing current-products fields for ' + kind); process.exit(1);
        }
      }
      for (const kind of ['surgeries','physical_trauma','psychoemotional_trauma',
                           'toxins','vaccinations','family_history','diagnoses',
                           'allergies','dental','sleep']) {
        if (!htmlNotDone.includes('name="' + kind + '_yes"') ||
            !htmlNotDone.includes('name="' + kind + '_text"')) {
          console.error('missing extended-history fields for ' + kind); process.exit(1);
        }
      }
      if (!/parent or grandparent/.test(htmlNotDone) ||
          !/current or past/.test(htmlNotDone) ||
          !/age at the time/.test(htmlNotDone)) {
        console.error('extended history is missing intake-parallel details'); process.exit(1);
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


@pytest.mark.skipif(not shutil.which("node"), reason="node not available")
def test_triage_success_message_adds_biofield_nudge_when_consult_recommended():
    """Consult nudge (cataract/macular triage): the success message stays the
    short baseline copy normally, and additionally invites a Biofield
    Analysis with Dr. Glen when the triage response's consult_recommended
    flag is true (e.g. a wet-AMD resolve)."""
    js = textwrap.dedent('''
      const fs = require('fs');
      const src = fs.readFileSync('static/js/portal-onboarding.js','utf8');
      const mod = {exports:{}};
      new Function('module','exports','window', src)(mod, mod.exports, {});
      const buildMsg = mod.exports._triageSuccessMessage;

      const plain = buildMsg(false);
      if (!/starter remedies are ready/.test(plain)) { console.error('missing baseline copy'); process.exit(1); }
      if (/Biofield/.test(plain)) { console.error('non-consult message should not mention Biofield'); process.exit(1); }

      const nudged = buildMsg(true);
      if (!/starter remedies are ready/.test(nudged)) { console.error('nudged message dropped baseline copy'); process.exit(1); }
      if (!/Biofield Analysis/.test(nudged)) { console.error('missing Biofield Analysis nudge'); process.exit(1); }
      if (!/Dr\\. Glen/.test(nudged)) { console.error('missing Dr. Glen mention'); process.exit(1); }
      const nudgeOnly = nudged.split('ready.')[1] || '';
      if (/—/.test(nudgeOnly)) { console.error('nudge text must not use an em dash'); process.exit(1); }
      if (/\\b[A-Z]{4,}\\b/.test(nudgeOnly)) { console.error('nudge text must not use ALL CAPS'); process.exit(1); }

      console.log('ok');
    ''')
    out = subprocess.run(["node", "-e", js], cwd=".", capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
