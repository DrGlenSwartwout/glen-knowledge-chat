// Tests the escGreeting helper inside static/client-portal.html.
// Extracts the real definition from the HTML and asserts its behavior,
// so paragraph/bold/escaping rendering can't silently regress.
const fs = require("fs");
const path = require("path");

const html = fs.readFileSync(
  path.join(__dirname, "..", "static", "client-portal.html"), "utf8");

const m = html.match(/const escGreeting\s*=\s*[\s\S]*?\n};?/);
if (!m) { console.error("FAIL: escGreeting not found in client-portal.html"); process.exit(1); }

// esc is a dependency of escGreeting; pull it in too.
const escDef = html.match(/const esc\s*=\s*\(s\)=>[^\n]*/)[0];
const escGreeting = eval("(function(){ " + escDef + "\n" + m[0] + "\n return escGreeting; })()");

let failed = 0;
function eq(got, want, label) {
  if (got !== want) { failed++; console.error(`FAIL ${label}\n  got:  ${JSON.stringify(got)}\n  want: ${JSON.stringify(want)}`); }
  else console.log(`ok   ${label}`);
}

eq(escGreeting("a\n\nb"), "<p>a</p><p>b</p>", "blank line -> separate paragraphs");
eq(escGreeting("**Hi** there"), "<p><strong>Hi</strong> there</p>", "**bold** -> <strong>");
eq(escGreeting("line1\nline2"), "<p>line1<br>line2</p>", "single newline -> <br>");
eq(escGreeting("<script>x"), "<p>&lt;script&gt;x</p>", "html is escaped (no raw <)");
eq(escGreeting(""), "", "empty string -> empty (default fallback still applies)");
eq(escGreeting("one"), "<p>one</p>", "plain text -> one paragraph");
eq(escGreeting("a\n\n\n\nb"), "<p>a</p><p>b</p>", "multiple blank lines collapse");

if (failed) { console.error(`\n${failed} failure(s)`); process.exit(1); }
console.log("\nall passed");
