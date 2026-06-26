# Task 3 Report — Cut Over: Redirects, Nav Collapse, Delete Old Pages

**Status:** COMPLETE
**Commit:** `dbd6f3c`
**Branch:** `sess/6a686b75-pages`

---

## Step 1: Redirect the 4 old routes (app.py)

Replaced each route body with a simple 302 redirect. The `_sales_console_ok()` guard and
cache-control headers were removed since the redirect is unconditional.

| Route | New body |
|---|---|
| `/console/sales-pages` | `return redirect("/console/pages#sales", code=302)` |
| `/console/ingredient-pages` | `return redirect("/console/pages#ingredient", code=302)` |
| `/console/topic-pages` | `return redirect("/console/pages#topic", code=302)` |
| `/console/topic-suggestions` | `return redirect("/console/pages#suggestions", code=302)` |

---

## Step 2: Collapse op-nav.js

**bosMods** — removed 4 entries (`sales`, `ingredients` Ingredient Pages, `topic-pages`,
`topic-suggestions`), added one in their place where `sales` was:
```js
{ id: "pages", label: "Pages", href: "/console/pages" + qs }
```

**NAV_PROFILES.glen.bos** — replaced `"sales","ingredients","topic-pages"` (consecutive)
with `"pages"`. `topic-suggestions` was not present in glen.bos (already in owner-More),
no change needed there. `rae.bos` untouched.

`node --check static/op-nav.js` → SYNTAX OK

---

## Step 3: Delete old pages + verify

```bash
git rm static/console-sales-pages.html static/console-ingredient-pages.html \
       static/console-topic-pages.html static/console-topic-suggestions.html
```

### Curl redirect verification (PORT=5098, CONSOLE_SECRET=test-secret)

```
sales-pages       -> 302 http://127.0.0.1:5098/console/pages#sales
ingredient-pages  -> 302 http://127.0.0.1:5098/console/pages#ingredient
topic-pages       -> 302 http://127.0.0.1:5098/console/pages#topic
topic-suggestions -> 302 http://127.0.0.1:5098/console/pages#suggestions
```

All 4 correct.

### Headless Playwright render

```
BOS ids: ['orders', 'money', 'crm', 'products', 'biofield', 'pages',
          'biofield-reveals', 'biofield-intake', 'reviews', 'shipping',
          'neworder', 'practitioners', 'top-products', 'remedy-meanings',
          'ingredients-ops', 'cert', 'coaching', 'studio-credits',
          'membership', 'atlas', 'wholesale', 'clips']
OK
```

- `pages` present: YES
- `ingredients-ops` present: YES
- None of `sales`, `ingredients`, `topic-pages`, `topic-suggestions` present: CONFIRMED
- Zero JS errors: CONFIRMED

### Dangling reference grep

Two developer provenance comments in `static/console-pages.html` referenced old filenames
(`console-topic-pages.html`, `console-ingredient-pages.html`). Updated to neutral wording.

```
grep -rn "console-sales-pages\.html|console-ingredient-pages\.html|
          console-topic-pages\.html|console-topic-suggestions\.html" app.py static/
-> CLEAN — no dangling refs
```

---

## Step 4: Commit

```
commit dbd6f3c
feat(console): cut over to Pages board — redirect old routes, collapse nav, drop old pages
7 files changed, 9 insertions(+), 1113 deletions(-)
delete mode 100644 static/console-ingredient-pages.html
delete mode 100644 static/console-sales-pages.html
delete mode 100644 static/console-topic-pages.html
delete mode 100644 static/console-topic-suggestions.html
```

---

## Concerns

Minor: two `//` comment lines in `static/console-pages.html` (not in the brief's change
list) referenced the deleted filenames. Updated comments to neutral wording — no logic
touched. Required to satisfy the grep-clean constraint.
