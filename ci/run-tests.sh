#!/usr/bin/env bash
# CI test runner.
#
# Deliberately a script rather than inline steps in .github/workflows/tests.yml: pushing
# anything under .github/workflows/ requires the `workflow` OAuth scope, which the usual
# gh token lacks, so every inline tweak needs a web-UI edit. Keeping the logic here means
# CI changes ship as ordinary pushes.
set -euo pipefail

# Test-only dependencies.
#   pytest  -- not a runtime dep, so not in requirements.txt.
#   pillow  -- scripts/build_journey_assets.py is a LOCAL build step whose outputs are
#              committed; its docstring says Pillow is intentionally kept out of
#              requirements.txt. But tests/test_journey_assets.py imports it, so CI needs
#              it. Do NOT "fix" this by adding Pillow to requirements.txt.
pip install --quiet pytest pillow

# Not a real key. It only has to be non-empty: the Pinecone client validates that in its
# constructor without making a network call, which is what app.py needs at import so the
# ~265 app-importing suites can run. Anything that actually reaches Pinecone 401s, which
# is why those suites are listed in ci/excluded-tests.txt.
export PINECONE_API_KEY="${PINECONE_API_KEY:-dummy-ci-key}"

if [ -z "$(grep -vE '^[[:space:]]*(#|$)' ci/excluded-tests.txt)" ]; then
  echo "::error::ci/excluded-tests.txt is empty -- refusing to run an unbounded suite"
  exit 1
fi

# Turn each excluded file into --ignore=, and let xargs pass them as separate arguments
# (not relying on shell word-splitting, which zsh does not do).
grep -vE '^[[:space:]]*(#|$)' ci/excluded-tests.txt \
  | sed 's#^#--ignore=#' \
  | xargs pytest tests/ -q -p no:cacheprovider --tb=short -rfE
