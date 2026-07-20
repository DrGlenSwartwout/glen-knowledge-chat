#!/usr/bin/env python3
"""Run the test suite and fail only on NEW breakage.

Why a ratchet instead of "the suite must be green": the suite has 153 failures
that predate any CI, concentrated in auth-gated route tests. Blocking every PR
on fixing all of them would mean CI never gets adopted, and the status quo —
no CI at all, breakage discovered in production — is worse than a ratchet.

So: `tests/known_failures.txt` is the accepted baseline. CI fails if a test
that was passing starts failing. It also reports tests that were fixed, and
tells you to drop them from the baseline so the ratchet only tightens.

Usage:
    python3 scripts/ci_check.py                 # run + compare
    python3 scripts/ci_check.py --update        # rewrite the baseline

Regenerate the baseline the same way CI runs, so the two agree:
    env -u DOPPLER_TOKEN PINECONE_API_KEY=pcsk_fake OPENAI_API_KEY=sk-fake \
        ANTHROPIC_API_KEY=sk-ant-fake SECRET_KEY=ci \
        python3 scripts/ci_check.py --update
"""
import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tests" / "known_failures.txt"
# Only real test ids: "FAILED tests/foo.py::test_bar". Log lines can also start
# with ERROR, which is why the "tests/...::" shape is required.
LINE = re.compile(r"^(?:FAILED|ERROR) (tests/\S+::\S+)")


def run_suite():
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "-p", "no:randomly"],
        cwd=ROOT, capture_output=True, text=True,
    )
    out = proc.stdout + proc.stderr
    failures = {m.group(1) for line in out.splitlines() if (m := LINE.match(line))}
    summary = ""
    for line in out.splitlines():
        if re.search(r"\d+ (passed|failed)", line) and "warning" in line or \
           re.match(r"^\d+ (passed|failed)", line):
            summary = line.strip()
    # A collection crash yields few/no FAILED lines but a nonzero exit; do not
    # let that read as "nothing broke".
    if proc.returncode != 0 and not failures:
        print("PYTEST FAILED WITHOUT PARSEABLE FAILURES — treating as broken:\n")
        print(out[-4000:])
        sys.exit(1)
    return failures, summary


def read_baseline():
    if not BASELINE.exists():
        return set()
    return {l.strip() for l in BASELINE.read_text().splitlines()
            if l.strip() and not l.startswith("#")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true",
                    help="rewrite the baseline from the current run")
    args = ap.parse_args()

    failures, summary = run_suite()
    print(summary or "(no pytest summary line found)")

    if args.update:
        # Preserve any leading #-comment block. Without this every --update wiped the
        # header explaining WHY entries are in the baseline, so the reasoning survived
        # exactly one regeneration.
        header = []
        if BASELINE.exists():
            for line in BASELINE.read_text().splitlines():
                if line.startswith("#") or not line.strip():
                    header.append(line)
                else:
                    break
        body = "\n".join(sorted(failures)) + "\n"
        BASELINE.write_text(("\n".join(header) + "\n" + body) if header else body)
        print(f"baseline updated: {len(failures)} known failures")
        return 0

    known = read_baseline()
    new = sorted(failures - known)
    fixed = sorted(known - failures)

    print(f"\nknown failures: {len(known)}   this run: {len(failures)}")
    if fixed:
        print(f"\n{len(fixed)} test(s) now PASS and should leave the baseline:")
        for t in fixed:
            print(f"  + {t}")
        print("  run: python3 scripts/ci_check.py --update")
    if new:
        print(f"\n{len(new)} NEW failure(s) — this is the regression CI exists to catch:")
        for t in new:
            print(f"  - {t}")
        return 1
    print("\nNo new failures.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
