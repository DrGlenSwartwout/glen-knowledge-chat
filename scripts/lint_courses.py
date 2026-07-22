#!/usr/bin/env python3
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # repo root, for `dashboard`
from dashboard.courses_lint import lint_courses  # noqa: E402


def main() -> int:
    errors = lint_courses()
    if errors:
        print("Course content lint FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("Course content lint OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
