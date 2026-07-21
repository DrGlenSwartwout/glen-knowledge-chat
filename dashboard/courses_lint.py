from __future__ import annotations

import os

import frontmatter
import yaml

from dashboard.courses_content import VALID_ACCESS, courses_root


def lint_courses(root: str | None = None) -> list:
    root = root or courses_root()
    errors: list = []
    if not os.path.isdir(root):
        # The content tree doesn't exist yet in every checkout (it lands in Task 7).
        # Absent is not an error: nothing to lint, so this must exit clean in CI.
        return []
    for name in sorted(os.listdir(root)):
        cdir = os.path.join(root, name)
        yml = os.path.join(cdir, "course.yaml")
        if not os.path.isfile(yml):
            continue
        try:
            with open(yml) as f:
                spec = yaml.safe_load(f) or {}
        except Exception as e:
            errors.append(f"[{name}] course.yaml not parseable: {e}")
            continue
        if not str(spec.get("title", "")).strip():
            errors.append(f"[{name}] course.yaml missing title")
        seen = set()
        for m in spec.get("modules", []) or []:
            mslug = m.get("slug", "")
            mdir = os.path.join(cdir, mslug)
            if not os.path.isdir(mdir):
                errors.append(f"[{name}] module dir missing: {mslug}")
                continue
            for lslug in m.get("lessons", []) or []:
                key = (mslug, lslug)
                if key in seen:
                    errors.append(f"[{name}/{mslug}] duplicate lesson: {lslug}")
                seen.add(key)
                lp = os.path.join(mdir, f"{lslug}.md")
                if not os.path.isfile(lp):
                    errors.append(f"[{name}/{mslug}] lesson file missing: {lslug}")
                    continue
                errors.extend(_lint_lesson(name, mslug, lslug, lp, cdir))
    return errors


def _lint_lesson(course, mslug, lslug, path, cdir) -> list:
    errs = []
    try:
        post = frontmatter.load(path)
    except Exception as e:
        return [f"[{course}/{mslug}/{lslug}] frontmatter not parseable: {e}"]
    meta = post.metadata or {}
    tag = f"[{course}/{mslug}/{lslug}]"
    if not str(meta.get("title") or "").strip():
        errs.append(f"{tag} missing title")
    access = str(meta.get("access") or "").strip().lower()
    if access not in VALID_ACCESS:
        errs.append(f"{tag} invalid access '{access}' (must be one of {VALID_ACCESS})")
    if not str(meta.get("rumble_id") or "").strip():
        errs.append(f"{tag} missing rumble_id")
    for d in meta.get("downloads") or []:
        if not str(d.get("url") or "").strip():
            errs.append(f"{tag} download missing url: {d}")
    return errs
