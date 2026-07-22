from __future__ import annotations

import os
from dataclasses import dataclass

import frontmatter
import markdown as _md
import yaml

VALID_ACCESS = ("public", "member", "paid")


@dataclass
class Lesson:
    slug: str
    title: str
    access: str
    rumble_id: str
    downloads: list
    body_md: str
    module_slug: str
    course_slug: str


@dataclass
class Module:
    slug: str
    title: str
    lessons: list


@dataclass
class Course:
    slug: str
    title: str
    description: str
    modules: list


def courses_root() -> str:
    env = os.environ.get("COURSES_ROOT")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "courses")


def load_lesson(path: str, course_slug: str, module_slug: str) -> Lesson:
    post = frontmatter.load(path)
    meta = post.metadata or {}
    return Lesson(
        slug=os.path.splitext(os.path.basename(path))[0],
        title=str(meta.get("title", "")).strip(),
        access=str(meta.get("access", "")).strip().lower(),
        rumble_id=str(meta.get("rumble_id", "")).strip(),
        downloads=list(meta.get("downloads") or []),
        body_md=post.content,
        module_slug=module_slug,
        course_slug=course_slug,
    )


def load_course(course_slug: str, root: str | None = None) -> Course:
    root = root or courses_root()
    cdir = os.path.join(root, course_slug)
    with open(os.path.join(cdir, "course.yaml")) as f:
        spec = yaml.safe_load(f) or {}
    modules = []
    for m in spec.get("modules", []) or []:
        lessons = []
        for lslug in m.get("lessons", []) or []:
            lp = os.path.join(cdir, m["slug"], f"{lslug}.md")
            lessons.append(load_lesson(lp, course_slug, m["slug"]))
        modules.append(Module(slug=m["slug"], title=str(m.get("title", "")), lessons=lessons))
    return Course(
        slug=course_slug,
        title=str(spec.get("title", "")),
        description=str(spec.get("description", "")),
        modules=modules,
    )


def list_courses(root: str | None = None) -> list:
    root = root or courses_root()
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        if os.path.isfile(os.path.join(root, name, "course.yaml")):
            out.append(load_course(name, root))
    return out


def render_body(body_md: str) -> str:
    return _md.markdown(body_md or "", extensions=["extra"])
