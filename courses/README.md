# MentorshipU course content

These are the launch courses served on mentorshipu.com. This is a SCAFFOLD: every file with a `REPLACE_WITH_...` value or a "Paste the ... here" body is a placeholder for you to fill in.

Full instructions, per-course checklist, and go-live steps:
`00 System/deploy-chat-specs/2026-07-21-mentorshipu-content-authoring-guide.md` (in the AI-Training vault).

Quick rules:
- Layout is uniform: `<course>/<module-slug>/<lesson-slug>.md`, listed in the course's `course.yaml`.
- Each lesson's frontmatter needs `title`, `access` (`public` | `member` | `paid`), `rumble_id` (the `v...` from `rumble.com/embed/<id>/`), optional `downloads`.
- The body is the formatted transcript (run the course-transcript-formatter skill on the Descript export).
- Before deploying, run `python3 scripts/lint_courses.py` — it fails on a bad `access`, missing `rumble_id`, unresolved download, or broken `course.yaml`.
