#!/usr/bin/env python3
"""Mechanical gates for skill edits: frontmatter, size budget, load-table links, leaked local details."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILL_MD_MAX_LINES = 160
LOCAL_ONLY = {"worklog.md"}
LEAKS = {
    "host name": re.compile(r"\bbh-glx-\S+"),
    "personal path": re.compile(r"/(?:data|home)/[a-z][a-z0-9_-]+/"),
    "token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9-]{20,})"),
}


def frontmatter(text):
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    if not m:
        return None
    return dict(line.split(":", 1) for line in m.group(1).splitlines() if ":" in line)


def check_skill(skill_dir):
    errors = []
    md = skill_dir / "SKILL.md"
    text = md.read_text()
    fm = frontmatter(text)
    if fm is None:
        return [f"{md}: missing frontmatter"]
    if fm.get("name", "").strip() != skill_dir.name:
        errors.append(f"{md}: frontmatter name != directory name {skill_dir.name}")
    if not fm.get("description", "").strip():
        errors.append(f"{md}: empty description")
    n = text.count("\n")
    if n > SKILL_MD_MAX_LINES:
        errors.append(f"{md}: {n} lines > {SKILL_MD_MAX_LINES}; move detail into sub-files")
    for ref in set(re.findall(r"\|\s*`([\w./-]+\.md)`\s*\|", text)):
        if ref not in LOCAL_ONLY and not (skill_dir / ref).is_file():
            errors.append(f"{md}: load table names missing file {ref}")
    return errors


def published_files():
    # Same set git would publish: tracked plus untracked-but-not-ignored.
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [ROOT / p for p in out.split() if p.endswith((".md", ".py", ".json", ".sh"))]


def check_leaks():
    errors = []
    for path in published_files():
        if path.resolve() == Path(__file__).resolve() or not path.is_file():
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            for kind, pattern in LEAKS.items():
                if pattern.search(line):
                    errors.append(f"{path.relative_to(ROOT)}:{lineno}: {kind}: {line.strip()[:80]}")
    return errors


def main():
    errors = []
    for skill_dir in sorted(p.parent for p in (ROOT / "skills").glob("*/SKILL.md")):
        errors += check_skill(skill_dir)
    errors += check_leaks()
    for e in errors:
        print(e)
    print(f"check_skills: {'FAIL' if errors else 'OK'} ({len(errors)} issues)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
