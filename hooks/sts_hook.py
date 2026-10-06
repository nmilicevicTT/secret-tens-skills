#!/usr/bin/env python3
"""Hook entry point for the sts plugin: sts_hook.py <session-start|post-skill|prompt>."""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILLS = ROOT / "skills"
PREFIX = "sts:"
STALE_SECONDS = 24 * 3600


def emit(event, text):
    print(json.dumps({"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}}))


def flag_path(session_id):
    return Path(tempfile.gettempdir()) / f"sts-active-{session_id}"


def last_skill_commit(name):
    # Worklogs are untracked, so the last commit under the skill dir dates its reference files.
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "log", "-1", "--format=%ct", "--", f"skills/{name}"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return int(out) if out else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def stale_worklogs():
    stale = []
    for wl in sorted(SKILLS.glob("*/worklog.md")):
        name = wl.parent.name
        ref = last_skill_commit(name)
        if ref is None:
            ref = max((p.stat().st_mtime for p in wl.parent.glob("*.md") if p != wl), default=0)
        age = wl.stat().st_mtime - ref
        if age > STALE_SECONDS:
            stale.append(f"{name} (worklog {age / 86400:.0f} d ahead of its reference files)")
    return stale


def session_start(_):
    rule = (ROOT / "rules" / "skill-evolution.md").read_text()
    text = f"sts plugin root (git checkout): {ROOT}\n\n{rule}"
    stale = stale_worklogs()
    if stale:
        text += "\nUnpromoted worklog entries: " + "; ".join(stale) + ". Offer to run /sts:evolve.\n"
    emit("SessionStart", text)


def post_skill(data):
    skill = (data.get("tool_input") or {}).get("skill", "")
    name = skill[len(PREFIX) :] if skill.startswith(PREFIX) else skill
    if not name or not (SKILLS / name / "SKILL.md").is_file() or not data.get("session_id"):
        return
    flag = flag_path(data["session_id"])
    names = set(flag.read_text().split()) if flag.exists() else set()
    names.add(name)
    flag.write_text("\n".join(sorted(names)))


def prompt(data):
    session_id = data.get("session_id")
    if not session_id:
        return
    flag = flag_path(session_id)
    if not flag.exists():
        return
    # Re-stated every prompt because a once-per-session rule fades in long sessions and after compaction.
    names = ", ".join(flag.read_text().split())
    emit(
        "UserPromptSubmit",
        f"sts skills active this session: {names}. Skill-evolution rule: record root causes, retractions "
        f"and user corrections in the skill when they happen (checkout {ROOT}).",
    )


def main():
    handlers = {"session-start": session_start, "post-skill": post_skill, "prompt": prompt}
    if len(sys.argv) != 2 or sys.argv[1] not in handlers:
        sys.exit(f"usage: {sys.argv[0]} {{{'|'.join(handlers)}}}")
    try:
        data = json.load(sys.stdin)
    except ValueError:
        data = {}
    # A broken hook must never block the session.
    try:
        handlers[sys.argv[1]](data)
    except Exception as exc:  # noqa: BLE001
        print(f"sts hook {sys.argv[1]} failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
