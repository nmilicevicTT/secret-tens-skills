---
name: evolve
description: Curation pass that turns what a task taught into edits of this plugin's skills. Use at the end of a task or PR that used an sts:* skill, when worklog.md has grown past its skill's reference files, when the session-start check reports stale worklogs, or when the user says "update the skill" / "self-improve" / "evolve".
---

# evolve

Applies `rules/skill-evolution.md` in one deliberate pass. The rule handles single events as they
happen; this pass catches what was missed and keeps the skills lean.

## Steps

1. **Collect evidence.** Scan the session (root causes, retractions, user corrections, validated
   techniques) and every `skills/*/worklog.md` entry newer than the last skill commit
   (`git -C <root> log -1 --format=%ci -- skills/<name>`).
2. **Extract candidates.** One line each: claim, evidence pointer, target file.
3. **Gate.** Apply the gates of the rule. Drop or merge candidates that fail. Cluster candidates that
   share a cause.
4. **Edit.** Smallest change that makes the skill correct: replace, sharpen, or delete before adding.
   Keep the load table in sync with the files.
5. **Audit (only when touching a skill).** Spot-check that 3 of its file paths or symbols still exist
   in the target repo HEAD (`git grep`). Fix or delete stale ones.
6. **Check.** `python3 <root>/scripts/check_skills.py` must pass.
7. **Commit.** `git -C <root> pull --rebase -q`, one commit per skill: `<skill>: <verb> <what>`.
8. **Report.** List each edit as `file — entry`, plus the candidates you rejected and why (one line
   each). Ask before pushing unless the user has said to push automatically.

## Output budget

- Report: at most 15 lines.
- If nothing passes the gates, say "no skill changes" and give the strongest rejected candidate.

## New skill

Only on user request, or when one subject recurs across two tasks. Copy `templates/skill/` to `skills/<name>/`, fill
in the frontmatter `description` with trigger phrases, and add the skill to the README table.
