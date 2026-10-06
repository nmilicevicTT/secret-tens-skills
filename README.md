# secret-tens-skills

Claude Code skills for Tenstorrent work, packaged as one plugin (`sts`). The skills improve
while they are used.

| Skill | Use |
|---|---|
| `sts:tt-metal-trace` | Make a tt-metal model (or part of one) run under Metal trace capture/replay; debug traces that replay wrong, hang, or corrupt. |
| `sts:evolve` | Curation pass that turns what a task taught into skill edits (end of task / PR). |

## Install

Clone, then add the clone as a **local** marketplace. Claude Code loads plugins from a local
marketplace in place, so the skill edits Claude makes land in this git checkout. A plugin installed
from the GitHub URL is a cache copy instead, and updates overwrite edits made to it.

```bash
git clone https://github.com/nmilicevicTT/secret-tens-skills.git ~/secret-tens-skills
claude plugin marketplace add ~/secret-tens-skills
claude plugin install sts@secret-tens-skills
```

Start a new session. On a shared filesystem, one clone serves every host.

Requires `python3` and `git` on PATH (used by the hooks).

## Layout

```
.claude-plugin/   marketplace.json + plugin.json
skills/<name>/    SKILL.md (+ sub-files in its load table); worklog.md is local-only, gitignored
rules/            skill-evolution.md: the always-on rule
hooks/            sts_hook.py, wired in hooks.json
scripts/          check_skills.py: mechanical gates run before every skill commit
templates/skill/  starting point for a new skill
```

## How self-improvement works

A plugin cannot ship a CLAUDE.md or rules file, so the always-on part is a rule injected by hooks:

- **SessionStart** (startup, clear, compact) injects `rules/skill-evolution.md` with the checkout path.
  It also flags a skill whose `worklog.md` is more than a day newer than its last commit, because
  lessons are stuck in task notes.
- **PostToolUse(Skill)** records which `sts:*` skills the session loaded.
- **UserPromptSubmit** restates a one-line reminder on every prompt, but only in sessions that loaded
  an `sts:*` skill.

The rule sets triggers (root cause, retraction, user correction, stale instruction, validated
technique, milestone) and gates every edit must pass before it is committed:

| Gate | Source |
|---|---|
| Evidence or it didn't happen; absence of a needed rule is a finding; cluster symptoms with one cause | tt_ops_code_gen `agents/self-reflection.md` |
| Not worse than now: the current text competes with the edit, conflicts are resolved, not appended | Dream-RSI (arXiv 2609.14858) §3: the candidate set includes the current policy, so V* ≥ V0 |
| New: never a repeat or rename of an existing entry | Dream-RSI App. B.2 novelty constraint |
| Classify the failure first: shape/layout/resource/mismatch errors are normally repairable | Dream-RSI App. B.2 failure classes |
| Facts, not steering: no unevidenced "try X next" | Dream-RSI §5.1: explicit directional guidance from history underperformed unguided search |
| General, no host names / run IDs / personal paths in tracked files | Dream-RSI prefix-only constraint (no hardcoded winners); enforced by `check_skills.py` |
| Progressive disclosure: short SKILL.md + load table | tt-buddy skill layout |

Dream-RSI improves search policies by replaying recorded discovery trees; it does not edit prose
skills. Only its acceptance and novelty rules carry over here.

## Add a skill

Ask Claude for it, or copy `templates/skill/` to `skills/<name>/`, then run
`python3 scripts/check_skills.py` and commit.
