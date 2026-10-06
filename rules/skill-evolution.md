# Skill evolution rule (always on)

Skills in this plugin (`sts:*`) improve while you work. They live in a git checkout (path in the
header above); edits there are live from the next session.

## Triggers — update the skill before the turn ends

Context compaction erases details, so write at the event, not "later":

1. A bug is root-caused (symptom → cause → fix, with evidence).
2. A claim in the skill, or one you made, turns out wrong → fix the text; record the dead theory under
   "Retracted" so it is not re-chased.
3. The user corrects the method or states a preference that generalizes beyond this task.
4. A skill instruction was missing, wrong, or stale (file:line moved) and cost time.
5. A technique was validated that the skill does not describe.
6. Task or PR milestone → promote durable lessons from `worklog.md` into the reference files.

## Gates — an edit must pass all of them

- **Evidence.** Cite a test, log, commit, or file:line. Unverified ideas stay in `worklog.md` or are
  marked `(unverified)`.
- **General.** Record every finding, phrased so it transfers to the next task and the next model: name
  the scenario pattern (draft model, prediction heads, pipeline ranks, eager code between replays), not
  the product. Task state, host names, run IDs, dates and personal paths stay in `worklog.md`. The
  skills are public.
- **Not worse than now.** Sharpen or replace existing text; never append a line that contradicts
  another. On conflict, newer evidence wins and the old text is deleted or moved to "Retracted" with
  the reason.
- **New.** Search the skill first. A repeat or rename of an existing entry is merged into it. Several
  symptoms with one cause make one entry.
- **Facts, not steering.** Record what is true, what breaks, and how to check it. Avoid "next time try
  X" directions without evidence; they over-constrain later work.
- **Classify the failure first.** One shape, layout, resource, or mismatch error is usually a
  repairable implementation slip, not proof that the approach is wrong.
- **Absence counts.** If a needed invariant was nowhere in the skill, add it.

## Where

- `SKILL.md`: hard rules, workflow, validation, load table. Keep it under ~150 lines and move detail
  into sub-files listed in the load table.
- Sub-files: catalogues (`bugs.md`: symptom | cause | fix, plus "Retracted" and "Not <subject>"),
  recipes, internals.
- `worklog.md`: in-flight task state. Gitignored, local only.

## How

1. Edit in the checkout and run `python3 <root>/scripts/check_skills.py`.
2. `git -C <root> pull --rebase -q`, then commit `<skill>: <verb> <what>`.
3. Tell the user in one line: `skill updated: <file> — <entry>`. Push only when the user agrees or has
   said to push automatically.
4. Start a new skill only when the user asks, or when the same subject recurs over two tasks.

At the end of a task that used an `sts:*` skill, run `/sts:evolve`.
