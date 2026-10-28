# Lesson format (agents: follow exactly)

Every track writes one lesson: `learn/<NN>-<track>.md` (for example `learn/03-env-rewards.md`).
The reader is JP: strong product/engineering background, new to RL training internals.
The goal is that he can explain and modify your code without you. A lesson he can skim has failed.

## Hard rules

1. **Never more than ~6 lines of prose without something for him to do**: a question, a command, or a sabotage.
2. **Predict before reveal.** Before every explanation or experiment, ask him to guess. Answers always go in a hidden block.
3. **Everything points at real code in this repo** with a clickable link (`[env.py:42](../fdagym/env.py#L42)`). No toy examples when the real thing exists.
4. **Real numbers only.** Paste actual outputs from commands you ran (counts, rewards, test names). Never invent output.
5. **15 minutes max** to read and do. Cut breadth, keep depth. Link out for the rest.
6. **Quiz blocks must use this exact syntax** so `scripts/quiz.py` can parse them:

```html
<details>
<summary><b>[predict]</b> Question text on one line?</summary>

Answer. Explain *why*, not just what. Mention the common wrong answer if there is one.
</details>
```

Allowed kinds: `predict` (guess before an experiment), `recall` (check a concept just taught),
`why` (justify a design decision), `transfer` (apply the idea to a new situation), `debug` (find the bug in a snippet).

## Sections, in order

### 1. Header
- Track, branch, time budget.
- **By the end you can:** 3 objectives, each an action verb you could test ("compute the reward of a trajectory by hand", not "understand rewards").

### 2. Where this fits
One mermaid diagram of the system with this track's part marked `YOU_ARE_HERE`. Two sentences on what breaks downstream if this part is wrong.

### 3. Concepts (2 to 4, no more)
Each concept follows this loop:
1. A `predict` question that exposes the naive intuition.
2. Short explanation (max 6 lines).
3. The real code that implements it, linked, with a 3 to 10 line excerpt.
4. A `recall` or `transfer` question.

### 4. Experiments (2 to 3)
A command he runs. A `predict` question before it. The real output you got, hidden in the answer.
Prefer experiments that surprise: the result contradicts the obvious guess.

### 5. Break it
One deliberate sabotage he applies by hand (a single line change, exact file and line).
Ask him to `predict` which test fails and why, then run the tests. Tell him how to revert (`git checkout -- <file>`).
If no test catches it, say so: that is a finding and should be an issue.

### 6. Decisions I made
2 to 4 decisions. For each: the options, what you chose, and one `why` or `transfer` question
("If openFDA had no rate limit, would you still cache? Why?").

### 7. Gotchas in the real data or tools
Surprises you hit while building, with evidence. These are the most valuable part; do not skip.

### 8. Checkpoint
- Tell him to run `python scripts/quiz.py learn/<this-file>.md`.
- 3 open questions for him to answer *in chat* to you (not in the quiz), harder than the quiz.
- Then **stop and wait**. When he says he's done, read `learn/answers/<this-file-stem>.jsonl`,
  address every `p` or `n` self-grade and every answer that reveals a misconception, and update the lesson
  if a question was unclear.

### 9. Read next
Max 3 links, each with one line on why it's worth his time.
