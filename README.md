# T2I automated evaluator — orchestrator

Runs the multi-agent pipeline from `t2i-automated-evaluator.md` over a batch of image pairs. Every result is labelled `"evaluator": "automated"`.

## Setup

Python 3.10 or newer.

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # not needed for --mock
```

PowerShell: `$env:ANTHROPIC_API_KEY = "..."`

## Run

```bash
python orchestrator.py examples/tasks.jsonl --mock      # offline dry run, fake answers
python orchestrator.py my_tasks.jsonl                   # real run
python orchestrator.py my_tasks.jsonl --workers 3 --skip-done
python -m pytest -q                                     # tests
```

Tasks file: UTF-8, one JSON object per line (blank lines and `#` lines are ignored). Image paths are relative to the tasks file.

```json
{"task_id": "t001", "prompt": "...", "image_a": "imgs/t001_a.png", "image_b": "imgs/t001_b.png",
 "axes": ["instruction_following", "visual_quality", "ai_artifacts"]}
```

Add `"ps2_style"` to `axes` for tasks that score it. `task_id` may use letters, digits, `.`, `_` and `-` only, because it becomes the result file name.

The whole batch is checked before any model call. Invalid JSON (the line is named), repeated task ids, unknown or duplicate axes, a missing `instruction_following`, an empty prompt, missing image files, and `--only` ids that aren't in the file all stop the run with every problem listed.

Results are saved as each task finishes. Ctrl+C stops queued tasks from starting, lets the running ones finish, and keeps everything completed. `--skip-done` then resumes.

## Readable report

```bash
python report.py results                 # writes results/report.md
python report.py results --print --only t001
```

One section per task: overall result, ratings, the justification, the requirement checklist with A/B results, findings with boxes and verifier notes, why a result was flagged, and where the two inspections disagreed.

## Writing the feedback

`feedback.py` helps with the short text that goes into a rating form's open-feedback box. It doesn't write the text. It gives the facts, checks the rules, and remembers which style you pick.

```bash
python feedback.py points results/t001.json            # verdict check: suggested verdict, deciding reasons, flags
python feedback.py shapes results/t001.json            # three different shapes to write
python feedback.py lint results/t001.json drafts.txt   # check drafts, one per line; exit code 1 on a problem
python feedback.py log --task t001 --verdict B --size slightly --pair-type small_gap \
    --shape gap_key_point --shown gap_key_point,gap_compare,gap_summary --text-file picked.txt
python feedback.py pool                                # how often each shape was shown and picked
```

There are four verdicts: A is better, B is better, both good, both bad. `points` suggests one from the result: the pipeline's winner and strength, a confirmed major flaw in each image means both flawed, and a pair with no winner is both good or both bad. The kind of pair sets the sentence count (3 for a clear win, 2 when every win is slight and neither image has a confirmed major flaw).

`lint` enforces the hard rules: only periods, commas, apostrophes and hyphens inside words, no comma before "and", no banned words (reads as, thanks to, weaker, barely, and others), no first person, the exact first line for the verdict, and the sentence order. `log` appends your pick to `feedback_history.jsonl`. A shape that was shown three times and never picked drops out of the pool, but a pool never goes below three shapes.

## Pipeline

| Stage | Who | Sees | Output |
|---|---|---|---|
| 1 | Describer ×2 | one image, no prompt | blind description |
| 1 | Requirements | prompt only | checklist, central/secondary, style frame (always `ps2` when PS2 style is scored), and an interpretation of what the prompt really asks for |
| 2 | Inspector ×2 | both images as "Image 1/2", in both orders | requirement checks, findings with boxes, axis calls |
| 3 | Judge (code) | both runs | reconciled ratings; flags axes where the two runs disagree; "Image 1/2" in free text rewritten to A/B |
| 4 | Verifier | full image + crop of the cited region + same region of the other image | confirmed / rejected / unverifiable per finding |
| 4b | Artifact reviewer | each confirmed artifact finding: images, crop, prompt, interpretation | `ai_defect` / `natural` / `unclear`: a generative defect, or detail any photograph would show |
| 5 | Judge (code) | ratings | Overall by count rule, IF tiebreak, `unresolved` when the guide doesn't cover it |
| 6 | Writer | verified findings, final ratings, the checklist with results, and why each tied axis is tied | justification |
| 7 | Auditor + code checks | finished result, no images | pass/fail, review reasons |

Stages 1 and 2 run their calls in parallel. A rejected major finding lowers that axis to low confidence and flags it; ratings are never silently flipped.

**Reading the prompt.** Before anything is judged, the requirements stage writes down what the prompt is really asking for: the intent, the look the wording implies ("unstaged" or "phone photo" ask for an unpolished image, and a tidy stock-photo result is a miss), which imperfections are normal for that look (mild softness, a blown-out window, clutter, shallow depth of field), and the one deciding question. The inspectors judge against it, so an image isn't marked down on visual quality for looking like the kind of photo that was requested, and the writer and auditor use it too. It is saved in each result and shown in the report.

**Second opinion on artifacts.** A finding can be real and still not be an AI artifact: a blurry face on a photo behind the subject or an unreadable caption is usually just out-of-focus or small detail. Each confirmed artifact finding goes to a separate reviewer that decides whether it is a generative defect (wrong structure where the image is sharp) or ordinary photographic detail. Findings judged natural are not counted or presented as flaws. If that removes every flaw reported against the losing image, the artifacts call has no support left, so the axis becomes a low-confidence tie and the result is flagged for review. The Overall result is then recomputed from the new ratings.

**Findings.** The two runs' findings are merged when they have the same image, axis and region and their boxes overlap (or the other run used identical wording), keeping the higher severity. Two different defects in one region, such as two bad hands, stay as two findings. A box that isn't in 0–1 units (pixels, percentages) can't be cropped, so the Verifier gets the full image and is told no crop exists. Crops have generous padding and a minimum size, because Inspector boxes are approximate.

**Checks that flag a result for human review** (they never change a rating):

- an artifacts call that lost its support after the second opinion;
- an axis whose two runs disagreed (A vs B is reported as suspected position bias; tie vs a winner is reported as a disagreement);
- `instruction_following` favouring an image that did worse on the central requirements, or tied when they differ;
- `ai_artifacts` or `visual_quality` favouring an image that has worse verified findings (major before minor);
- a checklist that is empty, has repeated ids, has no central requirement, or lacks a result for an image; results for requirements that aren't on the checklist;
- low confidence, an unresolved Overall, findings left unverified by the cap, an Overall that doesn't match the count rule, and anything the Auditor fails.

**Retries.** A call cut off at `max_tokens` is retried with a larger limit (`MAX_TOKENS_GROWTH`, up to `MAX_TOKENS_CEILING`); other transient errors back off and retry with the same limit.

## Files

| File | Purpose |
|---|---|
| `config.py` | model, effort and token budget per agent; image, crop, merge and retry settings |
| `prompts/*.md` | one system prompt per agent; edit these to change behaviour |
| `schemas.py` | every agent's output shape (sent as structured-output schemas) and the result shape |
| `agents.py` | what each agent is given, and the call/retry wrapper |
| `judge.py` | all deterministic rules |
| `providers.py` | Anthropic, mock, and a Gemini placeholder |
| `imaging.py` | resize, encode, box validation, crop |
| `orchestrator.py` | CLI and pipeline |
| `report.py` | turns a results folder into a readable Markdown report |
| `feedback.py` | verdict check, shape pool, rule checker and pick log for the feedback text |
| `feedback_history.jsonl` | the log of picked feedback (created by the first `log`) |
| `tests/` | rules, agents, imaging, feedback helper, and mock end-to-end batch runs |

## Outputs (`results/`)

| File | Contents |
|---|---|
| `<task_id>.json` | full result for one task; the source of truth, written as the task finishes |
| `results.jsonl` | all results, rebuilt from the `<task_id>.json` files after every run, so re-running a task replaces its row |
| `summary.csv` | one row per task, rebuilt the same way; UTF-8 with a BOM so Excel reads it correctly |
| `calls.jsonl` | tokens and latency per call; appended, because it is a spend log |
| `errors.jsonl` | failed tasks and the reason; appended |

A result also carries what a reviewer needs to see why it was flagged: each finding's box and the Verifier's note, each axis's reason from both inspections (`axis_reasons`), and any disagreement with the blind descriptions (`description_conflicts`).
