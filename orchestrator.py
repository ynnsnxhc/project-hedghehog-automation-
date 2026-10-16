"""Runs the T2I evaluation pipeline over a batch of tasks.

    python orchestrator.py examples/tasks.jsonl                # real run (needs ANTHROPIC_API_KEY)
    python orchestrator.py examples/tasks.jsonl --mock         # offline dry run, no API calls
    python orchestrator.py tasks.jsonl --only task_007 --workers 2

Each line of the tasks file is JSON:
    {"task_id": "...", "prompt": "...", "image_a": "path", "image_b": "path",
     "axes": ["instruction_following", "visual_quality", "ai_artifacts"]}
Image paths are relative to the tasks file. Add "ps2_style" to axes when the task scores it.
Lines that are blank or start with # are ignored. The file is read as UTF-8.

The whole batch is checked before any model call: bad JSON, repeated or unsafe task ids,
unknown axes and missing image files stop the run with every problem listed.

Outputs (in --out, default results/):
    <task_id>.json      full result for one task (the source of truth; written as each task finishes)
    results.jsonl       one line per task, rebuilt from the <task_id>.json files after every run
    summary.csv         one row per task, rebuilt the same way; UTF-8 with a BOM so Excel reads it
    calls.jsonl         every model call: agent, model, tokens, seconds (appended; it is a spend log)
    errors.jsonl        tasks that failed, with the reason (appended)
"""
import argparse
import csv
import json
import os
import re
import sys
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import agents
import config
import imaging
import judge
import schemas as s
from providers import RefusalError, Usage

VALID_AXES = set(s.SCORED_AXES)
DEFAULT_AXES = ["instruction_following", "visual_quality", "ai_artifacts"]
TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")  # becomes a file name, so no separators or '..'

SUMMARY_COLUMNS = ["task_id", "evaluator", "overall", "strength", "rule", *s.SCORED_AXES,
                   "flipped_axes", "needs_review", "review_reasons", "calls", "input_tokens", "output_tokens"]


# ------------------------------------------------------------------ one task
def task_axes(task):
    return list(task.get("axes") or DEFAULT_AXES)


def validate_task(task, base_dir):
    """Everything that can be wrong with a task before any model is called. Returns a list of problems."""
    problems = []
    task_id = task.get("task_id")
    if not isinstance(task_id, str) or not TASK_ID.fullmatch(task_id):
        problems.append(f"task_id {task_id!r} must be letters, digits, '.', '_' or '-' (it names the result file)")
    if not isinstance(task.get("prompt"), str) or not task["prompt"].strip():
        problems.append("prompt is empty")
    if "axes" in task and not isinstance(task["axes"], list):
        problems.append("axes must be a list")
    else:
        axes = task_axes(task)
        bad = sorted(set(map(str, axes)) - VALID_AXES)
        if bad:
            problems.append(f"unknown axes {bad}")
        if len(set(map(str, axes))) != len(axes):
            problems.append("axes lists the same axis twice")
        if "instruction_following" not in axes:
            problems.append("instruction_following must be scored (it is the Overall tiebreaker)")
    for key in ("image_a", "image_b"):
        path = base_dir / str(task.get(key, ""))
        if not path.is_file():
            problems.append(f"{key} not found: {path}")
    return problems


def run_task(task, base_dir, mock=False):
    """Evaluate one task. Returns the result and the per-call usage log."""
    problems = validate_task(task, base_dir)
    if problems:
        raise ValueError("; ".join(problems))
    task_id = task["task_id"]
    axes = task_axes(task)
    prompt = task["prompt"]
    usage = Usage()
    review = []

    images = {"A": imaging.load(base_dir / task["image_a"]), "B": imaging.load(base_dir / task["image_b"])}
    parts = {k: imaging.encode(v) for k, v in images.items()}

    with ThreadPoolExecutor(max_workers=3) as pool:
        # Stage 1: blind descriptions and the checklist are independent.
        f_desc_a = pool.submit(agents.describe, parts["A"], "A", usage, mock)
        f_desc_b = pool.submit(agents.describe, parts["B"], "B", usage, mock)
        f_reqs = pool.submit(agents.requirements, prompt, axes, usage, mock)
        desc = {"A": f_desc_a.result(), "B": f_desc_b.result()}
        reqs = f_reqs.result()

        # Stage 2: two inspections with the image order swapped.
        f1 = pool.submit(agents.inspect, prompt, axes, reqs, parts["A"], parts["B"],
                         desc["A"], desc["B"], 1, usage, mock)
        f2 = pool.submit(agents.inspect, prompt, axes, reqs, parts["B"], parts["A"],
                         desc["B"], desc["A"], 2, usage, mock)
        run1 = judge.map_run(f1.result(), ("A", "B"), run=1)
        run2 = judge.map_run(f2.result(), ("B", "A"), run=2)

    # Stage 3: reconcile (code).
    ratings, agreed, flipped, reasons = judge.reconcile(run1, run2, axes)
    review += [judge.flip_message(axis, run1, run2) for axis in flipped]
    requirement_results = judge.merge_requirements(run1, run2)
    stray = sorted(set(requirement_results) - {r.id for r in reqs.requirements})
    if stray:
        review.append(f"inspection returned results for requirements that are not on the checklist: {stray}")
    findings = [f for f in judge.merge_findings(run1, run2) if f["axis"] in axes]

    # Stage 4: verify deciding findings.
    to_check = judge.select_for_verification(findings, config.MAX_FINDINGS_TO_VERIFY)
    if len(findings) > len(to_check):
        review.append(f"{len(findings) - len(to_check)} findings were not verified (cap reached)")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {i: pool.submit(agents.verify, findings[i], images, parts, usage, mock) for i in to_check}
        for i, fut in futures.items():
            v = fut.result()
            findings[i]["verdict"] = v.verdict
            findings[i]["verifier_note"] = v.note
            if v.verdict == "confirmed":
                findings[i]["observation"] = v.observation

    # Stage 4b: a second opinion on every confirmed artifact. A blurry face or unreadable caption can
    # be a real generative defect, or just out-of-focus detail that any photograph would show.
    to_review = [i for i in to_check if findings[i]["axis"] == "ai_artifacts" and findings[i]["verdict"] == "confirmed"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {i: pool.submit(agents.review_artifact, findings[i], images, parts, prompt, reqs, usage, mock)
                   for i in to_review}
        for i, fut in futures.items():
            rv = fut.result()
            findings[i]["artifact_review"] = rv.verdict
            findings[i]["artifact_note"] = rv.reason

    before = {axis: r.value for axis, r in ratings.items()}
    review += judge.apply_verification(ratings, findings)
    lost_support = [axis for axis, r in ratings.items() if r.value != before[axis]]

    # Stage 5: overall (code).
    ov = judge.overall(ratings, axes)
    if ov.rule_applied == "unresolved":
        review.append("overall unresolved: axis wins split evenly and instruction following is tied")
    for axis, r in ratings.items():
        if r.confidence == "low" and axis not in flipped:
            review.append(f"{axis}: low confidence")

    verified = judge.to_verified(findings)
    reqs_out = [{**r.model_dump(), "A": requirement_results.get(r.id, {}).get("A"),
                 "B": requirement_results.get(r.id, {}).get("B")} for r in reqs.requirements]

    # Stage 6: justification.
    interpretation = reqs.interpretation.model_dump()
    justification = agents.write(prompt, axes, ratings, ov, verified, reqs_out,
                                 judge.tie_notes(ratings, flipped, reasons, lost_support),
                                 interpretation, reasons, usage, mock)

    result = s.Result(
        task_id=task_id,
        instruction_set_version=config.INSTRUCTION_SET_VERSION,
        models={role: f"{p}/{m}@{e}" + (" (mock)" if mock else "") for role, (p, m, e) in config.AGENT_MODELS.items()},
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        prompt=prompt, axes=axes, style_frame=reqs.style_frame, interpretation=interpretation,
        requirements=reqs_out, findings=verified, ratings=ratings, overall=ov,
        justification=justification,
        order_swap={"agreed_axes": agreed, "flipped_axes": flipped},
        axis_reasons=reasons, description_conflicts=judge.merge_conflicts(run1, run2),
        audit=None, needs_human_review=False, review_reasons=[],
        cost={"calls": 0, "input_tokens": 0, "output_tokens": 0},
    )

    # Stage 7: audit (model) + mechanical checks (code).
    audit = agents.audit(json.loads(result.model_dump_json()), usage, mock)
    result.audit = audit
    if not audit.passed:
        review += [f"audit: {p}" for p in audit.problems]
    review += [f"check: {p}" for p in judge.code_checks(result)]

    result.review_reasons = review
    result.needs_human_review = bool(review)
    result.cost = {"calls": usage.calls, "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}
    return result, usage.log


# ------------------------------------------------------------------ batch
def load_tasks(path):
    """Read the tasks file (UTF-8, with or without a BOM). Raises ValueError naming the line at fault."""
    tasks = []
    first_seen = {}
    for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            t = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{n} is not valid JSON ({e})") from e
        if not isinstance(t, dict):
            raise ValueError(f"{path}:{n} must be a JSON object")
        missing = {"task_id", "prompt", "image_a", "image_b"} - t.keys()
        if missing:
            raise ValueError(f"{path}:{n} missing {sorted(missing)}")
        key = str(t["task_id"])
        if key in first_seen:
            raise ValueError(f"{path}:{n} repeats task_id {key!r} (first used on line {first_seen[key]}); "
                             f"the second result would overwrite the first")
        first_seen[key] = n
        tasks.append(t)
    return tasks


def main(argv=None):
    _tolerate_odd_characters()
    ap = argparse.ArgumentParser(description="Automated T2I evaluator (results are labelled automated).")
    ap.add_argument("tasks", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--mock", action="store_true", help="offline dry run with fake model answers")
    ap.add_argument("--only", nargs="*", help="task_ids to run")
    ap.add_argument("--workers", type=int, default=1, help="tasks processed in parallel")
    ap.add_argument("--skip-done", action="store_true", help="skip tasks that already have a result file")
    args = ap.parse_args(argv)

    try:
        tasks = load_tasks(args.tasks)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.only:
        unknown = sorted(set(args.only) - {str(t["task_id"]) for t in tasks})
        if unknown:
            print(f"error: --only names tasks that are not in {args.tasks}: {unknown}", file=sys.stderr)
            return 2
        tasks = [t for t in tasks if str(t["task_id"]) in set(args.only)]
    if args.skip_done:
        tasks = [t for t in tasks if not (args.out / f"{t['task_id']}.json").exists()]

    base = args.tasks.parent
    problems = [f"{t.get('task_id')}: {p}" for t in tasks for p in validate_task(t, base)]
    if problems:
        print("error: nothing was run; fix these first:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)

    def one(task):
        try:
            result, call_log = run_task(task, base, mock=args.mock)
            return task, result, call_log, None
        except RefusalError as e:
            return task, None, [], f"refusal: {e}"
        except Exception as e:  # keep the batch going; record the failure
            return task, None, [], f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"

    counts = {"done": 0, "failed": 0, "flagged": 0}

    def record(task, result, call_log, err):
        if err:
            counts["failed"] += 1
            _append(args.out / "errors.jsonl", {"task_id": task["task_id"], "error": err,
                                                "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            print(f"[fail] {task['task_id']}: {err.splitlines()[0]}", file=sys.stderr)
            return
        counts["done"] += 1
        counts["flagged"] += result.needs_human_review
        _write_text(args.out / f"{result.task_id}.json", result.model_dump_json(indent=2))
        for c in call_log:
            _append(args.out / "calls.jsonl", {"task_id": result.task_id, **c})
        print(f"[ok]   {result.task_id}: overall {result.overall.winner} ({result.overall.strength})"
              f"{'  [review]' if result.needs_human_review else ''}")

    # Results are saved as each task finishes (not in submission order), so an interrupted run
    # keeps everything that completed, and Ctrl+C stops queued tasks from starting.
    pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
    futures = [pool.submit(one, t) for t in tasks]
    pending = set(futures)
    handled = set()
    interrupted = False
    try:
        while pending:
            # A bounded wait, because on Windows an untimed wait can't be interrupted by Ctrl+C.
            finished, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for fut in finished:
                record(*fut.result())
                handled.add(fut)  # after saving, so an interrupt mid-save is picked up again below
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted: queued tasks cancelled; waiting for the ones already running", file=sys.stderr)
        pool.shutdown(wait=True, cancel_futures=True)
        for fut in futures:
            if fut not in handled and not fut.cancelled():
                record(*fut.result())
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    total = rebuild_outputs(args.out)
    print(f"\n{counts['done']} done, {counts['flagged']} flagged for review, {counts['failed']} failed"
          f"{', run interrupted' if interrupted else ''}. {total} results in {args.out}/")
    return 0 if counts["failed"] == 0 and not interrupted else 1


# ------------------------------------------------------------------ outputs
def load_results(out_dir):
    """Every finished result in the output folder, sorted by task_id."""
    results = []
    for path in sorted(out_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("evaluator") == "automated" and "task_id" in data:
            results.append(data)
    return results


def summary_row(r):
    ratings = r["ratings"]
    return [r["task_id"], r["evaluator"], r["overall"]["winner"], r["overall"]["strength"],
            r["overall"]["rule_applied"],
            *[ratings[a]["value"] if a in ratings else "" for a in s.SCORED_AXES],
            ";".join(r["order_swap"]["flipped_axes"]), r["needs_human_review"],
            " | ".join(r["review_reasons"]),
            r["cost"]["calls"], r["cost"]["input_tokens"], r["cost"]["output_tokens"]]


def rebuild_outputs(out_dir):
    """Regenerate results.jsonl and summary.csv from the per-task files.

    Re-running a task replaces its row instead of adding a second one. Returns the result count.
    """
    results = load_results(out_dir)
    _write_text(out_dir / "results.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
    tmp = out_dir / "summary.csv.tmp"
    with tmp.open("w", newline="", encoding="utf-8-sig") as fh:  # BOM so Excel on Windows reads UTF-8
        w = csv.writer(fh)
        w.writerow(SUMMARY_COLUMNS)
        for r in results:
            w.writerow(summary_row(r))
    try:
        os.replace(tmp, out_dir / "summary.csv")
    except PermissionError:  # Windows won't replace a file that is open in Excel
        print(f"warning: summary.csv is open in another program, so it was not updated; "
              f"the new summary is in {tmp.name}", file=sys.stderr)
    return len(results)


def _write_text(path, text):
    """Write through a temp file, so a crash never leaves a half-written file that --skip-done would trust.

    Lines end in \\n on every platform, so the .jsonl files are strict JSON Lines.
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _append(path, obj):
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _tolerate_odd_characters():
    """A console that can't show a character (cp1252 on Windows) should print '?', not crash the run."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


if __name__ == "__main__":
    sys.exit(main())
