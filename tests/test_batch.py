"""Batch behaviour with the mock provider: input checks, reruns, resume, encoding, interruption."""
import csv
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orchestrator
import providers

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

# Mock task: 2 describers, requirements, 2 inspectors, 1 verifier, 1 artifact review, writer, auditor.
CALLS_PER_TASK = 9


def task(task_id="t1", **over):
    base = {"task_id": task_id, "prompt": "a kneeling badger fanning five cards",
            "image_a": "badger_a.png", "image_b": "badger_b.png"}
    base.update(over)
    return base


def write_tasks(folder, tasks, bom=False):
    """A tasks file next to copies of the example images. UTF-8, optionally with a BOM like Notepad writes."""
    for name in ("badger_a.png", "badger_b.png"):
        shutil.copy(EXAMPLES / name, folder / name)
    text = "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks)
    path = folder / "tasks.jsonl"
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8"))
    return path


def run(tasks_path, out, *extra):
    return orchestrator.main([str(tasks_path), "--out", str(out), "--mock", *extra])


def lines(path):
    return path.read_text(encoding="utf-8").splitlines()


# ---------------------------------------------------------------- clean mock run
def test_mock_run_raises_no_review_flags(tmp_path):
    out = tmp_path / "out"
    assert orchestrator.main([str(EXAMPLES / "tasks.jsonl"), "--out", str(out), "--mock"]) == 0
    for line in lines(out / "results.jsonl"):
        r = json.loads(line)
        assert r["needs_human_review"] is False, r["review_reasons"]
        assert r["audit"]["passed"] is True


def test_ps2_task_uses_the_ps2_style_frame(tmp_path):
    out = tmp_path / "out"
    orchestrator.main([str(EXAMPLES / "tasks.jsonl"), "--out", str(out), "--mock"])
    assert json.loads((out / "demo_001.json").read_text(encoding="utf-8"))["style_frame"] == "photographic"
    assert json.loads((out / "demo_002.json").read_text(encoding="utf-8"))["style_frame"] == "ps2"


def test_result_keeps_the_evidence_a_reviewer_needs(tmp_path):
    out = tmp_path / "out"
    orchestrator.main([str(EXAMPLES / "tasks.jsonl"), "--out", str(out), "--mock", "--only", "demo_001"])
    r = json.loads((out / "demo_001.json").read_text(encoding="utf-8"))
    f = r["findings"][0]
    assert f["bbox"] == {"x": 0.4, "y": 0.4, "w": 0.2, "h": 0.2}
    assert f["verifier_note"] == "clear in crop"
    assert set(r["axis_reasons"]) == {"instruction_following", "visual_quality", "ai_artifacts"}
    assert "description_conflicts" in r


# ---------------------------------------------------------------- reruns and resume
def test_rerunning_a_task_replaces_its_rows_instead_of_duplicating_them(tmp_path):
    tasks = write_tasks(tmp_path, [task("a"), task("b")])
    out = tmp_path / "out"
    assert run(tasks, out) == 0
    assert run(tasks, out) == 0
    assert len(lines(out / "results.jsonl")) == 2
    with (out / "summary.csv").open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["task_id"] for r in rows] == ["a", "b"]
    assert len(lines(out / "calls.jsonl")) == 2 * 2 * CALLS_PER_TASK  # the spend log keeps every call from both runs


def test_skip_done_only_runs_what_is_missing(tmp_path):
    out = tmp_path / "out"
    run(write_tasks(tmp_path, [task("a")]), out)
    tasks = write_tasks(tmp_path, [task("a"), task("b")])
    assert run(tasks, out, "--skip-done") == 0
    assert len(lines(out / "calls.jsonl")) == 2 * CALLS_PER_TASK  # a once, b once
    assert [json.loads(x)["task_id"] for x in lines(out / "results.jsonl")] == ["a", "b"]


def test_no_temp_files_are_left_behind(tmp_path):
    out = tmp_path / "out"
    run(write_tasks(tmp_path, [task("a")]), out)
    assert not list(out.glob("*.tmp"))


def test_json_files_use_plain_newlines_on_every_platform(tmp_path):
    out = tmp_path / "out"
    run(write_tasks(tmp_path, [task("a")]), out)
    for name in ("a.json", "results.jsonl", "calls.jsonl"):
        assert b"\r\n" not in (out / name).read_bytes(), name


def test_a_summary_open_in_excel_does_not_crash_a_finished_run(tmp_path, monkeypatch, capsys):
    real_replace = orchestrator.os.replace

    def locked(src, dst):
        if Path(dst).name == "summary.csv":
            raise PermissionError("in use")
        return real_replace(src, dst)

    monkeypatch.setattr(orchestrator.os, "replace", locked)
    out = tmp_path / "out"
    assert run(write_tasks(tmp_path, [task("a")]), out) == 0
    assert "summary.csv is open in another program" in capsys.readouterr().err
    assert (out / "summary.csv.tmp").exists()
    assert (out / "a.json").exists() and len(lines(out / "results.jsonl")) == 1


def test_failed_tasks_are_logged_and_the_batch_continues(tmp_path, monkeypatch):
    real = orchestrator.run_task

    def flaky(t, base, mock=False):
        if t["task_id"] == "bad":
            raise RuntimeError("model fell over")
        return real(t, base, mock=mock)

    monkeypatch.setattr(orchestrator, "run_task", flaky)
    out = tmp_path / "out"
    assert run(write_tasks(tmp_path, [task("bad"), task("good")]), out) == 1
    errors = [json.loads(x) for x in lines(out / "errors.jsonl")]
    assert errors[0]["task_id"] == "bad" and "model fell over" in errors[0]["error"] and errors[0]["at"]
    assert [json.loads(x)["task_id"] for x in lines(out / "results.jsonl")] == ["good"]


# ---------------------------------------------------------------- input checks
def test_a_bad_batch_stops_before_any_work_and_lists_every_problem(tmp_path, capsys):
    tasks = write_tasks(tmp_path, [task("ok"), task("../escape"), task("axes", axes=["style"]),
                                   task("noimg", image_b="missing.png"), task("noif", axes=["visual_quality"]),
                                   task("twice", axes=["instruction_following", "instruction_following"]),
                                   task("empty", prompt="  ")])
    out = tmp_path / "out"
    assert run(tasks, out) == 2
    err = capsys.readouterr().err
    for needle in ("../escape", "unknown axes", "missing.png", "instruction_following must be scored",
                   "same axis twice", "prompt is empty"):
        assert needle in err
    assert not out.exists()


def test_repeated_task_ids_are_rejected(tmp_path, capsys):
    assert run(write_tasks(tmp_path, [task("a"), task("a")]), tmp_path / "out") == 2
    assert "repeats task_id 'a'" in capsys.readouterr().err


def test_invalid_json_names_the_line(tmp_path, capsys):
    tasks = write_tasks(tmp_path, [task("a")])
    with tasks.open("a", encoding="utf-8") as fh:
        fh.write("{not json}\n")
    assert run(tasks, tmp_path / "out") == 2
    assert "tasks.jsonl:2 is not valid JSON" in capsys.readouterr().err


def test_a_missing_tasks_file_is_an_error_not_a_traceback(tmp_path, capsys):
    assert run(tmp_path / "nope.jsonl", tmp_path / "out") == 2
    assert capsys.readouterr().err.startswith("error:")


def test_only_with_an_unknown_id_is_an_error(tmp_path, capsys):
    assert run(write_tasks(tmp_path, [task("a")]), tmp_path / "out", "--only", "typo") == 2
    assert "typo" in capsys.readouterr().err


# ---------------------------------------------------------------- text encoding
def test_non_ascii_prompts_survive_a_bom_and_every_output(tmp_path):
    prompt = "a café sign reading “CLOSED” → ☕ and 中文"
    tasks = write_tasks(tmp_path, [task("a", prompt=prompt)], bom=True)
    assert orchestrator.load_tasks(tasks)[0]["prompt"] == prompt
    out = tmp_path / "out"
    assert run(tasks, out) == 0
    assert json.loads((out / "a.json").read_text(encoding="utf-8"))["prompt"] == prompt
    assert json.loads(lines(out / "results.jsonl")[0])["prompt"] == prompt


def test_prompt_files_are_read_as_utf8():
    import agents
    text = agents._prompt("inspector")
    assert " · " in text and "Â·" not in text  # the region list's middle dots, not mojibake


# ---------------------------------------------------------------- the order swap, end to end
def test_disagreeing_inspections_become_a_flagged_low_confidence_tie(tmp_path, monkeypatch):
    # With the mock's "swap the answer for run 2" step removed, the two runs contradict each other.
    monkeypatch.setattr(providers, "_swap_slots", lambda ins: ins)
    out = tmp_path / "out"
    assert run(write_tasks(tmp_path, [task("a")]), out) == 0
    r = json.loads((out / "a.json").read_text(encoding="utf-8"))
    assert set(r["order_swap"]["flipped_axes"]) == {"instruction_following", "ai_artifacts"}
    for axis in r["order_swap"]["flipped_axes"]:
        assert r["ratings"][axis] == {"value": "tie", "confidence": "low"}
        assert len(r["axis_reasons"][axis]) == 2
    assert r["needs_human_review"] is True
    assert any("position bias suspected" in x for x in r["review_reasons"])
    assert r["overall"]["winner"] == "unresolved"


# ---------------------------------------------------------------- interruption
def test_ctrl_c_stops_queued_tasks_but_keeps_what_finished(tmp_path, monkeypatch, capsys):
    real = orchestrator.run_task

    def slow(t, base, mock=False):
        time.sleep(0.4)
        return real(t, base, mock=mock)

    def interrupt(pending, timeout=None, return_when=None):
        time.sleep(0.1)  # let the first task start
        raise KeyboardInterrupt

    monkeypatch.setattr(orchestrator, "run_task", slow)
    monkeypatch.setattr(orchestrator, "wait", interrupt)
    out = tmp_path / "out"
    assert run(write_tasks(tmp_path, [task("a"), task("b"), task("c")]), out) == 1
    assert "interrupted" in capsys.readouterr().err
    done = sorted(p.stem for p in out.glob("*.json"))
    assert done == ["a"]  # the running task finished and was saved; b and c never started
    assert len(lines(out / "results.jsonl")) == 1


# ---------------------------------------------------------------- how the prompt was read, and the artifact second opinion
def test_result_records_how_the_prompt_was_read_and_the_artifact_review(tmp_path):
    out = tmp_path / "out"
    assert run(write_tasks(tmp_path, [task("a")]), out) == 0
    r = json.loads((out / "a.json").read_text(encoding="utf-8"))
    assert r["interpretation"]["deciding_question"] and r["interpretation"]["acceptable_imperfections"]
    f = r["findings"][0]
    assert f["artifact_review"] == "ai_defect" and f["artifact_note"]
    assert r["ratings"]["ai_artifacts"]["value"] == "strongly_A"  # a real defect keeps the win


def test_artifact_judged_natural_removes_the_win_and_flags_the_result(tmp_path, monkeypatch):
    import agents
    import schemas

    def dismiss(finding, images, parts, prompt, reqs, usage, mock=False):
        return schemas.ArtifactReview(verdict="natural", reason="out of focus, like the rest of the background")

    seen = {}
    real_write = agents.write

    def spy(*args, **kwargs):
        seen["tied_axes"], seen["payload_findings"] = args[6], args[4]
        return real_write(*args, **kwargs)

    monkeypatch.setattr(agents, "review_artifact", dismiss)
    monkeypatch.setattr(agents, "write", spy)
    out = tmp_path / "out"
    assert run(write_tasks(tmp_path, [task("a")]), out) == 0
    r = json.loads((out / "a.json").read_text(encoding="utf-8"))

    assert r["ratings"]["ai_artifacts"] == {"value": "tie", "confidence": "low"}
    assert r["findings"][0]["artifact_review"] == "natural"
    assert r["needs_human_review"] is True
    assert any("lost its support" in x for x in r["review_reasons"])
    assert "ordinary photographic detail" in seen["tied_axes"]["ai_artifacts"]["why"]
    # instruction following still decides the overall result, and the count rule still applies
    assert r["overall"]["winner"] == "A" and r["overall"]["rule_applied"] == "count"
