"""The report reads real pipeline output and says what the result says."""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orchestrator
import providers
import report

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def make_results(folder):
    out = folder / "out"
    orchestrator.main([str(EXAMPLES / "tasks.jsonl"), "--out", str(out), "--mock"])
    return out


def test_report_covers_every_section(tmp_path):
    out = make_results(tmp_path)
    assert report.main([str(out)]) == 0
    text = (out / "report.md").read_text(encoding="utf-8")
    for needle in ("# T2I evaluation report", "## demo_001", "## demo_002", "### Ratings", "### Why",
                   "### Requirements from the prompt", "### Findings", "Image A wins, strongly.",
                   "| Instruction following | **A**, slightly | high |", "right paw has six digits",
                   "five cards", "Audit: passed"):
        assert needle in text, needle
    assert "Needs human review" not in text


def test_only_limits_the_report_to_chosen_tasks(tmp_path):
    out = make_results(tmp_path)
    report.main([str(out), "--only", "demo_002", "--out", str(tmp_path / "one.md")])
    text = (tmp_path / "one.md").read_text(encoding="utf-8")
    assert "## demo_002" in text and "## demo_001" not in text and "PS2 style" in text


def test_flagged_results_show_why_and_both_inspections(tmp_path, monkeypatch):
    monkeypatch.setattr(providers, "_swap_slots", lambda ins: ins)  # makes the two runs contradict
    out = make_results(tmp_path)
    report.main([str(out)])
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "### Needs human review" in text and "position bias suspected" in text
    assert "### Where the two inspections disagreed" in text and "second inspection (images swapped)" in text
    assert "No winner" in text


def test_table_cells_survive_pipes_and_newlines():
    assert report.cell("a | b\nc") == "a \\| b c"


def test_missing_or_empty_folder_is_an_error(tmp_path, capsys):
    assert report.main([str(tmp_path / "nope")]) == 2
    assert report.main([str(tmp_path)]) == 2
    assert capsys.readouterr().err.count("error:") == 2


def test_non_ascii_text_is_written_as_utf8(tmp_path):
    out = make_results(tmp_path)
    path = out / "demo_001.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["prompt"] = "a café → sign"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    report.main([str(out)])
    assert "a café → sign" in (out / "report.md").read_text(encoding="utf-8")


def test_report_shows_how_the_prompt_was_read_and_the_second_opinion(tmp_path):
    out = make_results(tmp_path)
    report.main([str(out)])
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "### How the prompt was read" in text and "**Deciding question.**" in text
    assert "Second opinion: a real AI defect" in text


def test_report_marks_flaws_dismissed_as_ordinary_detail(tmp_path):
    out = make_results(tmp_path)
    path = out / "demo_001.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["findings"][0].update(artifact_review="natural", artifact_note="out of focus")
    path.write_text(json.dumps(data), encoding="utf-8")
    report.main([str(out), "--only", "demo_001"])
    assert "Second opinion: ordinary photographic detail, not counted (out of focus)" in (
        out / "report.md").read_text(encoding="utf-8")


def test_old_results_without_the_new_fields_still_make_a_report(tmp_path):
    out = make_results(tmp_path)
    path = out / "demo_001.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["interpretation"]
    for f in data["findings"]:
        del f["artifact_review"], f["artifact_note"]
    path.write_text(json.dumps(data), encoding="utf-8")
    assert report.main([str(out), "--only", "demo_001"]) == 0
    assert "How the prompt was read" not in (out / "report.md").read_text(encoding="utf-8")
