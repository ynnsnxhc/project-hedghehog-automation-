"""End-to-end dry run with the mock provider (no API calls)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orchestrator

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_mock_batch(tmp_path):
    code = orchestrator.main([str(EXAMPLES / "tasks.jsonl"), "--out", str(tmp_path), "--mock"])
    assert code == 0
    lines = (tmp_path / "results.jsonl").read_text().splitlines()
    assert len(lines) == 2
    r = json.loads(lines[0])
    assert r["evaluator"] == "automated"
    assert r["overall"]["winner"] == "A"
    assert r["order_swap"]["flipped_axes"] == []
    assert r["findings"][0]["verdict"] == "confirmed"
    assert (tmp_path / "summary.csv").exists() and (tmp_path / "calls.jsonl").exists()


def test_ps2_task_scores_ps2_axis(tmp_path):
    orchestrator.main([str(EXAMPLES / "tasks.jsonl"), "--out", str(tmp_path), "--mock", "--only", "demo_002"])
    r = json.loads((tmp_path / "demo_002.json").read_text())
    assert "ps2_style" in r["ratings"]
