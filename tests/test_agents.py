"""Tests for the agent call wrapper and what each agent sends."""
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from PIL import Image

import agents
import config
import imaging
import schemas as s
from providers import RefusalError, RetryableError, TruncatedError, Usage


class Scripted:
    """A provider that raises or returns what it is told, one entry per call, and records max_tokens."""

    def __init__(self, script):
        self.script = list(script)
        self.max_tokens_seen = []

    def complete(self, **kwargs):
        self.max_tokens_seen.append(kwargs["max_tokens"])
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


@pytest.fixture
def sleeps(monkeypatch):
    slept = []
    monkeypatch.setattr(agents.time, "sleep", slept.append)
    return slept


def use(monkeypatch, provider):
    monkeypatch.setattr(agents, "make_provider", lambda *a, **k: provider)
    return provider


# ---------------------------------------------------------------- Usage
class SpyLock:
    def __init__(self):
        self.entries = 0

    def __enter__(self):
        self.entries += 1

    def __exit__(self, *exc):
        return False


def test_usage_updates_take_the_lock():
    usage = Usage()
    usage._lock = SpyLock()
    usage.add("a", "m", 1, 2, 0.0)
    assert usage._lock.entries == 1


def test_usage_counts_stay_exact_across_threads():
    usage = Usage()

    def work():
        for _ in range(2000):
            usage.add("a", "m", 1, 2, 0.0)

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # switch threads constantly, so an unlocked += would lose counts
    try:
        threads = [threading.Thread(target=work) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(old)
    assert (usage.calls, usage.input_tokens, usage.output_tokens, len(usage.log)) == (16000, 16000, 32000, 16000)


# ---------------------------------------------------------------- _call
def test_truncated_call_is_retried_with_more_room_and_no_wait(monkeypatch, sleeps):
    p = use(monkeypatch, Scripted([TruncatedError("cut"), TruncatedError("cut"), "answer"]))
    assert agents._call("x", "inspector", [], None, Usage(), False) == "answer"
    base = config.MAX_TOKENS["inspector"]
    assert p.max_tokens_seen[0] == base
    assert p.max_tokens_seen[1] == min(int(base * config.MAX_TOKENS_GROWTH), config.MAX_TOKENS_CEILING)
    assert p.max_tokens_seen[2] >= p.max_tokens_seen[1] > base
    assert max(p.max_tokens_seen) <= config.MAX_TOKENS_CEILING
    assert sleeps == []


def test_token_growth_stops_at_the_ceiling(monkeypatch, sleeps):
    monkeypatch.setitem(config.MAX_TOKENS, "writer", 80)
    monkeypatch.setattr(config, "MAX_TOKENS_CEILING", 100)
    monkeypatch.setattr(config, "MAX_RETRIES", 4)
    p = use(monkeypatch, Scripted([TruncatedError("cut")] * 3 + ["answer"]))
    agents._call("x", "writer", [], None, Usage(), False)
    assert p.max_tokens_seen == [80, 100, 100, 100]


def test_a_ceiling_below_the_configured_limit_never_lowers_it(monkeypatch, sleeps):
    monkeypatch.setitem(config.MAX_TOKENS, "writer", 500)
    monkeypatch.setattr(config, "MAX_TOKENS_CEILING", 100)
    p = use(monkeypatch, Scripted([TruncatedError("cut"), "answer"]))
    agents._call("x", "writer", [], None, Usage(), False)
    assert p.max_tokens_seen == [500, 500]


def test_other_transient_errors_back_off_and_keep_the_same_limit(monkeypatch, sleeps):
    p = use(monkeypatch, Scripted([RetryableError("429"), RetryableError("529"), "answer"]))
    assert agents._call("x", "writer", [], None, Usage(), False) == "answer"
    assert sleeps == [config.RETRY_BACKOFF_SECONDS, config.RETRY_BACKOFF_SECONDS * 2]
    assert len(set(p.max_tokens_seen)) == 1


def test_gives_up_after_the_retry_limit(monkeypatch, sleeps):
    use(monkeypatch, Scripted([RetryableError("down")] * config.MAX_RETRIES))
    with pytest.raises(RuntimeError, match="failed after"):
        agents._call("x", "writer", [], None, Usage(), False)


def test_a_refusal_is_not_retried(monkeypatch, sleeps):
    p = use(monkeypatch, Scripted([RefusalError("no"), "never reached"]))
    with pytest.raises(RefusalError):
        agents._call("x", "writer", [], None, Usage(), False)
    assert len(p.max_tokens_seen) == 1


# ---------------------------------------------------------------- what each agent sends
def sent(monkeypatch, result=None):
    """Replace agents._call and capture the parts an agent builds."""
    captured = {}

    def fake(agent, role, parts, output_model, usage, mock):
        captured.update(agent=agent, role=role, parts=parts)
        return result

    monkeypatch.setattr(agents, "_call", fake)
    return captured


def interpretation(**over):
    fields = dict(intent="a candid photo of a man at a kitchen table", implied_look="candid phone-snapshot realism",
                  acceptable_imperfections=["mild softness", "blown-out window"],
                  deciding_question="Does it look unposed rather than styled?")
    fields.update(over)
    return s.Interpretation(**fields)


def test_ps2_axis_forces_the_ps2_style_frame(monkeypatch):
    sent(monkeypatch, s.RequirementList(style_frame="photographic", interpretation=interpretation(), requirements=[]))
    out = agents.requirements("a badger", ["instruction_following", "ps2_style"], Usage())
    assert out.style_frame == "ps2"


def test_other_tasks_keep_the_style_frame_the_model_chose(monkeypatch):
    sent(monkeypatch, s.RequirementList(style_frame="non_photo", interpretation=interpretation(), requirements=[]))
    assert agents.requirements("a badger", ["instruction_following"], Usage()).style_frame == "non_photo"


def _images():
    images = {"A": Image.new("RGB", (400, 400), "white"), "B": Image.new("RGB", (400, 400), "black")}
    return images, {k: imaging.encode(v) for k, v in images.items()}


def _finding(bbox):
    return {"image": "B", "region": "hands", "observation": "six fingers", "bbox": bbox}


def _image_parts(captured):
    return [p for p in captured["parts"] if p["type"] == "image"]


def test_verifier_gets_full_image_crop_and_comparison_crop(monkeypatch):
    captured = sent(monkeypatch)
    images, parts = _images()
    agents.verify(_finding({"x": 0.4, "y": 0.4, "w": 0.2, "h": 0.2}), images, parts, Usage())
    assert len(_image_parts(captured)) == 3
    assert _image_parts(captured)[0] is parts["B"]  # the exact bytes the Inspector saw


def test_verifier_falls_back_to_the_full_image_when_the_box_is_unusable(monkeypatch):
    captured = sent(monkeypatch)
    images, parts = _images()
    agents.verify(_finding({"x": 120, "y": 80, "w": 60, "h": 40}), images, parts, Usage())
    assert len(_image_parts(captured)) == 1
    assert any("No usable crop" in p.get("text", "") for p in captured["parts"])


def test_writer_payload_has_checklist_text_tie_reasons_and_no_boxes(monkeypatch):
    captured = sent(monkeypatch)
    finding = s.VerifiedFinding(image="B", region="text", bbox=s.BBox(x=0.1, y=0.1, w=0.2, h=0.1),
                                observation="sign reads CAFÉ", axis="ai_artifacts", severity="minor",
                                verdict="confirmed", seen_in_runs=[1, 2])
    rejected = finding.model_copy(update={"verdict": "rejected", "observation": "not real"})
    requirements = [{"id": "R1", "text": "a sign that says CAFÉ", "type": "text", "central": True,
                     "A": {"status": "met", "evidence": "x"}, "B": {"status": "met", "evidence": "x"}}]
    tied = {"visual_quality": {"why": "both inspections found no meaningful difference", "inspector_reasons": ["a", "b"]}}
    ratings = {"visual_quality": s.AxisRating(value="tie", confidence="high")}
    overall = s.Overall(winner="unresolved", strength="none", wins={"A": 0, "B": 0, "tie": 1},
                        rule_applied="unresolved", strength_basis="x")
    natural = finding.model_copy(update={"observation": "blurry fridge photos", "artifact_review": "natural",
                                         "artifact_note": "out of focus behind the subject"})
    agents.write("a CAFÉ sign", ["visual_quality"], ratings, overall, [finding, rejected, natural], requirements,
                 tied, interpretation().model_dump(), {"visual_quality": ["A is sharper", "A is sharper"]}, Usage())

    text = captured["parts"][0]["text"]
    assert "CAFÉ" in text  # not escaped to É
    payload = json.loads(text.split("\n\n", 1)[1])
    assert payload["requirements"][0]["text"] == "a sign that says CAFÉ"
    assert payload["tied_axes"]["visual_quality"]["inspector_reasons"] == ["a", "b"]
    assert payload["interpretation"]["implied_look"] == "candid phone-snapshot realism"
    assert payload["inspection_reasons"] == {"visual_quality": ["A is sharper", "A is sharper"]}
    assert [f["observation"] for f in payload["verified_findings"]] == ["sign reads CAFÉ"]
    assert "bbox" not in payload["verified_findings"][0] and "seen_in_runs" not in payload["verified_findings"][0]
    dismissed = payload["reviewed_and_dismissed_as_ordinary_photographic_detail"]
    assert dismissed == [{"image": "B", "observation": "blurry fridge photos", "why": "out of focus behind the subject"}]


def test_auditor_sees_the_order_swap_record_and_verdicts_but_not_boxes(monkeypatch):
    captured = sent(monkeypatch)
    result = {"prompt": "p", "interpretation": {"intent": "x"}, "axes": [], "ratings": {}, "overall": {},
              "requirements": [], "justification": {}, "axis_reasons": {"visual_quality": ["a", "b"]},
              "order_swap": {"agreed_axes": [], "flipped_axes": ["visual_quality"]},
              "findings": [{"image": "A", "bbox": {"x": 0}, "seen_in_runs": [1], "verdict": "rejected"}],
              "cost": {"calls": 3}}
    agents.audit(result, Usage())
    payload = json.loads(captured["parts"][0]["text"].split("\n\n", 1)[1])
    assert payload["order_swap"]["flipped_axes"] == ["visual_quality"]
    assert payload["axis_reasons"] == {"visual_quality": ["a", "b"]}
    assert payload["findings"] == [{"image": "A", "verdict": "rejected"}]
    assert "cost" not in payload


def test_artifact_reviewer_gets_the_images_the_prompt_and_how_it_was_read(monkeypatch):
    captured = sent(monkeypatch)
    images, parts = _images()
    reqs = s.RequirementList(style_frame="photographic", interpretation=interpretation(), requirements=[])
    finding = {**_finding({"x": 0.4, "y": 0.4, "w": 0.2, "h": 0.2}), "observation": "blurry faces on the fridge"}
    agents.review_artifact(finding, images, parts, "an unstaged photo of a man", reqs, Usage())
    assert captured["role"] == "artifact_reviewer"
    assert len(_image_parts(captured)) == 3
    text = captured["parts"][-1]["text"]
    assert "an unstaged photo of a man" in text and "candid phone-snapshot realism" in text
    assert "blown-out window" in text and "blurry faces on the fridge" in text


def test_inspector_is_given_the_interpretation_apart_from_the_checklist(monkeypatch):
    captured = sent(monkeypatch)
    reqs = s.RequirementList(style_frame="photographic", interpretation=interpretation(),
                             requirements=[s.Requirement(id="R1", text="a man", type="subject", central=True)])
    desc = s.Description(subjects=[], actions=[], relationships=[], text=[], setting="", lighting="", style="")
    agents.inspect("p", ["instruction_following"], reqs, {"type": "image"}, {"type": "image"}, desc, desc, 1, Usage())
    text = captured["parts"][-1]["text"]
    assert text.count("Does it look unposed rather than styled?") == 1  # once, not repeated in the checklist
    assert "Interpretation of the prompt" in text and '"id": "R1"' in text
