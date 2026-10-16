"""Tests for the deterministic rules, including the guide's own examples."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import judge
import schemas as s

AXES3 = ["instruction_following", "visual_quality", "ai_artifacts"]


def R(value, conf="high"):
    return s.AxisRating(value=value, confidence=conf)


def call(winner, strength="slight", conf="high"):
    return {"winner": winner, "strength": strength, "confidence": conf, "reason": "x"}


def run(calls):
    return {"calls": calls, "findings": [], "requirements": {}, "conflicts": []}


# ---------------------------------------------------------------- overall: guide examples
def test_dog_and_cat_example():
    # T2I page: IF tie, VQ A, artifacts A -> Overall A, by count.
    ov = judge.overall({"instruction_following": R("tie"), "visual_quality": R("slightly_A"),
                        "ai_artifacts": R("strongly_A")}, AXES3)
    assert ov.winner == "A" and ov.rule_applied == "count"


def test_dark_gray_cat_count():
    # A black but clean, B gray but warped: B wins IF, A wins VQ + artifacts -> A 2-1.
    ov = judge.overall({"instruction_following": R("strongly_B"), "visual_quality": R("slightly_A"),
                        "ai_artifacts": R("strongly_A")}, AXES3)
    assert ov.winner == "A" and ov.wins == {"A": 2, "B": 1, "tie": 0}
    assert ov.strength == "slightly"


def test_dark_gray_cat_tiebreak():
    # Even everywhere except gray vs black -> IF decides for B.
    ov = judge.overall({"instruction_following": R("slightly_B"), "visual_quality": R("tie"),
                        "ai_artifacts": R("tie")}, AXES3)
    assert ov.winner == "B" and ov.rule_applied == "count"  # 1-0 is already a count win


def test_even_split_uses_tiebreak():
    axes = AXES3 + ["ps2_style"]
    ov = judge.overall({"instruction_following": R("slightly_B"), "visual_quality": R("slightly_A"),
                        "ai_artifacts": R("strongly_A"), "ps2_style": R("slightly_B")}, axes)
    assert ov.winner == "B" and ov.rule_applied == "instruction_following_tiebreak"


def test_split_with_if_tie_is_unresolved():
    axes = AXES3 + ["ps2_style"]
    ov = judge.overall({"instruction_following": R("tie"), "visual_quality": R("slightly_A"),
                        "ai_artifacts": R("slightly_B"), "ps2_style": R("tie")}, axes)
    assert ov.winner == "unresolved"


def test_strongly_when_clean_sweep_with_decisive_axis():
    ov = judge.overall({"instruction_following": R("strongly_A"), "visual_quality": R("slightly_A"),
                        "ai_artifacts": R("tie")}, AXES3)
    assert ov.winner == "A" and ov.strength == "strongly"


# ---------------------------------------------------------------- reconcile
def test_reconcile_agreement_takes_weaker_strength_and_lower_confidence():
    r1 = run({a: call("A", "strong", "high") for a in AXES3})
    r2 = run({a: call("A", "slight", "medium") for a in AXES3})
    ratings, agreed, flipped, _ = judge.reconcile(r1, r2, AXES3)
    assert flipped == [] and set(agreed) == set(AXES3)
    assert ratings["ai_artifacts"].value == "slightly_A"
    assert ratings["ai_artifacts"].confidence == "medium"


def test_reconcile_flip_becomes_low_confidence_tie():
    r1 = run({a: call("A") for a in AXES3})
    r2 = run({**{a: call("A") for a in AXES3}, "visual_quality": call("B")})
    ratings, agreed, flipped, _ = judge.reconcile(r1, r2, AXES3)
    assert flipped == ["visual_quality"]
    assert ratings["visual_quality"].value == "tie" and ratings["visual_quality"].confidence == "low"


def test_map_run_swaps_labels_back():
    ins = s.Inspection(
        requirement_results=[s.RequirementResult(
            id="R1", image_1=s.ReqCheck(status="met", evidence="one"),
            image_2=s.ReqCheck(status="missed", evidence="two"))],
        regions_checked=[], description_conflicts=[],
        findings=[s.Finding(image="1", region="hands", observation="six fingers", axis="ai_artifacts",
                            severity="major", bbox=s.BBox(x=0, y=0, w=1, h=1))],
        calls=s.AxisCalls(**{a: s.AxisCall(winner="2", strength="slight", confidence="high", reason="")
                             for a in s.SCORED_AXES}))
    m = judge.map_run(ins, ("B", "A"), run=2)  # Image 1 was B
    assert m["findings"][0]["image"] == "B"
    assert m["calls"]["ai_artifacts"]["winner"] == "A"
    assert m["requirements"]["R1"]["B"]["status"] == "met"
    assert m["requirements"]["R1"]["A"]["status"] == "missed"


# ---------------------------------------------------------------- findings / verification
def _f(image, region="hands", axis="ai_artifacts", sev="major", run_=1, verdict=None):
    d = {"image": image, "region": region, "observation": "obs", "axis": axis, "severity": sev,
         "bbox": {"x": 0, "y": 0, "w": 1, "h": 1}, "run": run_}
    if verdict:
        d["verdict"] = verdict
    return d


def test_merge_findings_dedupes_across_runs():
    merged = judge.merge_findings({"findings": [_f("B", run_=1)]}, {"findings": [_f("B", run_=2)]})
    assert len(merged) == 1 and merged[0]["seen_in_runs"] == [1, 2]


def test_rejected_major_finding_against_loser_lowers_confidence():
    ratings = {"ai_artifacts": R("strongly_A")}
    f = {**_f("B", verdict="rejected"), "seen_in_runs": [1]}
    reasons = judge.apply_verification(ratings, [f])
    assert ratings["ai_artifacts"].confidence == "low" and reasons


def test_verification_order_puts_major_first():
    fs = [{**_f("A", sev="minor"), "seen_in_runs": [1]}, {**_f("B", sev="major"), "seen_in_runs": [1]}]
    assert judge.select_for_verification(fs, 1) == [1]
