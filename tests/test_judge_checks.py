"""Tests for finding merge, label translation, tie notes and the consistency checks."""
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


# ---------------------------------------------------------------- Image 1/2 -> A/B in free text
def test_relabel_rewrites_image_numbers_to_real_labels():
    assert judge.relabel("Image 2 is sharper than image 1.", ("B", "A")) == "Image A is sharper than image B."
    assert judge.relabel("image_1: hands", ("A", "B")) == "image A: hands"


def test_relabel_leaves_other_text_alone():
    assert judge.relabel("Image 12 and imagery and image 3", ("A", "B")) == "Image 12 and imagery and image 3"


def test_map_run_relabels_every_free_text_field():
    ins = s.Inspection(
        requirement_results=[s.RequirementResult(
            id="R1", image_1=s.ReqCheck(status="met", evidence="Image 1 has five cards"),
            image_2=s.ReqCheck(status="missed", evidence="Image 2 has three"))],
        regions_checked=[], description_conflicts=["Blind description of Image 1 said four cards"],
        findings=[s.Finding(image="1", region="hands", observation="Image 1's right paw has six digits",
                            axis="ai_artifacts", severity="major", bbox=s.BBox(x=0, y=0, w=1, h=1))],
        calls=s.AxisCalls(**{a: s.AxisCall(winner="2", strength="slight", confidence="high",
                                           reason="Image 2 is better") for a in s.SCORED_AXES}))
    m = judge.map_run(ins, ("B", "A"), run=2)  # in run 2, Image 1 was B and Image 2 was A
    assert m["findings"][0]["observation"] == "Image B's right paw has six digits"
    assert m["calls"]["visual_quality"]["reason"] == "Image A is better"
    assert m["requirements"]["R1"]["B"]["evidence"] == "Image B has five cards"
    assert m["requirements"]["R1"]["A"]["evidence"] == "Image A has three"
    assert m["conflicts"] == ["Blind description of Image B said four cards"]


# ---------------------------------------------------------------- merging findings
def _f(image, x=0.4, y=0.4, obs="obs", run_=1, sev="major", region="hands", axis="ai_artifacts"):
    return {"image": image, "region": region, "observation": obs, "axis": axis, "severity": sev,
            "bbox": {"x": x, "y": y, "w": 0.1, "h": 0.1}, "run": run_}


def _merge(first, second=()):
    return judge.merge_findings({"findings": list(first)}, {"findings": list(second)})


def test_two_different_defects_in_one_region_stay_separate():
    merged = _merge([_f("B", 0.1, 0.5, "left hand fused"), _f("B", 0.7, 0.5, "right hand six fingers")])
    assert [m["observation"] for m in merged] == ["left hand fused", "right hand six fingers"]


def test_same_defect_in_both_runs_merges_at_the_higher_severity():
    merged = _merge([_f("B", 0.40, 0.40, "six fingers", run_=1, sev="minor")],
                    [_f("B", 0.42, 0.41, "extra finger on right paw", run_=2, sev="major")])
    assert len(merged) == 1
    assert merged[0]["severity"] == "major"
    assert merged[0]["observation"] == "extra finger on right paw"
    assert merged[0]["seen_in_runs"] == [1, 2]


def test_same_wording_in_one_run_is_two_defects():
    merged = _merge([_f("B", 0.1, 0.5, "six fingers"), _f("B", 0.7, 0.5, "six fingers")])
    assert len(merged) == 2


def test_same_wording_from_the_other_run_merges_even_when_boxes_differ():
    merged = _merge([_f("B", 0.1, 0.5, "six fingers", run_=1)], [_f("B", 0.6, 0.5, "Six fingers", run_=2)])
    assert len(merged) == 1 and merged[0]["seen_in_runs"] == [1, 2]


def test_different_image_or_axis_never_merges():
    merged = _merge([_f("A"), _f("B"), _f("B", axis="instruction_following")])
    assert len(merged) == 3


# ---------------------------------------------------------------- reconcile messages and tie notes
def test_flip_message_tells_a_true_flip_from_tie_vs_winner():
    r1 = run({a: call("A") for a in AXES3})
    r2 = run({"instruction_following": call("A"), "visual_quality": call("B"), "ai_artifacts": call("tie", "none")})
    assert "position bias" in judge.flip_message("visual_quality", r1, r2)
    assert "position bias" not in judge.flip_message("ai_artifacts", r1, r2)


def test_tie_notes_explain_each_tied_axis():
    ratings = {"instruction_following": R("slightly_A"), "visual_quality": R("tie"), "ai_artifacts": R("tie", "low")}
    notes = judge.tie_notes(ratings, ["ai_artifacts"], {a: ["r1", "r2"] for a in AXES3})
    assert set(notes) == {"visual_quality", "ai_artifacts"}
    assert "disagreed" in notes["ai_artifacts"]["why"]
    assert "no meaningful difference" in notes["visual_quality"]["why"]
    assert notes["visual_quality"]["inspector_reasons"] == ["r1", "r2"]


def test_merge_conflicts_drops_blanks_and_repeats():
    r1 = {"conflicts": ["count differs", "None"]}
    r2 = {"conflicts": ["count differs", " ", "text differs"]}
    assert judge.merge_conflicts(r1, r2) == ["count differs", "text differs"]


# ---------------------------------------------------------------- consistency checks
def _req(rid, central, a, b):
    return {"id": rid, "text": rid, "type": "subject", "central": central,
            "A": {"status": a, "evidence": "x"}, "B": {"status": b, "evidence": "x"}}


def test_instruction_following_flagged_when_central_results_point_the_other_way():
    reqs = [_req("R1", True, "missed", "met"), _req("R2", False, "met", "missed")]
    out = judge.check_instruction_following({"instruction_following": R("slightly_A")}, reqs, [])
    assert len(out) == 1 and "favors A" in out[0] and "B 1/1" in out[0]


def test_instruction_following_accepts_a_rating_that_matches_the_checklist():
    reqs = [_req("R1", True, "missed", "met")]
    assert judge.check_instruction_following({"instruction_following": R("strongly_B")}, reqs, []) == []


def test_instruction_following_ignores_secondary_requirements():
    reqs = [_req("R1", True, "met", "met"), _req("R2", False, "missed", "met")]
    assert judge.check_instruction_following({"instruction_following": R("slightly_A")}, reqs, []) == []


def test_instruction_following_tie_is_flagged_unless_the_runs_disagreed():
    reqs = [_req("R1", True, "met", "missed")]
    ratings = {"instruction_following": R("tie")}
    assert judge.check_instruction_following(ratings, reqs, [])
    assert judge.check_instruction_following(ratings, reqs, ["instruction_following"]) == []


def test_requirements_that_could_not_be_checked_are_skipped():
    reqs = [_req("R1", True, "unverifiable", "met")]
    assert judge.check_instruction_following({"instruction_following": R("slightly_A")}, reqs, []) == []


def _vf(image, axis="ai_artifacts", sev="major", verdict="confirmed"):
    return s.VerifiedFinding(image=image, region="hands", bbox=s.BBox(x=0, y=0, w=1, h=1), observation="o",
                             axis=axis, severity=sev, verdict=verdict, seen_in_runs=[1])


def test_findings_check_flags_a_winner_with_worse_defects():
    out = judge.check_findings({"ai_artifacts": R("strongly_A")}, [_vf("A"), _vf("B", sev="minor")], [])
    assert len(out) == 1 and "favors A" in out[0]


def test_findings_check_puts_severity_before_count():
    # B has one major; A has two minors. A is still the better image on artifacts.
    findings = [_vf("B"), _vf("A", sev="minor"), _vf("A", sev="minor")]
    assert judge.check_findings({"ai_artifacts": R("slightly_A")}, findings, []) == []


def test_findings_check_ignores_rejected_and_unverifiable_findings():
    findings = [_vf("A", verdict="rejected"), _vf("A", verdict="unverifiable"), _vf("B")]
    assert judge.check_findings({"ai_artifacts": R("strongly_A")}, findings, []) == []


def test_findings_check_tie_flagged_only_when_majors_differ_and_runs_agreed():
    ratings = {"ai_artifacts": R("tie")}
    assert judge.check_findings(ratings, [_vf("B")], [])
    assert judge.check_findings(ratings, [_vf("B")], ["ai_artifacts"]) == []
    assert judge.check_findings(ratings, [_vf("B", sev="minor")], []) == []


def test_checklist_problems():
    assert judge.check_checklist([]) == ["requirement checklist is empty"]
    twice = judge.check_checklist([_req("R1", False, "met", "met"), _req("R1", False, "met", "met")])
    assert any("duplicate" in p for p in twice) and any("central" in p for p in twice)
    unchecked = judge.check_checklist([{**_req("R2", True, "met", "met"), "B": None}])
    assert any("R2" in p for p in unchecked)


# ---------------------------------------------------------------- code_checks on a whole result
def make_result(**over):
    ratings = {"instruction_following": R("slightly_A"), "visual_quality": R("tie"), "ai_artifacts": R("strongly_A")}
    fields = dict(
        task_id="t", instruction_set_version="v", models={}, created_at="now", prompt="p", axes=AXES3,
        style_frame="photographic", requirements=[_req("R1", True, "met", "missed")],
        findings=[_vf("B")], ratings=ratings, overall=judge.overall(ratings, AXES3),
        justification=s.Justification(instruction_following="x", visual_quality="x", ai_artifacts="x",
                                      ps2_style="", overall="x"),
        order_swap={"agreed_axes": AXES3, "flipped_axes": []}, audit=None,
        needs_human_review=False, review_reasons=[], cost={})
    fields.update(over)
    return s.Result(**fields)


def test_code_checks_pass_a_consistent_result():
    assert judge.code_checks(make_result()) == []


def test_code_checks_catch_empty_justification_fields():
    j = s.Justification(instruction_following="x", visual_quality="x", ai_artifacts=" ", ps2_style="", overall="")
    out = judge.code_checks(make_result(justification=j))
    assert "justification missing for ai_artifacts" in out
    assert "justification missing overall explanation" in out


def test_code_checks_catch_a_ps2_justification_for_an_unscored_axis():
    j = s.Justification(instruction_following="x", visual_quality="x", ai_artifacts="x", ps2_style="y", overall="x")
    assert "PS2 justification written but PS2 is not scored" in judge.code_checks(make_result(justification=j))


def test_code_checks_catch_a_wrong_overall():
    wrong = s.Overall(winner="B", strength="slightly", wins={"A": 2, "B": 0, "tie": 1}, rule_applied="count",
                      strength_basis="x")
    assert "overall winner does not match the count rule" in judge.code_checks(make_result(overall=wrong))


def test_code_checks_catch_a_wrong_overall_strength():
    ratings = {"instruction_following": R("slightly_A"), "visual_quality": R("tie"), "ai_artifacts": R("strongly_A")}
    wrong = judge.overall(ratings, AXES3).model_copy(update={"strength": "slightly"})
    assert "overall strength does not match the count rule" in judge.code_checks(make_result(overall=wrong))


def test_code_checks_pass_requirement_and_finding_contradictions_to_review():
    ratings = {"instruction_following": R("slightly_A"), "visual_quality": R("tie"), "ai_artifacts": R("strongly_A")}
    out = judge.code_checks(make_result(requirements=[_req("R1", True, "missed", "met")],
                                        findings=[_vf("A")], ratings=ratings))
    assert any(p.startswith("instruction_following favors A") for p in out)
    assert any(p.startswith("ai_artifacts favors A") for p in out)


# ---------------------------------------------------------------- findings that did not hold up
def _d(image, axis="ai_artifacts", sev="minor", verdict="confirmed", review=""):
    """A merged finding as the pipeline holds it before the result is built."""
    return {"image": image, "axis": axis, "severity": sev, "observation": f"{image} flaw", "verdict": verdict,
            "artifact_review": review}


def test_artifact_win_with_every_flaw_judged_natural_becomes_a_low_confidence_tie():
    ratings = {"ai_artifacts": R("slightly_B", "medium")}
    reasons = judge.apply_verification(ratings, [_d("A", review="natural"), _d("A", review="natural")])
    assert ratings["ai_artifacts"] == s.AxisRating(value="tie", confidence="low")
    assert len(reasons) == 1 and "lost its support" in reasons[0]


def test_artifact_win_survives_while_one_real_flaw_remains():
    ratings = {"ai_artifacts": R("slightly_B", "medium")}
    reasons = judge.apply_verification(ratings, [_d("A", review="natural"), _d("A", review="ai_defect")])
    assert ratings["ai_artifacts"] == R("slightly_B", "medium") and reasons == []


def test_artifact_win_also_loses_support_when_the_verifier_rejected_every_flaw():
    ratings = {"ai_artifacts": R("strongly_A")}
    judge.apply_verification(ratings, [_d("B", verdict="rejected")])
    assert ratings["ai_artifacts"].value == "tie"


def test_unverifiable_or_unclear_flaws_keep_the_win():
    ratings = {"ai_artifacts": R("slightly_A")}
    judge.apply_verification(ratings, [_d("B", verdict="unverifiable"), _d("B", review="unclear")])
    assert ratings["ai_artifacts"].value == "slightly_A"


def test_a_win_with_no_findings_at_all_is_left_to_the_other_checks():
    ratings = {"ai_artifacts": R("slightly_A")}
    assert judge.apply_verification(ratings, []) == []
    assert ratings["ai_artifacts"].value == "slightly_A"


def test_only_the_artifact_axis_loses_support_this_way():
    ratings = {"visual_quality": R("slightly_A")}
    judge.apply_verification(ratings, [_d("B", axis="visual_quality", verdict="rejected")])
    assert ratings["visual_quality"].value == "slightly_A"


def test_tie_note_explains_a_tie_that_lost_its_support():
    ratings = {"ai_artifacts": R("tie", "low"), "visual_quality": R("tie")}
    notes = judge.tie_notes(ratings, [], {a: ["r1", "r2"] for a in ratings}, lost_support=["ai_artifacts"])
    assert "ordinary photographic detail" in notes["ai_artifacts"]["why"]
    assert "no meaningful difference" in notes["visual_quality"]["why"]


def test_flaws_judged_natural_do_not_count_in_the_findings_check():
    flaw = _vf("A").model_copy(update={"artifact_review": "natural"})
    assert judge.check_findings({"ai_artifacts": R("slightly_A")}, [flaw], []) == []


def test_review_fields_reach_the_final_finding():
    f = {**_d("A", review="natural"), "region": "objects", "bbox": {"x": 0, "y": 0, "w": 1, "h": 1},
         "artifact_note": "out of focus", "seen_in_runs": [1]}
    out = judge.to_verified([f])[0]
    assert (out.artifact_review, out.artifact_note) == ("natural", "out of focus")
