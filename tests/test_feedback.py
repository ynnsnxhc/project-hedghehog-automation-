"""Tests for the feedback helper: the verdict, the shape pool, the pick log and the hard rules.

Most lint cases are corrections that were given while writing feedback, kept so a slip can't come back.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import feedback
import orchestrator

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

CLEAR = ("Overall, B is slightly better than A. B is better because she is seated on the Vespa and turning back "
         "to smile in a close-medium shot that looks like 1978 Rome, so it follows the prompt. A is worse because "
         "the pose is off, so it doesn't follow the prompt.")
GAP = ("Overall, B is a little better than A. The pose is what separates them, since B has her seated on the Vespa "
       "and turning back to smile while A has her leaning over the seat and laughing.")
FLAWED = ("Overall, B is slightly better than A. Both are bad because A has a hand with six fingers and B has a "
          "warped sign, so neither is clean. B is less bad because the sign is small and far away, so it's the "
          "better pick.")
TIE_GOOD = "Overall, both are good. Both do what was asked and look natural, so nothing separates them."
TIE_BAD = ("Overall, both are bad. A has a hand with six fingers, so it misses what was asked. "
           "B has a warped sign, so it misses it too.")


# ---------------------------------------------------------------- building small results
def rating(value, conf="high"):
    return {"value": value, "confidence": conf}


def req(rid, text, central, a, b, kind="subject"):
    return {"id": rid, "text": text, "type": kind, "central": central,
            "A": {"status": a, "evidence": "x"}, "B": {"status": b, "evidence": "x"}}


def finding(image, axis="instruction_following", severity="minor", verdict="confirmed", review="",
            obs="something visible"):
    return {"image": image, "axis": axis, "severity": severity, "verdict": verdict, "artifact_review": review,
            "observation": obs, "region": "whole_frame", "bbox": {"x": 0, "y": 0, "w": 1, "h": 1}}


def make(winner="B", strength="slightly", ratings=None, findings=None, requirements=None, **over):
    base = {"task_id": "t1", "prompt": "a prompt",
            "interpretation": {"deciding_question": "Does she sit on the Vespa and turn to smile?"},
            "ratings": ratings or {"instruction_following": rating("slightly_B"),
                                   "visual_quality": rating("slightly_A"), "ai_artifacts": rating("tie")},
            "overall": {"winner": winner, "strength": strength, "wins": {"A": 0, "B": 0, "tie": 0},
                        "rule_applied": "count", "strength_basis": "x"},
            "requirements": requirements or [], "findings": findings or [],
            "review_reasons": [], "needs_human_review": False,
            "order_swap": {"agreed_axes": [], "flipped_axes": []}, "axis_reasons": {}}
    base.update(over)
    return base


def tie_ratings():
    return {"instruction_following": rating("tie"), "visual_quality": rating("tie"), "ai_artifacts": rating("tie")}


# ---------------------------------------------------------------- the verdict
def test_small_gap_when_every_win_is_slight_and_no_major_flaw():
    info = feedback.analyze(make())
    assert (info["verdict"], info["size"], info["pair_type"]) == ("B", "slightly", "small_gap")


def test_clear_win_when_an_axis_is_strong():
    ratings = {"instruction_following": rating("strongly_B"), "visual_quality": rating("slightly_A"),
               "ai_artifacts": rating("tie")}
    assert feedback.analyze(make(ratings=ratings))["pair_type"] == "clear_win"


def test_clear_win_when_only_one_image_has_a_major_flaw():
    info = feedback.analyze(make(findings=[finding("A", severity="major")]))
    assert info["pair_type"] == "clear_win" and info["verdict"] == "B"


def test_both_flawed_when_each_image_has_a_kept_major_flaw():
    info = feedback.analyze(make(findings=[finding("A", severity="major"), finding("B", severity="major")]))
    assert info["pair_type"] == "both_flawed" and info["verdict"] == "B"


def test_unresolved_without_major_flaws_is_both_good():
    info = feedback.analyze(make(winner="unresolved", strength="none", ratings=tie_ratings()))
    assert (info["verdict"], info["size"], info["pair_type"]) == ("both-good", None, "tie_good")


def test_unresolved_with_a_major_flaw_in_each_is_both_bad():
    info = feedback.analyze(make(winner="unresolved", strength="none", ratings=tie_ratings(),
                                 findings=[finding("A", severity="major"), finding("B", severity="major")]))
    assert (info["verdict"], info["pair_type"]) == ("both-bad", "tie_bad")


def test_unresolved_with_one_major_flaw_suggests_the_other_image_and_says_to_check():
    info = feedback.analyze(make(winner="unresolved", strength="none", ratings=tie_ratings(),
                                 findings=[finding("A", severity="major")]))
    assert info["verdict"] == "B" and "Check this one" in " ".join(info["why"])


def test_picture_quality_alone_makes_it_both_good():
    ratings = {"instruction_following": rating("tie"), "visual_quality": rating("slightly_A"),
               "ai_artifacts": rating("tie")}
    info = feedback.analyze(make(winner="A", ratings=ratings))
    assert (info["verdict"], info["pair_type"], info["size"]) == ("both-good", "tie_good", None)
    assert "picture quality" in " ".join(info["why"])


def test_a_pair_separated_by_artifacts_is_not_a_picture_quality_tie():
    ratings = {"instruction_following": rating("tie"), "visual_quality": rating("slightly_A"),
               "ai_artifacts": rating("slightly_A")}
    info = feedback.analyze(make(winner="A", ratings=ratings))
    assert info["verdict"] == "A" and info["pair_type"] == "small_gap"


@pytest.mark.parametrize("axis, verdict, review, kept", [
    ("instruction_following", "confirmed", "", True),
    ("visual_quality", "confirmed", "", True),
    ("ai_artifacts", "confirmed", "ai_defect", True),
    ("ai_artifacts", "confirmed", "", False),
    ("ai_artifacts", "confirmed", "natural", False),
    ("ai_artifacts", "confirmed", "unclear", False),
    ("instruction_following", "confirmed", "natural", False),
    ("instruction_following", "rejected", "", False),
    ("visual_quality", "unverifiable", "", False),
])
def test_which_findings_count_as_kept(axis, verdict, review, kept):
    assert feedback.is_kept(finding("A", axis=axis, verdict=verdict, review=review)) is kept


def test_a_dismissed_artifact_does_not_make_a_pair_both_flawed():
    dismissed = [finding("A", axis="ai_artifacts", severity="major", review="natural"),
                 finding("B", axis="ai_artifacts", severity="major", review="natural")]
    assert feedback.analyze(make(findings=dismissed))["pair_type"] == "small_gap"


# ---------------------------------------------------------------- the reasons
def test_the_requirement_nearest_the_deciding_question_leads():
    requirements = [req("R1", "The street is in Rome", True, "missed", "met"),
                    req("R2", "She is on the Vespa", True, "partial", "met"),
                    req("R3", "She will turn and smile", True, "partial", "met")]
    order = [d["id"] for d in feedback.requirement_differences(make(requirements=requirements))]
    assert order == ["R3", "R2", "R1"]


def test_unverifiable_requirements_are_flagged_and_not_counted():
    result = make(requirements=[req("R1", "She will turn and smile", True, "met", "unverifiable")])
    assert feedback.requirement_differences(result) == []
    assert any("R1" in item and "unverifiable for B" in item for item in feedback.flag_items(result))


def test_only_shot_and_film_points_both_images_meet_are_named_as_true_for_both():
    requirements = [req("R1", "Close-medium shot", False, "met", "met", kind="framing"),
                    req("R2", "Ektachrome film look", True, "met", "met", kind="style"),
                    req("R3", "A woman", True, "met", "met"),
                    req("R4", "1978 look", False, "met", "partial", kind="style"),
                    req("R5", "A photograph, not an illustration", True, "met", "met", kind="style")]
    result = make(requirements=requirements, prompt="A close-medium shot on Ektachrome film of a woman on a Vespa")
    assert feedback.true_for_both(result) == ["Close-medium shot", "Ektachrome film look"]


def test_a_size_note_appears_when_instruction_following_was_strong_but_the_overall_size_is_slight():
    ratings = {"instruction_following": rating("strongly_B"), "visual_quality": rating("slightly_A"),
               "ai_artifacts": rating("tie")}
    info = feedback.analyze(make(ratings=ratings))
    assert any("'clearly better' is also fair" in w for w in info["why"])
    assert not any("clearly better" in w for w in feedback.analyze(make())["why"])


def test_instruction_concerns_fill_in_when_no_requirement_differs():
    info = feedback.analyze(make(findings=[finding("A", severity="major", obs="A looks like a styled shoot")]))
    assert any("A looks like a styled shoot" in r for r in info["reasons"])


def test_the_inspectors_words_are_used_when_nothing_else_decides():
    info = feedback.analyze(make(axis_reasons={"instruction_following": ["A shows the effect on a body figure"]}))
    assert any("inspector on instruction following" in r for r in info["reasons"])


def test_kept_artifacts_are_listed_and_dismissed_ones_are_left_out():
    result = make(findings=[finding("A", axis="ai_artifacts", review="ai_defect", obs="a merged finger"),
                            finding("B", axis="ai_artifacts", review="natural", obs="soft fridge photos")])
    info = feedback.analyze(result)
    assert any("a merged finger" in r for r in info["reasons"])
    assert any("soft fridge photos" in item for item in info["leave_out"])


def test_the_review_flag_and_low_confidence_are_reported():
    ratings = {"instruction_following": rating("slightly_B", "low"), "visual_quality": rating("slightly_A"),
               "ai_artifacts": rating("tie")}
    info = feedback.analyze(make(ratings=ratings, needs_human_review=True))
    assert info["flagged"] is True
    assert "instruction_following is low confidence" in info["flags"]
    assert "Flagged for review: yes" in feedback.format_points(info)


def test_it_works_on_a_real_pipeline_result(tmp_path):
    orchestrator.main([str(EXAMPLES / "tasks.jsonl"), "--out", str(tmp_path), "--mock"])
    info = feedback.analyze(feedback.load_result(tmp_path / "demo_001.json"))
    assert (info["verdict"], info["size"], info["pair_type"]) == ("A", "strongly", "clear_win")
    assert info["reasons"]


# ---------------------------------------------------------------- the shape pool
def test_every_kind_of_pair_has_at_least_three_shapes():
    for pair_type in feedback.PAIR_TYPES:
        assert len([n for n, s in feedback.SHAPES.items() if pair_type in s["types"]]) >= 3


def test_three_different_shapes_from_the_pool_and_the_same_seed_gives_the_same_three():
    first = feedback.pick_three("clear_win", [], "seed")
    assert len(set(first)) == 3 and all("clear_win" in feedback.SHAPES[n]["types"] for n in first)
    assert feedback.pick_three("clear_win", [], "seed") == first


def test_the_last_run_of_this_kind_of_pair_is_avoided_when_it_can_be():
    last = ["short_second_half", "detail_rich", "general_summary"]
    history = [{"pair_type": "clear_win", "shown": last, "shape": "detail_rich"}]
    for seed in range(20):  # no seed may bring one of the last run's shapes ahead of the two fresh ones
        chosen = feedback.pick_three("clear_win", history, str(seed))
        assert "requirement_check" in chosen and "mirrored" in chosen


def showings(shape, times, picked=False):
    return [{"pair_type": "clear_win", "shown": [shape], "shape": shape if picked else None} for _ in range(times)]


def test_a_shape_shown_three_times_and_never_picked_retires():
    assert "mirrored" not in feedback.eligible_shapes("clear_win", showings("mirrored", 3))
    assert "mirrored" in feedback.eligible_shapes("clear_win", showings("mirrored", 2))


def test_a_shape_that_was_picked_does_not_retire():
    assert "mirrored" in feedback.eligible_shapes("clear_win", showings("mirrored", 5, picked=True))


def test_retirement_never_goes_below_three_shapes():
    history = []
    for name in ("gap_key_point", "gap_compare", "gap_summary"):
        history += [{"pair_type": "small_gap", "shown": [name], "shape": None}] * 3
    assert len(feedback.eligible_shapes("small_gap", history)) == 3


def test_none_fit_counts_as_a_showing_without_a_pick():
    stats = feedback.shape_stats([{"shown": ["mirrored", "detail_rich"], "shape": None}])
    assert stats["mirrored"] == {"shown": 1, "picked": 0}


def test_the_pick_log_round_trips_text_and_keeps_plain_line_ends(tmp_path):
    path = tmp_path / "history.jsonl"
    feedback.append_history(path, {"task_id": "t1", "text": "café → ok", "shown": ["a"], "shape": "a"})
    feedback.append_history(path, {"task_id": "t2", "text": "two", "shown": [], "shape": None})
    assert b"\r\n" not in path.read_bytes()
    assert [r["task_id"] for r in feedback.load_history(path)] == ["t1", "t2"]
    assert feedback.load_history(path)[0]["text"] == "café → ok"


def test_a_missing_history_file_is_an_empty_history(tmp_path):
    assert feedback.load_history(tmp_path / "nope.jsonl") == []


# ---------------------------------------------------------------- the hard rules
@pytest.mark.parametrize("text, verdict, size, pair_type", [
    (CLEAR, "B", "slightly", "clear_win"),
    (GAP, "B", "slightly", "small_gap"),
    (FLAWED, "B", "slightly", "both_flawed"),
    (TIE_GOOD, "both-good", None, "tie_good"),
    (TIE_BAD, "both-bad", None, "tie_bad"),
    (CLEAR.replace("A is worse", "B is worse").replace("Overall, B is slightly better than A",
                                                       "Overall, A is clearly better than B").replace(
        "B is better because she", "A is better because she"), "A", "strongly", "clear_win"),
])
def test_good_drafts_pass(text, verdict, size, pair_type):
    problems, _ = feedback.lint_text(text, verdict, size, pair_type)
    assert problems == []


def problems_for(text, verdict="B", size="slightly", pair_type="clear_win", avoid=()):
    return feedback.lint_text(text, verdict, size, pair_type, avoid)[0]


@pytest.mark.parametrize("bad, expected", [
    (CLEAR.replace("Vespa and turning", "Vespa, and turning"), 'a comma before "and"'),
    (CLEAR.replace("so it follows", "— so it follows"), "characters you don't use"),
    (CLEAR.replace("the pose is off", "the pose (the angle) is off"), "characters you don't use"),
    (CLEAR.replace("A is worse because", "A is worse because:"), "characters you don't use"),
    (CLEAR.replace(", so it doesn't", "; so it doesn't"), "characters you don't use"),
    (CLEAR.replace("doesn't", "doesn’t"), "characters you don't use"),
    (CLEAR[:-1] + "?", "characters you don't use"),
    (CLEAR[:-1], "must end with a period"),
    (CLEAR.replace("looks like 1978 Rome", "reads as 1978 Rome"), '"reads as"'),
    (CLEAR.replace("looks like 1978 Rome", "comes across as 1978 Rome"), '"comes across as"'),
    (CLEAR.replace("in a close-medium shot", "thanks to a close-medium shot"), '"thanks to"'),
    (CLEAR.replace("in a close-medium shot", "owing to a close-medium shot"), '"owing to"'),
    (CLEAR.replace("A is worse because", "A is weaker because"), '"weaker"'),
    (CLEAR.replace("B is better because", "B is stronger because"), '"stronger"'),
    (CLEAR.replace("B is better because", "B is more convincing because"), '"more convincing"'),
    (CLEAR.replace("the pose is off", "the pose is barely right"), '"barely"'),
    (CLEAR.replace("so it follows", "so it actually follows"), '"actually"'),
    (CLEAR.replace("the pose is off", "the thing is off"), '"thing"'),
    (CLEAR.replace("the pose is off", "the pose is nicer"), '"nicer"'),
    (CLEAR.replace("B is better because she", "B is better because I think she"), "first person"),
    (CLEAR.replace("A is worse because the", "A is worse because I'd say the"), "first person"),
    (CLEAR.replace("because the pose", "because my pose"), "first person"),
    (CLEAR.replace(". A is", ".\nA is"), "no line breaks"),
    (CLEAR.replace("Overall, B is slightly better than A.", "Overall, B is better than A."), "the first line must be"),
    (CLEAR.replace("slightly better", "clearly better"), "the first line must be"),
    (CLEAR.replace("Overall, B is slightly better than A.", "Overall, A is slightly better than B."),
     "the first line must be"),
    (CLEAR.replace(" A is worse because the pose is off, so it doesn't follow the prompt.", ""), "expected 3 sentences"),
    (CLEAR + " It is fine.", "expected 3 sentences"),
    (CLEAR.replace("A is worse because", "A is off because"), 'sentence 3 must say why the other is "worse"'),
    (CLEAR.replace("B is better because she", "B is better since she"), 'sentence 2 must say why one is "better"'),
])
def test_known_slips_are_caught(bad, expected):
    assert any(expected in p for p in problems_for(bad)), problems_for(bad)


def test_a_small_gap_is_two_sentences_and_a_clear_win_is_not():
    assert any("expected 2 sentences" in p for p in problems_for(CLEAR, pair_type="small_gap"))
    assert any("expected 3 sentences" in p for p in problems_for(GAP, pair_type="clear_win"))


def test_ties_use_the_exact_lines():
    bad = TIE_GOOD.replace("Overall, both are good.", "Overall, they are both good.")
    assert any("the first line must be" in p for p in problems_for(bad, "both-good", None, "tie_good"))
    assert any("the first line must be" in p for p in problems_for(TIE_GOOD, "both-bad", None, "tie_bad"))


def test_a_pair_with_no_size_expects_the_plain_first_line():
    plain = CLEAR.replace("slightly better", "better")
    assert problems_for(plain, size=None) == []
    assert any("the first line must be" in p for p in problems_for(CLEAR, size=None))


def test_both_bad_then_less_bad_needs_both_bad_and_the_lesser_one():
    no_bad = FLAWED.replace("Both are bad because", "Both have trouble because")
    assert any('both are "bad"' in p for p in problems_for(no_bad, pair_type="both_flawed"))
    no_pick = FLAWED.replace("B is less bad because", "B is fine because").replace("it's the better pick", "it's fine")
    assert any("less bad" in p for p in problems_for(no_pick, pair_type="both_flawed"))


def test_words_to_leave_out_are_checked():
    assert any("Vespa" in p for p in problems_for(CLEAR, avoid=("Vespa",)))
    assert problems_for(CLEAR, avoid=("fridge",)) == []


def test_hyphens_inside_words_and_apostrophes_are_fine_but_other_marks_are_not():
    assert feedback.bad_characters("close-medium and doesn't, lived-in 1978.") == []
    assert feedback.bad_characters("- start, end -, a - b") == ["-", "-", "-"]


def test_prompt_wording_and_quality_words_only_give_notes():
    text = CLEAR.replace("so it follows the prompt", "so it matches the prompt").replace(
        "turning back to smile", "turning back to smile in a sharper picture")
    problems, warnings = feedback.lint_text(text, "B", "slightly", "clear_win")
    assert problems == [] and len(warnings) == 2


# ---------------------------------------------------------------- the command line
def write_result(tmp_path, **over):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(make(**over)), encoding="utf-8")
    return path


def test_points_prints_the_verdict_check(tmp_path, capsys):
    assert feedback.main(["points", str(write_result(tmp_path))]) == 0
    out = capsys.readouterr().out
    assert "Suggested verdict: B is better than A" in out and "small_gap" in out and "2 sentences" in out


def test_points_can_print_json(tmp_path, capsys):
    feedback.main(["points", str(write_result(tmp_path)), "--json"])
    assert json.loads(capsys.readouterr().out)["verdict"] == "B"


def test_shapes_prints_three_and_follows_a_changed_verdict(tmp_path, capsys):
    path = write_result(tmp_path)
    history = tmp_path / "h.jsonl"
    feedback.main(["shapes", str(path), "--history", str(history)])
    assert capsys.readouterr().out.count("\n  ") == 3
    feedback.main(["shapes", str(path), "--verdict", "both-good", "--history", str(history)])
    assert "tie_good" in capsys.readouterr().out


def test_lint_reports_each_line_and_fails_on_a_problem(tmp_path, capsys):
    path = write_result(tmp_path)
    candidates = tmp_path / "c.txt"
    candidates.write_text(GAP + "\n" + CLEAR + "\n", encoding="utf-8")
    assert feedback.main(["lint", str(path), str(candidates)]) == 1
    out = capsys.readouterr().out
    assert "1: ok" in out and "2: FAIL" in out and "expected 2 sentences" in out


def test_lint_passes_when_every_line_is_clean_and_uses_a_changed_verdict(tmp_path):
    path = write_result(tmp_path)
    good = tmp_path / "good.txt"
    good.write_text(GAP + "\n", encoding="utf-8")
    assert feedback.main(["lint", str(path), str(good)]) == 0
    tie = tmp_path / "tie.txt"
    tie.write_text(TIE_GOOD + "\n", encoding="utf-8")
    assert feedback.main(["lint", str(path), str(tie)]) == 1
    assert feedback.main(["lint", str(path), str(tie), "--verdict", "both-good"]) == 0


def test_log_appends_a_record_and_none_fit_has_no_shape(tmp_path):
    history = tmp_path / "h.jsonl"
    base = ["log", "--task", "t1", "--verdict", "B", "--size", "slightly", "--pair-type", "small_gap",
            "--shown", "gap_key_point,gap_compare,gap_summary", "--history", str(history)]
    assert feedback.main(base + ["--shape", "gap_compare", "--text", GAP]) == 0
    assert feedback.main(base + ["--text", "none fit"]) == 0
    first, second = feedback.load_history(history)
    assert first["shape"] == "gap_compare" and first["text"] == GAP and first["size"] == "slightly"
    assert second["shape"] is None and second["shown"] == ["gap_key_point", "gap_compare", "gap_summary"]


def test_log_reads_the_text_from_a_file(tmp_path):
    history, text = tmp_path / "h.jsonl", tmp_path / "t.txt"
    text.write_text(TIE_GOOD + "\n", encoding="utf-8")
    feedback.main(["log", "--task", "t", "--verdict", "both-good", "--pair-type", "tie_good",
                   "--shape", "tie_summary", "--text-file", str(text), "--history", str(history)])
    assert feedback.load_history(history)[0]["text"] == TIE_GOOD


def test_pool_marks_retired_shapes(tmp_path, capsys):
    history = tmp_path / "h.jsonl"
    for _ in range(3):
        feedback.append_history(history, {"pair_type": "clear_win", "shown": ["mirrored"], "shape": None})
    feedback.main(["pool", "--history", str(history)])
    out = capsys.readouterr().out
    assert "mirrored: shown 3, picked 0  (retired)" in out and "detail_rich: shown 0, picked 0" in out
