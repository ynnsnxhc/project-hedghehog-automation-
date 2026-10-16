"""Deterministic rules: mapping runs back to A/B, reconciling the order-swapped
runs, applying verification, computing Overall, and output checks.

No model calls here. Everything is unit-tested in tests/test_judge.py.
"""
import re

import config
from schemas import AxisRating, BBox, Overall, VerifiedFinding

_STRENGTH_RANK = {"slight": 1, "strong": 2}
_CONF_RANK = {"low": 0, "medium": 1, "high": 2}
_CONF_NAME = {v: k for k, v in _CONF_RANK.items()}
_SEVERITY_RANK = {"minor": 1, "major": 2}
_REQ_POINTS = {"met": 1.0, "partial": 0.5, "missed": 0.0}
_NO_CONFLICT = {"", "none", "n/a", "no conflicts", "no conflict", "no conflicts."}

# "Image 1", "image 2", "Image_1": the labels the Inspector saw. Never meaningful once mapped to A/B.
_IMAGE_REF = re.compile(r"\bimage[\s_-]?([12])\b", re.IGNORECASE)


# ------------------------------------------------------------------ mapping
def relabel(text, order):
    """Rewrite "Image 1"/"Image 2" in free text as the real A/B label.

    The Inspector's slots are arbitrary (swapped in run 2), so text that says "Image 2"
    would otherwise point at the wrong picture once it is shown next to A/B ratings.
    """
    return _IMAGE_REF.sub(lambda m: f"{m.group(0)[:5]} {order[int(m.group(1)) - 1]}", text)


def map_run(ins, order, run):
    """Translate an Inspection from Image 1/2 into A/B. order[0] is what Image 1 was."""
    slot = {"1": order[0], "2": order[1], "tie": "tie"}
    calls = {}
    for axis, c in ins.calls.model_dump().items():
        calls[axis] = {"winner": slot[c["winner"]], "strength": c["strength"],
                       "confidence": c["confidence"], "reason": relabel(c["reason"], order)}
    findings = [{**f.model_dump(), "image": slot[f.image],
                 "observation": relabel(f.observation, order), "run": run} for f in ins.findings]
    reqs = {}
    for r in ins.requirement_results:
        by_slot = {"1": r.image_1.model_dump(), "2": r.image_2.model_dump()}
        for check in by_slot.values():
            check["evidence"] = relabel(check["evidence"], order)
        reqs[r.id] = {"A": by_slot["1" if order[0] == "A" else "2"],
                      "B": by_slot["1" if order[0] == "B" else "2"]}
    return {"calls": calls, "findings": findings, "requirements": reqs,
            "conflicts": [relabel(c, order) for c in ins.description_conflicts]}


def to_value(winner, strength):
    if winner == "tie":
        return "tie"
    return ("strongly_" if strength == "strong" else "slightly_") + winner


# ------------------------------------------------------------------ reconcile
def reconcile(run1, run2, axes):
    """Combine the two order-swapped runs, axis by axis.

    Returns (ratings, agreed axes, flipped axes, {axis: [run 1 reason, run 2 reason]}).

    Same winner  -> keep it; use the weaker strength and the lower confidence.
    Different    -> position bias suspected: record a tie at low confidence and flag.
    """
    ratings, agreed, flipped, reasons = {}, [], [], {}
    for axis in axes:
        a, b = run1["calls"][axis], run2["calls"][axis]
        conf = _CONF_NAME[min(_CONF_RANK[a["confidence"]], _CONF_RANK[b["confidence"]])]
        if a["winner"] == b["winner"]:
            if a["winner"] == "tie":
                value = "tie"
            else:
                weaker = min(_STRENGTH_RANK.get(a["strength"], 1), _STRENGTH_RANK.get(b["strength"], 1))
                value = to_value(a["winner"], "strong" if weaker == 2 else "slight")
            ratings[axis] = AxisRating(value=value, confidence=conf)
            agreed.append(axis)
        else:
            ratings[axis] = AxisRating(value="tie", confidence="low")
            flipped.append(axis)
        reasons[axis] = [a["reason"], b["reason"]]
    return ratings, agreed, flipped, reasons


def flip_message(axis, run1, run2):
    """Review reason for an axis whose two runs disagreed."""
    w1, w2 = run1["calls"][axis]["winner"], run2["calls"][axis]["winner"]
    # A vs B is a true flip (position bias). Tie vs a winner is a weaker disagreement.
    suspicion = "" if "tie" in (w1, w2) else "; position bias suspected"
    return (f"{axis}: the two inspections disagree after the image order was swapped "
            f"({w1} vs {w2}); recorded as a low-confidence tie{suspicion}")


def tie_notes(ratings, flipped, reasons, lost_support=()):
    """Why each tied axis is tied, for the Writer. The Inspector reasons are the only source for this.

    lost_support = axes that started as a win and became a tie because the findings behind them were disproved.
    """
    notes = {}
    for axis, rating in ratings.items():
        if rating.value != "tie":
            continue
        if axis in flipped:
            why = "the two order-swapped inspections disagreed, so no winner could be established"
        elif axis in lost_support:
            why = ("the inspections favored one image, but every flaw reported against the other image was "
                   "rejected on checking or judged to be ordinary photographic detail, so neither image has a "
                   "counted flaw on this axis")
        else:
            why = "both inspections found no meaningful difference"
        notes[axis] = {"why": why, "inspector_reasons": reasons[axis]}
    return notes


def merge_requirements(run1, run2):
    """Where the runs disagree on a requirement status, mark it unverifiable."""
    merged = {}
    for rid in run1["requirements"].keys() | run2["requirements"].keys():
        r1, r2 = run1["requirements"].get(rid), run2["requirements"].get(rid)
        if r1 is None or r2 is None:
            merged[rid] = r1 or r2
            continue
        merged[rid] = {}
        for img in ("A", "B"):
            if r1[img]["status"] == r2[img]["status"]:
                merged[rid][img] = r1[img]
            else:
                merged[rid][img] = {"status": "unverifiable",
                                    "evidence": f"runs disagree: {r1[img]['status']} vs {r2[img]['status']}; "
                                                f"{r1[img]['evidence']} / {r2[img]['evidence']}"}
    return merged


def merge_conflicts(run1, run2):
    """Places where an inspector disagreed with a blind description, de-duplicated."""
    out = []
    for c in run1["conflicts"] + run2["conflicts"]:
        c = c.strip()
        if c.lower() not in _NO_CONFLICT and c not in out:
            out.append(c)
    return out


# ------------------------------------------------------------------ findings
def _iou(a, b):
    """Intersection over union of two {x, y, w, h} boxes."""
    iw = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
    ih = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def _same_defect(merged, f):
    """Is `f` the defect already recorded in `merged`?

    Needs the same image, axis and region, and then either overlapping boxes, or the same
    wording from the other run (identical wording within one run can be two real defects,
    such as six fingers on each hand).
    """
    if (merged["image"], merged["axis"], merged["region"].strip().lower()) != \
            (f["image"], f["axis"], f["region"].strip().lower()):
        return False
    if f["run"] not in merged["seen_in_runs"] and \
            merged["observation"].strip().lower() == f["observation"].strip().lower():
        return True
    return _iou(merged["bbox"], f["bbox"]) >= config.FINDING_MERGE_IOU


def merge_findings(run1, run2):
    """De-duplicate findings across runs.

    Keyed on image, axis and region plus box overlap, so two different defects in one
    region (a bad left hand and a bad right hand) stay as two findings, while the same
    defect reported by both runs, even at different severities, becomes one. The merged
    finding keeps the higher severity.
    """
    merged = []
    for f in run1["findings"] + run2["findings"]:
        for m in merged:
            if _same_defect(m, f):
                if f["run"] not in m["seen_in_runs"]:
                    m["seen_in_runs"].append(f["run"])
                if _SEVERITY_RANK[f["severity"]] > _SEVERITY_RANK[m["severity"]]:
                    m["severity"], m["observation"], m["bbox"] = f["severity"], f["observation"], f["bbox"]
                break
        else:
            merged.append({**f, "seen_in_runs": [f["run"]]})
    return merged


def select_for_verification(findings, cap):
    """Indexes of findings to verify: major first, then those seen in both runs."""
    order = sorted(range(len(findings)), key=lambda i: (
        findings[i]["severity"] != "major", len(findings[i]["seen_in_runs"]) < 2))
    return order[:cap]


def _disproved(f):
    """Rejected by the Verifier, or judged ordinary photographic detail by the artifact reviewer."""
    return f.get("verdict") == "rejected" or f.get("artifact_review") == "natural"


def apply_verification(ratings, findings):
    """Adjust ratings for findings that did not hold up.

    - ai_artifacts is decided by findings. If it favors one image but every finding against the
      other image was disproved, the call has no support left: it becomes a low-confidence tie.
    - Lower confidence where a major finding against the losing image failed verification.

    Returns review reasons. A rating is never flipped to the other image: a rejected finding
    means the call needs a second look, not that the other image wins.
    """
    reasons = []
    rating = ratings.get("ai_artifacts")
    if rating is not None and rating.value != "tie":
        loser = "B" if rating.value.endswith("A") else "A"
        against = [f for f in findings if f["axis"] == "ai_artifacts" and f["image"] == loser]
        if against and all(_disproved(f) for f in against):
            ratings["ai_artifacts"] = AxisRating(value="tie", confidence="low")
            dropped = "; ".join(f"'{f['observation']}'" for f in against)
            reasons.append(f"ai_artifacts: every finding against image {loser} was rejected or judged ordinary "
                           f"photographic detail ({dropped}), so the {rating.value} call lost its support and is "
                           f"recorded as a low-confidence tie")
    for f in findings:
        if f.get("verdict") != "rejected" or f["severity"] != "major":
            continue
        rating = ratings.get(f["axis"])
        if rating is None or rating.value == "tie":
            continue
        winner = rating.value.split("_")[1]
        if f["image"] != winner:  # the finding was evidence against the loser
            ratings[f["axis"]] = AxisRating(value=rating.value, confidence="low")
            reasons.append(f"{f['axis']}: a major finding against image {f['image']} failed verification "
                           f"('{f['observation']}')")
    return reasons


# ------------------------------------------------------------------ overall
def overall(ratings, axes):
    wins = {"A": 0, "B": 0, "tie": 0}
    strong = {"A": 0, "B": 0}
    for axis in axes:
        v = ratings[axis].value
        if v == "tie":
            wins["tie"] += 1
        else:
            side = v.split("_")[1]
            wins[side] += 1
            strong[side] += v.startswith("strongly")

    if wins["A"] != wins["B"]:
        winner = "A" if wins["A"] > wins["B"] else "B"
        loser = "B" if winner == "A" else "A"
        if wins[loser] == 0 and strong[winner] > 0:
            strength, basis = "strongly", f"{winner} wins every decided axis, at least one decisively."
        else:
            strength, basis = "slightly", (f"{winner} wins {wins[winner]} axes to {wins[loser]}"
                                           + (", with offsetting weaknesses." if wins[loser] else
                                              ", but no axis is a decisive win."))
        return Overall(winner=winner, strength=strength, wins=wins, rule_applied="count", strength_basis=basis)

    if_value = ratings["instruction_following"].value if "instruction_following" in ratings else "tie"
    if if_value != "tie":
        winner = if_value.split("_")[1]
        return Overall(winner=winner, strength="slightly", wins=wins,
                       rule_applied="instruction_following_tiebreak",
                       strength_basis=f"Axis wins split evenly; instruction following goes to {winner}.")
    return Overall(winner="unresolved", strength="none", wins=wins, rule_applied="unresolved",
                   strength_basis="Axis wins split evenly and instruction following is tied; the guidance doesn't cover this.")


# ------------------------------------------------------------------ checks
def central_scores(requirements):
    """Points each image earned on the central requirements (met 1, partial 0.5, missed 0).

    A requirement that either image couldn't be checked on is skipped for both.
    Returns ({"A": points, "B": points}, how many requirements were scored).
    """
    score = {"A": 0.0, "B": 0.0}
    counted = 0
    for r in requirements:
        if not r.get("central"):
            continue
        a, b = r.get("A"), r.get("B")
        if not a or not b or a["status"] not in _REQ_POINTS or b["status"] not in _REQ_POINTS:
            continue
        score["A"] += _REQ_POINTS[a["status"]]
        score["B"] += _REQ_POINTS[b["status"]]
        counted += 1
    return score, counted


def check_checklist(requirements):
    """Problems with the requirement checklist itself."""
    if not requirements:
        return ["requirement checklist is empty"]
    problems = []
    ids = [r["id"] for r in requirements]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        problems.append(f"duplicate requirement ids: {dupes}")
    if not any(r.get("central") for r in requirements):
        problems.append("no requirement is marked central")
    unchecked = [r["id"] for r in requirements if not r.get("A") or not r.get("B")]
    if unchecked:
        problems.append(f"no inspection result for one or both images on: {unchecked}")
    return problems


def check_instruction_following(ratings, requirements, flipped):
    """The instruction-following rating should agree with how the central requirements came out.

    Flag only, never change the rating: the inspector may have weighed something the
    checklist doesn't capture, but a person should look.
    """
    rating = ratings.get("instruction_following")
    score, counted = central_scores(requirements)
    if rating is None or not counted:
        return []
    detail = f"central requirements met: A {score['A']:g}/{counted}, B {score['B']:g}/{counted}"
    if rating.value == "tie":
        if score["A"] != score["B"] and "instruction_following" not in flipped:
            return [f"instruction_following is a tie but {detail}"]
        return []
    side = rating.value.split("_")[1]
    other = "B" if side == "A" else "A"
    if score[other] > score[side]:
        return [f"instruction_following favors {side} but {detail}"]
    return []


def check_findings(ratings, findings, flipped):
    """The ai_artifacts and visual_quality ratings should agree with the findings that survived verification.

    Fewer and less severe defects wins, severity before count. The Writer can only
    explain a rating from these findings, so a rating they contradict can't be justified.
    """
    problems = []
    for axis in ("ai_artifacts", "visual_quality"):
        rating = ratings.get(axis)
        if rating is None:
            continue
        tally = {"A": [0, 0], "B": [0, 0]}  # [major, minor]
        for f in findings:
            if f.axis == axis and f.verdict in ("confirmed", "not_checked") and f.artifact_review != "natural":
                tally[f.image][0 if f.severity == "major" else 1] += 1
        detail = (f"findings (major/minor): A {tally['A'][0]}/{tally['A'][1]}, "
                  f"B {tally['B'][0]}/{tally['B'][1]}")
        if rating.value == "tie":
            if tally["A"][0] != tally["B"][0] and axis not in flipped:
                problems.append(f"{axis} is a tie but {detail}")
            continue
        side = rating.value.split("_")[1]
        other = "B" if side == "A" else "A"
        if tally[side] > tally[other]:  # lists compare majors first, then minors
            problems.append(f"{axis} favors {side} but {detail}")
    return problems


def code_checks(result):
    """Mechanical checks from section 4 of the instruction set, plus consistency checks."""
    problems = []
    if result.evaluator != "automated":
        problems.append("missing automated label")
    j = result.justification
    if j is None:
        problems.append("no justification")
    else:
        for axis in result.axes:
            if not getattr(j, axis).strip():
                problems.append(f"justification missing for {axis}")
        if "ps2_style" not in result.axes and j.ps2_style.strip():
            problems.append("PS2 justification written but PS2 is not scored")
        if not j.overall.strip():
            problems.append("justification missing overall explanation")
    recomputed = overall(result.ratings, result.axes)
    if recomputed.winner != result.overall.winner:
        problems.append("overall winner does not match the count rule")
    elif recomputed.strength != result.overall.strength:
        problems.append("overall strength does not match the count rule")

    flipped = result.order_swap.get("flipped_axes", [])
    problems += check_checklist(result.requirements)
    problems += check_instruction_following(result.ratings, result.requirements, flipped)
    problems += check_findings(result.ratings, result.findings, flipped)
    return problems


def to_verified(findings):
    return [VerifiedFinding(image=f["image"], region=f["region"], bbox=BBox(**f["bbox"]),
                            observation=f["observation"], axis=f["axis"], severity=f["severity"],
                            verdict=f.get("verdict", "not_checked"),
                            verifier_note=f.get("verifier_note", ""),
                            artifact_review=f.get("artifact_review", ""), artifact_note=f.get("artifact_note", ""),
                            seen_in_runs=f["seen_in_runs"])
            for f in findings]
