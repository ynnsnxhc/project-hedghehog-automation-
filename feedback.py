"""Helpers for the feedback text that goes into the rating form.

    python feedback.py points my_tests/vespa_results/vespa_001.json
    python feedback.py shapes my_tests/vespa_results/vespa_001.json
    python feedback.py lint my_tests/vespa_results/vespa_001.json candidates.txt
    python feedback.py log --task vespa_001 --verdict B --size slightly --pair-type small_gap \\
        --shape gap_key_point --shown gap_key_point,gap_compare,gap_summary --text-file picked.txt
    python feedback.py pool

points  reads a result file and prints the verdict check: the suggested verdict (one of four), the deciding
        reasons, what both images get right, and anything flagged or doubtful.
shapes  picks three different shapes for the pair type, skipping shapes that were shown three times and never picked.
lint    checks drafts (one per line of the file) against the hard rules for the verdict. Exit code 1 on failure.
log     records which candidate was picked in feedback_history.jsonl, so unpicked shapes can retire.
pool    shows how often each shape was shown and picked.

The four verdicts are "A", "B", "both-good" and "both-bad". Who is ahead is only said for A or B.
"""
import argparse
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HISTORY = Path(__file__).with_name("feedback_history.jsonl")

VERDICTS = ("A", "B", "both-good", "both-bad")
PAIR_TYPES = ("clear_win", "small_gap", "both_flawed", "tie_good", "tie_bad")

# How many sentences each kind of pair gets: (fewest, most).
SENTENCES = {"clear_win": (3, 3), "small_gap": (2, 2), "both_flawed": (3, 3), "tie_good": (2, 3), "tie_bad": (3, 3)}

# The pipeline's strength word, and the plain words allowed in the first line for it.
SIZE_WORDS = {"slightly": ("slightly", "a little"), "strongly": ("clearly", "a lot")}

# Each shape says which kinds of pair it fits and how to build it. Every pair type keeps at least three.
SHAPES = {
    "short_second_half": {"types": ["clear_win"],
                          "how": "Sentence 2 gives the lead reason with one or two details. Sentence 3 is short and built differently."},
    "detail_rich": {"types": ["clear_win"],
                    "how": "Sentence 2 carries a short run of concrete details that decided it. Sentence 3 says what the other image misses."},
    "general_summary": {"types": ["clear_win"],
                        "how": "No object names. Judge in general terms, because ... so ... in both sentences."},
    "requirement_check": {"types": ["clear_win"],
                          "how": "Sentence 2 names the prompt points the better image gets right. Sentence 3 names the ones the other misses."},
    "mirrored": {"types": ["clear_win"],
                 "how": "Sentences 2 and 3 are built the same way, because ..., so ..., for contrast."},
    "gap_key_point": {"types": ["small_gap"],
                      "how": "Two sentences. Sentence 2 leads with the prompt's key point and what separates the images."},
    "gap_compare": {"types": ["small_gap"],
                    "how": "Two sentences. Sentence 2 sets what A does against what B does in one line."},
    "gap_summary": {"types": ["small_gap"],
                    "how": "Two sentences. Sentence 2 states the one difference in general terms, no object names."},
    "flawed_both_first": {"types": ["both_flawed"],
                          "how": "Sentence 2 says both are bad and why. Sentence 3 says why one is less bad and the better pick."},
    "flawed_each_issue": {"types": ["both_flawed"],
                          "how": "Sentence 2 names each image's issue, so both are bad. Sentence 3 says why one issue is the lesser."},
    "flawed_summary": {"types": ["both_flawed"],
                       "how": "General terms. Both are bad, then why one is less bad, no object names."},
    "tie_same_points": {"types": ["tie_good", "tie_bad"],
                        "how": "Name the same points both images get right, or both get wrong, and the issues if bad."},
    "tie_summary": {"types": ["tie_good", "tie_bad"],
                    "how": "General terms, no object names."},
    "tie_requirements": {"types": ["tie_good", "tie_bad"],
                         "how": "Go through the prompt points both images meet, or both miss."},
}

# Words and phrases the feedback never uses.
BANNED = ["reads as", "comes across as", "thanks to", "owing to", "weaker", "stronger", "more convincing", "barely",
          "actually", "thing", "stuff", "nicer", "reads quickly", "whereas", "furthermore"]

# Not banned, but not what was asked for. Shown as warnings.
SOFT = [("matches the prompt", "say 'follows the prompt' or 'does what was asked'"),
        ("match the prompt", "say 'follows the prompt' or 'does what was asked'"),
        ("fits the prompt", "say 'follows the prompt' or 'does what was asked'"),
        ("fit the prompt", "say 'follows the prompt' or 'does what was asked'"),
        ("sharper", "picture quality isn't one of the reasons you allowed"),
        ("sharpness", "picture quality isn't one of the reasons you allowed"),
        ("exposed", "picture quality isn't one of the reasons you allowed"),
        ("exposure", "picture quality isn't one of the reasons you allowed")]

STATUS_RANK = {"met": 0, "partial": 1, "missed": 2}


# ------------------------------------------------------------------ small helpers
def shorten(text, limit):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


def words_of(text):
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 3}


def load_result(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ reading a result
def is_kept(finding):
    """A finding the pipeline confirmed and, for artifacts, the second reviewer kept as a real defect."""
    if finding.get("verdict") != "confirmed" or finding.get("artifact_review") == "natural":
        return False
    if finding.get("axis") == "ai_artifacts":
        return finding.get("artifact_review") == "ai_defect"
    return True


def major_kept(result, image):
    return [f for f in result.get("findings", [])
            if f.get("image") == image and f.get("severity") == "major" and is_kept(f)]


def kept_artifacts(result):
    return [f for f in result.get("findings", []) if f.get("axis") == "ai_artifacts" and is_kept(f)]


def instruction_concerns(result):
    found = [f for f in result.get("findings", []) if f.get("axis") == "instruction_following" and is_kept(f)]
    found.sort(key=lambda f: f.get("severity") != "major")
    return found


def requirement_differences(result):
    """Central requirements where one image does better, the one closest to the deciding question first."""
    deciding = words_of((result.get("interpretation") or {}).get("deciding_question", ""))
    out = []
    for q in result.get("requirements", []):
        if not q.get("central"):
            continue
        a = (q.get("A") or {}).get("status")
        b = (q.get("B") or {}).get("status")
        if a not in STATUS_RANK or b not in STATUS_RANK or a == b:
            continue  # unverifiable ones are flagged instead
        out.append({"id": q["id"], "text": q["text"], "A": a, "B": b,
                    "better": "A" if STATUS_RANK[a] < STATUS_RANK[b] else "B",
                    "gap": abs(STATUS_RANK[a] - STATUS_RANK[b]),
                    "overlap": len(words_of(q["text"]) & deciding)})
    out.sort(key=lambda d: (-d["overlap"], -d["gap"], d["id"]))
    return out


def true_for_both(result):
    """Shot, film and era points both images meet, kept only when they echo the prompt's own words.
    Named once as true for both."""
    prompt = words_of(result.get("prompt", ""))
    return [shorten(q["text"], 90) for q in result.get("requirements", [])
            if q.get("type") in ("style", "framing")
            and (q.get("A") or {}).get("status") == "met" and (q.get("B") or {}).get("status") == "met"
            and len(words_of(q["text"]) & prompt) >= 2]


def flag_items(result):
    items = []
    for q in result.get("requirements", []):
        if q.get("central"):
            for image in ("A", "B"):
                if (q.get(image) or {}).get("status") == "unverifiable":
                    items.append(f"{q['id']} ({shorten(q['text'], 60)}) is unverifiable for {image}")
    for axis, r in result.get("ratings", {}).items():
        if r.get("confidence") == "low":
            items.append(f"{axis} is low confidence")
    for axis in (result.get("order_swap") or {}).get("flipped_axes", []):
        items.append(f"the two inspections disagreed on {axis}")
    audit = [r for r in result.get("review_reasons", []) if r.startswith("audit:")]
    items += ["auditor: " + shorten(r[len("audit:"):], 120) for r in audit[:3]]
    return items


def leave_out(result):
    """Things that were dismissed, rejected or unverifiable. They stay out of the feedback."""
    return [f"{f['image']}: {shorten(f['observation'], 80)}" for f in result.get("findings", [])
            if f.get("artifact_review") in ("natural", "unclear") or f.get("verdict") in ("rejected", "unverifiable")]


def quality_only(result):
    """The pipeline's pick rests only on picture quality: instruction following and artifacts are both ties."""
    r = result.get("ratings", {})
    def value(axis):
        return (r.get(axis) or {}).get("value", "tie")
    return value("instruction_following") == "tie" and value("ai_artifacts") == "tie" and value("visual_quality") != "tie"


# ------------------------------------------------------------------ the verdict
def analyze(result):
    """The verdict check: suggested verdict, kind of pair, deciding reasons, what is flagged."""
    ratings = result.get("ratings", {})
    overall = result.get("overall", {})
    winner = overall.get("winner", "unresolved")
    size = overall.get("strength") if overall.get("strength") in SIZE_WORDS else None
    major = {"A": major_kept(result, "A"), "B": major_kept(result, "B")}
    both_major = bool(major["A"]) and bool(major["B"])
    neither_major = not major["A"] and not major["B"]
    any_strong = any(r.get("value", "").startswith("strongly") for r in ratings.values())
    why = []

    if winner == "unresolved":
        size = None
        if both_major:
            verdict, pair = "both-bad", "tie_bad"
            why.append("The pipeline found no winner and each image has a confirmed major flaw.")
        elif neither_major:
            verdict, pair = "both-good", "tie_good"
            why.append("The pipeline found no winner and neither image has a confirmed major flaw.")
        else:
            lone = "A" if major["A"] else "B"
            verdict = "B" if lone == "A" else "A"
            pair, size = "clear_win", "slightly"
            why.append(f"The pipeline found no winner, but only image {lone} has a confirmed major flaw, "
                       f"so {verdict} is suggested. Check this one.")
    elif quality_only(result) and neither_major:
        verdict, pair, size = "both-good", "tie_good", None
        why.append(f"The pipeline picked {winner} only on picture quality and the prompt points are even, so both good.")
    else:
        verdict = winner
        if both_major:
            pair = "both_flawed"
            why.append("Each image has a confirmed major flaw, so both are bad and one is less bad.")
        elif not any_strong and neither_major:
            pair = "small_gap"
            why.append("Every axis win is slight and neither image has a confirmed major flaw, so only the deciding gap.")
        else:
            pair = "clear_win"
            why.append("One image clearly does better, or only one has a confirmed major flaw.")
        if size == "slightly" and (ratings.get("instruction_following") or {}).get("value") == f"strongly_{verdict}":
            why.append(f"Instruction following was strongly {verdict}, so 'clearly better' is also fair. "
                       f"Confirm the size at the verdict check.")

    differences = requirement_differences(result)
    reasons = []
    for d in differences[:3]:
        reasons.append(f"{d['id']} ({shorten(d['text'], 90)}): A {d['A']}, B {d['B']}, so {d['better']} does better")
    if not differences:
        for f in instruction_concerns(result)[:2]:
            reasons.append(f"{f['image']} ({f['severity']}, confirmed): {shorten(f['observation'], 140)}")
    if not reasons and (result.get("ratings", {}).get("instruction_following", {}).get("value", "tie") != "tie"):
        said = (result.get("axis_reasons") or {}).get("instruction_following", [])
        reasons += ["inspector on instruction following: " + shorten(s, 420) for s in said[:2]]
    for f in kept_artifacts(result)[:1]:
        reasons.append(f"{f['image']} artifact the reviewer kept: {shorten(f['observation'], 120)}")

    return {
        "task_id": result.get("task_id", ""),
        "prompt": result.get("prompt", ""),
        "verdict": verdict,
        "size": size,
        "pair_type": pair,
        "why": why,
        "pipeline": ", ".join(f"{axis} {r.get('value')} ({r.get('confidence')})" for axis, r in ratings.items())
                    + f"; overall {winner} {overall.get('strength', '')} by {overall.get('rule_applied', '')}",
        "reasons": reasons[:3],
        "true_for_both": true_for_both(result),
        "flagged": bool(result.get("needs_human_review")),
        "flags": flag_items(result),
        "leave_out": leave_out(result),
    }


def verdict_words(info):
    verdict, size = info["verdict"], info["size"]
    if verdict == "both-good":
        return "both good"
    if verdict == "both-bad":
        return "both bad"
    other = "B" if verdict == "A" else "A"
    return f"{verdict} is better than {other}" + (f" (pipeline size: {size}, so the first line says {' or '.join(SIZE_WORDS[size])})" if size else "")


def sentence_count_words(pair_type):
    fewest, most = SENTENCES[pair_type]
    return f"{fewest} sentences" if fewest == most else f"{fewest} or {most} sentences"


def format_points(info):
    lines = [f"Task: {info['task_id']}", f"Prompt: {info['prompt']}", "",
             f"Suggested verdict: {verdict_words(info)}",
             f"Kind of pair: {info['pair_type']}, {sentence_count_words(info['pair_type'])}"]
    lines += [f"  {w}" for w in info["why"]]
    lines += [f"Pipeline: {info['pipeline']}", "", "Deciding reasons (the prompt's key point leads):"]
    lines += [f"  {i}. {r}" for i, r in enumerate(info["reasons"], 1)] or ["  none found, look at the images"]
    if info["true_for_both"]:
        lines += ["", "True for both (name once if it matters):"] + [f"  {t}" for t in info["true_for_both"]]
    lines += ["", "Flagged for review: " + ("yes, so ask about doubtful items" if info["flagged"] else "no")]
    lines += [f"  {f}" for f in info["flags"]]
    if info["leave_out"]:
        lines += ["", "Leave out (dismissed, rejected or unverifiable):"] + [f"  {t}" for t in info["leave_out"]]
    return "\n".join(lines)


# ------------------------------------------------------------------ the shape pool and the pick log
def load_history(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_history(path, record):
    with Path(path).open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def shape_stats(history):
    stats = {}
    for rec in history:
        for name in rec.get("shown", []):
            stats.setdefault(name, {"shown": 0, "picked": 0})["shown"] += 1
        if rec.get("shape"):
            stats.setdefault(rec["shape"], {"shown": 0, "picked": 0})["picked"] += 1
    return stats


def eligible_shapes(pair_type, history):
    """Shapes for this kind of pair. One shown three times and never picked is retired, but never below three."""
    pool = [name for name, s in SHAPES.items() if pair_type in s["types"]]
    stats = shape_stats(history)

    def retired(name):
        s = stats.get(name, {"shown": 0, "picked": 0})
        return s["shown"] >= 3 and s["picked"] == 0

    keep = [n for n in pool if not retired(n)]
    if len(keep) < 3:
        back = sorted((n for n in pool if n not in keep), key=lambda n: stats[n]["shown"])
        keep += back[:3 - len(keep)]
    return keep


def pick_three(pair_type, history, seed):
    """Three different shapes, none repeated from the last run of this kind of pair when avoidable."""
    pool = eligible_shapes(pair_type, history)
    last = []
    for rec in reversed(history):
        if rec.get("pair_type") == pair_type:
            last = rec.get("shown", [])
            break
    rng = random.Random(seed)
    fresh = [n for n in pool if n not in last]
    rng.shuffle(fresh)
    chosen = fresh[:3]
    if len(chosen) < 3:
        rest = [n for n in pool if n not in chosen]
        rng.shuffle(rest)
        chosen += rest[:3 - len(chosen)]
    return chosen


# ------------------------------------------------------------------ the hard rules
def bad_characters(text):
    """Anything other than letters, digits, spaces, periods, commas, straight apostrophes and hyphens inside words."""
    found = []
    for i, ch in enumerate(text):
        if ch.isalnum() or ch in " .,'":
            continue
        if ch == "-" and 0 < i < len(text) - 1 and text[i - 1].isalnum() and text[i + 1].isalnum():
            continue
        found.append(ch)
    return found


def split_sentences(text):
    return [p for p in re.split(r"(?<=\.)\s+", text.strip()) if p]


def has_word(sentence, word):
    return re.search(r"\b" + re.escape(word) + r"\b", sentence, re.IGNORECASE) is not None


def first_line_problems(first, verdict, size):
    if verdict in ("both-good", "both-bad"):
        want = "Overall, both are good." if verdict == "both-good" else "Overall, both are bad."
        return [] if first == want else [f'the first line must be exactly "{want}"']
    other = "B" if verdict == "A" else "A"
    if size in SIZE_WORDS:
        pattern = rf"Overall, {verdict} is ({'|'.join(SIZE_WORDS[size])}) better than {other}\."
        shown = f"Overall, {verdict} is {' or '.join(SIZE_WORDS[size])} better than {other}."
    else:
        pattern = rf"Overall, {verdict} is better than {other}\."
        shown = f"Overall, {verdict} is better than {other}."
    return [] if re.fullmatch(pattern, first) else [f'the first line must be "{shown}"']


def lint_text(text, verdict, size, pair_type, avoid=()):
    """Returns (problems, warnings). Problems fail the draft, warnings are only noted."""
    problems, warnings = [], []
    text = text.strip()
    if not text:
        return ["the text is empty"], warnings
    if "\n" in text:
        problems.append("it must be one paragraph with no line breaks")
    bad = bad_characters(text)
    if bad:
        problems.append("characters you don't use: " + " ".join(sorted(set(bad))))
    if re.search(r",\s*and\b", text, re.IGNORECASE):
        problems.append('a comma before "and"')
    low = text.lower()
    for phrase in BANNED:
        if re.search(r"\b" + re.escape(phrase) + r"s?\b", low):
            problems.append(f'banned word or phrase: "{phrase}"')
    if re.search(r"\bI\b|\bI'[A-Za-z]+", text) or re.search(r"\b(?:me|my|mine|we|us|our)\b", low):
        problems.append("first person (the feedback never says I)")
    if not text.endswith("."):
        problems.append("it must end with a period")
    for word in avoid:
        if word and word.lower() in low:
            problems.append(f'mentions something to leave out: "{word}"')

    sentences = split_sentences(text)
    fewest, most = SENTENCES.get(pair_type, (1, 99))
    if not fewest <= len(sentences) <= most:
        want = str(fewest) if fewest == most else f"{fewest} or {most}"
        problems.append(f"expected {want} sentences for this kind of pair, found {len(sentences)}")
    if sentences:
        problems += first_line_problems(sentences[0], verdict, size)
    if verdict in ("A", "B") and len(sentences) >= 3:
        if pair_type == "clear_win":
            if not (has_word(sentences[1], "better") and has_word(sentences[1], "because")):
                problems.append('sentence 2 must say why one is "better" with "because"')
            if not (has_word(sentences[2], "worse") and has_word(sentences[2], "because")):
                problems.append('sentence 3 must say why the other is "worse" with "because"')
        elif pair_type == "both_flawed":
            if not has_word(sentences[1], "bad"):
                problems.append('sentence 2 must say both are "bad"')
            if not (has_word(sentences[2], "better") or "less bad" in sentences[2].lower()):
                problems.append('sentence 3 must say why one is "less bad" or the better pick')
    for phrase, note in SOFT:
        if phrase in low:
            warnings.append(f'"{phrase}": {note}')
    return problems, warnings


# ------------------------------------------------------------------ command line
def resolve(info, verdict=None, pair_type=None, size=None):
    """Apply what the user changed at the verdict check on top of the suggestion."""
    verdict = verdict or info["verdict"]
    if pair_type is None:
        if verdict == "both-good":
            pair_type = "tie_good"
        elif verdict == "both-bad":
            pair_type = "tie_bad"
        elif info["verdict"] in ("A", "B"):
            pair_type = info["pair_type"]
        else:
            pair_type = "clear_win"
    if size is None:
        size = info["size"] if verdict in ("A", "B") else None
    return verdict, pair_type, (None if size == "none" else size)


def run_points(args):
    info = analyze(load_result(args.result))
    print(json.dumps(info, indent=2, ensure_ascii=False) if args.json else format_points(info))
    return 0


def run_shapes(args):
    info = analyze(load_result(args.result))
    verdict, pair_type, _ = resolve(info, args.verdict, args.pair_type)
    chosen = pick_three(pair_type, load_history(args.history), args.seed or info["task_id"] or "feedback")
    print(f"Kind of pair: {pair_type}, verdict {verdict}. Three shapes:")
    for i, name in enumerate(chosen, 1):
        print(f"  {i}. {name}: {SHAPES[name]['how']}")
    return 0


def run_lint(args):
    info = analyze(load_result(args.result))
    verdict, pair_type, size = resolve(info, args.verdict, args.pair_type, args.size)
    avoid = [w.strip() for w in args.avoid.split(",") if w.strip()]
    texts = [line for line in args.file.read_text(encoding="utf-8").splitlines() if line.strip()]
    failed = 0
    for i, text in enumerate(texts, 1):
        problems, warnings = lint_text(text, verdict, size, pair_type, avoid)
        failed += bool(problems)
        print(f"{i}: " + ("FAIL" if problems else "ok"))
        for p in problems:
            print(f"   problem: {p}")
        for w in warnings:
            print(f"   note: {w}")
    return 1 if failed else 0


def run_log(args):
    text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8").strip()
    record = {"task_id": args.task, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "verdict": args.verdict, "size": None if args.size == "none" else args.size,
              "pair_type": args.pair_type, "shape": args.shape or None,
              "shown": [s for s in args.shown.split(",") if s], "text": text, "note": args.note}
    append_history(args.history, record)
    print(f"logged {args.task}: " + (f"picked {args.shape}" if args.shape else "none fit"))
    return 0


def run_pool(args):
    history = load_history(args.history)
    stats = shape_stats(history)
    for pair_type in PAIR_TYPES:
        keep = eligible_shapes(pair_type, history)
        print(pair_type)
        for name, s in SHAPES.items():
            if pair_type in s["types"]:
                st = stats.get(name, {"shown": 0, "picked": 0})
                print(f"  {name}: shown {st['shown']}, picked {st['picked']}" + ("" if name in keep else "  (retired)"))
    return 0


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Helpers for the feedback text that goes into the rating form.")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("points", help="the verdict check for a result file")
    p.add_argument("result", type=Path)
    p.add_argument("--json", action="store_true")
    p.set_defaults(run=run_points)

    p = sub.add_parser("shapes", help="three different shapes for this pair")
    p.add_argument("result", type=Path)
    p.add_argument("--verdict", choices=VERDICTS, help="the verdict you settled on, if you changed it")
    p.add_argument("--pair-type", choices=PAIR_TYPES)
    p.add_argument("--history", type=Path, default=HISTORY)
    p.add_argument("--seed")
    p.set_defaults(run=run_shapes)

    p = sub.add_parser("lint", help="check drafts, one per line of a file")
    p.add_argument("result", type=Path)
    p.add_argument("file", type=Path)
    p.add_argument("--verdict", choices=VERDICTS)
    p.add_argument("--pair-type", choices=PAIR_TYPES)
    p.add_argument("--size", choices=("slightly", "strongly", "none"))
    p.add_argument("--avoid", default="", help="comma separated words that must not appear")
    p.set_defaults(run=run_lint)

    p = sub.add_parser("log", help="record which candidate was picked")
    p.add_argument("--task", required=True)
    p.add_argument("--verdict", required=True, choices=VERDICTS)
    p.add_argument("--size", default="none", choices=("slightly", "strongly", "none"))
    p.add_argument("--pair-type", required=True, choices=PAIR_TYPES)
    p.add_argument("--shape", help="the shape picked; leave out when none fit")
    p.add_argument("--shown", default="", help="comma separated shapes that were shown")
    p.add_argument("--text", help="the picked text")
    p.add_argument("--text-file", type=Path, help="a file holding the picked text")
    p.add_argument("--note", default="")
    p.add_argument("--history", type=Path, default=HISTORY)
    p.set_defaults(run=run_log)

    p = sub.add_parser("pool", help="how often each shape was shown and picked")
    p.add_argument("--history", type=Path, default=HISTORY)
    p.set_defaults(run=run_pool)

    args = ap.parse_args(argv)
    if args.command == "log" and args.text is None and args.text_file is None:
        ap.error("log needs --text or --text-file")
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
