"""Turns a results folder into a readable report.

    python report.py my_tests/results                 # writes my_tests/results/report.md
    python report.py my_tests/results --print         # also prints it
    python report.py my_tests/results --only man_001  # one task
    python report.py my_tests/results --out summary.md

Reads the <task_id>.json files the pipeline writes, so it works on any finished run.
"""
import argparse
import sys
from pathlib import Path

from orchestrator import load_results

AXIS_NAMES = {"instruction_following": "Instruction following", "visual_quality": "Visual quality",
              "ai_artifacts": "AI artifacts", "ps2_style": "PS2 style"}
VALUE_WORDS = {"strongly_A": "**A**, strongly", "slightly_A": "**A**, slightly", "tie": "tie",
               "slightly_B": "**B**, slightly", "strongly_B": "**B**, strongly"}
REVIEW_WORDS = {"ai_defect": "a real AI defect", "natural": "ordinary photographic detail, not counted",
                "unclear": "could not tell"}
STATUS_MARK = {"met": "met", "partial": "partial", "missed": "MISSED", "unverifiable": "unverifiable"}


def cell(text):
    """Make text safe inside a Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def overall_line(r):
    ov = r["overall"]
    if ov["winner"] == "unresolved":
        return "**No winner**: the axis wins split evenly and instruction following is tied."
    return f"**Image {ov['winner']} wins, {ov['strength']}.** {ov['strength_basis']}"


def task_section(r):
    out = [f"## {r['task_id']}", "", f"**Prompt:** {r['prompt']}", ""]
    out += [overall_line(r), ""]

    if r["needs_human_review"]:
        out += ["### Needs human review", ""]
        out += [f"- {reason}" for reason in r["review_reasons"]]
        out.append("")

    reading = r.get("interpretation") or {}
    if reading:
        out += ["### How the prompt was read", "",
                f"**What it asks for.** {reading.get('intent', '')}", "",
                f"**Look it implies.** {reading.get('implied_look', '')}", ""]
        normal = reading.get("acceptable_imperfections") or []
        if normal:
            out += [f"**Normal for that look, so not held against an image.** {'; '.join(normal)}", ""]
        out += [f"**Deciding question.** {reading.get('deciding_question', '')}", ""]

    out += ["### Ratings", "", "| Axis | Winner | Confidence |", "|---|---|---|"]
    for axis in r["axes"]:
        rating = r["ratings"][axis]
        out.append(f"| {AXIS_NAMES.get(axis, axis)} | {VALUE_WORDS[rating['value']]} | {rating['confidence']} |")
    out.append("")

    j = r.get("justification")
    if j:
        out += ["### Why", ""]
        for axis in r["axes"]:
            if j.get(axis, "").strip():
                out += [f"**{AXIS_NAMES.get(axis, axis)}.** {j[axis].strip()}", ""]
        if j.get("overall", "").strip():
            out += [f"**Overall.** {j['overall'].strip()}", ""]

    flipped = r["order_swap"]["flipped_axes"]
    if flipped:
        out += ["### Where the two inspections disagreed", ""]
        for axis in flipped:
            first, second = r.get("axis_reasons", {}).get(axis, ["", ""])
            out += [f"- **{AXIS_NAMES.get(axis, axis)}**", f"  - first inspection: {first}",
                    f"  - second inspection (images swapped): {second}"]
        out.append("")

    out += ["### Requirements from the prompt", "", "| | Requirement | A | B |", "|---|---|---|---|"]
    for q in r["requirements"]:
        a, b = q.get("A") or {}, q.get("B") or {}
        out.append(f"| {'central' if q['central'] else ''} | {cell(q['text'])} | "
                   f"{STATUS_MARK.get(a.get('status'), '?')} | {STATUS_MARK.get(b.get('status'), '?')} |")
    out.append("")

    findings = r["findings"]
    out += ["### Findings", ""]
    if not findings:
        out += ["None recorded.", ""]
    for image in ("A", "B"):
        mine = [f for f in findings if f["image"] == image]
        if not mine:
            continue
        out += [f"**Image {image}**", ""]
        for f in mine:
            box = f["bbox"]
            where = f"x {box['x']:.2f}, y {box['y']:.2f}, w {box['w']:.2f}, h {box['h']:.2f}"
            note = f" Verifier: {f['verifier_note']}" if f.get("verifier_note") else ""
            second = REVIEW_WORDS.get(f.get("artifact_review", ""), "")
            if second:
                note += f" Second opinion: {second}" + (f" ({f['artifact_note']})" if f.get("artifact_note") else "")
            out += [f"- [{f['severity']}, {f['verdict']}] {AXIS_NAMES.get(f['axis'], f['axis'])}, {f['region']} "
                    f"({where}): {f['observation']}{note}"]
        out.append("")

    if r.get("description_conflicts"):
        out += ["### Disagreements with the blind descriptions", ""]
        out += [f"- {c}" for c in r["description_conflicts"]]
        out.append("")

    audit = r.get("audit") or {}
    cost = r.get("cost", {})
    out += ["### Checks", "",
            f"- Audit: {'passed' if audit.get('passed') else 'FAILED'}"
            + "".join(f"\n  - {p}" for p in audit.get("problems", [])),
            f"- Calls: {cost.get('calls', '?')}, tokens in/out: {cost.get('input_tokens', '?')} / "
            f"{cost.get('output_tokens', '?')}",
            f"- Run: {r['created_at']}, instruction set {r['instruction_set_version']}", ""]
    return "\n".join(out)


def build_report(results):
    wins = {"A": 0, "B": 0, "unresolved": 0}
    for r in results:
        wins[r["overall"]["winner"]] += 1
    flagged = sum(r["needs_human_review"] for r in results)
    out = ["# T2I evaluation report", "",
           f"{len(results)} task(s): A wins {wins['A']}, B wins {wins['B']}, no winner {wins['unresolved']}. "
           f"{flagged} flagged for human review. All results are labelled automated.", ""]
    if len(results) > 1:
        out += ["| Task | Overall | Review |", "|---|---|---|"]
        for r in results:
            ov = r["overall"]
            winner = "none" if ov["winner"] == "unresolved" else f"{ov['winner']} ({ov['strength']})"
            out.append(f"| {r['task_id']} | {winner} | {'yes' if r['needs_human_review'] else ''} |")
        out.append("")
    out += [task_section(r) for r in results]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Readable report from a results folder.")
    ap.add_argument("results", type=Path, help="the folder the pipeline wrote (has the <task_id>.json files)")
    ap.add_argument("--only", nargs="*", help="task_ids to include")
    ap.add_argument("--out", type=Path, help="report file (default: report.md inside the results folder)")
    ap.add_argument("--print", action="store_true", help="also print the report")
    args = ap.parse_args(argv)

    if not args.results.is_dir():
        print(f"error: {args.results} is not a folder", file=sys.stderr)
        return 2
    results = load_results(args.results)
    if args.only:
        results = [r for r in results if r["task_id"] in set(args.only)]
    if not results:
        print(f"error: no results found in {args.results}", file=sys.stderr)
        return 2

    text = build_report(results)
    path = args.out or args.results / "report.md"
    path.write_text(text, encoding="utf-8", newline="\n")
    if args.print:
        try:
            sys.stdout.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
        print(text)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
