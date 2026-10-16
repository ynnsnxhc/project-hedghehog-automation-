"""One function per agent. Each builds its inputs, calls its provider, returns a schema object.

What each agent is allowed to see is decided here, deliberately:
- Describer: one image, no prompt.
- Requirements: prompt and axes, no images. Also writes the interpretation of what the prompt really asks for.
- Inspector: both images as "Image 1"/"Image 2" (no A/B labels), prompt, interpretation, checklist, blind descriptions.
- Verifier: one claim, the full image, a crop, the matching crop of the other image.
- Artifact reviewer: one confirmed artifact finding, the same images, the prompt and the interpretation.
- Writer: final ratings, verified findings, the checklist with results, the interpretation, the reason for
  any tied axis, the flaws dismissed as ordinary photographic detail, and the inspections' own reasons
  (for whole-image qualities that no finding can show); no images.
- Auditor: the finished evaluation; no images.
"""
import json
import time
from pathlib import Path

import config
import imaging
import schemas as s
from providers import RefusalError, RetryableError, TruncatedError, make_provider

PROMPTS = Path(__file__).parent / "prompts"


def _prompt(name):
    return (PROMPTS / f"{name}.md").read_text(encoding="utf-8")


def _text(t):
    return {"type": "text", "text": t}


def _dump(obj):
    """JSON for a prompt. Keeps non-ASCII text as written, so transcribed text reaches the model intact."""
    return json.dumps(obj, indent=1, ensure_ascii=False)


def _call(agent, role, parts, output_model, usage, mock):
    provider_name, model, effort = config.AGENT_MODELS[role]
    provider = make_provider(provider_name, model, mock=mock)
    max_tokens = config.MAX_TOKENS[role]
    last = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            return provider.complete(agent=agent, system=_prompt(role), parts=parts,
                                     output_model=output_model, max_tokens=max_tokens,
                                     effort=effort, usage=usage)
        except RefusalError:
            raise
        except TruncatedError as e:
            # Asking again with the same limit would be cut off at the same place.
            last = e
            max_tokens = min(int(max_tokens * config.MAX_TOKENS_GROWTH),
                             max(config.MAX_TOKENS_CEILING, max_tokens))
        except RetryableError as e:
            last = e
            time.sleep(config.RETRY_BACKOFF_SECONDS * attempt * (0 if mock else 1))
    raise RuntimeError(f"{agent} failed after {config.MAX_RETRIES} attempts: {last}")


# ------------------------------------------------------------------ agents
def describe(img_part, label, usage, mock=False):
    parts = [img_part, _text("Describe this image.")]
    return _call(f"describer_{label}", "describer", parts, s.Description, usage, mock)


def requirements(prompt, axes, usage, mock=False):
    parts = [_text(f"Scored axes: {', '.join(axes)}\n\nPrompt:\n<prompt>\n{prompt}\n</prompt>")]
    out = _call("requirements", "requirements", parts, s.RequirementList, usage, mock)
    if "ps2_style" in axes:
        out.style_frame = "ps2"  # the rule is mechanical, so don't leave it to the model
    return out


def inspect(prompt, axes, reqs, first, second, desc_first, desc_second, run, usage, mock=False):
    parts = [
        _text("Image 1:"), first,
        _text("Image 2:"), second,
        _text(
            f"Prompt:\n<prompt>\n{prompt}\n</prompt>\n\n"
            f"Scored axes: {', '.join(axes)}\n"
            f"Style frame: {reqs.style_frame}\n\n"
            f"Interpretation of the prompt:\n{reqs.interpretation.model_dump_json(indent=1)}\n\n"
            f"Requirement checklist:\n{reqs.model_dump_json(indent=1, exclude={'interpretation'})}\n\n"
            f"Blind description of Image 1 (written without the prompt):\n{desc_first.model_dump_json(indent=1)}\n\n"
            f"Blind description of Image 2 (written without the prompt):\n{desc_second.model_dump_json(indent=1)}"),
    ]
    return _call(f"inspector_run{run}", "inspector", parts, s.Inspection, usage, mock)


def _claim_images(finding, images, parts):
    """The full image, then a crop of the cited region and the same region of the other image."""
    box = s.BBox(**finding["bbox"])
    own = finding["image"]
    other = "B" if own == "A" else "A"
    content = [_text("Full image the claim is about:"), parts[own]]
    if imaging.bbox_ok(box):
        content += [
            _text("Zoomed crop of the region the claim refers to:"), imaging.encode(imaging.crop(images[own], box)),
            _text("Same region in the other image, for comparison only:"), imaging.encode(imaging.crop(images[other], box)),
        ]
    else:
        # A box outside 0-1 would crop the wrong place and make a true claim look false.
        content.append(_text("No usable crop is available for this claim. Judge it from the full image, "
                             "and answer unverifiable if you can't tell."))
    return content


def verify(finding, images, parts, usage, mock=False):
    """images = {"A": PIL.Image, "B": PIL.Image}; parts = the same two, already encoded."""
    content = _claim_images(finding, images, parts)
    content.append(_text(f"Claim about the full image ({finding['region']}): {finding['observation']}"))
    return _call("verifier", "verifier", content, s.Verdict, usage, mock)


def review_artifact(finding, images, parts, prompt, reqs, usage, mock=False):
    """Second opinion on a confirmed ai_artifacts finding: a real generative defect, or ordinary photographic detail?"""
    content = _claim_images(finding, images, parts)
    content.append(_text(
        f"Prompt:\n<prompt>\n{prompt}\n</prompt>\n\n"
        f"Style frame: {reqs.style_frame}\n"
        f"Interpretation of the prompt:\n{reqs.interpretation.model_dump_json(indent=1)}\n\n"
        f"Reported as an AI artifact ({finding['region']}): {finding['observation']}"))
    return _call("artifact_reviewer", "artifact_reviewer", content, s.ArtifactReview, usage, mock)


def write(prompt, axes, ratings, overall, findings, requirements, tied_axes, interpretation, inspection_reasons,
          usage, mock=False):
    """requirements = checklist rows with text, type, central and each image's result.
    tied_axes = {axis: {"why", "inspector_reasons"}} for every axis that ended as a tie.
    interpretation = how the prompt was read (a dict).
    inspection_reasons = {axis: [first inspection's reason, second inspection's reason]}, with A/B labels."""
    drop = {"bbox", "seen_in_runs"}
    usable = [f.model_dump(exclude=drop) for f in findings
              if f.verdict in ("confirmed", "not_checked") and f.artifact_review != "natural"]
    dismissed = [{"image": f.image, "observation": f.observation, "why": f.artifact_note}
                 for f in findings if f.artifact_review == "natural"]
    payload = {
        "prompt": prompt,
        "interpretation": interpretation,
        "scored_axes": axes,
        "ratings": {k: v.model_dump() for k, v in ratings.items()},
        "overall": overall.model_dump(),
        "requirements": requirements,
        "tied_axes": tied_axes,
        "inspection_reasons": {axis: inspection_reasons[axis] for axis in axes},
        "verified_findings": usable,
        "reviewed_and_dismissed_as_ordinary_photographic_detail": dismissed,
    }
    parts = [_text("Write the justification for this evaluation.\n\n" + _dump(payload))]
    return _call("writer", "writer", parts, s.Justification, usage, mock)


def audit(result_json, usage, mock=False):
    keep = {k: result_json[k] for k in ("prompt", "interpretation", "axes", "ratings", "overall", "requirements",
                                        "findings", "justification", "order_swap", "axis_reasons")}
    keep["findings"] = [{k: v for k, v in f.items() if k not in ("bbox", "seen_in_runs")}
                        for f in keep["findings"]]
    parts = [_text("Audit this evaluation.\n\n" + _dump(keep))]
    return _call("auditor", "auditor", parts, s.Audit, usage, mock)
