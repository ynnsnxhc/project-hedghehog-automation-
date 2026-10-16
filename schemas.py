"""Data shapes for every agent's output and for the final result.

Agent output models are sent to the API as JSON schemas (structured outputs), so:
- every field is required (optional fields count against the API's schema limits),
- no numeric min/max constraints (the API doesn't enforce them; we check in code).

Pydantic builds each model from its field declarations (`name: type`), so those stay.
Nothing else in this project uses type annotations.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------- shared enums
ImageSlot = Literal["1", "2"]           # what agents see; code maps to A/B
Axis = Literal["instruction_following", "visual_quality", "ai_artifacts", "ps2_style"]
Severity = Literal["major", "minor"]
Confidence = Literal["high", "medium", "low"]
ReqStatus = Literal["met", "partial", "missed", "unverifiable"]
ReqType = Literal["subject", "count", "attribute", "action", "relationship",
                  "text", "setting", "style", "framing"]

SCORED_AXES = ("instruction_following", "visual_quality", "ai_artifacts", "ps2_style")


# ---------------------------------------------------------------- Describer
class SubjectSeen(BaseModel):
    name: str
    count: int
    location: str = Field(description="Where in the frame, e.g. 'left foreground'")


class TextSeen(BaseModel):
    location: str
    transcription: str = Field(description="Exactly what is visible, character by character. 'UNREADABLE' if it can't be read.")
    legible: bool


class Description(BaseModel):
    subjects: list[SubjectSeen]
    actions: list[str]
    relationships: list[str]
    text: list[TextSeen]
    setting: str
    lighting: str = Field(description="Light direction, softness, color temperature")
    style: str = Field(description="Medium/aesthetic, e.g. 'photograph', 'cartoon', 'PS2-era 3D render'")


# ---------------------------------------------------------------- Requirements
class Requirement(BaseModel):
    id: str = Field(description="R1, R2, ...")
    text: str
    type: ReqType
    central: bool


class Interpretation(BaseModel):
    """What the prompt is really asking for. The Inspectors judge against this."""
    intent: str = Field(description="What the person who wrote the prompt wants to see, in one or two plain sentences")
    implied_look: str = Field(description="The look or quality the wording asks for, e.g. 'candid phone-snapshot realism, not polished stock photography', or 'no particular look implied'")
    acceptable_imperfections: list[str] = Field(description="Things that are normal for that look and must not count against an image as quality flaws or AI tells")
    deciding_question: str = Field(description="The one question that most separates a good answer to this prompt from a bad one")


class RequirementList(BaseModel):
    style_frame: Literal["photographic", "non_photo", "ps2"]
    interpretation: Interpretation
    requirements: list[Requirement]


# ---------------------------------------------------------------- Inspector
class ReqCheck(BaseModel):
    status: ReqStatus
    evidence: str


class RequirementResult(BaseModel):
    id: str
    image_1: ReqCheck
    image_2: ReqCheck


class BBox(BaseModel):
    """Normalized 0-1 coordinates of the region a finding refers to."""
    x: float
    y: float
    w: float
    h: float


class Finding(BaseModel):
    image: ImageSlot
    region: str = Field(description="faces, hands, bodies, clothing, objects, text, light, surfaces, background, whole_frame")
    observation: str
    axis: Axis
    severity: Severity
    bbox: BBox


class AxisCall(BaseModel):
    winner: Literal["1", "2", "tie"]
    strength: Literal["strong", "slight", "none"] = Field(description="'none' only with a tie")
    confidence: Confidence
    reason: str


class AxisCalls(BaseModel):
    instruction_following: AxisCall
    visual_quality: AxisCall
    ai_artifacts: AxisCall
    ps2_style: AxisCall = Field(description="If PS2 style is not scored: winner tie, strength none, reason 'not scored'")


class Inspection(BaseModel):
    requirement_results: list[RequirementResult]
    regions_checked: list[str] = Field(description="Every region swept, as 'image N: region: finding or no issue'")
    description_conflicts: list[str] = Field(description="Places where your view disagrees with the blind descriptions, and what you concluded")
    findings: list[Finding]
    calls: AxisCalls


# ---------------------------------------------------------------- Verifier
class Verdict(BaseModel):
    verdict: Literal["confirmed", "rejected", "unverifiable"]
    observation: str = Field(description="The observation as it actually holds up (corrected if needed)")
    note: str


# ---------------------------------------------------------------- Artifact reviewer
class ArtifactReview(BaseModel):
    verdict: Literal["ai_defect", "natural", "unclear"]
    reason: str = Field(description="Which of focus, size or structure decided it, in one or two sentences")


# ---------------------------------------------------------------- Writer
class Justification(BaseModel):
    instruction_following: str
    visual_quality: str
    ai_artifacts: str
    ps2_style: str = Field(description="Empty string if PS2 style is not scored")
    overall: str


# ---------------------------------------------------------------- Auditor
class Audit(BaseModel):
    passed: bool
    problems: list[str]


# ---------------------------------------------------------------- final result (not sent to the API)
class AxisRating(BaseModel):
    value: Literal["strongly_A", "slightly_A", "tie", "slightly_B", "strongly_B"]
    confidence: Confidence


class VerifiedFinding(BaseModel):
    image: Literal["A", "B"]
    region: str
    bbox: BBox
    observation: str
    axis: Axis
    severity: Severity
    verdict: Literal["confirmed", "rejected", "unverifiable", "not_checked"]
    verifier_note: str = ""
    artifact_review: Literal["", "ai_defect", "natural", "unclear"] = ""   # second opinion, ai_artifacts findings only
    artifact_note: str = ""
    seen_in_runs: list[int]


class Overall(BaseModel):
    winner: Literal["A", "B", "unresolved"]
    strength: Literal["strongly", "slightly", "none"]
    wins: dict[str, int]
    rule_applied: Literal["count", "instruction_following_tiebreak", "unresolved"]
    strength_basis: str


class Result(BaseModel):
    evaluator: Literal["automated"] = "automated"
    task_id: str
    instruction_set_version: str
    models: dict[str, str]
    created_at: str
    prompt: str
    axes: list[str]
    style_frame: str
    interpretation: dict = {}                     # how the prompt was read: intent, implied look, acceptable imperfections
    requirements: list[dict]
    findings: list[VerifiedFinding]
    ratings: dict[str, AxisRating]
    overall: Overall
    justification: Optional[Justification]
    order_swap: dict[str, list[str]]
    axis_reasons: dict[str, list[str]] = {}       # what each of the two inspections said, per axis (A/B labels)
    description_conflicts: list[str] = []         # where an inspector disagreed with a blind description
    audit: Optional[Audit]
    needs_human_review: bool
    review_reasons: list[str]
    cost: dict[str, int]
