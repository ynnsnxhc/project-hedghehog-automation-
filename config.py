"""Central configuration: which model runs each agent, and run-level settings.

Change models here, not in agent code. Each entry is (provider, model_id, effort).
Effort applies to Claude 5.5 models (low | medium | high | xhigh | max); other
providers ignore it.
"""

INSTRUCTION_SET_VERSION = "2026-10-15"
EVALUATOR_LABEL = "automated"  # written into every result; never remove

AGENT_MODELS = {
    "describer":    ("anthropic", "claude-sonnet-5-5", "medium"),
    # Reads the prompt for what it really asks for, which every later stage judges against.
    "requirements": ("anthropic", "claude-sonnet-5-5", "medium"),
    "inspector":    ("anthropic", "claude-opus-5-5",   "high"),
    "verifier":     ("anthropic", "claude-opus-5-5",   "high"),
    # Second opinion: is a confirmed artifact a generative defect, or ordinary photographic detail?
    "artifact_reviewer": ("anthropic", "claude-opus-5-5", "medium"),
    "writer":       ("anthropic", "claude-sonnet-5-5", "medium"),
    "auditor":      ("anthropic", "claude-sonnet-5-5", "medium"),
}

# max_tokens covers thinking + answer on 5.5 models (thinking is always on).
MAX_TOKENS = {
    "describer": 8000,
    "requirements": 6000,
    "inspector": 16000,
    "verifier": 6000,
    "artifact_reviewer": 6000,
    "writer": 6000,
    "auditor": 6000,
}

# When a call is cut off at max_tokens, the retry gets more room instead of repeating
# the same request. The SDK refuses non-streaming requests above ~21,300 tokens, so the
# ceiling stays under that.
MAX_TOKENS_GROWTH = 1.5
MAX_TOKENS_CEILING = 20000

# Images: Claude 4.7+ reads up to 2576px on the long edge. Larger is downscaled
# by the API anyway, so we resize first to control cost and keep crops consistent.
MAX_IMAGE_EDGE = 2576
MAX_IMAGE_BYTES = 3_600_000  # per encoded image; base64 adds a third, and the API caps images at 5 MB
JPEG_QUALITIES = (95, 90, 85, 75)  # tried in order until the image fits MAX_IMAGE_BYTES

# Crops sent to the Verifier. Inspector boxes are approximate, so a crop gets generous
# padding and is never smaller than MIN_CROP_FRACTION of the frame in either direction.
CROP_PADDING = 0.25          # extra context around a cited region (fraction of box size)
MIN_CROP_FRACTION = 0.15     # smallest crop side, as a fraction of the image side
MIN_CROP_EDGE = 384          # crops are upscaled to at least this so small details resolve

MAX_RETRIES = 3              # per agent call, for transient API / validation errors
RETRY_BACKOFF_SECONDS = 4

# Every finding that could decide a rating is sent to the Verifier. This caps how
# many per task (major findings first); the rest are marked unverified.
MAX_FINDINGS_TO_VERIFY = 12

# Two findings on the same image, axis and region are the same defect when their boxes
# overlap by at least this much (intersection over union). Distinct defects in one
# region (two hands, say) stay separate.
FINDING_MERGE_IOU = 0.3
