"""Model providers behind one interface.

Every agent calls `provider.complete(system, parts, output_model, max_tokens, effort)`
and gets back a validated pydantic object. `parts` is a list of
{"type": "text", "text": ...} and {"type": "image", "media_type": ..., "data": <b64>}.

Providers:
- AnthropicProvider: Claude models via the Messages API with structured outputs.
- MockProvider: deterministic fake answers, for dry runs and tests (no API key).
- GeminiProvider: placeholder; see the class docstring.
"""
import json
import threading
import time

from pydantic import ValidationError


class RefusalError(Exception):
    """The model declined. Not retried."""


class RetryableError(Exception):
    """Transient failure (rate limit, overload, invalid output)."""


class TruncatedError(RetryableError):
    """The answer was cut off at max_tokens. The retry should be given more room."""


class Usage:
    """Call and token counts for one task.

    Several agent calls run on worker threads at once and all report here, so updates
    take a lock; an unlocked `+=` can lose counts.
    """

    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.log = []
        self._lock = threading.Lock()

    def add(self, agent, model, inp, out, seconds):
        with self._lock:
            self.calls += 1
            self.input_tokens += inp
            self.output_tokens += out
            self.log.append({"agent": agent, "model": model, "input_tokens": inp,
                             "output_tokens": out, "seconds": round(seconds, 2)})


class Provider:
    name = "base"

    def __init__(self, model):
        self.model = model

    def complete(self, *, agent, system, parts, output_model, max_tokens, effort, usage):
        raise NotImplementedError


# ------------------------------------------------------------------ Anthropic
class AnthropicProvider(Provider):
    """Claude 5.5 rules that matter here (from the Opus 5.5 migration guide):
    - no temperature/top_p/top_k and no prefill: both are rejected;
    - thinking is always on; `output_config.effort` controls depth; max_tokens covers both;
    - responses can start with thinking blocks, so read text blocks by type;
    - JSON output goes through `output_config.format` (structured outputs);
    - `stop_reason == "refusal"` must be handled.
    """
    name = "anthropic"
    _client = None
    _client_lock = threading.Lock()

    def __init__(self, model):
        super().__init__(model)
        with AnthropicProvider._client_lock:
            if AnthropicProvider._client is None:
                import anthropic
                AnthropicProvider._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def complete(self, *, agent, system, parts, output_model, max_tokens, effort, usage):
        import anthropic
        from anthropic import transform_schema

        content = []
        for p in parts:
            if p["type"] == "image":
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": p["media_type"], "data": p["data"]}})
            else:
                content.append({"type": "text", "text": p["text"]})

        start = time.time()
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": content}],
                output_config={
                    "effort": effort,
                    "format": {"type": "json_schema", "schema": transform_schema(output_model)},
                },
            )
        except (anthropic.RateLimitError, anthropic.APIConnectionError,
                anthropic.InternalServerError) as e:
            raise RetryableError(str(e)) from e

        usage.add(agent, self.model, resp.usage.input_tokens, resp.usage.output_tokens,
                  time.time() - start)

        if resp.stop_reason == "refusal":
            raise RefusalError(f"{agent}: model declined ({getattr(resp, 'stop_details', None)})")
        if resp.stop_reason == "max_tokens":
            raise TruncatedError(f"{agent}: hit max_tokens={max_tokens}; the retry gets more room "
                                 f"(or raise it in config.MAX_TOKENS)")

        text = "".join(b.text for b in resp.content if b.type == "text")
        try:
            return output_model.model_validate(json.loads(text))
        except (json.JSONDecodeError, ValidationError) as e:
            raise RetryableError(f"{agent}: invalid output: {e}") from e


# ------------------------------------------------------------------ Gemini
class GeminiProvider(Provider):
    """Placeholder. To add Gemini (e.g. for cheaper Describer/Inspector runs):
    1. pip install google-genai
    2. Implement `complete` with the current SDK's structured-output option
       (response schema = output_model's JSON schema) and image parts.
    3. Check the SDK docs at implementation time; its API changes often, so this
       is left unwritten rather than guessed.
    """
    name = "gemini"

    def complete(self, **kwargs):
        raise NotImplementedError("GeminiProvider is not implemented yet; see its docstring.")


# ------------------------------------------------------------------ Mock
class MockProvider(Provider):
    """Returns plausible fixed answers so the whole pipeline can run offline."""
    name = "mock"

    def complete(self, *, agent, system, parts, output_model, max_tokens, effort, usage):
        import schemas as s
        usage.add(agent, f"mock/{self.model}", 0, 0, 0.0)
        n = output_model.__name__
        if n == "Description":
            return s.Description(
                subjects=[s.SubjectSeen(name="badger", count=1, location="center")],
                actions=["badger kneels holding playing cards"], relationships=[],
                text=[], setting="forest clearing", lighting="soft daylight from left",
                style="photograph")
        if n == "RequirementList":
            return s.RequirementList(style_frame="photographic", interpretation=s.Interpretation(
                intent="A photo of a badger kneeling and fanning five cards in its paw.",
                implied_look="no particular look implied",
                acceptable_imperfections=["out-of-focus background"],
                deciding_question="Does the badger hold exactly five cards fanned in an outstretched paw?"),
                requirements=[
                s.Requirement(id="R1", text="a badger", type="subject", central=True),
                s.Requirement(id="R2", text="kneeling", type="action", central=False),
                s.Requirement(id="R3", text="five cards", type="count", central=False),
            ])
        if n == "Inspection":
            ins = self._mock_inspection(s)
            if agent.endswith("run2"):  # images were swapped; answer consistently
                ins = _swap_slots(ins)
            return ins
        if n == "Verdict":
            return s.Verdict(verdict="confirmed", observation="right paw has six digits", note="clear in crop")
        if n == "ArtifactReview":
            return s.ArtifactReview(verdict="ai_defect", reason="the paw is sharp and the extra digit is a structural error")
        if n == "Justification":
            # Read the payload the way the real Writer does, so a change to its shape breaks the dry run.
            payload = json.loads(parts[0]["text"].split("\n\n", 1)[1])
            tied = payload["tied_axes"]
            assert all(r["text"] for r in payload["requirements"])
            assert payload["interpretation"]["deciding_question"]
            assert all(len(v) == 2 for v in payload["inspection_reasons"].values())
            assert "reviewed_and_dismissed_as_ordinary_photographic_detail" in payload
            return s.Justification(
                instruction_following="A shows five cards fanned in the paw; B shows three.",
                visual_quality=("Tied: " + tied["visual_quality"]["why"] + "; both are sharp and evenly lit."
                                if "visual_quality" in tied else "A is sharper."),
                ai_artifacts="B's right paw has six digits; A's paws are intact.",
                ps2_style=("Neither image is more convincing as a PS2 game screenshot."
                           if "ps2_style" in payload["scored_axes"] else ""),
                overall="A wins two axes and one is tied, so A is preferred.")
        if n == "Audit":
            return s.Audit(passed=True, problems=[])
        raise ValueError(f"MockProvider has no answer for {n}")

    @staticmethod
    def _mock_inspection(s):
        box = s.BBox(x=0.4, y=0.4, w=0.2, h=0.2)
        ok = s.ReqCheck(status="met", evidence="visible")
        miss = s.ReqCheck(status="missed", evidence="three cards counted")
        return s.Inspection(
            requirement_results=[
                s.RequirementResult(id="R1", image_1=ok, image_2=ok),
                s.RequirementResult(id="R2", image_1=ok, image_2=ok),
                s.RequirementResult(id="R3", image_1=ok, image_2=miss),
            ],
            regions_checked=["image 1: hands: no issue", "image 2: hands: six fingers on right paw"],
            description_conflicts=[],
            findings=[s.Finding(image="2", region="hands", observation="right paw has six digits",
                                axis="ai_artifacts", severity="major", bbox=box)],
            calls=s.AxisCalls(
                instruction_following=s.AxisCall(winner="1", strength="slight", confidence="high", reason="card count"),
                visual_quality=s.AxisCall(winner="tie", strength="none", confidence="medium", reason="both sharp"),
                ai_artifacts=s.AxisCall(winner="1", strength="strong", confidence="high", reason="extra digit in 2"),
                ps2_style=s.AxisCall(winner="tie", strength="none", confidence="high", reason="not scored"),
            ))


def _swap_slots(ins):
    """Swap image 1 and image 2 throughout an Inspection (used by the mock for run 2)."""
    flip = {"1": "2", "2": "1", "tie": "tie"}
    d = ins.model_dump()
    for r in d["requirement_results"]:
        r["image_1"], r["image_2"] = r["image_2"], r["image_1"]
    for f in d["findings"]:
        f["image"] = flip[f["image"]]
    for c in d["calls"].values():
        c["winner"] = flip[c["winner"]]
    return type(ins).model_validate(d)


def make_provider(provider, model, mock=False):
    if mock:
        return MockProvider(model)
    return {"anthropic": AnthropicProvider, "gemini": GeminiProvider}[provider](model)
