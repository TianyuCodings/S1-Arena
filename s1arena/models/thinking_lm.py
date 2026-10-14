"""Local models that think within a token budget before their answer is read, at several budgets from one thought.

thinklm: is lm: with think=B, extended to Gemma 4's thought format.
- **Format.** The model writes up to B tokens of reasoning in its own thought format: Qwen's <think> block or Gemma 4's
  thought channel (<|channel>thought ... <channel|>). lm: knows only the <think> block.
- **Opening and sampling.** The thought is opened in the prompt, so every question gets one. It is sampled with the
  model's generation config, seeded per sample; lm: always uses Qwen3's settings.
- **The cut.** A thought still open after B tokens is cut there and closed as the chat template closes one. The answer
  labels are then read right after it, as in lm:.
- **think=0** reads the answer after an empty thought under the thinking prompt. This zero of the budget keeps the
  thinking prompt, which for some models differs from lm: think=0 (Qwen3.8 adds a reasoning-effort line, Gemma 4 a
  <|think|> flag).

thinkdgemma: is dgemma: with think=B, answered from the first token the model writes after its closed thought.
- **dgemma: with think > 0** puts the answer slot first on the canvas. Its read of the row before the slot wraps around
  to the canvas's last row, so the labels get almost no probability.
- **The slot read does not work after a thought either.** Pinning the close tag first on the canvas, as the think=0
  layout of dgemma: does, leaves a median label mass of 0.002 on 93 pilot questions, with the close tag itself on top.
- **Here the prompt ends with the closed thought**, and the model writes one answer token.
  - Its distribution over the labels is the answer.
  - The labels then get a median mass of 1.000; without a thought, this read and the slot read agree (accuracy 0.806
    and 0.817 on the same questions).
  - The answer block starts from a seeded random canvas, so the read repeats and a label set read in chunks of 128 ids
    shares one canvas.
- The thought is written as dgemma: writes it, in canvas blocks of 16 tokens and without seeds.
- think=0 answers the same way after an empty thought under the thinking prompt; with mode=nothink, after the empty
  thought of the non-thinking prompt, the prompt of dgemma: think=0.

For both:
- usage logs per question how many thought tokens were kept and whether the model ended the thought itself.
- decide_budgets answers requests at several budgets from one thought per question, written up to the largest budget;
  the thought at budget B is its first B tokens.
  - An autoregressive model's first B tokens do not depend on how many may follow, nor do DiffusionGemma's first
    blocks. So each budget is distributed as its own run, and the budgets differ only in where the thought is cut.
  - A budget the prompt leaves no room for is rejected, as lm: and dgemma: reject it.
  - The largest budget's rows keep the thoughts as raw_text: a JSON object, question -> thought.
- scripts/run_think_budgets.py runs a model this way and writes one run per budget.

Specs: thinklm:<Hugging Face id>@think=<tokens>[,tp=...,maxlen=...,gpu=...] (think=0 allowed)
       thinkdgemma:<Hugging Face id>@think=<tokens>[,reads=...,maxlen=...,gpu=...,mode=nothink] (think=0 allowed)
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict

from ..registry import register_model
from ..schema import Answer, ModelResponse
from .diffusion_gemma import CANVAS, THOUGHT_CLOSE, THOUGHT_OPEN, DiffusionGemma
from .local_lm import MAX_TOKEN_IDS, REJECTED, THINK_SAMPLING, LocalLM, parse_spec

FORMATS = {  # how a chat template opens and closes a thought, and the token that ends one
    "qwen": ("<think>\n", "\n</think>\n\n", "</think>"),
    "gemma": ("<|channel>thought\n", "\n<channel|>", "<channel|>"),
}


class Budgets:
    """Answers at several thinking budgets from one thought per question; the model class writes and closes thoughts."""

    def decide_many(self, jobs):
        return self.decide_budgets(jobs, [self.think])[self.think]

    def decide_budgets(self, jobs, budgets: list[int]) -> dict[int, list[ModelResponse]]:
        """Responses per budget. batch_ms_per_request counts the shared thought and the reads of every budget."""
        self._load()
        started = time.time()
        groups = {(j, name): self._units(sample, request, question)
                  for j, (request, sample) in enumerate(jobs) for name, question in request.questions.items()}
        units = [u for group in groups.values() for u in group]
        for unit in units:
            if "error" not in unit:
                unit["prompt_ids"] = self._encode(self._opened(unit))
                unit["room"] = self._context - len(unit["prompt_ids"]) - self._reserve
        todo = [u for u in units if "error" not in u and u["room"] >= 0]
        if max(budgets) > 0:
            self._write([u for u in todo if u["room"] > 0], max(budgets))
        for unit in todo:
            unit.setdefault("thought", [])
            unit.setdefault("ended", True)
        cuts = {b: {key: [self._cut(u, b) for u in group] for key, group in groups.items()} for b in budgets}
        # one pass of reads for every budget, a question's budgets next to each other: they share its prompt in vLLM's
        # prefix cache
        self._score([cuts[b][key][k] for key, group in groups.items() for k in range(len(group)) for b in budgets])
        ms = (time.time() - started) * 1000 / max(1, len(jobs))
        return {b: self._responses(jobs, cuts[b], ms, keep_thoughts=b == max(budgets)) for b in budgets}

    def _cut(self, unit: dict, budget: int) -> dict:
        """A copy of the unit whose thought is cut to `budget` tokens, closed and ready to score."""
        cut = {k: v for k, v in unit.items() if k != "thought"}
        if "error" in unit:
            return cut
        if unit["room"] < budget:
            cut["error"] = REJECTED + f"prompt longer than the {self._context}-token context"
            return cut
        thought = unit["thought"][:budget]
        cut["think_tokens"], cut["ended"] = len(thought), unit["ended"] and len(unit["thought"]) <= budget
        cut["text"] = self._tok.decode(thought, skip_special_tokens=True)
        self._close_thought(cut)
        return cut

    def _generate_thoughts(self, units: list[dict], budget: int, sampling) -> None:
        """Let the model write each unit's thought after its opened prompt, up to `budget` tokens."""
        from vllm import SamplingParams

        params = [SamplingParams(max_tokens=min(budget, u["room"]), stop_token_ids=[self._end], **sampling(u))
                  for u in units]
        for unit, out in zip(units, self._generate([u["prompt_ids"] for u in units], params)):
            ids, ended = list(out.outputs[0].token_ids), out.outputs[0].finish_reason == "stop"
            unit["thought"], unit["ended"] = ids[:-1] if ended and ids else ids, ended  # drop the end token

    def _responses(self, jobs, groups, ms: float, keep_thoughts: bool) -> list[ModelResponse]:
        responses = []
        for j, (request, _) in enumerate(jobs):
            answers, errors, thoughts = {}, [], {}
            usage = {"think_tokens": 0, "candidate_mass": {}, "thought": {}}
            for name, question in request.questions.items():
                group = groups[(j, name)]
                probs, error = self._combine(group, question)
                if error:
                    errors.append((name, error))
                    continue
                answers[name] = Answer(probs=probs)
                unit = group[0]
                usage["think_tokens"] += unit["think_tokens"]
                usage["candidate_mass"][name] = round(unit["mass"], 4)
                usage["thought"][name] = {"tokens": unit["think_tokens"], "ended": unit["ended"]}
                thoughts[name] = unit["text"]
            error = None
            if errors:
                final = all(e.startswith(REJECTED) for _, e in errors)
                error = (REJECTED if final else "") + "; ".join(f"{n}: {e}" for n, e in errors)[:300]
            usage["batch_ms_per_request"] = round(ms, 1)
            raw = json.dumps(thoughts, ensure_ascii=False) if keep_thoughts and any(thoughts.values()) else None
            responses.append(ModelResponse(answers=answers, usage=usage, error=error, raw_text=raw))
        return responses


@register_model
class ThinkingLM(Budgets, LocalLM):
    kind = "thinklm"
    _reserve = 64  # tokens kept free after the thought, as lm: keeps them

    def __init__(self, arg: str = "Qwen/Qwen3-8B@think=256") -> None:
        super().__init__(arg)
        if self.mode != "label" or self.think < 0:
            raise ValueError(f"thinklm needs mode=label and think >= 0: {arg!r}")

    @property
    def model_id(self) -> str:
        return f"{self.hf_id}@label-budget{self.think}"

    def _load(self) -> None:
        if self._llm is not None:
            return
        super()._load()
        template = self._tok.chat_template or ""
        style = "gemma" if "<|channel>thought" in template else "qwen" if "<think>" in template else None
        if style is None or not self._thinking:
            raise ValueError(f"{self.hf_id} has no thinking mode in its chat template")
        self._style = style
        self._opening, self._closing, end = FORMATS[style]
        self._end = self._tok.convert_tokens_to_ids(end)
        config = getattr(self._llm, "model_config", None) or self._llm.llm_engine.model_config
        config = config.get_diff_sampling_param()
        self._sampling = {k: config[k] for k in ("temperature", "top_p", "top_k") if k in config} or THINK_SAMPLING

    def _opened(self, unit: dict) -> str:
        prompt = self._prompt(unit["system"], unit["user"], thinking=True)
        unit["prompt"] = prompt if prompt.endswith(self._opening) else prompt + self._opening
        return unit["prompt"]

    def _write(self, units: list[dict], budget: int) -> None:
        self._generate_thoughts(units, budget, lambda u: {"seed": 1000 + u["sample"], **self._sampling})

    def _close_thought(self, cut: dict) -> None:
        thought = cut["text"].rstrip()
        closing = self._closing if thought or self._style == "qwen" else self._closing.lstrip("\n")  # Gemma's empty one
        cut["prefix"] = cut["prompt"] + thought + closing
        cut["prefix_ids"] = self._encode(cut["prefix"])


@register_model
class ThinkingDiffusionGemma(Budgets, DiffusionGemma):
    kind = "thinkdgemma"
    _reserve = CANVAS + 8  # the answer block and a margin, as dgemma: keeps them

    def __init__(self, arg: str = "google/diffusiongemma-26B-A4B-it@think=256") -> None:
        super().__init__(arg)
        self.nothink = parse_spec(arg)[1].get("mode") == "nothink"
        if self.think < 0 or (self.nothink and self.think):
            raise ValueError(f"thinkdgemma needs think >= 0, and think=0 with mode=nothink: {arg!r}")

    @property
    def model_id(self) -> str:
        suffix = "nothink" if self.nothink else f"budget{self.think}"
        return f"{self.hf_id}@reads{self.reads}-{suffix}-firsttoken"

    def _load(self) -> None:
        if self._llm is not None:
            return
        super()._load()
        self._end = self._close[0]

    def _opened(self, unit: dict) -> str:
        unit["prompt"] = self._prompt(unit["system"], unit["user"], thinking=not self.nothink) + THOUGHT_OPEN
        return unit["prompt"]

    def _write(self, units: list[dict], budget: int) -> None:
        self._generate_thoughts(units, budget, lambda u: {})  # no seeds for diffusion, as dgemma:

    def _close_thought(self, cut: dict) -> None:
        thought = cut["text"].rstrip()
        cut["prefix_ids"] = self._encode(cut["prompt"] + (thought + "\n" if thought else "") + THOUGHT_CLOSE)

    def _score(self, units: list[dict]) -> None:
        """Label log-probabilities of the first answer token, from a seeded canvas per (sample, read)."""
        from vllm import SamplingParams

        reads, owners = [], []
        for unit in (u for u in units if "error" not in u):
            ids = [self._encode(label) for label in unit["labels"]]
            if any(len(t) != 1 for t in ids) or len({t[0] for t in ids}) != len(ids):
                unit["error"] = REJECTED + "labels are not distinct single tokens"
                continue
            unit["label_ids"] = [t[0] for t in ids]
            for r in range(self.reads):
                rng = random.Random(f"{unit['sample']}-{r}")
                extra = {"diffusion_seed_canvas": [rng.randrange(self._vocab) for _ in range(CANVAS)],
                         "diffusion_canvas_length": CANVAS}
                for k in range(0, len(ids), MAX_TOKEN_IDS):  # one canvas for every chunk of the label set
                    chunk = unit["label_ids"][k:k + MAX_TOKEN_IDS]
                    reads.append(SamplingParams(max_tokens=1, logprob_token_ids=chunk, extra_args=extra))
                    owners.append((unit, r))
        values = defaultdict(dict)
        for (unit, r), out in zip(owners, self._generate([u["prefix_ids"] for u, _ in owners], reads)):
            values[(id(unit), r)].update({t: lp.logprob for t, lp in out.outputs[0].logprobs[0].items()})
        for unit in {id(u): u for u, _ in owners}.values():
            per_read = []
            unit["mass"] = 0.0
            for r in range(self.reads):
                got = values[(id(unit), r)]
                floor = min(got.values()) - 5.0
                logps = [got.get(i, floor) for i in unit["label_ids"]]
                top = max(logps)
                exps = [math.exp(x - top) for x in logps]
                per_read.append([e / sum(exps) for e in exps])
                unit["mass"] += sum(math.exp(x) for x in logps) / self.reads
            mean = [sum(col) / len(col) for col in zip(*per_read)]
            unit["scores"] = [math.log(max(p, 1e-12)) for p in mean]
