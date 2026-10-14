"""DiffusionGemma as a decision model: a read-only denoising pass over a single-token answer slot.

This follows vLLM's structured reads for DiffusionGemma (vllm-project/vllm#57250; it needs a vLLM build that includes
that change, e.g. a nightly after 2026-09-22). Each question gets the prompt of the autoregressive adapter (the
state, the question and its labeled options). After the prompt, DiffusionGemma denoises a canvas; here the canvas
holds the answer template: an empty thought block and the answer label, then the end-of-turn token and padding. The
label position starts as a random token (DiffusionGemma denoises uniform noise, not masks), every other position is
pinned, and a read-only pass returns the exact log-probabilities of every label token at that position; a softmax
over them is the answer. In vLLM's output, the row that predicts canvas position p is row p - 1 (the model keeps the
autoregressive shift), so the read takes the row before the slot (offset=-1). steps=S lets the slot be refined for up to S denoising steps before it is read (one step by
default); reads=R averages R reads with independent noise. think=B first lets the model write up to B tokens in its
thought channel as an ordinary generation; the canvas then starts right at the answer.

Every label must be a single token at the slot. Choices use A-Z and, past 26 options, two-letter labels that the
tokenizer keeps whole; a question whose options cannot all get such labels is rejected (a final failure).

layout=prompt (the default) ends the prompt with the thought channel's open tag and starts the canvas with its close
tag; layout=canvas puts the whole empty thought block on the canvas. On 150 knowledge items (5 per source), reading
the row before the slot with layout=prompt and one step gave the highest label mass (median 0.91) and the lowest
Brier score; reading the slot's own row left a median label mass of 0.08, and four denoising steps did not help.

Spec: dgemma:<Hugging Face id>[@key=value,...]; keys: think, steps, reads, offset, layout, gpu, maxlen, tp.
"""

from __future__ import annotations

import itertools
import math
import random
import string
from collections import defaultdict

from ..registry import register_model
from ..schema import Question
from .local_lm import MAX_TOKEN_IDS, REJECTED, SYSTEM, LocalLM, parse_spec, question_text, state_text

CANVAS = 16  # served canvas length: the answer template of one question, the turn close and padding
THOUGHT_OPEN, THOUGHT_CLOSE = "<|channel>thought\n", "<channel|>"
TURN_CLOSE_TEXT = "<turn|>"  # DiffusionGemma's end-of-turn token


@register_model
class DiffusionGemma(LocalLM):
    kind = "dgemma"
    probability_source = "answer-slot log-probabilities of a diffusion model (local)"

    def __init__(self, arg: str = "google/diffusiongemma-26B-A4B-it") -> None:
        self.hf_id, opts = parse_spec(arg)
        self.mode, self.norm = "label", "sum"
        self.think = int(opts.get("think", 0))
        self.steps = int(opts.get("steps", 1))
        self.reads = int(opts.get("reads", 1))
        self.offset = int(opts.get("offset", -1))
        self.layout = opts.get("layout", "prompt")
        self.tp = int(opts.get("tp", 1))
        self.maxlen = int(opts["maxlen"]) if "maxlen" in opts else None
        self.gpu = float(opts.get("gpu", 0.88))
        self._llm = self._tok = None

    @property
    def model_id(self) -> str:
        return f"{self.hf_id}@steps{self.steps}-reads{self.reads}-think{self.think}-{self.layout}{self.offset}"

    def _load(self) -> None:
        if self._llm is not None:
            return
        from vllm import LLM

        self._llm = LLM(model=self.hf_id, tensor_parallel_size=self.tp, max_model_len=self.maxlen, enable_prefix_caching=True,
                        gpu_memory_utilization=self.gpu, max_logprobs=64, seed=0, diffusion_config={"canvas_length": CANVAS})
        self._tok = self._llm.get_tokenizer()
        config = getattr(self._llm, "model_config", None) or self._llm.llm_engine.model_config
        self._context = config.max_model_len
        self._thinking = True
        self._scaffold = self._encode(THOUGHT_OPEN + THOUGHT_CLOSE)
        self._close = self._encode(THOUGHT_CLOSE)
        self._turn_close = self._encode(TURN_CLOSE_TEXT)
        if len(self._turn_close) != 1:
            raise ValueError(f"{TURN_CLOSE_TEXT!r} is not one token in {self.hf_id}")
        self._pad = self._tok.pad_token_id or 0
        self._vocab = len(self._tok)
        pairs = ("".join(p) for p in itertools.product(string.ascii_uppercase, repeat=2))
        self._letters = list(string.ascii_uppercase) + [p for p in pairs if len(self._encode(p)) == 1]

    def _labels(self, question: Question) -> list[str] | None:
        n = len(question.option_ids())
        if question.type == "boolean":
            return ["yes", "no"]
        if question.type == "score":
            return [str(i) for i in range(n)]
        return self._letters[:n] if n <= len(self._letters) else None

    def _units(self, sample: int, request, question: Question) -> list[dict]:
        labels = self._labels(question)
        if labels is None:
            return [{"sample": sample, "kind": "label", "labels": [], "error": REJECTED + "more options than single-token labels",
                     "system": SYSTEM, "user": ""}]
        return [{"sample": sample, "kind": "label", "labels": labels, "system": SYSTEM,
                 "user": f"{state_text(request)}\n\n{question_text(question, labels)}"}]

    def _think(self, units: list[dict]) -> None:
        """Set each unit's prompt ids and the canvas head: the empty thought block, or nothing after a written thought."""
        from vllm import SamplingParams

        limit = self._context - max(self.think, 0) - CANVAS - 8
        for unit in units:
            if "error" in unit:
                continue
            unit["prompt"] = self._prompt(unit["system"], unit["user"], thinking=self.think > 0)
            unit["prompt_ids"] = self._encode(unit["prompt"])
            if len(unit["prompt_ids"]) > limit:
                unit["error"] = REJECTED + f"prompt longer than the {self._context}-token context"
        todo = [u for u in units if "error" not in u]
        if self.think <= 0:
            opening = self._encode(THOUGHT_OPEN)
            for unit in todo:
                if self.layout == "prompt":
                    unit["prefix_ids"], unit["head"] = unit["prompt_ids"] + opening, self._close
                else:
                    unit["prefix_ids"], unit["head"] = unit["prompt_ids"], self._scaffold
            return
        opened = [u["prompt_ids"] + self._encode(THOUGHT_OPEN) for u in todo]
        params = [SamplingParams(max_tokens=self.think, stop_token_ids=self._close[:1]) for _ in todo]  # no seeds for diffusion
        for unit, ids, out in zip(todo, opened, self._generate(opened, params)):
            thought = [t for t in out.outputs[0].token_ids if t != self._close[0]]
            unit["think_tokens"] = len(thought)
            unit["prefix_ids"], unit["head"] = ids + thought + self._close, []

    def _score(self, units: list[dict]) -> None:
        from vllm import SamplingParams

        reads, owners = [], []
        for unit in (u for u in units if "error" not in u):
            label_ids = []
            for label in unit["labels"]:
                tokens = self._encode(label)
                if len(tokens) != 1:
                    unit["error"] = REJECTED + f"label {label!r} is not a single token"
                    break
                label_ids.append(tokens[0])
            if "error" in unit or len(set(label_ids)) != len(label_ids):
                unit.setdefault("error", REJECTED + "two labels share a token")
                continue
            unit["label_ids"] = label_ids
            slot = len(unit["head"])
            template = unit["head"] + [label_ids[0]] + self._turn_close
            for r in range(self.reads):
                rng = random.Random(f"{unit['sample']}-{r}")
                canvas = template + [self._pad] * (CANVAS - len(template))
                canvas[slot] = rng.randrange(self._vocab)
                extra = {"diffusion_seed_canvas": canvas, "diffusion_canvas_length": CANVAS, "diffusion_max_steps": self.steps,
                         "diffusion_read_only": True}
                if self.steps > 1:
                    extra["diffusion_pinned"] = [p for p in range(CANVAS) if p != slot]
                # vLLM returns at most 128 token ids per request: larger label sets are read in chunks of the same canvas,
                # whose read-only pass gives the same distribution every time.
                for k in range(0, len(label_ids), MAX_TOKEN_IDS):
                    reads.append(SamplingParams(max_tokens=CANVAS, logprob_token_ids=label_ids[k:k + MAX_TOKEN_IDS], extra_args=extra))
                    owners.append((unit, r, slot))
        if not reads:
            return
        values = defaultdict(dict)
        for (unit, r, slot), out in zip(owners, self._generate([u["prefix_ids"] for u, _, _ in owners], reads)):
            row = out.outputs[0].logprobs[slot + self.offset]
            values[(id(unit), r)].update({t: lp.logprob for t, lp in row.items()})
        for unit in {id(u): u for u, _, _ in owners}.values():
            for r in range(self.reads):
                got = values[(id(unit), r)]
                floor = min(got.values()) - 5.0
                logps = [got.get(i, floor) for i in unit["label_ids"]]
                top = max(logps)
                probs = [math.exp(x - top) for x in logps]
                total = sum(probs)
                unit.setdefault("read_probs", []).append([p / total for p in probs])
                unit["mass"] = unit.get("mass", 0.0) + sum(math.exp(x) for x in logps) / self.reads
            mean = [sum(col) / len(col) for col in zip(*unit["read_probs"])]
            unit["scores"] = [math.log(max(p, 1e-12)) for p in mean]
