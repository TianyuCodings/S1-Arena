"""Laya, an open System-One decision model, run locally through its own library (pip install laya).

Laya is an encoder that reads the question, its options and the state as one sequence and scores every option at a
mask token placed before it: [CLS] <type> question: <instructions> [SEP] [MASK] <option> ... [SEP] <state> [SEP]. The
library builds that sequence, one per question. Choice and score questions are passed as they are, a boolean question
is the library's "noul" (options false / true), and a dict state is serialized by the library with
json.dumps(state, ensure_ascii=False). The answer to a question is the softmax over its option logits divided by the
checkpoint's temperature for the question type and option count: the distribution the library's predict() computes,
read before predict() rounds it to four decimals (the multilingual checkpoint ships temperature 1, the two English
ones fitted temperatures). Nothing is sampled, so the sample index of a job is ignored.

Context: by default the model is run as released, limits included. The English checkpoint reads 512 tokens, and the
library cuts what does not fit: an option to 48 tokens, the options together to the head budget (192 tokens), and
the state to what is left, dropping its end. Such answers are counted in usage["cut_questions"]. A question whose
option markers do not all fit even then cannot be asked and is rejected.

With strict=1 nothing is cut: a question whose input does not fit the context is rejected instead. The library would
cut without an error in three places, so each is checked before the model is called:
- an option (" <id>: <description>") is cut to its first 48 tokens whatever the budgets are, so a question with a
  longer option is rejected;
- the instructions and options share a head budget (head_max_len, 256 tokens by default); past it the options are cut
  to a few tokens each and the instructions to what is left. The library takes a budget per call, and the adapter
  passes one just large enough to keep every question of the request whole, so this cut never happens;
- the state gets what is left of max_len and loses its end, so a question is rejected when its sequence (instructions,
  options, state and four special tokens) is longer than max_len.
After encoding, the library's own report (no state token dropped, no option capped) and the length of its sequence
are compared with the adapter's count; a difference raises instead of scoring a cut input.

One forward pass holds rows (questions) of similar length: rows x longest row^2 <= batch^2 tokens, as the attention
memory grows with the square of the length, and at most 64 rows.

Spec: laya:<Hugging Face repo>[@key=value,...]; keys: ckpt (english | multilingual | typed-decisions), maxlen (context
length in tokens), strict (1: reject instead of cutting), device, batch. The default is the English checkpoint with
the library's own limits. With strict=1 the context defaults to 512 (english), 8192 (multilingual, the setting the
authors document for long documents) and 1,024 (typed-decisions) tokens. Needs
the laya package (written against 0.3.27, whose private Agent methods it calls) with torch and transformers; they are
imported only when a model is first used, so specs can be listed without them.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

from ..registry import register_model
from ..schema import Answer, DecisionRequest, ModelResponse, Question
from .base import ModelAdapter
from .local_lm import REJECTED, parse_spec, softmax

# checkpoint -> (subfolder of the repo, default context length)
CHECKPOINTS = {"english": (None, 512), "multilingual": ("multilingual", 8192),
               "typed-decisions": ("typed-decisions", 1024)}
OPTION_TOKENS = 48  # the library keeps the first 48 tokens of an option (laya.common.build_head)
HEAD_SLACK = 16  # build_head cuts the options once fewer than 16 tokens of the head budget are left
MAX_ROWS = 64  # questions per forward pass
WINDOW = 256  # requests encoded at a time, so that a batch of 2,000 long requests is never held as token lists


def to_laya(question: Question) -> dict[str, Any]:
    if question.type == "boolean":
        return {"type": "noul", "instructions": question.instructions}
    return {"type": question.type, "instructions": question.instructions, "criteria": question.criteria}


@register_model
class Laya(ModelAdapter):
    kind = "laya"
    native_probs = True
    probability_source = "softmax over the option logits of one forward pass (laya library)"

    def __init__(self, arg: str = "convaiinnovations/laya") -> None:
        self.hf_id, opts = parse_spec(arg)
        self.ckpt = opts.get("ckpt", "english")
        if self.ckpt not in CHECKPOINTS:
            raise ValueError(f"unknown checkpoint in {arg!r}; known: {sorted(CHECKPOINTS)}")
        self.strict = opts.get("strict", "0") == "1"
        # as released (not strict), the context is the library's own unless maxlen is given
        self.maxlen = int(opts["maxlen"]) if "maxlen" in opts else CHECKPOINTS[self.ckpt][1] if self.strict else None
        self.device = opts.get("device")
        self.batch = int(opts.get("batch", 8192))
        self._agent: Any = None
        self._lib: Any = None
        self._lock = threading.Lock()  # closed-loop environments call decide() from several threads

    @property
    def model_id(self) -> str:
        if self.strict:
            return f"{self.hf_id}@{self.ckpt}-maxlen{self.maxlen}"
        if self.ckpt == "english" and self.maxlen is None:
            return self.hf_id  # the model as released
        return f"{self.hf_id}@{self.ckpt}" + (f"-maxlen{self.maxlen}" if self.maxlen else "") + "-cut"

    def _load(self) -> None:
        if self._agent is not None:
            return
        import laya
        import laya.common

        agent = laya.load(self.hf_id, subfolder=CHECKPOINTS[self.ckpt][0], device=self.device)
        limit = agent.model.encoder.config.max_position_embeddings
        if (self.maxlen or 0) > limit:
            raise ValueError(f"maxlen={self.maxlen} is longer than the encoder's {limit} positions")
        self._agent, self._lib = agent, laya.common

    def decide(self, request: DecisionRequest, sample: int = 0) -> ModelResponse:
        return self.decide_many([(request, sample)])[0]

    def decide_many(self, jobs: list[tuple[DecisionRequest, int]]) -> list[ModelResponse]:
        """Answer many requests at once: every question of every request is one row of a forward pass."""
        with self._lock:
            self._load()
            started = time.time()
            responses: list[ModelResponse] = []
            for start in range(0, len(jobs), WINDOW):
                responses += self._answer([request for request, _ in jobs[start:start + WINDOW]])
            per_job = round((time.time() - started) * 1000 / max(1, len(jobs)), 1)
        for response in responses:  # as for lm: models, latency_ms stays empty: realtime environments charge the
            response.usage["batch_ms_per_request"] = per_job  # latency of a single request, not a batch's share
        return responses

    def _encode(self, request: DecisionRequest) -> tuple[dict[str, dict], dict[str, str]]:
        """The library's sequence for every question that fits whole, and the reason for every one that does not."""
        agent, lib, tok = self._agent, self._lib, self._agent.tok

        def count(text: str) -> int:  # as the library tokenizes a piece of the question (under its tokenizer lock)
            return len(lib.encode_text(tok, text.replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"])

        internal, heads, errors, budget = {}, {}, {}, 0
        for name, question in request.questions.items():
            definition = to_laya(question)
            try:
                agent._check_question(name, definition)
            except ValueError as error:  # the library refuses the question itself
                errors[name] = REJECTED + str(error)[:200]
                continue
            q = agent._to_internal(definition)
            if not self.strict:
                internal[name] = q
                continue
            instructions = count(f"{q['t']} question: {q['ins']}")
            options = [1 + count(" " + text) for text in lib.render_options(q)]  # a mask token, then the option
            head = 3 + instructions + sum(options)  # [CLS] instructions [SEP] options [SEP]
            if max(options) > 1 + OPTION_TOKENS:
                errors[name] = REJECTED + f"an option has {max(options) - 1} tokens, the library keeps {OPTION_TOKENS}"
            elif head >= self.maxlen:
                errors[name] = REJECTED + f"the question with its options has {head} tokens, the context {self.maxlen}"
            else:
                internal[name], heads[name] = q, head
                budget = max(budget, sum(options) + max(HEAD_SLACK, instructions))
        items = {}
        if internal and not self.strict:  # as released: the library's budgets and cuts
            for name in internal:  # one question per call: a question that cannot be asked raises
                try:
                    items[name] = agent._encode_state(request.state, [name], internal, max_len=self.maxlen)[0]
                except ValueError as error:
                    errors[name] = REJECTED + str(error).removeprefix(f"question {name!r}: ")[:200]
        elif internal:
            encoded = agent._encode_state(request.state, list(internal), internal, max_len=self.maxlen,
                                          head_max_len=budget)
            for name, item in zip(internal, encoded):
                state = item["state_stats"]["state_tokens"]
                total = heads[name] + state + 1  # the state and its closing [SEP]
                if item["state_stats"]["truncated"]:
                    errors[name] = REJECTED + (f"input of {total} tokens (state {state}, question {heads[name]}) is "
                                               f"longer than the {self.maxlen}-token context")
                elif len(item["ids"]) != total or item["options"]["tokens_per_option"] is not None:
                    raise RuntimeError(f"laya kept {len(item['ids'])} of the {total} tokens of question {name!r}, "
                                       f"which was expected to fit: the adapter does not match the installed laya")
                else:
                    items[name] = item
        return items, errors

    def _answer(self, requests: list[DecisionRequest]) -> list[ModelResponse]:
        import torch

        agent, lib = self._agent, self._lib
        errors: list[dict[str, str]] = []
        usage: list[dict[str, int]] = []
        rows = []  # (request index, question name, the library's encoded sequence)
        for r, request in enumerate(requests):
            items, failed = self._encode(request)
            errors.append(failed)
            usage.append({"input_tokens": sum(len(item["ids"]) for item in items.values())})
            if not self.strict:  # questions whose state lost its end or whose options were shortened
                usage[-1]["cut_questions"] = sum(item["state_stats"]["truncated"] or item["options"]["tokens_per_option"]
                                                 is not None for item in items.values())
            rows += [(r, name, item) for name, item in items.items()]
        rows.sort(key=lambda row: -len(row[2]["ids"]))
        probs: dict[tuple[int, str], list[float]] = {}
        start = 0
        while start < len(rows):
            longest = len(rows[start][2]["ids"])
            chunk = rows[start:start + max(1, min(MAX_ROWS, self.batch ** 2 // longest ** 2))]
            with torch.no_grad():
                logits, _ = agent._forward(lib.collate_items([[item for _, _, item in chunk]], agent.tok.pad_token_id))
            for (r, name, item), row in zip(chunk, logits):
                kind, k = item["qtype"], len(item["markers"])
                temperature = agent.temperature_by_options.get(lib.temp_bucket(kind, k), agent.temperature[kind])
                if all(math.isfinite(z) for z in row[:k]):
                    probs[r, name] = softmax([float(z) / temperature for z in row[:k]])
                else:
                    errors[r][name] = "the model returned logits that are not finite"
            start += len(chunk)
        responses = []
        for r, request in enumerate(requests):
            answers = {}
            for name, question in request.questions.items():
                if (r, name) not in probs:
                    continue
                p = probs[r, name]
                if question.type == "boolean":  # the library's noul options are (false, true)
                    answers[name] = Answer(probs={"true": p[1], "false": 1.0 - p[1]})
                else:
                    answers[name] = Answer(probs=dict(zip(question.option_ids(), p)))
            error = None
            if errors[r]:
                final = all(e.startswith(REJECTED) for e in errors[r].values())
                text = "; ".join(f"{n}: {e.removeprefix(REJECTED)}" for n, e in errors[r].items())
                error = (REJECTED if final else "") + text[:300]
            responses.append(ModelResponse(answers=answers, usage=usage[r], error=error))
        return responses
