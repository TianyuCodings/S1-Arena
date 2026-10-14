"""CLM (Contrastive Language Model): two projection heads on a frozen encoder, scored by state-option similarity.

The probabilities are the model's own. Every question becomes one state text (the state rendered as prose, a blank
line, the question's instructions) and one text per option: a choice option's description, a score level, or, for a
boolean question, the statement affirmed and denied. The frozen encoder (Qwen3-8B with last-token pooling, served by
vLLM behind an OpenAI-compatible /v1/embeddings endpoint) embeds each text, the state head and the action head
project the embeddings, and a softmax over logit_scale * cosine(state, option) is the answer distribution. All of
this is the official library's code (contrastive-lm), called in-process: clm.Engine.answer, the function behind its
HTTP API, with clm.Embedder as the embeddings client. A batch is answered in chunks: the texts of a chunk are
embedded first, each distinct text once and `batch` texts per request, and stay in the client's cache, so option
texts that repeat across requests are embedded once.

Context: by default texts have at most 2,048 encoder tokens, the limit the model is released with: of a longer text
the last 2,048 tokens are embedded (the end of a state holds the latest steps and the question; training keeps the
same end), and usage["cut_texts"] counts them. The text is cut here, because the released client's request makes
the endpoint keep the first 2,048 tokens instead (https://github.com/Contrastive-LM/CLM/pull/29). With strict=1 nothing is cut: a question whose state text or one of whose option texts has more than
maxlen tokens is rejected, a final failure, and is not sent (default maxlen then: 8192, the context of the authors'
agentic recipe). A text without tokens is always rejected. Tokens are counted with the encoder's tokenizer file, as
the endpoint counts them; the encoder weights are never loaded here. A failed embeddings request fails its chunk of
requests, which are retried on the next run.

Spec: clm:<Hugging Face id>[@key=value,...]; keys: maxlen (tokens per text), strict (1: reject instead of cutting), emb (URL of the
embeddings endpoint; default $CLM_EMB_URL, else http://127.0.0.1:8090/v1/embeddings), served (the encoder's name on
that endpoint, default qwen3-8b), ckpt (a local copy of the checkpoint; default: the Hugging Face cache), batch
(texts per embeddings request, default 32). Needs contrastive-lm (installed without its dependencies), torch and
tokenizers; they are imported only when a model is first used. The heads run on CUDA when torch sees it, else on
the CPU ($CLM_DEVICE overrides).
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from ..registry import register_model
from ..schema import Answer, DecisionRequest, ModelResponse, Question
from .base import ModelAdapter
from .local_lm import REJECTED, parse_spec

REPO = "Contrastive-LM/CLM-v0.1-8B"
ENCODER = "Qwen/Qwen3-8B"  # tokenizer of checkpoints that do not name their encoder
EMB_URL = "http://127.0.0.1:8090/v1/embeddings"  # the library's default
OPTIONS = ("maxlen", "strict", "emb", "served", "ckpt", "batch")
CHUNK = 256  # requests embedded and answered together; the client's cache (200,000 texts) holds their texts


def cached_file(repo: str, filename: str) -> str:
    """Path of a Hub file: from the local cache without a network call if it is there, else downloaded."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return hf_hub_download(repo, filename, local_files_only=True)
    except LocalEntryNotFoundError:
        return hf_hub_download(repo, filename)


def to_clm(question: Question) -> dict[str, Any]:
    """The library's typed question (boolean -> Noul, choice -> Choice, score -> Score) in its wire format."""
    from clm.client import Choice, Noul, Score

    kind = {"boolean": Noul, "choice": Choice, "score": Score}[question.type]
    return kind(instructions=question.instructions, criteria=question.criteria).to_dict()


@register_model
class CLM(ModelAdapter):
    kind = "clm"
    native_probs = True
    probability_source = "softmax over state-option similarities (projection heads on a frozen encoder)"

    def __init__(self, arg: str = REPO) -> None:
        hf_id, opts = parse_spec(arg)
        if set(opts) - set(OPTIONS):
            raise ValueError(f"unknown options {sorted(set(opts) - set(OPTIONS))} in {arg!r}; known: {OPTIONS}")
        self.hf_id = hf_id or REPO
        self.strict = opts.get("strict", "0") == "1"
        self.maxlen = int(opts.get("maxlen", 8192 if self.strict else 2048))
        self._cuts: dict[str, str] = {}  # long texts of the chunk being answered -> their last maxlen tokens
        self.emb_url = opts.get("emb") or os.environ.get("CLM_EMB_URL", EMB_URL)
        self.served = opts.get("served", "qwen3-8b")
        self.ckpt = opts.get("ckpt")
        self.batch = int(opts.get("batch", 32))
        self._engine = self._tok = None
        self._lock = threading.Lock()  # environments with several agents call decide from threads

    @property
    def model_id(self) -> str:
        if self.strict:
            return f"{self.hf_id}@maxlen{self.maxlen}"
        return self.hf_id if self.maxlen == 2048 else f"{self.hf_id}@maxlen{self.maxlen}-cut"

    # ---------- engine ----------

    def _load(self) -> None:
        with self._lock:
            if self._engine is not None:
                return
            from clm.embedder import Embedder
            from clm.engine import DEFAULT_MODEL, Engine
            from tokenizers import Tokenizer

            ckpt = self.ckpt
            if not ckpt:  # the repository's config.json names its checkpoint file
                with open(cached_file(self.hf_id, "config.json"), encoding="utf-8") as f:
                    ckpt = cached_file(self.hf_id, json.load(f)["checkpoints"][0])
            # max_tokens=None: no truncation is requested, so the endpoint refuses a text it cannot take whole.
            # action_cache=0: no device memory is reserved for projections; embeddings are cached by the client.
            embedder = Embedder(self.emb_url, self.served, max_tokens=None, batch=self.batch)
            engine = Engine(embedder, checkpoint=ckpt, action_cache=0)
            encoder = engine.heads[DEFAULT_MODEL].cfg.get("model", ENCODER)
            self._tok = Tokenizer.from_file(cached_file(encoder, "tokenizer.json"))
            self._check_endpoint(embedder)
            if not self.strict:  # the engine asks for the embedding of a text; a long one is embedded cut
                embed = embedder.embed
                embedder.embed = lambda texts: embed([self._cuts.get(text, text) for text in texts])
            self._engine = engine

    def _check_endpoint(self, embedder) -> None:
        """The endpoint must serve the encoder with a context of at least maxlen tokens: with less it would refuse
        texts that fit here, and the whole batch with them."""
        reply = embedder.session.get(self.emb_url.rsplit("/v1/", 1)[0] + "/v1/models", timeout=30)
        cards = {card["id"]: card for card in reply.json()["data"]}
        if self.served not in cards:
            raise ValueError(f"{self.emb_url} serves {sorted(cards)}, not {self.served!r}")
        context = cards[self.served].get("max_model_len")  # reported by vLLM
        if context is not None and context < self.maxlen:
            raise ValueError(f"{self.emb_url} serves {self.served!r} with a context of {context} tokens, "
                             f"less than maxlen={self.maxlen}; restart it with --max-model-len {self.maxlen}")

    # ---------- decisions ----------

    def decide(self, request: DecisionRequest, sample: int = 0) -> ModelResponse:
        return self.decide_many([(request, sample)])[0]

    def decide_many(self, jobs: list[tuple[DecisionRequest, int]]) -> list[ModelResponse]:
        """Answer many requests at once, in order. The model is deterministic, so the sample index is ignored."""
        self._load()
        started = time.perf_counter()
        responses: list[ModelResponse] = []
        for start in range(0, len(jobs), CHUNK):
            chunk = [request for request, _ in jobs[start:start + CHUNK]]
            try:
                with self._lock:  # one chunk at a time: the cut texts of a chunk are shared state
                    responses += self._answer(chunk)
            except Exception as error:  # e.g. the endpoint is down: recorded as failed calls, retried next time
                responses += [ModelResponse(error=f"{type(error).__name__}: {str(error)[:300]}") for _ in chunk]
        per_job = round((time.perf_counter() - started) * 1000 / max(1, len(jobs)), 1)
        for response in responses:  # as for lm: models, latency_ms stays empty: realtime environments charge the
            if response.usage is not None:  # latency of a single request, which a batch does not measure
                response.usage["batch_ms_per_request"] = per_job
        return responses

    def _answer(self, requests: list[DecisionRequest]) -> list[ModelResponse]:
        from clm.schema import build_pairs, probabilities_of

        asked = [{name: to_clm(question) for name, question in r.questions.items()} for r in requests]
        # per request and question: the texts to embed, the state text first, then one per option
        texts = [{name: [state, *options] for name, (state, _, options) in build_pairs(r.state, questions).items()}
                 for r, questions in zip(requests, asked)]
        distinct = list(dict.fromkeys(t for per in texts for group in per.values() for t in group))
        encoded = dict(zip(distinct, self._tok.encode_batch_fast(distinct)))
        tokens = {text: len(e.ids) for text, e in encoded.items()}
        if not self.strict:
            self._cuts = {text: self._tail(e.ids) for text, e in encoded.items() if len(e.ids) > self.maxlen}
        errors = [{name: reason for name, group in per.items() if (reason := self._unfit(group, tokens))}
                  for per in texts]
        fitting = dict.fromkeys(t for per, error in zip(texts, errors) for name, group in per.items()
                                if name not in error for t in group)
        if fitting:  # one pass over the distinct texts, `batch` per request; Engine.answer then reads the cache
            self._engine.embedder.embed(list(fitting))
        responses = []
        for request, questions, per, error in zip(requests, asked, texts, errors):
            fit = {name: question for name, question in questions.items() if name not in error}
            answers = {}
            for name, reply in (self._engine.answer(request.state, fit)["answers"] if fit else {}).items():
                probs = probabilities_of(reply)
                answers[name] = Answer(probs={o: probs[o] for o in request.questions[name].option_ids()})
            usage = {"state_tokens": sum(tokens[t] for t in {group[0] for group in per.values()}),
                     "option_tokens": sum(tokens[t] for t in {t for group in per.values() for t in group[1:]})}
            if not self.strict:
                usage["cut_texts"] = len({t for group in per.values() for t in group if t in self._cuts})
            reasons = "; ".join(f"{name}: {reason}" for name, reason in error.items())[:300]
            responses.append(ModelResponse(answers=answers, usage=usage, error=REJECTED + reasons if error else None))
        return responses

    def _tail(self, ids: list[int]) -> str:
        """The text of the last maxlen tokens (fewer if that text would tokenize into more than maxlen again)."""
        keep = self.maxlen
        while True:
            text = self._tok.decode(ids[-keep:], skip_special_tokens=False)
            if len(self._tok.encode(text).ids) <= self.maxlen:
                return text
            keep -= 8

    def _unfit(self, texts: list[str], tokens: dict[str, int]) -> str | None:
        """Why the encoder cannot take the texts of a question (the state text, then the options) whole, if so."""
        for index, text in enumerate(texts):
            what = "an option text" if index else "the state text"
            if tokens[text] > self.maxlen and self.strict:
                return f"{what} has {tokens[text]} tokens, more than the {self.maxlen}-token context"
            if not tokens[text]:
                return f"{what} is empty"
        return None
