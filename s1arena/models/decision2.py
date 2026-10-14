"""Decision 2.0 models of the vLLM Semantic Router team (Decision-2.0-Sol-2B, Decision-2.0-Kai-0.6B).

These are System-One models: a Qwen text backbone with a candidate head, loaded through the code in the model's own
repository (AutoModel.from_pretrained(repo, trust_remote_code=True)) and asked through its system_one(state=...,
questions=...) call, which takes the request format of Jev. A boolean question is sent as the model's "noul" type,
choice and score questions as they are, and the state as the dict it is (the model's code writes a dict as compact
JSON with sorted keys). The model's code encodes every question together with the state and the question's options
as one input row, without a chat template, and a softmax over the candidate logits of that row is the answer
(temperature 1; Kai adds its packaged offsets to the logits of five-level scores). The adapter returns these
probabilities as system_one gives them: "probabilities" for choice and score, "noul" = P(true) for boolean.

Context: a row may hold at most the model's max_input_tokens (16,384 for Sol, 8,192 for Kai). The model's code never
truncates and refuses a longer row, so the adapter first encodes every row with the model's own encoder and reports
the refused questions as a final "rejected: ..." error; the other questions of the request are still answered.

The runtime answers one request per call and runs its questions as one padded batch; no call takes several requests,
so decide_many loops over the requests. A request whose batch would hold more than `batch` padded tokens (default:
the context length, so that a batch never needs more memory than one full-length row) is split into several calls.
The runtime is not exactly batch-invariant, so a split moves probabilities by rounding noise.

Spec: decision2:<Hugging Face id>[@key=value,...]; keys: rev (repository revision, part of the model id), device
(default: the runtime's own choice, cuda:0 if there is a GPU, else cpu), batch (padded tokens per call), gpus (hold
the backbone's decoder layers on this many visible GPUs, in contiguous blocks; for a model whose long inputs do not fit
one GPU, e.g. Vega-27B on an 80 GB A100 above about 28,000 tokens; not part of the model id). The runtime
fixes the numerics: BF16 autocast with an FP32 head on a GPU, FP32 elsewhere. Needs torch, transformers >= 5.17 and
safetensors, imported only when a model is first used. Runs offline (HF_HUB_OFFLINE=1) from a cache filled by
`hf download <id>`; the download must be complete, since the runtime checks every file against its manifest.
"""

from __future__ import annotations

import importlib
import threading
import time
from typing import Any

from ..registry import register_model
from ..schema import Answer, DecisionRequest, ModelResponse, Question
from .base import ModelAdapter
from .jev import to_jev
from .local_lm import REJECTED, parse_spec


def to_decision2(question: Question) -> dict[str, Any]:
    return {**to_jev(question), "type": "noul" if question.type == "boolean" else question.type}


def from_decision2(answer: dict[str, Any], question: Question) -> dict[str, float]:
    if question.type == "boolean":
        p = float(answer["noul"])
        return {"true": p, "false": 1.0 - p}
    return {o: float(answer["probabilities"][o]) for o in question.option_ids()}


@register_model
class Decision2(ModelAdapter):
    kind = "decision2"
    native_probs = True
    probability_source = "native (softmax over the candidate logits of a local decision model)"

    def __init__(self, arg: str = "vllm-sr/Decision-2.0-Sol-2B") -> None:
        self.hf_id, opts = parse_spec(arg)
        if set(opts) - {"rev", "device", "batch", "gpus"}:
            raise ValueError(f"unknown option in {arg!r}")
        self.rev = opts.get("rev")
        self.device = opts.get("device")
        self.batch = int(opts["batch"]) if "batch" in opts else None
        self.gpus = int(opts.get("gpus", 1))
        self._model = None
        self._lock = threading.Lock()  # closed-loop environments call decide() from several threads

    @property
    def model_id(self) -> str:
        return self.hf_id + (f"@rev={self.rev}" if self.rev else "")

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel

        options = {key: value for key, value in (("revision", self.rev), ("device", self.device)) if value}
        self._model = AutoModel.from_pretrained(self.hf_id, trust_remote_code=True, **options)
        runtime = self._model.runtime  # the repository's own runtime (decision2/api.py); its backend holds the model
        vendored = type(runtime).__module__.rpartition(".")[0] + "._vendor.dev2model."
        self._row = importlib.import_module(vendored + "infer").question_to_row
        self._encode = runtime.backend.encode_fn or importlib.import_module(vendored + "decision_model").encode
        self._tok = runtime.backend.tokenizer
        self._context = self._model.max_input_tokens
        if self.gpus > 1:
            self._spread(self.gpus)

    def _spread(self, gpus: int) -> None:
        """Move the backbone's decoder layers, loaded on cuda:0, onto GPUs 0..gpus-1 in contiguous blocks of about equal
        bytes; the embedding, the final norm and the head stay on cuda:0. A layer on another GPU gets its inputs moved
        there (the masks and rotary embeddings once per forward), and the last layer's output goes back to cuda:0 for the
        norm: the same modules run in the same order on the same dtypes."""
        import torch

        model = self._model.runtime.backend.model  # the runtime's DecisionModel: backbone (PEFT) + candidate head
        core = model.backbone.get_base_model() if hasattr(model.backbone, "get_base_model") else model.backbone
        layers = list(core.layers)
        sizes = [sum(p.numel() * p.element_size() for p in layer.parameters()) for layer in layers]
        load = sum(p.numel() * p.element_size() for p in model.parameters()) - sum(sizes)  # cuda:0's fixed part
        share = (load + sum(sizes)) / gpus
        moved: dict = {}  # id(tensor), device -> (tensor, its copy), for one forward

        def to(value, device):
            if torch.is_tensor(value):
                if value.device == device:
                    return value
                key = (id(value), device)
                if key not in moved:
                    moved[key] = (value, value.to(device))
                return moved[key][1]
            if isinstance(value, (tuple, list)):
                return type(value)(to(v, device) for v in value)
            if isinstance(value, dict):
                return {k: to(v, device) for k, v in value.items()}
            return value

        gpu = 0
        for layer, size in zip(layers, sizes):
            if gpu < gpus - 1 and load + size / 2 > share * (gpu + 1):
                gpu += 1
            load += size
            if gpu:
                device = torch.device("cuda", gpu)
                layer.to(device)
                layer.register_forward_pre_hook(lambda m, args, kwargs, d=device: (to(args, d), to(kwargs, d)),
                                                with_kwargs=True)
        core.norm.register_forward_pre_hook(lambda m, args, d=torch.device("cuda", 0): to(args, d))
        model.backbone.register_forward_pre_hook(lambda m, args: moved.clear())
        model.backbone.register_forward_hook(lambda m, args, out: moved.clear())
        torch.cuda.empty_cache()

    def decide(self, request: DecisionRequest, sample: int = 0) -> ModelResponse:
        return self.decide_many([(request, sample)])[0]

    def decide_many(self, jobs: list[tuple[DecisionRequest, int]]) -> list[ModelResponse]:
        """Answer the requests one after another; the model is deterministic, so the sample number is not used."""
        with self._lock:
            self._load()
            started = time.perf_counter()
            responses = []
            for request, _ in jobs:
                try:
                    responses.append(self._decide(request))
                except Exception as error:  # e.g. out of memory: recorded as a failed call, retried on the next run
                    responses.append(ModelResponse(error=f"{type(error).__name__}: {str(error)[:300]}"))
            per_job = round((time.perf_counter() - started) * 1000 / max(1, len(jobs)), 1)
        for response in responses:  # as for lm: models, latency_ms stays empty: realtime environments charge the
            if response.usage is not None:  # latency of a single request, which a batch does not measure
                response.usage["batch_ms_per_request"] = per_job
        return responses

    def _decide(self, request: DecisionRequest) -> ModelResponse:
        questions = {name: to_decision2(question) for name, question in request.questions.items()}
        item = {"id": "request", "state": request.state}
        lengths: dict[str, int] = {}
        errors: dict[str, str] = {}
        for name, payload in questions.items():
            try:  # the runtime's own encoding of state + question + options; it raises instead of truncating
                lengths[name] = len(self._encode(self._row(item, name, payload), self._tok, self._context)["ids"])
            except ValueError as error:
                errors[name] = str(error).removeprefix(f"request/{name}: ")
        answers: dict[str, Answer] = {}
        for names in self._calls(lengths):
            out = self._model.system_one(state=request.state, questions={name: questions[name] for name in names})
            for name in names:
                answer = out["answers"][name]
                if "error" in answer:
                    errors[name] = answer["error"]
                else:
                    answers[name] = Answer(probs=from_decision2(answer, request.questions[name]))
        usage = {"input_tokens": sum(lengths[name] for name in answers)}
        error = REJECTED + "; ".join(f"{name}: {reason}" for name, reason in errors.items())[:300] if errors else None
        return ModelResponse(answers=answers, usage=usage, error=error)

    def _calls(self, lengths: dict[str, int]) -> list[list[str]]:
        """The questions of each system_one call: all in one, the runtime's own batch, while it fits the budget."""
        budget = self.batch or self._context
        calls: list[list[str]] = []
        for name in lengths:
            rows = (calls[-1] if calls else []) + [name]
            padded = -(-max(lengths[n] for n in rows) // 8) * 8  # the runtime pads a batch to a multiple of 8
            if calls and len(rows) * padded <= budget:
                calls[-1] = rows
            else:
                calls.append([name])
        return calls
