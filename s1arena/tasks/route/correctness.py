"""Leaf: will a given model answer this request correctly (模型能否答对这题)?

One boolean question per candidate model, asked together in one request. Gold is each
model's recorded, automatically scored outcome on exactly this prompt (one reply per
model, so a correct answer can also be luck, and a wrong one a formatting slip).
"""

from __future__ import annotations

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import slug
from . import outcomes as oc

LEAF = "route/model_correctness"


class ModelCorrectness(TaskAdapter):
    leaf = LEAF
    tier = "C"
    MODELS: dict[str, str] = {}  # model id -> description
    PROTOCOL = ""

    def rows(self) -> list[dict]:
        """Records with 'id', 'group', 'prompt' and 'score' {model: 0/1}."""
        raise NotImplementedError

    def load(self):
        return [r for r in self.rows() if all(m in r["score"] for m in self.MODELS)]

    def label_of(self, record):
        return f"{record['group']}|{sum(record['score'][m] for m in self.MODELS):.0f}"

    def to_item(self, record, rng):
        questions, gold = {}, {}
        for model, description in self.MODELS.items():
            key = slug(model.split("/")[-1])
            questions[key] = Question(type="boolean", instructions=f"Will {description} answer this prompt correctly? {self.PROTOCOL}")
            gold[key] = Gold(label="true" if record["score"][model] >= 1 else "false")
        return self.item({"prompt": record["prompt"]}, questions, gold, group=record["group"], original_id=record["id"])


@register_task
class RouterBenchCorrectness(ModelCorrectness):
    name = f"{LEAF}/routerbench"
    source = "RouterBench, zero-shot MMLU, HellaSwag and ARC-Challenge prompts (Hu et al., 2024)"
    url = oc.ROUTERBENCH_URL
    license = oc.ROUTERBENCH_LICENSE
    criteria_source = "Written; gold is RouterBench's recorded 0/1 score of each model's single reply"
    notes = ("Six of the 11 models: Llama-2-70B, Code Llama 34B and Mistral 7B often break the answer format, so their scores "
             "reflect formatting more than ability, and WizardLM 13B and Claude 1 are left out to keep six questions per item. "
             "Other evals are left out: GSM8K scores correct replies 0.25-0.75, MBPP prompts lack the tested function "
             "signature, MT-Bench uses a GPT-4 judge, WinoGrande is already used in know/commonsense. "
             "The source file is a pandas pickle; it is read with an unpickler that only admits DataFrame classes.")
    MODELS = oc.ROUTERBENCH_MODELS
    PROTOCOL = oc.ROUTERBENCH_PROTOCOL

    def rows(self):
        return list(oc.routerbench())


@register_task
class RouterArenaCorrectness(ModelCorrectness):
    name = f"{LEAF}/routerarena"
    source = "RouterArena sub_10 split with cached replies of three models (Lu et al., 2025)"
    url = oc.ROUTERARENA_URL
    license = oc.ROUTERARENA_LICENSE
    criteria_source = "Written; gold is RouterArena's automatic score of each model's cached reply"
    notes = ("Only requests scored 0/1 (METEOR-scored translation and NarrativeQA items are left out). The authors ask "
             "that RouterArena be used for evaluation only, not for training routers.")
    MODELS = oc.ROUTERARENA_MODELS
    PROTOCOL = oc.ROUTERARENA_PROTOCOL

    def rows(self):
        return [{"id": r["id"], "group": r["family"], "prompt": r["prompt"], "score": {m: o["score"] for m, o in r["outcome"].items()}}
                for r in oc.routerarena().values()]


LIVEBENCH_MODELS = {
    "gemini-1.5-flash-002": "Gemini 1.5 Flash (Google, gemini-1.5-flash-002, September 2024)",
    "gpt-4o-mini-2024-07-18": "GPT-4o mini (OpenAI, gpt-4o-mini-2024-07-18)",
    "claude-3-5-haiku-20241022": "Claude 3.5 Haiku (Anthropic, claude-3-5-haiku-20241022)",
    "Qwen2.5-72B-Instruct-Turbo": "Qwen2.5 72B Instruct (Alibaba, open weights, September 2024)",
    "gpt-4o-2024-08-06": "GPT-4o (OpenAI, gpt-4o-2024-08-06)",
    "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet (Anthropic, claude-3-5-sonnet-20241022)",
    "o1-preview-2024-09-12": "o1-preview (OpenAI reasoning model, o1-preview-2024-09-12)",
}


@register_task
class LiveBenchCorrectness(ModelCorrectness):
    name = f"{LEAF}/livebench"
    source = "LiveBench code generation, code completion, typo fixing and Connections puzzles (White et al., 2025)"
    url = oc.LIVEBENCH_URL
    license = oc.LIVEBENCH_LICENSE
    criteria_source = "Written; gold is LiveBench's objective score of each model's output (no LLM judge)"
    notes = ("Questions released June to November 2024 whose texts are public. Connections counts as correct only when "
             "all groups are right. The license of the LiveBench question sets is not stated.")
    MODELS = LIVEBENCH_MODELS
    PROTOCOL = oc.LIVEBENCH_PROTOCOL

    def rows(self):
        return list(oc.livebench())
