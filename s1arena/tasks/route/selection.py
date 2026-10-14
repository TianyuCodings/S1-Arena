"""Leaf: route the request to a model or tier under a cost constraint (路由到哪个模型或档位).

The candidates form a price ladder and the instruction asks for the cheapest candidate that
still answers correctly. Gold is that candidate, read off the recorded outcomes of every
candidate on the same prompt (RouterBench's "oracle" router, with the cost order fixed by the
listed prices). Requests that no candidate answered correctly are left out, because then no
choice is correct.
"""

from __future__ import annotations

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import slug
from . import outcomes as oc

LEAF = "route/model_selection"
INSTRUCTIONS = ("Route this request to one of the candidate models, listed with their prices. Choose the cheapest candidate "
                "that will answer the request correctly; a more expensive model is only worth it if every cheaper one would get it wrong.")


class CheapestCorrect(TaskAdapter):
    leaf = LEAF
    tier = "B"
    criteria_source = "Adapted from RouterBench's oracle router (the cheapest model that answers correctly); prices from the listed sources"
    LADDER: dict[str, str] = {}  # model id -> description with price, from cheapest to most expensive
    PROTOCOL = ""

    def rows(self) -> list[dict]:
        raise NotImplementedError

    def load(self):
        out = []
        for r in self.rows():
            if not all(m in r["score"] for m in self.LADDER):
                continue
            best = next((m for m in self.LADDER if r["score"][m] >= 1), None)
            if best is not None:
                out.append(dict(r, best=best))
        return out

    def label_of(self, record):
        return f"{record['best']}|{record['group']}"

    def to_item(self, record, rng):
        options = {slug(m.split("/")[-1]): description for m, description in self.LADDER.items()}
        question = Question(type="choice", instructions=f"{INSTRUCTIONS} {self.PROTOCOL}", criteria=shuffled_options(options, rng))
        gold = Gold(label=slug(record["best"].split("/")[-1]))
        outcomes = {slug(m.split("/")[-1]): int(record["score"][m]) for m in self.LADDER}
        return self.item({"prompt": record["prompt"]}, {"model": question}, {"model": gold}, group=record["group"], outcomes=outcomes, original_id=record["id"])


RB_LADDER = ["mistralai/mixtral-8x7b-chat", "gpt-3.5-turbo-1106", "claude-v2", "gpt-4-1106-preview"]


@register_task
class RouterBenchSelection(CheapestCorrect):
    name = f"{LEAF}/routerbench"
    source = "RouterBench, zero-shot MMLU, HellaSwag and ARC-Challenge prompts (Hu et al., 2024)"
    url = oc.ROUTERBENCH_URL
    license = oc.ROUTERBENCH_LICENSE
    notes = ("Four-model ladder; the price shown is each model's average cost per request measured in RouterBench on these "
             "evals. About 8% of requests, which no candidate answered correctly, are left out. Mixtral alone answers about "
             "60% of the kept requests, so always choosing the cheapest model is a strong baseline, as in real traffic.")
    PROTOCOL = oc.ROUTERBENCH_PROTOCOL

    LADDER = {m: oc.ROUTERBENCH_MODELS[m] for m in RB_LADDER}  # the cost is added in load(), from the data

    def load(self):
        self.LADDER = {m: f"{oc.ROUTERBENCH_MODELS[m]}; average cost per request ${oc.routerbench_mean_cost(m):.6f}" for m in RB_LADDER}
        return super().load()

    def rows(self):
        return list(oc.routerbench())


LB_LADDER = {
    "gemini-1.5-flash-002": "Gemini 1.5 Flash (Google, gemini-1.5-flash-002); $0.075 per million input tokens, $0.30 per million output tokens",
    "gpt-4o-mini-2024-07-18": "GPT-4o mini (OpenAI, gpt-4o-mini-2024-07-18); $0.15 per million input tokens, $0.60 per million output tokens",
    "gpt-4o-2024-08-06": "GPT-4o (OpenAI, gpt-4o-2024-08-06); $2.50 per million input tokens, $10 per million output tokens",
    "o1-preview-2024-09-12": "o1-preview (OpenAI reasoning model, o1-preview-2024-09-12); $15 per million input tokens, $60 per million output tokens, reasoning tokens billed as output",
}


@register_task
class LiveBenchSelection(CheapestCorrect):
    name = f"{LEAF}/livebench"
    source = "LiveBench code generation, code completion, typo fixing and Connections puzzles (White et al., 2025)"
    url = oc.LIVEBENCH_URL
    license = oc.LIVEBENCH_LICENSE
    notes = ("Four-model ladder with the providers' list prices at the time of evaluation (Gemini 1.5 Flash after the "
             "October 2024 price cut). Requests no candidate solved are left out. The license of the LiveBench question sets is not stated.")
    LADDER = LB_LADDER
    PROTOCOL = oc.LIVEBENCH_PROTOCOL

    def rows(self):
        return list(oc.livebench())
