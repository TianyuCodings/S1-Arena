"""ACPBench (Kokel et al., AAAI 2025; ACPBench Hard, 2025): reasoning about action, change and planning.

Thirteen PDDL domains, most with robots or agents acting in the world (blocksworld, grippers,
visitall, floortile, rovers, depot, alfworld, ...), with states and goals rendered in natural
language by hand-written templates. Every answer is computed symbolically: applicability and
plan validity by PDDL semantics, the next action by an optimal planner. Used here for four
leaves: feasibility (applicability), task success (plan validation), failure type (validation
category and the first inapplicable step) and next skill (next action towards the goal).
"""

from __future__ import annotations

import functools
import re

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import hf_rows

REPO, REV = "ibm-research/acp_bench", "05e5883d9afbcfa3bdc8f270cb345ef0b9526d4a"
URL = "https://huggingface.co/datasets/ibm-research/acp_bench"
LICENSE = "CDLA-Permissive-2.0"
SOURCE = "ACPBench, test and validation splits (Kokel et al., AAAI 2025)"
DOMAIN_NOTE = ("All 13 ACPBench domains are used (blocksworld, grippers, visitall, floortile, rovers, depot, goldminer, grid, "
               "ferry, logistics, satellite, swap and a PDDL encoding of ALFWorld): 10 test problems per domain and question "
               "type, plus the validation split (5 problems in each of 8 domains).")


@functools.lru_cache(maxsize=None)
def acp(config: str) -> tuple[dict, ...]:
    rows = []
    for split in ("test", "validation"):
        for r in hf_rows(REPO, config, split, revision=REV):
            rows.append(dict(r, split=split, domain=domain_of(r["context"])))
    return tuple(rows)


DOMAINS = {"This is a ferry": "ferry", "There are several cities": "logistics", "This is a blocksworld": "blocksworld",
           "A robot is in a grid": "grid", "A set of robots use different colors": "floortile", "This is a grippers": "grippers",
           "This is a Rovers": "rovers", "This is a visitall": "visitall", "This is a depot": "depot",
           "A robotic arm is in a grid": "goldminer", "This domain consists of satellite": "satellite", "This is a swap": "swap",
           "This is an alfworld": "alfworld"}


def domain_of(context: str) -> str:
    return next((name for prefix, name in DOMAINS.items() if context.startswith(prefix)), "other")


def quoted(question: str) -> str:
    """The action or action sequence quoted in an ACPBench question."""
    m = re.search(r'"(.+)"', question, re.S)
    return m.group(1).strip() if m else ""


class ACPTask(TaskAdapter):
    url = URL
    license = LICENSE
    tier = "B"


@register_task
class ACPApplicability(ACPTask):
    name = "embodied/feasibility/acpbench_applicability"
    leaf = "embodied/feasibility"
    source = SOURCE + ": action applicability, yes/no and multiple-choice forms"
    criteria_source = "ACPBench applicability questions, with the action moved from the question into the state"
    notes = ("Gold follows from the preconditions of the PDDL action in the given state. Two forms: is this action applicable "
             "(yes/no), and which one of four actions is applicable. " + DOMAIN_NOTE)

    def load(self):
        rows = [dict(r, kind="bool") for r in acp("acp_app_bool")]
        return rows + [dict(r, kind="mcq") for r in acp("acp_app_mcq")]

    def label_of(self, record):
        return f"{record['kind']}|{record['answer'] if record['kind'] == 'bool' else 'mcq'}"

    def to_item(self, record, rng):
        if record["kind"] == "bool":
            action = record["question"].split("applicable in this state:", 1)[1].strip().rstrip("?").strip()
            question = Question(type="boolean", instructions="Is the following action applicable in this state?")
            state = {"context": record["context"], "action": action}
            return self.item(state, {"applicable": question}, {"applicable": Gold(label="true" if record["answer"] == "yes" else "false")},
                             original_id=record["id"], domain=record["domain"], split=record["split"], form="yes/no")
        choices = record["choices"]
        options = dict(zip(choices["label"], choices["text"]))
        question = Question(type="choice", instructions="Which of the following actions will be applicable in this state?", criteria=options)
        return self.item({"context": record["context"]}, {"applicable_action": question}, {"applicable_action": Gold(label=record["answer"])},
                         original_id=record["id"], domain=record["domain"], split=record["split"], form="multiple choice")


@register_task
class ACPPlanValidity(ACPTask):
    name = "embodied/task_success/acpbench_validation"
    leaf = "embodied/task_success"
    source = SOURCE + ": plan validation, yes/no form"
    criteria_source = "ACPBench validation question ('Is the following sequence of actions ... a plan for the current state?'), with the sequence moved into the state"
    notes = ("A sequence is a plan when every action is applicable in turn and the final state satisfies the goal, as checked "
             "symbolically. " + DOMAIN_NOTE)

    def load(self):
        return list(acp("acp_val_bool"))

    def label_of(self, record):
        return record["answer"]

    def to_item(self, record, rng):
        question = Question(type="boolean", instructions=(
            "Is the following sequence of actions a plan for the current state, that is, can the actions be applied one after "
            "another and does the final state satisfy the goal?"))
        state = {"context": record["context"], "action_sequence": quoted(record["question"])}
        return self.item(state, {"is_plan": question}, {"is_plan": Gold(label="true" if record["answer"] == "yes" else "false")},
                         original_id=record["id"], domain=record["domain"], split=record["split"])


CLAIMS = {  # ACPBench's claim texts -> option ids
    "The sequence is not valid": "invalid_action",
    "The sequence is not applicable": "not_applicable",
    "The sequence is applicable, but does not achieve the goal": "goal_not_reached",
    "The sequence is a plan": "plan",
}
CLAIM_TEXT = {
    "invalid_action": "The sequence is not valid (it contains an action that does not exist in this domain)",
    "not_applicable": "The sequence is not applicable (some action cannot be applied when its turn comes)",
    "goal_not_reached": "The sequence is applicable, but does not achieve the goal",
    "plan": "The sequence is a plan",
}


@register_task
class ACPFailure(ACPTask):
    name = "embodied/failure_type/acpbench_validation"
    leaf = "embodied/failure_type"
    source = SOURCE + ": plan validation, multiple-choice form (failure category) and the first inapplicable action (failure step)"
    criteria_source = "ACPBench validation claims verbatim (with a short gloss) and the 'first inapplicable action' question"
    notes = ("Two question forms. Category: the four ACPBench claims (a malformed action, an inapplicable action, applicable but "
             "the goal is missed, a plan). Step: which action of a failing sequence is the first that cannot be applied, with "
             "the sequence's actions as options. Gold is computed symbolically. " + DOMAIN_NOTE)

    def load(self):
        rows = [dict(r, kind="category") for r in acp("acp_val_mcq")]
        steps = [dict(r, kind="step") for r in acp("acp_val_gen")]
        return rows + [r for r in steps if 0 <= int(r["answer"]) < len(re.findall(r"\([^()]*\)", quoted(r["question"])))]

    def label_of(self, record):
        if record["kind"] == "category":
            choices = dict(zip(record["choices"]["label"], record["choices"]["text"]))
            return f"category|{CLAIMS[choices[record['answer']]]}"
        return "step"

    def to_item(self, record, rng):
        sequence = quoted(record["question"])
        if record["kind"] == "category":
            choices = dict(zip(record["choices"]["label"], record["choices"]["text"]))
            gold = CLAIMS[choices[record["answer"]]]
            options = shuffled_options(CLAIM_TEXT, rng)
            question = Question(type="choice", criteria=options, instructions=(
                "Which of the following claims is true with regard to the following sequence of actions?"))
            state = {"context": record["context"], "action_sequence": sequence}
            return self.item(state, {"failure_category": question}, {"failure_category": Gold(label=gold)},
                             original_id=record["id"], domain=record["domain"], split=record["split"], form="category")
        actions = re.findall(r"\([^()]*\)", sequence)
        answer = int(record["answer"])
        options = {f"step_{i + 1}": f"step {i + 1}: {a}" for i, a in enumerate(actions)}
        question = Question(type="choice", criteria=options, instructions=(
            "The actions below are applied one after another, starting from the current state. What is the first inapplicable "
            "action in the sequence?"))
        state = {"context": record["context"], "action_sequence": " ".join(actions)}
        return self.item(state, {"first_failing_step": question}, {"first_failing_step": Gold(label=f"step_{answer + 1}")},
                         original_id=record["id"], domain=record["domain"], split=record["split"], form="step")


@register_task
class ACPNextAction(ACPTask):
    name = "embodied/next_skill/acpbench_next_action"
    leaf = "embodied/next_skill"
    source = "ACPBench Hard, next action towards the goal, test and validation splits (Kokel et al., 2025)"
    criteria_source = "ACPBench Hard question ('What is the next action that takes us towards the goal?'); options from its answer sets"
    notes = ("Options: one action that lowers the optimal plan length by one (chosen at random when there are several) and the "
             "actions ACPBench lists as applicable but not bringing the goal closer; actions the benchmark left undecided are not "
             "shown. Items without such distractors are left out. Actions are in PDDL notation; the PDDL domain is shown. "
             + DOMAIN_NOTE)

    def load(self):
        return [r for r in acp("acp_nexta_gen") if r["answer"]["yes"] and r["answer"]["no"]]

    def label_of(self, record):
        return record["domain"]

    def to_item(self, record, rng):
        answer = record["answer"]
        right = rng.choice(sorted(answer["yes"]))
        actions = sorted({right, *answer["no"]})
        rng.shuffle(actions)
        options = {f"action_{i + 1}": a for i, a in enumerate(actions)}
        gold = next(k for k, a in options.items() if a == right)
        question = Question(type="choice", criteria=options, instructions="What is the next action that takes us towards the goal?")
        state = {"context": record["context"], "pddl_domain": record["PDDL_domain"].strip()}
        return self.item(state, {"next_action": question}, {"next_action": Gold(label=gold)},
                         original_id=record["id"], domain=record["domain"], split=record["split"], optimal_plan_length=answer["opt"])
