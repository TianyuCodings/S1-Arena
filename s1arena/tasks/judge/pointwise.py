"""Leaf: pointwise rating of a single response (逐条打分).

HelpSteer2 publishes every annotator's rating (the "disagreements" file), so its gold
carries the rating distribution as a soft label next to the official score. BiGGen
Bench has one human score per response, judged against an instance-specific rubric,
so its gold is a hard label only.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import hf_rows

LEAF = "judge/pointwise_rating"

# HelpSteer2 annotation guidelines (Wang et al., 2024, Appendix G.3), verbatim; levels ordered 0..4.
HELPSTEER2 = {
    "helpfulness": (
        "Helpfulness/Understanding. How useful and helpful the response is (“overall quality rating”).",
        [
            "0 – The response is not useful or helpful at all. The response completely missed the essence of what the user wanted.",
            "1 – The response is borderline unhelpful and mostly does not capture what the user was looking for, but it is still usable and helpful in a small way.",
            "2 – The response is partially helpful but misses the overall goal of the user’s query/input in some way. The response did not fully satisfy what the user was looking for.",
            "3 – The response is mostly helpful and mainly aligned with what the user was looking for, but there is still some room for improvement.",
            "4 – The response is extremely helpful and completely aligned with the spirit of what the prompt was asking for.",
        ],
    ),
    "correctness": (
        "Correctness/Completeness. The response is based on facts, no hallucinations, no mistakes. The response covers everything required in the instruction.",
        [
            "0 – The response is completely incorrect. All information provided is wrong, false or hallucinated. If the prompt asks the assistant to do a task, the task is not at all attempted, or the wrong task was attempted in the response. The response is completely irrelevant to the prompt.",
            "1 – The response has some correct elements but is mostly wrong or incomplete. The response may contain multiple instances of hallucinations, false information, misleading information, or irrelevant information. If the prompt asks the assistant to do a task, the task was attempted with a small amount of success.",
            "2 – The response contains a mix of correct and incorrect information. The response may miss some details, contain misleading information, or minor hallucinations, but is more or less aligned with what the prompt asks for. If the prompt asks the assistant to perform a task, the task is attempted with moderate success but still has clear room for improvement.",
            "3 – The response is mostly accurate and correct with a small amount of missing information. It contains no misleading information or hallucinations. If the prompt asks the assistant to perform a task, the task is mostly successfully attempted.",
            "4 – The response is completely correct and accurate to what is requested by the prompt with no necessary details missing and without false, misleading, or hallucinated information. If the prompt asks the assistant to do a task, the task is completely done and addressed in the response.",
        ],
    ),
    "coherence": (
        "Coherence/Clarity. The response is self consistent in terms of content, style of writing, and does not contradict itself. "
        "The response can be logically followed and understood by a human. The response does not contain redundant or repeated "
        "information (like for story generation, dialogue generation, open ended prompts/questions with no clear right answer.) "
        "With this attribute we measure how lucid, cogent, and self-consistent the model’s response is.",
        [
            "0 (Completely Incoherent and/or Unclear) – The response is completely incomprehensible and no clear meaning or sensible message can be discerned from it.",
            "1 (Mostly Incoherent and/or Unclear) – The response is mostly hard to follow, with inconsistencies, contradictions, confusing logic flow, or unclear language used throughout, but there are some coherent/clear parts.",
            "2 (A Little Unclear and/or Incoherent) – The response is a little unclear. There are some inconsistencies or contradictions, run on sentences, confusing statements, or hard to follow sections of the response.",
            "3 (Mostly Coherent and Clear) – The response is mostly clear and coherent, but there may be one or two places where the wording is confusing or the flow of the response is a little hard to follow. Over all, the response can mostly be followed with a little room for improvement.",
            "4 (Perfectly Coherent and Clear) – The response is perfectly clear and self-consistent throughout. There are no contradictory assertions or statements, the writing flows logically and following the train of thought/story is not challenging.",
        ],
    ),
    "complexity": (
        "Simple vs. Complex Language. Rate the response along a simple → complex spectrum. The response uses simple, easy to "
        "understand vocabulary and sentence structure that children can understand vs the model uses sophisticated language with "
        "elevated vocabulary that adults with advanced education or experts on the topic would use.",
        [
            "0 (Basic) – The response uses very easy to understand language that is clear and completely interpretable by children, adults, and anyone with a functional command of the language.",
            "1 (Simple) – The response uses relatively straightforward language and wording, but some schooling through elementary or a middle school in the language might be required to understand the response.",
            "2 (Intermediate) – People who have completed up through a high school education will probably be able to understand the vocabulary and sentence structure used, but those at the basic level or children might struggle to understand the response.",
            "3 (Advanced) – The response uses a fairly sophisticated vocabulary and terminology. Someone majoring in this subject at a college or university could have written it and would understand the response. An average adult who does not work or study in this area could not have written the response.",
            "4 (Expert) – An expert in the field or area could have written the response. It uses specific and technically relevant vocabulary. Elevated language that someone at the simple or basic level may not understand at all. The professional language of a lawyer, scientist, engineer, or doctor falls into this category.",
        ],
    ),
    "verbosity": (
        "Succinct vs. Verbose Language. The response is direct to the point without extra wordings. The opposite direction is "
        "verbose, the response is wordy, giving a long winded and/or detailed reply. The goal here is to place the response on a "
        "spectrum from the most short, crisp answers, to the most lengthy, detailed, and/or wordy answers under the context of "
        "what a user is expecting as a response to the prompt.",
        [
            "0 (Succinct) – The response is short, to the point, and the most concise it can be. No additional information is provided outside of what is requested by the prompt (regardless of if the information or response itself is incorrect, hallucinated, or misleading. A response that gives an incorrect answer can still be succinct.).",
            "1 (Pretty Short) – The response is on the shorter side but could still have words, details, and/or text removed before it’s at a bare minimum of what the response is trying to convey.",
            "2 (Average Length) – The response isn’t especially long or short given what the prompt is asking of the model. The length is adequate for conveying a full response but isn’t particularly wordy nor particularly concise.",
            "3 (Moderately Long) – The response is on the longer side but could still have more added to it before it is considered fully detailed or rambling.",
            "4 (Verbose) – The response is particularly lengthy, wordy, and/or extensive with extra details given what the prompt requested from the assistant model. The response can be verbose regardless of if the length is due to repetition and incoherency or if it is due to rich and insightful detail.",
        ],
    ),
}


def render_helpsteer_conversation(prompt: str) -> str:
    """HelpSteer2 marks later turns with '<extra_id_1>User' / '<extra_id_1>Assistant'; the first turn is the user's."""
    turns = prompt.split("\n<extra_id_1>")
    out = [f"User: {turns[0].strip()}"]
    for turn in turns[1:]:
        role, _, text = turn.partition("\n")
        out.append(f"{role.strip()}: {text.strip()}")
    return "\n\n".join(out)


@register_task
class HelpSteer2(TaskAdapter):
    name = f"{LEAF}/helpsteer2"
    leaf = LEAF
    source = "HelpSteer2, validation split with individual annotator ratings (Wang et al., 2024)"
    url = "https://huggingface.co/datasets/nvidia/HelpSteer2"
    license = "CC BY 4.0"
    tier = "A"
    criteria_source = "Attribute definitions and 0-4 level descriptions verbatim from the HelpSteer2 annotation guidelines (Wang et al., 2024, Appendix G.3)"
    notes = ("Five 0-4 ratings per response. Soft label = distribution of the individual ratings (3 to 5 annotators, from "
             "the released disagreements file); hard label = the released score, which is the mean after the authors "
             "removed outlier ratings, so it can differ from the most common individual rating.")

    def load(self):
        from huggingface_hub import hf_hub_download

        from ...env import load_env

        load_env()
        rows = {}
        for name in ("validation.jsonl.gz", "disagreements/disagreements.jsonl.gz"):
            path = hf_hub_download("nvidia/HelpSteer2", name, repo_type="dataset")
            with gzip.open(path, "rt", encoding="utf-8") as f:
                rows[name] = [json.loads(line) for line in f]
        ratings = {(r["prompt"], r["response"]): r for r in rows["disagreements/disagreements.jsonl.gz"]}
        out = []
        for r in rows["validation.jsonl.gz"]:
            individual = ratings.get((r["prompt"], r["response"]))
            if individual and all(individual[a] for a in HELPSTEER2):
                out.append({**r, "individual": {a: individual[a] for a in HELPSTEER2}})
        return out

    def label_of(self, record):
        return str(record["helpfulness"])

    def to_item(self, record, rng):
        questions, gold = {}, {}
        for attribute, (definition, levels) in HELPSTEER2.items():
            questions[attribute] = Question(type="score", instructions=f"Rate the response on this attribute. {definition}", criteria=levels)
            counts = Counter(int(x) for x in record["individual"][attribute])
            total = sum(counts.values())
            dist = {str(level): counts.get(level, 0) / total for level in range(5)}
            gold[attribute] = Gold(label=str(int(record[attribute])), dist=dist)
        state = {"conversation": render_helpsteer_conversation(record["prompt"]), "response": record["response"]}
        return self.item(state, questions, gold, annotators=len(record["individual"]["helpfulness"]))


# Prometheus absolute-grading prompt (prometheus_eval/prompts.py, ABSOLUTE_PROMPT), output-format steps removed.
PROMETHEUS_ABSOLUTE = (
    "An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score "
    "of 5, and a score rubric representing a evaluation criteria are given. Assess the quality of the response strictly "
    "based on the given score rubric, not evaluating in general. Score rubric: {criteria}"
)


@register_task
class BiGGenBench(TaskAdapter):
    name = f"{LEAF}/biggen_bench"
    leaf = LEAF
    source = "BiGGen Bench, human evaluation split (Kim et al., 2025)"
    url = "https://huggingface.co/datasets/prometheus-eval/BiGGen-Bench-Results"
    license = "CC BY-SA 4.0 (BiGGen Bench); results repository: not stated"
    tier = "B"
    criteria_source = "Instance-specific rubric (criterion and five score descriptions) verbatim from BiGGen Bench; task wording from the Prometheus absolute-grading prompt, output-format steps removed"
    notes = ("One human score (1-5) per response, so the gold is a hard label. Responses from Llama-2-13B, "
             "Mistral-7B-Instruct-v0.2, Mixtral-8x7B-Instruct and GPT-3.5-Turbo; the 4 responses scored -1 are dropped.")

    def load(self):
        return [r for r in hf_rows("prometheus-eval/BiGGen-Bench-Results", None, "human_eval") if 1 <= int(r["human_score"]) <= 5]

    def label_of(self, record):
        return f"{record['capability']}|{record['human_score']}"

    def to_item(self, record, rng):
        rubric = record["score_rubric"]
        levels = [f"Score {i}: {rubric[f'score{i}_description']}" for i in range(1, 6)]
        question = Question(type="score", instructions=PROMETHEUS_ABSOLUTE.format(criteria=rubric["criteria"]), criteria=levels)
        state = {"system_prompt": record["system_prompt"], "instruction": record["input"], "response": record["response"], "reference_answer": record["reference_answer"]}
        gold = Gold(label=str(int(record["human_score"]) - 1))
        meta = {"capability": record["capability"], "task_name": record["task"], "response_model": record["model_name"], "original_id": record["id"]}
        return self.item(state, {"score": question}, {"score": gold}, **meta)
