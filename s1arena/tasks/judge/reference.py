"""Leaf: answer correctness against a reference answer (答案正确性，有参考答案).

A judge sees a question, the reference (gold) answer and a candidate answer, and decides
whether the candidate is equivalent to / consistent with the reference. All labels are
human judgments; know/math_verification already uses VerifyBench and ProcessBench, so
this leaf uses other data (extractive QA, open-domain QA and a multi-domain verifier set).
"""

from __future__ import annotations

import json
from collections import defaultdict

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import fetch

LEAF = "judge/reference_answer"

# Answer Equivalence rating question Q2 (Bulian et al., 2022, Section 3.1), verbatim.
AE_INSTRUCTIONS = (
    "The first answer is the reference answer to the question; the second answer is a candidate. Would using the second "
    "answer in place of the first answer convey at least the same information to someone asking this question, without "
    "leaving out any important information nor adding any misleading or superfluous information? (Note that either adding "
    "important information or removing superfluous or misleading information in the second answer from the first answer "
    "would still convey at least the same information.)"
)


@register_task
class AnswerEquivalence(TaskAdapter):
    name = f"{LEAF}/answer_equivalence"
    leaf = LEAF
    source = "Answer Equivalence (AE) dataset, test split, SQuAD questions (Bulian et al., 2022)"
    url = "https://github.com/google-research-datasets/answer-equivalence-dataset"
    license = "Apache-2.0 (questions and passages from SQuAD, CC BY-SA 4.0)"
    tier = "A"
    criteria_source = "Rating question Q2 verbatim from Bulian et al. (2022), Section 3.1"
    notes = ("Candidates are non-exact-match predictions of BiDAF, XLNet and Luke. About a third of the examples have 2-4 "
             "ratings: soft label = share of raters judging the candidate equivalent, hard label = majority; the few "
             "tied examples (which the authors broke at random) are dropped.")

    def load(self):
        path = fetch("https://raw.githubusercontent.com/google-research-datasets/answer-equivalence-dataset/main/v1/ae_test.jsonl", "ae_test.jsonl")
        groups = defaultdict(list)
        with path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                groups[(r["qid"], r["reference"], r["candidate"])].append(r)
        out = []
        for ratings in groups.values():
            share = sum(r["label"] == "equivalent" for r in ratings) / len(ratings)
            if share == 0.5:
                continue
            first = ratings[0]
            out.append({"qid": first["qid"], "question": first["question"], "context": first["context"], "reference": first["reference"],
                        "candidate": first["candidate"], "share": share, "raters": len(ratings)})
        return out

    def label_of(self, record):
        return f"{record['share'] > 0.5}|{record['raters'] > 1}"

    def to_item(self, record, rng):
        question = Question(type="boolean", instructions=AE_INSTRUCTIONS)
        state = {"context": record["context"], "question": record["question"], "first_answer": record["reference"], "second_answer": record["candidate"]}
        share = record["share"]
        gold = Gold(label="true" if share > 0.5 else "false", dist={"true": share, "false": 1.0 - share} if record["raters"] > 1 else None)
        return self.item(state, {"equivalent": question}, {"equivalent": gold}, original_id=record["qid"], raters=record["raters"])


# EVOUNA human annotation guidelines, task (3) (Wang et al., 2023, Appendix C.2), verbatim.
EVOUNA_INSTRUCTIONS = (
    "Here is a question, a set of golden answers (split with /), an AI-generated answer. Judge whether the AI-generated "
    "answer is correct according to the question and golden answers. Guidelines: "
    "If the question specifies a number (e.g., names of four people), and the response does not meet this requirement "
    "(e.g., provides only one name), the answer is deemed incorrect. "
    "Spelling errors in the responses are considered mistakes. For example, if \"golden answer\" is misspelled as "
    "\"gloden answer,\" the response is marked as incorrect. "
    "For questions related to specific times, such as \"When was the term social justice first used?\" a response of "
    "\"1840s\" would be considered correct. However, if the answer needs to be precise to a specific day, month, and year, "
    "each time component needs to be factually accurate for the response to be marked as correct. "
    "For location-based queries, like \"Where was Oak Island filmed?\", a response of \"Canada\" would be deemed correct. "
    "But, if the answer requires specific details like state, city, or county, each geographical component must be "
    "accurate for the answer to be considered correct. "
    "If there is a direct answer and subsequent explanation in the response, then only focus on whether the direct answer "
    "is correct, not whether the subsequent explanation is correct."
)


@register_task
class EVOUNA(TaskAdapter):
    name = f"{LEAF}/evouna"
    leaf = LEAF
    source = "EVOUNA, Natural Questions and TriviaQA answers of five QA systems judged by humans (Wang et al., 2023)"
    url = "https://github.com/wangcunxiang/QA-Eval"
    license = "Apache-2.0"
    tier = "A"
    criteria_source = "Human annotation guidelines, task (3), verbatim from Wang et al. (2023), Appendix C.2"
    notes = ("Answers from DPR+FiD, GPT-3.5 (text-davinci-003), ChatGPT-3.5, GPT-4 and Bing Chat to NQ-open and TriviaQA "
             "test questions; questions the annotators marked improper are dropped.")
    MODELS = {"fid": "DPR+FiD", "gpt35": "GPT-3.5", "chatgpt": "ChatGPT-3.5", "gpt4": "ChatGPT-4", "newbing": "Bing Chat"}

    def load(self):
        out = []
        for subset in ("NQ", "TQ"):
            path = fetch(f"https://raw.githubusercontent.com/wangcunxiang/QA-Eval/main/EVOUNA/{subset}.json", f"evouna_{subset}.json")
            with path.open(encoding="utf-8") as f:
                rows = json.load(f)
            for i, r in enumerate(rows):
                if r.get("improper"):
                    continue
                for key in self.MODELS:
                    answer, judge = r.get(f"answer_{key}"), r.get(f"judge_{key}")
                    if isinstance(judge, bool) and isinstance(answer, str) and answer.strip():
                        out.append({"subset": subset, "index": i, "model": key, "question": r["question"], "golden": r["golden_answer"], "answer": answer.strip(), "correct": judge})
        return out

    def label_of(self, record):
        return f"{record['subset']}|{record['model']}|{record['correct']}"

    def to_item(self, record, rng):
        question = Question(type="boolean", instructions=EVOUNA_INSTRUCTIONS)
        state = {"question": record["question"], "golden_answers": record["golden"], "ai_generated_answer": record["answer"]}
        gold = Gold(label="true" if record["correct"] else "false")
        return self.item(state, {"correct": question}, {"correct": gold}, subset=record["subset"], answer_model=self.MODELS[record["model"]])


# CompassVerifier grading prompt (src/prompts.py, CV_PROMPT), output-format instructions removed.
COMPASS_INSTRUCTIONS = (
    "Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the "
    "standard answers, that is, whether the candidates answered correctly. Here are some evaluation criteria: "
    "1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the "
    "standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard "
    "answer according to the form of the question. THE STANDARD ANSWER IS ALWAYS CORRECT AND THE QUESTION IS PERFECTLY "
    "VALID. NEVER QUESTION THEM. "
    "2. ONLY compare the FINAL ANSWER - COMPLETELY IGNORE any potential errors in the REASONING PROCESSES. "
    "3. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers "
    "may be a textual description, as long as the meaning expressed is the same. Before making a judgment, please understand "
    "the question and the standard answer first, and then judge whether the candidate's answer is correct. "
    "4. Some answers may consist of multiple items, such as multiple-choice questions, multiple-select questions, "
    "fill-in-the-blank questions, etc. Regardless of the question type, the final answer will be considered correct as long "
    "as it matches the standard answer, regardless of whether the reasoning process is correct. For multiple-select questions "
    "and multi-blank fill-in-the-blank questions, all corresponding options or blanks must be answered correctly and match "
    "the standard answer exactly to be deemed correct. "
    "5. If the prediction is given with \\boxed{}, please ignore the \\boxed{} and only judge whether the candidate's answer "
    "is consistent with the standard answer. "
    "6. If the candidate's answer is invalid (e.g., incomplete (cut off mid-response), lots of unnormal repetitive content, "
    "or irrelevant to the question, saying it can't answer the question because some irresistible factors, like ethical "
    "issues, no enough information, etc.), select option C (INVALID). "
    "Grade the candidate's answer as one of the options."
)
COMPASS_OPTIONS = {"A": "CORRECT", "B": "INCORRECT", "C": "INVALID"}


@register_task
class VerifierBench(TaskAdapter):
    name = f"{LEAF}/verifierbench"
    leaf = LEAF
    source = "VerifierBench (CompassVerifier; Liu et al., 2025)"
    url = "https://huggingface.co/datasets/opencompass/VerifierBench"
    license = "CC BY-SA 4.0"
    tier = "B"
    criteria_source = "CompassVerifier grading prompt (src/prompts.py, CV_PROMPT), output-format instructions removed"
    notes = ("Responses of 50+ models on 15 datasets (math, knowledge, science, general reasoning) that automatic verifiers "
             "disagreed on, labelled by human annotators as correct, incorrect or invalid.")

    def load(self):
        from huggingface_hub import hf_hub_download

        from ...env import load_env

        load_env()
        path = hf_hub_download("opencompass/VerifierBench", "verifierbench.json", repo_type="dataset")
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        return [r for r in rows if r["gold_judgment"] in COMPASS_OPTIONS]

    def label_of(self, record):
        return f"{record['domain']}|{record['gold_judgment']}"

    def to_item(self, record, rng):
        question = Question(type="choice", instructions=COMPASS_INSTRUCTIONS, criteria=shuffled_options(COMPASS_OPTIONS, rng))
        state = {"original_question": record["question"], "standard_answer": record["gold_answer"], "candidate_answer": record["llm_response"]}
        meta = {"domain": record["domain"], "answer_type": record["answer_type"], "data_source": record["data_source"]}
        return self.item(state, {"grade": question}, {"grade": Gold(label=record["gold_judgment"])}, **meta)
