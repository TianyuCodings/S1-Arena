"""Knowledge and reasoning, dataset-based sources for leaves 1-6.

Subject MCQ, factual QA recast as answer checking, commonsense, math answer
verification, logic puzzles (ZebraLogic) and temporal reasoning (BBH).
"""

from __future__ import annotations

import random
import re
from collections import defaultdict

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import hf_rows

LETTERS = "ABCDEFGHIJKLMNOP"


def lettered(options: list[str]) -> dict[str, str]:
    return {LETTERS[i]: str(text) for i, text in enumerate(options)}


# ---------- 1. subject-knowledge MCQ ----------

@register_task
class MMLUPro(TaskAdapter):
    name = "know/subject_mcq/mmlu_pro"
    leaf = "know/subject_mcq"
    source = "MMLU-Pro (Wang et al., 2024)"
    url = "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro"
    license = "MIT"
    tier = "B"
    criteria_source = "Adapted from the MMLU-Pro evaluation prompt; options are the original answer texts"
    notes = "Mostly 10 options. Widely used, so contamination is likely."

    def load(self):
        return hf_rows("TIGER-Lab/MMLU-Pro", None, "test")

    def label_of(self, record):
        return record["category"]

    def to_item(self, record, rng):
        question = Question(type="choice", instructions=f"The following is a multiple choice question about {record['category']}. Which option is correct?", criteria=lettered(record["options"]))
        return self.item({"question": record["question"]}, {"answer": question}, {"answer": Gold(label=record["answer"])}, category=record["category"], original_id=record["question_id"])


@register_task
class SuperGPQA(TaskAdapter):
    name = "know/subject_mcq/supergpqa"
    leaf = "know/subject_mcq"
    source = "SuperGPQA (M-A-P, 2025)"
    url = "https://huggingface.co/datasets/m-a-p/SuperGPQA"
    license = "ODC-BY"
    tier = "B"
    criteria_source = "Adapted from the SuperGPQA evaluation prompt; options are the original answer texts"
    notes = "285 graduate-level disciplines; the whole benchmark is the 'train' split."

    def load(self):
        return hf_rows("m-a-p/SuperGPQA", None, "train")

    def label_of(self, record):
        return record["discipline"]

    def to_item(self, record, rng):
        question = Question(type="choice", instructions=f"The following is a multiple choice question about {record['field']}. Which option is correct?", criteria=lettered(record["options"]))
        meta = {"discipline": record["discipline"], "difficulty": record["difficulty"], "original_id": record["uuid"]}
        return self.item({"question": record["question"]}, {"answer": question}, {"answer": Gold(label=record["answer_letter"])}, **meta)


# ---------- 2. factual QA recast as "is this answer correct?" ----------

class AnswerCheck(TaskAdapter):
    """Pairs each question with its gold answer or with another question's answer of the same kind."""

    tier = "C"
    criteria_source = "Written; wrong answers are gold answers of other questions of the same answer type"

    def pool(self) -> list[dict]:
        raise NotImplementedError

    def load(self):
        rows = self.pool()
        by_type = defaultdict(list)
        for r in rows:
            by_type[r["kind"]].append(r["answer"])
        rng = random.Random(f"{self.name}:pairs")
        out = []
        for i, r in enumerate(rows):
            others = [a for a in by_type[r["kind"]] if a.strip().lower() != r["answer"].strip().lower()]
            if i % 2 == 0 or not others:
                out.append(dict(r, candidate=r["answer"], label="true"))
            else:
                out.append(dict(r, candidate=rng.choice(others), label="false"))
        return out

    def label_of(self, record):
        return f"{record['label']}|{record['kind']}"

    def to_item(self, record, rng):
        question = Question(type="boolean", instructions="Is the proposed answer to this question correct?")
        state = {"question": record["question"], "proposed_answer": record["candidate"]}
        return self.item(state, {"correct": question}, {"correct": Gold(label=record["label"])}, answer_type=record["kind"])


@register_task
class SimpleQAVerified(AnswerCheck):
    name = "know/factual_judgment/simpleqa_verified"
    leaf = "know/factual_judgment"
    source = "SimpleQA Verified (Google, 2025; after Wei et al., 2024)"
    url = "https://huggingface.co/datasets/google/simpleqa-verified"
    license = "MIT"

    def pool(self):
        rows = hf_rows("google/simpleqa-verified", "simpleqa_verified", "eval")
        return [{"question": r["problem"], "answer": str(r["answer"]), "kind": r["answer_type"]} for r in rows]


@register_task
class NQOpen(AnswerCheck):
    name = "know/factual_judgment/nq_open"
    leaf = "know/factual_judgment"
    source = "Natural Questions open, validation (Lee et al., 2019)"
    url = "https://huggingface.co/datasets/google-research-datasets/nq_open"
    license = "CC BY-SA 3.0"
    notes = "Answer kind is approximated by the question word (who / when / where / how many / other)."

    @staticmethod
    def kind(question: str) -> str:
        q = question.lower()
        for word in ("how many", "who", "when", "where", "what year"):
            if q.startswith(word) or f" {word} " in f" {q} ":
                return word
        return "other"

    def pool(self):
        rows = hf_rows("google-research-datasets/nq_open", "nq_open", "validation")
        return [{"question": r["question"], "answer": r["answer"][0], "kind": self.kind(r["question"])} for r in rows if r["answer"]]


# ---------- 3. commonsense ----------

@register_task
class CommonsenseQA(TaskAdapter):
    name = "know/commonsense/commonsense_qa"
    leaf = "know/commonsense"
    source = "CommonsenseQA, validation (Talmor et al., 2019)"
    url = "https://huggingface.co/datasets/tau/commonsense_qa"
    license = "MIT"
    tier = "C"
    criteria_source = "Written; options are the original answer texts"
    notes = "Test labels are hidden, so the validation split is used."

    def load(self):
        return hf_rows("tau/commonsense_qa", None, "validation")

    def label_of(self, record):
        return record["answerKey"]

    def to_item(self, record, rng):
        options = dict(zip(record["choices"]["label"], record["choices"]["text"]))
        question = Question(type="choice", instructions="Which option is the most sensible answer to the question?", criteria=options)
        return self.item({"question": record["question"]}, {"answer": question}, {"answer": Gold(label=record["answerKey"])}, original_id=record["id"])


@register_task
class WinoGrande(TaskAdapter):
    name = "know/commonsense/winogrande"
    leaf = "know/commonsense"
    source = "WinoGrande XL, validation (Sakaguchi et al., 2020)"
    url = "https://huggingface.co/datasets/allenai/winogrande"
    license = "CC BY"
    tier = "C"
    criteria_source = "Written"
    notes = "Test labels are hidden, so the validation split is used."

    def load(self):
        return hf_rows("allenai/winogrande", "winogrande_xl", "validation")

    def label_of(self, record):
        return record["answer"]

    def to_item(self, record, rng):
        options = {"A": record["option1"], "B": record["option2"]}
        question = Question(type="choice", instructions="Which option correctly fills the blank (_) in the sentence?", criteria=options)
        gold = Gold(label="A" if record["answer"] == "1" else "B")
        return self.item({"sentence": record["sentence"]}, {"answer": question}, {"answer": gold})


@register_task
class PIQA(TaskAdapter):
    name = "know/commonsense/piqa"
    leaf = "know/commonsense"
    source = "PIQA, validation (Bisk et al., 2020)"
    url = "https://huggingface.co/datasets/baber/piqa"
    license = "AFL-3.0"
    tier = "C"
    criteria_source = "Written"
    notes = "Third-party parquet mirror; test labels are hidden, so the validation split is used."

    def load(self):
        return hf_rows("baber/piqa", None, "validation")

    def label_of(self, record):
        return str(record["label"])

    def to_item(self, record, rng):
        options = {"A": record["sol1"], "B": record["sol2"]}
        question = Question(type="choice", instructions="Which solution is the more sensible way to achieve the goal?", criteria=options)
        gold = Gold(label="A" if int(record["label"]) == 0 else "B")
        return self.item({"goal": record["goal"]}, {"answer": question}, {"answer": gold})


# ---------- 4. math answer verification ----------

@register_task
class ProcessBench(TaskAdapter):
    name = "know/math_verification/processbench"
    leaf = "know/math_verification"
    source = "ProcessBench model solutions (Zheng et al., 2024)"
    url = "https://huggingface.co/datasets/Qwen/ProcessBench"
    license = "Apache-2.0"
    tier = "C"
    criteria_source = "Written; gold is the dataset's final-answer correctness label"
    notes = "Solutions from Qwen and Llama models; half have a correct final answer."

    def load(self):
        rows = []
        for split in ("gsm8k", "math", "olympiadbench", "omnimath"):
            rows += [dict(r, split=split) for r in hf_rows("Qwen/ProcessBench", None, split)]
        return rows

    def label_of(self, record):
        return f"{record['split']}|{record['final_answer_correct']}"

    def to_item(self, record, rng):
        question = Question(type="boolean", instructions="Is the final answer of this solution correct?")
        state = {"problem": record["problem"], "solution": "\n\n".join(record["steps"])}
        gold = Gold(label="true" if record["final_answer_correct"] else "false")
        return self.item(state, {"correct": question}, {"correct": gold}, split=record["split"], original_id=record["id"])


@register_task
class VerifyBenchMath(TaskAdapter):
    name = "know/math_verification/verifybench"
    leaf = "know/math_verification"
    source = "VerifyBench, math problems (Yan et al., 2025)"
    url = "https://huggingface.co/datasets/ZJU-REAL/VerifyBench"
    license = "MIT"
    tier = "C"
    criteria_source = "Written; gold is the dataset's human-checked correctness label"
    notes = "Only rows whose source starts with math_reasoning are used."

    def load(self):
        return [r for r in hf_rows("ZJU-REAL/VerifyBench", None, "VerifyBench") if r["source"].startswith("math_reasoning")]

    def label_of(self, record):
        return str(record["gold_correct"])

    def to_item(self, record, rng):
        question = Question(type="boolean", instructions="Does the final answer of the completion match the reference answer?")
        state = {"question": record["question"], "reference_answer": str(record["answer"]), "completion": record["completion"]}
        gold = Gold(label="true" if record["gold_correct"] else "false")
        return self.item(state, {"matches": question}, {"matches": gold}, original_id=record["completion_id"])


# ---------- 5. logic puzzles ----------

@register_task
class ZebraLogicMC(TaskAdapter):
    name = "know/logic_puzzles/zebralogic"
    leaf = "know/logic_puzzles"
    source = "ZebraLogic, multiple-choice mode (Lin et al., 2025)"
    url = "https://huggingface.co/datasets/WildEval/ZebraLogic"
    license = "Not stated"
    tier = "C"
    criteria_source = "Written; options are the original answer texts"
    notes = "The authors state no license; included in the repository pending a license check."

    def load(self):
        return hf_rows("WildEval/ZebraLogic", "mc_mode", "test")

    def label_of(self, record):
        return str(len(record["choices"]))

    def to_item(self, record, rng):
        options = lettered(record["choices"])
        answer = next((k for k, v in options.items() if v == str(record["answer"])), None)
        if answer is None:
            return None
        question = Question(type="choice", instructions=record["question"], criteria=options)
        return self.item({"puzzle": record["puzzle"]}, {"answer": question}, {"answer": Gold(label=answer)}, original_id=record["id"])


FOLIO_OPTIONS = {
    "true": "The conclusion follows from the premises.",
    "false": "The premises imply that the conclusion is false.",
    "uncertain": "The premises determine neither the conclusion nor its negation.",
}


@register_task
class FOLIO(TaskAdapter):
    name = "know/logic_puzzles/folio"
    leaf = "know/logic_puzzles"
    source = "FOLIO v2, first-order logic reasoning in natural language, validation (Han et al., 2024)"
    url = "https://huggingface.co/datasets/yale-nlp/FOLIO"
    license = "MIT"
    tier = "B"
    criteria_source = "Adapted from the FOLIO task definition: true, false or uncertain given the premises"
    notes = "Gated on Hugging Face: accept the terms with the account of HF_API_KEY first. The test set is not public, so the validation split is used."

    def load(self):
        return hf_rows("yale-nlp/FOLIO", None, "validation")

    def label_of(self, record):
        return str(record["label"]).strip().lower()

    def to_item(self, record, rng):
        label = self.label_of(record)
        if label not in FOLIO_OPTIONS:
            return None
        premises = record["premises"] if isinstance(record["premises"], str) else "\n".join(record["premises"])
        question = Question(type="choice", instructions="Based only on the premises, is the conclusion true, false, or uncertain?", criteria=FOLIO_OPTIONS)
        state = {"premises": premises.strip(), "conclusion": str(record["conclusion"]).strip()}
        return self.item(state, {"answer": question}, {"answer": Gold(label=label)}, original_id=record.get("example_id"), story_id=record.get("story_id"))


# ---------- 6. temporal reasoning ----------

@register_task
class BBHDateUnderstanding(TaskAdapter):
    name = "know/date_reasoning/bbh_date_understanding"
    leaf = "know/date_reasoning"
    source = "BIG-Bench Hard, date understanding (Suzgun et al., 2023)"
    url = "https://huggingface.co/datasets/lukaemon/bbh"
    license = "MIT"
    tier = "C"
    criteria_source = "Options parsed from the original task input"

    def load(self):
        return hf_rows("lukaemon/bbh", "date_understanding", "test")

    def label_of(self, record):
        return record["target"]

    def to_item(self, record, rng):
        question_text, _, options_text = record["input"].partition("Options:")
        options = dict(re.findall(r"\(([A-F])\)\s*(.+)", options_text))
        label = record["target"].strip("() ")
        if label not in options:
            return None
        question = Question(type="choice", instructions="Which option answers the question?", criteria={k: v.strip() for k, v in options.items()})
        return self.item({"question": question_text.strip()}, {"answer": question}, {"answer": Gold(label=label)})
