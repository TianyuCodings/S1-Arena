"""Knowledge and reasoning, dataset-based sources for leaves 7-12.

Quantitative NLI, multi-hop QA, causal reasoning, fact verification with evidence,
citation support and answerability.
"""

from __future__ import annotations

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import hf_rows

# Label wording shared by the NLI-style sources (ChaosNLI crowdworker instructions, Nie et al., 2020).
NLI_INSTRUCTIONS = (
    "Given a context, a statement can be either: Definitely correct (Entailment); or Definitely incorrect "
    "(Contradiction); or Neither (Neutral). Your goal is to choose the correct category for a given pair of "
    "context and statement."
)
NLI_OPTIONS = {"entailment": "Definitely correct", "contradiction": "Definitely incorrect", "neutral": "Neither"}


# ---------- 7. numeric comparison: quantitative NLI ----------

@register_task
class EQUATE(TaskAdapter):
    name = "know/numeric_comparison/equate"
    leaf = "know/numeric_comparison"
    source = "EQUATE quantitative NLI: AWPNLI, NewsNLI, RedditNLI (Ravichander et al., 2019)"
    url = "https://huggingface.co/datasets/tasksource/equate"
    license = "MIT"
    tier = "B"
    criteria_source = "NLI label wording from the ChaosNLI crowdworker instructions"
    notes = "Three of the five EQUATE subsets are in this copy."

    def load(self):
        return [r for r in hf_rows("tasksource/equate", None, "train") if str(r["gold_label"]).lower() in NLI_OPTIONS]

    def label_of(self, record):
        return str(record["gold_label"]).lower()

    def to_item(self, record, rng):
        question = Question(type="choice", instructions=NLI_INSTRUCTIONS, criteria=shuffled_options(NLI_OPTIONS, rng))
        state = {"context": record["sentence1"], "statement": record["sentence2"]}
        return self.item(state, {"relation": question}, {"relation": Gold(label=self.label_of(record))})


# ---------- 8. multi-hop QA ----------

@register_task
class StrategyQA(TaskAdapter):
    name = "know/multihop/strategyqa"
    leaf = "know/multihop"
    source = "StrategyQA, re-split test (Geva et al., 2021)"
    url = "https://huggingface.co/datasets/ChilleD/StrategyQA"
    license = "MIT"
    tier = "C"
    criteria_source = "Written; the gold facts are withheld, so the reasoning steps are implicit"
    notes = "A re-split of the original labelled train set; widely used, so contamination is likely."

    def load(self):
        return hf_rows("ChilleD/StrategyQA", None, "test")

    def label_of(self, record):
        return str(record["answer"])

    def to_item(self, record, rng):
        question = Question(type="boolean", instructions="Is the answer to this yes/no question yes?")
        gold = Gold(label="true" if record["answer"] else "false")
        return self.item({"question": record["question"]}, {"yes": question}, {"yes": gold}, original_id=record["qid"])


class ComparisonQA(TaskAdapter):
    """HotpotQA-layout comparison questions: yes/no ones become boolean items, 'X or Y' ones a two-way choice."""

    tier = "C"
    criteria_source = "Written; the two options are the entities named in the question"
    TYPES: tuple[str, ...] = ("comparison",)

    def rows(self) -> list[dict]:
        raise NotImplementedError

    def load(self):
        out = []
        for r in self.rows():
            if r["type"] not in self.TYPES:
                continue
            answer = str(r["answer"]).strip()
            titles = list(dict.fromkeys(r["supporting_facts"]["title"]))
            named = [t for t in titles if t.lower() in r["question"].lower()]
            if answer.lower() in ("yes", "no"):
                out.append(dict(r, form="yes_no", answer=answer.lower()))
            elif len(named) == 2 and answer in named:
                out.append(dict(r, form="which", answer=answer, candidates=named))
        return out

    def label_of(self, record):
        return f"{record['form']}|{record['answer'] if record['form'] == 'yes_no' else 'entity'}"

    def to_item(self, record, rng):
        context = "\n\n".join(f"{t}: {' '.join(s)}" for t, s in zip(record["context"]["title"], record["context"]["sentences"]))
        state = {"context": context, "question": record["question"]}
        if record["form"] == "yes_no":
            question = Question(type="boolean", instructions="Using the context paragraphs, is the answer to the question yes?")
            return self.item(state, {"yes": question}, {"yes": Gold(label="true" if record["answer"] == "yes" else "false")}, qtype=record["type"])
        candidates = list(record["candidates"])
        rng.shuffle(candidates)
        options = {"A": candidates[0], "B": candidates[1]}
        question = Question(type="choice", instructions="Using the context paragraphs, which option answers the question?", criteria=options)
        label = "A" if candidates[0] == record["answer"] else "B"
        return self.item(state, {"answer": question}, {"answer": Gold(label=label)}, qtype=record["type"])


@register_task
class HotpotQAComparison(ComparisonQA):
    name = "know/multihop/hotpotqa_comparison"
    leaf = "know/multihop"
    source = "HotpotQA, distractor setting, comparison questions (Yang et al., 2018)"
    url = "https://huggingface.co/datasets/hotpotqa/hotpot_qa"
    license = "CC BY-SA 4.0"
    notes = "Validation split; the input holds all 10 paragraphs (2 gold, 8 distractors)."

    def rows(self):
        return hf_rows("hotpotqa/hotpot_qa", "distractor", "validation")


@register_task
class TwoWikiComparison(ComparisonQA):
    name = "know/multihop/2wikimultihopqa"
    leaf = "know/multihop"
    source = "2WikiMultihopQA, comparison and bridge-comparison questions (Ho et al., 2020)"
    url = "https://huggingface.co/datasets/framolfese/2WikiMultihopQA"
    license = "Apache-2.0"
    notes = "Validation split (test answers are hidden); the input holds all 10 paragraphs."
    TYPES = ("comparison", "bridge_comparison")

    def rows(self):
        return hf_rows("framolfese/2WikiMultihopQA", None, "validation")


# ---------- 9. causal reasoning ----------

class CausalChoice(TaskAdapter):
    tier = "B"
    criteria_source = "COPA question wording ('What was the CAUSE of this?' / 'What happened as a RESULT?')"

    def label_of(self, record):
        return f"{record['question']}|{record['label']}"

    def to_item(self, record, rng):
        ask = "What was the CAUSE of this?" if record["question"] == "cause" else "What happened as a RESULT?"
        options = {"A": record["choice1"], "B": record["choice2"]}
        question = Question(type="choice", instructions=f"{ask} Choose the more plausible alternative.", criteria=options)
        gold = Gold(label="A" if int(record["label"]) == 0 else "B")
        return self.item({"premise": record["premise"]}, {"alternative": question}, {"alternative": gold}, relation=record["question"])


@register_task
class COPA(CausalChoice):
    name = "know/causal/copa"
    leaf = "know/causal"
    source = "COPA, original test set with labels (Roemmele et al., 2011), via Balanced COPA"
    url = "https://huggingface.co/datasets/pkavumba/balanced-copa"
    license = "CC BY 4.0 (card); original COPA BSD-2-Clause"
    notes = "The SuperGLUE copy hides these test labels; this copy includes them. Heavily contaminated."

    def load(self):
        return hf_rows("pkavumba/balanced-copa", None, "test")


@register_task
class ECARE(CausalChoice):
    name = "know/causal/e_care"
    leaf = "know/causal"
    source = "e-CARE, validation (Du et al., 2022)"
    url = "https://huggingface.co/datasets/12ml/e-CARE"
    license = "MIT"
    notes = "Test labels are hidden, so the validation split is used."

    def load(self):
        return hf_rows("12ml/e-CARE", None, "validation")


# ---------- 10. fact verification with evidence ----------

VERDICTS = {
    "supports": "The evidence supports the claim.",
    "refutes": "The evidence refutes the claim.",
    "not_enough_info": "The evidence neither supports nor refutes the claim.",
}
VERIFY = "Given the evidence, which verdict applies to the claim?"


@register_task
class VitaminC(TaskAdapter):
    name = "know/fact_verification/vitaminc"
    leaf = "know/fact_verification"
    source = "VitaminC contrastive fact verification (Schuster et al., 2021)"
    url = "https://huggingface.co/datasets/tals/vitaminc"
    license = "CC BY-SA 3.0"
    tier = "B"
    criteria_source = "Adapted from the FEVER label definitions"
    MAP = {"SUPPORTS": "supports", "REFUTES": "refutes", "NOT ENOUGH INFO": "not_enough_info"}

    def load(self):
        return hf_rows("tals/vitaminc", None, "test")

    def label_of(self, record):
        return f"{record['label']}|{record['revision_type']}"

    def to_item(self, record, rng):
        question = Question(type="choice", instructions=VERIFY, criteria=shuffled_options(VERDICTS, rng))
        state = {"claim": record["claim"], "evidence": record["evidence"]}
        return self.item(state, {"verdict": question}, {"verdict": Gold(label=self.MAP[record["label"]])}, original_id=record["unique_id"])


@register_task
class SciFact(TaskAdapter):
    name = "know/fact_verification/scifact"
    leaf = "know/fact_verification"
    source = "SciFact claims against paper abstracts, validation (Wadden et al., 2020)"
    url = "https://huggingface.co/datasets/allenai/scifact_entailment"
    license = "CC BY-NC 2.0"
    tier = "B"
    criteria_source = "Adapted from the SciFact label definitions"
    notes = "Read from the Hub's parquet conversion; test labels are hidden."
    MAP = {"SUPPORT": "supports", "CONTRADICT": "refutes", "NEI": "not_enough_info"}

    def load(self):
        return hf_rows("allenai/scifact_entailment", None, "validation", revision="refs/convert/parquet")

    def label_of(self, record):
        return record["verdict"]

    def to_item(self, record, rng):
        if record["verdict"] not in self.MAP:
            return None
        question = Question(type="choice", instructions=VERIFY, criteria=shuffled_options(VERDICTS, rng))
        state = {"claim": record["claim"], "evidence": f"{record['title']}. " + " ".join(record["abstract"])}
        return self.item(state, {"verdict": question}, {"verdict": Gold(label=self.MAP[record["verdict"]])}, original_id=record["claim_id"])


@register_task
class AVeriTeC(TaskAdapter):
    name = "know/fact_verification/averitec"
    leaf = "know/fact_verification"
    source = "AVeriTeC real-world claims with question-answer evidence, dev (Schlichtkrull et al., 2023)"
    url = "https://huggingface.co/datasets/pminervini/averitec"
    license = "CC BY-NC 4.0"
    tier = "B"
    criteria_source = "Adapted from the AVeriTeC label definitions"
    notes = "The justification and fact-checking article are withheld because they reveal the label."
    OPTIONS = {
        "supported": "The evidence supports the claim.",
        "refuted": "The evidence refutes the claim.",
        "not_enough_evidence": "There is not enough evidence to support or refute the claim.",
        "conflicting_or_cherrypicking": "The evidence both supports and refutes the claim, or the claim is technically true but misleading through cherry-picking.",
    }
    MAP = {"Supported": "supported", "Refuted": "refuted", "Not Enough Evidence": "not_enough_evidence", "Conflicting Evidence/Cherrypicking": "conflicting_or_cherrypicking"}

    def load(self):
        return [r for r in hf_rows("pminervini/averitec", None, "dev") if r["label"] in self.MAP]

    def label_of(self, record):
        return record["label"]

    def to_item(self, record, rng):
        lines = []
        for qa in record["questions"] or []:
            answers = "; ".join(str(a.get("answer", "")).strip() + (f" ({a['boolean_explanation']})" if a.get("boolean_explanation") else "") for a in qa.get("answers") or [])
            lines.append(f"Q: {qa.get('question', '').strip()}\nA: {answers}")
        question = Question(type="choice", instructions=VERIFY, criteria=shuffled_options(self.OPTIONS, rng))
        state = {"claim": record["claim"], "claim_date": record.get("claim_date") or "", "evidence": "\n".join(lines)}
        return self.item(state, {"verdict": question}, {"verdict": Gold(label=self.MAP[record["label"]])})


# ---------- 11. citation support ----------

@register_task
class WiCE(TaskAdapter):
    name = "know/citation_support/wice"
    leaf = "know/citation_support"
    source = "WiCE, Wikipedia citation entailment, sub-claims (Kamoi et al., 2023)"
    url = "https://huggingface.co/datasets/jon-tow/wice"
    license = "ODC-BY (annotations); Wikipedia CC BY-SA and Common Crawl terms (text)"
    tier = "B"
    criteria_source = "Adapted from the WiCE label definitions"
    OPTIONS = {
        "supported": "All of the information in the claim is supported by the cited source.",
        "partially_supported": "Some, but not all, of the information in the claim is supported by the cited source.",
        "not_supported": "None of the information in the claim is supported by the cited source.",
    }

    def load(self):
        return hf_rows("jon-tow/wice", "subclaim", "test")

    def label_of(self, record):
        return record["label"]

    def to_item(self, record, rng):
        if record["label"] not in self.OPTIONS:
            return None
        question = Question(type="choice", instructions="How much of the claim does the cited source support?", criteria=shuffled_options(self.OPTIONS, rng))
        state = {"claim": record["claim"], "cited_source": " ".join(record["evidence"])}
        return self.item(state, {"support": question}, {"support": Gold(label=record["label"])})


@register_task
class AttributionBench(TaskAdapter):
    name = "know/citation_support/attributionbench"
    leaf = "know/citation_support"
    source = "AttributionBench, in-distribution test (Li et al., 2024)"
    url = "https://huggingface.co/datasets/osunlp/AttributionBench"
    license = "Apache-2.0 (component datasets keep their own licenses)"
    tier = "B"
    criteria_source = "Adapted from the AttributionBench task definition"
    notes = "Balanced attributable / not attributable; drawn from ExpertQA, Stanford-GenSearch, AttributedQA and LFQA."

    def load(self):
        return hf_rows("osunlp/AttributionBench", "full_data", "test")

    def label_of(self, record):
        return f"{record['attribution_label']}|{record['src_dataset']}"

    def to_item(self, record, rng):
        question = Question(type="boolean", instructions="Is the claim fully supported by the references?")
        state = {"question": record["question"], "claim": record["claim"], "references": "\n\n".join(record["references"] or [])}
        gold = Gold(label="true" if record["attribution_label"] == "attributable" else "false")
        return self.item(state, {"attributable": question}, {"attributable": gold}, src_dataset=record["src_dataset"])


# ---------- 12. answerability ----------

@register_task
class SQuAD2(TaskAdapter):
    name = "know/answerability/squad_v2"
    leaf = "know/answerability"
    source = "SQuAD 2.0, validation (Rajpurkar et al., 2018)"
    url = "https://huggingface.co/datasets/rajpurkar/squad_v2"
    license = "CC BY-SA 4.0"
    tier = "C"
    criteria_source = "Written"

    def load(self):
        return hf_rows("rajpurkar/squad_v2", "squad_v2", "validation")

    def label_of(self, record):
        return str(bool(record["answers"]["text"]))

    def to_item(self, record, rng):
        question = Question(type="boolean", instructions="Can the question be answered from the passage alone?")
        state = {"passage": record["context"], "question": record["question"]}
        gold = Gold(label="true" if record["answers"]["text"] else "false")
        return self.item(state, {"answerable": question}, {"answerable": gold}, original_id=record["id"])


@register_task
class MuSiQue(TaskAdapter):
    name = "know/answerability/musique"
    leaf = "know/answerability"
    source = "MuSiQue-Full, answerable and unanswerable multi-hop questions, validation (Trivedi et al., 2022)"
    url = "https://huggingface.co/datasets/bdsaglam/musique"
    license = "CC BY 4.0"
    tier = "B"
    criteria_source = "Adapted from the MuSiQue definition of answerability"
    notes = "Unanswerable questions are contrastive twins whose supporting paragraph was removed."

    def load(self):
        return hf_rows("bdsaglam/musique", None, "validation")

    def label_of(self, record):
        return str(record["answerable"])

    def to_item(self, record, rng):
        paragraphs = "\n\n".join(f"{p['title']}: {p['paragraph_text']}" for p in record["paragraphs"])
        question = Question(type="boolean", instructions="Can the question be answered using only the given paragraphs?")
        state = {"paragraphs": paragraphs, "question": record["question"]}
        gold = Gold(label="true" if record["answerable"] else "false")
        return self.item(state, {"answerable": question}, {"answerable": gold}, original_id=record["id"])
