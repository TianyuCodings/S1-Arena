"""Leaf: legal decisions (法律).

LegalBench classification tasks (each item keeps its own task's instructions and label
definitions), CUAD clause-type identification and ContractNLI document-level NDA review.
"""

from __future__ import annotations

import html
import json
import random
import re
import zipfile

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import fetch, hf_rows, slug

LEAF = "pro/legal"


def _clean(text: str) -> str:
    """One line: collapse all whitespace (clause spans extracted from PDFs)."""
    return re.sub(r"\s+", " ", str(text).replace("\xa0", " ")).strip()


def _text(text: str, task: str) -> str:
    """Keep line breaks; undo the HTML escaping of Reddit posts and SARA's <br> line breaks."""
    text = str(text)
    if task.startswith("learned_hands"):
        text = html.unescape(html.unescape(text))
    text = text.replace("<br>", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r" ?\n ?", "\n", text).strip()


# ---------- LegalBench: a curated pool of classification tasks ----------

# CLAUDETTE category definitions, verbatim from Lippi et al. (2019), Section 3.
UNFAIR_TOS = {
    "arbitration": "Arbitration: The arbitration clause requires or allows the parties to resolve their disputes through an arbitration process, before the case could go to court.",
    "unilateral_change": "Unilateral change: The unilateral change clause specifies the conditions under which the service provider could amend and modify the terms of service and/or the service itself.",
    "content_removal": "Content removal: The content removal gives the provider a right to modify/delete user's content, including in-app purchases, and sometimes specifies the conditions under which the service provider may do so.",
    "jurisdiction": "Jurisdiction: The jurisdiction clause stipulates what courts will have the competence to adjudicate disputes under the contract.",
    "choice_of_law": "Choice of law: The choice of law clause specifies what law will govern the contract, meaning also what law will be applied in potential adjudication of a dispute arising under the contract.",
    "limitation_of_liability": "Limitation of liability: The limitation of liability clause stipulates that the duty to pay damages is limited or excluded, for certain kind of losses, under certain conditions.",
    "unilateral_termination": "Unilateral termination: The unilateral termination clause gives provider the right to suspend and/or terminate the service and/or the contract, and sometimes details the circumstances under which the provider claims to have a right to do so.",
    "contract_by_using": "Contract by using: The contract by using clause stipulates that the consumer is bound by the terms of use of a specific service, simply by using the service, without even being required to mark that he or she has read and accepted them.",
    "other": "Other: the clause belongs to none of the categories above.",
}

MAUD = "Read the segment of a merger agreement and answer the multiple-choice question by choosing the option that best characterizes the agreement. Question: "

# task -> (question name, question type, instructions, options or None, {state key: column}, {answer: option id} or None)
LEGALBENCH = {
    "function_of_decision_section": ("function", "choice", "Classify the following paragraph from a decision of a U.S. Federal Court of Appeals using the following definitions.", {
        "facts": "Facts: The paragraph describes the factual background that led up to the present lawsuit.",
        "procedural_history": "Procedural History: The paragraph describes the course of litigation that led to the current proceeding before the court.",
        "issue": "Issue: The paragraph describes the legal or factual issue that must be resolved by the court.",
        "rule": "Rule: The paragraph describes a rule of law relevant to resolving the issue.",
        "analysis": "Analysis: The paragraph analyzes the legal issue by applying the relevant legal principles to the facts of the present dispute.",
        "conclusion": "Conclusion: The paragraph presents a conclusion of the court.",
        "decree": "Decree: The paragraph constitutes a decree resolving the dispute.",
    }, {"paragraph": "Paragraph"}, None),
    "oral_argument_question_purpose": ("purpose", "choice", "Classify the purpose of this question asked by a Supreme Court justice at oral argument.", {
        "background": "Background: questions seeking factual or procedural information that is missing or not clear in the briefing",
        "clarification": "Clarification: questions seeking to get an advocate to clarify her position or the scope of the rule being advocated for",
        "implications": "Implications: questions about the limits of a rule or its implications for future cases",
        "support": "Support: questions offering support for the advocate's position",
        "criticism": "Criticism: questions criticizing an advocate's position",
        "communicate": "Communicate: question designed primarily to communicate with other justices",
        "humor": "Humor: questions designed to interject humor into the argument and relieve tension",
    }, {"question": "question"}, None),
    "overruling": ("overrules", "boolean", "Does the sentence from a judicial opinion contain language overruling a previous case?", None, {"sentence": "text"}, {"Yes": "true", "No": "false"}),
    "definition_classification": ("defines_term", "boolean", "Does this sentence from a Supreme Court opinion define a term?", None, {"sentence": "text"}, {"Yes": "true", "No": "false"}),
    "canada_tax_court_outcomes": ("outcome", "choice", "Indicate whether the following judgment excerpt from a Tax Court of Canada decision allows the appeal or dismisses the appeal. Where the result is mixed, indicate that the appeal was allowed. Ignore costs orders. Where the outcome is unclear indicate other.", {
        "allowed": "allowed: the appeal is allowed (also when the result is mixed)",
        "dismissed": "dismissed: the appeal is dismissed",
        "other": "other: the excerpt does not clearly state the outcome",
    }, {"judgment_excerpt": "text"}, None),
    "unfair_tos": ("clause_type", "choice", "Classify this clause from an online terms-of-service agreement by type.", UNFAIR_TOS, {"clause": "text"}, None),
    "insurance_policy_interpretation": ("covered", "choice", "Read the insurance policy and the insurance claim. Answer whether the claim is covered under the policy.", {
        "yes": "Yes: the claim is covered.", "no": "No: the claim is not covered.", "ambiguous": "It's ambiguous.",
    }, {"policy": "policy", "claim": "claim"}, {"A": "yes", "B": "no", "C": "ambiguous"}),
    "proa": ("private_right_of_action", "boolean", "A private right of action is when a regular person, a private citizen, is legally entitled to enforce their rights under a given statute. Does the clause specify a private right of action?", None, {"clause": "text"}, {"Yes": "true", "No": "false"}),
    "maud_type_of_consideration": ("answer", "choice", MAUD + "What type of consideration is specified in this agreement?", {
        "all_cash": "All Cash", "all_stock": "All Stock", "mixed_cash_stock": "Mixed Cash/Stock", "mixed_cash_stock_election": "Mixed Cash/Stock: Election",
    }, {"merger_agreement_excerpt": "text"}, {"A": "all_cash", "B": "all_stock", "C": "mixed_cash_stock", "D": "mixed_cash_stock_election"}),
    "maud_ordinary_course_efforts_standard": ("answer", "choice", MAUD + "What is the efforts standard?", {
        "commercially_reasonable": "Commercially reasonable efforts", "flat_covenant": "Flat covenant (no efforts standard)", "reasonable_best": "Reasonable best efforts",
    }, {"merger_agreement_excerpt": "text"}, {"A": "commercially_reasonable", "B": "flat_covenant", "C": "reasonable_best"}),
    "maud_fiduciary_exception__board_determination_standard": ("answer", "choice", MAUD + "Under what circumstances could the Board take actions on a different acquisition proposal notwithstanding the no-shop provision?", {
        "breach": 'If failure to take actions would lead to "breach" of fiduciary duties',
        "inconsistent": 'If failure to take actions would be "inconsistent" with fiduciary duties',
        "likely_breach": 'If failure to take actions would lead to "reasonably likely/expected breach" of fiduciary duties',
        "likely_inconsistent": 'If failure to take actions would lead to "reasonably likely/expected to be inconsistent" with fiduciary duties',
        "likely_violation": 'If failure to take actions would lead to "reasonably likely/expected violation" of fiduciary duties',
        "required_to_comply": 'If taking such actions is "required to comply" with fiduciary duties',
        "violation": 'If failure to take actions would lead to "violation" of fiduciary duties',
        "never": "Under no circumstances could the Board do so.",
        "other": "Other circumstances",
    }, {"merger_agreement_excerpt": "text"}, dict(zip("ABCDEFGHI", ("breach", "inconsistent", "likely_breach", "likely_inconsistent", "likely_violation", "required_to_comply", "violation", "never", "other")))),
    "hearsay": ("hearsay", "boolean", "Hearsay is an out-of-court statement introduced to prove the truth of the matter asserted. Is there hearsay?", None, {"fact_pattern": "text"}, {"Yes": "true", "No": "false"}),
    "ucc_v_common_law": ("governed_by", "choice", "The UCC (through Article 2) governs the sale of goods, which are defined as moveable tangible things (cars, apples, books, etc.), whereas the common law governs contracts for real estate and services. Determine if the contract is governed by the UCC or by common law.", {
        "ucc": "UCC", "common_law": "Common Law",
    }, {"contract": "contract"}, {"UCC": "ucc", "Common Law": "common_law"}),
    "abercrombie": ("mark_type", "choice", "What is the type of mark?", {
        "generic": "generic: A mark is generic if it is the common name for the product.",
        "descriptive": "descriptive: A mark is descriptive if it describes a purpose, nature, or attribute of the product.",
        "suggestive": "suggestive: A mark is suggestive if it suggests or implies a quality or characteristic of the product.",
        "arbitrary": "arbitrary: A mark is arbitrary if it is a real English word that has no relation to the product.",
        "fanciful": "fanciful: A mark is fanciful if it is an invented word.",
    }, {"mark": "text"}, None),
    "learned_hands_housing": ("issue", "boolean", "Does the post discuss issues with paying your rent or mortgage, landlord-tenant issues, housing subsidies and public housing, eviction, and other problems with your apartment, mobile home, or house?", None, {"post": "text"}, {"Yes": "true", "No": "false"}),
    "learned_hands_employment": ("issue", "boolean", "Does the post discuss issues related to working at a job, including discrimination and harassment, worker's compensation, workers rights, unions, getting paid, pensions, being fired, and more?", None, {"post": "text"}, {"Yes": "true", "No": "false"}),
    "sara_entailment": ("entailed", "boolean", "Is the statement entailed under the statute, given the facts? (The alternative is that the statute and facts contradict it.)", None, {"statute": "statute", "facts": "description", "statement": "question"}, {"Entailment": "true", "Contradiction": "false"}),
}


@register_task
class LegalBench(TaskAdapter):
    name = f"{LEAF}/legalbench"
    leaf = LEAF
    source = "LegalBench, 17 classification tasks, test splits (Guha et al., 2023)"
    url = "https://huggingface.co/datasets/nguha/legalbench"
    license = "Per task: CC BY 4.0 (most), CC BY-SA 4.0 (definition_classification), CC BY-NC 4.0 (canada_tax_court_outcomes), CC BY-NC-SA 4.0 (learned_hands_*), MIT (sara_entailment)"
    tier = "B"
    criteria_source = ("Each task's own instructions and label definitions from its LegalBench README and base_prompt.txt (few-shot examples "
                       "dropped); unfair_tos category definitions verbatim from Lippi et al. (2019)")
    sampling = "balanced"
    notes = ("Balanced across the 17 tasks (rhetorical analysis, interpretation incl. 3 MAUD deal-point questions, rule application, "
             "issue spotting) so large tasks do not dominate; labels follow each task's natural distribution. In unfair_tos the "
             "'Other' class (91% of clauses) is subsampled to 100. Label-revealing columns (hearsay slice, SARA case id) are withheld.")

    def load(self):
        rows = []
        for task in LEGALBENCH:
            records = [dict(r, task=task) for r in hf_rows("nguha/legalbench", task, "test")]
            if task == "unfair_tos":
                other = [r for r in records if r["answer"] == "Other"]
                keep = random.Random(f"{self.name}:unfair_tos").sample(other, 100)
                records = [r for r in records if r["answer"] != "Other"] + keep
            rows += records
        return rows

    def label_of(self, record):
        return record["task"]

    def to_item(self, record, rng):
        qname, qtype, instructions, options, fields, mapping = LEGALBENCH[record["task"]]
        answer = str(record["answer"]).strip()
        label = mapping[answer] if mapping else slug(answer)
        if qtype == "choice":
            if label not in options:
                return None
            question = Question(type="choice", instructions=instructions, criteria=shuffled_options(options, rng))
        else:
            question = Question(type="boolean", instructions=instructions)
        state = {key: _text(record[column], record["task"]) for key, column in fields.items()}
        return self.item(state, {qname: question}, {qname: Gold(label=label)}, legalbench_task=record["task"], original_id=str(record["index"]))


# ---------- CUAD: which kind of clause is this? ----------

# Categories whose answers are names or bare dates, which cannot be told apart without the whole contract.
CUAD_SKIP = {"Document Name", "Parties", "Agreement Date", "Effective Date"}


@register_task
class CUAD(TaskAdapter):
    name = f"{LEAF}/cuad"
    leaf = LEAF
    source = "CUAD v1, clause-type identification on the 102 test contracts (Hendrycks et al., 2021)"
    url = "https://github.com/TheAtticusProject/cuad"
    license = "CC BY 4.0"
    tier = "B"
    criteria_source = "Category names and descriptions verbatim from the CUAD question 'Details' (the annotators' category definitions); question written"
    notes = ("Each item is one clause span that expert annotators labelled with exactly one category; spans overlapping a span of another "
             "category are dropped, as are the name and date categories (Document Name, Parties, Agreement Date, Effective Date). "
             "37 options. LegalBench's binary cuad_* tasks are not used, to avoid double counting.")

    def load(self):
        with zipfile.ZipFile(fetch("https://github.com/TheAtticusProject/cuad/raw/main/data.zip", "cuad_data.zip")) as zf:
            docs = json.load(zf.open("test.json"))["data"]  # the paper's test split: 102 contracts
        self.options, rows = {}, []
        for doc in docs:
            para = doc["paragraphs"][0]
            spans = []
            for qa in para["qas"]:
                category = re.search(r'related to "(.+?)"', qa["question"]).group(1)
                details = _clean(qa["question"].split("Details:", 1)[1])
                if category not in CUAD_SKIP:
                    self.options[slug(category)] = f"{category}: {details}"
                for answer in qa["answers"]:
                    spans.append((answer["answer_start"], answer["answer_start"] + len(answer["text"]), category, answer["text"]))
            seen = set()
            for start, end, category, text in spans:
                overlaps = any(c != category and s < end and start < e for s, e, c, _ in spans)
                key = (category, _clean(text))
                if category in CUAD_SKIP or overlaps or key in seen or len(key[1]) < 20:
                    continue
                seen.add(key)
                rows.append({"contract": doc["title"], "category": category, "clause": key[1]})
        return rows

    def label_of(self, record):
        return record["category"]

    def to_item(self, record, rng):
        question = Question(
            type="choice",
            instructions="This clause comes from a commercial contract. Which category of clause that lawyers look for when reviewing contracts does it belong to?",
            criteria=shuffled_options(self.options, rng),
        )
        return self.item({"clause": record["clause"]}, {"category": question}, {"category": Gold(label=slug(record["category"]))}, contract=record["contract"])


# ---------- ContractNLI: does the NDA entail, contradict or not mention the hypothesis? ----------

@register_task
class ContractNLI(TaskAdapter):
    name = f"{LEAF}/contractnli"
    leaf = LEAF
    source = "ContractNLI, document-level NLI on non-disclosure agreements, test split (Koreeda and Manning, 2021)"
    url = "https://stanfordnlp.github.io/contract-nli/"
    license = "CC BY 4.0"
    tier = "B"
    criteria_source = "Hypotheses verbatim from the dataset; label wording adapted from the ContractNLI task definition (README)"
    notes = "123 test NDAs x 17 fixed hypotheses; the whole contract is the input (median about 1,800 tokens). Evidence spans are not shown."
    OPTIONS = {
        "entailment": "Entailment: the contract entails the hypothesis.",
        "contradiction": "Contradiction: the contract contradicts the hypothesis.",
        "not_mentioned": "NotMentioned: the contract neither entails nor contradicts the hypothesis (it is not mentioned).",
    }
    MAP = {"Entailment": "entailment", "Contradiction": "contradiction", "NotMentioned": "not_mentioned"}

    def load(self):
        path = fetch("https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip", "contract-nli.zip")
        with zipfile.ZipFile(path) as zf:
            data = json.load(zf.open("contract-nli/test.json"))
        rows = []
        for doc in data["documents"]:
            for hyp, annotation in doc["annotation_sets"][0]["annotations"].items():
                rows.append({"doc_id": doc["id"], "text": doc["text"], "hypothesis_id": hyp, "hypothesis": data["labels"][hyp]["hypothesis"], "choice": annotation["choice"]})
        return rows

    def label_of(self, record):
        return f"{record['choice']}|{record['hypothesis_id']}"

    def to_item(self, record, rng):
        question = Question(
            type="choice",
            instructions="Given the non-disclosure agreement, is the hypothesis entailed by, contradicting to, or not mentioned by (neutral to) the contract?",
            criteria=shuffled_options(self.OPTIONS, rng),
        )
        state = {"contract": record["text"].strip(), "hypothesis": record["hypothesis"]}
        gold = Gold(label=self.MAP[record["choice"]])
        return self.item(state, {"relation": question}, {"relation": gold}, hypothesis_id=record["hypothesis_id"], doc_id=record["doc_id"])
