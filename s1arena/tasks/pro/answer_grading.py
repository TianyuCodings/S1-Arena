"""Leaf: answer grading (作答评分): short answers against a reference answer, essays against a rubric.

Short answers use the SemEval-2013 Student Response Analysis labels and the UNT
data-structures answers graded by two people; essays use the ELLIPSE corpus, whose
raw file keeps both raters' scores, so two-rater items carry soft labels.
"""

from __future__ import annotations

import csv
import io
import re
import xml.etree.ElementTree as ET
import zipfile

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import fetch, hf_rows

LEAF = "pro/answer_grading"


def two_rater_gold(a: int, b: int, levels: int, offset: int = 0) -> Gold:
    """Soft label from two ratings; hard label = their mean rounded half down (the lower level on a one-point split)."""
    i, j = a - offset, b - offset
    if not (0 <= i < levels and 0 <= j < levels):
        raise ValueError(f"rating out of range: {a}, {b}")
    dist = {str(k): [i, j].count(k) / 2 for k in range(levels) if k in (i, j)}
    return Gold(label=str((i + j) // 2), dist=dist)


# ---------- SemEval-2013 Task 7: Student Response Analysis (SciEntsBank and Beetle) ----------

# Dzikovska et al. (2013), Section 2, verbatim.
SRA_INSTRUCTIONS = (
    "Given a question, a known correct 'reference answer' and a 1- or 2-sentence 'student answer', label the "
    "student answer with one of the following judgments."
)
SRA_OPTIONS = {
    "correct": "Correct: the student answer is a complete and correct paraphrase of the reference answer.",
    "partially_correct_incomplete": "Partially correct incomplete: it is a partially correct answer containing some but not all information from the reference answer.",
    "contradictory": "Contradictory: the student answer explicitly contradicts the reference answer.",
    "irrelevant": "Irrelevant: the student answer is talking about domain content but not providing the necessary information.",
    "non_domain": "Non domain: the student utterance does not include domain content, e.g., \"I don't know\", \"what the book says\", \"you are stupid\".",
}


@register_task
class SemEval2013SRA(TaskAdapter):
    name = f"{LEAF}/semeval2013_sra"
    leaf = LEAF
    source = "SemEval-2013 Task 7 Student Response Analysis, SciEntsBank and Beetle, 5-way (Dzikovska et al., 2013)"
    url = "https://github.com/myrosia/semeval-2013-task7"
    license = "CC BY-SA 3.0"
    tier = "A"
    criteria_source = "Task wording and the five judgment definitions verbatim from Dzikovska et al. (2013), Section 2"
    notes = (
        "Test sets for unseen answers and unseen questions of both corpora (SciEntsBank's unseen-domains set is not used, "
        "which keeps the two corpora at similar size). Beetle questions list all reference answers with their original "
        "quality tags (BEST, GOOD, MINIMAL, KEYWORD). Labels are by trained annotators. Repeated answers to the same question "
        "are kept once, and the one answer labelled two different ways is dropped."
    )
    ZIP = "https://raw.githubusercontent.com/myrosia/semeval-2013-task7/5157c69f914e8c581045c03e98f96398d40a91b8/semeval-5way.zip"
    SETS = ("test-unseen-answers", "test-unseen-questions")

    def load(self):
        rows = []
        with zipfile.ZipFile(fetch(self.ZIP, "semeval2013_task7_5way.zip")) as zf:
            for path in sorted(zf.namelist()):
                parts = path.split("/")
                if len(parts) != 4 or parts[1] not in self.SETS or parts[2] != "Core" or not path.endswith(".xml"):
                    continue
                root = ET.fromstring(zf.read(path))
                refs = [(r.get("category"), (r.text or "").strip()) for r in root.find("referenceAnswers")]
                for answer in root.find("studentAnswers"):
                    rows.append({
                        "corpus": parts[0], "test_set": parts[1], "question_id": root.get("id"),
                        "question": (root.findtext("questionText") or "").strip(), "references": refs,
                        "answer": (answer.text or "").strip(), "label": answer.get("accuracy"), "answer_id": answer.get("id"),
                    })
        labels: dict[tuple, set] = {}
        for r in rows:
            labels.setdefault((r["question"], " ".join(r["answer"].lower().split())), set()).add(r["label"])
        out, seen = [], set()
        for r in rows:  # repeated answers to a question are kept once; the one answer labelled two ways is dropped
            key = (r["question"], " ".join(r["answer"].lower().split()))
            if r["label"] in SRA_OPTIONS and r["answer"] and len(labels[key]) == 1 and key not in seen:
                seen.add(key)
                out.append(r)
        return out

    def label_of(self, record):
        return f"{record['corpus']}|{record['label']}"

    def to_item(self, record, rng):
        refs = record["references"]
        if len(refs) == 1:
            reference = refs[0][1]
        else:
            reference = "\n".join(f"- {text}" + (f" [{tag}]" if tag else "") for tag, text in refs)
        state = {"question": record["question"], "reference_answer": reference, "student_answer": record["answer"]}
        question = Question(type="choice", instructions=SRA_INSTRUCTIONS, criteria=shuffled_options(SRA_OPTIONS, rng))
        meta = {"corpus": record["corpus"], "test_set": record["test_set"], "original_id": record["answer_id"]}
        return self.item(state, {"judgment": question}, {"judgment": Gold(label=record["label"])}, **meta)


# ---------- Mohler et al. (2011): data-structures short answers, two graders ----------

@register_task
class MohlerASAG(TaskAdapter):
    name = f"{LEAF}/mohler_asag"
    leaf = LEAF
    source = "UNT Computer Science Short Answer Dataset v2.0 (Mohler, Bunescu and Mihalcea, 2011)"
    url = "https://huggingface.co/datasets/nkazi/MohlerASAG"
    license = "Not stated upstream (UNT release); CC BY 4.0 on the Hugging Face copy"
    tier = "B"
    criteria_source = "Adapted from Mohler et al. (2011): answers graded for correctness on an integer scale from 0 (completely incorrect) to 5 (perfect answer)"
    notes = (
        "Assignments 1-10 only (the two exams were graded on 0-10). Soft label = the two graders' scores (a teaching "
        "assistant and an author); hard label = their mean rounded half down. The whole release is evaluation data (no official "
        "split); repeated identical answers to the same question are kept once."
    )
    LEVELS = ["0: completely incorrect", "1", "2", "3", "4", "5: perfect answer"]

    def load(self):
        rows, seen = [], set()
        for r in hf_rows("nkazi/MohlerASAG", "raw", "open_ended", revision="dac06732b42d01a5169f543f6d272f1476cced3e"):
            key = (r["question"], " ".join(self.clean(r["student_answer"]).lower().split()))
            if r["id"].split(".")[0] not in ("E11", "E12") and key not in seen:  # identical answers to a question are kept once
                seen.add(key)
                rows.append(r)
        return rows

    def label_of(self, record):
        return str(int(record["score_grader_1"] + record["score_grader_2"]) // 2)

    @staticmethod
    def clean(text: str) -> str:
        text = re.sub(r"<br\s*/?>", "\n", text or "")
        return re.sub(r"\n{2,}", "\n", text).strip()

    def to_item(self, record, rng):
        a, b = int(record["score_grader_1"]), int(record["score_grader_2"])
        question = Question(
            type="score",
            instructions="Grade the student's answer to this data-structures question for correctness, compared with the "
            "instructor's answer, on an integer scale from 0 (completely incorrect) to 5 (perfect answer).",
            criteria=self.LEVELS,
        )
        state = {"question": self.clean(record["question"]), "instructor_answer": self.clean(record["instructor_answer"]), "student_answer": self.clean(record["student_answer"])}
        if not state["student_answer"]:
            return None
        return self.item(state, {"grade": question}, {"grade": two_rater_gold(a, b, 6)}, original_id=record["id"])


# ---------- ELLIPSE: English-learner essays, holistic and analytic rubric scores from two raters ----------

# English Proficiency Scoring Rubric for English Language Learners (ELL_Rubrics.docx, ELLIPSE corpus), verbatim.
ELLIPSE_RUBRIC = {
    "Overall": [
        "A limited range of familiar words or phrases loosely strung together; frequent errors in grammar (including syntax) and usage. Communication impeded in most cases by language inaccuracies.",
        "Inconsistent facility in sentence formation, word choice, and mechanics; organization partially developed but may be missing or unsuccessful. Communication impeded in many instances by language inaccuracies.",
        "Facility limited to the use of common structures and generic vocabulary; organization generally controlled although connection sometimes absent or unsuccessful; errors in grammar and syntax and usage. Communication is impeded by language inaccuracies in some cases.",
        "Facility in the use of language with syntactic variety and range of words and phrases; controlled organization; accuracy in grammar and conventions; occasional language inaccuracies that rarely impede communication.",
        "Native-like facility in the use of language with syntactic variety, Appropriate word choice and phrases; well-controlled text organization; precise use of grammar and conventions; rare language inaccuracies that do not impede communication.",
    ],
    "Cohesion": [
        "No clear control of organization; cohesive devices not present or unsuccessfully used; presentation of ideas unclear.",
        "Organization only partially developed with a lack of logical sequencing of ideas; some basic cohesive devices used but with inaccuracy or repetition.",
        "Organization generally controlled; cohesive devices used but limited in type; Some repetitive, mechanical, or faulty use of cohesion use within and/or between sentences and paragraphs.",
        "Organization generally well controlled; a range of cohesive devices used appropriately such as reference and transitional words and phrases to connect ideas; generally appropriate overlap of ideas",
        "Text organization consistently well controlled using a variety of effective linguistic features such as reference and transitional words and phrases to connect ideas across sentences and paragraphs; appropriate overlap of ideas.",
    ],
    "Syntax": [
        "Pervasive and basic errors in sentence structure and word order that cause confusion; basic sentences errors common.",
        "Some sentence variation used; many sentence structure problems.",
        "Simple, compound, and complex syntactic structures present although the range may be limited; some apparent errors in sentence formation, especially in more complex sentences.",
        "Appropriate use of a variety of syntactic structures, such as simple, compound, and complex sentences; occasional errors or inappropriateness in sentence formation.",
        "Flexible and effective use of a full range of syntactic structures including simple, compound, and complex sentences; There may be rare minor and negligible errors in sentence formation.",
    ],
    "Vocabulary": [
        "Limited vocabulary often inappropriately used; limited control of word choice and word forms; little attempt to use topic-related terms",
        "Narrow range of vocabulary to convey basic and elementary meaning; topic related terms used inappropriately; errors in word formation and word choice that may distort meanings",
        "Minimally adequate range of vocabulary for the topic; no precise use of subtle word meanings; topic related terms only used occasionally; attempts to use less common vocabulary but with some inaccuracy",
        "Sufficient range of vocabulary to allow flexibility and precision; appropriate use of topic-related terms and less common lexical items",
        "Wide range of vocabulary flexibly and effectively used to convey precise meanings; skillful use of topic-related terms and less common words; rare negligible inaccuracies in word use.",
    ],
    "Phraseology": [
        "Memorized chunks of language, or simple phrasal patterns predominate; many repetitions and misuses of phrases.",
        "Narrow range of phrases, such as collocations and lexical bundles, used to convey basic and elementary meaning; many repetitions and /or misuses of phrases.",
        "Evident use of phrases such as idioms, collocations, and lexical bundles but without much variety; some noticeable repetitions and misuses.",
        "Appropriate use of a variety of phrases, such as idioms, collocations, and lexical bundles; occasional inaccuracies and colloquialisms.",
        "Flexible and effective use of a variety of phrases, such as idioms, collocations, and lexical bundles, to convey precise and subtle meanings; rare minor inaccuracies that are negligible.",
    ],
    "Grammar": [
        "Errors in grammar and usage throughout.",
        "Many errors in grammar and usage.",
        "Some errors in grammar and usage.",
        "Minimal errors in grammar and usage.",
        "Command of grammar and usage with few or no errors.",
    ],
    "Conventions": [
        "Minimal use of conventions; spelling, capitalization, and punctuation errors throughout.",
        "Variable use of conventions; spelling, capitalization, and punctuation errors frequent and distracting.",
        "Developing use of conventions to convey meaning; errors in spelling, capitalization, and punctuation that are sometimes distracting.",
        "Generally consistent use of appropriate conventions to convey meaning; spelling, capitalization, and punctuation errors few and not distracting.",
        "Consistent use of appropriate conventions to convey meaning; spelling, capitalization, and punctuation errors nonexistent or negligible.",
    ],
}
# "Key Terms and Definitions" of the same rubric, verbatim, attached to the dimensions that use them.
ELLIPSE_TERMS = {
    "Cohesion": "Cohesive device: Cohesive devices are used as links between two or more items (e.g., words, phrases, clauses) in a text to enhance text cohesion. These include the use of conjunctions (and, but, if, on the other hand), transitions (first, next, finally, for example), repetition of words, phrases, and ideas across sentences and paragraphs, and the use of anaphor (pronouns replacing nouns).",
    "Syntax": "Syntax: Structuring sentences according to syntactic rules related to coordinating clauses, developing syntactic phrases (noun, verb, preposition phrases), phrasal and clausal dependency, and transformations such as passives, relative clauses, and negations. Simple: Independent clause. Complex: Independent and dependent clauses. Compound: Two of more independent clauses.",
    "Phraseology": "Phrase: Multiple word units. Chunks: Multiple words that combine to have a single meaning. Often memorized without knowing what the individual words mean (e.g., \"How are you\" for \"Hello\"). Lexical bundles: Multiple word units that are common in English but are not idiomatic (\"There is\"). More common than collocations. Collocations: Two or more words that are often used together (e.g., save time, go to bed, fast food). Idioms: multi-word unit where meaning not deducible from those of the individual words (kick the bucket, rain cats and dogs).",
    "Grammar": "Grammar: The rules by which words change their forms, including the use of word classes and grammatical morphology in English. Word classes include prepositions, pronouns, nouns, verbs, etc... Grammatical morphology includes third person, plural, possessive, etc...",
}


@register_task
class ELLIPSE(TaskAdapter):
    name = f"{LEAF}/ellipse"
    leaf = LEAF
    source = "ELLIPSE corpus of English-learner essays, test set (Crossley et al., 2023)"
    url = "https://github.com/scrosseye/ELLIPSE-Corpus"
    license = "CC BY-NC-SA 4.0"
    redistributable = False  # the test essays ship in a password-protected zip to keep them off the open web
    tier = "A"
    criteria_source = "Level descriptors and key-term definitions verbatim from the ELLIPSE 'English Proficiency Scoring Rubric for English Language Learners' (ELL_Rubrics.docx)"
    notes = (
        "Seven score questions per essay (overall proficiency and six analytic scales, 1-5). Soft labels are the two raters' "
        "scores from the corpus's raw rating file; hard label = their mean rounded half down. Twelve test essays whose final "
        "overall score is not the mean of the two raw ratings are left out. Essays by students in grades 8-12."
    )
    BASE = "https://raw.githubusercontent.com/scrosseye/ELLIPSE-Corpus/dc3b8f0b3b4332fc9f64302c4ccfc4ed582f4b43/"

    @staticmethod
    def _csv(path, member: str, password: bytes) -> list[dict]:
        with zipfile.ZipFile(path) as zf:
            return list(csv.DictReader(io.StringIO(zf.read(member, pwd=password).decode("utf-8"))))

    def load(self):
        test = self._csv(fetch(self.BASE + "ELLIPSE_Final_github_test.zip", "ellipse_test.zip"), "ELLIPSE_Final_github_test.csv", b"ellipse_test")
        raw = self._csv(
            fetch(self.BASE + "ellipsis_raw_rater_scores_anon_all_essay.zip", "ellipse_raw_rater_scores.zip"),
            "ellipsis_raw_rater_scores_anon_all_essay.csv", b"ellipse_raw_data",
        )
        ratings = {}
        for r in raw:
            ratings.setdefault(r["text_id_kaggle"], r)  # one id occurs twice; the first row is kept
        rows = []
        for t in test:
            r = ratings.get(t["text_id_kaggle"])
            if r is None:
                continue
            scores = {d: (int(float(r[f"{d}_1"])), int(float(r[f"{d}_2"]))) for d in ELLIPSE_RUBRIC}
            if abs(sum(scores["Overall"]) / 2 - float(t["Overall"])) > 1e-9:
                continue
            rows.append({"id": t["text_id_kaggle"], "essay": t["full_text"].strip(), "prompt": t["prompt"], "scores": scores})
        return rows

    def label_of(self, record):
        return str(sum(record["scores"]["Overall"]) // 2)

    def to_item(self, record, rng):
        questions, gold = {}, {}
        for dim, levels in ELLIPSE_RUBRIC.items():
            if dim == "Overall":
                text = "Using the holistic column of the English proficiency rubric for English language learners, rate the overall language proficiency shown in this essay from 1 to 5."
            else:
                text = f"Using the analytic rubric for English language learners, rate the {dim.lower()} of this essay from 1 to 5."
                if dim in ELLIPSE_TERMS:
                    text += " Key terms: " + ELLIPSE_TERMS[dim]
            questions[dim.lower()] = Question(type="score", instructions=text, criteria=[f"{i + 1}: {d}" for i, d in enumerate(levels)])
            gold[dim.lower()] = two_rater_gold(*record["scores"][dim], levels=5, offset=1)
        state = {"prompt_topic": record["prompt"], "essay": record["essay"]}
        return self.item(state, questions, gold, original_id=record["id"])
