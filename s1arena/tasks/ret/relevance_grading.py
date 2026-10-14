"""Leaf: graded relevance judgments (相关性分级). One query-document pair per item, answered on the
source collection's own relevance scale."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from ...env import load_env
from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import fetch
from .sources import consistent_judgments, trec_dl_judgments

LEAF = "ret/relevance_grading"

# TREC 2019/2020 Deep Learning Track, passage judging scale (Craswell et al., 2020; 2021), verbatim.
TREC_DL_LEVELS = [
    "[0] Irrelevant: The passage has nothing to do with the query.",
    "[1] Related: The passage seems related to the query but does not answer it.",
    "[2] Highly relevant: The passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information.",
    "[3] Perfectly relevant: The passage is dedicated to the query and contains the exact answer.",
]


@register_task
class TrecDLRelevance(TaskAdapter):
    name = f"{LEAF}/trec_dl"
    rank_key = "meta:qid"  # NDCG groups
    leaf = LEAF
    source = "TREC 2019 and 2020 Deep Learning Track, passage judgments on MS MARCO (Craswell et al., 2020; 2021)"
    url = "https://microsoft.github.io/msmarco/TREC-Deep-Learning"
    license = "MS MARCO terms: non-commercial research use; NIST relevance judgments"
    tier = "A"
    criteria_source = "Four-point passage judging scale verbatim from the TREC DL 2019/2020 overview papers"
    sampling = "balanced"
    notes = ("NIST assessors' grades for the 97 judged queries; passages from the official MS MARCO collection (1 GB download). "
             "Near-duplicate passages of a query (same opening) that received different grades are dropped. Sampled over "
             "(grade, query) cells, so the four grades are about equally frequent (56-68% of judged passages are irrelevant) "
             "and no query dominates.")

    def load(self):
        return consistent_judgments(trec_dl_judgments())

    def label_of(self, record):
        return f"{record['grade']}|{record['year']}|{record['qid']}"

    def to_item(self, record, rng):
        question = Question(type="score", instructions="How relevant is the passage to the search query? Judge on the four-point scale.", criteria=TREC_DL_LEVELS)
        state = {"query": record["query"], "passage": record["passage"]}
        return self.item(state, {"relevance": question}, {"relevance": Gold(label=str(record["grade"]))}, year=record["year"], qid=record["qid"], pid=record["pid"])


# TREC-COVID relevance scale (Round 5 call for participation, ir.nist.gov/trec-covid/round5.html), verbatim.
TREC_COVID_LEVELS = [
    "Not Relevant: everything else.",
    "Partially Relevant: the article answers part of the question but would need to be combined with other information to get a complete answer.",
    "Relevant: the article is fully responsive to the information need as expressed by the topic, i.e. answers the Question in the topic. "
    "The article need not contain all information on the topic, but must, on its own, provide an answer to the question.",
]


@register_task
class TrecCovidRelevance(TaskAdapter):
    name = f"{LEAF}/trec_covid"
    rank_key = "meta:topic"  # NDCG groups
    leaf = LEAF
    source = "TREC-COVID, final cumulative judgments (Voorhees et al., 2020); article texts from the BEIR corpus (Thakur et al., 2021)"
    url = "https://ir.nist.gov/trec-covid/"
    license = "CORD-19 Dataset License Agreement (text and data mining only, non-transferable); judgments by NIST"
    redistributable = False
    tier = "A"
    criteria_source = "Three-way relevance scale verbatim from the TREC-COVID Round 5 guidelines; topics (query, question, narrative) from topics-rnd5.xml"
    sampling = "balanced"
    notes = ("Biomedical expert judgments of CORD-19 articles from the official qrels-covid_d5_j0.5-5; the article is shown as title "
             "and abstract (BEIR corpus, CORD-19 of 2020-07-16), articles without an abstract are skipped. Sampled evenly "
             "across (grade, topic) cells. Loader only: the CORD-19 license grants a non-transferable text-mining license.")

    def load(self):
        load_env()
        from datasets import load_dataset

        topics = {}
        for topic in ET.parse(fetch("https://ir.nist.gov/trec-covid/data/topics-rnd5.xml", "trec_covid_topics-rnd5.xml")).getroot():
            topics[topic.get("number")] = {k: (topic.findtext(k) or "").strip() for k in ("query", "question", "narrative")}
        with fetch("https://ir.nist.gov/trec-covid/data/qrels-covid_d5_j0.5-5.txt", "trec_covid_qrels-covid_d5_j0.5-5.txt").open() as f:
            judged = [(t, doc, int(s)) for t, _, doc, s in (line.split() for line in f if line.strip()) if int(s) in (0, 1, 2)]
        wanted = {doc for _, doc, _ in judged}
        corpus = load_dataset("BeIR/trec-covid", "corpus", split="corpus")
        docs = {r["_id"]: (r["title"].strip(), r["text"].strip()) for r in corpus if r["_id"] in wanted}
        rows = []
        for qid, doc, grade in judged:
            title, abstract = docs.get(doc, ("", ""))
            if qid in topics and len(abstract) >= 200:
                rows.append({"qid": qid, **topics[qid], "doc": doc, "title": title, "abstract": abstract, "grade": grade})
        return rows

    def label_of(self, record):
        return f"{record['grade']}|{record['qid']}"

    def to_item(self, record, rng):
        question = Question(
            type="score",
            instructions="How relevant is the article to the information need expressed by the topic (its query, question and narrative)?",
            criteria=TREC_COVID_LEVELS,
        )
        state = {
            "topic_query": record["query"],
            "topic_question": record["question"],
            "topic_narrative": record["narrative"],
            "article": f"{record['title']}\n\n{record['abstract']}" if record["title"] else record["abstract"],
        }
        return self.item(state, {"relevance": question}, {"relevance": Gold(label=str(record["grade"]))}, topic=record["qid"], cord_uid=record["doc"])


# ANTIQUE label definitions (Hashemi et al., 2020, Section 3.1), verbatim; the scale is 1-4, shown here as levels 0-3.
ANTIQUE_LEVELS = [
    "Label 1: It is completely out of context or does not make any sense.",
    "Label 2: It does not answer the question or if it does, it provides an unreasonable answer, however, it is not out of context. "
    "Therefore, you cannot accept it as an answer to the question.",
    "Label 3: It can be an answer to the question, however, it is not sufficiently convincing. There should be an answer with much better quality for the question.",
    "Label 4: It looks reasonable and convincing. Its quality is on par with or better than the \"Possibly Correct Answer\". "
    "Note that it does not have to provide the same answer as the \"Possibly Correct Answer\".",
]
ANTIQUE = "https://ciir.cs.umass.edu/downloads/Antique/"


@register_task
class AntiqueRelevance(TaskAdapter):
    name = f"{LEAF}/antique"
    rank_key = "meta:qid"  # NDCG groups
    leaf = LEAF
    source = "ANTIQUE non-factoid answer retrieval, test judgments (Hashemi et al., 2020)"
    url = "https://ciir.cs.umass.edu/downloads/Antique/"
    license = "Research use (ANTIQUE readme); answers from Yahoo! Answers (Webscope L6), whose terms do not allow redistribution"
    redistributable = False
    tier = "A"
    criteria_source = "Label definitions verbatim from Hashemi et al. (2020), Section 3.1; raters also saw the asker's chosen answer as the 'possibly correct answer', which is included"
    sampling = "balanced"
    notes = ("Crowd labels (majority of 3 workers, expert on persistent disagreement). Answers chosen by the asker (label set by rule, "
             "not judged) are shown as reference only, and the queries on the official offensive-content blacklist are dropped. "
             "Sampled over (label, question) cells, so labels are about equally frequent and questions vary. Loader only.")

    def load(self):
        def table(name):
            with fetch(ANTIQUE + name).open(encoding="utf-8") as f:
                return [line.rstrip("\n").split("\t") for line in f if line.strip()]

        queries = {q: text for q, text in table("antique-test-queries.txt")}
        blacklist = {line[0].strip() for line in table("test-queries-blacklist.txt")}
        with fetch(ANTIQUE + "antique-test.qrel").open(encoding="utf-8") as f:
            qrels = [line.split() for line in f if line.strip()]
        wanted = {doc for _, _, doc, _ in qrels}
        answers = {}
        with fetch(ANTIQUE + "antique-collection.txt").open(encoding="utf-8") as f:
            for line in f:
                doc, _, text = line.rstrip("\n").partition("\t")
                if doc in wanted:
                    answers[doc] = text.strip()
        chosen = {q: answers.get(doc) for q, kind, doc, _ in qrels if kind == "U0"}
        rows = []
        for q, kind, doc, label in qrels:
            if kind == "U0" or q in blacklist or q not in queries or not chosen.get(q) or not answers.get(doc):
                continue
            if answers[doc] == chosen[q]:
                continue
            rows.append({"qid": q, "question": queries[q].strip(), "reference": chosen[q], "doc": doc, "answer": answers[doc],
                         "label": int(label), "judged_by": "expert" if kind == "E0" else "crowd"})
        return rows

    def label_of(self, record):
        return f"{record['label']}|{record['qid']}"

    def to_item(self, record, rng):
        question = Question(
            type="score",
            instructions=("Judge the relevance of the given answer to the question. The \"possibly correct answer\" was chosen by the person "
                          "who asked; other answers can differ from it and still be correct."),
            criteria=ANTIQUE_LEVELS,
        )
        state = {"question": record["question"], "possibly_correct_answer": record["reference"], "answer": record["answer"]}
        gold = Gold(label=str(record["label"] - 1))
        return self.item(state, {"relevance": question}, {"relevance": gold}, qid=record["qid"], doc=record["doc"], judged_by=record["judged_by"])
