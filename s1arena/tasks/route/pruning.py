"""Leaf: context compression, keep or drop each passage (上下文压缩：保留还是丢弃).

One boolean per paragraph, asked together. Two definitions of "needed":

- generated fact chains: a paragraph is needed when, with it removed and every other paragraph
  kept, the question can no longer be answered (leave-one-out ablation, computed by a solver);
- QASPER: a paragraph is needed when annotators chose it as part of the minimal evidence for
  the answer (soft label = share of annotators who chose it).
"""

from __future__ import annotations

import random
from collections import defaultdict

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import hf_rows

LEAF = "route/context_pruning"

# ---------- generated fact chains with leave-one-out ablation ----------

FIRST = ["Alma", "Bruno", "Celia", "Dario", "Edith", "Felix", "Greta", "Hugo", "Irene", "Jonas", "Klara", "Leon", "Mira",
         "Nils", "Olga", "Pavel", "Quinn", "Rosa", "Stefan", "Tilda", "Ugo", "Vera", "Walter", "Xenia", "Yusuf", "Zora"]
LAST = ["Abend", "Brask", "Corwin", "Delacroix", "Eskola", "Fenwick", "Garrow", "Holm", "Ivers", "Jansen", "Kovac",
        "Lindqvist", "Marlow", "Nyberg", "Oduya", "Petrakis", "Quade", "Rasmus", "Solberg", "Tamura", "Ulloa", "Varga"]
FIRM_A = ["Brightwell", "Kestrel", "Norvane", "Halcyon", "Ostrel", "Pinecrest", "Quillon", "Redmere", "Silverline",
          "Tamsin", "Umbrel", "Vantor", "Wexmoor", "Yarrowby", "Zephyrine", "Ashgrove", "Bramblet", "Cobalto"]
FIRM_B = ["Analytics", "Foods", "Robotics", "Textiles", "Logistics", "Pharma", "Media", "Energy", "Software", "Shipping"]
CITIES = ["Varrowby", "Lindmere", "Ostvale", "Calvenor", "Dunstrand", "Esterholm", "Falkenmoor", "Gavrish", "Hollowen",
          "Irkavan", "Jastrel", "Korvath", "Lusmere", "Morvaine", "Nerrick", "Oskaland", "Pellmar", "Quorvale", "Rusken",
          "Sallowmere", "Tarnholt", "Ulvenstad", "Vesperan", "Wendrift"]
COUNTRIES = ["Veldania", "Ostrenia", "Karvassa", "Lumeria", "Norvikstan", "Pallador", "Rethmark", "Sulvania", "Tovaria", "Zembria"]

TEMPLATES = {  # relation -> sentence templates over (a, b)
    "works_at": ["{a} works at {b}.", "{a} is currently employed by {b}.", "These days {a} works for {b}."],
    "worked_before": ["{a} worked at {b} until {year}.", "Earlier in their career, {a} spent several years at {b}."],
    "hq_in": ["{a} is headquartered in {b}.", "The head office of {a} is in {b}."],
    "office_in": ["{a} also runs a branch office in {b}.", "{a} opened a sales office in {b} last year."],
    "city_in": ["{a} is a city in {b}.", "The city of {a} lies in {b}."],
    "born_in": ["{a} was born in {b}.", "{a} was born and raised in {b}."],
    "lives_in": ["{a} now lives in {b}.", "{a} moved to {b} a few years ago."],
    "married_to": ["{a} is married to {b}.", "{a} and {b} are married."],
    "founded_by": ["{a} was founded by {b}.", "{b} founded {a}."],
}
QUESTIONS = {  # question kind -> (question template, chain of relations followed from the subject)
    "employer_hq_country": ("In which country is the head office of the company where {s} currently works?", ["works_at", "hq_in", "city_in"]),
    "founder_birth_country": ("In which country was the founder of {s} born?", ["founded_by", "born_in", "city_in"]),
    "spouse_birth_city": ("In which city was the spouse of {s} born?", ["married_to", "born_in"]),
    "spouse_employer_hq": ("In which city is the head office of the company where the spouse of {s} currently works?", ["married_to", "works_at", "hq_in"]),
    "birth_country": ("In which country was {s} born?", ["born_in", "city_in"]),
}


def solve(facts: set[tuple[str, str, str]], kind: str, subject: str) -> str | None:
    """Follow the question's chain of relations; None if a link is missing. married_to is symmetric."""
    current = subject
    for relation in QUESTIONS[kind][1]:
        nxt = [b for r, a, b in facts if r == relation and a == current]
        if relation == "married_to":
            nxt += [a for r, a, b in facts if r == relation and b == current]
        if len(set(nxt)) != 1:
            return None
        current = nxt[0]
    return current


class World:
    """A small consistent world: every person, company and city has at most one value per relation."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.people = [f"{f} {l}" for f, l in zip(rng.sample(FIRST, 10), rng.sample(LAST, 10))]
        self.firms = [f"{a} {b}" for a, b in zip(rng.sample(FIRM_A, 6), rng.sample(FIRM_B * 2, 6))]
        self.cities = rng.sample(CITIES, 8)
        self.countries = rng.sample(COUNTRIES, 4)
        self.country = {c: rng.choice(self.countries) for c in self.cities}
        self.hq = {f: rng.choice(self.cities) for f in self.firms}
        self.born = {p: rng.choice(self.cities) for p in self.people}
        self.employer = {p: rng.choice(self.firms) for p in self.people}
        self.founder = {f: p for f, p in zip(self.firms, rng.sample(self.people, len(self.firms)))}
        a, b, c, d = rng.sample(self.people, 4)
        self.spouse = {a: b, c: d}

    def fact(self, relation: str, a: str) -> tuple[str, str, str]:
        value = {"works_at": self.employer, "hq_in": self.hq, "city_in": self.country, "born_in": self.born,
                 "married_to": self.spouse, "founded_by": self.founder}[relation][a]
        return (relation, a, value)


def _render(fact: tuple[str, str, str], rng: random.Random) -> str:
    relation, a, b = fact
    return rng.choice(TEMPLATES[relation]).format(a=a, b=b, year=rng.randint(2008, 2021))


def generate(rng: random.Random) -> dict | None:
    world = World(rng)
    kind = rng.choice(sorted(QUESTIONS))
    chain = QUESTIONS[kind][1]
    subjects = {"works_at": world.people, "founded_by": world.firms, "married_to": list(world.spouse), "born_in": world.people}[chain[0]]
    subject = rng.choice(subjects)
    needed, current = [], subject
    for relation in chain:
        f = world.fact(relation, current)
        needed.append(f)
        current = f[2]
    # distractors: true facts about the same entities that the question does not ask about
    pool = []
    for p in world.people:
        pool += [world.fact("born_in", p), world.fact("works_at", p), ("lives_in", p, rng.choice(world.cities)),
                 ("worked_before", p, rng.choice([f for f in world.firms if f != world.employer[p]]))]
    for f in world.firms:
        pool += [world.fact("hq_in", f), world.fact("founded_by", f), ("office_in", f, rng.choice([c for c in world.cities if c != world.hq[f]]))]
    pool += [world.fact("city_in", c) for c in world.cities] + [("married_to", a, b) for a, b in world.spouse.items()]
    chain_entities = {subject} | {f[2] for f in needed}
    close = [f for f in pool if f not in needed and (f[1] in chain_entities or f[2] in chain_entities)]
    far = [f for f in pool if f not in needed and f not in close]
    distractors = rng.sample(close, min(len(close), rng.randint(4, 7))) + rng.sample(far, rng.randint(1, 3))
    sentences = [(f, _render(f, rng)) for f in needed + distractors]
    if rng.random() < 0.35:  # restate one needed fact in other words, so neither copy is needed on its own
        f = rng.choice(needed)
        sentences.append((f, _render(f, rng)))
    rng.shuffle(sentences)
    n_par = rng.randint(6, min(9, len(sentences)))
    cuts = sorted(rng.sample(range(1, len(sentences)), n_par - 1))
    paragraphs = [sentences[i:j] for i, j in zip([0] + cuts, cuts + [len(sentences)])]
    facts = [set(f for f, _ in par) for par in paragraphs]
    answer = solve(set().union(*facts), kind, subject)
    if answer is None or answer != needed[-1][2]:
        return None
    keep = [solve(set().union(*(facts[:i] + facts[i + 1:])), kind, subject) != answer for i in range(len(paragraphs))]
    return {"question": QUESTIONS[kind][0].format(s=subject), "kind": kind, "answer": answer,
            "paragraphs": [" ".join(text for _, text in par) for par in paragraphs], "keep": keep}


@register_task
class FactChainPruning(TaskAdapter):
    name = f"{LEAF}/fact_chains"
    leaf = LEAF
    source = "Leave-one-out paragraph ablation on generated multi-hop fact chains (generated)"
    url = "https://github.com/TianyuCodings/S1-Arena"
    license = "CC BY 4.0 (generated by S1-Arena)"
    tier = "C"
    criteria_source = "Written; gold is computed by removing each paragraph in turn and re-solving the question"
    notes = ("Fictional people, companies, cities and countries, so world knowledge cannot fill a gap. Each question needs a chain "
             "of 2-3 facts; distractors mention the same entities (past employers, branch offices, where someone lives). In about "
             "a third of the items one needed fact is stated twice, so neither copy is needed on its own. Checked by an independent "
             "re-parser and solver.")
    POOL = 4000

    def load(self):
        rng = random.Random(f"{self.name}:pool")
        rows = []
        while len(rows) < self.POOL:
            row = generate(rng)
            if row is not None:
                rows.append(row)
        return rows

    def label_of(self, record):
        return f"{record['kind']}|{sum(record['keep'])}"

    def to_item(self, record, rng):
        state = {"question": record["question"]}
        questions, gold = {}, {}
        for i, (text, keep) in enumerate(zip(record["paragraphs"], record["keep"]), 1):
            state[f"P{i}"] = text
            questions[f"keep_P{i}"] = Question(type="boolean", instructions=(
                f"A context-compression step must decide whether paragraph P{i} can be dropped. Is P{i} needed: with P{i} "
                "removed and every other paragraph kept, could the question no longer be answered from the paragraphs?"))
            gold[f"keep_P{i}"] = Gold(label="true" if keep else "false")
        return self.item(state, questions, gold, kind=record["kind"], needed=sum(record["keep"]))


# ---------- QASPER: minimal evidence chosen by NLP-practitioner annotators ----------

MAX_PARAGRAPHS = 8


@register_task
class QasperPruning(TaskAdapter):
    name = f"{LEAF}/qasper"
    leaf = LEAF
    source = "QASPER, questions on NLP papers with annotated evidence paragraphs, test split (Dasigi et al., 2021)"
    url = "https://huggingface.co/datasets/allenai/qasper"
    license = "CC BY 4.0"
    tier = "B"
    criteria_source = "Adapted from the QASPER annotation instructions ('select the minimal set of evidence snippets that contains the answer')"
    notes = ("Each item shows up to 8 paragraphs of one paper in their original order: every paragraph any annotator chose as "
             "evidence, plus other paragraphs of the same paper. Hard label: chosen by at least one annotator; soft label: share "
             "of the answering annotators who chose it (annotator agreement is low, Jaccard about 0.47). Questions answered from "
             "figures or tables, or with more than 5 evidence paragraphs, are left out.")

    def load(self):
        rows = []
        for paper in hf_rows("allenai/qasper", None, "test", revision="refs/convert/parquet"):
            sections = paper["full_text"]["section_name"]
            paragraphs = []
            for name, paras in zip(sections, paper["full_text"]["paragraphs"]):
                paragraphs += [(str(name or "").strip(), p.strip()) for p in paras if p and len(p.strip()) >= 40]
            texts = [p for _, p in paragraphs]
            for qi, question in enumerate(paper["qas"]["question"]):
                evidence_sets = []
                for answer in paper["qas"]["answers"][qi]["answer"]:
                    if answer["unanswerable"]:
                        continue
                    evidence = {e.strip() for e in answer["evidence"] if e.strip()}
                    if not evidence or not evidence <= set(texts):
                        evidence_sets = None  # figure/table evidence or text we cannot locate
                        break
                    evidence_sets.append(evidence)
                if not evidence_sets:
                    continue
                share = defaultdict(float)
                for evidence in evidence_sets:
                    for e in evidence:
                        share[e] += 1 / len(evidence_sets)
                if len(share) > 5:
                    continue
                rows.append({"id": paper["qas"]["question_id"][qi], "title": paper["title"], "question": question.strip(),
                             "paragraphs": paragraphs, "share": dict(share), "annotators": len(evidence_sets)})
        return rows

    def label_of(self, record):
        return f"{len(record['share'])}|{record['annotators']}"

    def to_item(self, record, rng):
        index = {text: i for i, (_, text) in enumerate(record["paragraphs"])}
        chosen = {index[e] for e in record["share"]}
        others = [i for i in range(len(record["paragraphs"])) if i not in chosen]
        chosen |= set(rng.sample(others, min(len(others), MAX_PARAGRAPHS - len(chosen))))
        state = {"paper_title": record["title"], "question": record["question"]}
        questions, gold = {}, {}
        for k, i in enumerate(sorted(chosen), 1):
            section, text = record["paragraphs"][i]
            state[f"P{k}"] = (f"[{section}] " if section else "") + text
            share = min(1.0, record["share"].get(text, 0.0))
            questions[f"keep_P{k}"] = Question(type="boolean", instructions=(
                f"Keep paragraph P{k} for answering the question? Keep it if it belongs to the minimal set of paragraphs that "
                "contains the answer to the question. When several paragraphs could serve as evidence, prefer the one that "
                "adequately answers the question, then the one that occurs earlier in the paper."))
            gold[f"keep_P{k}"] = Gold(label="true" if share > 0 else "false", dist={"true": share, "false": 1.0 - share})
        return self.item(state, questions, gold, original_id=record["id"], annotators=record["annotators"])
