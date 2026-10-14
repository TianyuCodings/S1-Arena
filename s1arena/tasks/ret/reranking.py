"""Leaf: reranking (重排序). Pick the most relevant of five candidates; every candidate carries a relevance
judgment from the source collection, so the gold is the only top-graded one."""

from __future__ import annotations

import random
import re
from collections import defaultdict

from ...env import load_env
from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from .sources import consistent_judgments, fix_mojibake, opening, trec_dl_judgments

LEAF = "ret/reranking"
LETTERS = "ABCDE"
K = 5


def candidate_options(positive: str, negatives: list[str], rng: random.Random) -> tuple[dict[str, str], str]:
    texts = [positive, *negatives]
    rng.shuffle(texts)
    return {LETTERS[i]: t for i, t in enumerate(texts)}, LETTERS[texts.index(positive)]


@register_task
class TrecDLRerank(TaskAdapter):
    name = f"{LEAF}/trec_dl"
    leaf = LEAF
    source = "TREC 2019 and 2020 Deep Learning Track, passage judgments on MS MARCO (Craswell et al., 2020; 2021)"
    url = "https://microsoft.github.io/msmarco/TREC-Deep-Learning"
    license = "MS MARCO terms: non-commercial research use; NIST relevance judgments"
    tier = "B"
    criteria_source = "Adapted from the TREC DL passage judging scale: the answer is the one 'perfectly relevant' passage"
    sampling = "balanced"
    notes = ("Each item has one passage NIST graded 3 (perfectly relevant) and four judged passages of the same query: up to two "
             "graded 1 (related), the rest 0 (irrelevant); passages graded 2 are never used, and near-duplicate passages (same "
             "opening) with conflicting grades are dropped. Sampled evenly across queries, because a few queries have 60-160 "
             "perfectly relevant passages each. The same judgments also feed ret/relevance_grading/trec_dl.")

    def load(self):
        by_query = defaultdict(list)
        for r in consistent_judgments(trec_dl_judgments()):
            by_query[(r["year"], r["qid"])].append(r)
        rows = []
        for (year, qid), judged in sorted(by_query.items()):
            perfect = [r for r in judged if r["grade"] == 3]
            pool = {1: [], 0: []}
            seen = {opening(r["passage"]) for r in judged if r["grade"] >= 2}
            for r in sorted(judged, key=lambda r: r["pid"]):
                if r["grade"] in pool and opening(r["passage"]) not in seen:  # no duplicate text within a list
                    pool[r["grade"]].append(r["passage"])
                    seen.add(opening(r["passage"]))
            unique = {}
            for p in sorted(perfect, key=lambda r: r["pid"]):
                unique.setdefault(opening(p["passage"]), p)
            for p in unique.values():
                rng = random.Random(f"{self.name}:{year}:{qid}:{p['pid']}")
                related = rng.sample(pool[1], min(len(pool[1]), 2))
                rest = rng.sample(pool[0], min(len(pool[0]), K - 1 - len(related)))
                if len(related) + len(rest) < K - 1:
                    rest += rng.sample([t for t in pool[1] if t not in related], K - 1 - len(related) - len(rest))
                negatives = related + rest
                if len(negatives) == K - 1:
                    rows.append({"year": year, "qid": qid, "query": p["query"], "pid": p["pid"], "positive": p["passage"],
                                 "negatives": negatives, "related": len(related) + sum(1 for t in rest if t in pool[1])})
        return rows

    def label_of(self, record):
        return f"{record['year']}|{record['qid']}"  # the answer letter is set by the shuffle in to_item

    def to_item(self, record, rng):
        options, answer = candidate_options(record["positive"], record["negatives"], rng)
        question = Question(
            type="choice",
            instructions=("Which candidate passage is the most relevant to the search query? Exactly one candidate is perfectly "
                          "relevant: it is dedicated to the query and contains the exact answer."),
            criteria=options,
        )
        return self.item({"query": record["query"]}, {"best_passage": question}, {"best_passage": Gold(label=answer)},
                         year=record["year"], qid=record["qid"], pid=record["pid"], related_distractors=record["related"])


def readable(text: str) -> bool:
    """Excludes page debris (HTML, CSS, link lists, lists of names, file listings): at least 20 words averaging at most
    12 characters, at least two sentences, and few markdown links, braces or tags."""
    words = text.split()
    if len(words) < 20 or sum(map(len, words)) / len(words) > 12:
        return False
    sentences = len(re.findall(r"[a-z)][.?!](\s|$)", text))
    return sentences >= 2 and text.count("](") <= len(words) / 15 and text.count("{") + text.count("<") <= len(words) / 10


BRIGHT_DOMAINS = ("biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow", "sustainable_living")


@register_task
class BrightRerank(TaskAdapter):
    name = f"{LEAF}/bright"
    leaf = LEAF
    source = "BRIGHT, StackExchange domains (Su et al., 2025)"
    url = "https://huggingface.co/datasets/xlangai/BRIGHT"
    license = "CC BY 4.0"
    tier = "B"
    criteria_source = "Adapted from the BRIGHT relevance definition for StackExchange and its annotator instructions (Appendix F.1)"
    notes = ("Query = the full StackExchange post (7 domains). One passage from a web page cited in an accepted or highly voted answer "
             "and confirmed helpful by the annotators, against four passages from the hard-negative pages that the annotators "
             "collected for the same post (topically similar, verified unhelpful). Candidates are passages of 150-6,000 characters "
             "that read as text (page debris such as HTML, CSS, link lists and name lists is skipped, for positives and negatives "
             "alike); the longest annotated passage is the positive, and other passages of the cited pages are never used.")

    def load(self):
        load_env()
        import pandas as pd
        from huggingface_hub import hf_hub_download

        rows = []
        for domain in BRIGHT_DOMAINS:
            read = lambda part: pd.read_parquet(hf_hub_download("xlangai/BRIGHT", f"{part}/{domain}-00000-of-00001.parquet", repo_type="dataset"))
            docs = read("documents")
            content = {doc_id: fix_mojibake(text) for doc_id, text in zip(docs["id"], docs["content"])}
            folders = defaultdict(list)
            for doc_id in sorted(content):
                folders[doc_id.split("/")[0]].append(doc_id)
            usable = lambda doc_id: 150 <= len(content.get(doc_id, "").strip()) <= 6000 and readable(content[doc_id])
            for ex in read("examples").to_dict("records"):
                gold = list(ex["gold_ids"])
                folder = {g.split("/")[0] for g in gold}
                if len(folder) != 1:
                    continue
                folder = folder.pop()
                pages = [g.split("/", 1)[1].rsplit(".", 1)[0] for g in ex["gold_ids_long"]]
                on_cited_page = lambda doc_id: any(doc_id.split("/", 1)[1].startswith(p + "_") or doc_id.split("/", 1)[1] == p + ".txt" for p in pages)
                positives = sorted(g for g in gold if usable(g))
                negatives = [d for d in folders[folder] if d not in gold and not on_cited_page(d) and usable(d)]
                if not positives or len(negatives) < K - 1:
                    continue
                rng = random.Random(f"{self.name}:{domain}:{ex['id']}")
                positive = max(positives, key=lambda g: (len(content[g]), g))  # the fullest annotated passage
                seen, pool = {opening(content[positive])}, []
                for doc_id in negatives:  # no candidate may repeat another one's text
                    if opening(content[doc_id]) not in seen:
                        seen.add(opening(content[doc_id]))
                        pool.append(doc_id)
                if len(pool) < K - 1:
                    continue
                chosen = rng.sample(pool, K - 1)
                rows.append({"domain": domain, "id": str(ex["id"]), "query": fix_mojibake(ex["query"].strip()), "positive": content[positive].strip(),
                             "negatives": [content[d].strip() for d in chosen], "positive_id": positive})
        return rows

    def label_of(self, record):
        return record["domain"]

    def to_item(self, record, rng):
        options, answer = candidate_options(record["positive"], record["negatives"], rng)
        question = Question(
            type="choice",
            instructions=("Which candidate passage helps answer the StackExchange post? The helpful passage provides critical "
                          "information for understanding the post or addressing its question, such as key concepts, theories or "
                          "code. The other candidates cover similar topics but do not help."),
            criteria=options,
        )
        return self.item({"post": record["query"]}, {"helpful_passage": question}, {"helpful_passage": Gold(label=answer)},
                         domain=record["domain"], original_id=record["id"])
