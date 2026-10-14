"""Leaf: code search relevance (代码搜索相关性): graded relevance of a code snippet to a query.

CodeSearchNet Challenge: expert 0-3 relevance ratings of functions for 99 real search queries;
pairs rated by several annotators get the rating distribution as a soft label. CoSQA: web search
queries labelled by at least three annotators as answered or not by a Python function (the
released label is the majority), asked as a two-level scale.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ...env import load_env
from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import CACHE, fetch

LEAF = "code/code_search"
CSN_ANNOTATIONS = "https://raw.githubusercontent.com/github/CodeSearchNet/master/resources/annotationStore.csv"
CSN_LANGUAGES = {"Go": "go", "Java": "java", "JavaScript": "javascript", "PHP": "php", "Python": "python", "Ruby": "ruby"}

# CodeSearchNet Challenge annotator instructions (Husain et al., 2019, Figure 1), verbatim.
CSN_INSTRUCTIONS = (
    "Through the annotations, we want to measure how relevant would these results be to you. You don't have to be "
    "absolutely certain about the correctness of the code. You might be interested in copy-pasting the code, finding a "
    "library to use or just getting some understanding about how something is implemented. You might be searching "
    "within your project (e.g. to reuse code within your project), your company or all of GitHub. Please annotate the "
    "results according to the following scheme:"
)
CSN_LEVELS = [
    "0: Totally irrelevant. I would never want to see this for this query.",
    "1: Weak match. That's not exactly what I was looking for, but there are some useful elements/pointers to things "
    "that I would use (e.g. APIs, code structure) and can form the basis of a new query or exploration towards solving my query.",
    "2: Strong match. This does more or less what I was looking for. I would use the code in here as a backbone for my "
    "purpose, but I won't necessarily copy-paste it or use this library.",
    "3: Exact match. This seems exactly what I was looking for. I would copy-paste the code and make minor adaptations "
    "or will use this functionality of the library in my code.",
]


def csn_code(urls: set[str]) -> dict[str, str]:
    """Function source for the given CodeSearchNet URLs, read from the corpus on the Hub (only matching row groups)."""
    path = CACHE / "codesearchnet_challenge_code.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    load_env()
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = [f"datasets/code-search-net/code_search_net/{lang}/{split}-00000-of-00001.parquet"
             for lang in CSN_LANGUAGES.values() for split in ("train", "validation", "test")]

    def scan(name):
        with fs.open(name, block_size=8 * 2**20) as handle:
            parquet = pq.ParquetFile(handle, pre_buffer=True)
            groups = []
            for g in range(parquet.metadata.num_row_groups):
                column = parquet.read_row_group(g, columns=["func_code_url"]).column(0).to_pylist()
                if urls.intersection(column):
                    groups.append(g)
            found = {}
            for g in groups:
                table = parquet.read_row_group(g, columns=["func_code_url", "whole_func_string"])
                for url, code in zip(table.column(0).to_pylist(), table.column(1).to_pylist()):
                    if url in urls:
                        found[url] = code
            return found

    code = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for found in pool.map(scan, files):
            code.update(found)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".part")
    tmp.write_text(json.dumps(code, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp.rename(path)
    return code


@register_task
class CodeSearchNetChallenge(TaskAdapter):
    name = f"{LEAF}/codesearchnet_challenge"
    rank_key = "state:query"  # NDCG groups
    leaf = LEAF
    source = "CodeSearchNet Challenge expert relevance annotations, six languages (Husain et al., 2019)"
    url = "https://github.com/github/CodeSearchNet"
    license = "MIT (annotations); code from public GitHub repositories under their own licenses"
    tier = "A"
    criteria_source = "Annotator instructions and the 0-3 scale verbatim from Husain et al. (2019), Figure 1"
    notes = (
        "Only query-function pairs whose function is in the documented CodeSearchNet corpus on the Hub (about half of the "
        "annotated pairs; the rest point to undocumented functions not in that corpus). Pairs rated by two or more "
        "annotators get the rating distribution as a soft label; the hard label is the most frequent rating (lower on ties)."
    )

    def load(self):
        with fetch(CSN_ANNOTATIONS, "codesearchnet_annotationStore.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        ratings = defaultdict(list)
        for r in rows:
            if r["Relevance"].strip() in ("0", "1", "2", "3"):
                ratings[(r["Language"], r["Query"].strip(), r["GitHubUrl"].strip())].append(int(r["Relevance"]))
        code = csn_code({url for _, _, url in ratings})
        records = []
        for (language, query, url), scores in sorted(ratings.items()):
            if url in code and code[url].strip():
                records.append({"language": language, "query": query, "url": url, "code": code[url], "ratings": scores})
        return records

    def label_of(self, record):
        return f"{record['language']}|{self.hard(record)}"

    @staticmethod
    def hard(record) -> int:
        counts = Counter(record["ratings"])
        return min(counts, key=lambda level: (-counts[level], level))

    def to_item(self, record, rng):
        question = Question(type="score", instructions="Rate how relevant the code is to the search query. " + CSN_INSTRUCTIONS, criteria=CSN_LEVELS)
        counts = Counter(record["ratings"])
        dist = {str(level): counts.get(level, 0) / len(record["ratings"]) for level in range(4)} if len(record["ratings"]) > 1 else None
        state = {"language": record["language"], "query": record["query"], "code": record["code"]}
        gold = Gold(label=str(self.hard(record)), dist=dist)
        return self.item(state, {"relevance": question}, {"relevance": gold}, url=record["url"], annotators=len(record["ratings"]))


# CoSQA answer-annotation guideline (Huang et al., 2021, Section 3.2), verbatim.
COSQA_INSTRUCTIONS = (
    "Judge whether the code can answer the query. Label the instance with \"1\" if the code is a correct answer; "
    "otherwise, it is labeled \"0\". Judgment should be made after comprehensively considering the relevance between "
    "query with documentation, query with function header, and query with function body. If code can answer the query "
    "and even exceed the demand of the query, it is a correct answer. If code can meet a certain category of the query "
    "demands, it is also a correct answer. If code satisfies no more than 50% of the query demands, the code can not "
    "correctly answer the query. If a small part of the code is relevant to the query, the code can not be a correct answer."
)
COSQA_LEVELS = ["0: The code is not a correct answer to the query.", "1: The code is a correct answer to the query."]


@register_task
class CoSQA(TaskAdapter):
    name = f"{LEAF}/cosqa"
    rank_key = "state:query"  # NDCG groups
    leaf = LEAF
    source = "CoSQA, Bing web queries paired with Python functions, dev split (Huang et al., 2021)"
    url = "https://github.com/Jun-jie-Huang/CoCLR"
    license = "MIT (CoCLR repository); C-UDA 1.0 as part of CodeXGLUE; functions from the CodeSearchNet corpus"
    tier = "A"
    criteria_source = "Answer-annotation guideline verbatim from Huang et al. (2021), Section 3.2"
    notes = (
        "Binary relevance asked as a two-level scale. Each pair was labelled by at least three annotators; only the final "
        "label is released, so it is a hard label. Read from the Hugging Face copy gonglinyuan/CoSQA of the CodeXGLUE files; the test "
        "labels (WebQueryTest) are not public, so the dev split is used."
    )

    def load(self):
        load_env()
        from huggingface_hub import hf_hub_download

        with open(hf_hub_download("gonglinyuan/CoSQA", "cosqa-dev.json", repo_type="dataset"), encoding="utf-8") as f:
            return json.load(f)

    def label_of(self, record):
        return str(record["label"])

    def to_item(self, record, rng):
        question = Question(type="score", instructions=COSQA_INSTRUCTIONS, criteria=COSQA_LEVELS)
        state = {"query": record["doc"], "code": record["code"]}
        return self.item(state, {"answers": question}, {"answers": Gold(label=str(record["label"]))}, original_id=record["idx"])
