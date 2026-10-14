"""Shared loaders for the retrieval domain (no tasks are registered here)."""

from __future__ import annotations

import gzip
import json
import re
import tarfile

from ..common import CACHE, fetch

MSMARCO = "https://msmarco.z22.web.core.windows.net/msmarcoranking/"
TREC_DL = {  # year -> (queries, passage qrels)
    2019: (MSMARCO + "msmarco-test2019-queries.tsv.gz", "https://trec.nist.gov/data/deep/2019qrels-pass.txt"),
    2020: (MSMARCO + "msmarco-test2020-queries.tsv.gz", "https://trec.nist.gov/data/deep/2020qrels-pass.txt"),
}


def fix_mojibake(text: str) -> str:
    """Some MS MARCO passages are UTF-8 that was decoded as Latin-1/CP1252 once too often; undo that."""
    if not any(c in text for c in "âÃÂ"):
        return text
    for codec in ("cp1252", "latin-1"):
        try:
            return text.encode(codec).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return text


def msmarco_passages(pids: set[str]) -> dict[str, str]:
    """Texts of the given MS MARCO v1 passages, read once from the official collection (1 GB download)
    and cached as a small JSON file next to it."""
    key = "msmarco_trec_dl_judged_passages.json"
    path = CACHE / key
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if pids <= set(cached):
            return {p: cached[p] for p in pids}
    archive = fetch(MSMARCO + "collection.tar.gz", "msmarco_passage_collection.tar.gz")
    found: dict[str, str] = {}
    wanted = {p.encode() for p in pids}
    with tarfile.open(archive, "r:gz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("collection.tsv"))
        with tar.extractfile(member) as f:
            for line in f:
                pid, _, text = line.partition(b"\t")
                if pid in wanted:
                    found[pid.decode()] = fix_mojibake(text.decode("utf-8").strip())
    path.write_text(json.dumps(found, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return found


def opening(text: str, n: int = 150) -> str:
    """Normalized start of a passage, ignoring leading list numbers and punctuation, to spot near-duplicates."""
    return re.sub(r"^[^a-z]+", "", " ".join(text.lower().split()))[:n]


def consistent_judgments(rows: list[dict]) -> list[dict]:
    """Drop judgments of near-duplicate passages (same query, same opening) that received different grades."""
    grades: dict[tuple, set] = {}
    for r in rows:
        grades.setdefault((r["year"], r["qid"], opening(r["passage"])), set()).add(r["grade"])
    return [r for r in rows if len(grades[(r["year"], r["qid"], opening(r["passage"]))]) == 1]


def trec_dl_judgments() -> list[dict]:
    """All TREC DL 2019 and 2020 passage judgments (grades 0-3) with query and passage text."""
    rows = []
    for year, (queries_url, qrels_url) in TREC_DL.items():
        with gzip.open(fetch(queries_url), "rt", encoding="utf-8") as f:
            queries = dict(line.rstrip("\n").split("\t", 1) for line in f if line.strip())
        with fetch(qrels_url).open(encoding="utf-8") as f:
            for line in f:
                qid, _, pid, grade = line.split()
                rows.append({"year": year, "qid": qid, "query": queries[qid].strip(), "pid": pid, "grade": int(grade)})
    passages = msmarco_passages({r["pid"] for r in rows})
    return [dict(r, passage=passages[r["pid"]]) for r in rows if passages.get(r["pid"])]
