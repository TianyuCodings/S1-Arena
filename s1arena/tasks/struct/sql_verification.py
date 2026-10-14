"""Leaf: SQL verification (SQL 是否回答了问题).

A question, the database schema and a model-generated SQL query; does the query correctly
answer the question? Both sources use SQL written by real text-to-SQL systems:

- NL2SQL-BUGs: RESDSQL, CodeS and CHESS outputs on BIRD dev, labelled correct or incorrect
  by execution against expert-cleaned gold SQL and annotated with error types by experts.
- Spider dev beam predictions of three parsers (SmBoP, RESDSQL+NatSQL, NatSQL) released by
  Chen et al. (2023) with execution-based labels. We re-execute every kept candidate: a
  correct label must also hold on every database of the Spider test suite (Zhong et al.,
  2020), and an incorrect label must show a different result on the dev database.
"""

from __future__ import annotations

import itertools
import json
import random
import re
import sqlite3
import zipfile
from collections import Counter
from pathlib import Path

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import CACHE, fetch

LEAF = "struct/sql_verification"

# Adapted from the NL2SQL-BUGs evaluation prompt (Liu et al., 2025): the task sentence and the
# main categories of its semantic-error taxonomy.
ERRORS = (
    "A query is incorrect if it has any semantic error: wrong, missing or extra attributes; wrong, missing or extra "
    "tables or join conditions; wrong values or data formats; wrong comparison or logical operators; missing, wrong "
    "or extra conditions; misused functions (aggregate, date, string, conversion, conditional); missing or extra "
    "clauses such as GROUP BY or ORDER BY; wrong subqueries; wrong sort direction or DISTINCT."
)


# ---------- NL2SQL-BUGs (BIRD dev) ----------

BUGS = "https://raw.githubusercontent.com/HKUSTDial/NL2SQL-Bugs-Benchmark/9cb75ddbf3f64120cdc254df6048601e85b039ee/"


@register_task
class NL2SQLBugs(TaskAdapter):
    name = f"{LEAF}/nl2sql_bugs"
    leaf = LEAF
    source = "NL2SQL-BUGs, BIRD dev questions with SQL from RESDSQL, CodeS and CHESS (Liu et al., 2025)"
    url = "https://github.com/HKUSTDial/NL2SQL-Bugs-Benchmark"
    license = "Not stated (NL2SQL-BUGs labels); BIRD questions, evidence and schemas CC BY-SA 4.0"
    tier = "B"
    criteria_source = "Adapted from the NL2SQL-BUGs evaluation prompt: task sentence and the main error categories of its taxonomy"
    notes = ("Labels: execution against expert-cleaned BIRD gold SQL, with error types annotated by two experts "
             "(Cohen's kappa 0.78). The schema is the benchmark's enhanced DDL with column descriptions. "
             "The repository states no license.")

    def load(self):
        rows = json.loads(fetch(BUGS + "data/NL2SQL-Bugs-with-evidence.json", "nl2sql_bugs_with_evidence.json").read_text(encoding="utf-8"))
        schemas = json.loads(fetch(BUGS + "prompts/schema_dev.json", "nl2sql_bugs_schema_dev.json").read_text(encoding="utf-8"))
        for row in rows:
            row["schema"] = schemas[row["db_id"]]["enhanced_ddl"].strip()
        return rows

    def label_of(self, record):
        if record["label"]:
            return "true"
        return "false|" + (record["error_types"][0]["error_type"] if record["error_types"] else "unknown")

    def to_item(self, record, rng):
        question = Question(
            type="boolean",
            instructions="Does the SQL query correctly answer the question, given the evidence and the database schema? " + ERRORS,
        )
        state = {"database_schema": record["schema"], "question": record["question"], "evidence": record["evidence"] or "", "sql": record["sql"]}
        errors = sorted({e["error_type"] for e in record["error_types"]})
        gold = Gold(label="true" if record["label"] else "false")
        return self.item(state, {"correct": question}, {"correct": gold}, db_id=record["db_id"], error_types=errors, original_id=record["id"])


# ---------- Spider dev, recorded parser beams (Chen et al., 2023) ----------

ED_ZIP = "https://drive.usercontent.google.com/download?id=1E_ojxiX8m-8M4-o2wajL9o6FrWEHccuR&export=download&confirm=t"
SPIDER_ZIP = "https://drive.usercontent.google.com/download?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download&confirm=t"
TESTSUITE_ZIP = "https://drive.usercontent.google.com/download?id=1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w&export=download&confirm=t"
PARSERS = {
    "smbop": "datasets/smbop/ed_smbop_beam_test.json",
    "resdsql_natsql": "datasets/resdnatsql/ed_resdnatsql_beam_test.json",
    "natsql": "datasets/natsql/ed_natsql_beam_test.json",
}
SKIP_DBS = {"wta_1"}  # its test suite unpacks to 3.3 GB
POOL = 1200  # verified candidates kept for sampling
MAX_STEPS = 20_000_000  # SQLite VM steps before a query is abandoned
SAMPLE_ROWS = 3


def _normalize(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, bytes):
        return value.hex()
    return value


def run_sql(path: Path, sql: str):
    """Rows of a query, or None if it fails or runs too long."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    steps = {"n": 0}

    def progress():
        steps["n"] += 1
        return 1 if steps["n"] * 1000 > MAX_STEPS else 0

    con.set_progress_handler(progress, 1000)
    try:
        return [tuple(_normalize(v) for v in row) for row in con.execute(sql).fetchall()]
    except Exception:
        return None
    finally:
        con.close()


def _key(row):
    return tuple((type(v).__name__ if v is not None else "", str(v)) for v in row)


def results_equal(gold, pred, ordered: bool) -> bool:
    """Result equality as in test-suite evaluation: rows as a multiset (or a list when the gold query
    orders them), allowing the predicted columns to be a permutation of the gold columns."""
    if gold is None or pred is None or len(gold) != len(pred):
        return False
    if not gold:
        return True
    width = len(gold[0])
    if any(len(r) != width for r in pred):
        return False
    gold_cols = [Counter(_key((r[i],)) for r in gold) for i in range(width)]
    pred_cols = [Counter(_key((r[i],)) for r in pred) for i in range(width)]
    options = [[j for j in range(width) if pred_cols[j] == gold_cols[i]] for i in range(width)]
    if any(not o for o in options):
        return False
    target = gold if ordered else sorted(gold, key=_key)
    for perm in itertools.islice(itertools.product(*options), 2000):
        if len(set(perm)) != width:
            continue
        rows = [tuple(r[j] for j in perm) for r in pred]
        if (rows if ordered else sorted(rows, key=_key)) == target:
            return True
    return False


def top_level_order_by(sql: str) -> bool:
    """True when ORDER BY occurs outside parentheses (i.e. orders the final result)."""
    depth, text = 0, re.sub(r"'[^']*'|\"[^\"]*\"", "''", sql)
    for i, ch in enumerate(text):
        depth += (ch == "(") - (ch == ")")
        if depth == 0 and text[i:i + 8].lower() == "order by":
            return True
    return False


def compact(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower().replace("( ", "(").replace(" )", ")")


class SpiderDatabases:
    """Extracts the Spider dev databases and their test suites once, into the download cache."""

    def __init__(self):
        self.root = CACHE / "spider_extracted"
        self.spider = zipfile.ZipFile(fetch(SPIDER_ZIP, "spider_data.zip"))
        self.suite = zipfile.ZipFile(fetch(TESTSUITE_ZIP, "spider_testsuite_databases.zip"))

    def original(self, db: str) -> Path:
        return self._extract(self.spider, f"spider_data/database/{db}/{db}.sqlite")

    def variants(self, db: str) -> list[Path]:
        names = sorted(n for n in self.suite.namelist() if n.startswith(f"database/{db}/") and n.endswith(".sqlite"))
        return [self._extract(self.suite, n) for n in names]

    def _extract(self, archive: zipfile.ZipFile, member: str) -> Path:
        path = self.root / member
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part")
            tmp.write_bytes(archive.read(member))
            tmp.rename(path)
        return path


def schema_text(path: Path) -> str:
    """CREATE TABLE statements, each followed by a few example rows."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    parts = []
    for name, ddl in con.execute("SELECT name, sql FROM sqlite_master WHERE type = 'table' AND sql IS NOT NULL ORDER BY rowid"):
        cursor = con.execute(f'SELECT * FROM "{name}" LIMIT {SAMPLE_ROWS}')
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        block = ddl.strip() + ";"
        if rows:
            lines = [" | ".join(columns)] + [" | ".join("NULL" if v is None else str(v)[:60] for v in r) for r in rows]
            block += f"\n/* {len(rows)} example rows from {name}:\n" + "\n".join(lines) + "\n*/"
        parts.append(block)
    con.close()
    return "\n\n".join(parts)


@register_task
class SpiderParserBeams(TaskAdapter):
    name = f"{LEAF}/spider_parser_beams"
    leaf = LEAF
    source = "Spider dev with beam predictions of SmBoP, RESDSQL+NatSQL and NatSQL (Yu et al., 2018; Chen et al., 2023)"
    url = "https://github.com/OSU-NLP-Group/Text2SQL-Error-Detection"
    license = "Spider CC BY-SA 4.0; parser predictions: not stated"
    tier = "B"
    criteria_source = "Adapted from the NL2SQL-BUGs evaluation prompt; gold from execution against the Spider gold query"
    notes = ("Executable beam predictions labelled with the fixed Spider execution evaluation of Chen et al. (2023); "
             "candidates whose gold query returns no rows or whose label changed under that fix are dropped. "
             "Correct labels are re-checked on all Spider test-suite databases (Zhong et al., 2020), incorrect labels "
             "on the dev database. The wta_1 database is left out (its test suite is 3.3 GB). The schema shows three "
             "example rows per table.")

    def load(self):
        archive = zipfile.ZipFile(fetch(ED_ZIP, "text2sql_error_detection_datasets.zip"))
        candidates, seen = [], set()
        for parser, member in PARSERS.items():
            for beam in json.loads(archive.read(member)):
                for c in beam:
                    if not c["valid_sql"] or c["empty_gold"] or c["exec_disagree"] or c["db_id"] in SKIP_DBS:
                        continue
                    key = (c["question"], compact(c["sql"]))
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append({"parser": parser, "idx": c["idx"], "db_id": c["db_id"], "question": c["question"],
                                       "sql": c["sql"].strip(), "gold_sql": c["gold"].strip(), "label": int(c["label"]), "em": int(c["em"])})
        random.Random(f"{self.name}:pool").shuffle(candidates)
        dbs, schemas, variants, cache, out = SpiderDatabases(), {}, {}, {}, []
        per_label = Counter()
        for c in candidates:
            if per_label[c["label"]] >= POOL // 2:
                continue
            db = c["db_id"]
            if db not in schemas:
                schemas[db] = schema_text(dbs.original(db))
                variants[db] = dbs.variants(db)
            if self.verified(c, dbs.original(db), variants[db], cache):
                out.append(dict(c, schema=schemas[db]))
                per_label[c["label"]] += 1
            if min(per_label[0], per_label[1]) >= POOL // 2:
                break
        return out

    @staticmethod
    def verified(c, original: Path, suite: list[Path], cache: dict) -> bool:
        ordered = top_level_order_by(c["gold_sql"])

        def gold_on(path):
            key = (c["gold_sql"], str(path))
            if key not in cache:
                cache[key] = run_sql(path, c["gold_sql"])
            return cache[key]

        if c["label"] == 0:  # must differ on the dev database
            gold, pred = gold_on(original), run_sql(original, c["sql"])
            return gold is not None and pred is not None and not results_equal(gold, pred, ordered)
        for path in suite:  # must agree on every database of the test suite
            gold = gold_on(path)
            if gold is None:
                continue
            if not results_equal(gold, run_sql(path, c["sql"]), ordered):
                return False
        return True

    def label_of(self, record):
        return f"{record['label']}|{record['parser']}"

    def to_item(self, record, rng):
        question = Question(type="boolean", instructions="Does the SQL query correctly answer the question on this database? " + ERRORS)
        state = {"database_schema": record["schema"], "question": record["question"], "sql": record["sql"]}
        gold = Gold(label="true" if record["label"] else "false")
        return self.item(state, {"correct": question}, {"correct": gold}, db_id=record["db_id"], parser=record["parser"], spider_dev_index=record["idx"])
