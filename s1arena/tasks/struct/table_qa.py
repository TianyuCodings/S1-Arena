"""Leaf: table question answering recast as multiple choice (表格问答，改写成选择题).

WikiTableQuestions and TAT-QA give one free-form answer per question. Each becomes one option
among distractors that are wrong by construction: other values from the same table column
(or row) that differ from the gold answer after the official answer normalisation, or nearby
numbers when the answer is computed. TabMWP comes with its own multiple-choice questions.
"""

from __future__ import annotations

import json
import re
import unicodedata
import zipfile

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import fetch

LEAF = "struct/table_qa"
LETTERS = "ABCDEFGH"
ASK = "Which option answers the question, based on the table?"


def md_table(header: list[str], rows: list[list[str]], max_cell: int | None = None) -> str:
    """A markdown table; pipes inside cells are escaped and line breaks collapsed."""

    def cell(value) -> str:
        text = " ".join(str(value).split())
        if max_cell and len(text) > max_cell:
            text = text[: max_cell - 1] + "…"
        return text.replace("|", "\\|")

    width = max([len(header)] + [len(r) for r in rows])
    header = list(header) + [""] * (width - len(header))
    lines = ["| " + " | ".join(cell(h) for h in header) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(cell(v) for v in list(r) + [""] * (width - len(r))) + " |" for r in rows]
    return "\n".join(lines)


def lettered(values: list[str], gold: str, rng) -> tuple[dict[str, str], str]:
    """Shuffle the option texts with the record rng and letter them."""
    values = list(values)
    rng.shuffle(values)
    return {LETTERS[i]: v for i, v in enumerate(values)}, LETTERS[values.index(gold)]


# ---------- numbers ----------

NUMBER = re.compile(r"^[-+]?(\d{1,3}(,\d{3})+|\d+)?(\.\d+)?$")


def parse_number(text) -> float | None:
    """'75,287' -> 75287.0; None unless the whole string is a plain number."""
    s = str(text).strip().replace("−", "-")
    if not s or not any(ch.isdigit() for ch in s) or not NUMBER.match(s):
        return None
    return float(s.replace(",", ""))


def decimals_of(text: str) -> int:
    s = str(text).strip()
    return len(s.split(".", 1)[1]) if "." in s else 0


def format_number(value: float, decimals: int, commas: bool) -> str:
    return f"{value:,.{decimals}f}" if commas else f"{value:.{decimals}f}"


def near_numbers(value: float, decimals: int, rng, n: int = 3, nonneg: bool = False, year: bool = False) -> list[float]:
    """n numbers that differ from value at its displayed precision. Steps are 1 for small
    integers and years, else about 4% of the value; how many lie below the value is drawn at
    random, so the gold is not always the middle option."""
    unit = 10.0 ** -decimals
    if year or (decimals == 0 and abs(value) < 25):
        step = 1.0
    else:
        step = max(unit, round(abs(value) * 0.04 / unit) * unit)
    below = [round(value - m * step, decimals) for m in range(1, 7)]
    above = [round(value + m * step, decimals) for m in range(1, 7)]
    if nonneg:
        below = [v for v in below if v >= 0]
    k = rng.randint(max(0, n - len(above)), min(n, len(below)))
    return rng.sample(below, k) + rng.sample(above, n - k)


# ---------- WikiTableQuestions ----------

def wtq_normalize(x: str) -> str:
    """String normalisation of the official WikiTableQuestions evaluator (v1.0.2), ported to Python 3."""
    x = "".join(c for c in unicodedata.normalize("NFKD", x) if unicodedata.category(c) != "Mn")
    x = re.sub(r"[‘’´`]", "'", x)
    x = re.sub(r"[“”]", '"', x)
    x = re.sub(r"[‐‑‒–—−]", "-", x)
    while True:
        old = x
        x = re.sub(r"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())
        x = re.sub(r"(?<!^)( \([^)]*\))*$", "", x.strip())
        x = re.sub(r'^"([^"]*)"$', r"\1", x.strip())
        if x == old:
            break
    if x and x[-1] == ".":
        x = x[:-1]
    return re.sub(r"\s+", " ", x).lower().strip()


def tsv_unescape(s: str) -> str:
    """WTQ TSV escapes: \\n newline, \\p pipe, \\\\ backslash."""
    out, i = [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            out.append({"n": "\n", "p": "|", "\\": "\\"}.get(s[i + 1], s[i + 1]))
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def tsv_rows(text: str) -> list[dict[str, str]]:
    lines = [line for line in text.split("\n") if line.strip()]
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:]]


def pick_distractors(pool: list[str], gold: str, question: str, rng, n: int = 3) -> list[str]:
    """Values named in the question first (for 'was X or Y ...?'), then values with or without
    digits like the gold, then the rest; random within each group."""
    q = wtq_normalize(question)
    digits = any(ch.isdigit() for ch in gold)

    def group(value: str) -> int:
        v = wtq_normalize(value)
        if len(v) >= 3 and re.search(r"(?<!\w)" + re.escape(v) + r"(?!\w)", q):
            return 0
        return 1 if any(ch.isdigit() for ch in value) == digits else 2

    shuffled = list(pool)
    rng.shuffle(shuffled)
    return sorted(shuffled, key=group)[:n]


# Questions that accept any one of several answers ("name a team that ...") are dropped:
# another value from the same column could then also be correct.
NAME_ONE = re.compile(r"\b(name (a|an|one|any|some)|one of the|give (a|an|one)|an example|list (a|an|one)|tell me (a|an|one))\b", re.I)


@register_task
class WikiTableQuestions(TaskAdapter):
    name = f"{LEAF}/wikitablequestions"
    leaf = LEAF
    source = "WikiTableQuestions, pristine-unseen-tables test set (Pasupat and Liang, 2015)"
    url = "https://github.com/ppasupat/WikiTableQuestions"
    license = "CC BY-SA 4.0"
    tier = "C"
    criteria_source = ("Written; the correct option is the dataset's answer, distractors are other values of the answer's "
                       "table column (different under the official evaluator's normalisation) or nearby numbers for counts")
    notes = ("Single-answer questions; four options: the answer and three other values of its table column, or nearby "
             "numbers (steps of 1, or about 4%) for counts and other computed numbers. List answers, answers with units and "
             "'name a/one ...' questions (several answers acceptable) are dropped; yes/no questions keep two options.")

    ZIP = "https://github.com/ppasupat/WikiTableQuestions/releases/download/v1.0.2/WikiTableQuestions-1.0.2-compact.zip"
    ROOT = "WikiTableQuestions/"

    def load(self):
        z = zipfile.ZipFile(fetch(self.ZIP, "wtq-1.0.2-compact.zip"))
        examples = tsv_rows(z.read(self.ROOT + "data/pristine-unseen-tables.tsv").decode("utf-8"))
        canon = {r["id"]: r for r in tsv_rows(z.read(self.ROOT + "tagged/data/pristine-unseen-tables.tagged").decode("utf-8"))}
        tables: dict[str, tuple] = {}
        out = []
        for ex in examples:
            if "|" in ex["targetValue"] or NAME_ONE.search(ex["utterance"]):
                continue
            path = ex["context"]
            if path not in tables:
                tables[path] = self.read_table(z, path)
            header, body, values = tables[path]
            record = self.make_record(ex, canon[ex["id"]], header, body, values)
            if record:
                out.append(record)
        return out

    def read_table(self, z: zipfile.ZipFile, path: str):
        """Header, body rows and the CoreNLP number/date values of every body cell."""
        rows = [[tsv_unescape(c) for c in line.split("\t")] for line in z.read(self.ROOT + path.replace(".csv", ".tsv")).decode("utf-8").split("\n") if line != ""]
        tagged = path.replace("csv/", "tagged/", 1).replace("-csv/", "-tagged/").replace(".csv", ".tagged")
        values = {}
        for cell in tsv_rows(z.read(self.ROOT + tagged).decode("utf-8")):
            if cell["row"] != "-1":
                values[(int(cell["row"]), int(cell["col"]))] = (cell.get("number", ""), cell.get("date", ""))
        return rows[0], rows[1:], values

    def make_record(self, ex, tagged, header, body, values):
        gold = tsv_unescape(ex["targetValue"]).strip()
        g = wtq_normalize(gold)
        if not g:
            return None
        base = {"id": ex["id"], "question": ex["utterance"], "gold": gold, "header": header, "rows": body}
        if g in ("yes", "no"):
            return dict(base, gold=g, mode="yes_no", distractors=["no" if g == "yes" else "yes"])
        gold_num = parse_number(gold)
        canon_num = parse_number(tagged["targetCanon"]) if tagged["targetCanonType"] == "number" else None
        canon_date = tagged["targetCanon"] if tagged["targetCanonType"] == "date" else None

        def same_as_gold(text: str, number: str, date: str) -> bool:
            if wtq_normalize(text) == g:
                return True
            values = {parse_number(text), parse_number(number)} - {None}
            if {gold_num, canon_num} & values:
                return True
            return bool(canon_date) and date == canon_date

        hit_cols = sorted({j for i, row in enumerate(body) for j, cell in enumerate(row) if wtq_normalize(cell) == g or (gold_num is not None and parse_number(cell) == gold_num)})
        pool: dict[str, str] = {}
        for j in hit_cols:
            for i, row in enumerate(body):
                if j >= len(row) or not row[j].strip() or not wtq_normalize(row[j]):
                    continue
                number, date = values.get((i, j), ("", ""))
                if gold_num is not None and parse_number(row[j]) is None:
                    continue  # a numeric answer only gets plain numbers as distractors
                if not same_as_gold(row[j], number, date):
                    pool.setdefault(wtq_normalize(row[j]), row[j].strip())
        if len(pool) >= 3:
            return dict(base, mode="table_value", distractors=sorted(pool.values()))
        if gold_num is not None:
            year = decimals_of(gold) == 0 and "," not in gold and 1000 <= gold_num <= 2100
            return dict(base, mode="year" if year else "number", number=gold_num, decimals=decimals_of(gold), commas="," in gold)
        return None  # a string that is not a table value, or a number with a unit

    def label_of(self, record):
        return record["mode"]

    def to_item(self, record, rng):
        gold = record["gold"]
        if record["mode"] in ("number", "year"):
            value = record["number"]
            others = near_numbers(value, record["decimals"], rng, nonneg=value >= 0, year=record["mode"] == "year")
            distractors = [format_number(v, record["decimals"], record["commas"]) for v in others]
            gold = format_number(value, record["decimals"], record["commas"])
        elif record["mode"] == "table_value":
            distractors = pick_distractors(record["distractors"], gold, record["question"], rng)
        else:
            distractors = record["distractors"]
        if len({gold, *distractors}) < 1 + len(distractors):
            return None
        options, answer = lettered([gold] + distractors, gold, rng)
        state = {"table": md_table(record["header"], record["rows"]), "question": record["question"]}
        question = Question(type="choice", instructions=ASK, criteria=options)
        return self.item(state, {"answer": question}, {"answer": Gold(label=answer)}, mode=record["mode"], original_id=record["id"])


# ---------- TabMWP ----------

@register_task
class TabMWP(TaskAdapter):
    name = f"{LEAF}/tabmwp"
    leaf = LEAF
    source = "TabMWP, multiple-choice questions of the test split (Lu et al., 2023)"
    url = "https://github.com/lupantech/PromptPG"
    license = "CC BY-NC-SA 4.0"
    tier = "B"
    criteria_source = "Questions and answer options verbatim from TabMWP; the one-line answer instruction is written"
    notes = "Grade 1-8 math word problems over small tables; only the dataset's multiple-choice questions (2-4 options) are used."

    URL = "https://raw.githubusercontent.com/lupantech/PromptPG/main/data/tabmwp/problems_test.json"

    def load(self):
        with fetch(self.URL, "tabmwp_problems_test.json").open(encoding="utf-8") as f:
            problems = json.load(f)
        return [dict(p, pid=pid) for pid, p in sorted(problems.items(), key=lambda kv: int(kv[0]))
                if p["ques_type"] == "multi_choice" and p["answer"] in (p["choices"] or [])]

    def label_of(self, record):
        return record["ans_type"]

    def to_item(self, record, rng):
        choices = list(dict.fromkeys(record["choices"]))  # a few problems repeat an option
        options, answer = lettered(choices, record["answer"], rng)
        table = (record["table_title"] + "\n" if record.get("table_title") else "") + record["table"]
        question = Question(type="choice", instructions=ASK, criteria=options)
        state = {"table": table, "question": record["question"]}
        return self.item(state, {"answer": question}, {"answer": Gold(label=answer)}, ans_type=record["ans_type"], grade=record["grade"], original_id=record["pid"])


# ---------- TAT-QA ----------

SCALES = ("thousand", "million", "billion")


def tat_number(text) -> float | None:
    """'$ 1,452.4' -> 1452.4, '(3.1)' -> -3.1, '17.7%' -> 17.7; None for other text."""
    s = str(text).strip().replace("$", "").replace("%", "").replace(" ", "").replace("−", "-")
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace(",", "")
    try:
        value = float(s)
    except ValueError:
        return None
    return -value if negative else value


def cell_kind(text: str) -> str:
    """Distractors must look like the gold: a year, a percentage, an amount of money, another number, or text."""
    text = text.strip()
    if re.fullmatch(r"(19|20)\d\d", text):
        return "year"
    if tat_number(text) is None:
        return "text"
    return "percent" if "%" in text else "money" if "$" in text else "number"


def with_scale(text: str, scale: str) -> str:
    """Attach the answer's scale to numeric options: '685' -> '685 million', '17.7' -> '17.7%'."""
    if tat_number(text) is None:
        return text
    if scale in SCALES:
        return f"{text} {scale}"
    if scale == "percent" and not text.rstrip().endswith("%") and "$" not in text:
        return f"{text}%"
    return text


@register_task
class TATQA(TaskAdapter):
    name = f"{LEAF}/tatqa"
    leaf = LEAF
    source = "TAT-QA, test set with released gold answers (Zhu et al., 2021)"
    url = "https://github.com/NExTplusplus/TAT-QA"
    license = "CC BY 4.0"
    tier = "C"
    criteria_source = ("Written; the correct option is the dataset's answer with its scale, distractors are other cells of the "
                       "answer's row or column, or nearby numbers for arithmetic and counting answers")
    notes = ("Financial-report tables with their paragraphs. Span answers get other cells of the same kind from the answer's "
             "row or column as distractors; arithmetic and counting answers get nearby numbers (steps of about 4%, or 1) at the "
             "dataset's rounding. Questions answered from the text alone and multi-span answers are dropped.")

    URL = "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_test_gold.json"

    def load(self):
        with fetch(self.URL, "tatqa_dataset_test_gold.json").open(encoding="utf-8") as f:
            docs = json.load(f)
        out = []
        for doc in docs:
            table = doc["table"]["table"]
            paragraphs = [p["text"] for p in sorted(doc["paragraphs"], key=lambda p: p["order"])]
            for q in doc["questions"]:
                record = self.make_record(q, table)
                if record:
                    out.append(dict(record, id=q["uid"], question=q["question"], table=table, paragraphs=paragraphs, scale=q["scale"]))
        return out

    def make_record(self, q, table):
        kind, source, answer, scale = q["answer_type"], q["answer_from"], q["answer"], q["scale"]
        if source == "text" or kind == "multi-span":
            return None
        if kind in ("arithmetic", "count"):
            value = tat_number(answer)
            if value is None:
                return None
            text = str(answer)
            return {"mode": kind, "number": value, "decimals": decimals_of(text) if kind == "arithmetic" else 0}
        gold = str(answer[0]).strip()  # span
        gold_value = tat_number(gold)
        if scale and gold_value is None:
            return None  # a text answer annotated with a numeric scale

        def same(cell: str) -> bool:
            return cell.strip() == gold or (gold_value is not None and tat_number(cell) == gold_value)

        hits = [(i, j) for i, row in enumerate(table) for j, cell in enumerate(row) if same(cell)]
        if not hits:
            return None
        kind = cell_kind(gold)
        pool: dict[str, str] = {}
        for i, j in hits:
            if i == 0:  # a header cell (e.g. a year): other header cells
                cells = [(0, jj) for jj in range(1, len(table[0]))]
            elif j == 0:  # a row label: other row labels
                cells = [(ii, 0) for ii in range(1, len(table))]
            else:  # a value: other values of its row and column
                cells = [(i, jj) for jj in range(1, len(table[i]))] + [(ii, j) for ii in range(1, len(table)) if j < len(table[ii])]
            for ii, jj in cells:
                text = table[ii][jj].strip()
                if text and any(ch.isalnum() for ch in text) and not same(text) and cell_kind(text) == kind:
                    pool.setdefault(" ".join(text.lower().split()), text)
        if not pool:
            return None
        return {"mode": "table_value", "gold": gold, "distractors": sorted(pool.values())}

    def label_of(self, record):
        return f"{record['mode']}|{record['scale'] or 'none'}"

    def to_item(self, record, rng):
        scale = record["scale"]
        if record["mode"] == "table_value":
            others = rng.sample(record["distractors"], min(3, len(record["distractors"])))
            texts = [with_scale(t, scale) for t in [record["gold"]] + others]
        else:
            value, decimals = record["number"], record["decimals"]
            others = near_numbers(value, decimals, rng, nonneg=record["mode"] == "count" or value >= 0)
            texts = [with_scale(format_number(v, decimals, abs(v) >= 1000), scale) for v in [value] + others]
        if len(set(texts)) < len(texts):
            return None
        options, answer = lettered(texts, texts[0], rng)
        rows = record["table"]
        state = {"table": md_table(rows[0], rows[1:]), "paragraphs": "\n\n".join(record["paragraphs"]), "question": record["question"]}
        question = Question(type="choice", instructions="Which option answers the question, based on the table and the paragraphs?", criteria=options)
        return self.item(state, {"answer": question}, {"answer": Gold(label=answer)}, mode=record["mode"], scale=scale or "none", original_id=record["id"])
