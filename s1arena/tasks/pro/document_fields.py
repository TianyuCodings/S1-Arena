"""Leaf: document fields (票据 / 表单字段抽取): pick each field's value from a document's own OCR text.

Receipts (SROIE, CORD) and TV political-ad invoices (VRDU ad-buy forms), mirroring an
invoice-processing workflow. For every annotated field the options are all values of the
same kind found in the document (amounts, dates, identifiers or header lines) and the gold
is the one the annotators marked; on the invoices a stated advertiser is also checked.
Candidate extraction is deterministic; an independent checker re-derives every gold value.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import re
from decimal import Decimal, InvalidOperation

from ...env import load_env
from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import fetch

LEAF = "pro/document_fields"


def option_id(i: int) -> str:
    """A..Z, then AA, AB, ..."""
    return chr(65 + i) if i < 26 else chr(64 + i // 26) + chr(65 + i % 26)


def lettered(values: dict[str, str], rng) -> tuple[dict[str, str], dict[str, str]]:
    """Shuffle candidate values (key -> text as printed); return criteria (id -> text) and key -> id."""
    keys = list(values)
    rng.shuffle(keys)
    ids = {k: option_id(i) for i, k in enumerate(keys)}
    return {ids[k]: values[k] for k in keys}, ids


def hf_file(repo: str, filename: str, revision: str):
    load_env()
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset", revision=revision)


# ---------- SROIE: Malaysian receipts, key fields company / date / total ----------

SROIE_AMOUNT = re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+|\d+)\.(\d{2})(?!\d)")
SROIE_DATE = re.compile(
    r"(?<!\d)(\d{1,2}[/.-]\d{1,2}[/.-](?:\d{4}|\d{2})|\d{4}[/.-]\d{1,2}[/.-]\d{1,2}"
    r"|\d{1,2}[ -]?(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*[ ,-]*(?:\d{4}|\d{2}))(?!\d)",
    re.I,
)
HEADER_LINES = 6


def squash(text: str) -> str:
    return " ".join(str(text).upper().split())


def sroie_total_key(text: str) -> str | None:
    if "-" in (text or ""):  # negative totals (refunds) are left out
        return None
    t = re.sub(r"[^\d.]", "", text or "")
    if re.fullmatch(r"\d+\.\d{2}", t):
        whole, cents = t.split(".")
        return f"{int(whole)}.{cents}"
    return f"{int(t)}.00" if re.fullmatch(r"\d+", t) else None


def sroie_fields(lines: list[str], entities: dict) -> dict:
    """Candidates and gold for each usable field; a field is left out when its annotated value is not found unambiguously."""
    out = {}
    amounts: dict[str, str] = {}
    for line in lines:
        for m in SROIE_AMOUNT.finditer(line):
            amounts.setdefault(f"{int(m.group(1).replace(',', ''))}.{m.group(2)}", m.group(0))
    total = sroie_total_key(entities.get("total") or "")
    if total in amounts and len(amounts) >= 2:
        out["total"] = {"options": amounts, "gold": total}
    dates: dict[str, str] = {}
    for line in lines:
        for m in SROIE_DATE.finditer(line):
            dates.setdefault(m.group(1), m.group(1))
    if (entities.get("date") or "") in dates and len(dates) >= 2:
        out["date"] = {"options": dates, "gold": entities["date"]}
    header: dict[str, str] = {}
    for line in lines[:HEADER_LINES]:
        if squash(line):
            header.setdefault(squash(line), line.strip())
    company, keys = squash(entities.get("company") or ""), list(header)
    matches = [
        k for i, k in enumerate(keys)
        if company and (k == company or k.startswith(company + " ") or k.startswith(company + "(") or (i + 1 < len(keys) and f"{k} {keys[i + 1]}" == company))
    ]
    if len(matches) == 1 and len(header) >= 2:
        out["company"] = {"options": header, "gold": matches[0]}
    return out


SROIE_QUESTIONS = {
    "company": "Which line of this receipt gives the name of the company that issued it (if the name runs over two lines, its first line)?",
    "date": "Which date is the date of this receipt?",
    "total": "Which amount is the total amount of this receipt?",
}


@register_task
class SROIE(TaskAdapter):
    name = f"{LEAF}/sroie"
    leaf = LEAF
    source = "SROIE, ICDAR 2019 Scanned Receipts OCR and Information Extraction, test set with the 14 restored annotations (Huang et al., 2019)"
    url = "https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE"
    license = "CC BY 4.0"
    tier = "C"
    criteria_source = "Written from the SROIE key-field names (company, date, total); options are strings found in the receipt's own OCR lines"
    notes = (
        "The state is the receipt's ground-truth OCR text lines. Options: every amount with two decimals (total), every date "
        "string (date, only when a receipt has two or more), the first six lines (company). A field is asked only when its "
        "annotated value is found exactly once among the options; the address field is not asked (it is often not recoverable from the lines)."
    )
    REVISION = "bffe40c26759f3376ec2b3ae9031dbba54cd587c"

    def load(self):
        import pandas as pd

        path = hf_file("jsdnrs/ICDAR2019-SROIE", "data/test-00000-of-00001.parquet", self.REVISION)
        rows = []
        for r in pd.read_parquet(path, columns=["key", "entities", "words"]).to_dict("records"):
            lines = [str(w) for w in r["words"]]
            fields = sroie_fields(lines, dict(r["entities"]))
            if fields:
                rows.append({"key": r["key"], "lines": lines, "fields": fields})
        return rows

    def label_of(self, record):
        return "+".join(sorted(record["fields"]))

    def to_item(self, record, rng):
        questions, gold = {}, {}
        for field in ("company", "date", "total"):
            if field in record["fields"]:
                spec = record["fields"][field]
                criteria, ids = lettered(spec["options"], rng)
                questions[field] = Question(type="choice", instructions=SROIE_QUESTIONS[field], criteria=criteria)
                gold[field] = Gold(label=ids[spec["gold"]])
        return self.item({"receipt_text": "\n".join(record["lines"])}, questions, gold, original_id=record["key"])


# ---------- CORD v2: Indonesian receipts, subtotal and total amounts ----------

# CORD class definitions (github.com/clovaai/cord, "Class Definition"), verbatim descriptions.
CORD_FIELDS = {
    "sub_total.subtotal_price": "subtotal price",
    "sub_total.discount_price": "discounted price in total",
    "sub_total.service_price": "service charge",
    "sub_total.othersvc_price": "added charge other than service charge",
    "sub_total.tax_price": "tax amount",
    "total.total_price": "total price",
    "total.cashprice": "amount of price paid in cash",
    "total.changeprice": "amount of change in cash",
    "total.creditcardprice": "amount of price paid in credit/debit card",
    "total.emoneyprice": "amount of price paid in emoney, point",
}
CORD_PRICES = set(CORD_FIELDS) | {
    "menu.price", "menu.unitprice", "menu.discountprice", "menu.itemsubtotal", "menu.sub_unitprice", "menu.sub_price",
    "void_menu.price", "sub_total.etc", "total.total_etc",
}


def cord_amount_key(text: str) -> str | None:
    """Digits with sign ('-60000' for '-60.000'); None if the text is not an amount."""
    t = re.sub(r"(?i)\b(rp|idr)\.?", "", text).strip().strip("@*:[] ")
    if t.endswith("-") and not t.startswith("-"):  # '60.000,-' style
        t = re.sub(r"[.,]?-$", "", t).strip()
    if not re.fullmatch(r"[-(]?\s*[\d.,]*\d[\d., ]*\s*\)?", t):
        return None
    digits = re.sub(r"\D", "", t).lstrip("0") or "0"
    return ("-" if t.startswith(("-", "(")) and digits != "0" else "") + digits


def cord_document(ground_truth: dict) -> tuple[str, dict]:
    """Receipt text rebuilt line by line from the annotated words, plus candidate amounts and gold keys per field."""
    rows: dict[int, list] = {}
    for line in ground_truth["valid_line"]:
        for w in line["words"]:
            rows.setdefault(w["row_id"], []).append(w)
    ordered = sorted(rows.values(), key=lambda ws: (min(w["quad"]["y1"] for w in ws), min(w["quad"]["x1"] for w in ws)))
    text = "\n".join(" ".join(w["text"] for w in sorted(ws, key=lambda w: w["quad"]["x1"])) for ws in ordered)
    amounts: dict[str, str] = {}
    by_field: dict[str, set] = {}
    lines = sorted((l for l in ground_truth["valid_line"] if l["words"]), key=lambda l: min((w["quad"]["y1"], w["quad"]["x1"]) for w in l["words"]))
    for line in lines:
        if line["category"] not in CORD_PRICES:
            continue
        value = " ".join(w["text"] for w in line["words"] if not w["is_key"]).strip()
        key = cord_amount_key(value) if value else None
        if key is None:
            continue
        amounts.setdefault(key, value)
        if line["category"] in CORD_FIELDS:
            by_field.setdefault(line["category"], set()).add(key)
    fields = {f: next(iter(keys)) for f, keys in by_field.items() if len(keys) == 1}
    return text, {"options": amounts, "fields": fields}


@register_task
class CORD(TaskAdapter):
    name = f"{LEAF}/cord"
    leaf = LEAF
    source = "CORD v2 Consolidated Receipt Dataset, test and validation sets (Park et al., 2019)"
    url = "https://huggingface.co/datasets/naver-clova-ix/cord-v2"
    license = "CC BY 4.0"
    tier = "B"
    criteria_source = "Field descriptions verbatim from the CORD class definitions (subtotal and total classes), in a written question"
    notes = (
        "Receipt text rebuilt line by line from the annotated OCR words (store information is not part of the public CORD). "
        "Options are all distinct amounts in the receipt (menu prices, subtotals, totals, payments), shown as printed. Test "
        "and validation sets are both used (100 receipts each) because the test set alone is too small."
    )
    REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"
    FILES = ("data/test-00000-of-00001-9c204eb3f4e11791.parquet", "data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet")

    def load(self):
        import pandas as pd

        rows, seen = [], set()
        for filename in self.FILES:
            for r in pd.read_parquet(hf_file("naver-clova-ix/cord-v2", filename, self.REVISION), columns=["ground_truth"]).to_dict("records"):
                gt = json.loads(r["ground_truth"])
                text, spec = cord_document(gt)
                if len(spec["options"]) >= 2 and spec["fields"] and text not in seen:  # a few receipts occur twice
                    seen.add(text)
                    rows.append({"split": gt["meta"]["split"], "image_id": gt["meta"]["image_id"], "text": text, **spec})
        return rows

    def label_of(self, record):
        return f"{len(record['fields'])}"

    def to_item(self, record, rng):
        criteria, ids = lettered(record["options"], rng)
        questions, gold = {}, {}
        for field, description in CORD_FIELDS.items():
            if field in record["fields"]:
                name = field.split(".")[1]
                questions[name] = Question(type="choice", instructions=f"Which amount on this receipt is the {description}?", criteria=criteria)
                gold[name] = Gold(label=ids[record["fields"][field]])
        meta = {"split": record["split"], "original_id": f"{record['split']}:{record['image_id']}"}
        return self.item({"receipt_text": record["text"]}, questions, gold, **meta)


# ---------- VRDU ad-buy forms: TV political-ad invoices and contracts (DeepForm) ----------

VRDU_AMOUNT = re.compile(r"(?<![\d.,])(-?\$?\s?\d{1,3}(?:,\d{3})+(?:\.\d{2})?|-?\$?\s?\d+\.\d{2})(?!\d)")
VRDU_NUMERIC_DATE = re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})(?![\d/])")
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
VRDU_TEXT_DATE = re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)\.? (\d{1,2}),? (\d{4})\b")
VRDU_DMY_DATE = re.compile(r"(?<![\w-])(\d{1,2})-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-(\d{4}|\d{2})(?![\w-])", re.I)
TOKEN = re.compile(r"\S+")
STRIP = "#:;,.()[]{}\"'*"


def year4(y: str) -> int:
    y = int(y)
    return y if y >= 100 else (2000 + y if y < 50 else 1900 + y)


def safe_date(y: int, m: int, d: int) -> str | None:
    try:
        return dt.date(y, m, d).isoformat()
    except ValueError:
        return None


def vrdu_candidates(text: str) -> dict[str, list]:
    """(key, text as printed, start, end) of every amount, date and identifier-like token in the OCR text."""
    out = {"amount": [], "date": [], "id": []}
    for m in VRDU_AMOUNT.finditer(text):
        try:
            key = str(Decimal(re.sub(r"[$,\s]", "", m.group(1))).quantize(Decimal("0.01")))
        except InvalidOperation:
            continue
        out["amount"].append((key, m.group(1).strip(), m.start(1), m.end(1)))
    for m in VRDU_NUMERIC_DATE.finditer(text):
        key = safe_date(year4(m.group(3)), int(m.group(1)), int(m.group(2)))  # US forms: month/day/year
        if key:
            out["date"].append((key, m.group(0), m.start(), m.end()))
    for m in VRDU_TEXT_DATE.finditer(text):
        key = safe_date(int(m.group(3)), MONTHS.index(m.group(1)[:3].lower()) + 1, int(m.group(2)))
        if key:
            out["date"].append((key, m.group(0), m.start(), m.end()))
    for m in VRDU_DMY_DATE.finditer(text):
        key = safe_date(year4(m.group(3)), MONTHS.index(m.group(2).lower()) + 1, int(m.group(1)))
        if key:
            out["date"].append((key, m.group(0), m.start(), m.end()))
    out["date"].sort(key=lambda c: c[2])
    for m in TOKEN.finditer(text):
        raw = m.group(0)
        token = raw.strip(STRIP)
        if sum(ch.isdigit() for ch in token) < 4 or any(c in token for c in "/$:") or VRDU_AMOUNT.fullmatch(token):
            continue
        start = m.start() + raw.index(token)
        out["id"].append((token.upper(), token, start, start + len(token)))
    return out


VRDU_FIELDS = {  # question name -> (annotated field, candidate kind, question); wording adapted from the DeepForm field list
    "contract_number": ("contract_num", "id", "Which value is the contract number of this ad buy? (Several documents can share a contract number when a contract is revised.)"),
    "flight_start_date": ("flight_from", "date", "On which date does the flight start, i.e. the first air date of the ads (the start of the 'flight dates')?"),
    "flight_end_date": ("flight_to", "date", "On which date does the flight end, i.e. the last air date of the ads (the end of the 'flight dates')?"),
    "gross_amount": ("gross_amount", "amount", "Which amount is the gross amount, i.e. the total amount paid for the ads?"),
}
NEGATIVE_SOURCES = ("agency", "product", "property")


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def vrdu_document(doc: dict) -> dict | None:
    text = doc["ocr"]["text"]
    cands = vrdu_candidates(text)
    spans: dict[str, list] = {}
    values: dict[str, list] = {}
    for ann in doc["annotations"]:
        if isinstance(ann[0], str):  # line items (repeated nested fields) are not used
            for value, _box, segments in ann[1]:
                spans.setdefault(ann[0], []).extend(tuple(s) for s in segments)
                values.setdefault(ann[0], []).append(" ".join(value.split()))
    fields = {}
    for question, (field, kind, _) in VRDU_FIELDS.items():
        if field not in spans:
            continue
        hit = {key for key, _, start, end in cands[kind] for s, e in spans[field] if start < e and s < end}
        options: dict[str, str] = {}
        for key, printed, _, _ in cands[kind]:
            options.setdefault(key, printed)
        if len(hit) == 1 and len(options) >= 2:
            fields[question] = {"options": options, "gold": hit.pop()}
    advertiser = [v for v in values.get("advertiser", []) if v]
    check = None
    if advertiser:
        taken = set().union(*(words(v) for v in advertiser))
        negatives = [v for f in NEGATIVE_SOURCES for v in values.get(f, [])[:1] if v and words(v) and not (words(v) & taken)]
        check = {"positive": advertiser[0], "negative": negatives[0] if negatives else None}
    if not fields and not check:
        return None
    return {"filename": doc["filename"], "text": text, "pages": len(doc["ocr"]["pages"]), "fields": fields, "advertiser": check}


@register_task
class VRDUAdBuy(TaskAdapter):
    name = f"{LEAF}/vrdu_ad_buy"
    leaf = LEAF
    source = "VRDU ad-buy forms (DeepForm political TV-ad invoices and contracts from FCC public files), mixed-template test split (Wang et al., 2023)"
    url = "https://github.com/google-research-datasets/vrdu"
    license = "Not stated (documents are US public records from the FCC)"
    tier = "B"
    criteria_source = "Adapted from the DeepForm field list (contract number, advertiser, flight dates, total amount paid for the ads)"
    notes = (
        "The state is the OCR text shipped with VRDU (multi-page; line items included). Gold values come from the annotators' "
        "boxes: the amount, date or identifier in the OCR text that the box covers. Options are all amounts, dates (read as "
        "month/day/year) or identifier-like tokens (4+ digits, not dates, times or amounts) in the document. The advertiser check states the annotated "
        "advertiser or, for about half the documents, another annotated party (agency, product or station) that shares no word with it."
    )
    BASE = "https://raw.githubusercontent.com/google-research-datasets/vrdu/04a4df7ac5cc694e51c48115c03fe428f0e240cc/ad-buy-form/"
    SPLIT = "few_shot-splits/DeepForm-mixed_template-train_10-test_300-valid_100-SD_0.json"

    def load(self):
        test = set(json.loads(fetch(self.BASE + self.SPLIT, "vrdu_adbuy_mixed_sd0_split.json").read_text(encoding="utf-8"))["test"])
        rows = []
        with gzip.open(fetch(self.BASE + "main/dataset.jsonl.gz", "vrdu_adbuy_dataset.jsonl.gz"), "rt", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                if doc["filename"] in test:
                    record = vrdu_document(doc)
                    if record:
                        rows.append(record)
        return rows

    def label_of(self, record):
        return "1 page" if record["pages"] == 1 else "2-3 pages" if record["pages"] <= 3 else "4+ pages"

    def to_item(self, record, rng):
        questions, gold = {}, {}
        for question, (_, _, text) in VRDU_FIELDS.items():
            if question in record["fields"]:
                spec = record["fields"][question]
                criteria, ids = lettered(spec["options"], rng)
                questions[question] = Question(type="choice", instructions=text, criteria=criteria)
                gold[question] = Gold(label=ids[spec["gold"]])
        check = record["advertiser"]
        if check:
            negative = rng.random() < 0.5
            if not negative or check["negative"]:
                stated = check["negative"] if negative else check["positive"]
                questions["advertiser_check"] = Question(
                    type="boolean",
                    instructions=f"Is \"{stated}\" the advertiser named on this form? The advertiser is the party buying the ads "
                    "(often the name of a political committee, but not always).",
                )
                gold["advertiser_check"] = Gold(label="false" if negative else "true")
        if not questions:
            return None
        return self.item({"document_text": record["text"]}, questions, gold, original_id=record["filename"], pages=record["pages"])
