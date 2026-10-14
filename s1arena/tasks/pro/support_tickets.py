"""Leaf: support tickets (客服 / 工单): routing to a queue and ticket severity.

CFPB consumer complaints and Banking77 customer messages are routed to a product or
topic; Mozilla Bugzilla defect reports get the severity their triage owner assigned.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re

from ...env import load_env
from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import fetch, humanize

LEAF = "pro/support_tickets"


def hash_keep(key: str, modulus: int) -> bool:
    """Deterministic thinning by a hash of the record id (independent of the other records)."""
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % modulus == 0


# ---------- CFPB consumer complaints: route to a product ----------

# CFPB product taxonomy in use in 2022, with each product's sub-products as they appear in the data.
CFPB_PRODUCTS = {
    "Checking or savings account": "Checking account; savings account; CD (Certificate of Deposit); other banking product or service",
    "Credit card or prepaid card": "General-purpose credit card or charge card; store credit card; general-purpose prepaid card; "
    "government benefit card; gift card; payroll card; student prepaid card",
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting; credit repair services; "
    "other personal consumer report",
    "Debt collection": "Credit card debt; medical debt; auto debt; payday loan debt; mortgage debt; private or federal student "
    "loan debt; other debt; debt of unknown type",
    "Money transfer, virtual currency, or money service": "Mobile or digital wallet; domestic (US) or international money "
    "transfer; virtual currency; check cashing service; traveler's check or cashier's check; debt settlement; foreign "
    "currency exchange; refund anticipation check; money order",
    "Mortgage": "Conventional home mortgage; FHA mortgage; VA mortgage; home equity loan or line of credit (HELOC); reverse "
    "mortgage; other type of mortgage",
    "Payday loan, title loan, or personal loan": "Installment loan; personal line of credit; payday loan; title loan",
    "Student loan": "Federal student loan servicing; private student loan",
    "Vehicle loan or lease": "Loan; lease",
}
CFPB_IDS = {
    "Checking or savings account": "checking_or_savings",
    "Credit card or prepaid card": "credit_or_prepaid_card",
    "Credit reporting, credit repair services, or other personal consumer reports": "credit_reporting",
    "Debt collection": "debt_collection",
    "Money transfer, virtual currency, or money service": "money_transfer",
    "Mortgage": "mortgage",
    "Payday loan, title loan, or personal loan": "payday_title_personal_loan",
    "Student loan": "student_loan",
    "Vehicle loan or lease": "vehicle_loan_or_lease",
}
CFPB_REPO = "BEE-spoke-data/consumer-finance-complaints"
CFPB_REVISION = "088cc7308d4afc2a880f2329d08e2f7a09188ec6"


@register_task
class CFPBComplaints(TaskAdapter):
    name = f"{LEAF}/cfpb_complaints"
    leaf = LEAF
    source = "CFPB Consumer Complaint Database, narratives of complaints received in 2022 (via a Hugging Face snapshot)"
    url = "https://huggingface.co/datasets/BEE-spoke-data/consumer-finance-complaints"
    license = "CC0 1.0 (US government public data)"
    tier = "B"
    criteria_source = ("Product names verbatim from the CFPB complaint taxonomy used in 2022; each option lists that product's "
                       "CFPB sub-products; question written")
    sampling = "balanced"
    notes = ("Gold is the product the consumer selected when filing, which routes the complaint. The CFPB stopped publishing "
             "narratives in its live database, so a snapshot is used (file has-text/train-00000-of-00003 at a pinned revision), "
             "thinned to a fixed 10% by a hash of the complaint id; duplicate narratives removed. Sampled evenly across the "
             "nine products because the 2022 stream is 72% credit reporting.")

    def load(self):
        load_env()
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(CFPB_REPO, "has-text/train-00000-of-00003.parquet", repo_type="dataset", revision=CFPB_REVISION)
        table = pq.read_table(path, columns=["Date received", "Product", "Consumer complaint narrative", "Complaint ID"])
        table = table.filter(pc.starts_with(table["Date received"], "2022"))
        seen, rows = set(), []
        for r in sorted(table.to_pylist(), key=lambda r: r["Complaint ID"]):
            text = (r["Consumer complaint narrative"] or "").strip()
            if r["Product"] not in CFPB_IDS or not text or text in seen or not hash_keep(str(r["Complaint ID"]), 10):
                continue
            seen.add(text)
            rows.append({"id": r["Complaint ID"], "product": r["Product"], "narrative": text})
        return rows

    def label_of(self, record):
        return record["product"]

    def to_item(self, record, rng):
        options = shuffled_options({CFPB_IDS[p]: f"{p} (for example: {subs})" for p, subs in CFPB_PRODUCTS.items()}, rng)
        question = Question(
            type="choice",
            instructions="A consumer submitted this complaint to the U.S. Consumer Financial Protection Bureau. Which product "
            "is the complaint about? The product decides where the complaint is routed.",
            criteria=options,
        )
        gold = Gold(label=CFPB_IDS[record["product"]])
        return self.item({"complaint": record["narrative"]}, {"product": question}, {"product": gold}, original_id=record["id"])


# ---------- Banking77: route a customer message to a support topic ----------

B77_BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"


@register_task
class Banking77(TaskAdapter):
    name = f"{LEAF}/banking77"
    leaf = LEAF
    source = "BANKING77, online banking customer queries, test split (Casanueva et al., 2020)"
    url = "https://github.com/PolyAI-LDN/task-specific-datasets"
    license = "CC BY 4.0"
    tier = "C"
    criteria_source = "Written from the 77 intent names"
    notes = "Single-domain, fine-grained routing: many intents differ only in the step of a process (e.g. pending vs failed top-up)."

    def load(self):
        self.intents = json.loads(fetch(B77_BASE + "categories.json", "banking77_categories.json").read_text(encoding="utf-8"))
        with fetch(B77_BASE + "test.csv", "banking77_test.csv").open(encoding="utf-8") as f:
            return [r for r in csv.DictReader(f) if r["category"] in self.intents]

    def label_of(self, record):
        return record["category"]

    def to_item(self, record, rng):
        options = shuffled_options({i: humanize(i) for i in self.intents}, rng)
        question = Question(
            type="choice",
            instructions="A customer of an online bank sent this message to customer support. Which support topic should it be routed to?",
            criteria=options,
        )
        return self.item({"message": record["text"].strip()}, {"topic": question}, {"topic": Gold(label=record["category"])})


# ---------- Mozilla Bugzilla: defect severity ----------

# BMO user guide, "Severity" field (wiki.mozilla.org/BMO/UserGuide/BugFields), verbatim; least to most severe.
BMO_LEVELS = [
    "S4 (Small/Trivial) minor significance, cosmetic issues, low or no impact to users",
    "S3 (Normal) Blocks non-critical functionality or a work around exists",
    "S2 (Serious) Major functionality/product severely impaired or a high impact issue and a satisfactory workaround does not exist",
    "S1 (Catastrophic) Blocks development/testing, may impact more than 25% of users, causes data loss, likely dot release "
    "driver, and no workaround available",
]
BMO_SEVERITY = {"S4": "0", "S3": "1", "S2": "2", "S1": "3"}
BMO_QUERY = (
    "https://bugzilla.mozilla.org/rest/bug?product=Firefox&product=Core&product=DevTools&product=Toolkit&product=Fenix"
    "&product=Thunderbird&type=defect&resolution=FIXED&severity=S1&severity=S2&severity=S3&severity=S4"
    "&f1=creation_ts&o1=greaterthaneq&v1=2024-01-01&f2=creation_ts&o2=lessthan&v2=2024-07-01"
    "&f3=delta_ts&o3=lessthan&v3=2026-01-01"  # not changed since 2026-01-01, so the recorded severity is frozen
    "&include_fields=id,summary,severity,component,product,creation_time,keywords,creator,description&limit=10000"
)
BMO_SKIP_KEYWORDS = {"intermittent-failure", "intermittent-testcase", "test-verify-fail", "perf-alert"}
# Reports that state a severity ("setting it as S3", "access-S2", "high severity", "sec-high") would leak the label.
# A token followed by ':' is left alone: "[s3:gecko-generated-sources:...]" appears in stack traces.
BMO_LEAK = re.compile(r"(?<![\w/.])s[1-4](?![\w:])|\bsev(erity)?\b|\bsec-(critical|high|moderate|low)\b", re.I)


@register_task
class MozillaBugSeverity(TaskAdapter):
    name = f"{LEAF}/mozilla_bug_severity"
    leaf = LEAF
    source = "Mozilla Bugzilla defect reports filed January-June 2024 and later fixed (bugzilla.mozilla.org)"
    url = "https://bugzilla.mozilla.org"
    license = "Not stated (public bug reports; Mozilla Websites & Communications Terms of Use)"
    tier = "A"
    criteria_source = ("Severity field description and S1-S4 level definitions verbatim from the BMO user guide "
                       "(wiki.mozilla.org/BMO/UserGuide/BugFields)")
    notes = ("Gold is the severity recorded on the bug, which component triage owners set (the guide says reporters should not). "
             "The model sees the product, component, summary and the reporter's first comment; triagers may also have used "
             "later discussion. Bugs filed by bots, intermittent or perf-alert bugs, and reports that mention a severity "
             "(S1-S4, 'severity', sec-high etc.) are excluded. The pool is one live REST query cached on first build; only bugs "
             "last changed before 2026-01-01 are included, so their severity can no longer change, but a later rebuild can "
             "lose bugs that were touched since.")

    def load(self):
        data = json.loads(fetch(BMO_QUERY, "bmo_fixed_defects_2024h1_unchanged_since_2026.json").read_text(encoding="utf-8"))
        rows = []
        for bug in data["bugs"]:
            creator = bug.get("creator", "")
            if bug.get("severity") not in BMO_SEVERITY or not (bug.get("description") or "").strip():
                continue
            if creator.endswith(".bugs") or "bot" in creator.split("@")[0] or BMO_SKIP_KEYWORDS & set(bug.get("keywords") or []):
                continue
            if BMO_LEAK.search(bug["summary"]) or BMO_LEAK.search(bug["description"]):
                continue
            rows.append({k: bug[k] for k in ("id", "product", "component", "summary", "severity", "description")})
        return sorted(rows, key=lambda r: r["id"])

    def label_of(self, record):
        return record["severity"]

    def to_item(self, record, rng):
        question = Question(
            type="score",
            instructions="Severity: This field describes the impact of a bug. Which severity should the component's triage "
            "owner give this bug report?",
            criteria=BMO_LEVELS,
        )
        state = {"product": record["product"], "component": record["component"], "summary": record["summary"],
                 "description": record["description"].strip()}
        gold = Gold(label=BMO_SEVERITY[record["severity"]])
        return self.item(state, {"severity": question}, {"severity": gold}, original_id=record["id"])
