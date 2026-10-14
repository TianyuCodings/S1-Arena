"""Leaf: e-commerce (电商): query-product relations and product categories.

ESCI (Amazon) and WANDS (Wayfair) ask how a product relates to a shopping query, with
each dataset's own label definitions; WDC-25 asks for the category of a product offer
from a web shop.
"""

from __future__ import annotations

import gzip
import hashlib
import html
import json
import re

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import fetch

LEAF = "pro/ecommerce"


def strip_html(text: str | None) -> str:
    text = re.sub(r"<br\s*/?>|</p>|</li>", "\n", text or "", flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return re.sub(r"\s*\n\s*", "\n", text).strip()


# ---------- Amazon Shopping Queries (ESCI) ----------

# Reddy et al. (2022), Section 1, verbatim.
ESCI_OPTIONS = {
    "exact": "Exact (E): the item is relevant for the query, and satisfies all the query specifications (e.g., a water bottle "
    "matching all attributes of a query “plastic water bottle 24oz”, such as material and size)",
    "substitute": "Substitute (S): the item is somewhat relevant, i.e., it fails to fulfill some aspects of the query but the "
    "item can be used as a functional substitute (e.g., fleece for a “sweater” query)",
    "complement": "Complement (C): the item does not fulfill the query, but could be used in combination with an exact item "
    "(e.g., track pants for “running shoes” query)",
    "irrelevant": "Irrelevant (I): the item is irrelevant, or it fails to fulfill a central aspect of the query (e.g., socks for "
    "a “telescope” query, or a wheat flour bread for a “gluten–free bread” query)",
}
ESCI_LABELS = {"E": "exact", "S": "substitute", "C": "complement", "I": "irrelevant"}
ESCI_BASE = "https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset/"


@register_task
class ESCI(TaskAdapter):
    name = f"{LEAF}/esci"
    leaf = LEAF
    source = "Shopping Queries Dataset (ESCI), English test split (Reddy et al., 2022)"
    url = "https://github.com/amazon-science/esci-data"
    license = "Apache-2.0"
    tier = "A"
    criteria_source = "The four ESCI class definitions verbatim from Reddy et al. (2022), Section 1"
    sampling = "balanced"
    notes = ("US locale, test split (large version). Sampled evenly across E/S/C/I because the leaf tests the four relation "
             "definitions; the natural test mix (65% E, 22% S, 10% I, 2.6% C) would leave only a handful of complements. "
             "Queries are first thinned to a fixed 2% by a hash of the query id (keeps loading light). Labels are majority "
             "votes of at least three trained annotators, who saw the amazon.com detail page; the paper estimates 91% "
             "label accuracy. HTML is stripped from descriptions.")

    def load(self):
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        examples = pq.read_table(fetch(ESCI_BASE + "shopping_queries_dataset_examples.parquet", "esci_examples.parquet"),
                                 columns=["example_id", "query", "query_id", "product_id", "product_locale", "esci_label", "large_version", "split"])
        examples = examples.filter(pc.and_(pc.equal(examples["split"], "test"), pc.equal(examples["product_locale"], "us")))
        rows = [r for r in examples.to_pylist() if r["large_version"] == 1 and self.keep_query(r["query_id"])]
        ids = pa.array(sorted({r["product_id"] for r in rows}))
        products = pq.read_table(fetch(ESCI_BASE + "shopping_queries_dataset_products.parquet", "esci_products.parquet"),
                                 filters=[("product_locale", "=", "us"), ("product_id", "in", ids)])
        self.products = {p["product_id"]: p for p in products.to_pylist()}
        return [dict(example_id=r["example_id"], query=r["query"], product_id=r["product_id"], label=r["esci_label"])
                for r in rows if r["product_id"] in self.products and r["esci_label"] in ESCI_LABELS]

    @staticmethod
    def keep_query(query_id: int) -> bool:
        return int(hashlib.sha1(str(query_id).encode()).hexdigest(), 16) % 50 == 0

    def label_of(self, record):
        return record["label"]

    def to_item(self, record, rng):
        p = self.products[record["product_id"]]
        state = {"query": record["query"]}
        for key, field in (("product_title", "product_title"), ("brand", "product_brand"), ("color", "product_color"),
                           ("bullet_points", "product_bullet_point"), ("description", "product_description")):
            value = strip_html(p.get(field)) if field == "product_description" else (p.get(field) or "").strip()
            if value:
                state[key] = value
        question = Question(
            type="choice",
            instructions="A customer searched an online store with this query and the product below was among the results. "
            "Which ESCI relevance class does the product have for the query?",
            criteria=shuffled_options(ESCI_OPTIONS, rng),
        )
        gold = Gold(label=ESCI_LABELS[record["label"]])
        return self.item(state, {"relevance": question}, {"relevance": gold}, original_id=record["example_id"])


# ---------- WANDS (Wayfair) ----------

# WANDS annotation guidelines ("Relevance Annotations"), verbatim; ordered from least to most relevant.
WANDS_LEVELS = [
    "Irrelevant: this label indicates the product is not relevant to the query.",
    "Partial match: this label represents the surfaced product that does not fully match the search query. It only matches "
    "the target entity of the query, but does not satisfy the modifiers for the query.",
    "Exact match: this label represents the surfaced product fully matches the search query.",
]
WANDS_LABELS = {"Irrelevant": "0", "Partial": "1", "Exact": "2"}
WANDS_BASE = "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/"


@register_task
class WANDS(TaskAdapter):
    name = f"{LEAF}/wands"
    rank_key = "state:query"  # NDCG groups
    leaf = LEAF
    source = "WANDS, Wayfair product search relevance judgements (Chen et al., 2022)"
    url = "https://github.com/wayfair/WANDS"
    license = "MIT"
    tier = "A"
    criteria_source = ("Level definitions and the misspelling note verbatim from the WANDS annotation guidelines "
                       "(Product Search Relevance Annotation Guidelines.pdf); 'not' in the Irrelevant definition is set in bold there")
    notes = ("WANDS has no train/test split; it is an evaluation set (480 queries, 233,448 judgements). The model sees the "
             "product name, class, category path, description and attributes; annotators saw the product page with images.")

    def load(self):
        import pandas as pd

        products = pd.read_csv(fetch(WANDS_BASE + "product.csv", "wands_product.csv"), sep="\t", dtype=str, keep_default_na=False)
        queries = pd.read_csv(fetch(WANDS_BASE + "query.csv", "wands_query.csv"), sep="\t", dtype=str, keep_default_na=False)
        labels = pd.read_csv(fetch(WANDS_BASE + "label.csv", "wands_label.csv"), sep="\t", dtype=str, keep_default_na=False)
        self.products = {r["product_id"]: r for r in products.to_dict("records")}
        self.queries = {r["query_id"]: r["query"] for r in queries.to_dict("records")}
        return [dict(id=r["id"], query_id=r["query_id"], product_id=r["product_id"], label=r["label"])
                for r in labels.to_dict("records") if r["label"] in WANDS_LABELS and r["product_id"] in self.products]

    def label_of(self, record):
        return record["label"]

    def to_item(self, record, rng):
        p = self.products[record["product_id"]]
        features = [f.strip() for f in p["product_features"].split("|") if f.strip()]
        state = {"query": self.queries[record["query_id"]], "product_name": p["product_name"], "product_class": p["product_class"],
                 "category_path": p["category hierarchy"], "description": p["product_description"], "attributes": "\n".join(features)}
        state = {k: v.strip() for k, v in state.items() if v and v.strip()}
        question = Question(
            type="score",
            instructions="Rate the relevance of this Wayfair product to the customer's search query. For any misspelled "
            "query/products, please ignore the mis-spellings and proceed to annotate the example as-is.",
            criteria=WANDS_LEVELS,
        )
        gold = Gold(label=WANDS_LABELS[record["label"]])
        return self.item(state, {"relevance": question}, {"relevance": gold}, original_id=record["id"])


# ---------- WDC-25 product categorization ----------

WDC_BASE = "https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/"
WDC_CATEGORIES = [
    "Automotive", "Baby", "Books", "CDs_and_Vinyl", "Camera_and_Photo", "Cell_Phones_and_Accessories", "Clothing",
    "Computers_and_Accessories", "Grocery_and_Gourmet_Food", "Health_and_Beauty", "Home_and_Garden", "Jewelry",
    "Luggage_and_Travel_Gear", "Movies_and_TV", "Musical_Instruments", "Office_Products", "Other_Electronics", "Others",
    "Pet_Supplies", "Shoes", "Sports_and_Outdoors", "Tools_and_Home_Improvement", "Toys_and_Games", "Video_Games",
]


def wdc_option(category: str) -> str:
    if category == "Others":
        return "Others: the product fits none of the other categories"
    if category == "Other_Electronics":
        return "Other electronics: electronics not covered by the camera, cell phone, computer or video game categories"
    return category.replace("_and_", " and ").replace("_", " ")


def wdc_value(raw: str) -> str:
    """'[ wd my cloud nas server 6 tb ]' -> 'wd my cloud nas server 6 tb'; drops null placeholders."""
    text = re.sub(r'"@\w+$', "", raw.strip()).strip('"')
    text = text.strip().removeprefix("[").removesuffix("]")
    parts = [p.strip() for p in text.split(" , ") if p.strip() and p.strip().lower() not in ("null", "none")]
    return re.sub(r"\s+", " ", " ; ".join(parts)).strip()


@register_task
class WDCCategories(TaskAdapter):
    name = f"{LEAF}/wdc25_categories"
    leaf = LEAF
    source = "WDC-25 Gold Standard for Product Categorization, test clusters (Web Data Commons, 2019)"
    url = "http://webdatacommons.org/categorization/"
    license = "Not stated (Web Data Commons; offers extracted from the Common Crawl)"
    tier = "C"
    criteria_source = "Written from the 25 WDC-25 category names; 'Others' and 'Other electronics' glosses written"
    notes = ("Manually labelled offers from many e-shops (schema.org annotations). One offer per test cluster (the one with the "
             "most text), clusters labelled 'not found' excluded. Text is lower-cased and stripped of punctuation as "
             "distributed. URLs, prices and images are withheld (URLs often name the category).")

    FIELDS = ("/name", "/title", "/brand", "/manufacturer", "/description")

    def load(self):
        test = json.load(gzip.open(fetch(WDC_BASE + "categories_clusters_testing.json.gzip", "wdc25_categories_clusters_testing.json.gz"), "rt", encoding="utf-8"))
        offers = json.load(gzip.open(fetch(WDC_BASE + "categories_gold_standard_offers.json.gzip", "wdc25_categories_gold_standard_offers.json.gz"), "rt", encoding="utf-8"))
        clusters = {str(c["cluster_id"]) for c in test}
        best: dict[str, dict] = {}
        for offer in offers:
            cluster = str(offer["cluster_id"])
            if cluster not in clusters or offer["categoryLabel"] not in WDC_CATEGORIES:
                continue
            record = self.render(offer)
            if not record["offer"]:
                continue
            size = len(record["offer"]) + len(record["page_title"])
            if cluster not in best or (size, record["offer_id"]) > (best[cluster]["size"], best[cluster]["offer_id"]):
                best[cluster] = dict(record, size=size, cluster_id=cluster, label=offer["categoryLabel"])
        return [best[c] for c in sorted(best)]

    def render(self, offer: dict) -> dict:
        own = {k: wdc_value(v) for p in offer["schema.org_properties"] for k, v in p.items() if k in self.FIELDS}
        parent = {k: wdc_value(v) for p in (offer.get("parent_schema.org_properties") or []) for k, v in p.items() if k in self.FIELDS}
        lines = [f"{k.strip('/')}: {v}" for k, v in own.items() if v]
        page_title = parent.get("/title") or parent.get("/name") or ""
        return {"offer": "\n".join(lines), "page_title": page_title, "offer_id": offer["nodeID"]}

    def label_of(self, record):
        return record["label"]

    def to_item(self, record, rng):
        options = shuffled_options({c: wdc_option(c) for c in WDC_CATEGORIES}, rng)
        question = Question(type="choice", instructions="Which product category does this product offer from a web shop belong to?", criteria=options)
        state = {"offer": record["offer"]}
        if record["page_title"]:
            state["web_page_title"] = record["page_title"]
        return self.item(state, {"category": question}, {"category": Gold(label=record["label"])}, cluster_id=record["cluster_id"])
