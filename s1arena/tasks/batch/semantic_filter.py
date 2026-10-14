"""Leaf: semantic filter (语义过滤): does this row satisfy a condition stated in natural language?

Rows are records with several fields. Each source has several conditions; every row is paired
with one of them (chosen per row), and the gold comes from a column of the dataset that is
withheld from the row. Sampling is balanced over (condition, answer), so every condition
appears about equally often and is true for half of its rows.
"""

from __future__ import annotations

import random

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import hf_rows

LEAF = "batch/semantic_filter"
FILTER = "Does this row satisfy the condition? Condition: {condition}"


def assign(conditions: dict, row_key: str, task: str, row: dict):
    """Pick one condition for a row (deterministically) and evaluate it; None when it does not apply."""
    names = sorted(conditions)
    name = random.Random(f"{task}:{row_key}").choice(names)
    value = conditions[name][1](row)
    return None if value is None else (name, value)


CLOTHING = {
    "recommends": ("The reviewer recommends the product.", lambda r: r["recommended_ind"] == 1),
    "five_stars": ("The reviewer rated the product 5 out of 5 stars.", lambda r: r["rating"] == 5),
    "one_or_two_stars": ("The reviewer rated the product 1 or 2 out of 5 stars.", lambda r: r["rating"] <= 2),
    "dress": ("The reviewed product is a dress.", lambda r: r["class_name"] == "Dresses"),
    "bottoms": ("The reviewed product is a bottom: pants, jeans, a skirt or shorts.", lambda r: r["department_name"] == "Bottoms"),
    "tops": ("The reviewed product is a top: a knit top, blouse or sweater.", lambda r: r["department_name"] == "Tops"),
    "jacket": ("The reviewed product is a jacket or other outerwear.", lambda r: r["department_name"] == "Jackets"),
    "intimates": ("The reviewed product is loungewear, sleepwear, swimwear, legwear or other intimate apparel.",
                  lambda r: r["department_name"] == "Intimate"),
}


@register_task
class ClothingReviewsFilter(TaskAdapter):
    name = f"{LEAF}/clothing_reviews"
    leaf = LEAF
    source = "Women's E-Commerce Clothing Reviews, 23,486 reviews from an online retailer (nicapotato, Kaggle)"
    url = "https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews"
    license = "CC0-1.0"
    tier = "C"
    criteria_source = "Conditions written from the dataset's column definitions (Recommended IND, Rating, Department Name, Class Name); gold is the withheld column"
    sampling = "balanced"
    notes = (
        "Eight conditions: the reviewer's own recommendation flag and star rating, and the retailer's department and class "
        "of the product. The row shows the review title and text, the reviewer's age and the helpful-vote count; rating, "
        "recommendation, product ids and categories are withheld. Read from the Hugging Face repack "
        "chibifire/kaggle-womens-ecom-clothing-reviews. Balanced over (condition, answer) by design."
    )

    def load(self):
        records = []
        for r in hf_rows("chibifire/kaggle-womens-ecom-clothing-reviews", None, "train"):
            if not (r["review_text"] or "").strip() or r["department_name"] is None:
                continue
            picked = assign(CLOTHING, str(r["row_id"]), self.name, r)
            if picked:
                records.append({"row": r, "condition": picked[0], "label": picked[1]})
        return records

    def label_of(self, record):
        return f"{record['condition']}|{record['label']}"

    def to_item(self, record, rng):
        r = record["row"]
        state = {"review_title": (r["title"] or "").strip(), "review_text": r["review_text"].strip(),
                 "reviewer_age": str(r["age"]), "positive_feedback_count": str(r["positive_feedback_count"])}
        question = Question(type="boolean", instructions=FILTER.format(condition=CLOTHING[record["condition"]][0]))
        gold = Gold(label="true" if record["label"] else "false")
        return self.item(state, {"satisfies": question}, {"satisfies": gold}, condition=record["condition"], original_id=r["row_id"])


DEGREE_OR_HIGHER = {"Bachelor's Degree", "Master's Degree", "Doctorate"}
BELOW_DEGREE = {"High School or equivalent", "Some High School Coursework", "Some College Coursework Completed",
                "Associate Degree", "Certification", "Vocational", "Vocational - HS Diploma", "Vocational - Degree"}


def among(value, yes: set, no: set):
    return True if value in yes else False if value in no else None


JOBS = {
    "full_time": ("The position is full-time.",
                  lambda r: among(r["employment_type"], {"Full-time"}, {"Part-time", "Contract", "Temporary"})),
    "bachelor_or_higher": ("The position requires at least a bachelor's degree.",
                           lambda r: among(r["required_education"], DEGREE_OR_HIGHER, BELOW_DEGREE)),
    "entry_level": ("The position is for someone at the start of their career: an internship or an entry-level job.",
                    lambda r: among(r["required_experience"], {"Internship", "Entry level"}, {"Associate", "Mid-Senior level", "Director", "Executive"})),
    "remote": ("The position can be done remotely (telecommuting).", lambda r: r["telecommuting"] == 1),
    "sales": ("The job is a sales role.", lambda r: None if not r["function"] or r["function"] == "Other" else r["function"] == "Sales"),
    "fraudulent": ("The posting is fraudulent: a fake job ad posted by scammers.", lambda r: r["fraudulent"] == 1),
}
JOB_FIELDS = ("title", "location", "department", "salary_range", "company_profile", "description", "requirements", "benefits")


@register_task
class JobPostingsFilter(TaskAdapter):
    name = f"{LEAF}/job_postings"
    leaf = LEAF
    source = "EMSCAD, Employment Scam Aegean Dataset: 17,880 job ads from Workable, 866 of them fraudulent (Vidros et al., 2017)"
    url = "https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction"
    license = "CC0-1.0 (Kaggle release)"
    tier = "C"
    criteria_source = "Conditions written from the dataset's structured fields (employment type, required education and experience, telecommuting, function) and its fraud label; gold is the withheld field"
    sampling = "balanced"
    notes = (
        "Six conditions. The structured fields were filled in by the employers; the fraud label comes from Workable's "
        "specialists. The row shows the ad's text fields and whether it has a company logo; the field a condition is about "
        "is withheld, as are all other structured fields. Rows whose field is empty or 'Other'/'Unspecified' are not asked "
        "that condition. Read from the Hugging Face copy victor/real-or-fake-fake-jobposting-prediction. Balanced over "
        "(condition, answer) by design."
    )

    def load(self):
        records = []
        for r in hf_rows("victor/real-or-fake-fake-jobposting-prediction", None, "train"):
            picked = assign(JOBS, str(r["job_id"]), self.name, r)
            if picked:
                records.append({"row": r, "condition": picked[0], "label": picked[1]})
        return records

    def label_of(self, record):
        return f"{record['condition']}|{record['label']}"

    def to_item(self, record, rng):
        r = record["row"]
        state = {field: str(r[field]).strip() for field in JOB_FIELDS if r.get(field) and str(r[field]).strip()}
        state["has_company_logo"] = "yes" if r["has_company_logo"] == 1 else "no"
        question = Question(type="boolean", instructions=FILTER.format(condition=JOBS[record["condition"]][0]))
        gold = Gold(label="true" if record["label"] else "false")
        return self.item(state, {"satisfies": question}, {"satisfies": gold}, condition=record["condition"], original_id=r["job_id"])
