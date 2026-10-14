"""Leaf: semantic ranking (语义排序): score each row against a criterion on a graded scale; rows are then
ranked by score and evaluated with NDCG per criterion.

- Hotel reviews scored for one aspect (cleanliness, value, location, rooms, sleep quality or the
  hotel overall); gold is the reviewer's own 1-5 rating of that aspect.
- Clinical trials scored against a patient case; gold is the TREC 2022 Clinical Trials track
  judgment (eligible / excluded / not relevant).
"""

from __future__ import annotations

import random

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..code._hub import hub_columns
from ..common import hf_rows

LEAF = "batch/semantic_ranking"

ASPECTS = {
    "overall": "the hotel overall",
    "cleanliness": "the hotel's cleanliness",
    "value": "the value for money",
    "location": "the hotel's location",
    "rooms": "the rooms",
    "sleep_quality": "the quality of sleep",
}
# TripAdvisor's labels for its 1-5 bubble ratings.
STARS = ["1: Terrible", "2: Poor", "3: Average", "4: Very good", "5: Excellent"]


@register_task
class TripAdvisorAspects(TaskAdapter):
    name = f"{LEAF}/tripadvisor_aspects"
    rank_key = "meta:criterion"  # NDCG groups
    leaf = LEAF
    source = "TripAdvisor hotel reviews with the reviewers' aspect ratings, English reviews of 2012 (Li et al., 2013; processed by Niimi, 2024)"
    url = "https://huggingface.co/datasets/jniimi/tripadvisor-review-rating"
    license = "Apache-2.0 (Hugging Face release); original crawl by Li, Ott and Cardie (2013), terms not stated"
    tier = "B"
    criteria_source = "Criteria written from the rating fields; the five levels are TripAdvisor's own rating labels (Terrible to Excellent)"
    sampling = "balanced"
    notes = (
        "Six criteria: the reviewer's ratings of the hotel overall, cleanliness, value, location, rooms and sleep quality; "
        "each review is asked about one of them (chosen per review). The ratings themselves are withheld. Balanced over "
        "(criterion, rating) so that every rating level is represented for ranking. Only the first 20 row groups "
        "(20,000 reviews) of the Hugging Face parquet file are read."
    )

    def load(self):
        frame = hub_columns(
            "jniimi/tripadvisor-review-rating", ["data/train-00000-of-00001.parquet"],
            ["hotel_id", "user_id", "title", "text", "stay_year", "post_date", *ASPECTS], "tripadvisor_reviews_head", row_groups=20,
        )
        records = []
        for r in frame.to_dict("records"):
            if not str(r["text"] or "").strip():
                continue
            key = f"{r['hotel_id']}:{r['user_id']}:{r['post_date']}"
            aspect = random.Random(f"{self.name}:{key}").choice(sorted(ASPECTS))
            rating = r[aspect]
            if rating != rating or rating not in (1, 2, 3, 4, 5):  # NaN or out of range
                continue
            records.append({"key": key, "title": str(r["title"] or "").strip(), "text": str(r["text"]).strip(),
                            "stay_year": str(r["stay_year"]), "aspect": aspect, "rating": int(rating)})
        return records

    def label_of(self, record):
        return f"{record['aspect']}|{record['rating']}"

    def to_item(self, record, rng):
        question = Question(
            type="score",
            instructions=f"Score this hotel review against the criterion: how satisfied was the reviewer with {ASPECTS[record['aspect']]}? "
            "Use the reviewer's 1-5 rating scale.",
            criteria=STARS,
        )
        state = {"review_title": record["title"], "review_text": record["text"], "year_of_stay": record["stay_year"]}
        gold = Gold(label=str(record["rating"] - 1))
        return self.item(state, {"rating": question}, {"rating": gold}, criterion=record["aspect"], original_id=record["key"])


# TREC 2022 Clinical Trials track relevance definitions (trec-cds.org/2022.html), verbatim, ordered by gain.
TRIAL_LEVELS = [
    "Not relevant: patient does not have sufficient information to qualify for the trial.",
    "Excluded: patient meets inclusion criteria, but is excluded on the grounds of the trial's exclusion criteria.",
    "Eligible: patient meets inclusion criteria and exclusion criteria do not apply.",
]
TRIAL_LABELS = {"not relevant": 0, "possibly relevant": 1, "definitely relevant": 2}  # names used by the HF copy for 0/1/2


@register_task
class TRECClinicalTrials(TaskAdapter):
    name = f"{LEAF}/trec_clinical_trials"
    rank_key = "meta:topic"  # NDCG groups
    leaf = LEAF
    source = "TREC 2022 Clinical Trials track: synthetic patient cases and ClinicalTrials.gov records with NIST judgments (Roberts et al., 2022)"
    url = "https://huggingface.co/datasets/araag2/TREC_Clinical-Trials"
    license = "CC BY-SA 4.0 (Hugging Face copy); trial records from ClinicalTrials.gov (public domain)"
    tier = "A"
    criteria_source = "Relevance definitions (eligible / excluded / not relevant) verbatim from the TREC 2022 Clinical Trials track guidelines"
    sampling = "balanced"
    notes = (
        "Criterion: one of the 50 patient cases (admission-note style). Row: a trial's title, brief summary and eligibility "
        "criteria. Levels are ordered by TREC's gains (not relevant 0, excluded 1, eligible 2). Balanced over the three "
        "levels; 80% of the judged pairs are not relevant."
    )

    def load(self):
        rows = hf_rows("araag2/TREC_Clinical-Trials", "2022_processed", "test",
                       data_files={"test": "2022_processed/test-00000-of-00001.parquet"}, verification_mode="no_checks")
        return [r for r in rows if r["Label"] in TRIAL_LABELS and (r["Eligibility"] or "").strip() and (r["Patient"] or "").strip()]

    def label_of(self, record):
        return record["Label"]

    def to_item(self, record, rng):
        question = Question(
            type="score",
            instructions="Score this clinical trial against the criterion: is it a match for the following patient?\n\nPatient: "
            + record["Patient"].strip(),
            criteria=TRIAL_LEVELS,
        )
        state = {"trial_id": record["doc_id"], "title": (record["Title"] or record["title"] or "").strip(),
                 "brief_summary": (record["Summary"] or "").strip(), "eligibility_criteria": record["Eligibility"].strip()}
        gold = Gold(label=str(TRIAL_LABELS[record["Label"]]))
        return self.item(state, {"match": question}, {"match": gold}, topic=record["query_id"], criterion=f"patient {record['query_id']}")
