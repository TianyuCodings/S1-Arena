"""Leaf: semantic grouping (语义分组): assign each row to one of several categories defined in natural language.

Gold is the category recorded in the source: the complaint reason chosen by crowd contributors
for airline tweets, and the catalogue product type of Amazon listings (a few near-identical
types are merged into one group).
"""

from __future__ import annotations

import gzip
import json
import tarfile

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import fetch, hf_rows

LEAF = "batch/semantic_grouping"
GROUP = "Assign the row to exactly one of these categories: {what}"

AIRLINE_REASONS = {
    "Customer Service Issue": ("customer_service", "Customer service: unhelpful, rude or unreachable staff, long hold times, no reply to requests."),
    "Late Flight": ("late_flight", "Late flight: the flight is delayed or arrived late."),
    "Cancelled Flight": ("cancelled_flight", "Cancelled flight: the flight was cancelled, including the rebooking that followed."),
    "Lost Luggage": ("lost_luggage", "Lost luggage: checked bags are missing, delayed or were sent to the wrong place."),
    "Damaged Luggage": ("damaged_luggage", "Damaged luggage: bags or their contents were damaged."),
    "Bad Flight": ("bad_flight", "Bad flight: a poor experience on board, e.g. seats, cabin, food, wifi, entertainment or cleanliness."),
    "Flight Booking Problems": ("booking_problems", "Booking problems: trouble booking, changing or paying for a reservation, fares, the website or app."),
    "Flight Attendant Complaints": ("flight_attendants", "Flight attendants: complaints about the cabin crew."),
    "longlines": ("long_lines", "Long lines: long queues at check-in, security, the gate or the service desk."),
}


@register_task
class AirlineComplaints(TaskAdapter):
    name = f"{LEAF}/airline_complaints"
    leaf = LEAF
    source = "Twitter US Airline Sentiment, negative tweets with the complaint reason (Figure Eight / CrowdFlower Data for Everyone, 2015)"
    url = "https://huggingface.co/datasets/osanseviero/twitter-airline-sentiment"
    license = "CC BY-NC-SA 4.0"
    tier = "C"
    criteria_source = "Category descriptions written from the nine complaint-reason names used by the crowd contributors"
    notes = (
        "Negative tweets from February 2015 about six US airlines; contributors first rated sentiment, then chose the "
        "reason. Only tweets whose reason has a contributor-agreement confidence of at least 0.6 and is not 'Can't Tell'. "
        "Only the final label is released, so it is a hard label."
    )

    def load(self):
        rows = hf_rows("osanseviero/twitter-airline-sentiment", None, "train")
        return [r for r in rows if r["airline_sentiment"] == "negative" and r["negativereason"] in AIRLINE_REASONS
                and float(r["negativereason_confidence"] or 0) >= 0.6 and (r["text"] or "").strip()]

    def label_of(self, record):
        return AIRLINE_REASONS[record["negativereason"]][0]

    def to_item(self, record, rng):
        options = dict(AIRLINE_REASONS.values())
        question = Question(type="choice", instructions=GROUP.format(what="what is the customer complaining about?"),
                            criteria=shuffled_options(options, rng))
        state = {"airline": record["airline"], "tweet": record["text"].strip(), "posted_at": record["tweet_created"],
                 "user_timezone": record["user_timezone"] or ""}
        gold = Gold(label=self.label_of(record))
        return self.item(state, {"category": question}, {"category": gold}, original_id=str(record["tweet_id"]),
                         confidence=float(record["negativereason_confidence"]))


ABO_URL = "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar"
# group id -> (ABO product types, description); a few near-identical types are merged
ABO_GROUPS = {
    "footwear": (("SHOES", "BOOT", "SANDAL"), "Footwear: shoes, boots, sandals or slippers."),
    "chair": (("CHAIR",), "Chair: a seat for one person with a backrest, e.g. dining, office, accent or lounge chairs and recliners."),
    "sofa": (("SOFA",), "Sofa, couch, loveseat or sectional."),
    "ottoman": (("OTTOMAN",), "Ottoman or pouf."),
    "stool": (("STOOL_SEATING",), "Stool, e.g. bar, counter or vanity stools."),
    "bench": (("BENCH",), "Bench."),
    "table": (("TABLE",), "Table, e.g. dining, coffee, side, console or accent tables."),
    "desk": (("DESK",), "Desk."),
    "bed": (("BED", "BED_FRAME"), "Bed or bed frame."),
    "headboard": (("HEADBOARD",), "Headboard."),
    "shelf": (("SHELF",), "Shelf, bookcase or shelving unit."),
    "cabinet": (("CABINET",), "Cabinet, e.g. storage, accent or bathroom cabinets and sideboards."),
    "dresser": (("DRESSER",), "Dresser or chest of drawers."),
    "rug": (("RUG",), "Rug or mat."),
    "wall_art": (("WALL_ART",), "Wall art, e.g. framed prints, canvases and wall decor."),
    "pillow": (("PILLOW",), "Pillow or cushion, including decorative throw pillows."),
    "planter": (("PLANTER",), "Planter or plant pot."),
    "mirror": (("HOME_MIRROR",), "Mirror."),
    "lamp": (("LAMP",), "Lamp that stands on a table, desk or the floor."),
    "light_fixture": (("LIGHT_FIXTURE",), "Light fixture mounted on a ceiling or wall, e.g. pendants, chandeliers, flush mounts, sconces."),
    "light_bulb": (("LIGHT_BULB",), "Light bulb."),
    "ring": (("FINERING", "RING"), "Ring (fine or fashion jewelry)."),
    "earrings": (("FINEEARRING", "EARRING"), "Earrings (fine or fashion jewelry)."),
    "necklace_bracelet": (("FINENECKLACEBRACELETANKLET", "NECKLACE", "BRACELET"), "Necklace, pendant, bracelet or anklet (fine or fashion jewelry)."),
    "hat": (("HAT",), "Hat or cap."),
    "handbag": (("HANDBAG",), "Handbag, purse or tote."),
    "luggage": (("SUITCASE", "LUGGAGE"), "Suitcase or other luggage."),
    "coffee": (("COFFEE",), "Coffee (beans, ground coffee or pods)."),
    "tea": (("TEA",), "Tea."),
    "phone_case": (("CELLULAR_PHONE_CASE",), "Mobile phone case or cover."),
}
TYPE_TO_GROUP = {t: g for g, (types, _) in ABO_GROUPS.items() for t in types}


def english(values) -> list[str]:
    return [v["value"].strip() for v in values or [] if str(v.get("language_tag", "")).startswith("en_") and str(v.get("value", "")).strip()]


@register_task
class ABOProductTypes(TaskAdapter):
    name = f"{LEAF}/abo_product_types"
    leaf = LEAF
    source = "Amazon Berkeley Objects (ABO) product listings, English listings of amazon.com, .co.uk and .ca (Collins et al., CVPR 2022)"
    url = "https://amazon-berkeley-objects.s3.amazonaws.com/index.html"
    license = "CC BY 4.0"
    tier = "C"
    criteria_source = "Category descriptions written from ABO product-type names; gold is the listing's catalogue product type"
    sampling = "balanced"
    notes = (
        "30 groups built from 38 specific ABO product types; types that the catalogue does not keep apart reliably are "
        "merged (shoes, boots and sandals: about a quarter of SHOES listings are named boots or sandals; fine and fashion "
        "rings, earrings and necklaces; suitcase and luggage; bed and bed frame). Catch-all types such as HOME or GROCERY "
        "are not used. The row shows the listing's English name, brand, bullet points, colour, material and style; the "
        "product type, browse node and keywords are withheld. Balanced over groups so that every category is tested "
        "(footwear and chairs make up almost half of the catalogue)."
    )

    def load(self):
        records = []
        with tarfile.open(fetch(ABO_URL, "abo-listings.tar")) as tar:
            for member in sorted(tar.getnames()):
                if not member.endswith(".json.gz"):
                    continue
                with gzip.open(tar.extractfile(member)) as f:
                    for line in f:
                        r = json.loads(line)
                        if r.get("domain_name") not in ("amazon.com", "amazon.co.uk", "amazon.ca") or not r.get("product_type"):
                            continue
                        group = TYPE_TO_GROUP.get(r["product_type"][0]["value"])
                        name = english(r.get("item_name"))
                        if group and name:
                            records.append({"item_id": r["item_id"], "domain": r["domain_name"], "group": group, "name": name[0],
                                            "brand": english(r.get("brand")), "bullets": english(r.get("bullet_point")),
                                            "color": english(r.get("color")), "material": english(r.get("material")),
                                            "style": english(r.get("style"))})
        return records

    def label_of(self, record):
        return record["group"]

    def to_item(self, record, rng):
        state = {"product_name": record["name"]}
        for field, key in (("brand", "brand"), ("features", "bullets"), ("color", "color"), ("material", "material"), ("style", "style")):
            if record[key]:
                state[field] = "\n".join(record[key]) if key == "bullets" else "; ".join(dict.fromkeys(record[key]))
        options = {g: description for g, (_, description) in ABO_GROUPS.items()}
        question = Question(type="choice", instructions=GROUP.format(what="which kind of product is this listing?"),
                            criteria=shuffled_options(options, rng))
        return self.item(state, {"category": question}, {"category": Gold(label=record["group"])}, original_id=record["item_id"], marketplace=record["domain"])
