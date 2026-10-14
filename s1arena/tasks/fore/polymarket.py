"""Polymarket events with several mutually exclusive outcomes, for fore/multi_outcome and fore/numeric_bins.

Events are read from Polymarket's public gamma API: closed events with scheduled end dates in
January-August 2026, at least $100k traded, in politics, world affairs, economy, business,
technology, science, culture, awards, weather and finance (sports and crypto left out). An
event qualifies when its markets are mutually exclusive (negRisk) and exactly one of them
resolved Yes. Only outcomes that were listed on the event's start date are offered, and the
event is skipped if the winner was added later. Events whose outcomes are numeric ranges
(ordered by Polymarket's own threshold) go to numeric_bins; the others to multi_outcome.

The state shows the event title, its resolution rules, the as-of date (the event's start
date) and the scheduled end date. Final prices, volumes and closing times are never shown.
Polymarket's terms reserve its content, so items are built locally and not redistributed.
"""

from __future__ import annotations

import calendar
import functools
import json
import re
from collections import defaultdict

from ...registry import register_task
from ...schema import MAX_LEVELS, Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import fetch, slug

API = "https://gamma-api.polymarket.com/events"
MONTHS = [(2026, m) for m in range(1, 9)]
TAGS = ["politics", "geopolitics", "world", "economy", "business", "tech", "science", "pop-culture", "awards", "weather", "finance"]
EXCLUDE = {"sports", "esports", "crypto", "crypto-prices", "games", "soccer", "basketball", "nba", "nfl", "mlb", "nhl", "tennis",
           "golf", "cricket", "ufc", "boxing", "f1", "bitcoin", "ethereum", "solana", "mentions"}
VOLUME_MIN = 100_000
PER_SERIES = 2  # recurring events (daily temperatures, weekly counts) are capped per series
MONTH_WORDS = r"January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
LICENSE = "Polymarket Terms of Use (content reserved by Polymarket); not redistributed"
URL = "https://docs.polymarket.com/developers/gamma-markets-api/overview"
CONTAMINATION = ("Events ended January-August 2026; a model whose training data postdates an event may know the outcome. "
                 "The events come from a live API, cached locally on first build; the market price on the as-of date is not included.")


def _pages():
    for year, month in MONTHS:
        lo, hi = f"{year}-{month:02d}-01", f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]}"
        for tag in TAGS:
            for offset in range(0, 2000, 100):
                url = f"{API}?closed=true&tag_slug={tag}&end_date_min={lo}&end_date_max={hi}&volume_min={VOLUME_MIN}&limit=100&offset={offset}"
                with open(fetch(url, f"polymarket_events_{tag}_{lo}_{offset}.json"), encoding="utf-8") as f:
                    page = json.load(f)
                yield page
                if len(page) < 100:
                    break


def _yes_price(market: dict) -> str | None:
    try:
        return str(json.loads(market.get("outcomePrices") or "[]")[0])
    except (ValueError, IndexError, TypeError):
        return None


SCALE = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}
NUMBER = re.compile(r"(?<![\w.])(-?\d[\d,]*(?:\.\d+)?)\s*([kmbt])?(?![a-z])", re.I)


def _value(label: str) -> float | None:
    """First number in a range label, with k/m/b/t suffixes; negative for decreases or cuts ('25 bps decrease')."""
    numbers = NUMBER.findall(label.replace("\u2212", "-").replace("\u2013", "-"))
    if not numbers:
        return None
    suffix = numbers[0][1] or next((s for _, s in numbers if s), "")  # '350-360m': the unit is written once, at the end
    value = float(numbers[0][0].replace(",", "")) * SCALE.get(suffix.lower(), 1.0)
    return -value if re.search(r"decrease|\bcut\b|\blower\b", label, re.I) else value


RANGE_WORDS = {"or", "below", "above", "less", "more", "higher", "lower", "fewer", "under", "over", "than", "between", "and", "to",
               "bps", "increase", "decrease", "change", "no", "before", "after", "on", "prior", "by", "st", "nd", "rd", "th", "et",
               "pt", "am", "pm", "s", "k", "m", "b", "t", "c", "f", "day", "days", "week", "weeks", "end", "of", "the", "at", "least"}


def _range_like(label: str) -> bool:
    """'70-71°F', '<$4,750', '25 bps decrease', 'March 15' are ranges; 'claude-opus-4-6' or 'Player 123' are names."""
    words = re.findall(r"[a-z]+", label.lower())
    months = {w.lower() for w in MONTH_WORDS.split("|")}
    return bool(re.search(r"\d", label)) and all(w in RANGE_WORDS or w in months for w in words)


def _parse(event: dict) -> dict | None:
    tags = {t.get("slug") for t in event.get("tags") or [] if isinstance(t, dict)}
    if tags & EXCLUDE or not event.get("negRisk"):
        return None
    as_of = str(event.get("startDate") or event.get("creationDate") or "")[:10]
    markets = [m for m in event.get("markets") or [] if float(m.get("volumeNum") or 0) > 0 and (m.get("groupItemTitle") or "").strip()]
    listed = [m for m in markets if str(m.get("createdAt") or "")[:10] <= as_of]
    winners = [m for m in markets if _yes_price(m) == "1" and m.get("umaResolutionStatus") == "resolved"]
    if len(winners) != 1 or winners[0] not in listed or len(listed) < 3 or len(listed) > 255:
        return None
    if any(_yes_price(m) not in ("0", "1") for m in listed):
        return None  # split or unresolved outcomes
    labels = [m["groupItemTitle"].strip() for m in listed]
    if len(set(labels)) != len(labels):
        return None
    try:
        thresholds = [float(m.get("groupItemThreshold")) for m in listed]
    except (TypeError, ValueError):
        thresholds = []
    ordered = False
    if len(set(thresholds)) == len(thresholds) == len(listed):
        by_threshold = [label for _, label in sorted(zip(thresholds, labels))]
        values = [_value(label) for label in by_threshold]
        known = [v for v in values if v is not None]
        # a range event: nearly every label carries a number, and the numbers rise with Polymarket's threshold order
        ranged = sum(_range_like(label) for label in by_threshold)
        if ranged >= len(labels) - 1 and len(known) >= len(labels) - 1 and all(a <= b for a, b in zip(known, known[1:])):
            labels, ordered = by_threshold, True
    series = re.sub(r"\s+", " ", re.sub(rf"\d[\d,.]*|\b(?:{MONTH_WORDS})\b", "#", event["title"])).strip()
    category = next((t for t in TAGS if t in tags), "other")
    return {"id": str(event["id"]), "title": event["title"].strip(), "rules": (event.get("description") or "").strip(),
            "as_of": as_of, "end": str(event.get("endDate") or "")[:10], "labels": labels, "winner": winners[0]["groupItemTitle"].strip(),
            "kind": "bracket" if ordered else "categorical", "series": series, "category": category}


@functools.lru_cache(maxsize=1)
def polymarket_events() -> tuple[dict, ...]:
    events = {}
    for page in _pages():
        for event in page:
            events[str(event["id"])] = event
    parsed = [p for p in (_parse(e) for _, e in sorted(events.items(), key=lambda kv: int(kv[0]))) if p]
    per_series, out = defaultdict(int), []
    for p in parsed:
        key = (p["kind"], p["series"])
        if per_series[key] < PER_SERIES:
            per_series[key] += 1
            out.append(p)
    return tuple(out)


def _state(record: dict) -> dict:
    return {"question": record["title"], "resolution_rules": record["rules"] or "(not stated)",
            "as_of_date": record["as_of"], "scheduled_end_date": record["end"] or "(not stated)"}


@register_task
class PolymarketMultiOutcome(TaskAdapter):
    name = "fore/multi_outcome/polymarket"
    leaf = "fore/multi_outcome"
    source = "Polymarket multi-outcome events ended January-August 2026 (gamma API)"
    url = URL
    license = LICENSE
    redistributable = False
    tier = "C"
    criteria_source = "Written; options are Polymarket's outcome names, the rules text is Polymarket's own"
    notes = ("Elections, appointments, awards, central-bank decisions, company and AI-model rankings. Options are the outcomes "
             "listed on the event's start date; events whose winner was added later are left out. At most 2 events per "
             "recurring series. " + CONTAMINATION)

    def load(self):
        return [e for e in polymarket_events() if e["kind"] == "categorical"]

    def label_of(self, record):
        return record["category"]

    def to_item(self, record, rng):
        ids = {}
        for label in record["labels"]:
            key = slug(label)[:60] or "option"
            while key in ids:
                key += "_"
            ids[key] = label
        options = shuffled_options(ids, rng)
        gold = next(k for k, v in ids.items() if v == record["winner"])
        instructions = ("Which of these mutually exclusive outcomes will happen? Exactly one of them resolves Yes under the "
                        "resolution rules. Use only information available on the as-of date.")
        question = Question(type="choice", instructions=instructions, criteria=options)
        return self.item(_state(record), {"outcome": question}, {"outcome": Gold(label=gold)},
                         category=record["category"], original_id=record["id"])


@register_task
class PolymarketNumericBins(TaskAdapter):
    name = "fore/numeric_bins/polymarket"
    leaf = "fore/numeric_bins"
    source = "Polymarket range ('bracket') events ended January-August 2026 (gamma API)"
    url = URL
    license = LICENSE
    redistributable = False
    tier = "C"
    criteria_source = "Written; the ordered levels are Polymarket's own ranges, ordered by Polymarket's threshold values"
    notes = ("Temperatures, precipitation, economic releases, prices at a date, counts (posts, ships, earthquakes), turnouts and "
             "dates. The levels are the event's ranges in increasing order; at most 2 events per recurring series (e.g. one "
             "city's daily high); only events with at most 10 ranges (the score-question limit). " + CONTAMINATION)

    def load(self):  # at most MAX_LEVELS ranges, the most a score question may have
        return [e for e in polymarket_events() if e["kind"] == "bracket" and len(e["labels"]) <= MAX_LEVELS]

    def label_of(self, record):
        position = record["labels"].index(record["winner"]) / max(1, len(record["labels"]) - 1)
        return f"{record['category']}|{'low' if position < 1 / 3 else 'high' if position > 2 / 3 else 'middle'}"

    def to_item(self, record, rng):
        instructions = ("Which range will the outcome fall in? The levels are the event's mutually exclusive ranges in increasing "
                        "order; exactly one resolves Yes under the resolution rules. Use only information available on the as-of date.")
        question = Question(type="score", instructions=instructions, criteria=record["labels"])
        gold = Gold(label=str(record["labels"].index(record["winner"])))
        return self.item(_state(record), {"range": question}, {"range": gold}, category=record["category"], original_id=record["id"])
