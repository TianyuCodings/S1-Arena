"""Forecasts resolved from published daily FRED data (numeric_bins and multi_outcome).

numeric_bins: where will a daily FRED series stand on a later date?

For a fixed schedule of as-of dates (every 14 days from October 2025 to June 2026) and horizons
of 7, 30 and 90 days, the question asks for the value of a public-domain daily series (Treasury
yields, breakeven inflation, exchange rates, oil and gas prices, the broad dollar index) on the
resolution date. The five levels are cut around the last value published before the as-of date,
at the 10th, 30th, 70th and 90th percentiles of the series' own changes over the same horizon in
the three years before the as-of date, so a forecaster who knows only the history would give
them 10/20/40/20/10%. Gold is the level holding the published value, which is correct by
construction.

multi_outcome: which of several related series will move most between the as-of date and the
resolution date (which currency gains most against the dollar, which energy price rises most,
which Treasury yield rises most)? Gold is the series with the largest move in the published data.

An independent checker re-derives every item from the raw CSV files.
"""

from __future__ import annotations

import bisect
import datetime as dt
import functools

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import fetch

LEAF = "fore/numeric_bins"
SERIES = {  # id -> (title, unit shown after values, decimals, source)
    "DGS10": ("10-year US Treasury constant-maturity yield", "%", 2, "Federal Reserve Board, H.15"),
    "DGS2": ("2-year US Treasury constant-maturity yield", "%", 2, "Federal Reserve Board, H.15"),
    "DGS3MO": ("3-month US Treasury constant-maturity yield", "%", 2, "Federal Reserve Board, H.15"),
    "T10Y2Y": ("spread between the 10-year and 2-year US Treasury yields", " percentage points", 2, "Federal Reserve Bank of St. Louis from H.15 data"),
    "T10YIE": ("10-year US breakeven inflation rate", "%", 2, "Federal Reserve Bank of St. Louis from H.15 data"),
    "DFII10": ("10-year US Treasury inflation-indexed (TIPS) yield", "%", 2, "Federal Reserve Board, H.15"),
    "DEXUSEU": ("euro exchange rate in US dollars per euro", " USD", 4, "Federal Reserve Board, H.10"),
    "DEXUSUK": ("pound sterling exchange rate in US dollars per pound", " USD", 4, "Federal Reserve Board, H.10"),
    "DEXJPUS": ("exchange rate in Japanese yen per US dollar", " JPY", 2, "Federal Reserve Board, H.10"),
    "DEXCHUS": ("exchange rate in Chinese yuan per US dollar", " CNY", 4, "Federal Reserve Board, H.10"),
    "DEXMXUS": ("exchange rate in Mexican pesos per US dollar", " MXN", 4, "Federal Reserve Board, H.10"),
    "DTWEXBGS": ("nominal broad US dollar index", "", 2, "Federal Reserve Board, H.10"),
    "DCOILWTICO": ("West Texas Intermediate crude oil spot price (US dollars per barrel)", "", 2, "U.S. Energy Information Administration"),
    "DCOILBRENTEU": ("Brent crude oil spot price (US dollars per barrel)", "", 2, "U.S. Energy Information Administration"),
    "DHHNGSP": ("Henry Hub natural gas spot price (US dollars per million BTU)", "", 2, "U.S. Energy Information Administration"),
}
AS_OF = [dt.date(2025, 10, 1) + dt.timedelta(days=14 * k) for k in range(20)]  # 2025-10-01 ... 2026-06-24
HORIZONS = (7, 30, 90)
LAST_RESOLUTION = dt.date(2026, 9, 1)
SNAPSHOT = "2026-09"  # cache tag of the downloaded CSV files
QUANTILES = (0.10, 0.30, 0.70, 0.90)


@functools.lru_cache(maxsize=None)
def series_values(series_id: str) -> tuple[tuple[dt.date, ...], tuple[float, ...]]:
    """Observation dates and values of a FRED series (missing days skipped)."""
    path = fetch(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}", f"fred_{series_id}_{SNAPSHOT}.csv")
    dates, values = [], []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            day, _, value = line.strip().partition(",")
            try:
                values.append(float(value))
            except ValueError:
                continue
            dates.append(dt.date.fromisoformat(day))
    return tuple(dates), tuple(values)


def value_on(dates, values, day: dt.date) -> tuple[dt.date, float] | None:
    """Last observation on or before the day."""
    i = bisect.bisect_right(dates, day) - 1
    return (dates[i], values[i]) if i >= 0 else None


def quantile(sorted_values: list[float], q: float) -> float:
    """Linear interpolation between order statistics (numpy's default)."""
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def make_record(series_id: str, as_of: dt.date, horizon: int) -> dict | None:
    dates, values = series_values(series_id)
    resolution = as_of + dt.timedelta(days=horizon)
    if resolution > LAST_RESOLUTION or dates[-1] < resolution:
        return None
    start = value_on(dates, values, as_of - dt.timedelta(days=1))  # published before the as-of date
    end = value_on(dates, values, resolution)
    if start is None or end is None or end[0] <= start[0]:
        return None
    changes = []
    first = as_of - dt.timedelta(days=3 * 365)
    for i in range(bisect.bisect_left(dates, first), len(dates)):
        t = dates[i]
        later = t + dt.timedelta(days=horizon)
        if later >= as_of:
            break
        v = value_on(dates, values, later)
        changes.append(v[1] - values[i])
    if len(changes) < 200:
        return None
    changes.sort()
    decimals = SERIES[series_id][2]
    edges = [round(start[1] + quantile(changes, q), decimals) for q in QUANTILES]
    if any(b <= a for a, b in zip(edges, edges[1:])):
        return None
    level = bisect.bisect_right(edges, end[1])  # a value equal to an edge belongs to the higher level
    return {"series": series_id, "as_of": as_of.isoformat(), "horizon": horizon, "resolution": resolution.isoformat(),
            "start_date": start[0].isoformat(), "start": start[1], "end_date": end[0].isoformat(), "end": end[1],
            "edges": edges, "level": level}


def levels(series_id: str, edges: list[float]) -> list[str]:
    unit, decimals = SERIES[series_id][1], SERIES[series_id][2]
    f = lambda v: f"{v:.{decimals}f}{unit}"
    return [f"below {f(edges[0])}"] + [f"from {f(a)} to below {f(b)}" for a, b in zip(edges, edges[1:])] + [f"{f(edges[-1])} or higher"]


@register_task
class FredNumericBins(TaskAdapter):
    name = f"{LEAF}/fred_daily"
    leaf = LEAF
    source = "Daily FRED series (Treasury yields, breakeven inflation, exchange rates, oil and gas prices), values published up to September 2026"
    url = "https://fred.stlouisfed.org/"
    license = "Public-domain series from the Federal Reserve Board and the U.S. EIA via FRED; items generated by S1-Arena (CC BY 4.0)"
    tier = "C"
    criteria_source = "Written; levels are cut at percentiles of the series' own past changes, so they are fixed before the outcome"
    notes = ("15 series, 20 as-of dates (every 14 days from 2025-10-01) and horizons of 7, 30 and 90 days, resolved by 2026-09-01. "
             "The value published before the as-of date is shown; the levels would get 10/20/40/20/10% from the series' own "
             "history. Static items: a model trained on data after the resolution date may know the value. Checked by an "
             "independent re-derivation from the raw CSV files.")

    def load(self):
        rows = []
        for series_id in SERIES:
            for as_of in AS_OF:
                for horizon in HORIZONS:
                    record = make_record(series_id, as_of, horizon)
                    if record is not None:
                        rows.append(record)
        return rows

    def label_of(self, record):
        return f"{record['series']}|{record['horizon']}"

    def to_item(self, record, rng):
        title, unit, decimals, source = SERIES[record["series"]]
        state = {
            "question": f"What will the {title} be on {record['resolution']}?",
            "resolution_criteria": (f"Resolves to the value of FRED series {record['series']} ({source}) for {record['resolution']}, "
                                    "or for the last earlier day with a published value if there is none for that day (weekends, holidays)."),
            "as_of_date": record["as_of"],
            "latest_value_before_as_of_date": f"{record['start']:.{decimals}f}{unit} (for {record['start_date']})",
        }
        question = Question(type="score", criteria=levels(record["series"], record["edges"]),
                            instructions="Which range will the value fall in? Use only information available on the as-of date.")
        return self.item(state, {"range": question}, {"range": Gold(label=str(record["level"]))},
                         series=record["series"], horizon_days=record["horizon"], resolved_value=record["end"], resolved_value_date=record["end_date"])


# ---------- multi_outcome: which of several related series moves most ----------

GROUPS = {  # group -> (question, {option id: (label, series, direction)}); direction +1: a rise counts as a gain
    "currencies": ("Between {as_of} and {resolution}, which of these currencies will gain the most against the US dollar "
                   "(largest percentage appreciation, or smallest depreciation)?",
                   {"eur": ("Euro", "DEXUSEU", +1), "gbp": ("British pound", "DEXUSUK", +1), "jpy": ("Japanese yen", "DEXJPUS", -1),
                    "cny": ("Chinese yuan", "DEXCHUS", -1), "mxn": ("Mexican peso", "DEXMXUS", -1)}),
    "energy": ("Between {as_of} and {resolution}, which of these spot prices will rise the most in percentage terms (or fall the least)?",
               {"wti": ("West Texas Intermediate crude oil", "DCOILWTICO", +1), "brent": ("Brent crude oil", "DCOILBRENTEU", +1),
                "henry_hub": ("Henry Hub natural gas", "DHHNGSP", +1)}),
    "yields": ("Between {as_of} and {resolution}, which of these US Treasury yields will rise the most in percentage points (or fall the least)?",
               {"3_month": ("3-month Treasury yield", "DGS3MO", +1), "2_year": ("2-year Treasury yield", "DGS2", +1),
                "10_year": ("10-year Treasury yield", "DGS10", +1), "10_year_tips": ("10-year TIPS (inflation-indexed) yield", "DFII10", +1)}),
}


def move(group: str, series_id: str, direction: int, start: float, end: float) -> float:
    """Percent gain for prices and currencies (from the foreign currency's side), change in points for yields."""
    if group == "yields":
        return end - start
    return (end / start - 1.0) if direction > 0 else (start / end - 1.0)


def make_race(group: str, as_of: dt.date, horizon: int) -> dict | None:
    resolution = as_of + dt.timedelta(days=horizon)
    if resolution > LAST_RESOLUTION:
        return None
    moves, latest = {}, {}
    for option, (label, series_id, direction) in GROUPS[group][1].items():
        dates, values = series_values(series_id)
        start = value_on(dates, values, as_of - dt.timedelta(days=1))
        end = value_on(dates, values, resolution)
        if start is None or end is None or end[0] <= start[0] or dates[-1] < resolution:
            return None
        moves[option] = move(group, series_id, direction, start[1], end[1])
        decimals, unit = SERIES[series_id][2], SERIES[series_id][1]
        latest[option] = f"{label}: {start[1]:.{decimals}f}{unit} ({SERIES[series_id][0]}; FRED {series_id}, {start[0].isoformat()})"
    ranked = sorted(moves.values(), reverse=True)
    if ranked[0] - ranked[1] < 1e-9:
        return None  # a tie has no single winner
    return {"group": group, "as_of": as_of.isoformat(), "horizon": horizon, "resolution": resolution.isoformat(),
            "latest": latest, "moves": {k: round(v, 6) for k, v in moves.items()}, "winner": max(moves, key=moves.get)}


@register_task
class FredMultiOutcome(TaskAdapter):
    name = "fore/multi_outcome/fred_races"
    leaf = "fore/multi_outcome"
    source = "Which related daily FRED series moves most (currencies against the dollar, energy prices, Treasury yields), resolved by September 2026"
    url = "https://fred.stlouisfed.org/"
    license = FredNumericBins.license
    tier = "C"
    criteria_source = "Written; gold is the option with the largest move in the published data"
    notes = ("Three question groups (5 currencies, 3 energy prices, 4 Treasury yields), 20 as-of dates (every 14 days from "
             "2025-10-01) and horizons of 7, 30 and 90 days, resolved by 2026-09-01. Currency moves are measured from the "
             "foreign currency's side, prices in percent, yields in percentage points. Static items: a model trained on data "
             "after the resolution date may know the outcome. Checked by an independent re-derivation from the raw CSV files.")

    def load(self):
        rows = []
        for group in GROUPS:
            for as_of in AS_OF:
                for horizon in HORIZONS:
                    record = make_race(group, as_of, horizon)
                    if record is not None:
                        rows.append(record)
        return rows

    def label_of(self, record):
        return f"{record['group']}|{record['horizon']}"

    def to_item(self, record, rng):
        text, options = GROUPS[record["group"]]
        state = {
            "question": text.format(as_of=record["as_of"], resolution=record["resolution"]),
            "resolution_criteria": ("Each option is measured with its FRED series: the last value published before the as-of date "
                                    f"against the value for {record['resolution']}, or for the last earlier day with a published value."),
            "as_of_date": record["as_of"],
            "latest_values_before_as_of_date": "\n".join(record["latest"][k] for k in options),
        }
        criteria = shuffled_options({k: label for k, (label, _, _) in options.items()}, rng)
        question = Question(type="choice", criteria=criteria, instructions="Which option will it be? Use only information available on the as-of date.")
        return self.item(state, {"winner": question}, {"winner": Gold(label=record["winner"])},
                         group=record["group"], horizon_days=record["horizon"], moves=record["moves"])
