"""Leaf: binary forecasts of future events (未来事件二元预测).

One boolean question: will the question resolve Yes? The state holds the question, its
background and resolution criteria, and the as-of date; gold is the recorded resolution.
The crowd or market probability at the as-of date is kept in meta as a calibration reference.
"""

from __future__ import annotations

import functools
import html
import json
import re

from ...env import load_env
from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter

LEAF = "fore/binary_events"
CONTAMINATION = ("Static items: a model whose training data postdates a question's resolution may know the outcome; filter "
                 "by resolution date against each model's cutoff.")

# ---------- ForecastBench ----------

FB_REPO = "forecastingresearch/forecastbench-datasets"
FB_REV = "c1360cb34bdf2d6b77f86f800af436c9ebbc649b"  # 2026-09-22 snapshot; the repository is updated nightly
MARKETS = {"manifold": "Manifold", "metaculus": "Metaculus", "polymarket": "Polymarket", "infer": "INFER (RAND Forecasting Initiative)"}
DATASETS = ("fred", "dbnomics", "wikipedia", "yfinance")  # ACLED left out: restrictive data terms and almost always 'No'
COPYRIGHT = re.compile(r"all rights reserved|may not be copied|not be (?:copied|reproduced)", re.I)


def _clean(text) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", str(text or "")))
    return re.sub(r"[ \t\r\f\v]+", " ", re.sub(r"\n\s*\n+", "\n\n", text)).strip()


def _fb_file(name: str) -> dict:
    load_env()
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(FB_REPO, name, repo_type="dataset", revision=FB_REV), encoding="utf-8") as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def forecastbench() -> tuple[dict, dict]:
    """(question sets by forecast due date, resolution entries by due date) at the pinned revision."""
    load_env()
    from huggingface_hub import HfApi

    files = [s.rfilename for s in HfApi().dataset_info(FB_REPO, revision=FB_REV).siblings]
    qsets, rsets = {}, {}
    for name in sorted(files):
        m = re.fullmatch(r"datasets/question_sets/(\d{4}-\d{2}-\d{2})-llm\.json", name)
        if m:
            qsets[m.group(1)] = _fb_file(name)["questions"]
        m = re.fullmatch(r"datasets/resolution_sets/(\d{4}-\d{2}-\d{2})_resolution_set\.json", name)
        if m:
            rsets[m.group(1)] = _fb_file(name)["resolutions"]
    return qsets, rsets


class ForecastBenchTask(TaskAdapter):
    leaf = LEAF
    url = "https://huggingface.co/datasets/forecastingresearch/forecastbench-datasets"
    license = "CC BY-SA 4.0"
    tier = "B"
    criteria_source = "ForecastBench's per-source question introduction (source_intro), verbatim, plus a written question sentence"

    def label_of(self, record):
        return f"{record['source']}|{record['label']}"

    def to_item(self, record, rng):
        instructions = f"{record['intro']} Using only information available on the as-of date, will this question resolve Yes?"
        return self.item(record["state"], {"resolves_yes": Question(type="boolean", instructions=instructions)},
                         {"resolves_yes": Gold(label=record["label"])}, **record["meta"])


@register_task
class ForecastBenchMarkets(ForecastBenchTask):
    name = f"{LEAF}/forecastbench_markets"
    source = "ForecastBench question sets, prediction-market and forecasting-platform questions (Karger et al., 2025)"
    notes = ("Manifold, Metaculus, Polymarket and INFER questions asked between July 2024 and August 2026 that have resolved Yes "
             "or No (questions still open are left out). As-of date = the freeze date of the first question set that asked it; "
             "the market or crowd probability on that date is in meta (market_prob_at_as_of). Only resolved questions are used, "
             "so questions that resolve early (often Yes) are over-represented among long-horizon ones. " + CONTAMINATION)

    def load(self):
        qsets, rsets = forecastbench()
        first = {}
        for due in sorted(qsets):
            for q in qsets[due]:
                if isinstance(q["id"], str) and q["source"] in MARKETS and (q["source"], q["id"]) not in first:
                    first[(q["source"], q["id"])] = q
        outcome, resolved_on = {}, {}
        for due in sorted(rsets):
            for r in rsets[due]:
                key = (r["source"], r["id"]) if isinstance(r["id"], str) else None
                if key in first and r.get("resolved") and r.get("resolved_to") in (0, 1, 0.0, 1.0):
                    outcome.setdefault(key, set()).add(int(r["resolved_to"]))
                    resolved_on[key] = min(resolved_on.get(key, r["resolution_date"]), r["resolution_date"])
        rows = []
        for key, values in sorted(outcome.items()):
            if len(values) != 1:
                continue
            q = first[key]
            criteria = f"Resolves to the outcome of the question on {MARKETS[q['source']]}."
            if str(q.get("market_info_resolution_criteria") or "N/A") != "N/A":
                criteria += "\n\n" + _clean(q["market_info_resolution_criteria"])
            state = {"question": _clean(q["question"]), "background": _clean(q.get("background")) or "(none)",
                     "resolution_criteria": criteria, "as_of_date": q["freeze_datetime"][:10],
                     "question_close_date": str(q.get("market_info_close_datetime") or "")[:10] or "(not stated)"}
            meta = {"platform": q["source"], "market_prob_at_as_of": _float(q.get("freeze_datetime_value")),
                    "resolved_by": resolved_on[key], "original_id": q["id"]}
            rows.append({"source": q["source"], "label": "true" if values == {1} else "false", "intro": q["source_intro"].strip(),
                         "state": state, "meta": meta})
        return rows


@register_task
class ForecastBenchDatasets(ForecastBenchTask):
    name = f"{LEAF}/forecastbench_datasets"
    source = "ForecastBench question sets, questions resolved from FRED, DBnomics, Wikipedia and Yahoo Finance data (Karger et al., 2025)"
    notes = ("Will a series be higher on the resolution date than on the forecast due date? One item per question set, series "
             "and horizon (7 days to several years), resolved from the published data. The latest value on the as-of date is "
             "shown. Series whose ForecastBench background carries a no-copying notice are left out. " + CONTAMINATION)

    def load(self):
        qsets, rsets = forecastbench()
        rows = []
        for due in sorted(rsets):
            questions = {(q["source"], q["id"]): q for q in qsets.get(due, []) if isinstance(q["id"], str)}
            for r in rsets[due]:
                if r.get("source") not in DATASETS or not isinstance(r["id"], str) or r.get("direction") not in (None, "", []):
                    continue
                q = questions.get((r["source"], r["id"]))
                if q is None or not r.get("resolved") or r.get("resolved_to") not in (0, 1, 0.0, 1.0) or COPYRIGHT.search(str(q.get("background"))):
                    continue
                question = q["question"].replace("{forecast_due_date}", due).replace("{resolution_date}", r["resolution_date"])
                state = {"question": _clean(question), "background": _clean(q.get("background")) or "(none)",
                         "resolution_criteria": _clean(q["resolution_criteria"]), "as_of_date": q["freeze_datetime"][:10],
                         "latest_value_on_as_of_date": f"{q['freeze_datetime_value']} ({_clean(q['freeze_datetime_value_explanation'])})"}
                meta = {"platform": q["source"], "forecast_due_date": due, "resolution_date": r["resolution_date"], "original_id": q["id"]}
                rows.append({"source": q["source"], "label": "true" if int(r["resolved_to"]) == 1 else "false",
                             "intro": q["source_intro"].strip(), "state": state, "meta": meta})
        return rows


def _float(value):
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


# ---------- Halawi et al. (2024) test split ----------

@register_task
class HalawiForecasting(TaskAdapter):
    name = f"{LEAF}/halawi_2024"
    leaf = LEAF
    source = "Binary forecasting questions from Metaculus, Manifold, Polymarket, Good Judgment Open and INFER, test split (Halawi et al., 2024)"
    url = "https://huggingface.co/datasets/YuehHanChen/forecasting"
    license = "Apache-2.0"
    tier = "C"
    criteria_source = "Written; background and resolution criteria are the platforms' own texts"
    notes = ("Questions opened in 2023 and resolved between June 2023 and early 2024, so most current models have seen the "
             "outcomes. As-of date = the question's begin date; the first community forecast is in meta. " + CONTAMINATION)

    def load(self):
        load_env()
        from huggingface_hub import hf_hub_download

        path = hf_hub_download("YuehHanChen/forecasting", "test.json", repo_type="dataset", revision="c88acc8705f03c8a6526b0b057200c1f0b1b8505")
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        return [r for r in rows if str(r.get("resolution")) in ("0", "1", "0.0", "1.0")]

    def label_of(self, record):
        return f"{record['data_source']}|{float(record['resolution']):.0f}"

    def to_item(self, record, rng):
        state = {"question": _clean(record["question"]), "background": _clean(record.get("background")) or "(none)",
                 "resolution_criteria": _clean(record.get("resolution_criteria")) or "(not stated)",
                 "as_of_date": str(record["date_begin"])[:10], "question_close_date": str(record.get("date_close") or "")[:10] or "(not stated)"}
        try:
            first = json.loads(record.get("community_predictions") or "[]")
            crowd = float(first[0][1]) if first else None
        except (ValueError, TypeError, IndexError):
            crowd = None
        instructions = "Using only information available on the as-of date, will this question resolve Yes?"
        gold = Gold(label="true" if float(record["resolution"]) == 1 else "false")
        return self.item(state, {"resolves_yes": Question(type="boolean", instructions=instructions)}, {"resolves_yes": gold},
                         platform=record["data_source"], crowd_prob_first=crowd, resolved_by=str(record.get("date_resolve_at"))[:10])
