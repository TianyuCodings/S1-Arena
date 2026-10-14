"""Base class for task adapters.

To add a task, subclass TaskAdapter in a new module under s1arena/tasks/, fill in
the class attributes, implement load(), label_of() and to_item(), and decorate the
class with @register_task. The builder, runner, scorer and site export pick it up
automatically.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from typing import Any, Callable, Iterable

from ..schema import DecisionRequest, Gold, Item, Question

# There is no shared input budget: long items are kept and each model is limited only by its own context length
# (a request a model cannot take counts as that model's failure). approx_tokens() reports sizes.


class TaskAdapter:
    name: str = ""  # "basic_cls/intent/clinc150"
    leaf: str = ""  # "basic_cls/intent"
    source: str = ""  # dataset name as cited
    url: str = ""  # dataset page
    license: str = ""
    tier: str = "C"  # A: original instructions verbatim; B: adapted; C: newly written
    criteria_source: str = ""  # where the instructions and option descriptions come from
    redistributable: bool = True  # False -> built items go to data_local/ (not committed)
    sampling: str = "proportional"  # or "balanced": equal share per stratum
    items_per_record: int = 1  # e.g. 2 when a pair is asked in both orders (see to_items); such items share meta["pair"]
    rank_key: str | None = None  # "meta:<key>" or "state:<key>": items ranked together for NDCG (score questions)
    notes: str = ""

    def load(self) -> Iterable[dict[str, Any]]:
        """Yield raw records from the evaluation split."""
        raise NotImplementedError

    def label_of(self, record: dict[str, Any]) -> str:
        """Key used for stratified sampling (usually the gold label)."""
        raise NotImplementedError

    def to_item(self, record: dict[str, Any], rng: random.Random) -> Item | None:
        """Convert one raw record into an Item; return None to skip it."""
        raise NotImplementedError

    def to_items(self, record: dict[str, Any], rng: random.Random) -> list[Item]:
        """Items made from one record: by default the single item from to_item(). Override it
        (and set items_per_record) to emit several, e.g. a pairwise comparison in both orders."""
        item = self.to_item(record, rng)
        return [] if item is None else [item]

    def build(self, n: int = 500, seed: int = 0) -> list[Item]:
        records = [r for r in self.load() if r is not None]
        sample = stratified_sample(records, self.label_of, max(1, n // self.items_per_record), seed, balanced=self.sampling == "balanced")
        spares = None  # computed on the first duplicate only
        items, seen, self.excluded, self.duplicates = [], set(), 0, 0
        queue = list(sample)
        while queue:
            record = queue.pop(0)
            produced = self.to_items(record, self.record_rng(record, seed))
            if any(item.id in seen for item in produced):  # same content as an item already taken: replace the record
                self.duplicates += 1
                spares = spares if spares is not None else self._spares(records, sample, seed)
                key = self.label_of(record)
                pool = spares.get(key) or next((group for group in spares.values() if group), None)
                if pool:
                    queue.append(pool.pop(0))
                continue
            if len(produced) > 1:  # a pair (or group) from one record: scored together by pair accuracy
                group = hashlib.sha256("|".join(i.id for i in produced).encode()).hexdigest()[:12]
                for item in produced:
                    item.meta.setdefault("pair", group)
            for item in produced:
                seen.add(item.id)
                items.append(item)
        return items

    def _spares(self, records: list[dict[str, Any]], sample: list[dict[str, Any]], seed: int) -> dict[str, list]:
        """Records left out of the sample, by stratum, in a fixed order: replacements for duplicates."""
        taken = {id(r) for r in sample}
        spares: dict[str, list] = defaultdict(list)
        for record in records:
            if id(record) not in taken:
                spares[self.label_of(record)].append(record)
        order = lambda r: hashlib.sha256(f"{seed}:{self.name}:{json.dumps(r, sort_keys=True, default=str, ensure_ascii=False)}".encode()).hexdigest()
        return {key: sorted(group, key=order) for key, group in spares.items()}

    def record_rng(self, record: dict[str, Any], seed: int) -> random.Random:
        """Randomness tied to the record (e.g. option order), so an item does not change when the sample size does."""
        key = json.dumps(record, sort_keys=True, default=str, ensure_ascii=False)
        return random.Random(hashlib.sha256(f"{seed}:{self.name}:{key}".encode()).hexdigest())

    def item(self, state: dict[str, Any], questions: dict[str, Question], gold: dict[str, Gold], **meta: Any) -> Item:
        """Assemble an Item with this task's identity and provenance filled in."""
        return Item(
            leaf=self.leaf,
            task=self.name,
            request=DecisionRequest(state=state, questions=questions),
            gold=gold,
            meta={"source": self.source, "license": self.license, "tier": self.tier, **meta},
        )

    def card(self) -> dict[str, Any]:
        keys = ("name", "leaf", "source", "url", "license", "tier", "criteria_source", "redistributable", "notes")
        return {k: getattr(self, k) for k in keys}


_ENCODER = None


def approx_tokens(request: DecisionRequest) -> int:
    """Token count of the request under a fixed reference tokenizer (tiktoken o200k_base)."""
    global _ENCODER
    if _ENCODER is None:
        import tiktoken

        _ENCODER = tiktoken.get_encoding("o200k_base")
    return len(_ENCODER.encode(request.model_dump_json(), disallowed_special=()))


def stratified_sample(
    records: list[dict[str, Any]],
    key: Callable[[dict[str, Any]], str],
    n: int,
    seed: int,
    min_per_class: int = 1,
    balanced: bool = False,
) -> list[dict[str, Any]]:
    """Stratified sample of size n: proportional to class size (at least min_per_class per class
    when possible), or with an equal share per class when balanced."""
    if len(records) <= n:
        out = list(records)
        random.Random(seed).shuffle(out)
        return out
    rng = random.Random(seed)
    groups: dict[str, list] = defaultdict(list)
    for record in records:
        groups[key(record)].append(record)
    for members in groups.values():
        rng.shuffle(members)
    if len(groups) > n:  # more classes than items: one item from each of n randomly chosen classes
        sample = [groups[k][0] for k in rng.sample(sorted(groups), n)]
        rng.shuffle(sample)
        return sample
    total = len(records)
    if balanced:
        alloc = {k: min(len(v), n // len(groups)) for k, v in groups.items()}
    else:
        alloc = {k: min(len(v), max(min_per_class, int(n * len(v) / total))) for k, v in groups.items()}
    while sum(alloc.values()) > n:
        largest = max(alloc, key=lambda k: (alloc[k], k))
        alloc[largest] -= 1
    spare = sorted((k for k in groups if alloc[k] < len(groups[k])), key=lambda k: -len(groups[k]))
    while sum(alloc.values()) < n and spare:
        for k in list(spare):
            if sum(alloc.values()) >= n:
                break
            if alloc[k] < len(groups[k]):
                alloc[k] += 1
            else:
                spare.remove(k)
    sample = [r for k in sorted(groups) for r in groups[k][: alloc[k]]]
    rng.shuffle(sample)
    return sample


def shuffled_options(options: dict[str, str], rng: random.Random) -> dict[str, str]:
    """Present choice options in a random but reproducible order."""
    keys = list(options)
    rng.shuffle(keys)
    return {k: options[k] for k in keys}
