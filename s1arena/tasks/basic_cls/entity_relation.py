"""Leaf: entity typing and relation classification (实体类型 / 关系分类)."""

from __future__ import annotations

import json
import re

from ...env import load_env
from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import fetch, hf_rows, slug

LEAF = "basic_cls/entity_relation"


def _relation_text(x: str, y: str) -> dict[str, str]:
    return {
        "Cause-Effect": f"{x} is the cause of {y} ({y} is the effect).",
        "Instrument-Agency": f"{x} is an instrument used by the agent {y}.",
        "Product-Producer": f"{x} is a product made or produced by {y}.",
        "Content-Container": f"{x} is or was contained in the container {y}.",
        "Entity-Origin": f"{x} originates or comes from {y}.",
        "Entity-Destination": f"{x} moves towards the destination {y}.",
        "Component-Whole": f"{x} is a component or part of the whole {y}.",
        "Member-Collection": f"{x} is a member of the collection {y}.",
        "Message-Topic": f"{x} is a message or communication about the topic {y}.",
    }


@register_task
class SemEval2010Task8(TaskAdapter):
    name = f"{LEAF}/semeval2010_task8"
    leaf = LEAF
    source = "SemEval-2010 Task 8, relations between nominals (Hendrickx et al., 2010)"
    url = "https://huggingface.co/datasets/SemEvalWorkshop/sem_eval_2010_task_8"
    license = "CC BY 3.0"
    tier = "B"
    criteria_source = "Adapted from the official relation definitions of SemEval-2010 Task 8"

    def load(self):
        rows = hf_rows("SemEvalWorkshop/sem_eval_2010_task_8", None, "test")
        self.options = {}
        for name in sorted({r["relation_name"] for r in rows}):
            if name == "Other":
                self.options[name] = "None of the nine relations below holds between e1 and e2 in either direction."
                continue
            base, first, second = re.match(r"(.+)\((e\d),(e\d)\)", name).groups()
            self.options[name] = _relation_text(first, second)[base]
        return rows

    def label_of(self, record):
        return record["relation_name"]

    def to_item(self, record, rng):
        question = Question(
            type="choice",
            instructions="Which semantic relation holds between the marked nominals <e1> and <e2> in this sentence? The direction matters.",
            criteria=shuffled_options(self.options, rng),
        )
        return self.item({"sentence": record["sentence"]}, {"relation": question}, {"relation": Gold(label=record["relation_name"])})


@register_task
class FewRel(TaskAdapter):
    name = f"{LEAF}/fewrel"
    leaf = LEAF
    source = "FewRel 1.0, the 80 public Wikidata relations (Han et al., 2018)"
    url = "https://github.com/thunlp/FewRel"
    license = "MIT"
    tier = "A"
    criteria_source = "Wikidata relation names and descriptions from pid2name.json, as shown to the FewRel annotators"

    BASE = "https://raw.githubusercontent.com/thunlp/FewRel/master/data/"

    def load(self):
        pid2name = json.loads(fetch(self.BASE + "pid2name.json", "fewrel_pid2name.json").read_text(encoding="utf-8"))
        rows = []
        for part in ("train_wiki", "val_wiki"):
            data = json.loads(fetch(self.BASE + f"{part}.json", f"fewrel_{part}.json").read_text(encoding="utf-8"))
            for pid, instances in data.items():
                rows += [dict(inst, pid=pid) for inst in instances]
        self.ids = {}
        self.options = {}
        for pid in sorted({r["pid"] for r in rows}):
            name, description = pid2name[pid]
            key = slug(name) if slug(name) not in self.options else f"{slug(name)}_{pid.lower()}"
            self.ids[pid] = key
            self.options[key] = f"{name}: {description}"
        return rows

    def label_of(self, record):
        return record["pid"]

    def to_item(self, record, rng):
        tokens = record["tokens"]
        head = " ".join(tokens[i] for i in record["h"][2][0])
        tail = " ".join(tokens[i] for i in record["t"][2][0])
        question = Question(
            type="choice",
            instructions="Which Wikidata relation holds from the head entity to the tail entity, as expressed in this sentence (head -> relation -> tail)?",
            criteria=shuffled_options(self.options, rng),
        )
        state = {"sentence": " ".join(tokens), "head": head, "tail": tail}
        return self.item(state, {"relation": question}, {"relation": Gold(label=self.ids[record["pid"]])}, wikidata_property=record["pid"])


FINE_WORDS = {
    "broadcastprogram": "broadcast program", "writtenart": "written art", "sportsfacility": "sports facility",
    "sportsleague": "sports league", "sportsteam": "sports team", "showorganization": "show organization",
    "livingthing": "living thing", "educationaldegree": "educational degree", "chemicalthing": "chemical thing",
    "biologything": "biology thing", "astronomything": "astronomy thing", "bodiesofwater": "bodies of water",
    "politicalparty": "political party", "governmentagency": "government agency", "militaryconflict": "military conflict",
    "sportsevent": "sports event", "protest": "protest", "restaurant": "restaurant", "medical": "medical",
    "mountain": "mountain", "island": "island", "park": "park", "road": "road", "railway": "railway",
    "transit": "transit", "currency": "currency", "disease": "disease", "god": "god", "language": "language",
    "law": "law", "award": "award", "game": "game", "ship": "ship", "software": "software", "train": "train",
    "weapon": "weapon", "airplane": "airplane", "car": "car", "food": "food", "company": "company", "media": "media",
    "newspaper": "newspaper", "education": "education", "religion": "religion", "actor": "actor", "artist": "artist",
    "athlete": "athlete", "director": "director", "politician": "politician", "scholar": "scholar", "soldier": "soldier",
    "election": "election", "attack": "attack", "battle": "battle", "war": "war", "disaster": "disaster",
}


def fine_description(name: str) -> str:
    coarse, _, fine = name.partition("-")
    words = " / ".join(FINE_WORDS.get(part.lower(), part) for part in fine.split("/"))
    return f"{coarse}: {words}"


@register_task
class FewNERD(TaskAdapter):
    name = f"{LEAF}/fewnerd"
    leaf = LEAF
    source = "Few-NERD supervised, fine-grained entity types (Ding et al., 2021)"
    url = "https://huggingface.co/datasets/DFKI-SLT/few-nerd"
    license = "CC BY-SA 4.0"
    tier = "C"
    criteria_source = "Written from the coarse and fine type names"

    def load(self):
        load_env()
        from datasets import load_dataset

        ds = load_dataset("DFKI-SLT/few-nerd", "supervised", split="test")
        names = ds.features["fine_ner_tags"].feature.names
        self.options = {n: fine_description(n) for n in names if n != "O"}
        rows = []
        for sentence in ds:
            tokens, tags = sentence["tokens"], sentence["fine_ner_tags"]
            start = None
            for i in range(len(tags) + 1):
                tag = tags[i] if i < len(tags) else 0
                if start is not None and (tag != tags[start]):
                    rows.append({"id": sentence["id"], "tokens": tokens, "start": start, "end": i, "type": names[tags[start]]})
                    start = None
                if tag != 0 and start is None:
                    start = i
        return rows

    def label_of(self, record):
        return record["type"]

    def to_item(self, record, rng):
        tokens, s, e = record["tokens"], record["start"], record["end"]
        marked = " ".join(tokens[:s] + ["[["] + tokens[s:e] + ["]]"] + tokens[e:])
        question = Question(
            type="choice",
            instructions="What is the fine-grained type of the entity mention marked with [[ ]] in this sentence?",
            criteria=shuffled_options(self.options, rng),
        )
        state = {"sentence": marked, "mention": " ".join(tokens[s:e])}
        return self.item(state, {"entity_type": question}, {"entity_type": Gold(label=record["type"])}, original_id=record["id"])
