"""Leaf: topic and hierarchical topic classification (主题 / 层级分类)."""

from __future__ import annotations

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import hf_rows, humanize, slug

LEAF = "basic_cls/topic"


@register_task
class DBpediaL2(TaskAdapter):
    name = f"{LEAF}/dbpedia_l2"
    leaf = LEAF
    source = "DBpedia hierarchical classes, level 2 (DeveloperOats/DBPedia_Classes)"
    url = "https://huggingface.co/datasets/DeveloperOats/DBPedia_Classes"
    license = "CC0 (card); text from Wikipedia (CC BY-SA)"
    tier = "C"
    criteria_source = "Written from the DBpedia class names and their level-1 parent classes"

    def load(self):
        rows = hf_rows("DeveloperOats/DBPedia_Classes", None, "test")
        self.options = {r["l2"]: f"{humanize(r['l2'])} (a kind of {humanize(r['l1'])})" for r in rows}
        return rows

    def label_of(self, record):
        return record["l2"]

    def to_item(self, record, rng):
        question = Question(
            type="choice",
            instructions="Which DBpedia ontology class best describes the subject of this Wikipedia abstract?",
            criteria=shuffled_options(dict(sorted(self.options.items())), rng),
        )
        return self.item({"abstract": record["text"]}, {"class": question}, {"class": Gold(label=record["l2"])}, l1=record["l1"], l3=record["l3"])


@register_task
class Newsgroups20(TaskAdapter):
    name = f"{LEAF}/20newsgroups"
    leaf = LEAF
    source = "20 Newsgroups (Lang, 1995)"
    url = "https://huggingface.co/datasets/SetFit/20_newsgroups"
    license = "CC BY 4.0 (UCI)"
    tier = "C"
    criteria_source = "Written from the newsgroup names"

    GROUPS = {
        "alt.atheism": "Discussion of atheism and arguments about religion from a non-believer's perspective.",
        "comp.graphics": "Computer graphics: image formats, rendering and graphics software.",
        "comp.os.ms-windows.misc": "The Microsoft Windows operating system.",
        "comp.sys.ibm.pc.hardware": "IBM PC-compatible hardware.",
        "comp.sys.mac.hardware": "Apple Macintosh hardware.",
        "comp.windows.x": "The X Window System.",
        "misc.forsale": "Items offered for sale or wanted.",
        "rec.autos": "Cars and automobiles.",
        "rec.motorcycles": "Motorcycles and riding.",
        "rec.sport.baseball": "Baseball.",
        "rec.sport.hockey": "Ice hockey.",
        "sci.crypt": "Cryptography, encryption and data security.",
        "sci.electronics": "Electronics and circuits.",
        "sci.med": "Medicine and health.",
        "sci.space": "Space, spaceflight and astronomy.",
        "soc.religion.christian": "Christianity and Christian belief.",
        "talk.politics.guns": "Politics of guns and gun control.",
        "talk.politics.mideast": "Politics of the Middle East.",
        "talk.politics.misc": "Other political topics.",
        "talk.religion.misc": "Other discussion of religion.",
    }

    def load(self):
        return [r for r in hf_rows("SetFit/20_newsgroups", None, "test") if r["text"].strip()]

    def label_of(self, record):
        return record["label_text"]

    def to_item(self, record, rng):
        options = {k: f"{k}: {v}" for k, v in self.GROUPS.items()}
        if record["label_text"] not in options:
            return None
        question = Question(type="choice", instructions="Which newsgroup was this post published in?", criteria=shuffled_options(options, rng))
        return self.item({"post": record["text"]}, {"newsgroup": question}, {"newsgroup": Gold(label=record["label_text"])})


@register_task
class WOS46985(TaskAdapter):
    name = f"{LEAF}/wos46985"
    leaf = LEAF
    source = "Web of Science WOS-46985 (Kowsari et al., 2017)"
    url = "https://huggingface.co/datasets/river-martin/web-of-science-with-label-texts"
    license = "CC BY 4.0"
    tier = "C"
    criteria_source = "Written from the area and domain names"

    def load(self):
        rows = hf_rows("river-martin/web-of-science-with-label-texts", None, "test")
        self.ids, self.options = {}, {}
        for r in rows:
            key = (r["domain"].strip(), r["area"].strip())
            if key not in self.ids:
                option = slug(key[1]) if slug(key[1]) not in self.options else slug(f"{key[0]} {key[1]}")
                self.ids[key] = option
                self.options[option] = f"{key[1]} (a research area in {key[0]})"
        return rows

    def label_of(self, record):
        return self.ids[(record["domain"].strip(), record["area"].strip())]

    def to_item(self, record, rng):
        label = self.label_of(record)
        question = Question(
            type="choice",
            instructions="Which research area is this paper abstract from?",
            criteria=shuffled_options(dict(sorted(self.options.items())), rng),
        )
        return self.item({"abstract": record["abstract"]}, {"area": question}, {"area": Gold(label=label)}, domain=record["domain"].strip())
