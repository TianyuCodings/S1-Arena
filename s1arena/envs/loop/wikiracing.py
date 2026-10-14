"""Wikiracing on Wikispeedia: get from a start article to a target article by clicking links.

Offline on the Wikispeedia data (West & Leskovec, 2012; SNAP): the 4,604 articles of the 2007
Wikipedia Selection for Schools, their link graph and 51,318 finished human games. Start/target pairs
are long: their shortest path is 5 or 6 clicks, and each was finished in at least MIN_HUMAN_GAMES human game, so
every task is solvable by people and the outcome reports how many clicks they needed. The budget is
tight: the shortest path plus SLACK clicks, so a long detour cannot be repaired.

Each decision is one choice question: the links of the current article (titles, sorted, numbered
1..n; self-links left out) plus "back" to the previous article, as in the original game. The state shows
the target and the current article (title and opening text) and the path so far. An article with more
links than fit in one question (only "United States" and "Driving on the left or right") is split into
pages of PAGE_SIZE links with "more"/"previous" options; changing the page is not a click. An article
without links (17 exist) sends the player back automatically, which counts as a click, as in the game.

The data (two archives, 45 MB) is downloaded into .cache/downloads on first use and reduced to one
index file in .cache/wikispeedia/.
"""

from __future__ import annotations

import collections
import json
import random
import re
import statistics
import tarfile
import threading
import urllib.parse
from typing import Any

from ...env import ROOT
from ...registry import register_env
from ...schema import DecisionRequest, MAX_OPTIONS, Question
from ..base import Environment

GRAPH_URL = "https://snap.stanford.edu/data/wikispeedia/wikispeedia_paths-and-graph.tar.gz"
TEXT_URL = "https://snap.stanford.edu/data/wikispeedia/wikispeedia_articles_plaintext.tar.gz"
INDEX = ROOT / ".cache" / "wikispeedia" / "index.json"
INDEX_VERSION = 2
MIN_HUMAN_GAMES = 1  # a pair qualifies when at least this many human games reached the target (few humans finish long pairs)
STRATA = {5: 10, 6: 10}  # shortest-path distance -> number of pairs
SLACK = 2  # clicks allowed beyond the shortest path
EXCERPT_CHARS = 300
PAGE_SIZE = MAX_OPTIONS - 3  # leaves room for back, more and previous
BACK, MORE, PREVIOUS = "back", "more", "previous"

_LOCK = threading.Lock()
_DATA: dict[str, Any] | None = None


# ---------------------------------------------------------------------------------------------------
# Data: titles, links, opening text of every article, and the task list.


def _title(name: str) -> str:
    return urllib.parse.unquote(name).replace("_", " ")


def _rows(tar: tarfile.TarFile, member: str) -> list[list[str]]:
    text = tar.extractfile(member).read().decode("utf-8")
    return [line.split("\t") for line in text.splitlines() if line.strip() and not line.startswith("#")]


_VERB = re.compile(r"\b(is|was|are|were|refers|means|has|had|became|comprises|consists)\b")


def _drop_captions(lines: list[str]) -> list[str]:
    """Image captions appear as the caption, 'Enlarge' and the caption again; remove all three (and Commons boxes)."""
    lines = list(lines)
    for i in reversed(range(len(lines))):
        if lines[i].strip().startswith("Wikimedia Commons has media related to"):
            del lines[i : i + 2]
    while True:
        marks = [i for i, line in enumerate(lines) if line.strip() == "Enlarge"]
        if not marks:
            break
        i = marks[0]
        k = max([k for k in range(1, i + 1) if lines[i - k : i] == lines[i + 1 : i + 1 + k]], default=0)
        del lines[i - k : i + 1 + k]
    half = len(lines) // 2
    if lines and len(lines) % 2 == 0 and lines[:half] == lines[half:]:
        return []  # a caption printed twice
    return lines


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    out = ""
    for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text):
        if out and len(out) + 1 + len(sentence) > limit:
            break
        out = (out + " " + sentence).strip()
    if len(out) > limit:
        out = out[:limit].rsplit(" ", 1)[0].rstrip(",;:") + " …"
    return out


def excerpt(text: str, title: str, limit: int = EXCERPT_CHARS) -> str:
    """The opening prose of a plaintext article: the first paragraph that reads like the article's lead sentence.

    Paragraphs are indented blocks; headings, lists, tables, infoboxes and image captions are skipped. The
    first paragraph that mentions the title and contains a verb such as "is"/"was" wins, else the first with
    such a verb, else the first paragraph. Cut after whole sentences at `limit` characters.
    """
    words = [w for w in re.findall(r"[a-z0-9]+", re.sub(r"\(.*?\)", "", title).lower()) if len(w) > 2 or w.isdigit()]
    blocks, block = [], []
    for line in text.splitlines():
        if line.strip():
            block.append(line)
        elif block:
            blocks.append(block)
            block = []
    blocks.append(block)
    found = []
    for lines in map(_drop_captions, blocks):
        if not lines:
            continue
        first = lines[0].strip()
        if first.startswith(("#copyright", "2007 Schools Wikipedia Selection", "This is a featured article", "SOS Children")):
            continue
        if not all(line.startswith("   ") for line in lines) or first.startswith(("*", "+", "#", "-", "|")):
            continue  # titles and headings are not indented; lists
        if any(re.search(r"\S {3,}\S", line.strip()) for line in lines):
            continue  # tables and infoboxes
        wrapped = [line.strip() for line in lines[:-1]]
        if wrapped and sum(len(line) >= 50 for line in wrapped) < 0.6 * len(wrapped):
            continue  # prose is wrapped at about 70 characters; short lines are infobox rows or captions
        joined = re.sub(r"\s+", " ", " ".join(line.strip() for line in lines)).strip()
        joined = re.sub(r" ([,.;:)])", r"\1", joined).replace("( ", "(")
        if len(joined.split()) >= 12 and re.search(r"[.!?]", joined):
            found.append(joined)
    tests = (lambda j: _VERB.search(j) and any(w in j[:200].lower() for w in words[:3]), lambda j: _VERB.search(j), lambda j: True)
    for test in tests:
        for joined in found:
            if test(joined):
                return _shorten(joined, limit)
    return ""


def _distances(links: list[list[int]], source: int) -> list[int]:
    dist = [-1] * len(links)
    dist[source] = 0
    queue = collections.deque([source])
    while queue:
        a = queue.popleft()
        for b in links[a]:
            if dist[b] < 0:
                dist[b] = dist[a] + 1
                queue.append(b)
    return dist


def _build() -> dict[str, Any]:
    from ...tasks.common import fetch

    with tarfile.open(fetch(GRAPH_URL), "r:gz") as tar:
        names = [row[0] for row in _rows(tar, "wikispeedia_paths-and-graph/articles.tsv")]
        pairs = _rows(tar, "wikispeedia_paths-and-graph/links.tsv")
        games = _rows(tar, "wikispeedia_paths-and-graph/paths_finished.tsv")
    index = {name: i for i, name in enumerate(names)}
    titles = [_title(name) for name in names]
    links: list[set[int]] = [set() for _ in names]
    for source, target in pairs:
        if source != target:  # a self-link opens the same article again
            links[index[source]].add(index[target])
    links_sorted = [sorted(targets, key=lambda j: (titles[j], j)) for targets in links]

    texts = [""] * len(names)
    with tarfile.open(fetch(TEXT_URL), "r:gz") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".txt"):
                name = member.name.rsplit("/", 1)[-1][: -len(".txt")]
                if name in index:
                    texts[index[name]] = excerpt(tar.extractfile(member).read().decode("utf-8"), _title(name))

    clicks: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    for row in games:
        path = row[3].split(";")
        if path[0] in index and path[-1] in index and path[0] != path[-1]:
            clicks[(index[path[0]], index[path[-1]])].append(len(path) - 1)  # back clicks ("<") count, as in the game
    candidates: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    dist_cache: dict[int, list[int]] = {}
    for (start, target), counts in sorted(clicks.items()):
        if len(counts) < MIN_HUMAN_GAMES or len(links_sorted[start]) < 2:
            continue
        if start not in dist_cache:
            dist_cache[start] = _distances(links_sorted, start)
        d = dist_cache[start][target]
        if d in STRATA:
            candidates[d].append((start, target))
    rng = random.Random("wikiracing")
    tasks = []
    for stratum, count in STRATA.items():
        for start, target in rng.sample(candidates[stratum], count):
            counts = clicks[(start, target)]
            tasks.append({"start": start, "target": target, "shortest_path": stratum, "budget": stratum + SLACK,
                          "human_games": len(counts), "human_median_clicks": float(statistics.median(counts)),
                          "human_min_clicks": min(counts)})
    rng.shuffle(tasks)  # any prefix of the seeds mixes the distances
    return {"version": INDEX_VERSION, "titles": titles, "links": links_sorted, "excerpts": texts, "tasks": tasks}


def data() -> dict[str, Any]:
    global _DATA
    with _LOCK:
        if _DATA is None:
            if INDEX.exists():
                loaded = json.loads(INDEX.read_text(encoding="utf-8"))
                _DATA = loaded if loaded.get("version") == INDEX_VERSION else None
            if _DATA is None:
                _DATA = _build()
                INDEX.parent.mkdir(parents=True, exist_ok=True)
                tmp = INDEX.with_suffix(".part")
                tmp.write_text(json.dumps(_DATA, ensure_ascii=False), encoding="utf-8")
                tmp.replace(INDEX)
        return _DATA


# ---------------------------------------------------------------------------------------------------
# Environment.


@register_env
class Wikiracing(Environment):
    name = "loop/wikiracing/wikispeedia"
    leaf = "loop/wikiracing"
    source = "Wikispeedia (West & Leskovec, 2012): 2007 Wikipedia Selection for Schools, its link graph and finished human games"
    url = "https://snap.stanford.edu/data/wikispeedia.html"
    license = "CC BY-SA 3.0 / GFDL (Wikipedia text); paths and graph from SNAP (cite West & Leskovec 2012; West et al. 2009)"
    tier = "C"
    criteria_source = "Rules written from the Wikispeedia game (links only, back allowed); article titles and opening text from the dataset"
    mode = "paused"
    version = 2  # version 1: pairs at 2-4+ clicks finished by 10 humans, a fixed budget of 20 clicks
    episodes = 20
    max_decisions = 30  # clicks end the episode at the pair's budget; the rest leaves room for page changes on the two largest articles
    score_name = "clicks (lower is better)"
    notes = ("20 long start/target pairs, 10 whose shortest path is 5 clicks and 10 at 6 clicks, each finished by at least one human "
             f"(pairs at 5 or more clicks are about 5% of all pairs and rarely finished), fixed sample. Budget = shortest path + {SLACK} "
             "clicks (link or back), so 7 or 8. Score = clicks used; success = the target is reached within the budget. The outcome also "
             "reports the humans' median and best click counts for the pair (over their finished games, back clicks included, as in "
             "the game), the shortest path in the link graph and the budget.")

    def reset(self, seed: int) -> None:
        self.data = data()
        tasks = self.data["tasks"]
        if not 0 <= seed < len(tasks):
            raise ValueError(f"seed {seed} out of range: {len(tasks)} tasks")
        self.task = tasks[seed]
        self.stack = [self.task["start"]]  # articles back to the start, as the browser's back button sees them
        self.path = [self._name(self.task["start"])]
        self.clicks = 0
        self.page = 0

    def _name(self, i: int) -> str:
        return self.data["titles"][i]

    def _about(self, i: int) -> str:
        return self.data["excerpts"][i]

    def _options(self) -> dict[str, str]:
        here = self.stack[-1]
        links = self.data["links"][here]
        pages = max(1, (len(links) + PAGE_SIZE - 1) // PAGE_SIZE)
        self.page = min(self.page, pages - 1)
        start = self.page * PAGE_SIZE if pages > 1 else 0
        shown = links[start : start + PAGE_SIZE] if pages > 1 else links
        options = {str(start + k + 1): self._name(j) for k, j in enumerate(shown)}
        if len(self.stack) > 1:
            options[BACK] = f"Back to the previous article ({self._name(self.stack[-2])})"
        if self.page < pages - 1:
            options[MORE] = f"Show more links of this article (it has {len(links)}; these are {start + 1}-{start + len(shown)})"
        if self.page > 0:
            options[PREVIOUS] = "Show the previous links of this article"
        return options

    def observe(self) -> DecisionRequest:
        self.current = self._options()
        here, target = self.stack[-1], self.task["target"]
        state = {
            "game": ("Wikiracing on Wikispeedia, a 2007 selection of 4,604 Wikipedia articles: get from the start article to the target "
                     "article by clicking links. A click opens one of the articles the current article links to, or goes back to the "
                     f"previous article. At most {self.task['budget']} clicks."),
            "target article": self._name(target),
            "target article, opening text": self._about(target),
            "current article": self._name(here),
            "current article, opening text": self._about(here),
            "path so far": " → ".join(self.path),
            "clicks used": f"{self.clicks} of {self.task['budget']}",
        }
        question = Question(type="choice", instructions="Which link do you click next?", criteria=self.current)
        return DecisionRequest(state=state, questions={"link": question})

    def _back(self) -> None:
        self.stack.pop()
        self.clicks += 1
        self.path.append(f"(back) {self._name(self.stack[-1])}")

    def act(self, choice: dict[str, str]) -> None:
        option = choice["link"]
        if option == MORE:
            self.page += 1
            return
        if option == PREVIOUS:
            self.page = max(0, self.page - 1)
            return
        self.page = 0
        if option == BACK:
            self._back()
            return
        article = self.data["links"][self.stack[-1]][int(option) - 1]
        self.stack.append(article)
        self.clicks += 1
        if article == self.task["target"]:
            self.path.append(self._name(article))
            return
        if not self.data["links"][article]:  # no links: the only way on is back
            self.path.append(f"{self._name(article)} (no links)")
            if self.clicks < self.task["budget"]:
                self._back()
            return
        self.path.append(self._name(article))

    def done(self) -> bool:
        return self.stack[-1] == self.task["target"] or self.clicks >= self.task["budget"]

    def default_choice(self, request: DecisionRequest) -> dict[str, str]:
        return {"link": next(iter(request.questions["link"].criteria))}  # the first listed link

    def outcome(self) -> dict[str, Any]:
        task = self.task
        return {"success": self.stack[-1] == task["target"], "score": self.clicks, "start": self._name(task["start"]),
                "target": self._name(task["target"]), "shortest_path": task["shortest_path"], "budget": task["budget"],
                "human_median_clicks": task["human_median_clicks"],
                "human_min_clicks": task["human_min_clicks"], "human_games": task["human_games"], "path": self.path}
