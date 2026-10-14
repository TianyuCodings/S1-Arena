"""Leaf: log anomalies and alert categories (日志异常 / 根因类别).

Loghub system logs (Zhu et al., 2023) with their original labels:

- HDFS_v1: every message about one data block, labelled normal or anomalous per block.
- BGL (Blue Gene/L): windows of 40-120 consecutive lines. Every line carries an alert
  category tag assigned by the administrators' expert rules ("-" = not an alert); the tag
  column is removed from the input. One task asks whether a window contains an alert, the
  other which alert category it contains. Most lines in a window are irrelevant to the answer.
"""

from __future__ import annotations

import bisect
import csv
import io
import random
import re
import zipfile
from collections import defaultdict

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import fetch

LEAF = "struct/log_anomaly"
ZENODO = "https://zenodo.org/api/records/8196385/files/{}/content"
LICENSE = "CC BY 4.0 (Loghub, Zenodo record 8196385)"


def loghub(name: str):
    return fetch(ZENODO.format(name), "loghub_" + name)


# ---------- HDFS_v1: block sessions ----------

@register_task
class HDFSBlocks(TaskAdapter):
    name = f"{LEAF}/hdfs"
    leaf = LEAF
    source = "Loghub HDFS_v1, block-level traces labelled normal or anomalous (Xu et al., 2009)"
    url = "https://github.com/logpai/loghub/tree/master/HDFS"
    license = LICENSE
    tier = "C"
    criteria_source = "Written; labels are the dataset's per-block normal/anomaly labels (handcrafted expert rules)"
    sampling = "balanced"
    notes = ("Only 2.9% of the 575,061 blocks are anomalous, so the sample is balanced to measure both classes. "
             "The input is every raw log line that mentions the block, in log order.")
    POOL = 1500  # blocks per label read from the 1.5 GB log

    def load(self):
        archive = zipfile.ZipFile(loghub("HDFS_v1.zip"))
        with archive.open("preprocessed/anomaly_label.csv") as f:
            labels = {r["BlockId"]: r["Label"] for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))}
        rng = random.Random(f"{self.name}:pool")
        pool = {}
        for label in ("Anomaly", "Normal"):
            for block in rng.sample(sorted(b for b, l in labels.items() if l == label), self.POOL):
                pool[block] = label
        lines = defaultdict(list)
        pattern = re.compile(rb"blk_-?\d+")
        with archive.open("HDFS.log") as f:
            for raw in f:
                for block in dict.fromkeys(pattern.findall(raw)):
                    block = block.decode()
                    if block in pool:
                        lines[block].append(raw.decode("utf-8", errors="replace").rstrip("\r\n"))
        return [{"block": b, "label": pool[b], "lines": lines[b]} for b in sorted(pool) if lines[b]]

    def label_of(self, record):
        return record["label"]

    def to_item(self, record, rng):
        question = Question(
            type="boolean",
            instructions="These are all messages of a Hadoop HDFS log that mention one data block, in log order. Is this "
            "block's trace anomalous, i.e. do the messages show a problem in handling the block rather than its normal "
            "life cycle (allocation, writing and replication, verification, serving, deletion)?",
        )
        state = {"block_id": record["block"], "log": "\n".join(record["lines"])}
        gold = Gold(label="true" if record["label"] == "Anomaly" else "false")
        return self.item(state, {"anomalous": question}, {"anomalous": gold}, lines=len(record["lines"]))


# ---------- BGL: windows of consecutive lines ----------

# Alert definition verbatim from Oliner and Stearley (2007), Section 1.
ALERT = ("An alert is a message in the system logs that merits the attention of the system administrator, either because "
         "immediate action must be taken or because there is an indication of an underlying problem.")
# Category names and example message bodies from Oliner and Stearley (2007), Table 4 ("machine check interrupt" is the
# example given there for the remaining categories; the KERNSOCK example is taken from the log itself).
CATEGORIES = {
    "KERNDTLB": "Kernel alert whose messages read like 'data TLB error interrupt'.",
    "KERNSTOR": "Kernel alert whose messages read like 'data storage interrupt'.",
    "APPSEV": "Application alert whose messages read like 'ciod: Error reading message prefix after LOGIN_MESSAGE on CioStream [...]'.",
    "KERNMNTF": "Kernel alert whose messages read like 'Lustre mount FAILED : bglio11 : block_id : location'.",
    "KERNTERM": "Kernel alert whose messages read like 'rts: kernel terminated for reason 1004rts: bad message header: [...]'.",
    "KERNRTSP": "Kernel alert whose messages read like 'rts panic! - stopping execution'.",
    "KERNMC": "Kernel alert whose messages read like 'machine check interrupt'.",
    "KERNSOCK": "Kernel alert whose messages read like 'MailboxMonitor::serviceMailboxes() lib_ido_error: -1019 socket closed'.",
}
ERROR_LEVELS = {b"FATAL", b"SEVERE", b"ERROR", b"FAILURE"}
MIN_W, MAX_W = 40, 120
_BGL: dict | None = None


def _bgl() -> dict:
    """Scan BGL.log once: choose windows for both tasks, then read their lines (tag column removed)."""
    global _BGL
    if _BGL is not None:
        return _BGL
    import numpy as np

    archive = zipfile.ZipFile(loghub("BGL.zip"))
    names, tags, errors = {"-": 0}, [], []
    with archive.open("BGL.log") as f:
        for raw in f:
            parts = raw.split(b" ", 9)
            tag = parts[0].decode()
            tags.append(names.setdefault(tag, len(names)))
            errors.append(tag == "-" and len(parts) > 8 and parts[8] in ERROR_LEVELS)
    tags = np.array(tags, dtype=np.int16)
    inverse = {v: k for k, v in names.items()}
    alert_sum = np.concatenate([[0], np.cumsum(tags > 0)])
    error_sum = np.concatenate([[0], np.cumsum(np.array(errors, dtype=np.int32))])

    def count(prefix, s, w):
        return int(prefix[s + w] - prefix[s])

    taken: dict[str, list[tuple[int, int]]] = {"detect": [], "category": []}  # sorted, non-overlapping per task

    def free(task, s, w):
        spans = taken[task]
        i = bisect.bisect_left(spans, (s, s + w))
        return (i == 0 or spans[i - 1][1] <= s) and (i == len(spans) or s + w <= spans[i][0])

    def propose(task, anchors, rng, accept, want, tries):
        found = []
        for _ in range(tries):
            if len(found) >= want or len(anchors) == 0:
                break
            p = int(anchors[rng.randrange(len(anchors))])
            w = rng.randint(MIN_W, MAX_W)
            s = min(max(0, p - rng.randrange(w)), len(tags) - w)
            info = accept(s, w)
            if info is not None and free(task, s, w):
                bisect.insort(taken[task], (s, s + w))
                found.append({"start": s, "size": w, **info})
        return found

    rng = random.Random("struct/log_anomaly/bgl:pool")
    alert_pos, error_pos = np.flatnonzero(tags > 0), np.flatnonzero(np.array(errors))
    # alerts near the edge of a burst or isolated: few other alerts within half a maximal window
    nearby = alert_sum[np.minimum(alert_pos + MAX_W // 2, len(tags))] - alert_sum[np.maximum(alert_pos - MAX_W // 2, 0)]

    def positive(s, w):
        n = count(alert_sum, s, w)
        return {"label": "true", "alerts": n} if 1 <= n <= 10 else None

    def negative(s, w):
        return {"label": "false", "alerts": 0} if count(alert_sum, s, w) == 0 and count(error_sum, s, w) > 0 else None

    detect = propose("detect", alert_pos[nearby <= 10], rng, positive, 600, 50000)
    detect += propose("detect", error_pos, rng, negative, 600, 50000)

    category = []
    for code in CATEGORIES:
        ci = names[code]
        cat_sum = np.concatenate([[0], np.cumsum(tags == ci)])

        def single(s, w, ci=ci, cat_sum=cat_sum, code=code):
            n = count(cat_sum, s, w)
            return {"category": code, "alerts": n} if 1 <= n <= w // 2 and n == count(alert_sum, s, w) else None

        anchors = alert_pos[(tags[alert_pos] == ci) & (nearby <= MAX_W // 2)]
        category += propose("category", anchors, rng, single, 60, 50000)

    wanted = {}
    for w in detect + category:
        for i in range(w["start"], w["start"] + w["size"]):
            wanted[i] = None
    with archive.open("BGL.log") as f:
        for i, raw in enumerate(f):
            if i in wanted:
                wanted[i] = raw.decode("utf-8", errors="replace").rstrip("\r\n").split(" ", 1)[1]
    for w in detect + category:
        w["lines"] = [wanted[i] for i in range(w["start"], w["start"] + w["size"])]
        w["tags"] = sorted({inverse[int(t)] for t in tags[w["start"]:w["start"] + w["size"]] if t > 0})
    _BGL = {"detect": detect, "category": category}
    return _BGL


class BGLWindows(TaskAdapter):
    leaf = LEAF
    url = "https://github.com/logpai/loghub/tree/master/BGL"
    license = LICENSE
    sampling = "balanced"


@register_task
class BGLAlertDetection(BGLWindows):
    name = f"{LEAF}/bgl"
    source = "Loghub BGL, Blue Gene/L RAS log with alert tags from the administrators' rules (Oliner and Stearley, 2007)"
    tier = "B"
    criteria_source = "Alert definition verbatim from Oliner and Stearley (2007); question wording adapted"
    notes = ("Windows of 40-120 consecutive lines. Alert windows hold 1-10 alert lines; every non-alert window holds at "
             "least one non-alert line of level FATAL, SEVERE, ERROR or FAILURE, so the level alone does not decide. "
             "Balanced by design; non-overlapping windows.")

    def load(self):
        return _bgl()["detect"]

    def label_of(self, record):
        return record["label"]

    def to_item(self, record, rng):
        question = Question(type="boolean", instructions="Does this excerpt of the Blue Gene/L supercomputer's RAS log contain at least one alert? " + ALERT)
        state = {"log": "\n".join(record["lines"])}
        return self.item(state, {"alert": question}, {"alert": Gold(label=record["label"])}, alert_lines=record["alerts"], categories=record["tags"], start_line=record["start"])


@register_task
class BGLAlertCategory(BGLWindows):
    name = f"{LEAF}/bgl_category"
    source = "Loghub BGL, alert categories assigned by the administrators' rules (Oliner and Stearley, 2007)"
    tier = "B"
    criteria_source = "Category names and example messages from Oliner and Stearley (2007), Table 4; KERNSOCK example taken from the log"
    notes = ("Windows of 40-120 consecutive lines whose alert lines (at most half of the window) all belong to one of the "
             "8 categories with enough separate incidents. Balanced across categories, since raw counts mostly reflect "
             "repeated reporting of the same incident.")

    def load(self):
        return _bgl()["category"]

    def label_of(self, record):
        return record["category"]

    def to_item(self, record, rng):
        question = Question(
            type="choice",
            instructions="This excerpt of the Blue Gene/L supercomputer's RAS log contains alert messages, all of one alert "
            "category; the other lines are not alerts. " + ALERT + " Which category do the alerts belong to?",
            criteria=shuffled_options(CATEGORIES, rng),
        )
        state = {"log": "\n".join(record["lines"])}
        return self.item(state, {"category": question}, {"category": Gold(label=record["category"])}, alert_lines=record["alerts"], start_line=record["start"])
