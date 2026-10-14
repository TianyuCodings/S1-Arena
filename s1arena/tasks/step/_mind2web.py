"""Mind2Web helpers: the MindAct DOM pruning and text rendering (src/data_utils/dom_utils.py of
OSU-NLP-Group/Mind2Web, MIT), ported from lxml to the standard library's ElementTree, and a reader for
the test splits of Multimodal-Mind2Web that skips the screenshot columns."""

from __future__ import annotations

import copy
import json
import pickle
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from ..common import CACHE

M2W_REPO = "osunlp/Multimodal-Mind2Web"
M2W_SCORES = ("osunlp/Mind2Web", "scores_all_data.pkl")  # MindAct candidate-generation (DeBERTa ranker) scores and ranks
M2W_SPLITS = ("test_task", "test_website", "test_domain")
M2W_COLUMNS = ["action_uid", "cleaned_html", "operation", "pos_candidates", "neg_candidates", "website", "domain", "subdomain",
               "annotation_id", "confirmed_task", "action_reprs", "target_action_index", "target_action_reprs"]


def _split_cache(split: str) -> Path:
    """One Multimodal-Mind2Web test split without the screenshot and raw HTML columns, as a local parquet file sorted by
    action_uid in small row groups, so that one step's HTML can be read without loading the whole split."""
    path = CACHE / f"multimodal_mind2web_{split}.parquet"
    if not path.exists():
        import pyarrow as pa
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi, HfFileSystem

        from ...env import load_env

        load_env()
        files = sorted(f for f in HfApi().list_repo_files(M2W_REPO, repo_type="dataset") if f.startswith(f"data/{split}-"))
        fs, tables = HfFileSystem(), []
        for name in files:
            with fs.open(f"datasets/{M2W_REPO}/{name}", block_size=8 * 2**20) as handle:
                tables.append(pq.read_table(handle, columns=M2W_COLUMNS, pre_buffer=True))
        table = pa.concat_tables(tables).sort_by("action_uid")
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".part")
        pq.write_table(table, tmp, row_group_size=32)
        tmp.rename(path)
    return path


def mind2web_rows(split: str) -> list[dict]:
    """Rows of a test split without the HTML (see mind2web_html)."""
    import pyarrow.parquet as pq

    columns = [c for c in M2W_COLUMNS if c != "cleaned_html"]
    return pq.read_table(_split_cache(split), columns=columns).to_pylist()


def mind2web_html(split: str, action_uid: str) -> str:
    import pyarrow.parquet as pq

    table = pq.read_table(_split_cache(split), columns=["cleaned_html"], filters=[("action_uid", "=", action_uid)])
    return table.column("cleaned_html")[0].as_py()


class _NoGlobals(pickle.Unpickler):
    """The scores file holds only dicts, lists, strings and numbers; refuse anything else."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"global {module}.{name} is not allowed")


def mind2web_ranks() -> dict[str, dict[str, int]]:
    """Rank of every candidate element by MindAct's candidate generator, keyed by '<annotation_id>_<action_uid>'."""
    from ._sources import hub_file

    with open(hub_file(*M2W_SCORES), "rb") as f:
        data = _NoGlobals(f).load()
    return {key: dict(value) for key, value in data["ranks"].items()}


# ---- MindAct DOM utilities (ElementTree port) ----


def _parents(root: ET.Element) -> dict:
    return {child: parent for parent in root.iter() for child in parent}


def _descendants(node: ET.Element, max_depth: int, depth: int = 0) -> list:
    if depth > max_depth:
        return []
    out = []
    for child in node:
        out.append(child)
        out.extend(_descendants(child, max_depth, depth + 1))
    return out


def prune_tree(dom: ET.Element, candidates: set[str], max_depth: int = 5, max_children: int = 50, max_sibling: int = 3) -> ET.Element:
    parent_of = _parents(dom)
    by_id = {n.get("backend_node_id"): n for n in dom.iter() if n.get("backend_node_id")}
    keep: set[str] = set()
    for cid in candidates:
        node = by_id[cid]
        keep.add(cid)
        up = parent_of.get(node)
        while up is not None:  # ancestors
            keep.add(up.get("backend_node_id", ""))
            up = parent_of.get(up)
        keep.update([x.get("backend_node_id", "") for x in _descendants(node, max_depth)][:max_children])
        parent = parent_of.get(node)
        if parent is not None:
            siblings = [x for x in parent if x.tag != "text"]
            i = siblings.index(node)
            keep.update(x.get("backend_node_id", "") for x in siblings[max(0, i - max_sibling): i + max_sibling + 1])
    tree = copy.deepcopy(dom)
    parent_of = _parents(tree)
    for node in list(tree.iter())[::-1]:
        parent = parent_of.get(node)
        if node.tag != "text":
            is_keep = node.get("backend_node_id", "") in keep
            is_candidate = node.get("backend_node_id", "") in candidates
        else:
            is_keep = parent is not None and parent.get("backend_node_id", "") in keep
            is_candidate = parent is not None and parent.get("backend_node_id", "") in candidates
        if not is_keep and parent is not None:
            parent.remove(node)
            continue
        if not is_candidate or node.tag == "text":
            node.attrib.pop("backend_node_id", None)
        if (len(node.attrib) == 0 and not any(x.tag == "text" for x in node) and parent is not None
                and node.tag != "text" and len(node) <= 1):
            index = list(parent).index(node)
            for offset, child in enumerate(list(node)):  # insert all children into parent
                parent.insert(index + offset, child)
                parent_of[child] = parent
            parent.remove(node)
    return tree


_ATTRS = ["role", "aria_role", "type", "alt", "aria_description", "aria_label", "label", "title", "name", "text_value", "value",
          "placeholder", "input_checked", "input_value", "option_selected", "class"]


def _attribute_repr(node: ET.Element, max_value_length: int = 5, max_length: int = 20) -> None:
    seen, values = set(), ""
    for attr in _ATTRS:
        if node.get(attr) is None:
            continue
        value = node.get(attr).lower()
        if value in ("hidden", "none", "presentation", "null", "undefined") or value.startswith("http"):
            continue
        value = " ".join([v for v in value.split() if len(v) < 15][:max_value_length])
        if value and value not in seen:
            seen.add(value)
            values += value + " "
    uid = node.get("backend_node_id", "")
    node.attrib.clear()
    if uid:
        node.set("id", uid)
    if values:
        node.set("meta", " ".join(values.split()[:max_length]))


_ESCAPES = [("&quot;", '"'), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&nbsp;", " "), ("&ndash;", "-"), ("&rsquo;", "'"),
            ("&lsquo;", "'"), ("&ldquo;", '"'), ("&rdquo;", '"'), ("&#39;", "'"), ("&#40;", "("), ("&#41;", ")")]


def tree_repr(tree: ET.Element, id_mapping: dict[str, int], max_value_length: int = 5, max_length: int = 20) -> str:
    """MindAct's get_tree_repr with keep_html_brackets=False: '(button id=3 search (span search))'."""
    tree = copy.deepcopy(tree)
    for node in tree.iter():
        if node.tag != "text":
            if "backend_node_id" in node.attrib:
                bid = node.get("backend_node_id")
                if bid not in id_mapping:
                    id_mapping[bid] = len(id_mapping)
                node.set("backend_node_id", str(id_mapping[bid]))
            _attribute_repr(node, max_value_length, max_length)
        else:
            node.text = " ".join((node.text or "").split()[:max_length])
    text = ET.tostring(tree, encoding="unicode")
    text = text.replace(" />", "/>")  # ElementTree writes '<a />', lxml '<a/>'
    text = text.replace('"', " ")
    text = text.replace("meta= ", "").replace("id= ", "id=").replace(" >", ">")
    text = re.sub(r"<text>(.*?)</text>", r"\1", text)
    text = text.replace("/>", "$/$>")
    text = re.sub(r"</(.+?)>", r")", text)
    text = re.sub(r"<(.+?)>", r"(\1", text)
    text = text.replace("$/$", ")")
    for k, v in _ESCAPES:
        text = text.replace(k, v)
    return re.sub(r"\s+", " ", text).strip()


def multichoice(cleaned_html: str, candidate_ids: list[str]) -> tuple[str, list[tuple[str, int, str]]]:
    """MindAct's format_input_multichoice: the pruned page and, per candidate in page order, (backend id, shown id, choice text)."""
    dom = ET.fromstring(cleaned_html)
    pruned = prune_tree(dom, set(candidate_ids))
    mapping: dict[str, int] = {}
    page = tree_repr(pruned, mapping)
    choices = []
    for node in pruned.iter():
        bid = node.get("backend_node_id")
        if bid:
            choices.append((bid, mapping[bid], " ".join(tree_repr(node, mapping).split()[:10])))
    return page, choices


def load_candidates(raw: list) -> list[dict]:
    return [json.loads(c) if isinstance(c, str) else c for c in raw]
