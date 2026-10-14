"""Leaf: next GUI action (Web / GUI 下一步动作).

The state is the task, the earlier actions and the current page or screen as text (Mind2Web's
cleaned HTML, pruned around the candidate elements as in MindAct; AndroidControl's accessibility
tree). One request asks two questions: the operation of the next action and its target element,
chosen among numbered candidates. Gold is the human demonstration.
"""

from __future__ import annotations

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ._mind2web import M2W_SPLITS, load_candidates, mind2web_html, mind2web_ranks, mind2web_rows, multichoice

LEAF = "step/next_action"

# ---------- Mind2Web ----------

M2W_CANDIDATES = 20  # the demonstrated element plus the 19 other elements ranked highest by MindAct's candidate generator
# Adapted from MindAct's multiple-choice prompt (src/action_prediction/dataloader.py, format_input_multichoice).
M2W_CONTEXT = ("Based on the HTML webpage in the state (pruned to the candidate elements and their context; the numbers are element "
               "ids), try to complete the following task. The task and the previous actions are given in the state. ")
M2W_ELEMENT = "What should be the next action? Choose the element the next action operates on."
M2W_OPERATION = "What should be the next action? Choose the operation it performs on its element."
M2W_OPS = {
    "CLICK": "CLICK: click the element (buttons, links, checkboxes, menu entries, date cells and the like; hovering and pressing Enter count as clicks)",
    "TYPE": "TYPE: type a text value into the element (a text field or search box)",
    "SELECT": "SELECT: select one of the options of the element (a drop-down list)",
}


@register_task
class Mind2WebNextAction(TaskAdapter):
    name = f"{LEAF}/mind2web"
    leaf = LEAF
    source = "Mind2Web test splits (cross-task, cross-website, cross-domain) in the human-verified Multimodal-Mind2Web release (Deng et al., 2023)"
    url = "https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web"
    license = "CC BY 4.0 (Mind2Web; the authors ask not to redistribute the unzipped test data); the Multimodal-Mind2Web card lists OpenRAIL"
    redistributable = False
    tier = "B"
    criteria_source = "Adapted from MindAct's multiple-choice prompt; operation names from the Mind2Web action space"
    notes = ("Crowdworkers' demonstrations on 137 real websites. Each item is one step: the page is the cleaned HTML pruned around "
             "20 candidate elements exactly as MindAct does (the demonstrated element plus the 19 others its candidate ranker scores "
             "highest; elements that contain or are contained in a demonstrated element are left out), with the last five previous "
             "actions. Gold is the demonstrated element and operation (HOVER and ENTER count as CLICK, as in Mind2Web); the typed or "
             "selected value is not asked. Only the HTML is used, not the screenshots. Steps whose demonstrated element was removed "
             "by Mind2Web's HTML cleaning are skipped, and so are the Mind2Web tasks that step/action_critic uses (through "
             "WebRewardBench). Loader only: Mind2Web's authors ask that the test data not be redistributed.")

    def load(self):
        from .action_critic import webrewardbench_tasks

        ranks = mind2web_ranks()
        taken = webrewardbench_tasks()  # Mind2Web tasks that step/action_critic uses
        rows = []
        for split in M2W_SPLITS:
            for r in mind2web_rows(split):
                if r["annotation_id"] in taken:
                    continue
                pos = load_candidates(r["pos_candidates"])
                neg = load_candidates(r["neg_candidates"])
                rank = ranks.get(f"{r['annotation_id']}_{r['action_uid']}")
                if not pos or not neg or rank is None:
                    continue
                ids = [c["backend_node_id"] for c in pos + neg]
                rows.append({"split": split, "row": {k: (list(v) if not isinstance(v, str) else v) for k, v in r.items()
                                                     if k not in ("pos_candidates", "neg_candidates")},
                             "pos": [c["backend_node_id"] for c in pos], "neg": [c["backend_node_id"] for c in neg],
                             "rank": {b: rank[b] for b in ids if b in rank}})
        return rows

    def label_of(self, record):
        import json

        return f"{record['split']}|{json.loads(record['row']['operation'])['op']}"

    def to_item(self, record, rng):
        import json
        import xml.etree.ElementTree as ET

        r, rank = record["row"], record["rank"]
        worst = 10**9
        positive = min(record["pos"], key=lambda b: (rank.get(b, worst), b))
        positives = set(record["pos"])
        html = mind2web_html(record["split"], r["action_uid"])
        dom = ET.fromstring(html)
        parent_of = {child: parent for parent in dom.iter() for child in parent}
        by_id = {n.get("backend_node_id"): n for n in dom.iter() if n.get("backend_node_id")}
        if positive not in by_id:
            return None
        related = set()  # ancestors and descendants of demonstrated elements: clicking them may do the same thing
        for pid in positives & set(by_id):
            node = by_id[pid]
            related.update(n.get("backend_node_id") for n in node.iter())
            up = parent_of.get(node)
            while up is not None:
                related.add(up.get("backend_node_id"))
                up = parent_of.get(up)
        negatives = [b for b in record["neg"] if b in by_id and b not in related]
        negatives.sort(key=lambda b: (rank.get(b, worst), b))
        chosen = [positive] + negatives[: M2W_CANDIDATES - 1]
        if len(chosen) < 2:
            return None
        page, choices = multichoice(html, chosen)
        shown = {bid: str(i) for bid, i, _ in choices}
        if positive not in shown:
            return None
        elements = {str(i): text for _, i, text in sorted(choices, key=lambda c: c[1])}
        operation = json.loads(r["operation"])["op"]
        index = int(r["target_action_index"])
        previous = list(r["action_reprs"])[:index][-5:]
        state = {"html": page, "task": r["confirmed_task"], "previous_actions": "\n".join(previous) if previous else "None"}
        questions = {
            "element": Question(type="choice", instructions=M2W_CONTEXT + M2W_ELEMENT, criteria=elements),
            "operation": Question(type="choice", instructions=M2W_CONTEXT + M2W_OPERATION, criteria=shuffled_options(M2W_OPS, rng)),
        }
        gold = {"element": Gold(label=shown[positive]), "operation": Gold(label=operation)}
        return self.item(state, questions, gold, split=record["split"], website=r["website"], domain=r["domain"],
                         annotation_id=r["annotation_id"], action_uid=r["action_uid"], step=index,
                         backend_node_ids={shown[b]: b for b in shown})


# ---------- AndroidControl ----------

AC_OPS = {  # action types and descriptions from the AndroidControl README
    "click": "click: click at a specific point on the screen (an element)",
    "long_press": "long_press: long press at a specific point on the screen (an element)",
    "scroll": "scroll: scroll in a direction (up, down, left or right)",
    "open_app": "open_app: open an app by its name",
    "input_text": "input_text: type text into the text field that has focus (it does not click or press enter)",
    "navigate_home": "navigate_home: go to the home screen",
    "navigate_back": "navigate_back: go back to the previous screen",
    "wait": "wait: wait a few seconds, when nothing should be done but wait for the screen to change (e.g. a loading bar)",
}
AC_CONTEXT = ("A person is operating an Android phone to reach the goal in the state; the actions taken so far and the current "
              "screen (its accessibility tree as text; interactive elements are numbered) are given. ")
AC_OPERATION = "Which type of action should come next?"
AC_TARGET = ("Which element does the next action press? For a click or a long press, choose the numbered element it presses; for any "
             "other action (scroll, typing, opening an app, back, home, wait), choose 'none'.")
AC_NONE = "none: the next action does not press an element (scroll, input_text, open_app, navigate_back, navigate_home or wait)"


def ac_screen(elements: list[dict]) -> tuple[str, dict[int, str]]:
    """Screen text with interactive elements numbered from 1, and the number of each interactive element (by index)."""
    from ._android import describe

    lines, numbers = [], {}
    for i, e in enumerate(elements):
        if e["interactive"]:
            numbers[i] = str(len(numbers) + 1)
            lines.append(f"[{numbers[i]}] {describe(e)}")
        else:
            lines.append(f"- {describe(e)}")
    return "\n".join(lines), numbers


def ac_past(action: dict, elements: list[dict] | None) -> str:
    from ._android import element_at

    kind = action.get("action_type")
    if kind in ("click", "long_press"):
        target = element_at(elements or [], action["x"], action["y"]) or element_at(elements or [], action["x"], action["y"], False)
        where = f"\"{target['label']}\" ({target['cls']})" if target and target["label"] else f"the point ({action['x']}, {action['y']})"
        return f"{kind} on {where}"
    if kind == "scroll":
        return f"scroll {action.get('direction')}"
    if kind == "input_text":
        return f"input_text \"{action.get('text')}\""
    if kind == "open_app":
        return f"open_app \"{action.get('app_name')}\""
    return str(kind)


@register_task
class AndroidControlNextAction(TaskAdapter):
    name = f"{LEAF}/androidcontrol"
    leaf = LEAF
    source = "AndroidControl official test split, human demonstrations on 833 Android apps (Li et al., 2024)"
    url = "https://github.com/google-research/google-research/tree/master/android_control"
    license = "Apache-2.0 (google-research repository); read through public conversions (ckg/AndroidControlParsed-20k: MIT; HarrytheOrange/parsed_AndroidControl: Apache-2.0)"
    tier = "B"
    criteria_source = "Action types and their descriptions from the AndroidControl README; question wording adapted"
    notes = ("Steps of the 1,543 official test episodes in the high-level setting: the model sees the episode's goal, the actions "
             "taken so far and the current screen, not the rater's instruction for the current step (kept only in the item's "
             "metadata). The screen is the accessibility tree rendered as text: interactive elements (clickable, long-clickable, editable "
             "or checkable) are numbered, other text is listed for context. The target of a click or long press is the smallest "
             "interactive element that contains the recorded point; other interactive elements containing the point are left out of "
             "the options so that exactly one is correct, and steps whose point hits no element are skipped. The official TFRecords "
             "(50 GB with screenshots) are read through two public text conversions joined by the official episode ids; their actions "
             "agree on every step used.")

    def load(self):
        from ._android import action_matches, element_at, screen_elements, test_episodes

        rows = []
        for episode in test_episodes():
            screens, actions = episode["screens"], episode["actions"]
            parsed: dict[int, list] = {}
            for step, action in enumerate(actions):
                if step not in screens or step >= len(episode["instructions"]) or not action_matches(screens[step]["action"], action):
                    continue
                parsed[step] = screen_elements(screens[step]["tree"])
                target = None
                if action["action_type"] in ("click", "long_press"):
                    target = element_at(parsed[step], action["x"], action["y"])
                    if target is None:
                        continue
                past = [ac_past(a, parsed.get(s) if s in parsed else (screen_elements(screens[s]["tree"]) if s in screens else None))
                        for s, a in enumerate(actions[:step])]
                rows.append({"episode_id": episode["episode_id"], "step": step, "goal": episode["goal"],
                             "instruction": episode["instructions"][step], "past": past, "app": screens[step]["app"],
                             "action": action, "tree": screens[step]["tree"]})
        return rows

    def label_of(self, record):
        return record["action"]["action_type"]

    def to_item(self, record, rng):
        from ._android import describe, element_at, screen_elements

        elements = screen_elements(record["tree"])
        action = record["action"]
        target = element_at(elements, action["x"], action["y"]) if action["action_type"] in ("click", "long_press") else None
        if target is not None:  # keep only the smallest interactive element containing the point
            x, y = action["x"], action["y"]
            elements = [e for e in elements if e is target or not (e["interactive"] and e["bounds"][0] <= x <= e["bounds"][2]
                                                                   and e["bounds"][1] <= y <= e["bounds"][3])]
        screen, numbers = ac_screen(elements)
        if len(numbers) > 250:
            return None
        options = {numbers[i]: describe(elements[i]) for i in sorted(numbers)}
        options["none"] = AC_NONE
        gold_target = numbers[next(i for i, e in enumerate(elements) if e is target)] if target is not None else "none"
        state = {"goal": record["goal"],
                 "previous_actions": "\n".join(f"{i + 1}. {p}" for i, p in enumerate(record["past"])) or "None",
                 "current_app": record["app"], "screen": screen}
        questions = {"action_type": Question(type="choice", instructions=AC_CONTEXT + AC_OPERATION, criteria=shuffled_options(AC_OPS, rng))}
        gold = {"action_type": Gold(label=action["action_type"])}
        if numbers:  # a screen without interactive elements (e.g. while loading) asks only for the action type
            questions["target"] = Question(type="choice", instructions=AC_CONTEXT + AC_TARGET, criteria=options)
            gold["target"] = Gold(label=gold_target)
        bounds = {numbers[i]: list(elements[i]["bounds"]) for i in sorted(numbers)}
        return self.item(state, questions, gold, episode_id=record["episode_id"], step=record["step"], action=action, option_bounds=bounds,
                         step_instruction=record["instruction"])
