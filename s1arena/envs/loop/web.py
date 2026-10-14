"""Web tasks in a real browser: MiniWoB++ pages driven by clicks and typing at grid coordinates.

Each MiniWoB++ task (Liu et al., 2018; the Farama Foundation's miniwob 1.1 package) is played in headless
Chrome. The state is the task instruction verbatim (MiniWoB's utterance), the page as an indented outline
of its visible elements, each led by the box of its visible part in page pixels (web_page.py), how to read
the boxes and the grid, and the actions taken so far (the model has no other memory). Actions land on
MiniWoB's 160 x 210 px page, the screen of MiniWoB's coordinate actions, divided into a grid of 5 x 5 px
cells. A request asks:

- operation: "click"; "type" when the page has a field or a list that accepts text; "scroll down" and
  "scroll up" when a scroll box of the page hides part of its content. When only a click is possible,
  this question is not asked;
- row and column (42 rows, 32 columns): the cell at whose centre the action lands. The element on top at
  that point (document.elementFromPoint) receives it, as with MiniWoB's coordinate click (CLICK_COORDS);
- text: what to type, asked when "type" is possible. A classifier cannot write free text, so this is a
  choice over the quoted phrases of the instruction, MiniWoB's own field values for the instruction (the
  TYPE_FIELD action of Liu et al., 2018) and the texts and field values shown on the page. Tasks whose
  answer text cannot be built this way (arithmetic results, a word inside a paragraph, case changes) are
  not included.

A call that gives no usable answer does nothing (the page waits); it still counts as a decision.

Version 2 (2026-09-28). Version 1 asked for one of the page's
numbered elements, MiniWoB's element click (CLICK_ELEMENT). The pages have few elements and the instruction
names the target, so a uniform-random policy succeeded in 20% or more of the episodes of ten tasks
(click-button 60%) and every model solved many tasks in every episode. Numbering every visible element made
random clicks succeed more often (a click on a container reaches the element at its centre), so that attempt
was reverted (#93). Version 2 uses MiniWoB's coordinate actions instead: the model has to locate the target
from the boxes, and a random click reaches it only in proportion to its area. The cells are 5 px because the
narrowest elements a task needs are about 8 px wide (two-letter links, page numbers), and every element at
least 5 px wide and high covers a cell centre. Content hidden in a scroll box (the email inbox, the social
media feed, the flight results, the multi-select list) can be clicked only once it is scrolled into view,
so version 2 adds MiniWoB's scroll actions; a scroll moves the box by 7/8 of its height, as Page Down does,
so the decision budgets still suffice. The options of a closed drop-down list have no place on the page, so
a list accepts typing: the option whose text is typed gets selected, as with the keyboard.
scripts/miniwob_replay.py replays the 178 successful version-1 episodes of Jev, GPT-4.1 mini and Claude
Haiku 4.5 in version 2, each chosen element turned into the cell at the centre of its box (scrolled into
view first when needed): all 178 succeed within the budgets. The random policy succeeds in 6.1% of 100
episodes per task (10.7% in version 1); it still reaches 20% on three tasks, through MiniWoB's own rules:
click-widget 44% (the first widget clicked ends the episode, and a widget takes its whole row),
navigate-tree 25% (a whole tree row is clickable, and clicks on other folders do nothing) and
click-checkboxes-soft 25% (the partial reward counts boxes rightly left unticked, so submitting nothing
often scores above 0; 1% with full credit).

The world waits for every decision (paused mode): the page's own countdown, which would end the episode
after 7-30 s of wall-clock time, is switched off right after the episode starts, and the score is
MiniWoB's raw reward (without the time discount), so a slow model loses nothing. After each action the
harness waits until the page is still (identical snapshots 50 ms apart, no jQuery animation and no
pending timeout of up to 2 s); the "-nodelay" variants, which MiniWoB provides without animations, are
used where they exist.

Outcome: the raw reward of the page when it ends the episode, success = reward > 0 and full_credit =
reward 1. Most tasks give 1 or -1; partial rewards: the click-checkboxes tasks give (right - wrong boxes) /
boxes, phone-book 0.4 for the right person's wrong detail, buy-ticket -0.5 for a ticket between the
extremes, tic-tac-toe -0.5 for a draw and -0.75 for a loss. In the two checkbox tasks and phone-book, whose
partial credit a random policy collects (submitting nothing counts the boxes rightly left unticked),
success = full credit (version 3 of those tasks). An episode the page has not ended when the
decision budget runs out scores -1, as MiniWoB scores a timed-out episode.

Determinism: the page is seeded with Math.seedrandom(seed) (the same problem as the miniwob gym for that
seed), every episode starts in a fresh browser (no state leaks between episodes), the time zone and locale
are fixed (UTC, en-US), and actions are dispatched as DOM events (a click at a point is a click event on the
element there, a scroll sets the box's scroll position), so no pointer hovers anywhere.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ...registry import register_env
from ...schema import DecisionRequest, Question
from ..base import Environment
from .web_page import CLICK_JS, IMAGES_READY_JS, PRELOAD_JS, SCROLL_JS, SELECT_TEXT_JS, SNAPSHOT_JS, TIMERS_JS, accepts_text, render, short, typing_candidates

URL = "https://miniwob.farama.org/"
LICENSE = "MIT"
SOURCE = "MiniWoB++ (Liu et al., 2018), miniwob 1.1 package by the Farama Foundation"
DATA_MODE = "train"  # MiniWoB's default. In "test" mode the natural-language email tasks use held-out instruction
# templates and no longer hand over their field values, so the recipient could not be offered as a typing candidate.
TIMEZONE, LOCALE = "UTC", "en-US"
SETTLE_POLL_S, SETTLE_MAX_S = 0.05, 3.0

START_JS = f"""
core.endEpisode(0);
Math.seedrandom(arguments[0]);
core.setDataMode({json.dumps(DATA_MODE)});
core.startEpisodeReal();
"""
# Paused mode: stop the countdown that would end the episode after EPISODE_MAX_TIME ms of wall-clock time.
# endEpisode() still works because EP_TIMER stays non-null.
PAUSE_TIMER_JS = "clearTimeout(core.EP_TIMER); core.EP_TIMER = -1; clearInterval(core.CD_TIMER);"
UTTERANCE_JS = "return core.getUtterance();"
METADATA_JS = "return {done: WOB_DONE_GLOBAL, raw_reward: WOB_RAW_REWARD_GLOBAL, reason: WOB_REWARD_REASON};"

AREA_W, AREA_H = 160, 210  # MiniWoB's page (#wrap), the screen of its coordinate actions (miniwob.constants.TASK_WIDTH, TASK_HEIGHT)
CELL = 5  # px: the narrowest elements a task needs are about 8 px wide
ROWS = {str(r): f"y {CELL * (r - 1)} to {CELL * r}" for r in range(1, AREA_H // CELL + 1)}
COLUMNS = {str(c): f"x {CELL * (c - 1)} to {CELL * c}" for c in range(1, AREA_W // CELL + 1)}
LAYOUT = (f"The page is {AREA_W} x {AREA_H} px, with x measured rightwards and y downwards from its top-left corner; the task instruction "
          "fills y 0 to 50. The page lists its visible elements in document order, indented under their parent: the box of the element's "
          "visible part (x, y, width, height), tag[input type]#id.classes, the visible text in quotes, then the element's state. An element "
          "out of view (scrolled away in a scroll box, or outside the page) shows the box it would take and cannot be reached until it is "
          "scrolled into view; the options of a closed drop-down list have no box. "
          f"Every action lands at the centre of one cell of a grid of {CELL} x {CELL} px cells: row r covers y from {CELL}(r - 1) to {CELL}r and "
          f"column c covers x from {CELL}(c - 1) to {CELL}c, so the action lands at x = {CELL}c - {CELL / 2:g}, y = {CELL}r - {CELL / 2:g}, on "
          "whatever element is on top at that point.")

OPERATIONS = {
    "click": "Click at the chosen cell. Clicking an option of a list selects it; in a list that allows several selections, clicking an option toggles it.",
    "type": ("Click at the chosen cell to focus the element there, then type the chosen text, replacing any text the element already contains. "
             "Typing the text of an option into a list selects that option."),
    "scroll down": "Scroll the scroll box under the chosen cell down by 7/8 of its visible height, as the Page Down key does.",
    "scroll up": "Scroll the scroll box under the chosen cell up by 7/8 of its visible height, as the Page Up key does.",
}
NO_OP = "none"  # used only when a model gives no usable answer: the page is left as it is


class MiniWoBTask(Environment):
    leaf = "loop/web"
    source = SOURCE
    url = URL
    license = LICENSE
    tier = "A"
    criteria_source = "Task instruction verbatim (MiniWoB utterance); page with element boxes, grid, actions and typing candidates rendered by the fixed harness"
    mode = "paused"
    episodes = 5
    score_name = "reward"
    version = 2
    task = ""  # MiniWoB task id, e.g. "click-button"
    full_credit_only = False  # success needs reward 1 (FULL_CREDIT_TASKS), not only a reward above 0

    def reset(self, seed: int) -> None:
        self.close()
        self.seed = seed
        self.history: list[str] = []
        self.ended = False
        self.metadata: dict[str, Any] = {}
        self.driver = _start_browser()
        try:
            caps = self.driver.capabilities
            self.browser = f"{caps.get('browserName', '')} {caps.get('browserVersion', '')}".strip()
            self._start_episode(seed)
        except BaseException:
            self.close()
            raise

    def _start_episode(self, seed: int) -> None:
        from miniwob.fields import get_field_extractor
        from miniwob.selenium_instance import HTML_DIR

        self.driver.get(f"file://{HTML_DIR}/miniwob/{self.task}.html")
        _wait(lambda: self.driver.execute_script("return !!document.getElementById('sync-task-cover') && typeof genProblem === 'function';"), "page load")
        self.driver.execute_script(TIMERS_JS)
        self.driver.execute_script(PRELOAD_JS)
        _wait(lambda: self.driver.execute_script(IMAGES_READY_JS), "images")
        self.driver.execute_script(START_JS, seed)
        _wait(lambda: self.driver.execute_script("return WOB_TASK_READY;"), "task start")
        self.driver.execute_script(PAUSE_TIMER_JS)
        utterance = self.driver.execute_script(UTTERANCE_JS)
        if isinstance(utterance, dict):  # natural-language tasks hand their fields over with the utterance
            self.utterance, self.field_values = utterance["utterance"], [str(v) for v in utterance.get("fields", {}).values()]
        else:
            self.utterance = utterance
            try:
                self.field_values = [str(v) for _, v in get_field_extractor(self.task)(utterance)]
            except (KeyError, ValueError):
                self.field_values = []
        self._settle()

    def _settle(self) -> None:
        """Wait until the page is still, keep its last snapshot and whether the page ended the episode."""
        previous, deadline = None, time.monotonic() + SETTLE_MAX_S
        while True:
            snap = self.driver.execute_script(SNAPSHOT_JS, AREA_W, AREA_H)
            if (snap["nodes"] == previous and not snap["busy"]) or time.monotonic() > deadline:
                break
            previous = snap["nodes"]
            time.sleep(SETTLE_POLL_S)
        self.nodes = snap["nodes"]
        self.page = render(self.nodes)
        self.texts = typing_candidates(self.utterance, self.field_values, self.nodes) + [""]
        typing = any(accepts_text(node) for node in self.nodes) and len(self.texts) >= 2
        scrolling = any(node.get("scroll_up") or node.get("scroll_down") for node in self.nodes)
        self.operations = [op for op, possible in (("click", True), ("type", typing), ("scroll down", scrolling), ("scroll up", scrolling)) if possible]
        self.metadata = self.driver.execute_script(METADATA_JS)
        self.ended = bool(self.metadata.get("done"))

    def observe(self) -> DecisionRequest:
        history = "\n".join(f"{i}. {h}" for i, h in enumerate(self.history, 1)) or "none yet"
        state = {"task": self.utterance, "page": self.page, "layout": LAYOUT, "actions so far": history}
        questions = {}
        if len(self.operations) > 1:
            questions["operation"] = Question(type="choice", criteria={op: OPERATIONS[op] for op in self.operations}, instructions=(
                "Which action to take next on the web page to accomplish the task? It lands at the centre of the grid cell chosen in \"row\" "
                "and \"column\"" + ("; \"type\" also uses the text chosen in \"text\"." if "type" in self.operations else ".")))
        what = "act" if len(self.operations) > 1 else "click"
        questions["row"] = Question(type="choice", criteria=ROWS, instructions=(
            f"Where to {what} next on the web page to accomplish the task: in which row of the grid? The action lands at the centre of "
            "the cell in this row and the chosen column (see \"layout\")."))
        questions["column"] = Question(type="choice", criteria=COLUMNS, instructions=(
            f"Where to {what} next on the web page to accomplish the task: in which column of the grid? The action lands at the centre of "
            "the cell in this column and the chosen row (see \"layout\")."))
        if "type" in self.operations:
            texts = {f"t{k}": json.dumps(t, ensure_ascii=False) if t else "(nothing: the element's text is cleared)" for k, t in enumerate(self.texts, 1)}
            questions["text"] = Question(type="choice", criteria=texts, instructions=(
                "Which text to type, if the action is \"type\" (ignored otherwise)? The options are the quoted phrases "
                "and values of the task instruction and the texts and values shown on the page."))
        return DecisionRequest(state=state, questions=questions)

    def default_choice(self, request: DecisionRequest) -> dict[str, str]:
        """No usable answer: do nothing this step (a failed call never clicks or types anything)."""
        return {name: NO_OP for name in request.questions}

    def act(self, choice: dict[str, str]) -> None:
        operation = choice.get("operation", "click")  # pages where only a click is possible do not ask for the operation
        row, column, text_id = choice.get("row", NO_OP), choice.get("column", NO_OP), choice.get("text", NO_OP)
        if operation not in self.operations or NO_OP in (row, column) or (operation == "type" and text_id == NO_OP):
            self.history.append("(no action)")
            return
        x, y, where = (int(column) - 0.5) * CELL, (int(row) - 0.5) * CELL, f"at row {row}, column {column}"
        if operation.startswith("scroll"):
            box = self.driver.execute_script(SCROLL_JS, x, y, 1 if operation == "scroll down" else -1)
            self.history.append(f"{operation} {short(self.nodes[box])} {where}" if box is not None else f"{operation} {where}: nothing to scroll there")
        else:
            what = self._name(self.driver.execute_script(CLICK_JS, x, y))
            if operation == "type":
                text = self.texts[int(text_id[1:]) - 1]
                self._type(text)
                self.history.append(f"type {json.dumps(text, ensure_ascii=False)} into {what} {where}" if text else f"clear {what} {where}")
            else:
                self.history.append(f"click {what} {where}")
        self._settle()

    def _name(self, hit: int | str | None) -> str:
        """The element a click reached, as named in the list of past actions."""
        if hit == "instruction":
            return "the task instruction"
        return short(self.nodes[hit]) if isinstance(hit, int) else "the page background"

    def _type(self, text: str) -> None:
        from selenium.webdriver.common.action_chains import ActionChains
        from selenium.webdriver.common.keys import Keys

        if self._done_now():  # the click ended the episode
            return
        selected = self.driver.execute_script(SELECT_TEXT_JS)
        chain = ActionChains(self.driver, duration=0)
        if not text:
            if selected:
                chain.send_keys(Keys.BACKSPACE).perform()
            return
        for key in text:  # one key press per character, as MiniWoB's typing action
            chain.key_down(key).key_up(key)
        chain.perform()

    def _done_now(self) -> bool:
        return bool(self.driver.execute_script("return WOB_DONE_GLOBAL;"))

    def done(self) -> bool:
        return self.ended

    def outcome(self) -> dict[str, Any]:
        reason = self.metadata.get("reason")
        if self.ended:
            reward = float(self.metadata["raw_reward"])
            full = reward >= 1.0
            out = {"success": full if self.full_credit_only else reward > 0, "score": reward, "full_credit": full, "ended_by": "page"}
        else:  # decision budget exhausted: scored like a MiniWoB episode that timed out
            out = {"success": False, "score": -1.0, "full_credit": False, "ended_by": "decision limit"}
        if isinstance(reason, (str, int, float)):
            out["reason"] = reason
        return {**out, "actions": self.history, "instruction": self.utterance, "browser": self.browser}

    def close(self) -> None:
        driver = getattr(self, "driver", None)
        self.driver = None
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass

    def card(self) -> dict[str, Any]:
        return {**super().card(), "miniwob_task": self.task}


def _start_browser():
    from selenium import webdriver

    options = webdriver.ChromeOptions()
    for arg in ("--headless=new", "--window-size=800,600", "--force-device-scale-factor=1", f"--lang={LOCALE}", "--disable-gpu",
                "--disable-extensions", "--no-first-run", "--no-default-browser-check", "--disable-background-networking",
                "--disable-features=Translate,OptimizationHints,MediaRouter", "--mute-audio"):
        options.add_argument(arg)
    driver = webdriver.Chrome(options=options)
    try:
        driver.execute_cdp_cmd("Emulation.setTimezoneOverride", {"timezoneId": TIMEZONE})
        driver.execute_cdp_cmd("Emulation.setLocaleOverride", {"locale": LOCALE})
    except BaseException:
        driver.quit()
        raise
    return driver


def _wait(condition, what: str, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError(f"MiniWoB page: {what} took longer than {timeout_s:.0f} s")
        time.sleep(0.05)


# (MiniWoB task, decision budget, note). Budgets are about twice an efficient solution plus a few steps.
TASKS: list[tuple[str, int, str]] = [
    ("click-button", 5, "Click the button with a given label among buttons, text fields and text."),
    ("click-button-sequence", 6, "Click two buttons in order."),
    ("click-checkboxes-large", 25, "Tick the named checkboxes (5-12 of up to 12 boxes) and submit; partial credit."),
    ("click-checkboxes-soft", 12, "Tick the checkboxes whose words are similar in meaning to the given ones and submit; partial credit."),
    ("click-collapsible-2-nodelay", 10, "Expand accordion sections to find and click a link."),
    ("click-dialog-2", 5, "Click the named button of a dialog."),
    ("click-option", 6, "Pick the named radio button and submit."),
    ("click-scroll-list", 12, "Select the named items in a list that allows several selections and submit."),
    ("click-tab-2-hard", 10, "Switch tabs to find and click a link."),
    ("click-widget", 5, "Click a widget of the given kind (button, checkbox, text field, ...)."),
    ("choose-list", 6, "Choose the named option of a drop-down list and submit."),
    ("choose-date-nodelay", 30, "Pick a date in 2016 with a date picker (month by month) and submit."),
    ("enter-password", 8, "Type the password into both fields and submit."),
    ("enter-date", 6, "Type a date into a date field and submit."),
    ("login-user-popup", 12, "Log in with a user name and password; a popup may interrupt."),
    ("multi-layouts", 10, "Fill a search form whose fields appear in varying layouts and submit."),
    ("form-sequence-2", 8, "Check a radio button, type a number into one of three fields and submit."),
    ("copy-paste-2", 6, "Copy the text of one of several text areas into a field and submit."),
    ("read-table-2", 8, "Fill two fields with values read from a table and submit."),
    ("use-autocomplete-nodelay", 8, "Enter an item with given start and end letters with the help of an autocomplete list."),
    ("email-inbox", 10, "Delete, star, reply to or forward the email of a given sender."),
    ("email-inbox-forward-nl-turk", 10, "Forward an email; the instruction was written by crowd workers."),
    ("social-media", 6, "Click an action (reply, retweet, like, or a menu item) on a given user's post."),
    ("social-media-some", 20, "Click an action on a given number of a user's posts and submit."),
    ("search-engine", 10, "Search for a term and click the n-th result, across result pages."),
    ("navigate-tree", 10, "Expand folders of a file tree to find and click a named file or folder."),
    ("book-flight-nodelay", 25, "Book the cheapest or shortest flight: two autocomplete fields, a date picker and a result list."),
    ("phone-book", 15, "Page through a contact book to find a person and click their phone, email or address; partial credit."),
    ("buy-ticket", 5, "Buy the ticket with the lowest or highest price or the shortest or longest duration."),
    ("tic-tac-toe", 10, "Win a game of tic-tac-toe against the page (draw -0.5, loss -0.75)."),
]


# Tasks whose partial credit a random policy collects: the checkbox tasks count boxes rightly left unticked (submitting
# nothing often scores above 0), phone-book gives 0.4 for a wrong detail of the right person. Success = full credit
# there (version 3, as the success rule changed).
FULL_CREDIT_TASKS = {"click-checkboxes-large", "click-checkboxes-soft", "phone-book"}


def _register(task: str, max_decisions: int, note: str) -> None:
    attrs = {"name": f"loop/web/miniwob/{task}", "task": task, "max_decisions": max_decisions,
             "source": f"MiniWoB++ {task} (Liu et al., 2018; miniwob 1.1)", "notes": note, "__doc__": f"MiniWoB++ task {task}: {note}"}
    if task in FULL_CREDIT_TASKS:
        attrs.update(full_credit_only=True, version=3, notes=f"{note} Success = full credit (reward 1).")
    register_env(type("MiniWoB_" + task.replace("-", "_"), (MiniWoBTask,), attrs))


for _task, _budget, _note in TASKS:
    _register(_task, _budget, _note)
