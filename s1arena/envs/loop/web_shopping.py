"""Online shopping in WebShop (Yao et al., 2022), text mode: search, open products, pick options, buy.

WebShop simulates a shopping website with 1,181,430 products scraped from amazon.com and 12,087 instructions written by
crowd workers, each for one of the products ("i need a long clip-in hair extension which is natural looking, and price
lower than 30.00 dollars"). The agent searches, opens products, selects options (size, color, ...) and buys; WebShop's
reward compares the purchase with the instruction. Nothing in the harness calls a language model.

Simulator. WebShop's own code at commit 64fa2a5 (princeton-nlp/WebShop, MIT): its text environment (WebAgentTextEnv,
observation mode "text") over one SimServer that holds the full product set, the human-instruction goals and the Lucene
BM25 index built by WebShop's own search_engine scripts. WebShop pins the packages its search and reward depend on
(pyserini 0.17 with a JDK 11, spaCy 3.3 with en_core_web_sm 3.3.0, thefuzz 0.19, Flask 2.1), which do not install on
Python 3.12, so WebShop runs in its own Python 3.8 environment behind web_shopping_server.py, a small HTTP server on
127.0.0.1. This module talks to it at $S1ARENA_WEBSHOP_URL (default http://127.0.0.1:3000) and refuses a server whose
500 test goals differ from the expected ones (TEST_GOALS_SHA256).

State (plain strings):
- website: what WebShop is and how a turn works, one paragraph (no strategy);
- instruction: the goal's instruction verbatim, WebShop's price clause included;
- page: WebShop's text observation of the current page verbatim (the visible texts joined by " [SEP] "), minus the
  "Instruction: [SEP] ..." header that repeats the instruction;
- selected options (product pages): the options chosen so far for the product, from WebShop's session (the site shows
  them as pressed buttons, which the text observation leaves out);
- actions so far: the WebShop actions taken, a product click with the product's title.

Action: one choice question per decision; each option is one WebShop action valid on the current page, in WebShop's
syntax:
- search page: search[query] for each search candidate (below);
- other pages: click[...] for each clickable WebShop lists for the page (get_available_actions: Back to Search,
  < Prev, Next >, Description, Features, Reviews, Buy Now, the product links of a results page, the option buttons of
  a product page), in page order; a product link is shown with the product's title and price, an option button with
  its option's name. On a product page with more than 255 clickables (a few products have hundreds of sizes or colors)
  the option buttons come in lists, turned with "more"/"previous" options, while the page's other buttons stay in every
  list; turning the list is a decision but not a WebShop action.
A decision without a usable answer does nothing (WebShop's own treatment of an invalid action: the page stays); it still
counts as a decision.

Search. The query is WebShop's only free-text action and a classifier cannot write text, so the search page offers
queries derived from the instruction by fixed rules, in this order (duplicates dropped, so 2 to 4 options):
1. the instruction verbatim: the query of WebShop's rule baseline, and the query WebShop's IL and RL agents always keep
   among their candidates (baseline_models/env.py);
2. the instruction without its price clause (", and price lower than X dollars");
3. (2) without the request that opens it ("i am looking for a", "i would like", "can you find me", ...);
4. (3) up to its first punctuation mark, " - " or clause word (with, that, which, who, whose, where, for, in, without,
   made, so, because), roughly the product phrase.
The harness never searches on its own initiative: which query to run, and whether to go back and search again with
another, is the model's decision, as WebShop's IL and RL agents choose among candidate queries. (If the rules gave a
single query it would be run without asking, as a forced move; none of the 500 test instructions does.) A search
returns up to 50 products, 10 per results page (WebShop's defaults).

Outcome: WebShop's reward for the purchase (get_reward: the share of the instruction's attributes, options and price
cap the purchase meets, times a product-type factor; 0 to 1) is the score, and success = reward 1 (WebShop's success
rate). Without a purchase when the decision budget runs out the score is 0, as in WebShop. The outcome also reports the
reward's components, the product and options bought, whether it is the instruction's own product, and the actions.

Episodes: seed i is WebShop's test goal i (the test split is goals 0-499 of WebShop's fixed shuffle, as its baseline
agents use it); the benchmark plays the first 100. Budget: 15 decisions, the step limit of ReAct's WebShop runs; a
direct purchase takes 3 to 7 decisions here (a search, a product, 0 to 4 options, Buy Now), so 15 leaves room to read
a description, turn a results page, go back or search again, but not to browse without end.

Determinism: the product, attribute and instruction files and the index are fixed; the server seeds the random draws
WebShop makes while loading (price caps; prices of products listed with a price range); the index is built
single-threaded, as WebShop's run_indexing.sh does, so documents keep the file's order and BM25 ties break the same
way; WebShop is otherwise deterministic given the actions. The same seed and decisions give the same episode.

Setup (on the machine that runs the server):
- WebShop at commit 64fa2a5 and a Python 3.8 environment with its requirements.txt pins (plus faiss-cpu 1.7.2, which
  its setup.sh installs with conda, Werkzeug 2.1.2 for Flask 2.1.2, and spaCy's en_core_web_sm 3.3.0, which its
  goal.py loads); a JDK 11 on JAVA_HOME (pyserini);
- data from the Hugging Face mirror YWZBrandon/webshop-data at revision ce990ff (the Google Drive links in WebShop's
  setup.sh are gone): items_shuffle.json (5.48 GB), items_ins_v2.json (186 MB), items_human_ins.json (5.1 MB);
- the index: WebShop's search_engine/convert_product_file_format.py with the full files, then its pyserini command for
  "indexes" (run_indexing.sh), into <WebShop>/search_engine/indexes (3-4 GB);
- the server: <WebShop python> s1arena/envs/loop/web_shopping_server.py --webshop <WebShop> --data <data>; it takes
  about 2 minutes and 15-19 GB of memory to load.
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.request
from typing import Any

from ...registry import register_env
from ...schema import MAX_OPTIONS, DecisionRequest, Question
from ..base import Environment

URL = "https://github.com/princeton-nlp/WebShop"
WEBSHOP_COMMIT = "64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd"
DATA_MIRROR = "https://huggingface.co/datasets/YWZBrandon/webshop-data (revision ce990fff5aee388db2706f07820c578ab68e0453)"
SERVER_VAR = "S1ARENA_WEBSHOP_URL"
DEFAULT_SERVER = "http://127.0.0.1:3000"
REQUEST_TIMEOUT_S = 300
PRODUCTS = 1_181_430  # after WebShop's load_products drops duplicate ASINs
TEST_GOALS = 500
TEST_GOALS_SHA256 = "6621d3af242515335833bfab1f41098e19f8de9674a81fcf56daf93a3a4a6ecf"  # web_shopping_server.goals_fingerprint(goals[:500])
NO_OP = "none"  # a decision without a usable answer: nothing is done
MORE, PREVIOUS = "more", "previous"
TITLE_CHARS = 80  # a product's title in the list of past actions

MAX_DECISIONS = 15
RULES = ("WebShop, a simulated shopping website with 1.18 million products from amazon.com. You shop for the product the "
         "instruction describes. Each turn you take one action on the current page: on the search page you search; on the "
         "other pages you click one of the page's buttons, product links or product options (clicking an option selects it "
         "for the product shown). Buy Now buys the product shown with the selected options and ends the episode; the episode "
         f"also ends after {MAX_DECISIONS} decisions without a purchase.")
QUESTION = ("Which action do you take next? Each option is a WebShop action on the current page: search[query] searches "
            "for the query; click[...] clicks a button, a product link (shown with the product's title and price) or an "
            "option button of the product (shown with the option's name).")

# Search candidates. WebShop appends the price clause itself: ", and price lower than {price_upper:.2f} dollars".
PRICE_CLAUSE = re.compile(r",? and price lower than \d+(?:\.\d+)? dollars$")
GREETING = re.compile(r"^(?:hey|hi|hello|ok|okay|please)\b[\s,.!]*")
REQUESTS = sorted((  # phrases that open a request; the longest match is removed
    "i am looking", "i'm looking", "im looking", "i am searching", "i'm searching", "im searching", "i am shopping",
    "i'm shopping", "i am trying", "i'm trying", "im trying", "i am hoping", "i'm hoping", "i am interested in",
    "i'm interested in", "im interested in", "i am in need of", "i'm in need of", "i would like", "i'd like", "id like",
    "i would love", "i'd love", "i want", "i wanna", "i need", "i nee", "i require", "i must have", "can you find",
    "can you get", "can you help me find", "could you find", "could you get", "can i get", "could i get", "can i have",
    "may i have", "please find", "please get", "help me find", "find", "get", "buy", "order", "locate", "search for",
    "show", "look for", "looking for", "searching for", "shopping for", "need",
), key=len, reverse=True)
FILLERS = ("to buy", "to find", "to get", "to purchase", "to order", "to have", "to see", "to try", "buying", "finding",
           "getting", "purchasing", "for", "me", "a", "an", "the", "some")  # removed after the request, repeatedly
HEAD_END = re.compile(r"[,.;:!?](?=\s|$)|\s-\s|\s(?:with|that|which|who|whose|where|for|in|without|made|so|because)\s")
SPECIAL = ("<r>", "<a>", "<c>", "<q>")  # WebShop reads a query starting with these as a command (random items, attribute...)


def _clean(text: str) -> str:
    """Lowercase, without square brackets (WebShop's action parser splits on them), single spaces."""
    return " ".join(re.sub(r"[\[\]]", " ", text.lower()).split())


def _strip_request(text: str) -> str:
    text = GREETING.sub("", text)
    for phrase in REQUESTS:
        if text.startswith(phrase + " "):
            text = text[len(phrase) + 1:]
            break
    stripped = True
    while stripped:
        stripped = False
        for word in FILLERS:
            if text.startswith(word + " "):
                text, stripped = text[len(word) + 1:], True
    return text.strip(" ,.")


def search_candidates(instruction: str) -> list[str]:
    """The queries the search page offers, in order (see the module docstring)."""
    full = _clean(instruction)
    no_price = _clean(PRICE_CLAUSE.sub("", instruction.strip()))
    request = _strip_request(no_price)
    head = HEAD_END.split(request, maxsplit=1)[0].strip(" ,.")
    out: list[str] = []
    for query in (full, no_price, request, head):
        if query and query not in out and query.split()[0] not in SPECIAL and re.search(r"[a-z0-9]", query):
            out.append(query)
    return out


# ---------------------------------------------------------------------------------------------------
# The WebShop server (web_shopping_server.py), reached over HTTP.

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # the server is local: never through a proxy
_SERVER_LOCK = threading.Lock()
_SERVER: dict[str, Any] = {}


def _server_url() -> str:
    return os.environ.get(SERVER_VAR, DEFAULT_SERVER).rstrip("/")


def _request(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    url = _server_url() + path
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with _OPENER.open(request, timeout=REQUEST_TIMEOUT_S) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"WebShop server {url}: HTTP {error.code}: {detail}") from None
    except (urllib.error.URLError, OSError) as error:
        raise RuntimeError(f"WebShop server not reachable at {url} ({error}); start web_shopping_server.py (see {__name__})") from None


def server_info() -> dict[str, Any]:
    """The server's /info, checked once per server URL: it must serve the expected products and test goals."""
    with _SERVER_LOCK:
        url = _server_url()
        if _SERVER.get("url") != url:
            info = _request("/info")
            if info.get("products") != PRODUCTS or info.get("test_goals_sha256") != TEST_GOALS_SHA256:
                raise RuntimeError(f"the WebShop server at {url} does not serve the expected data: {info.get('products')} products "
                                   f"(expected {PRODUCTS}), test goals sha256 {info.get('test_goals_sha256')} (expected {TEST_GOALS_SHA256})")
            _SERVER.update(url=url, info=info)
        return _SERVER["info"]


# ---------------------------------------------------------------------------------------------------
# Environment.


@register_env
class WebShop(Environment):
    name = "loop/web_shopping/webshop"
    leaf = "loop/web_shopping"
    source = "WebShop (Yao et al., 2022), text mode, all 1.18M products, human instructions (test split); WebShop code at commit 64fa2a5"
    url = URL
    license = ("MIT (WebShop code); product data scraped from amazon.com by the WebShop authors, no license stated (the Hugging Face "
               "mirror YWZBrandon/webshop-data has no dataset card)")
    tier = "A"
    criteria_source = ("Instruction (crowd-written, with WebShop's price clause) and page text verbatim from WebShop's text mode; options are "
                       "WebShop's clickables for the page and search queries derived from the instruction by fixed rules; one paragraph of "
                       "rules added; reward from WebShop's own get_reward")
    notes = ("First 100 of WebShop's 500 test goals (seed i = test goal i) over the full product set (1,181,430 products; Lucene BM25 "
             "index built by WebShop's own scripts). One choice per decision: on the search page a query among 2-4 derived from the "
             "instruction (verbatim; without the price clause; without the opening request; the product phrase), elsewhere a click among "
             "the page's buttons, product links (with title and price) and option buttons. 15 decisions, ReAct's WebShop step limit. "
             "Score = WebShop's reward in [0, 1] (share of the instruction's attributes, options and price cap met, times a product-type "
             "factor); success = reward 1; no purchase within the budget scores 0. WebShop runs in its own Python 3.8 environment behind "
             "web_shopping_server.py (about 18 GB of memory).")
    mode = "paused"
    episodes = 100
    max_decisions = MAX_DECISIONS
    score_name = "reward"

    def reset(self, seed: int) -> None:
        self.close()
        server_info()
        if not 0 <= seed < TEST_GOALS:
            raise ValueError(f"seed {seed} out of range: WebShop's test split has {TEST_GOALS} goals")
        self.seed = seed
        self.history: list[str] = []
        self.list_page = 0
        self.current: dict[str, tuple[str, str, str, bool]] = {}
        self.view = _request("/reset", {"goal": seed})
        self.session = self.view["session"]
        self.instruction = self.view["instruction"]
        self.queries = search_candidates(self.instruction)
        self._forced_search()

    # --- requests ---

    def _actions(self) -> list[tuple[str, str, str, bool]]:
        """(WebShop action, option text, text in the list of past actions, is an option button) per action valid on the
        current page, in page order."""
        view = self.view
        if view["has_search_bar"]:
            return [(f"search[{q}]", f"search[{q}]", f"search[{q}]", False) for q in self.queries]
        out = []
        for c in view["clickables"]:
            if not c["valid"]:  # e.g. an option value with square brackets, which WebShop's action parser cannot read
                continue
            action = f"click[{c['key']}]"
            if c["kind"] == "product":
                title = c["title"] if len(c["title"]) <= TITLE_CHARS else c["title"][: TITLE_CHARS - 1] + "…"
                out.append((action, f"click[{c['label']}]: {c['title']} ({c['price']})", f"click[{c['label']}] ({title})", False))
            elif c["kind"] == "option":
                selected = view["options"].get(c["group"].lower()) == c["key"]
                text = f"click[{c['label']}]: option \"{c['group']}\"" + (" (selected)" if selected else "")
                out.append((action, text, f"click[{c['label']}] (option \"{c['group']}\")", True))
            else:
                out.append((action, f"click[{c['label']}]", f"click[{c['label']}]", False))
        return out

    def _page_text(self) -> str:
        """WebShop's observation without the header that repeats the instruction."""
        return self.view["observation"].replace(f"Instruction: [SEP] {self.instruction.strip()} [SEP] ", "", 1)

    def observe(self) -> DecisionRequest:
        numbered = list(enumerate(self._actions(), 1))  # option ids are positions in the page's full list
        extra: dict[str, str] = {}
        if len(numbered) > MAX_OPTIONS:  # more than one question holds: the option buttons come in lists, the rest stays
            fixed = [(i, a) for i, a in numbered if not a[3]]
            paged = [(i, a) for i, a in numbered if a[3]]
            size = MAX_OPTIONS - len(fixed) - 2
            lists = (len(paged) + size - 1) // size
            self.list_page = min(self.list_page, lists - 1)
            chunk = paged[self.list_page * size : (self.list_page + 1) * size]
            numbered = sorted(fixed + chunk)
            if self.list_page < lists - 1:
                extra[MORE] = (f"Show more option buttons (the product has {len(paged)}; these are "
                               f"{self.list_page * size + 1}-{self.list_page * size + len(chunk)})")
            if self.list_page > 0:
                extra[PREVIOUS] = "Show the previous option buttons"
        else:
            self.list_page = 0
        self.current = {str(i): action for i, action in numbered}
        criteria = {key: action[1] for key, action in self.current.items()} | extra
        state = {"website": RULES, "instruction": self.instruction, "page": self._page_text()}
        if self.view["page"] in ("item_page", "item_sub_page"):
            has_options = any(c["kind"] == "option" for c in self.view["clickables"])
            if has_options or self.view["options"]:
                state["selected options"] = "; ".join(f"{group}: {value}" for group, value in self.view["options"].items()) or "none"
        state["actions so far"] = "\n".join(f"{i}. {h}" for i, h in enumerate(self.history, 1)) or "none yet"
        return DecisionRequest(state=state, questions={"action": Question(type="choice", instructions=QUESTION, criteria=criteria)})

    def default_choice(self, request: DecisionRequest) -> dict[str, str]:
        """No usable answer: nothing is done this step (WebShop leaves the page as it is for an invalid action)."""
        return {name: NO_OP for name in request.questions}

    # --- actions ---

    def act(self, choice: dict[str, str]) -> None:
        option = choice.get("action", NO_OP)
        if option == MORE:
            self.list_page += 1
            return
        if option == PREVIOUS:
            self.list_page = max(0, self.list_page - 1)
            return
        picked = self.current.get(option)
        if picked is None:
            self.history.append("(no usable answer: nothing was done)")
            return
        self._send(picked[0], picked[2])

    def _send(self, action: str, record: str) -> None:
        self.view = _request("/step", {"session": self.session, "action": action})
        self.history.append(record)
        self.list_page = 0
        self._forced_search()

    def _forced_search(self) -> None:
        """On the search page with a single candidate query, that search is run at once (a forced move, not a decision)."""
        if not self.view["done"] and self.view["has_search_bar"] and len(self.queries) == 1:
            self._send(f"search[{self.queries[0]}]", f"search[{self.queries[0]}] (the only query: run without a decision)")

    def done(self) -> bool:
        return bool(self.view.get("done"))

    def outcome(self) -> dict[str, Any]:
        base = {"goal": self.seed, "instruction": self.instruction, "actions": list(self.history)}
        if not self.done():  # the decision budget ran out before Buy Now: WebShop gives no reward without a purchase
            return {"success": False, "score": 0.0, "ended_by": "decision limit", **base}
        reward = float(self.view["reward"])
        purchase, goal = self.view["purchase"], self.view["goal"]
        return {"success": reward == 1.0, "score": reward, "ended_by": "buy now", "bought": purchase["asin"], "options": purchase["options"],
                "price": purchase["price"], "same_product": purchase["asin"] == goal["asin"], "reward_info": self.view["reward_info"],
                "goal_product": goal["asin"], "goal_attributes": goal["attributes"], "goal_options": goal["goal_options"],
                "price_upper": goal["price_upper"], **base}

    def close(self) -> None:
        session = getattr(self, "session", None)
        self.session = None
        if session:
            try:
                _request("/close", {"session": session})
            except RuntimeError:  # a server that went away holds no session
                pass

    def card(self) -> dict[str, Any]:
        return {**super().card(), "webshop_commit": WEBSHOP_COMMIT, "data": DATA_MIRROR, "products": PRODUCTS, "test_goals": TEST_GOALS,
                "test_goals_sha256": TEST_GOALS_SHA256}
