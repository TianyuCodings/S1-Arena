"""WebShop behind a small local HTTP server, for the loop/web_shopping environment (web_shopping.py).

WebShop (Yao et al., 2022; princeton-nlp/WebShop, MIT) pins the packages its search and its reward depend on:
pyserini 0.17 (Lucene BM25, needs a JDK 11), spaCy 3.3 with en_core_web_sm 3.3.0 (product-type part of the reward),
thefuzz 0.19 (attribute and option matching), Flask 2.1 (the pages), gym 0.24, torch 1.11. They do not install on
Python 3.12, so WebShop runs in its own Python 3.8 environment and this script serves it over HTTP on 127.0.0.1; the
S1-Arena environment is a thin client. Start it with WebShop's Python, not S1-Arena's:

    <WebShop venv>/bin/python web_shopping_server.py --webshop <WebShop checkout> --data <data dir> [--port 3000]

<data dir> holds items_shuffle.json, items_ins_v2.json and items_human_ins.json; the Lucene index must have been built
at <WebShop checkout>/search_engine/indexes with WebShop's own search_engine scripts. The file uses only the standard
library at module level and Python 3.8 syntax (S1-Arena's environment discovery imports every module under
s1arena/envs); nothing here calls a language model.

Everything goes through WebShop's own objects at the pinned commit: one SimServer (products, the human-instruction
goals, the search engine) and, per episode, one WebAgentTextEnv in "text" observation mode, stepped with WebShop's action
strings (search[...] and click[...]). The full product set is loaded, as in the paper: WebShop's utils.py defaults to a
1,000-product preview and its README says to switch the product and attribute files to the full ones, which --data
does here without editing the checkout.

Two things WebShop leaves to chance are fixed:
- While loading, WebShop draws the price of every product listed with a price range (random.uniform) and the price cap
  of every instruction (random.sample) from Python's global generator, which it never seeds, so the "price lower than X
  dollars" clause changes from one WebShop start to the next. The server seeds that generator with LOAD_SEED right
  before loading. The goal order is WebShop's own (a shuffle seeded with 233 after loading).
- Each episode gets its own session id, so episodes of the same goal can run side by side.

All WebShop calls run on one worker thread (pyjnius attaches that thread to the JVM once), which also serializes them.
The JVM's heap is capped (--java-heap, 2 GB; searching uses a few hundred MB), so the process stays near 15-17 GB
after loading (18.5 GB at the peak of loading). Endpoints, JSON in and out: GET /info; POST /reset {"goal": i};
POST /step {"session": s, "action": "search[...]" or "click[...]"}; POST /close {"session": s}.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOAD_SEED = 233  # WebShop's own seed constant (it seeds its goal shuffle with 233)
TEST_GOALS = 500  # WebShop's test split: the first 500 goals (baseline_models/env.py)
BASE_URL = "http://127.0.0.1:3000"  # WebShop's SimServer base URL; it only appears inside WebShop's page URLs
WEBSHOP_COMMIT = "64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd"

_WORKER = ThreadPoolExecutor(max_workers=1)
_STATE = {"envs": {}}
_COUNTER = itertools.count(1)


def goals_fingerprint(goals) -> str:
    """sha256 of the goals as the agent meets them: product, instruction, attributes, options, price cap."""
    rows = [[g["asin"], g["instruction_text"], g["attributes"], g["goal_options"], g["price_upper"]] for g in goals]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def load(webshop: str, data: str, java_heap: str) -> None:
    sys.path.insert(0, webshop)
    import warnings

    import jnius_config  # pyserini starts a JVM through pyjnius when imported: cap its heap first (searches need little)

    if not jnius_config.vm_running:
        jnius_config.add_options("-Xmx" + java_heap)
    warnings.filterwarnings("ignore")  # thefuzz warns about its pure-Python matcher, as in any WebShop install
    from web_agent_site.engine import engine
    from web_agent_site.engine.goal import nlp
    from web_agent_site.envs import web_agent_text_env

    files = {"products": "items_shuffle.json", "attributes": "items_ins_v2.json", "human_instructions": "items_human_ins.json"}
    paths = {key: os.path.join(data, name) for key, name in files.items()}
    engine.DEFAULT_ATTR_PATH = paths["attributes"]
    engine.HUMAN_ATTR_PATH = paths["human_instructions"]
    started = time.time()
    random.seed(LOAD_SEED)
    server = web_agent_text_env.SimServer(BASE_URL, paths["products"], filter_goals=None, limit_goals=-1, num_products=None,
                                          human_goals=1, show_attrs=False)
    versions = {}
    for package in ("pyserini", "spacy", "thefuzz", "flask", "bs4", "gym"):
        try:
            versions[package] = getattr(__import__(package), "__version__", "?")
        except ImportError:
            versions[package] = None
    _STATE.update(server=server, env_class=web_agent_text_env.WebAgentTextEnv, parse_action=engine.parse_action)
    _STATE["info"] = {
        "webshop_commit": WEBSHOP_COMMIT,
        "files": {key: {"path": path, "bytes": os.path.getsize(path)} for key, path in paths.items()},
        "products": len(server.all_products),
        "index_docs": int(server.search_engine.num_docs),
        "goals": len(server.goals),
        "test_goals": TEST_GOALS,
        "test_goals_sha256": goals_fingerprint(server.goals[:TEST_GOALS]),
        "load_seed": LOAD_SEED,
        "java_heap": java_heap,
        "load_seconds": round(time.time() - started),
        "python": sys.version.split()[0],
        "spacy_model": f"{nlp.meta['name']} {nlp.meta['version']}",
        "versions": versions,
    }


def _view(env) -> dict:
    """The current page: WebShop's text observation, its clickables in page order, and the session's selections."""
    server = env.server
    session = server.user_sessions[env.session]
    available = env.get_available_actions()  # also sets env.text_to_clickable, as WebShop's step() does
    items = sorted(env.text_to_clickable.items(), key=lambda kv: (kv[1].sourceline or 0, kv[1].sourcepos or 0))
    clickables = []
    for key, tag in items:
        if key == "search":  # the search page's button: WebShop's step() never clicks it
            continue
        entry = {"key": key}
        classes = tag.get("class") or []
        if classes and classes[0] == "product-link":
            product = server.product_item_dict.get(key.upper(), {})
            entry.update(kind="product", label=key.upper(), title=product.get("Title", ""), price=product.get("Price", ""))
        elif tag.name == "input":
            entry.update(kind="option", label=key, group=tag.get("name", ""))
        else:
            entry.update(kind="button", label=tag.get_text().strip())
        name, arg = _STATE["parse_action"](f"click[{key}]")
        entry["valid"] = name == "click" and arg is not None and arg.lower() == key  # WebShop's parser gives the key back
        clickables.append(entry)
    page = server.get_page_name(env.browser.current_url)
    return {
        "session": env.session,
        "page": page or "search",
        "url": env.browser.current_url,
        "observation": env.observation,
        "has_search_bar": bool(available["has_search_bar"]),
        "clickables": clickables,
        "instruction": session["goal"]["instruction_text"],
        "asin": session.get("asin"),
        "options": dict(session.get("options") or {}),
    }


def reset(body: dict) -> dict:
    server = _STATE["server"]
    goal = int(body["goal"])
    if not 0 <= goal < len(server.goals):
        raise ValueError(f"goal {goal} out of range: {len(server.goals)} goals")
    env = _STATE["env_class"](observation_mode="text", server=server, session_prefix=f"jv{next(_COUNTER)}_")
    server.user_sessions.pop(env.session, None)  # the constructor opens a session with a random goal; not used
    env.reset(session=goal)  # session id = prefix + goal index; the goal is server.goals[goal]
    _STATE["envs"][env.session] = env
    return dict(_view(env), reward=0.0, done=False)


def step(body: dict) -> dict:
    env = _STATE["envs"][body["session"]]
    _, reward, done, _ = env.step(str(body["action"]))
    out = _view(env)
    out.update(reward=float(reward), done=bool(done))
    if done:
        server = env.server
        session = server.user_sessions[env.session]
        goal = session["goal"]
        out["purchase"] = {"asin": session["asin"], "options": dict(session["options"]), "price": server.product_prices.get(session["asin"])}
        out["reward_info"] = session.get("verbose_info")
        out["goal"] = {key: goal[key] for key in ("asin", "attributes", "goal_options", "price_upper", "query", "category")}
    return out


def close(body: dict) -> dict:
    env = _STATE["envs"].pop(body["session"], None)
    if env is not None:
        env.server.user_sessions.pop(env.session, None)
    return {"closed": env is not None}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet: one line per request would flood the log
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/info":
            self._send(200, dict(_STATE["info"], sessions=len(_STATE["envs"])))
        else:
            self._send(404, {"error": f"unknown path {self.path}"})

    def do_POST(self) -> None:
        handler = {"/reset": reset, "/step": step, "/close": close}.get(self.path)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if handler is None:
                self._send(404, {"error": f"unknown path {self.path}"})
                return
            result = _WORKER.submit(handler, body).result()
        except KeyError as error:
            self._send(400, {"error": f"unknown session or missing field: {error}"})
            return
        except Exception as error:  # noqa: BLE001 - every failure is answered with a 500 and its message
            traceback.print_exc()
            self._send(500, {"error": f"{type(error).__name__}: {error}"})
            return
        self._send(200, result)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Serve WebShop's text environment over HTTP for S1-Arena.")
    parser.add_argument("--webshop", required=True, help=f"WebShop checkout (princeton-nlp/WebShop at {WEBSHOP_COMMIT[:7]})")
    parser.add_argument("--data", required=True, help="folder with items_shuffle.json, items_ins_v2.json, items_human_ins.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--java-heap", default="2g", help="maximum heap of the JVM that runs Lucene (the JVM default is a quarter of RAM)")
    args = parser.parse_args(argv)
    _WORKER.submit(load, os.path.abspath(args.webshop), os.path.abspath(args.data), args.java_heap).result()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ready on {args.host}:{args.port} {json.dumps(_STATE['info'])}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
