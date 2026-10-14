"""Score episodes at a lower decision cap by replaying the first decisions of longer episodes, without calling the model.

The runner never shows an environment's decision cap to the model, so the first N decisions of an episode played at a
higher cap are a sample of the episode the same model plays at cap N (a sample, not a copy: a new run can choose
differently wherever the model's replies do not repeat exactly). Each source episode is first replayed in full: only
when the replay reproduces the recorded outcome, and the first N observations repeat on a second pass, is a row written at
cap N, marked "replayed_from". The row keeps the source episode's median latency; cost and wall time are left empty.
Paused, single-agent episodes only.

Usage (on a machine with the environments installed):
    python scripts/replay_prefix.py --model jev --env game/rule_puzzles --episodes 2 --max-decisions 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from s1arena.envs import runner  # noqa: E402
from s1arena.registry import envs_under, make_model  # noqa: E402


def replay(env_cls, seed: int, trace: list[dict], limit: int) -> tuple[dict, list[str]]:
    """The outcome after the first `limit` decisions of the trace, and a hash of each observation on the way."""
    env = env_cls()
    env.reset(seed)
    hashes = []
    try:
        for step in trace[:limit]:  # the runner's loop, with the recorded choices in place of the model's
            if env.done():
                break
            request = env.observe()
            if env.done():
                break
            hashes.append(hashlib.sha1(request.model_dump_json().encode()).hexdigest())
            env.act(step["c"])
        return json.loads(json.dumps(env.outcome(), ensure_ascii=False)), hashes
    finally:
        env.close()


def derive(env_cls, source: dict, limit: int) -> dict | str:
    """The row at the lower cap, or why the source episode does not replay faithfully."""
    trace = source["trace"]
    try:
        full, full_hashes = replay(env_cls, source["seed"], trace, len(trace))
        if len(full_hashes) != len(trace) or full != source["outcome"]:
            return (f"full replay differs: {len(full_hashes)} of {len(trace)} decisions, outcome {json.dumps(full)[:160]} "
                    f"vs recorded {json.dumps(source['outcome'])[:160]}")
        cut, hashes = replay(env_cls, source["seed"], trace, limit)
    except Exception as error:
        return f"replay failed: {type(error).__name__}: {str(error)[:200]}"
    if hashes != full_hashes[: len(hashes)]:
        return "observations differ between two replays"
    n = len(hashes)
    return {**source, "max_decisions": limit, "outcome": cut, "decisions": n,
            "failed_decisions": sum(step.get("x", 0) for step in trace[:n]), "cost_usd": None, "wall_s": None,
            "trace": trace[:n], "replayed_from": {"max_decisions": source["max_decisions"], "decisions": source["decisions"]}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--env", required=True, help="environment name or prefix")
    parser.add_argument("--episodes", type=int, required=True, help="seeds 0 .. episodes-1, as in env-score")
    parser.add_argument("--max-decisions", type=int, required=True, help="the lower cap")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    slug, cap = make_model(args.model).slug, args.max_decisions
    for cls in envs_under(args.env):
        if cls.agents or cls.max_decisions <= cap:
            continue
        have = {r["seed"] for r in runner.latest_episodes(slug, cls.name, "paused", cap, version=cls.version) if "outcome" in r}
        sources = {r["seed"]: r for r in runner.latest_episodes(slug, cls.name, "paused", cls.max_decisions, version=cls.version)
                   if "outcome" in r}
        seeds = [s for s in range(args.episodes) if s not in have]
        if missing := [s for s in seeds if s not in sources]:
            print(f"{cls.name}: no episode at cap {cls.max_decisions} for seeds {missing}", flush=True)
        todo = [sources[s] for s in seeds if s in sources]
        if not todo:
            continue
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(lambda row: derive(cls, row, cap), todo))
        with runner.episode_path(slug, cls.name).open("a", encoding="utf-8") as out:
            for source, row in zip(todo, results):
                if isinstance(row, str):
                    print(f"  {cls.name} seed {source['seed']}: not replayed, {row}", flush=True)
                    continue
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"  {cls.name} seed {row['seed']}: score={row['outcome'].get('score')} success={row['outcome'].get('success')} "
                      f"decisions={row['decisions']} of {source['decisions']}", flush=True)


if __name__ == "__main__":
    main()
