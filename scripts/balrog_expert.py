"""Check the per-seed budgets of the BALROG-derived game environments with expert play.

For every benchmark seed of BabyAI (game/roguelike/babyai_*), MiniHack MazeWalk (game/roguelike/minihack_mazewalk_*)
and Baba Is AI (game/rule_puzzles/baba_is_ai/*), an expert plays the episode through the harness, following the
solution the seed's budget was computed from: BabyAI's bot, the shortest path to the stairs, the searched Baba Is AI
solution. It should succeed within the budget. Prints one line per environment (expert success, mean shortest
solution L*, mean budget) and writes every episode to --out as JSON lines.

    python scripts/balrog_expert.py --workers 8 --out results/summary/balrog_expert.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from s1arena.envs.game.baba_is_ai import BabaIsAIPuzzle, shortest_solution  # noqa: E402
from s1arena.envs.game.babyai_text import ACTIONS, BabyAILevel  # noqa: E402
from s1arena.envs.game.nethack_text import MazeWalkTask, maze_solution  # noqa: E402
from s1arena.registry import ENVS, envs_under  # noqa: E402

SUITES = (BabyAILevel, MazeWalkTask, BabaIsAIPuzzle)
SEARCH_STATES = 2_000_000  # enough for the hardest Baba Is AI level, which needs several minutes and gigabytes


def play(name: str, seed: int) -> dict:
    """One expert episode, played as the runner plays it (a decision per act until done())."""
    env, started = ENVS[name](), time.time()
    try:
        env.reset(seed)
        if isinstance(env, BabyAILevel):
            from minigrid.utils.baby_ai_bot import BabyAIBot

            bot, options = BabyAIBot(env.env), {action: option for option, (action, _) in ACTIONS.items()}
            while not env.done():
                action = int(bot.replan())  # the bot's 'done' is not an option: drop with empty hands or pick up with full ones
                env.act({"action": options.get(action, "pickup" if env.game.carrying else "drop")})
        else:
            if isinstance(env, MazeWalkTask):
                solution = maze_solution(env.env_id, seed)
            else:  # searched again, up to the table's L*: a shorter solution or none at all shows a wrong entry
                solution = shortest_solution(env.task_id, seed, env.shortest or env.max_decisions // 2, SEARCH_STATES)
            for move in solution or ():
                if env.done():
                    break
                env.act({"action": move})
        outcome = env.outcome()
    finally:
        env.close()
    return {"env": name, "version": env.version, "seed": seed, "success": bool(outcome["success"]),
            "shortest": outcome["shortest"], "budget": outcome.get("budget", outcome.get("max_steps")),
            "decisions": outcome.get("steps", outcome.get("decisions")), "wall_s": round(time.time() - started, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", default="game", help="environment name or prefix (only the three suites are played)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    classes = [cls for cls in envs_under(args.env) if issubclass(cls, SUITES)]
    jobs = [(cls.name, seed) for cls in classes for seed in range(cls.episodes)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(play, [name for name, _ in jobs], [seed for _, seed in jobs]))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(row) + "\n" for row in rows)
    for cls in classes:
        mine = [row for row in rows if row["env"] == cls.name]
        known = [row["shortest"] for row in mine if row["shortest"] is not None]
        unsolved = [row["seed"] for row in mine if row["shortest"] is None]
        print(f"{cls.name:<76} v{cls.version} expert {sum(row['success'] for row in mine)}/{len(mine)}  "
              f"L*={statistics.mean(known) if known else float('nan'):5.1f}  budget={statistics.mean(row['budget'] for row in mine):5.1f}"
              + (f"  no L* for seeds {unsolved}" if unsolved else ""), flush=True)


if __name__ == "__main__":
    main()
