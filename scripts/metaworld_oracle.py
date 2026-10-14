"""Check that the Meta-World harness can still solve every MT10 task with the options the model gets.

A scripted policy plays each task through the environment's own interface (observe / act), choosing only among the
options of the current harness (axis shifts of 1 or 5 cm and gripper open / close in version 2). Its plan per task is a
list of waypoints computed from the named points the model also sees; each waypoint is reached axis by axis. The
script reports the success rate and the decisions used per task, so a harness change that makes a task impossible or
too long for the decision budget shows up.

Usage (on a machine with Meta-World installed): python scripts/metaworld_oracle.py [--seeds 10] [--task reach]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from s1arena.registry import envs_under  # noqa: E402

TOL = 0.006  # a waypoint coordinate counts as reached within this distance (m); the smallest shift is 1 cm


class Budget(Exception):
    pass


class Done(Exception):
    pass


class Oracle:
    def __init__(self, env):
        self.env = env

    def act(self, option: str) -> None:
        if self.env.done():
            raise Done
        request = self.env.observe()
        options = request.questions["action"].criteria
        if option not in options:
            raise ValueError(f"{option} is not an option: {sorted(options)}")
        if self.env.decisions >= self.env.max_decisions:
            raise Budget
        self.env.act({"action": option})

    def goto(self, target, order: str = "xyz") -> None:
        """Shift the fingertip centre to target, one axis at a time in the given order."""
        for axis in order:
            i = "xyz".index(axis)
            for _ in range(40):
                d = float(np.asarray(target)[i] - self.env._tcp()[i])
                if abs(d) < TOL:
                    break
                step = 5 if abs(d) >= 0.045 else 1
                self.act(f"{axis}{'+' if d > 0 else '-'}{step}cm")

    def grip(self, close: bool) -> None:
        self.act("close_gripper" if close else "open_gripper")


def plan(task: str, o: Oracle) -> None:
    env, p = o.env, o.env._point
    up = np.array([0.0, 0.0, 1.0])
    if task == "reach":
        o.goto(p("goal"), "zxy")
    elif task in ("push", "pick_place"):
        puck = p("puck")
        o.goto(puck + 0.10 * up, "zxy")
        o.goto(puck + 0.005 * up, "z")
        o.grip(True)
        if task == "push":
            o.goto(np.r_[p("goal")[:2], env._tcp()[2]], "xy")
        else:
            o.goto(env._tcp() + 0.10 * up, "z")
            o.goto(p("goal"), "xyz")
    elif task == "button_press_topdown":
        o.goto(p("button") + 0.03 * up, "zxy")
        o.goto(p("goal") - 0.015 * up, "z")
    elif task in ("drawer_open", "drawer_close"):
        handle = p("handle")
        if task == "drawer_open":
            o.goto(handle + 0.10 * up, "zxy")
            o.goto(handle, "z")
            o.grip(True)
            o.goto(handle + np.array([0.0, -0.18, 0.0]), "y")
        else:
            o.goto(handle + np.array([0.0, -0.06, 0.10]), "zxy")
            o.goto(handle + np.array([0.0, -0.06, 0.0]), "z")
            o.goto(p("goal") + np.array([0.0, 0.02, 0.0]), "y")
    elif task in ("window_open", "window_close"):
        handle = p("handle")
        o.goto(handle + np.array([0.0, -0.08, 0.10]), "zxy")
        o.goto(handle + np.array([0.0, -0.08, 0.0]), "z")
        o.goto(handle, "y")
        o.goto(np.r_[p("goal")[0], env._tcp()[1:]], "x")
    elif task == "door_open":  # as Meta-World's scripted policy: closed fingers slipped between handle and panel, then pulled
        o.grip(True)
        handle = p("handle")
        o.goto(handle + np.array([0.01, 0.02, 0.06]), "zxy")
        o.goto(handle + np.array([0.01, 0.02, 0.0]), "z")
        for _ in range(30):  # keep pulling toward a point 5 cm to the -x side of where the handle is now
            d = p("handle") + np.array([-0.05, 0.0, 0.0]) - env._tcp()
            axis = int(np.argmax(np.abs(d[:2])))
            if abs(d[axis]) < 0.005:
                break
            step = 5 if abs(d[axis]) >= 0.045 else 1
            o.act(f"{'xy'[axis]}{'+' if d[axis] > 0 else '-'}{step}cm")
    elif task == "peg_insert_side":
        peg = p("peg")  # grasp the peg at its centre, at its own height (1 cm higher, the fingers slip off)
        o.goto(peg + 0.10 * up, "zxy")
        o.goto(peg, "z")
        o.grip(True)
        o.goto(env._tcp() + 0.12 * up, "z")
        offset = p("goal") - p("peg_head")
        o.goto(env._tcp() + offset + np.array([0.08, 0.0, 0.0]), "zyx")
        o.goto(env._tcp() + (p("goal") - p("peg_head")), "x")
    else:
        raise ValueError(task)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--task", default="", help="one MT10 task, e.g. reach (default: all)")
    parser.add_argument("--tolerance", type=float, default=None, help="try another stopping tolerance of the controller (m)")
    args = parser.parse_args()
    if args.tolerance is not None:
        import s1arena.envs.embodied.tabletop_loop as tabletop

        tabletop.TOLERANCE = args.tolerance
    for cls in envs_under("embodied/tabletop_loop"):
        task = cls.name.rsplit("/", 1)[-1]
        if args.task and task != args.task:
            continue
        wins, used, notes = 0, [], []
        for seed in range(args.seeds):
            env = cls()
            env.reset(seed)
            oracle = Oracle(env)
            try:
                plan(task, oracle)
                for _ in range(5):  # hold still for a moment (repeating the gripper command) while things settle
                    if env.done():
                        break
                    oracle.act("close_gripper" if env.grip > 0 else "open_gripper")
            except Done:
                pass
            except Budget:
                notes.append(f"seed {seed}: decision budget")
            wins += env.success
            used.append(env.decisions)
            env.close()
        print(f"{task:<22} v{cls.version} success {wins}/{args.seeds}  decisions mean {statistics.mean(used):.1f} max {max(used)} "
              f"(budget {cls.max_decisions})" + (f"  [{'; '.join(notes)}]" if notes else ""), flush=True)


if __name__ == "__main__":
    main()
