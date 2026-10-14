"""Skill-level tabletop manipulation: Meta-World v3 tasks (MuJoCo) driven through motion primitives.

The harness is the minimal layer a classifier needs to act: Meta-World's action is a continuous
displacement of the hand plus a gripper command, and the harness only discretizes it. It reads the
privileged simulator state (the gripper's fingertip centre and opening, the task's object points and
the goal point, which Meta-World's goal-observable environments expose) and writes it out as numbers.
It asks one choice question whose options shift the fingertip centre by 1 or 5 cm along one axis, or
close or open the fingers. The chosen primitive runs a proportional controller for a bounded number of
simulation steps, then the next decision is asked. The task text is Meta-World's own task description;
the scene text says what stands on the table and how it moves. Neither says how to act.

Version 2 (2026-09-28) removed the version-1 help that was not needed to act: primitives
that drove the gripper straight onto or above a named point, and the success condition restated from the
environment code (kept in the card as "success_condition"). The decision budget rose from 30 to 80 and a
move now stops 5 mm (not 2 mm) from its target, so that the many short moves fit in Meta-World's 500-step
horizon: scripts/metaworld_oracle.py, a scripted policy restricted to these options, solves every task
(peg-insert-side on most seeds; see its output). Uniform random choice, which succeeded in 30-80% of the
episodes of reach, button press, window open/close and drawer close in version 1, succeeds only in
drawer-close (30%) and reach (10%) in version 2.

The tasks are Meta-World's MT10 set. Seed s builds Meta-World's goal-observable environment with
seed=s, as gym.make("Meta-World/goal_observable", env_name=..., seed=s) does, so the object and goal
positions are Meta-World's own. The score is Meta-World's success flag: the episode ends at the first
successful simulation step, as in Meta-World's evaluation, and fails when the task's 500-step horizon
or the decision budget runs out.
"""

from __future__ import annotations

import threading
from typing import Any

from ...registry import register_env
from ...schema import DecisionRequest, Question
from ..base import Environment

URL = "https://github.com/Farama-Foundation/Metaworld"
LICENSE = "MIT (Meta-World); Apache-2.0 (MuJoCo)"

# Building a seeded Meta-World environment reseeds NumPy's global random generator for a moment; the
# runner plays episodes in threads, so construction is serialized to keep every seed deterministic.
_BUILD_LOCK = threading.Lock()

GAIN = 25.0  # proportional gain on the fingertip error, as in Meta-World's scripted policies
LEAD = 0.05  # the mocap driving the hand stays within this distance of the hand (m); bounds the pushing force
TOLERANCE = 0.005  # a move ends when the fingertip centre is this close to its target (m); 2 mm in version 1
MOVE_STEPS = 80  # at most this many simulation steps per move
STALL_STEPS = 8  # a move also ends when the fingertip centre moved less than 2 mm over this many steps
GRIP_STEPS = 15  # simulation steps to close or open the fingers (they settle in about 13)
SHIFTS = (0.05, 0.01)  # the fixed axis steps (m)

SCENE = ("A Sawyer robot arm with a two-finger gripper works over a table. Coordinates are in metres: x runs across the table "
         "(positive = to the robot's right), y points away from the robot's base, z is the height above the table top. The two "
         "fingers close toward each other along the y axis.")
PUCK = "A puck (a cylinder 0.04 m across and 0.04 m tall) lies on the table"
NOTES = ("MT10 task. Score = Meta-World's success flag; the episode ends at the first success, and fails when the 500-step "
         "horizon or the decision budget runs out.")
AXES = {"x": ("+x (to the robot's right)", "-x (to the robot's left)"),
        "y": ("+y (away from the robot)", "-y (toward the robot)"),
        "z": ("+z (up)", "-z (down)")}


def fmt(v) -> str:
    return f"({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f})"


class MetaWorldTabletop(Environment):
    leaf = "embodied/tabletop_loop"
    url = URL
    license = LICENSE
    tier = "B"
    criteria_source = ("Task wording from Meta-World's task table (Yu et al., CoRL 2019, Table 2); success = the v3 environment's "
                       "success flag (not shown to the model)")
    notes = NOTES
    mode = "paused"
    control_dt = 0.0125  # one Meta-World simulation step (5 MuJoCo steps of 2.5 ms)
    episodes = 10
    max_decisions = 80
    score_name = "success"
    version = 2
    task_id = ""  # Meta-World v3 environment name, e.g. "reach-v3"
    description = ""  # the task's description in Meta-World's task table
    success_text = ""  # the environment's success condition in words (in the card, not shown to the model)
    scene_text = ""  # what stands on the table and how it moves
    points: dict[str, str] = {}  # named point -> what it is (shown with its coordinates); the move targets, in order

    def reset(self, seed: int) -> None:
        import mujoco
        from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE

        with _BUILD_LOCK:
            self.env = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[self.task_id + "-goal-observable"](seed=seed)
        self.obs, _ = self.env.reset()
        mujoco.mj_forward(self.env.model, self.env.data)  # some resets leave site positions stale until the first step
        self.horizon = int(self.env.max_path_length)
        self.control_dt = float(self.env.dt)
        self.grip = -1.0  # finger command held between primitives: -1 open, +1 closed
        self.steps, self.decisions, self.success, self.last = 0, 0, False, ""

    # simulator access -------------------------------------------------------------------------

    def _tcp(self):
        return self.env.tcp_center.copy()

    def _point(self, name: str):
        raise NotImplementedError

    def scene(self) -> str:
        return self.scene_text

    def _step(self, move) -> None:
        import numpy as np

        action = np.array([move[0], move[1], move[2], self.grip], dtype=np.float32)
        self.obs, _, _, _, info = self.env.step(action)
        self.steps += 1
        self.success = self.success or bool(info["success"])

    def _drive(self, target) -> None:
        """One simulation step of proportional control of the fingertip centre toward target.

        Meta-World moves a mocap body, welded to the hand, by up to 1 cm per step. The mocap moves by
        GAIN times the fingertip error (as in Meta-World's scripted policies) but is kept within LEAD
        of the hand, so a blocked hand pushes with a bounded force instead of winding up.
        """
        import numpy as np

        mocap = self.env.data.mocap_pos[0].copy()
        scale = self.env.action_scale
        want = mocap + np.clip(GAIN * (target - self._tcp()), -1.0, 1.0) * scale
        hand = self.env.get_endeff_pos()
        ahead = want - hand
        distance = float(np.linalg.norm(ahead))
        if distance > LEAD:
            want = hand + ahead * (LEAD / distance)
        self._step(np.clip((want - mocap) / scale, -1.0, 1.0))

    def _move(self, target) -> str:
        """Drive the fingertip centre toward target; returns how the move ended."""
        import numpy as np

        target = np.asarray(target, dtype=float)
        trail = [self._tcp()]
        for _ in range(MOVE_STEPS):
            if np.linalg.norm(target - trail[-1]) < TOLERANCE or self.done():
                break
            self._drive(target)
            trail.append(self._tcp())
            if len(trail) > STALL_STEPS and np.linalg.norm(trail[-1] - trail[-1 - STALL_STEPS]) < 0.002:
                break
        short = float(np.linalg.norm(target - trail[-1]))
        return "reached" if short < 0.01 else f"stopped {short:.3f} m short of its target"

    def _hold(self, steps: int) -> None:
        here = self._tcp()
        for _ in range(steps):
            if self.done():
                break
            self._drive(here)

    # the Environment interface ----------------------------------------------------------------

    def options(self) -> dict[str, str]:
        out = {}
        for size in SHIFTS:
            for axis, (plus, minus) in AXES.items():
                out[f"{axis}+{size * 100:.0f}cm"] = f"Shift the gripper {size:.2f} m along {plus}."
                out[f"{axis}-{size * 100:.0f}cm"] = f"Shift the gripper {size:.2f} m along {minus}."
        out["close_gripper"] = ("Close the fingers and keep them closed during the following moves; whatever is between the "
                                "fingers is held.")
        out["open_gripper"] = "Open the fingers fully and keep them open during the following moves; anything held is released."
        return out

    def observe(self) -> DecisionRequest:
        points = []
        for name, what in self.points.items():
            points.append(f"- {name.replace('_', ' ')}" + (f" ({what})" if what else "") + f": {fmt(self._point(name))}")
        progress = f"decision {self.decisions + 1}; {self.steps} of {self.horizon} simulation steps used"
        if self.last:
            progress += f"; last action: {self.last}"
        state = {
            "task": f"{self.task_id} (Meta-World): {self.description}",
            "scene": f"{SCENE} {self.scene()}",
            "gripper": (f"fingertip centre at {fmt(self._tcp())}; opening {float(self.obs[3]):.2f} (1.00 = fully open, about "
                        "0.30 = fully closed)"),
            "points": "\n".join(points),
            "progress": progress,
        }
        question = Question(
            type="choice",
            instructions=(f"Choose the robot's next motion primitive. A shift moves the fingertip centre by a fixed step along one "
                          f"axis under a proportional controller, for at most {MOVE_STEPS} simulation steps "
                          f"({MOVE_STEPS * self.control_dt:.1f} s of simulated time); it ends early once the step is done or the "
                          f"gripper is blocked. Then the next decision is asked."),
            criteria=self.options(),
        )
        return DecisionRequest(state=state, questions={"action": question})

    def act(self, choice: dict[str, str]) -> None:
        option = choice["action"]
        self.decisions += 1
        if option in ("close_gripper", "open_gripper"):
            self.grip = 1.0 if option == "close_gripper" else -1.0
            self._hold(GRIP_STEPS)
            result = f"opening now {float(self.obs[3]):.2f}"
        else:  # an axis shift such as "x+5cm"
            target = self._tcp()
            target["xyz".index(option[0])] += (1.0 if option[1] == "+" else -1.0) * float(option[2:-2]) / 100
            result = self._move(target)
        self.last = f"{option} ({result})"

    def idle(self) -> None:
        self._step((0.0, 0.0, 0.0))

    def done(self) -> bool:
        return self.success or self.steps >= self.horizon

    def outcome(self) -> dict[str, Any]:
        return {"success": self.success, "score": float(self.success), "env_steps": self.steps}

    def close(self) -> None:
        env = getattr(self, "env", None)
        if env is not None:
            env.close()

    def card(self) -> dict[str, Any]:
        return {**super().card(), "success_condition": self.success_text}


class _PuckTask(MetaWorldTabletop):
    points = {"puck": "centre of the puck", "goal": ""}

    def _point(self, name):
        return self.env._get_pos_objects()[:3].copy() if name == "puck" else self.env._target_pos.copy()


@register_env
class Reach(_PuckTask):
    name = "embodied/tabletop_loop/reach"
    source = "Meta-World reach-v3 (MuJoCo)"
    task_id = "reach-v3"
    description = "reach a goal position."
    success_text = "the gripper's fingertip centre is within 0.05 m of the goal point."
    scene_text = f"{PUCK}; the goal is a point in the air above the table."


@register_env
class Push(_PuckTask):
    name = "embodied/tabletop_loop/push"
    source = "Meta-World push-v3 (MuJoCo)"
    task_id = "push-v3"
    description = "Push the puck to a goal."
    success_text = "the centre of the puck is within 0.05 m of the goal point."
    scene_text = f"{PUCK}; the goal is a point on the table."


@register_env
class PickPlace(_PuckTask):
    name = "embodied/tabletop_loop/pick_place"
    source = "Meta-World pick-place-v3 (MuJoCo)"
    task_id = "pick-place-v3"
    description = "Pick and place a puck to a goal."
    success_text = "the centre of the puck is within 0.07 m of the goal point."
    scene_text = f"{PUCK}; the goal is a point in the air above the table."


@register_env
class DoorOpen(MetaWorldTabletop):
    name = "embodied/tabletop_loop/door_open"
    source = "Meta-World door-open-v3 (MuJoCo)"
    task_id = "door-open-v3"
    description = "Open a door with a revolving joint."
    success_text = "the x coordinate of the handle point is within 0.08 m of the goal point's x coordinate."
    points = {"handle": "centre of the door handle", "goal": ""}

    def scene(self):
        x, y = self.env.data.xanchor[self.env.model.joint("doorjoint").id][:2]
        return (f"A door stands upright in a frame on the table, facing the robot, with its panel along the x axis. It turns "
                f"about a vertical hinge at its left (-x) end, at (x, y) = ({x:+.3f}, {y:+.3f}), and swings open toward the "
                f"robot. The handle sticks out of the panel toward the robot near the panel's right end.")

    def _point(self, name):
        return self.env.data.geom("handle").xpos.copy() if name == "handle" else self.env._target_pos.copy()


class _Drawer(MetaWorldTabletop):
    points = {"handle": "front of the drawer handle", "goal": ""}
    handle_offset = 0.0  # Meta-World's reported object point minus the front of the handle, along z

    def _point(self, name):
        if name == "handle":
            out = self.env._get_pos_objects().copy()
            out[2] -= self.handle_offset
            return out
        return self.env._target_pos.copy()


@register_env
class DrawerOpen(_Drawer):
    name = "embodied/tabletop_loop/drawer_open"
    source = "Meta-World drawer-open-v3 (MuJoCo)"
    task_id = "drawer-open-v3"
    description = "Open a drawer."
    success_text = "the handle point is within 0.03 m of the goal point."
    scene_text = ("A chest stands on the table with its front facing the robot. Its drawer is closed; it slides along the y axis "
                  "and opens toward the robot (-y), by up to 0.16 m. The handle is a horizontal bar (along x) across the drawer "
                  "front, with a gap between the bar and the front.")


@register_env
class DrawerClose(_Drawer):
    name = "embodied/tabletop_loop/drawer_close"
    source = "Meta-World drawer-close-v3 (MuJoCo)"
    task_id = "drawer-close-v3"
    description = "Push and close a drawer."
    success_text = "the handle point is within 0.0415 m of the goal point."
    scene_text = ("A chest stands on the table with its front facing the robot. Its drawer is pulled out toward the robot; it "
                  "slides along the y axis and closes away from the robot (+y). The handle is a horizontal bar (along x) across "
                  "the drawer front, with a gap between the bar and the front.")
    handle_offset = 0.05  # Meta-World reports a point 0.05 m above the handle for this task
    max_decisions = 40  # twice what the scripted policy needs (19 on average); random choice: 30% of episodes with 80, 20% with 40


@register_env
class ButtonPressTopdown(MetaWorldTabletop):
    name = "embodied/tabletop_loop/button_press_topdown"
    source = "Meta-World button-press-topdown-v3 (MuJoCo)"
    task_id = "button-press-topdown-v3"
    description = "Press a button from the top."
    success_text = "the top of the button is within 0.024 m of the goal point's height."
    scene_text = "A box stands on the table with a round button on its top, facing up; pressing the button moves it down (-z)."
    points = {"button": "centre of the button's top surface", "goal": ""}

    def _point(self, name):
        return self.env._get_pos_objects().copy() if name == "button" else self.env._target_pos.copy()


@register_env
class WindowOpen(MetaWorldTabletop):
    name = "embodied/tabletop_loop/window_open"
    source = "Meta-World window-open-v3 (MuJoCo)"
    task_id = "window-open-v3"
    description = "Push and open a window."
    success_text = "the x coordinate of the handle point is within 0.05 m of the goal point's x coordinate."
    scene_text = ("A window frame stands upright on the table, facing the robot. Its pane is closed; it slides along the x axis "
                  "and opens toward +x, by up to 0.20 m. The handle is a vertical bar that sticks out of the pane toward the "
                  "robot.")
    points = {"handle": "0.025 m to the -x side of the handle bar's axis, at its mid-height", "goal": ""}

    def _point(self, name):
        return self.env._get_pos_objects().copy() if name == "handle" else self.env._target_pos.copy()


@register_env
class WindowClose(WindowOpen):
    name = "embodied/tabletop_loop/window_close"
    source = "Meta-World window-close-v3 (MuJoCo)"
    task_id = "window-close-v3"
    description = "Push and close a window."
    scene_text = ("A window frame stands upright on the table, facing the robot. Its pane is open; it slides along the x axis "
                  "and closes toward -x, by up to 0.20 m. The handle is a vertical bar that sticks out of the pane toward the "
                  "robot.")
    points = {"handle": "0.025 m to the +x side of the handle bar's axis, at its mid-height", "goal": ""}


@register_env
class PegInsertSide(MetaWorldTabletop):
    name = "embodied/tabletop_loop/peg_insert_side"
    source = "Meta-World peg-insert-side-v3 (MuJoCo)"
    task_id = "peg-insert-side-v3"
    description = "Insert a peg sideways."
    success_text = ("sqrt(dx^2 + (2*dy)^2 + (2*dz)^2) is at most 0.07 m, where (dx, dy, dz) is the peg head point minus the "
                    "goal point.")
    scene_text = ("A square peg (0.03 m thick, 0.24 m long) lies on the table along the x axis, its head end toward -x. A box "
                  "stands on the table with a square hole in its +x side; the goal point is inside the hole.")
    points = {"peg": "centre of the peg", "peg_head": "on the peg's axis, 0.02 m in from its head end",
              "hole": "centre of the hole's mouth", "goal": ""}

    def _point(self, name):
        if name == "peg":
            return self.env.data.geom("peg").xpos.copy()
        if name in ("peg_head", "hole"):
            return self.env._get_site_pos({"peg_head": "pegHead", "hole": "hole"}[name])
        return self.env._target_pos.copy()
