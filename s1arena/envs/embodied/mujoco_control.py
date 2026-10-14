"""Low-level control of MuJoCo bodies (Gymnasium v5 environments) with discretized actions.

The state is the environment's observation vector written out as named quantities with units and
sign conventions; each actuator is one score question with five ordered levels spanning its control
range, asked together in one request. The task text states the goal and the reward as documented
by Gymnasium; it never says how to act. Scores are the raw episode returns.
"""

from __future__ import annotations

import math
from typing import Any

from ...registry import register_env
from ...schema import DecisionRequest
from ..base import Environment, level_question, levels

URL = "https://gymnasium.farama.org/environments/mujoco/"
LICENSE = "MIT (Gymnasium); Apache-2.0 (MuJoCo)"


class GymMujoco(Environment):
    leaf = "embodied/mujoco_control"
    license = LICENSE
    url = URL
    tier = "B"
    criteria_source = "Goal and reward adapted from the Gymnasium environment documentation; observation names and signs checked in the simulator"
    mode = "paused"
    episodes = 10
    gym_id = ""
    action_repeat = 1  # env steps per decision
    goal = ""
    actuators: list[tuple[str, str]] = []  # (question name, description) in actuator order
    n_levels = 5

    def reset(self, seed: int) -> None:
        import gymnasium as gym

        self.env = gym.make(self.gym_id)
        self.obs, _ = self.env.reset(seed=seed)
        self.low, self.high = self.env.action_space.low, self.env.action_space.high
        self.values = [levels(float(lo), float(hi), self.n_levels) for lo, hi in zip(self.low, self.high)]
        self.action = [0.0] * len(self.actuators)
        self.total, self.steps, self.ended, self.terminated = 0.0, 0, False, False
        self.x0 = float(self.env.unwrapped.data.qpos[0])
        self.control_dt = float(self.env.unwrapped.dt) * self.action_repeat

    def _step(self) -> None:
        import numpy as np

        self.obs, reward, terminated, truncated, _ = self.env.step(np.array(self.action, dtype=np.float32))
        self.total += float(reward)
        self.steps += 1
        self.terminated = self.terminated or bool(terminated)  # fell over (not just out of time)
        self.ended = self.ended or bool(terminated) or bool(truncated)

    def observe(self) -> DecisionRequest:
        t = self.steps * float(self.env.unwrapped.dt)
        lines = "\n".join(f"- {label}: {value}" for label, value in self.describe(self.obs))
        state = {"task": self.goal, "time": f"{t:.2f} s since the start", "observation": lines}
        questions = {
            name: level_question(f"{text} Choose the command for the next {self.control_dt:.2f} s.", values)
            for (name, text), values in zip(self.actuators, self.values)
        }
        return DecisionRequest(state=state, questions=questions)

    def act(self, choice: dict[str, str]) -> None:
        self.action = [values[int(choice[name])] for (name, _), values in zip(self.actuators, self.values)]
        for _ in range(self.action_repeat):
            if self.ended:
                break
            self._step()

    def idle(self) -> None:
        self._step()

    def done(self) -> bool:
        return self.ended

    def outcome(self) -> dict[str, Any]:
        x = float(self.env.unwrapped.data.qpos[0])
        return {"success": None, "score": round(self.total, 3), "env_steps": self.steps, "x_travelled": round(x - self.x0, 3)}

    def close(self) -> None:
        env = getattr(self, "env", None)
        if env is not None:
            env.close()

    def describe(self, obs) -> list[tuple[str, str]]:
        raise NotImplementedError


def f(v: float, unit: str = "") -> str:
    return f"{v:+.3f}{unit}"


@register_env
class InvertedPendulum(GymMujoco):
    name = "embodied/mujoco_control/inverted_pendulum"
    source = "Gymnasium InvertedPendulum-v5 (MuJoCo)"
    gym_id = "InvertedPendulum-v5"
    max_decisions = 500
    notes = "25 decisions per simulated second; success = the pole is still up after 500 decisions (20 s)."
    goal = ("A cart moves along a horizontal track (x axis, positive = right) with a pole hinged on top. Keep the pole upright: "
            "the episode ends as soon as the pole leans more than 0.2 rad from vertical. Reward: +1 for every step the pole stays up.")
    actuators = [("cart_force", "Horizontal force on the cart (positive pushes the cart to the right).")]

    def describe(self, obs):
        return [("cart position", f(obs[0], " m (0 = track centre, positive = right)")),
                ("pole angle from vertical", f(obs[1], " rad (positive = pole leaning to the right)")),
                ("cart velocity", f(obs[2], " m/s (positive = moving right)")),
                ("pole angular velocity", f(obs[3], " rad/s (positive = tipping further to the right)"))]

    def outcome(self):
        return {**super().outcome(), "success": not self.terminated}


@register_env
class InvertedDoublePendulum(GymMujoco):
    name = "embodied/mujoco_control/inverted_double_pendulum"
    source = "Gymnasium InvertedDoublePendulum-v5 (MuJoCo)"
    gym_id = "InvertedDoublePendulum-v5"
    max_decisions = 500
    notes = "20 decisions per simulated second; success = both poles still up after 500 decisions (25 s)."
    goal = ("A cart moves along a horizontal track (x axis, positive = right) with two poles stacked on hinges. Keep both poles "
            "upright: the episode ends when the tip of the upper pole drops to 1 m or less above the track. Reward per step: 10 for "
            "staying up, minus a penalty for the tip's distance from straight above the cart and for the hinges' angular velocities.")
    actuators = [("cart_force", "Horizontal force on the cart (positive pushes the cart to the right).")]

    def describe(self, obs):
        a1, a2 = math.atan2(obs[1], obs[3]), math.atan2(obs[2], obs[4])
        return [("cart position", f(obs[0], " m (positive = right)")),
                ("lower pole angle from vertical", f(a1, " rad (positive = leaning right)")),
                ("upper pole angle relative to the lower pole", f(a2, " rad (positive = bent further to the right)")),
                ("cart velocity", f(obs[5], " m/s")),
                ("lower pole angular velocity", f(obs[6], " rad/s (positive = tipping right)")),
                ("upper pole relative angular velocity", f(obs[7], " rad/s"))]

    def outcome(self):
        return {**super().outcome(), "success": not self.terminated}


@register_env
class Reacher(GymMujoco):
    name = "embodied/mujoco_control/reacher"
    source = "Gymnasium Reacher-v5 (MuJoCo)"
    gym_id = "Reacher-v5"
    max_decisions = 50
    notes = "A 2-link arm in a horizontal plane; 50 decisions (1 s). The score is the return; final distance is reported too."
    goal = ("A two-link arm lies in a horizontal plane, its base fixed at the origin. Move the fingertip onto the target and keep it "
            "there. Reward per step: minus the fingertip-target distance (m), minus 0.1 times the squared torques.")
    actuators = [("shoulder_torque", "Torque on the shoulder joint at the base (positive turns the arm counter-clockwise seen from above)."),
                 ("elbow_torque", "Torque on the elbow joint between the links (positive bends the outer link counter-clockwise).")]

    def describe(self, obs):
        q0, q1 = math.atan2(obs[2], obs[0]), math.atan2(obs[3], obs[1])
        return [("shoulder angle", f(q0, " rad (counter-clockwise from the +x axis)")),
                ("elbow angle", f(q1, " rad (outer link relative to the inner link, counter-clockwise)")),
                ("shoulder angular velocity", f(obs[6], " rad/s")),
                ("elbow angular velocity", f(obs[7], " rad/s")),
                ("target position (x, y)", f"({obs[4]:+.3f}, {obs[5]:+.3f}) m"),
                ("fingertip minus target (x, y)", f"({obs[8]:+.3f}, {obs[9]:+.3f}) m")]

    def outcome(self):
        out = super().outcome()
        out.pop("x_travelled")  # qpos[0] is a joint angle here
        return {**out, "final_distance": round(math.hypot(self.obs[8], self.obs[9]), 4)}


@register_env
class Hopper(GymMujoco):
    name = "embodied/mujoco_control/hopper"
    source = "Gymnasium Hopper-v5 (MuJoCo)"
    gym_id = "Hopper-v5"
    action_repeat = 2
    max_decisions = 250
    notes = "One-legged hopper in the x-z plane; each decision holds for 2 simulator steps (0.016 s); up to 250 decisions (4 s)."
    goal = ("A one-legged robot (torso, thigh, leg, foot) moves in the vertical x-z plane. Hop forward (+x) as fast as possible "
            "without falling: the episode ends when the torso drops below 0.7 m, tilts more than 0.2 rad, or the state becomes "
            "extreme. Reward per step: forward velocity + 1 for staying healthy - 0.001 times the squared torques.")
    actuators = [("thigh_torque", "Torque at the hip joint between torso and thigh."),
                 ("leg_torque", "Torque at the knee joint between thigh and leg."),
                 ("foot_torque", "Torque at the ankle joint between leg and foot.")]

    def describe(self, obs):
        return [("torso height", f(obs[0], " m")),
                ("torso angle", f(obs[1], " rad (0 = upright)")),
                ("hip (thigh) joint angle", f(obs[2], " rad")),
                ("knee (leg) joint angle", f(obs[3], " rad")),
                ("ankle (foot) joint angle", f(obs[4], " rad")),
                ("forward velocity", f(obs[5], " m/s (positive = forward)")),
                ("vertical velocity", f(obs[6], " m/s")),
                ("torso angular velocity", f(obs[7], " rad/s")),
                ("hip angular velocity", f(obs[8], " rad/s")),
                ("knee angular velocity", f(obs[9], " rad/s")),
                ("ankle angular velocity", f(obs[10], " rad/s"))]


@register_env
class HalfCheetah(GymMujoco):
    name = "embodied/mujoco_control/half_cheetah"
    source = "Gymnasium HalfCheetah-v5 (MuJoCo)"
    gym_id = "HalfCheetah-v5"
    max_decisions = 300
    notes = "Planar two-legged runner; 20 decisions per simulated second; 300 decisions (15 s); it cannot fall, so no early end."
    goal = ("A planar two-legged robot (a back and a front leg, each with thigh, shin and foot joints) runs along the x axis. Run "
            "forward (+x) as fast as possible. Reward per step: forward velocity - 0.1 times the squared torques.")
    actuators = [("back_thigh", "Torque at the back hip joint."), ("back_shin", "Torque at the back knee joint."),
                 ("back_foot", "Torque at the back ankle joint."), ("front_thigh", "Torque at the front hip joint."),
                 ("front_shin", "Torque at the front knee joint."), ("front_foot", "Torque at the front ankle joint.")]

    def describe(self, obs):
        names = ["back hip", "back knee", "back ankle", "front hip", "front knee", "front ankle"]
        out = [("body height", f(obs[0], " m")), ("body pitch angle", f(obs[1], " rad"))]
        out += [(f"{n} angle", f(v, " rad")) for n, v in zip(names, obs[2:8])]
        out += [("forward velocity", f(obs[8], " m/s")), ("vertical velocity", f(obs[9], " m/s")), ("pitch angular velocity", f(obs[10], " rad/s"))]
        out += [(f"{n} angular velocity", f(v, " rad/s")) for n, v in zip(names, obs[11:17])]
        return out
