"""Flying a quadrotor through waypoints in PyFlyt (QuadX-Waypoints), on a real-time clock.

The drone is PyFlyt's QuadX (a Crazyflie 2.x model in PyBullet) in its QuadX-Waypoints task: four
random waypoints inside a 5 m dome, to be reached in order within 0.2 m. The task's default action
space (body rates and thrust at 30 Hz) cannot be flown by a model that answers a few times per
second, so the drone uses PyFlyt's own position flight mode (mode 7: x, y, yaw, z set-points), and
each decision is one manoeuvre that changes the position set-point, the "target point": move it
0.5 m along one of the six axis directions, move it to where the drone is (stop), or keep it. The
onboard controllers fly the drone to the target point and hold it there; the heading set-point is
kept at its start value (the task has no heading targets). One env step is PyFlyt's default agent
step of 1/30 s, so control_dt = 1/30 s; while the model is deciding, the drone keeps flying towards
its last target point.

The state is the drone's position, velocity and attitude read from the simulator, the target point,
and the remaining waypoints, each as map coordinates and as an offset from the drone. Scores are
PyFlyt's sparse return (+100 per waypoint, -100 for touching the floor or leaving the dome, -0.1 per
step otherwise): the default dense return adds shaping terms, including 0.1 / distance to the next
waypoint per control step, which pay more for hovering just outside a waypoint than for reaching
it. The time limit is 30 s instead of PyFlyt's 10 s, which is too short for discrete set-point
manoeuvres: a scripted policy that reads only the request text finishes every seed in 14 to 24 s at
300 ms per decision (at 1 s per decision it needs 26 to 48 s). The limit stays at 30 s because with
-3 per second a longer episode would score an early crash (-100) above hovering without progress
until the end.
"""

from __future__ import annotations

import math
from typing import Any

from ...registry import register_env
from ...schema import DecisionRequest, Question
from ..base import Environment

URL = "https://github.com/jjshoots/PyFlyt"
LICENSE = "MIT"
AGENT_HZ = 30  # PyFlyt's default agent rate: one env step = 4 control steps of 1/120 s
DURATION = 30.0  # seconds (PyFlyt's default for this task is 10 s)
STEP = 0.5  # metres per move of the target point
NUM_TARGETS = 4  # PyFlyt defaults below
REACH = 0.2
DOME = 5.0

INSTRUCTIONS = ("Choose the drone's next manoeuvre. The simulation does not pause while you decide: until your answer "
                "takes effect, the drone keeps flying towards its current target point.")

MOVES = {  # option -> (set-point index in PyFlyt's mode-7 set-point [x, y, yaw, z], sign)
    "move_east": (0, +1), "move_west": (0, -1), "move_north": (1, +1),
    "move_south": (1, -1), "move_up": (3, +1), "move_down": (3, -1),
}

OPTIONS = {
    "move_east": f"Move the target point {STEP} m east (x + {STEP} m).",
    "move_west": f"Move the target point {STEP} m west (x - {STEP} m).",
    "move_north": f"Move the target point {STEP} m north (y + {STEP} m).",
    "move_south": f"Move the target point {STEP} m south (y - {STEP} m).",
    "move_up": f"Move the target point {STEP} m up (z + {STEP} m).",
    "move_down": f"Move the target point {STEP} m down (z - {STEP} m).",
    "stop_here": "Move the target point to where the drone is at the moment this choice takes effect; the drone brakes "
                 "and holds that position.",
    "keep_target": "Leave the target point where it is.",
}

GOAL = ("The goal of this environment is to fly a quadrotor aircraft towards a set of waypoints as fast as possible. "
        "Coordinates are in metres: x east, y north, z height above the floor; the drone starts hovering at (0, 0, 1). "
        f"There are {NUM_TARGETS} waypoints, to be reached in order: a waypoint counts as reached when the drone's centre "
        f"comes within {REACH} m of it, and only the next waypoint can be reached. An onboard position controller flies "
        "the drone towards its target point and holds it there: it commands a velocity of 1 m/s per metre of remaining "
        "distance along each axis, capped at 2 m/s east-west and north-south and 1 m/s up-down, and the drone follows "
        "with some lag. You do not steer the drone directly; each manoeuvre changes the target point. The drone's "
        "heading is 0° when its nose points east and grows counter-clockwise seen from above. The episode ends "
        f"when all waypoints are reached, when the drone touches the floor, when it is more than {DOME:g} m from (0, 0, 0) "
        f"(the flight area is a dome of radius {DOME:g} m), or after {DURATION:g} s. Reward: +100 for each waypoint "
        "reached and -100 for touching the floor or leaving the flight area; otherwise -0.1 per 1/30 s.")

COMPASS8 = ["east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east"]


def offset_words(dx: float, dy: float, dz: float, unit: str = "m") -> str:
    """A vector as distances along the three axes with direction words, e.g. '1.20 m east, 0.40 m south, 0.05 m up'."""
    parts = [(dx, "east", "west"), (dy, "north", "south"), (dz, "up", "down")]
    return ", ".join(f"{abs(v):.2f} {unit} {pos if v > -0.005 else neg}" for v, pos, neg in parts)


def coords(x: float, y: float, z: float) -> str:
    x, y, z = (round(v, 2) + 0.0 for v in (x, y, z))  # no "-0.00"
    return f"x = {x:+.2f}, y = {y:+.2f}, z = {z:.2f} m"


@register_env
class DroneWaypoints(Environment):
    name = "realtime/drone/quadx_waypoints"
    leaf = "realtime/drone"
    source = "PyFlyt QuadX-Waypoints-v4 (Tai et al., 2023)"
    url = URL
    license = LICENSE
    tier = "B"
    criteria_source = ("Task sentence from the PyFlyt documentation; rules, reward and controller behaviour written from "
                       "the PyFlyt 0.29 source (QuadX-Waypoints defaults except flight mode 7, sparse reward and a 30 s "
                       "time limit)")
    mode = "realtime"
    control_dt = 1.0 / AGENT_HZ
    episodes = 10
    max_decisions = int(DURATION * AGENT_HZ) + 2  # PyFlyt truncates after max_steps + 2 env steps
    score_name = "return"
    notes = ("Quadrotor (Crazyflie 2.x model) in PyBullet with 4 random waypoints inside a 5 m dome, reached in order "
             "within 0.2 m. The drone flies in PyFlyt's position mode; each decision moves the target point 0.5 m east, "
             "west, north, south, up or down, moves it to the drone's position, or keeps it (8 options). One env step "
             "is 1/30 s, and while the model decides the drone keeps flying to its last target point. The episode ends "
             "when all waypoints are reached, at a floor contact or on leaving the dome, or after 30 s (PyFlyt's default "
             "is 10 s). Score = PyFlyt's sparse return (+100 per waypoint, -100 for a crash, -0.1 per step otherwise); "
             "success = all 4 waypoints reached. Also reported: waypoints reached.")

    # --- episode -------------------------------------------------------------------------------

    def reset(self, seed: int) -> None:
        import numpy as np
        import PyFlyt.core.aviary as aviary
        from PyFlyt.gym_envs.quadx_envs.quadx_waypoints_env import QuadXWaypointsEnv

        # The Aviary prints a terminal escape sequence (cursor up, blanks) on every construction, which erases the
        # previous line of the runner's progress output; shadowing print in that module silences only this.
        aviary.print = _silent
        self.env = QuadXWaypointsEnv(sparse_reward=True, num_targets=NUM_TARGETS, goal_reach_distance=REACH,
                                     flight_mode=7, flight_dome_size=DOME, max_duration_seconds=DURATION,
                                     angle_representation="euler", agent_hz=AGENT_HZ)
        self.env.reset(seed=seed)
        self.sim = self.env.env  # the PyBullet Aviary
        self.drone = self.sim.drones[0]
        self.setpoint = np.array(self.drone.setpoint, dtype=np.float64)  # mode 7: [x, y, yaw, z], the start pose
        self.total, self.steps, self.ended = 0.0, 0, False
        self.info: dict[str, Any] = {}

    def _step(self) -> None:
        _, reward, terminated, truncated, info = self.env.step(self.setpoint.copy())
        self.total += float(reward)
        self.steps += 1
        self.info = info
        self.ended = self.ended or bool(terminated) or bool(truncated)

    def act(self, choice: dict[str, str]) -> None:
        option = choice["manoeuvre"]
        if option in MOVES:
            index, sign = MOVES[option]
            self.setpoint[index] += sign * STEP
        elif option == "stop_here":
            x, y, z = self.position()
            self.setpoint[0], self.setpoint[1], self.setpoint[3] = x, y, z
        self._step()

    def idle(self) -> None:
        self._step()

    def done(self) -> bool:
        return self.ended

    def default_choice(self, request: DecisionRequest) -> dict[str, str]:
        return {"manoeuvre": "keep_target"}

    def outcome(self) -> dict[str, Any]:
        reached = int(self.env.waypoints.num_targets_reached)
        complete = bool(self.env.waypoints.all_targets_reached)
        out = {
            "success": complete,
            "score": round(self.total, 2),
            "waypoints_reached": reached,
            "crashed": bool(self.info.get("collision", False)),
            "out_of_bounds": bool(self.info.get("out_of_bounds", False)),
            "sim_time_s": round(self.steps / AGENT_HZ, 2),
            "env_steps": self.steps,
        }
        if not complete:
            out["distance_to_next_m"] = round(self.distance(self.env.waypoints.targets[0]), 3)
        return out

    def close(self) -> None:
        env = getattr(self, "env", None)
        if env is not None:
            env.close()
            self.env = None

    # --- simulator state -----------------------------------------------------------------------

    def position(self) -> tuple[float, float, float]:
        pos, _ = self.sim.getBasePositionAndOrientation(self.drone.Id)
        return float(pos[0]), float(pos[1]), float(pos[2])

    def distance(self, point) -> float:
        x, y, z = self.position()
        return math.dist((x, y, z), (float(point[0]), float(point[1]), float(point[2])))

    # --- request -------------------------------------------------------------------------------

    def observe(self) -> DecisionRequest:
        x, y, z = self.position()
        _, orientation = self.sim.getBasePositionAndOrientation(self.drone.Id)
        velocity, _ = self.sim.getBaseVelocity(self.drone.Id)
        vx, vy, vz = (float(v) for v in velocity)
        rotation = self.sim.getMatrixFromQuaternion(orientation)  # row-major 3x3, body to world
        up_x, up_y, up_z = rotation[2], rotation[5], rotation[8]  # the body's up axis (rotor thrust) in the world
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, up_z))))
        heading = round(math.degrees(self.sim.getEulerFromQuaternion(orientation)[2]), 1) + 0.0  # no "-0.0"
        if tilt < 0.05:
            attitude = "level"
        else:
            lean = COMPASS8[int(((math.degrees(math.atan2(up_y, up_x)) % 360) + 22.5) // 45) % 8]
            attitude = f"tilted {tilt:.1f}° from level, leaning towards the {lean}"
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        drone = (f"position: {coords(x, y, z)}; {math.sqrt(x * x + y * y + z * z):.2f} m from (0, 0, 0)\n"
                 f"velocity: {offset_words(vx, vy, vz, 'm/s')} (speed {speed:.2f} m/s)\n"
                 f"attitude: {attitude}; heading {heading:+.1f}°")
        tx, ty, tz = self.setpoint[0], self.setpoint[1], self.setpoint[3]
        target = f"{coords(tx, ty, tz)}; from the drone: {offset_words(tx - x, ty - y, tz - z)}"
        reached = int(self.env.waypoints.num_targets_reached)
        lines = [f"reached {reached} of {NUM_TARGETS}"]
        for k, point in enumerate(self.env.waypoints.targets):
            px, py, pz = (float(v) for v in point)
            label = f"next: waypoint {reached + 1}" if k == 0 else f"then waypoint {reached + 1 + k}"
            lines.append(f"{label}: {coords(px, py, pz)}; from the drone: {offset_words(px - x, py - y, pz - z)}; "
                         f"{math.dist((x, y, z), (px, py, pz)):.2f} m away")
        state = {
            "task": GOAL,
            "time": f"{self.steps / AGENT_HZ:.2f} s of {DURATION:g} s",
            "drone": drone,
            "target point": target,
            "waypoints": "\n".join(lines),
        }
        question = Question(type="choice", instructions=INSTRUCTIONS, criteria=dict(OPTIONS))
        return DecisionRequest(state=state, questions={"manoeuvre": question})


def _silent(*args, **kwargs) -> None:
    return None
