"""Driving in highway-env with its discrete meta-actions, on a real-time clock.

The model drives the ego car of a highway-env scenario (default configuration unless a scenario
says otherwise). The car has highway-env's own steering and speed controllers; each decision is one
meta-action that changes their set-points (target lane, target speed), exactly highway-env's
DiscreteMetaAction. One env step is one policy step of highway-env's default policy frequency (1 Hz,
15 simulation frames), so control_dt = 1 s. While the model is still deciding, no new meta-action is
issued: the car keeps tracking the lane and speed set by its last decision (the IDLE meta-action).

The state lists the objects of the scenario's default Kinematics observation (same objects, same
order), in raw units, with the lane each one is on. Positions follow that observation's convention:
relative to the ego car on the highway and the merge, map coordinates at the roundabout and the
intersection. Scores are the raw returns: the sum of highway-env's rewards over every simulated
second, including the seconds that pass while the model decides.

The multi-agent intersection is a deadlock test: two cars (highway-env places them on the south and
the west approach) are each driven by an independent request per step, answered by separate calls
of the same model; each request shows that car's own state and its own Kinematics observation, in
which the other controlled car is marked, but never the other car's decisions. highway-env gives
both cars the destination "o1", which makes their paths concentric arcs that never meet (a left turn
from the south and a right turn from the west around the same centre), so the destinations are
replaced by pairs whose paths cross or join the same exit, cycling with the seed. A car that has
arrived leaves the road (highway-env keeps it driving under whatever action it is given, and nobody
decides for it any more). The time limit is 20 s instead of 13 s so that a car that waits for the
other can still arrive.

Version 2 of the roundabout and the merge (a random policy succeeded in 60% and 30% of
their ten version-1 episodes; over 1,000 and 400 seeds, 66.5% and 17.5%). Success was little more than
"no collision"; now it needs progress: at the roundabout the car must have driven 25 m along its exit
road within the 11 s, at the merge the vehicle from the access ramp must have merged before the car
passes the merge (the scenario's own objective). Random success drops to 11.0% and 12.5%. The two-car
intersection keeps version 1: its 2 of 10 random wins were noise (17.2% over 1,000 seeds), and denser
traffic, tried, left a policy in which each car decides on its own only 51% of the episodes.
"""

from __future__ import annotations

import math
from typing import Any

from ...registry import register_env
from ...schema import DecisionRequest, Question
from ..base import Environment

URL = "https://github.com/Farama-Foundation/HighwayEnv"
LICENSE = "MIT"

# intersection-v0 overwrites these IDMVehicle class attributes (globally) when it builds its scene;
# every reset restores highway-env's defaults so an episode never depends on what ran before it.
IDM_DEFAULTS = {"DISTANCE_WANTED": 10.0, "COMFORT_ACC_MAX": 3.0, "COMFORT_ACC_MIN": -5.0}

CONTROLS = ("Your car steers along its lane and holds a target speed by itself; you only change these set-points with "
            "meta-actions. The controllers track the set-points only; they do not react to other vehicles.")

META = {
    "LANE_LEFT": "Set the target lane to the next lane on your left (no effect if there is none).",
    "IDLE": "Keep the current target lane and target speed.",
    "LANE_RIGHT": "Set the target lane to the next lane on your right (no effect if there is none).",
    "FASTER": "Set the target speed one level above the level closest to your current speed (no effect at the top level).",
    "SLOWER": "Set the target speed one level below the level closest to your current speed (no effect at the bottom level).",
}

INSTRUCTIONS = ("Choose your car's next meta-action. The simulation does not pause while you decide: until your answer "
                "takes effect, your car keeps its current target lane and target speed.")

COMPASS = {"s": "south", "e": "east", "n": "north", "w": "west"}
INTERSECTION_ROADS = {"0": "south", "1": "west", "2": "north", "3": "east"}


def bearing(heading: float) -> str:
    """highway-env heading (radians, x east, y south) as a compass bearing."""
    deg = (90.0 + math.degrees(heading)) % 360.0
    names = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]
    return f"{deg:.0f}° ({names[int((deg + 22.5) // 45) % 8]})"


class HighwayDriving(Environment):
    leaf = "realtime/highway"
    url = URL
    license = LICENSE
    tier = "B"
    criteria_source = ("Task sentence from the highway-env documentation; rules, reward and meta-actions written from the "
                       "highway-env 1.12 source (default scenario configuration)")
    mode = "realtime"
    control_dt = 1.0  # highway-env's default policy frequency: one meta-action per simulated second
    episodes = 10
    score_name = "return"
    gym_id = ""
    goal = ""
    frame = "road"  # "road": positions relative to you along/across the road; "map": x east, y north
    time_limit = 0.0  # extra bound on simulated seconds for scenarios without one (0 = the scenario's own)
    config_overrides: dict = {}  # highway-env configuration keys changed from the scenario's defaults

    # --- episode -------------------------------------------------------------------------------

    def make_env(self, seed: int):
        import gymnasium as gym
        import highway_env  # noqa: F401  (registers the scenarios)
        from gymnasium.envs.registration import load_env_creator
        from highway_env.vehicle.behavior import IDMVehicle

        for key, value in IDM_DEFAULTS.items():
            setattr(IDMVehicle, key, value)
        spec = gym.spec(self.gym_id)  # these ids register no wrappers; building from the entry point skips
        kwargs = dict(spec.kwargs)  # gym.make's "newer version exists" warning
        if self.config_overrides:
            kwargs["config"] = {**kwargs.get("config", {}), **self.config_overrides}
        env = load_env_creator(spec.entry_point)(**kwargs)
        env.reset(seed=seed)
        return env

    def reset(self, seed: int) -> None:
        self.env = self.make_env(seed)
        self.u = self.env.unwrapped
        self.ego = self.u.vehicle
        self.names = dict(self.u.action_type.actions)  # action index -> meta-action name
        self.index = {name: i for i, name in self.names.items()}
        self.total, self.steps, self.distance, self.speeds = 0.0, 0, 0.0, []
        self.ended = False
        self.arrived_at: float | None = None

    def _step(self, action: int) -> None:
        import numpy as np

        before = self.ego.position.copy()
        _, reward, terminated, truncated, _ = self.env.step(action)
        self.total += float(reward)
        self.steps += 1
        self.distance += float(np.linalg.norm(self.ego.position - before))
        self.speeds.append(float(self.ego.speed))
        if self.arrived_at is None and self.arrived():
            self.arrived_at = round(float(self.u.time), 1)
        over_time = bool(self.time_limit) and self.u.time >= self.time_limit
        self.ended = self.ended or bool(terminated) or bool(truncated) or over_time

    def act(self, choice: dict[str, str]) -> None:
        self._step(self.index[choice["action"]])

    def idle(self) -> None:
        self._step(self.index["IDLE"])

    def done(self) -> bool:
        return self.ended

    def default_choice(self, request: DecisionRequest) -> dict[str, str]:
        return {"action": "IDLE"}

    def completed(self) -> bool:
        """The episode ran to its natural end (not cut short by the runner's decision cap)."""
        return self.u.time >= self.u.config["duration"]

    def arrived(self) -> bool:
        """The car has reached the goal that success requires (scenarios whose success needs progress)."""
        return False

    def outcome(self) -> dict[str, Any]:
        crashed = bool(self.ego.crashed)
        return {
            "success": self.completed() and not crashed,
            "score": round(self.total, 4),
            "crashed": crashed,
            "distance_m": round(self.distance, 1),
            "mean_speed_mps": round(sum(self.speeds) / len(self.speeds), 2) if self.speeds else 0.0,
            "sim_time_s": round(float(self.u.time), 1),
            "env_steps": self.steps,
        }

    def close(self) -> None:
        env = getattr(self, "env", None)
        if env is not None:
            env.close()

    # --- request -------------------------------------------------------------------------------

    def observe(self) -> DecisionRequest:
        return self.request(self.goal)

    def request(self, goal: str) -> DecisionRequest:
        """The request for the car in self.ego."""
        limit = self.u.config.get("duration")  # merge-v0 has none (time_limit is only a safeguard there)
        clock = f"{self.u.time:.0f} s of {limit:.0f} s" if limit else f"{self.u.time:.0f} s since the start"
        rows = [self.row(v) for v in self.observed()]
        state = {
            "task": goal,
            "time": clock,
            "you": self.describe_ego(),
            "others": self.legend() + ("\n" + "\n".join(rows) if rows else "\n(none)"),
        }
        options = {name: META[name] for name in self.names.values()}
        question = Question(type="choice", instructions=INSTRUCTIONS, criteria=options)
        return DecisionRequest(state=state, questions={"action": question})

    def observed(self) -> list:
        """The objects of the scenario's Kinematics observation, selected and ordered as highway-env does."""
        obs = self.u.observation_type
        return self.u.road.close_objects_to(self.ego, self.u.PERCEPTION_DISTANCE, count=obs.vehicles_count - 1,
                                            see_behind=obs.see_behind, sort=obs.order == "sorted",
                                            vehicles_only=not obs.include_obstacles)

    def speed_levels(self) -> str:
        return ", ".join(f"{s:g}" for s in self.ego.target_speeds) + " m/s"

    def lane_text(self) -> str:
        ego = self.ego
        text = self.place(ego.lane_index)
        if ego.target_lane_index[:2] == ego.lane_index[:2] and ego.target_lane_index[2] != ego.lane_index[2]:
            text += f", changing to lane {ego.target_lane_index[2]}"
        return text

    def describe_ego(self) -> str:
        ego = self.ego
        if self.frame == "road":
            where = self.lane_text()
            return (f"{where}; speed {ego.speed:.1f} m/s; target speed {ego.target_speed:g} m/s (levels {self.speed_levels()})")
        return (f"position x={ego.position[0]:+.1f} m, y={-ego.position[1]:+.1f} m ({self.lane_text()}); heading "
                f"{bearing(ego.heading)}; speed {ego.speed:.1f} m/s; target speed {ego.target_speed:g} m/s "
                f"(levels {self.speed_levels()})")

    def legend(self) -> str:
        if self.frame == "road":
            return ("Nearest first. dx = metres ahead of you (negative = behind), dy = metres to your right (negative = left), "
                    "dvx and dvy = the object's velocity minus yours along and across the road (m/s).")
        return ("Nearest first. Map coordinates in metres (x east, y north), velocity (vx, vy) in m/s, heading as a compass "
                "bearing, and the distance from your car.")

    def kind(self, v) -> str:
        from highway_env.vehicle.objects import Obstacle

        return "obstacle" if isinstance(v, Obstacle) else "vehicle"

    def row(self, v) -> str:
        import numpy as np

        what = self.kind(v)
        where = self.place(v.lane_index)
        if self.frame == "road":
            d = v.to_dict(self.ego)
            return f"- {what}, {where}: dx={d['x']:+.1f}, dy={d['y']:+.1f}, dvx={d['vx']:+.1f}, dvy={d['vy']:+.1f}"
        vx, vy = (v.velocity if what != "obstacle" else (0.0, 0.0))
        gap = float(np.linalg.norm(v.position - self.ego.position))
        return (f"- {what}, {where}: x={v.position[0]:+.1f}, y={-v.position[1]:+.1f}, vx={vx:+.1f}, vy={-vy:+.1f}, "
                f"heading {bearing(v.heading)}, {gap:.1f} m from you")

    def place(self, lane_index) -> str:
        return f"lane {lane_index[2]}"


@register_env
class Highway(HighwayDriving):
    name = "realtime/highway/highway"
    source = "highway-env highway-v0 (Leurent, 2018)"
    gym_id = "highway-v0"
    max_decisions = 40
    notes = ("Four-lane highway with 50 vehicles; one meta-action per simulated second, and while the model decides the car "
             "keeps its last target lane and speed; the episode ends after 40 s or at the first collision. Success = no "
             "collision. Also reported: distance driven and mean speed.")
    goal = ("In this task, the ego-vehicle (your car) is driving on a multilane highway populated with other vehicles. The "
            "agent's objective is to reach a high speed while avoiding collisions with neighbouring vehicles. Driving on "
            "the right side of the road is also rewarded. The road is straight, all traffic drives in your direction, and "
            "there are 4 lanes, numbered 0 (leftmost) to 3 (rightmost), each 4 m wide. " + CONTROLS + " The episode ends after "
            "40 s or at the first collision. Reward per second, between 0 and 1: (0.4 × speed score + 0.1 × lane score − 1 "
            "if crashed + 1) / 1.5, where the speed score rises linearly from 0 at a forward speed of 20 m/s to 1 at 30 "
            "m/s (clipped) and the lane score is your target lane number divided by 3; 0 while off the road.")


@register_env
class Merge(HighwayDriving):
    name = "realtime/highway/merge"
    source = "highway-env merge-v0 (Leurent, 2018)"
    gym_id = "merge-v0"
    max_decisions = 30
    time_limit = 30.0
    version = 2
    notes = ("Two-lane highway with an access ramp; one meta-action per simulated second, and while the model decides the "
             "car keeps its last target lane and speed; the episode ends once the car has passed the merge (x > 370 m, about "
             "11 to 17 s) or at the first collision; 30 s bound as a safeguard. Success = the car passes x = 370 m without a "
             "collision after merge-v0's merging vehicle (the car that starts on the access ramp) has merged, i.e. its "
             "closest lane is lane 0 or 1 of the highway at the car's pass. Version 2: version 1 counted "
             "every collision-free pass; the car cannot drive slower than 20 m/s, so it always passes within 17 s and "
             "success was only 'no collision' (random 3 of 10). The scenario's own objective is to keep a high speed while "
             "making room for the merging vehicle, so version 2 requires the merge to happen before the pass (the ego "
             "starts on the highway, so it is the ramp car that merges). merge-v0 has no traffic-density parameter.")
    goal = ("In this task, the ego-vehicle (your car) starts on a main highway but soon approaches a road junction with "
            "incoming vehicles on the access ramp. The agent's objective is now to maintain a high speed while making room "
            "for the vehicles so that they can safely merge in the traffic. The highway is straight with 2 lanes, 0 (left) "
            "and 1 (right), each 4 m wide; x is the distance along the highway. The ramp joins from the right: its merging "
            "lane runs beside lane 1 from x = 230 m to x = 310 m, where it ends at an obstacle; ramp vehicles move into "
            "lane 1 there. You start at x = 30 m in lane 1. " + CONTROLS + " The episode ends when you pass x = 370 m or at "
            "the first collision. The task is done if you pass x = 370 m without a collision after the vehicle coming from "
            "the access ramp has merged into the highway (it must not still be on the access ramp or the merging lane when "
            "you pass). Reward per second: (0.2 × speed score + 0.1 × lane score − 0.05 if you chose a lane change − 0.5 × "
            "ramp slowdown − 1 if crashed + 1.5) / 1.8, where the speed score is (speed − 20 m/s) / 10 m/s, the lane score "
            "is 1 in lane 1 and 0 in lane 0, and the ramp slowdown sums (desired speed − speed) / desired speed over the "
            "vehicles on the merging lane.")

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.ramp = next(v for v in self.u.road.vehicles if v.lane_index[:2] == ("j", "k"))  # merge-v0's merging vehicle
        self.ramp_merged_at: float | None = None

    def _step(self, action: int) -> None:
        super()._step(action)
        a, b, k = self.ramp.lane_index
        if self.ramp_merged_at is None and (a, b) in {("b", "c"), ("c", "d")} and k < 2:  # lane 0 or 1, not the merging lane
            self.ramp_merged_at = round(float(self.u.time), 1)

    def place(self, lane_index) -> str:
        a, b, k = lane_index
        if (a, b) in {("j", "k"), ("k", "b")}:
            return "access ramp"
        if (a, b, k) == ("b", "c", 2):
            return "merging lane"
        return f"lane {k}"

    def describe_ego(self) -> str:
        return f"x = {self.ego.position[0]:.1f} m; " + super().describe_ego()

    def completed(self) -> bool:
        return bool(self.ego.position[0] > 370)  # merge-v0's own end: the car has passed the merge

    def arrived(self) -> bool:
        return self.completed()

    def outcome(self) -> dict[str, Any]:
        out = super().outcome()
        merged = self.ramp_merged_at is not None and self.arrived_at is not None and self.ramp_merged_at <= self.arrived_at
        return {**out, "success": out["success"] and merged, "passed_s": self.arrived_at, "ramp_merged_s": self.ramp_merged_at}


@register_env
class Roundabout(HighwayDriving):
    name = "realtime/highway/roundabout"
    source = "highway-env roundabout-v0 (Leurent, 2018)"
    gym_id = "roundabout-v0"
    frame = "map"
    max_decisions = 11
    version = 2
    exit_distance = 25.0  # metres along the north exit road that count as arrived (highway-env's intersection: 25 m)
    notes = ("Two-lane roundabout; the car follows its route (south entry to north exit) by itself; one meta-action per "
             "simulated second, and while the model decides the car keeps its last target lane and speed; the episode ends "
             "after 11 s or at the first collision. Success = within the 11 s the car has driven 25 m along the north exit "
             "road (lane nxs-nxr, past the exit bend: 67.5 m north of the centre), with no collision in the episode; 25 m "
             "is where highway-env's intersection counts a car as arrived. Version 2: version 1 counted "
             "every collision-free episode, so stopping before the roundabout succeeded (random 6 of 10). Keeping the "
             "initial 8 m/s the car would get there after about 16 s, at 16 m/s after about 8 s. roundabout-v0 has no "
             "traffic-density parameter.")
    goal = ("In this task, the ego-vehicle (your car) is approaching a roundabout with flowing traffic. It will follow its "
            "planned route automatically, but has to handle lane changes and longitudinal control to pass the roundabout "
            "as fast as possible while avoiding collisions. Map coordinates are in metres, x east and y north. The "
            "roundabout is centred at (0, 0); traffic circulates counter-clockwise on two lanes: lane 0 (inner, left, "
            "radius 20 m) and lane 1 (outer, right, radius 24 m). Roads enter and leave it from the north, east, south and "
            "west, one lane each way. You come from the south and your route leaves by the north exit. " + CONTROLS + " The "
            "episode ends after 11 s or at the first collision. The task is done if within the 11 s you have driven 25 m "
            "along the north exit road, which starts after the exit bend, 42.5 m north of the centre (so y ≥ 67.5 m), "
            "without a collision. Reward per second: (0.2 × target speed level / 2 − 0.05 if you chose a lane change − 1 "
            "if crashed + 1) / 1.2, where the target speed level is 0, 1 or 2 for 0, 8 or 16 m/s; 0 while off the road.")

    def arrived(self) -> bool:
        on_exit = self.ego.lane_index[:2] == ("nxs", "nxr")  # the end of its planned route (node nxs): the exit road
        return on_exit and self.ego.lane.local_coordinates(self.ego.position)[0] >= self.exit_distance

    def outcome(self) -> dict[str, Any]:
        out = super().outcome()
        arrived = self.arrived_at is not None
        return {**out, "success": arrived and not out["crashed"], "arrived": arrived, "arrival_s": self.arrived_at}

    def place(self, lane_index) -> str:
        a, b, k = lane_index
        ring = {"se", "ex", "ee", "nx", "ne", "wx", "we", "sx"}
        if a in ring and b in ring:
            return f"roundabout lane {k} ({'inner' if k == 0 else 'outer'})"
        if a[1:] in ("er", "es"):
            return f"{COMPASS[a[0]]} entry road"
        return f"{COMPASS[a[0]]} exit road"


@register_env
class Intersection(HighwayDriving):
    name = "realtime/highway/intersection"
    source = "highway-env intersection-v0 (Leurent, 2018)"
    gym_id = "intersection-v0"
    frame = "map"
    max_decisions = 13
    notes = ("Four-way intersection with dense traffic; the car turns left along its route by itself and the model controls "
             "the speed only (3 meta-actions, one per simulated second; while the model decides the car keeps its last "
             "target speed); the episode ends on arrival (25 m along the exit road), after 13 s, or at the first collision. "
             "Success = arrived without a collision.")
    goal = ("An intersection negotiation task with dense traffic; you drive the ego-vehicle. Map coordinates are in "
            "metres, x east and y north. A four-way intersection is centred at (0, 0); each road has one lane in each "
            "direction and traffic drives on the right. You approach from the south, heading north, and your route turns "
            "left onto the west road. Your car follows the route and holds a target speed by itself; you only change the "
            "target speed with meta-actions, and the controllers do not react to other vehicles. Right of way, which the "
            "other drivers respect: the east-west road has priority over the north-south road, and on each road straight "
            "and right-turning traffic has priority over left-turning traffic. The episode ends when you arrive (25 m "
            "along the west road), after 13 s, or at the first collision. Reward per second: 1 once you have arrived; "
            "otherwise a speed score (0 at 7 m/s or less, rising linearly to 1 at 9 m/s) minus 5 if crashed; 0 while off "
            "the road.")

    def place(self, lane_index) -> str:
        a, b, _ = lane_index
        if a.startswith("o"):
            return f"{INTERSECTION_ROADS[a[1]]} approach"
        if a.startswith("ir"):
            return "inside the intersection"
        return f"{INTERSECTION_ROADS[b[1]]} exit road"

    def lane_text(self) -> str:
        return self.place(self.ego.lane_index)

    def outcome(self) -> dict[str, Any]:
        out = super().outcome()
        arrived = bool(self.u.has_arrived(self.ego))
        return {**out, "success": arrived and not out["crashed"], "arrived": arrived}


# Multi-agent intersection. highway-env puts controlled car i on the approach of road i % 4: south, then west.
MA_START = {"south_car": ("south", "north"), "west_car": ("west", "east")}  # approach, heading on it
# (south car, west car) destinations, cycled by seed: crossing paths (straight/straight, left/straight, left/left) or two
# routes into the same exit (straight and a left turn into the north road, a right turn and straight into the east road)
MA_ROUTES = [("o2", "o3"), ("o1", "o3"), ("o1", "o2"), ("o2", "o2"), ("o3", "o3")]
MA_MOVES = {("south_car", "o1"): "left", ("south_car", "o2"): "straight", ("south_car", "o3"): "right",
            ("west_car", "o0"): "right", ("west_car", "o2"): "left", ("west_car", "o3"): "straight"}
STOPPED_MPS = 0.5  # a controlled car slower than this counts as stopped
DEADLOCK_S = 3.0  # no controlled car arrived and all of them stopped for this long: deadlock
MA_DURATION = 20  # seconds (highway-env's default is 13 s)


@register_env
class IntersectionMultiAgent(Intersection):
    name = "realtime/highway/intersection_multi_agent"
    source = "highway-env intersection-multi-agent-v0 (Leurent, 2018), destinations changed so that the paths conflict"
    gym_id = "intersection-multi-agent-v0"
    agents = list(MA_START)
    config_overrides = {"duration": MA_DURATION}
    max_decisions = MA_DURATION
    criteria_source = ("Task sentence from the highway-env documentation; rules, reward and meta-actions written from the "
                       "highway-env 1.12 source (intersection-multi-agent-v0, duration 20 s, destinations set per seed)")
    notes = ("Deadlock test: the intersection-v0 crossing with two controlled cars, from the south and from the west, each "
             "driven by its own request per simulated second (independent calls of the same model; each car sees its own "
             "state and the traffic, with the other controlled car marked, but not the other car's decisions). "
             "highway-env's default destinations give paths that never meet, so each seed uses one of five destination "
             "pairs whose paths cross or join the same exit. Speed only (3 meta-actions per car); while the model "
             "decides, both cars keep their target speeds. A car that has arrived (25 m along its exit road) leaves the "
             "road. The episode ends when both cars have arrived, at the first collision of either, or after 20 s "
             "(highway-env's default is 13 s). Score = the shared return (the mean of the two cars' rewards per "
             "second); success = both arrived without a collision. Also reported: deadlock (no car arrived and both "
             "below 0.5 m/s for at least 3 s of simulated time) and crashes.")

    def reset(self, seed: int) -> None:
        self.env = self.make_env(seed)
        self.u = self.env.unwrapped
        self.cars = dict(zip(self.agents, self.u.controlled_vehicles))
        self.routes = dict(zip(self.agents, MA_ROUTES[seed % len(MA_ROUTES)]))
        for agent, car in self.cars.items():  # before the first step, so this equals configuring the destinations
            car.plan_route_to(self.routes[agent])
        self.views = dict(zip(self.agents, self.u.observation_type.agents_observation_types))
        self.names = dict(self.u.action_type.agents_action_types[0].actions)
        self.index = {name: i for i, name in self.names.items()}
        self.ego, self.view = self.cars[self.agents[0]], self.views[self.agents[0]]
        self.total, self.steps, self.ended = 0.0, 0, False
        self.frames = self.standstill = self.longest = 0
        self.deadlock = self.collided = False
        self.arrived_at: dict[str, float | None] = {agent: None for agent in self.agents}
        self.crashed_at: dict[str, float | None] = {agent: None for agent in self.agents}
        road = self.u.road
        step = road.step

        def watched_step(dt: float) -> None:  # every simulation frame (1/15 s): arrivals, crashes, standstills
            step(dt)
            self._frame(dt)

        road.step = watched_step

    def _frame(self, dt: float) -> None:
        self.frames += 1
        t = round(self.frames * dt, 2)
        cars = list(self.cars.values())
        on_road = [car for car in cars if any(v is car for v in self.u.road.vehicles)]
        if len(on_road) == len(cars):
            # highway-env's own test, repeated for the two controlled cars: a pair that overlaps, or will within dt, crashes
            intersecting, will_intersect, _ = cars[0]._is_colliding(cars[1], dt)
            self.collided = self.collided or bool(intersecting) or bool(will_intersect)
        for agent, car in self.cars.items():
            if self.crashed_at[agent] is None and car.crashed:
                self.crashed_at[agent] = t
            if self.arrived_at[agent] is None and self.u.has_arrived(car):
                self.arrived_at[agent] = t
                if not car.crashed:  # it leaves the road, as simulated vehicles do at the end of their exit road
                    self.u.road.vehicles = [v for v in self.u.road.vehicles if v is not car]
        if all(car.speed < STOPPED_MPS for car in cars) and not any(self.u.has_arrived(car) for car in cars):
            self.standstill += 1
            self.longest = max(self.longest, self.standstill)
            self.deadlock = self.deadlock or self.standstill * dt >= DEADLOCK_S - 1e-9
        else:
            self.standstill = 0

    def _step(self, actions: tuple) -> None:
        _, reward, terminated, truncated, _ = self.env.step(actions)
        self.total += float(reward)
        self.steps += 1
        self.ended = self.ended or bool(terminated) or bool(truncated)

    def active_agents(self) -> list[str]:
        return [agent for agent, car in self.cars.items() if not self.u.has_arrived(car) and not car.crashed]

    def act(self, choice: dict[str, str]) -> None:
        raise NotImplementedError("multi-agent environment: use act_all")

    def act_all(self, choices: dict[str, dict[str, str]]) -> None:
        idle = self.index["IDLE"]  # for cars that have arrived: they have left the road and make no more decisions
        self._step(tuple(self.index[choices[agent]["action"]] if agent in choices else idle for agent in self.agents))

    def idle(self) -> None:
        self._step(tuple(self.index["IDLE"] for _ in self.agents))

    def observe(self) -> DecisionRequest:
        """The first active car's request (the runner asks every active car through observe_agent)."""
        return self.observe_agent((self.active_agents() or self.agents)[0])

    def observe_agent(self, agent: str) -> DecisionRequest:
        self.ego, self.view = self.cars[agent], self.views[agent]
        return self.request(self.goal_for(agent))

    def observed(self) -> list:
        """The objects of this car's own Kinematics observation, selected and ordered as highway-env does."""
        obs = self.view
        return self.u.road.close_objects_to(self.ego, self.u.PERCEPTION_DISTANCE, count=obs.vehicles_count - 1,
                                            see_behind=obs.see_behind, sort=obs.order == "sorted",
                                            vehicles_only=not obs.include_obstacles)

    def kind(self, v) -> str:
        if v is not self.ego and any(v is car for car in self.cars.values()):
            return "other controlled car"
        return super().kind(v)

    def goal_for(self, agent: str) -> str:
        come, head = MA_START[agent]
        other = next(a for a in self.agents if a != agent)
        return ("An intersection negotiation task with dense traffic; two of the cars are controlled, and you drive one of "
                "them. Map coordinates are in metres, x east and y north. A four-way intersection is centred at (0, 0); "
                "each road has one lane in each direction and traffic drives on the right. "
                f"You approach from the {come}, heading {head}, and your route {self.route_words(agent)}. "
                "Your car follows the route and holds a target speed by itself; you only change the target speed with "
                "meta-actions, and the controllers do not react to other vehicles. "
                f"The other controlled car approaches from the {MA_START[other][0]}; another driver decides for it "
                "independently, from that car's own view of the traffic, and you do not see those decisions. All other "
                "vehicles are simulated drivers who respect the right of way: the east-west road has priority over the "
                "north-south road, and on each road straight and right-turning traffic has priority over left-turning "
                "traffic. A controlled car arrives 25 m along its exit road; it then leaves the road and makes no more "
                "decisions. The episode ends when both controlled cars have arrived, at the first collision of either of "
                f"them, or after {MA_DURATION} s. Reward per second, shared by the two "
                "controlled cars: the mean of their rewards, each 1 once that car has arrived, otherwise its speed score "
                "(0 at 7 m/s or less, rising linearly to 1 at 9 m/s) minus 5 if it has crashed, and 0 while it is off "
                "the road.")

    def route_words(self, agent: str) -> str:
        move, road = MA_MOVES[(agent, self.routes[agent])], INTERSECTION_ROADS[self.routes[agent][1]]
        return f"goes straight on to the {road} road" if move == "straight" else f"turns {move} onto the {road} road"

    def outcome(self) -> dict[str, Any]:
        arrived = {agent: bool(self.u.has_arrived(car)) for agent, car in self.cars.items()}
        crashed = [agent for agent, car in self.cars.items() if car.crashed]
        routes = {}
        for agent, dest in self.routes.items():
            move = MA_MOVES[(agent, dest)]
            kind = "straight" if move == "straight" else f"{move} turn"
            routes[agent] = f"{MA_START[agent][0]} to {INTERSECTION_ROADS[dest[1]]} ({kind})"
        return {
            "success": all(arrived.values()) and not crashed,
            "score": round(self.total, 4),
            "deadlock": self.deadlock,
            "crashed": bool(crashed),
            "crashed_cars": crashed,
            "cars_collided": self.collided,  # the two controlled cars hit each other
            "arrived_cars": sum(arrived.values()),
            "arrival_s": self.arrived_at,
            "crash_s": self.crashed_at,
            "longest_standstill_s": round(self.longest / self.u.config["simulation_frequency"], 2),
            "routes": routes,
            "sim_time_s": round(float(self.u.time), 1),
            "env_steps": self.steps,
        }
