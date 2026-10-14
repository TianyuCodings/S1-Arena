"""Check that the realtime Doom and highway harnesses can still be solved with the options the model gets.

A scripted policy plays each environment through its own interface (observe / act, or observe_agent / act_all for
the two-car intersection), in realtime mode with the same latency charge as the runner (s1arena/envs/runner.py
play_episode: round(latency / control_dt) idle steps, the previous action held, before each decision takes effect),
and chooses only among the options of the current request. It reports the success rate per environment, so a
harness change that makes a task unsolvable shows up (a too-easy environment is made harder, then
checked to be still solvable).

- realtime/doom/basic: aim, then shoot once. The policy knows its own button history and Doom's movement physics
  (per tic: strafe thrust 0.75 map units per tic, then friction 0.90625) and the pistol's timing (it fires on the 5th
  tic of a press and fires again if the button is still held 14 tics later), all measured in the simulator. At each
  decision it searches the next three decisions (every option, the hold included) for the earliest shot predicted to
  pass within 20 units of the monster's centre (its radius is 31); the monster's lateral offset is the one the request
  shows (distance x sine of the bearing). It shoots only when a hit is predicted.
- realtime/highway/merge: IDM car following and MOBIL-like lane changes that make room for the ramp car (MergeDrive).
- realtime/highway/roundabout: speed choice by predicted conflicts with the listed vehicles (SpeedOnly, Predictor).
- realtime/highway/intersection and intersection_multi_agent: gap acceptance at the stop line (Crossing, TwoCars),
  with the other vehicles' planned routes, which highway-env's own drivers use to predict each other; --planner plays
  the two-car intersection with a foresighted joint planner instead (an upper bound).
- realtime/atari/freeway: waits for gaps (FreewayDodge). It searches the next two decisions (every option, the latency
  hold included) in a copy of the emulator and takes the first input of the sequence that ends highest. Freeway has no
  success criterion, so the score is reported; --baseline plays the trivial policy instead (always up).

Usage (on a machine with the realtime extras installed):
    python scripts/realtime_oracle.py --env realtime/doom/basic [--seeds 10] [--latency-ms 320] [--workers 8]
"""

from __future__ import annotations

import argparse
import copy
import itertools
import math
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from s1arena.registry import envs_under  # noqa: E402

# --- driver ------------------------------------------------------------------------------------


def play(env, seed: int, policy, latency_ms: float) -> dict:
    """One episode as runner.play_episode plays it, with the scripted policy in place of the model."""
    env.reset(seed)
    hold = round(latency_ms / 1000 / env.control_dt)
    policy.reset(env, hold)
    decisions = 0
    try:
        while not env.done() and decisions < env.max_decisions:
            if env.agents:
                requests = {agent: env.observe_agent(agent) for agent in env.active_agents()}
                choice = {agent: policy.decide(env, request, agent) for agent, request in requests.items()}
                for agent, request in requests.items():
                    check(choice[agent], request)
            else:
                request = env.observe()
                choice = policy.decide(env, request)
                check(choice, request)
            for _ in range(hold):
                if env.done():
                    break
                env.idle()
            if env.done():
                break
            env.act_all(choice) if env.agents else env.act(choice)
            decisions += 1
        out = env.outcome()
    finally:
        env.close()
    return {**out, "decisions": decisions}


def check(choice: dict[str, str], request) -> None:
    for name, option in choice.items():
        if option not in request.questions[name].option_ids():
            raise ValueError(f"{option} is not an option of {name}: {request.questions[name].option_ids()}")


# --- Doom basic --------------------------------------------------------------------------------

THRUST, FRICTION = 0.75, 0.90625  # per tic: strafe thrust (map units per tic) added to the speed, then friction
FIRE_DELAY, REFIRE_CHECK, READY = 4, 14, 19  # pistol, in tics after a firing sequence starts: shot, refire check, ready
AIM = 20.0  # shoot only if the shot is predicted to pass this close to the monster's centre (its radius is 31 units)
DIRECTION = {"MOVE_LEFT": 1, "MOVE_RIGHT": -1}  # facing along +x, strafing left moves toward +y
MONSTER = re.compile(r"Cacodemon \(Monster\): (?P<distance>\d+) units away, (?:straight ahead|(?P<degrees>\d+)° to your "
                     r"(?P<side>left|right))")
BULLETS = re.compile(r"ammunition \(pistol bullets\) (\d+)")


def direction(buttons: list[str]) -> int:
    return sum(DIRECTION.get(b, 0) for b in buttons)


def pressed(option: str) -> list[str]:
    """The buttons of an option id ("NOOP", "ATTACK", "MOVE_LEFT+ATTACK", ...)."""
    return [] if option == "NOOP" else option.split("+")


class Pistol:
    """The pistol's firing sequence as measured in the simulator: a press on a ready pistol starts a sequence that
    fires on its 5th tic; if the button is still held on the 15th tic the sequence restarts (a second shot 14 tics
    after the first), otherwise the pistol is ready again from the 20th tic."""

    def __init__(self):
        self.start, self.ready = None, 0  # tics relative to now

    def copy(self) -> "Pistol":
        other = Pistol()
        other.start, other.ready = self.start, self.ready
        return other

    def tick(self, held: bool, bullets: int) -> bool:
        """Advance one tic with the shoot button held or not; True if a shot is fired on this tic."""
        shot = False
        if self.start is not None:
            age = -self.start
            if age == FIRE_DELAY:
                shot = True
            if age == REFIRE_CHECK:
                if held and bullets - shot > 0:
                    self.start = 0
                else:
                    self.start, self.ready = None, READY - REFIRE_CHECK
        elif self.ready <= 0 and held and bullets > 0:
            self.start = 0
        if self.start is not None:
            self.start -= 1
        self.ready -= 1
        return shot


class DoomAim:
    """Model-predictive aiming over the options of the request, with the latency hold taken into account."""

    def reset(self, env, hold: int) -> None:
        self.hold_tics = hold * 4  # one env step is 4 tics
        self.v = 0.0  # own lateral speed (map units per tic, + = left), dead-reckoned from the buttons held
        self.prev: list[str] = []  # buttons of the last decision, held during the next idle steps
        self.pistol = Pistol()

    def lateral(self, request) -> float | None:
        """The monster's offset to the left of the line of fire, in map units: distance x sin(bearing), read from the
        request (distance in whole units, bearing in whole degrees)."""
        found = MONSTER.search(request.state["visible"])
        if not found:
            return None
        distance, degrees, side = float(found["distance"]), float(found["degrees"] or 0), found["side"]
        return distance * math.sin(math.radians(degrees if side == "left" else -degrees))

    def rollout(self, schedule: list[tuple[int, list[str]]], bullets: int):
        """Per-tic physics and pistol over (tics, buttons) segments: shots as (tic, own lateral travel), final rest."""
        v, y, tic, pistol, shots = self.v, 0.0, 0, self.pistol.copy(), []
        for tics, buttons in schedule:
            d, held = direction(buttons), "ATTACK" in buttons
            for i in range(tics):
                if i % 4 == 0 and bullets - len(shots) <= 0:
                    held = False  # the harness releases the shoot button once the bullets are spent
                tic += 1
                v += THRUST * d
                y += v
                v *= FRICTION
                if pistol.tick(held, bullets - len(shots)):
                    shots.append((tic, y))
        return shots, y + v / (1 - FRICTION), pistol, v

    def decide(self, env, request, agent=None) -> dict[str, str]:
        offered = request.questions["buttons"].option_ids()
        lat = self.lateral(request)
        left = BULLETS.search(request.state["you"])
        bullets = int(left[1]) if left else 10**6
        best = None
        if lat is not None:
            for seq in itertools.product(offered, repeat=3):
                schedule = [(self.hold_tics, self.prev)]
                for name in seq:
                    schedule += [(4, pressed(name)), (self.hold_tics, pressed(name))]
                shots, rest, _, _ = self.rollout(schedule, bullets)
                hits = [i for i, (_, y) in enumerate(shots) if abs(lat - y) <= AIM]
                if hits:
                    key = (0, hits[0], shots[hits[0]][0])  # fewest bullets wasted before the hit, then the earliest hit
                else:
                    key = (1, len(shots), abs(lat - rest))  # no hit in sight: no shots, end up closest to the line
                if best is None or key < best[0]:
                    best = (key, seq[0])
        name = best[1] if best else "NOOP"
        _, _, self.pistol, self.v = self.rollout([(self.hold_tics, self.prev), (4, pressed(name))], bullets)
        self.prev = pressed(name)
        return {"buttons": name}


# --- highway: shared --------------------------------------------------------------------------

IDM_A, IDM_B, IDM_D0, IDM_T = 3.0, 5.0, 10.0, 1.5  # highway-env's IDM: comfort acceleration and braking, jam distance, headway


def idm(speed: float, target: float, gap: float | None = None, closing: float = 0.0) -> float:
    """highway-env's IDM acceleration (exponent 4) toward a target speed, behind a leader at centre distance gap (m)
    approached at the closing speed (m/s); the target speed is not clipped to the lane's speed limit."""
    a = IDM_A * (1 - (max(speed, 0.0) / max(target, 1e-3)) ** 4)
    if gap is not None:
        wanted = IDM_D0 + speed * IDM_T + speed * closing / (2 * math.sqrt(IDM_A * IDM_B))
        a -= IDM_A * (wanted / max(gap, 1e-3)) ** 2
    return a


# --- highway merge -----------------------------------------------------------------------------

MERGE_LANE_Y = (0.0, 4.0)  # lateral centres of lanes 0 and 1 (merge-v0; the merging lane is at 8 m)
RAMP_ROADS = {("j", "k"), ("k", "b")}


class MergeDrive:
    """IDM car following, a MOBIL-like lane choice, and room for the ramp car, over the vehicles the request lists.

    A vehicle is in lane k when its lateral position is within 2 m of the lane centre. The ego's IDM target is its top
    speed (30 m/s). While the ramp car (on the access ramp or the merging lane) has not merged and the ego is in lane 1
    before the end of the merging lane, the ego moves to lane 0 if that is safe, and otherwise treats the ramp car as its
    lane-1 leader (while it is between 15 m behind and 40 m ahead), so that it merges ahead. Speed: SLOWER if the IDM
    acceleration in the current lane (and in the target lane while changing) is below -1 m/s2, FASTER if above +0.5,
    else IDLE. Lane change for speed: to the other lane if the IDM acceleration there is at least 0.3 m/s2 better, or
    if the ego must brake at its lowest target speed (20 m/s; it cannot go slower). A lane change is safe if the new
    follower would not brake harder than 3 m/s2 (highway-env's IDM, speed limit 20 m/s) and the new leader is at least
    12 m ahead."""

    def reset(self, env, hold: int) -> None:
        pass

    def lanes_of(self, v) -> list[int]:
        if v.lane_index[:2] in RAMP_ROADS:
            return []
        return [k for k, centre in enumerate(MERGE_LANE_Y) if abs(v.position[1] - centre) < 2.0]

    def ramp_car(self, env):
        for v in env.observed():
            if v.lane_index[:2] in RAMP_ROADS or v.lane_index == ("b", "c", 2):
                return v
        return None

    def neighbours(self, env, lane: int, extra=None):
        x = env.ego.position[0]
        ahead = behind = None
        for v in env.observed():
            if lane not in self.lanes_of(v) and v is not extra:
                continue
            dx = v.position[0] - x
            if dx > 0 and (ahead is None or dx < ahead[0]):
                ahead = (dx, v)
            if dx <= 0 and (behind is None or -dx < behind[0]):
                behind = (-dx, v)
        return ahead, behind

    def accel(self, env, lane: int, extra=None) -> float:
        ego = env.ego
        ahead, _ = self.neighbours(env, lane, extra)
        if ahead is None:
            return idm(ego.speed, 30.0)
        gap, v = ahead
        return idm(ego.speed, 30.0, gap, ego.speed - v.speed)

    def safe_change(self, env, lane: int) -> bool:
        ego = env.ego
        ahead, behind = self.neighbours(env, lane)
        if ahead is not None and ahead[0] < 12.0:
            return False
        if behind is not None:
            gap, v = behind
            if gap < 8.0 or idm(v.speed, 20.0, gap, v.speed - ego.speed) < -3.0:
                return False
        return True

    def decide(self, env, request, agent=None) -> dict[str, str]:
        ego = env.ego
        options = request.questions["action"].option_ids()
        here = min(range(2), key=lambda k: abs(ego.position[1] - MERGE_LANE_Y[k]))
        changing = ego.target_lane_index[2] != ego.lane_index[2] or abs(ego.position[1] - MERGE_LANE_Y[here]) > 0.5
        target = min(ego.target_lane_index[2], 1)
        ramp = self.ramp_car(env)
        near = ramp is not None and ego.position[0] < 310 and -15 < ramp.position[0] - ego.position[0] < 40
        yield_to = ramp if near else None
        a = min(self.accel(env, here, yield_to if here == 1 else None),
                self.accel(env, target, yield_to if target == 1 else None))
        if not changing:
            other = 1 - here
            if yield_to is not None and here == 1 and self.safe_change(env, 0):
                return {"action": "LANE_LEFT"}
            gain = self.accel(env, other, yield_to if other == 1 else None) - a
            stuck = a < -1.0 and ego.target_speed <= ego.target_speeds[0]  # already at the lowest target speed
            if (gain > 0.3 or stuck) and self.safe_change(env, other) and not (other == 1 and yield_to is not None):
                return {"action": "LANE_LEFT" if other < here else "LANE_RIGHT"}
        if a < -1.0 and "SLOWER" in options:
            return {"action": "SLOWER"}
        if a > 0.5 and "FASTER" in options:
            return {"action": "FASTER"}
        return {"action": "IDLE"}


# --- highway: prediction ------------------------------------------------------------------

TAU_ACC = 0.6  # highway-env's speed controller time constant (s)


def lane_paths(network, lane_index, n: int = 5) -> list[list]:
    """Every path of n lanes a vehicle may drive from lane_index: at each node, each next road of the network (its
    route is not shown), on the lane highway-env would continue on (same lane number, or the closest lane)."""
    paths = [[lane_index]]
    for _ in range(n - 1):
        longer = []
        for path in paths:
            index = path[-1]
            end = network.get_lane(index).position(network.get_lane(index).length, 0)
            nexts = list(network.graph.get(index[1], {}))
            if not nexts:
                longer.append(path)
            for node in nexts:
                longer.append(path + [network.next_lane(index, route=[(index[1], node, None)], position=end)])
        paths = longer
    return paths


def discs(p, h) -> list:
    """A car (5 m by 2 m) as three discs along its heading."""
    c, s_ = math.cos(h), math.sin(h)
    return [(p[0] + d * c, p[1] + d * s_) for d in (-1.75, 0.0, 1.75)]


def overlap(p_ego, h_ego, p_other, h_other, gap: float = 3.0) -> bool:
    """Two cars too close: a disc of one within gap of a disc of the other (discs of radius 1.3 m touch at 2.6 m; cars
    side by side in lanes 4 m apart never count)."""
    return any(math.hypot(a[0] - b[0], a[1] - b[1]) < gap for a in discs(p_ego, h_ego) for b in discs(p_other, h_other))


class Predictor:
    """Predicted tracks over the next seconds. The ego: a copy of the car stepped alone with its own controllers as it
    follows its route, under one meta-action for the first second and another one after it. Each listed vehicle:
    steered by highway-env's lane controller at a constant speed along every path it may take at the next junctions
    (its route is not shown), a car entering a roundabout also slowing to half its speed; or, with routes=True, along
    its planned route (as highway-env's own drivers predict each other), a slow car that need not give way to the ego
    also starting off to the speed limit, and a car ahead going the same way also braking to a stop. A wreck stays
    where it is."""

    HORIZON, FPS = 4.0, 15  # seconds, simulation frames per second (highway-env's default)

    def __init__(self, env, ego, others, horizon: float | None = None, routes: bool = False, gap: float = 3.0):
        self.net = env.u.road.network
        self.ego = ego
        self.gap = gap
        frames = int((horizon or self.HORIZON) * self.FPS)
        self.times = [(k + 1) / self.FPS for k in range(frames)]
        self.others = []  # predicted tracks per vehicle, over its possible paths and speeds
        for v in others:
            if v.crashed:  # a wreck stays where it is
                self.others.append([(v.position.copy(), v.heading) for _ in self.times[::3]])
                continue
            if routes and getattr(v, "route", None):  # its planned route, as highway-env's drivers predict each other
                limit = getattr(v.lane, "speed_limit", None)
                may_go = limit and v.speed < 0.8 * limit and v.lane.priority >= ego.lane.priority  # it need not yield to us
                speeds = {max(v.speed, 0.0)} | ({float(limit)} if may_go else set())
                if self.ahead(v):  # a car ahead on our way may brake to a stop
                    speeds.add(0.0)
                path = [v.target_lane_index] + [r for r in v.route if r[:2] != v.target_lane_index[:2]]
                for speed in speeds:
                    self.others.append(self.follow(v, path, len(self.times), speed))
                continue
            speeds = {max(v.speed, 0.0)}
            if v.lane_index[0][1:] in ("er", "es"):  # entering a roundabout: may slow down to merge
                speeds.add(0.5 * v.speed)
            for path in lane_paths(self.net, v.lane_index):
                for speed in speeds:
                    self.others.append(self.follow(v, path, len(self.times), speed))

    def ahead(self, v) -> bool:
        """v is in front of the ego (within 30 m, less than 60 degrees off its heading) and drives the same way."""
        dx, dy = v.position[0] - self.ego.position[0], v.position[1] - self.ego.position[1]
        distance = math.hypot(dx, dy)
        if distance > 30.0 or distance < 1e-6:
            return False
        bearing = math.atan2(dy, dx) - self.ego.heading
        same_way = math.cos(v.heading - self.ego.heading) > 0.5
        return math.cos(bearing) > 0.5 and same_way

    def follow(self, v, path: list, frames: int, speed: float) -> list:
        """A vehicle's positions every 3 frames as it tracks the given speed and steers along path with highway-env's
        lane controller (as its own driver steers, so entries and exits are where they really are)."""
        from highway_env.vehicle.controller import ControlledVehicle

        car = ControlledVehicle(v.road, v.position, heading=v.heading, speed=v.speed, target_lane_index=path[0],
                                target_speed=speed, route=list(path))
        track = []
        for k in range(frames):
            car.act()
            car.step(1 / self.FPS)
            if k % 3 == 0:
                track.append((car.position.copy(), car.heading))
        return track

    def ego_track(self, first: str, then: str) -> list:
        """The ego's (position, heading) every 3 frames: meta-action first now, meta-action then after 1 s."""
        road, self.ego.road = self.ego.road, None
        try:
            car = copy.deepcopy(self.ego)
        finally:
            self.ego.road = road
        car.road = road
        track = []
        for k in range(len(self.times)):
            if k == 0 or k == self.FPS:
                car.act(first if k == 0 else then)
            car.act()
            car.step(1 / self.FPS)
            if k % 3 == 0:
                track.append((car.position.copy(), car.heading))
        return track

    def first_conflict(self, first: str, then: str) -> float:
        """Seconds until the ego's track first comes too close to a predicted track (infinity if never)."""
        for k, (p, h) in enumerate(self.ego_track(first, then)):
            if any(overlap(p, h, *other[k], self.gap) for other in self.others):
                return self.times[3 * k]
        return math.inf

    def safe(self, first: str, then: str) -> bool:
        return self.first_conflict(first, then) == math.inf


def speed_after(ego, action: str) -> float:
    """highway-env's MDPVehicle target speed after a speed meta-action."""
    if action == "IDLE":
        return float(ego.target_speed)
    step = 1 if action == "FASTER" else -1
    index = int(min(max(ego.speed_to_index(ego.speed) + step, 0), len(ego.target_speeds) - 1))
    return float(ego.target_speeds[index])


# --- highway roundabout and intersections ---------------------------------------------------

class SpeedOnly:
    """Roundabout: the fastest speed meta-action after which some speed meta-action one second later keeps the ego
    clear of every predicted track for 4 s (see Predictor), else the one that stays clear longest. The route is
    followed by the car itself; no lane changes."""

    SPEEDS = ("FASTER", "IDLE", "SLOWER")
    GAP = 2.6

    def reset(self, env, hold: int) -> None:
        pass

    def decide(self, env, request, agent=None) -> dict[str, str]:
        ego = env.ego
        options = [o for o in request.questions["action"].option_ids() if o in self.SPEEDS]
        predictor = Predictor(env, ego, env.observed(), gap=self.GAP)
        best = None
        for action in options:
            clear = max(predictor.first_conflict(action, then) for then in self.SPEEDS)
            key = (clear, speed_after(ego, action))
            if best is None or key > best[0]:
                best = (key, action)
        return {"action": best[1]}


class Crossing:
    """Gap acceptance at the intersection, as a careful driver who knows where the other vehicles are heading (their
    planned routes, which highway-env's own drivers use to predict each other; the request does not show them). On the
    approach lane: go (FASTER) if driving through at full speed is predicted clear for 6 s (every vehicle at constant
    speed along its route, a slow one possibly starting off to the speed limit; the car by running its own controller);
    otherwise keep coming while the car can still stop before the stop line one decision later, and stop there when it
    cannot. Inside the intersection or beyond: the fastest speed meta-action that stays clear for 4 s (after which some
    meta-action one second later keeps it clear), else the one that stays clear longest."""

    COMMIT_S = 6.0
    SPEEDS = ("FASTER", "IDLE", "SLOWER")
    GAP = 3.0

    def reset(self, env, hold: int) -> None:
        pass

    def others(self, env, ego) -> list:
        return [v for v in env.observed()]

    def decide(self, env, request, agent=None) -> dict[str, str]:
        if agent is not None:
            env.observe_agent(agent)  # the harness's view of this car: env.ego and its own observation
        ego = env.ego
        others = self.others(env, ego)
        options = request.questions["action"].option_ids()
        lane = env.u.road.network.get_lane(ego.lane_index)
        if ego.lane_index[0].startswith("o") and ego.lane_index[1].startswith("ir"):  # still on the approach
            if Predictor(env, ego, others, self.COMMIT_S, routes=True, gap=self.GAP).safe("FASTER", "FASTER"):
                return {"action": "FASTER"}
            to_line = lane.length - lane.local_coordinates(ego.position)[0]
            stopping = ego.speed * (1.0 + TAU_ACC) + 2.0  # one more decision at this speed, then the controller's stop
            if to_line > stopping + 4.5 * (1.0 + TAU_ACC) and "IDLE" in options:
                return {"action": "IDLE" if ego.target_speed <= 4.5 else "SLOWER"}
            return {"action": "SLOWER"}
        predictor = Predictor(env, ego, others, routes=True, gap=self.GAP)
        best = None
        for action in self.SPEEDS:
            clear = max(predictor.first_conflict(action, then) for then in self.SPEEDS)
            key = (clear, speed_after(ego, action))
            if best is None or key > best[0]:
                best = (key, action)
        return {"action": best[1]}


class TwoCars(Crossing):
    """The two-car intersection: each car decides alone with the Crossing rule. A car ignores the other controlled
    car while that car is on a lower-priority road and still on its approach (it has to yield, as the goal text says),
    and treats it as any other vehicle otherwise."""

    def others(self, env, ego) -> list:
        kept = []
        for v in env.observed():
            controlled = any(v is car for car in env.cars.values())
            on_approach = v.lane_index[0].startswith("o") and v.lane_index[1].startswith("ir")
            if controlled and on_approach and v.lane.priority < ego.lane.priority:
                to_line = v.lane.length - v.lane.local_coordinates(v.position)[0]
                if to_line > v.speed * (1.0 + TAU_ACC) + 2.0:  # it can still stop to give way
                    continue
            kept.append(v)
        return kept


class JointPlanner:
    """The two-car intersection, planned jointly and with foresight (an upper bound on what the harness allows, not a
    model of independent drivers): at each decision, for every joint speed meta-action of the cars still driving,
    followed by each of five fixed joint continuations for 3 more seconds, a copy of the highway-env environment (with
    its random state, so later traffic is known) is simulated; the first joint action of the best copy is taken
    (-1000 per crashed car, +100 per arrived car, plus the metres each car drove). The copies apply each action at
    once, as the runner does at 320 ms (no held step at 1 s per step)."""

    CONTINUATIONS = (("FASTER", "FASTER"), ("FASTER", "SLOWER"), ("SLOWER", "FASTER"), ("SLOWER", "SLOWER"), ("IDLE", "IDLE"))
    STEPS = 4

    def reset(self, env, hold: int) -> None:
        self.plan: dict[str, str] = {}

    def rollout_value(self, env, idx: list[int], first: tuple, then: tuple) -> float:
        sim = copy.deepcopy(env.env)
        road = sim.unwrapped.road
        road.__dict__.pop("step", None)  # the harness's per-frame hook belongs to the real episode
        u = sim.unwrapped
        start = [u.controlled_vehicles[i].position.copy() for i in idx]
        for step in range(self.STEPS):
            actions = [env.index["IDLE"]] * len(env.agents)
            for k, i in enumerate(idx):
                actions[i] = env.index[(first if step == 0 else then)[k]]
            sim.step(tuple(actions))
            if any(u.controlled_vehicles[i].crashed for i in idx):
                break
        value = 0.0
        for k, i in enumerate(idx):
            car = u.controlled_vehicles[i]
            value += -1000.0 if car.crashed else 100.0 if u.has_arrived(car) else 0.0
            if not car.crashed:
                value += float(math.hypot(*(car.position - start[k])))
        return value

    def decide(self, env, request, agent=None) -> dict[str, str]:
        agents = env.active_agents()
        if agent == agents[0]:  # plan once per decision, for every car still driving
            idx = [env.agents.index(a) for a in agents]
            best = None
            for first in itertools.product(("FASTER", "IDLE", "SLOWER"), repeat=len(agents)):
                for then in self.CONTINUATIONS:
                    value = self.rollout_value(env, idx, first, then[:len(agents)])
                    if best is None or value > best[0]:
                        best = (value, first)
            self.plan = dict(zip(agents, best[1]))
        return {"action": self.plan[agent]}


# --- main --------------------------------------------------------------------------------------

# --- Freeway ---------------------------------------------------------------------------------------


class FreewayDodge:
    """Search over the next two decisions in a copy of the emulator; a crossing counts 1,000, height 1 per unit."""

    def reset(self, env, hold: int) -> None:
        self.hold = hold

    def rollout(self, env, plan: tuple[str, ...]) -> int:
        actions = env.env._env.unwrapped._action_set  # the ALE actions of the minimal action set, in option order
        held = env.names[env.action]  # pressed until this decision takes effect
        gained = 0
        for name, steps in [(held, self.hold)] + [(name, 1 + self.hold) for name in plan]:
            for _ in range(steps * 4):  # 4 frames per env step
                gained += env.ale.act(actions[env.index[name]])
        return gained * 1000 + int(env.ale.getRAM()[14])  # RAM 14: the chicken's height, where OCAtari reads it

    def decide(self, env, request) -> dict[str, str]:
        start = env.ale.cloneState()
        best, choice = None, "UP"
        for plan in itertools.product(("UP", "NOOP", "DOWN"), repeat=2):  # UP first: ties keep moving on
            env.ale.restoreState(start)
            value = self.rollout(env, plan)
            if best is None or value > best:
                best, choice = value, plan[0]
        env.ale.restoreState(start)
        return {"action": choice}


class AlwaysUp:
    """The trivial Freeway policy that the game mode is chosen against."""

    def reset(self, env, hold: int) -> None:
        pass

    def decide(self, env, request) -> dict[str, str]:
        return {"action": "UP"}


POLICIES = {"realtime/doom/basic": DoomAim, "realtime/highway/merge": MergeDrive, "realtime/highway/roundabout": SpeedOnly,
            "realtime/highway/intersection": Crossing, "realtime/highway/intersection_multi_agent": TwoCars,
            "realtime/atari/freeway": FreewayDodge}
PLANNERS = {"realtime/highway/intersection_multi_agent": JointPlanner}
BASELINES = {"realtime/atari/freeway": AlwaysUp}


def run(job: tuple) -> tuple[int, dict]:
    name, seed, latency_ms, kind = job
    cls = next(c for c in envs_under(name) if c.name == name)
    return seed, play(cls(), seed, TABLES[kind][name](), latency_ms)


TABLES = {"scripted": POLICIES, "planner": PLANNERS, "baseline": BASELINES}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", default="", help="one environment (default: every one with a policy here)")
    parser.add_argument("--seeds", type=int, default=0, help="seeds 0..N-1 (default: the environment's episodes)")
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--latency-ms", type=float, default=320.0)
    parser.add_argument("--planner", action="store_true", help="the foresighted joint planner, where there is one")
    parser.add_argument("--baseline", action="store_true", help="the trivial policy, where there is one")
    parser.add_argument("--workers", type=int, default=1, help="episodes played in parallel processes")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    kind = "planner" if args.planner else "baseline" if args.baseline else "scripted"
    table = TABLES[kind]
    for cls in envs_under(""):
        if cls.name not in table or (args.env and cls.name != args.env):
            continue
        seeds = list(range(args.first_seed, args.first_seed + (args.seeds or cls.episodes)))
        jobs = [(cls.name, seed, args.latency_ms, kind) for seed in seeds]
        if args.workers > 1:
            with ProcessPoolExecutor(args.workers) as pool:
                results = list(pool.map(run, jobs))
        else:
            results = [run(job) for job in jobs]
        if args.verbose:
            for seed, out in results:
                print(f"  {cls.name} seed {seed}: {out}", flush=True)
        if all(out["success"] is None for _, out in results):  # score only (Atari)
            scores = [out["score"] for _, out in results]
            print(f"{cls.name} (version {cls.version}): {kind} mean score {sum(scores) / len(scores):.1f} at "
                  f"{args.latency_ms:.0f} ms; scores {scores}", flush=True)
            continue
        failed = [seed for seed, out in results if not out["success"]]
        print(f"{cls.name} (version {cls.version}): {kind} success {len(results) - len(failed)}/{len(results)} at "
              f"{args.latency_ms:.0f} ms; failed seeds {failed}", flush=True)


if __name__ == "__main__":
    main()
