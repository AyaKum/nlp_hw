# coding: utf-8
"""Simple RoboCup-like 2v2 soccer simulation with BDI opponents and
online Q-learning for Team A.

This version avoids external heavy dependencies so it can run in the
training environment while still respecting the structure from the
lecture slides.  Vector maths is implemented manually instead of using
NumPy, but matplotlib (available in the base environment) is used for
plotting results.
"""

from __future__ import annotations

import math
import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - fallback minimal plotter
    plt = None

# ---------------------------------------------------------------------------
# Physics constants (taken from the lecture description)
# ---------------------------------------------------------------------------
FIELD_WIDTH = 100.0
FIELD_HEIGHT = 65.0
GOAL_Y_RANGE = (FIELD_HEIGHT / 2 - 3.66, FIELD_HEIGHT / 2 + 3.66)
TIME_STEP = 0.05  # 50 ms
NUM_STEPS_PER_EPISODE = int(90 / TIME_STEP)  # 90 second timeout

PLAYER_RADIUS = 0.6
BALL_RADIUS = 0.11

FORCE_SCALE = 1.2
DAMPING = 0.4
KICK_FORCE = 45.0
BALL_FRICTION = 0.2
MAX_PLAYER_SPEED = 8.0

SMALL_ERROR = 0.15
TAU_THRESHOLD = 20
SIGMA_THRESHOLD = 2.5

# ---------------------------------------------------------------------------
# Simple vector helpers (replacement for NumPy functionality)
# ---------------------------------------------------------------------------


def vec(x: float = 0.0, y: float = 0.0) -> List[float]:
    return [float(x), float(y)]


def vec_copy(v: List[float]) -> List[float]:
    return [v[0], v[1]]


def vec_add(a: List[float], b: List[float]) -> List[float]:
    return [a[0] + b[0], a[1] + b[1]]


def vec_add_inplace(v: List[float], w: List[float], scale: float = 1.0) -> None:
    v[0] += w[0] * scale
    v[1] += w[1] * scale


def vec_sub(a: List[float], b: List[float]) -> List[float]:
    return [a[0] - b[0], a[1] - b[1]]


def vec_scale(v: List[float], s: float) -> List[float]:
    return [v[0] * s, v[1] * s]


def vec_norm(v: List[float]) -> float:
    return math.hypot(v[0], v[1])


def vec_normalise(v: List[float]) -> List[float]:
    norm = vec_norm(v)
    if norm < 1e-6:
        return [0.0, 0.0]
    return [v[0] / norm, v[1] / norm]


def clamp_position(pos: List[float]) -> None:
    pos[0] = max(0.0, min(FIELD_WIDTH, pos[0]))
    pos[1] = max(0.0, min(FIELD_HEIGHT, pos[1]))


def limit_speed(vec_value: List[float], max_speed: float) -> None:
    speed = vec_norm(vec_value)
    if speed > max_speed and speed > 1e-6:
        scale = max_speed / speed
        vec_value[0] *= scale
        vec_value[1] *= scale


def distance(a: List[float], b: List[float]) -> float:
    return vec_norm(vec_sub(a, b))


def simple_plot_png(values: List[float], path: str) -> None:
    if not values:
        values = [0.0]
    width = len(values)
    scale_x = max(1, (300 + width - 1) // width)
    width_scaled = width * scale_x
    height = 200
    points = set()
    for i, value in enumerate(values):
        v = max(0.0, min(1.0, float(value)))
        x_base = i * scale_x
        y = height - 1 - int(v * (height - 1))
        for sx in range(scale_x):
            for dy in (-1, 0, 1):
                py = y + dy
                if 0 <= py < height:
                    points.add((x_base + sx, py))

    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width_scaled):
            if (x, y) in points:
                raw.extend(b"\x00\x00\x00")
            elif y == height - 1 or x == 0:
                raw.extend(b"\x88\x88\x88")
            else:
                raw.extend(b"\xff\xff\xff")

    compressed = zlib.compress(bytes(raw))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        fh.write(chunk(b"IHDR", struct.pack(">IIBBBBB", width_scaled, height, 8, 2, 0, 0, 0)))
        fh.write(chunk(b"IDAT", compressed))
        fh.write(chunk(b"IEND", b""))


# ---------------------------------------------------------------------------
# Soccer entities
# ---------------------------------------------------------------------------


@dataclass
class Player:
    name: str
    team: str
    role: str
    position: List[float]
    velocity: List[float] = field(default_factory=lambda: vec())
    acceleration: List[float] = field(default_factory=lambda: vec())
    has_ball: bool = False

    def reset(self, position: Tuple[float, float]):
        self.position = vec(*position)
        self.velocity = vec()
        self.acceleration = vec()
        self.has_ball = False


@dataclass
class Ball:
    position: List[float]
    velocity: List[float] = field(default_factory=lambda: vec())
    acceleration: List[float] = field(default_factory=lambda: vec())

    def reset(self, position: Tuple[float, float]):
        self.position = vec(*position)
        self.velocity = vec()
        self.acceleration = vec()


@dataclass
class Action:
    target_force: List[float]
    kick: bool = False
    kick_direction: Optional[List[float]] = None


# ---------------------------------------------------------------------------
# World model helpers
# ---------------------------------------------------------------------------


def make_empty_world_model() -> Dict[str, Dict[str, object]]:
    return {
        "ball": {"mu": vec(FIELD_WIDTH / 2, FIELD_HEIGHT / 2), "sigma": 5.0, "tau": 1000},
        "opponents": {},
        "teammates": {},
    }


shared_information: Dict[str, Dict[str, object]] = {}


class World:
    def __init__(self):
        self.players: List[Player] = []
        self.ball = Ball(position=vec(FIELD_WIDTH / 2, FIELD_HEIGHT / 2))

    def add_player(self, player: Player) -> None:
        self.players.append(player)

    def reset(self) -> None:
        self.ball.reset((FIELD_WIDTH / 2, FIELD_HEIGHT / 2))
        for p in self.players:
            if p.team == "A":
                if p.role == "striker":
                    p.reset((30.0, FIELD_HEIGHT / 2 + 5))
                else:
                    p.reset((20.0, FIELD_HEIGHT / 2 - 5))
            else:
                if p.role == "striker":
                    p.reset((70.0, FIELD_HEIGHT / 2 + 5))
                else:
                    p.reset((80.0, FIELD_HEIGHT / 2 - 5))
        for info in shared_information.values():
            info["ball"] = None
            info.setdefault("teammates", {}).clear()

    # ---------------------------------------------------------------
    # Physics update
    # ---------------------------------------------------------------
    def step(self, actions: Dict[str, Action]) -> None:
        for p in self.players:
            action = actions.get(p.name, Action(vec()))
            desired_force = action.target_force
            p.acceleration = vec_add(vec_scale(desired_force, FORCE_SCALE), vec_scale(p.velocity, -DAMPING))

        for p in self.players:
            vec_add_inplace(p.velocity, p.acceleration, TIME_STEP)
            limit_speed(p.velocity, MAX_PLAYER_SPEED)
            vec_add_inplace(p.position, p.velocity, TIME_STEP)
            clamp_position(p.position)

        for i in range(len(self.players)):
            for j in range(i + 1, len(self.players)):
                a = self.players[i]
                b = self.players[j]
                diff = vec_sub(a.position, b.position)
                dist = vec_norm(diff)
                min_dist = 2 * PLAYER_RADIUS
                if dist < min_dist and dist > 1e-6:
                    overlap = min_dist - dist
                    direction = vec_scale(diff, 1.0 / dist)
                    vec_add_inplace(a.position, direction, overlap / 2)
                    vec_add_inplace(b.position, direction, -overlap / 2)
                    a.velocity = vec_scale(a.velocity, -0.1)
                    b.velocity = vec_scale(b.velocity, -0.1)

        ball_action = None
        for p in self.players:
            if distance(p.position, self.ball.position) <= PLAYER_RADIUS + BALL_RADIUS + 0.05:
                act = actions.get(p.name)
                if act and act.kick and act.kick_direction is not None:
                    ball_action = act
                    break

        if ball_action is not None:
            self.ball.velocity = vec()
            self.ball.acceleration = vec_scale(vec_normalise(ball_action.kick_direction), KICK_FORCE)
        else:
            self.ball.acceleration = vec_scale(self.ball.velocity, -BALL_FRICTION)

        vec_add_inplace(self.ball.velocity, self.ball.acceleration, TIME_STEP)
        vec_add_inplace(self.ball.position, self.ball.velocity, TIME_STEP)
        clamp_position(self.ball.position)

    def goal_scored(self) -> Optional[str]:
        if self.ball.position[0] >= FIELD_WIDTH - BALL_RADIUS:
            if GOAL_Y_RANGE[0] <= self.ball.position[1] <= GOAL_Y_RANGE[1]:
                return "A"
        if self.ball.position[0] <= BALL_RADIUS:
            if GOAL_Y_RANGE[0] <= self.ball.position[1] <= GOAL_Y_RANGE[1]:
                return "B"
        return None


# ---------------------------------------------------------------------------
# World model update functions (faithful to lecture pseudo-code)
# ---------------------------------------------------------------------------


def UPDATEWORLD(agent: Player, wm: Dict[str, Dict[str, object]], world_state: World) -> None:
    UPDATEVISION(agent, wm, world_state)
    UPDATESHAREDINFORMATION(agent, wm, world_state)
    UPDATETIME(wm)


def UPDATEVISION(agent: Player, wm: Dict[str, Dict[str, object]], world_state: World) -> None:
    ball = world_state.ball
    if distance(agent.position, ball.position) < 40.0:
        obs = vec_add(ball.position, [random.gauss(0, 0.5), random.gauss(0, 0.5)])
        wm_ball = wm["ball"]
        wm_ball["mu"] = vec_add(vec_scale(wm_ball["mu"], 0.8), vec_scale(obs, 0.2))
        wm_ball["sigma"] = max(0.5, wm_ball["sigma"] * 0.8)
        wm_ball["tau"] = 0

    opponents = [p for p in world_state.players if p.team != agent.team]
    for op in opponents:
        if distance(agent.position, op.position) < 35.0:
            obs = vec_add(op.position, [random.gauss(0, 0.7), random.gauss(0, 0.7)])
            prev = wm["opponents"].get(op.name, {"mu": obs, "sigma": 3.0, "tau": 0})
            new_mu = vec_add(vec_scale(prev["mu"], 0.6), vec_scale(obs, 0.4))
            wm["opponents"][op.name] = {"mu": new_mu, "sigma": max(0.5, prev["sigma"] * 0.85), "tau": 0}


def UPDATESHAREDINFORMATION(agent: Player, wm: Dict[str, Dict[str, object]], world_state: World) -> None:
    team_store = shared_information.setdefault(agent.team, {"ball": None, "teammates": {}})
    team_store["teammates"][agent.name] = vec_copy(agent.position)
    wm["teammates"] = {name: vec_copy(pos) for name, pos in team_store["teammates"].items() if name != agent.name}

    wm_ball = wm["ball"]
    if wm_ball["tau"] > TAU_THRESHOLD:
        best = GETBALLLOCATION(agent, team_store)
        if best is not None:
            wm_ball["mu"] = vec_copy(best["mu"])
            wm_ball["sigma"] = best["sigma"]
            wm_ball["tau"] = best["tau"]

    if ISVALID(wm_ball):
        existing = team_store.get("ball")
        if existing is None or existing["tau"] > wm_ball["tau"]:
            team_store["ball"] = {"mu": vec_copy(wm_ball["mu"]), "sigma": wm_ball["sigma"], "tau": wm_ball["tau"]}


def UPDATETIME(wm: Dict[str, Dict[str, object]]) -> None:
    wm_ball = wm["ball"]
    wm_ball["tau"] += 1
    wm_ball["sigma"] += SMALL_ERROR
    for info in wm["opponents"].values():
        info["tau"] = info.get("tau", 0) + 1
        info["sigma"] = info.get("sigma", 1.0) + SMALL_ERROR


def GETBALLLOCATION(agent: Player, shared_store: Dict[str, Dict[str, object]]) -> Optional[Dict[str, object]]:
    best = None
    best_score = float("inf")
    candidate = shared_store.get("ball")
    if candidate is not None and ISVALID(candidate):
        score = candidate["tau"] + candidate["sigma"]
        if score < best_score:
            best = {"mu": vec_copy(candidate["mu"]), "sigma": candidate["sigma"], "tau": candidate["tau"]}
    return best


def ISVALID(ball_info: Optional[Dict[str, object]]) -> bool:
    if ball_info is None:
        return False
    if ball_info.get("sigma", SIGMA_THRESHOLD + 1) > SIGMA_THRESHOLD:
        return False
    if ball_info.get("tau", TAU_THRESHOLD + 1) > TAU_THRESHOLD:
        return False
    return True


# ---------------------------------------------------------------------------
# Team B policy (BDI-like)
# ---------------------------------------------------------------------------


def team_b_policy(player: Player, world: World) -> Action:
    ball = world.ball
    teammates = [p for p in world.players if p.team == player.team and p.name != player.name]
    opponent_goal = vec(0.0, FIELD_HEIGHT / 2)
    ball_dist = distance(player.position, ball.position)

    desire = "MOVE"
    kick = False
    kick_dir = None
    force = vec()

    if ball_dist < 2.0:
        if ball.position[0] < 30.0:
            desire = "SHOOT"
            goal_vec = vec_sub(vec(0.0, FIELD_HEIGHT / 2), ball.position)
            kick_dir = vec_normalise(goal_vec)
            kick = True
        else:
            desire = "PASS"
            mate = min(teammates, key=lambda t: distance(t.position, opponent_goal))
            pass_vec = vec_sub(mate.position, ball.position)
            kick_dir = vec_normalise(pass_vec)
            kick = True
    else:
        desire = "MOVE"

    if desire == "MOVE":
        if player.role == "striker":
            target = ball.position
        else:
            target = vec(FIELD_WIDTH * 0.75, FIELD_HEIGHT / 2)
        direction = vec_sub(target, player.position)
        force = vec_normalise(direction)
    else:
        force = vec()

    return Action(force, kick=kick, kick_direction=kick_dir)


# ---------------------------------------------------------------------------
# Team A: Q-learning agents
# ---------------------------------------------------------------------------

ActionOptions = ["TO_BALL", "TO_GOAL", "DEFEND", "WAIT"]


class QLearningAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.q_table: Dict[Tuple[int, ...], List[float]] = {}
        self.epsilon = 1.0
        self.alpha = 0.2
        self.gamma = 0.9

    def discretize(self, player: Player, world: World, holder: Optional[str]) -> Tuple[int, ...]:
        px = min(4, int(player.position[0] / (FIELD_WIDTH / 5)))
        py = min(2, int(player.position[1] / (FIELD_HEIGHT / 3)))
        bx = min(4, int(world.ball.position[0] / (FIELD_WIDTH / 5)))
        by = min(2, int(world.ball.position[1] / (FIELD_HEIGHT / 3)))
        holder_index = {
            None: 0,
            "A_striker": 1,
            "A_defender": 2,
            "B_striker": 3,
            "B_defender": 4,
        }.get(holder, 0)
        return (px, py, bx, by, holder_index)

    def ensure_state(self, state: Tuple[int, ...]) -> None:
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in ActionOptions]

    def select_action(self, state: Tuple[int, ...]) -> int:
        self.ensure_state(state)
        if random.random() < self.epsilon:
            return random.randrange(len(ActionOptions))
        values = self.q_table[state]
        return max(range(len(values)), key=lambda i: values[i])

    def update(self, state: Tuple[int, ...], action: int, reward: float, next_state: Tuple[int, ...], done: bool) -> None:
        self.ensure_state(state)
        self.ensure_state(next_state)
        target = reward
        if not done:
            target += self.gamma * max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])

    def decay_epsilon(self) -> None:
        self.epsilon = max(0.05, self.epsilon * 0.995)


# Convert RL decision to actual command

def rl_action_to_command(agent: Player, action_idx: int, world: World) -> Action:
    action = ActionOptions[action_idx]
    ball = world.ball
    if action == "TO_BALL":
        direction = vec_sub(ball.position, agent.position)
        force = vec_normalise(direction)
        kick = distance(agent.position, ball.position) < 1.2
        kick_dir = vec_normalise(vec_sub(vec(FIELD_WIDTH, FIELD_HEIGHT / 2), ball.position))
        return Action(force, kick=kick, kick_direction=kick_dir if kick else None)
    if action == "TO_GOAL":
        target = vec(FIELD_WIDTH, FIELD_HEIGHT / 2)
        direction = vec_sub(target, agent.position)
        force = vec_normalise(direction)
        kick = distance(agent.position, ball.position) < 1.2
        kick_dir = vec_normalise(vec_sub(target, ball.position))
        return Action(force, kick=kick, kick_direction=kick_dir if kick else None)
    if action == "DEFEND":
        if agent.role == "striker":
            target = vec(FIELD_WIDTH * 0.4, FIELD_HEIGHT / 2)
        else:
            target = vec(FIELD_WIDTH * 0.2, FIELD_HEIGHT / 2)
        direction = vec_sub(target, agent.position)
        force = vec_normalise(direction)
        kick = False
        if distance(agent.position, ball.position) < 1.0:
            kick = True
            kick_dir = vec_normalise(vec_sub(vec(FIELD_WIDTH, FIELD_HEIGHT / 2), ball.position))
            return Action(force, kick=True, kick_direction=kick_dir)
        return Action(force, kick=False, kick_direction=None)
    # WAIT
    kick = False
    kick_dir = None
    if distance(agent.position, ball.position) < 1.0:
        kick = True
        kick_dir = vec_normalise(vec_sub(vec(FIELD_WIDTH, FIELD_HEIGHT / 2), ball.position))
    return Action(vec(), kick=kick, kick_direction=kick_dir)


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------


def determine_ball_holder(world: World) -> Optional[str]:
    for p in world.players:
        if distance(p.position, world.ball.position) < 1.0:
            return p.name
    return None


def run_episode(world: World, rl_agents: Dict[str, QLearningAgent], train: bool = True) -> Tuple[float, str]:
    world.reset()
    world_models = {p.name: make_empty_world_model() for p in world.players if p.team == "A"}

    total_reward = 0.0
    result = "timeout"

    for step in range(NUM_STEPS_PER_EPISODE):
        for p in world.players:
            if p.team == "A":
                UPDATEWORLD(p, world_models[p.name], world)

        ball_holder = determine_ball_holder(world)
        actions: Dict[str, Action] = {}
        states: Dict[str, Tuple[int, ...]] = {}
        chosen_actions: Dict[str, int] = {}

        for p in world.players:
            if p.team == "A":
                agent = rl_agents[p.name]
                state = agent.discretize(p, world, ball_holder)
                action_idx = agent.select_action(state)
                actions[p.name] = rl_action_to_command(p, action_idx, world)
                states[p.name] = state
                chosen_actions[p.name] = action_idx
            else:
                actions[p.name] = team_b_policy(p, world)

        world.step(actions)
        ball_holder_next = determine_ball_holder(world)

        reward = -0.01
        scorer = world.goal_scored()
        done = scorer is not None or step == NUM_STEPS_PER_EPISODE - 1
        if scorer == "A":
            reward = 1.0
            result = "A"
        elif scorer == "B":
            reward = -1.0
            result = "B"
        elif done:
            result = "timeout"

        for p in world.players:
            if p.team == "A":
                agent = rl_agents[p.name]
                next_state = agent.discretize(p, world, ball_holder_next)
                if train:
                    agent.update(states[p.name], chosen_actions[p.name], reward, next_state, done)
        total_reward += reward

        if done:
            break

    if train:
        for agent in rl_agents.values():
            agent.decay_epsilon()

    return total_reward, result


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------


def main():
    random.seed(0)

    world = World()
    players = [
        Player("A_striker", "A", "striker", vec(30.0, FIELD_HEIGHT / 2 + 5)),
        Player("A_defender", "A", "defender", vec(20.0, FIELD_HEIGHT / 2 - 5)),
        Player("B_striker", "B", "striker", vec(70.0, FIELD_HEIGHT / 2 + 5)),
        Player("B_defender", "B", "defender", vec(80.0, FIELD_HEIGHT / 2 - 5)),
    ]
    for p in players:
        world.add_player(p)

    rl_agents = {
        "A_striker": QLearningAgent("A_striker", "striker"),
        "A_defender": QLearningAgent("A_defender", "defender"),
    }

    episodes = 300
    results: List[str] = []

    for _ in range(episodes):
        _, result = run_episode(world, rl_agents, train=True)
        results.append(result)

    window = 50
    win_rate = []
    for i in range(len(results)):
        start = max(0, i - window + 1)
        segment = results[start : i + 1]
        wins = sum(1 for r in segment if r == "A")
        win_rate.append(wins / len(segment))

    if plt is not None:
        plt.figure(figsize=(8, 4))
        plt.plot(win_rate)
        plt.title("Team A Win Rate (rolling window of 50 episodes)")
        plt.xlabel("Episode")
        plt.ylabel("Win Rate")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("win_rate.png")
    else:
        simple_plot_png(win_rate, "win_rate.png")

    early = results[: episodes // 3]
    late = results[-(episodes // 3) :]
    early_rate = sum(1 for r in early if r == "A") / max(1, len(early))
    late_rate = sum(1 for r in late if r == "A") / max(1, len(late))

    print("Early episodes win rate (Team A): {:.2f}".format(early_rate))
    print("Late episodes win rate (Team A): {:.2f}".format(late_rate))
    print("Plot saved to win_rate.png")


if __name__ == "__main__":
    main()
