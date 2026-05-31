"""
maze_dqn_v2_0_5_normalized_margin_preference_rm.py

v2.0.5: Normalized Margin Preference Reward Model -> QCNN

This file is intentionally written as a new program structure instead of a patch
on v2.0.3. It has two separated stages:

1) Offline reward-model learning:
   - Build trajectories from BFS, noisy-BFS, random, and safe-loop policies.
   - Rank trajectories lexicographically:
       success > timeout > wall
     If the primary outcome is the same:
       success: smaller BFS gap is better
       timeout: smaller final BFS distance to goal is better
       wall: smaller BFS distance before/at wall position is better
   - Convert ordered trajectories into preference pairs.
   - Train R_phi(S, a, S') with the Bradley-Terry-Luce loss:
       L_BTL = -log sigmoid(R_phi(tau+) - R_phi(tau-))

2) QCNN training:
   - Freeze R_phi.
   - Use r_t = rm_scale * R_phi(S_t, a_t, S_{t+1}) as the DQN reward.
   - Optional terminal anchors exist for debugging, but default is off.

No demo replay, no BFS margin, no action mask, no progress reward.
BFS is used only to create trajectory preferences and evaluation metrics.
"""

import argparse
import csv
import json
import math
import random
from collections import deque, namedtuple
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Small-grid experiments are often faster and more stable with one CPU thread.
# CUDA is unaffected.
torch.set_num_threads(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VERSION = "v2.0.5_normalized_margin_preference_rm"
SIZE = 8
MAX_STEPS = 64
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]

TEST_MAZE_TEXT = """
S.......
.######.
.....#..
##.#...#
.....#..
.######.
.#...#..
...#...G
"""

DQNTransition = namedtuple("DQNTransition", ["state", "action", "reward", "next_state", "done"])


@dataclass
class RMTransition:
    state: np.ndarray
    action: int
    next_state: np.ndarray


@dataclass
class Trajectory:
    transitions: List[RMTransition]
    outcome: str  # "success" or "timeout"; wall is an event, not outcome
    steps: int
    bfs_len: int
    start_bfs_dist: int
    final_bfs_dist: int
    gap: int
    wall_hits: int
    repeat_count: int
    # Lexicographic preference key:
    # success/timeout > no-wall/has-wall > path quality > wall count > repeat count
    quality_key: Tuple[int, int, int, int, int]
    badness: float
    source: str


@dataclass
class PreferencePair:
    pos: Trajectory
    neg: Trajectory
    delta_badness: float = 0.0
    margin: float = 0.0
    pair_source: str = "ranking"


@dataclass
class DQNConfig:
    episodes: int = 3000
    max_steps: int = 64
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 128
    replay_size: int = 100_000
    warmup_steps: int = 3000
    updates_per_episode: int = 64
    target_update_interval: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 2500
    eval_n: int = 100
    rm_scale: float = 2.0
    anchor_terminal: bool = False
    anchor_goal: float = 64.0
    anchor_wall: float = -32.0


# -----------------------------
# General utilities
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def moving_average(values: Sequence[float], window: int = 20) -> List[float]:
    out: List[float] = []
    buf: deque = deque(maxlen=window)
    for v in values:
        buf.append(float(v))
        out.append(float(np.mean(buf)))
    return out


def weighted_choice(weight_map: Dict[str, float]) -> str:
    keys = list(weight_map.keys())
    weights = np.array([weight_map[k] for k in keys], dtype=np.float64)
    weights = weights / weights.sum()
    return str(np.random.choice(keys, p=weights))


def parse_maze_text(text: str) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) != SIZE:
        raise ValueError("test maze has invalid row count")
    grid = np.zeros((SIZE, SIZE), dtype=np.int8)
    start = goal = None
    for r, line in enumerate(lines):
        if len(line) != SIZE:
            raise ValueError("test maze has invalid column count")
        for c, ch in enumerate(line):
            if ch == "#":
                grid[r, c] = 1
            elif ch == "S":
                start = (r, c)
            elif ch == "G":
                goal = (r, c)
    if start is None or goal is None:
        raise ValueError("test maze needs S and G")
    return grid, start, goal


def in_bounds(pos: Tuple[int, int]) -> bool:
    r, c = pos
    return 0 <= r < SIZE and 0 <= c < SIZE


def is_free(grid: np.ndarray, pos: Tuple[int, int]) -> bool:
    r, c = pos
    return in_bounds(pos) and int(grid[r, c]) == 0


# -----------------------------
# BFS and maze generation
# -----------------------------

def bfs_distances(grid: np.ndarray, goal: Tuple[int, int]) -> np.ndarray:
    dist = np.full((SIZE, SIZE), fill_value=10_000, dtype=np.int32)
    if not is_free(grid, goal):
        return dist
    q = deque([goal])
    dist[goal] = 0
    while q:
        r, c = q.popleft()
        for dr, dc in ACTIONS:
            nr, nc = r + dr, c + dc
            if is_free(grid, (nr, nc)) and dist[nr, nc] == 10_000:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def bfs_path(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    dist = bfs_distances(grid, goal)
    if dist[start] >= 10_000:
        return None
    path = [start]
    cur = start
    while cur != goal:
        best = None
        best_d = dist[cur]
        for dr, dc in ACTIONS:
            nxt = (cur[0] + dr, cur[1] + dc)
            if is_free(grid, nxt) and dist[nxt] < best_d:
                best = nxt
                best_d = int(dist[nxt])
        if best is None:
            return None
        cur = best
        path.append(cur)
    return path


def action_between(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    delta = (b[0] - a[0], b[1] - a[1])
    return ACTIONS.index(delta)


def random_free_cell(grid: np.ndarray) -> Tuple[int, int]:
    cells = [(r, c) for r in range(SIZE) for c in range(SIZE) if grid[r, c] == 0]
    return random.choice(cells)


def generate_maze(difficulty: str) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    # Rejection sampling keeps this simple and deterministic enough for experiments.
    specs = {
        "easy": (0.14, 6, 22),
        "medium": (0.23, 12, 36),
        "hard": (0.31, 18, 64),
    }
    wall_prob, min_len, max_len = specs[difficulty]
    start = (0, 0)
    goal = (SIZE - 1, SIZE - 1)
    for _ in range(10_000):
        grid = (np.random.rand(SIZE, SIZE) < wall_prob).astype(np.int8)
        grid[start] = 0
        grid[goal] = 0
        path = bfs_path(grid, start, goal)
        if path is None:
            continue
        length = len(path) - 1
        if min_len <= length <= max_len:
            return grid, start, goal
    # Fallback: carve a random monotone-ish path, then add walls.
    grid = np.zeros((SIZE, SIZE), dtype=np.int8)
    pos = start
    protected = {start, goal}
    while pos != goal:
        choices = []
        if pos[0] < goal[0]:
            choices.append((pos[0] + 1, pos[1]))
        if pos[1] < goal[1]:
            choices.append((pos[0], pos[1] + 1))
        pos = random.choice(choices)
        protected.add(pos)
    for r in range(SIZE):
        for c in range(SIZE):
            if (r, c) not in protected and random.random() < wall_prob:
                grid[r, c] = 1
    return grid, start, goal


def sample_difficulty(easy: float, medium: float, hard: float) -> str:
    return weighted_choice({"easy": easy, "medium": medium, "hard": hard})


# -----------------------------
# Environment with visited state
# -----------------------------

class MazeEnv:
    def __init__(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], max_steps: int = MAX_STEPS):
        self.grid = grid.copy().astype(np.int8)
        self.start = start
        self.goal = goal
        self.max_steps = max_steps
        self.dist = bfs_distances(self.grid, self.goal)
        self.reset()

    def reset(self) -> np.ndarray:
        self.pos = self.start
        self.steps = 0
        self.done = False
        self.visited = np.zeros((SIZE, SIZE), dtype=np.float32)
        self.visited[self.pos] = 1.0
        return self.state()

    def state(self) -> np.ndarray:
        wall = self.grid.astype(np.float32)
        agent = np.zeros((SIZE, SIZE), dtype=np.float32)
        goal = np.zeros((SIZE, SIZE), dtype=np.float32)
        agent[self.pos] = 1.0
        goal[self.goal] = 1.0
        return np.stack([wall, agent, goal, self.visited.astype(np.float32)], axis=0)

    def step_raw(self, action: int) -> Tuple[np.ndarray, Dict[str, object]]:
        if self.done:
            raise RuntimeError("step after done")
        before = self.pos
        dr, dc = ACTIONS[action]
        nxt = (before[0] + dr, before[1] + dc)
        hit_wall = (not in_bounds(nxt)) or (not is_free(self.grid, nxt))
        success = False
        timeout = False

        # v2.0.4 fix: wall hit is an event, not a terminal state.
        # The agent stays in place, consumes one step, and can continue.
        if not hit_wall:
            self.pos = nxt
            self.visited[self.pos] = 1.0
            success = self.pos == self.goal

        self.steps += 1
        if success:
            self.done = True
            outcome = "success"
        elif self.steps >= self.max_steps:
            self.done = True
            timeout = True
            outcome = "timeout"
        else:
            outcome = "running"

        next_state = self.state()
        info = {
            "before": before,
            "pos": self.pos,
            "hit_wall": hit_wall,
            "success": success,
            "timeout": timeout,
            "outcome": outcome,
            "steps": self.steps,
            "bfs_dist": int(self.dist[self.pos]) if self.dist[self.pos] < 10_000 else 10_000,
            "repeat_count": int(self.visited.sum()) if False else 0,
        }
        return next_state, info


def transition_tensor(state: np.ndarray, action: int, next_state: np.ndarray) -> np.ndarray:
    planes = np.zeros((4, SIZE, SIZE), dtype=np.float32)
    planes[action, :, :] = 1.0
    return np.concatenate([state.astype(np.float32), next_state.astype(np.float32), planes], axis=0)


# -----------------------------
# Trajectory generation
# -----------------------------


def normalized_log_count(x: int, max_steps: int = MAX_STEPS) -> float:
    return float(math.log1p(max(0, x)) / max(1e-6, math.log1p(max_steps)))


def trajectory_badness(traj: Trajectory, args: argparse.Namespace) -> float:
    has_wall = 1.0 if traj.wall_hits > 0 else 0.0
    wall_count = normalized_log_count(traj.wall_hits, args.max_steps)
    repeat_count = normalized_log_count(traj.repeat_count, args.max_steps)
    if traj.outcome == "success":
        denom = max(1.0, float(args.max_steps - traj.bfs_len))
        path_bad = max(0.0, float(traj.gap) / denom)
        return float(
            path_bad
            + args.badness_wall_presence_weight * has_wall
            + args.badness_wall_count_weight * wall_count
            + args.badness_repeat_weight * repeat_count
        )
    # timeout is strictly worse than any success by lexicographic ordering.
    # This badness only controls margin magnitude after the pair direction has been fixed.
    dist_bad = float(traj.final_bfs_dist) / max(1.0, float(traj.start_bfs_dist))
    return float(
        args.badness_timeout_base
        + dist_bad
        + args.badness_wall_presence_weight * has_wall
        + args.badness_wall_count_weight * wall_count
        + args.badness_repeat_weight * repeat_count
    )


def finalize_trajectory(
    transitions: List[RMTransition],
    outcome: str,
    bfs_len: int,
    start_bfs_dist: int,
    final_bfs_dist: int,
    wall_hits: int,
    repeat_count: int,
    source: str,
    args: argparse.Namespace,
) -> Trajectory:
    steps = len(transitions)
    if outcome == "success":
        gap = max(0, steps - bfs_len)
        # Any success is strictly better than timeout.
        # Within the same outcome, no-wall is the second lexicographic key:
        # no-wall >> has-wall, then path gap, wall count, repeat count.
        key = (2, -(1 if wall_hits > 0 else 0), -gap, -wall_hits, -repeat_count)
        final_bfs_dist = 0
    else:
        gap = args.max_steps
        # Wall is now an event, not an outcome.
        # Within timeout, no-wall is the second lexicographic key:
        # no-wall >> has-wall, then final BFS distance, wall count, repeat count.
        key = (1, -(1 if wall_hits > 0 else 0), -final_bfs_dist, -wall_hits, -repeat_count)
    dummy = Trajectory(
        transitions=transitions,
        outcome=outcome,
        steps=steps,
        bfs_len=bfs_len,
        start_bfs_dist=start_bfs_dist,
        final_bfs_dist=final_bfs_dist,
        gap=gap,
        wall_hits=wall_hits,
        repeat_count=repeat_count,
        quality_key=key,
        badness=0.0,
        source=source,
    )
    dummy.badness = trajectory_badness(dummy, args)
    return dummy


def best_bfs_action(env: MazeEnv) -> Optional[int]:
    cur_d = int(env.dist[env.pos])
    best_actions: List[int] = []
    for a, (dr, dc) in enumerate(ACTIONS):
        nxt = (env.pos[0] + dr, env.pos[1] + dc)
        if is_free(env.grid, nxt) and int(env.dist[nxt]) < cur_d:
            best_actions.append(a)
    return random.choice(best_actions) if best_actions else None


def wall_actions_at(env: MazeEnv) -> List[int]:
    out: List[int] = []
    for a, (dr, dc) in enumerate(ACTIONS):
        nxt = (env.pos[0] + dr, env.pos[1] + dc)
        if (not in_bounds(nxt)) or (not is_free(env.grid, nxt)):
            out.append(a)
    return out


def record_step(env: MazeEnv, state: np.ndarray, action: int) -> Tuple[np.ndarray, RMTransition, Dict[str, object], bool]:
    old_visited = env.visited.copy()
    ns, info = env.step_raw(action)
    repeat = (not info["hit_wall"]) and old_visited[env.pos] > 0.5
    return ns, RMTransition(state.copy(), action, ns.copy()), info, bool(repeat)


def make_trajectory(env: MazeEnv, policy: str, args: argparse.Namespace, noise_prob: float = 0.2) -> Trajectory:
    state = env.reset()
    path = bfs_path(env.grid, env.start, env.goal)
    if path is None:
        raise ValueError("trajectory generated on unsolvable maze")
    bfs_len = len(path) - 1
    start_bfs_dist = int(env.dist[env.start]) if env.dist[env.start] < 10_000 else bfs_len
    transitions: List[RMTransition] = []
    outcome = "timeout"
    repeat_visits = 0
    wall_hits = 0

    sticky_target_action: Optional[int] = None
    sticky_started = False
    sticky_hits_target = random.randint(args.sticky_wall_min_hits, args.sticky_wall_max_hits)
    sticky_prefix_steps = max(1, int(args.sticky_wall_prefix_ratio * max(1, bfs_len)))
    sticky_hits_done = 0

    for t in range(args.max_steps):
        if policy == "bfs":
            action = best_bfs_action(env)
            if action is None:
                action = random.randrange(4)
        elif policy == "noisy_bfs":
            if random.random() < noise_prob:
                action = random.randrange(4)
            else:
                action = best_bfs_action(env)
                if action is None:
                    action = random.randrange(4)
        elif policy == "random":
            action = random.randrange(4)
        elif policy == "safe_loop":
            legal: List[int] = []
            visited_legal: List[int] = []
            non_goal_legal: List[int] = []
            for a, (dr, dc) in enumerate(ACTIONS):
                nxt = (env.pos[0] + dr, env.pos[1] + dc)
                if is_free(env.grid, nxt):
                    legal.append(a)
                    if env.visited[nxt] > 0.5:
                        visited_legal.append(a)
                    if nxt != env.goal:
                        non_goal_legal.append(a)
            if visited_legal:
                action = random.choice(visited_legal)
            elif non_goal_legal:
                action = random.choice(non_goal_legal)
            elif legal:
                action = random.choice(legal)
            else:
                action = random.randrange(4)
        elif policy == "sticky_wall":
            # Follow BFS/noisy-BFS for a prefix, then repeatedly hit the same wall.
            if (not sticky_started) and t < sticky_prefix_steps:
                action = best_bfs_action(env)
                if action is None or random.random() < 0.10:
                    action = random.randrange(4)
            else:
                if not sticky_started:
                    candidates = wall_actions_at(env)
                    if not candidates:
                        # Move around until a wall action becomes available.
                        action = best_bfs_action(env)
                        if action is None:
                            action = random.randrange(4)
                    else:
                        sticky_target_action = random.choice(candidates)
                        sticky_started = True
                        action = sticky_target_action
                else:
                    action = sticky_target_action if sticky_target_action is not None else random.randrange(4)
        else:
            raise ValueError(f"unknown policy {policy}")

        s = state.copy()
        ns, tr, info, repeated = record_step(env, s, action)
        if info["hit_wall"]:
            wall_hits += 1
            if policy == "sticky_wall" and sticky_started:
                sticky_hits_done += 1
        if repeated:
            repeat_visits += 1
        transitions.append(tr)
        state = ns
        if info["outcome"] != "running":
            outcome = str(info["outcome"])
            break
        # For sticky-wall, after enough wall hits, keep the behavior until timeout to expose the exploit.
        if policy == "sticky_wall" and sticky_started and sticky_hits_done >= sticky_hits_target:
            pass
    else:
        outcome = "timeout"

    final_dist = int(env.dist[env.pos]) if int(env.dist[env.pos]) < 10_000 else 10_000
    return finalize_trajectory(
        transitions=transitions,
        outcome=outcome,
        bfs_len=bfs_len,
        start_bfs_dist=start_bfs_dist,
        final_bfs_dist=final_dist,
        wall_hits=wall_hits,
        repeat_count=repeat_visits,
        source=policy,
        args=args,
    )


def make_cpl_pair(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], args: argparse.Namespace) -> Optional[PreferencePair]:
    # Construct same-prefix + continue-vs-sticky-wall pair. This is not a single-step label:
    # both branches are full trajectories, but the difference is localized.
    path = bfs_path(grid, start, goal)
    if path is None or len(path) < 4:
        return None
    prefix_len = max(1, min(len(path) - 2, int(args.sticky_wall_prefix_ratio * (len(path) - 1))))

    def run_prefix(env: MazeEnv) -> np.ndarray:
        state = env.reset()
        for i in range(prefix_len):
            if env.pos == goal:
                break
            action = action_between(env.pos, path[i + 1]) if i + 1 < len(path) else (best_bfs_action(env) or 0)
            state, _tr, _info, _rep = record_step(env, state, action)
            if _info["outcome"] != "running":
                break
        return state

    # Positive branch: continue BFS after the same prefix.
    env_pos = MazeEnv(grid, start, goal, max_steps=args.max_steps)
    state = run_prefix(env_pos)
    pos_trans: List[RMTransition] = []
    pos_wall = 0
    pos_rep = 0
    outcome = "timeout"
    while len(pos_trans) + env_pos.steps < args.max_steps and env_pos.pos != goal:
        a = best_bfs_action(env_pos)
        if a is None:
            a = random.randrange(4)
        ns, tr, info, rep = record_step(env_pos, state, a)
        pos_trans.append(tr)
        if info["hit_wall"]:
            pos_wall += 1
        if rep:
            pos_rep += 1
        state = ns
        if info["outcome"] != "running":
            outcome = str(info["outcome"])
            break
    if env_pos.pos == goal:
        outcome = "success"
    bfs_len = len(path) - 1
    start_dist = int(env_pos.dist[start])
    pos_final = int(env_pos.dist[env_pos.pos]) if env_pos.dist[env_pos.pos] < 10_000 else 10_000
    pos = finalize_trajectory(pos_trans, outcome, bfs_len, start_dist, pos_final, pos_wall, pos_rep, "cpl_forward", args)

    # Negative branch: same prefix, then repeated wall hits.
    env_neg = MazeEnv(grid, start, goal, max_steps=args.max_steps)
    state = run_prefix(env_neg)
    neg_trans: List[RMTransition] = []
    neg_wall = 0
    neg_rep = 0
    wall_as = wall_actions_at(env_neg)
    if not wall_as:
        return None
    wall_a = random.choice(wall_as)
    for _ in range(args.max_steps - env_neg.steps):
        ns, tr, info, rep = record_step(env_neg, state, wall_a)
        neg_trans.append(tr)
        if info["hit_wall"]:
            neg_wall += 1
        if rep:
            neg_rep += 1
        state = ns
        if info["outcome"] != "running":
            break
    neg_final = int(env_neg.dist[env_neg.pos]) if env_neg.dist[env_neg.pos] < 10_000 else 10_000
    neg = finalize_trajectory(neg_trans, "timeout", bfs_len, start_dist, neg_final, neg_wall, neg_rep, "cpl_sticky_wall", args)
    if pos.quality_key <= neg.quality_key:
        # It should normally be positive, but keep generation robust.
        return None
    return PreferencePair(pos, neg, max(0.0, neg.badness - pos.badness), 0.0, "cpl")


def assign_adaptive_margins(pairs: List[PreferencePair], args: argparse.Namespace) -> None:
    if not pairs:
        return
    if not args.use_margin:
        for pr in pairs:
            pr.margin = 0.0
        return
    deltas = np.asarray([max(0.0, pr.delta_badness) for pr in pairs], dtype=np.float32)
    q_low = float(np.quantile(deltas, args.margin_q_low))
    q_high = float(np.quantile(deltas, args.margin_q_high))
    denom = max(1e-6, q_high - q_low)
    for pr in pairs:
        # Any success > timeout is a strict qualitative jump; use full normalized margin.
        if pr.pos.outcome == "success" and pr.neg.outcome == "timeout":
            m = args.margin_max
        else:
            z = (max(0.0, pr.delta_badness) - q_low) / denom
            m = args.margin_max * float(np.clip(z, 0.0, 1.0))
            if pr.pos.outcome == pr.neg.outcome:
                # Same-outcome no-wall versus has-wall is qualitative. Ensure a clear gap.
                pos_has_wall = pr.pos.wall_hits > 0
                neg_has_wall = pr.neg.wall_hits > 0
                if (not pos_has_wall) and neg_has_wall:
                    m = max(m, args.wall_presence_margin_floor)
        pr.margin = max(args.margin_min, min(args.margin_max, m))


def build_preference_dataset(args: argparse.Namespace) -> Tuple[List[PreferencePair], Dict[str, int], Dict[str, float]]:
    pairs: List[PreferencePair] = []
    stats = {
        "mazes": 0,
        "trajectories": 0,
        "pairs": 0,
        "success": 0,
        "timeout": 0,
        "bfs": 0,
        "noisy_bfs": 0,
        "random": 0,
        "safe_loop": 0,
        "sticky_wall": 0,
        "cpl_forward": 0,
        "cpl_sticky_wall": 0,
        "skipped_ties": 0,
        "cpl_pairs": 0,
    }
    noise_values = [0.10, 0.20, 0.35]
    weights = {"easy": args.easy_ratio, "medium": args.medium_ratio, "hard": args.hard_ratio}

    for _ in range(args.rm_mazes):
        difficulty = weighted_choice(weights)
        grid, start, goal = generate_maze(difficulty)
        env = MazeEnv(grid, start, goal, max_steps=args.max_steps)
        trajs: List[Trajectory] = []

        trajs.append(make_trajectory(env, "bfs", args))
        for _n in range(args.noisy_bfs_per_maze):
            trajs.append(make_trajectory(env, "noisy_bfs", args, noise_prob=random.choice(noise_values)))
        for _n in range(args.random_per_maze):
            trajs.append(make_trajectory(env, "random", args))
        for _n in range(args.safe_loop_per_maze):
            trajs.append(make_trajectory(env, "safe_loop", args))
        for _n in range(args.sticky_wall_per_maze):
            trajs.append(make_trajectory(env, "sticky_wall", args))

        stats["mazes"] += 1
        stats["trajectories"] += len(trajs)
        for tr in trajs:
            stats[tr.outcome] += 1
            stats[tr.source] += 1

        candidate_pairs: List[PreferencePair] = []
        for i in range(len(trajs)):
            for j in range(i + 1, len(trajs)):
                a, b = trajs[i], trajs[j]
                if a.quality_key == b.quality_key:
                    stats["skipped_ties"] += 1
                    continue
                if a.quality_key > b.quality_key:
                    pos, neg = a, b
                else:
                    pos, neg = b, a
                candidate_pairs.append(PreferencePair(pos, neg, max(0.0, neg.badness - pos.badness), 0.0, "ranking"))
        random.shuffle(candidate_pairs)
        pairs.extend(candidate_pairs[:args.pairs_per_maze])
        stats["pairs"] += min(len(candidate_pairs), args.pairs_per_maze)

        if args.use_cpl:
            for _ in range(args.cpl_pairs_per_maze):
                pr = make_cpl_pair(grid, start, goal, args)
                if pr is not None:
                    pairs.append(pr)
                    stats["pairs"] += 1
                    stats["cpl_pairs"] += 1

    assign_adaptive_margins(pairs, args)
    margins = [p.margin for p in pairs]
    deltas = [p.delta_badness for p in pairs]
    margin_stats = {
        "delta_badness_mean": float(np.mean(deltas)) if deltas else 0.0,
        "delta_badness_p10": float(np.percentile(deltas, 10)) if deltas else 0.0,
        "delta_badness_p50": float(np.percentile(deltas, 50)) if deltas else 0.0,
        "delta_badness_p90": float(np.percentile(deltas, 90)) if deltas else 0.0,
        "margin_mean": float(np.mean(margins)) if margins else 0.0,
        "margin_p10": float(np.percentile(margins, 10)) if margins else 0.0,
        "margin_p50": float(np.percentile(margins, 50)) if margins else 0.0,
        "margin_p90": float(np.percentile(margins, 90)) if margins else 0.0,
    }
    return pairs, stats, margin_stats

# -----------------------------
# Models
# -----------------------------

class TransitionRewardCNN(nn.Module):
    def __init__(self, in_channels: int = 12):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * SIZE * SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x)).squeeze(-1)


class QNetwork(nn.Module):
    def __init__(self, in_channels: int = 4, n_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * SIZE * SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Reward model training
# -----------------------------


def transition_rewards_tensor(model: TransitionRewardCNN, traj: Trajectory, device: torch.device) -> torch.Tensor:
    if not traj.transitions:
        return torch.zeros((1,), dtype=torch.float32, device=device)
    xs = [transition_tensor(t.state, t.action, t.next_state) for t in traj.transitions]
    x = torch.tensor(np.stack(xs), dtype=torch.float32, device=device)
    return model(x)


def trajectory_score(model: TransitionRewardCNN, traj: Trajectory, device: torch.device, normalizer: str = "bfs_len") -> torch.Tensor:
    rs = transition_rewards_tensor(model, traj, device)
    denom = 1.0
    if normalizer == "bfs_len":
        denom = float(max(1, traj.bfs_len))
    elif normalizer == "episode_len":
        denom = float(max(1, traj.steps))
    elif normalizer == "raw_sum":
        denom = 1.0
    else:
        raise ValueError(f"unknown score normalizer: {normalizer}")
    return rs.sum() / denom


def batch_btl_loss(model: TransitionRewardCNN, batch: Sequence[PreferencePair], device: torch.device, args: argparse.Namespace) -> Tuple[torch.Tensor, float, float, float, float]:
    diffs: List[torch.Tensor] = []
    pos_scores = []
    neg_scores = []
    margins = []
    reg_terms: List[torch.Tensor] = []
    correct = 0
    for pair in batch:
        rp = transition_rewards_tensor(model, pair.pos, device)
        rn = transition_rewards_tensor(model, pair.neg, device)
        denom_p = float(max(1, pair.pos.bfs_len)) if args.score_normalizer == "bfs_len" else (float(max(1, pair.pos.steps)) if args.score_normalizer == "episode_len" else 1.0)
        denom_n = float(max(1, pair.neg.bfs_len)) if args.score_normalizer == "bfs_len" else (float(max(1, pair.neg.steps)) if args.score_normalizer == "episode_len" else 1.0)
        sp = rp.sum() / denom_p
        sn = rn.sum() / denom_n
        margin = float(pair.margin if args.use_margin else 0.0)
        diffs.append(sp - sn - margin)
        reg_terms.append(torch.cat([rp, rn]).pow(2).mean())
        pos_scores.append(float(sp.detach().cpu()))
        neg_scores.append(float(sn.detach().cpu()))
        margins.append(margin)
        if float((sp - sn).detach().cpu()) > margin:
            correct += 1
    diff = torch.stack(diffs)
    loss_rank = -F.logsigmoid(diff).mean()
    loss_reg = args.reward_l2 * torch.stack(reg_terms).mean() if reg_terms and args.reward_l2 > 0 else torch.zeros((), device=device)
    loss = loss_rank + loss_reg
    acc = correct / max(1, len(batch))
    return loss, acc, float(np.mean(pos_scores)), float(np.mean(neg_scores)), float(np.mean(margins))

def evaluate_rm(model: TransitionRewardCNN, pairs: Sequence[PreferencePair], device: torch.device, batch_size: int, args: argparse.Namespace) -> Dict[str, float]:
    model.eval()
    losses = []
    accs = []
    pos_vals = []
    neg_vals = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            loss, acc, pos_m, neg_m, _margin_m = batch_btl_loss(model, batch, device, args)
            losses.append(float(loss.cpu()))
            accs.append(acc)
            pos_vals.append(pos_m)
            neg_vals.append(neg_m)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(np.mean(accs)) if accs else 0.0,
        "pos_score": float(np.mean(pos_vals)) if pos_vals else 0.0,
        "neg_score": float(np.mean(neg_vals)) if neg_vals else 0.0,
    }


def save_rm_checkpoint(
    path: Path,
    model: TransitionRewardCNN,
    args: argparse.Namespace,
    stats: Dict[str, int],
    margin_stats: Dict[str, float],
    history: List[Dict[str, float]],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "version": VERSION,
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "dataset_stats": stats,
        "margin_stats": margin_stats,
        "history": history,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def train_reward_model(args: argparse.Namespace, device: torch.device, out_dir: Path) -> Path:
    print("\n=== Stage A: Train v2.0.5 Normalized Margin Preference Reward Model ===")
    print("Preference key: success > timeout; wall is an event, not outcome.")
    print("Success is always strictly preferred over timeout, even with wall/repeat/long path.")
    print("Trajectory score: sum R_phi / BFS_len. Loss: -log sigmoid(score+ - score- - margin).")
    print("Safe-loop and sticky-wall trajectories are explicitly included.")

    pairs, stats, margin_stats = build_preference_dataset(args)
    if len(pairs) < 10:
        raise RuntimeError("not enough preference pairs generated")
    random.shuffle(pairs)
    split = int(len(pairs) * (1.0 - args.rm_val_ratio))
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]
    print(f"[RM dataset] stats={stats}")
    print(f"[RM dataset] margin_stats={margin_stats}")
    print(f"[RM dataset] train_pairs={len(train_pairs)} val_pairs={len(val_pairs)}")

    model = TransitionRewardCNN().to(device)
    opt = optim.Adam(model.parameters(), lr=args.rm_lr)
    history: List[Dict[str, float]] = []
    best_val_acc = float("-inf")
    best_epoch = 0
    best_path = out_dir / "v2.0.5_reward_model_best.pt"
    last_path = out_dir / "v2.0.5_reward_model_last.pt"
    use_val_metric = len(val_pairs) > 0

    for epoch in range(1, args.rm_epochs + 1):
        model.train()
        random.shuffle(train_pairs)
        losses = []
        accs = []
        pos_vals = []
        neg_vals = []
        for i in range(0, len(train_pairs), args.rm_batch_size):
            batch = train_pairs[i:i + args.rm_batch_size]
            opt.zero_grad(set_to_none=True)
            loss, acc, pos_m, neg_m, margin_m = batch_btl_loss(model, batch, device, args)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.rm_grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            accs.append(acc)
            pos_vals.append(pos_m)
            neg_vals.append(neg_m)
        val = evaluate_rm(model, val_pairs, device, args.rm_batch_size, args) if val_pairs else {"loss": 0.0, "accuracy": 0.0, "pos_score": 0.0, "neg_score": 0.0}
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_accuracy": float(np.mean(accs)),
            "train_pos_score": float(np.mean(pos_vals)),
            "train_neg_score": float(np.mean(neg_vals)),
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_pos_score": val["pos_score"],
            "val_neg_score": val["neg_score"],
        }
        history.append(row)
        print(
            f"rm epoch={epoch:03d}/{args.rm_epochs} "
            f"loss={row['train_loss']:.4f} acc={row['train_accuracy']:.3f} "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_accuracy']:.3f} "
            f"pos={row['val_pos_score']:.2f} neg={row['val_neg_score']:.2f}"
        )

        metric = row["val_accuracy"] if use_val_metric else row["train_accuracy"]
        if metric > best_val_acc:
            best_val_acc = metric
            best_epoch = epoch
            save_rm_checkpoint(
                best_path,
                model,
                args,
                stats,
                margin_stats,
                history,
                extra={
                    "best_epoch": best_epoch,
                    "best_val_accuracy": best_val_acc,
                    "selection_metric": "val_accuracy" if use_val_metric else "train_accuracy",
                },
            )
            print(
                f"[Save] best reward model (epoch={best_epoch}, "
                f"{'val' if use_val_metric else 'train'}_acc={best_val_acc:.4f}): {best_path}"
            )

    save_rm_checkpoint(
        last_path,
        model,
        args,
        stats,
        margin_stats,
        history,
        extra={"checkpoint_kind": "last_epoch", "epoch": args.rm_epochs},
    )
    history_meta = {
        "stats": stats,
        "margin_stats": margin_stats,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc if math.isfinite(best_val_acc) else None,
        "selection_metric": "val_accuracy" if use_val_metric else "train_accuracy",
        "best_model_path": str(best_path),
        "last_model_path": str(last_path),
    }
    with open(out_dir / "v2.0.5_reward_model_history.json", "w", encoding="utf-8") as f:
        json.dump(history_meta, f, indent=2, ensure_ascii=False)
    plot_rm_history(history, out_dir / "v2.0.5_reward_model_curves.png")
    if best_epoch > 0:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    if not getattr(args, "no_debug", False):
        run_reward_model_debug(model, args, device, out_dir, tag="rm_after_train")
    print(f"[Save] last reward model: {last_path}")
    if best_epoch > 0:
        print(f"[Save] best reward model: {best_path} (epoch={best_epoch}, val_acc={best_val_acc:.4f})")
        return best_path
    print("[Warn] no best checkpoint saved; using last epoch for downstream QCNN")
    return last_path


def plot_rm_history(history: List[Dict[str, float]], path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [h["train_loss"] for h in history], label="train")
    plt.plot(epochs, [h["val_loss"] for h in history], label="val")
    plt.title("BTL loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [h["train_accuracy"] for h in history], label="train")
    plt.plot(epochs, [h["val_accuracy"] for h in history], label="val")
    plt.title("Preference accuracy")
    plt.xlabel("epoch")
    plt.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)




# -----------------------------
# Reward-model diagnostics
# -----------------------------

def argmax_pos(plane: np.ndarray) -> Tuple[int, int]:
    idx = int(np.argmax(plane))
    return divmod(idx, SIZE)


def infer_transition_features(tr: RMTransition) -> Dict[str, object]:
    s = tr.state
    ns = tr.next_state
    grid = s[0]
    pos = argmax_pos(s[1])
    next_pos = argmax_pos(ns[1])
    goal = argmax_pos(s[2])
    visited = s[3]
    dr, dc = ACTIONS[tr.action]
    target = (pos[0] + dr, pos[1] + dc)
    hit_wall = (not in_bounds(target)) or (grid[target] > 0.5)
    success = next_pos == goal
    repeat = (not hit_wall) and (visited[next_pos] > 0.5)
    dist = bfs_distances(grid.astype(np.int8), goal)
    d0 = int(dist[pos]) if dist[pos] < 10_000 else 10_000
    d1 = int(dist[next_pos]) if dist[next_pos] < 10_000 else 10_000
    delta_d = d0 - d1
    if success:
        category = "goal"
    elif hit_wall:
        category = "wall"
    elif repeat:
        category = "repeat"
    elif delta_d > 0:
        category = "toward_goal"
    elif delta_d < 0:
        category = "away_goal"
    else:
        category = "flat"
    return {
        "pos": pos,
        "next_pos": next_pos,
        "goal": goal,
        "action": tr.action,
        "hit_wall": hit_wall,
        "success": success,
        "repeat": repeat,
        "d0": d0,
        "d1": d1,
        "delta_d": delta_d,
        "category": category,
    }


def rm_step_reward(model: TransitionRewardCNN, tr: RMTransition, device: torch.device, scale: float = 1.0) -> float:
    with torch.no_grad():
        x = torch.tensor(transition_tensor(tr.state, tr.action, tr.next_state)[None, ...], dtype=torch.float32, device=device)
        return float(model(x).item() * scale)


def score_trajectory_raw(model: TransitionRewardCNN, traj: Trajectory, device: torch.device, scale: float = 1.0) -> Tuple[float, List[float]]:
    vals = [rm_step_reward(model, tr, device, scale=scale) for tr in traj.transitions]
    return float(sum(vals)), vals


def summarize_values(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"n": 0, "mean": 0.0, "std": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(vals, dtype=np.float32)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def simulate_transition_from_state(state: np.ndarray, action: int) -> RMTransition:
    grid = state[0].astype(np.int8)
    pos = argmax_pos(state[1])
    goal = argmax_pos(state[2])
    visited = state[3].copy()
    dr, dc = ACTIONS[action]
    target = (pos[0] + dr, pos[1] + dc)
    hit_wall = (not in_bounds(target)) or (not is_free(grid, target))
    next_pos = pos if hit_wall else target
    next_state = state.copy()
    next_state[1, :, :] = 0.0
    next_state[1, next_pos[0], next_pos[1]] = 1.0
    if not hit_wall:
        next_state[3, next_pos[0], next_pos[1]] = 1.0
    next_state[2, :, :] = 0.0
    next_state[2, goal[0], goal[1]] = 1.0
    return RMTransition(state.copy(), action, next_state.copy())


def collect_probe_trajectories(args: argparse.Namespace, probe_mazes: int) -> List[Trajectory]:
    trajectories: List[Trajectory] = []
    weights = {"easy": args.easy_ratio, "medium": args.medium_ratio, "hard": args.hard_ratio}
    for _ in range(probe_mazes):
        diff = weighted_choice(weights)
        grid, start, goal = generate_maze(diff)
        for policy, noise in [("bfs", 0.0), ("noisy_bfs", 0.15), ("noisy_bfs", 0.35), ("random", 0.0), ("safe_loop", 0.0), ("sticky_wall", 0.0)]:
            env = MazeEnv(grid, start, goal, max_steps=args.max_steps)
            trajectories.append(make_trajectory(env, policy, args, noise_prob=noise))
    return trajectories


def run_reward_model_debug(
    model: TransitionRewardCNN,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
    tag: str = "rm",
) -> Dict[str, object]:
    model.eval()
    probe_n = max(1, int(getattr(args, "debug_probe_mazes", 30)))
    trajectories = collect_probe_trajectories(args, probe_n)

    # A. Trajectory score diagnostics: detects length bias and safe-loop exploitation.
    traj_rows: List[Dict[str, object]] = []
    step_by_cat: Dict[str, List[float]] = {}
    step_by_source: Dict[str, List[float]] = {}
    for idx, traj in enumerate(trajectories):
        total, vals = score_trajectory_raw(model, traj, device, scale=1.0)
        avg = total / max(1, traj.steps)
        bfsnorm = total / max(1, traj.bfs_len)
        traj_rows.append({
            "idx": idx,
            "source": traj.source,
            "outcome": traj.outcome,
            "steps": traj.steps,
            "bfs_len": traj.bfs_len,
            "gap": traj.gap,
            "final_bfs_dist": traj.final_bfs_dist,
            "wall_hits": traj.wall_hits,
            "score_sum": total,
            "score_avg": avg,
            "score_bfsnorm": bfsnorm,
            "badness": traj.badness,
            "repeat_count": traj.repeat_count,
            "quality_key": list(traj.quality_key),
        })
        for tr, r in zip(traj.transitions, vals):
            feat = infer_transition_features(tr)
            step_by_cat.setdefault(str(feat["category"]), []).append(r)
            step_by_source.setdefault(traj.source, []).append(r)

    by_traj_group: Dict[str, List[float]] = {}
    by_traj_group_avg: Dict[str, List[float]] = {}
    by_traj_group_bfsnorm: Dict[str, List[float]] = {}
    for row in traj_rows:
        key = f"{row['source']}|{row['outcome']}"
        by_traj_group.setdefault(key, []).append(float(row["score_sum"]))
        by_traj_group_avg.setdefault(key, []).append(float(row["score_avg"]))
        by_traj_group_bfsnorm.setdefault(key, []).append(float(row["score_bfsnorm"]))

    # B. Counterfactual action sensitivity: fixed state, enumerate four actions.
    action_rows: List[Dict[str, object]] = []
    sample_states: List[np.ndarray] = []
    for traj in trajectories:
        if traj.transitions:
            # Sample the first, middle, and last-ish states when available.
            inds = sorted(set([0, len(traj.transitions) // 2, max(0, len(traj.transitions) - 1)]))
            for i in inds:
                if len(sample_states) < 60:
                    sample_states.append(traj.transitions[i].state)
    for sid, state in enumerate(sample_states):
        for action in range(4):
            tr = simulate_transition_from_state(state, action)
            r = rm_step_reward(model, tr, device, scale=1.0)
            feat = infer_transition_features(tr)
            action_rows.append({
                "state_id": sid,
                "action": action,
                "reward": r,
                "category": feat["category"],
                "hit_wall": bool(feat["hit_wall"]),
                "repeat": bool(feat["repeat"]),
                "success": bool(feat["success"]),
                "d0": int(feat["d0"]),
                "d1": int(feat["d1"]),
                "delta_d": int(feat["delta_d"]),
            })

    # C. Action-shuffle sensitivity: if reward barely changes after replacing action planes,
    # the model is likely ignoring action encoding.
    shuffle_deltas: List[float] = []
    for traj in trajectories[: min(40, len(trajectories))]:
        for tr in traj.transitions[: min(10, len(traj.transitions))]:
            base = rm_step_reward(model, tr, device, scale=1.0)
            alt_action = random.choice([a for a in range(4) if a != tr.action])
            fake = RMTransition(tr.state, alt_action, tr.next_state)
            alt = rm_step_reward(model, fake, device, scale=1.0)
            shuffle_deltas.append(abs(base - alt))

    report: Dict[str, object] = {
        "tag": tag,
        "probe_mazes": probe_n,
        "trajectory_group_score_sum": {k: summarize_values(v) for k, v in sorted(by_traj_group.items())},
        "trajectory_group_score_avg": {k: summarize_values(v) for k, v in sorted(by_traj_group_avg.items())},
        "trajectory_group_score_bfsnorm": {k: summarize_values(v) for k, v in sorted(by_traj_group_bfsnorm.items())},
        "step_reward_by_category": {k: summarize_values(v) for k, v in sorted(step_by_cat.items())},
        "step_reward_by_source": {k: summarize_values(v) for k, v in sorted(step_by_source.items())},
        "action_sensitivity_abs_delta": summarize_values(shuffle_deltas),
        "trajectory_rows_sample": traj_rows[:100],
        "counterfactual_action_rows_sample": action_rows[:200],
    }

    json_path = out_dir / f"v2.0.5_{tag}_debug_report.json"
    csv_traj = out_dir / f"v2.0.5_{tag}_debug_trajectory_scores.csv"
    csv_action = out_dir / f"v2.0.5_{tag}_debug_counterfactual_actions.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    if traj_rows:
        with open(csv_traj, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(traj_rows[0].keys()))
            writer.writeheader()
            writer.writerows(traj_rows)
    if action_rows:
        with open(csv_action, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(action_rows[0].keys()))
            writer.writeheader()
            writer.writerows(action_rows)

    print(f"[Debug:{tag}] saved report: {json_path}")
    print(f"[Debug:{tag}] step_reward_by_category={report['step_reward_by_category']}")
    print(f"[Debug:{tag}] action_sensitivity_abs_delta={report['action_sensitivity_abs_delta']}")
    return report


def run_q_policy_debug(
    q_net: QNetwork,
    rm: TransitionRewardCNN,
    cfg: DQNConfig,
    device: torch.device,
    out_dir: Path,
    tag: str = "q_policy",
) -> Dict[str, object]:
    q_net.eval()
    rm.eval()
    grid, start, goal = parse_maze_text(TEST_MAZE_TEXT)
    env = MazeEnv(grid, start, goal, max_steps=cfg.max_steps)
    state = env.reset()
    rows: List[Dict[str, object]] = []
    total_r = 0.0
    for t in range(cfg.max_steps):
        with torch.no_grad():
            qs = q_net(torch.tensor(state[None, ...], dtype=torch.float32, device=device))[0].detach().cpu().numpy().astype(float).tolist()
        action_rewards = []
        for a in range(4):
            tr = simulate_transition_from_state(state, a)
            rr = rm_step_reward(rm, tr, device, scale=cfg.rm_scale)
            feat = infer_transition_features(tr)
            action_rewards.append({"a": a, "rm_reward": rr, "category": feat["category"], "delta_d": feat["delta_d"]})
        action = int(np.argmax(qs))
        next_state, info = env.step_raw(action)
        r = learned_reward(rm, state, action, next_state, device, cfg.rm_scale)
        total_r += r
        chosen_feat = infer_transition_features(RMTransition(state, action, next_state))
        rows.append({
            "t": t,
            "pos": list(info["before"]),
            "action": action,
            "q_values": qs,
            "chosen_rm_reward": r,
            "chosen_category": chosen_feat["category"],
            "chosen_delta_d": int(chosen_feat["delta_d"]),
            "hit_wall": bool(info["hit_wall"]),
            "success": bool(info["success"]),
            "timeout": bool(info["timeout"]),
            "bfs_dist": int(info["bfs_dist"]),
            "action_rewards": action_rewards,
        })
        state = next_state
        if info["outcome"] != "running":
            break
    report = {"tag": tag, "total_reward": total_r, "steps": len(rows), "rows": rows}
    path = out_dir / f"v2.0.5_{tag}_debug_rollout.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[Debug:{tag}] saved rollout: {path}")
    return report


# -----------------------------
# DQN training with frozen reward model
# -----------------------------

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data: deque = deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.data.append(DQNTransition(*args))

    def sample(self, batch_size: int) -> DQNTransition:
        batch = random.sample(self.data, batch_size)
        return DQNTransition(*zip(*batch))

    def __len__(self) -> int:
        return len(self.data)


def epsilon_by_episode(ep: int, cfg: DQNConfig) -> float:
    if ep >= cfg.epsilon_decay_episodes:
        return cfg.epsilon_end
    frac = ep / max(1, cfg.epsilon_decay_episodes)
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def select_action(q_net: QNetwork, state: np.ndarray, epsilon: float, device: torch.device) -> int:
    if random.random() < epsilon:
        return random.randrange(4)
    with torch.no_grad():
        x = torch.tensor(state[None, ...], dtype=torch.float32, device=device)
        q = q_net(x)[0]
        return int(torch.argmax(q).item())


def learned_reward(
    rm: TransitionRewardCNN,
    state: np.ndarray,
    action: int,
    next_state: np.ndarray,
    device: torch.device,
    scale: float,
) -> float:
    with torch.no_grad():
        x = torch.tensor(transition_tensor(state, action, next_state)[None, ...], dtype=torch.float32, device=device)
        return float(rm(x).item() * scale)


def optimize_dqn(q_net: QNetwork, target_net: QNetwork, opt: optim.Optimizer, replay: ReplayBuffer, cfg: DQNConfig, device: torch.device) -> Optional[float]:
    if len(replay) < max(cfg.batch_size, cfg.warmup_steps):
        return None
    batch = replay.sample(cfg.batch_size)
    states = torch.tensor(np.stack(batch.state), dtype=torch.float32, device=device)
    actions = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.tensor(np.stack(batch.next_state), dtype=torch.float32, device=device)
    dones = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)

    q = q_net(states).gather(1, actions)
    with torch.no_grad():
        next_actions = torch.argmax(q_net(next_states), dim=1, keepdim=True)
        next_q = target_net(next_states).gather(1, next_actions)
        target = rewards + cfg.gamma * (1.0 - dones) * next_q
    loss = F.smooth_l1_loss(q, target)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
    opt.step()
    return float(loss.detach().cpu())


def rollout_eval(
    q_net: QNetwork,
    rm: TransitionRewardCNN,
    difficulty: str,
    n: int,
    cfg: DQNConfig,
    device: torch.device,
    fixed_test: bool = False,
) -> Dict[str, float]:
    q_net.eval()
    rm.eval()
    totals = {"success": 0, "wall": 0, "timeout": 0, "steps": 0.0, "repeat": 0.0, "bfs_gap": 0.0, "reward": 0.0, "bfs_agree": 0.0}
    for _ in range(n):
        if fixed_test:
            grid, start, goal = parse_maze_text(TEST_MAZE_TEXT)
        else:
            grid, start, goal = generate_maze(difficulty)
        env = MazeEnv(grid, start, goal, max_steps=cfg.max_steps)
        path = bfs_path(grid, start, goal)
        bfs_len = len(path) - 1 if path else cfg.max_steps
        state = env.reset()
        total_r = 0.0
        repeats = 0
        wall_hits = 0
        bfs_agree_count = 0
        outcome = "timeout"
        for _t in range(cfg.max_steps):
            action = select_action(q_net, state, 0.0, device)
            old_visited = env.visited.copy()
            old_pos = env.pos
            cur_d = int(env.dist[old_pos])
            dr, dc = ACTIONS[action]
            cand = (old_pos[0] + dr, old_pos[1] + dc)
            if is_free(env.grid, cand) and int(env.dist[cand]) < cur_d:
                bfs_agree_count += 1
            next_state, info = env.step_raw(action)
            if info["hit_wall"]:
                wall_hits += 1
            if not info["hit_wall"] and old_visited[env.pos] > 0.5:
                repeats += 1
            r = learned_reward(rm, state, action, next_state, device, cfg.rm_scale)
            if cfg.anchor_terminal:
                if info["success"]:
                    r += cfg.anchor_goal
                elif info["hit_wall"]:
                    r += cfg.anchor_wall
            total_r += r
            state = next_state
            if info["outcome"] != "running":
                outcome = str(info["outcome"])
                break

        if outcome == "success":
            totals["success"] += 1
        elif wall_hits > 0:
            totals["wall"] += 1
        else:
            totals["timeout"] += 1
        actual_steps = env.steps
        totals["steps"] += float(actual_steps)
        totals["repeat"] += float(repeats)
        totals["bfs_gap"] += float(max(0, actual_steps - bfs_len) if env.pos == env.goal else cfg.max_steps - bfs_len)
        totals["reward"] += float(total_r)
        totals["bfs_agree"] += float(bfs_agree_count / max(1, actual_steps))
    denom = float(n)
    return {
        "success": totals["success"] / denom,
        "wall": totals["wall"] / denom,
        "timeout": totals["timeout"] / denom,
        "steps": totals["steps"] / denom,
        "repeat": totals["repeat"] / denom,
        "bfs_gap": totals["bfs_gap"] / denom,
        "reward": totals["reward"] / denom,
        "bfs_agree": totals["bfs_agree"] / denom,
    }




def estimate_auto_rm_scale(rm: TransitionRewardCNN, args: argparse.Namespace, device: torch.device) -> Tuple[float, float]:
    trajectories = collect_probe_trajectories(args, max(5, min(args.debug_probe_mazes, 50)))
    vals: List[float] = []
    rm.eval()
    with torch.no_grad():
        for traj in trajectories:
            for tr in traj.transitions:
                vals.append(rm_step_reward(rm, tr, device, scale=1.0))
    std = float(np.std(vals)) if vals else 1.0
    scale = float(args.target_rm_std / (std + 1e-6))
    return scale, std

def train_qcnn(args: argparse.Namespace, device: torch.device, out_dir: Path, reward_model_path: Path) -> Path:
    print("\n=== Stage B: Train QCNN with frozen preference reward model ===")
    cfg = DQNConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        warmup_steps=args.warmup_steps,
        updates_per_episode=args.updates_per_episode,
        target_update_interval=args.target_update_interval,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
        eval_n=args.eval_n,
        rm_scale=args.rm_scale,
        anchor_terminal=args.anchor_terminal,
        anchor_goal=args.anchor_goal,
        anchor_wall=args.anchor_wall,
    )
    rm = TransitionRewardCNN().to(device)
    ckpt = torch.load(reward_model_path, map_location=device)
    rm.load_state_dict(ckpt["model_state_dict"])
    rm.eval()
    for p in rm.parameters():
        p.requires_grad_(False)

    if args.auto_rm_scale:
        auto_scale, rm_std = estimate_auto_rm_scale(rm, args, device)
        cfg.rm_scale = auto_scale
        print(f"[RSC] auto rm_scale={auto_scale:.4f} from rm_std={rm_std:.4f}, target_std={args.target_rm_std}")
    else:
        print(f"[RSC] manual rm_scale={cfg.rm_scale:.4f}")

    q_net = QNetwork().to(device)
    target_net = QNetwork().to(device)
    target_net.load_state_dict(q_net.state_dict())
    opt = optim.Adam(q_net.parameters(), lr=cfg.lr)
    replay = ReplayBuffer(cfg.replay_size)

    history: List[Dict[str, float]] = []
    global_step = 0
    difficulty_weights = {"easy": args.easy_ratio, "medium": args.medium_ratio, "hard": args.hard_ratio}

    for ep in range(1, cfg.episodes + 1):
        diff = weighted_choice(difficulty_weights)
        grid, start, goal = generate_maze(diff)
        env = MazeEnv(grid, start, goal, max_steps=cfg.max_steps)
        state = env.reset()
        eps = epsilon_by_episode(ep, cfg)
        ep_reward = 0.0
        repeats = 0
        wall_hits = 0
        bfs_agree_count = 0
        outcome = "timeout"
        losses = []
        path = bfs_path(grid, start, goal)
        bfs_len = len(path) - 1 if path else cfg.max_steps

        for t in range(cfg.max_steps):
            action = select_action(q_net, state, eps, device)
            old_visited = env.visited.copy()
            old_pos = env.pos
            cur_d = int(env.dist[old_pos])
            cand = (old_pos[0] + ACTIONS[action][0], old_pos[1] + ACTIONS[action][1])
            if is_free(env.grid, cand) and int(env.dist[cand]) < cur_d:
                bfs_agree_count += 1
            next_state, info = env.step_raw(action)
            if info["hit_wall"]:
                wall_hits += 1
            if not info["hit_wall"] and old_visited[env.pos] > 0.5:
                repeats += 1
            r = learned_reward(rm, state, action, next_state, device, cfg.rm_scale)
            if cfg.anchor_terminal:
                if info["success"]:
                    r += cfg.anchor_goal
                elif info["hit_wall"]:
                    r += cfg.anchor_wall
            done = bool(info["outcome"] != "running")
            replay.push(state, action, r, next_state, done)
            state = next_state
            ep_reward += r
            global_step += 1
            for _ in range(cfg.updates_per_episode // cfg.max_steps if cfg.max_steps > 0 else 1):
                loss = optimize_dqn(q_net, target_net, opt, replay, cfg, device)
                if loss is not None:
                    losses.append(loss)
            if global_step % cfg.target_update_interval == 0:
                target_net.load_state_dict(q_net.state_dict())
            if done:
                outcome = str(info["outcome"])
                break

        # Episode-end updates to keep total updates stable.
        for _ in range(cfg.updates_per_episode):
            loss = optimize_dqn(q_net, target_net, opt, replay, cfg, device)
            if loss is not None:
                losses.append(loss)
        actual_steps = env.steps if env.steps > 0 else cfg.max_steps
        row = {
            "episode": ep,
            "epsilon": eps,
            "reward": ep_reward,
            "success": 1.0 if outcome == "success" else 0.0,
            "wall": 1.0 if (outcome != "success" and wall_hits > 0) else 0.0,
            "timeout": 1.0 if (outcome != "success" and wall_hits == 0) else 0.0,
            "steps": float(actual_steps),
            "repeat": float(repeats),
            "bfs_gap": float(max(0, actual_steps - bfs_len) if outcome == "success" else cfg.max_steps - bfs_len),
            "bfs_agree": float(bfs_agree_count / max(1, actual_steps)),
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "buffer": len(replay),
        }
        history.append(row)
        if ep % 100 == 0 or ep == 1:
            recent = history[-100:]
            print(
                f"ep={ep:4d}/{cfg.episodes} eps={eps:.3f} "
                f"reward={np.mean([x['reward'] for x in recent]):8.2f} "
                f"success={100*np.mean([x['success'] for x in recent]):6.2f}% "
                f"wall={100*np.mean([x['wall'] for x in recent]):6.2f}% "
                f"steps={np.mean([x['steps'] for x in recent]):6.2f} "
                f"repeat={np.mean([x['repeat'] for x in recent]):6.2f} "
                f"bfsGap={np.mean([x['bfs_gap'] for x in recent]):6.2f} "
                f"loss={row['loss']:.4f} buf={len(replay)}"
            )

    model_path = out_dir / "v2.0.5_qcnn_from_preference_reward.pt"
    torch.save({
        "version": VERSION,
        "model_state_dict": q_net.state_dict(),
        "reward_model": str(reward_model_path),
        "args": vars(args),
        "history": history,
    }, model_path)
    with open(out_dir / "v2.0.5_qcnn_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    plot_q_history(history, out_dir / "v2.0.5_qcnn_curves.png")

    evals = evaluate_all(q_net, rm, cfg, device, args.eval_n)
    with open(out_dir / "v2.0.5_qcnn_eval.json", "w", encoding="utf-8") as f:
        json.dump(evals, f, indent=2, ensure_ascii=False)
    print("\n=== Evaluation Suite ===")
    for name, m in evals.items():
        print(
            f"{name:7s} success={100*m['success']:6.2f}% wall={100*m['wall']:6.2f}% "
            f"timeout={100*m['timeout']:6.2f}% steps={m['steps']:6.2f} "
            f"repeat={m['repeat']:6.2f} bfsGap={m['bfs_gap']:6.2f} reward={m['reward']:7.2f}"
        )
    if not getattr(args, "no_debug", False):
        run_reward_model_debug(rm, args, device, out_dir, tag="rm_after_q_train")
        run_q_policy_debug(q_net, rm, cfg, device, out_dir, tag="q_policy_test")
    print(f"[Save] qcnn model: {model_path}")
    return model_path


def evaluate_all(q_net: QNetwork, rm: TransitionRewardCNN, cfg: DQNConfig, device: torch.device, n: int) -> Dict[str, Dict[str, float]]:
    return {
        "easy": rollout_eval(q_net, rm, "easy", n, cfg, device),
        "medium": rollout_eval(q_net, rm, "medium", n, cfg, device),
        "hard": rollout_eval(q_net, rm, "hard", n, cfg, device),
        "test": rollout_eval(q_net, rm, "easy", 1, cfg, device, fixed_test=True),
    }


def plot_q_history(history: List[Dict[str, float]], path: Path) -> None:
    episodes = [h["episode"] for h in history]
    panels = [
        ("Reward moving average", "reward"),
        ("Success moving average", "success"),
        ("Wall moving average", "wall"),
        ("BFS-gap moving average", "bfs_gap"),
        ("Repeat moving average", "repeat"),
    ]
    fig = plt.figure(figsize=(12, 10))
    for idx, (title, key) in enumerate(panels, start=1):
        plt.subplot(3, 2, idx)
        ys = moving_average([h[key] for h in history], 50)
        plt.plot(episodes, ys)
        plt.title(title)
        plt.xlabel("episode")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# -----------------------------
# Test mode
# -----------------------------

def load_q_model(path: str, device: torch.device) -> QNetwork:
    model = QNetwork().to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_rm_model(path: str, device: torch.device) -> TransitionRewardCNN:
    model = TransitionRewardCNN().to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# -----------------------------
# CLI
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v2.0.5 normalized margin preference reward model -> QCNN")
    p.add_argument("--mode", choices=["all", "train-rm", "train-q", "test"], default="all")
    p.add_argument("--output-dir", default="./v2.0.5")
    p.add_argument("--run-name", default="default")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")

    # Difficulty distribution defaults are the established v2.0 baseline.
    p.add_argument("--easy-ratio", type=float, default=0.4)
    p.add_argument("--medium-ratio", type=float, default=0.3)
    p.add_argument("--hard-ratio", type=float, default=0.3)
    p.add_argument("--max-steps", type=int, default=64)

    # Recommended reward-model defaults.
    p.add_argument("--rm-mazes", type=int, default=1000)
    p.add_argument("--trajectories-per-maze", type=int, default=16)
    p.add_argument("--pairs-per-maze", type=int, default=48)
    p.add_argument("--noisy-bfs-per-maze", type=int, default=3)
    p.add_argument("--random-per-maze", type=int, default=3)
    p.add_argument("--safe-loop-per-maze", type=int, default=1)
    p.add_argument("--sticky-wall-per-maze", type=int, default=2)
    p.add_argument("--sticky-wall-min-hits", type=int, default=8)
    p.add_argument("--sticky-wall-max-hits", type=int, default=64)
    p.add_argument("--sticky-wall-prefix-ratio", type=float, default=0.4)
    p.add_argument("--use-cpl", action="store_true", default=True)
    p.add_argument("--no-cpl", dest="use_cpl", action="store_false")
    p.add_argument("--cpl-pairs-per-maze", type=int, default=4)
    p.add_argument("--rm-epochs", type=int, default=30)
    p.add_argument("--rm-batch-size", type=int, default=16)
    p.add_argument("--rm-lr", type=float, default=1e-4)
    p.add_argument("--rm-grad-clip", type=float, default=5.0)
    p.add_argument("--rm-val-ratio", type=float, default=0.15)
    p.add_argument("--score-normalizer", choices=["raw_sum", "bfs_len", "episode_len"], default="bfs_len")
    p.add_argument("--use-margin", action="store_true", default=True)
    p.add_argument("--no-margin", dest="use_margin", action="store_false")
    p.add_argument("--margin-max", type=float, default=2.0)
    p.add_argument("--margin-min", type=float, default=0.05)
    p.add_argument("--margin-q-low", type=float, default=0.10)
    p.add_argument("--margin-q-high", type=float, default=0.90)
    p.add_argument("--wall-presence-margin-floor", type=float, default=1.25)
    p.add_argument("--badness-timeout-base", type=float, default=1.0)
    p.add_argument("--badness-wall-presence-weight", type=float, default=1.0)
    p.add_argument("--badness-wall-count-weight", type=float, default=1.0)
    p.add_argument("--badness-repeat-weight", type=float, default=0.5)
    p.add_argument("--reward-l2", type=float, default=0.01)

    # Stage control.
    p.add_argument(
        "--skip-rm",
        action="store_true",
        help="Skip stage A (reward-model training). Use --reward-model or an existing "
        "v2.0.5_reward_model_best.pt under the run output dir for stage B (QCNN).",
    )

    # Recommended QCNN defaults.
    p.add_argument("--reward-model", default="")
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--replay-size", type=int, default=100000)
    p.add_argument("--warmup-steps", type=int, default=3000)
    p.add_argument("--updates-per-episode", type=int, default=64)
    p.add_argument("--target-update-interval", type=int, default=500)
    p.add_argument("--epsilon-decay-episodes", type=int, default=2500)
    p.add_argument("--rm-scale", type=float, default=2.0)
    p.add_argument("--auto-rm-scale", action="store_true", default=True)
    p.add_argument("--no-auto-rm-scale", dest="auto_rm_scale", action="store_false")
    p.add_argument("--target-rm-std", type=float, default=1.0)
    p.add_argument("--eval-n", type=int, default=100)

    # Diagnostics are on by default in this debug version.
    p.add_argument("--debug-probe-mazes", type=int, default=30)
    p.add_argument("--no-debug", action="store_true")

    # Debug-only anchors, default off by design.
    p.add_argument("--anchor-terminal", action="store_true")
    p.add_argument("--anchor-goal", type=float, default=64.0)
    p.add_argument("--anchor-wall", type=float, default=-32.0)

    # Test mode. --test is kept as a compatibility alias for --mode test.
    p.add_argument("--model", default="")
    p.add_argument("--test", action="store_true")
    return p




def format_value_for_name(v: object) -> str:
    if isinstance(v, bool):
        return "on" if v else "off"
    if isinstance(v, float):
        return (f"{v:g}").replace(".", "p").replace("-", "m")
    return str(v).replace(".", "p").replace("-", "m").replace("/", "_")


def resolve_reward_model_path(args: argparse.Namespace, out_dir: Path) -> Path:
    """Pick frozen R checkpoint for stage B: explicit path > best > last epoch."""
    if args.reward_model:
        path = Path(args.reward_model)
        if not path.exists():
            raise FileNotFoundError(f"reward model not found: {path}")
        return path

    best_path = out_dir / "v2.0.5_reward_model_best.pt"
    last_path = out_dir / "v2.0.5_reward_model.pt"
    if best_path.exists():
        return best_path
    if last_path.exists():
        return last_path
    raise FileNotFoundError(
        "no reward model found. Pass --reward-model, or run stage A first so that "
        f"{best_path.name} (or {last_path.name}) exists under {out_dir}"
    )


def build_run_name(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    if getattr(args, "run_name", ""):
        return args.run_name
    tracked = [
        "rm_mazes", "trajectories_per_maze", "pairs_per_maze", "rm_epochs",
        "episodes", "easy_ratio", "medium_ratio", "hard_ratio",
        "score_normalizer", "use_margin", "margin_max", "margin_min",
        "sticky_wall_per_maze", "use_cpl", "auto_rm_scale", "target_rm_std",
        "badness_timeout_base", "badness_wall_presence_weight", "badness_wall_count_weight",
        "badness_repeat_weight", "reward_l2",
    ]
    defaults = {a.dest: a.default for a in parser._actions if a.dest != "help"}
    parts = []
    for key in tracked:
        if not hasattr(args, key):
            continue
        val = getattr(args, key)
        default = defaults.get(key, None)
        if val != default:
            parts.append(f"{key}-{format_value_for_name(val)}")
    return "default" if not parts else "__".join(parts)


# ============================================================
# v2.0.5 distribution/count-map override block
# This block intentionally overrides several earlier definitions while keeping
# the user's v2.0.5 file structure, best-RM saving, and --skip-rm stage control.
# ============================================================

class MazeEnv:
    def __init__(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], max_steps: int = MAX_STEPS):
        self.grid = grid.copy().astype(np.int8)
        self.start = start
        self.goal = goal
        self.max_steps = max_steps
        self.dist = bfs_distances(self.grid, self.goal)
        self.reset()

    def reset(self) -> np.ndarray:
        self.pos = self.start
        self.steps = 0
        self.done = False
        self.visited = np.zeros((SIZE, SIZE), dtype=np.float32)
        self.visited[self.pos] = 1.0
        return self.state()

    def state(self) -> np.ndarray:
        wall = self.grid.astype(np.float32)
        agent = np.zeros((SIZE, SIZE), dtype=np.float32)
        goal = np.zeros((SIZE, SIZE), dtype=np.float32)
        agent[self.pos] = 1.0
        goal[self.goal] = 1.0
        count_map = np.log1p(self.visited.astype(np.float32)) / max(1e-6, math.log1p(float(self.max_steps)))
        count_map = np.clip(count_map, 0.0, 1.0)
        return np.stack([wall, agent, goal, count_map], axis=0)

    def step_raw(self, action: int) -> Tuple[np.ndarray, Dict[str, object]]:
        if self.done:
            raise RuntimeError("step after done")
        before = self.pos
        dr, dc = ACTIONS[action]
        nxt = (before[0] + dr, before[1] + dc)
        hit_wall = (not in_bounds(nxt)) or (not is_free(self.grid, nxt))
        success = False
        timeout = False
        if not hit_wall:
            self.pos = nxt
            success = self.pos == self.goal
        # Count-map state: every time step increases the count of the cell the
        # agent actually occupies. Therefore staying at a wall also increases
        # the current cell's count and makes self-loop exploitation observable.
        self.visited[self.pos] += 1.0
        self.steps += 1
        if success:
            self.done = True
            outcome = "success"
        elif self.steps >= self.max_steps:
            self.done = True
            timeout = True
            outcome = "timeout"
        else:
            outcome = "running"
        next_state = self.state()
        info = {
            "before": before,
            "pos": self.pos,
            "hit_wall": hit_wall,
            "success": success,
            "timeout": timeout,
            "outcome": outcome,
            "steps": self.steps,
            "bfs_dist": int(self.dist[self.pos]) if self.dist[self.pos] < 10_000 else 10_000,
            "visit_count": int(self.visited[self.pos]),
        }
        return next_state, info


def record_step(env: MazeEnv, state: np.ndarray, action: int) -> Tuple[np.ndarray, RMTransition, Dict[str, object], bool]:
    old_counts = env.visited.copy()
    ns, info = env.step_raw(action)
    repeated = bool(old_counts[env.pos] > 0.0)
    return ns, RMTransition(state.copy(), action, ns.copy()), info, repeated


def valid_actions(env: MazeEnv, avoid_goal: bool = False, allow_visited: bool = True) -> List[int]:
    out: List[int] = []
    for a, (dr, dc) in enumerate(ACTIONS):
        nxt = (env.pos[0] + dr, env.pos[1] + dc)
        if not is_free(env.grid, nxt):
            continue
        if avoid_goal and nxt == env.goal:
            continue
        if (not allow_visited) and env.visited[nxt] > 0.0:
            continue
        out.append(a)
    return out


def actions_decreasing_bfs(env: MazeEnv, avoid_goal: bool = False) -> List[int]:
    cur_d = int(env.dist[env.pos]) if env.dist[env.pos] < 10_000 else 10_000
    out: List[int] = []
    for a, (dr, dc) in enumerate(ACTIONS):
        nxt = (env.pos[0] + dr, env.pos[1] + dc)
        if is_free(env.grid, nxt) and (not avoid_goal or nxt != env.goal) and int(env.dist[nxt]) < cur_d:
            out.append(a)
    return out


def actions_increasing_or_flat_bfs(env: MazeEnv, avoid_goal: bool = True) -> List[int]:
    cur_d = int(env.dist[env.pos]) if env.dist[env.pos] < 10_000 else 10_000
    out: List[int] = []
    for a, (dr, dc) in enumerate(ACTIONS):
        nxt = (env.pos[0] + dr, env.pos[1] + dc)
        if is_free(env.grid, nxt) and (not avoid_goal or nxt != env.goal) and int(env.dist[nxt]) >= cur_d:
            out.append(a)
    return out


def transition_visit_delta(tr: RMTransition) -> float:
    pos = argmax_pos(tr.next_state[1])
    before_count = float(tr.state[3, pos[0], pos[1]])
    after_count = float(tr.next_state[3, pos[0], pos[1]])
    return max(0.0, after_count - before_count)


def make_controlled_success_trajectory(
    env: MazeEnv,
    args: argparse.Namespace,
    noise_prob: float,
    prefix_ratio: float,
    target_wall_hits: int = 0,
    max_extra_ratio: float = 0.75,
) -> Trajectory:
    state = env.reset()
    path = bfs_path(env.grid, env.start, env.goal)
    if path is None:
        raise ValueError("trajectory generated on unsolvable maze")
    bfs_len = len(path) - 1
    start_bfs_dist = int(env.dist[env.start]) if env.dist[env.start] < 10_000 else bfs_len
    transitions: List[RMTransition] = []
    wall_hits = 0
    repeat_visits = 0
    prefix_steps = max(0, min(bfs_len - 1, int(prefix_ratio * max(1, bfs_len))))
    max_extra = max(0, int(max_extra_ratio * max(1, bfs_len)))
    detour_budget = max(1, max_extra)

    for _ in range(args.max_steps):
        remaining_needed = int(env.dist[env.pos]) if env.dist[env.pos] < 10_000 else args.max_steps
        remaining_slots = args.max_steps - env.steps
        force_bfs = remaining_slots <= remaining_needed + 1
        action: Optional[int] = None

        if env.steps < prefix_steps:
            action = best_bfs_action(env)
        elif (not force_bfs) and wall_hits < target_wall_hits and random.random() < args.success_wall_try_prob:
            walls = wall_actions_at(env)
            if walls:
                action = random.choice(walls)
        elif (not force_bfs) and detour_budget > 0 and random.random() < noise_prob:
            # Prefer a legal non-goal detour. This produces controlled long-success
            # trajectories without turning them into uncontrolled failures.
            candidates = actions_increasing_or_flat_bfs(env, avoid_goal=True)
            if not candidates:
                candidates = valid_actions(env, avoid_goal=True, allow_visited=True)
            if candidates:
                action = random.choice(candidates)
                detour_budget -= 1
        if action is None:
            action = best_bfs_action(env)
            if action is None:
                candidates = valid_actions(env, avoid_goal=False, allow_visited=True)
                action = random.choice(candidates) if candidates else random.randrange(4)

        ns, tr, info, repeated = record_step(env, state, int(action))
        transitions.append(tr)
        if info["hit_wall"]:
            wall_hits += 1
        if repeated:
            repeat_visits += 1
        state = ns
        if info["outcome"] != "running":
            outcome = str(info["outcome"])
            break
    else:
        outcome = "timeout"

    # The success half is intended to end in success. If a controlled-success
    # attempt fails, force a final BFS recovery only when there is room.
    while outcome != "success" and (not env.done) and env.steps < args.max_steps:
        a = best_bfs_action(env)
        if a is None:
            break
        ns, tr, info, repeated = record_step(env, state, int(a))
        transitions.append(tr)
        if info["hit_wall"]:
            wall_hits += 1
        if repeated:
            repeat_visits += 1
        state = ns
        if info["outcome"] != "running":
            outcome = str(info["outcome"])
            break

    final_dist = int(env.dist[env.pos]) if env.dist[env.pos] < 10_000 else 10_000
    return finalize_trajectory(
        transitions=transitions,
        outcome=outcome,
        bfs_len=bfs_len,
        start_bfs_dist=start_bfs_dist,
        final_bfs_dist=final_dist,
        wall_hits=wall_hits,
        repeat_count=repeat_visits,
        source="controlled_success" if outcome == "success" else "controlled_success_failed",
        args=args,
    )


def make_controlled_failure_trajectory(env: MazeEnv, args: argparse.Namespace, failure_kind: str) -> Trajectory:
    state = env.reset()
    path = bfs_path(env.grid, env.start, env.goal)
    if path is None:
        raise ValueError("trajectory generated on unsolvable maze")
    bfs_len = len(path) - 1
    start_bfs_dist = int(env.dist[env.start]) if env.dist[env.start] < 10_000 else bfs_len
    transitions: List[RMTransition] = []
    wall_hits = 0
    repeat_visits = 0
    prefix_len = max(0, min(bfs_len - 1, int(args.failure_prefix_ratio * max(1, bfs_len))))
    sticky_action: Optional[int] = None

    for _ in range(args.max_steps):
        action: Optional[int] = None
        if failure_kind in {"sticky_wall", "prefix_sticky_wall"} and env.steps < prefix_len:
            action = best_bfs_action(env)
        elif failure_kind in {"sticky_wall", "prefix_sticky_wall"}:
            if sticky_action is None:
                walls = wall_actions_at(env)
                if walls:
                    sticky_action = random.choice(walls)
                else:
                    # Move legally without entering goal until a wall is available.
                    candidates = valid_actions(env, avoid_goal=True, allow_visited=True)
                    action = random.choice(candidates) if candidates else random.randrange(4)
            if action is None:
                action = sticky_action if sticky_action is not None else random.randrange(4)
        elif failure_kind == "safe_loop":
            visited_legal = []
            legal = []
            for a, (dr, dc) in enumerate(ACTIONS):
                nxt = (env.pos[0] + dr, env.pos[1] + dc)
                if is_free(env.grid, nxt) and nxt != env.goal:
                    legal.append(a)
                    if env.visited[nxt] > 0.0:
                        visited_legal.append(a)
            action = random.choice(visited_legal or legal) if (visited_legal or legal) else random.choice(wall_actions_at(env) or [random.randrange(4)])
        elif failure_kind == "random_avoid_goal":
            if random.random() < args.failure_wall_prob:
                walls = wall_actions_at(env)
                if walls:
                    action = random.choice(walls)
            if action is None:
                candidates = valid_actions(env, avoid_goal=True, allow_visited=True)
                # Bias toward non-progress or repeated states but do not force only wall.
                nonprog = actions_increasing_or_flat_bfs(env, avoid_goal=True)
                repeated = []
                for a in candidates:
                    dr, dc = ACTIONS[a]
                    nxt = (env.pos[0] + dr, env.pos[1] + dc)
                    if env.visited[nxt] > 0.0:
                        repeated.append(a)
                bag = repeated or nonprog or candidates
                action = random.choice(bag) if bag else random.choice(wall_actions_at(env) or [random.randrange(4)])
        else:
            raise ValueError(f"unknown failure_kind {failure_kind}")

        ns, tr, info, repeated = record_step(env, state, int(action))
        transitions.append(tr)
        if info["hit_wall"]:
            wall_hits += 1
        if repeated:
            repeat_visits += 1
        state = ns
        # Failures are constructed to avoid goal. If a rare accidental success happens,
        # stop and let its outcome be recorded honestly; build_preference_dataset will
        # still maintain the target mix by retrying.
        if info["outcome"] != "running":
            outcome = str(info["outcome"])
            break
    else:
        outcome = "timeout"
    final_dist = int(env.dist[env.pos]) if env.dist[env.pos] < 10_000 else 10_000
    return finalize_trajectory(
        transitions=transitions,
        outcome=outcome,
        bfs_len=bfs_len,
        start_bfs_dist=start_bfs_dist,
        final_bfs_dist=final_dist,
        wall_hits=wall_hits,
        repeat_count=repeat_visits,
        source=failure_kind,
        args=args,
    )


def make_trajectory(env: MazeEnv, policy: str, args: argparse.Namespace, noise_prob: float = 0.2) -> Trajectory:
    if policy == "bfs":
        tr = make_controlled_success_trajectory(env, args, noise_prob=0.0, prefix_ratio=1.0, target_wall_hits=0, max_extra_ratio=0.0)
        tr.source = "bfs"
        return tr
    if policy == "noisy_bfs":
        tr = make_controlled_success_trajectory(env, args, noise_prob=noise_prob, prefix_ratio=args.success_shared_prefix_ratio, target_wall_hits=0, max_extra_ratio=args.success_max_extra_ratio)
        tr.source = "controlled_success" if tr.outcome == "success" else "controlled_success_failed"
        return tr
    if policy == "controlled_success_wall":
        tr = make_controlled_success_trajectory(env, args, noise_prob=noise_prob, prefix_ratio=args.success_shared_prefix_ratio, target_wall_hits=random.randint(1, args.success_max_wall_hits), max_extra_ratio=args.success_max_extra_ratio)
        tr.source = "controlled_success_wall" if tr.outcome == "success" else "controlled_success_failed"
        return tr
    if policy == "random":
        return make_controlled_failure_trajectory(env, args, "random_avoid_goal")
    if policy == "safe_loop":
        return make_controlled_failure_trajectory(env, args, "safe_loop")
    if policy == "sticky_wall":
        return make_controlled_failure_trajectory(env, args, "prefix_sticky_wall")
    if policy in {"prefix_sticky_wall", "random_avoid_goal"}:
        return make_controlled_failure_trajectory(env, args, policy)
    raise ValueError(f"unknown policy {policy}")


def make_cpl_pair(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], args: argparse.Namespace) -> Optional[PreferencePair]:
    path = bfs_path(grid, start, goal)
    if path is None or len(path) < 4:
        return None
    prefix_len = max(1, min(len(path) - 2, int(args.success_shared_prefix_ratio * (len(path) - 1))))

    def run_prefix(env: MazeEnv) -> np.ndarray:
        state = env.reset()
        for i in range(prefix_len):
            if env.pos == goal:
                break
            action = action_between(env.pos, path[i + 1]) if i + 1 < len(path) else (best_bfs_action(env) or 0)
            state, _tr, info, _rep = record_step(env, state, action)
            if info["outcome"] != "running":
                break
        return state

    env_pos = MazeEnv(grid, start, goal, max_steps=args.max_steps)
    state = run_prefix(env_pos)
    pos_trans: List[RMTransition] = []
    pos_wall = 0
    pos_rep = 0
    outcome = "timeout"
    while env_pos.steps < args.max_steps and env_pos.pos != goal:
        a = best_bfs_action(env_pos)
        if a is None:
            break
        ns, tr, info, rep = record_step(env_pos, state, a)
        pos_trans.append(tr)
        if info["hit_wall"]:
            pos_wall += 1
        if rep:
            pos_rep += 1
        state = ns
        if info["outcome"] != "running":
            outcome = str(info["outcome"])
            break
    if env_pos.pos == goal:
        outcome = "success"
    bfs_len = len(path) - 1
    start_dist = int(env_pos.dist[start])
    pos_final = int(env_pos.dist[env_pos.pos]) if env_pos.dist[env_pos.pos] < 10_000 else 10_000
    pos = finalize_trajectory(pos_trans, outcome, bfs_len, start_dist, pos_final, pos_wall, pos_rep, "cpl_forward", args)

    env_neg = MazeEnv(grid, start, goal, max_steps=args.max_steps)
    state = run_prefix(env_neg)
    neg_trans: List[RMTransition] = []
    neg_wall = 0
    neg_rep = 0
    walls = wall_actions_at(env_neg)
    if not walls:
        return None
    wall_a = random.choice(walls)
    for _ in range(args.max_steps - env_neg.steps):
        ns, tr, info, rep = record_step(env_neg, state, wall_a)
        neg_trans.append(tr)
        if info["hit_wall"]:
            neg_wall += 1
        if rep:
            neg_rep += 1
        state = ns
        if info["outcome"] != "running":
            break
    neg_final = int(env_neg.dist[env_neg.pos]) if env_neg.dist[env_neg.pos] < 10_000 else 10_000
    neg = finalize_trajectory(neg_trans, "timeout", bfs_len, start_dist, neg_final, neg_wall, neg_rep, "cpl_sticky_wall", args)
    if pos.quality_key <= neg.quality_key:
        return None
    return PreferencePair(pos, neg, max(0.0, neg.badness - pos.badness), 0.0, "cpl")


def infer_transition_features(tr: RMTransition) -> Dict[str, object]:
    s = tr.state
    ns = tr.next_state
    grid = s[0]
    pos = argmax_pos(s[1])
    next_pos = argmax_pos(ns[1])
    goal = argmax_pos(s[2])
    dr, dc = ACTIONS[tr.action]
    target = (pos[0] + dr, pos[1] + dc)
    hit_wall = (not in_bounds(target)) or (grid[target] > 0.5)
    success = next_pos == goal
    repeat = (not hit_wall) and (s[3, next_pos[0], next_pos[1]] > 0.0)
    dist = bfs_distances(grid.astype(np.int8), goal)
    d0 = int(dist[pos]) if dist[pos] < 10_000 else 10_000
    d1 = int(dist[next_pos]) if dist[next_pos] < 10_000 else 10_000
    delta_d = d0 - d1
    visit_before = float(s[3, next_pos[0], next_pos[1]])
    visit_after = float(ns[3, next_pos[0], next_pos[1]])
    if success:
        category = "goal"
    elif hit_wall:
        category = "wall"
    elif repeat and delta_d > 0:
        category = "recovery_repeat"
    elif repeat:
        category = "nonprogress_repeat"
    elif delta_d > 0:
        category = "toward_goal"
    elif delta_d < 0:
        category = "away_goal"
    else:
        category = "flat"
    return {
        "pos": pos,
        "next_pos": next_pos,
        "goal": goal,
        "action": tr.action,
        "hit_wall": hit_wall,
        "success": success,
        "repeat": repeat,
        "visit_before": visit_before,
        "visit_after": visit_after,
        "visit_delta": max(0.0, visit_after - visit_before),
        "d0": d0,
        "d1": d1,
        "delta_d": delta_d,
        "category": category,
    }


def simulate_transition_from_state(state: np.ndarray, action: int) -> RMTransition:
    grid = state[0].astype(np.int8)
    pos = argmax_pos(state[1])
    goal = argmax_pos(state[2])
    dr, dc = ACTIONS[action]
    target = (pos[0] + dr, pos[1] + dc)
    hit_wall = (not in_bounds(target)) or (not is_free(grid, target))
    next_pos = pos if hit_wall else target
    next_state = state.copy()
    next_state[1, :, :] = 0.0
    next_state[1, next_pos[0], next_pos[1]] = 1.0
    old_norm = float(next_state[3, next_pos[0], next_pos[1]])
    # Reconstruct approximately in normalized count space by adding one raw step
    # represented as a small monotone increase. This is only for diagnostics.
    next_state[3, next_pos[0], next_pos[1]] = min(1.0, old_norm + 1.0 / max(1.0, float(MAX_STEPS)))
    next_state[2, :, :] = 0.0
    next_state[2, goal[0], goal[1]] = 1.0
    return RMTransition(state.copy(), action, next_state.copy())


def build_preference_dataset(args: argparse.Namespace) -> Tuple[List[PreferencePair], Dict[str, int], Dict[str, float]]:
    pairs: List[PreferencePair] = []
    stats: Dict[str, int] = {
        "mazes": 0,
        "trajectories": 0,
        "pairs": 0,
        "success": 0,
        "timeout": 0,
        "bfs": 0,
        "controlled_success": 0,
        "controlled_success_wall": 0,
        "controlled_success_failed": 0,
        "random_avoid_goal": 0,
        "safe_loop": 0,
        "prefix_sticky_wall": 0,
        "cpl_forward": 0,
        "cpl_sticky_wall": 0,
        "skipped_ties": 0,
        "cpl_pairs": 0,
    }
    weights = {"easy": args.easy_ratio, "medium": args.medium_ratio, "hard": args.hard_ratio}
    success_target = max(1, int(round(args.trajectories_per_maze * args.success_ratio)))
    failure_target = max(1, args.trajectories_per_maze - success_target)
    for _ in range(args.rm_mazes):
        difficulty = weighted_choice(weights)
        grid, start, goal = generate_maze(difficulty)
        trajs: List[Trajectory] = []
        env = MazeEnv(grid, start, goal, max_steps=args.max_steps)
        bfs_traj = make_trajectory(env, "bfs", args)
        trajs.append(bfs_traj)

        success_noise_grid = np.linspace(args.success_noise_start, args.success_noise_end, max(1, success_target - 1)).tolist()
        for idx, noise in enumerate(success_noise_grid):
            made: Optional[Trajectory] = None
            for _retry in range(args.success_retry):
                env_s = MazeEnv(grid, start, goal, max_steps=args.max_steps)
                policy = "controlled_success_wall" if random.random() < args.success_wall_fraction else "noisy_bfs"
                cand = make_trajectory(env_s, policy, args, noise_prob=float(noise))
                if cand.outcome == "success":
                    made = cand
                    break
            if made is None:
                made = make_trajectory(MazeEnv(grid, start, goal, max_steps=args.max_steps), "bfs", args)
            trajs.append(made)

        failure_kinds = ["prefix_sticky_wall", "safe_loop", "random_avoid_goal"]
        for idx in range(failure_target):
            kind = failure_kinds[idx % len(failure_kinds)]
            made = None
            for _retry in range(args.failure_retry):
                env_f = MazeEnv(grid, start, goal, max_steps=args.max_steps)
                cand = make_controlled_failure_trajectory(env_f, args, kind)
                if cand.outcome != "success":
                    made = cand
                    break
            if made is None:
                made = make_controlled_failure_trajectory(MazeEnv(grid, start, goal, max_steps=args.max_steps), args, "prefix_sticky_wall")
            trajs.append(made)

        stats["mazes"] += 1
        stats["trajectories"] += len(trajs)
        for tr in trajs:
            stats[tr.outcome] = stats.get(tr.outcome, 0) + 1
            stats[tr.source] = stats.get(tr.source, 0) + 1

        candidate_pairs: List[PreferencePair] = []
        for i in range(len(trajs)):
            for j in range(i + 1, len(trajs)):
                a, b = trajs[i], trajs[j]
                if a.quality_key == b.quality_key:
                    stats["skipped_ties"] += 1
                    continue
                pos, neg = (a, b) if a.quality_key > b.quality_key else (b, a)
                src = "same_maze_ranking"
                # Most trajectories intentionally share the initial BFS prefix; this
                # label helps later diagnostics distinguish these pairs.
                if (pos.source in {"bfs", "controlled_success", "controlled_success_wall"} and neg.source in {"controlled_success", "controlled_success_wall", "prefix_sticky_wall"}):
                    src = "shared_prefix_ranking"
                candidate_pairs.append(PreferencePair(pos, neg, max(0.0, neg.badness - pos.badness), 0.0, src))
        random.shuffle(candidate_pairs)
        pairs.extend(candidate_pairs[:args.pairs_per_maze])
        stats["pairs"] += min(len(candidate_pairs), args.pairs_per_maze)

        if args.use_cpl:
            for _k in range(args.cpl_pairs_per_maze):
                pr = make_cpl_pair(grid, start, goal, args)
                if pr is not None:
                    pairs.append(pr)
                    stats["pairs"] += 1
                    stats["cpl_pairs"] += 1

    assign_adaptive_margins(pairs, args)
    margins = [p.margin for p in pairs]
    deltas = [p.delta_badness for p in pairs]
    margin_stats = {
        "delta_badness_mean": float(np.mean(deltas)) if deltas else 0.0,
        "delta_badness_p10": float(np.percentile(deltas, 10)) if deltas else 0.0,
        "delta_badness_p50": float(np.percentile(deltas, 50)) if deltas else 0.0,
        "delta_badness_p90": float(np.percentile(deltas, 90)) if deltas else 0.0,
        "margin_mean": float(np.mean(margins)) if margins else 0.0,
        "margin_p10": float(np.percentile(margins, 10)) if margins else 0.0,
        "margin_p50": float(np.percentile(margins, 50)) if margins else 0.0,
        "margin_p90": float(np.percentile(margins, 90)) if margins else 0.0,
    }
    return pairs, stats, margin_stats


_old_build_arg_parser = build_arg_parser

def build_arg_parser() -> argparse.ArgumentParser:
    p = _old_build_arg_parser()
    # Distribution/count-map controls. Defaults implement the requested 0.5/0.5
    # success/failure RM dataset while keeping v2.0.5's stage control and output logic.
    p.add_argument("--success-ratio", type=float, default=0.5)
    p.add_argument("--success-noise-start", type=float, default=0.05)
    p.add_argument("--success-noise-end", type=float, default=0.55)
    p.add_argument("--success-shared-prefix-ratio", type=float, default=0.45)
    p.add_argument("--success-max-extra-ratio", type=float, default=1.25)
    p.add_argument("--success-wall-fraction", type=float, default=0.25)
    p.add_argument("--success-wall-try-prob", type=float, default=0.20)
    p.add_argument("--success-max-wall-hits", type=int, default=4)
    p.add_argument("--success-retry", type=int, default=6)
    p.add_argument("--failure-retry", type=int, default=4)
    p.add_argument("--failure-prefix-ratio", type=float, default=0.45)
    p.add_argument("--failure-wall-prob", type=float, default=0.35)
    return p


def build_run_name(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    if getattr(args, "run_name", ""):
        return args.run_name
    tracked = [
        "rm_mazes", "trajectories_per_maze", "pairs_per_maze", "rm_epochs",
        "episodes", "easy_ratio", "medium_ratio", "hard_ratio",
        "success_ratio", "success_noise_start", "success_noise_end", "success_shared_prefix_ratio",
        "success_wall_fraction", "failure_wall_prob", "score_normalizer", "use_margin",
        "margin_max", "margin_min", "use_cpl", "auto_rm_scale", "target_rm_std",
        "badness_timeout_base", "badness_wall_presence_weight", "badness_wall_count_weight",
        "badness_repeat_weight", "reward_l2",
    ]
    defaults = {a.dest: a.default for a in parser._actions if a.dest != "help"}
    parts = []
    for key in tracked:
        if not hasattr(args, key):
            continue
        val = getattr(args, key)
        default = defaults.get(key, None)
        if val != default:
            parts.append(f"{key}-{format_value_for_name(val)}")
    return "default" if not parts else "__".join(parts)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if getattr(args, "test", False):
        args.mode = "test"
    set_seed(args.seed)
    device = get_device(args.device)
    run_name = build_run_name(args, parser)
    out_dir = ensure_dir(Path(args.output_dir) / run_name)
    args.run_name = run_name
    print(f"=== {VERSION} ===")
    print(f"Device: {device}")
    print(f"Output dir: {out_dir}")
    print("Default design: normalized-margin offline preference reward model, then frozen-RM QCNN.")
    print("No demo, no BFS margin, no action mask, no handcrafted step/repeat reward in main QCNN reward.")

    reward_model_path: Optional[Path] = None
    if args.mode in ["all", "train-rm"]:
        if args.skip_rm:
            if args.mode == "train-rm":
                raise ValueError("--skip-rm cannot be used with --mode train-rm")
            reward_model_path = resolve_reward_model_path(args, out_dir)
            print(f"\n[Skip RM] Stage A skipped. Using frozen reward model: {reward_model_path}")
        else:
            reward_model_path = train_reward_model(args, device, out_dir)

    if args.mode in ["all", "train-q"]:
        if reward_model_path is None:
            reward_model_path = resolve_reward_model_path(args, out_dir)
            print(f"[Load RM] Using reward model for QCNN: {reward_model_path}")
        train_qcnn(args, device, out_dir, reward_model_path)

    if args.mode == "test":
        if not args.model:
            args.model = str(out_dir / "v2.0.5_qcnn_from_preference_reward.pt")
        if not args.reward_model:
            args.reward_model = str(resolve_reward_model_path(args, out_dir))
        if not Path(args.model).exists() or not Path(args.reward_model).exists():
            raise ValueError("test mode needs --model and --reward-model, or the default files in --output-dir")
        q = load_q_model(args.model, device)
        rm = load_rm_model(args.reward_model, device)
        cfg = DQNConfig(episodes=args.episodes, eval_n=args.eval_n, rm_scale=args.rm_scale, anchor_terminal=args.anchor_terminal)
        metrics = rollout_eval(q, rm, "easy", 1, cfg, device, fixed_test=True)
        print("=== Fixed canonical test maze ===")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
