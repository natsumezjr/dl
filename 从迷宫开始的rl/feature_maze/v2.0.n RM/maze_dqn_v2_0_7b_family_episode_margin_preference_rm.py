"""
maze_dqn_v2_0_7b_family_episode_margin_preference_rm.py

v2.0.7b: Family + Episode-normalized Bucket Margin Preference Reward Model -> QCNN

Core changes from the v2.0.6 long-training version:
1. Data unit is a trajectory family: same maze, same prefix, same branch point,
   multiple controlled branches.
2. Every generated trajectory is validated. No generator may silently fall back to BFS.
3. Visit statistics are explicit fields: revisit/visit2/visit3+/visit4+/maxVisit/penalty.
4. Pair count is intentionally small and structured: about 40 pairs per maze.
5. Margin is bucket-specific and bounded with a tanh function; no global quantile clip.
6. Dataset summaries, badness reports, pair bucket reports, margin saturation checks,
   RM probes, QCNN best checkpoints, and test GIF are built in.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict, deque, namedtuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **kwargs: x

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

# Small grid experiments are often faster and more stable with one CPU thread.
torch.set_num_threads(1)

VERSION = "v2.0.7b_family_episode_margin_preference_rm"
SIZE = 8
DEFAULT_MAX_STEPS = 64
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]
DQNTransition = namedtuple("DQNTransition", ["state", "action", "reward", "next_state", "done"])

TEST_MAZE_TEXT = """
S.......
.######.
.....#..
####.#.#
.....#..
.######.
.#...#..
...#...G
"""

# -----------------------------
# Dataclasses
# -----------------------------

@dataclass
class RMTransition:
    state: np.ndarray
    action: int
    next_state: np.ndarray
    is_wall: bool
    is_goal: bool
    pos_before: Tuple[int, int]
    pos_after: Tuple[int, int]
    bfs_before: int
    bfs_after: int
    delta_bfs: int
    visit_count_after: int


@dataclass
class Trajectory:
    maze_id: int
    family_id: int
    source: str
    transitions: List[RMTransition]
    outcome: str
    success: bool
    steps: int
    bfs_len: int
    start_bfs_dist: int
    final_bfs_dist: int
    gap: int
    wall_hits: int
    wall_rate: float
    revisit_steps: int
    visit2_steps: int
    visit3plus_steps: int
    visit4plus_steps: int
    max_visit: int
    visit_penalty_sum: float
    visit_penalty_norm: float
    quality_key: Tuple[float, float, float, float, float, float]
    b_outcome: float
    b_wall_presence: float
    b_wall_count: float
    b_visit: float
    b_path: float
    b_total: float
    prefix_actions: List[int] = field(default_factory=list)
    branch_pos: Tuple[int, int] = (0, 0)


@dataclass
class PreferencePair:
    pos: Trajectory
    neg: Trajectory
    bucket: str
    margin: float
    delta_component: float
    pair_source: str


@dataclass
class Family:
    maze_id: int
    family_id: int
    grid: np.ndarray
    start: Tuple[int, int]
    goal: Tuple[int, int]
    prefix_actions: List[int]
    branch_pos: Tuple[int, int]
    trajectories: List[Trajectory]


@dataclass
class EvalMetrics:
    success: float
    explore_timeout: float
    wall_timeout: float
    reward: float
    steps: float
    wall_hits: float
    wall_step_rate: float
    visit2_steps: float
    visit3plus_steps: float
    visit4plus_steps: float
    max_visit: float
    bfs_agree: float
    bfs_gap: float

# -----------------------------
# Basic utilities
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


def moving_average(values: Sequence[float], window: int = 50) -> List[float]:
    out: List[float] = []
    buf: deque = deque(maxlen=window)
    for v in values:
        buf.append(float(v))
        out.append(float(np.mean(buf)))
    return out


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def weighted_choice(weight_map: Dict[str, float]) -> str:
    keys = list(weight_map.keys())
    weights = np.array([max(0.0, float(weight_map[k])) for k in keys], dtype=np.float64)
    if weights.sum() <= 0:
        return random.choice(keys)
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


def apply_action(pos: Tuple[int, int], action: int) -> Tuple[int, int]:
    dr, dc = ACTIONS[action]
    return (pos[0] + dr, pos[1] + dc)


def opposite_action(action: int) -> int:
    return {0: 1, 1: 0, 2: 3, 3: 2}[action]

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


def shortest_action_from_dist(grid: np.ndarray, pos: Tuple[int, int], dist: np.ndarray) -> Optional[int]:
    best: List[int] = []
    best_d = int(dist[pos])
    for a in range(4):
        np0 = apply_action(pos, a)
        if is_free(grid, np0) and int(dist[np0]) < best_d:
            best_d = int(dist[np0])
            best = [a]
        elif is_free(grid, np0) and int(dist[np0]) == best_d:
            best.append(a)
    return random.choice(best) if best else None


def shortest_actions_from_pos(grid: np.ndarray, pos: Tuple[int, int], goal: Tuple[int, int]) -> List[int]:
    dist = bfs_distances(grid, goal)
    if int(dist[pos]) >= 10_000:
        return []
    actions: List[int] = []
    cur = pos
    guard = 0
    while cur != goal and guard < SIZE * SIZE:
        a = shortest_action_from_dist(grid, cur, dist)
        if a is None:
            break
        actions.append(a)
        cur = apply_action(cur, a)
        guard += 1
    return actions


def enumerate_shortest_action_paths(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    max_paths: int = 8,
) -> List[List[int]]:
    dist = bfs_distances(grid, goal)
    if int(dist[start]) >= 10_000:
        return []
    out: List[List[int]] = []

    def rec(pos: Tuple[int, int], acc: List[int]) -> None:
        if len(out) >= max_paths:
            return
        if pos == goal:
            out.append(list(acc))
            return
        cur_d = int(dist[pos])
        candidates = []
        for a in range(4):
            nxt = apply_action(pos, a)
            if is_free(grid, nxt) and int(dist[nxt]) == cur_d - 1:
                candidates.append(a)
        random.shuffle(candidates)
        for a in candidates:
            acc.append(a)
            rec(apply_action(pos, a), acc)
            acc.pop()
            if len(out) >= max_paths:
                return

    rec(start, [])
    return out


def generate_maze(difficulty: str) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    # Wall probabilities chosen to keep enough valid 8x8 mazes while giving varied layouts.
    p_map = {"easy": 0.10, "medium": 0.20, "hard": 0.30}
    p = p_map.get(difficulty, 0.20)
    for _ in range(500):
        grid = (np.random.rand(SIZE, SIZE) < p).astype(np.int8)
        free = list(zip(*np.where(grid == 0)))
        if len(free) < 10:
            continue
        start = random.choice(free)
        goal = random.choice(free)
        if start == goal:
            continue
        dist = bfs_distances(grid, goal)
        d = int(dist[start])
        if d >= 10_000 or d < 4:
            continue
        if difficulty == "easy" and d > 14:
            continue
        if difficulty == "medium" and not (8 <= d <= 24):
            continue
        if difficulty == "hard" and d < 12:
            continue
        return grid, start, goal
    # Fallback open maze.
    grid = np.zeros((SIZE, SIZE), dtype=np.int8)
    return grid, (0, 0), (SIZE - 1, SIZE - 1)


def sample_difficulty(args: argparse.Namespace) -> str:
    return weighted_choice({"easy": args.easy_ratio, "medium": args.medium_ratio, "hard": args.hard_ratio})

# -----------------------------
# Visit map and trajectory execution
# -----------------------------

def visit_count_penalty(c: int, args: argparse.Namespace) -> float:
    if c <= 1:
        return 0.0
    if c == 2:
        return float(args.visit_second_penalty)
    val = float(args.visit_second_penalty) + float(args.visit_exp_scale) * (float(args.visit_exp_base) ** (c - 2) - 1.0)
    return float(min(args.visit_penalty_cap, val))


def encode_state(
    grid: np.ndarray,
    pos: Tuple[int, int],
    goal: Tuple[int, int],
    visit_counts: np.ndarray,
    visit_count_cap: float,
) -> np.ndarray:
    s = np.zeros((4, SIZE, SIZE), dtype=np.float32)
    s[0] = grid.astype(np.float32)
    s[1, pos[0], pos[1]] = 1.0
    s[2, goal[0], goal[1]] = 1.0
    s[3] = np.minimum(visit_counts.astype(np.float32), float(visit_count_cap)) / float(visit_count_cap)
    return s


def transition_input(state: np.ndarray, action: int, next_state: np.ndarray) -> np.ndarray:
    a_planes = np.zeros((4, SIZE, SIZE), dtype=np.float32)
    a_planes[action, :, :] = 1.0
    return np.concatenate([state, next_state, a_planes], axis=0)


def action_path_end(grid: np.ndarray, start: Tuple[int, int], actions: Sequence[int]) -> Tuple[int, int]:
    cur = start
    for a in actions:
        nxt = apply_action(cur, a)
        if is_free(grid, nxt):
            cur = nxt
    return cur


def run_actions_as_trajectory(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    actions: Sequence[int],
    source: str,
    args: argparse.Namespace,
    maze_id: int,
    family_id: int,
    prefix_actions: Optional[List[int]] = None,
    branch_pos: Optional[Tuple[int, int]] = None,
) -> Trajectory:
    dist = bfs_distances(grid, goal)
    bfs_len = int(dist[start]) if int(dist[start]) < 10_000 else args.max_steps
    visit_counts = np.zeros((SIZE, SIZE), dtype=np.int32)
    pos = start
    visit_counts[pos] = 1
    transitions: List[RMTransition] = []
    wall_hits = 0
    revisit_steps = 0
    visit2_steps = 0
    visit3plus_steps = 0
    visit4plus_steps = 0
    visit_penalty_sum = 0.0
    success = False
    steps = 0

    for a in list(actions)[: args.max_steps]:
        state = encode_state(grid, pos, goal, visit_counts, args.visit_count_cap)
        before = pos
        bfs_before = int(dist[before]) if int(dist[before]) < 10_000 else args.max_steps
        attempted = apply_action(pos, a)
        is_wall = not is_free(grid, attempted)
        if is_wall:
            nxt = pos
            wall_hits += 1
        else:
            nxt = attempted
        visit_counts[nxt] += 1
        c_after = int(visit_counts[nxt])
        if c_after >= 2:
            revisit_steps += 1
        if c_after == 2:
            visit2_steps += 1
        if c_after >= 3:
            visit3plus_steps += 1
        if c_after >= 4:
            visit4plus_steps += 1
        visit_penalty_sum += visit_count_penalty(c_after, args)
        next_state = encode_state(grid, nxt, goal, visit_counts, args.visit_count_cap)
        bfs_after = int(dist[nxt]) if int(dist[nxt]) < 10_000 else args.max_steps
        is_goal = nxt == goal
        transitions.append(
            RMTransition(
                state=state,
                action=int(a),
                next_state=next_state,
                is_wall=bool(is_wall),
                is_goal=bool(is_goal),
                pos_before=before,
                pos_after=nxt,
                bfs_before=bfs_before,
                bfs_after=bfs_after,
                delta_bfs=bfs_before - bfs_after,
                visit_count_after=c_after,
            )
        )
        pos = nxt
        steps += 1
        if is_goal:
            success = True
            break
    # If action list ended early and not success, hold position with a valid/invalid no-op-like wall choice until max_steps.
    # Generators normally provide max_steps for timeouts; this fallback only standardizes rare short failures.
    while not success and steps < args.max_steps and source.endswith("timeout"):
        # Choose a wall action if possible for wall timeout; otherwise choose a valid back-and-forth action.
        cand = None
        if "wall" in source:
            for a in range(4):
                if not is_free(grid, apply_action(pos, a)):
                    cand = a
                    break
        if cand is None:
            valid = [a for a in range(4) if is_free(grid, apply_action(pos, a))]
            cand = random.choice(valid) if valid else 0
        # Execute one filler action.
        filler = run_actions_as_trajectory(grid, pos, goal, [cand], source + "_filler", args, maze_id, family_id)
        # Manually merge one step while keeping visit_counts would be complex; avoid recursive filler by breaking.
        break

    outcome = "success" if success else "timeout"
    final_bfs_dist = 0 if success else (int(dist[pos]) if int(dist[pos]) < 10_000 else args.max_steps)
    gap = max(0, steps - bfs_len) if success else args.max_steps
    max_visit = int(visit_counts.max())
    wall_rate = wall_hits / max(1, steps)
    visit_penalty_norm = visit_penalty_sum / max(1e-9, float(args.visit_penalty_cap) * float(args.max_steps))
    traj = build_trajectory_stats(
        maze_id=maze_id,
        family_id=family_id,
        source=source,
        transitions=transitions,
        outcome=outcome,
        success=success,
        steps=steps,
        bfs_len=bfs_len,
        start_bfs_dist=int(dist[start]) if int(dist[start]) < 10_000 else args.max_steps,
        final_bfs_dist=final_bfs_dist,
        gap=gap,
        wall_hits=wall_hits,
        wall_rate=wall_rate,
        revisit_steps=revisit_steps,
        visit2_steps=visit2_steps,
        visit3plus_steps=visit3plus_steps,
        visit4plus_steps=visit4plus_steps,
        max_visit=max_visit,
        visit_penalty_sum=visit_penalty_sum,
        visit_penalty_norm=visit_penalty_norm,
        prefix_actions=list(prefix_actions or []),
        branch_pos=branch_pos if branch_pos is not None else action_path_end(grid, start, prefix_actions or []),
        args=args,
    )
    return traj


def visit_tier(traj_or_visit3: Any, args: argparse.Namespace) -> int:
    v = int(traj_or_visit3.visit3plus_steps if hasattr(traj_or_visit3, "visit3plus_steps") else traj_or_visit3)
    if v == 0:
        return 0
    if v <= args.visit_tier_mid_threshold:
        return 1
    return 2


def build_trajectory_stats(
    maze_id: int,
    family_id: int,
    source: str,
    transitions: List[RMTransition],
    outcome: str,
    success: bool,
    steps: int,
    bfs_len: int,
    start_bfs_dist: int,
    final_bfs_dist: int,
    gap: int,
    wall_hits: int,
    wall_rate: float,
    revisit_steps: int,
    visit2_steps: int,
    visit3plus_steps: int,
    visit4plus_steps: int,
    max_visit: int,
    visit_penalty_sum: float,
    visit_penalty_norm: float,
    prefix_actions: List[int],
    branch_pos: Tuple[int, int],
    args: argparse.Namespace,
) -> Trajectory:
    outcome_rank = 2 if success else 1
    has_wall = 1 if wall_hits > 0 else 0
    vt = visit_tier(visit3plus_steps, args)
    if success:
        denom = max(1e-9, float(args.max_steps - bfs_len))
        path_quality = -float(max(0, steps - bfs_len)) / denom
        b_path = float(max(0, steps - bfs_len)) / denom
    else:
        path_quality = -float(final_bfs_dist) / max(1e-9, float(start_bfs_dist))
        b_path = float(final_bfs_dist) / max(1e-9, float(start_bfs_dist))
    b_outcome = 0.0 if success else 1.0
    b_wall_presence = 1.0 if wall_hits > 0 else 0.0
    b_wall_count = float(wall_hits) / max(1, args.max_steps)
    b_visit = float(visit_penalty_norm)
    b_total = b_outcome + b_wall_presence + b_wall_count + b_visit + 0.5 * b_path
    quality_key = (
        float(outcome_rank),
        -float(has_wall),
        -float(vt),
        float(path_quality),
        -float(wall_rate),
        -float(visit_penalty_norm),
    )
    return Trajectory(
        maze_id=maze_id,
        family_id=family_id,
        source=source,
        transitions=transitions,
        outcome=outcome,
        success=bool(success),
        steps=int(steps),
        bfs_len=int(bfs_len),
        start_bfs_dist=int(start_bfs_dist),
        final_bfs_dist=int(final_bfs_dist),
        gap=int(gap),
        wall_hits=int(wall_hits),
        wall_rate=float(wall_rate),
        revisit_steps=int(revisit_steps),
        visit2_steps=int(visit2_steps),
        visit3plus_steps=int(visit3plus_steps),
        visit4plus_steps=int(visit4plus_steps),
        max_visit=int(max_visit),
        visit_penalty_sum=float(visit_penalty_sum),
        visit_penalty_norm=float(visit_penalty_norm),
        quality_key=quality_key,
        b_outcome=float(b_outcome),
        b_wall_presence=float(b_wall_presence),
        b_wall_count=float(b_wall_count),
        b_visit=float(b_visit),
        b_path=float(b_path),
        b_total=float(b_total),
        prefix_actions=list(prefix_actions),
        branch_pos=branch_pos,
    )

# -----------------------------
# Validators and key tests
# -----------------------------

def validate_trajectory(tr: Trajectory, label: str, args: argparse.Namespace, strict: bool = True) -> bool:
    ok = True
    if label == "bfs_success":
        ok = tr.success and tr.steps == tr.bfs_len and tr.gap == 0 and tr.wall_hits == 0 and tr.visit2_steps == 0 and tr.visit3plus_steps == 0
    elif label == "success_explore":
        ok = tr.success and tr.gap > 0 and tr.wall_hits <= args.success_explore_max_wall and tr.visit4plus_steps <= args.success_explore_max_visit4plus
    elif label == "recovery_success":
        ok = tr.success and tr.visit2_steps >= 1 and tr.visit4plus_steps <= args.recovery_max_visit4plus and tr.wall_hits <= args.recovery_max_wall and tr.gap > 0
    elif label == "explore_timeout":
        ok = (not tr.success) and tr.wall_hits == 0
    elif label == "loop_timeout":
        ok = (not tr.success) and tr.wall_hits == 0 and tr.visit3plus_steps >= args.loop_min_visit3plus and tr.max_visit >= args.loop_min_max_visit
    elif label == "wall_timeout":
        ok = (not tr.success) and tr.wall_hits >= 1
    else:
        ok = True
    if strict and not ok:
        return False
    return bool(ok)


def synthetic_key(
    success: bool,
    has_wall: bool,
    visit3plus: int,
    steps: int,
    bfs_len: int,
    final_dist: int,
    start_dist: int,
    wall_hits: int,
    visit_penalty_norm: float,
    args: argparse.Namespace,
) -> Tuple[float, float, float, float, float, float]:
    outcome_rank = 2 if success else 1
    vt = visit_tier(visit3plus, args)
    if success:
        path_quality = -float(max(0, steps - bfs_len)) / max(1e-9, float(args.max_steps - bfs_len))
    else:
        path_quality = -float(final_dist) / max(1e-9, float(start_dist))
    wall_rate = float(wall_hits) / max(1, steps)
    return (float(outcome_rank), -float(int(has_wall)), -float(vt), float(path_quality), -float(wall_rate), -float(visit_penalty_norm))


def run_key_unit_tests(args: argparse.Namespace) -> None:
    # success_wall > timeout_clean
    success_wall = synthetic_key(True, True, 0, 20, 10, 0, 10, 1, 0.05, args)
    timeout_clean = synthetic_key(False, False, 0, 64, 10, 1, 10, 0, 0.02, args)
    assert success_wall > timeout_clean, "key test failed: success_wall > timeout_clean"
    # success_clean > success_wall
    success_clean = synthetic_key(True, False, 0, 14, 10, 0, 10, 0, 0.02, args)
    assert success_clean > success_wall, "key test failed: success_clean > success_wall"
    # timeout_lowVisit > timeout_highVisit
    low_visit_to = synthetic_key(False, False, 0, 64, 10, 3, 10, 0, 0.02, args)
    high_visit_to = synthetic_key(False, False, 10, 64, 10, 1, 10, 0, 0.7, args)
    assert low_visit_to > high_visit_to, "key test failed: timeout_lowVisit > timeout_highVisit"
    # success_short > success_long
    short_s = synthetic_key(True, False, 0, 12, 10, 0, 10, 0, 0.01, args)
    long_s = synthetic_key(True, False, 0, 30, 10, 0, 10, 0, 0.01, args)
    assert short_s > long_s, "key test failed: success_short > success_long"
    # timeout_near > timeout_far
    near_t = synthetic_key(False, False, 0, 64, 10, 1, 10, 0, 0.01, args)
    far_t = synthetic_key(False, False, 0, 64, 10, 8, 10, 0, 0.01, args)
    assert near_t > far_t, "key test failed: timeout_near > timeout_far"
    # visit2_recovery > loop_timeout because success beats timeout and low visit beats high visit.
    visit2_recovery = synthetic_key(True, False, 0, 18, 10, 0, 10, 0, 0.03, args)
    visit3_loop = synthetic_key(False, False, 8, 64, 10, 1, 10, 0, 0.5, args)
    assert visit2_recovery > visit3_loop, "key test failed: visit2_recovery > loop_timeout"

# -----------------------------
# Trajectory generators
# -----------------------------

def choose_family_prefix(bfs_actions: List[int], args: argparse.Namespace) -> List[int]:
    if len(bfs_actions) <= 3:
        return []
    max_len = max(1, min(len(bfs_actions) - 2, int(round(len(bfs_actions) * args.family_prefix_max_ratio))))
    min_len = min(max_len, max(0, int(round(len(bfs_actions) * args.family_prefix_min_ratio))))
    k = random.randint(min_len, max_len)
    return list(bfs_actions[:k])


def valid_actions(grid: np.ndarray, pos: Tuple[int, int]) -> List[int]:
    return [a for a in range(4) if is_free(grid, apply_action(pos, a))]


def wall_actions(grid: np.ndarray, pos: Tuple[int, int]) -> List[int]:
    return [a for a in range(4) if not is_free(grid, apply_action(pos, a))]


def bfs_continue_actions(grid: np.ndarray, pos: Tuple[int, int], goal: Tuple[int, int]) -> List[int]:
    return shortest_actions_from_pos(grid, pos, goal)


def detour_actions_no_revisit(
    grid: np.ndarray,
    branch: Tuple[int, int],
    goal: Tuple[int, int],
    dist: np.ndarray,
    max_extra: int,
) -> Optional[List[int]]:
    # Try to create a small simple path from branch to a point that reconnects to BFS without revisits.
    for _ in range(80):
        cur = branch
        seen = {cur}
        prefix: List[int] = []
        for _step in range(random.randint(1, max(1, max_extra))):
            acts = [a for a in valid_actions(grid, cur) if apply_action(cur, a) not in seen]
            if not acts:
                break
            # Prefer non-decreasing or sideways moves to create real detour.
            random.shuffle(acts)
            a = acts[0]
            nxt = apply_action(cur, a)
            prefix.append(a)
            cur = nxt
            seen.add(cur)
            # If can reconnect to goal with no overlap except current, accept.
            cont = bfs_continue_actions(grid, cur, goal)
            pos = cur
            ok = True
            cont_positions = []
            for ca in cont:
                pos = apply_action(pos, ca)
                cont_positions.append(pos)
                if pos in seen and pos != goal:
                    ok = False
                    break
            if ok and cont and (len(prefix) + len(cont)) > int(dist[branch]):
                return prefix + cont
    return None


def generate_success_explore_branch(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    prefix: List[int],
    args: argparse.Namespace,
    maze_id: int,
    family_id: int,
) -> Optional[Trajectory]:
    branch = action_path_end(grid, start, prefix)
    dist = bfs_distances(grid, goal)
    for _ in range(args.generator_retry):
        extra = detour_actions_no_revisit(grid, branch, goal, dist, args.success_explore_max_extra)
        if extra is None:
            continue
        actions = prefix + extra
        tr = run_actions_as_trajectory(grid, start, goal, actions, "success_explore", args, maze_id, family_id, prefix, branch)
        if validate_trajectory(tr, "success_explore", args, strict=False):
            return tr
    return None


def generate_recovery_success_branch(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    prefix: List[int],
    args: argparse.Namespace,
    maze_id: int,
    family_id: int,
) -> Optional[Trajectory]:
    branch = action_path_end(grid, start, prefix)
    bfs_cont = bfs_continue_actions(grid, branch, goal)
    if not bfs_cont:
        return None
    bfs_first = bfs_cont[0]
    # Need a wrong free action that moves to a side cell, then backtracks to branch. That creates visit2 at branch.
    candidates = [a for a in valid_actions(grid, branch) if a != bfs_first]
    random.shuffle(candidates)
    for wrong in candidates:
        side = apply_action(branch, wrong)
        back = opposite_action(wrong)
        if apply_action(side, back) != branch:
            continue
        actions = prefix + [wrong, back] + bfs_cont
        tr = run_actions_as_trajectory(grid, start, goal, actions, "recovery_success", args, maze_id, family_id, prefix, branch)
        if validate_trajectory(tr, "recovery_success", args, strict=False):
            return tr
    return None


def generate_explore_timeout_branch(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    prefix: List[int],
    args: argparse.Namespace,
    maze_id: int,
    family_id: int,
) -> Optional[Trajectory]:
    branch = action_path_end(grid, start, prefix)
    for _ in range(args.generator_retry):
        cur = branch
        actions = list(prefix)
        visits = defaultdict(int)
        # approximate visit tracking after prefix
        p = start
        visits[p] += 1
        for a in prefix:
            p2 = apply_action(p, a)
            if is_free(grid, p2):
                p = p2
            visits[p] += 1
        for _step in range(args.max_steps - len(prefix)):
            acts = valid_actions(grid, cur)
            # avoid goal, avoid high revisits, no wall
            scored = []
            for a in acts:
                nxt = apply_action(cur, a)
                if nxt == goal:
                    continue
                c = visits[nxt] + 1
                scored.append((c, random.random(), a, nxt))
            if not scored:
                break
            scored.sort(key=lambda x: (x[0], x[1]))
            _, _, a, nxt = scored[0]
            actions.append(a)
            cur = nxt
            visits[cur] += 1
            if len(actions) >= args.max_steps:
                break
        tr = run_actions_as_trajectory(grid, start, goal, actions, "explore_timeout", args, maze_id, family_id, prefix, branch)
        if validate_trajectory(tr, "explore_timeout", args, strict=False):
            return tr
    return None


def generate_loop_timeout_branch(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    prefix: List[int],
    args: argparse.Namespace,
    maze_id: int,
    family_id: int,
) -> Optional[Trajectory]:
    branch = action_path_end(grid, start, prefix)
    acts = [a for a in valid_actions(grid, branch) if apply_action(branch, a) != goal]
    random.shuffle(acts)
    for a in acts:
        nxt = apply_action(branch, a)
        back = opposite_action(a)
        if not is_free(grid, nxt) or apply_action(nxt, back) != branch:
            continue
        loop = [a, back] * ((args.max_steps - len(prefix)) // 2 + 2)
        actions = (prefix + loop)[: args.max_steps]
        tr = run_actions_as_trajectory(grid, start, goal, actions, "loop_timeout", args, maze_id, family_id, prefix, branch)
        if validate_trajectory(tr, "loop_timeout", args, strict=False):
            return tr
    return None


def generate_wall_timeout_branch(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    prefix: List[int],
    args: argparse.Namespace,
    maze_id: int,
    family_id: int,
    repeated: bool = False,
) -> Optional[Trajectory]:
    branch = action_path_end(grid, start, prefix)
    walls = wall_actions(grid, branch)
    if not walls:
        # Move to a neighbor that has wall action.
        for a in valid_actions(grid, branch):
            nb = apply_action(branch, a)
            if wall_actions(grid, nb):
                prefix2 = prefix + [a]
                branch = nb
                walls = wall_actions(grid, branch)
                break
    if not walls:
        return None
    wall_a = random.choice(walls)
    if repeated:
        tail = [wall_a] * max(args.repeated_wall_min_hits + 4, args.max_steps - len(prefix))
        source = "wall_timeout"
    else:
        # one/few walls then wander; still timeout.
        valid = valid_actions(grid, branch)
        wander: List[int] = []
        cur = branch
        for _ in range(max(0, args.max_steps - len(prefix) - 3)):
            acts = valid_actions(grid, cur)
            if not acts:
                break
            a = random.choice(acts)
            if apply_action(cur, a) == goal:
                continue
            wander.append(a)
            cur = apply_action(cur, a)
        tail = [wall_a] * random.randint(1, 3) + wander
        source = "wall_timeout"
    actions = (prefix + tail)[: args.max_steps]
    tr = run_actions_as_trajectory(grid, start, goal, actions, source, args, maze_id, family_id, prefix, branch)
    label = "repeated_wall_timeout" if repeated else "wall_timeout"
    if validate_trajectory(tr, label, args, strict=False):
        return tr
    return None


def build_family(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    maze_id: int,
    family_id: int,
    args: argparse.Namespace,
    bfs_actions_ref: List[int],
) -> Family:
    prefix = choose_family_prefix(bfs_actions_ref, args)
    branch = action_path_end(grid, start, prefix)
    trajs: List[Trajectory] = []
    # Branch-level BFS success: prefix + shortest continuation.
    cont = bfs_continue_actions(grid, branch, goal)
    if cont:
        tr = run_actions_as_trajectory(grid, start, goal, prefix + cont, "bfs_success", args, maze_id, family_id, prefix, branch)
        if validate_trajectory(tr, "bfs_success", args, strict=False):
            trajs.append(tr)
    for gen_fn in [
        generate_success_explore_branch,
        generate_recovery_success_branch,
        generate_explore_timeout_branch,
        generate_loop_timeout_branch,
    ]:
        tr = gen_fn(grid, start, goal, prefix, args, maze_id, family_id)
        if tr is not None:
            trajs.append(tr)
    trw = generate_wall_timeout_branch(
        grid, start, goal, prefix, args, maze_id, family_id,
        repeated=(random.random() < 0.5),
    )
    if trw is not None:
        # repeated wall is now a subtype of wall_timeout, not a separate trajectory label.
        trw.source = "wall_timeout"
        trajs.append(trw)
    return Family(maze_id=maze_id, family_id=family_id, grid=grid, start=start, goal=goal, prefix_actions=prefix, branch_pos=branch, trajectories=trajs)

# -----------------------------
# Pair construction and margins
# -----------------------------

BUCKETS = ["outcome", "wall", "revisit_loop", "path", "cpl_recover_wall", "cpl_explore_wall", "cpl_recover_loop"]

def better_worse(a: Trajectory, b: Trajectory) -> Optional[Tuple[Trajectory, Trajectory]]:
    if a.quality_key > b.quality_key:
        return a, b
    if b.quality_key > a.quality_key:
        return b, a
    return None


def component_for_bucket(tr: Trajectory, bucket: str) -> float:
    if bucket == "outcome":
        return tr.b_outcome
    if bucket == "wall":
        # Wall pair compares no-wall vs has-wall primarily.
        return tr.b_wall_presence
    if bucket == "revisit_loop":
        return tr.b_visit
    if bucket == "path":
        return tr.b_path
    if bucket in ["cpl_recover_wall", "cpl_explore_wall"]:
        return tr.b_wall_count + tr.b_visit
    if bucket == "cpl_recover_loop":
        return tr.b_visit
    return tr.b_total


def bucket_margin_params(args: argparse.Namespace, bucket: str) -> Tuple[float, float, float]:
    if bucket == "outcome":
        return args.margin_outcome_min, args.margin_outcome_max, args.margin_outcome_s
    if bucket == "wall":
        return args.margin_wall_min, args.margin_wall_max, args.margin_wall_s
    if bucket == "revisit_loop":
        return args.margin_revisit_min, args.margin_revisit_max, args.margin_revisit_s
    if bucket == "path":
        return args.margin_path_min, args.margin_path_max, args.margin_path_s
    if bucket == "cpl_recover_wall":
        return args.margin_cpl_recover_wall_min, args.margin_cpl_recover_wall_max, args.margin_cpl_recover_wall_s
    if bucket == "cpl_explore_wall":
        return args.margin_cpl_explore_wall_min, args.margin_cpl_explore_wall_max, args.margin_cpl_explore_wall_s
    if bucket == "cpl_recover_loop":
        return args.margin_cpl_recover_loop_min, args.margin_cpl_recover_loop_max, args.margin_cpl_recover_loop_s
    return args.margin_path_min, args.margin_path_max, args.margin_path_s


def compute_margin(pos: Trajectory, neg: Trajectory, bucket: str, args: argparse.Namespace) -> Tuple[float, float]:
    b_pos = component_for_bucket(pos, bucket)
    b_neg = component_for_bucket(neg, bucket)
    delta = max(0.0, float(b_neg - b_pos))
    m_min, m_max, s = bucket_margin_params(args, bucket)
    margin = float(m_min + (m_max - m_min) * math.tanh(delta / (s + 1e-9)))
    return margin, delta


def make_pair(pos: Trajectory, neg: Trajectory, bucket: str, source: str, args: argparse.Namespace) -> PreferencePair:
    m, d = compute_margin(pos, neg, bucket, args)
    return PreferencePair(pos=pos, neg=neg, bucket=bucket, margin=m, delta_component=d, pair_source=source)


def find_by_source(trajs: List[Trajectory], source: str) -> List[Trajectory]:
    return [t for t in trajs if t.source == source]


def add_candidate_pair(
    out: Dict[str, List[PreferencePair]],
    a: Optional[Trajectory],
    b: Optional[Trajectory],
    bucket: str,
    source: str,
    args: argparse.Namespace,
) -> None:
    if a is None or b is None:
        return
    bw = better_worse(a, b)
    if bw is None:
        return
    pos, neg = bw
    out[bucket].append(make_pair(pos, neg, bucket, source, args))


def first_traj(fam: Family, source: str) -> Optional[Trajectory]:
    return next(iter(find_by_source(fam.trajectories, source)), None)


def generate_pair_candidates_for_maze(families: List[Family], args: argparse.Namespace) -> Dict[str, List[PreferencePair]]:
    """Generate exactly the family-level mandatory pair candidates.

    Each valid family contributes up to 12 pairs:
      outcome: S>E, R>E, R>L
      wall:    E>W, L>W
      revisit: R>L, S>L
      path:    B>S, B>R
      cpl:     R>W, S>W, R>L
    """
    cands: Dict[str, List[PreferencePair]] = {b: [] for b in BUCKETS}
    for fam in families:
        B = first_traj(fam, "bfs_success")
        S = first_traj(fam, "success_explore")
        R = first_traj(fam, "recovery_success")
        E = first_traj(fam, "explore_timeout")
        L = first_traj(fam, "loop_timeout")
        W = first_traj(fam, "wall_timeout")
        # outcome pairs: success vs timeout, but not dominated by BFS.
        add_candidate_pair(cands, S, E, "outcome", "family_mandatory", args)
        add_candidate_pair(cands, R, E, "outcome", "family_mandatory", args)
        add_candidate_pair(cands, R, L, "outcome", "family_mandatory", args)
        # wall pairs: timeout vs timeout, wall is worse.
        add_candidate_pair(cands, E, W, "wall", "family_mandatory", args)
        add_candidate_pair(cands, L, W, "wall", "family_mandatory", args)
        # revisit/loop pairs: recovery/explore after revisits beats loop failure.
        add_candidate_pair(cands, R, L, "revisit_loop", "family_mandatory", args)
        add_candidate_pair(cands, S, L, "revisit_loop", "family_mandatory", args)
        # path pairs: BFS is the anchor, but path margin is intentionally weak.
        add_candidate_pair(cands, B, S, "path", "family_mandatory", args)
        add_candidate_pair(cands, B, R, "path", "family_mandatory", args)
        # CPL pairs: same-prefix counterfactuals.
        add_candidate_pair(cands, R, W, "cpl_recover_wall", "family_mandatory", args)
        add_candidate_pair(cands, S, W, "cpl_explore_wall", "family_mandatory", args)
        add_candidate_pair(cands, R, L, "cpl_recover_loop", "family_mandatory", args)
    return cands


def sample_bucket_pairs(cands: Dict[str, List[PreferencePair]], args: argparse.Namespace) -> List[PreferencePair]:
    # v2.0.7b does not use complete closure or random fill. It keeps the family-level mandatory pairs.
    selected: List[PreferencePair] = []
    seen = set()
    for bucket in BUCKETS:
        for pp in cands.get(bucket, []):
            key = (id(pp.pos), id(pp.neg), pp.bucket, pp.pair_source)
            if key in seen:
                continue
            seen.add(key)
            selected.append(pp)
    return selected[: args.pairs_per_maze]


def print_expected_gradient_pressure(args: argparse.Namespace) -> None:
    rows = [
        ("outcome", 3, 0.80),
        ("wall", 2, 0.45),
        ("revisit_loop", 2, 0.30),
        ("path", 2, 0.08),
        ("cpl", 3, 0.45),
    ]
    pressures = [(name, n, m, n*m) for name, n, m in rows]
    total = sum(p for *_rest, p in pressures) or 1.0
    print("\n[Expected Gradient Pressure]")
    print(f"{'bucket':14s} {'n_per_family':>12s} {'typical_margin':>15s} {'pressure':>10s} {'pct':>8s}")
    for name, n, m, p in pressures:
        print(f"{name:14s} {n:12d} {m:15.3f} {p:10.3f} {100*p/total:7.2f}%")

# -----------------------------
# Dataset reports
# -----------------------------

def fmt_pct(x: float) -> str:
    return f"{100.0 * x:6.2f}%"


def print_dataset_reports(trajectories: List[Trajectory], pairs: List[PreferencePair], args: argparse.Namespace) -> Dict[str, Any]:
    print("\n==== RM DATASET SUMMARY ====")
    by_type: Dict[str, List[Trajectory]] = defaultdict(list)
    for t in trajectories:
        by_type[t.source].append(t)
    order = ["bfs_success", "success_explore", "recovery_success", "explore_timeout", "loop_timeout", "wall_timeout"]
    print("\n[Trajectory Label Validation]")
    print(f"{'type':26s} {'n':>6s} {'succ%':>8s} {'wall_mean':>10s} {'visit2_mean':>12s} {'visit3+_mean':>13s} {'maxVisit_mean':>14s} {'gap_mean':>10s} {'finalDist_mean':>15s}")
    summary: Dict[str, Any] = {"trajectory_types": {}, "badness": {}, "pair_buckets": {}}
    for typ in order:
        arr = by_type.get(typ, [])
        n = len(arr)
        if n == 0:
            print(f"{typ:26s} {0:6d}")
            continue
        vals = {
            "n": n,
            "success_rate": safe_mean([1.0 if t.success else 0.0 for t in arr]),
            "wall_mean": safe_mean([t.wall_hits for t in arr]),
            "visit2_mean": safe_mean([t.visit2_steps for t in arr]),
            "visit3plus_mean": safe_mean([t.visit3plus_steps for t in arr]),
            "max_visit_mean": safe_mean([t.max_visit for t in arr]),
            "gap_mean": safe_mean([t.gap if t.success else 0.0 for t in arr]),
            "final_dist_mean": safe_mean([t.final_bfs_dist if not t.success else 0.0 for t in arr]),
        }
        summary["trajectory_types"][typ] = vals
        print(f"{typ:26s} {n:6d} {fmt_pct(vals['success_rate']):>8s} {vals['wall_mean']:10.2f} {vals['visit2_mean']:12.2f} {vals['visit3plus_mean']:13.2f} {vals['max_visit_mean']:14.2f} {vals['gap_mean']:10.2f} {vals['final_dist_mean']:15.2f}")
    print("\n[Badness Component Report]")
    print(f"{'type':26s} {'B_outcome':>10s} {'B_wallPres':>11s} {'B_wallCount':>12s} {'B_visit':>9s} {'B_path':>9s} {'B_total':>9s}")
    for typ in order:
        arr = by_type.get(typ, [])
        if not arr:
            continue
        comps = {
            "B_outcome": safe_mean([t.b_outcome for t in arr]),
            "B_wallPres": safe_mean([t.b_wall_presence for t in arr]),
            "B_wallCount": safe_mean([t.b_wall_count for t in arr]),
            "B_visit": safe_mean([t.b_visit for t in arr]),
            "B_path": safe_mean([t.b_path for t in arr]),
            "B_total": safe_mean([t.b_total for t in arr]),
        }
        summary["badness"][typ] = comps
        print(f"{typ:26s} {comps['B_outcome']:10.3f} {comps['B_wallPres']:11.3f} {comps['B_wallCount']:12.3f} {comps['B_visit']:9.3f} {comps['B_path']:9.3f} {comps['B_total']:9.3f}")
    print("\n[Badness Distribution by Type]")
    print(f"{'type':26s} {'B_total_mean':>13s} {'B_total_p50':>12s} {'B_total_min':>12s} {'B_total_max':>12s}")
    for typ in order:
        arr = by_type.get(typ, [])
        if not arr:
            continue
        bt = [t.b_total for t in arr]
        print(f"{typ:26s} {safe_mean(bt):13.3f} {percentile(bt,50):12.3f} {min(bt):12.3f} {max(bt):12.3f}")
    # Sanity warnings.
    def mean_b(typ: str) -> float:
        return safe_mean([t.b_total for t in by_type.get(typ, [])])
    if by_type.get("bfs_success") and by_type.get("success_explore") and mean_b("bfs_success") > mean_b("success_explore"):
        print("[WARN] bfs_success B_total is not lower than success_explore.")
    if by_type.get("explore_timeout") and by_type.get("loop_timeout") and mean_b("loop_timeout") <= mean_b("explore_timeout"):
        print("[WARN] loop_timeout B_total is not higher than explore_timeout.")
    if by_type.get("wall_timeout") and by_type.get("explore_timeout") and mean_b("wall_timeout") <= mean_b("explore_timeout"):
        print("[WARN] wall_timeout B_total is not higher than explore_timeout.")
    # Pair buckets and margins.
    print("\n[Pair Bucket Report]")
    print(f"{'bucket':14s} {'n':>7s} {'margin_mean':>12s} {'margin_p10':>11s} {'margin_p50':>11s} {'margin_p90':>11s} {'margin_min':>11s} {'margin_max':>11s}")
    by_bucket: Dict[str, List[PreferencePair]] = defaultdict(list)
    for p in pairs:
        by_bucket[p.bucket].append(p)
    for b in BUCKETS:
        arr = by_bucket.get(b, [])
        if not arr:
            continue
        ms = [p.margin for p in arr]
        vals = {
            "n": len(arr),
            "mean": safe_mean(ms),
            "p10": percentile(ms, 10),
            "p50": percentile(ms, 50),
            "p90": percentile(ms, 90),
            "min": min(ms),
            "max": max(ms),
        }
        summary["pair_buckets"][b] = vals
        print(f"{b:14s} {len(arr):7d} {vals['mean']:12.3f} {vals['p10']:11.3f} {vals['p50']:11.3f} {vals['p90']:11.3f} {vals['min']:11.3f} {vals['max']:11.3f}")
    print("\n[Margin Saturation Check]")
    print(f"{'bucket':14s} {'pct_near_min':>14s} {'pct_near_max':>14s}")
    for b in BUCKETS:
        arr = by_bucket.get(b, [])
        if not arr:
            continue
        m_min, m_max, _ = bucket_margin_params(args, b if b != "random" else "path")
        near_min = safe_mean([1.0 if p.margin <= m_min + 0.02 else 0.0 for p in arr])
        near_max = safe_mean([1.0 if p.margin >= m_max - 0.02 else 0.0 for p in arr])
        print(f"{b:14s} {fmt_pct(near_min):>14s} {fmt_pct(near_max):>14s}")
        if near_max > 0.40:
            print(f"[WARN] bucket={b} has pct_near_max > 40%; margin may be saturating.")
    print_expected_gradient_pressure(args)
    return summary

# -----------------------------
# Build RM dataset
# -----------------------------

def build_preference_dataset(args: argparse.Namespace) -> Tuple[List[Trajectory], List[PreferencePair], Dict[str, Any]]:
    run_key_unit_tests(args)
    trajectories: List[Trajectory] = []
    pairs: List[PreferencePair] = []
    maze_summaries: List[Dict[str, Any]] = []
    for maze_id in tqdm(range(args.rm_mazes), desc="Building RM families"):
        difficulty = sample_difficulty(args)
        grid, start, goal = generate_maze(difficulty)
        bfs_paths = enumerate_shortest_action_paths(grid, start, goal, max_paths=args.max_bfs_paths_per_maze)
        if not bfs_paths:
            continue
        # Add all/capped BFS samples at maze family -1.
        for k, path in enumerate(bfs_paths):
            tr = run_actions_as_trajectory(grid, start, goal, path, "bfs_success", args, maze_id, -1, [], start)
            if validate_trajectory(tr, "bfs_success", args, strict=False):
                trajectories.append(tr)
        families: List[Family] = []
        for fam_id in range(args.families_per_maze):
            ref = random.choice(bfs_paths)
            fam = build_family(grid, start, goal, maze_id, fam_id, args, ref)
            families.append(fam)
            trajectories.extend(fam.trajectories)
        cands = generate_pair_candidates_for_maze(families, args)
        pairs.extend(sample_bucket_pairs(cands, args))
        maze_summaries.append({"maze_id": maze_id, "difficulty": difficulty, "n_bfs_paths": len(bfs_paths), "n_family_traj": sum(len(f.trajectories) for f in families)})
    summary = print_dataset_reports(trajectories, pairs, args)
    summary["maze_summaries"] = maze_summaries[:100]
    summary["total_trajectories"] = len(trajectories)
    summary["total_pairs"] = len(pairs)
    return trajectories, pairs, summary

# -----------------------------
# Reward model
# -----------------------------

class RewardModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * SIZE * SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def rm_inputs_for_traj(tr: Trajectory, device: torch.device) -> torch.Tensor:
    arr = np.stack([transition_input(t.state, t.action, t.next_state) for t in tr.transitions], axis=0).astype(np.float32)
    return torch.from_numpy(arr).to(device)


def trajectory_score(model: RewardModel, tr: Trajectory, device: torch.device, normalizer: str = "bfs_len") -> Tuple[torch.Tensor, torch.Tensor]:
    if not tr.transitions:
        z = torch.tensor(0.0, device=device)
        return z, z.view(1)
    x = rm_inputs_for_traj(tr, device)
    rewards = model(x)
    denom = 1.0
    if normalizer == "bfs_len":
        denom = max(1.0, float(tr.bfs_len))
    elif normalizer == "episode_len":
        denom = max(1.0, float(tr.steps))
    return rewards.sum() / denom, rewards


def train_reward_model(pairs: List[PreferencePair], args: argparse.Namespace, device: torch.device, out_dir: Path) -> Tuple[RewardModel, Dict[str, Any]]:
    model = RewardModel().to(device)
    opt = optim.Adam(model.parameters(), lr=args.rm_lr)
    idx = list(range(len(pairs)))
    random.shuffle(idx)
    split = int(len(idx) * 0.85)
    train_idx = idx[:split]
    val_idx = idx[split:]
    hist: Dict[str, List[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_loss = float("inf")
    best_acc = -1.0
    best_loss_path = out_dir / f"{VERSION}_reward_model_best_loss.pt"
    best_acc_path = out_dir / f"{VERSION}_reward_model_best_acc.pt"
    last_path = out_dir / f"{VERSION}_reward_model_last.pt"

    def run_epoch(indices: List[int], train: bool) -> Tuple[float, float]:
        if train:
            random.shuffle(indices)
        losses: List[float] = []
        accs: List[float] = []
        for start in range(0, len(indices), args.rm_batch_size):
            batch_ids = indices[start:start + args.rm_batch_size]
            if train:
                opt.zero_grad()
            batch_losses = []
            batch_rewards = []
            correct = 0
            total = 0
            for pi in batch_ids:
                pp = pairs[pi]
                sp, rp = trajectory_score(model, pp.pos, device, args.score_normalizer)
                sn, rn = trajectory_score(model, pp.neg, device, args.score_normalizer)
                margin = torch.tensor(float(pp.margin), device=device)
                diff = sp - sn - margin
                batch_losses.append(-F.logsigmoid(diff))
                batch_rewards.append(rp)
                batch_rewards.append(rn)
                correct += int((sp.detach() > sn.detach()).item())
                total += 1
            if not batch_losses:
                continue
            loss_rank = torch.stack(batch_losses).mean()
            if batch_rewards:
                all_rewards = torch.cat(batch_rewards)
                loss_reg = float(args.reward_l2) * (all_rewards ** 2).mean()
            else:
                loss_reg = torch.tensor(0.0, device=device)
            loss = loss_rank + loss_reg
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            losses.append(float(loss.detach().cpu().item()))
            accs.append(correct / max(1, total))
        return safe_mean(losses), safe_mean(accs)

    print("\n==== RM TRAINING ====")
    print(f"pairs={len(pairs)} train={len(train_idx)} val={len(val_idx)} batch={args.rm_batch_size} epochs={args.rm_epochs} device={device}")
    for ep in range(1, args.rm_epochs + 1):
        model.train()
        tr_loss, tr_acc = run_epoch(train_idx, True)
        model.eval()
        with torch.no_grad():
            va_loss, va_acc = run_epoch(val_idx, False)
        hist["train_loss"].append(tr_loss); hist["train_acc"].append(tr_acc)
        hist["val_loss"].append(va_loss); hist["val_acc"].append(va_acc)
        print(f"[RM ep {ep:02d}/{args.rm_epochs}] train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={va_loss:.4f} acc={va_acc:.4f}")
        if va_loss < best_loss:
            best_loss = va_loss
            torch.save({"model": model.state_dict(), "args": vars(args), "best_val_loss": best_loss}, best_loss_path)
            print(f"  [save] best loss -> {best_loss_path.name}")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model": model.state_dict(), "args": vars(args), "best_val_acc": best_acc}, best_acc_path)
            print(f"  [save] best acc  -> {best_acc_path.name}")
    torch.save({"model": model.state_dict(), "args": vars(args)}, last_path)
    hist_obj = {"history": hist, "best_val_loss": best_loss, "best_val_acc": best_acc, "best_loss_path": str(best_loss_path), "best_acc_path": str(best_acc_path), "last_path": str(last_path)}
    with open(out_dir / f"{VERSION}_reward_model_history.json", "w", encoding="utf-8") as f:
        json.dump(hist_obj, f, indent=2)
    plot_rm_curves(hist, out_dir / f"{VERSION}_reward_model_curves.png")
    return model, hist_obj


def plot_rm_curves(hist: Dict[str, List[float]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist["train_loss"], label="train")
    axes[0].plot(hist["val_loss"], label="val")
    axes[0].set_title("BTL loss")
    axes[0].legend()
    axes[1].plot(hist["train_acc"], label="train")
    axes[1].plot(hist["val_acc"], label="val")
    axes[1].set_title("Preference accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def load_reward_model(path: str | Path, device: torch.device) -> RewardModel:
    model = RewardModel().to(device)
    ckpt = torch.load(path, map_location=device)
    sd = ckpt.get("model", ckpt)
    model.load_state_dict(sd)
    model.eval()
    return model

# -----------------------------
# RM diagnostic
# -----------------------------

def probe_step_rewards(model: RewardModel, args: argparse.Namespace, device: torch.device, out_dir: Path) -> Dict[str, Any]:
    cats: Dict[str, List[float]] = defaultdict(list)
    model.eval()
    for maze_id in range(args.debug_probe_mazes):
        grid, start, goal = generate_maze(sample_difficulty(args))
        bfs_paths = enumerate_shortest_action_paths(grid, start, goal, max_paths=1)
        if not bfs_paths:
            continue
        # Collect family trajectories plus a repeated wall trajectory.
        fam = build_family(grid, start, goal, maze_id, 0, args, bfs_paths[0])
        for tr in fam.trajectories:
            if not tr.transitions:
                continue
            x = rm_inputs_for_traj(tr, device)
            with torch.no_grad():
                rs = model(x).detach().cpu().numpy().tolist()
            for t, r in zip(tr.transitions, rs):
                if t.is_goal:
                    cats["goal"].append(float(r))
                if t.is_wall:
                    cats["wall"].append(float(r))
                    if t.visit_count_after >= 4:
                        cats["repeated_wall"].append(float(r))
                elif t.delta_bfs > 0:
                    cats["toward_goal"].append(float(r))
                elif t.delta_bfs < 0:
                    cats["away_goal"].append(float(r))
                if t.visit_count_after == 2:
                    cats["visit2"].append(float(r))
                if t.visit_count_after >= 3:
                    cats["visit3plus"].append(float(r))
                if t.visit_count_after >= 4:
                    cats["visit4plus"].append(float(r))
    order = ["goal", "toward_goal", "away_goal", "visit2", "visit3plus", "visit4plus", "wall", "repeated_wall"]
    report: Dict[str, Any] = {}
    print("\n[RM Step Reward Probe]")
    print(f"{'category':18s} {'mean':>9s} {'p10':>9s} {'p50':>9s} {'p90':>9s} {'n':>7s}")
    for c in order:
        vals = cats.get(c, [])
        if vals:
            row = {"mean": safe_mean(vals), "p10": percentile(vals, 10), "p50": percentile(vals, 50), "p90": percentile(vals, 90), "n": len(vals)}
        else:
            row = {"mean": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan"), "n": 0}
        report[c] = row
        print(f"{c:18s} {row['mean']:9.3f} {row['p10']:9.3f} {row['p50']:9.3f} {row['p90']:9.3f} {row['n']:7d}")
    warnings: List[str] = []
    def mean(c: str) -> float:
        return float(report.get(c, {}).get("mean", float("nan")))
    if not math.isnan(mean("visit2")) and not math.isnan(mean("wall")) and mean("visit2") <= mean("wall"):
        warnings.append("visit2_mean <= wall_mean")
    if not math.isnan(mean("repeated_wall")) and not math.isnan(mean("visit3plus")) and mean("repeated_wall") > mean("visit3plus"):
        warnings.append("repeated_wall_mean > visit3plus_mean")
    if not math.isnan(mean("toward_goal")) and not math.isnan(mean("away_goal")) and mean("toward_goal") <= mean("away_goal"):
        warnings.append("toward_goal_mean <= away_goal_mean")
    for w in warnings:
        print(f"[STRONG WARN] RM diagnostic failed: {w}")
    report["warnings"] = warnings
    with open(out_dir / f"{VERSION}_rm_diagnostic.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    if warnings and args.stop_if_rm_diagnostic_fails:
        raise RuntimeError("RM diagnostic failed: " + "; ".join(warnings))
    return report

# -----------------------------
# DQN environment and network
# -----------------------------

class MazeEnv:
    def __init__(self, args: argparse.Namespace, difficulty: str = "easy", fixed: Optional[Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]] = None) -> None:
        self.args = args
        if fixed is not None:
            self.grid, self.start, self.goal = fixed
        else:
            self.grid, self.start, self.goal = generate_maze(difficulty)
        self.dist = bfs_distances(self.grid, self.goal)
        self.pos = self.start
        self.visit_counts = np.zeros((SIZE, SIZE), dtype=np.int32)
        self.visit_counts[self.pos] = 1
        self.steps = 0
        self.wall_hits = 0
        self.revisit_steps = 0
        self.visit2_steps = 0
        self.visit3plus_steps = 0
        self.visit4plus_steps = 0
        self.visit_penalty_sum = 0.0
        self.max_visit = 1

    def state(self) -> np.ndarray:
        return encode_state(self.grid, self.pos, self.goal, self.visit_counts, self.args.visit_count_cap)

    def step(self, action: int) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        before = self.pos
        bfs_before = int(self.dist[before]) if int(self.dist[before]) < 10_000 else self.args.max_steps
        attempted = apply_action(self.pos, action)
        is_wall = not is_free(self.grid, attempted)
        if is_wall:
            nxt = self.pos
            self.wall_hits += 1
        else:
            nxt = attempted
        self.visit_counts[nxt] += 1
        c_after = int(self.visit_counts[nxt])
        self.max_visit = max(self.max_visit, c_after)
        if c_after >= 2:
            self.revisit_steps += 1
        if c_after == 2:
            self.visit2_steps += 1
        if c_after >= 3:
            self.visit3plus_steps += 1
        if c_after >= 4:
            self.visit4plus_steps += 1
        self.visit_penalty_sum += visit_count_penalty(c_after, self.args)
        self.pos = nxt
        self.steps += 1
        done = self.pos == self.goal or self.steps >= self.args.max_steps
        bfs_after = int(self.dist[nxt]) if int(self.dist[nxt]) < 10_000 else self.args.max_steps
        info = {
            "is_wall": bool(is_wall),
            "is_goal": bool(self.pos == self.goal),
            "pos_before": before,
            "pos_after": nxt,
            "bfs_before": bfs_before,
            "bfs_after": bfs_after,
            "delta_bfs": bfs_before - bfs_after,
            "visit_count_after": c_after,
        }
        return self.state(), done, info


class QNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * SIZE * SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buf: deque = deque(maxlen=capacity)

    def push(self, *args: Any) -> None:
        self.buf.append(DQNTransition(*args))

    def sample(self, batch_size: int) -> List[DQNTransition]:
        return random.sample(self.buf, batch_size)

    def __len__(self) -> int:
        return len(self.buf)


def rm_reward_for_transition(model: RewardModel, state: np.ndarray, action: int, next_state: np.ndarray, device: torch.device, scale: float) -> float:
    x = torch.from_numpy(transition_input(state, action, next_state)[None].astype(np.float32)).to(device)
    with torch.no_grad():
        r = float(model(x).item())
    return scale * r


def epsilon_by_episode(ep: int, args: argparse.Namespace) -> float:
    if ep >= args.eps_decay_episodes:
        return args.eps_end
    frac = ep / max(1, args.eps_decay_episodes)
    return args.eps_start + frac * (args.eps_end - args.eps_start)


def optimize_q(q: QNetwork, target: QNetwork, replay: ReplayBuffer, opt: optim.Optimizer, args: argparse.Namespace, device: torch.device) -> float:
    if len(replay) < args.batch_size:
        return float("nan")
    batch = replay.sample(args.batch_size)
    states = torch.from_numpy(np.stack([b.state for b in batch]).astype(np.float32)).to(device)
    actions = torch.tensor([b.action for b in batch], dtype=torch.long, device=device)
    rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=device)
    next_states = torch.from_numpy(np.stack([b.next_state for b in batch]).astype(np.float32)).to(device)
    dones = torch.tensor([b.done for b in batch], dtype=torch.float32, device=device)
    qv = q(states).gather(1, actions.view(-1, 1)).squeeze(1)
    with torch.no_grad():
        next_actions = q(next_states).argmax(dim=1)
        next_q = target(next_states).gather(1, next_actions.view(-1, 1)).squeeze(1)
        y = rewards + args.gamma * (1.0 - dones) * next_q
    loss = F.smooth_l1_loss(qv, y)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q.parameters(), 5.0)
    opt.step()
    return float(loss.detach().cpu().item())


def rollout_policy(
    q: QNetwork,
    rm: RewardModel,
    args: argparse.Namespace,
    device: torch.device,
    difficulty: str = "easy",
    fixed: Optional[Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]] = None,
    epsilon: float = 0.0,
    collect_debug: bool = False,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], List[np.ndarray]]:
    env = MazeEnv(args, difficulty=difficulty, fixed=fixed)
    frames: List[np.ndarray] = []
    debug: List[Dict[str, Any]] = []
    total_reward = 0.0
    bfs_agree = 0
    state = env.state()
    for t in range(args.max_steps):
        if random.random() < epsilon:
            action = random.randrange(4)
        else:
            with torch.no_grad():
                qs = q(torch.from_numpy(state[None].astype(np.float32)).to(device)).detach().cpu().numpy()[0]
            action = int(np.argmax(qs))
        next_state, done, info = env.step(action)
        r = rm_reward_for_transition(rm, state, action, next_state, device, args.rm_scale_used)
        total_reward += r
        if info["delta_bfs"] > 0:
            bfs_agree += 1
        if collect_debug:
            action_rows = []
            with torch.no_grad():
                qs_now = q(torch.from_numpy(state[None].astype(np.float32)).to(device)).detach().cpu().numpy()[0]
            # simulate each action for RM reward without mutating env by using current visit map copy
            for a in range(4):
                pos = tuple(env.visit_counts.shape)  # dummy overwritten below
                # Construct one-step next state from state by approximate copy from current env before chosen action is hard after step.
                # Instead, use current env after step for chosen action details; for alternatives use basic wall prediction.
                attempted = apply_action(info["pos_before"], a)
                is_wall = not is_free(env.grid, attempted)
                next_pos = info["pos_before"] if is_wall else attempted
                vc = env.visit_counts.copy()
                # approximate pre-step count by subtracting chosen action visit, then add alternative
                vc[info["pos_after"]] = max(0, vc[info["pos_after"]] - 1)
                vc[next_pos] += 1
                alt_next_state = encode_state(env.grid, next_pos, env.goal, vc, args.visit_count_cap)
                alt_r = rm_reward_for_transition(rm, state, a, alt_next_state, device, args.rm_scale_used)
                b_before = int(env.dist[info["pos_before"]]) if int(env.dist[info["pos_before"]]) < 10_000 else args.max_steps
                b_after = int(env.dist[next_pos]) if int(env.dist[next_pos]) < 10_000 else args.max_steps
                action_rows.append({
                    "action": ACTION_NAMES[a],
                    "q": float(qs_now[a]),
                    "rm_reward": float(alt_r),
                    "is_wall": bool(is_wall),
                    "next_position": list(next_pos),
                    "visit_count_after": int(vc[next_pos]),
                    "delta_bfs_dist": int(b_before - b_after),
                })
            debug.append({
                "step": t,
                "position": list(info["pos_before"]),
                "chosen_action": ACTION_NAMES[action],
                "is_wall": bool(info["is_wall"]),
                "visit_count": int(info["visit_count_after"]),
                "rm_reward": float(r),
                "actions": action_rows,
            })
            frames.append(render_maze_frame(env.grid, info["pos_after"], env.goal, env.visit_counts, title=f"step={t} action={ACTION_NAMES[action]} wall={info['is_wall']} visit={info['visit_count_after']} r={r:.2f}"))
        state = next_state
        if done:
            break
    success = env.pos == env.goal
    metrics = {
        "success": float(success),
        "explore_timeout": float((not success) and env.wall_hits == 0),
        "wall_timeout": float((not success) and env.wall_hits > 0),
        "reward": float(total_reward),
        "steps": float(env.steps),
        "wall_hits": float(env.wall_hits),
        "wall_step_rate": float(env.wall_hits / max(1, env.steps)),
        "visit2_steps": float(env.visit2_steps),
        "visit3plus_steps": float(env.visit3plus_steps),
        "visit4plus_steps": float(env.visit4plus_steps),
        "max_visit": float(env.max_visit),
        "bfs_agree": float(bfs_agree / max(1, env.steps)),
        "bfs_gap": float(max(0, env.steps - (int(env.dist[env.start]) if int(env.dist[env.start]) < 10_000 else args.max_steps)) if success else args.max_steps - (int(env.dist[env.start]) if int(env.dist[env.start]) < 10_000 else args.max_steps)),
    }
    return metrics, debug, frames


def evaluate_policy(q: QNetwork, rm: RewardModel, args: argparse.Namespace, device: torch.device, n: int, include_test: bool = True) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for diff in ["easy", "medium", "hard"]:
        rows = [rollout_policy(q, rm, args, device, difficulty=diff, epsilon=0.0)[0] for _ in range(n)]
        out[diff] = aggregate_metrics(rows)
    if include_test:
        fixed = parse_maze_text(TEST_MAZE_TEXT)
        rows = [rollout_policy(q, rm, args, device, fixed=fixed, epsilon=0.0)[0] for _ in range(1)]
        out["test"] = aggregate_metrics(rows)
    return out


def aggregate_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = rows[0].keys() if rows else []
    return {k: safe_mean([r[k] for r in rows]) for k in keys}


def print_eval(eval_dict: Dict[str, Dict[str, float]]) -> None:
    print("\n==== EVALUATION ====")
    for name, m in eval_dict.items():
        print(f"{name}")
        print(f"  outcome : success={fmt_pct(m['success'])} explore_timeout={fmt_pct(m['explore_timeout'])} wall_timeout={fmt_pct(m['wall_timeout'])}")
        print(f"  wall    : wallStep={m['wall_step_rate']:.3f} wallHits={m['wall_hits']:.2f}")
        print(f"  visits  : visit2={m['visit2_steps']:.2f} visit3+={m['visit3plus_steps']:.2f} visit4+={m['visit4plus_steps']:.2f} maxVisit={m['max_visit']:.2f}")
        print(f"  path    : bfsAgree={m['bfs_agree']:.3f} bfsGap={m['bfs_gap']:.2f} steps={m['steps']:.2f}")
        print(f"  reward  : {m['reward']:.2f}")


def overall_metrics(eval_dict: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    weights = {"easy": 0.4, "medium": 0.3, "hard": 0.3}
    out = {}
    for key in ["success", "wall_step_rate", "visit4plus_steps", "bfs_gap", "reward"]:
        out[key] = sum(weights[d] * eval_dict[d][key] for d in weights if d in eval_dict)
    return out


def train_qcnn(rm: RewardModel, args: argparse.Namespace, device: torch.device, out_dir: Path) -> Tuple[QNetwork, Dict[str, Any]]:
    q = QNetwork().to(device)
    target = QNetwork().to(device)
    target.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=args.q_lr)
    replay = ReplayBuffer(args.replay_size)
    hist: Dict[str, List[float]] = defaultdict(list)
    best = {
        "success": {"key": (-1e9,), "path": out_dir / f"{VERSION}_qcnn_best_success.pt"},
        "safe": {"key": -1e9, "path": out_dir / f"{VERSION}_qcnn_best_safe.pt"},
        "hard": {"key": -1e9, "path": out_dir / f"{VERSION}_qcnn_best_hard.pt"},
    }
    total_updates = 0
    print("\n==== QCNN TRAINING ====")
    for ep in range(1, args.episodes + 1):
        diff = sample_difficulty(args)
        env = MazeEnv(args, difficulty=diff)
        state = env.state()
        eps = epsilon_by_episode(ep, args)
        ep_reward = 0.0
        losses = []
        bfs_agree = 0
        for _ in range(args.max_steps):
            if random.random() < eps:
                action = random.randrange(4)
            else:
                with torch.no_grad():
                    qs = q(torch.from_numpy(state[None].astype(np.float32)).to(device))
                    action = int(qs.argmax(dim=1).item())
            next_state, done, info = env.step(action)
            r = rm_reward_for_transition(rm, state, action, next_state, device, args.rm_scale_used)
            replay.push(state, action, r, next_state, done)
            ep_reward += r
            if info["delta_bfs"] > 0:
                bfs_agree += 1
            state = next_state
            if len(replay) >= args.warmup_steps:
                for _u in range(args.updates_per_step):
                    loss = optimize_q(q, target, replay, opt, args, device)
                    if not math.isnan(loss):
                        losses.append(loss)
                        total_updates += 1
                        if total_updates % args.target_update_interval == 0:
                            target.load_state_dict(q.state_dict())
            if done:
                break
        success = env.pos == env.goal
        hist["reward"].append(ep_reward)
        hist["success"].append(float(success))
        hist["wall_step_rate"].append(env.wall_hits / max(1, env.steps))
        hist["wall_hits"].append(float(env.wall_hits))
        hist["visit2_steps"].append(float(env.visit2_steps))
        hist["visit3plus_steps"].append(float(env.visit3plus_steps))
        hist["visit4plus_steps"].append(float(env.visit4plus_steps))
        hist["max_visit"].append(float(env.max_visit))
        hist["bfs_agree"].append(bfs_agree / max(1, env.steps))
        start_d = int(env.dist[env.start]) if int(env.dist[env.start]) < 10_000 else args.max_steps
        hist["bfs_gap"].append(float(max(0, env.steps - start_d) if success else args.max_steps - start_d))
        hist["steps"].append(float(env.steps))
        hist["loss"].append(safe_mean(losses))
        if ep % args.log_interval == 0:
            w = min(args.log_interval, len(hist["reward"]))
            print(f"\n[QCNN ep {ep:4d}/{args.episodes}] eps={eps:.3f}")
            print(f"  outcome : succ={fmt_pct(safe_mean(hist['success'][-w:]))} cleanTO={fmt_pct(safe_mean([1.0 if (s==0 and wh==0) else 0.0 for s,wh in zip(hist['success'][-w:], hist['wall_hits'][-w:])]))} wallTO={fmt_pct(safe_mean([1.0 if (s==0 and wh>0) else 0.0 for s,wh in zip(hist['success'][-w:], hist['wall_hits'][-w:])]))}")
            print(f"  wall    : wallStep={safe_mean(hist['wall_step_rate'][-w:]):.3f} avgWallHits={safe_mean(hist['wall_hits'][-w:]):.2f}")
            print(f"  visits  : visit2={safe_mean(hist['visit2_steps'][-w:]):.2f} visit3+={safe_mean(hist['visit3plus_steps'][-w:]):.2f} visit4+={safe_mean(hist['visit4plus_steps'][-w:]):.2f} maxVisit={safe_mean(hist['max_visit'][-w:]):.2f}")
            print(f"  path    : bfsAgree={safe_mean(hist['bfs_agree'][-w:]):.3f} bfsGap={safe_mean(hist['bfs_gap'][-w:]):.2f} steps={safe_mean(hist['steps'][-w:]):.2f}")
            print(f"  reward  : avg={safe_mean(hist['reward'][-w:]):.2f} loss={safe_mean([x for x in hist['loss'][-w:] if not math.isnan(x)]):.4f}")
            print(f"  best    : success={best['success']['key']} safe={best['safe']['key']:.3f} hard={best['hard']['key']:.3f}")
        if ep % args.eval_interval == 0 or ep == args.episodes:
            eval_dict = evaluate_policy(q, rm, args, device, n=args.eval_n, include_test=False)
            om = overall_metrics(eval_dict)
            success_key = (om["success"], -om["wall_step_rate"], -om["visit4plus_steps"], -om["bfs_gap"], om["reward"])
            safe_score = om["success"] - 0.5 * om["wall_step_rate"] - 0.25 * (om["visit4plus_steps"] / args.max_steps)
            hard_score = eval_dict["hard"]["success"] - 0.5 * eval_dict["hard"]["wall_step_rate"] - 0.25 * (eval_dict["hard"]["visit4plus_steps"] / args.max_steps)
            if success_key > best["success"]["key"]:
                best["success"]["key"] = success_key
                torch.save({"model": q.state_dict(), "args": vars(args), "eval": eval_dict, "key": success_key}, best["success"]["path"])
                with open(out_dir / "best_success_eval.json", "w", encoding="utf-8") as f:
                    json.dump(eval_dict, f, indent=2)
                print(f"[save] best_success ep={ep} key={success_key}")
            if safe_score > best["safe"]["key"]:
                best["safe"]["key"] = safe_score
                torch.save({"model": q.state_dict(), "args": vars(args), "eval": eval_dict, "score": safe_score}, best["safe"]["path"])
                with open(out_dir / "best_safe_eval.json", "w", encoding="utf-8") as f:
                    json.dump(eval_dict, f, indent=2)
                print(f"[save] best_safe ep={ep} score={safe_score:.4f}")
            if hard_score > best["hard"]["key"]:
                best["hard"]["key"] = hard_score
                torch.save({"model": q.state_dict(), "args": vars(args), "eval": eval_dict, "score": hard_score}, best["hard"]["path"])
                with open(out_dir / "best_hard_eval.json", "w", encoding="utf-8") as f:
                    json.dump(eval_dict, f, indent=2)
                print(f"[save] best_hard ep={ep} score={hard_score:.4f}")
    last_path = out_dir / f"{VERSION}_qcnn_last.pt"
    torch.save({"model": q.state_dict(), "args": vars(args)}, last_path)
    hist_obj = {"history": {k: list(v) for k, v in hist.items()}, "best": {k: str(v["path"]) for k, v in best.items()}, "last_path": str(last_path)}
    with open(out_dir / f"{VERSION}_qcnn_history.json", "w", encoding="utf-8") as f:
        json.dump(hist_obj, f, indent=2)
    plot_q_curves(hist, out_dir / f"{VERSION}_qcnn_curves.png")
    eval_dict = evaluate_policy(q, rm, args, device, n=args.eval_n, include_test=True)
    print_eval(eval_dict)
    with open(out_dir / f"{VERSION}_qcnn_eval.json", "w", encoding="utf-8") as f:
        json.dump(eval_dict, f, indent=2)
    return q, hist_obj


def plot_q_curves(hist: Dict[str, List[float]], path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    panels = [
        ("reward", "Reward moving average"),
        ("success", "Success moving average"),
        ("wall_step_rate", "Wall-step moving average"),
        ("bfs_gap", "BFS-gap moving average"),
        ("visit4plus_steps", "visit4+ moving average"),
        ("loss", "Loss moving average"),
    ]
    for ax, (key, title) in zip(axes.ravel(), panels):
        vals = hist.get(key, [])
        ax.plot(moving_average(vals, 50) if vals else [])
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

# -----------------------------
# Test GIF
# -----------------------------

def render_maze_frame(grid: np.ndarray, pos: Tuple[int, int], goal: Tuple[int, int], visit_counts: np.ndarray, title: str = "") -> np.ndarray:
    img = np.ones((SIZE, SIZE, 3), dtype=np.float32)
    img[grid == 1] = np.array([0.02, 0.02, 0.02])
    # visit tint
    vc = np.minimum(visit_counts, 8) / 8.0
    img[:, :, 1] = np.minimum(img[:, :, 1], 1.0 - 0.35 * vc)
    img[goal] = np.array([1.0, 0.85, 0.0])
    img[pos] = np.array([1.0, 0.1, 0.1])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, SIZE, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, SIZE, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba())
    arr = np.asarray(buf[:, :, :3], dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return arr


def save_gif(frames: List[np.ndarray], path: Path, fps: int = 3) -> None:
    if not frames:
        return
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.axis("off")
    im = ax.imshow(frames[0])
    def update(i: int):
        im.set_data(frames[i])
        return [im]
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=int(1000 / max(1, fps)), blit=True)
    ani.save(path, writer="pillow")
    plt.close(fig)


def load_q_model(path: str | Path, device: torch.device) -> QNetwork:
    q = QNetwork().to(device)
    ckpt = torch.load(path, map_location=device)
    q.load_state_dict(ckpt.get("model", ckpt))
    q.eval()
    return q


def run_test(args: argparse.Namespace, device: torch.device, out_dir: Path) -> None:
    if not hasattr(args, "rm_scale_used"):
        args.rm_scale_used = float(args.rm_scale)
    rm_path = args.reward_model or str(out_dir / f"{VERSION}_reward_model_best_loss.pt")
    q_path = args.model or str(out_dir / f"{VERSION}_qcnn_best_success.pt")
    if not Path(rm_path).exists():
        raise FileNotFoundError(f"Reward model not found: {rm_path}")
    if not Path(q_path).exists():
        raise FileNotFoundError(f"Q model not found: {q_path}")
    rm = load_reward_model(rm_path, device)
    q = load_q_model(q_path, device)
    fixed = parse_maze_text(TEST_MAZE_TEXT)
    metrics, debug, frames = rollout_policy(q, rm, args, device, fixed=fixed, collect_debug=True)
    gif_path = out_dir / f"{VERSION}_test_rollout.gif"
    json_path = out_dir / f"{VERSION}_test_rollout_debug.json"
    save_gif(frames, gif_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "steps": debug}, f, indent=2)
    print("\n==== TEST ROLLOUT ====")
    print(json.dumps(metrics, indent=2))
    print(f"[Save] gif: {gif_path}")
    print(f"[Save] rollout json: {json_path}")

# -----------------------------
# Argument parsing and run dirs
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=VERSION)
    # mode and paths
    p.add_argument("--mode", choices=["all", "train-rm", "train-q", "debug-rm"], default="all")
    p.add_argument("--test", action="store_true")
    p.add_argument("--output-dir", default="./v2.0.7b")
    p.add_argument("--run-name", default="default")
    p.add_argument("--reward-model", default="")
    p.add_argument("--model", default="")
    p.add_argument("--skip-rm", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    # environment
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--easy-ratio", type=float, default=0.4)
    p.add_argument("--medium-ratio", type=float, default=0.3)
    p.add_argument("--hard-ratio", type=float, default=0.3)
    # visit
    p.add_argument("--visit-count-cap", type=float, default=8.0)
    p.add_argument("--visit-second-penalty", type=float, default=0.10)
    p.add_argument("--visit-exp-scale", type=float, default=0.25)
    p.add_argument("--visit-exp-base", type=float, default=1.60)
    p.add_argument("--visit-penalty-cap", type=float, default=4.0)
    p.add_argument("--visit-tier-mid-threshold", type=int, default=4)
    # dataset
    p.add_argument("--rm-mazes", type=int, default=400)
    p.add_argument("--families-per-maze", type=int, default=3)
    p.add_argument("--random-behavior-ratio", type=float, default=0.20)
    p.add_argument("--structured-family-ratio", type=float, default=0.80)
    p.add_argument("--max-bfs-paths-per-maze", type=int, default=8)
    p.add_argument("--family-prefix-min-ratio", type=float, default=0.20)
    p.add_argument("--family-prefix-max-ratio", type=float, default=0.65)
    p.add_argument("--generator-retry", type=int, default=40)
    p.add_argument("--success-explore-max-wall", type=int, default=2)
    p.add_argument("--success-explore-max-visit4plus", type=int, default=4)
    p.add_argument("--success-explore-max-extra", type=int, default=6)
    p.add_argument("--recovery-max-visit4plus", type=int, default=4)
    p.add_argument("--recovery-max-wall", type=int, default=1)
    p.add_argument("--clean-timeout-max-visit3plus", type=int, default=2)
    p.add_argument("--loop-min-visit3plus", type=int, default=8)
    p.add_argument("--loop-min-max-visit", type=int, default=4)
    p.add_argument("--repeated-wall-min-hits", type=int, default=8)
    p.add_argument("--repeated-wall-min-max-visit", type=int, default=4)
    # pairs
    p.add_argument("--pairs-per-maze", type=int, default=36)
    p.add_argument("--pairs-per-family", type=int, default=12)
    p.add_argument("--bfs-pair-cap-per-maze", type=int, default=4)
    # margins
    p.add_argument("--margin-outcome-min", type=float, default=0.65)
    p.add_argument("--margin-outcome-max", type=float, default=0.85)
    p.add_argument("--margin-outcome-s", type=float, default=1.00)
    p.add_argument("--margin-wall-min", type=float, default=0.30)
    p.add_argument("--margin-wall-max", type=float, default=0.55)
    p.add_argument("--margin-wall-s", type=float, default=1.00)
    p.add_argument("--margin-revisit-min", type=float, default=0.18)
    p.add_argument("--margin-revisit-max", type=float, default=0.40)
    p.add_argument("--margin-revisit-s", type=float, default=0.20)
    p.add_argument("--margin-path-min", type=float, default=0.02)
    p.add_argument("--margin-path-max", type=float, default=0.15)
    p.add_argument("--margin-path-s", type=float, default=0.25)
    p.add_argument("--margin-cpl-recover-wall-min", type=float, default=0.35)
    p.add_argument("--margin-cpl-recover-wall-max", type=float, default=0.60)
    p.add_argument("--margin-cpl-recover-wall-s", type=float, default=0.20)
    p.add_argument("--margin-cpl-explore-wall-min", type=float, default=0.28)
    p.add_argument("--margin-cpl-explore-wall-max", type=float, default=0.50)
    p.add_argument("--margin-cpl-explore-wall-s", type=float, default=0.20)
    p.add_argument("--margin-cpl-recover-loop-min", type=float, default=0.22)
    p.add_argument("--margin-cpl-recover-loop-max", type=float, default=0.42)
    p.add_argument("--margin-cpl-recover-loop-s", type=float, default=0.15)
    # RM training
    p.add_argument("--rm-epochs", type=int, default=6)
    p.add_argument("--rm-batch-size", type=int, default=64)
    p.add_argument("--rm-lr", type=float, default=1e-3)
    p.add_argument("--reward-l2", type=float, default=0.01)
    p.add_argument("--score-normalizer", choices=["bfs_len", "episode_len", "raw"], default="episode_len")
    p.add_argument("--debug-probe-mazes", type=int, default=30)
    p.add_argument("--stop-if-rm-diagnostic-fails", action="store_true")
    p.add_argument("--rm-min-raw-std", type=float, default=0.05)
    p.add_argument("--rm-scale-cap", type=float, default=10.0)
    p.add_argument("--stop-if-rm-std-too-low", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stop-if-rm-scale-too-high", action=argparse.BooleanOptionalAction, default=True)
    # Q training
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--q-lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--replay-size", type=int, default=100000)
    p.add_argument("--warmup-steps", type=int, default=3000)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--target-update-interval", type=int, default=500)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-episodes", type=int, default=2000)
    p.add_argument("--eval-n", type=int, default=100)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--rm-scale", type=float, default=1.0)
    p.add_argument("--auto-rm-scale", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--target-rm-std", type=float, default=1.0)
    return p


def auto_run_name(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    if args.run_name != "default":
        return args.run_name
    defaults = vars(parser.parse_args([]))
    ignore = {"mode", "test", "run_name", "output_dir", "device", "reward_model", "model", "skip_rm"}
    changed = []
    for k, v in vars(args).items():
        if k in ignore:
            continue
        if k in defaults and v != defaults[k]:
            changed.append(f"{k}-{str(v).replace('.', 'p')}")
    return "default" if not changed else "__".join(changed[:8])


def estimate_rm_scale(model: RewardModel, trajectories: List[Trajectory], args: argparse.Namespace, device: torch.device) -> float:
    if not args.auto_rm_scale:
        return float(args.rm_scale)
    vals: List[float] = []
    sample_trajs = random.sample(trajectories, min(len(trajectories), 200)) if trajectories else []
    model.eval()
    for tr in sample_trajs:
        if not tr.transitions:
            continue
        x = rm_inputs_for_traj(tr, device)
        with torch.no_grad():
            vals.extend(model(x).detach().cpu().numpy().tolist())
    std = float(np.std(vals)) if vals else 0.0
    scale = float(args.target_rm_std / (std + 1e-6)) if std > 0 else float("inf")
    print("\n[RM Collapse Check]")
    print(f"raw_rm_std={std:.6f}  target={args.target_rm_std:.4f}  proposed_scale={scale:.4f}  cap={args.rm_scale_cap:.4f}")
    if std < args.rm_min_raw_std:
        msg = f"RM collapsed: raw_rm_std={std:.6f} < rm_min_raw_std={args.rm_min_raw_std:.6f}. QCNN training skipped."
        print(f"[STOP] {msg}" if args.stop_if_rm_std_too_low else f"[WARN] {msg}")
        if args.stop_if_rm_std_too_low:
            raise RuntimeError(msg)
    if scale > args.rm_scale_cap:
        msg = f"RM scale too high: {scale:.4f} > rm_scale_cap={args.rm_scale_cap:.4f}."
        print(f"[STOP] {msg}" if args.stop_if_rm_scale_too_high else f"[WARN] {msg} Clipping scale.")
        if args.stop_if_rm_scale_too_high:
            raise RuntimeError(msg)
        scale = float(args.rm_scale_cap)
    print(f"[RSC] auto_rm_scale=True raw_rm_std={std:.4f} target={args.target_rm_std:.4f} scale={scale:.4f}")
    return scale

# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    run_name = auto_run_name(args, parser)
    out_dir = ensure_dir(Path(args.output_dir) / run_name)
    device = get_device(args.device)
    print(f"\n=== {VERSION} ===")
    print(f"Device: {device}")
    print(f"Output: {out_dir}")
    print("Kept: two-stage RM->QCNN, --skip-rm, --reward-model, best checkpoints, --test GIF, visit map, Double DQN.")
    print("Changed: episode_len RM score normalization, 6-branch trajectory families, lighter CPL margins, collapse-gated RSC.")
    if args.test:
        run_test(args, device, out_dir)
        return
    rm_model: RewardModel
    trajectories: List[Trajectory] = []
    if args.skip_rm or args.mode == "train-q":
        if not args.reward_model:
            raise ValueError("--skip-rm or --mode train-q requires --reward-model")
        rm_model = load_reward_model(args.reward_model, device)
        print(f"[Load] reward model: {args.reward_model}")
    else:
        trajectories, pairs, dataset_summary = build_preference_dataset(args)
        with open(out_dir / f"{VERSION}_dataset_debug.json", "w", encoding="utf-8") as f:
            json.dump(dataset_summary, f, indent=2)
        rm_model, rm_hist = train_reward_model(pairs, args, device, out_dir)
        # Load best-loss checkpoint for downstream unless user wants last. Best-loss is the conservative default.
        best_loss_path = rm_hist.get("best_loss_path", "")
        if best_loss_path and Path(best_loss_path).exists():
            rm_model = load_reward_model(best_loss_path, device)
            print(f"[Load] downstream RM from best loss: {best_loss_path}")
        probe_step_rewards(rm_model, args, device, out_dir)
        if args.mode in ["train-rm", "debug-rm"]:
            return
    # Need trajectories for auto scale. If skipped, build a small probe dataset quickly.
    if not trajectories:
        old_rm_mazes = args.rm_mazes
        args.rm_mazes = min(50, max(10, old_rm_mazes))
        trajectories, _, _ = build_preference_dataset(args)
        args.rm_mazes = old_rm_mazes
    args.rm_scale_used = estimate_rm_scale(rm_model, trajectories, args, device)
    if args.mode in ["all", "train-q"]:
        q, qhist = train_qcnn(rm_model, args, device, out_dir)


if __name__ == "__main__":
    main()
