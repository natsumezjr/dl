"""
maze_dqn_v2_0_7debug_rm_context_probe_fixed.py

v2.0.7debug: RM-only context probe for family episode-normalized preference reward model


Patch marker for user-visible renamed build:
- fixed wall_timeout validator pollution
- widened wall_exp / CPL margin scales
- added pure_argmax RM greedy probe outputs

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

VERSION = "v2.0.8_local_segment_preference_rm_patched"
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
    b_wall_exp: float
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
    meta: Dict[str, Any] = field(default_factory=dict)


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
    return weighted_choice({"easy": getattr(args, "difficulty_easy", None) if getattr(args, "difficulty_easy", None) is not None else args.easy_ratio, "medium": getattr(args, "difficulty_medium", None) if getattr(args, "difficulty_medium", None) is not None else args.medium_ratio, "hard": getattr(args, "difficulty_hard", None) if getattr(args, "difficulty_hard", None) is not None else args.hard_ratio})

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
    alpha = float(getattr(args, "wall_exp_alpha", 4.0))
    b_wall_exp = (math.exp(alpha * b_wall_count) - 1.0) / max(1e-9, (math.exp(alpha) - 1.0))
    b_visit = float(visit_penalty_norm)
    b_total = b_outcome + b_wall_presence + b_wall_exp + b_visit + 0.5 * b_path
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
        b_wall_exp=float(b_wall_exp),
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
        ok = (not tr.success) and tr.wall_hits == 0 and tr.visit3plus_steps < args.loop_min_visit3plus and tr.max_visit < args.loop_min_max_visit
    elif label == "revisit_timeout":
        ok = (not tr.success) and tr.wall_hits == 0 and tr.visit2_steps >= args.revisit_timeout_min_visit2
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



def generate_revisit_timeout_branch(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    prefix: List[int],
    args: argparse.Namespace,
    maze_id: int,
    family_id: int,
) -> Optional[Trajectory]:
    """High visit2 timeout that is not necessarily a severe visit3+ loop."""
    branch = action_path_end(grid, start, prefix)
    for _ in range(args.generator_retry):
        actions = list(prefix)
        cur = branch
        # Create several out-and-back excursions from branch to generate visit2 without wall.
        free_as = [a for a in valid_actions(grid, cur) if apply_action(cur, a) != goal]
        if not free_as:
            continue
        excursions = random.randint(max(2, args.revisit_timeout_min_visit2 // 2), max(3, args.revisit_timeout_min_visit2 + 2))
        for _j in range(excursions):
            a = random.choice(free_as)
            nb = apply_action(cur, a)
            if not is_free(grid, nb) or nb == goal:
                continue
            actions.extend([a, opposite_action(a)])
            # Occasionally move one safe step away and back to vary context.
            if random.random() < 0.35:
                side_as = [x for x in valid_actions(grid, cur) if apply_action(cur, x) != goal]
                if side_as:
                    b = random.choice(side_as)
                    actions.extend([b, opposite_action(b)])
        # Fill remaining with mostly non-wall random moves avoiding goal.
        cur = action_path_end(grid, start, actions)
        guard = 0
        while len(actions) < args.max_steps and guard < args.max_steps * 4:
            guard += 1
            acts = [a for a in valid_actions(grid, cur) if apply_action(cur, a) != goal]
            if not acts:
                break
            # Bias toward returning to earlier cells, but allow wandering.
            a = random.choice(acts)
            actions.append(a)
            cur = apply_action(cur, a)
        actions = actions[: args.max_steps]
        tr = run_actions_as_trajectory(grid, start, goal, actions, "revisit_timeout", args, maze_id, family_id, prefix, branch)
        if validate_trajectory(tr, "revisit_timeout", args, strict=False):
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
    """Generate a true wall-timeout branch.

    v2.0.7debug patch:
    - Never allow wall_timeout to become success.
    - Never validate repeated-wall with an unknown label.
    - Retry generation rather than silently accepting polluted samples.
    """
    base_branch = action_path_end(grid, start, prefix)
    for _attempt in range(max(1, int(args.generator_retry))):
        branch = base_branch
        prefix2 = list(prefix)
        walls = wall_actions(grid, branch)
        if not walls:
            # Move to a nearby free cell that has at least one wall action.
            candidates = valid_actions(grid, branch)
            random.shuffle(candidates)
            moved = False
            for a in candidates:
                nb = apply_action(branch, a)
                if nb == goal:
                    continue
                if wall_actions(grid, nb):
                    prefix2 = prefix + [a]
                    branch = nb
                    walls = wall_actions(grid, branch)
                    moved = True
                    break
            if not moved or not walls:
                continue

        wall_a = random.choice(walls)
        if repeated:
            # Sticky wall branch: stay on the same cell by repeatedly selecting the wall action.
            tail = [wall_a] * max(args.repeated_wall_min_hits + 4, args.max_steps - len(prefix2))
        else:
            # One/few wall hits, then non-goal wandering. Reject if it accidentally reaches goal.
            tail = [wall_a] * random.randint(1, 3)
            cur = branch
            for _ in range(max(0, args.max_steps - len(prefix2) - len(tail))):
                acts = valid_actions(grid, cur)
                random.shuffle(acts)
                chosen = None
                for a in acts:
                    nxt = apply_action(cur, a)
                    if nxt != goal:
                        chosen = a
                        break
                if chosen is None:
                    # If every valid action reaches goal, keep hitting the original wall.
                    chosen = wall_a
                    tail.append(chosen)
                    continue
                tail.append(chosen)
                cur = apply_action(cur, chosen)

        actions = (prefix2 + tail)[: args.max_steps]
        tr = run_actions_as_trajectory(grid, start, goal, actions, "wall_timeout", args, maze_id, family_id, prefix2, branch)
        if tr.success:
            continue
        if validate_trajectory(tr, "wall_timeout", args, strict=True):
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
        generate_revisit_timeout_branch,
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

BUCKETS = [
    # Long episode/family buckets.
    "episode_outcome", "episode_wall", "episode_loop", "episode_path",
    # Legacy aliases kept for compatibility with existing family construction.
    "outcome", "wall_presence", "wall_exp", "revisit_loop", "path",
    "cpl_recover_wall", "cpl_explore_wall", "cpl_recover_loop", "cpl_visit_toward",
    # Local segment buckets.
    "local_toward", "local_safe_wall", "local_visit2_direction",
    "local_recover_loop", "local_visit_penalty",
    # Failure-mined / failure-inspired local segment buckets.
    "failure_mined_wall", "failure_mined_away", "failure_mined_visit2",
    "failure_mined_loop", "failure_mined_saturated",
]


def better_worse(a: Trajectory, b: Trajectory) -> Optional[Tuple[Trajectory, Trajectory]]:
    if a.quality_key > b.quality_key:
        return a, b
    if b.quality_key > a.quality_key:
        return b, a
    return None


def component_for_bucket(tr: Trajectory, bucket: str) -> float:
    if bucket in ["outcome", "episode_outcome"]:
        return tr.b_outcome
    if bucket in ["wall_presence", "episode_wall"]:
        return tr.b_wall_presence
    if bucket == "wall_exp":
        return tr.b_wall_exp
    if bucket in ["revisit_loop", "episode_loop"]:
        return tr.b_visit
    if bucket in ["path", "episode_path", "local_toward", "local_visit2_direction", "failure_mined_away", "failure_mined_visit2", "failure_mined_saturated"]:
        return tr.b_path
    if bucket in ["local_safe_wall", "failure_mined_wall"]:
        return tr.b_wall_presence + tr.b_wall_exp
    if bucket in ["local_recover_loop", "failure_mined_loop", "local_visit_penalty"]:
        return tr.b_visit
    if bucket in ["cpl_recover_wall", "cpl_explore_wall"]:
        return tr.b_wall_exp + tr.b_visit
    if bucket == "cpl_recover_loop":
        return tr.b_visit
    if bucket == "cpl_visit_toward":
        return tr.b_visit + tr.b_path
    return tr.b_total


def bucket_margin_params(args: argparse.Namespace, bucket: str) -> Tuple[float, float, float]:
    if bucket == "outcome":
        return args.margin_outcome_min, args.margin_outcome_max, args.margin_outcome_s
    if bucket == "wall_presence":
        return args.margin_wall_presence_min, args.margin_wall_presence_max, args.margin_wall_presence_s
    if bucket == "wall_exp":
        return args.margin_wall_exp_min, args.margin_wall_exp_max, args.margin_wall_exp_s
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
    if bucket == "cpl_visit_toward":
        return args.margin_cpl_visit_toward_min, args.margin_cpl_visit_toward_max, args.margin_cpl_visit_toward_s
    if bucket == "episode_outcome":
        return args.episode_outcome_min, args.episode_outcome_max, args.episode_outcome_scale
    if bucket == "episode_wall":
        return args.episode_wall_min, args.episode_wall_max, args.episode_wall_scale
    if bucket == "episode_loop":
        return args.episode_loop_min, args.episode_loop_max, args.episode_loop_scale
    if bucket == "episode_path":
        return args.episode_path_min, args.episode_path_max, args.episode_path_scale
    if bucket == "local_toward":
        return args.local_toward_min, args.local_toward_max, args.local_toward_scale
    if bucket == "local_safe_wall":
        return args.local_safe_wall_min, args.local_safe_wall_max, args.local_safe_wall_scale
    if bucket == "local_visit2_direction":
        return args.local_visit2_direction_min, args.local_visit2_direction_max, args.local_visit2_direction_scale
    if bucket == "local_recover_loop":
        return args.local_recover_loop_min, args.local_recover_loop_max, args.local_recover_loop_scale
    if bucket == "local_visit_penalty":
        return args.local_visit_penalty_min, args.local_visit_penalty_max, args.local_visit_penalty_scale
    if bucket == "failure_mined_wall":
        return args.failure_mined_wall_min, args.failure_mined_wall_max, args.failure_mined_wall_scale
    if bucket == "failure_mined_away":
        return args.failure_mined_away_min, args.failure_mined_away_max, args.failure_mined_away_scale
    if bucket == "failure_mined_visit2":
        return args.failure_mined_visit2_min, args.failure_mined_visit2_max, args.failure_mined_visit2_scale
    if bucket == "failure_mined_loop":
        return args.failure_mined_loop_min, args.failure_mined_loop_max, args.failure_mined_loop_scale
    if bucket == "failure_mined_saturated":
        return args.failure_mined_saturated_min, args.failure_mined_saturated_max, args.failure_mined_saturated_scale
    return args.margin_path_min, args.margin_path_max, args.margin_path_s


def compute_margin(pos: Trajectory, neg: Trajectory, bucket: str, args: argparse.Namespace) -> Tuple[float, float]:
    b_pos = component_for_bucket(pos, bucket)
    b_neg = component_for_bucket(neg, bucket)
    delta = max(0.0, float(b_neg - b_pos))
    m_min, m_max, s = bucket_margin_params(args, bucket)
    margin = float(m_min + (m_max - m_min) * math.tanh(delta / (s + 1e-9)))
    return margin, delta


def first_action_meta(tr: Trajectory) -> Tuple[Optional[int], Optional[float], Optional[bool]]:
    if not tr.transitions:
        return None, None, None
    t0 = tr.transitions[0]
    # Store delta_bfs as next_dist - current_dist to match the pair-validity report convention.
    # Thus toward is usually negative. Internally RMTransition.delta_bfs uses current-next.
    return int(t0.action), float(t0.bfs_after - t0.bfs_before), bool(t0.is_wall)


def traj_progress_score(tr: Trajectory) -> float:
    if not tr.transitions:
        return 0.0
    # Positive means average progress toward the goal.
    return float(np.mean([float(t.delta_bfs) for t in tr.transitions]))


def make_pair_meta(pos: Trajectory, neg: Trajectory, bucket: str, source: str, margin: float, delta: float, args: argparse.Namespace) -> Dict[str, Any]:
    m_min, m_max, m_scale = bucket_margin_params(args, bucket)
    pos_a, pos_d, pos_w = first_action_meta(pos)
    neg_a, neg_d, neg_w = first_action_meta(neg)
    return {
        "bucket": bucket,
        "pos_label": pos.source,
        "neg_label": neg.source,
        "pos_kind": pos.source,
        "neg_kind": neg.source,
        "pos_outcome": pos.outcome,
        "neg_outcome": neg.outcome,
        "pos_steps": pos.steps,
        "neg_steps": neg.steps,
        "pos_wall_hits": pos.wall_hits,
        "neg_wall_hits": neg.wall_hits,
        "pos_visit2_steps": pos.visit2_steps,
        "neg_visit2_steps": neg.visit2_steps,
        "pos_visit3plus_steps": pos.visit3plus_steps,
        "neg_visit3plus_steps": neg.visit3plus_steps,
        "pos_visit4plus_steps": pos.visit4plus_steps,
        "neg_visit4plus_steps": neg.visit4plus_steps,
        "pos_max_visit": pos.max_visit,
        "neg_max_visit": neg.max_visit,
        "pos_final_bfs_dist": pos.final_bfs_dist,
        "neg_final_bfs_dist": neg.final_bfs_dist,
        "pos_progress_score": traj_progress_score(pos),
        "neg_progress_score": traj_progress_score(neg),
        "pos_first_action": pos_a,
        "neg_first_action": neg_a,
        "pos_first_delta_bfs": pos_d,
        "neg_first_delta_bfs": neg_d,
        "pos_first_is_wall": pos_w,
        "neg_first_is_wall": neg_w,
        "delta_b": float(delta),
        "margin": float(margin),
        "margin_min": float(m_min),
        "margin_max": float(m_max),
        "margin_scale": float(m_scale),
        "near_min": bool(margin <= m_min + 0.02),
        "near_max": bool(margin >= m_max - 0.02),
        "pair_source": source,
    }


def make_pair(pos: Trajectory, neg: Trajectory, bucket: str, source: str, args: argparse.Namespace) -> PreferencePair:
    m, d = compute_margin(pos, neg, bucket, args)
    meta = make_pair_meta(pos, neg, bucket, source, m, d, args)
    return PreferencePair(pos=pos, neg=neg, bucket=bucket, margin=m, delta_component=d, pair_source=source, meta=meta)


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
    """Family-level mandatory pair candidates for RM-only context debug.

    B=bfs_success, S=success_explore, R=recovery_success,
    E=explore_timeout, V=revisit_timeout, L=loop_timeout, W=wall_timeout.
    """
    cands: Dict[str, List[PreferencePair]] = {b: [] for b in BUCKETS}
    for fam in families:
        B = first_traj(fam, "bfs_success")
        S = first_traj(fam, "success_explore")
        R = first_traj(fam, "recovery_success")
        E = first_traj(fam, "explore_timeout")
        V = first_traj(fam, "revisit_timeout")
        L = first_traj(fam, "loop_timeout")
        W = first_traj(fam, "wall_timeout")
        # outcome: success exploration/recovery > failed exploration/revisit/loop.
        add_candidate_pair(cands, S, E, "episode_outcome", "family_mandatory", args)
        add_candidate_pair(cands, R, E, "episode_outcome", "family_mandatory", args)
        add_candidate_pair(cands, R, V, "episode_outcome", "family_mandatory", args)
        add_candidate_pair(cands, R, L, "episode_outcome", "family_mandatory", args)
        # wall: timeout with wall is worse than no-wall timeouts.
        add_candidate_pair(cands, E, W, "episode_wall", "family_mandatory", args)
        add_candidate_pair(cands, V, W, "episode_wall", "family_mandatory", args)
        add_candidate_pair(cands, L, W, "episode_wall", "family_mandatory", args)
        add_candidate_pair(cands, E, W, "wall_exp", "family_mandatory", args)
        add_candidate_pair(cands, V, W, "wall_exp", "family_mandatory", args)
        add_candidate_pair(cands, L, W, "wall_exp", "family_mandatory", args)
        # revisit / loop: recover > revisit fail > loop; success explore > loop.
        add_candidate_pair(cands, R, V, "episode_loop", "family_mandatory", args)
        add_candidate_pair(cands, R, L, "episode_loop", "family_mandatory", args)
        add_candidate_pair(cands, S, L, "episode_loop", "family_mandatory", args)
        add_candidate_pair(cands, V, L, "episode_loop", "family_mandatory", args)
        # path: weak BFS anchor.
        add_candidate_pair(cands, B, S, "episode_path", "family_mandatory", args)
        add_candidate_pair(cands, B, R, "episode_path", "family_mandatory", args)
        # CPL family-level counterfactuals.
        add_candidate_pair(cands, R, W, "cpl_recover_wall", "family_mandatory", args)
        add_candidate_pair(cands, S, W, "cpl_explore_wall", "family_mandatory", args)
        add_candidate_pair(cands, R, L, "cpl_recover_loop", "family_mandatory", args)
        add_candidate_pair(cands, R, V, "cpl_visit_toward", "family_mandatory", args)
    return cands

def sample_bucket_pairs(cands: Dict[str, List[PreferencePair]], args: argparse.Namespace) -> List[PreferencePair]:
    # v2.0.8 keeps family-level mandatory pairs and adds local segment pairs elsewhere.
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
        ("episode_outcome", 4, 0.50),
        ("episode_wall", 3, 0.25),
        ("episode_loop", 4, 0.14),
        ("episode_path", 2, 0.02),
        ("local_toward", 10, 0.08),
        ("local_safe_wall", 8, 0.25),
        ("local_visit2_direction", 8, 0.06),
        ("local_recover_loop", 8, 0.14),
        ("local_visit_penalty", 6, 0.03),
        ("failure_mined_wall", 2, 0.25),
        ("failure_mined_away", 2, 0.08),
        ("failure_mined_visit2", 2, 0.06),
        ("failure_mined_loop", 1, 0.14),
        ("failure_mined_saturated", 1, 0.05),
    ]
    pressures = [(name, n, m, n*m) for name, n, m in rows]
    total = sum(p for *_rest, p in pressures) or 1.0
    print("\n[Expected Gradient Pressure]")
    print(f"{'bucket':24s} {'n_per_family':>12s} {'typical_margin':>15s} {'pressure':>10s} {'pct':>8s}")
    for name, n, m, p in pressures:
        print(f"{name:24s} {n:12d} {m:15.3f} {p:10.3f} {100*p/total:7.2f}%")


def state_after_prefix(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], prefix: Sequence[int], args: argparse.Namespace) -> Tuple[Tuple[int, int], np.ndarray]:
    visit_counts = np.zeros((SIZE, SIZE), dtype=np.int32)
    pos = start
    visit_counts[pos] = 1
    for a in list(prefix)[: args.max_steps]:
        nxt = apply_action(pos, a)
        if is_free(grid, nxt):
            pos = nxt
        visit_counts[pos] += 1
        if pos == goal:
            break
    return pos, visit_counts


def run_segment_as_trajectory(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    prefix_actions: Sequence[int],
    segment_actions: Sequence[int],
    source: str,
    args: argparse.Namespace,
    maze_id: int,
    family_id: int,
) -> Trajectory:
    dist = bfs_distances(grid, goal)
    seg_start, visit_counts = state_after_prefix(grid, start, goal, prefix_actions, args)
    pos = seg_start
    transitions: List[RMTransition] = []
    wall_hits = revisit_steps = visit2_steps = visit3plus_steps = visit4plus_steps = 0
    visit_penalty_sum = 0.0
    success = False
    for a in list(segment_actions)[: args.local_segment_max_len]:
        state = encode_state(grid, pos, goal, visit_counts, args.visit_count_cap)
        before = pos
        bfs_before = int(dist[before]) if int(dist[before]) < 10000 else args.max_steps
        attempted = apply_action(pos, int(a))
        is_wall = not is_free(grid, attempted)
        nxt = pos if is_wall else attempted
        wall_hits += int(is_wall)
        visit_counts[nxt] += 1
        c_after = int(visit_counts[nxt])
        revisit_steps += int(c_after >= 2)
        visit2_steps += int(c_after == 2)
        visit3plus_steps += int(c_after >= 3)
        visit4plus_steps += int(c_after >= 4)
        visit_penalty_sum += visit_count_penalty(c_after, args)
        next_state = encode_state(grid, nxt, goal, visit_counts, args.visit_count_cap)
        bfs_after = int(dist[nxt]) if int(dist[nxt]) < 10000 else args.max_steps
        is_goal = nxt == goal
        transitions.append(RMTransition(state, int(a), next_state, bool(is_wall), bool(is_goal), before, nxt, bfs_before, bfs_after, bfs_before - bfs_after, c_after))
        pos = nxt
        if is_goal:
            success = True
            break
    steps = len(transitions)
    start_bfs_dist = int(dist[seg_start]) if int(dist[seg_start]) < 10000 else args.max_steps
    final_bfs_dist = 0 if success else (int(dist[pos]) if int(dist[pos]) < 10000 else args.max_steps)
    bfs_len = max(1, start_bfs_dist)
    gap = max(0, steps - bfs_len) if success else args.max_steps
    visit_penalty_norm = visit_penalty_sum / max(1e-9, float(args.visit_penalty_cap) * float(args.max_steps))
    return build_trajectory_stats(
        maze_id=maze_id, family_id=family_id, source=source, transitions=transitions, outcome=("success" if success else "segment"),
        success=success, steps=steps, bfs_len=bfs_len, start_bfs_dist=start_bfs_dist, final_bfs_dist=final_bfs_dist, gap=gap,
        wall_hits=wall_hits, wall_rate=wall_hits / max(1, steps), revisit_steps=revisit_steps, visit2_steps=visit2_steps,
        visit3plus_steps=visit3plus_steps, visit4plus_steps=visit4plus_steps, max_visit=int(visit_counts.max()),
        visit_penalty_sum=visit_penalty_sum, visit_penalty_norm=visit_penalty_norm, prefix_actions=list(prefix_actions), branch_pos=seg_start, args=args,
    )


def rollout_segment_policy(grid: np.ndarray, pos: Tuple[int, int], goal: Tuple[int, int], first_action: int, k: int, mode: str) -> List[int]:
    dist = bfs_distances(grid, goal)
    actions = [int(first_action)]
    cur = apply_action(pos, first_action) if is_free(grid, apply_action(pos, first_action)) else pos
    for _ in range(max(0, k - 1)):
        valid = valid_actions(grid, cur)
        if not valid:
            actions.append(0); continue
        if mode == "toward":
            best_d = min(int(dist[apply_action(cur, a)]) if is_free(grid, apply_action(cur, a)) else 10000 for a in valid)
            cand = [a for a in valid if int(dist[apply_action(cur, a)]) == best_d]
        elif mode == "away":
            best_d = max(int(dist[apply_action(cur, a)]) if is_free(grid, apply_action(cur, a)) else -1 for a in valid)
            cand = [a for a in valid if int(dist[apply_action(cur, a)]) == best_d]
        elif mode == "loop":
            cand = [opposite_action(actions[-1])] if actions else valid
            cand = [a for a in cand if a in valid] or valid
        else:
            cand = valid
        a = random.choice(cand)
        actions.append(a)
        nxt = apply_action(cur, a)
        if is_free(grid, nxt):
            cur = nxt
    return actions


def local_pair(pos: Trajectory, neg: Trajectory, bucket: str, args: argparse.Namespace, source: str) -> PreferencePair:
    return make_pair(pos, neg, bucket, source, args)


def generate_local_segment_pairs_for_maze(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], maze_id: int, bfs_paths: List[List[int]], args: argparse.Namespace) -> Tuple[List[Trajectory], List[PreferencePair]]:
    out_traj: List[Trajectory] = []
    out_pairs: List[PreferencePair] = []
    dist = bfs_distances(grid, goal)
    target_local = int(args.local_segment_pairs_per_maze)
    target_failure = 0  # true failure-mined pairs are added after warmup epochs
    attempts = 0
    while len(out_pairs) < target_local + target_failure and attempts < (target_local + target_failure + 10) * 80:
        attempts += 1
        ref = random.choice(bfs_paths)
        if len(ref) < 2:
            continue
        prefix_len = random.randint(0, max(0, len(ref) - 2))
        prefix = list(ref[:prefix_len])
        pos, vc = state_after_prefix(grid, start, goal, prefix, args)
        if pos == goal or int(dist[pos]) >= 10000:
            continue
        valid = valid_actions(grid, pos)
        walls = wall_actions(grid, pos)
        toward = [a for a in valid if int(dist[apply_action(pos, a)]) < int(dist[pos])]
        away = [a for a in valid if int(dist[apply_action(pos, a)]) >= int(dist[pos])]
        k = random.randint(args.local_segment_min_len, args.local_segment_max_len)
        bucket_cycle = len(out_pairs) % 10
        pair = None
        if bucket_cycle in (0, 1) and toward and away:
            ap, an = random.choice(toward), random.choice(away)
            tp = run_segment_as_trajectory(grid, start, goal, prefix, rollout_segment_policy(grid, pos, goal, ap, k, "toward"), "local_toward_pos", args, maze_id, 9000 + attempts)
            tn = run_segment_as_trajectory(grid, start, goal, prefix, rollout_segment_policy(grid, pos, goal, an, k, "away"), "local_toward_neg", args, maze_id, 9000 + attempts)
            pair = local_pair(tp, tn, "local_toward", args, "local_segment")
        elif bucket_cycle in (2, 3) and valid and walls:
            ap, an = random.choice(valid), random.choice(walls)
            tp = run_segment_as_trajectory(grid, start, goal, prefix, rollout_segment_policy(grid, pos, goal, ap, k, "away"), "local_safe_wall_pos", args, maze_id, 9000 + attempts)
            tn = run_segment_as_trajectory(grid, start, goal, prefix, [an] * k, "local_safe_wall_neg", args, maze_id, 9000 + attempts)
            pair = local_pair(tp, tn, "local_safe_wall", args, "local_segment")
        elif bucket_cycle in (4, 5):
            # Create a visit2 context by stepping out and back when possible.
            va = random.choice(valid) if valid else None
            if va is not None:
                back = opposite_action(va)
                pref2 = prefix + [va, back]
                pos2, _ = state_after_prefix(grid, start, goal, pref2, args)
                valid2 = valid_actions(grid, pos2)
                toward2 = [a for a in valid2 if int(dist[apply_action(pos2, a)]) < int(dist[pos2])]
                away2 = [a for a in valid2 if int(dist[apply_action(pos2, a)]) >= int(dist[pos2])]
                if toward2 and away2:
                    tp = run_segment_as_trajectory(grid, start, goal, pref2, rollout_segment_policy(grid, pos2, goal, random.choice(toward2), k, "toward"), "local_visit2_direction_pos", args, maze_id, 9000 + attempts)
                    tn = run_segment_as_trajectory(grid, start, goal, pref2, rollout_segment_policy(grid, pos2, goal, random.choice(away2), k, "away"), "local_visit2_direction_neg", args, maze_id, 9000 + attempts)
                    pair = local_pair(tp, tn, "local_visit2_direction", args, "local_segment")
        elif bucket_cycle in (6, 7) and valid:
            va = random.choice(valid); back = opposite_action(va)
            pref2 = prefix + [va, back]
            pos2, _ = state_after_prefix(grid, start, goal, pref2, args)
            valid2 = valid_actions(grid, pos2)
            toward2 = [a for a in valid2 if int(dist[apply_action(pos2, a)]) < int(dist[pos2])]
            if toward2:
                tp = run_segment_as_trajectory(grid, start, goal, pref2, rollout_segment_policy(grid, pos2, goal, random.choice(toward2), k, "toward"), "local_recover_loop_pos", args, maze_id, 9000 + attempts)
                tn = run_segment_as_trajectory(grid, start, goal, pref2, rollout_segment_policy(grid, pos2, goal, va if va in valid2 else random.choice(valid2), k, "loop"), "local_recover_loop_neg", args, maze_id, 9000 + attempts)
                pair = local_pair(tp, tn, "local_recover_loop", args, "local_segment")
        else:
            if valid:
                a = random.choice(valid)
                pref_low = prefix + [a, opposite_action(a)]
                pref_high = prefix + [a, opposite_action(a), a, opposite_action(a), a, opposite_action(a)]
                tp = run_segment_as_trajectory(grid, start, goal, pref_low, rollout_segment_policy(grid, pos, goal, a, k, "away"), "local_visit_penalty_pos", args, maze_id, 9000 + attempts)
                tn = run_segment_as_trajectory(grid, start, goal, pref_high, rollout_segment_policy(grid, pos, goal, a, k, "away"), "local_visit_penalty_neg", args, maze_id, 9000 + attempts)
                pair = local_pair(tp, tn, "local_visit_penalty", args, "local_segment")
        if pair is None:
            continue
        # Convert the last fraction into failure-mined buckets by relabeling analogous local pairs.
        if len(out_pairs) >= target_local:
            fmap = {
                "local_safe_wall": "failure_mined_wall",
                "local_toward": "failure_mined_away",
                "local_visit2_direction": "failure_mined_visit2",
                "local_recover_loop": "failure_mined_loop",
                "local_visit_penalty": "failure_mined_saturated",
            }
            new_bucket = fmap.get(pair.bucket, "failure_mined_saturated")
            pair = local_pair(pair.pos, pair.neg, new_bucket, args, "failure_mined_segment")
        out_traj.extend([pair.pos, pair.neg])
        out_pairs.append(pair)
    return out_traj, out_pairs


def generate_failure_mined_pairs_from_model(model: RewardModel, args: argparse.Namespace, device: torch.device) -> List[PreferencePair]:
    """Mine local counterfactual pairs from pure-argmax RM failures after warmup epochs."""
    if not args.enable_failure_mining or args.failure_mined_pairs_per_maze <= 0:
        return []
    mined: List[PreferencePair] = []
    model.eval()
    max_total = int(args.failure_mining_probe_mazes) * int(args.failure_mined_pairs_per_maze)
    for maze_id in tqdm(range(args.failure_mining_probe_mazes), desc="Failure mining", leave=False):
        difficulty = sample_difficulty(args)
        grid, start, goal = generate_maze(difficulty)
        dist = bfs_distances(grid, goal)
        pos = start
        visits = np.zeros((SIZE, SIZE), dtype=np.int32); visits[pos] = 1
        prefix: List[int] = []
        per_maze = 0
        for _t in range(args.rm_greedy_max_steps):
            state = encode_state(grid, pos, goal, visits, args.visit_count_cap)
            bfs_before = int(dist[pos]) if int(dist[pos]) < 10000 else args.max_steps
            infos = []
            for a in range(4):
                attempted = apply_action(pos, a)
                is_wall = not is_free(grid, attempted)
                nxt = pos if is_wall else attempted
                vc_after = int(visits[nxt]) + 1
                vv = visits.copy(); vv[nxt] += 1
                ns = encode_state(grid, nxt, goal, vv, args.visit_count_cap)
                r = rm_probe_reward_for_transition(model, state, a, ns, device)
                bfs_after = int(dist[nxt]) if int(dist[nxt]) < 10000 else args.max_steps
                infos.append({"a": a, "r": r, "is_wall": is_wall, "nxt": nxt, "delta": bfs_before - bfs_after, "vc_after": vc_after})
            chosen = max(infos, key=lambda z: z["r"])
            valid = [z for z in infos if not z["is_wall"]]
            toward = [z for z in valid if z["delta"] > 0]
            away = [z for z in valid if z["delta"] <= 0]
            all_sat = (max(z["r"] for z in infos) - min(z["r"] for z in infos)) < 0.05
            k = random.randint(args.local_segment_min_len, args.local_segment_max_len)
            pair = None
            fam_id = 100000 + maze_id * 1000 + _t
            if chosen["is_wall"] and valid:
                good = random.choice(toward or valid)
                tp = run_segment_as_trajectory(grid, start, goal, prefix, rollout_segment_policy(grid, pos, goal, good["a"], k, "toward"), "failure_mined_wall_pos", args, maze_id, fam_id)
                tn = run_segment_as_trajectory(grid, start, goal, prefix, [chosen["a"]] * k, "failure_mined_wall_neg", args, maze_id, fam_id)
                pair = local_pair(tp, tn, "failure_mined_wall", args, "failure_mined")
            elif chosen["delta"] <= 0 and toward:
                good = random.choice(toward)
                tp = run_segment_as_trajectory(grid, start, goal, prefix, rollout_segment_policy(grid, pos, goal, good["a"], k, "toward"), "failure_mined_away_pos", args, maze_id, fam_id)
                tn = run_segment_as_trajectory(grid, start, goal, prefix, rollout_segment_policy(grid, pos, goal, chosen["a"], k, "away"), "failure_mined_away_neg", args, maze_id, fam_id)
                bucket = "failure_mined_visit2" if visits[pos] >= 2 else "failure_mined_away"
                pair = local_pair(tp, tn, bucket, args, "failure_mined")
            elif all_sat and len(valid) >= 2:
                good = random.choice(toward or valid)
                bad = random.choice(away or valid)
                if good["a"] != bad["a"]:
                    tp = run_segment_as_trajectory(grid, start, goal, prefix, rollout_segment_policy(grid, pos, goal, good["a"], k, "toward"), "failure_mined_saturated_pos", args, maze_id, fam_id)
                    tn = run_segment_as_trajectory(grid, start, goal, prefix, rollout_segment_policy(grid, pos, goal, bad["a"], k, "away"), "failure_mined_saturated_neg", args, maze_id, fam_id)
                    pair = local_pair(tp, tn, "failure_mined_saturated", args, "failure_mined")
            elif visits[pos] >= 3 and toward and away:
                tp = run_segment_as_trajectory(grid, start, goal, prefix, rollout_segment_policy(grid, pos, goal, random.choice(toward)["a"], k, "toward"), "failure_mined_loop_pos", args, maze_id, fam_id)
                tn = run_segment_as_trajectory(grid, start, goal, prefix, rollout_segment_policy(grid, pos, goal, random.choice(away)["a"], k, "loop"), "failure_mined_loop_neg", args, maze_id, fam_id)
                pair = local_pair(tp, tn, "failure_mined_loop", args, "failure_mined")
            if pair is not None:
                mined.append(pair); per_maze += 1
                if per_maze >= args.failure_mined_pairs_per_maze or len(mined) >= max_total:
                    break
            # execute chosen action to continue mining rollout
            nxt = chosen["nxt"]
            visits[nxt] += 1
            prefix.append(int(chosen["a"]))
            pos = nxt
            if pos == goal:
                break
        if len(mined) >= max_total:
            break
    print(f"[Failure Mining] added {len(mined)} local counterfactual pairs after warmup.")
    return mined

# -----------------------------
# Dataset reports
# -----------------------------

def fmt_pct(x: float) -> str:
    return f"{100.0 * x:6.2f}%"



def _num_or_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)
    except Exception:
        return None


def _mean_meta(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [_num_or_none(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    return safe_mean(vals)


def _rate_meta(rows: List[Dict[str, Any]], key: str, pred) -> float:
    vals = [_num_or_none(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return 0.0
    return safe_mean([1.0 if pred(v) else 0.0 for v in vals])


def _side_stats(rows: List[Dict[str, Any]], side: str) -> Dict[str, float]:
    return {
        "success_rate": _rate_meta(rows, f"{side}_outcome_success", lambda v: v > 0.5),
        "wall_mean": _mean_meta(rows, f"{side}_wall_hits"),
        "visit2_mean": _mean_meta(rows, f"{side}_visit2_steps"),
        "visit3plus_mean": _mean_meta(rows, f"{side}_visit3plus_steps"),
        "visit4plus_mean": _mean_meta(rows, f"{side}_visit4plus_steps"),
        "max_visit_mean": _mean_meta(rows, f"{side}_max_visit"),
        "final_bfs_dist_mean": _mean_meta(rows, f"{side}_final_bfs_dist"),
        "progress_mean": _mean_meta(rows, f"{side}_progress_score"),
        "first_toward_rate": _rate_meta(rows, f"{side}_first_delta_bfs", lambda v: v < 0),
        "first_away_rate": _rate_meta(rows, f"{side}_first_delta_bfs", lambda v: v >= 0),
        "first_wall_rate": _rate_meta(rows, f"{side}_first_is_wall_num", lambda v: v > 0.5),
    }


def _pair_row_meta(p: PreferencePair) -> Dict[str, Any]:
    m = dict(p.meta) if getattr(p, "meta", None) else {}
    # Normalize fields for report aggregation.
    m["pos_outcome_success"] = 1.0 if p.pos.success else 0.0
    m["neg_outcome_success"] = 1.0 if p.neg.success else 0.0
    m["pos_first_is_wall_num"] = 1.0 if m.get("pos_first_is_wall") else 0.0 if m.get("pos_first_is_wall") is not None else None
    m["neg_first_is_wall_num"] = 1.0 if m.get("neg_first_is_wall") else 0.0 if m.get("neg_first_is_wall") is not None else None
    return m


def build_pair_direction_report(pairs: List[PreferencePair]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in pairs:
        m = _pair_row_meta(p)
        key = f"{p.bucket}|{m.get('pos_label', p.pos.source)}|{m.get('neg_label', p.neg.source)}"
        grouped[key].append(m)
    report: Dict[str, Any] = {}
    for key, rows in sorted(grouped.items()):
        bucket, pos_label, neg_label = key.split("|", 2)
        margins = [float(r.get("margin", 0.0)) for r in rows]
        deltas = [float(r.get("delta_b", 0.0)) for r in rows]
        report[key] = {
            "bucket": bucket,
            "pos_label": pos_label,
            "neg_label": neg_label,
            "n": len(rows),
            "margin": {
                "mean": safe_mean(margins),
                "p50": percentile(margins, 50),
                "p90": percentile(margins, 90),
                "pct_near_min": safe_mean([1.0 if r.get("near_min") else 0.0 for r in rows]),
                "pct_near_max": safe_mean([1.0 if r.get("near_max") else 0.0 for r in rows]),
            },
            "delta_b": {"mean": safe_mean(deltas), "p50": percentile(deltas, 50)},
            "pos_stats": _side_stats(rows, "pos"),
            "neg_stats": _side_stats(rows, "neg"),
        }
    return report


def validate_pair(p: PreferencePair, args: argparse.Namespace) -> Tuple[bool, List[str]]:
    m = _pair_row_meta(p)
    reasons: List[str] = []
    b = p.bucket
    pos_success = bool(p.pos.success)
    neg_success = bool(p.neg.success)
    pos_wall = int(p.pos.wall_hits)
    neg_wall = int(p.neg.wall_hits)
    pos_first_wall = bool(m.get("pos_first_is_wall")) if m.get("pos_first_is_wall") is not None else False
    neg_first_wall = bool(m.get("neg_first_is_wall")) if m.get("neg_first_is_wall") is not None else False
    pos_fd = _num_or_none(m.get("pos_first_delta_bfs"))
    neg_fd = _num_or_none(m.get("neg_first_delta_bfs"))
    pos_prog = float(m.get("pos_progress_score", 0.0))
    neg_prog = float(m.get("neg_progress_score", 0.0))
    pos_v2 = int(p.pos.visit2_steps)
    neg_v2 = int(p.neg.visit2_steps)
    pos_v3 = int(p.pos.visit3plus_steps)
    neg_v3 = int(p.neg.visit3plus_steps)
    pos_v4 = int(p.pos.visit4plus_steps)
    neg_v4 = int(p.neg.visit4plus_steps)

    if b == "episode_outcome":
        if not (pos_success and not neg_success):
            reasons.append("invalid_same_or_better_neg")
    elif b in {"episode_wall", "wall_exp", "local_safe_wall", "failure_mined_wall"}:
        if pos_wall > neg_wall or pos_wall > 0 and b.startswith("local"):
            reasons.append("invalid_pos_wall")
        if neg_wall <= 0:
            reasons.append("invalid_neg_no_wall_when_wall_expected")
        if b in {"local_safe_wall", "failure_mined_wall"}:
            if pos_first_wall:
                reasons.append("invalid_pos_wall")
            if not neg_first_wall:
                reasons.append("invalid_neg_no_wall_when_wall_expected")
    elif b in {"local_toward", "failure_mined_away"}:
        if pos_fd is None or not (pos_fd < 0):
            reasons.append("invalid_pos_not_toward")
        if neg_fd is None or not (neg_fd >= 0):
            reasons.append("invalid_neg_not_away")
        if not (pos_prog > neg_prog or (pos_fd is not None and neg_fd is not None and pos_fd < neg_fd)):
            reasons.append("invalid_progress_order")
    elif b in {"local_visit2_direction", "failure_mined_visit2"}:
        if pos_v2 <= 0 or neg_v2 <= 0:
            reasons.append("invalid_no_visit_context")
        if pos_fd is None or not (pos_fd < 0):
            reasons.append("invalid_pos_not_toward")
        if neg_fd is None or not (neg_fd >= 0):
            reasons.append("invalid_neg_not_away")
        if not (pos_prog > neg_prog):
            reasons.append("invalid_progress_order")
    elif b in {"local_recover_loop", "failure_mined_loop"}:
        if not (pos_prog > neg_prog):
            reasons.append("invalid_progress_order")
        if not (pos_v3 <= neg_v3 or pos_v4 <= neg_v4):
            reasons.append("invalid_visit_order")
    elif b == "local_visit_penalty":
        tol = float(getattr(args, "local_visit_penalty_progress_tolerance", 0.20))
        if abs(pos_prog - neg_prog) > tol:
            reasons.append("invalid_progress_order")
        if not (pos_v3 <= neg_v3 and pos_v4 <= neg_v4):
            reasons.append("invalid_visit_order")
    elif b == "failure_mined_saturated":
        if not (pos_prog >= neg_prog or pos_wall <= neg_wall):
            reasons.append("invalid_same_or_better_neg")
    return (len(reasons) == 0), reasons


def build_pair_validity_report(pairs: List[PreferencePair], args: argparse.Namespace) -> Dict[str, Any]:
    reason_keys = [
        "invalid_no_visit_context", "invalid_pos_not_toward", "invalid_neg_not_away",
        "invalid_pos_wall", "invalid_neg_no_wall_when_wall_expected", "invalid_progress_order",
        "invalid_visit_order", "invalid_same_or_better_neg",
    ]
    by_bucket: Dict[str, List[PreferencePair]] = defaultdict(list)
    for p in pairs:
        by_bucket[p.bucket].append(p)
    report: Dict[str, Any] = {}
    for b, arr in sorted(by_bucket.items()):
        counts = {k: 0 for k in reason_keys}
        valid = 0
        for p in arr:
            ok, reasons = validate_pair(p, args)
            if ok:
                valid += 1
            for r in reasons:
                if r in counts:
                    counts[r] += 1
        n = len(arr)
        report[b] = {
            "n": n,
            "valid_rate": float(valid / n) if n else 0.0,
            "invalid_rate": float((n - valid) / n) if n else 0.0,
            "invalid_reasons": counts,
        }
    return report


def print_pair_direction_report(report: Dict[str, Any], args: argparse.Namespace) -> List[str]:
    warnings: List[str] = []
    print("\n[Pair Direction Report]")
    print(f"{'bucket':24s} {'pos_label':28s} {'neg_label':28s} {'n':>6s} {'m_mean':>8s} {'m_p50':>8s} {'nearMax':>8s} {'dB_mean':>8s} {'posProg':>8s} {'negProg':>8s} {'posTow':>7s} {'negTow':>7s}")
    for key, r in sorted(report.items(), key=lambda kv: (kv[1]['bucket'], -kv[1]['n'])):
        ps = r["pos_stats"]; ns = r["neg_stats"]
        print(f"{r['bucket'][:24]:24s} {r['pos_label'][:28]:28s} {r['neg_label'][:28]:28s} {r['n']:6d} {r['margin']['mean']:8.3f} {r['margin']['p50']:8.3f} {fmt_pct(r['margin']['pct_near_max']):>8s} {r['delta_b']['mean']:8.3f} {ps['progress_mean']:8.3f} {ns['progress_mean']:8.3f} {fmt_pct(ps['first_toward_rate']):>7s} {fmt_pct(ns['first_toward_rate']):>7s}")
        if r["bucket"] in {"local_toward", "local_visit2_direction", "failure_mined_away", "failure_mined_visit2"} and ps["progress_mean"] < ns["progress_mean"]:
            msg = f"[PAIR WARNING] possible reversed pair direction in {r['bucket']}: pos_progress_mean < neg_progress_mean. key={key}"
            print(msg); warnings.append(msg)
    return warnings


def print_pair_validity_report(report: Dict[str, Any], args: argparse.Namespace) -> List[str]:
    warnings: List[str] = []
    threshold = float(getattr(args, "pair_valid_rate_warning_threshold", 0.80))
    print("\n[Pair Validity Report]")
    print(f"{'bucket':24s} {'n':>7s} {'valid':>9s} {'invalid':>9s} {'noVisit':>8s} {'posNotTow':>9s} {'negNotAway':>10s} {'posWall':>8s} {'negNoWall':>10s} {'progOrd':>8s} {'visitOrd':>8s} {'sameBetterNeg':>13s}")
    for b, r in sorted(report.items()):
        inv = r["invalid_reasons"]
        print(f"{b[:24]:24s} {r['n']:7d} {fmt_pct(r['valid_rate']):>9s} {fmt_pct(r['invalid_rate']):>9s} {inv['invalid_no_visit_context']:8d} {inv['invalid_pos_not_toward']:9d} {inv['invalid_neg_not_away']:10d} {inv['invalid_pos_wall']:8d} {inv['invalid_neg_no_wall_when_wall_expected']:10d} {inv['invalid_progress_order']:8d} {inv['invalid_visit_order']:8d} {inv['invalid_same_or_better_neg']:13d}")
        if r["valid_rate"] < threshold:
            msg = f"[PAIR WARNING] bucket={b} valid_rate={r['valid_rate']:.3f} is low. Fix pair generator before tuning margin."
            print(msg); warnings.append(msg)
        if b == "local_visit2_direction" and r["valid_rate"] < threshold:
            msg = "[PAIR WARNING] local_visit2_direction is not clean enough; toward_visit2 vs away_visit2 will not improve by margin tuning alone."
            print(msg); warnings.append(msg)
    return warnings

def print_dataset_reports(trajectories: List[Trajectory], pairs: List[PreferencePair], args: argparse.Namespace) -> Dict[str, Any]:
    print("\n==== RM DATASET SUMMARY ====")
    by_type: Dict[str, List[Trajectory]] = defaultdict(list)
    for t in trajectories:
        by_type[t.source].append(t)
    order = ["bfs_success", "success_explore", "recovery_success", "explore_timeout", "revisit_timeout", "loop_timeout", "wall_timeout"]
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
    print(f"{'type':26s} {'B_outcome':>10s} {'B_wallPres':>11s} {'B_wallCount':>12s} {'B_wallExp':>10s} {'B_visit':>9s} {'B_path':>9s} {'B_total':>9s}")
    for typ in order:
        arr = by_type.get(typ, [])
        if not arr:
            continue
        comps = {
            "B_outcome": safe_mean([t.b_outcome for t in arr]),
            "B_wallPres": safe_mean([t.b_wall_presence for t in arr]),
            "B_wallCount": safe_mean([t.b_wall_count for t in arr]),
            "B_wallExp": safe_mean([t.b_wall_exp for t in arr]),
            "B_visit": safe_mean([t.b_visit for t in arr]),
            "B_path": safe_mean([t.b_path for t in arr]),
            "B_total": safe_mean([t.b_total for t in arr]),
        }
        summary["badness"][typ] = comps
        print(f"{typ:26s} {comps['B_outcome']:10.3f} {comps['B_wallPres']:11.3f} {comps['B_wallCount']:12.3f} {comps['B_wallExp']:10.3f} {comps['B_visit']:9.3f} {comps['B_path']:9.3f} {comps['B_total']:9.3f}")
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
    pair_direction_report = build_pair_direction_report(pairs)
    pair_validity_report = build_pair_validity_report(pairs, args)
    pair_warnings: List[str] = []
    pair_warnings.extend(print_pair_direction_report(pair_direction_report, args))
    pair_warnings.extend(print_pair_validity_report(pair_validity_report, args))
    summary["pair_direction_report"] = pair_direction_report
    summary["pair_validity_report"] = pair_validity_report
    summary["pair_warning_summary"] = pair_warnings

    print("\n[Margin Saturation Check]")
    print(f"{'bucket':24s} {'pct_near_min':>14s} {'pct_near_max':>14s}")
    tuning: Dict[str, Any] = {}
    weak_buckets = {"episode_path", "path", "local_visit_penalty"}
    for b in BUCKETS:
        arr = by_bucket.get(b, [])
        if not arr:
            continue
        m_min, m_max, m_scale = bucket_margin_params(args, b if b != "random" else "path")
        margins = [p.margin for p in arr]
        deltas = [p.delta_component for p in arr]
        near_min = safe_mean([1.0 if p.margin <= m_min + 0.02 else 0.0 for p in arr])
        near_max = safe_mean([1.0 if p.margin >= m_max - 0.02 else 0.0 for p in arr])
        print(f"{b:24s} {fmt_pct(near_min):>14s} {fmt_pct(near_max):>14s}")
        action = "keep"; rec_min = m_min; rec_max = m_max; rec_scale = m_scale; reason = "margin distribution looks acceptable."
        valid_rate = pair_validity_report.get(b, {}).get("valid_rate", 1.0)
        if near_max > 0.40:
            if valid_rate < float(getattr(args, "pair_valid_rate_warning_threshold", 0.80)):
                action = "fix pair generator / pair direction before tuning margin"
                reason = "margin is saturating, but pair direction validity is low; increasing or decreasing margin may reinforce noisy preferences."
                print(f"[PAIR WARNING] bucket={b} margin is saturated but pair validity is low.")
            else:
                action = "increase scale or decrease max"; rec_scale = m_scale * 1.5; rec_max = m_max * 0.9
                reason = "margin is saturating; bucket pushes reward too hard and may cause +/-1 reward saturation."
                print(f"[WARN] bucket={b} has pct_near_max > 40%; margin may be saturating.")
        elif near_min > 0.70 and b not in weak_buckets:
            action = "decrease scale or increase min"; rec_scale = m_scale * 0.75; rec_min = m_min + 0.02
            reason = "margin too weak; pair may not provide enough separation."
        if b == "episode_path" and near_max > 0.10:
            action = "strongly weaken path margin"; rec_max = min(m_max, 0.02); rec_scale = m_scale * 2.0
            reason = "path margin may reintroduce length shortcut."
        tuning[b] = {
            "n": len(arr), "margin_min": m_min, "margin_max": m_max, "margin_scale": m_scale,
            "delta_b": {"mean": safe_mean(deltas), "p50": percentile(deltas,50), "p90": percentile(deltas,90)},
            "margin": {"mean": safe_mean(margins), "p10": percentile(margins,10), "p50": percentile(margins,50), "p90": percentile(margins,90), "pct_near_min": near_min, "pct_near_max": near_max},
            "recommendation": {"action": action, "recommended_min": rec_min, "recommended_max": rec_max, "recommended_scale": rec_scale, "reason": reason},
        }
    summary["margin_tuning_report"] = tuning
    print("\n[Margin Tunability Report]")
    print(f"{'bucket':24s} {'n':>7s} {'dB_p50':>8s} {'m_p50':>8s} {'nearMin':>9s} {'nearMax':>9s}  recommendation")
    for b, info in tuning.items():
        print(f"{b:24s} {info['n']:7d} {info['delta_b']['p50']:8.3f} {info['margin']['p50']:8.3f} {fmt_pct(info['margin']['pct_near_min']):>9s} {fmt_pct(info['margin']['pct_near_max']):>9s}  {info['recommendation']['action']}")
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
        episode_pairs = sample_bucket_pairs(cands, args)
        local_traj, local_pairs = generate_local_segment_pairs_for_maze(grid, start, goal, maze_id, bfs_paths, args)
        trajectories.extend(local_traj)
        pairs.extend(episode_pairs)
        pairs.extend(local_pairs)
        maze_summaries.append({"maze_id": maze_id, "difficulty": difficulty, "n_bfs_paths": len(bfs_paths), "n_family_traj": sum(len(f.trajectories) for f in families), "n_episode_pairs": len(episode_pairs), "n_local_pairs": len(local_pairs)})
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
    best_contextual_gap_path = out_dir / f"{VERSION}_reward_model_best_contextual_gap.pt"
    best_pure_argmax_path = out_dir / f"{VERSION}_reward_model_best_pure_argmax.pt"
    best_low_saturation_path = out_dir / f"{VERSION}_reward_model_best_low_saturation.pt"
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
        if args.enable_failure_mining and ep == args.failure_mining_after_epochs and args.failure_mined_pairs_per_maze > 0:
            mined = generate_failure_mined_pairs_from_model(model, args, device)
            if mined:
                pairs.extend(mined)
                idx = list(range(len(pairs)))
                random.shuffle(idx)
                split = int(len(idx) * 0.85)
                train_idx = idx[:split]
                val_idx = idx[split:]
                print(f"  [failure-mining] dataset expanded: pairs={len(pairs)} train={len(train_idx)} val={len(val_idx)}")
        if va_loss < best_loss:
            best_loss = va_loss
            torch.save({"model": model.state_dict(), "args": vars(args), "best_val_loss": best_loss}, best_loss_path)
            print(f"  [save] best loss -> {best_loss_path.name}")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model": model.state_dict(), "args": vars(args), "best_val_acc": best_acc}, best_acc_path)
            print(f"  [save] best acc  -> {best_acc_path.name}")
    torch.save({"model": model.state_dict(), "args": vars(args)}, last_path)
    # v2.0.8 model-selection placeholders. Full contextual and greedy metrics are computed after training;
    # these checkpoints intentionally preserve the final post-local-segment RM for those criteria.
    torch.save({"model": model.state_dict(), "args": vars(args), "selection_note": "final model saved for contextual gap selection"}, best_contextual_gap_path)
    torch.save({"model": model.state_dict(), "args": vars(args), "selection_note": "final model saved for pure argmax selection"}, best_pure_argmax_path)
    torch.save({"model": model.state_dict(), "args": vars(args), "selection_note": "final model saved for low saturation selection"}, best_low_saturation_path)
    hist_obj = {"history": hist, "best_val_loss": best_loss, "best_val_acc": best_acc, "best_loss_path": str(best_loss_path), "best_acc_path": str(best_acc_path), "best_contextual_gap_path": str(best_contextual_gap_path), "best_pure_argmax_path": str(best_pure_argmax_path), "best_low_saturation_path": str(best_low_saturation_path), "last_path": str(last_path)}
    with open(out_dir / f"{VERSION}_reward_model_history.json", "w", encoding="utf-8") as f:
        json.dump(hist_obj, f, indent=2)
    plot_rm_curves(hist, out_dir / f"{VERSION}_training_curves.png")
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



def _reward_stats(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan"), "pct_neg_one": float("nan"), "pct_pos_one": float("nan")}
    arr = np.array(vals, dtype=np.float64)
    return {
        "n": int(len(vals)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "pct_neg_one": float(np.mean(arr <= -0.98)),
        "pct_pos_one": float(np.mean(arr >= 0.98)),
    }


def probe_contextual_rewards(model: RewardModel, args: argparse.Namespace, device: torch.device, out_dir: Path) -> Dict[str, Any]:
    cats: Dict[str, List[float]] = defaultdict(list)
    model.eval()
    for maze_id in range(args.debug_probe_mazes):
        grid, start, goal = generate_maze(sample_difficulty(args))
        bfs_paths = enumerate_shortest_action_paths(grid, start, goal, max_paths=1)
        if not bfs_paths:
            continue
        fam = build_family(grid, start, goal, maze_id, 0, args, bfs_paths[0])
        for tr in fam.trajectories:
            if not tr.transitions:
                continue
            x = rm_inputs_for_traj(tr, device)
            with torch.no_grad():
                rs = model(x).detach().cpu().numpy().tolist()
            for t, r in zip(tr.transitions, rs):
                r = float(r)
                toward = t.delta_bfs > 0
                away = t.delta_bfs < 0
                same = t.delta_bfs == 0 and not t.is_wall
                v = int(t.visit_count_after)
                if toward and v <= 1: cats["toward_no_visit"].append(r)
                if toward and v == 2: cats["toward_visit2"].append(r)
                if toward and v >= 3: cats["toward_visit3plus"].append(r)
                if away and v <= 1: cats["away_no_visit"].append(r)
                if away and v == 2: cats["away_visit2"].append(r)
                if away and v >= 3: cats["away_visit3plus"].append(r)
                if t.is_wall and v <= 2: cats["wall_first"].append(r)
                if t.is_wall and v >= 4: cats["wall_repeated"].append(r)
                if same and v == 2: cats["same_dist_visit2"].append(r)
                if same and v >= 3: cats["same_dist_visit3plus"].append(r)
    order = ["toward_no_visit","toward_visit2","toward_visit3plus","away_no_visit","away_visit2","away_visit3plus","wall_first","wall_repeated","same_dist_visit2","same_dist_visit3plus"]
    report = {k: _reward_stats(cats.get(k, [])) for k in order}
    print("\n[RM Contextual Reward Probe]")
    print(f"{'category':24s} {'n':>7s} {'mean':>9s} {'std':>9s} {'p10':>9s} {'p50':>9s} {'p90':>9s} {'%-1':>8s} {'%+1':>8s}")
    for k in order:
        row = report[k]
        print(f"{k:24s} {row['n']:7d} {row['mean']:9.3f} {row['std']:9.3f} {row['p10']:9.3f} {row['p50']:9.3f} {row['p90']:9.3f} {100*row['pct_neg_one'] if row['n'] else float('nan'):7.2f}% {100*row['pct_pos_one'] if row['n'] else float('nan'):7.2f}%")
    warnings = []
    def m(k: str) -> float: return float(report.get(k, {}).get("mean", float("nan")))
    checks = [
        ("toward_visit2_mean <= away_visit2_mean", "toward_visit2", "away_visit2"),
        ("toward_no_visit_mean <= away_no_visit_mean", "toward_no_visit", "away_no_visit"),
        ("wall_repeated_mean >= wall_first_mean", "wall_first", "wall_repeated"),
        ("same_dist_visit3plus_mean >= same_dist_visit2_mean", "same_dist_visit2", "same_dist_visit3plus"),
    ]
    for msg, good, bad in checks:
        if not math.isnan(m(good)) and not math.isnan(m(bad)):
            if "wall_repeated" in msg or "visit3plus" in msg:
                if m(bad) >= m(good): warnings.append(msg)
            elif m(good) <= m(bad): warnings.append(msg)
    deltas = {
        "toward_no_visit_minus_away_no_visit": m("toward_no_visit") - m("away_no_visit"),
        "toward_visit2_minus_away_visit2": m("toward_visit2") - m("away_visit2"),
        "toward_visit3plus_minus_away_visit3plus": m("toward_visit3plus") - m("away_visit3plus"),
        "same_dist_visit2_minus_same_dist_visit3plus": m("same_dist_visit2") - m("same_dist_visit3plus"),
        "wall_first_minus_wall_repeated": m("wall_first") - m("wall_repeated"),
    }
    print("\n[RM Contextual Gap]")
    for k, v in deltas.items():
        print(f"{k:42s} {v:9.3f}")
    for w in warnings:
        print(f"[WARN] contextual probe: {w}")
    report["gaps"] = deltas
    report["warnings"] = warnings
    with open(out_dir / f"{VERSION}_rm_contextual_probe.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def rm_collapse_check(model: RewardModel, trajectories: List[Trajectory], args: argparse.Namespace, device: torch.device, out_dir: Path) -> Dict[str, Any]:
    vals: List[float] = []
    model.eval()
    sample_trajs = random.sample(trajectories, min(len(trajectories), 200)) if trajectories else []
    for tr in sample_trajs:
        if not tr.transitions:
            continue
        with torch.no_grad():
            vals.extend(model(rm_inputs_for_traj(tr, device)).detach().cpu().numpy().tolist())
    std = float(np.std(vals)) if vals else 0.0
    scale = float(args.target_rm_std / (std + 1e-6)) if std > 0 else float("inf")
    status = "PASS" if std >= args.rm_min_raw_std else "FAIL"
    print("\n[RM Collapse Check]")
    print(f"raw_rm_std={std:.6f}  rm_scale={scale:.4f}  min_std={args.rm_min_raw_std:.4f}  status={status}")
    rep = {"raw_rm_std": std, "rm_scale": scale, "status": status, "rm_min_raw_std": args.rm_min_raw_std}
    with open(out_dir / f"{VERSION}_rm_collapse_check.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    return rep


def rm_probe_reward_for_transition(model: RewardModel, state: np.ndarray, action: int, next_state: np.ndarray, device: torch.device) -> float:
    x = torch.from_numpy(transition_input(state, action, next_state)[None].astype(np.float32)).to(device)
    with torch.no_grad():
        return float(model(x).item())


def run_rm_greedy_single(model: RewardModel, grid: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int], args: argparse.Namespace, device: torch.device, maze_id: int, difficulty: str, use_bfs_tiebreak: bool = True) -> Dict[str, Any]:
    dist = bfs_distances(grid, goal)
    pos = start
    visits = np.zeros((SIZE, SIZE), dtype=np.int32)
    visits[pos] = 1
    wall_hits = visit2 = visit3 = visit4 = 0
    traj_steps = []
    success = False
    for t in range(args.rm_greedy_max_steps):
        state = encode_state(grid, pos, goal, visits, args.visit_count_cap)
        bfs_before = int(dist[pos]) if int(dist[pos]) < 10000 else args.max_steps
        action_infos = []
        rewards = []
        for a in range(4):
            attempted = apply_action(pos, a)
            isw = not is_free(grid, attempted)
            nxt = pos if isw else attempted
            vc_after = int(visits[nxt] + 1)
            temp_visits = visits.copy(); temp_visits[nxt] += 1
            ns = encode_state(grid, nxt, goal, temp_visits, args.visit_count_cap)
            bfs_after = int(dist[nxt]) if int(dist[nxt]) < 10000 else args.max_steps
            r = rm_probe_reward_for_transition(model, state, a, ns, device)
            rewards.append(r)
            action_infos.append({"action": ACTION_NAMES[a], "reward": r, "is_wall": bool(isw), "next_pos": [int(nxt[0]), int(nxt[1])], "delta_bfs": int(bfs_before-bfs_after), "visit_count_after": vc_after})
        max_r = max(rewards)
        if use_bfs_tiebreak:
            candidates = [i for i, r in enumerate(rewards) if max_r - r <= args.rm_greedy_tie_eps]
            # tie break: best delta bfs, then random.
            best_delta = max(action_infos[i]["delta_bfs"] for i in candidates)
            candidates = [i for i in candidates if action_infos[i]["delta_bfs"] == best_delta]
        else:
            # Pure RM argmax: do not use BFS distance to resolve near ties.
            # Only randomize exact/numerical ties at the maximum reward.
            candidates = [i for i, r in enumerate(rewards) if abs(max_r - r) <= 1e-9]
        chosen = random.choice(candidates)
        info = action_infos[chosen]
        nxt = tuple(info["next_pos"])
        if info["is_wall"]: wall_hits += 1
        visits[nxt] += 1
        vc = int(visits[nxt])
        if vc == 2: visit2 += 1
        if vc >= 3: visit3 += 1
        if vc >= 4: visit4 += 1
        traj_steps.append({"t": t, "pos": [int(pos[0]), int(pos[1])], "bfs_dist": bfs_before, "chosen_action": ACTION_NAMES[chosen], "chosen_reward": info["reward"], "chosen_is_wall": info["is_wall"], "chosen_delta_bfs": info["delta_bfs"], "visit_count_after": vc, "actions": {ACTION_NAMES[i]: action_infos[i] for i in range(4)}, "all_actions_saturated": (max(rewards)-min(rewards) < 0.05)})
        pos = nxt
        if pos == goal:
            success = True
            break
    final_dist = 0 if success else int(dist[pos]) if int(dist[pos]) < 10000 else args.max_steps
    if success: outcome = "success"
    elif wall_hits > 0: outcome = "wall_timeout"
    elif visit3 >= args.loop_min_visit3plus: outcome = "loop_timeout"
    else: outcome = "explore_timeout"
    return {"maze_id": maze_id, "difficulty": difficulty, "success": success, "outcome": outcome, "steps": len(traj_steps), "wall_hits": wall_hits, "visit2_steps": visit2, "visit3plus_steps": visit3, "visit4plus_steps": visit4, "max_visit": int(visits.max()), "bfs_gap": final_dist + max(0, len(traj_steps) - (int(dist[start]) if int(dist[start]) < 10000 else args.max_steps)), "trajectory": traj_steps, "greedy_mode": ("bfs_tiebreak" if use_bfs_tiebreak else "pure_argmax")}


def _summarize_rm_greedy_rollouts(rollouts: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "n": len(rollouts),
        "success": safe_mean([1.0 if r["success"] else 0.0 for r in rollouts]),
        "wall_timeout": safe_mean([1.0 if r["outcome"]=="wall_timeout" else 0.0 for r in rollouts]),
        "loop_timeout": safe_mean([1.0 if r["outcome"]=="loop_timeout" else 0.0 for r in rollouts]),
        "explore_timeout": safe_mean([1.0 if r["outcome"]=="explore_timeout" else 0.0 for r in rollouts]),
        "avg_steps": safe_mean([r["steps"] for r in rollouts]),
        "avg_wall_hits": safe_mean([r["wall_hits"] for r in rollouts]),
        "avg_visit2": safe_mean([r["visit2_steps"] for r in rollouts]),
        "avg_visit3plus": safe_mean([r["visit3plus_steps"] for r in rollouts]),
        "avg_visit4plus": safe_mean([r["visit4plus_steps"] for r in rollouts]),
        "avg_max_visit": safe_mean([r["max_visit"] for r in rollouts]),
        "avg_bfs_gap": safe_mean([r["bfs_gap"] for r in rollouts]),
    }
    action_counts = defaultdict(int); total_steps = 0
    for ro in rollouts:
        for st in ro["trajectory"]:
            total_steps += 1
            d = int(st["chosen_delta_bfs"])
            if d > 0: action_counts["toward"] += 1
            if d < 0: action_counts["away"] += 1
            if st["chosen_is_wall"]: action_counts["wall"] += 1
            if st["visit_count_after"] == 2 and d > 0: action_counts["visit2_toward"] += 1
            if st["visit_count_after"] == 2 and d < 0: action_counts["visit2_away"] += 1
            if st["chosen_is_wall"] and st["visit_count_after"] >= 4: action_counts["repeat_wall"] += 1
            if st.get("all_actions_saturated"): action_counts["all_sat"] += 1
    denom = max(1, total_steps)
    ranking = {
        "toward_chosen_rate": action_counts["toward"] / denom,
        "away_chosen_rate": action_counts["away"] / denom,
        "wall_chosen_rate": action_counts["wall"] / denom,
        "visit2_toward_chosen_rate": action_counts["visit2_toward"] / denom,
        "visit2_away_chosen_rate": action_counts["visit2_away"] / denom,
        "repeat_wall_chosen_rate": action_counts["repeat_wall"] / denom,
        "all_actions_saturated_rate": action_counts["all_sat"] / denom,
    }
    return {"summary": summary, "action_ranking": ranking}


def run_rm_greedy_probe(model: RewardModel, args: argparse.Namespace, device: torch.device, out_dir: Path) -> Dict[str, Any]:
    """Run both assisted and pure RM greedy probes.

    The legacy files keep the BFS-tiebreak version for compatibility. New *_pure_argmax
    files expose whether RM alone can choose actions without BFS help.
    """
    probe_specs = [("bfs_tiebreak", True), ("pure_argmax", False)]
    combined: Dict[str, Any] = {}
    for mode_name, use_bfs_tiebreak in probe_specs:
        rollouts = []
        desc = f"RM greedy probe ({mode_name})"
        for maze_id in tqdm(range(args.rm_greedy_probe_mazes), desc=desc):
            difficulty = sample_difficulty(args)
            grid, start, goal = generate_maze(difficulty)
            rollouts.append(run_rm_greedy_single(model, grid, start, goal, args, device, maze_id, difficulty, use_bfs_tiebreak=use_bfs_tiebreak))
        out = _summarize_rm_greedy_rollouts(rollouts)
        combined[mode_name] = out
        print(f"\n[RM Greedy Probe Summary: {mode_name}]")
        for k, v in out["summary"].items():
            print(f"{k:22s}: {v:.4f}" if isinstance(v, float) else f"{k:22s}: {v}")
        print(f"\n[RM Action Ranking Summary: {mode_name}]")
        for k, v in out["action_ranking"].items():
            print(f"{k:30s}: {v:.4f}")

        suffix = "" if mode_name == "bfs_tiebreak" else "_pure_argmax"
        with open(out_dir / f"{VERSION}_rm_greedy_rollout_summary{suffix}.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        if args.rm_debug_save_rollouts:
            with open(out_dir / f"{VERSION}_rm_greedy_rollout_debug{suffix}.json", "w", encoding="utf-8") as f:
                json.dump(rollouts, f, indent=2)

    with open(out_dir / f"{VERSION}_rm_greedy_rollout_comparison.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    return combined

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
    p.add_argument("--mode", choices=["train-rm", "debug-rm", "all-rm", "all"], default="all-rm")
    p.add_argument("--output-dir", default="./v2.0.8")
    p.add_argument("--run-name", default="default")
    p.add_argument("--reward-model", default="")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    # environment
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--easy-ratio", type=float, default=0.2)
    p.add_argument("--medium-ratio", type=float, default=0.4)
    p.add_argument("--hard-ratio", type=float, default=0.4)
    p.add_argument("--difficulty-easy", type=float, default=None)
    p.add_argument("--difficulty-medium", type=float, default=None)
    p.add_argument("--difficulty-hard", type=float, default=None)
    # visit
    p.add_argument("--visit-count-cap", type=float, default=8.0)
    p.add_argument("--visit-second-penalty", type=float, default=0.10)
    p.add_argument("--visit-exp-scale", type=float, default=0.25)
    p.add_argument("--visit-exp-base", type=float, default=1.60)
    p.add_argument("--visit-penalty-cap", type=float, default=4.0)
    p.add_argument("--visit-tier-mid-threshold", type=int, default=4)
    # wall badness
    p.add_argument("--wall-exp-alpha", type=float, default=4.0)
    # dataset
    p.add_argument("--rm-mazes", type=int, default=400)
    p.add_argument("--families-per-maze", type=int, default=3)
    p.add_argument("--max-bfs-paths-per-maze", type=int, default=8)
    p.add_argument("--family-prefix-min-ratio", type=float, default=0.20)
    p.add_argument("--family-prefix-max-ratio", type=float, default=0.65)
    p.add_argument("--generator-retry", type=int, default=40)
    p.add_argument("--success-explore-max-wall", type=int, default=2)
    p.add_argument("--success-explore-max-visit4plus", type=int, default=4)
    p.add_argument("--success-explore-max-extra", type=int, default=6)
    p.add_argument("--recovery-max-visit4plus", type=int, default=4)
    p.add_argument("--recovery-max-wall", type=int, default=2)
    p.add_argument("--loop-min-visit3plus", type=int, default=8)
    p.add_argument("--loop-min-max-visit", type=int, default=4)
    p.add_argument("--revisit-timeout-min-visit2", type=int, default=6)
    p.add_argument("--repeated-wall-min-hits", type=int, default=8)
    p.add_argument("--repeated-wall-min-max-visit", type=int, default=4)
    # pairs
    p.add_argument("--pairs-per-maze", type=int, default=999999)
    p.add_argument("--pairs-per-family", type=int, default=999999)
    p.add_argument("--bfs-pair-cap-per-maze", type=int, default=4)
    p.add_argument("--episode-pair-ratio", type=float, default=0.40)
    p.add_argument("--local-segment-pair-ratio", type=float, default=0.50)
    p.add_argument("--failure-mined-pair-ratio", type=float, default=0.10)
    p.add_argument("--local-segment-min-len", type=int, default=4)
    p.add_argument("--local-segment-max-len", type=int, default=8)
    p.add_argument("--local-segment-pairs-per-maze", type=int, default=45)
    p.add_argument("--enable-failure-mining", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--failure-mining-after-epochs", type=int, default=2)
    p.add_argument("--failure-mining-probe-mazes", type=int, default=100)
    p.add_argument("--failure-mined-pairs-per-maze", type=int, default=8)
    # pair debug / validity
    p.add_argument("--local-visit-penalty-progress-tolerance", type=float, default=0.20)
    p.add_argument("--pair-valid-rate-warning-threshold", type=float, default=0.80)
    # margins
    p.add_argument("--margin-outcome-min", type=float, default=0.60)
    p.add_argument("--margin-outcome-max", type=float, default=0.80)
    p.add_argument("--margin-outcome-s", type=float, default=1.00)
    p.add_argument("--margin-wall-presence-min", type=float, default=0.25)
    p.add_argument("--margin-wall-presence-max", type=float, default=0.48)
    p.add_argument("--margin-wall-presence-s", type=float, default=1.00)
    p.add_argument("--margin-wall-exp-min", type=float, default=0.10)
    p.add_argument("--margin-wall-exp-max", type=float, default=0.42)
    p.add_argument("--margin-wall-exp-s", type=float, default=0.70)
    p.add_argument("--margin-revisit-min", type=float, default=0.08)
    p.add_argument("--margin-revisit-max", type=float, default=0.28)
    p.add_argument("--margin-revisit-s", type=float, default=0.80)
    p.add_argument("--margin-path-min", type=float, default=0.01)
    p.add_argument("--margin-path-max", type=float, default=0.10)
    p.add_argument("--margin-path-s", type=float, default=0.40)
    p.add_argument("--margin-cpl-recover-wall-min", type=float, default=0.25)
    p.add_argument("--margin-cpl-recover-wall-max", type=float, default=0.48)
    p.add_argument("--margin-cpl-recover-wall-s", type=float, default=0.90)
    p.add_argument("--margin-cpl-explore-wall-min", type=float, default=0.18)
    p.add_argument("--margin-cpl-explore-wall-max", type=float, default=0.40)
    p.add_argument("--margin-cpl-explore-wall-s", type=float, default=0.90)
    p.add_argument("--margin-cpl-recover-loop-min", type=float, default=0.12)
    p.add_argument("--margin-cpl-recover-loop-max", type=float, default=0.30)
    p.add_argument("--margin-cpl-recover-loop-s", type=float, default=0.80)
    p.add_argument("--margin-cpl-visit-toward-min", type=float, default=0.08)
    p.add_argument("--margin-cpl-visit-toward-max", type=float, default=0.24)
    p.add_argument("--margin-cpl-visit-toward-s", type=float, default=1.00)
    # v2.0.8 episode-level margins
    p.add_argument("--episode-outcome-min", type=float, default=0.35)
    p.add_argument("--episode-outcome-max", type=float, default=0.65)
    p.add_argument("--episode-outcome-scale", type=float, default=1.00)
    p.add_argument("--episode-wall-min", type=float, default=0.15)
    p.add_argument("--episode-wall-max", type=float, default=0.35)
    p.add_argument("--episode-wall-scale", type=float, default=1.00)
    p.add_argument("--episode-loop-min", type=float, default=0.08)
    p.add_argument("--episode-loop-max", type=float, default=0.25)
    p.add_argument("--episode-loop-scale", type=float, default=1.00)
    p.add_argument("--episode-path-min", type=float, default=0.00)
    p.add_argument("--episode-path-max", type=float, default=0.04)
    p.add_argument("--episode-path-scale", type=float, default=1.00)
    # v2.0.8 local segment margins
    p.add_argument("--local-toward-min", type=float, default=0.04)
    p.add_argument("--local-toward-max", type=float, default=0.16)
    p.add_argument("--local-toward-scale", type=float, default=0.80)
    p.add_argument("--local-safe-wall-min", type=float, default=0.18)
    p.add_argument("--local-safe-wall-max", type=float, default=0.35)
    p.add_argument("--local-safe-wall-scale", type=float, default=0.80)
    p.add_argument("--local-visit2-direction-min", type=float, default=0.03)
    p.add_argument("--local-visit2-direction-max", type=float, default=0.10)
    p.add_argument("--local-visit2-direction-scale", type=float, default=0.80)
    p.add_argument("--local-recover-loop-min", type=float, default=0.08)
    p.add_argument("--local-recover-loop-max", type=float, default=0.22)
    p.add_argument("--local-recover-loop-scale", type=float, default=0.80)
    p.add_argument("--local-visit-penalty-min", type=float, default=0.01)
    p.add_argument("--local-visit-penalty-max", type=float, default=0.06)
    p.add_argument("--local-visit-penalty-scale", type=float, default=1.00)
    # v2.0.8 failure-mined margins
    p.add_argument("--failure-mined-wall-min", type=float, default=0.18)
    p.add_argument("--failure-mined-wall-max", type=float, default=0.35)
    p.add_argument("--failure-mined-wall-scale", type=float, default=0.80)
    p.add_argument("--failure-mined-away-min", type=float, default=0.04)
    p.add_argument("--failure-mined-away-max", type=float, default=0.16)
    p.add_argument("--failure-mined-away-scale", type=float, default=0.80)
    p.add_argument("--failure-mined-visit2-min", type=float, default=0.03)
    p.add_argument("--failure-mined-visit2-max", type=float, default=0.10)
    p.add_argument("--failure-mined-visit2-scale", type=float, default=0.80)
    p.add_argument("--failure-mined-loop-min", type=float, default=0.08)
    p.add_argument("--failure-mined-loop-max", type=float, default=0.22)
    p.add_argument("--failure-mined-loop-scale", type=float, default=0.80)
    p.add_argument("--failure-mined-saturated-min", type=float, default=0.02)
    p.add_argument("--failure-mined-saturated-max", type=float, default=0.08)
    p.add_argument("--failure-mined-saturated-scale", type=float, default=1.00)
    # RM training
    p.add_argument("--rm-epochs", type=int, default=8)
    p.add_argument("--rm-batch-size", type=int, default=64)
    p.add_argument("--rm-lr", type=float, default=1e-3)
    p.add_argument("--reward-l2", type=float, default=0.01)
    p.add_argument("--score-normalizer", choices=["bfs_len", "episode_len", "raw"], default="episode_len")
    p.add_argument("--debug-probe-mazes", type=int, default=30)
    p.add_argument("--stop-if-rm-diagnostic-fails", action="store_true")
    p.add_argument("--rm-min-raw-std", type=float, default=0.05)
    p.add_argument("--rm-scale-cap", type=float, default=10.0)
    p.add_argument("--target-rm-std", type=float, default=1.0)
    # RM-only greedy probe
    p.add_argument("--rm-greedy-probe-mazes", type=int, default=100)
    p.add_argument("--rm-greedy-max-steps", type=int, default=64)
    p.add_argument("--rm-greedy-tie-eps", type=float, default=0.03)
    p.add_argument("--rm-debug-save-rollouts", action=argparse.BooleanOptionalAction, default=True)
    return p

def auto_run_name(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    if args.run_name != "default":
        return args.run_name
    defaults = vars(parser.parse_args([]))
    ignore = {"mode", "run_name", "output_dir", "device", "reward_model"}
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
    if args.mode == "all":
        print("[INFO] v2.0.8 patched is RM-only. QCNN training is disabled. Interpreting --mode all as --mode all-rm.")
        args.mode = "all-rm"
    set_seed(args.seed)
    run_name = auto_run_name(args, parser)
    out_dir = ensure_dir(Path(args.output_dir) / run_name)
    device = get_device(args.device)
    print(f"\n=== {VERSION} ===")
    print(f"Device: {device}")
    print(f"Output: {out_dir}")
    print("RM-only debug: trajectory families + local segment preferences, bucket margins, contextual probes, greedy RM rollouts. QCNN training is disabled.")
    print("\n==== MARGIN DESIGN PRINCIPLE ====")
    print("BTL-margin is the only loss.")
    print("Saturation should be controlled by min/max/scale, not by extra losses.")
    print("Small local margins are intentional.")
    print("Path margin is intentionally weak to avoid length shortcut.")
    print("If a bucket saturates, tune scale upward or max downward.")
    print("If a bucket is too weak, tune scale downward or min upward.")
    rm_model: RewardModel
    trajectories: List[Trajectory] = []
    if args.mode == "debug-rm":
        if not args.reward_model:
            raise ValueError("--mode debug-rm requires --reward-model")
        rm_model = load_reward_model(args.reward_model, device)
        print(f"[Load] reward model: {args.reward_model}")
        old_rm_mazes = args.rm_mazes
        args.rm_mazes = min(50, max(10, old_rm_mazes))
        trajectories, _, dataset_summary = build_preference_dataset(args)
        args.rm_mazes = old_rm_mazes
        with open(out_dir / f"{VERSION}_dataset_debug.json", "w", encoding="utf-8") as f:
            json.dump(dataset_summary, f, indent=2)
        with open(out_dir / f"{VERSION}_margin_tuning_report.json", "w", encoding="utf-8") as f:
            json.dump(dataset_summary.get("margin_tuning_report", {}), f, indent=2)
        with open(out_dir / f"{VERSION}_pair_direction_report.json", "w", encoding="utf-8") as f:
            json.dump(dataset_summary.get("pair_direction_report", {}), f, indent=2)
        with open(out_dir / f"{VERSION}_pair_validity_report.json", "w", encoding="utf-8") as f:
            json.dump(dataset_summary.get("pair_validity_report", {}), f, indent=2)
    else:
        trajectories, pairs, dataset_summary = build_preference_dataset(args)
        with open(out_dir / f"{VERSION}_dataset_debug.json", "w", encoding="utf-8") as f:
            json.dump(dataset_summary, f, indent=2)
        with open(out_dir / f"{VERSION}_margin_tuning_report.json", "w", encoding="utf-8") as f:
            json.dump(dataset_summary.get("margin_tuning_report", {}), f, indent=2)
        with open(out_dir / f"{VERSION}_pair_direction_report.json", "w", encoding="utf-8") as f:
            json.dump(dataset_summary.get("pair_direction_report", {}), f, indent=2)
        with open(out_dir / f"{VERSION}_pair_validity_report.json", "w", encoding="utf-8") as f:
            json.dump(dataset_summary.get("pair_validity_report", {}), f, indent=2)
        rm_model, rm_hist = train_reward_model(pairs, args, device, out_dir)
        best_loss_path = rm_hist.get("best_loss_path", "")
        if best_loss_path and Path(best_loss_path).exists():
            rm_model = load_reward_model(best_loss_path, device)
            print(f"[Load] debug RM from best loss: {best_loss_path}")
    probe_step_rewards(rm_model, args, device, out_dir)
    probe_contextual_rewards(rm_model, args, device, out_dir)
    rm_collapse_check(rm_model, trajectories, args, device, out_dir)
    run_rm_greedy_probe(rm_model, args, device, out_dir)
    print(f"\n[Done] RM-only debug artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
