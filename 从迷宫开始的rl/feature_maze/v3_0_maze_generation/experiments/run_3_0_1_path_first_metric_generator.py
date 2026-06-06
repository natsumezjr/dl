#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3.0.1 — Path-First Metric-Constrained Maze Generator (PF-MCG).

Generate 8x8 solvable mazes by:
  1. sampling a self-avoiding main path (Path-First skeleton),
  2. injecting structural primitives to steer metrics toward target ranges,
  3. quantizing final structure and recording operation deltas.

No CNN / RL / Reward Model / GAN / PCGML.
Stable code lives in this file until extracted to common/.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# =============================================================================
# constants
# =============================================================================

VERSION = "3.0.1"
EXPERIMENT_NAME = "path_first_metric_generator"
GENERATOR_NAME = "pf_mcg"

ACTIONS: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_VECTORS = ACTIONS
OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}
INF = 10**9

CORE_METRICS = (
    "reachable_free_ratio",
    "wall_ratio",
    "bfs_len_norm",
    "detour_ratio",
    "dead_end_ratio",
    "junction_ratio",
    "dead_end_depth_mean",
    "junction_on_solution_ratio",
)

DELTA_METRICS = CORE_METRICS + (
    "island_free_ratio",
    "solution_turn_ratio",
)

DEFAULTS: Dict[str, Any] = dict(
    seed=42,
    grid_size=8,
    n_mazes=100,
    run_name=None,
    output_dir=None,
    no_tqdm=False,
    no_viz=False,
    n_trace_samples=20,
    n_top_k=5,
    n_sample_images=6,
    target_wall_ratio_min=0.18,
    target_wall_ratio_max=0.35,
    target_bfs_len_norm_min=0.15,
    target_bfs_len_norm_max=0.45,
    target_detour_ratio_min=1.3,
    target_detour_ratio_max=4.5,
    target_dead_end_ratio_min=0.04,
    target_dead_end_ratio_max=0.18,
    target_junction_ratio_min=0.30,
    target_junction_ratio_max=0.75,
    target_dead_end_depth_mean_min=1.0,
    target_dead_end_depth_mean_max=4.0,
    target_junction_on_solution_ratio_min=0.35,
    target_junction_on_solution_ratio_max=0.85,
    target_island_free_ratio_min=0.00,
    target_island_free_ratio_max=0.08,
    target_manhattan_min=4,
    target_manhattan_max=12,
    target_main_path_len_min=8,
    target_main_path_len_max=30,
    target_main_path_turn_ratio_min=0.15,
    target_main_path_turn_ratio_max=0.75,
    max_restarts=100,
    max_edit_steps=80,
    max_main_path_attempts=500,
    max_operation_attempts=30,
    allow_worse_prob=0.0,
)

SCRIPT_PATH = Path(__file__).resolve()
V3_ROOT = SCRIPT_PATH.parents[1]


# =============================================================================
# io helpers
# =============================================================================


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def json_safe(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            return None
        return fv
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k: json_safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    return v


def save_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, ensure_ascii=False, indent=2)


def maybe_tqdm(iterable, args, desc: str = "", total=None):
    if getattr(args, "no_tqdm", False) or tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=False, file=sys.stderr, dynamic_ncols=True)


def is_valid_number(x: Any) -> bool:
    if x is None:
        return False
    try:
        v = float(x)
        return not (math.isnan(v) or math.isinf(v))
    except (TypeError, ValueError):
        return False


def fmt_num(v: Any, width: int = 10, prec: int = 4) -> str:
    if not is_valid_number(v):
        return f"{'—':>{width}s}"
    return f"{float(v):>{width}.{prec}f}"


# =============================================================================
# grid / BFS utilities
# =============================================================================


def in_bounds(p: Tuple[int, int], n: int) -> bool:
    return 0 <= p[0] < n and 0 <= p[1] < n


def neighbors(p: Tuple[int, int], n: int):
    for q in neighbors_idx(p, n):
        yield q


def neighbors_idx(p: Tuple[int, int], n: int):
    for dr, dc in ACTIONS:
        q = (p[0] + dr, p[1] + dc)
        if in_bounds(q, n):
            yield q


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def direction_between(a: Tuple[int, int], b: Tuple[int, int]) -> Optional[int]:
    dr, dc = b[0] - a[0], b[1] - a[1]
    for i, (ar, ac) in enumerate(ACTIONS):
        if (dr, dc) == (ar, ac):
            return i
    return None


def count_turns(path: Sequence[Tuple[int, int]]) -> int:
    if len(path) < 3:
        return 0
    turns = 0
    prev_dir = direction_between(path[0], path[1])
    for i in range(1, len(path) - 1):
        d = direction_between(path[i], path[i + 1])
        if d is not None and prev_dir is not None and d != prev_dir:
            turns += 1
        prev_dir = d
    return turns


def bfs_shortest_path(
    maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]
) -> Optional[List[Tuple[int, int]]]:
    if maze[start] == 1 or maze[goal] == 1:
        return None
    if start == goal:
        return [start]
    n = maze.shape[0]
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    dq = deque([start])
    while dq:
        p = dq.popleft()
        if p == goal:
            break
        for q in sorted(neighbors(p, n)):
            if maze[q] == 0 and q not in parent:
                parent[q] = p
                dq.append(q)
    if goal not in parent:
        return None
    path: List[Tuple[int, int]] = []
    cur: Optional[Tuple[int, int]] = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def reachable_free_cells(maze: np.ndarray, start: Tuple[int, int]) -> set:
    if maze[start] == 1:
        return set()
    n = maze.shape[0]
    seen = {start}
    dq = deque([start])
    while dq:
        p = dq.popleft()
        for q in neighbors(p, n):
            if maze[q] == 0 and q not in seen:
                seen.add(q)
                dq.append(q)
    return seen


def free_cells_all(maze: np.ndarray) -> List[Tuple[int, int]]:
    n = maze.shape[0]
    return [(r, c) for r in range(n) for c in range(n) if maze[r, c] == 0]


def wall_cells_all(maze: np.ndarray) -> List[Tuple[int, int]]:
    n = maze.shape[0]
    return [(r, c) for r in range(n) for c in range(n) if maze[r, c] == 1]


def build_adjacency(maze: np.ndarray, cells: set) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    n = maze.shape[0]
    adj: Dict[Tuple[int, int], List[Tuple[int, int]]] = {c: [] for c in cells}
    for p in cells:
        for q in neighbors(p, n):
            if q in cells:
                adj[p].append(q)
    return adj


def is_straight_corridor(adj: Dict, v: Tuple[int, int]) -> bool:
    nbs = adj[v]
    if len(nbs) != 2:
        return False
    a, b = nbs[0], nbs[1]
    return a[0] + b[0] == 2 * v[0] and a[1] + b[1] == 2 * v[1]


def compute_dead_end_depths(
    adj: Dict[Tuple[int, int], List[Tuple[int, int]]],
    degrees: Dict[Tuple[int, int], int],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    solution_set: set,
    reachable: set,
) -> List[int]:
    depths: List[int] = []
    for de in sorted(reachable):
        if degrees.get(de, 0) != 1:
            continue
        depth = 0
        prev: Optional[Tuple[int, int]] = None
        cur = de
        while True:
            nbs = [w for w in adj[cur] if w != prev]
            if not nbs:
                break
            nxt = nbs[0]
            depth += 1
            if nxt == start or nxt == goal or nxt in solution_set or degrees.get(nxt, 0) >= 3:
                break
            if degrees.get(nxt, 0) != 2:
                break
            prev = cur
            cur = nxt
        depths.append(depth)
    return depths


# =============================================================================
# MazeQuantizer (3.0.0 extended)
# =============================================================================


def quantize_maze(maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> dict:
    n = maze.shape[0]
    all_free = free_cells_all(maze)
    free_count = len(all_free)
    wall_count = n * n - free_count

    start_free = maze[start] == 0
    goal_free = maze[goal] == 0
    path = bfs_shortest_path(maze, start, goal) if start_free and goal_free else None
    is_solvable = path is not None

    reachable = reachable_free_cells(maze, start) if start_free else set()
    if free_count == 0:
        reachable_free_ratio = float("nan")
        island_free_ratio = float("nan")
    else:
        reachable_free_ratio = len(reachable) / free_count
        island_free_ratio = 1.0 - reachable_free_ratio

    validity = {"is_solvable": is_solvable, "reachable_free_ratio": reachable_free_ratio}

    bfs_len: Optional[int] = (len(path) - 1) if path else None
    manhattan_dist = manhattan(start, goal)
    wall_ratio = wall_count / (n * n)
    bfs_len_norm = (bfs_len / (n * n - 1)) if bfs_len is not None else float("nan")
    if manhattan_dist == 0:
        detour_ratio = float("nan")
    elif bfs_len is not None:
        detour_ratio = bfs_len / manhattan_dist
    else:
        detour_ratio = float("nan")

    wall_and_path = {
        "wall_ratio": wall_ratio,
        "bfs_len": bfs_len,
        "bfs_len_norm": bfs_len_norm,
        "manhattan_dist": manhattan_dist,
        "detour_ratio": detour_ratio,
    }

    degree_histogram = {f"degree_{d}": 0 for d in range(5)}
    dead_end_ratio = junction_ratio = corridor_ratio = float("nan")
    straight_corridor_ratio = turn_corridor_ratio = float("nan")
    off_solution_junction_ratio = float("nan")
    solution_turn_count = 0
    solution_turn_ratio = float("nan")

    if is_solvable and reachable:
        adj = build_adjacency(maze, reachable)
        degrees = {v: len(adj[v]) for v in reachable}
        for v in reachable:
            degree_histogram[f"degree_{min(degrees[v], 4)}"] += 1
        n_reach = len(reachable)
        dead_end_ratio = sum(1 for v in reachable if degrees[v] == 1) / n_reach
        junction_ratio = sum(1 for v in reachable if degrees[v] >= 3) / n_reach
        corridor_nodes = [v for v in reachable if degrees[v] == 2]
        corridor_ratio = len(corridor_nodes) / n_reach
        if corridor_nodes:
            straight_n = sum(1 for v in corridor_nodes if is_straight_corridor(adj, v))
            turn_n = len(corridor_nodes) - straight_n
            straight_corridor_ratio = straight_n / n_reach
            turn_corridor_ratio = turn_n / n_reach
        junction_nodes = [v for v in reachable if degrees[v] >= 3]
        if path:
            solution_set = set(path)
            off_j = sum(1 for v in junction_nodes if v not in solution_set)
            off_solution_junction_ratio = off_j / max(1, len(junction_nodes))

    graph_structure = {
        "degree_histogram": degree_histogram,
        "dead_end_ratio": dead_end_ratio,
        "junction_ratio": junction_ratio,
        "corridor_ratio": corridor_ratio,
        "straight_corridor_ratio": straight_corridor_ratio,
        "turn_corridor_ratio": turn_corridor_ratio,
        "off_solution_junction_ratio": off_solution_junction_ratio,
    }

    dead_end_depth_mean = 0.0
    dead_end_depth_max = 0
    junction_on_solution_ratio = float("nan")

    if is_solvable and path and reachable:
        solution_set = set(path)
        adj = build_adjacency(maze, reachable)
        degrees = {v: len(adj[v]) for v in reachable}
        depths = compute_dead_end_depths(adj, degrees, start, goal, solution_set, reachable)
        if depths:
            dead_end_depth_mean = float(np.mean(depths))
            dead_end_depth_max = int(max(depths))
        junction_on_solution_ratio = sum(1 for v in path if degrees.get(v, 0) >= 3) / len(path)
        solution_turn_count = count_turns(path)
        solution_turn_ratio = solution_turn_count / max(1, bfs_len - 1) if bfs_len and bfs_len > 0 else 0.0

    playability_proxy = {
        "dead_end_depth_mean": dead_end_depth_mean,
        "dead_end_depth_max": dead_end_depth_max,
        "junction_on_solution_ratio": junction_on_solution_ratio,
        "solution_path_selection": "one_bfs_path",
        "solution_turn_count": solution_turn_count,
        "solution_turn_ratio": solution_turn_ratio,
    }

    core_metrics = {
        "reachable_free_ratio": reachable_free_ratio,
        "wall_ratio": wall_ratio,
        "bfs_len_norm": bfs_len_norm,
        "detour_ratio": detour_ratio,
        "dead_end_ratio": dead_end_ratio,
        "junction_ratio": junction_ratio,
        "dead_end_depth_mean": dead_end_depth_mean,
        "junction_on_solution_ratio": junction_on_solution_ratio,
    }

    aux_metrics = {
        "bfs_len": bfs_len,
        "manhattan_dist": manhattan_dist,
        "corridor_ratio": corridor_ratio,
        "straight_corridor_ratio": straight_corridor_ratio,
        "turn_corridor_ratio": turn_corridor_ratio,
        "solution_turn_count": solution_turn_count,
        "solution_turn_ratio": solution_turn_ratio,
        "off_solution_junction_ratio": off_solution_junction_ratio,
        "island_free_ratio": island_free_ratio,
        "degree_histogram": degree_histogram,
        "dead_end_depth_max": dead_end_depth_max,
        "free_cell_count": free_count,
        "wall_cell_count": wall_count,
    }

    return {
        "validity": validity,
        "wall_and_path": wall_and_path,
        "graph_structure": graph_structure,
        "playability_proxy": playability_proxy,
        "core_metrics": core_metrics,
        "aux_metrics": aux_metrics,
    }


# =============================================================================
# target ranges & metric error
# =============================================================================


def build_target_ranges(args) -> dict:
    i_min, i_max = args.target_island_free_ratio_min, args.target_island_free_ratio_max
    return {
        "reachable_free_ratio": (1.0 - i_max, 1.0 - i_min),
        "wall_ratio": (args.target_wall_ratio_min, args.target_wall_ratio_max),
        "bfs_len_norm": (args.target_bfs_len_norm_min, args.target_bfs_len_norm_max),
        "detour_ratio": (args.target_detour_ratio_min, args.target_detour_ratio_max),
        "dead_end_ratio": (args.target_dead_end_ratio_min, args.target_dead_end_ratio_max),
        "junction_ratio": (args.target_junction_ratio_min, args.target_junction_ratio_max),
        "dead_end_depth_mean": (
            args.target_dead_end_depth_mean_min,
            args.target_dead_end_depth_mean_max,
        ),
        "junction_on_solution_ratio": (
            args.target_junction_on_solution_ratio_min,
            args.target_junction_on_solution_ratio_max,
        ),
        "island_free_ratio": (i_min, i_max),
    }


def build_target_ranges_record(args) -> dict:
    tr = build_target_ranges(args)
    return {k: [lo, hi] for k, (lo, hi) in tr.items()}


def single_metric_error(m: Any, lo: float, hi: float) -> float:
    if not is_valid_number(m):
        return 100.0
    mv = float(m)
    if mv < lo:
        return lo - mv
    if mv > hi:
        return mv - hi
    return 0.0


def compute_metric_error(
    core_metrics: dict,
    aux_metrics: dict,
    target_ranges: dict,
    weights: Optional[dict] = None,
) -> dict:
    weights = weights or {}
    per_metric_error: Dict[str, float] = {}
    in_range_flags: Dict[str, bool] = {}

    for k in CORE_METRICS:
        lo, hi = target_ranges[k]
        e = single_metric_error(core_metrics.get(k), lo, hi)
        per_metric_error[k] = e
        in_range_flags[k] = e == 0.0

    i_lo, i_hi = target_ranges["island_free_ratio"]
    island = aux_metrics.get("island_free_ratio")
    ie = single_metric_error(island, i_lo, i_hi)
    per_metric_error["island_free_ratio"] = ie
    in_range_flags["island_free_ratio"] = ie == 0.0

    total = sum(weights.get(k, 1.0) * per_metric_error[k] for k in per_metric_error)
    core_in_range = all(in_range_flags[k] for k in CORE_METRICS)
    all_in_range = core_in_range and in_range_flags["island_free_ratio"]

    return {
        "total_error": total,
        "per_metric_error": per_metric_error,
        "in_range_flags": in_range_flags,
        "all_in_range": all_in_range,
        "core_in_range": core_in_range,
    }


def most_violated_metric(error_dict: dict) -> str:
    per = error_dict["per_metric_error"]
    return max(per, key=lambda k: per[k])


def metrics_snapshot(q: dict) -> dict:
    snap = dict(q["core_metrics"])
    snap["island_free_ratio"] = q["aux_metrics"].get("island_free_ratio")
    snap["solution_turn_ratio"] = q["aux_metrics"].get("solution_turn_ratio")
    return snap


def delta_metrics(before: dict, after: dict) -> dict:
    out = {}
    for k in DELTA_METRICS:
        bv, av = before.get(k), after.get(k)
        if is_valid_number(bv) and is_valid_number(av):
            out[k] = float(av) - float(bv)
        else:
            out[k] = None
    return out


# =============================================================================
# main path generation
# =============================================================================


def goal_candidates_for_manhattan(start: Tuple[int, int], D: int, n: int) -> List[Tuple[int, int]]:
    sr, sc = start
    out = []
    for r in range(n):
        for c in range(n):
            if (r, c) == start:
                continue
            if manhattan(start, (r, c)) == D:
                out.append((r, c))
    return out


def path_reachable_in_steps(pos: Tuple[int, int], goal: Tuple[int, int], steps_left: int) -> bool:
    d = manhattan(pos, goal)
    if steps_left < d:
        return False
    return (steps_left - d) % 2 == 0


def generate_self_avoiding_main_path(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    target_len: int,
    turn_ratio_range: Tuple[float, float],
    grid_size: int,
    rng: random.Random,
    max_attempts: int = 500,
) -> Tuple[Optional[List[Tuple[int, int]]], Optional[str]]:
    n = grid_size
    D = manhattan(start, goal)
    if target_len < D:
        return None, "main_path_failed_len"
    if (target_len - D) % 2 != 0:
        return None, "main_path_failed_len"
    if target_len > n * n - 1:
        return None, "main_path_failed_len"

    lo_turn, hi_turn = turn_ratio_range
    last_reason = "main_path_failed_no_candidate"

    for _ in range(max_attempts):
        path: List[Tuple[int, int]] = [start]
        visited = {start}

        def dfs(pos: Tuple[int, int], steps_left: int) -> bool:
            if steps_left == 0:
                return pos == goal
            if not path_reachable_in_steps(pos, goal, steps_left):
                return False
            nbs = list(neighbors(pos, n))
            rng.shuffle(nbs)
            nbs.sort(key=lambda q: (-manhattan(q, goal), rng.random()))
            for q in nbs:
                if q in visited:
                    continue
                if steps_left - 1 < manhattan(q, goal):
                    continue
                visited.add(q)
                path.append(q)
                if dfs(q, steps_left - 1):
                    return True
                path.pop()
                visited.remove(q)
            return False

        if not dfs(start, target_len):
            last_reason = "main_path_failed_no_candidate"
            continue

        turn_count = count_turns(path)
        turn_ratio = turn_count / max(1, target_len - 1) if target_len > 1 else 0.0
        if lo_turn <= turn_ratio <= hi_turn:
            return path, None
        last_reason = "main_path_failed_turn_ratio"

    return None, last_reason if last_reason else "main_path_failed_timeout"


def sample_main_path_len(rng: random.Random, lo: int, hi: int, D: int) -> Optional[int]:
    candidates = [L for L in range(lo, hi + 1) if L >= D and (L - D) % 2 == 0]
    if not candidates:
        return None
    return rng.choice(candidates)


# =============================================================================
# primitive operations
# =============================================================================


def _make_op_result(
    op_type: str,
    success: bool,
    reject_reason: Optional[str],
    before_q: dict,
    after_q: Optional[dict],
    before_err: dict,
    after_err: Optional[dict],
    accepted: bool,
) -> dict:
    before_m = metrics_snapshot(before_q)
    after_m = metrics_snapshot(after_q) if after_q else None
    return {
        "operation_type": op_type,
        "success": success,
        "reject_reason": reject_reason,
        "before_metrics": before_m,
        "after_metrics": after_m,
        "delta_metrics": delta_metrics(before_m, after_m) if after_m else None,
        "before_error": before_err,
        "after_error": after_err,
        "accepted": accepted,
    }


def _carve_branch(
    maze: np.ndarray,
    anchor: Tuple[int, int],
    branch_len: int,
    start: Tuple[int, int],
    rng: random.Random,
    allow_reconnect: bool = False,
) -> Tuple[bool, Optional[str]]:
    n = maze.shape[0]
    reachable = reachable_free_cells(maze, start)
    tip = anchor
    prev = None
    carved = []
    for _ in range(branch_len):
        wall_nbs = []
        for q in neighbors(tip, n):
            if maze[q] == 1:
                if not allow_reconnect and q in reachable and q != anchor:
                    continue
                wall_nbs.append(q)
        if not wall_nbs:
            break
        rng.shuffle(wall_nbs)
        nxt = wall_nbs[0]
        maze[nxt] = 0
        carved.append(nxt)
        prev = tip
        tip = nxt
        reachable = reachable_free_cells(maze, start)
    if not carved:
        return False, "no_branch_carved"
    if allow_reconnect:
        return True, None
    tip_reach = reachable_free_cells(maze, start)
    if carved[-1] in tip_reach and len(carved) > 1:
        for c in reversed(carved):
            maze[c] = 1
        return False, "branch_reconnected"
    return True, None


def op_dead_end_inject(ctx: dict, rng: random.Random) -> dict:
    maze, start, goal = ctx["maze"], ctx["start"], ctx["goal"]
    before_q, before_err = ctx["q"], ctx["error"]
    path = bfs_shortest_path(maze, start, goal)
    if not path or len(path) < 3:
        return _make_op_result("dead_end_inject", False, "no_path", before_q, None, before_err, None, False)

    anchors = [p for p in path[1:-1]]
    rng.shuffle(anchors)
    branch_len = rng.randint(2, max(2, int(ctx["args"].target_dead_end_depth_mean_max)))
    for anchor in anchors[:8]:
        backup = maze.copy()
        ok, reason = _carve_branch(maze, anchor, branch_len, start, rng, allow_reconnect=False)
        if not ok:
            maze[:] = backup
            continue
        after_q = quantize_maze(maze, start, goal)
        if not after_q["validity"]["is_solvable"]:
            maze[:] = backup
            continue
        after_err = compute_metric_error(
            after_q["core_metrics"], after_q["aux_metrics"], ctx["target_ranges"]
        )
        tr = ctx["target_ranges"]
        if after_q["core_metrics"]["bfs_len_norm"] < tr["bfs_len_norm"][0] - 0.05:
            maze[:] = backup
            continue
        if after_q["core_metrics"]["wall_ratio"] < tr["wall_ratio"][0] * 0.5:
            maze[:] = backup
            continue
        return _make_op_result(
            "dead_end_inject", True, None, before_q, after_q, before_err, after_err, False
        )
    return _make_op_result("dead_end_inject", False, "no_valid_anchor", before_q, None, before_err, None, False)


def op_branch_inject(ctx: dict, rng: random.Random) -> dict:
    maze, start, goal = ctx["maze"], ctx["start"], ctx["goal"]
    before_q, before_err = ctx["q"], ctx["error"]
    path = bfs_shortest_path(maze, start, goal)
    if not path:
        return _make_op_result("branch_inject", False, "no_path", before_q, None, before_err, None, False)

    anchors = list(path[1:-1])
    off_path_free = [c for c in free_cells_all(maze) if c not in path]
    rng.shuffle(anchors)
    branch_len = rng.randint(2, 5)
    tr = ctx["target_ranges"]

    for anchor in anchors[:6]:
        backup = maze.copy()
        reconnect = rng.random() < 0.3
        ok, _ = _carve_branch(maze, anchor, branch_len, start, rng, allow_reconnect=reconnect)
        if not ok:
            maze[:] = backup
            continue
        after_q = quantize_maze(maze, start, goal)
        if not after_q["validity"]["is_solvable"]:
            maze[:] = backup
            continue
        if after_q["core_metrics"]["bfs_len_norm"] < tr["bfs_len_norm"][0]:
            maze[:] = backup
            continue
        after_err = compute_metric_error(
            after_q["core_metrics"], after_q["aux_metrics"], ctx["target_ranges"]
        )
        return _make_op_result("branch_inject", True, None, before_q, after_q, before_err, after_err, False)

    for anchor in off_path_free[:6]:
        backup = maze.copy()
        ok, _ = _carve_branch(maze, anchor, branch_len, start, rng, allow_reconnect=False)
        if not ok:
            maze[:] = backup
            continue
        after_q = quantize_maze(maze, start, goal)
        if not after_q["validity"]["is_solvable"]:
            maze[:] = backup
            continue
        after_err = compute_metric_error(
            after_q["core_metrics"], after_q["aux_metrics"], ctx["target_ranges"]
        )
        return _make_op_result("branch_inject", True, None, before_q, after_q, before_err, after_err, False)

    return _make_op_result("branch_inject", False, "no_valid_anchor", before_q, None, before_err, None, False)


def op_island_inject(ctx: dict, rng: random.Random) -> dict:
    maze, start, goal = ctx["maze"], ctx["start"], ctx["goal"]
    before_q, before_err = ctx["q"], ctx["error"]
    i_lo, i_hi = ctx["target_ranges"]["island_free_ratio"]
    if i_hi <= 0.0:
        return _make_op_result("island_inject", False, "island_disabled", before_q, None, before_err, None, False)

    n = maze.shape[0]
    reachable = reachable_free_cells(maze, start)
    island_size = max(1, int(rng.randint(1, 4) * max(i_lo, 0.02) * 20))
    wall_cands = [
        w for w in wall_cells_all(maze)
        if w not in reachable and all(nb in reachable or maze[nb] == 1 for nb in neighbors(w, n))
    ]
    rng.shuffle(wall_cands)
    for seed_cell in wall_cands[:30]:
        backup = maze.copy()
        island = [seed_cell]
        maze[seed_cell] = 0
        for _ in range(island_size - 1):
            frontier = []
            for cell in island:
                for q in neighbors(cell, n):
                    if maze[q] == 1 and q not in island and q not in reachable:
                        frontier.append(q)
            if not frontier:
                break
            nxt = rng.choice(frontier)
            maze[nxt] = 0
            island.append(nxt)
        after_q = quantize_maze(maze, start, goal)
        island_ratio = after_q["aux_metrics"].get("island_free_ratio")
        if not is_valid_number(island_ratio) or island_ratio > i_hi:
            maze[:] = backup
            continue
        if not after_q["validity"]["is_solvable"]:
            maze[:] = backup
            continue
        after_err = compute_metric_error(
            after_q["core_metrics"], after_q["aux_metrics"], ctx["target_ranges"]
        )
        return _make_op_result("island_inject", True, None, before_q, after_q, before_err, after_err, False)
    return _make_op_result("island_inject", False, "no_island_site", before_q, None, before_err, None, False)


def op_safe_wall_add(ctx: dict, rng: random.Random) -> dict:
    maze, start, goal = ctx["maze"], ctx["start"], ctx["goal"]
    before_q, before_err = ctx["q"], ctx["error"]
    main_path_set = set(ctx.get("main_path", []))
    cands = [
        c for c in free_cells_all(maze)
        if c != start and c != goal and c not in main_path_set
    ]
    rng.shuffle(cands)
    for cell in cands[:ctx["args"].max_operation_attempts]:
        backup = maze.copy()
        maze[cell] = 1
        after_q = quantize_maze(maze, start, goal)
        if not after_q["validity"]["is_solvable"]:
            maze[:] = backup
            continue
        after_err = compute_metric_error(
            after_q["core_metrics"], after_q["aux_metrics"], ctx["target_ranges"]
        )
        return _make_op_result("safe_wall_add", True, None, before_q, after_q, before_err, after_err, False)
    return _make_op_result("safe_wall_add", False, "no_candidate", before_q, None, before_err, None, False)


def op_safe_wall_remove(ctx: dict, rng: random.Random) -> dict:
    maze, start, goal = ctx["maze"], ctx["start"], ctx["goal"]
    before_q, before_err = ctx["q"], ctx["error"]
    tr = ctx["target_ranges"]
    n = maze.shape[0]
    cands = [w for w in wall_cells_all(maze) if any(maze[q] == 0 for q in neighbors(w, n))]
    rng.shuffle(cands)
    for cell in cands[: ctx["args"].max_operation_attempts]:
        backup = maze.copy()
        maze[cell] = 0
        after_q = quantize_maze(maze, start, goal)
        if not after_q["validity"]["is_solvable"]:
            maze[:] = backup
            continue
        cm = after_q["core_metrics"]
        if cm["bfs_len_norm"] < tr["bfs_len_norm"][0] or cm["detour_ratio"] < tr["detour_ratio"][0]:
            maze[:] = backup
            continue
        after_err = compute_metric_error(
            after_q["core_metrics"], after_q["aux_metrics"], ctx["target_ranges"]
        )
        return _make_op_result("safe_wall_remove", True, None, before_q, after_q, before_err, after_err, False)
    return _make_op_result("safe_wall_remove", False, "no_candidate", before_q, None, before_err, None, False)


def op_path_block_optional(ctx: dict, rng: random.Random) -> dict:
    maze, start, goal = ctx["maze"], ctx["start"], ctx["goal"]
    before_q, before_err = ctx["q"], ctx["error"]
    path = bfs_shortest_path(maze, start, goal)
    if not path or len(path) < 4:
        return _make_op_result("path_block_optional", False, "no_path", before_q, None, before_err, None, False)
    interior = [p for p in path[1:-1] if maze[p] == 0]
    rng.shuffle(interior)
    for cell in interior[: ctx["args"].max_operation_attempts]:
        backup = maze.copy()
        maze[cell] = 1
        after_q = quantize_maze(maze, start, goal)
        if not after_q["validity"]["is_solvable"]:
            maze[:] = backup
            continue
        after_err = compute_metric_error(
            after_q["core_metrics"], after_q["aux_metrics"], ctx["target_ranges"]
        )
        return _make_op_result(
            "path_block_optional", True, None, before_q, after_q, before_err, after_err, False
        )
    return _make_op_result("path_block_optional", False, "no_candidate", before_q, None, before_err, None, False)


OPERATIONS: Dict[str, Callable[[dict, random.Random], dict]] = {
    "dead_end_inject": op_dead_end_inject,
    "branch_inject": op_branch_inject,
    "island_inject": op_island_inject,
    "safe_wall_add": op_safe_wall_add,
    "safe_wall_remove": op_safe_wall_remove,
    "path_block_optional": op_path_block_optional,
}


def select_operation(
    core_metrics: dict,
    aux_metrics: dict,
    error_dict: dict,
    target_ranges: dict,
) -> Tuple[str, str, str]:
    per = error_dict["per_metric_error"]
    violated = max(
        list(CORE_METRICS) + ["island_free_ratio"],
        key=lambda k: per.get(k, 0.0),
    )
    if per.get(violated, 0) <= 0:
        violated = "wall_ratio"

    cm = core_metrics
    tr = target_ranges

    if violated == "wall_ratio":
        w = cm.get("wall_ratio")
        if is_valid_number(w) and w < tr["wall_ratio"][0]:
            return "safe_wall_add", "wall_ratio below target", violated
        return "safe_wall_remove", "wall_ratio above target", violated

    if violated == "bfs_len_norm":
        b = cm.get("bfs_len_norm")
        if is_valid_number(b) and b < tr["bfs_len_norm"][0]:
            return "path_block_optional", "bfs_len_norm below target", violated
        return "safe_wall_add", "bfs_len_norm above target", violated

    if violated == "detour_ratio":
        d = cm.get("detour_ratio")
        if is_valid_number(d) and d < tr["detour_ratio"][0]:
            return "path_block_optional", "detour_ratio below target", violated
        return "safe_wall_add", "detour_ratio above target", violated

    if violated in ("dead_end_ratio", "dead_end_depth_mean"):
        return "dead_end_inject", f"{violated} below target", violated

    if violated in ("junction_ratio", "junction_on_solution_ratio"):
        return "branch_inject", f"{violated} below target", violated

    if violated == "island_free_ratio":
        isl = aux_metrics.get("island_free_ratio")
        if is_valid_number(isl) and isl < tr["island_free_ratio"][0]:
            return "island_inject", "island_free_ratio below target", violated
        return "safe_wall_remove", "island_free_ratio above target", violated

    if violated == "reachable_free_ratio":
        r = cm.get("reachable_free_ratio")
        if is_valid_number(r) and r < tr["reachable_free_ratio"][0]:
            return "safe_wall_remove", "reachable_free_ratio below target", violated
        return "safe_wall_add", "reachable_free_ratio above target", violated

    return "safe_wall_add", f"fallback for {violated}", violated


def try_accept_operation(
    op_result: dict,
    maze: np.ndarray,
    backup: np.ndarray,
    args,
    rng: random.Random,
) -> bool:
    if not op_result["success"] or op_result["after_error"] is None:
        maze[:] = backup
        return False

    before_err = op_result["before_error"]
    after_err = op_result["after_error"]
    after_total = after_err["total_error"]
    before_total = before_err["total_error"]

    if after_err["all_in_range"]:
        op_result["accepted"] = True
        return True
    if after_total < before_total:
        op_result["accepted"] = True
        return True
    if after_total == before_total and after_err["core_in_range"]:
        op_result["accepted"] = True
        return True
    if args.allow_worse_prob > 0 and rng.random() < args.allow_worse_prob:
        op_result["accepted"] = True
        return True

    maze[:] = backup
    op_result["accepted"] = False
    if not op_result.get("reject_reason"):
        op_result["reject_reason"] = "error_not_improved"
    return False


# =============================================================================
# PF-MCG generator
# =============================================================================


def generate_metric_constrained_maze(args, rng: random.Random, target_ranges: dict) -> dict:
    n = args.grid_size
    operation_log: List[dict] = []
    operation_counts: Dict[str, int] = defaultdict(int)
    restart_count = 0
    edit_steps_used = 0
    main_path_fail_reason = None

    best: Optional[dict] = None
    best_error = float("inf")

    turn_ratio_range = (
        args.target_main_path_turn_ratio_min,
        args.target_main_path_turn_ratio_max,
    )

    for restart in range(args.max_restarts):
        restart_count = restart
        sr, sc = rng.randint(0, n - 1), rng.randint(0, n - 1)
        start = (sr, sc)
        D = rng.randint(args.target_manhattan_min, args.target_manhattan_max)
        goals = goal_candidates_for_manhattan(start, D, n)
        if not goals:
            for dd in (D - 1, D + 1):
                if dd >= 1:
                    goals = goal_candidates_for_manhattan(start, dd, n)
                    if goals:
                        D = dd
                        break
        if not goals:
            continue
        goal = rng.choice(goals)

        L = sample_main_path_len(
            rng, args.target_main_path_len_min, args.target_main_path_len_max, D
        )
        if L is None:
            main_path_fail_reason = "main_path_failed_len"
            continue

        implied_detour = L / D if D > 0 else float("nan")
        d_lo, d_hi = target_ranges["detour_ratio"]
        if not (d_lo <= implied_detour <= d_hi):
            b_lo = max(args.target_main_path_len_min, int(math.ceil(D * d_lo)))
            b_hi = min(args.target_main_path_len_max, int(math.floor(D * d_hi)))
            L2 = sample_main_path_len(rng, b_lo, b_hi, D)
            if L2 is not None:
                L = L2
                implied_detour = L / D
            elif not (d_lo <= implied_detour <= d_hi):
                continue

        main_path, fail_reason = generate_self_avoiding_main_path(
            start, goal, L, turn_ratio_range, n, rng, args.max_main_path_attempts
        )
        if main_path is None:
            main_path_fail_reason = fail_reason
            continue

        maze = np.ones((n, n), dtype=np.int8)
        for cell in main_path:
            maze[cell] = 0

        main_path_len = len(main_path) - 1
        main_path_turn_count = count_turns(main_path)
        main_path_detour = main_path_len / D if D > 0 else float("nan")

        q = quantize_maze(maze, start, goal)
        err = compute_metric_error(q["core_metrics"], q["aux_metrics"], target_ranges)

        local_log: List[dict] = []
        local_counts: Dict[str, int] = defaultdict(int)

        if err["all_in_range"]:
            return _pack_result(
                maze, start, goal, main_path, main_path_len, main_path_turn_count,
                main_path_detour, D, L, implied_detour, True, False, err, restart_count,
                0, local_log, local_counts, target_ranges, args,
            )

        for step in range(args.max_edit_steps):
            if err["all_in_range"]:
                break
            op_name, sel_reason, violated = select_operation(
                q["core_metrics"], q["aux_metrics"], err, target_ranges
            )
            ctx = {
                "maze": maze,
                "start": start,
                "goal": goal,
                "q": q,
                "error": err,
                "target_ranges": target_ranges,
                "args": args,
                "main_path": main_path,
            }
            backup = maze.copy()
            op_fn = OPERATIONS[op_name]
            op_result = op_fn(ctx, rng)
            accepted = try_accept_operation(op_result, maze, backup, args, rng)
            local_counts[op_name] += 1
            if accepted:
                q = quantize_maze(maze, start, goal)
                err = compute_metric_error(q["core_metrics"], q["aux_metrics"], target_ranges)
            trace = {
                "step": step,
                "selected_operation": op_name,
                "selection_reason": sel_reason,
                "most_violated_metric": violated,
                "before_metrics": op_result["before_metrics"],
                "after_metrics": op_result["after_metrics"],
                "delta_metrics": op_result["delta_metrics"],
                "before_error": op_result["before_error"],
                "after_error": op_result["after_error"],
                "accepted": accepted,
                "reject_reason": op_result.get("reject_reason"),
            }
            local_log.append(trace)

        edit_steps_used = len(local_log)
        if err["total_error"] < best_error:
            best_error = err["total_error"]
            best = _pack_result(
                maze.copy(), start, goal, main_path, main_path_len, main_path_turn_count,
                main_path_detour, D, L, implied_detour,
                err["all_in_range"], not err["all_in_range"], err, restart_count,
                edit_steps_used, local_log, local_counts, target_ranges, args,
            )
            operation_log = local_log
            operation_counts = local_counts

        if err["all_in_range"]:
            return _pack_result(
                maze, start, goal, main_path, main_path_len, main_path_turn_count,
                main_path_detour, D, L, implied_detour, True, False, err, restart_count,
                edit_steps_used, local_log, local_counts, target_ranges, args,
            )

    if best is None:
        raise RuntimeError("PF-MCG: all restarts failed to produce a maze")

    best["generation_metadata"]["main_path_fail_reason"] = main_path_fail_reason
    best["operation_trace"] = operation_log
    best["operation_counts"] = dict(operation_counts)
    return best


def _pack_result(
    maze, start, goal, main_path, main_path_len, main_path_turn_count,
    main_path_detour, D, L, implied_detour, success, failed_target, err,
    restart_count, edit_steps, op_log, op_counts, target_ranges, args,
) -> dict:
    q = quantize_maze(maze, start, goal)
    err = compute_metric_error(q["core_metrics"], q["aux_metrics"], target_ranges)
    failed_metrics = [k for k, v in err["in_range_flags"].items() if not v]
    return {
        "maze": maze,
        "start": start,
        "goal": goal,
        "main_path": main_path,
        "main_path_len": main_path_len,
        "main_path_turn_count": main_path_turn_count,
        "main_path_detour_ratio": main_path_detour,
        "generation_success": success and err["all_in_range"],
        "generation_failed_target": failed_target or not err["all_in_range"],
        "target_ranges": {k: list(v) for k, v in target_ranges.items()},
        "sampled_target_values": {
            "manhattan_D": D,
            "main_path_len_L": L,
            "implied_detour_ratio": implied_detour,
            "main_path_turn_ratio": main_path_turn_count / max(1, main_path_len - 1)
            if main_path_len > 1
            else 0.0,
        },
        "quantization": q,
        "metric_error": err,
        "generation_metadata": {
            "restart_count": restart_count,
            "edit_steps": edit_steps,
            "operation_counts": dict(op_counts),
            "final_error": err["total_error"],
            "failed_metrics": failed_metrics,
            "best_candidate_metrics": q["core_metrics"],
        },
        "operation_trace": op_log,
        "operation_counts": dict(op_counts),
    }


# =============================================================================
# aggregation & reporting
# =============================================================================


def metric_stats(values: Sequence[Any]) -> dict:
    nums = [float(x) for x in values if is_valid_number(x)]
    nan_count = len(values) - len(nums)
    if not nums:
        return {
            "mean": None, "std": None, "min": None,
            "p25": None, "p50": None, "p75": None, "max": None, "nan_count": nan_count,
        }
    arr = np.array(nums)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
        "nan_count": nan_count,
    }


def build_generation_summary(records: List[dict], quant_results: List[dict]) -> dict:
    n = len(records)
    success = sum(1 for r in records if r.get("generation_success"))
    target_hit = sum(
        1 for q in quant_results if q.get("metric_error", {}).get("all_in_range")
    )
    restarts = [r["generation_metadata"].get("restart_count", 0) for r in records]
    edits = [r["generation_metadata"].get("edit_steps", 0) for r in records]
    errors = [r["generation_metadata"].get("final_error", 0) for r in records]
    return {
        "n_requested": n,
        "n_generated": n,
        "success_count": success,
        "success_rate": success / n if n else None,
        "target_hit_rate": target_hit / n if n else None,
        "restart_mean": float(np.mean(restarts)) if restarts else None,
        "restart_p50": float(np.quantile(restarts, 0.5)) if restarts else None,
        "restart_max": int(max(restarts)) if restarts else None,
        "edit_steps_mean": float(np.mean(edits)) if edits else None,
        "edit_steps_p50": float(np.quantile(edits, 0.5)) if edits else None,
        "edit_steps_max": int(max(edits)) if edits else None,
        "final_error_mean": float(np.mean(errors)) if errors else None,
        "final_error_p50": float(np.quantile(errors, 0.5)) if errors else None,
        "final_error_max": float(max(errors)) if errors else None,
    }


def build_operation_delta_report(all_ops: List[dict]) -> List[dict]:
    by_type: Dict[str, List[dict]] = defaultdict(list)
    for op in all_ops:
        op_type = op.get("operation_type") or op.get("selected_operation") or "unknown"
        by_type[op_type].append(op)

    rows = []
    for op_type, ops in sorted(by_type.items()):
        attempts = len(ops)
        accepted = [o for o in ops if o.get("accepted")]
        acc_n = len(accepted)
        row: dict = {
            "operation_type": op_type,
            "attempt_count": attempts,
            "accept_count": acc_n,
            "accept_rate": acc_n / attempts if attempts else None,
        }
        for metric in DELTA_METRICS:
            deltas = []
            for o in accepted:
                dm = o.get("delta_metrics") or {}
                if dm.get(metric) is not None:
                    deltas.append(dm[metric])
            row[f"mean_delta_{metric}"] = float(np.mean(deltas)) if deltas else None
        rows.append(row)
    return rows


def build_target_hit_report(quant_results: List[dict], target_ranges: dict) -> List[dict]:
    rows = []
    check_keys = list(CORE_METRICS) + ["island_free_ratio"]
    for metric in check_keys:
        lo, hi = target_ranges[metric]
        if metric in CORE_METRICS:
            vals = [q["core_metrics"].get(metric) for q in quant_results]
        else:
            vals = [q["aux_metrics"].get(metric) for q in quant_results]
        valid = [float(v) for v in vals if is_valid_number(v)]
        n = len(valid)
        if n == 0:
            rows.append({
                "metric": metric, "target_min": lo, "target_max": hi,
                "hit_rate": None, "below_rate": None, "above_rate": None,
                "mean": None, "p25": None, "p50": None, "p75": None,
            })
            continue
        hit = sum(1 for v in valid if lo <= v <= hi)
        below = sum(1 for v in valid if v < lo)
        above = sum(1 for v in valid if v > hi)
        st = metric_stats(valid)
        rows.append({
            "metric": metric,
            "target_min": lo,
            "target_max": hi,
            "hit_rate": hit / n,
            "below_rate": below / n,
            "above_rate": above / n,
            "mean": st["mean"],
            "p25": st["p25"],
            "p50": st["p50"],
            "p75": st["p75"],
        })
    return rows


def build_metric_summary(quant_results: List[dict]) -> dict:
    out = {}
    for metric in CORE_METRICS:
        vals = [q["core_metrics"].get(metric) for q in quant_results]
        out[metric] = metric_stats(vals)
    return out


def build_top_k_examples(records: List[dict], quant_results: List[dict], k: int) -> dict:
    by_id = {r["maze_id"]: r for r in records}
    q_by_id = {q["maze_id"]: q for q in quant_results}

    def top_ids(key_fn, reverse=True):
        pairs = [(q["maze_id"], key_fn(q)) for q in quant_results]
        pairs = [(mid, v) for mid, v in pairs if is_valid_number(v)]
        pairs.sort(key=lambda x: x[1], reverse=reverse)
        return [{"maze_id": mid, "value": v} for mid, v in pairs[:k]]

    return {
        "highest_wall_ratio": top_ids(lambda q: q["core_metrics"]["wall_ratio"]),
        "lowest_wall_ratio": top_ids(lambda q: q["core_metrics"]["wall_ratio"], reverse=False),
        "highest_detour_ratio": top_ids(lambda q: q["core_metrics"]["detour_ratio"]),
        "highest_dead_end_depth_mean": top_ids(lambda q: q["core_metrics"]["dead_end_depth_mean"]),
        "highest_junction_ratio": top_ids(lambda q: q["core_metrics"]["junction_ratio"]),
        "lowest_reachable_free_ratio": top_ids(
            lambda q: q["core_metrics"]["reachable_free_ratio"], reverse=False
        ),
        "highest_final_error": top_ids(lambda q: q["metric_error"]["total_error"]),
        "lowest_final_error": top_ids(lambda q: q["metric_error"]["total_error"], reverse=False),
    }


def print_terminal_report(
    gen_summary: dict,
    target_hit: List[dict],
    metric_summary: dict,
    op_delta: List[dict],
    quant_results: List[dict],
    output_dir: Path,
) -> None:
    print("\n[3.0.1 Path-First Metric-Constrained Maze Generator]")
    print("\n=== 1. Generation Summary ===")
    for k in (
        "n_requested", "n_generated", "success_count", "success_rate", "target_hit_rate",
        "restart_mean", "edit_steps_mean", "final_error_mean",
    ):
        print(f"  {k}: {gen_summary.get(k)}")

    print("\n=== 2. Target Hit Report ===")
    print(f"{'metric':<32} {'hit%':>8} {'below%':>8} {'above%':>8} {'mean':>10} {'p50':>10}")
    print("-" * 78)
    for row in target_hit:
        print(
            f"{row['metric']:<32} {fmt_num(row.get('hit_rate'), 8, 3)} "
            f"{fmt_num(row.get('below_rate'), 8, 3)} {fmt_num(row.get('above_rate'), 8, 3)} "
            f"{fmt_num(row.get('mean'), 10)} {fmt_num(row.get('p50'), 10)}"
        )

    print("\n=== 3. Core Metric Summary ===")
    print(f"{'metric':<32} {'mean':>10} {'std':>10} {'p50':>10}")
    print("-" * 64)
    for metric in CORE_METRICS:
        st = metric_summary[metric]
        print(f"{metric:<32} {fmt_num(st['mean'], 10)} {fmt_num(st['std'], 10)} {fmt_num(st['p50'], 10)}")

    print("\n=== 4. Operation Delta Report ===")
    print(f"{'operation':<22} {'attempts':>8} {'accepts':>8} {'rate':>8} {'Δdead_end':>12} {'Δjunction':>12}")
    print("-" * 74)
    for row in op_delta:
        print(
            f"{row['operation_type']:<22} {row['attempt_count']:>8} {row['accept_count']:>8} "
            f"{fmt_num(row.get('accept_rate'), 8, 3)} "
            f"{fmt_num(row.get('mean_delta_dead_end_ratio'), 12)} "
            f"{fmt_num(row.get('mean_delta_junction_ratio'), 12)}"
        )

    print("\n=== 5. Top Failed Metrics (by above/below rate) ===")
    failed = sorted(
        target_hit,
        key=lambda r: (r.get("hit_rate") or 0),
    )
    for row in failed[:5]:
        print(
            f"  {row['metric']}: hit={fmt_num(row.get('hit_rate'), 8, 3)} "
            f"below={fmt_num(row.get('below_rate'), 8, 3)} "
            f"above={fmt_num(row.get('above_rate'), 8, 3)}"
        )

    print(f"\n=== 6. Output Directory ===\n  {output_dir}")


# =============================================================================
# visualization
# =============================================================================


def save_maze_figure(
    maze: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    main_path: Optional[List[Tuple[int, int]]],
    out_path: Path,
    title_lines: List[str],
) -> None:
    n = maze.shape[0]
    fig, ax = plt.subplots(figsize=(4.2, 4.8))
    display = np.where(maze == 1, 0.0, 1.0)
    ax.imshow(display, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    if main_path:
        pr, pc = zip(*main_path)
        ax.plot(pc, pr, color="orange", linewidth=1.2, alpha=0.6, linestyle="--")
    path = bfs_shortest_path(maze, start, goal)
    if path:
        pr, pc = zip(*path)
        ax.plot(pc, pr, color="cyan", linewidth=1.5, alpha=0.8)
    ax.plot(start[1], start[0], "go", markersize=8)
    ax.plot(goal[1], goal[0], "r*", markersize=10)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.grid(True, color="lightgray", linewidth=0.5)
    ax.set_title("\n".join(title_lines), fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_visualizations(
    records: List[dict],
    quant_results: List[dict],
    output_dir: Path,
    n_samples: int,
    top_k: dict,
) -> None:
    q_by_id = {q["maze_id"]: q for q in quant_results}
    rec_by_id = {r["maze_id"]: r for r in records}

    def render(rec, subdir: Path):
        ensure_dir(subdir)
        q = q_by_id[rec["maze_id"]]
        cm = q["core_metrics"]
        aux = q["aux_metrics"]
        title = [
            rec["maze_id"],
            f"success={rec.get('generation_success')} err={rec['generation_metadata'].get('final_error', 0):.3f}",
            f"bfs={aux.get('bfs_len')} wall={cm['wall_ratio']:.3f} detour={cm['detour_ratio']:.3f}",
            f"de={cm['dead_end_ratio']:.3f} junc={cm['junction_ratio']:.3f}",
            f"de_depth={cm['dead_end_depth_mean']:.2f} j_on_sol={cm['junction_on_solution_ratio']:.3f}",
            f"island={aux.get('island_free_ratio', 0):.3f}",
        ]
        maze = np.array(rec["maze"], dtype=np.int8)
        mp = [tuple(x) for x in rec.get("main_path", [])]
        save_maze_figure(
            maze, tuple(rec["start"]), tuple(rec["goal"]), mp,
            subdir / f"{rec['maze_id']}.png", title,
        )

    sample_dir = output_dir / "samples"
    ensure_dir(sample_dir)
    for rec in records[:n_samples]:
        render(rec, sample_dir)

    top_dir = output_dir / "top_k_samples"
    seen = set()
    for key, items in top_k.items():
        sub = top_dir / key
        for item in items[:3]:
            mid = item["maze_id"]
            if mid in seen:
                continue
            seen.add(mid)
            if mid in rec_by_id:
                render(rec_by_id[mid], sub)

    fail_dir = output_dir / "failed_samples"
    for rec in records:
        if rec.get("generation_failed_target"):
            render(rec, fail_dir)


# =============================================================================
# CLI
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"{VERSION} PF-MCG experiment")
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--grid-size", type=int, default=DEFAULTS["grid_size"])
    p.add_argument("--n-mazes", type=int, default=DEFAULTS["n_mazes"])
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--no-tqdm", action="store_true")
    p.add_argument("--no-viz", action="store_true")
    p.add_argument("--n-trace-samples", type=int, default=DEFAULTS["n_trace_samples"])
    p.add_argument("--n-top-k", type=int, default=DEFAULTS["n_top_k"])
    p.add_argument("--n-sample-images", type=int, default=DEFAULTS["n_sample_images"])

    for prefix in (
        "target_wall_ratio", "target_bfs_len_norm", "target_detour_ratio",
        "target_dead_end_ratio", "target_junction_ratio", "target_dead_end_depth_mean",
        "target_junction_on_solution_ratio", "target_island_free_ratio",
    ):
        p.add_argument(f"--{prefix.replace('_', '-')}-min", type=float, default=DEFAULTS[f"{prefix}_min"])
        p.add_argument(f"--{prefix.replace('_', '-')}-max", type=float, default=DEFAULTS[f"{prefix}_max"])

    p.add_argument("--target-manhattan-min", type=int, default=DEFAULTS["target_manhattan_min"])
    p.add_argument("--target-manhattan-max", type=int, default=DEFAULTS["target_manhattan_max"])
    p.add_argument("--target-main-path-len-min", type=int, default=DEFAULTS["target_main_path_len_min"])
    p.add_argument("--target-main-path-len-max", type=int, default=DEFAULTS["target_main_path_len_max"])
    p.add_argument("--target-main-path-turn-ratio-min", type=float, default=DEFAULTS["target_main_path_turn_ratio_min"])
    p.add_argument("--target-main-path-turn-ratio-max", type=float, default=DEFAULTS["target_main_path_turn_ratio_max"])

    p.add_argument("--max-restarts", type=int, default=DEFAULTS["max_restarts"])
    p.add_argument("--max-edit-steps", type=int, default=DEFAULTS["max_edit_steps"])
    p.add_argument("--max-main-path-attempts", type=int, default=DEFAULTS["max_main_path_attempts"])
    p.add_argument("--max-operation-attempts", type=int, default=DEFAULTS["max_operation_attempts"])
    p.add_argument("--allow-worse-prob", type=float, default=DEFAULTS["allow_worse_prob"])
    return p


def default_run_name(args) -> str:
    return f"n_mazes-{args.n_mazes}__grid-{args.grid_size}"


def resolve_output_dir(args) -> Path:
    run_name = args.run_name or default_run_name(args)
    if args.output_dir:
        return Path(args.output_dir)
    return V3_ROOT / "outputs" / VERSION / run_name


def resolved_config_from_args(args) -> dict:
    cfg = {k: getattr(args, k.replace("-", "_"), None) for k in DEFAULTS if hasattr(args, k)}
    cfg.update({
        "version": VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "generator_name": GENERATOR_NAME,
        "seed": args.seed,
        "grid_size": args.grid_size,
        "n_mazes": args.n_mazes,
        "target_ranges": build_target_ranges_record(args),
        "no_tqdm": args.no_tqdm,
        "no_viz": args.no_viz,
    })
    for k, v in DEFAULTS.items():
        attr = k
        if hasattr(args, attr):
            cfg[attr] = getattr(args, attr)
    return cfg


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    output_dir = resolve_output_dir(args)
    ensure_dir(output_dir)

    target_ranges = build_target_ranges(args)
    target_ranges_record = build_target_ranges_record(args)

    run_metadata = {
        "version": VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "generator_name": GENERATOR_NAME,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(SCRIPT_PATH),
        "output_dir": str(output_dir),
        "assumptions": [
            "Path-First: all cells start as wall; main_path carved first.",
            "Final BFS metrics from MazeQuantizer override main_path skeleton.",
            "island_inject enabled when target_island_free_ratio_max > 0.",
            "reachable_free_ratio range derived from island_free_ratio range.",
            "No DP / incremental BFS — full BFS each quantize step.",
            "Operation acceptance: greedy error reduction (allow_worse_prob default 0).",
        ],
    }

    maze_records: List[dict] = []
    quant_results: List[dict] = []
    all_operation_events: List[dict] = []
    trace_samples: List[dict] = []

    for i in maybe_tqdm(range(args.n_mazes), args, desc="pf_mcg", total=args.n_mazes):
        result = generate_metric_constrained_maze(args, rng, target_ranges)
        maze_id = f"pf_{i:05d}"
        q = result["quantization"]
        q_with_err = {**q, "metric_error": result["metric_error"], "maze_id": maze_id}

        record = {
            "maze_id": maze_id,
            "maze": result["maze"].tolist(),
            "start": list(result["start"]),
            "goal": list(result["goal"]),
            "generator_name": GENERATOR_NAME,
            "generation_success": result["generation_success"],
            "generation_failed_target": result["generation_failed_target"],
            "target_ranges": target_ranges_record,
            "sampled_target_values": result["sampled_target_values"],
            "main_path": [list(c) for c in result["main_path"]],
            "main_path_len": result["main_path_len"],
            "main_path_turn_count": result["main_path_turn_count"],
            "main_path_detour_ratio": result["main_path_detour_ratio"],
            "generation_metadata": result["generation_metadata"],
        }
        maze_records.append(record)
        quant_results.append(q_with_err)

        for step_trace in result.get("operation_trace", []):
            evt = {"maze_id": maze_id, **step_trace}
            all_operation_events.append(evt)

        if len(trace_samples) < args.n_trace_samples:
            trace_samples.append({"maze_id": maze_id, "trace": result.get("operation_trace", [])})

    failed_ids = [r["maze_id"] for r in maze_records if r["generation_failed_target"]]
    for mid in failed_ids[: max(0, args.n_trace_samples - len(trace_samples))]:
        rec = next(r for r in maze_records if r["maze_id"] == mid)
        if not any(t["maze_id"] == mid for t in trace_samples):
            trace_samples.append({"maze_id": mid, "trace": []})

    generation_summary = build_generation_summary(maze_records, quant_results)
    metric_summary = build_metric_summary(quant_results)
    operation_delta_report = build_operation_delta_report(all_operation_events)
    target_hit_report = build_target_hit_report(quant_results, target_ranges)
    top_k_examples = build_top_k_examples(maze_records, quant_results, args.n_top_k)

    run_metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    run_metadata["n_maze_records"] = len(maze_records)

    save_json(output_dir / "resolved_config.json", resolved_config_from_args(args))
    save_json(output_dir / "run_metadata.json", run_metadata)
    save_json(output_dir / "maze_records.json", maze_records)
    save_json(output_dir / "quantization_results.json", quant_results)
    save_json(output_dir / "metric_summary.json", metric_summary)
    save_json(output_dir / "generation_summary.json", generation_summary)
    save_json(output_dir / "operation_delta_report.json", operation_delta_report)
    save_json(output_dir / "operation_trace_samples.json", trace_samples)
    save_json(output_dir / "target_hit_report.json", target_hit_report)
    save_json(output_dir / "top_k_examples_by_metric.json", top_k_examples)

    if not args.no_viz:
        save_visualizations(maze_records, quant_results, output_dir, args.n_sample_images, top_k_examples)

    print_terminal_report(
        generation_summary, target_hit_report, metric_summary,
        operation_delta_report, quant_results, output_dir,
    )


if __name__ == "__main__":
    main()
