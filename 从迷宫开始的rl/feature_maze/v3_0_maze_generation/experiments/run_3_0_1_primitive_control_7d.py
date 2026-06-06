#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3.0.1-v2 — Primitive Control + 7D Truncated-Normal Sampling Test.

Default mode ``full``: run all primitive-specific tests, then normal_7d.

Engineering correction over PF-MCG:
  - primitive_test: isolate each structural primitive's controllability
  - normal_7d: sample 7D target vector, track target vs final distribution
  - full: all primitives sequentially, then normal_7d

No CNN / RL / GAN / PCGML. Full BFS quantization only.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

VERSION = "3.0.1_v2"
EXPERIMENT_NAME = "primitive_control_7d_v2"
GENERATOR_NAME = "pf_mcg"

ACTIONS: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
INF = 10**9
ALL_OPS = (
    "dead_end_inject",
    "branch_inject",
    "island_inject",
    "safe_wall_add",
    "safe_wall_remove",
    "path_block_optional",
)

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

D7_KEYS = (
    "L_BFS",
    "detour_ratio",
    "wall_ratio",
    "dead_end_ratio",
    "dead_end_depth_mean",
    "junction_ratio",
    "island_free_ratio",
)

PRIMITIVE_TEST_TARGETS: Dict[str, List[Any]] = {
    "wall": [0.20, 0.25, 0.30, 0.35, 0.40],
    "path": [(10, 1.3), (14, 1.8), (18, 2.4), (22, 3.0), (26, 2.0)],
    "dead_end": [0.05, 0.10, 0.15, 0.20],
    "dead_end_depth": [1.0, 2.0, 3.0, 4.0],
    "junction": [0.30, 0.45, 0.60, 0.75],
    "island": [0.00, 0.03, 0.06, 0.10],
}

PRIMITIVE_TEST_NAMES: Tuple[str, ...] = ("path", "dead_end", "dead_end_depth", "junction", "island", "wall")

PRIMARY_METRICS_BY_PRIMITIVE = {
    "path": ("L_BFS", "detour_ratio"),
    "dead_end": ("dead_end_ratio",),
    "dead_end_depth": ("dead_end_depth_mean",),
    "junction": ("junction_ratio",),
    "island": ("island_free_ratio",),
    "wall": ("wall_ratio",),
}

PRIMITIVE_ROLE = {
    "path": "primary_structure",
    "dead_end": "primary_structure",
    "dead_end_depth": "primary_structure",
    "junction": "primary_structure",
    "island": "primary_structure",
    "wall": "secondary_final_adjustment",
}

DEFAULTS: Dict[str, Any] = dict(
    mode="full",
    primitive_test_name="dead_end",
    seed=42,
    grid_size=8,
    n_mazes=30,
    n_mazes_per_primitive=8,
    run_name=None,
    output_dir=None,
    no_tqdm=False,
    no_viz=False,
    max_restarts=100,
    max_edit_steps=80,
    max_main_path_attempts=500,
    max_operation_attempts=30,
    allow_worse_prob=0.0,
    reject_shortcut_preservation_min=0.0,
    target_manhattan_min=3,
    target_manhattan_max=13,
    target_main_path_turn_ratio_min=0.15,
    target_main_path_turn_ratio_max=0.75,
    L_mean=16,
    L_std=4,
    L_min=8,
    L_max=28,
    detour_mean=2.0,
    detour_std=0.6,
    detour_min=1.2,
    detour_max=4.0,
    wall_ratio_mean=0.30,
    wall_ratio_std=0.06,
    wall_ratio_min=0.18,
    wall_ratio_max=0.42,
    dead_end_ratio_mean=0.10,
    dead_end_ratio_std=0.04,
    dead_end_ratio_min=0.03,
    dead_end_ratio_max=0.22,
    dead_end_depth_mean_mean=2.0,
    dead_end_depth_mean_std=0.8,
    dead_end_depth_mean_min=1.0,
    dead_end_depth_mean_max=4.0,
    junction_ratio_mean=0.55,
    junction_ratio_std=0.12,
    junction_ratio_min=0.25,
    junction_ratio_max=0.80,
    island_free_ratio_mean=0.03,
    island_free_ratio_std=0.03,
    island_free_ratio_min=0.00,
    island_free_ratio_max=0.12,
    eps_L=1,
    eps_detour=0.20,
    eps_wall_ratio=0.03125,
    eps_dead_end_ratio=0.025,
    eps_dead_end_depth_mean=0.35,
    eps_junction_ratio=0.05,
    eps_island_free_ratio=0.025,
    sampling_max_attempts=200,
)

SCRIPT_PATH = Path(__file__).resolve()
V3_ROOT = SCRIPT_PATH.parents[1]


# =============================================================================
# io / utils
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


def maybe_tqdm(it, args, desc="", total=None):
    if getattr(args, "no_tqdm", False) or tqdm is None:
        return it
    return tqdm(it, desc=desc, total=total, leave=False, file=sys.stderr, dynamic_ncols=True)


def is_valid_number(x: Any) -> bool:
    if x is None:
        return False
    try:
        v = float(x)
        return not (math.isnan(v) or math.isinf(v))
    except (TypeError, ValueError):
        return False


def fmt_num(v: Any, w: int = 10, p: int = 4) -> str:
    if not is_valid_number(v):
        return f"{'—':>{w}s}"
    return f"{float(v):>{w}.{p}f}"


def metric_stats(values: Sequence[Any]) -> dict:
    nums = [float(x) for x in values if is_valid_number(x)]
    if not nums:
        return {"mean": None, "std": None, "p25": None, "p50": None, "p75": None, "min": None, "max": None}
    a = np.array(nums)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "p25": float(np.quantile(a, 0.25)),
        "p50": float(np.quantile(a, 0.50)),
        "p75": float(np.quantile(a, 0.75)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


# =============================================================================
# grid / BFS
# =============================================================================


def in_bounds(p: Tuple[int, int], n: int) -> bool:
    return 0 <= p[0] < n and 0 <= p[1] < n


def neighbors(p: Tuple[int, int], n: int):
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


def bfs_shortest_path(maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]):
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
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def reachable_free_cells(maze: np.ndarray, start: Tuple[int, int]) -> set:
    if maze[start] == 1:
        return set()
    n = maze.shape[0]
    seen, dq = {start}, deque([start])
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


def build_adjacency(maze: np.ndarray, cells: set):
    n = maze.shape[0]
    adj = {c: [] for c in cells}
    for p in cells:
        for q in neighbors(p, n):
            if q in cells:
                adj[p].append(q)
    return adj


def is_straight_corridor(adj: dict, v: Tuple[int, int]) -> bool:
    nbs = adj[v]
    if len(nbs) != 2:
        return False
    a, b = nbs[0], nbs[1]
    return a[0] + b[0] == 2 * v[0] and a[1] + b[1] == 2 * v[1]


def compute_dead_end_depths(adj, degrees, start, goal, solution_set, reachable):
    depths = []
    for de in sorted(reachable):
        if degrees.get(de, 0) != 1:
            continue
        depth, prev, cur = 0, None, de
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
            prev, cur = cur, nxt
        depths.append(depth)
    return depths


def island_components(maze: np.ndarray, start: Tuple[int, int]) -> Tuple[int, float, int]:
    """Return (count, mean_size, max_size) of free components NOT reachable from start."""
    n = maze.shape[0]
    reachable = reachable_free_cells(maze, start)
    all_free = set(free_cells_all(maze))
    island_cells = all_free - reachable
    if not island_cells:
        return 0, 0.0, 0
    unvisited = set(island_cells)
    sizes = []
    while unvisited:
        seed = unvisited.pop()
        comp = {seed}
        dq = deque([seed])
        while dq:
            p = dq.popleft()
            for q in neighbors(p, n):
                if q in unvisited:
                    unvisited.remove(q)
                    comp.add(q)
                    dq.append(q)
        sizes.append(len(comp))
    return len(sizes), float(np.mean(sizes)), int(max(sizes))


# =============================================================================
# MazeQuantizer extended
# =============================================================================


def quantize_maze(
    maze: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    main_path_len: Optional[int] = None,
) -> dict:
    n = maze.shape[0]
    all_free = free_cells_all(maze)
    free_count = len(all_free)
    wall_count = n * n - free_count
    start_free = maze[start] == 0
    goal_free = maze[goal] == 0
    path = bfs_shortest_path(maze, start, goal) if start_free and goal_free else None
    is_solvable = path is not None
    reachable = reachable_free_cells(maze, start) if start_free else set()
    reachable_free_ratio = len(reachable) / free_count if free_count else float("nan")
    island_free_ratio = 1.0 - reachable_free_ratio if free_count else float("nan")

    bfs_len = (len(path) - 1) if path else None
    manhattan_dist = manhattan(start, goal)
    wall_ratio = wall_count / (n * n)
    bfs_len_norm = bfs_len / (n * n - 1) if bfs_len is not None else float("nan")
    detour_ratio = (
        bfs_len / manhattan_dist
        if manhattan_dist > 0 and bfs_len is not None
        else float("nan")
    )

    degree_histogram = {f"degree_{d}": 0 for d in range(5)}
    dead_end_ratio = junction_ratio = corridor_ratio = float("nan")
    straight_corridor_ratio = turn_corridor_ratio = float("nan")
    degree_3_ratio = degree_4_ratio = float("nan")
    off_solution_junction_ratio = float("nan")
    solution_turn_count = 0
    solution_turn_ratio = float("nan")
    dead_end_depth_mean = 0.0
    dead_end_depth_max = 0
    junction_on_solution_ratio = float("nan")

    if is_solvable and reachable:
        adj = build_adjacency(maze, reachable)
        degrees = {v: len(adj[v]) for v in reachable}
        for v in reachable:
            degree_histogram[f"degree_{min(degrees[v], 4)}"] += 1
        n_reach = len(reachable)
        dead_end_ratio = sum(1 for v in reachable if degrees[v] == 1) / n_reach
        junction_ratio = sum(1 for v in reachable if degrees[v] >= 3) / n_reach
        degree_3_ratio = sum(1 for v in reachable if degrees[v] == 3) / n_reach
        degree_4_ratio = sum(1 for v in reachable if degrees[v] >= 4) / n_reach
        corridor_nodes = [v for v in reachable if degrees[v] == 2]
        corridor_ratio = len(corridor_nodes) / n_reach
        if corridor_nodes:
            straight_n = sum(1 for v in corridor_nodes if is_straight_corridor(adj, v))
            straight_corridor_ratio = straight_n / n_reach
            turn_corridor_ratio = (len(corridor_nodes) - straight_n) / n_reach
        if path:
            solution_set = set(path)
            junc_nodes = [v for v in reachable if degrees[v] >= 3]
            off_solution_junction_ratio = sum(1 for v in junc_nodes if v not in solution_set) / max(
                1, len(junc_nodes)
            )
            depths = compute_dead_end_depths(adj, degrees, start, goal, solution_set, reachable)
            if depths:
                dead_end_depth_mean = float(np.mean(depths))
                dead_end_depth_max = int(max(depths))
            junction_on_solution_ratio = sum(1 for v in path if degrees.get(v, 0) >= 3) / len(path)
            solution_turn_count = count_turns(path)
            solution_turn_ratio = solution_turn_count / max(1, (bfs_len or 1) - 1)

    mpl = main_path_len if main_path_len is not None else (bfs_len if bfs_len is not None else 0)
    final_bfs = bfs_len if bfs_len is not None else 0
    shortcut_gain = mpl - final_bfs if mpl else 0
    main_path_preservation_ratio = final_bfs / max(1, mpl) if mpl else float("nan")
    main_path_detour = (
        mpl / manhattan_dist if manhattan_dist > 0 and mpl else float("nan")
    )
    final_detour = detour_ratio

    ic_count, ic_mean, ic_max = island_components(maze, start)

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
        "island_free_ratio": island_free_ratio,
        "bfs_len": bfs_len,
        "manhattan_dist": manhattan_dist,
        "corridor_ratio": corridor_ratio,
        "straight_corridor_ratio": straight_corridor_ratio,
        "turn_corridor_ratio": turn_corridor_ratio,
        "solution_turn_count": solution_turn_count,
        "solution_turn_ratio": solution_turn_ratio,
        "off_solution_junction_ratio": off_solution_junction_ratio,
        "degree_3_ratio": degree_3_ratio,
        "degree_4_ratio": degree_4_ratio,
        "dead_end_depth_max": dead_end_depth_max,
        "free_cell_count": free_count,
        "wall_cell_count": wall_count,
        "main_path_len": mpl,
        "final_bfs_len": final_bfs,
        "main_path_detour_ratio": main_path_detour,
        "final_detour_ratio": final_detour,
        "shortcut_gain": shortcut_gain,
        "main_path_preservation_ratio": main_path_preservation_ratio,
        "island_component_count": ic_count,
        "island_size_mean": ic_mean,
        "island_size_max": ic_max,
        "degree_histogram": degree_histogram,
    }
    return {
        "validity": {"is_solvable": is_solvable, "reachable_free_ratio": reachable_free_ratio},
        "wall_and_path": {
            "wall_ratio": wall_ratio,
            "bfs_len": bfs_len,
            "bfs_len_norm": bfs_len_norm,
            "manhattan_dist": manhattan_dist,
            "detour_ratio": detour_ratio,
        },
        "graph_structure": {
            "degree_histogram": degree_histogram,
            "dead_end_ratio": dead_end_ratio,
            "junction_ratio": junction_ratio,
            "corridor_ratio": corridor_ratio,
            "degree_3_ratio": degree_3_ratio,
            "degree_4_ratio": degree_4_ratio,
        },
        "playability_proxy": {
            "dead_end_depth_mean": dead_end_depth_mean,
            "dead_end_depth_max": dead_end_depth_max,
            "junction_on_solution_ratio": junction_on_solution_ratio,
            "solution_path_selection": "one_bfs_path",
        },
        "core_metrics": core_metrics,
        "aux_metrics": aux_metrics,
    }


# =============================================================================
# sampling
# =============================================================================


def sample_truncated_normal(mean: float, std: float, low: float, high: float, rng: random.Random) -> Tuple[float, bool, Optional[str]]:
    if std <= 0:
        v = float(np.clip(mean, low, high))
        return v, mean != v or std == 0, "zero_std_clip" if std == 0 else None
    for _ in range(200):
        v = rng.gauss(mean, std)
        if low <= v <= high:
            return v, False, None
    v = float(np.clip(rng.gauss(mean, std), low, high))
    return v, True, "truncation_fallback_clip"


def get_epsilons(args) -> dict:
    n = args.grid_size
    return {
        "L_BFS": args.eps_L,
        "detour_ratio": args.eps_detour,
        "wall_ratio": args.eps_wall_ratio,
        "dead_end_ratio": args.eps_dead_end_ratio,
        "dead_end_depth_mean": args.eps_dead_end_depth_mean,
        "junction_ratio": args.eps_junction_ratio,
        "island_free_ratio": args.eps_island_free_ratio,
        "bfs_len_norm": args.eps_L / max(1, n * n - 1),
    }




def feasible_manhattan_values_for_L(args, L: int) -> List[int]:
    """Return feasible Manhattan D values for a self-avoiding path length L.

    A target path length L and Manhattan distance D must satisfy L >= D and
    parity (L-D) even. This function makes feasibility explicit instead of
    silently accepting impossible L / detour combinations.
    """
    n = args.grid_size
    max_grid_D = 2 * (n - 1)
    lo = max(1, int(args.target_manhattan_min))
    hi = min(int(args.target_manhattan_max), max_grid_D, int(L))
    return [d for d in range(lo, hi + 1) if (L - d) % 2 == 0]


def choose_adjusted_D(args, L: int, detour: float) -> Tuple[Optional[int], dict]:
    raw_D = max(1, int(round(float(L) / max(float(detour), 1e-9))))
    feasible = feasible_manhattan_values_for_L(args, int(L))
    meta = {
        "raw_D": raw_D,
        "adjusted_D": raw_D,
        "D_adjusted": False,
        "infeasible_path_target": False,
        "feasible_D_values": feasible[:],
        "path_feasibility_rule": "L>=D, (L-D)%2==0, D within manhattan/grid bounds",
    }
    if raw_D in feasible:
        return raw_D, meta
    if feasible:
        adjusted = min(feasible, key=lambda d: (abs(d - raw_D), d))
        meta["adjusted_D"] = adjusted
        meta["D_adjusted"] = True
        return adjusted, meta
    meta["adjusted_D"] = None
    meta["D_adjusted"] = True
    meta["infeasible_path_target"] = True
    return None, meta


def primary_metrics_for_primitive(name: str) -> Tuple[str, ...]:
    return PRIMARY_METRICS_BY_PRIMITIVE.get(name, tuple())


def primary_hit_from_pair(pair: dict, primitive_name: str) -> bool:
    metrics = primary_metrics_for_primitive(primitive_name)
    if not metrics:
        return False
    flags = pair.get("range_hit_flags", {})
    normalized = pair.get("normalized_errors", {})
    # For L_BFS there is no range_hit_flags key, so use normalized error <= 1.
    for m in metrics:
        if m in flags:
            if not bool(flags[m]):
                return False
        else:
            if not is_valid_number(normalized.get(m)) or float(normalized[m]) > 1.0:
                return False
    return True


def side_effect_failed_metrics(pair: dict, primitive_name: str) -> List[str]:
    primary = set(primary_metrics_for_primitive(primitive_name))
    flags = pair.get("range_hit_flags", {})
    out = []
    for k, ok in flags.items():
        if k not in primary and not ok:
            out.append(k)
    return out

def sampled_7d_to_ranges(targets: dict, eps: dict, n: int) -> dict:
    L = targets["L_BFS"]
    det = targets["detour_ratio"]
    return {
        "reachable_free_ratio": (
            1.0 - (targets["island_free_ratio"] + eps["island_free_ratio"]),
            1.0 - max(0.0, targets["island_free_ratio"] - eps["island_free_ratio"]),
        ),
        "wall_ratio": (
            targets["wall_ratio"] - eps["wall_ratio"],
            targets["wall_ratio"] + eps["wall_ratio"],
        ),
        "bfs_len_norm": (
            (L - eps["L_BFS"]) / (n * n - 1),
            (L + eps["L_BFS"]) / (n * n - 1),
        ),
        "detour_ratio": (det - eps["detour_ratio"], det + eps["detour_ratio"]),
        "dead_end_ratio": (
            targets["dead_end_ratio"] - eps["dead_end_ratio"],
            targets["dead_end_ratio"] + eps["dead_end_ratio"],
        ),
        "dead_end_depth_mean": (
            targets["dead_end_depth_mean"] - eps["dead_end_depth_mean"],
            targets["dead_end_depth_mean"] + eps["dead_end_depth_mean"],
        ),
        "junction_ratio": (
            targets["junction_ratio"] - eps["junction_ratio"],
            targets["junction_ratio"] + eps["junction_ratio"],
        ),
        "junction_on_solution_ratio": (0.0, 1.0),
        "island_free_ratio": (
            max(0.0, targets["island_free_ratio"] - eps["island_free_ratio"]),
            targets["island_free_ratio"] + eps["island_free_ratio"],
        ),
    }


def final_value_for_d7(q: dict, key: str) -> Optional[float]:
    if key == "L_BFS":
        return q["aux_metrics"].get("bfs_len")
    if key == "island_free_ratio":
        return q["aux_metrics"].get("island_free_ratio")
    if key in q["core_metrics"]:
        return q["core_metrics"].get(key)
    return None


def compute_range_error(core: dict, aux: dict, ranges: dict) -> dict:
    per, flags = {}, {}
    keys = list(CORE_METRICS) + ["island_free_ratio"]
    for k in keys:
        lo, hi = ranges[k]
        m = aux.get("island_free_ratio") if k == "island_free_ratio" else core.get(k)
        if not is_valid_number(m):
            e = 100.0
        else:
            mv = float(m)
            e = max(0.0, lo - mv, mv - hi)
        per[k] = e
        flags[k] = e == 0.0
    total = sum(per.values())
    core_ok = all(flags[k] for k in CORE_METRICS)
    return {
        "total_error": total,
        "per_metric_error": per,
        "in_range_flags": flags,
        "all_in_range": core_ok and flags["island_free_ratio"],
        "core_in_range": core_ok,
    }


def compute_sampled_target_error(q: dict, targets: dict, eps: dict, weights: Optional[dict] = None) -> dict:
    weights = weights or {}
    per_abs, per_norm, flags = {}, {}, {}
    for k in D7_KEYS:
        z = targets[k]
        m = final_value_for_d7(q, k)
        scale = max(eps.get(k, 0.01), 1e-6)
        if not is_valid_number(m):
            per_abs[k] = None
            per_norm[k] = 1.0
        else:
            ae = abs(float(m) - float(z))
            per_abs[k] = ae
            per_norm[k] = ae / scale
        lo = targets.get(f"_range_{k}", (z - eps[k], z + eps[k]))[0] if False else z - eps[k]
        hi = z + eps[k]
        flags[k] = is_valid_number(m) and lo <= float(m) <= hi
    wsum = sum(weights.get(k, 1.0) * per_norm[k] for k in D7_KEYS)
    wden = sum(weights.get(k, 1.0) for k in D7_KEYS)
    return {
        "total_error": wsum / wden if wden else wsum,
        "per_metric_abs_error": per_abs,
        "per_metric_normalized_error": per_norm,
        "in_range_flags": flags,
    }


def sample_7d_target(args, rng: random.Random, debug: dict) -> Tuple[Optional[dict], Optional[str]]:
    eps = get_epsilons(args)
    meta = {"sampling_fallback_used": False, "sampling_fallback_reason": None, "reject_reasons": []}
    n = args.grid_size

    for attempt in range(args.sampling_max_attempts):
        fb_log = []
        L, fb, r = sample_truncated_normal(args.L_mean, args.L_std, args.L_min, args.L_max, rng)
        if fb:
            fb_log.append(r)
        det, fb2, r2 = sample_truncated_normal(args.detour_mean, args.detour_std, args.detour_min, args.detour_max, rng)
        if fb2:
            fb_log.append(r2)
        L = int(round(L))
        L = int(np.clip(L, args.L_min, args.L_max))
        if det <= 0:
            debug["parity_fail_count"] = debug.get("parity_fail_count", 0) + 1
            continue
        D, d_meta = choose_adjusted_D(args, L, det)
        if D is None:
            debug["path_target_infeasible_count"] = debug.get("path_target_infeasible_count", 0) + 1
            meta["reject_reasons"].append("infeasible_L_D_detour")
            continue
        if d_meta.get("D_adjusted"):
            meta["sampling_fallback_used"] = True
            meta["sampling_fallback_reason"] = (meta.get("sampling_fallback_reason") or "") + ";adjusted_D_for_feasibility"
            meta["D_adjustment"] = d_meta

        wr, fb3, _ = sample_truncated_normal(args.wall_ratio_mean, args.wall_ratio_std, args.wall_ratio_min, args.wall_ratio_max, rng)
        de, _, _ = sample_truncated_normal(args.dead_end_ratio_mean, args.dead_end_ratio_std, args.dead_end_ratio_min, args.dead_end_ratio_max, rng)
        dd, _, _ = sample_truncated_normal(args.dead_end_depth_mean_mean, args.dead_end_depth_mean_std, args.dead_end_depth_mean_min, args.dead_end_depth_mean_max, rng)
        jr, _, _ = sample_truncated_normal(args.junction_ratio_mean, args.junction_ratio_std, args.junction_ratio_min, args.junction_ratio_max, rng)
        isl, _, _ = sample_truncated_normal(args.island_free_ratio_mean, args.island_free_ratio_std, args.island_free_ratio_min, args.island_free_ratio_max, rng)

        if fb_log:
            meta["sampling_fallback_used"] = True
            meta["sampling_fallback_reason"] = ";".join(x for x in fb_log if x)

        targets = {
            "L_BFS": float(L),
            "detour_ratio": float(det),
            "wall_ratio": float(wr),
            "dead_end_ratio": float(de),
            "dead_end_depth_mean": float(dd),
            "junction_ratio": float(jr),
            "island_free_ratio": float(max(0.0, isl)),
            "derived_D": D,
        }
        ranges = sampled_7d_to_ranges(targets, eps, n)
        debug["sampling_success_count"] = debug.get("sampling_success_count", 0) + 1
        return {"targets": targets, "ranges": ranges, "eps": eps, "sampling_metadata": meta}, None

    debug["sampling_reject_count"] = debug.get("sampling_reject_count", 0) + 1
    return None, "sampling_exhausted"


def primitive_test_target(args, idx: int) -> dict:
    name = args.primitive_test_name
    eps = get_epsilons(args)
    n = args.grid_size
    targets = {
        "L_BFS": 16.0,
        "detour_ratio": 2.0,
        "wall_ratio": 0.30,
        "dead_end_ratio": 0.10,
        "dead_end_depth_mean": 2.0,
        "junction_ratio": 0.55,
        "island_free_ratio": 0.03,
    }
    tv = None
    if name == "wall":
        tv = PRIMITIVE_TEST_TARGETS["wall"][idx % len(PRIMITIVE_TEST_TARGETS["wall"])]
        targets["wall_ratio"] = tv
    elif name == "path":
        tv = PRIMITIVE_TEST_TARGETS["path"][idx % len(PRIMITIVE_TEST_TARGETS["path"])]
        targets["L_BFS"], targets["detour_ratio"] = float(tv[0]), float(tv[1])
    elif name == "dead_end":
        tv = PRIMITIVE_TEST_TARGETS["dead_end"][idx % len(PRIMITIVE_TEST_TARGETS["dead_end"])]
        targets["dead_end_ratio"] = tv
    elif name == "dead_end_depth":
        tv = PRIMITIVE_TEST_TARGETS["dead_end_depth"][idx % len(PRIMITIVE_TEST_TARGETS["dead_end_depth"])]
        targets["dead_end_depth_mean"] = tv
    elif name == "junction":
        tv = PRIMITIVE_TEST_TARGETS["junction"][idx % len(PRIMITIVE_TEST_TARGETS["junction"])]
        targets["junction_ratio"] = tv
    elif name == "island":
        tv = PRIMITIVE_TEST_TARGETS["island"][idx % len(PRIMITIVE_TEST_TARGETS["island"])]
        targets["island_free_ratio"] = tv
    else:
        raise ValueError(f"unknown primitive_test_name: {name}")

    raw_D = max(1, int(round(targets["L_BFS"] / max(targets["detour_ratio"], 1e-9))))
    D, d_meta = choose_adjusted_D(args, int(round(targets["L_BFS"])), targets["detour_ratio"])
    if D is None:
        # Keep a placeholder D so ranges can still be serialized. generate_one
        # will return a failure record instead of raising.
        D = raw_D
    targets["derived_D"] = D
    targets["raw_D"] = raw_D
    targets["adjusted_D"] = d_meta.get("adjusted_D")
    ranges = sampled_7d_to_ranges(targets, eps, n)
    return {
        "targets": targets,
        "ranges": ranges,
        "eps": eps,
        "target_value": tv,
        "sampling_metadata": {
            "primitive_test_name": name,
            "target_value": tv,
            "primitive_role": PRIMITIVE_ROLE.get(name, "primary_structure"),
            "D_adjustment": d_meta,
            "infeasible_path_target": bool(d_meta.get("infeasible_path_target")),
        },
    }


# =============================================================================
# main path
# =============================================================================


def goal_candidates_for_manhattan(start, D, n):
    return [(r, c) for r in range(n) for c in range(n) if (r, c) != start and manhattan(start, (r, c)) == D]


def path_reachable_in_steps(pos, goal, steps_left):
    d = manhattan(pos, goal)
    return steps_left >= d and (steps_left - d) % 2 == 0


def generate_self_avoiding_main_path(start, goal, target_len, turn_ratio_range, grid_size, rng, max_attempts=500):
    n = grid_size
    D = manhattan(start, goal)
    if target_len < D or (target_len - D) % 2 != 0 or target_len > n * n - 1:
        return None, "main_path_failed_len"
    lo_turn, hi_turn = turn_ratio_range
    last = "main_path_failed_no_candidate"
    for _ in range(max_attempts):
        path, visited = [start], {start}

        def dfs(pos, steps_left):
            if steps_left == 0:
                return pos == goal
            if not path_reachable_in_steps(pos, goal, steps_left):
                return False
            nbs = list(neighbors(pos, n))
            rng.shuffle(nbs)
            nbs.sort(key=lambda q: (-manhattan(q, goal), rng.random()))
            for q in nbs:
                if q in visited or steps_left - 1 < manhattan(q, goal):
                    continue
                visited.add(q)
                path.append(q)
                if dfs(q, steps_left - 1):
                    return True
                path.pop()
                visited.remove(q)
            return False

        if not dfs(start, target_len):
            last = "main_path_failed_no_candidate"
            continue
        tr = count_turns(path) / max(1, target_len - 1)
        if lo_turn <= tr <= hi_turn:
            return path, None
        last = "main_path_failed_turn_ratio"
    return None, last


# =============================================================================
# primitives (corrected)
# =============================================================================


def metrics_snap(q: dict) -> dict:
    s = dict(q["core_metrics"])
    aux = q["aux_metrics"]
    for k in ("island_free_ratio", "bfs_len", "degree_3_ratio", "degree_4_ratio", "solution_turn_ratio"):
        s[k] = aux.get(k)
    return s


def delta_snap(before: dict, after: dict) -> dict:
    keys = list(D7_KEYS) + ["degree_3_ratio", "degree_4_ratio", "bfs_len", "sampled_target_error"]
    out = {}
    for k in keys:
        bv, av = before.get(k), after.get(k)
        out[k] = float(av) - float(bv) if is_valid_number(bv) and is_valid_number(av) else None
    return out


def _op_result(op_type, success, reject_reason, before_q, after_q, before_re, before_se, after_re, after_se, accepted, extra=None):
    bm, am = metrics_snap(before_q), metrics_snap(after_q) if after_q else None
    r = {
        "operation_type": op_type,
        "success": success,
        "reject_reason": reject_reason,
        "before_metrics": bm,
        "after_metrics": am,
        "delta_metrics": delta_snap(bm, am) if am else None,
        "before_error": {"range_error": before_re, "sampled_target_error": before_se},
        "after_error": {"range_error": after_re, "sampled_target_error": after_se} if after_re else None,
        "accepted": accepted,
    }
    if extra:
        r.update(extra)
    return r


def carve_dead_end_branch(maze, anchor, branch_len, start, rng):
    """Carve branch from anchor; reject if connects to existing free region besides anchor."""
    n = maze.shape[0]
    pre_reach = reachable_free_cells(maze, start)
    pre_free = set(free_cells_all(maze))
    tip = anchor
    carved = []
    for _ in range(branch_len):
        cands = []
        for q in neighbors(tip, n):
            if maze[q] == 1:
                cands.append(q)
        if not cands:
            break
        rng.shuffle(cands)
        nxt = cands[0]
        if nxt in pre_free and nxt != anchor:
            return False, carved, "branch_reconnected_existing_region"
        maze[nxt] = 0
        carved.append(nxt)
        tip = nxt
    if len(carved) < 1:
        return False, carved, "branch_too_short"
    post_reach = reachable_free_cells(maze, start)
    adj = build_adjacency(maze, post_reach)
    tip_deg = len(adj.get(carved[-1], []))
    if tip_deg != 1:
        return False, carved, "branch_no_dead_end_created"
    return True, carved, None


def op_dead_end_inject(ctx, rng):
    maze, start, goal = ctx["maze"], ctx["start"], ctx["goal"]
    bq, br, bs = ctx["q"], ctx["range_err"], ctx["sampled_err"]
    path = bfs_shortest_path(maze, start, goal)
    if not path:
        return _op_result("dead_end_inject", False, "no_anchor", bq, None, br, bs, None, None, False)
    anchors = [p for p in path[1:-1] if True] + [p for p in ctx.get("main_path", [])[1:-1]]
    rng.shuffle(anchors)
    depth_tgt = ctx["targets"]["dead_end_depth_mean"]
    blen = max(1, int(round(depth_tgt)))
    for anchor in anchors[:10]:
        backup = maze.copy()
        ok, carved, reason = carve_dead_end_branch(maze, anchor, blen, start, rng)
        if not ok:
            maze[:] = backup
            continue
        aq = quantize_maze(maze, start, goal, ctx.get("main_path_len"))
        if not aq["validity"]["is_solvable"]:
            maze[:] = backup
            continue
        ar = compute_range_error(aq["core_metrics"], aq["aux_metrics"], ctx["ranges"])
        ase = compute_sampled_target_error(aq, ctx["targets"], ctx["eps"])
        return _op_result("dead_end_inject", True, reason, bq, aq, br, bs, ar, ase, False)
    return _op_result("dead_end_inject", False, "no_candidate", bq, None, br, bs, None, None, False)


def op_branch_inject(ctx, rng):
    maze, start, goal = ctx["maze"], ctx["start"], ctx["goal"]
    bq, br, bs = ctx["q"], ctx["range_err"], ctx["sampled_err"]
    path = bfs_shortest_path(maze, start, goal) or ctx.get("main_path", [])
    anchors = list(set(path[1:-1] if len(path) > 2 else []))
    rng.shuffle(anchors)
    for anchor in anchors[:8]:
        backup = maze.copy()
        ok, _, reason = carve_dead_end_branch(maze, anchor, rng.randint(2, 4), start, rng)
        if not ok and reason != "branch_no_dead_end_created":
            maze[:] = backup
            continue
        aq = quantize_maze(maze, start, goal, ctx.get("main_path_len"))
        if not aq["validity"]["is_solvable"]:
            maze[:] = backup
            continue
        ar = compute_range_error(aq["core_metrics"], aq["aux_metrics"], ctx["ranges"])
        ase = compute_sampled_target_error(aq, ctx["targets"], ctx["eps"])
        return _op_result("branch_inject", True, None, bq, aq, br, bs, ar, ase, False)
    return _op_result("branch_inject", False, "no_candidate", bq, None, br, bs, None, None, False)


def op_island_inject(ctx, rng):
    maze, start, goal = ctx["maze"], ctx["start"], ctx["goal"]
    bq, br, bs = ctx["q"], ctx["range_err"], ctx["sampled_err"]
    tgt_isl = ctx["targets"]["island_free_ratio"]
    if tgt_isl <= 0 and ctx["ranges"]["island_free_ratio"][1] <= 0:
        return _op_result("island_inject", False, "island_disabled", bq, None, br, bs, None, None, False)
    n = maze.shape[0]
    reachable = reachable_free_cells(maze, start)
    free_n = len(free_cells_all(maze))
    desired = max(1, int(round(tgt_isl * max(free_n, 8))))
    wall_cands = [w for w in wall_cells_all(maze) if w not in reachable]
    rng.shuffle(wall_cands)
    for seed in wall_cands[:40]:
        backup = maze.copy()
        island = [seed]
        maze[seed] = 0
        for _ in range(desired - 1):
            frontier = []
            for c in island:
                for q in neighbors(c, n):
                    if maze[q] == 1 and q not in island and q not in reachable:
                        frontier.append(q)
            if not frontier:
                break
            nxt = rng.choice(frontier)
            maze[nxt] = 0
            island.append(nxt)
        new_reach = reachable_free_cells(maze, start)
        if any(c in new_reach for c in island if c != seed):
            maze[:] = backup
            continue
        aq = quantize_maze(maze, start, goal, ctx.get("main_path_len"))
        isl = aq["aux_metrics"].get("island_free_ratio")
        if not is_valid_number(isl) or isl > ctx["ranges"]["island_free_ratio"][1]:
            maze[:] = backup
            continue
        ar = compute_range_error(aq["core_metrics"], aq["aux_metrics"], ctx["ranges"])
        ase = compute_sampled_target_error(aq, ctx["targets"], ctx["eps"])
        return _op_result("island_inject", True, None, bq, aq, br, bs, ar, ase, False)
    return _op_result("island_inject", False, "island_no_space", bq, None, br, bs, None, None, False)


def op_safe_wall_add(ctx, rng):
    maze, start, goal = ctx["maze"], ctx["start"], ctx["goal"]
    bq, br, bs = ctx["q"], ctx["range_err"], ctx["sampled_err"]
    mp = set(ctx.get("main_path", []))
    cands = [c for c in free_cells_all(maze) if c not in mp and c != start and c != goal]
    rng.shuffle(cands)
    for cell in cands[: ctx["args"].max_operation_attempts]:
        backup = maze.copy()
        maze[cell] = 1
        aq = quantize_maze(maze, start, goal, ctx.get("main_path_len"))
        if not aq["validity"]["is_solvable"]:
            maze[:] = backup
            continue
        ar = compute_range_error(aq["core_metrics"], aq["aux_metrics"], ctx["ranges"])
        ase = compute_sampled_target_error(aq, ctx["targets"], ctx["eps"])
        return _op_result("safe_wall_add", True, None, bq, aq, br, bs, ar, ase, False)
    return _op_result("safe_wall_add", False, "no_candidate", bq, None, br, bs, None, None, False)


def op_safe_wall_remove(ctx, rng):
    maze, start, goal = ctx["maze"], ctx["start"], ctx["goal"]
    bq, br, bs = ctx["q"], ctx["range_err"], ctx["sampled_err"]
    n = maze.shape[0]
    before_bfs = bq["aux_metrics"].get("bfs_len")
    before_det = bq["core_metrics"].get("detour_ratio")
    before_pres = bq["aux_metrics"].get("main_path_preservation_ratio")
    cands = [w for w in wall_cells_all(maze) if any(maze[q] == 0 for q in neighbors(w, n))]
    rng.shuffle(cands)
    for cell in cands[: ctx["args"].max_operation_attempts]:
        backup = maze.copy()
        maze[cell] = 0
        aq = quantize_maze(maze, start, goal, ctx.get("main_path_len"))
        if not aq["validity"]["is_solvable"]:
            maze[:] = backup
            continue
        after_bfs = aq["aux_metrics"].get("bfs_len")
        after_det = aq["core_metrics"].get("detour_ratio")
        after_pres = aq["aux_metrics"].get("main_path_preservation_ratio")
        shortcut = (
            is_valid_number(before_bfs) and is_valid_number(after_bfs) and after_bfs < before_bfs - 1
        )
        extra = {
            "delta_final_bfs_len": (after_bfs - before_bfs) if is_valid_number(after_bfs) and is_valid_number(before_bfs) else None,
            "delta_detour_ratio": (after_det - before_det) if is_valid_number(after_det) and is_valid_number(before_det) else None,
            "delta_main_path_preservation_ratio": (after_pres - before_pres) if is_valid_number(after_pres) and is_valid_number(before_pres) else None,
            "shortcut_created": shortcut,
        }
        if ctx["args"].reject_shortcut_preservation_min > 0 and is_valid_number(after_pres):
            if after_pres < ctx["args"].reject_shortcut_preservation_min:
                maze[:] = backup
                continue
        ar = compute_range_error(aq["core_metrics"], aq["aux_metrics"], ctx["ranges"])
        ase = compute_sampled_target_error(aq, ctx["targets"], ctx["eps"])
        return _op_result("safe_wall_remove", True, None, bq, aq, br, bs, ar, ase, False, extra)
    return _op_result("safe_wall_remove", False, "no_candidate", bq, None, br, bs, None, None, False)


def op_path_block_optional(ctx, rng):
    maze, start, goal = ctx["maze"], ctx["start"], ctx["goal"]
    bq, br, bs = ctx["q"], ctx["range_err"], ctx["sampled_err"]
    path = bfs_shortest_path(maze, start, goal)
    if not path or len(path) < 4:
        return _op_result("path_block_optional", False, "no_path", bq, None, br, bs, None, None, False)
    interior = [p for p in path[1:-1]]
    rng.shuffle(interior)
    for cell in interior[: ctx["args"].max_operation_attempts]:
        backup = maze.copy()
        maze[cell] = 1
        aq = quantize_maze(maze, start, goal, ctx.get("main_path_len"))
        if not aq["validity"]["is_solvable"]:
            maze[:] = backup
            continue
        ar = compute_range_error(aq["core_metrics"], aq["aux_metrics"], ctx["ranges"])
        ase = compute_sampled_target_error(aq, ctx["targets"], ctx["eps"])
        return _op_result("path_block_optional", True, None, bq, aq, br, bs, ar, ase, False)
    return _op_result("path_block_optional", False, "no_candidate", bq, None, br, bs, None, None, False)


OPERATIONS = {
    "dead_end_inject": op_dead_end_inject,
    "branch_inject": op_branch_inject,
    "island_inject": op_island_inject,
    "safe_wall_add": op_safe_wall_add,
    "safe_wall_remove": op_safe_wall_remove,
    "path_block_optional": op_path_block_optional,
}

REPAIR_OPS = ("safe_wall_add", "safe_wall_remove", "path_block_optional")


def select_operation_normal(ctx) -> Tuple[str, str, str]:
    q, re, se = ctx["q"], ctx["range_err"], ctx["sampled_err"]
    per = re["per_metric_error"]
    violated = max(list(CORE_METRICS) + ["island_free_ratio"], key=lambda k: per.get(k, 0))
    cm, aux, tr = q["core_metrics"], q["aux_metrics"], ctx["ranges"]

    if per.get(violated, 0) <= 0:
        sn = se["per_metric_normalized_error"]
        violated = max(D7_KEYS, key=lambda k: sn.get(k, 0))

    if violated in ("wall_ratio",) or violated == "wall_ratio":
        w = cm.get("wall_ratio")
        if is_valid_number(w) and w < tr["wall_ratio"][0]:
            return "safe_wall_add", "wall below range", violated
        return "safe_wall_remove", "wall above range", violated
    if violated in ("bfs_len_norm", "detour_ratio", "L_BFS"):
        return "path_block_optional", f"{violated} low", violated
    if violated in ("dead_end_ratio", "dead_end_depth_mean"):
        return "dead_end_inject", f"{violated} low", violated
    if violated in ("junction_ratio", "junction_on_solution_ratio"):
        return "branch_inject", f"{violated} low", violated
    if violated == "island_free_ratio":
        isl = aux.get("island_free_ratio")
        if is_valid_number(isl) and isl < tr["island_free_ratio"][0]:
            return "island_inject", "island low", violated
        return "safe_wall_remove", "island high", violated
    if violated == "reachable_free_ratio":
        return "safe_wall_remove", "reachable low", violated
    return "safe_wall_add", "fallback", violated


def select_operation_primitive(ctx) -> Tuple[str, str, str]:
    name = ctx["args"].primitive_test_name
    step = ctx.get("edit_step", 0)
    primary = {
        "wall": ["safe_wall_add", "safe_wall_remove"],
        "path": ["path_block_optional", "safe_wall_add", "safe_wall_remove"],
        "dead_end": ["dead_end_inject"],
        "dead_end_depth": ["dead_end_inject"],
        "junction": ["branch_inject", "safe_wall_remove"],
        "island": ["island_inject"],
    }
    ops = primary.get(name, ["safe_wall_add"])
    if step % 4 == 3:
        op = REPAIR_OPS[step % len(REPAIR_OPS)]
        return op, f"repair step for {name}", "repair"
    op = ops[step % len(ops)]
    return op, f"primitive_test primary {name}", name


def try_accept(op_result, maze, backup, args, rng) -> bool:
    if not op_result["success"] or op_result["after_error"] is None:
        maze[:] = backup
        op_result["accepted"] = False
        if not op_result.get("reject_reason"):
            op_result["reject_reason"] = "invalid_after"
        return False
    br = op_result["before_error"]["range_error"]
    ar = op_result["after_error"]["range_error"]
    bs = op_result["before_error"]["sampled_target_error"]
    ase = op_result["after_error"]["sampled_target_error"]
    if ar["all_in_range"]:
        op_result["accepted"] = True
        return True
    if ar["total_error"] < br["total_error"]:
        op_result["accepted"] = True
        return True
    if ase["total_error"] < bs["total_error"]:
        op_result["accepted"] = True
        return True
    if args.allow_worse_prob > 0 and rng.random() < args.allow_worse_prob:
        op_result["accepted"] = True
        return True
    maze[:] = backup
    op_result["accepted"] = False
    op_result["reject_reason"] = op_result.get("reject_reason") or "metric_not_improved"
    return False


# =============================================================================
# generator
# =============================================================================


def generate_one(args, rng, sample_bundle: dict, gen_debug: dict) -> dict:
    targets = sample_bundle["targets"]
    ranges = sample_bundle["ranges"]
    eps = sample_bundle["eps"]
    n = args.grid_size
    D = int(targets["derived_D"])
    L = int(round(targets["L_BFS"]))
    turn_rng = (args.target_main_path_turn_ratio_min, args.target_main_path_turn_ratio_max)

    metadata = sample_bundle.get("sampling_metadata", {})
    if metadata.get("infeasible_path_target"):
        gen_debug["infeasible_path_target_count"] = gen_debug.get("infeasible_path_target_count", 0) + 1
        return _pack_failure(targets, ranges, eps, "infeasible_path_target", sample_bundle)

    best = None
    best_score = float("inf")
    sampling_debug = gen_debug

    for restart in range(args.max_restarts):
        sr, sc = rng.randint(0, n - 1), rng.randint(0, n - 1)
        start = (sr, sc)
        goals = goal_candidates_for_manhattan(start, D, n)
        if not goals:
            for dd in (D - 1, D + 1, D - 2, D + 2):
                if dd >= 1:
                    goals = goal_candidates_for_manhattan(start, dd, n)
                    if goals:
                        D = dd
                        break
        if not goals:
            sampling_debug["no_goal_candidate_count"] = sampling_debug.get("no_goal_candidate_count", 0) + 1
            continue
        goal = rng.choice(goals)

        main_path, fail = generate_self_avoiding_main_path(
            start, goal, L, turn_rng, n, rng, args.max_main_path_attempts
        )
        if main_path is None:
            sampling_debug["main_path_generation_fail_count"] = sampling_debug.get(
                "main_path_generation_fail_count", 0
            ) + 1
            continue

        maze = np.ones((n, n), dtype=np.int8)
        for c in main_path:
            maze[c] = 0
        mpl = len(main_path) - 1
        q = quantize_maze(maze, start, goal, mpl)
        re = compute_range_error(q["core_metrics"], q["aux_metrics"], ranges)
        se = compute_sampled_target_error(q, targets, eps)
        op_log = []
        op_stats = {op: {"selected_count": 0, "attempt_count": 0, "success_count": 0, "accept_count": 0, "reject_count": 0, "reject_reason_counts": defaultdict(int)} for op in ALL_OPS}

        if re["all_in_range"] and se["total_error"] < 0.01:
            return _pack(maze, start, goal, main_path, mpl, targets, ranges, eps, True, False, re, se, restart, 0, op_log, op_stats, sample_bundle)

        for step in range(args.max_edit_steps):
            if re["all_in_range"] and se["total_error"] < 0.05:
                break
            ctx = {
                "maze": maze, "start": start, "goal": goal, "q": q,
                "range_err": re, "sampled_err": se, "ranges": ranges, "targets": targets, "eps": eps,
                "args": args, "main_path": main_path, "main_path_len": mpl, "edit_step": step,
            }
            if args.mode == "primitive_test":
                op_name, reason, viol = select_operation_primitive(ctx)
            else:
                op_name, reason, viol = select_operation_normal(ctx)
            op_stats[op_name]["selected_count"] += 1
            backup = maze.copy()
            op_stats[op_name]["attempt_count"] += 1
            op_result = OPERATIONS[op_name](ctx, rng)
            if op_result["success"]:
                op_stats[op_name]["success_count"] += 1
            accepted = try_accept(op_result, maze, backup, args, rng)
            if accepted:
                op_stats[op_name]["accept_count"] += 1
                q = quantize_maze(maze, start, goal, mpl)
                re = compute_range_error(q["core_metrics"], q["aux_metrics"], ranges)
                se = compute_sampled_target_error(q, targets, eps)
            else:
                op_stats[op_name]["reject_count"] += 1
                rr = op_result.get("reject_reason") or "rejected"
                op_stats[op_name]["reject_reason_counts"][rr] += 1
            op_log.append({
                "step": step,
                "selected_operation": op_name,
                "selection_reason": reason,
                "primitive_test_name": getattr(args, "primitive_test_name", None),
                "most_violated_metric": viol,
                "before_metrics": op_result["before_metrics"],
                "after_metrics": op_result["after_metrics"],
                "delta_metrics": op_result["delta_metrics"],
                "range_error_before": op_result["before_error"]["range_error"]["total_error"],
                "range_error_after": op_result["after_error"]["range_error"]["total_error"] if op_result["after_error"] else None,
                "sampled_target_error_before": op_result["before_error"]["sampled_target_error"]["total_error"],
                "sampled_target_error_after": op_result["after_error"]["sampled_target_error"]["total_error"] if op_result["after_error"] else None,
                "accepted": accepted,
                "reject_reason": op_result.get("reject_reason"),
                **{k: op_result[k] for k in ("delta_final_bfs_len", "delta_detour_ratio", "shortcut_created") if k in op_result},
            })

        score = re["total_error"] + se["total_error"]
        failed = not re["all_in_range"]
        cand = _pack(maze, start, goal, main_path, mpl, targets, ranges, eps, not failed, failed, re, se, restart, len(op_log), op_log, op_stats, sample_bundle)
        if score < best_score:
            best_score = score
            best = cand
        if re["all_in_range"]:
            return cand

    if best is None:
        gen_debug["generation_hard_failure_count"] = gen_debug.get("generation_hard_failure_count", 0) + 1
        return _pack_failure(targets, ranges, eps, "generation_failed_all_restarts", sample_bundle)
    best["generation_failed_target"] = True
    return best




def _empty_quantization_for_failure(targets, ranges):
    core = {k: None for k in CORE_METRICS}
    aux = {k: None for k in D7_KEYS}
    aux.update({
        "bfs_len": None,
        "island_free_ratio": None,
        "main_path_len": None,
        "final_bfs_len": None,
        "main_path_preservation_ratio": None,
    })
    return {
        "validity": {"is_solvable": False, "reachable_free_ratio": None},
        "wall_and_path": {"wall_ratio": None, "bfs_len": None, "bfs_len_norm": None, "manhattan_dist": None, "detour_ratio": None},
        "graph_structure": {"degree_histogram": {}, "dead_end_ratio": None, "junction_ratio": None, "corridor_ratio": None, "degree_3_ratio": None, "degree_4_ratio": None},
        "playability_proxy": {"dead_end_depth_mean": None, "dead_end_depth_max": None, "junction_on_solution_ratio": None, "solution_path_selection": "none"},
        "core_metrics": core,
        "aux_metrics": aux,
    }


def _failure_error_payload(ranges):
    keys = list(CORE_METRICS) + ["island_free_ratio"]
    return {
        "total_error": 100.0,
        "per_metric_error": {k: 100.0 for k in keys},
        "in_range_flags": {k: False for k in keys},
        "all_in_range": False,
        "core_in_range": False,
    }


def _failure_sampled_error(targets, eps):
    return {
        "total_error": 100.0,
        "per_metric_abs_error": {k: None for k in D7_KEYS},
        "per_metric_normalized_error": {k: 100.0 for k in D7_KEYS},
        "in_range_flags": {k: False for k in D7_KEYS},
    }


def _pack_failure(targets, ranges, eps, reason, bundle):
    q = _empty_quantization_for_failure(targets, ranges)
    re = _failure_error_payload(ranges)
    se = _failure_sampled_error(targets, eps)
    pair = {k: {"target": targets.get(k), "final": None, "abs_error": None} for k in D7_KEYS}
    return {
        "maze": np.ones((1, 1), dtype=np.int8),
        "start": (0, 0),
        "goal": (0, 0),
        "main_path": [],
        "main_path_len": 0,
        "main_path_turn_count": 0,
        "main_path_detour_ratio": None,
        "generation_success": False,
        "generation_failed_target": True,
        "hard_failure": True,
        "hard_failure_reason": reason,
        "targets": targets,
        "target_ranges": {k: list(v) for k, v in ranges.items()},
        "sample_bundle": bundle,
        "quantization": q,
        "range_error": re,
        "sampled_target_error": se,
        "target_final_pair": pair,
        "generation_metadata": {
            "restart_count": 0,
            "edit_steps": 0,
            "final_range_error": re["total_error"],
            "final_sampled_target_error": se["total_error"],
            "failed_metrics": list(re["in_range_flags"].keys()),
            "operation_stats": empty_op_report() if "empty_op_report" in globals() else {},
            "hard_failure": True,
            "hard_failure_reason": reason,
        },
        "operation_trace": [],
    }

def _pack(maze, start, goal, main_path, mpl, targets, ranges, eps, success, failed, re, se, restart, edits, op_log, op_stats, bundle):
    q = quantize_maze(maze, start, goal, mpl)
    re = compute_range_error(q["core_metrics"], q["aux_metrics"], ranges)
    se = compute_sampled_target_error(q, targets, eps)
    pair = {}
    for k in D7_KEYS:
        z = targets[k]
        m = final_value_for_d7(q, k)
        pair[k] = {"target": z, "final": m, "abs_error": abs(float(m) - float(z)) if is_valid_number(m) else None}
    return {
        "maze": maze,
        "start": start,
        "goal": goal,
        "main_path": main_path,
        "main_path_len": mpl,
        "main_path_turn_count": count_turns(main_path),
        "main_path_detour_ratio": mpl / max(1, manhattan(start, goal)),
        "generation_success": success and re["all_in_range"],
        "generation_failed_target": failed or not re["all_in_range"],
        "hard_failure": False,
        "hard_failure_reason": None,
        "targets": targets,
        "target_ranges": {k: list(v) for k, v in ranges.items()},
        "sample_bundle": bundle,
        "quantization": q,
        "range_error": re,
        "sampled_target_error": se,
        "target_final_pair": pair,
        "generation_metadata": {
            "restart_count": restart,
            "edit_steps": edits,
            "final_range_error": re["total_error"],
            "final_sampled_target_error": se["total_error"],
            "failed_metrics": [k for k, v in re["in_range_flags"].items() if not v],
            "operation_stats": {k: {**{kk: vv for kk, vv in v.items() if kk != "reject_reason_counts"}, "reject_reason_counts": dict(v["reject_reason_counts"])} for k, v in op_stats.items()},
        },
        "operation_trace": op_log,
    }


# =============================================================================
# reports
# =============================================================================


def empty_op_report() -> dict:
    return {op: {"selected_count": 0, "attempt_count": 0, "success_count": 0, "accept_count": 0, "reject_count": 0, "reject_reason_counts": {}} for op in ALL_OPS}


def merge_op_reports(results: List[dict]) -> List[dict]:
    agg = empty_op_report()
    deltas = {op: defaultdict(list) for op in ALL_OPS}
    for res in results:
        stats = res["generation_metadata"].get("operation_stats", {})
        for op in ALL_OPS:
            s = stats.get(op, {})
            for k in ("selected_count", "attempt_count", "success_count", "accept_count", "reject_count"):
                agg[op][k] += s.get(k, 0)
            for rr, cnt in s.get("reject_reason_counts", {}).items():
                agg[op]["reject_reason_counts"][rr] = agg[op]["reject_reason_counts"].get(rr, 0) + cnt
        for tr in res.get("operation_trace", []):
            op = tr.get("selected_operation")
            if tr.get("accepted") and op in deltas:
                dm = tr.get("delta_metrics") or {}
                for k, v in dm.items():
                    if v is not None:
                        deltas[op][k].append(v)
    rows = []
    for op in ALL_OPS:
        row = dict(agg[op])
        row["operation_type"] = op
        for k, vals in deltas[op].items():
            row[f"mean_delta_{k}"] = float(np.mean(vals)) if vals else None
        rows.append(row)
    return rows


def _sampled_target_value(s: dict, dim: str) -> Any:
    if "targets" in s:
        return s["targets"].get(dim)
    return s.get(f"target_{dim}")


def build_distribution_match(sampled: List[dict], finals: List[dict]) -> List[dict]:
    rows = []
    fin_by_id = {f.get("maze_id"): f for f in finals}
    for dim in D7_KEYS:
        tvals = [_sampled_target_value(s, dim) for s in sampled]
        fvals = []
        for s in sampled:
            fin = fin_by_id.get(s.get("maze_id"))
            if fin is not None:
                v = fin.get(dim)
                if v is None and "core_metrics" in fin:
                    v = final_value_for_d7(fin, dim)
                if v is not None:
                    fvals.append(v)
        tvals = [float(x) for x in tvals if is_valid_number(x)]
        fvals = [float(x) for x in fvals if is_valid_number(x)]
        ts, fs = metric_stats(tvals), metric_stats(fvals)
        hist_dist = None
        if len(tvals) >= 5 and len(fvals) >= 5:
            bins = np.linspace(min(min(tvals), min(fvals)), max(max(tvals), max(fvals)), 8)
            th, _ = np.histogram(tvals, bins=bins)
            fh, _ = np.histogram(fvals, bins=bins)
            thn = th / max(1, th.sum())
            fhn = fh / max(1, fh.sum())
            hist_dist = float(np.sum(np.abs(thn - fhn)))
        rows.append({
            "dimension": dim,
            "target_mean": ts["mean"],
            "final_mean": fs["mean"],
            "mean_error": (fs["mean"] - ts["mean"]) if ts["mean"] is not None and fs["mean"] is not None else None,
            "target_std": ts["std"],
            "final_std": fs["std"],
            "std_error": (fs["std"] - ts["std"]) if ts["std"] is not None and fs["std"] is not None else None,
            "target_p25": ts["p25"], "final_p25": fs["p25"],
            "target_p50": ts["p50"], "final_p50": fs["p50"],
            "target_p75": ts["p75"], "final_p75": fs["p75"],
            "histogram_distance": hist_dist,
        })
    return rows


def build_correlation(pairs: List[dict]) -> List[dict]:
    rows = []
    for dim in D7_KEYS:
        ts, fs = [], []
        for p in pairs:
            if dim in p.get("target_final", {}):
                t = p["target_final"][dim]["target"]
                f = p["target_final"][dim]["final"]
                if is_valid_number(t) and is_valid_number(f):
                    ts.append(float(t))
                    fs.append(float(f))
        corr = r2 = None
        if len(ts) >= 3:
            ta, fa = np.array(ts), np.array(fs)
            if np.std(ta) > 1e-9 and np.std(fa) > 1e-9:
                corr = float(np.corrcoef(ta, fa)[0, 1])
                ss_res = float(np.sum((fa - ta) ** 2))
                ss_tot = float(np.sum((ta - np.mean(ta)) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else None
        rows.append({"dimension": dim, "target_final_correlation": corr, "r2_like_score": r2})
    return rows


def build_primitive_summary(records, pairs) -> List[dict]:
    name = records[0]["generation_metadata"].get("primitive_test_name") if records else "unknown"
    metrics = primary_metrics_for_primitive(name)
    role = PRIMITIVE_ROLE.get(name, "primary_structure")
    by_tv = defaultdict(list)
    for rec, pair in zip(records, pairs):
        tv = rec.get("target_value")
        by_tv[tv].append((rec, pair))
    rows = []
    for tv, items in sorted(by_tv.items(), key=lambda x: str(x[0])):
        n = len(items)
        primary_hits = sum(1 for _, p in items if p.get("primary_hit"))
        all_hits = sum(1 for _, p in items if p.get("range_hit_all"))
        primary_abs = []
        side_effect_counts = defaultdict(int)
        hard_failures = 0
        for rec, p in items:
            if rec.get("hard_failure"):
                hard_failures += 1
            for k in metrics:
                ae = p.get("abs_errors", {}).get(k)
                if ae is not None:
                    primary_abs.append(ae)
            for k in p.get("side_effect_failed_metrics", []):
                side_effect_counts[k] += 1
        rows.append({
            "primitive_test_name": name,
            "primitive_role": role,
            "primary_metrics": list(metrics),
            "target_value": tv,
            "n": n,
            "success_rate": sum(1 for r, _ in items if r.get("generation_success")) / n if n else None,
            "primary_hit_rate": primary_hits / n if n else None,
            "all_7d_hit_rate": all_hits / n if n else None,
            "target_hit_rate": primary_hits / n if n else None,  # backward-compatible alias
            "mean_primary_abs_error": float(np.mean(primary_abs)) if primary_abs else None,
            "p50_primary_abs_error": float(np.quantile(primary_abs, 0.5)) if primary_abs else None,
            "hard_failure_count": hard_failures,
            "side_effect_failed_metric_counts": dict(side_effect_counts),
            "main_failure_reason": items[0][0]["generation_metadata"].get("failed_metrics", []),
        })
    return rows

def build_failure_cases(records, pairs, k=10):
    scored = []
    for rec, pair in zip(records, pairs):
        scored.append({
            "maze_id": rec["maze_id"],
            "target_value": rec.get("target_value"),
            "sampled_target_error": pair.get("sampled_target_error"),
            "range_error": pair.get("range_error"),
            "failed_metrics": rec["generation_metadata"].get("failed_metrics"),
            "generation_failed_target": rec.get("generation_failed_target"),
        })
    scored.sort(key=lambda x: x.get("sampled_target_error") or 0, reverse=True)
    return scored[:k]


# =============================================================================
# CLI / main
# =============================================================================


def build_arg_parser():
    p = argparse.ArgumentParser(
        description=f"{VERSION} primitive control + 7D test (default: full = all primitives + 7D)"
    )
    p.add_argument(
        "--mode",
        choices=["full", "primitive_test", "normal_7d"],
        default=DEFAULTS["mode"],
        help="full: all primitive tests then normal_7d; or run a single phase",
    )
    p.add_argument(
        "--primitive-test-name",
        choices=list(PRIMITIVE_TEST_TARGETS.keys()),
        default=DEFAULTS["primitive_test_name"],
        help="used when mode=primitive_test, or to run one primitive inside full",
    )
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--grid-size", type=int, default=DEFAULTS["grid_size"])
    p.add_argument(
        "--n-mazes",
        type=int,
        default=DEFAULTS["n_mazes"],
        help="maze count for normal_7d (and for primitive_test when mode=primitive_test)",
    )
    p.add_argument(
        "--n-mazes-per-primitive",
        type=int,
        default=DEFAULTS["n_mazes_per_primitive"],
        help="mazes per primitive when mode=full",
    )
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--no-tqdm", action="store_true")
    p.add_argument("--no-viz", action="store_true")
    p.add_argument("--max-restarts", type=int, default=DEFAULTS["max_restarts"])
    p.add_argument("--max-edit-steps", type=int, default=DEFAULTS["max_edit_steps"])
    p.add_argument("--max-main-path-attempts", type=int, default=DEFAULTS["max_main_path_attempts"])
    p.add_argument("--max-operation-attempts", type=int, default=DEFAULTS["max_operation_attempts"])
    p.add_argument("--allow-worse-prob", type=float, default=DEFAULTS["allow_worse_prob"])
    p.add_argument("--reject-shortcut-preservation-min", type=float, default=DEFAULTS["reject_shortcut_preservation_min"])
    p.add_argument("--target-manhattan-min", type=int, default=DEFAULTS["target_manhattan_min"])
    p.add_argument("--target-manhattan-max", type=int, default=DEFAULTS["target_manhattan_max"])
    p.add_argument("--target-main-path-turn-ratio-min", type=float, default=DEFAULTS["target_main_path_turn_ratio_min"])
    p.add_argument("--target-main-path-turn-ratio-max", type=float, default=DEFAULTS["target_main_path_turn_ratio_max"])
    p.add_argument("--L-mean", type=float, default=DEFAULTS["L_mean"])
    p.add_argument("--L-std", type=float, default=DEFAULTS["L_std"])
    p.add_argument("--L-min", type=int, default=DEFAULTS["L_min"])
    p.add_argument("--L-max", type=int, default=DEFAULTS["L_max"])
    p.add_argument("--detour-mean", type=float, default=DEFAULTS["detour_mean"])
    p.add_argument("--detour-std", type=float, default=DEFAULTS["detour_std"])
    p.add_argument("--detour-min", type=float, default=DEFAULTS["detour_min"])
    p.add_argument("--detour-max", type=float, default=DEFAULTS["detour_max"])
    p.add_argument("--wall-ratio-mean", type=float, default=DEFAULTS["wall_ratio_mean"])
    p.add_argument("--wall-ratio-std", type=float, default=DEFAULTS["wall_ratio_std"])
    p.add_argument("--wall-ratio-min", type=float, default=DEFAULTS["wall_ratio_min"])
    p.add_argument("--wall-ratio-max", type=float, default=DEFAULTS["wall_ratio_max"])
    p.add_argument("--dead-end-ratio-mean", type=float, default=DEFAULTS["dead_end_ratio_mean"])
    p.add_argument("--dead-end-ratio-std", type=float, default=DEFAULTS["dead_end_ratio_std"])
    p.add_argument("--dead-end-ratio-min", type=float, default=DEFAULTS["dead_end_ratio_min"])
    p.add_argument("--dead-end-ratio-max", type=float, default=DEFAULTS["dead_end_ratio_max"])
    p.add_argument("--dead-end-depth-mean-mean", type=float, default=DEFAULTS["dead_end_depth_mean_mean"])
    p.add_argument("--dead-end-depth-mean-std", type=float, default=DEFAULTS["dead_end_depth_mean_std"])
    p.add_argument("--dead-end-depth-mean-min", type=float, default=DEFAULTS["dead_end_depth_mean_min"])
    p.add_argument("--dead-end-depth-mean-max", type=float, default=DEFAULTS["dead_end_depth_mean_max"])
    p.add_argument("--junction-ratio-mean", type=float, default=DEFAULTS["junction_ratio_mean"])
    p.add_argument("--junction-ratio-std", type=float, default=DEFAULTS["junction_ratio_std"])
    p.add_argument("--junction-ratio-min", type=float, default=DEFAULTS["junction_ratio_min"])
    p.add_argument("--junction-ratio-max", type=float, default=DEFAULTS["junction_ratio_max"])
    p.add_argument("--island-free-ratio-mean", type=float, default=DEFAULTS["island_free_ratio_mean"])
    p.add_argument("--island-free-ratio-std", type=float, default=DEFAULTS["island_free_ratio_std"])
    p.add_argument("--island-free-ratio-min", type=float, default=DEFAULTS["island_free_ratio_min"])
    p.add_argument("--island-free-ratio-max", type=float, default=DEFAULTS["island_free_ratio_max"])
    p.add_argument("--eps-L", type=int, default=DEFAULTS["eps_L"])
    p.add_argument("--eps-detour", type=float, default=DEFAULTS["eps_detour"])
    p.add_argument("--eps-wall-ratio", type=float, default=DEFAULTS["eps_wall_ratio"])
    p.add_argument("--eps-dead-end-ratio", type=float, default=DEFAULTS["eps_dead_end_ratio"])
    p.add_argument("--eps-dead-end-depth-mean", type=float, default=DEFAULTS["eps_dead_end_depth_mean"])
    p.add_argument("--eps-junction-ratio", type=float, default=DEFAULTS["eps_junction_ratio"])
    p.add_argument("--eps-island-free-ratio", type=float, default=DEFAULTS["eps_island_free_ratio"])
    p.add_argument("--sampling-max-attempts", type=int, default=DEFAULTS["sampling_max_attempts"])
    return p


def default_run_name(args):
    if args.mode == "full":
        return f"full__prim-{args.n_mazes_per_primitive}__7d-{args.n_mazes}"
    if args.mode == "primitive_test":
        return f"primitive_{args.primitive_test_name}__n-{args.n_mazes}"
    return f"normal_7d__n-{args.n_mazes}"


def resolve_output_dir(args):
    rn = args.run_name or default_run_name(args)
    return Path(args.output_dir) if args.output_dir else V3_ROOT / "outputs" / VERSION / rn


def print_normal_report(gen_sum, dist_match, corr, op_rep, worst, out_dir):
    print("\n[3.0.1 Primitive Control + 7D Normal Test]")
    print("\n=== 1. Generation Summary ===")
    for k, v in gen_sum.items():
        print(f"  {k}: {v}")
    print("\n=== 2. 7D Distribution Match ===")
    print(f"{'dim':<24} {'t_mean':>8} {'f_mean':>8} {'err':>8} {'hist_d':>8}")
    print("-" * 60)
    for r in dist_match:
        print(f"{r['dimension']:<24} {fmt_num(r['target_mean'],8,2)} {fmt_num(r['final_mean'],8,2)} {fmt_num(r['mean_error'],8,2)} {fmt_num(r.get('histogram_distance'),8,3)}")
    print("\n=== 3. Target-Final Correlation ===")
    for r in corr:
        print(f"  {r['dimension']:<24} corr={fmt_num(r['target_final_correlation'],8,3)} r2={fmt_num(r['r2_like_score'],8,3)}")
    print("\n=== 4. Operation Report ===")
    for r in op_rep:
        if r["attempt_count"] > 0:
            print(f"  {r['operation_type']:<22} sel={r['selected_count']} att={r['attempt_count']} acc={r['accept_count']}")
    print("\n=== 5. Worst Mismatch Examples ===")
    for w in worst[:5]:
        print(f"  {w['maze_id']} ste={fmt_num(w.get('sampled_target_error'),8,3)} re={fmt_num(w.get('range_error'),8,3)} failed={w.get('failed_metrics')}")
    print(f"\n=== 6. Output Directory ===\n  {out_dir}")


def print_primitive_report(summary, pairs, op_rep, failures, out_dir):
    print("\n[3.0.1 Primitive-Specific Control Test]")
    print("\n=== 1. Primitive Test Summary ===")
    for r in summary:
        print(f"  target={r['target_value']} n={r['n']} primary_hit={fmt_num(r.get('primary_hit_rate'),6,3)} all7d={fmt_num(r.get('all_7d_hit_rate'),6,3)} primary_mae={fmt_num(r.get('mean_primary_abs_error'),8,3)} role={r.get('primitive_role')} fail={r['main_failure_reason']}")
    print("\n=== 2. Target-Final (sample) ===")
    for p in pairs[:8]:
        print(f"  {p['maze_id']} ste={fmt_num(p.get('sampled_target_error'),8,3)} range={fmt_num(p.get('range_error'),8,3)}")
    print("\n=== 3. Operation Report ===")
    for r in op_rep:
        if r["attempt_count"]:
            print(f"  {r['operation_type']:<22} att={r['attempt_count']} acc={r['accept_count']} rej={r['reject_count']}")
    print("\n=== 4. Failure Cases ===")
    for f in failures[:5]:
        print(f"  {f['maze_id']} ste={fmt_num(f.get('sampled_target_error'),8,3)} failed={f.get('failed_metrics')}")
    print(f"\n=== 5. Output Directory ===\n  {out_dir}")


def print_full_report(full_summary: dict, root_dir: Path) -> None:
    print("\n[3.0.1 Full Test: All Primitives + 7D Normal]")
    print("\n=== Primitive Phases ===")
    for phase in full_summary.get("primitive_phases", []):
        print(
            f"  {phase['primitive_test_name']:<16} n={phase['n_generated']:<3} "
            f"primary_hit={fmt_num(phase.get('primary_hit_rate'),6,3)} "
            f"all7d={fmt_num(phase.get('all_7d_hit_rate'),6,3)} "
            f"mae={fmt_num(phase.get('mean_abs_error'),8,3)} "
            f"-> {phase['output_dir']}"
        )
    n7 = full_summary.get("normal_7d_phase", {})
    print("\n=== 7D Normal Phase ===")
    for k in ("n_requested", "n_generated", "range_hit_rate", "mean_sampled_target_error", "mean_range_error"):
        print(f"  {k}: {n7.get(k)}")
    print(f"\n=== Output Root ===\n  {root_dir}")


def run_phase(
    args,
    rng: random.Random,
    phase_mode: str,
    n_mazes: int,
    out_dir: Path,
    *,
    primitive_test_name: Optional[str] = None,
    maze_id_prefix: str = "pf7d",
) -> dict:
    """Run one experiment phase; save outputs under out_dir; return phase summary."""
    ensure_dir(out_dir)
    phase_args = argparse.Namespace(**vars(args))
    phase_args.mode = phase_mode
    if primitive_test_name is not None:
        phase_args.primitive_test_name = primitive_test_name

    run_metadata = {
        "version": VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "mode": phase_mode,
        "primitive_test_name": primitive_test_name if phase_mode == "primitive_test" else None,
        "n_mazes": n_mazes,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(SCRIPT_PATH),
        "output_dir": str(out_dir),
    }
    sampling_debug: dict = defaultdict(int)
    sampling_debug["sampling_attempts"] = 0

    records, quant_results, sampled_list, final_list, pairs_list, all_results = [], [], [], [], [], []
    desc = primitive_test_name if phase_mode == "primitive_test" else phase_mode

    for i in maybe_tqdm(range(n_mazes), phase_args, desc=desc, total=n_mazes):
        if phase_mode == "normal_7d":
            sampling_debug["sampling_attempts"] += 1
            bundle, _ = sample_7d_target(phase_args, rng, sampling_debug)
            if bundle is None:
                continue
        else:
            bundle = primitive_test_target(phase_args, i)

        result = generate_one(phase_args, rng, bundle, sampling_debug)
        mid = f"{maze_id_prefix}_{i:05d}"
        q = result["quantization"]
        targets = result["targets"]

        pair_entry = {
            "maze_id": mid,
            "target": {k: targets[k] for k in D7_KEYS},
            "final": {k: final_value_for_d7(q, k) for k in D7_KEYS},
            "abs_errors": {k: result["target_final_pair"][k]["abs_error"] for k in D7_KEYS},
            "normalized_errors": result["sampled_target_error"]["per_metric_normalized_error"],
            "range_hit_flags": result["range_error"]["in_range_flags"],
            "range_hit_all": result["range_error"]["all_in_range"],
            "range_error": result["range_error"]["total_error"],
            "sampled_target_error": result["sampled_target_error"]["total_error"],
            "target_final": result["target_final_pair"],
        }
        if phase_mode == "primitive_test":
            pair_entry["primary_metrics"] = list(primary_metrics_for_primitive(phase_args.primitive_test_name))
            pair_entry["primary_hit"] = primary_hit_from_pair(pair_entry, phase_args.primitive_test_name)
            pair_entry["side_effect_failed_metrics"] = side_effect_failed_metrics(pair_entry, phase_args.primitive_test_name)
            pair_entry["primitive_role"] = PRIMITIVE_ROLE.get(phase_args.primitive_test_name, "primary_structure")
        rec = {
            "maze_id": mid,
            "maze": result["maze"].tolist(),
            "start": list(result["start"]),
            "goal": list(result["goal"]),
            "generator_name": GENERATOR_NAME,
            "generation_success": result["generation_success"],
            "generation_failed_target": result["generation_failed_target"],
            "hard_failure": result.get("hard_failure", False),
            "hard_failure_reason": result.get("hard_failure_reason"),
            "target_ranges": result["target_ranges"],
            "sampled_targets": {k: targets[k] for k in D7_KEYS + ("derived_D",)},
            "derived_D": targets["derived_D"],
            "main_path": [list(c) for c in result["main_path"]],
            "main_path_len": result["main_path_len"],
            "main_path_detour_ratio": result["main_path_detour_ratio"],
            "generation_metadata": {
                **result["generation_metadata"],
                "primitive_test_name": primitive_test_name if phase_mode == "primitive_test" else None,
            },
            "target_value": bundle.get("target_value"),
        }
        records.append(rec)
        quant_results.append({
            **q,
            "maze_id": mid,
            "range_error": result["range_error"],
            "sampled_target_error": result["sampled_target_error"],
        })
        sampled_list.append({
            "maze_id": mid,
            **{f"target_{k}": targets[k] for k in D7_KEYS},
            "derived_D": targets["derived_D"],
            "target_ranges": result["target_ranges"],
            "sampling_metadata": bundle.get("sampling_metadata", {}),
        })
        final_list.append({"maze_id": mid, **{k: final_value_for_d7(q, k) for k in D7_KEYS}, **q["aux_metrics"]})
        pairs_list.append(pair_entry)
        all_results.append(result)

    run_metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    run_metadata["n_generated"] = len(records)

    cfg = {k: getattr(phase_args, k, None) for k in vars(phase_args) if not k.startswith("_")}
    cfg["phase_mode"] = phase_mode
    cfg["phase_n_mazes"] = n_mazes
    save_json(out_dir / "resolved_config.json", {"version": VERSION, "experiment_name": EXPERIMENT_NAME, **cfg})
    save_json(out_dir / "run_metadata.json", run_metadata)
    save_json(out_dir / "maze_records.json", records)
    save_json(out_dir / "quantization_results.json", quant_results)

    op_report = merge_op_reports(all_results)
    phase_summary: dict = {"output_dir": str(out_dir), "n_generated": len(records), "phase_mode": phase_mode}

    if phase_mode == "normal_7d":
        dist = build_distribution_match(sampled_list, quant_results)
        corr = build_correlation(pairs_list)
        save_json(out_dir / "sampled_targets.json", sampled_list)
        save_json(out_dir / "final_metrics.json", final_list)
        save_json(out_dir / "target_final_pairs.json", pairs_list)
        save_json(out_dir / "distribution_match_report.json", dist)
        save_json(out_dir / "correlation_report.json", corr)
        save_json(out_dir / "normal_sampling_debug.json", dict(sampling_debug))
        save_json(out_dir / "operation_delta_report.json", op_report)
        gen_sum = {
            "n_requested": n_mazes,
            "n_generated": len(records),
            "range_hit_rate": sum(1 for p in pairs_list if p["range_hit_all"]) / max(1, len(pairs_list)),
            "mean_sampled_target_error": float(np.mean([p["sampled_target_error"] for p in pairs_list])) if pairs_list else None,
            "mean_range_error": float(np.mean([p["range_error"] for p in pairs_list])) if pairs_list else None,
            "generation_failed_count": sum(1 for r in records if r["generation_failed_target"]),
        }
        save_json(out_dir / "generation_summary.json", gen_sum)
        worst = sorted(pairs_list, key=lambda x: x["sampled_target_error"], reverse=True)
        print_normal_report(gen_sum, dist, corr, op_report, worst, out_dir)
        phase_summary.update(gen_sum)
        phase_summary["distribution_match"] = dist
        phase_summary["correlation"] = corr
    else:
        prim_sum = build_primitive_summary(records, pairs_list)
        failures = build_failure_cases(records, pairs_list)
        save_json(out_dir / "primitive_test_summary.json", prim_sum)
        save_json(out_dir / "primitive_target_final_pairs.json", pairs_list)
        save_json(out_dir / "primitive_failure_cases.json", failures)
        save_json(out_dir / "primitive_operation_report.json", op_report)
        print_primitive_report(prim_sum, pairs_list, op_report, failures, out_dir)
        phase_summary["primitive_test_name"] = primitive_test_name
        phase_summary["primary_metrics"] = list(primary_metrics_for_primitive(phase_args.primitive_test_name))
        phase_summary["primitive_role"] = PRIMITIVE_ROLE.get(phase_args.primitive_test_name, "primary_structure")
        phase_summary["primary_hit_rate"] = (
            sum(1 for p in pairs_list if p.get("primary_hit")) / max(1, len(pairs_list))
        )
        phase_summary["target_hit_rate"] = phase_summary["primary_hit_rate"]  # backward-compatible alias
        phase_summary["all_7d_hit_rate"] = (
            sum(1 for p in pairs_list if p.get("range_hit_all")) / max(1, len(pairs_list))
        )
        abs_errs = []
        for p in pairs_list:
            for k in phase_summary["primary_metrics"]:
                v = p.get("abs_errors", {}).get(k)
                if v is not None:
                    abs_errs.append(v)
        phase_summary["mean_abs_error"] = float(np.mean(abs_errs)) if abs_errs else None
        phase_summary["primitive_test_summary"] = prim_sum

    return phase_summary


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    root_dir = resolve_output_dir(args)
    ensure_dir(root_dir)

    if args.mode == "full":
        print("\n[3.0.1 Full Test] Phase 1/7+: primitive-specific control (all primitives)")
        primitive_phases = []
        for idx, pname in enumerate(PRIMITIVE_TEST_NAMES, start=1):
            print(f"\n--- Primitive {idx}/{len(PRIMITIVE_TEST_NAMES)}: {pname} ---")
            phase_dir = root_dir / "primitives" / pname
            summary = run_phase(
                args,
                rng,
                "primitive_test",
                args.n_mazes_per_primitive,
                phase_dir,
                primitive_test_name=pname,
                maze_id_prefix=f"prim_{pname}",
            )
            primitive_phases.append(summary)

        print(f"\n[3.0.1 Full Test] Phase {len(PRIMITIVE_TEST_NAMES)+1}: normal_7d distribution")
        normal_dir = root_dir / "normal_7d"
        normal_summary = run_phase(
            args,
            rng,
            "normal_7d",
            args.n_mazes,
            normal_dir,
            maze_id_prefix="n7d",
        )
        full_summary = {
            "mode": "full",
            "primitive_phases": primitive_phases,
            "normal_7d_phase": normal_summary,
            "n_mazes_per_primitive": args.n_mazes_per_primitive,
            "n_mazes_7d": args.n_mazes,
            "primitive_names": list(PRIMITIVE_TEST_NAMES),
        }
        save_json(root_dir / "full_run_summary.json", full_summary)
        save_json(
            root_dir / "run_metadata.json",
            {
                "version": VERSION,
                "experiment_name": EXPERIMENT_NAME,
                "mode": "full",
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "output_dir": str(root_dir),
                "primitive_names": list(PRIMITIVE_TEST_NAMES),
                "n_mazes_per_primitive": args.n_mazes_per_primitive,
                "n_mazes_7d": args.n_mazes,
            },
        )
        save_json(
            root_dir / "resolved_config.json",
            {"version": VERSION, "experiment_name": EXPERIMENT_NAME, **{k: getattr(args, k) for k in vars(args)}},
        )
        print_full_report(full_summary, root_dir)
        return

    # single phase: primitive_test or normal_7d
    n = args.n_mazes
    run_phase(
        args,
        rng,
        args.mode,
        n,
        root_dir,
        primitive_test_name=args.primitive_test_name if args.mode == "primitive_test" else None,
        maze_id_prefix="prim_" + args.primitive_test_name if args.mode == "primitive_test" else "n7d",
    )


if __name__ == "__main__":
    main()
