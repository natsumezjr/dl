#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3.0.0 — Maze generation distribution test.

Compare RRG (Random-Rejection Generator) vs PBG (Path-Blocking Generator)
on 8x8 maze structural metrics via MazeQuantizer.

No CNN / RL / Reward Model — generator + quantizer only.
Stable utilities live in this file until extracted to common/.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import deque
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

VERSION = "3.0.0"
EXPERIMENT_NAME = "maze_distribution_test"

ACTIONS: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
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

DEFAULTS: Dict[str, Any] = dict(
    seed=42,
    grid_size=8,
    n_mazes=1000,
    generators="rrg,pbg",
    easy_bfs_min=6,
    easy_bfs_max=10,
    medium_bfs_min=11,
    medium_bfs_max=18,
    hard_bfs_min=19,
    hard_bfs_max=48,
    difficulty="mixed",
    difficulty_easy=0.2,
    difficulty_medium=0.4,
    difficulty_hard=0.4,
    max_retries=500,
    max_pbg_steps=200,
    pbg_initial_wall_p=0.05,
    output_dir=None,
    run_name=None,
    n_sample_images=6,
    no_tqdm=False,
    no_viz=False,
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


# =============================================================================
# BFS / grid utilities (copied stable subset — no v1/v2 import)
# =============================================================================


def in_bounds(p: Tuple[int, int], n: int) -> bool:
    return 0 <= p[0] < n and 0 <= p[1] < n


def neighbors(p: Tuple[int, int], n: int):
    for dr, dc in ACTIONS:
        q = (p[0] + dr, p[1] + dc)
        if in_bounds(q, n):
            yield q


def bfs_distances_from(maze: np.ndarray, source: Tuple[int, int]) -> np.ndarray:
    n = maze.shape[0]
    dist = np.full((n, n), INF, dtype=np.int32)
    if maze[source] == 1:
        return dist
    dq = deque([source])
    dist[source] = 0
    while dq:
        p = dq.popleft()
        for q in neighbors(p, n):
            if maze[q] == 0 and dist[q] > dist[p] + 1:
                dist[q] = dist[p] + 1
                dq.append(q)
    return dist


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


def difficulty_of_len(cfg, bfs_len: int) -> Optional[str]:
    if cfg.easy_bfs_min <= bfs_len <= cfg.easy_bfs_max:
        return "easy"
    if cfg.medium_bfs_min <= bfs_len <= cfg.medium_bfs_max:
        return "medium"
    if cfg.hard_bfs_min <= bfs_len <= cfg.hard_bfs_max:
        return "hard"
    return None


def choose_target_difficulty(cfg) -> str:
    if cfg.difficulty != "mixed":
        return cfg.difficulty
    r = random.random()
    if r < cfg.difficulty_easy:
        return "easy"
    if r < cfg.difficulty_easy + cfg.difficulty_medium:
        return "medium"
    return "hard"


def bfs_range_for_difficulty(cfg, target: str) -> Tuple[int, int]:
    mapping = {
        "easy": (cfg.easy_bfs_min, cfg.easy_bfs_max),
        "medium": (cfg.medium_bfs_min, cfg.medium_bfs_max),
        "hard": (cfg.hard_bfs_min, cfg.hard_bfs_max),
    }
    return mapping[target]


# =============================================================================
# MazeQuantizer
# =============================================================================


def build_adjacency(maze: np.ndarray, cells: set) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    n = maze.shape[0]
    adj: Dict[Tuple[int, int], List[Tuple[int, int]]] = {c: [] for c in cells}
    for p in cells:
        for q in neighbors(p, n):
            if q in cells:
                adj[p].append(q)
    return adj


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


def quantize_maze(maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> dict:
    """
    Quantize maze structure into layered metrics.

    Assumptions documented in run_metadata:
    - degree stats use reachable free cells from start only;
    - one deterministic BFS shortest path for solution_path_selection;
    - reachable_free_ratio denominator is all free cells in grid.
    """
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
    else:
        reachable_free_ratio = len(reachable) / free_count

    validity = {
        "is_solvable": is_solvable,
        "reachable_free_ratio": reachable_free_ratio,
    }

    bfs_len: Optional[int] = (len(path) - 1) if path else None
    manhattan_dist = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    wall_ratio = wall_count / (n * n)

    if bfs_len is not None:
        bfs_len_norm = bfs_len / (n * n - 1)
    else:
        bfs_len_norm = float("nan")

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
    dead_end_ratio = float("nan")
    junction_ratio = float("nan")
    corridor_ratio = float("nan")

    if is_solvable and reachable:
        adj = build_adjacency(maze, reachable)
        degrees = {v: len(adj[v]) for v in reachable}
        for v in reachable:
            d = min(degrees[v], 4)
            degree_histogram[f"degree_{d}"] = degree_histogram.get(f"degree_{d}", 0) + 1
        n_reach = len(reachable)
        dead_end_ratio = sum(1 for v in reachable if degrees[v] == 1) / n_reach
        junction_ratio = sum(1 for v in reachable if degrees[v] >= 3) / n_reach
        corridor_ratio = sum(1 for v in reachable if degrees[v] == 2) / n_reach

    graph_structure = {
        "degree_histogram": degree_histogram,
        "dead_end_ratio": dead_end_ratio,
        "junction_ratio": junction_ratio,
        "corridor_ratio": corridor_ratio,
    }

    solution_path_selection = "one_bfs_path"
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
        junction_on_path = sum(1 for v in path if degrees.get(v, 0) >= 3)
        junction_on_solution_ratio = junction_on_path / len(path)

    playability_proxy = {
        "dead_end_depth_mean": dead_end_depth_mean,
        "dead_end_depth_max": dead_end_depth_max,
        "junction_on_solution_ratio": junction_on_solution_ratio,
        "solution_path_selection": solution_path_selection,
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
# Generator A: RRG
# =============================================================================


def _rrg_corridor_fallback(n: int) -> np.ndarray:
    corridor: List[Tuple[int, int]] = []
    for r in range(n):
        if r % 2 == 0:
            cols = range(n) if (r // 2) % 2 == 0 else range(n - 1, -1, -1)
            for c in cols:
                corridor.append((r, c))
            if r + 1 < n:
                corridor.append((r + 1, n - 1 if (r // 2) % 2 == 0 else 0))
    maze = np.ones((n, n), dtype=np.int8)
    for cell in corridor:
        maze[cell] = 0
    return maze


def generate_rrg(cfg, target_difficulty: str) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], dict]:
    n = cfg.grid_size
    lo, hi = bfs_range_for_difficulty(cfg, target_difficulty)
    retry_count = 0
    wall_p_last = None

    for attempt in range(cfg.max_retries):
        retry_count = attempt + 1
        wall_p = random.uniform(0.05, 0.30)
        wall_p_last = wall_p
        maze = (np.random.rand(n, n) < wall_p).astype(np.int8)
        free = free_cells_all(maze)
        if len(free) < 2:
            continue
        start, goal = random.sample(free, 2)
        path = bfs_shortest_path(maze, start, goal)
        if not path:
            continue
        bfs_len = len(path) - 1
        if lo <= bfs_len <= hi:
            actual = difficulty_of_len(cfg, bfs_len) or target_difficulty
            meta = {
                "generator_name": "rrg",
                "retry_count": retry_count,
                "wall_p": wall_p,
                "fallback_used": False,
                "fallback_type": None,
                "bfs_len": bfs_len,
            }
            return maze, start, goal, meta

    # fallback: corridor maze
    maze = _rrg_corridor_fallback(n)
    free = free_cells_all(maze)
    possible = []
    for _ in range(4000):
        start, goal = random.sample(free, 2)
        path = bfs_shortest_path(maze, start, goal)
        if not path:
            continue
        bfs_len = len(path) - 1
        if lo <= bfs_len <= hi:
            possible.append((start, goal, bfs_len))
            if len(possible) >= 64:
                break

    if possible:
        start, goal, bfs_len = random.choice(possible)
        actual = difficulty_of_len(cfg, bfs_len) or target_difficulty
        meta = {
            "generator_name": "rrg",
            "retry_count": retry_count,
            "wall_p": "corridor",
            "fallback_used": True,
            "fallback_type": "corridor",
            "bfs_len": bfs_len,
        }
        return maze, start, goal, meta

    best = None
    target_mid = (lo + hi) // 2
    for _ in range(4000):
        start, goal = random.sample(free, 2)
        path = bfs_shortest_path(maze, start, goal)
        if not path:
            continue
        bfs_len = len(path) - 1
        score = abs(bfs_len - target_mid)
        if best is None or score < best[0]:
            best = (score, start, goal, bfs_len)

    if best is None:
        raise RuntimeError("RRG fallback maze generation failed")
    _, start, goal, bfs_len = best
    meta = {
        "generator_name": "rrg",
        "retry_count": retry_count,
        "wall_p": "corridor_closest",
        "fallback_used": True,
        "fallback_type": "corridor_closest",
        "bfs_len": bfs_len,
    }
    return maze, start, goal, meta


# =============================================================================
# Generator B: PBG
# =============================================================================


def generate_pbg(cfg, target_difficulty: str) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], dict]:
    n = cfg.grid_size
    lo, hi = bfs_range_for_difficulty(cfg, target_difficulty)
    restart_count = 0
    max_restarts = cfg.max_retries

    total_insert_attempts = 0
    total_accepted = 0
    total_rejected_unreachable = 0
    total_rejected_too_long = 0

    for restart in range(max_restarts):
        restart_count = restart
        maze = (np.random.rand(n, n) < cfg.pbg_initial_wall_p).astype(np.int8)
        free = free_cells_all(maze)
        if len(free) < 2:
            continue
        start, goal = random.sample(free, 2)

        insert_attempts = 0
        accepted_wall_inserts = 0
        rejected_unreachable = 0
        rejected_too_long = 0

        for _ in range(cfg.max_pbg_steps):
            path = bfs_shortest_path(maze, start, goal)
            if not path:
                break
            bfs_len = len(path) - 1
            if lo <= bfs_len <= hi:
                meta = {
                    "generator_name": "pbg",
                    "insert_attempts": insert_attempts,
                    "accepted_wall_inserts": accepted_wall_inserts,
                    "rejected_unreachable": rejected_unreachable,
                    "rejected_too_long": rejected_too_long,
                    "restart_count": restart_count,
                    "final_bfs_len": bfs_len,
                }
                total_insert_attempts += insert_attempts
                total_accepted += accepted_wall_inserts
                total_rejected_unreachable += rejected_unreachable
                total_rejected_too_long += rejected_too_long
                return maze, start, goal, meta

            if bfs_len > hi:
                break

            interior = [p for p in path[1:-1]]
            if not interior:
                break
            random.shuffle(interior)
            progressed = False
            for cell in interior:
                if maze[cell] == 1:
                    continue
                insert_attempts += 1
                r, c = cell
                maze[r, c] = 1
                new_path = bfs_shortest_path(maze, start, goal)
                if new_path is None:
                    maze[r, c] = 0
                    rejected_unreachable += 1
                    continue
                new_len = len(new_path) - 1
                if new_len > hi:
                    maze[r, c] = 0
                    rejected_too_long += 1
                    continue
                if new_len > bfs_len:
                    accepted_wall_inserts += 1
                    progressed = True
                    break
                maze[r, c] = 0

            if not progressed:
                break

        total_insert_attempts += insert_attempts
        total_accepted += accepted_wall_inserts
        total_rejected_unreachable += rejected_unreachable
        total_rejected_too_long += rejected_too_long

    path = bfs_shortest_path(maze, start, goal)
    final_bfs_len = (len(path) - 1) if path else None
    meta = {
        "generator_name": "pbg",
        "insert_attempts": total_insert_attempts,
        "accepted_wall_inserts": total_accepted,
        "rejected_unreachable": total_rejected_unreachable,
        "rejected_too_long": total_rejected_too_long,
        "restart_count": restart_count,
        "final_bfs_len": final_bfs_len,
        "generation_failed_target": True,
    }
    return maze, start, goal, meta


# =============================================================================
# statistics / reporting
# =============================================================================


def is_valid_number(x: Any) -> bool:
    if x is None:
        return False
    try:
        v = float(x)
        return not (math.isnan(v) or math.isinf(v))
    except (TypeError, ValueError):
        return False


def metric_stats(values: Sequence[Any]) -> dict:
    nums = [float(x) for x in values if is_valid_number(x)]
    nan_count = len(values) - len(nums)
    if not nums:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "max": None,
            "nan_count": nan_count,
        }
    arr = np.array(nums, dtype=np.float64)
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


def build_generator_summary(records: List[dict], quant_results: List[dict]) -> dict:
    by_gen: Dict[str, List[dict]] = {}
    quant_by_id = {q["maze_id"]: q for q in quant_results}
    for rec in records:
        by_gen.setdefault(rec["generator_name"], []).append(rec)

    summary: Dict[str, dict] = {}
    for gen, recs in by_gen.items():
        n_generated = len(recs)
        solvable = sum(
            1 for r in recs if quant_by_id[r["maze_id"]]["validity"]["is_solvable"]
        )
        entry: dict = {
            "n_generated": n_generated,
            "n_solvable": solvable,
            "solvable_rate": solvable / n_generated if n_generated else None,
            "retry_mean": None,
            "retry_p50": None,
            "retry_max": None,
            "fallback_rate": None,
            "pbg_insert_attempts_mean": None,
            "pbg_accepted_wall_inserts_mean": None,
            "pbg_rejected_unreachable_mean": None,
            "pbg_restart_mean": None,
        }
        if gen == "rrg":
            retries = [r["generation_metadata"].get("retry_count") for r in recs]
            retry_nums = [float(x) for x in retries if x is not None]
            if retry_nums:
                entry["retry_mean"] = float(np.mean(retry_nums))
                entry["retry_p50"] = float(np.quantile(retry_nums, 0.5))
                entry["retry_max"] = int(max(retry_nums))
            fallbacks = [r["generation_metadata"].get("fallback_used") for r in recs]
            fb = sum(1 for x in fallbacks if x)
            entry["fallback_rate"] = fb / n_generated if n_generated else None
        elif gen == "pbg":
            entry["pbg_insert_attempts_mean"] = float(
                np.mean([r["generation_metadata"].get("insert_attempts", 0) for r in recs])
            )
            entry["pbg_accepted_wall_inserts_mean"] = float(
                np.mean([r["generation_metadata"].get("accepted_wall_inserts", 0) for r in recs])
            )
            entry["pbg_rejected_unreachable_mean"] = float(
                np.mean([r["generation_metadata"].get("rejected_unreachable", 0) for r in recs])
            )
            entry["pbg_restart_mean"] = float(
                np.mean([r["generation_metadata"].get("restart_count", 0) for r in recs])
            )
        summary[gen] = entry
    return summary


def build_metric_summary(quant_results: List[dict]) -> dict:
    by_gen: Dict[str, List[dict]] = {}
    for q in quant_results:
        by_gen.setdefault(q["generator_name"], []).append(q)

    out: dict = {}
    for gen, rows in by_gen.items():
        out[gen] = {}
        for metric in CORE_METRICS:
            vals = [r["core_metrics"].get(metric) for r in rows]
            out[gen][metric] = metric_stats(vals)
    return out


def build_distribution_comparison(metric_summary: dict) -> List[dict]:
    gens = list(metric_summary.keys())
    if "rrg" not in gens or "pbg" not in gens:
        return []
    rows = []
    for metric in CORE_METRICS:
        rrg = metric_summary["rrg"].get(metric, {})
        pbg = metric_summary["pbg"].get(metric, {})
        rrg_mean = rrg.get("mean")
        pbg_mean = pbg.get("mean")
        rrg_p50 = rrg.get("p50")
        pbg_p50 = pbg.get("p50")
        rows.append({
            "metric": metric,
            "rrg_mean": rrg_mean,
            "pbg_mean": pbg_mean,
            "mean_diff": (pbg_mean - rrg_mean) if is_valid_number(rrg_mean) and is_valid_number(pbg_mean) else None,
            "rrg_p50": rrg_p50,
            "pbg_p50": pbg_p50,
            "p50_diff": (pbg_p50 - rrg_p50) if is_valid_number(rrg_p50) and is_valid_number(pbg_p50) else None,
        })
    return rows


def fmt_num(v: Any, width: int = 8, prec: int = 4) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return f"{'—':>{width}s}"
    return f"{float(v):>{width}.{prec}f}"


def print_terminal_report(
    generator_summary: dict,
    metric_summary: dict,
    distribution_comparison: List[dict],
) -> None:
    print("\n[3.0.0 Maze Distribution Test]")
    print("\n=== 1. Generator Summary ===")
    cols = ["generator", "n", "solvable", "rate", "retry_mean", "fallback_rate", "pbg_restarts"]
    header = f"{'generator':<10} {'n':>6} {'solvable':>8} {'rate':>8} {'retry_mean':>10} {'fallback':>10} {'pbg_rst':>8}"
    print(header)
    print("-" * len(header))
    for gen, s in sorted(generator_summary.items()):
        print(
            f"{gen:<10} {s['n_generated']:>6} {s['n_solvable']:>8} "
            f"{fmt_num(s['solvable_rate'], 8, 3)} "
            f"{fmt_num(s['retry_mean'], 10, 2)} "
            f"{fmt_num(s['fallback_rate'], 10, 3)} "
            f"{fmt_num(s['pbg_restart_mean'], 8, 2)}"
        )

    print("\n=== 2. Core Metric Summary by Generator ===")
    for gen in sorted(metric_summary.keys()):
        print(f"\n--- {gen.upper()} ---")
        print(f"{'metric':<32} {'mean':>10} {'std':>10} {'p50':>10} {'min':>10} {'max':>10}")
        print("-" * 84)
        for metric in CORE_METRICS:
            st = metric_summary[gen][metric]
            print(
                f"{metric:<32} {fmt_num(st['mean'], 10)} {fmt_num(st['std'], 10)} "
                f"{fmt_num(st['p50'], 10)} {fmt_num(st['min'], 10)} {fmt_num(st['max'], 10)}"
            )

    print("\n=== 3. RRG vs PBG Comparison ===")
    print(f"{'metric':<32} {'rrg_mean':>10} {'pbg_mean':>10} {'diff':>10} {'rrg_p50':>10} {'pbg_p50':>10} {'p50_diff':>10}")
    print("-" * 94)
    for row in distribution_comparison:
        print(
            f"{row['metric']:<32} {fmt_num(row['rrg_mean'], 10)} {fmt_num(row['pbg_mean'], 10)} "
            f"{fmt_num(row['mean_diff'], 10)} {fmt_num(row['rrg_p50'], 10)} "
            f"{fmt_num(row['pbg_p50'], 10)} {fmt_num(row['p50_diff'], 10)}"
        )


# =============================================================================
# visualization
# =============================================================================


def save_maze_sample(
    maze: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    path: Optional[List[Tuple[int, int]]],
    out_path: Path,
    title_lines: List[str],
) -> None:
    n = maze.shape[0]
    fig, ax = plt.subplots(figsize=(4, 4.5))
    display = np.where(maze == 1, 0.0, 1.0)
    ax.imshow(display, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    if path:
        pr, pc = zip(*path)
        ax.plot(pc, pr, color="cyan", linewidth=1.5, alpha=0.7)
    ax.plot(start[1], start[0], "go", markersize=8)
    ax.plot(goal[1], goal[0], "r*", markersize=10)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.grid(True, color="lightgray", linewidth=0.5)
    ax.set_title("\n".join(title_lines), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_sample_images(
    records: List[dict],
    quant_results: List[dict],
    output_dir: Path,
    n_per_gen: int,
) -> None:
    quant_by_id = {q["maze_id"]: q for q in quant_results}
    by_gen: Dict[str, List[dict]] = {}
    for rec in records:
        by_gen.setdefault(rec["generator_name"], []).append(rec)

    for gen, recs in by_gen.items():
        sample_dir = output_dir / f"samples_{gen}"
        ensure_dir(sample_dir)
        for rec in recs[:n_per_gen]:
            q = quant_by_id[rec["maze_id"]]
            cm = q["core_metrics"]
            maze = np.array(rec["maze"], dtype=np.int8)
            start = tuple(rec["start"])
            goal = tuple(rec["goal"])
            path = bfs_shortest_path(maze, start, goal)
            title = [
                rec["maze_id"],
                f"bfs_len={q['aux_metrics']['bfs_len']}",
                f"wall_ratio={cm['wall_ratio']:.3f}",
                f"detour={cm['detour_ratio']:.3f}" if is_valid_number(cm["detour_ratio"]) else "detour=—",
                f"dead_end={cm['dead_end_ratio']:.3f}" if is_valid_number(cm["dead_end_ratio"]) else "dead_end=—",
                f"junction={cm['junction_ratio']:.3f}" if is_valid_number(cm["junction_ratio"]) else "junction=—",
            ]
            out_path = sample_dir / f"{rec['maze_id']}.png"
            save_maze_sample(maze, start, goal, path, out_path, title)


# =============================================================================
# CLI
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"{VERSION} maze generation distribution test")
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--grid-size", type=int, default=DEFAULTS["grid_size"])
    p.add_argument("--n-mazes", type=int, default=DEFAULTS["n_mazes"])
    p.add_argument("--generators", type=str, default=DEFAULTS["generators"])
    p.add_argument("--easy-bfs-min", type=int, default=DEFAULTS["easy_bfs_min"])
    p.add_argument("--easy-bfs-max", type=int, default=DEFAULTS["easy_bfs_max"])
    p.add_argument("--medium-bfs-min", type=int, default=DEFAULTS["medium_bfs_min"])
    p.add_argument("--medium-bfs-max", type=int, default=DEFAULTS["medium_bfs_max"])
    p.add_argument("--hard-bfs-min", type=int, default=DEFAULTS["hard_bfs_min"])
    p.add_argument("--hard-bfs-max", type=int, default=DEFAULTS["hard_bfs_max"])
    p.add_argument("--difficulty", choices=["easy", "medium", "hard", "mixed"], default=DEFAULTS["difficulty"])
    p.add_argument("--difficulty-easy", type=float, default=DEFAULTS["difficulty_easy"])
    p.add_argument("--difficulty-medium", type=float, default=DEFAULTS["difficulty_medium"])
    p.add_argument("--difficulty-hard", type=float, default=DEFAULTS["difficulty_hard"])
    p.add_argument("--max-retries", type=int, default=DEFAULTS["max_retries"])
    p.add_argument("--max-pbg-steps", type=int, default=DEFAULTS["max_pbg_steps"])
    p.add_argument("--pbg-initial-wall-p", type=float, default=DEFAULTS["pbg_initial_wall_p"])
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--n-sample-images", type=int, default=DEFAULTS["n_sample_images"])
    p.add_argument("--no-tqdm", action="store_true")
    p.add_argument("--no-viz", action="store_true")
    return p


def resolved_config_from_args(args) -> dict:
    gens = [g.strip().lower() for g in args.generators.split(",") if g.strip()]
    return {
        "version": VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "seed": args.seed,
        "grid_size": args.grid_size,
        "n_mazes": args.n_mazes,
        "generators": gens,
        "easy_bfs_min": args.easy_bfs_min,
        "easy_bfs_max": args.easy_bfs_max,
        "medium_bfs_min": args.medium_bfs_min,
        "medium_bfs_max": args.medium_bfs_max,
        "hard_bfs_min": args.hard_bfs_min,
        "hard_bfs_max": args.hard_bfs_max,
        "difficulty": args.difficulty,
        "difficulty_easy": args.difficulty_easy,
        "difficulty_medium": args.difficulty_medium,
        "difficulty_hard": args.difficulty_hard,
        "max_retries": args.max_retries,
        "max_pbg_steps": args.max_pbg_steps,
        "pbg_initial_wall_p": args.pbg_initial_wall_p,
        "no_tqdm": args.no_tqdm,
        "no_viz": args.no_viz,
    }


def default_run_name(args) -> str:
    gens = args.generators.replace(",", "_")
    return f"n_mazes-{args.n_mazes}__generators-{gens}__difficulty-{args.difficulty}"


def resolve_output_dir(args) -> Path:
    run_name = args.run_name or default_run_name(args)
    if args.output_dir:
        return Path(args.output_dir)
    return V3_ROOT / "outputs" / VERSION / run_name


GENERATORS = {
    "rrg": generate_rrg,
    "pbg": generate_pbg,
}


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    gens = [g.strip().lower() for g in args.generators.split(",") if g.strip()]
    for g in gens:
        if g not in GENERATORS:
            raise SystemExit(f"Unknown generator: {g!r}. Choose from: {list(GENERATORS)}")

    output_dir = resolve_output_dir(args)
    ensure_dir(output_dir)

    resolved = resolved_config_from_args(args)
    run_metadata = {
        "version": VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(SCRIPT_PATH),
        "output_dir": str(output_dir),
        "assumptions": [
            "MazeQuantizer degree stats use reachable free cells from start.",
            "reachable_free_ratio denominator is all free cells in grid.",
            "solution_path_selection uses one deterministic BFS shortest path.",
            "RRG fallback types: corridor, corridor_closest.",
            "PBG blocks cells on current shortest path; reverts on unreachable or too-long.",
            "mixed difficulty uses difficulty_easy/medium/hard proportions.",
        ],
    }

    maze_records: List[dict] = []
    quant_results: List[dict] = []
    maze_counter = 0

    tasks = [(gen_name, i) for gen_name in gens for i in range(args.n_mazes)]
    for gen_name, _ in maybe_tqdm(tasks, args, desc="generating", total=len(tasks)):
        gen_fn = GENERATORS[gen_name]
        target = choose_target_difficulty(args)
        maze, start, goal, gen_meta = gen_fn(args, target)
        path = bfs_shortest_path(maze, start, goal)
        bfs_len = (len(path) - 1) if path else None
        actual = difficulty_of_len(args, bfs_len) if bfs_len is not None else None

        maze_id = f"{gen_name}_{maze_counter:05d}"
        maze_counter += 1

        record = {
            "maze_id": maze_id,
            "generator_name": gen_name,
            "maze": maze.tolist(),
            "start": list(start),
            "goal": list(goal),
            "difficulty_target": target,
            "difficulty_actual": actual,
            "generation_metadata": gen_meta,
        }
        maze_records.append(record)

        q = quantize_maze(maze, start, goal)
        quant_results.append({
            "maze_id": maze_id,
            "generator_name": gen_name,
            **q,
        })

    generator_summary = build_generator_summary(maze_records, quant_results)
    metric_summary = build_metric_summary(quant_results)
    distribution_comparison = build_distribution_comparison(metric_summary)

    run_metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    run_metadata["n_maze_records"] = len(maze_records)

    save_json(output_dir / "resolved_config.json", resolved)
    save_json(output_dir / "run_metadata.json", run_metadata)
    save_json(output_dir / "maze_records.json", maze_records)
    save_json(output_dir / "quantization_results.json", quant_results)
    save_json(output_dir / "generator_summary.json", generator_summary)
    save_json(output_dir / "metric_summary.json", metric_summary)
    save_json(output_dir / "distribution_comparison.json", distribution_comparison)

    if not args.no_viz:
        save_sample_images(maze_records, quant_results, output_dir, args.n_sample_images)

    print_terminal_report(generator_summary, metric_summary, distribution_comparison)
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
