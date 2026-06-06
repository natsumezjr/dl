#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3.0.2 Edge-Plan + Decision-Graph Maze Generator.

This version intentionally leaves the 3.0.1 metric-edit direction behind.  It
uses GraphPlan / EdgePlan as the primary generation target and keeps the old 7D
numbers only as cell-level validation metrics.

No CNN, DQN, Reward Model, GAN, PCGML, or reinforcement learning is used here.
This script only builds, extracts, visualizes, and verifies maze graph structure.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import networkx as nx
except Exception:
    nx = None

VERSION = "3.0.2(4)"
EXPERIMENT_NAME = "edge_plan_decision_graph_hint_first_alt_room_island"
DEFAULT_CONFIG_REL = "feature_maze/v3_0_maze_generation/configs/edge_plan_3_0_2_default(4).json"
ACTIONS: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
INF = 10**9
EDGE_TYPES = ("main_edge", "dead_edge", "alternative_edge", "island_edge")
NODE_TYPES = ("choice_node", "endpoint_node", "island_node")
D7_KEYS = ("L_BFS", "detour_ratio", "wall_ratio", "dead_end_ratio", "dead_end_depth_mean", "junction_ratio", "island_free_ratio")

# =============================================================================
# basic IO / formatting
# =============================================================================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_safe(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        x = float(v)
        return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(v, float):
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, tuple):
        return [json_safe(x) for x in v]
    if isinstance(v, list):
        return [json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): json_safe(val) for k, val in v.items()}
    if isinstance(v, Path):
        return str(v)
    return v


def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt_num(v: Any, width: int = 10, prec: int = 3) -> str:
    if v is None:
        return f"{'-':>{width}s}"
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return f"{'-':>{width}s}"
        return f"{x:>{width}.{prec}f}"
    except Exception:
        return f"{str(v):>{width}s}"


def stats(values: Sequence[Any]) -> Dict[str, Optional[float]]:
    xs: List[float] = []
    for v in values:
        try:
            x = float(v)
            if not math.isnan(x) and not math.isinf(x):
                xs.append(x)
        except Exception:
            pass
    if not xs:
        return {"mean": None, "p50": None, "p95": None, "min": None, "max": None}
    a = np.array(xs, dtype=np.float64)
    return {
        "mean": float(np.mean(a)),
        "p50": float(np.quantile(a, 0.50)),
        "p95": float(np.quantile(a, 0.95)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def print_kv(title: str, rows: Sequence[Tuple[str, Any]]) -> None:
    print(f"\n{title}")
    for k, v in rows:
        print(f"  {k:<36} {v}")


# =============================================================================
# grid and graph utilities
# =============================================================================


def in_bounds(p: Tuple[int, int], n: int) -> bool:
    return 0 <= p[0] < n and 0 <= p[1] < n


def neighbors(p: Tuple[int, int], n: int) -> Iterable[Tuple[int, int]]:
    for dr, dc in ACTIONS:
        q = (p[0] + dr, p[1] + dc)
        if in_bounds(q, n):
            yield q


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def free_cells(maze: np.ndarray) -> List[Tuple[int, int]]:
    n = maze.shape[0]
    return [(r, c) for r in range(n) for c in range(n) if int(maze[r, c]) == 0]


def build_free_adjacency(maze: np.ndarray, cells: Optional[Set[Tuple[int, int]]] = None) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    n = maze.shape[0]
    cell_set = set(free_cells(maze)) if cells is None else set(cells)
    return {p: [q for q in neighbors(p, n) if q in cell_set] for p in cell_set}


def connected_components_cells(cells: Set[Tuple[int, int]], n: int) -> List[Set[Tuple[int, int]]]:
    unseen = set(cells)
    comps: List[Set[Tuple[int, int]]] = []
    while unseen:
        seed = unseen.pop()
        comp = {seed}
        dq = deque([seed])
        while dq:
            p = dq.popleft()
            for q in neighbors(p, n):
                if q in unseen:
                    unseen.remove(q)
                    comp.add(q)
                    dq.append(q)
        comps.append(comp)
    return comps


def bfs_path(maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    if not in_bounds(start, maze.shape[0]) or not in_bounds(goal, maze.shape[0]):
        return None
    if int(maze[start]) != 0 or int(maze[goal]) != 0:
        return None
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    dq = deque([start])
    n = maze.shape[0]
    while dq:
        p = dq.popleft()
        if p == goal:
            break
        for q in sorted(neighbors(p, n)):
            if int(maze[q]) == 0 and q not in parent:
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



def count_shortest_paths_up_to(maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], limit: int = 2) -> int:
    """Count shortest start-goal paths up to a small cap for ambiguity debug."""
    if int(maze[start]) != 0 or int(maze[goal]) != 0:
        return 0
    dist_s = bfs_distances(maze, start)
    if goal not in dist_s:
        return 0
    shortest = dist_s[goal]
    # dynamic programming along increasing distance layers
    cells = sorted(dist_s, key=lambda x: dist_s[x])
    ways: Dict[Tuple[int, int], int] = {start: 1}
    n = maze.shape[0]
    for p in cells:
        if dist_s[p] >= shortest:
            continue
        for q in neighbors(p, n):
            if int(maze[q]) == 0 and dist_s.get(q) == dist_s[p] + 1:
                ways[q] = min(limit, ways.get(q, 0) + ways.get(p, 0))
    return min(limit, ways.get(goal, 0))

def bfs_distances(maze: np.ndarray, start: Tuple[int, int]) -> Dict[Tuple[int, int], int]:
    if int(maze[start]) != 0:
        return {}
    out = {start: 0}
    dq = deque([start])
    n = maze.shape[0]
    while dq:
        p = dq.popleft()
        for q in neighbors(p, n):
            if int(maze[q]) == 0 and q not in out:
                out[q] = out[p] + 1
                dq.append(q)
    return out


def is_induced_path(path: Sequence[Tuple[int, int]]) -> bool:
    pos = {p: i for i, p in enumerate(path)}
    for i, p in enumerate(path):
        for q in path:
            j = pos[q]
            if abs(i - j) > 1 and manhattan(p, q) == 1:
                return False
    return True


def path_reachable_in_steps(pos: Tuple[int, int], goal: Tuple[int, int], steps_left: int) -> bool:
    d = manhattan(pos, goal)
    return steps_left >= d and (steps_left - d) % 2 == 0


def count_turns(path: Sequence[Tuple[int, int]]) -> int:
    if len(path) < 3:
        return 0
    def direction(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
        return (b[0] - a[0], b[1] - a[1])
    return sum(1 for i in range(1, len(path) - 1) if direction(path[i - 1], path[i]) != direction(path[i], path[i + 1]))


# =============================================================================
# config / sampling
# =============================================================================


def deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def sample_uniform_int(spec: Dict[str, Any], rng: random.Random) -> int:
    return int(rng.randint(int(spec["min"]), int(spec["max"])))


def sample_truncated_int(spec: Dict[str, Any], rng: random.Random) -> int:
    lo, hi = int(spec["min"]), int(spec["max"])
    if spec.get("sampling") == "uniform_int":
        return rng.randint(lo, hi)
    mean = float(spec.get("mean", (lo + hi) / 2))
    std = float(spec.get("std", max(1.0, (hi - lo) / 4)))
    for _ in range(100):
        x = int(round(rng.gauss(mean, std)))
        if lo <= x <= hi:
            return x
    return int(np.clip(round(rng.gauss(mean, std)), lo, hi))


def weighted_choice(items: Dict[str, float], rng: random.Random) -> str:
    total = sum(max(0.0, float(v)) for v in items.values())
    if total <= 0:
        return rng.choice(list(items.keys()))
    x = rng.random() * total
    acc = 0.0
    for k, v in items.items():
        acc += max(0.0, float(v))
        if x <= acc:
            return k
    return list(items.keys())[-1]


def sample_graph_plan(cfg: Dict[str, Any], rng: random.Random, debug: Dict[str, Any]) -> Dict[str, Any]:
    gpd = cfg["graph_plan_distribution"]
    budget = sample_uniform_int(gpd["graph_budget"], rng)
    profile = weighted_choice(gpd["profile_probs"], rng)
    weights = gpd["profiles"][profile]
    raw = {
        "dead_edge_count": budget * float(weights["dead_edge_count_weight"]),
        "alternative_edge_count": budget * float(weights["alternative_edge_count_weight"]),
        "island_component_count": budget * float(weights["island_component_count_weight"]),
    }
    counts = {k: int(round(v)) for k, v in raw.items()}
    delta = budget - sum(counts.values())
    if delta != 0:
        debug["graph_budget_normalization_count"] += 1
        order = sorted(raw, key=lambda k: raw[k] - math.floor(raw[k]), reverse=(delta > 0))
        for i in range(abs(delta)):
            key = order[i % len(order)]
            counts[key] += 1 if delta > 0 else -1
            counts[key] = max(0, counts[key])
    if sum(counts.values()) != budget:
        debug["graph_budget_normalization_failed_count"] += 1
        while sum(counts.values()) < budget:
            counts["dead_edge_count"] += 1
        while sum(counts.values()) > budget and counts["island_component_count"] > 0:
            counts["island_component_count"] -= 1
    mp = cfg["main_path"]
    reject = debug["main_path_sampling_reject_reason_counts"]
    main_L = main_D = None
    for _ in range(int(mp.get("max_sampling_attempts", 200))):
        L = sample_truncated_int(mp["main_L"], rng)
        D = sample_truncated_int(mp["main_D"], rng)
        if mp.get("require_L_ge_D", True) and L < D:
            reject["main_path_L_lt_D"] += 1
            continue
        if mp.get("require_parity", True) and (L - D) % 2 != 0:
            reject["main_path_parity_invalid"] += 1
            continue
        if L > 30:
            reject["main_path_L_gt_30"] += 1
            continue
        main_L, main_D = L, D
        break
    if main_L is None:
        debug["main_path_sampling_failed_count"] += 1
        main_L = int(mp["main_L"]["max"])
        main_D = int(mp["main_D"]["min"])
    dead_edges = []
    for i in range(counts["dead_edge_count"]):
        L = sample_truncated_int(cfg["dead_edge"]["L"], rng)
        dead_edges.append({"edge_id": f"planned_dead_{i}", "edge_type": "dead_edge", "L": L})
    alternative_edges = []
    for i in range(counts["alternative_edge_count"]):
        extra = sample_truncated_int(cfg["alternative_edge"]["extra_L"], rng)
        if cfg["alternative_edge"].get("require_strictly_longer_than_main_interval", True):
            extra = max(extra, int(cfg["alternative_edge"].get("min_extra_L", 1)))
        alternative_edges.append({"edge_id": f"planned_alternative_{i}", "edge_type": "alternative_edge", "extra_L": extra})
    islands = []
    for i in range(counts["island_component_count"]):
        free_budget = rng.randint(int(cfg["island"].get("free_budget_min", 0)), int(cfg["island"].get("free_budget_max", 8)))
        islands.append({"component_id": f"planned_island_{i}", "edge_type": "island_edge", "free_budget": free_budget})
    return {
        "profile": profile,
        "graph_budget": budget,
        "main_L": main_L,
        "main_D": main_D,
        "main_detour": main_L / max(1, main_D),
        **counts,
        "planned_edges": [{"edge_id": "planned_main", "edge_type": "main_edge", "L": main_L, "D": main_D, "detour": main_L / max(1, main_D)}] + dead_edges + alternative_edges,
        "planned_islands": islands,
        "sampling_debug": {"raw_count_budget": raw, "normalized_counts": counts},
    }


# =============================================================================
# main path and edge embedding
# =============================================================================


def goal_candidates_for_D(start: Tuple[int, int], D: int, n: int) -> List[Tuple[int, int]]:
    return [(r, c) for r in range(n) for c in range(n) if (r, c) != start and manhattan(start, (r, c)) == D]


def generate_induced_main_path(n: int, L: int, D: int, cfg: Dict[str, Any], rng: random.Random, debug: Dict[str, Any]) -> Optional[List[Tuple[int, int]]]:
    max_attempts = int(cfg["main_path"].get("max_path_attempts", 1000))
    for attempt in range(max_attempts):
        debug["main_path_attempt_count"] += 1
        start = (rng.randrange(n), rng.randrange(n))
        goals = goal_candidates_for_D(start, D, n)
        if not goals:
            debug["main_path_fail_reason_counts"]["main_path_no_goal_candidate"] += 1
            continue
        goal = rng.choice(goals)
        path = [start]
        visited = {start}

        def dfs(pos: Tuple[int, int], steps_left: int) -> bool:
            if steps_left == 0:
                return pos == goal
            if not path_reachable_in_steps(pos, goal, steps_left):
                return False
            ns = list(neighbors(pos, n))
            rng.shuffle(ns)
            ns.sort(key=lambda q: (manhattan(q, goal), rng.random()))
            for q in ns:
                if q in visited:
                    continue
                if cfg["main_path"].get("require_induced_path", True):
                    ok = True
                    for old in path[:-1]:
                        if manhattan(q, old) == 1:
                            ok = False
                            break
                    if not ok:
                        continue
                if not path_reachable_in_steps(q, goal, steps_left - 1):
                    continue
                visited.add(q)
                path.append(q)
                if dfs(q, steps_left - 1):
                    return True
                path.pop()
                visited.remove(q)
            return False

        if dfs(start, L):
            if cfg["main_path"].get("require_induced_path", True) and not is_induced_path(path):
                debug["main_path_fail_reason_counts"]["main_path_not_induced"] += 1
                continue
            maze = np.ones((n, n), dtype=np.int8)
            for p in path:
                maze[p] = 0
            bp = bfs_path(maze, path[0], path[-1])
            debug["bfs_call_count"] += 1
            if not bp or len(bp) - 1 != L:
                debug["main_path_fail_reason_counts"]["main_path_bfs_shortcut"] += 1
                continue
            return path
        debug["main_path_fail_reason_counts"]["main_path_generation_failed"] += 1
    return None


def free_neighbor_count(maze: np.ndarray, p: Tuple[int, int]) -> int:
    return sum(1 for q in neighbors(p, maze.shape[0]) if int(maze[q]) == 0)


def can_place_cell(maze: np.ndarray, q: Tuple[int, int], allowed_touch: Set[Tuple[int, int]], current_path: Set[Tuple[int, int]], cfg: Dict[str, Any]) -> Tuple[bool, str]:
    n = maze.shape[0]
    if not in_bounds(q, n):
        return False, "out_of_bounds"
    if int(maze[q]) == 0 and q not in allowed_touch:
        return False, "hits_unrelated_free_cell"
    if q in current_path:
        return False, "edge_self_intersection"
    if cfg["edge_embedding"].get("forbid_touching_unrelated_free_cells", True):
        for nb in neighbors(q, n):
            if int(maze[nb]) == 0 and nb not in allowed_touch and nb not in current_path:
                return False, "touches_unrelated_free_cell"
    return True, "ok"


def carve_path_between(
    maze: np.ndarray,
    start: Tuple[int, int],
    end: Optional[Tuple[int, int]],
    target_L: int,
    cfg: Dict[str, Any],
    rng: random.Random,
    debug: Dict[str, Any],
    allowed_touch: Optional[Set[Tuple[int, int]]] = None,
) -> Tuple[Optional[List[Tuple[int, int]]], str]:
    n = maze.shape[0]
    allowed = set(allowed_touch or {start})
    if end is not None:
        allowed.add(end)
    max_steps = int(cfg["edge_embedding"].get("max_edge_path_search_steps", 2000))
    attempts = 0
    while attempts < max_steps:
        attempts += 1
        debug["edge_path_search_steps"] += 1
        path = [start]
        used = {start}
        reason = "edge_path_search_exhausted"

        def dfs(pos: Tuple[int, int], steps_left: int) -> bool:
            nonlocal reason
            if steps_left == 0:
                if end is None:
                    return True
                return pos == end
            ns = list(neighbors(pos, n))
            rng.shuffle(ns)
            if end is not None:
                ns.sort(key=lambda x: (manhattan(x, end), rng.random()))
            for q in ns:
                if end is not None and steps_left - 1 < manhattan(q, end):
                    continue
                ok, why = can_place_cell(maze, q, allowed, used, cfg)
                if not ok:
                    reason = why
                    continue
                used.add(q)
                path.append(q)
                if dfs(q, steps_left - 1):
                    return True
                path.pop()
                used.remove(q)
            return False

        if dfs(start, target_L):
            return path, "ok"
        if attempts > 10:
            break
    return None, reason


def apply_edge_path(maze: np.ndarray, path: Sequence[Tuple[int, int]]) -> None:
    for p in path:
        maze[p] = 0


def select_anchor_on_main(path: Sequence[Tuple[int, int]], cfg: Dict[str, Any], rng: random.Random, debug: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    candidates = list(path[1:-1]) if cfg["anchor_selection"].get("exclude_start_goal", True) else list(path)
    rng.shuffle(candidates)
    debug["anchor_candidate_count"] += len(candidates)
    return candidates[0] if candidates else None


def add_dead_edge(maze: np.ndarray, main_path: Sequence[Tuple[int, int]], plan_edge: Dict[str, Any], cfg: Dict[str, Any], rng: random.Random, debug: Dict[str, Any]) -> Tuple[bool, str, Optional[List[Tuple[int, int]]]]:
    for _ in range(int(cfg["edge_embedding"].get("max_edge_attempts", 200))):
        debug["edge_attempt_count"] += 1
        anchor = select_anchor_on_main(main_path, cfg, rng, debug)
        if anchor is None:
            debug["anchor_reject_reason_counts"]["no_anchor_candidate"] += 1
            return False, "dead_edge_no_anchor", None
        path, reason = carve_path_between(maze, anchor, None, int(plan_edge["L"]), cfg, rng, debug, allowed_touch={anchor})
        if path is None or len(path) < 2:
            debug["edge_reject_reason_counts"][f"dead_edge_{reason}"] += 1
            continue
        backup = maze.copy()
        apply_edge_path(maze, path)
        if free_neighbor_count(maze, path[-1]) != 1:
            maze[:, :] = backup
            debug["edge_reject_reason_counts"]["dead_edge_tip_not_endpoint"] += 1
            continue
        return True, "ok", path
    debug["edge_reject_reason_counts"]["dead_edge_no_candidate"] += 1
    return False, "dead_edge_no_candidate", None


def add_alternative_edge(maze: np.ndarray, main_path: Sequence[Tuple[int, int]], plan_edge: Dict[str, Any], cfg: Dict[str, Any], rng: random.Random, debug: Dict[str, Any]) -> Tuple[bool, str, Optional[List[Tuple[int, int]]]]:
    alt_cfg = cfg["alternative_edge"]
    min_i = int(alt_cfg.get("anchor_main_interval_min", 3))
    max_i = int(alt_cfg.get("anchor_main_interval_max", 10))
    Lmain = len(main_path) - 1
    for _ in range(int(cfg["edge_embedding"].get("max_edge_attempts", 200))):
        debug["edge_attempt_count"] += 1
        if Lmain < min_i:
            return False, "alternative_edge_main_too_short", None
        i = rng.randint(0, max(0, Lmain - min_i))
        interval = rng.randint(min_i, min(max_i, Lmain - i))
        j = i + interval
        a, b = main_path[i], main_path[j]
        target_L = interval + int(plan_edge.get("extra_L", 1))
        if alt_cfg.get("require_strictly_longer_than_main_interval", True):
            target_L = max(target_L, interval + int(alt_cfg.get("min_extra_L", 1)))
        if target_L < manhattan(a, b) or (target_L - manhattan(a, b)) % 2 != 0:
            target_L += 1
        path, reason = carve_path_between(maze, a, b, target_L, cfg, rng, debug, allowed_touch={a, b})
        if path is None:
            debug["edge_reject_reason_counts"][f"alternative_edge_{reason}"] += 1
            continue
        backup = maze.copy()
        apply_edge_path(maze, path)
        bp = bfs_path(maze, main_path[0], main_path[-1])
        debug["bfs_call_count"] += 1
        if bp and len(bp) - 1 < Lmain:
            maze[:, :] = backup
            debug["edge_reject_reason_counts"]["shortest_path_changed_goal"] += 1
            continue
        return True, "ok", path
    debug["edge_reject_reason_counts"]["alternative_edge_no_candidate"] += 1
    return False, "alternative_edge_no_candidate", None




def add_alternative_edge_coplanned(maze: np.ndarray, main_path: Sequence[Tuple[int, int]], plan_edge: Dict[str, Any], cfg: Dict[str, Any], rng: random.Random, debug: Dict[str, Any]) -> Tuple[bool, str, Optional[List[Tuple[int, int]]], Dict[str, Any]]:
    """Co-planned alternative: choose interval immediately after main path.

    Unlike post-main baseline, failure here rejects the whole main path attempt.
    The first implementation keeps the search local and records interval/route
    collision reasons rather than adding a global graph optimizer.
    """
    alt_cfg = cfg["alternative_edge"]
    min_i = int(alt_cfg.get("anchor_main_interval_min", 4))
    max_i = int(alt_cfg.get("anchor_main_interval_max", 12))
    max_attempts = int(alt_cfg.get("max_coplanned_interval_attempts", cfg["edge_embedding"].get("max_edge_attempts", 200)))
    Lmain = len(main_path) - 1
    meta: Dict[str, Any] = {"planning_mode": "co_planned_with_main", "main_interval": None}
    if Lmain < min_i:
        return False, "alternative_coplanned_main_too_short", None, meta
    # In co-planned mode we still preserve endpoint-only connection, but allow
    # adjacency to the main path during route search. This records whether the
    # extraction/classification can explain the resulting loop.
    local_cfg = json.loads(json.dumps(json_safe(cfg)))
    local_cfg.setdefault("edge_embedding", {})["forbid_touching_unrelated_free_cells"] = bool(
        alt_cfg.get("coplanned_forbid_touching_unrelated_free_cells", False)
    )
    for _ in range(max_attempts):
        debug["edge_attempt_count"] += 1
        max_start = max(0, Lmain - min_i)
        i = rng.randint(0, max_start)
        upper = min(max_i, Lmain - i)
        if upper < min_i:
            continue
        interval = rng.randint(min_i, upper)
        j = i + interval
        a, b = main_path[i], main_path[j]
        extra = int(plan_edge.get("extra_L", sample_truncated_int(alt_cfg.get("extra_L", {"min":0,"max":6}), rng)))
        if alt_cfg.get("require_strictly_longer_than_main_interval", False):
            extra = max(extra, int(alt_cfg.get("min_extra_L", 1)))
        else:
            extra = max(extra, int(alt_cfg.get("min_extra_L", 0)))
        target_L = interval + extra
        D = manhattan(a, b)
        if target_L < D:
            target_L = D
        if (target_L - D) % 2 != 0:
            target_L += 1
        interval_meta = {"a_index": i, "b_index": j, "main_interval_L": interval, "a": a, "b": b}
        meta.update({
            "main_interval": interval_meta,
            "expected_choice_anchors": [
                {"main_index": i, "cell": a},
                {"main_index": j, "cell": b},
            ],
            "target_extra_L": extra,
            "target_L": target_L,
            "alternative_equal_length": target_L == interval,
        })
        path, reason = carve_path_between(maze, a, b, target_L, local_cfg, rng, debug, allowed_touch={a, b})
        if path is None:
            debug["edge_reject_reason_counts"][f"alternative_coplanned_{reason}"] += 1
            continue
        backup = maze.copy()
        apply_edge_path(maze, path)
        bp = bfs_path(maze, main_path[0], main_path[-1])
        debug["bfs_call_count"] += 1
        if bp and len(bp) - 1 < Lmain:
            maze[:, :] = backup
            debug["edge_reject_reason_counts"]["alternative_coplanned_shortest_path_changed_goal"] += 1
            continue
        # Keep only routes that actually create some non-main cells.
        non_main = [c for c in path[1:-1] if c not in set(main_path)]
        if not non_main:
            maze[:, :] = backup
            debug["edge_reject_reason_counts"]["alternative_coplanned_no_new_cells"] += 1
            continue
        return True, "ok", path, meta
    return False, "alternative_coplanned_no_valid_route", None, meta
def add_island_component(maze: np.ndarray, free_budget: int, cfg: Dict[str, Any], rng: random.Random, debug: Dict[str, Any]) -> Tuple[bool, str, List[Tuple[int, int]]]:
    if free_budget <= 0:
        debug["edge_reject_reason_counts"]["skipped_zero_budget_island"] += 1
        return False, "skipped_zero_budget_island", []
    n = maze.shape[0]
    for _ in range(int(cfg["edge_embedding"].get("max_edge_attempts", 200))):
        debug["edge_attempt_count"] += 1
        walls = [(r, c) for r in range(n) for c in range(n) if int(maze[r, c]) == 1]
        rng.shuffle(walls)
        if not walls:
            return False, "island_no_wall_space", []
        seed = walls[0]
        comp = [seed]
        used = {seed}
        ok = True
        for _k in range(free_budget - 1):
            frontier = []
            for p in comp:
                for q in neighbors(p, n):
                    if int(maze[q]) == 1 and q not in used:
                        # Keep island disconnected from current free graph.
                        if all(int(maze[nb]) == 1 or nb in used for nb in neighbors(q, n)):
                            frontier.append(q)
            if not frontier:
                break
            q = rng.choice(frontier)
            used.add(q)
            comp.append(q)
        if len(comp) == 0:
            ok = False
        if ok:
            apply_edge_path(maze, comp)
            return True, "ok", comp
    debug["edge_reject_reason_counts"]["island_generation_failed"] += 1
    return False, "island_generation_failed", []


# =============================================================================
# decision graph extraction
# =============================================================================


def find_room_candidate_cells(maze: np.ndarray, cfg: Dict[str, Any]) -> Set[Tuple[int, int]]:
    n = maze.shape[0]
    rd = cfg["room_detection"]
    cands: Set[Tuple[int, int]] = set()
    if rd.get("use_2x2_free_block", True):
        for r in range(n - 1):
            for c in range(n - 1):
                block = [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]
                if all(int(maze[p]) == 0 for p in block):
                    cands.update(block)
    if rd.get("use_high_degree_cells", True):
        adj = build_free_adjacency(maze)
        th = int(rd.get("room_high_degree_threshold", 3))
        for p, ns in adj.items():
            if len(ns) >= th:
                cands.add(p)
    return cands


def room_exit_count(room: Set[Tuple[int, int]], maze: np.ndarray) -> Tuple[int, Dict[Tuple[int, int], int]]:
    n = maze.shape[0]
    outside_free = set(free_cells(maze)) - set(room)
    comps = connected_components_cells(outside_free, n)
    comp_id: Dict[Tuple[int, int], int] = {}
    for i, comp in enumerate(comps):
        for p in comp:
            comp_id[p] = i
    exits: Set[int] = set()
    exit_cells: Dict[Tuple[int, int], int] = {}
    for p in room:
        for q in neighbors(p, n):
            if q in comp_id:
                exits.add(comp_id[q])
                exit_cells[q] = comp_id[q]
    return len(exits), exit_cells


def component_id_by_start(maze: np.ndarray, start: Tuple[int, int]) -> Set[Tuple[int, int]]:
    if int(maze[start]) != 0:
        return set()
    adj = build_free_adjacency(maze)
    seen = {start}
    dq = deque([start])
    while dq:
        p = dq.popleft()
        for q in adj.get(p, []):
            if q not in seen:
                seen.add(q)
                dq.append(q)
    return seen


def choose_anchor(cells: Set[Tuple[int, int]], start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[int, int]:
    if start in cells:
        return start
    if goal in cells:
        return goal
    return sorted(cells)[0]


def extract_decision_graph(maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], cfg: Dict[str, Any], main_path_hint: Optional[Sequence[Tuple[int, int]]] = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    n = maze.shape[0]
    free = set(free_cells(maze))
    adj = build_free_adjacency(maze)
    reachable = component_id_by_start(maze, start) if start in free else set()
    room_candidates = find_room_candidate_cells(maze, cfg)
    raw_rooms = connected_components_cells(room_candidates, n)
    rooms = [r for r in raw_rooms if len(r) >= int(cfg["room_detection"].get("room_min_component_size", 3))]
    room_id_by_cell: Dict[Tuple[int, int], str] = {}
    room_records: Dict[str, Dict[str, Any]] = {}
    for i, comp in enumerate(rooms):
        rid = f"room_{i}"
        exits, exit_cells = room_exit_count(comp, maze)
        for cell in comp:
            room_id_by_cell[cell] = rid
        room_records[rid] = {
            "room_component_id": rid,
            "cells": sorted(comp),
            "anchor": choose_anchor(comp, start, goal),
            "room_exit_count": exits,
            "exit_cells": sorted(exit_cells.keys()),
            "pass_through_room": exits == 2,
        }
    structural_units: Dict[str, Dict[str, Any]] = {}
    cell_to_node: Dict[Tuple[int, int], str] = {}
    # Room nodes only if not pass-through; exit_count=2 rooms are folded into edges.
    for rid, rec in room_records.items():
        exits = int(rec["room_exit_count"])
        if exits == 2:
            continue
        if exits == 0:
            ntype = "island_node"
        elif exits == 1:
            ntype = "endpoint_node"
        else:
            ntype = "choice_node"
        nid = f"n_{len(structural_units)}"
        cells = [tuple(x) for x in rec["cells"]]
        anchor = tuple(rec["anchor"])
        structural_units[nid] = {
            "node_id": nid,
            "node_type": ntype,
            "anchor": anchor,
            "cells": sorted(cells),
            "is_start": start in cells,
            "is_goal": goal in cells,
            "effective_degree": exits,
            "room_component": True,
            "room_exit_count": exits,
        }
        for c in cells:
            cell_to_node[c] = nid
    # Non-room structural cells.
    for p in sorted(free):
        if p in cell_to_node:
            continue
        rid = room_id_by_cell.get(p)
        if rid is not None and room_records[rid].get("pass_through_room"):
            # Fold pass-through room cells into edges; they must not become nodes.
            continue
        deg = len(adj.get(p, []))
        is_start = p == start
        is_goal = p == goal
        if is_start or is_goal or deg != 2:
            if p not in reachable:
                ntype = "island_node"
            elif deg <= 1:
                ntype = "endpoint_node"
            else:
                ntype = "choice_node"
            nid = f"n_{len(structural_units)}"
            structural_units[nid] = {
                "node_id": nid,
                "node_type": ntype,
                "anchor": p,
                "cells": [p],
                "is_start": is_start,
                "is_goal": is_goal,
                "effective_degree": deg,
                "room_component": False,
                "room_exit_count": None,
            }
            cell_to_node[p] = nid
    # Guarantee start/goal nodes if they are free but hidden by a pass-through room.
    for special, flag in [(start, "is_start"), (goal, "is_goal")]:
        if special in free and special not in cell_to_node:
            nid = f"n_{len(structural_units)}"
            deg = len(adj.get(special, []))
            structural_units[nid] = {
                "node_id": nid,
                "node_type": "endpoint_node" if deg <= 1 else "choice_node",
                "anchor": special,
                "cells": [special],
                "is_start": special == start,
                "is_goal": special == goal,
                "effective_degree": deg,
                "room_component": False,
                "room_exit_count": None,
            }
            cell_to_node[special] = nid
    edges: List[Dict[str, Any]] = []
    visited_directed: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
    canonical_shortest_path = bfs_path(maze, start, goal)
    if main_path_hint is not None:
        start_goal_path = [tuple(x) for x in main_path_hint]
        main_path_source = "hinted"
        main_path_ambiguous = False
    else:
        start_goal_path = canonical_shortest_path
        main_path_source = "canonical_shortest"
        main_path_ambiguous = count_shortest_paths_up_to(maze, start, goal, 2) > 1 if canonical_shortest_path else False
    canonical_set = set(start_goal_path or [])
    canonical_pairs = set()
    if start_goal_path:
        for a, b in zip(start_goal_path[:-1], start_goal_path[1:]):
            canonical_pairs.add((a, b)); canonical_pairs.add((b, a))
    def node_of(cell: Tuple[int, int]) -> Optional[str]:
        return cell_to_node.get(cell)
    for nid, node in list(structural_units.items()):
        for c in node["cells"]:
            for nb in adj.get(tuple(c), []):
                if (tuple(c), nb) in visited_directed:
                    continue
                path = [tuple(c), nb]
                prev, cur = tuple(c), nb
                visited_directed.add((prev, cur))
                visited_directed.add((cur, prev))
                end_nid = node_of(cur)
                pass_rooms: Set[str] = set()
                steps = 0
                while end_nid is None and steps < n * n + 5:
                    steps += 1
                    rid = room_id_by_cell.get(cur)
                    if rid is not None and room_records.get(rid, {}).get("pass_through_room"):
                        pass_rooms.add(rid)
                    nxts = [x for x in adj.get(cur, []) if x != prev]
                    if not nxts:
                        break
                    if len(nxts) > 1:
                        # A hidden intersection not captured as a node; stop and make an implicit node.
                        implicit_id = f"n_{len(structural_units)}"
                        structural_units[implicit_id] = {
                            "node_id": implicit_id,
                            "node_type": "choice_node",
                            "anchor": cur,
                            "cells": [cur],
                            "is_start": cur == start,
                            "is_goal": cur == goal,
                            "effective_degree": len(adj.get(cur, [])),
                            "room_component": False,
                            "room_exit_count": None,
                        }
                        cell_to_node[cur] = implicit_id
                        end_nid = implicit_id
                        break
                    nxt = nxts[0]
                    visited_directed.add((cur, nxt))
                    visited_directed.add((nxt, cur))
                    prev, cur = cur, nxt
                    path.append(cur)
                    end_nid = node_of(cur)
                if end_nid is None or end_nid == nid:
                    continue
                a_anchor = tuple(structural_units[nid]["anchor"])
                b_anchor = tuple(structural_units[end_nid]["anchor"])
                path_start, path_end = tuple(path[0]), tuple(path[-1])
                L = len(path) - 1
                D_path = manhattan(path_start, path_end)
                detour_path = None if D_path == 0 else L / D_path
                D_anchor = manhattan(a_anchor, b_anchor)
                detour_anchor = None if D_anchor == 0 else L / D_anchor
                geometry_bug = (D_path == 0) or (L < D_path) or (detour_path is not None and detour_path < 1.0 - 1e-9)
                anchor_geometry_mismatch = (detour_anchor is not None and detour_anchor < 1.0 - 1e-9) or (D_anchor != D_path)
                if all((path[i], path[i + 1]) in canonical_pairs for i in range(len(path) - 1)):
                    etype = "main_edge"
                elif nid not in [node_of(x) for x in reachable] or end_nid not in [node_of(x) for x in reachable]:
                    etype = "island_edge"
                elif structural_units[nid]["node_type"] == "island_node" or structural_units[end_nid]["node_type"] == "island_node":
                    etype = "island_edge"
                elif structural_units[nid]["node_type"] == "endpoint_node" or structural_units[end_nid]["node_type"] == "endpoint_node":
                    etype = "dead_edge"
                else:
                    etype = "alternative_edge"
                edges.append({
                    "edge_id": f"e_{len(edges)}",
                    "edge_type": etype,
                    "source": nid,
                    "target": end_nid,
                    "cell_path": path,
                    "path_start": path_start,
                    "path_end": path_end,
                    "source_anchor": a_anchor,
                    "target_anchor": b_anchor,
                    "L": L,
                    "D": D_path,
                    "detour": detour_path,
                    "D_path": D_path,
                    "detour_path": detour_path,
                    "D_anchor": D_anchor,
                    "detour_anchor": detour_anchor,
                    "anchor_geometry_mismatch": bool(anchor_geometry_mismatch),
                    "extraction_geometry_bug": bool(geometry_bug),
                    "invalid_or_room_internal_edge": D_path == 0,
                    "passes_through_room": bool(pass_rooms),
                    "room_component_ids": sorted(pass_rooms),
                })
    nodes = list(structural_units.values())
    edge_counts = Counter(e["edge_type"] for e in edges)
    node_counts = Counter(nr["node_type"] for nr in nodes)
    pass_through_room_ids = {rid for rid, r in room_records.items() if r.get("pass_through_room")}
    pass_through_room_node_leak_count = 0
    for node in nodes:
        for c in node.get("cells", []):
            rid = room_id_by_cell.get(tuple(c))
            if rid in pass_through_room_ids:
                pass_through_room_node_leak_count += 1
                break
    detour_path_lt_1_count = sum(1 for e in edges if e.get("detour_path") is not None and e.get("detour_path") < 1.0 - 1e-9)
    detour_anchor_lt_1_count = sum(1 for e in edges if e.get("detour_anchor") is not None and e.get("detour_anchor") < 1.0 - 1e-9)
    geometry_bug_count = sum(1 for e in edges if e.get("extraction_geometry_bug"))
    main_route_total_L = sum(e.get("L", 0) for e in edges if e.get("edge_type") == "main_edge")
    main_route_total_D_path = sum(e.get("D_path", 0) for e in edges if e.get("edge_type") == "main_edge")
    island_components = []
    island_free = set(free) - set(reachable)
    for idx, comp in enumerate(connected_components_cells(island_free, n)):
        node_ids = [node["node_id"] for node in nodes if any(tuple(c) in comp for c in node.get("cells", []))]
        edge_ids = [e["edge_id"] for e in edges if any(tuple(c) in comp for c in e.get("cell_path", []))]
        island_components.append({
            "component_id": f"island_{idx}",
            "cells": sorted(comp),
            "cell_count": len(comp),
            "node_count": len(node_ids),
            "edge_count": len(edge_ids),
            "node_ids": node_ids,
            "edge_ids": edge_ids,
        })
    metrics = {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "choice_node_count": node_counts.get("choice_node", 0),
        "endpoint_node_count": node_counts.get("endpoint_node", 0),
        "island_node_count": node_counts.get("island_node", 0),
        "main_edge_count": edge_counts.get("main_edge", 0),
        "dead_edge_count": edge_counts.get("dead_edge", 0),
        "alternative_edge_count": edge_counts.get("alternative_edge", 0),
        "island_edge_count": edge_counts.get("island_edge", 0),
        "island_component_count": len(island_components),
        "main_route_total_L": main_route_total_L,
        "extracted_main_edge_sum_L": main_route_total_L,
        "main_route_total_D_path": main_route_total_D_path,
        "main_route_edge_count": edge_counts.get("main_edge", 0),
        "main_route_reconstruct_success": edge_counts.get("main_edge", 0) > 0,
        "geometry_bug_count": geometry_bug_count,
        "detour_path_lt_1_count": detour_path_lt_1_count,
        "detour_anchor_lt_1_count": detour_anchor_lt_1_count,
        "pass_through_room_node_leak_count": pass_through_room_node_leak_count,
        "room_component_count": len(room_records),
        "pass_through_room_count": sum(1 for r in room_records.values() if r.get("pass_through_room")),
        "old_7d_is_validation_not_primary": True,
    }
    return {
        "nodes": nodes,
        "edges": edges,
        "node_table": nodes,
        "edge_table": edges,
        "metrics": metrics,
        "rooms": list(room_records.values()),
        "canonical_path": canonical_shortest_path,
        "main_path": start_goal_path,
        "main_path_source": main_path_source,
        "main_path_ambiguous": bool(main_path_ambiguous),
        "island_components": island_components,
        "island_component_count": len(island_components),
        "debug_extraction_report": {
            "runtime_ms": (time.perf_counter() - t0) * 1000,
            "room_candidate_count": len(room_candidates),
            "room_component_count": len(room_records),
            "cell_to_node_count": len(cell_to_node),
            "pass_through_room_node_leak_count": pass_through_room_node_leak_count,
            "detour_path_lt_1_count": detour_path_lt_1_count,
            "detour_anchor_lt_1_count": detour_anchor_lt_1_count,
            "geometry_bug_count": geometry_bug_count,
            "main_path_source": main_path_source,
            "main_path_ambiguous": bool(main_path_ambiguous),
            "island_component_count": len(island_components),
            "note": "Degree-2 cells and pass-through rooms are folded into edges. Start and goal are node attributes. Core edge geometry uses D_path/detour_path. Main route may come from main_path_hint.",
        },
    }


# =============================================================================
# validation and consistency
# =============================================================================


def cell_level_validation_metrics(maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Dict[str, Any]:
    n = maze.shape[0]
    free = set(free_cells(maze))
    path = bfs_path(maze, start, goal)
    reachable = set(bfs_distances(maze, start).keys()) if start in free else set()
    adj = build_free_adjacency(maze, reachable)
    degrees = {p: len(ns) for p, ns in adj.items()}
    wall_ratio = 1.0 - len(free) / max(1, n * n)
    L = None if not path else len(path) - 1
    D = manhattan(start, goal)
    det = None if L is None or D == 0 else L / D
    dead_nodes = [p for p, d in degrees.items() if d == 1]
    junction_nodes = [p for p, d in degrees.items() if d >= 3]
    depths = []
    solution = set(path or [])
    for de in dead_nodes:
        depth, prev, cur = 0, None, de
        while True:
            ns = [q for q in adj.get(cur, []) if q != prev]
            if not ns:
                break
            nxt = ns[0]
            depth += 1
            if nxt in solution or degrees.get(nxt, 0) >= 3:
                break
            prev, cur = cur, nxt
        depths.append(depth)
    island_free_ratio = 1.0 - len(reachable) / max(1, len(free)) if free else None
    return {
        "note": "Old 7D metrics are validation metrics, not the primary generation target of 3.0.2.",
        "L_BFS": L,
        "detour_ratio": det,
        "wall_ratio": wall_ratio,
        "dead_end_ratio": len(dead_nodes) / max(1, len(reachable)),
        "dead_end_depth_mean": float(np.mean(depths)) if depths else 0.0,
        "junction_ratio": len(junction_nodes) / max(1, len(reachable)),
        "island_free_ratio": island_free_ratio,
    }


def plan_summary_from_paths(plan: Dict[str, Any], planned_paths: Sequence[Dict[str, Any]], accepted_only: bool) -> Dict[str, Any]:
    def ok(x: Dict[str, Any]) -> bool:
        return bool(x.get("accepted")) if accepted_only else bool(x.get("sampled", True))
    dead = [x for x in planned_paths if x.get("edge_type") == "dead_edge" and ok(x)]
    alt = [x for x in planned_paths if x.get("edge_type") == "alternative_edge" and ok(x)]
    isl = [x for x in planned_paths if x.get("edge_type") == "island_edge" and ok(x)]
    return {
        "main_L": int(plan.get("main_L", 0)),
        "main_D": int(plan.get("main_D", 0)),
        "dead_edge_count": len(dead),
        "alternative_edge_count": len(alt),
        "island_component_count": len(isl),
        "dead_L_list": sorted([int(x.get("actual_L", x.get("target_L", x.get("L", 0))) or 0) for x in dead]),
        "alternative_L_list": sorted([int(x.get("actual_L", 0) or 0) for x in alt]),
        "island_free_budget": sum(int(x.get("free_budget", len(x.get("cells", []))) or 0) for x in isl),
    }


def match_planned_edges_to_extracted(planned_paths: List[Dict[str, Any]], extracted: Dict[str, Any]) -> None:
    """Best-effort matching for debug only. Prefer same type and same L, then max cell overlap."""
    edges = list(extracted.get("edges", []))
    used: Set[str] = set()
    for pe in planned_paths:
        if pe.get("edge_type") in ("main_edge", "island_edge") or not pe.get("accepted"):
            pe.setdefault("matched_extracted_edge_id", None)
            pe.setdefault("matched_extracted_edge_type", None)
            continue
        p_cells = {tuple(x) for x in pe.get("actual_cell_path", pe.get("cell_path", []))}
        best = None
        best_score = (-1, -1, -999)
        for e in edges:
            if e.get("edge_id") in used:
                continue
            e_cells = {tuple(x) for x in e.get("cell_path", [])}
            same_type = 1 if e.get("edge_type") == pe.get("edge_type") else 0
            same_L = 1 if int(e.get("L", -1)) == int(pe.get("actual_L", -2)) else 0
            overlap = len(p_cells & e_cells)
            score = (same_type, same_L, overlap)
            if score > best_score:
                best = e
                best_score = score
        if best is not None and best_score[2] > 0:
            used.add(best["edge_id"])
            pe["matched_extracted_edge_id"] = best.get("edge_id")
            pe["matched_extracted_edge_type"] = best.get("edge_type")
            pe["matched_extracted_edge_type_mismatch"] = best.get("edge_type") != pe.get("edge_type")
        else:
            pe["matched_extracted_edge_id"] = None
            pe["matched_extracted_edge_type"] = None
            pe["matched_extracted_edge_type_mismatch"] = False


def compare_plan_extracted(plan: Dict[str, Any], extracted: Dict[str, Any], planned_paths: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    planned_paths = planned_paths or []
    sampled_summary = plan_summary_from_paths(plan, planned_paths, accepted_only=False)
    accepted_summary = plan_summary_from_paths(plan, planned_paths, accepted_only=True)
    m = extracted.get("metrics", {})
    main_rec = planned_paths[0] if planned_paths else {}
    planned_main_L = int(plan.get("main_L", main_rec.get("planned_main_L", 0)) or 0)
    generated_main_L = int(main_rec.get("generated_main_L", main_rec.get("actual_L", planned_main_L)) or 0)
    hinted_main_route_L = int(main_rec.get("hinted_main_route_L", generated_main_L) or 0)
    final_bfs_L = main_rec.get("final_bfs_L")
    extracted_main_edge_sum_L = int(m.get("extracted_main_edge_sum_L", m.get("main_route_total_L", 0)) or 0)
    generated_main_L_error = abs(generated_main_L - planned_main_L)
    hinted_main_route_L_error = abs(hinted_main_route_L - planned_main_L)
    final_bfs_L_error = None if final_bfs_L is None else abs(int(final_bfs_L) - planned_main_L)
    extracted_main_edge_sum_L_error = abs(extracted_main_edge_sum_L - planned_main_L)
    final_bfs_L_match = final_bfs_L == planned_main_L
    final_bfs_shorter = final_bfs_L is not None and final_bfs_L < planned_main_L
    extracted_dead_L = sorted([int(e.get("L", 0)) for e in extracted.get("edges", []) if e.get("edge_type") == "dead_edge"])
    accepted_dead_L = accepted_summary["dead_L_list"]
    dead_count_match = accepted_summary["dead_edge_count"] == int(m.get("dead_edge_count", 0) or 0)
    alt_count_match = accepted_summary["alternative_edge_count"] == int(m.get("alternative_edge_count", 0) or 0)
    island_count_match = accepted_summary["island_component_count"] == int(m.get("island_component_count", 0) or 0)
    geometry_bug_count = int(m.get("geometry_bug_count", 0) or 0)
    alternative_paths = [x for x in planned_paths if x.get("edge_type") == "alternative_edge"]
    accepted_alternatives = [x for x in alternative_paths if x.get("accepted")]
    equal_len_count = sum(1 for x in accepted_alternatives if x.get("alternative_equal_length"))
    alt_shortcut_violations = sum(1 for x in accepted_alternatives if x.get("alternative_shortcut_violation"))
    alt_extracted_match = accepted_summary["alternative_edge_count"] == int(m.get("alternative_edge_count", 0) or 0)
    island_paths = [x for x in planned_paths if x.get("edge_type") == "island_edge"]
    island_no_cells_generated_count = sum(1 for x in island_paths if x.get("reject_reason") == "island_no_cells_generated")
    checks = {
        "generated_main_L_match": generated_main_L == planned_main_L,
        "final_bfs_L_match": bool(final_bfs_L_match),
        "hinted_main_route_L_match": hinted_main_route_L == planned_main_L,
        "dead_edge_count_match_effective": dead_count_match,
        "alternative_edge_count_match_effective": alt_count_match,
        "island_component_count_match": island_count_match,
        "no_geometry_bug": geometry_bug_count == 0,
        "no_alternative_shortcut_violation": not final_bfs_shorter and alt_shortcut_violations == 0,
    }
    sampled_failures = {
        "missing_sampled_dead_edge_count": sampled_summary["dead_edge_count"] - accepted_summary["dead_edge_count"],
        "missing_sampled_alternative_edge_count": sampled_summary["alternative_edge_count"] - accepted_summary["alternative_edge_count"],
        "missing_sampled_island_count": sampled_summary["island_component_count"] - accepted_summary["island_component_count"],
    }
    edge_embedding_success = all(v == 0 for v in sampled_failures.values())
    # In 3.0.2(3), extracted_main_edge_sum_L mismatch is a representation issue, not a shortest-path failure.
    effective_plan_match = all(checks.values())
    strict_reasons: List[str] = []
    for k, ok in checks.items():
        if not ok:
            strict_reasons.append(k)
    if extracted_main_edge_sum_L != planned_main_L and final_bfs_L_match and hinted_main_route_L == planned_main_L:
        strict_reasons.append("extracted_main_edge_sum_L_representation_mismatch")
    for k, v in sampled_failures.items():
        if v > 0:
            strict_reasons.append(k.replace("missing_sampled_", "sampled_").replace("_count", "_not_embedded"))
    return {
        "hard_generation_success": True,
        "edge_embedding_success": edge_embedding_success,
        "strict_consistency_success": edge_embedding_success and effective_plan_match,
        "effective_plan_match": effective_plan_match,
        "sampled_plan_match": edge_embedding_success and effective_plan_match,
        "sampled_plan_summary": sampled_summary,
        "accepted_plan_summary": accepted_summary,
        "main_lengths": {
            "planned_main_L": planned_main_L,
            "generated_main_L": generated_main_L,
            "final_bfs_L": final_bfs_L,
            "hinted_main_route_L": hinted_main_route_L,
            "extracted_main_edge_sum_L": extracted_main_edge_sum_L,
        },
        "main_checks": {
            "generated_main_L_match": generated_main_L == planned_main_L,
            "final_bfs_L_match": bool(final_bfs_L_match),
            "hinted_main_route_L_match": hinted_main_route_L == planned_main_L,
            "extracted_main_edge_sum_L_match": extracted_main_edge_sum_L == planned_main_L,
            "main_path_source": extracted.get("main_path_source", main_rec.get("main_path_source")),
            "main_path_ambiguous": bool(extracted.get("main_path_ambiguous") or main_rec.get("main_path_ambiguous")),
        },
        "alternative_checks": {
            "sampled_alternative_count": sampled_summary["alternative_edge_count"],
            "accepted_alternative_count": accepted_summary["alternative_edge_count"],
            "extracted_alternative_count": int(m.get("alternative_edge_count", 0) or 0),
            "alternative_equal_length_count": equal_len_count,
            "alternative_shortcut_violation_count": alt_shortcut_violations + (1 if final_bfs_shorter else 0),
            "alternative_extracted_match": alt_extracted_match,
        },
        "island_checks": {
            "planned_island_component_count": sampled_summary["island_component_count"],
            "accepted_island_component_count": accepted_summary["island_component_count"],
            "extracted_island_component_count": int(m.get("island_component_count", 0) or 0),
            "island_component_count_match": island_count_match,
            "island_node_count": int(m.get("island_node_count", 0) or 0),
            "island_edge_count": int(m.get("island_edge_count", 0) or 0),
            "island_no_cells_generated_count": island_no_cells_generated_count,
        },
        "extracted_summary": {
            "extracted_main_edge_sum_L": extracted_main_edge_sum_L,
            "main_edge_count": m.get("main_edge_count"),
            "dead_edge_count": m.get("dead_edge_count"),
            "alternative_edge_count": m.get("alternative_edge_count"),
            "island_component_count": m.get("island_component_count"),
            "island_node_count": m.get("island_node_count"),
            "island_edge_count": m.get("island_edge_count"),
            "detour_path_lt_1_count": m.get("detour_path_lt_1_count", 0),
            "detour_anchor_lt_1_count": m.get("detour_anchor_lt_1_count", 0),
        },
        "checks": checks,
        "sampled_plan_failures": sampled_failures,
        "effective_plan_errors": {
            "extra_node_count": 0,
            "extra_edge_count": 0,
            "missing_edge_count": 0,
            "edge_type_mismatch_count": sum(1 for x in planned_paths if x.get("matched_extracted_edge_type_mismatch")),
            "geometry_bug_count": geometry_bug_count,
        },
        "main_route_total_L_error": extracted_main_edge_sum_L_error,
        "extracted_main_edge_sum_L_error": extracted_main_edge_sum_L_error,
        "generated_main_L_error": generated_main_L_error,
        "final_bfs_L_error": final_bfs_L_error,
        "hinted_main_route_L_error": hinted_main_route_L_error,
        "dead_L_multiset_match": accepted_dead_L == extracted_dead_L,
        "dead_L_error_mean": float(np.mean([abs(a-b) for a,b in zip(accepted_dead_L, extracted_dead_L)])) if accepted_dead_L and extracted_dead_L else None,
        "canonical_consistency_pass": effective_plan_match,
        "strict_failure_reasons": strict_reasons,
    }



# =============================================================================
# visualization
# =============================================================================


def plot_maze_overlay(maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], graph: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(maze, cmap="gray_r", vmin=0, vmax=1)
    ax.scatter([start[1]], [start[0]], marker="o", s=160, label="start")
    ax.scatter([goal[1]], [goal[0]], marker="*", s=220, label="goal")
    markers = {"choice_node": "s", "endpoint_node": "x", "island_node": "D"}
    for node in graph.get("nodes", []):
        r, c = node["anchor"]
        ax.scatter([c], [r], marker=markers.get(node["node_type"], "o"), s=90)
        label = f"{node['node_id']}\n{node['node_type'].replace('_node','')}"
        if node.get("is_start"):
            label += "\nS"
        if node.get("is_goal"):
            label += "\nG"
        ax.text(c + 0.05, r + 0.05, label, fontsize=7)
    linestyles = {"main_edge": "-", "dead_edge": "--", "alternative_edge": ":", "island_edge": "-."}
    for edge in graph.get("edges", []):
        pts = edge.get("cell_path", [])
        if len(pts) < 2:
            continue
        ys = [p[0] for p in pts]
        xs = [p[1] for p in pts]
        ax.plot(xs, ys, linestyle=linestyles.get(edge["edge_type"], "-"), linewidth=2)
        mid = pts[len(pts) // 2]
        ax.text(mid[1] + 0.05, mid[0] - 0.1, f"{edge['edge_id']}\n{edge['edge_type']}\nL={edge['L']} D={edge['D']}", fontsize=6)
    ax.set_xticks(range(maze.shape[1])); ax.set_yticks(range(maze.shape[0]))
    ax.grid(True, linewidth=0.4)
    ax.set_title("maze overlay: nodes and edges")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_decision_graph(graph: Dict[str, Any], path: Path, title: str = "decision graph") -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(7, 5))
    if nx is None:
        ax.text(0.5, 0.5, "networkx unavailable", ha="center")
    else:
        G = nx.Graph()
        for node in graph.get("nodes", []):
            G.add_node(node["node_id"], label=f"{node['node_id']}\n{node['node_type']}\ndeg={node.get('effective_degree')}")
        for edge in graph.get("edges", []):
            G.add_edge(edge["source"], edge["target"], label=f"{edge['edge_id']}\n{edge['edge_type']}\nL={edge['L']} D={edge['D']}")
        pos = nx.spring_layout(G, seed=42) if len(G.nodes) else {}
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=900)
        nx.draw_networkx_edges(G, pos, ax=ax)
        nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, "label"), font_size=7, ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "label"), font_size=6, ax=ax)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_plan_vs_extracted(samples: List[Dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if not samples:
        for ax in axes:
            ax.text(0.5, 0.5, "no samples", ha="center")
            ax.axis("off")
    else:
        s = samples[0]
        plan = s.get("planned_graph", {})
        comp = s.get("consistency_report", {})
        axes[0].text(0.05, 0.85, "Planned Graph", fontsize=12, weight="bold")
        axes[0].text(0.05, 0.65, json.dumps({k: plan.get(k) for k in ["profile", "graph_budget", "main_L", "main_D", "dead_edge_count", "alternative_edge_count", "island_component_count"]}, indent=2), family="monospace", fontsize=8)
        axes[1].text(0.05, 0.85, "Extracted Consistency", fontsize=12, weight="bold")
        axes[1].text(0.05, 0.55, json.dumps(comp, indent=2)[:1200], family="monospace", fontsize=8)
        for ax in axes:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# generation pipeline
# =============================================================================


def graph_counts_for_delta(graph: Dict[str, Any]) -> Dict[str, int]:
    m = graph.get("metrics", {})
    return {
        "choice_node_count": int(m.get("choice_node_count", 0) or 0),
        "endpoint_node_count": int(m.get("endpoint_node_count", 0) or 0),
        "island_node_count": int(m.get("island_node_count", 0) or 0),
        "main_edge_count": int(m.get("main_edge_count", 0) or 0),
        "dead_edge_count": int(m.get("dead_edge_count", 0) or 0),
        "alternative_edge_count": int(m.get("alternative_edge_count", 0) or 0),
        "island_edge_count": int(m.get("island_edge_count", 0) or 0),
    }


def make_planned_main_record(plan: Dict[str, Any], main_path: Sequence[Tuple[int, int]], maze: np.ndarray) -> Dict[str, Any]:
    start, goal = tuple(main_path[0]), tuple(main_path[-1])
    bp = bfs_path(maze, start, goal)
    L = len(main_path) - 1
    D_path = manhattan(start, goal)
    final_bfs_L = None if bp is None else len(bp) - 1
    return {
        "planned_edge_id": "planned_main",
        "edge_type": "main_edge",
        "sampled": True,
        "accepted": True,
        "planned_main_L": int(plan.get("main_L", L)),
        "generated_main_L": L,
        "hinted_main_route_L": L,
        "final_bfs_L": final_bfs_L,
        "main_path_hint_available": True,
        "main_path_source": "hinted",
        "main_path_ambiguous": False,
        "target_L": int(plan.get("main_L", L)),
        "target_D": int(plan.get("main_D", D_path)),
        "cell_path": list(main_path),
        "actual_cell_path": list(main_path),
        "actual_L": L,
        "D_path": D_path,
        "detour_path": None if D_path == 0 else L / D_path,
        "is_induced": is_induced_path(main_path),
        "bfs_start_goal": final_bfs_L,
        "reject_reason": None,
        "matched_extracted_edge_id": None,
        "matched_extracted_edge_type": None,
    }

def final_edge_record(pe: Dict[str, Any], accepted: bool, reason: str, path: Optional[List[Tuple[int, int]]], attempt_count: int) -> Dict[str, Any]:
    out = {
        "planned_edge_id": pe.get("edge_id") or pe.get("component_id"),
        "edge_type": pe.get("edge_type"),
        "planning_mode": pe.get("planning_mode") or pe.get("alternative_planning_mode"),
        "sampled": True,
        "accepted": bool(accepted),
        "target_L": pe.get("target_L", pe.get("L")),
        "target_D": pe.get("D"),
        "target_extra_L": pe.get("target_extra_L", pe.get("extra_L")),
        "main_interval": pe.get("main_interval"),
        "expected_choice_anchors": pe.get("expected_choice_anchors"),
        "alternative_equal_length": bool(pe.get("alternative_equal_length", False)),
        "main_path_attempt_rejected": bool(pe.get("main_path_attempt_rejected", False)),
        "free_budget": pe.get("free_budget"),
        "actual_cell_path": list(path) if path else None,
        "actual_L": (len(path) - 1) if path else None,
        "reject_reason": None if accepted else reason,
        "attempt_count": attempt_count,
        "matched_extracted_edge_id": None,
        "matched_extracted_edge_type": None,
    }
    if path:
        out["D_path"] = manhattan(tuple(path[0]), tuple(path[-1]))
        out["detour_path"] = None if out["D_path"] == 0 else out["actual_L"] / out["D_path"]
    return out


def edge_delta_record(maze_id: str, pe: Dict[str, Any], before_graph: Dict[str, Any], after_graph: Optional[Dict[str, Any]], accepted: bool, reason: str) -> Dict[str, Any]:
    before = graph_counts_for_delta(before_graph)
    after = graph_counts_for_delta(after_graph or {})
    actual = {k: after.get(k, 0) - before.get(k, 0) for k in before}
    et = pe.get("edge_type")
    if et == "dead_edge":
        expected = {"dead_edge_count": 1, "endpoint_node_count": 1}
    elif et == "alternative_edge":
        expected = {"alternative_edge_count": 1, "endpoint_node_count": 0}
    elif et == "island_edge":
        expected = {"island_node_count": 1}
    else:
        expected = {}
    status = "rejected"
    if accepted:
        status = "expected_with_main_split" if actual.get("main_edge_count", 0) != 0 else "accepted"
        if et == "dead_edge" and actual.get("dead_edge_count", 0) < 1:
            status = "accepted_but_expected_delta_missing"
        if et == "alternative_edge" and actual.get("alternative_edge_count", 0) < 1:
            status = "accepted_but_not_extracted_as_alternative"
    return {
        "maze_id": maze_id,
        "planned_edge_id": pe.get("edge_id") or pe.get("component_id"),
        "edge_type": et,
        "before_metrics": before,
        "after_metrics": after,
        "expected_delta": expected,
        "actual_delta": actual,
        "delta_status": status,
        "accepted": bool(accepted),
        "reject_reason": None if accepted else reason,
    }


def build_maze_from_plan(plan: Dict[str, Any], cfg: Dict[str, Any], rng: random.Random, maze_id: str = "maze_00000") -> Dict[str, Any]:
    debug: Dict[str, Any] = {
        "main_path_attempt_count": 0,
        "edge_attempt_count": 0,
        "edge_path_search_steps": 0,
        "extract_graph_call_count": 0,
        "bfs_call_count": 0,
        "main_path_fail_reason_counts": Counter(),
        "edge_reject_reason_counts": Counter(),
        "anchor_reject_reason_counts": Counter(),
        "anchor_candidate_count": 0,
    }
    t0 = time.perf_counter()
    n = int(cfg["grid"]["size"])
    alt_edges = [pe for pe in plan.get("planned_edges", []) if pe.get("edge_type") == "alternative_edge"]
    planning_mode = plan.get("alternative_planning_mode") or cfg.get("alternative_edge", {}).get("planning_mode", "post_main_baseline")
    use_coplanned = planning_mode == "co_planned_with_main" and bool(alt_edges)
    max_co_attempts = int(cfg.get("alternative_edge", {}).get("max_coplanned_main_attempts", 120))

    main_path = None
    maze = None
    start_cell = goal_cell = None
    planned_edge_paths: List[Dict[str, Any]] = []
    graph_delta_debug: List[Dict[str, Any]] = []
    coplanned_done: Set[str] = set()

    if use_coplanned:
        last_reason = "alternative_coplanned_no_valid_route"
        for main_try in range(max_co_attempts):
            candidate_main = generate_induced_main_path(n, int(plan["main_L"]), int(plan["main_D"]), cfg, rng, debug)
            if candidate_main is None:
                last_reason = "main_path_generation_failed"
                continue
            candidate_maze = np.ones((n, n), dtype=np.int8)
            apply_edge_path(candidate_maze, candidate_main)
            start_tmp, goal_tmp = candidate_main[0], candidate_main[-1]
            local_records = [make_planned_main_record(plan, candidate_main, candidate_maze)]
            local_delta: List[Dict[str, Any]] = []
            all_ok = True
            before_graph = extract_decision_graph(candidate_maze, start_tmp, goal_tmp, cfg, main_path_hint=candidate_main)
            debug["extract_graph_call_count"] += 1
            for pe0 in alt_edges:
                pe = dict(pe0)
                pe["planning_mode"] = "co_planned_with_main"
                attempts_before = int(debug.get("edge_attempt_count", 0))
                ok, reason, edge_path, meta = add_alternative_edge_coplanned(candidate_maze, candidate_main, pe, cfg, rng, debug)
                attempt_count = int(debug.get("edge_attempt_count", 0)) - attempts_before
                pe.update(meta)
                if not ok:
                    pe["main_path_attempt_rejected"] = True
                    rec = final_edge_record(pe, False, reason, None, attempt_count)
                    local_records.append(rec)
                    local_delta.append(edge_delta_record(maze_id, pe, before_graph, None, False, reason))
                    all_ok = False
                    last_reason = reason
                    break
                after_graph = extract_decision_graph(candidate_maze, start_tmp, goal_tmp, cfg, main_path_hint=candidate_main)
                debug["extract_graph_call_count"] += 1
                bp = bfs_path(candidate_maze, start_tmp, goal_tmp)
                debug["bfs_call_count"] += 1
                if cfg["shortcut_constraints"].get("require_goal_shortest_path_equal_main_L", True) and (not bp or len(bp) - 1 != plan["main_L"]):
                    pe["main_path_attempt_rejected"] = True
                    reason = "alternative_coplanned_shortest_path_changed_goal"
                    rec = final_edge_record(pe, False, reason, None, attempt_count)
                    local_records.append(rec)
                    local_delta.append(edge_delta_record(maze_id, pe, before_graph, after_graph, False, reason))
                    all_ok = False
                    last_reason = reason
                    break
                rec = final_edge_record(pe, True, "ok", edge_path, attempt_count)
                rec["alternative_shortcut_violation"] = False
                rec["alternative_equal_length"] = bool(pe.get("alternative_equal_length", False))
                local_records.append(rec)
                local_delta.append(edge_delta_record(maze_id, pe, before_graph, after_graph, True, "ok"))
                coplanned_done.add(pe.get("edge_id"))
                before_graph = after_graph
            if all_ok:
                main_path = candidate_main
                maze = candidate_maze
                start_cell, goal_cell = start_tmp, goal_tmp
                planned_edge_paths = local_records
                graph_delta_debug = local_delta
                break
        if main_path is None:
            # Return a hard generation record with sampled alternative failure visible.
            return {
                "success": False,
                "hard_generation_success": False,
                "edge_embedding_success": False,
                "strict_consistency_success": False,
                "hard_failure_reason": last_reason,
                "debug": debug,
                "runtime_ms": (time.perf_counter() - t0) * 1000,
                "planned_graph": {**plan, "alternative_planning_mode": planning_mode},
                "planned_edge_paths": planned_edge_paths,
                "graph_delta_debug": graph_delta_debug,
                "edge_geometry_debug": [],
            }
    else:
        main_path = generate_induced_main_path(n, int(plan["main_L"]), int(plan["main_D"]), cfg, rng, debug)
        if main_path is None:
            return {
                "success": False,
                "hard_generation_success": False,
                "edge_embedding_success": False,
                "strict_consistency_success": False,
                "hard_failure_reason": "main_path_generation_failed",
                "debug": debug,
                "runtime_ms": (time.perf_counter() - t0) * 1000,
                "planned_graph": {**plan, "alternative_planning_mode": planning_mode},
                "planned_edge_paths": [],
                "graph_delta_debug": [],
                "edge_geometry_debug": [],
            }
        maze = np.ones((n, n), dtype=np.int8)
        apply_edge_path(maze, main_path)
        start_cell, goal_cell = main_path[0], main_path[-1]
        planned_edge_paths = [make_planned_main_record(plan, main_path, maze)]

    assert maze is not None and main_path is not None and start_cell is not None and goal_cell is not None
    plan = {**plan, "alternative_planning_mode": planning_mode}

    for pe in plan.get("planned_edges", []):
        if pe.get("edge_type") == "main_edge" or pe.get("edge_id") in coplanned_done:
            continue
        before_maze = maze.copy()
        before_graph = extract_decision_graph(maze, start_cell, goal_cell, cfg, main_path_hint=main_path)
        debug["extract_graph_call_count"] += 1
        attempts_before = int(debug.get("edge_attempt_count", 0))
        if pe.get("edge_type") == "dead_edge":
            ok, reason, edge_path = add_dead_edge(maze, main_path, pe, cfg, rng, debug)
        elif pe.get("edge_type") == "alternative_edge":
            pe = {**pe, "planning_mode": "post_main_baseline"}
            ok, reason, edge_path = add_alternative_edge(maze, main_path, pe, cfg, rng, debug)
        else:
            continue
        attempt_count = int(debug.get("edge_attempt_count", 0)) - attempts_before
        if not ok:
            maze[:, :] = before_maze
            rec = final_edge_record(pe, False, reason, None, attempt_count)
            planned_edge_paths.append(rec)
            graph_delta_debug.append(edge_delta_record(maze_id, pe, before_graph, None, False, reason))
            continue
        after_graph = extract_decision_graph(maze, start_cell, goal_cell, cfg, main_path_hint=main_path)
        debug["extract_graph_call_count"] += 1
        bp = bfs_path(maze, start_cell, goal_cell)
        debug["bfs_call_count"] += 1
        if cfg["shortcut_constraints"].get("require_goal_shortest_path_equal_main_L", True) and (not bp or len(bp) - 1 != plan["main_L"]):
            maze[:, :] = before_maze
            debug["edge_reject_reason_counts"]["shortest_path_changed_goal"] += 1
            reason = "shortest_path_changed_goal"
            rec = final_edge_record(pe, False, reason, None, attempt_count)
            planned_edge_paths.append(rec)
            graph_delta_debug.append(edge_delta_record(maze_id, pe, before_graph, after_graph, False, reason))
            continue
        rec = final_edge_record(pe, True, "ok", edge_path, attempt_count)
        planned_edge_paths.append(rec)
        graph_delta_debug.append(edge_delta_record(maze_id, pe, before_graph, after_graph, True, "ok"))

    for island in plan.get("planned_islands", []):
        before_maze = maze.copy()
        before_graph = extract_decision_graph(maze, start_cell, goal_cell, cfg, main_path_hint=main_path)
        debug["extract_graph_call_count"] += 1
        attempts_before = int(debug.get("edge_attempt_count", 0))
        free_budget = int(island.get("free_budget", 0))
        ok, reason, cells = add_island_component(maze, free_budget, cfg, rng, debug)
        attempt_count = int(debug.get("edge_attempt_count", 0)) - attempts_before
        pe = {**island, "edge_type": "island_edge", "component_id": island.get("component_id")}
        after_graph = extract_decision_graph(maze, start_cell, goal_cell, cfg, main_path_hint=main_path)
        debug["extract_graph_call_count"] += 1
        if ok and free_budget > 0 and not cells:
            maze[:, :] = before_maze
            reason = "island_no_cells_generated"
            ok = False
            after_graph = before_graph
        if ok and free_budget > 0:
            before_islands = int(before_graph.get("metrics", {}).get("island_component_count", 0) or 0)
            after_islands = int(after_graph.get("metrics", {}).get("island_component_count", 0) or 0)
            if after_islands <= before_islands:
                maze[:, :] = before_maze
                reason = "island_component_not_extracted"
                ok = False
                after_graph = before_graph
        rec = final_edge_record(pe, ok, reason, cells if (ok and cells) else None, attempt_count)
        rec["cells"] = cells if ok else []
        planned_edge_paths.append(rec)
        graph_delta_debug.append(edge_delta_record(maze_id, pe, before_graph, after_graph, ok, reason))

    final_bp = bfs_path(maze, start_cell, goal_cell)
    debug["bfs_call_count"] += 1
    if planned_edge_paths:
        planned_edge_paths[0]["final_bfs_L"] = None if final_bp is None else len(final_bp) - 1
        planned_edge_paths[0]["planned_main_L"] = int(plan.get("main_L", planned_edge_paths[0].get("planned_main_L", 0)))
        planned_edge_paths[0]["generated_main_L"] = len(main_path) - 1
        planned_edge_paths[0]["hinted_main_route_L"] = len(main_path) - 1
        planned_edge_paths[0]["main_path_hint_available"] = True
        planned_edge_paths[0]["main_path_source"] = "hinted"
        planned_edge_paths[0]["main_path_ambiguous"] = any(pe.get("edge_type") == "alternative_edge" and pe.get("accepted") and pe.get("alternative_equal_length") for pe in planned_edge_paths)
    extracted = extract_decision_graph(maze, start_cell, goal_cell, cfg, main_path_hint=main_path)
    debug["extract_graph_call_count"] += 1
    match_planned_edges_to_extracted(planned_edge_paths, extracted)
    comp = compare_plan_extracted(plan, extracted, planned_edge_paths)
    comp["maze_id"] = maze_id
    cell_metrics = cell_level_validation_metrics(maze, start_cell, goal_cell)
    runtime_ms = (time.perf_counter() - t0) * 1000
    edge_geometry_debug = [{
        "edge_id": e.get("edge_id"),
        "edge_type": e.get("edge_type"),
        "L": e.get("L"),
        "D_path": e.get("D_path"),
        "detour_path": e.get("detour_path"),
        "D_anchor": e.get("D_anchor"),
        "detour_anchor": e.get("detour_anchor"),
        "anchor_geometry_mismatch": e.get("anchor_geometry_mismatch"),
        "extraction_geometry_bug": e.get("extraction_geometry_bug"),
    } for e in extracted.get("edges", [])]
    hard_success = True
    edge_embedding_success = bool(comp.get("edge_embedding_success"))
    strict_success = bool(comp.get("strict_consistency_success"))
    return {
        "success": hard_success,
        "hard_generation_success": hard_success,
        "edge_embedding_success": edge_embedding_success,
        "strict_consistency_success": strict_success,
        "maze": maze,
        "start": start_cell,
        "goal": goal_cell,
        "main_path": main_path,
        "planned_graph": plan,
        "planned_edge_paths": planned_edge_paths,
        "graph_delta_debug": graph_delta_debug,
        "edge_geometry_debug": edge_geometry_debug,
        "extracted_decision_graph": extracted,
        "consistency_report": comp,
        "cell_level_validation_metrics": cell_metrics,
        "debug": debug,
        "runtime_ms": runtime_ms,
    }


# =============================================================================
# reports and hints
# =============================================================================


def summarize_results(results: List[Dict[str, Any]], requested: int) -> Dict[str, Any]:
    hard_successes = [r for r in results if r.get("hard_generation_success", r.get("success"))]
    failures = [r for r in results if not r.get("hard_generation_success", r.get("success"))]
    plans = [r.get("planned_graph", {}) for r in hard_successes]
    extracted = [r.get("extracted_decision_graph", {}) for r in hard_successes]
    consistency = [r.get("consistency_report", {}) for r in hard_successes]
    perf = [r.get("performance", r) for r in results]
    summary = {
        "n_requested": requested,
        "n_success": len(hard_successes),  # backward-compatible alias for hard generation success
        "hard_generation_success_rate": sum(1 for r in results if r.get("hard_generation_success", r.get("success"))) / max(1, requested),
        "edge_embedding_success_rate": sum(1 for r in results if r.get("edge_embedding_success")) / max(1, requested),
        "strict_consistency_success_rate": sum(1 for r in results if r.get("strict_consistency_success")) / max(1, requested),
        "hard_failure_count": len(failures),
    }
    # Keep old keys but make them explicit aliases, not primary success semantics.
    summary["success_rate"] = summary["hard_generation_success_rate"]
    summary["canonical_consistency_rate"] = summary["strict_consistency_success_rate"]
    graph_plan = {
        "main_L": stats([p.get("main_L") for p in plans]),
        "main_D": stats([p.get("main_D") for p in plans]),
        "graph_budget": stats([p.get("graph_budget") for p in plans]),
        "sampled_dead_edge_count": stats([p.get("dead_edge_count") for p in plans]),
        "sampled_alternative_edge_count": stats([p.get("alternative_edge_count") for p in plans]),
        "sampled_island_component_count": stats([p.get("island_component_count") for p in plans]),
    }
    plan_embedding = {
        "sampled_dead_edge_count": stats([c.get("sampled_plan_summary", {}).get("dead_edge_count") for c in consistency]),
        "accepted_dead_edge_count": stats([c.get("accepted_plan_summary", {}).get("dead_edge_count") for c in consistency]),
        "sampled_alternative_edge_count": stats([c.get("sampled_plan_summary", {}).get("alternative_edge_count") for c in consistency]),
        "accepted_alternative_edge_count": stats([c.get("accepted_plan_summary", {}).get("alternative_edge_count") for c in consistency]),
        "sampled_island_count": stats([c.get("sampled_plan_summary", {}).get("island_component_count") for c in consistency]),
        "accepted_island_count": stats([c.get("accepted_plan_summary", {}).get("island_component_count") for c in consistency]),
    }
    main_route_lengths = {
        "planned_main_L": stats([c.get("main_lengths", {}).get("planned_main_L") for c in consistency]),
        "generated_main_L_error": stats([c.get("generated_main_L_error") for c in consistency]),
        "final_bfs_L_error": stats([c.get("final_bfs_L_error") for c in consistency]),
        "hinted_main_route_L_error": stats([c.get("hinted_main_route_L_error") for c in consistency]),
        "extracted_main_edge_sum_L_error": stats([c.get("extracted_main_edge_sum_L_error") for c in consistency]),
    }
    extraction = {
        "choice_node_count": stats([e.get("metrics", {}).get("choice_node_count") for e in extracted]),
        "endpoint_node_count": stats([e.get("metrics", {}).get("endpoint_node_count") for e in extracted]),
        "island_node_count": stats([e.get("metrics", {}).get("island_node_count") for e in extracted]),
        "main_edge_count": stats([e.get("metrics", {}).get("main_edge_count") for e in extracted]),
        "dead_edge_count": stats([e.get("metrics", {}).get("dead_edge_count") for e in extracted]),
        "alternative_edge_count": stats([e.get("metrics", {}).get("alternative_edge_count") for e in extracted]),
        "extracted_main_edge_sum_L_error": stats([c.get("extracted_main_edge_sum_L_error") for c in consistency]),
        "geometry_bug_count": stats([e.get("metrics", {}).get("geometry_bug_count") for e in extracted]),
        "detour_path_lt_1_count": stats([e.get("metrics", {}).get("detour_path_lt_1_count") for e in extracted]),
        "detour_anchor_lt_1_count": stats([e.get("metrics", {}).get("detour_anchor_lt_1_count") for e in extracted]),
    }
    consistency_summary = {
        "sampled_plan_match_rate": sum(1 for c in consistency if c.get("sampled_plan_match")) / max(1, len(consistency)),
        "effective_plan_match_rate": sum(1 for c in consistency if c.get("effective_plan_match")) / max(1, len(consistency)),
        "strict_consistency_success_rate": sum(1 for c in consistency if c.get("strict_consistency_success")) / max(1, len(consistency)),
        "generated_main_L_error_mean": stats([c.get("generated_main_L_error") for c in consistency])["mean"],
        "final_bfs_L_error_mean": stats([c.get("final_bfs_L_error") for c in consistency])["mean"],
        "hinted_main_route_L_error_mean": stats([c.get("hinted_main_route_L_error") for c in consistency])["mean"],
        "extracted_main_edge_sum_L_error_mean": stats([c.get("extracted_main_edge_sum_L_error") for c in consistency])["mean"],
        "dead_edge_count_effective_match_rate": sum(1 for c in consistency if c.get("checks", {}).get("dead_edge_count_match_effective")) / max(1, len(consistency)),
        "alternative_edge_count_effective_match_rate": sum(1 for c in consistency if c.get("checks", {}).get("alternative_edge_count_match_effective")) / max(1, len(consistency)),
        "island_component_match_rate": sum(1 for c in consistency if c.get("island_checks", {}).get("island_component_count_match")) / max(1, len(consistency)),
        "geometry_bug_count": sum(c.get("effective_plan_errors", {}).get("geometry_bug_count", 0) or 0 for c in consistency),
        "detour_path_lt_1_count": sum(e.get("metrics", {}).get("detour_path_lt_1_count", 0) or 0 for e in extracted),
        "detour_anchor_lt_1_count": sum(e.get("metrics", {}).get("detour_anchor_lt_1_count", 0) or 0 for e in extracted),
        "extra_node_mean": stats([c.get("effective_plan_errors", {}).get("extra_node_count") for c in consistency])["mean"],
        "extra_edge_mean": stats([c.get("effective_plan_errors", {}).get("extra_edge_count") for c in consistency])["mean"],
        "missing_edge_mean": stats([c.get("effective_plan_errors", {}).get("missing_edge_count") for c in consistency])["mean"],
        "edge_type_mismatch_count": sum(c.get("effective_plan_errors", {}).get("edge_type_mismatch_count", 0) or 0 for c in consistency),
    }
    performance = {
        "runtime_ms": stats([p.get("runtime_ms") for p in perf]),
        "main_path_attempt_count": stats([p.get("main_path_attempt_count") for p in perf]),
        "edge_attempt_count": stats([p.get("edge_attempt_count") for p in perf]),
        "extract_graph_call_count": stats([p.get("extract_graph_call_count") for p in perf]),
        "bfs_call_count": stats([p.get("bfs_call_count") for p in perf]),
    }
    all_planned_edges = [pe for r in hard_successes for pe in r.get("planned_edge_paths", [])[1:]]
    alt_edges = [pe for pe in all_planned_edges if pe.get("edge_type") == "alternative_edge"]
    islands = [pe for pe in all_planned_edges if pe.get("edge_type") == "island_edge"]
    alt_reasons = Counter(pe.get("reject_reason") or "accepted" for pe in alt_edges)
    island_reasons = Counter(pe.get("reject_reason") or "accepted" for pe in islands)
    accepted_alt_count = sum(1 for pe in alt_edges if pe.get("accepted"))
    alternative_planning = {
        "sampled_alternative_count": len(alt_edges),
        "accepted_alternative_count": accepted_alt_count,
        "extracted_alternative_count": sum(c.get("alternative_checks", {}).get("extracted_alternative_count", 0) or 0 for c in consistency),
        "alternative_equal_length_count": sum(1 for pe in alt_edges if pe.get("accepted") and pe.get("alternative_equal_length")),
        "alternative_shortcut_violation_count": sum(c.get("alternative_checks", {}).get("alternative_shortcut_violation_count", 0) or 0 for c in consistency),
        "alternative_extracted_match_rate": sum(1 for pe in alt_edges if pe.get("accepted") and pe.get("matched_extracted_edge_type") == "alternative_edge") / max(1, accepted_alt_count),
        "alternative_reject_top_reasons": alt_reasons.most_common(8),
    }
    island_planning = {
        "sampled_island_component_count": len(islands),
        "accepted_island_component_count": sum(1 for pe in islands if pe.get("accepted")),
        "extracted_island_component_count": sum(c.get("island_checks", {}).get("extracted_island_component_count", 0) or 0 for c in consistency),
        "island_component_match_rate": sum(1 for c in consistency if c.get("island_checks", {}).get("island_component_count_match")) / max(1, len(consistency)),
        "island_node_count_mean": stats([c.get("island_checks", {}).get("island_node_count") for c in consistency])["mean"],
        "island_edge_count_mean": stats([c.get("island_checks", {}).get("island_edge_count") for c in consistency])["mean"],
        "island_no_cells_generated_count": sum(1 for pe in islands if pe.get("reject_reason") == "island_no_cells_generated"),
        "island_reject_top_reasons": island_reasons.most_common(8),
    }
    return {"summary": summary, "graph_plan_sampling": graph_plan, "plan_embedding": plan_embedding, "main_route_lengths": main_route_lengths, "extraction_summary": extraction, "consistency_summary": consistency_summary, "alternative_planning": alternative_planning, "island_planning": island_planning, "performance": performance}


def counter_to_dict(c: Any) -> Dict[str, int]:
    if isinstance(c, Counter):
        return dict(c)
    if isinstance(c, defaultdict):
        return dict(c)
    return dict(c or {})


def performance_from_debug(result: Dict[str, Any]) -> Dict[str, Any]:
    dbg = result.get("debug", {})
    return {
        "runtime_ms": result.get("runtime_ms"),
        "main_path_attempt_count": dbg.get("main_path_attempt_count", 0),
        "edge_attempt_count": dbg.get("edge_attempt_count", 0),
        "extract_graph_call_count": dbg.get("extract_graph_call_count", 0),
        "bfs_call_count": dbg.get("bfs_call_count", 0),
        "failed_reason_counts": {"main_path": counter_to_dict(dbg.get("main_path_fail_reason_counts", {})), "edge": counter_to_dict(dbg.get("edge_reject_reason_counts", {}))},
    }


def error_hints(aggregate_debug: Dict[str, Any], report: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    th = cfg.get("debug", {}).get("hint_thresholds", {})
    hints: List[Dict[str, Any]] = []
    cs = report.get("consistency_summary", {})
    summary = report.get("summary", {})
    detour_path_lt_1 = int(cs.get("detour_path_lt_1_count", 0) or 0)
    detour_anchor_lt_1 = int(cs.get("detour_anchor_lt_1_count", 0) or 0)
    if detour_path_lt_1 > 0:
        hints.append({"hint": "detour_path < 1 detected.", "likely_direction": ["Extract edge geometry is inconsistent.", "L and D_path may not be computed from the same endpoints."], "suggested_next_json": ["extracted_decision_graphs.json", "edge_geometry_debug.json", "maze_overlay_samples/"], "suggested_config_fields": ["extract_decision_graph edge path endpoint logic", "room representative anchor handling"]})
    if detour_anchor_lt_1 > 0 and detour_path_lt_1 == 0:
        hints.append({"hint": "anchor geometry mismatch detected.", "likely_direction": ["Room representative anchor differs from actual edge exit cell.", "This is acceptable only if core metrics use D_path."], "suggested_next_json": ["edge_geometry_debug.json", "debug_extraction_report.json"], "suggested_config_fields": ["room_detection.use_2x2_free_block", "room_detection.room_min_component_size"]})
    edge_embedding_rate = float(summary.get("edge_embedding_success_rate", 0.0) or 0.0)
    effective_rate = float(cs.get("effective_plan_match_rate", 0.0) or 0.0)
    if edge_embedding_rate < float(th.get("edge_embedding_success_rate_low", 0.8)) and effective_rate >= float(th.get("effective_plan_match_rate_high", 0.8)):
        hints.append({"hint": "Effective plan matches, but sampled plan does not.", "likely_direction": ["Generation is best-effort.", "Planned edges are too hard to embed, especially alternative edges."], "suggested_next_json": ["planned_edge_paths.json", "edge_embedding_debug.json"], "suggested_config_fields": ["graph_plan_distribution.graph_budget.max", "alternative_edge.extra_L.min/max", "alternative_edge.anchor_main_interval_min/max"]})
    if effective_rate < float(th.get("effective_plan_match_rate_low", 0.8)):
        hints.append({"hint": "Effective plan does not match extracted graph.", "likely_direction": ["Accepted edges are not being extracted as intended.", "Graph delta validation is incomplete."], "suggested_next_json": ["planned_edge_paths.json", "graph_delta_debug.json", "plan_vs_extracted_pairs.json", "maze_overlay_samples/"], "suggested_config_fields": ["add_edge expected graph delta check", "edge classification rules"]})
    if cs.get("final_bfs_L_error_mean") is not None and float(cs.get("final_bfs_L_error_mean") or 0) > 0:
        hints.append({"hint": "Final BFS is shorter or longer than planned main path.", "likely_direction": ["Alternative or another structure changed the real start-goal shortest path."], "suggested_next_json": ["planned_edge_paths.json", "consistency_reports.json", "maze_overlay_samples/"], "suggested_config_fields": ["shortcut_constraints.require_goal_shortest_path_equal_main_L", "alternative_edge.extra_L.min/max"]})
    if (cs.get("final_bfs_L_error_mean") in (0, 0.0)) and (cs.get("hinted_main_route_L_error_mean") in (0, 0.0)) and cs.get("extracted_main_edge_sum_L_error_mean") is not None and float(cs.get("extracted_main_edge_sum_L_error_mean") or 0) > 0:
        hints.append({"hint": "Extracted main edge sum differs, but final BFS and hinted main path are correct.", "likely_direction": ["This is an Extract/Compare representation issue, not a shortest-path failure.", "Check main_path_hint handling and main/alternative edge classification."], "suggested_next_json": ["extracted_decision_graphs.json", "planned_edge_paths.json", "graph_delta_debug.json"], "suggested_config_fields": ["extract_decision_graph.main_path_hint", "edge_classification.alternative_edge_rule"]})
    ap = report.get("alternative_planning", {})
    ip = report.get("island_planning", {})
    if ap.get("sampled_alternative_count", 0) and ap.get("accepted_alternative_count", 0) == 0:
        hints.append({"hint": "Co-planned alternative still fails.", "likely_direction": ["Alternative loop may be too hard under current 8x8 and induced constraints.", "Inspect interval length, extra_L, and route collision reasons."], "suggested_next_json": ["planned_edge_paths.json", "graph_delta_debug.json", "maze_overlay_samples/"], "suggested_config_fields": ["alternative_edge.anchor_main_interval_min/max", "alternative_edge.extra_L.min/max", "main_path.main_L.max"]})
    if ap.get("accepted_alternative_count", 0) > ap.get("extracted_alternative_count", 0):
        hints.append({"hint": "Accepted alternative was not extracted as alternative.", "likely_direction": ["Extract may be treating alternative as main_edge because main_path_hint is missing or ignored."], "suggested_next_json": ["planned_edge_paths.json", "extracted_decision_graphs.json", "maze_overlay_samples/"], "suggested_config_fields": ["extract_decision_graph.main_path_hint", "edge classification rules"]})
    if ip.get("accepted_island_component_count", 0) != ip.get("extracted_island_component_count", 0):
        hints.append({"hint": "Island component mismatch.", "likely_direction": ["Compare should use island component count, not island node count.", "If cells=[], generation failed to create island component."], "suggested_next_json": ["planned_edge_paths.json", "extracted_decision_graphs.json", "graph_delta_debug.json"], "suggested_config_fields": ["island accepted condition", "island extraction component logic"]})
    # Preserve first-version generation hints as secondary signals.
    n = max(1, summary.get("n_requested", 1))
    main_failed = aggregate_debug.get("main_path_fail_reason_counts", {}).get("main_path_generation_failed", 0)
    if main_failed / n > float(th.get("main_path_generation_failed_rate", 0.3)):
        hints.append({"hint": "main_path_generation_failed is high.", "likely_direction": ["main_L may be too high for 8x8 or induced constraint.", "Check graph_plan_sampling_debug.json and main_path_fail_reason_counts."], "suggested_next_json": ["graph_plan_sampling_debug.json", "generation_failure_cases.json"], "suggested_config_fields": ["main_path.main_L.max", "main_path.main_L.mean", "main_path.main_D.min/max", "main_path.require_induced_path"]})
    if not hints:
        hints.append({"hint": "No dominant error direction crossed configured thresholds.", "likely_direction": ["Inspect planned_edge_paths.json, edge_geometry_debug.json and graph_delta_debug.json for local mismatches."], "suggested_next_json": ["planned_edge_paths.json", "edge_geometry_debug.json", "graph_delta_debug.json"], "suggested_config_fields": ["debug.hint_thresholds"]})
    return hints


def print_run_report(mode: str, out_dir: Path, report: Dict[str, Any], hints: List[Dict[str, Any]]) -> None:
    print("\n[3.0.2(4) Edge-Plan + Hint-First Alternative + Room Debug + Island Candidate Search]")
    print(f"Mode: {mode}")
    print(f"Output directory: {out_dir}")
    print_kv("\n=== Summary ===", [(k, v) for k, v in report["summary"].items() if k in ("n_requested", "hard_generation_success_rate", "edge_embedding_success_rate", "strict_consistency_success_rate", "hard_failure_count")])
    gp = report.get("graph_plan_sampling", {})
    print("\n=== GraphPlan Sampling ===")
    for k in ["main_L", "main_D", "graph_budget", "sampled_dead_edge_count", "sampled_alternative_edge_count", "sampled_island_component_count"]:
        s = gp.get(k, {})
        print(f"  {k:<34} mean={fmt_num(s.get('mean'),8,3)} p50={fmt_num(s.get('p50'),8,3)} min={fmt_num(s.get('min'),8,3)} max={fmt_num(s.get('max'),8,3)}")
    mrl = report.get("main_route_lengths", {})
    print("\n=== Main Route Lengths ===")
    for k in ["planned_main_L", "generated_main_L_error", "final_bfs_L_error", "hinted_main_route_L_error", "extracted_main_edge_sum_L_error"]:
        ss = mrl.get(k, {})
        print(f"  {k:<34} mean={fmt_num(ss.get('mean'),8,3)} p50={fmt_num(ss.get('p50'),8,3)} min={fmt_num(ss.get('min'),8,3)} max={fmt_num(ss.get('max'),8,3)}")
    pe = report.get("plan_embedding", {})
    print("\n=== Plan Embedding ===")
    for k in ["sampled_dead_edge_count", "accepted_dead_edge_count", "sampled_alternative_edge_count", "accepted_alternative_edge_count", "sampled_island_count", "accepted_island_count"]:
        s = pe.get(k, {})
        print(f"  {k:<34} mean={fmt_num(s.get('mean'),8,3)} p50={fmt_num(s.get('p50'),8,3)}")
    ex = report.get("extraction_summary", {})
    print("\n=== Extracted Graph ===")
    for k in ["extracted_main_edge_sum_L_error", "dead_edge_count", "alternative_edge_count", "geometry_bug_count", "detour_path_lt_1_count", "detour_anchor_lt_1_count"]:
        s = ex.get(k, {})
        print(f"  {k:<34} mean={fmt_num(s.get('mean'),8,3)} p50={fmt_num(s.get('p50'),8,3)}")
    cs = report.get("consistency_summary", {})
    print_kv("\n=== Compare ===", [(k, fmt_num(v, 10, 4) if isinstance(v, (int, float)) or v is None else v) for k, v in cs.items()])
    ap = report.get("alternative_planning", {})
    print_kv("\n=== Alternative Planning ===", [(k, v) for k, v in ap.items()])
    ip = report.get("island_planning", {})
    print_kv("\n=== Island Planning ===", [(k, v) for k, v in ip.items()])
    perf = report.get("performance", {})
    print("\n=== Runtime / Complexity ===")
    for k, s in perf.items():
        print(f"  {k:<34} mean={fmt_num(s.get('mean'),8,3)} p50={fmt_num(s.get('p50'),8,3)} p95={fmt_num(s.get('p95'),8,3)}")
    print("\n=== Error Direction Hints ===")
    for h in hints:
        print(f"[HINT] {h['hint']}")
        print("Likely direction:")
        for x in h.get("likely_direction", []):
            print(f"  - {x}")
        print("Suggested next JSON:")
        for x in h.get("suggested_next_json", []):
            print(f"  - {x}")
        print("Suggested config fields / code area:")
        for x in h.get("suggested_config_fields", []):
            print(f"  - {x}")


# =============================================================================
# modes
# =============================================================================


def resolve_root() -> Path:
    here = Path(__file__).resolve()
    # .../feature_maze/v3_0_maze_generation/experiments/script.py -> repo-ish root is parents[3]
    if len(here.parents) >= 3:
        return here.parents[3]
    return Path.cwd()


def resolve_out_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    root = resolve_root()
    run_name = args.run_name or f"{args.mode}__n-{args.n_mazes}__seed-{args.seed}"
    return root / "feature_maze" / "v3_0_maze_generation" / "outputs" / VERSION / run_name / ("test" if args.mode == "test" else "")


def aggregate_debugs(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {"main_path_fail_reason_counts": Counter(), "edge_reject_reason_counts": Counter(), "anchor_reject_reason_counts": Counter(), "dead_edge_attempt_count": 0, "alternative_edge_attempt_count": 0}
    for r in results:
        dbg = r.get("debug", {})
        out["main_path_fail_reason_counts"].update(counter_to_dict(dbg.get("main_path_fail_reason_counts", {})))
        edge = counter_to_dict(dbg.get("edge_reject_reason_counts", {}))
        out["edge_reject_reason_counts"].update(edge)
        out["anchor_reject_reason_counts"].update(counter_to_dict(dbg.get("anchor_reject_reason_counts", {})))
        out["dead_edge_attempt_count"] += sum(v for k, v in edge.items() if str(k).startswith("dead_edge")) + 1
        out["alternative_edge_attempt_count"] += sum(v for k, v in edge.items() if str(k).startswith("alternative_edge")) + edge.get("shortest_path_changed_goal", 0) + 1
    return out




def categorize_result_for_sample(result: Dict[str, Any], mode: str, diagnostic_profile: Optional[str]) -> List[str]:
    cats: List[str] = []
    comp = result.get("consistency_report", {})
    paths = result.get("planned_edge_paths", [])[1:]
    if diagnostic_profile == "main_only" and comp.get("strict_consistency_success"):
        cats.append("main_only_clean")
    if diagnostic_profile == "main_dead_only":
        if comp.get("strict_consistency_success"):
            cats.append("main_dead_clean")
        if any(pe.get("matched_extracted_edge_type_mismatch") for pe in paths):
            cats.append("main_dead_anchor_mismatch")
    if diagnostic_profile in ("main_alternative_only", "main_alternative_coplanned_only"):
        if any(pe.get("edge_type") == "alternative_edge" and pe.get("accepted") for pe in paths):
            cats.append("main_alternative_coplanned_success")
        if any(pe.get("edge_type") == "alternative_edge" and not pe.get("accepted") for pe in paths):
            cats.append("main_alternative_coplanned_failed")
    if diagnostic_profile == "main_island_only":
        if any(pe.get("edge_type") == "island_edge" and pe.get("accepted") for pe in paths):
            cats.append("main_island_success")
        if any(pe.get("edge_type") == "island_edge" and not pe.get("accepted") for pe in paths):
            cats.append("main_island_mismatch")
    if mode in ("generate", "roundtrip"):
        if comp.get("effective_plan_match"):
            cats.append("mixed_effective_match_true")
        else:
            cats.append("mixed_effective_match_false_or_main_route_error")
    return cats


def select_visualization_samples(results: List[Dict[str, Any]], cfg: Dict[str, Any], mode: str, diagnostic_profile: Optional[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ss = cfg.get("sample_selection", {})
    categories = ss.get("categories", ["mixed_effective_match_true", "mixed_effective_match_false_or_main_route_error"])
    max_samples = int(ss.get("max_samples", 10))
    selected: List[Dict[str, Any]] = []
    report: List[Dict[str, Any]] = []
    used: Set[str] = set()
    for cat in categories:
        found = None
        for r in results:
            if not r.get("hard_generation_success", r.get("success")):
                continue
            mid = r.get("maze_id")
            if mid in used:
                continue
            if cat in categorize_result_for_sample(r, mode, diagnostic_profile):
                found = r
                break
        if found is not None and len(selected) < max_samples:
            used.add(found.get("maze_id"))
            idx = len(selected) + 1
            selected.append({"category": cat, "index": idx, "result": found})
            report.append({"category": cat, "maze_id": found.get("maze_id"), "selected": True, "filename_prefix": f"{idx:02d}_{cat}__{found.get('maze_id')}"})
        else:
            report.append({"category": cat, "maze_id": None, "selected": False, "reason": "category_not_found"})
    if len(selected) < max_samples:
        for r in results:
            if not r.get("hard_generation_success", r.get("success")) or r.get("maze_id") in used:
                continue
            fallback_cat = "fallback_error" if not r.get("strict_consistency_success") else "fallback_clean"
            idx = len(selected) + 1
            selected.append({"category": fallback_cat, "index": idx, "result": r})
            report.append({"category": fallback_cat, "maze_id": r.get("maze_id"), "selected": True, "filename_prefix": f"{idx:02d}_{fallback_cat}__{r.get('maze_id')}"})
            used.add(r.get("maze_id"))
            if len(selected) >= max_samples:
                break
    return selected, report
def save_single_sample_outputs(result: Dict[str, Any], out_dir: Path, cfg: Dict[str, Any], prefix: str = "") -> None:
    maze = result["maze"]
    start = tuple(result["start"])
    goal = tuple(result["goal"])
    graph = result["extracted_decision_graph"]
    save_json(out_dir / f"{prefix}extracted_decision_graph.json", graph)
    save_json(out_dir / f"{prefix}node_table.json", graph.get("node_table", []))
    save_json(out_dir / f"{prefix}edge_table.json", graph.get("edge_table", []))
    save_json(out_dir / f"{prefix}decision_graph_metrics.json", graph.get("metrics", {}))
    save_json(out_dir / f"{prefix}cell_level_validation_metrics.json", result.get("cell_level_validation_metrics", cell_level_validation_metrics(maze, start, goal)))
    save_json(out_dir / f"{prefix}debug_extraction_report.json", graph.get("debug_extraction_report", {}))
    if cfg.get("visualization", {}).get("enabled", True):
        plot_maze_overlay(maze, start, goal, graph, out_dir / f"{prefix}maze_overlay.png")
        plot_decision_graph(graph, out_dir / f"{prefix}decision_graph.png")


def mode_test(args: argparse.Namespace, cfg: Dict[str, Any], out_dir: Path) -> None:
    if not args.maze_file:
        raise SystemExit("--maze-file is required for --mode test")
    data = load_json(Path(args.maze_file))
    maze = np.array(data["maze"], dtype=np.int8)
    start = tuple(data["start"])
    goal = tuple(data["goal"])
    t0 = time.perf_counter()
    extracted = extract_decision_graph(maze, start, goal, cfg)
    runtime_ms = (time.perf_counter() - t0) * 1000
    result = {
        "success": True,
        "maze": maze,
        "start": start,
        "goal": goal,
        "extracted_decision_graph": extracted,
        "cell_level_validation_metrics": cell_level_validation_metrics(maze, start, goal),
        "runtime_ms": runtime_ms,
        "debug": {"extract_graph_call_count": 1, "bfs_call_count": 1, "main_path_attempt_count": 0, "edge_attempt_count": 0},
    }
    result["performance"] = performance_from_debug(result)
    save_json(out_dir / "resolved_config.json", cfg)
    save_single_sample_outputs(result, out_dir, cfg)
    report = summarize_results([{**result, "planned_graph": {}, "consistency_report": {"canonical_consistency_pass": True}}], 1)
    hints = error_hints({}, report, cfg)
    save_json(out_dir / "performance_report.json", [performance_from_debug(result)])
    print_run_report("test", out_dir, report, hints)


def apply_diagnostic_profile_to_plan(plan: Dict[str, Any], profile: Optional[str], cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not profile:
        return plan
    plan = json.loads(json.dumps(json_safe(plan)))
    prof_cfg = (cfg or {}).get("diagnostic_profiles", {}).get(profile, {})
    if profile == "main_only":
        dead_n, alt_n, island_n = 0, 0, 0
    elif profile == "main_dead_only":
        dead_n, alt_n, island_n = 1, 0, 0
    elif profile == "main_alternative_only":
        dead_n, alt_n, island_n = 0, 1, 0
    elif profile == "main_alternative_coplanned_only":
        dead_n, alt_n, island_n = 0, 1, 0
    elif profile == "main_island_only":
        dead_n, alt_n, island_n = 0, 0, 1
    else:
        return plan
    dead_n = int(prof_cfg.get("dead_edge_count", dead_n))
    alt_n = int(prof_cfg.get("alternative_edge_count", alt_n))
    island_n = int(prof_cfg.get("island_component_count", island_n))
    alt_mode = prof_cfg.get("alternative_planning_mode")
    if alt_mode is None:
        alt_mode = "co_planned_with_main" if profile in ("main_alternative_only", "main_alternative_coplanned_only") else "co_planned_with_main"
    dead_edges = []
    for i in range(dead_n):
        dead_edges.append({"edge_id": f"planned_dead_{i}", "edge_type": "dead_edge", "L": 4})
    alt_edges = []
    alt_extra = int(prof_cfg.get("alternative_extra_L", 0))
    for i in range(alt_n):
        alt_edges.append({"edge_id": f"planned_alternative_{i}", "edge_type": "alternative_edge", "extra_L": alt_extra, "planning_mode": alt_mode})
    islands = []
    island_budget = int(prof_cfg.get("island_free_budget", 4))
    for i in range(island_n):
        islands.append({"component_id": f"planned_island_{i}", "edge_type": "island_edge", "free_budget": island_budget})
    plan["dead_edge_count"] = dead_n
    plan["alternative_edge_count"] = alt_n
    plan["island_component_count"] = island_n
    plan["graph_budget"] = dead_n + alt_n + island_n
    plan["profile"] = f"diagnostic_{profile}"
    plan["alternative_planning_mode"] = alt_mode
    plan["planned_edges"] = [{"edge_id": "planned_main", "edge_type": "main_edge", "L": plan["main_L"], "D": plan["main_D"], "detour": plan["main_L"] / max(1, plan["main_D"])}] + dead_edges + alt_edges
    plan["planned_islands"] = islands
    return plan


def run_generate_like(args: argparse.Namespace, cfg: Dict[str, Any], out_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    ensure_dir(out_dir)
    save_json(out_dir / "resolved_config.json", cfg)
    sampling_debug: Dict[str, Any] = {"main_path_sampling_reject_reason_counts": Counter(), "graph_budget_normalization_count": 0, "graph_budget_normalization_failed_count": 0, "main_path_sampling_failed_count": 0}
    results: List[Dict[str, Any]] = []
    failure_cases: List[Dict[str, Any]] = []
    for i in range(args.n_mazes):
        maze_id = f"maze_{i:05d}"
        plan = sample_graph_plan(cfg, rng, sampling_debug)
        plan = apply_diagnostic_profile_to_plan(plan, getattr(args, "diagnostic_profile", None) if args.mode == "diagnostic" else None, cfg)
        result = build_maze_from_plan(plan, cfg, rng, maze_id=maze_id)
        result["maze_id"] = maze_id
        result["performance"] = performance_from_debug(result)
        if not result.get("hard_generation_success", result.get("success")):
            failure_cases.append({"maze_id": result["maze_id"], "hard_failure_reason": result.get("hard_failure_reason"), "debug": json_safe(result.get("debug", {}))})
        results.append(result)
    overlay_dir = out_dir / "maze_overlay_samples"
    dg_dir = out_dir / "decision_graph_samples"
    ensure_dir(overlay_dir); ensure_dir(dg_dir)
    selected_samples, sample_selection_report = select_visualization_samples(results, cfg, args.mode, getattr(args, "diagnostic_profile", None) if args.mode == "diagnostic" else None)
    for item in selected_samples:
        r = item["result"]
        prefix = f"{item['index']:02d}_{item['category']}__{r['maze_id']}"
        plot_maze_overlay(r["maze"], tuple(r["start"]), tuple(r["goal"]), r["extracted_decision_graph"], overlay_dir / f"{prefix}.png")
        plot_decision_graph(r["extracted_decision_graph"], dg_dir / f"{prefix}.png")
    report = summarize_results(results, args.n_mazes)
    agg = aggregate_debugs(results)
    hints = error_hints(agg, report, cfg)
    save_json(out_dir / "generated_mazes.json", [{"maze_id": r.get("maze_id"), "maze": r.get("maze"), "start": r.get("start"), "goal": r.get("goal"), "hard_generation_success": r.get("hard_generation_success"), "edge_embedding_success": r.get("edge_embedding_success"), "strict_consistency_success": r.get("strict_consistency_success")} for r in results])
    save_json(out_dir / "planned_graphs.json", [{"maze_id": r.get("maze_id"), **r.get("planned_graph", {})} for r in results if r.get("hard_generation_success", r.get("success"))])
    save_json(out_dir / "extracted_decision_graphs.json", [{"maze_id": r.get("maze_id"), **r.get("extracted_decision_graph", {})} for r in results if r.get("hard_generation_success", r.get("success"))])
    save_json(out_dir / "consistency_reports.json", [r.get("consistency_report", {}) for r in results if r.get("hard_generation_success", r.get("success"))])
    save_json(out_dir / "planned_edge_paths.json", [{"maze_id": r.get("maze_id"), "sampled_plan_summary": r.get("consistency_report", {}).get("sampled_plan_summary", {}), "main_path": (r.get("planned_edge_paths") or [{}])[0], "planned_edges": (r.get("planned_edge_paths") or [])[1:]} for r in results if r.get("hard_generation_success", r.get("success"))])
    save_json(out_dir / "graph_delta_debug.json", [x for r in results for x in r.get("graph_delta_debug", [])])
    save_json(out_dir / "edge_geometry_debug.json", [{"maze_id": r.get("maze_id"), "edges": r.get("edge_geometry_debug", [])} for r in results if r.get("hard_generation_success", r.get("success"))])
    save_json(out_dir / "generation_failure_cases.json", failure_cases)
    save_json(out_dir / "sample_selection_report.json", sample_selection_report)
    save_json(out_dir / "graph_plan_sampling_debug.json", sampling_debug)
    save_json(out_dir / "edge_embedding_debug.json", agg)
    save_json(out_dir / "performance_report.json", [r.get("performance", {}) for r in results])
    save_json(out_dir / "run_summary.json", report)
    save_json(out_dir / "error_direction_hints.json", hints)
    return results, report, hints


def mode_generate(args: argparse.Namespace, cfg: Dict[str, Any], out_dir: Path) -> None:
    results, report, hints = run_generate_like(args, cfg, out_dir)
    print_run_report(args.mode, out_dir, report, hints)


def mode_roundtrip(args: argparse.Namespace, cfg: Dict[str, Any], out_dir: Path) -> None:
    results, report, hints = run_generate_like(args, cfg, out_dir)
    pairs = []
    for r in results:
        if not r.get("success"):
            continue
        pairs.append({"maze_id": r["maze_id"], "planned_graph": r["planned_graph"], "planned_edge_paths": r.get("planned_edge_paths", []), "extracted_metrics": r["extracted_decision_graph"].get("metrics", {}), "consistency_report": r["consistency_report"]})
    save_json(out_dir / "roundtrip_report.json", report)
    save_json(out_dir / "plan_vs_extracted_pairs.json", pairs)
    plot_plan_vs_extracted(pairs, out_dir / "plan_vs_extracted_samples.png")
    print_run_report("roundtrip", out_dir, report, hints)


# =============================================================================
# CLI
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="3.0.2 Edge-Plan + Decision-Graph Maze Generator")
    p.add_argument("--mode", choices=["test", "generate", "roundtrip", "diagnostic"], default="generate")
    p.add_argument("--diagnostic-profile", choices=["main_only", "main_dead_only", "main_alternative_only", "main_alternative_coplanned_only", "main_alternative_equal_length_only", "main_alternative_longer_only", "main_island_only", "roundtrip_layered_main_dead_alt", "roundtrip_layered_main_alt_island"], default="main_only")
    p.add_argument("--n-mazes", type=int, default=None)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--maze-file", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p


def default_config_path() -> Path:
    root = resolve_root()
    p = root / DEFAULT_CONFIG_REL
    if p.exists():
        return p
    local = Path(__file__).resolve().parents[1] / "configs" / "edge_plan_3_0_2_default.json"
    return local


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg_path = Path(args.config) if args.config else default_config_path()
    cfg = load_json(cfg_path)
    if args.seed is not None:
        cfg.setdefault("runtime_limits", {})["seed"] = args.seed
    args.seed = int(cfg.get("runtime_limits", {}).get("seed", 42) if args.seed is None else args.seed)
    if args.n_mazes is None:
        args.n_mazes = int(cfg.get("runtime_limits", {}).get("n_mazes_default", 20))
    out_dir = resolve_out_dir(args)
    ensure_dir(out_dir)
    cfg["_resolved"] = {"config_path": str(cfg_path), "version": VERSION, "experiment_name": EXPERIMENT_NAME, "mode": args.mode, "output_dir": str(out_dir)}
    if args.mode == "test":
        mode_test(args, cfg, out_dir)
    elif args.mode == "generate" or args.mode == "diagnostic":
        mode_generate(args, cfg, out_dir)
    elif args.mode == "roundtrip":
        mode_roundtrip(args, cfg, out_dir)
    else:
        raise SystemExit(f"unknown mode: {args.mode}")



# =============================================================================
# 3.0.2(4) overrides: hint-first alternative, room reports, island search, samples
# =============================================================================

_extract_decision_graph_v3 = extract_decision_graph
_build_maze_from_plan_v3 = build_maze_from_plan
_match_planned_edges_to_extracted_v3 = match_planned_edges_to_extracted
_add_island_component_v3 = add_island_component
_summarize_results_v3 = summarize_results
_error_hints_v3 = error_hints
_select_visualization_samples_v3 = select_visualization_samples
_run_generate_like_v3 = run_generate_like
_mode_test_v3 = mode_test


def _cells_set(path: Optional[Sequence[Any]]) -> Set[Tuple[int, int]]:
    if not path:
        return set()
    return {tuple(x) for x in path}


def _edge_in_main_hint(edge_path: Sequence[Tuple[int, int]], main_path_hint: Optional[Sequence[Tuple[int, int]]]) -> bool:
    if not main_path_hint or len(edge_path) < 2:
        return False
    hint = [tuple(x) for x in main_path_hint]
    idx = {p: i for i, p in enumerate(hint)}
    try:
        pos = [idx[tuple(p)] for p in edge_path]
    except KeyError:
        return False
    return all(pos[i + 1] == pos[i] + 1 or pos[i + 1] == pos[i] - 1 for i in range(len(pos) - 1))


def _planned_alt_records(planned_alternative_paths: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out = []
    for rec in planned_alternative_paths or []:
        if rec.get("edge_type") != "alternative_edge" or not rec.get("accepted"):
            continue
        path = rec.get("actual_cell_path") or rec.get("cell_path") or []
        cells = _cells_set(path)
        anchors = []
        for item in rec.get("expected_choice_anchors") or []:
            cell = item.get("cell") if isinstance(item, dict) else item
            if cell is not None:
                anchors.append(tuple(cell))
        if not anchors and path:
            anchors = [tuple(path[0]), tuple(path[-1])]
        out.append({"record": rec, "cells": cells, "interior": set(list(cells)[1:-1]) if len(cells) > 2 else cells, "anchors": set(anchors)})
    return out


def _build_room_detection_report(graph: Dict[str, Any], maze: np.ndarray, main_path_hint: Optional[Sequence[Tuple[int, int]]], planned_alternative_paths: Optional[Sequence[Dict[str, Any]]]) -> Dict[str, Any]:
    rooms = graph.get("rooms", []) or []
    main_cells = _cells_set(main_path_hint)
    alt_cells = set()
    for rec in planned_alternative_paths or []:
        alt_cells |= _cells_set(rec.get("actual_cell_path") or rec.get("cell_path") or [])
    edges = graph.get("edges", []) or []
    nodes = graph.get("nodes", []) or []
    node_cells = {tuple(c): n.get("node_id") for n in nodes for c in n.get("cells", [])}
    comps = []
    for room in rooms:
        cells = [tuple(x) for x in room.get("cells", [])]
        cell_set = set(cells)
        exit_count = int(room.get("room_exit_count", 0) or 0)
        if exit_count == 0:
            cls = "island_room"
        elif exit_count == 1:
            cls = "endpoint_room"
        elif exit_count == 2:
            cls = "pass_through_room"
        else:
            cls = "choice_room"
        projected_node_ids = sorted({node_cells[c] for c in cell_set if c in node_cells})
        edge_ids = [e.get("edge_id") for e in edges if any(tuple(c) in cell_set for c in e.get("cell_path", []))]
        overlaps_main = bool(cell_set & main_cells)
        overlaps_alt = bool(cell_set & alt_cells)
        suppression = None
        if not projected_node_ids:
            if exit_count == 2:
                suppression = "room_component_exit_count_2"
            elif overlaps_main:
                suppression = "room_suppressed_by_main_path_hint"
            elif overlaps_alt:
                suppression = "room_suppressed_by_planned_alternative_identity"
            else:
                suppression = "room_not_projected"
        comps.append({
            "room_component_id": room.get("room_component_id"),
            "cells": cells,
            "cell_count": len(cells),
            "exit_count": exit_count,
            "outside_component_count": exit_count,
            "classification": cls,
            "projected_as_node": bool(projected_node_ids),
            "projected_node_id": projected_node_ids[0] if projected_node_ids else None,
            "projected_node_ids": projected_node_ids,
            "suppression_reason": suppression,
            "overlaps_main_hint": overlaps_main,
            "overlaps_planned_alternative": overlaps_alt,
            "edge_ids_passing_through": sorted([x for x in edge_ids if x is not None]),
        })
    # Light no-room explanations.  These are diagnostic hints, not proof.
    no_room_reason_top: List[Tuple[str, int]] = []
    if not comps:
        n = maze.shape[0]
        free_block_count = 0
        for r in range(n - 1):
            for c in range(n - 1):
                if int(maze[r, c]) == 0 and int(maze[r + 1, c]) == 0 and int(maze[r, c + 1]) == 0 and int(maze[r + 1, c + 1]) == 0:
                    free_block_count += 1
        if free_block_count == 0:
            no_room_reason_top.append(("no_2x2_free_block", 1))
        else:
            no_room_reason_top.append(("component_below_min_size_or_suppressed", free_block_count))
    return {
        "room_detection_enabled": True,
        "room_candidate_cell_count": len({tuple(c) for room in rooms for c in room.get("cells", [])}),
        "room_component_count": len(comps),
        "room_projected_node_count": sum(1 for c in comps if c.get("projected_as_node")),
        "pass_through_room_count": sum(1 for c in comps if c.get("classification") == "pass_through_room"),
        "choice_room_count": sum(1 for c in comps if c.get("classification") == "choice_room"),
        "room_suppressed_by_hint_count": sum(1 for c in comps if c.get("suppression_reason") == "room_suppressed_by_main_path_hint"),
        "room_suppressed_by_alternative_count": sum(1 for c in comps if c.get("suppression_reason") == "room_suppressed_by_planned_alternative_identity"),
        "room_visualized_count": len(comps),
        "large_open_area_detected_by_free_blocks": 0 if comps else (no_room_reason_top[0][1] if no_room_reason_top else 0),
        "room_components": comps,
        "no_room_reason_top": no_room_reason_top,
    }


def extract_decision_graph(maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], cfg: Dict[str, Any], main_path_hint: Optional[Sequence[Tuple[int, int]]] = None, planned_alternative_paths: Optional[Sequence[Dict[str, Any]]] = None, expected_choice_anchors: Optional[Sequence[Tuple[int, int]]] = None) -> Dict[str, Any]:
    graph = _extract_decision_graph_v3(maze, start, goal, cfg, main_path_hint=main_path_hint)
    alt_records = _planned_alt_records(planned_alternative_paths)
    main_cells = _cells_set(main_path_hint)
    expected_anchor_cells = set(tuple(x) for x in (expected_choice_anchors or []))
    for ar in alt_records:
        expected_anchor_cells |= ar["anchors"]
    alt_cells = set()
    for ar in alt_records:
        alt_cells |= ar["cells"]
    identity_conflicts = sorted([c for c in (main_cells & alt_cells) if c not in expected_anchor_cells])
    threshold = float(cfg.get("compare", {}).get("alternative_cell_coverage_threshold", 0.8))
    # Hint-first edge identity: main exact path first; otherwise planned alternative cells win.
    for e in graph.get("edges", []):
        path = [tuple(x) for x in e.get("cell_path", [])]
        path_set = set(path)
        if _edge_in_main_hint(path, main_path_hint):
            e["edge_type"] = "main_edge"
            e["identity_source"] = "main_path_hint"
            continue
        best = None
        best_cov = 0.0
        best_anchor = False
        for ar in alt_records:
            alt_core = ar["cells"]
            if not alt_core:
                continue
            overlap = len(path_set & alt_core)
            cov = overlap / max(1, len(alt_core))
            edge_endpoints = {tuple(path[0]), tuple(path[-1])} if path else set()
            anchor_match = len(edge_endpoints & ar["anchors"]) >= 1 or bool(path_set & ar["anchors"])
            # For room-split alternative, one extracted edge may cover a portion only.
            if cov > best_cov or (cov == best_cov and anchor_match):
                best = ar
                best_cov = cov
                best_anchor = anchor_match
        if best is not None and (best_cov >= threshold or (best_cov > 0 and best_anchor)):
            e["edge_type"] = "alternative_edge"
            e["identity_source"] = "planned_alternative_hint"
            e["planned_alternative_cell_coverage"] = best_cov
            e["planned_alternative_anchor_match"] = bool(best_anchor)
    # If room folding swallowed the planned alternative into a room node, preserve
    # the planned edge identity as an extracted alternative edge.  This is a
    # hint-first diagnostic representation, not a new generation algorithm.
    node_by_cell = {}
    for nrec in graph.get("nodes", []):
        for c in nrec.get("cells", []):
            node_by_cell[tuple(c)] = nrec.get("node_id")
    for ar in alt_records:
        alt_cells = ar["cells"]
        if not alt_cells:
            continue
        already = set()
        for e in graph.get("edges", []):
            if e.get("edge_type") == "alternative_edge":
                already |= (_cells_set(e.get("cell_path") or []) & alt_cells)
        coverage = len(already) / max(1, len(alt_cells))
        if coverage >= threshold:
            continue
        rec = ar["record"]
        path = [tuple(x) for x in (rec.get("actual_cell_path") or rec.get("cell_path") or [])]
        if len(path) < 2:
            continue
        src = node_by_cell.get(tuple(path[0]), "hint_alt_source")
        tgt = node_by_cell.get(tuple(path[-1]), "hint_alt_target")
        L = len(path) - 1
        D_path = manhattan(tuple(path[0]), tuple(path[-1]))
        D_anchor = D_path
        graph.setdefault("edges", []).append({
            "edge_id": f"e_hint_alt_{len(graph.get('edges', []))}",
            "edge_type": "alternative_edge",
            "source": src,
            "target": tgt,
            "cell_path": path,
            "path_start": tuple(path[0]),
            "path_end": tuple(path[-1]),
            "source_anchor": tuple(path[0]),
            "target_anchor": tuple(path[-1]),
            "L": L,
            "D": D_path,
            "detour": None if D_path == 0 else L / D_path,
            "D_path": D_path,
            "detour_path": None if D_path == 0 else L / D_path,
            "D_anchor": D_anchor,
            "detour_anchor": None if D_anchor == 0 else L / D_anchor,
            "anchor_geometry_mismatch": False,
            "extraction_geometry_bug": D_path == 0 or L < D_path,
            "invalid_or_room_internal_edge": D_path == 0,
            "passes_through_room": True,
            "room_component_ids": [],
            "identity_source": "planned_alternative_hint_synthetic_after_room_folding",
            "planned_alternative_cell_coverage": 1.0,
            "planned_alternative_anchor_match": True,
            "synthetic_hint_edge": True,
        })
    # Recompute metrics after reclassification.
    edge_counts = Counter(e.get("edge_type") for e in graph.get("edges", []))
    node_counts = Counter(n.get("node_type") for n in graph.get("nodes", []))
    metrics = dict(graph.get("metrics", {}))
    metrics.update({
        "choice_node_count": node_counts.get("choice_node", 0),
        "endpoint_node_count": node_counts.get("endpoint_node", 0),
        "island_node_count": node_counts.get("island_node", 0),
        "main_edge_count": edge_counts.get("main_edge", 0),
        "dead_edge_count": edge_counts.get("dead_edge", 0),
        "alternative_edge_count": edge_counts.get("alternative_edge", 0),
        "island_edge_count": edge_counts.get("island_edge", 0),
        "main_route_total_L": sum(e.get("L", 0) for e in graph.get("edges", []) if e.get("edge_type") == "main_edge"),
        "extracted_main_edge_sum_L": sum(e.get("L", 0) for e in graph.get("edges", []) if e.get("edge_type") == "main_edge"),
        "planned_path_identity_conflict_count": len(identity_conflicts),
    })
    graph["metrics"] = metrics
    graph.setdefault("debug_extraction_report", {})["planned_path_identity_conflicts"] = identity_conflicts
    graph.setdefault("debug_extraction_report", {})["planned_path_identity_conflict_count"] = len(identity_conflicts)
    graph["room_detection_report"] = _build_room_detection_report(graph, maze, main_path_hint, planned_alternative_paths)
    return graph


def match_planned_edges_to_extracted(planned_paths: List[Dict[str, Any]], extracted: Dict[str, Any]) -> None:
    threshold = 0.8
    edges = list(extracted.get("edges", []))
    for pe in planned_paths:
        if pe.get("edge_type") == "main_edge" or not pe.get("accepted"):
            pe.setdefault("matched_extracted_edge_id", None)
            pe.setdefault("matched_extracted_edge_type", None)
            continue
        if pe.get("edge_type") == "island_edge":
            cells = _cells_set(pe.get("cells") or pe.get("actual_cell_path") or [])
            match_id = None
            for comp in extracted.get("island_components", []) or []:
                comp_cells = _cells_set(comp.get("cells") or [])
                if cells and len(cells & comp_cells) / max(1, len(cells)) >= threshold:
                    match_id = comp.get("component_id")
                    break
            pe["matched_extracted_component_id"] = match_id
            continue
        if pe.get("edge_type") == "alternative_edge":
            p_cells = _cells_set(pe.get("actual_cell_path") or pe.get("cell_path") or [])
            anchors = set()
            for item in pe.get("expected_choice_anchors") or []:
                cell = item.get("cell") if isinstance(item, dict) else item
                if cell is not None:
                    anchors.add(tuple(cell))
            if not anchors and pe.get("actual_cell_path"):
                path = pe.get("actual_cell_path")
                anchors = {tuple(path[0]), tuple(path[-1])}
            matched = []
            covered = set()
            anchor_hit = False
            for e in edges:
                e_cells = _cells_set(e.get("cell_path") or [])
                overlap = p_cells & e_cells
                if not overlap:
                    continue
                # Avoid counting pure hinted main pieces unless they have alternative identity.
                if e.get("edge_type") == "main_edge" and e.get("identity_source") != "planned_alternative_hint":
                    continue
                matched.append(e)
                covered |= overlap
                endpoints = set()
                if e.get("cell_path"):
                    endpoints = {tuple(e["cell_path"][0]), tuple(e["cell_path"][-1])}
                anchor_hit = anchor_hit or bool((endpoints | e_cells) & anchors)
            coverage = len(covered) / max(1, len(p_cells))
            types = [e.get("edge_type") for e in matched]
            pe["matched_extracted_edge_ids"] = [e.get("edge_id") for e in matched]
            pe["matched_extracted_edge_types"] = types
            pe["matched_extracted_edge_id"] = pe["matched_extracted_edge_ids"][0] if matched else None
            pe["matched_extracted_edge_type"] = types[0] if types else None
            pe["alternative_cell_coverage"] = coverage
            pe["anchor_match"] = bool(anchor_hit)
            pe["matched_as_multi_edge"] = len(matched) > 1
            pe["matched_extracted_edge_type_mismatch"] = bool(matched and "alternative_edge" not in types)
            pe["alternative_planned_edge_match"] = bool(pe.get("accepted") and not pe.get("alternative_shortcut_violation") and anchor_hit and coverage >= threshold and "alternative_edge" in types)
            continue
    # Fallback for dead edges remains the v3 matcher.
    dead_only = [p for p in planned_paths if p.get("edge_type") == "dead_edge"]
    if dead_only:
        _match_planned_edges_to_extracted_v3(planned_paths, extracted)


def add_island_component(maze: np.ndarray, free_budget: int, cfg: Dict[str, Any], rng: random.Random, debug: Dict[str, Any]) -> Tuple[bool, str, List[Tuple[int, int]]]:
    if free_budget <= 0:
        debug["edge_reject_reason_counts"]["skipped_zero_budget_island"] += 1
        return False, "skipped_zero_budget_island", []
    n = maze.shape[0]
    max_attempts = int(cfg.get("island", {}).get("max_island_attempts", 50))
    min_cells = int(cfg.get("island", {}).get("min_component_cells", 1))
    allow_single = bool(cfg.get("island", {}).get("allow_single_cell_island", True))
    main_free = set(free_cells(maze))
    # Candidate walls that do not touch the current main component are safer island seeds.
    for attempt in range(max_attempts):
        debug["edge_attempt_count"] += 1
        walls = [(r, c) for r in range(n) for c in range(n) if int(maze[r, c]) == 1]
        candidates = [p for p in walls if all(int(maze[nb]) == 1 for nb in neighbors(p, n))]
        if not candidates:
            candidates = walls
        if not candidates:
            debug["edge_reject_reason_counts"]["island_no_wall_space"] += 1
            return False, "island_no_wall_space", []
        seed = rng.choice(candidates)
        comp = [seed]
        used = {seed}
        while len(comp) < free_budget:
            frontier = []
            for p in comp:
                for q in neighbors(p, n):
                    if q in used or int(maze[q]) != 1:
                        continue
                    # Do not touch existing free cells except cells we are carving.
                    if all((int(maze[nb]) == 1 or nb in used) for nb in neighbors(q, n)):
                        frontier.append(q)
            if not frontier:
                break
            q = rng.choice(frontier)
            used.add(q)
            comp.append(q)
        if len(comp) < min_cells or (len(comp) == 1 and not allow_single):
            debug["edge_reject_reason_counts"]["island_no_cells_generated"] += 1
            continue
        # Verify this carved component will not connect to current free graph.
        if any(nb in main_free for cell in comp for nb in neighbors(cell, n)):
            debug["edge_reject_reason_counts"]["island_cells_touch_main_component"] += 1
            continue
        backup = maze.copy()
        apply_edge_path(maze, comp)
        reachable = component_id_by_start(maze, next(iter(main_free)) if main_free else comp[0]) if main_free else set()
        if any(c in reachable for c in comp):
            maze[:, :] = backup
            debug["edge_reject_reason_counts"]["island_cells_connected_to_main_component"] += 1
            continue
        return True, "ok", comp
    debug["edge_reject_reason_counts"]["island_no_cells_generated"] += 1
    return False, "island_no_cells_generated", []


def build_maze_from_plan(plan: Dict[str, Any], cfg: Dict[str, Any], rng: random.Random, maze_id: str = "maze_00000") -> Dict[str, Any]:
    # Force all alternatives to co-planned mode in 3.0.2(4).
    if plan.get("alternative_planning_mode") == "post_main_baseline" or cfg.get("alternative_edge", {}).get("planning_mode") == "post_main_baseline":
        print("[WARN] baseline alternative config is ignored in 3.0.2(4); using co_planned_with_main.")
    plan = json.loads(json.dumps(json_safe(plan)))
    plan["alternative_planning_mode"] = "co_planned_with_main"
    for pe in plan.get("planned_edges", []):
        if pe.get("edge_type") == "alternative_edge":
            pe["planning_mode"] = "co_planned_with_main"
    result = _build_maze_from_plan_v3(plan, cfg, rng, maze_id=maze_id)
    if not result.get("hard_generation_success", result.get("success")):
        return result
    # Re-extract with explicit planned alternative identity.
    planned_paths = result.get("planned_edge_paths", [])
    alt_paths = [p for p in planned_paths if p.get("edge_type") == "alternative_edge" and p.get("accepted")]
    expected_anchors = []
    for pe in alt_paths:
        for item in pe.get("expected_choice_anchors") or []:
            cell = item.get("cell") if isinstance(item, dict) else item
            if cell is not None:
                expected_anchors.append(tuple(cell))
    extracted = extract_decision_graph(result["maze"], tuple(result["start"]), tuple(result["goal"]), cfg, main_path_hint=result.get("main_path"), planned_alternative_paths=alt_paths, expected_choice_anchors=expected_anchors)
    match_planned_edges_to_extracted(planned_paths, extracted)
    comp = compare_plan_extracted(result.get("planned_graph", plan), extracted, planned_paths)
    comp["maze_id"] = maze_id
    result["extracted_decision_graph"] = extracted
    result["planned_edge_paths"] = planned_paths
    result["consistency_report"] = comp
    result["edge_embedding_success"] = bool(comp.get("edge_embedding_success"))
    result["strict_consistency_success"] = bool(comp.get("strict_consistency_success"))
    result["room_detection_report"] = extracted.get("room_detection_report", {})
    result["edge_geometry_debug"] = [{
        "edge_id": e.get("edge_id"), "edge_type": e.get("edge_type"), "identity_source": e.get("identity_source"),
        "L": e.get("L"), "D_path": e.get("D_path"), "detour_path": e.get("detour_path"),
        "D_anchor": e.get("D_anchor"), "detour_anchor": e.get("detour_anchor"),
        "anchor_geometry_mismatch": e.get("anchor_geometry_mismatch"), "extraction_geometry_bug": e.get("extraction_geometry_bug"),
    } for e in extracted.get("edges", [])]
    return result


_compare_plan_extracted_v3 = compare_plan_extracted

def compare_plan_extracted(plan: Dict[str, Any], extracted: Dict[str, Any], planned_paths: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    # Start from v3 report and add planned-edge matching fields.
    rep = globals().get('_compare_plan_extracted_v3')
    if rep is None:
        # Store original lazily if this block was loaded after original definition.
        pass
    base = _compare_plan_extracted_v3(plan, extracted, planned_paths) if '_compare_plan_extracted_v3' in globals() else None
    if base is None:
        # This branch should not happen, but keeps the override safe.
        base = {}
    planned_paths = planned_paths or []
    alt_paths = [p for p in planned_paths if p.get("edge_type") == "alternative_edge"]
    accepted_alt = [p for p in alt_paths if p.get("accepted")]
    alt_match_count = sum(1 for p in accepted_alt if p.get("alternative_planned_edge_match"))
    covs = [p.get("alternative_cell_coverage") for p in accepted_alt if p.get("alternative_cell_coverage") is not None]
    anchor_match_count = sum(1 for p in accepted_alt if p.get("anchor_match"))
    type_match_count = sum(1 for p in accepted_alt if "alternative_edge" in (p.get("matched_extracted_edge_types") or []))
    multi_count = sum(1 for p in accepted_alt if p.get("matched_as_multi_edge"))
    alt_checks = dict(base.get("alternative_checks", {}))
    alt_checks.update({
        "alternative_planned_edge_match_count": alt_match_count,
        "alternative_planned_edge_match_rate": alt_match_count / max(1, len(accepted_alt)),
        "alternative_cell_coverage_mean": float(np.mean(covs)) if covs else None,
        "alternative_anchor_match_rate": anchor_match_count / max(1, len(accepted_alt)),
        "alternative_type_match_rate": type_match_count / max(1, len(accepted_alt)),
        "alternative_multi_edge_match_rate": multi_count / max(1, len(accepted_alt)),
        "matched_extracted_edge_type_mismatch_count": sum(1 for p in accepted_alt if p.get("matched_extracted_edge_type_mismatch")),
    })
    base["alternative_checks"] = alt_checks
    # For strict consistency, use planned-edge match instead of raw global count only.
    if accepted_alt and alt_match_count < len(accepted_alt):
        base.setdefault("strict_failure_reasons", []).append("alternative_planned_edge_not_matched")
        base["strict_consistency_success"] = False
        base["effective_plan_match"] = False
    return base


# Save original compare before overriding it above. This assignment is intentionally
# after function body creation in source order via manual patch below.



def summarize_results(results: List[Dict[str, Any]], requested: int) -> Dict[str, Any]:
    report = _summarize_results_v3(results, requested)
    hard = [r for r in results if r.get("hard_generation_success", r.get("success"))]
    comps = [r.get("consistency_report", {}) for r in hard]
    alt_checks = [c.get("alternative_checks", {}) for c in comps]
    room_reports = [r.get("room_detection_report") or r.get("extracted_decision_graph", {}).get("room_detection_report", {}) for r in hard]
    paths = [pe for r in hard for pe in r.get("planned_edge_paths", [])[1:]]
    islands = [pe for pe in paths if pe.get("edge_type") == "island_edge"]
    island_attempts = [pe.get("attempt_count") for pe in islands if pe.get("attempt_count") is not None]
    accepted_alt = [pe for pe in paths if pe.get("edge_type") == "alternative_edge" and pe.get("accepted")]
    coverages = [pe.get("alternative_cell_coverage") for pe in accepted_alt if pe.get("alternative_cell_coverage") is not None]
    report["alternative_planned_edge_matching"] = {
        "accepted_alternative_count": len(accepted_alt),
        "alternative_planned_edge_match_rate": sum(1 for pe in accepted_alt if pe.get("alternative_planned_edge_match")) / max(1, len(accepted_alt)),
        "alternative_cell_coverage_mean": float(np.mean(coverages)) if coverages else None,
        "alternative_anchor_match_rate": sum(1 for pe in accepted_alt if pe.get("anchor_match")) / max(1, len(accepted_alt)),
        "alternative_type_match_rate": sum(1 for pe in accepted_alt if "alternative_edge" in (pe.get("matched_extracted_edge_types") or [])) / max(1, len(accepted_alt)),
        "alternative_multi_edge_match_rate": sum(1 for pe in accepted_alt if pe.get("matched_as_multi_edge")) / max(1, len(accepted_alt)),
        "matched_extracted_edge_type_mismatch_count": sum(1 for pe in accepted_alt if pe.get("matched_extracted_edge_type_mismatch")),
    }
    report["room_detection"] = {
        "room_detection_enabled": True,
        "room_component_count": stats([rr.get("room_component_count") for rr in room_reports]),
        "room_projected_node_count": stats([rr.get("room_projected_node_count") for rr in room_reports]),
        "pass_through_room_count": stats([rr.get("pass_through_room_count") for rr in room_reports]),
        "choice_room_count": stats([rr.get("choice_room_count") for rr in room_reports]),
        "room_suppressed_by_hint_count": sum(rr.get("room_suppressed_by_hint_count", 0) or 0 for rr in room_reports),
        "room_suppressed_by_alternative_count": sum(rr.get("room_suppressed_by_alternative_count", 0) or 0 for rr in room_reports),
        "room_visualized_count": sum(rr.get("room_visualized_count", 0) or 0 for rr in room_reports),
        "large_open_area_detected_by_free_blocks": sum(rr.get("large_open_area_detected_by_free_blocks", 0) or 0 for rr in room_reports),
    }
    report.setdefault("island_planning", {}).update({
        "island_no_cells_generated_count": sum(1 for pe in islands if pe.get("reject_reason") == "island_no_cells_generated"),
        "island_cells_connected_to_main_component_count": sum(1 for pe in islands if pe.get("reject_reason") == "island_cells_connected_to_main_component"),
        "island_cells_touch_main_component_count": sum(1 for pe in islands if pe.get("reject_reason") == "island_cells_touch_main_component"),
        "island_attempt_count": stats(island_attempts),
    })
    return report


def _success_categories_for_result(r: Dict[str, Any]) -> List[Tuple[str, str, Dict[str, Any]]]:
    comp = r.get("consistency_report", {})
    paths = r.get("planned_edge_paths", [])[1:]
    room = r.get("room_detection_report") or r.get("extracted_decision_graph", {}).get("room_detection_report", {})
    cats: List[Tuple[str, str, Dict[str, Any]]] = []
    final_ok = (comp.get("final_bfs_L_error") in (0, 0.0, None))
    dead = [p for p in paths if p.get("edge_type") == "dead_edge" and p.get("accepted")]
    alt = [p for p in paths if p.get("edge_type") == "alternative_edge" and p.get("accepted")]
    isl = [p for p in paths if p.get("edge_type") == "island_edge" and p.get("accepted")]
    if comp.get("strict_consistency_success") and not paths:
        cats.append(("success_main_only", "strict main-only consistency", {"final_bfs_L_error": comp.get("final_bfs_L_error")}))
    if dead and comp.get("checks", {}).get("dead_edge_count_match_effective"):
        cats.append(("success_main_dead", "accepted dead edge matched extracted dead edge", {"dead_edge_count": len(dead)}))
    for p in alt:
        if p.get("alternative_planned_edge_match") and p.get("alternative_equal_length"):
            cats.append(("success_alt_equal_length", "accepted alternative matched with equal length and no shortcut", {"alternative_cell_coverage": p.get("alternative_cell_coverage"), "alternative_equal_length": True}))
        if p.get("alternative_planned_edge_match") and not p.get("alternative_equal_length"):
            cats.append(("success_alt_longer", "accepted longer alternative matched as alternative_edge", {"alternative_cell_coverage": p.get("alternative_cell_coverage"), "alternative_equal_length": False}))
    if isl and comp.get("island_checks", {}).get("island_component_count_match"):
        cats.append(("success_island_component", "accepted island component matched extracted island component", {"island_count": len(isl)}))
    if dead and alt and any(p.get("alternative_planned_edge_match") for p in alt):
        cats.append(("success_main_dead_alt", "main + dead + alternative matched", {}))
    if alt and isl and any(p.get("alternative_planned_edge_match") for p in alt) and comp.get("island_checks", {}).get("island_component_count_match"):
        cats.append(("success_main_alt_island", "main + alternative + island matched", {}))
    if room.get("pass_through_room_count", 0):
        cats.append(("success_room_pass_through", "room component detected as pass-through and visualized", {"pass_through_room_count": room.get("pass_through_room_count")}))
    if room.get("choice_room_count", 0):
        cats.append(("success_room_choice_node", "choice-like room detected/projected", {"choice_room_count": room.get("choice_room_count")}))
    if comp.get("effective_plan_match"):
        cats.append(("success_mixed_effective_match", "effective plan matched extracted graph", {}))
    return cats


def categorize_result_for_sample(result: Dict[str, Any], mode: str, diagnostic_profile: Optional[str]) -> List[str]:
    cats = [c for c, _, _ in _success_categories_for_result(result)]
    comp = result.get("consistency_report", {})
    paths = result.get("planned_edge_paths", [])[1:]
    if any(p.get("edge_type") == "alternative_edge" and p.get("accepted") and not p.get("alternative_planned_edge_match") for p in paths):
        cats.append("failure_alt_identity_mismatch")
    if any(p.get("edge_type") == "island_edge" and not p.get("accepted") for p in paths):
        cats.append("failure_island_candidate")
    if not comp.get("effective_plan_match"):
        cats.append("failure_effective_mismatch")
    if comp.get("extracted_main_edge_sum_L_error") not in (0, 0.0, None) and comp.get("final_bfs_L_error") in (0, 0.0, None):
        cats.append("failure_extract_representation")
    return cats


def select_visualization_samples(results: List[Dict[str, Any]], cfg: Dict[str, Any], mode: str, diagnostic_profile: Optional[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    target_success = [
        "success_main_only", "success_main_dead", "success_alt_equal_length", "success_alt_longer", "success_island_component",
        "success_main_dead_alt", "success_main_alt_island", "success_room_pass_through", "success_room_choice_node", "success_mixed_effective_match",
    ]
    target_failure = ["failure_alt_identity_mismatch", "failure_island_candidate", "failure_effective_mismatch", "failure_extract_representation"]
    max_samples = int(cfg.get("sample_selection", {}).get("max_samples", 16))
    selected: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    used: Set[str] = set()
    available_success = defaultdict(list)
    for r in results:
        if not r.get("hard_generation_success", r.get("success")):
            continue
        for cat, why, metrics in _success_categories_for_result(r):
            available_success[cat].append((r, why, metrics))
    def add(cat: str, r: Dict[str, Any], why: str, metrics: Dict[str, Any]) -> None:
        if len(selected) >= max_samples or r.get("maze_id") in used:
            return
        used.add(r.get("maze_id"))
        idx = len(selected) + 1
        selected.append({"category": cat, "index": idx, "result": r})
        rows.append({"rank": idx, "category": cat, "maze_id": r.get("maze_id"), "why_selected": why, "key_metrics": metrics, "files": {"maze_overlay": f"maze_overlay_samples/{idx:02d}_{cat}__{r.get('maze_id')}.png", "decision_graph": f"decision_graph_samples/{idx:02d}_{cat}__{r.get('maze_id')}.png"}})
    for cat in target_success:
        vals = available_success.get(cat, [])
        if vals:
            r, why, metrics = vals[0]
            add(cat, r, why, metrics)
    for cat in target_failure:
        for r in results:
            if not r.get("hard_generation_success", r.get("success")) or r.get("maze_id") in used:
                continue
            if cat in categorize_result_for_sample(r, mode, diagnostic_profile):
                add(cat, r, f"selected failure/mismatch category {cat}", {})
                break
    # Fill with any unused examples.
    for r in results:
        if len(selected) >= max_samples:
            break
        if r.get("hard_generation_success", r.get("success")) and r.get("maze_id") not in used:
            cat = "fallback_success" if r.get("strict_consistency_success") else "fallback_mismatch"
            add(cat, r, f"fallback {cat}", {})
    success_selected = [row["category"] for row in rows if row["category"].startswith("success_")]
    available_cats = sorted(available_success.keys())
    missing = [c for c in target_success if c not in available_cats]
    report = {
        "success_sample_count": len(success_selected),
        "success_category_count": len(set(success_selected)),
        "success_categories_available": available_cats,
        "success_categories_selected": success_selected,
        "success_sample_shortage": len(set(success_selected)) < 5,
        "missing_success_categories": missing,
        "selected_samples": rows,
    }
    return selected, report


def plot_maze_overlay(maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], graph: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(maze, cmap="gray_r", vmin=0, vmax=1)
    # Draw room overlays first so nodes/edges remain visible.
    for room in graph.get("room_detection_report", {}).get("room_components", []):
        cells = [tuple(x) for x in room.get("cells", [])]
        if not cells:
            continue
        ys = [p[0] for p in cells]; xs = [p[1] for p in cells]
        ax.scatter(xs, ys, marker="s", s=260, alpha=0.18)
        ar = cells[len(cells)//2]
        label = room.get("classification", "room").replace("_room", "")
        if room.get("suppression_reason"):
            label = "room_suppressed"
        ax.text(ar[1]-0.25, ar[0]+0.25, label, fontsize=6)
    ax.scatter([start[1]], [start[0]], marker="o", s=160, label="start")
    ax.scatter([goal[1]], [goal[0]], marker="*", s=220, label="goal")
    markers = {"choice_node": "s", "endpoint_node": "x", "island_node": "D"}
    for node in graph.get("nodes", []):
        r, c = node["anchor"]
        ax.scatter([c], [r], marker=markers.get(node["node_type"], "o"), s=90)
        label = f"{node['node_id']}\n{node['node_type'].replace('_node','')}"
        if node.get("room_component"):
            label += "\nroom=true"
        if node.get("is_start"):
            label += "\nS"
        if node.get("is_goal"):
            label += "\nG"
        ax.text(c + 0.05, r + 0.05, label, fontsize=7)
    linestyles = {"main_edge": "-", "dead_edge": "--", "alternative_edge": ":", "island_edge": "-."}
    for edge in graph.get("edges", []):
        pts = edge.get("cell_path", [])
        if len(pts) < 2:
            continue
        ys = [p[0] for p in pts]; xs = [p[1] for p in pts]
        ax.plot(xs, ys, linestyle=linestyles.get(edge["edge_type"], "-"), linewidth=2)
        mid = pts[len(pts)//2]
        room_tag = "\npasses_room=true" if edge.get("passes_through_room") else ""
        ax.text(mid[1] + 0.05, mid[0] - 0.1, f"{edge['edge_id']}\n{edge['edge_type']}\nL={edge['L']} D={edge['D']}{room_tag}", fontsize=6)
    ax.set_xticks(range(maze.shape[1])); ax.set_yticks(range(maze.shape[0]))
    ax.grid(True, linewidth=0.4)
    ax.set_title("maze overlay: nodes, edges, room components")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def run_generate_like(args: argparse.Namespace, cfg: Dict[str, Any], out_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    # Execute v3 pipeline with overridden functions, then add 3.0.2(4) reports.
    results, report, hints = _run_generate_like_v3(args, cfg, out_dir)
    room_reports = [{"maze_id": r.get("maze_id"), **(r.get("room_detection_report") or r.get("extracted_decision_graph", {}).get("room_detection_report", {}))} for r in results if r.get("hard_generation_success", r.get("success"))]
    save_json(out_dir / "room_detection_report.json", room_reports)
    # Regenerate summary/hints after extra reports are available.
    report = summarize_results(results, args.n_mazes)
    # Merge sample selection coverage into summary if saved.
    ss_path = out_dir / "sample_selection_report.json"
    if ss_path.exists():
        try:
            ss = load_json(ss_path)
            if isinstance(ss, dict):
                report["success_sample_coverage"] = {k: ss.get(k) for k in ["success_sample_count", "success_category_count", "success_categories_available", "success_categories_selected", "success_sample_shortage", "missing_success_categories"]}
        except Exception:
            pass
    agg = aggregate_debugs(results)
    hints = error_hints(agg, report, cfg)
    save_json(out_dir / "run_summary.json", report)
    save_json(out_dir / "error_direction_hints.json", hints)
    return results, report, hints


def error_hints(aggregate_debug: Dict[str, Any], report: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    hints = _error_hints_v3(aggregate_debug, report, cfg)
    ss = report.get("success_sample_coverage", {})
    if ss.get("success_category_count", 0) < 5:
        hints.append({"hint": "Fewer than 5 success categories were selected.", "likely_direction": ["Some profiles do not yet produce enough successful cases."], "suggested_next_json": ["sample_selection_report.json"], "suggested_config_fields": ["diagnostic_profiles", "sample_selection.categories"]})
    rd = report.get("room_detection", {})
    room_mean = (rd.get("room_component_count") or {}).get("mean")
    visualized = rd.get("room_visualized_count", 0) or 0
    if room_mean and room_mean > 0 and visualized == 0:
        hints.append({"hint": "Room components were detected but not visualized.", "likely_direction": ["Visualization does not draw room overlays."], "suggested_next_json": ["room_detection_report.json", "maze_overlay_samples/"], "suggested_config_fields": ["visualization.enabled"]})
    if (room_mean in (0, 0.0, None)) and (rd.get("large_open_area_detected_by_free_blocks", 0) or 0) > 0:
        hints.append({"hint": "Large open free areas exist but no room component was detected.", "likely_direction": ["Room detection thresholds may be too strict or disabled."], "suggested_next_json": ["room_detection_report.json"], "suggested_config_fields": ["room_detection.use_2x2_free_block", "room_detection.room_min_component_size", "room_detection.room_high_degree_threshold"]})
    alt = report.get("alternative_planned_edge_matching", {})
    if alt.get("accepted_alternative_count", 0) > 0 and alt.get("matched_extracted_edge_type_mismatch_count", 0) > 0:
        hints.append({"hint": "Accepted alternative was matched as non-alternative edge.", "likely_direction": ["Planned path identity is being lost during extraction or room folding."], "suggested_next_json": ["planned_edge_paths.json", "room_detection_report.json", "extracted_decision_graphs.json"], "suggested_config_fields": ["compare.alternative_cell_coverage_threshold", "room_detection.room_min_component_size"]})
    ip = report.get("island_planning", {})
    if ip.get("island_no_cells_generated_count", 0) > 0:
        hints.append({"hint": "Island candidate generation is weak.", "likely_direction": ["Current island placement fails before extraction."], "suggested_next_json": ["planned_edge_paths.json", "graph_delta_debug.json"], "suggested_config_fields": ["island.max_island_attempts", "island.candidate_strategy", "island.free_budget_max"]})
    return hints




_apply_diagnostic_profile_to_plan_v3 = apply_diagnostic_profile_to_plan

def apply_diagnostic_profile_to_plan(plan: Dict[str, Any], profile: Optional[str], cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if profile in ("main_alternative_equal_length_only", "main_alternative_longer_only", "roundtrip_layered_main_dead_alt", "roundtrip_layered_main_alt_island"):
        plan = json.loads(json.dumps(json_safe(plan)))
        prof_cfg = (cfg or {}).get("diagnostic_profiles", {}).get(profile, {})
        if profile == "main_alternative_equal_length_only":
            dead_n, alt_n, island_n, alt_extra = 0, 1, 0, int(prof_cfg.get("alternative_extra_L", 0))
        elif profile == "main_alternative_longer_only":
            dead_n, alt_n, island_n, alt_extra = 0, 1, 0, int(prof_cfg.get("alternative_extra_L", 2))
        elif profile == "roundtrip_layered_main_dead_alt":
            dead_n, alt_n, island_n, alt_extra = 1, 1, 0, int(prof_cfg.get("alternative_extra_L", 2))
        else:
            dead_n, alt_n, island_n, alt_extra = 0, 1, 1, int(prof_cfg.get("alternative_extra_L", 2))
        dead_n = int(prof_cfg.get("dead_edge_count", dead_n)); alt_n = int(prof_cfg.get("alternative_edge_count", alt_n)); island_n = int(prof_cfg.get("island_component_count", island_n))
        dead_edges = [{"edge_id": f"planned_dead_{i}", "edge_type": "dead_edge", "L": int(prof_cfg.get("dead_L", 4))} for i in range(dead_n)]
        alt_edges = [{"edge_id": f"planned_alternative_{i}", "edge_type": "alternative_edge", "extra_L": alt_extra, "planning_mode": "co_planned_with_main", "target_extra_L": alt_extra} for i in range(alt_n)]
        island_budget = int(prof_cfg.get("island_free_budget", 4))
        islands = [{"component_id": f"planned_island_{i}", "edge_type": "island_edge", "free_budget": island_budget} for i in range(island_n)]
        plan.update({
            "dead_edge_count": dead_n,
            "alternative_edge_count": alt_n,
            "island_component_count": island_n,
            "graph_budget": dead_n + alt_n + island_n,
            "profile": f"diagnostic_{profile}",
            "alternative_planning_mode": "co_planned_with_main",
            "planned_edges": [{"edge_id": "planned_main", "edge_type": "main_edge", "L": plan["main_L"], "D": plan["main_D"], "detour": plan["main_L"] / max(1, plan["main_D"])}] + dead_edges + alt_edges,
            "planned_islands": islands,
        })
        return plan
    # main_alternative_only is co-planned in 3.0.2(4), and existing v3 handles it that way.
    return _apply_diagnostic_profile_to_plan_v3(plan, profile, cfg)

_print_run_report_v3 = print_run_report

def print_run_report(mode: str, out_dir: Path, report: Dict[str, Any], hints: List[Dict[str, Any]]) -> None:
    # Use v3 printer then append 3.0.2(4) sections.
    _print_run_report_v3(mode, out_dir, report, hints)
    sc = report.get("success_sample_coverage", {})
    print_kv("\n=== Success Sample Coverage ===", [(k, sc.get(k)) for k in ["success_sample_count", "success_category_count", "success_categories_available", "success_categories_selected", "success_sample_shortage"]])
    rd = report.get("room_detection", {})
    print("\n=== Room Detection ===")
    for k in ["room_component_count", "room_projected_node_count", "pass_through_room_count", "choice_room_count"]:
        s = rd.get(k, {})
        print(f"  {k:<36} mean={fmt_num(s.get('mean'),8,3)} p50={fmt_num(s.get('p50'),8,3)}")
    print(f"  {'room_suppressed_by_hint_count':<36} {rd.get('room_suppressed_by_hint_count')}")
    print(f"  {'room_suppressed_by_alternative_count':<36} {rd.get('room_suppressed_by_alternative_count')}")
    alt = report.get("alternative_planned_edge_matching", {})
    print_kv("\n=== Alternative Planned-Edge Matching ===", [(k, alt.get(k)) for k in ["accepted_alternative_count", "alternative_planned_edge_match_rate", "alternative_cell_coverage_mean", "alternative_anchor_match_rate", "alternative_type_match_rate", "alternative_multi_edge_match_rate", "matched_extracted_edge_type_mismatch_count"]])
    ip = report.get("island_planning", {})
    ia = ip.get("island_attempt_count", {})
    print_kv("\n=== Island Candidate Search ===", [("sampled_island_component_count", ip.get("sampled_island_component_count")), ("accepted_island_component_count", ip.get("accepted_island_component_count")), ("island_no_cells_generated_count", ip.get("island_no_cells_generated_count")), ("island_cells_connected_to_main_component_count", ip.get("island_cells_connected_to_main_component_count")), ("island_component_match_rate", ip.get("island_component_match_rate")), ("island_attempt_count_mean", ia.get("mean") if isinstance(ia, dict) else None), ("island_attempt_count_p50", ia.get("p50") if isinstance(ia, dict) else None), ("island_attempt_count_p95", ia.get("p95") if isinstance(ia, dict) else None)])


# Keep v3 printer alias after its definition is available.

if __name__ == "__main__":
    main()
