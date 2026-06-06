#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TDA-QD MazeForge 3.0.4

Clean distribution-quality validation factory built from the verified 3.0.4(13)
core algorithm. This script does not modify the stable 3.0.4 factory or the MVP
files. It exposes only formal validation/generation modes:
  generate_until_target
  multi_seed_until_target
  validate_quality

Final archive acceptance is based only on full rect-room macro extraction
full_bin_key + handdraw target quotas.
"""
from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import importlib.util
import json
import math
import random
import statistics
import sys
import time
import shutil
from itertools import count
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

VERSION = "3.0.4_tda_qd_mazeforge_final"
TITLE = "[TDA-QD MazeForge 3.0.4 Final] Handdraw-Quota Closure Patch"
DEFAULT_TARGET_DISTRIBUTION_REL = "feature_maze/v3_0_maze_generation/configs/handdraw_target_distribution_3_0_4.json"
DEFAULT_CONFIG_REL = "feature_maze/v3_0_maze_generation/configs/archive_factory_3_0_4_default(3).json"
DEFAULT_OUTPUT_ROOT_REL = "feature_maze/v3_0_maze_generation/outputs/3.0.4"
CRITICAL_LONG_BINS = [
    "tree|long|high_choice|high_endpoint",
    "tree|long|mid_choice|high_endpoint",
    "single_loop|long|high_choice|high_endpoint",
    "multi_loop|long|high_choice|high_endpoint",
    "tree|long|low_choice|mid_endpoint",
]
Cell = Tuple[int, int]
Grid = List[List[int]]
ACTIONS: List[Cell] = [(-1, 0), (1, 0), (0, -1), (0, 1)]

try:
    from tqdm import tqdm as _tqdm
    TQDM_AVAILABLE = True
except Exception:  # pragma: no cover
    _tqdm = None
    TQDM_AVAILABLE = False


def progress(it: Iterable[Any], desc: str = "", **kwargs: Any):
    if _tqdm is None or not sys.stderr.isatty():
        if desc and _tqdm is None:
            print(f"[INFO] {desc} (tqdm not installed; progress bar disabled)")
        return it
    opts: Dict[str, Any] = dict(
        desc=desc,
        dynamic_ncols=True,
        mininterval=2.0,
        maxinterval=10.0,
        smoothing=0.05,
        file=sys.stderr,
        leave=True,
        ascii=True,
    )
    opts.update(kwargs)
    return _tqdm(it, **opts)


iter_progress = progress


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(s))[:180]


# -----------------------------------------------------------------------------
# Self-contained 3.0.4 core
# -----------------------------------------------------------------------------
# The TDA-QD shell intentionally does not import common/ or previous experiment
# scripts. The only expected external files are the handdraw target distribution
# JSON and optional default JSON config. The minimal full extractor / quantizer /
# generators / archive utilities are embedded below.

@dataclass
class MazeCandidate:
    grid: Grid
    start: Cell
    goal: Cell
    origin_generator: str
    origin_seed_id: str
    transform_history: List[Dict[str, Any]] = field(default_factory=list)
    candidate_id: str = ""

    def clone(self) -> "MazeCandidate":
        return MazeCandidate(
            grid=[row[:] for row in self.grid],
            start=self.start,
            goal=self.goal,
            origin_generator=self.origin_generator,
            origin_seed_id=self.origin_seed_id,
            transform_history=copy.deepcopy(self.transform_history),
            candidate_id=self.candidate_id,
        )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def append_jsonl(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")


def in_bounds(r: int, c: int, n: int) -> bool:
    return 0 <= r < n and 0 <= c < n


def neighbors(cell: Cell, n: int) -> List[Cell]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in ACTIONS if in_bounds(r + dr, c + dc, n)]


def empty_grid(n: int, fill: int = 1) -> Grid:
    return [[fill for _ in range(n)] for _ in range(n)]


def free_cells(grid: Grid) -> List[Cell]:
    return [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == 0]


def cell_degree_in_set(grid: Grid, cell: Cell, allowed: Optional[set[Cell]] = None) -> int:
    n = len(grid)
    return sum(1 for nb in neighbors(cell, n) if grid[nb[0]][nb[1]] == 0 and (allowed is None or nb in allowed))


def local_free_count(grid: Grid, cell: Cell, use_8_neighborhood: bool = True) -> int:
    n = len(grid)
    r, c = cell
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)] if use_8_neighborhood else ACTIONS
    cnt = 0
    for dr, dc in dirs:
        rr, cc = r + dr, c + dc
        if in_bounds(rr, cc, n) and grid[rr][cc] == 0:
            cnt += 1
    return cnt


def bfs_dist(grid: Grid, start: Cell, goal: Optional[Cell] = None) -> Tuple[Dict[Cell, int], Dict[Cell, Cell]]:
    n = len(grid)
    dist: Dict[Cell, int] = {}
    parent: Dict[Cell, Cell] = {}
    if not in_bounds(start[0], start[1], n) or grid[start[0]][start[1]] != 0:
        return dist, parent
    q: deque[Cell] = deque([start])
    dist[start] = 0
    while q:
        cur = q.popleft()
        if goal is not None and cur == goal:
            break
        for nb in neighbors(cur, n):
            if grid[nb[0]][nb[1]] == 0 and nb not in dist:
                dist[nb] = dist[cur] + 1
                parent[nb] = cur
                q.append(nb)
    return dist, parent



def grid_shape(grid: Grid) -> Tuple[int, int]:
    return len(grid), max((len(row) for row in grid), default=0)

def in_grid(grid: Grid, r: int, c: int) -> bool:
    return 0 <= r < len(grid) and 0 <= c < len(grid[r])

def neighbors_grid(cell: Cell, grid: Grid) -> List[Cell]:
    r, c = cell
    out = []
    for dr, dc in ACTIONS:
        rr, cc = r + dr, c + dc
        if in_grid(grid, rr, cc):
            out.append((rr, cc))
    return out

def bfs_dist_grid(grid: Grid, start: Cell, goal: Optional[Cell] = None) -> Tuple[Dict[Cell, int], Dict[Cell, Cell]]:
    dist: Dict[Cell, int] = {}
    parent: Dict[Cell, Cell] = {}
    if not in_grid(grid, start[0], start[1]) or grid[start[0]][start[1]] != 0:
        return dist, parent
    q: deque[Cell] = deque([start])
    dist[start] = 0
    while q:
        cur = q.popleft()
        if goal is not None and cur == goal:
            break
        for nb in neighbors_grid(cur, grid):
            if grid[nb[0]][nb[1]] == 0 and nb not in dist:
                dist[nb] = dist[cur] + 1
                parent[nb] = cur
                q.append(nb)
    return dist, parent




def compute_three_quota_statuses(target: Dict[str, Any], archive_counts: Counter, full_eval_counts: Optional[Counter] = None) -> Dict[str, Dict[str, Any]]:
    full_eval_counts = full_eval_counts or Counter()
    return {
        "handdraw_quota_status": compute_hard_quota_status(target, archive_counts, "handdraw_observed", full_eval_counts),
        "exploration_quota_status": compute_hard_quota_status(target, archive_counts, "unobserved_possible", full_eval_counts),
        "all_nonzero_quota_status": compute_hard_quota_status(target, archive_counts, "all_nonzero", full_eval_counts),
    }


def compute_closure_status(
    target: Dict[str, Any],
    archive_counts: Counter,
    synthetic_metrics: Dict[str, Any],
    full_eval_counts: Optional[Counter],
    full_extraction_error_count: int,
    topology_diversity_report: Optional[Dict[str, Any]],
    closure_l1_threshold: float = 0.05,
) -> Dict[str, Any]:
    statuses = compute_three_quota_statuses(target, archive_counts, full_eval_counts or Counter())
    critical = {}
    critical_full = True
    for key in CRITICAL_LONG_BINS:
        q = int(target.get("bins", {}).get(key, {}).get("target_quota", 0))
        c = int(archive_counts.get(key, 0))
        critical[key] = {"target_quota": q, "confirmed_count": c, "full": c >= q}
        if c < q:
            critical_full = False
    topo_count = 0
    if topology_diversity_report:
        topo_count = int(topology_diversity_report.get("possible_topology_style_concentration_count", 0) or 0)
    conditions = {
        "handdraw_quota_met": bool(statuses["handdraw_quota_status"].get("quota_met")),
        "observed_handdraw_bin_fill_rate_full": float(synthetic_metrics.get("observed_handdraw_bin_fill_rate", 0.0)) >= 1.0,
        "critical_long_bins_full": critical_full,
        "l1_distance_within_threshold": float(synthetic_metrics.get("l1_distance_to_target_distribution", 999.0)) <= float(closure_l1_threshold),
        "full_extraction_error_zero": int(full_extraction_error_count) == 0,
        "structural_confirmed_zero": int(synthetic_metrics.get("structural_bin_accept_count", 0) or 0) == 0,
        "topology_style_concentration_zero": topo_count == 0,
    }
    unmet = [k for k, v in conditions.items() if not v]
    ready = not unmet
    info = []
    if conditions["handdraw_quota_met"] and not statuses["exploration_quota_status"].get("quota_met"):
        info.append("handdraw distribution completed; remaining bins are exploration-only")
    return {
        "ready_for_3_0_4_closure": ready,
        "conditions": conditions,
        "unmet_conditions": unmet,
        "closure_l1_threshold": closure_l1_threshold,
        "observed_handdraw_bin_fill_rate": synthetic_metrics.get("observed_handdraw_bin_fill_rate"),
        "l1_distance_to_target_distribution": synthetic_metrics.get("l1_distance_to_target_distribution"),
        "critical_long_bins": critical,
        "critical_long_bins_full": critical_full,
        "topology_style_concentration_count": topo_count,
        "full_extraction_error_count": int(full_extraction_error_count),
        "structural_confirmed_count": int(synthetic_metrics.get("structural_bin_accept_count", 0) or 0),
        "handdraw_quota_status": statuses["handdraw_quota_status"],
        "exploration_quota_status": statuses["exploration_quota_status"],
        "all_nonzero_quota_status": statuses["all_nonzero_quota_status"],
        "info": info,
    }


def ascii_from_record(record: Dict[str, Any]) -> str:
    grid = record.get("grid") or []
    start = tuple(record.get("start", [0, 0]))
    goal = tuple(record.get("goal", [0, 0]))
    if grid and isinstance(grid[0], str):
        return "\n".join(grid)
    try:
        return "\n".join(grid_to_ascii(grid, start, goal))
    except Exception:
        return ""


def grid_hashes_for_record(record: Dict[str, Any]) -> Tuple[str, str]:
    grid = record.get("grid") or []
    start = record.get("start", [0, 0])
    goal = record.get("goal", [0, 0])
    grid_payload = json.dumps(grid, ensure_ascii=False, separators=(",", ":"), default=str)
    sg_payload = json.dumps({"grid": grid, "start": start, "goal": goal}, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha1(grid_payload.encode("utf-8")).hexdigest()[:16], hashlib.sha1(sg_payload.encode("utf-8")).hexdigest()[:16]


def build_visualization_path_lookup(vis_index: Optional[Dict[str, Any]]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    if not vis_index:
        return lookup
    for _bin_key, entry in vis_index.items():
        paths = entry.get("paths", []) or []
        ids = entry.get("sample_ids", []) or []
        for sample_id, path in zip(ids, paths):
            if sample_id:
                lookup[str(sample_id)] = str(path)
    return lookup


def export_samples_csv(out_dir: Path, records: List[Dict[str, Any]], target: Dict[str, Any], vis_index: Optional[Dict[str, Any]], csv_name: str = "samples.csv") -> Dict[str, Any]:
    ensure_dir(out_dir)
    vis_lookup = build_visualization_path_lookup(vis_index)
    def sort_key(r: Dict[str, Any]):
        key = (r.get("full_bin") or {}).get("bin_key") or r.get("full_bin_key") or ""
        status = target.get("bins", {}).get(key, {}).get("status", "")
        rank = 0 if status == "handdraw_observed" else (1 if status == "unobserved_possible" else 2)
        return (rank, key, str(r.get("sample_id") or r.get("maze_id") or r.get("candidate_id") or ""))
    rows = []
    for idx, r in enumerate(sorted(records, key=sort_key), 1):
        full_bin = r.get("full_bin") or {}
        bin_key = full_bin.get("bin_key") or r.get("full_bin_key") or ""
        cycle, bfs, choice, endpoint = bin_parts(bin_key)
        tb = target.get("bins", {}).get(bin_key, {})
        metrics = r.get("metrics") or r.get("full_metrics") or {}
        sample_id = str(r.get("sample_id") or r.get("maze_id") or r.get("candidate_id") or f"sample_{idx:06d}")
        grid_hash, sg_hash = grid_hashes_for_record(r)
        start = r.get("start", ["", ""]); goal = r.get("goal", ["", ""])
        vis_path = vis_lookup.get(sample_id, "")
        rows.append({
            "sample_id": sample_id,
            "bin_key": bin_key,
            "cycle_bin": cycle,
            "bfs_bin": bfs,
            "choice_bin": choice,
            "endpoint_bin": endpoint,
            "target_status": tb.get("status", ""),
            "target_quota": int(tb.get("target_quota", 0) or 0),
            "target_count_final": sum(1 for x in records if ((x.get("full_bin") or {}).get("bin_key") or x.get("full_bin_key")) == bin_key),
            "quota_scope_member_handdraw": tb.get("status") == "handdraw_observed",
            "quota_scope_member_exploration": tb.get("status") == "unobserved_possible",
            "is_critical_long_bin": bin_key in CRITICAL_LONG_BINS,
            "bfs_len": metrics.get("bfs_len", ""),
            "rect_room_compressed_cycle_rank": metrics.get("rect_room_compressed_cycle_rank", ""),
            "rect_room_macro_choice_count": metrics.get("rect_room_macro_choice_count", ""),
            "rect_room_macro_endpoint_count": metrics.get("rect_room_macro_endpoint_count", ""),
            "free_ratio": metrics.get("free_ratio", ""),
            "reachable_free_ratio": metrics.get("reachable_free_ratio", ""),
            "origin_generator": r.get("origin_generator", ""),
            "origin_seed_id": r.get("origin_seed_id", ""),
            "route_id": r.get("route") or r.get("route_id") or "",
            "from_sfg": bool(r.get("from_sfg") or r.get("origin_generator") == "sfg_generator"),
            "from_rbr": bool(r.get("from_rbr")),
            "accepted_as_commit": bool(r.get("accepted_as_commit") or r.get("from_rbr")),
            "lineage_id": r.get("lineage_id") or r.get("origin_seed_id", ""),
            "grid_hash": grid_hash,
            "grid_start_goal_hash": sg_hash,
            "start_r": start[0] if len(start) > 0 else "",
            "start_c": start[1] if len(start) > 1 else "",
            "goal_r": goal[0] if len(goal) > 0 else "",
            "goal_c": goal[1] if len(goal) > 1 else "",
            "ascii_grid": ascii_from_record(r),
            "json_grid": json.dumps(r.get("grid", []), ensure_ascii=False, separators=(",", ":")),
            "visualization_path": vis_path,
        })
    fieldnames = [
        "sample_id","bin_key","cycle_bin","bfs_bin","choice_bin","endpoint_bin","target_status","target_quota","target_count_final",
        "quota_scope_member_handdraw","quota_scope_member_exploration","is_critical_long_bin","bfs_len","rect_room_compressed_cycle_rank",
        "rect_room_macro_choice_count","rect_room_macro_endpoint_count","free_ratio","reachable_free_ratio","origin_generator","origin_seed_id",
        "route_id","from_sfg","from_rbr","accepted_as_commit","lineage_id","grid_hash","grid_start_goal_hash","start_r","start_c","goal_r","goal_c",
        "ascii_grid","json_grid","visualization_path"
    ]
    csv_path = out_dir / csv_name
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    by_bin = []
    counts = Counter(row["bin_key"] for row in rows)
    vis_counts = Counter(row["bin_key"] for row in rows if row.get("visualization_path"))
    for key, b in sorted(target.get("bins", {}).items()):
        if int(b.get("target_quota", 0)) <= 0 and counts.get(key, 0) == 0:
            continue
        by_bin.append({
            "bin_key": key,
            "target_status": b.get("status"),
            "quota": int(b.get("target_quota", 0) or 0),
            "confirmed_count": int(counts.get(key, 0)),
            "csv_sample_count": int(counts.get(key, 0)),
            "visualization_count": int(vis_counts.get(key, 0)),
            "is_critical_long_bin": key in CRITICAL_LONG_BINS,
        })
    by_bin_path = out_dir / "samples_by_bin.csv"
    with by_bin_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bin_key","target_status","quota","confirmed_count","csv_sample_count","visualization_count","is_critical_long_bin"])
        writer.writeheader()
        writer.writerows(by_bin)
    report = {
        "csv_path": str(csv_path),
        "samples_by_bin_csv_path": str(by_bin_path),
        "n_rows": len(rows),
        "confirmed_samples_count": len(records),
        "row_count_matches_confirmed": len(rows) == len(records),
        "n_handdraw_observed_rows": sum(1 for row in rows if row["target_status"] == "handdraw_observed"),
        "n_exploration_rows": sum(1 for row in rows if row["target_status"] == "unobserved_possible"),
        "n_critical_long_rows": sum(1 for row in rows if row["is_critical_long_bin"]),
        "missing_visualization_path_count": sum(1 for row in rows if not row.get("visualization_path")),
        "visualizations_disabled": vis_index is None,
    }
    write_json(out_dir / "samples_csv_export_report.json", report)
    return report

def shortest_path(grid: Grid, start: Cell, goal: Cell) -> List[Cell]:
    dist, parent = bfs_dist(grid, start, goal)
    if goal not in dist:
        return []
    cur = goal
    path = [cur]
    while cur != start:
        cur = parent[cur]
        path.append(cur)
    return list(reversed(path))


def connected_components(grid: Grid) -> List[List[Cell]]:
    n = len(grid)
    seen: set[Cell] = set()
    comps: List[List[Cell]] = []
    for cell in free_cells(grid):
        if cell in seen:
            continue
        q = deque([cell])
        seen.add(cell)
        comp: List[Cell] = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nb in neighbors(cur, n):
                if grid[nb[0]][nb[1]] == 0 and nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        comps.append(comp)
    return comps


def count_edges_in_cells(grid: Grid, cells: Iterable[Cell]) -> int:
    cell_set = set(cells)
    e = 0
    for r, c in cell_set:
        for dr, dc in [(1, 0), (0, 1)]:
            nb = (r + dr, c + dc)
            if nb in cell_set and grid[nb[0]][nb[1]] == 0:
                e += 1
    return e


def two_by_two_free_block_count(grid: Grid) -> int:
    n = len(grid)
    cnt = 0
    for r in range(n - 1):
        for c in range(n - 1):
            if grid[r][c] == grid[r + 1][c] == grid[r][c + 1] == grid[r + 1][c + 1] == 0:
                cnt += 1
    return cnt


def count_room_components(grid: Grid, min_size: int = 3, high_degree_threshold: int = 3) -> Tuple[int, List[List[Cell]]]:
    n = len(grid)
    candidates: set[Cell] = set()
    for r in range(n - 1):
        for c in range(n - 1):
            block = [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]
            if all(grid[x][y] == 0 for x, y in block):
                candidates.update(block)
    for cell in free_cells(grid):
        if cell_degree_in_set(grid, cell) >= high_degree_threshold:
            candidates.add(cell)
    seen: set[Cell] = set()
    comps: List[List[Cell]] = []
    for cell in candidates:
        if cell in seen:
            continue
        q = deque([cell])
        seen.add(cell)
        comp: List[Cell] = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nb in neighbors(cur, n):
                if nb in candidates and nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        if len(comp) >= min_size:
            comps.append(comp)
    return len(comps), comps


def grid_hash(grid: Grid, start: Cell, goal: Cell) -> str:
    s = "".join("".join(str(v) for v in row) for row in grid) + f"|{start}|{goal}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def metric_stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"n": 0, "min": None, "p50": None, "mean": None, "max": None}
    vals = sorted(values)
    return {"n": len(vals), "min": vals[0], "p50": vals[len(vals) // 2], "mean": sum(vals) / len(vals), "max": vals[-1]}


def choose_weighted(rng: random.Random, weights: Dict[str, float]) -> str:
    items = [(k, max(0.0, float(v))) for k, v in weights.items() if float(v) > 0]
    if not items:
        raise ValueError("empty weights")
    total = sum(v for _, v in items)
    x = rng.random() * total
    acc = 0.0
    for k, v in items:
        acc += v
        if x <= acc:
            return k
    return items[-1][0]






class RectRoomExtractor:
    """Maximum non-overlapping rectangle room extractor.

    Room definition for 3.0.4(3): a maximal, non-overlapping, wall-free
    rectangular free-cell region with w>=2 and h>=2. This is a diagnostic
    quantizer component, not a strict construction validator.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cfg = config.get("rect_room_extraction", {})

    @staticmethod
    def _rect_cells(r0: int, c0: int, r1: int, c1: int) -> List[Cell]:
        return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]

    def enumerate_candidates(self, grid: Grid, main_cells: set[Cell]) -> List[Dict[str, Any]]:
        h, w = grid_shape(grid)
        candidates: List[Dict[str, Any]] = []
        # Prefix sum over cells that are free and start-reachable. Works for rectangular fixtures.
        ok = [[1 if (r, c) in main_cells else 0 for c in range(w)] for r in range(h)]
        ps = [[0] * (w + 1) for _ in range(h + 1)]
        for r in range(h):
            row_sum = 0
            for c in range(w):
                row_sum += ok[r][c]
                ps[r + 1][c + 1] = ps[r][c + 1] + row_sum
        def rect_sum(r0: int, c0: int, r1: int, c1: int) -> int:
            # inclusive bbox
            return ps[r1 + 1][c1 + 1] - ps[r0][c1 + 1] - ps[r1 + 1][c0] + ps[r0][c0]
        # Inclusive bbox convention: [r0, c0, r1, c1].
        for r0 in range(h):
            for c0 in range(w):
                if not ok[r0][c0]:
                    continue
                for r1 in range(r0 + 1, h):
                    hh = r1 - r0 + 1
                    for c1 in range(c0 + 1, w):
                        ww = c1 - c0 + 1
                        area = ww * hh
                        if rect_sum(r0, c0, r1, c1) != area:
                            continue
                        aspect_ratio = max(ww, hh) / float(min(ww, hh))
                        cells = self._rect_cells(r0, c0, r1, c1)
                        candidates.append({
                            "bbox": [r0, c0, r1, c1],
                            "bbox_convention": "inclusive",
                            "w": ww,
                            "h": hh,
                            "area": area,
                            "aspect_ratio": aspect_ratio,
                            "covered_cells": cells,
                            "sort_key": (-area, -min(ww, hh), aspect_ratio, r0, c0, r1, c1),
                        })
        candidates.sort(key=lambda x: x["sort_key"])
        return candidates

    def extract(self, grid: Grid, start: Cell, goal: Cell) -> Dict[str, Any]:
        dist, _ = bfs_dist_grid(grid, start)
        main_cells = set(dist.keys())
        if not main_cells or goal not in main_cells:
            return {
                "rect_room_extraction_status": "failed",
                "rect_room_extraction_warning": ["start_goal_not_connected"],
                "candidate_count": 0,
                "selected_rect_rooms": [],
                "rect_room_count": 0,
                "rect_room_area_sum": 0,
                "tiny_2x2_room_count": 0,
                "elongated_room_count": 0,
                "internal_rect_room_cycle_rank_sum": 0,
                "room_cell_count": 0,
                "room_cell_set": set(),
                "cell_to_room_id": {},
            }
        candidates = self.enumerate_candidates(grid, main_cells)
        occupied: set[Cell] = set()
        rooms: List[Dict[str, Any]] = []
        cell_to_room_id: Dict[Cell, str] = {}
        internal_cycle_sum = 0
        for cand in candidates:
            cells = set(cand["covered_cells"])
            if cells & occupied:
                continue
            rid = f"rr_{len(rooms):04d}"
            r0, c0, r1, c1 = cand["bbox"]
            anchor = ((r0 + r1) // 2, (c0 + c1) // 2)
            w, h, area = cand["w"], cand["h"], cand["area"]
            internal_edges = h * (w - 1) + w * (h - 1)
            internal_cycle = max(0, internal_edges - area + 1)
            row = {
                "room_id": rid,
                "bbox": cand["bbox"],
                "bbox_convention": "inclusive",
                "w": w,
                "h": h,
                "area": area,
                "aspect_ratio": cand["aspect_ratio"],
                "anchor": list(anchor),
                "covered_cells": [list(x) for x in sorted(cells)],
                "internal_cycle_rank": internal_cycle,
            }
            rooms.append(row)
            occupied.update(cells)
            internal_cycle_sum += internal_cycle
            for cell in cells:
                cell_to_room_id[cell] = rid
        return {
            "rect_room_extraction_status": "ok",
            "rect_room_extraction_warning": [],
            "candidate_count": len(candidates),
            "selected_rect_rooms": rooms,
            "rect_room_count": len(rooms),
            "rect_room_area_sum": sum(r["area"] for r in rooms),
            "tiny_2x2_room_count": sum(1 for r in rooms if r["w"] == 2 and r["h"] == 2),
            "elongated_room_count": sum(1 for r in rooms if max(r["w"], r["h"]) / max(1, min(r["w"], r["h"])) >= float(self.cfg.get("elongated_aspect_ratio", 3.0))),
            "internal_rect_room_cycle_rank_sum": internal_cycle_sum,
            "room_cell_count": len(occupied),
            "room_cell_set": occupied,
            "cell_to_room_id": cell_to_room_id,
        }


class RectRoomMacroGraphExtractor:
    """Rect-room macro graph extractor with parallel-edge preserving diagnostics."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.room_extractor = RectRoomExtractor(config)

    def _room_exit_info(self, grid: Grid, main_cells: set[Cell], room: Dict[str, Any], room_cells: set[Cell]) -> Dict[str, Any]:
        n = len(grid)
        cells = {tuple(x) for x in room["covered_cells"]}
        outside: set[Cell] = set()
        boundary_pairs: List[Tuple[Cell, Cell]] = []
        for cell in cells:
            for nb in neighbors(cell, n):
                if nb in main_cells and nb not in room_cells:
                    outside.add(nb)
                    boundary_pairs.append((cell, nb))
        seen: set[Cell] = set()
        groups: List[List[Cell]] = []
        for cell in sorted(outside):
            if cell in seen:
                continue
            q = deque([cell])
            seen.add(cell)
            group: List[Cell] = []
            while q:
                cur = q.popleft()
                group.append(cur)
                for nb in neighbors(cur, n):
                    if nb in outside and nb not in seen:
                        seen.add(nb)
                        q.append(nb)
            groups.append(group)
        return {
            "room_boundary_connection_count": len(boundary_pairs),
            "room_exit_port_count": len(groups),
            "room_outside_component_count": len(groups),
            "exit_groups": [[list(x) for x in g[:8]] for g in groups],
            "exit_cells_sample": [list(x) for x in sorted(outside)[:12]],
        }

    def extract(self, grid: Grid, start: Cell, goal: Cell) -> Dict[str, Any]:
        n = len(grid)
        dist, _ = bfs_dist_grid(grid, start)
        main_cells = set(dist.keys())
        warnings: List[str] = []
        if not main_cells or goal not in main_cells:
            return {
                "rect_macro_extraction_status": "failed",
                "rect_macro_extraction_warning": ["start_goal_not_connected"],
                "rect_room_macro_vertex_count": 0,
                "rect_room_macro_edge_count": 0,
                "rect_room_compressed_cycle_rank": 0,
                "rect_room_macro_choice_count": 0,
                "rect_room_macro_endpoint_count": 0,
                "rect_room_count": 0,
                "selected_rect_rooms": [],
                "nodes": [],
                "edges": [],
            }
        room_info = self.room_extractor.extract(grid, start, goal)
        rooms = room_info["selected_rect_rooms"]
        room_cells: set[Cell] = set(room_info.get("room_cell_set", set()))
        cell_to_room_id: Dict[Cell, str] = room_info.get("cell_to_room_id", {})
        room_by_id = {r["room_id"]: r for r in rooms}
        # Node candidates.
        node_cells: Dict[str, Cell] = {}
        node_type: Dict[str, str] = {}
        room_anchor_node: Dict[str, str] = {}
        for room in rooms:
            rid = room["room_id"]
            anchor = tuple(room["anchor"])
            nid = f"rn_{rid}"
            node_cells[nid] = anchor
            node_type[nid] = "rect_room_anchor"
            room_anchor_node[rid] = nid
        degree = {cell: cell_degree_in_set(grid, cell, main_cells) for cell in main_cells}
        terminal_inside_room_count = 0
        terminal_room_id: Dict[str, Optional[str]] = {"start": None, "goal": None}
        for label, cell in [("start", start), ("goal", goal)]:
            rid = cell_to_room_id.get(cell)
            if rid:
                terminal_inside_room_count += 1
                terminal_room_id[label] = rid
                # Keep terminal metadata on the room node; do not create duplicate same-cell node.
            else:
                nid = f"terminal_{label}"
                node_cells[nid] = cell
                deg = degree.get(cell, 0)
                node_type[nid] = "terminal_choice" if deg >= 3 else "terminal_endpoint" if deg <= 1 else "terminal_corridor_split"
        for cell, deg in degree.items():
            if cell in room_cells or cell in (start, goal):
                continue
            if deg != 2:
                nid = f"cn_{cell[0]}_{cell[1]}"
                node_cells[nid] = cell
                node_type[nid] = "macro_endpoint" if deg == 1 else "macro_choice" if deg >= 3 else "macro_corridor_split"
        cell_to_node: Dict[Cell, str] = {cell: nid for nid, cell in node_cells.items()}
        for rid, nid in room_anchor_node.items():
            for cell_list in room_by_id[rid]["covered_cells"]:
                cell_to_node[tuple(cell_list)] = nid
        # Trace from each node. Room nodes start from every outside boundary neighbor.
        frontiers: List[Tuple[str, Cell, Cell]] = []  # node id, node representative cell, first next cell
        for nid, cell in node_cells.items():
            rid_for_node = None
            if nid.startswith("rn_rr_"):
                rid_for_node = nid[len("rn_"):]
            if rid_for_node and rid_for_node in room_by_id:
                cells = {tuple(x) for x in room_by_id[rid_for_node]["covered_cells"]}
                for rc in sorted(cells):
                    for nb in neighbors(rc, n):
                        if nb in main_cells and nb not in room_cells:
                            frontiers.append((nid, rc, nb))
            else:
                for nb in neighbors(cell, n):
                    if nb in main_cells:
                        frontiers.append((nid, cell, nb))
        edges: List[Dict[str, Any]] = []
        seen_paths: set[Tuple[Cell, ...]] = set()
        dangling = 0
        loop_without_node = 0
        for src_nid, src_cell, first in frontiers:
            path = [src_cell, first]
            prev, cur = src_cell, first
            seen_cells = {src_cell}
            dst_nid = cell_to_node.get(cur)
            while dst_nid is None:
                if cur in seen_cells:
                    loop_without_node += 1
                    warnings.append("rect_macro_loop_without_node")
                    break
                seen_cells.add(cur)
                nexts = [x for x in neighbors(cur, n) if x in main_cells and x != prev]
                # Treat entering any room as hitting that room's anchor node.
                room_nexts = [x for x in nexts if x in room_cells]
                if room_nexts:
                    cur2 = sorted(room_nexts)[0]
                    path.append(cur2)
                    dst_nid = cell_to_node.get(cur2)
                    cur = cur2
                    break
                nexts = [x for x in nexts if x not in room_cells]
                if len(nexts) != 1:
                    if len(nexts) > 1:
                        warnings.append("rect_macro_ambiguous_non_room_junction")
                    else:
                        warnings.append("rect_macro_dangling_trace")
                        dangling += 1
                    break
                prev, cur = cur, nexts[0]
                path.append(cur)
                dst_nid = cell_to_node.get(cur)
            if dst_nid is None or dst_nid == src_nid:
                continue
            key1 = tuple(path)
            key2 = tuple(reversed(path))
            if key1 in seen_paths or key2 in seen_paths:
                continue
            seen_paths.add(key1)
            edges.append({
                "edge_id": f"rme_{len(edges):04d}",
                "u": src_nid,
                "v": dst_nid,
                "length": max(1, len(path) - 1),
                "path_cells": [list(x) for x in path],
                "edge_kind": "corridor",
            })
        deg: Counter[str] = Counter()
        pair_counts: Counter[Tuple[str, str]] = Counter()
        for e in edges:
            deg[e["u"]] += 1
            deg[e["v"]] += 1
            pair_counts[tuple(sorted((e["u"], e["v"])))] += 1
        parallel_macro_edge_count = sum(cnt for cnt in pair_counts.values() if cnt > 1)
        room_pair_parallel_edge_count = 0
        for (u, v), cnt in pair_counts.items():
            if cnt > 1 and node_type.get(u) == "rect_room_anchor" and node_type.get(v) == "rect_room_anchor":
                room_pair_parallel_edge_count += cnt
        # Before compression.
        node_rows_before: List[Dict[str, Any]] = []
        room_debug_rows: List[Dict[str, Any]] = []
        for nid, cell in sorted(node_cells.items()):
            rid = None
            if nid.startswith("rn_rr_"):
                rid = nid[len("rn_"):]
            row = {
                "node_id": nid,
                "cell": list(cell),
                "node_type": node_type[nid],
                "macro_degree": deg[nid],
                "is_start": terminal_room_id["start"] == rid or nid == "terminal_start",
                "is_goal": terminal_room_id["goal"] == rid or nid == "terminal_goal",
                "room_id": rid,
            }
            node_rows_before.append(row)
            if rid:
                room = room_by_id[rid]
                exit_info = self._room_exit_info(grid, main_cells, room, room_cells)
                role = "endpoint_like_room" if deg[nid] == 1 else "corridor_like_room" if deg[nid] == 2 else "macro_choice_room" if deg[nid] >= 3 else "isolated_room"
                room_debug_rows.append({**room, **exit_info, "macro_degree": deg[nid], "room_role_after_macro_graph": role})
        v_before = len(node_rows_before)
        e_before = len(edges)
        comp_count = 1 if v_before else 0
        cr_before = max(0, e_before - v_before + comp_count)
        choice_before = sum(1 for r in node_rows_before if r["macro_degree"] >= 3 and not r["is_start"] and not r["is_goal"])
        endpoint_before = sum(1 for r in node_rows_before if r["macro_degree"] == 1 and not r["is_start"] and not r["is_goal"])
        # Degree-2 room compression approximation. It preserves cycle rank for simple degree-2 pass-through nodes.
        compressible = [r for r in node_rows_before if r["node_type"] == "rect_room_anchor" and r["macro_degree"] == 2 and not r["is_start"] and not r["is_goal"]]
        compression_warning: List[str] = []
        if compressible:
            compression_warning.append("degree2_room_compression_is_metric_level_mvp")
        v_after = max(0, v_before - len(compressible))
        e_after = max(0, e_before - len(compressible))
        cr_after = max(0, e_after - v_after + (1 if v_after else 0))
        # For choice/endpoint, count after compression by ignoring compressible rooms.
        compress_ids = {r["node_id"] for r in compressible}
        node_rows_after = [r for r in node_rows_before if r["node_id"] not in compress_ids]
        choice_after = sum(1 for r in node_rows_after if r["macro_degree"] >= 3 and not r["is_start"] and not r["is_goal"])
        endpoint_after = sum(1 for r in node_rows_after if r["macro_degree"] == 1 and not r["is_start"] and not r["is_goal"])
        expected_after_internal = max(0, 0)  # filled by Quantizer using raw_cycle_rank.
        return {
            "rect_macro_extraction_status": "ok" if not warnings else "warning",
            "rect_macro_extraction_warning": sorted(set(warnings + compression_warning)),
            "selected_rect_rooms": room_debug_rows,
            "rect_room_count": room_info["rect_room_count"],
            "rect_room_area_sum": room_info["rect_room_area_sum"],
            "tiny_2x2_room_count": room_info["tiny_2x2_room_count"],
            "elongated_room_count": room_info["elongated_room_count"],
            "internal_rect_room_cycle_rank_sum": room_info["internal_rect_room_cycle_rank_sum"],
            "terminal_inside_room_count": terminal_inside_room_count,
            "terminal_room_id": terminal_room_id,
            "terminal_endpoint_count": sum(1 for r in node_rows_before if r["node_type"] == "terminal_endpoint"),
            "terminal_choice_count": sum(1 for r in node_rows_before if r["node_type"] == "terminal_choice"),
            "rect_room_macro_vertex_count_before": v_before,
            "rect_room_macro_edge_count_before": e_before,
            "rect_room_compressed_cycle_rank_before": cr_before,
            "rect_room_macro_choice_count_before": choice_before,
            "rect_room_macro_endpoint_count_before": endpoint_before,
            "rect_room_macro_vertex_count": v_after,
            "rect_room_macro_edge_count": e_after,
            "rect_room_compressed_cycle_rank": cr_after,
            "rect_room_macro_choice_count": choice_after,
            "rect_room_macro_endpoint_count": endpoint_after,
            "rect_room_macro_corridor_edge_count": e_after,
            "parallel_macro_edge_count": parallel_macro_edge_count,
            "room_pair_parallel_edge_count": room_pair_parallel_edge_count,
            "macro_cycle_from_parallel_room_edges": max(0, room_pair_parallel_edge_count - 1) if room_pair_parallel_edge_count else 0,
            "obstructed_open_component_count": 0,
            "non_rect_open_component_count": 0,
            "uncovered_open_area_count": 0,
            "nodes_before_compression": node_rows_before,
            "nodes_after_compression": node_rows_after,
            "edges": edges,
            "rect_macro_debug": {"dangling_trace_count": dangling, "loop_without_node_count": loop_without_node},
        }


class Quantizer:
    """Lightweight candidate evaluator for quality gates and transform diagnostics.

    Final archive binning is never based on these cheap/raw metrics. Accepted
    samples are confirmed only after full rect-room macro graph extraction.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.n = int(config["grid"].get("size", 8))
        self.rect_macro_extractor = RectRoomMacroGraphExtractor(config)

    def evaluate(self, cand: MazeCandidate) -> Dict[str, Any]:
        grid = cand.grid
        start = cand.start
        goal = cand.goal
        rows, cols = grid_shape(grid)
        free = free_cells(grid)
        free_count = len(free)
        free_ratio = free_count / float(max(1, rows * cols))
        dist_goal, _ = bfs_dist_grid(grid, start, goal)
        solvable = goal in dist_goal
        bfs_len = int(dist_goal[goal]) if solvable else None
        dist_all, _ = bfs_dist_grid(grid, start)
        reachable = set(dist_all.keys())
        reachable_count = len(reachable)
        reachable_free_ratio = reachable_count / free_count if free_count else 0.0
        raw_v = len(reachable)
        raw_e = count_edges_in_cells(grid, reachable)
        raw_cycle_rank = max(0, raw_e - raw_v + 1) if raw_v else 0
        degrees = {cell: cell_degree_in_set(grid, cell, reachable) for cell in reachable}
        raw_endpoint_cell_count = sum(1 for d in degrees.values() if d == 1)
        raw_choice_cell_count = sum(1 for d in degrees.values() if d >= 3)
        start_free_degree = cell_degree_in_set(grid, start, reachable) if start in reachable else 0
        shortest_action_count = 0
        if solvable and bfs_len is not None:
            for nb in neighbors_grid(start, grid):
                if grid[nb[0]][nb[1]] == 0:
                    d2, _ = bfs_dist_grid(grid, nb, goal)
                    if goal in d2 and d2[goal] == bfs_len - 1:
                        shortest_action_count += 1
        room_count, _ = count_room_components(grid, min_size=int(self.config.get("room_detection", {}).get("room_min_component_size", 3)))
        two_by_two_count = two_by_two_free_block_count(grid)
        comps = connected_components(grid)
        island_component_count = max(0, len(comps) - 1) if start in free else len(comps)
        bfs_len_norm = bfs_len / float(max(1, rows * cols)) if bfs_len is not None else None
        duplicate_hash = grid_hash(grid, start, goal)
        return {
            "solvable": solvable,
            "bfs_len": bfs_len,
            "bfs_len_norm": bfs_len_norm,
            "raw_vertex_count": raw_v,
            "raw_edge_count": raw_e,
            "raw_cycle_rank": raw_cycle_rank,
            "raw_endpoint_cell_count": raw_endpoint_cell_count,
            "raw_choice_cell_count": raw_choice_cell_count,
            "free_ratio": free_ratio,
            "reachable_free_ratio": reachable_free_ratio,
            "start_free_degree": start_free_degree,
            "shortest_action_count_at_start": shortest_action_count,
            "dead_end_count": raw_endpoint_cell_count,
            "room_count": room_count,
            "two_by_two_free_block_count": two_by_two_count,
            "island_component_count": island_component_count,
            "free_count": free_count,
            "reachable_free_count": reachable_count,
            "duplicate_hash": duplicate_hash,
        }

    def quality_gate(self, metrics: Dict[str, Any]) -> Tuple[float, Optional[str], List[str]]:
        q = self.config["quality_gate"]
        failures: List[str] = []
        if not metrics["solvable"]:
            failures.append("solvable_fail")
        if metrics.get("reachable_free_ratio", 0.0) < q.get("min_reachable_free_ratio", 0.9):
            failures.append("reachable_free_ratio_fail")
        if metrics.get("bfs_len") is None or metrics.get("bfs_len", 0) < q.get("min_bfs_len", 3):
            failures.append("bfs_len_fail")
        if metrics.get("start_free_degree", 0) < q.get("min_start_free_degree", 1):
            failures.append("start_free_degree_fail")
        fr = metrics.get("free_ratio", 0.0)
        if fr < q.get("min_free_ratio", 0.2) or fr > q.get("max_free_ratio", 0.85):
            failures.append("free_ratio_fail")
        return (0.0 if failures else 1.0), (failures[0] if failures else None), failures




class StartGoalAssigner:
    def __init__(self, config: Dict[str, Any], rng: random.Random):
        self.config = config
        self.rng = rng
        self.n = int(config["grid"]["size"])
        self.fallback_count = 0
        self.policy_counts: Counter[str] = Counter()
        self.target_bin_counts: Counter[str] = Counter()
        self.target_bin_hit_counts: Counter[str] = Counter()
        self.fallback_by_target_bin: Counter[str] = Counter()

    def assign(self, grid: Grid, archive: Optional[Any] = None) -> Tuple[Cell, Cell, Dict[str, Any]]:
        policy = self.config["grid"].get("start_goal_policy", "sample_by_bfs_bin")
        self.policy_counts[policy] += 1
        if policy == "fixed_corners":
            s = tuple(self.config["grid"].get("start", [0, 0]))
            g = tuple(self.config["grid"].get("goal", [self.n - 1, self.n - 1]))
            grid[s[0]][s[1]] = 0
            grid[g[0]][g[1]] = 0
            return s, g, {"policy": policy, "fallback": False, "target_bfs_bin": None, "sampled_pairs": 0}
        comps = connected_components(grid)
        if not comps:
            grid[0][0] = 0
            comps = [[(0, 0)]]
        largest = max(comps, key=len)
        if len(largest) < 2:
            # Open one neighbor to get a pair.
            a = largest[0]
            nb = neighbors(a, self.n)[0]
            grid[nb[0]][nb[1]] = 0
            largest = max(connected_components(grid), key=len)
        if policy == "random_free_pair":
            return (*self._random_pair(largest), {"policy": policy, "fallback": False, "target_bfs_bin": None, "sampled_pairs": 0})
        if policy == "random_boundary_pair":
            boundary = [c for c in largest if c[0] in (0, self.n - 1) or c[1] in (0, self.n - 1)]
            if len(boundary) >= 2:
                return (*self._random_pair(boundary), {"policy": policy, "fallback": False, "target_bfs_bin": None, "sampled_pairs": 0})
            self.fallback_count += 1
            s, g = self._random_pair(largest)
            return s, g, {"policy": policy, "fallback": True, "fallback_policy": "random_free_pair", "target_bfs_bin": None, "sampled_pairs": 0}
        if policy == "sample_by_bfs_bin":
            return self._sample_by_bfs_bin(grid, largest, archive)
        # Unsupported optional policy falls back explicitly.
        self.fallback_count += 1
        s, g = self._random_pair(largest)
        return s, g, {"policy": policy, "fallback": True, "fallback_policy": "random_free_pair", "target_bfs_bin": None, "sampled_pairs": 0}

    def _random_pair(self, cells: Sequence[Cell]) -> Tuple[Cell, Cell]:
        s = self.rng.choice(list(cells))
        g = self.rng.choice(list(cells))
        tries = 0
        while g == s and tries < 20:
            g = self.rng.choice(list(cells))
            tries += 1
        return s, g

    def _choose_target_bfs_bin(self, archive: Optional[Any]) -> str:
        bins = [b["name"] for b in self.config["archive"]["bfs_len_bins"]]
        if archive is None or self.config["start_goal"].get("target_bfs_bin_policy") != "prefer_sparse_bins":
            return self.rng.choice(bins)
        # Sum counts over cycle/endpoint dimensions and prefer sparse BFS bins.
        counts = Counter()
        for key, cnt in archive.counts.items():
            parts = key.split("|")
            if len(parts) >= 3:
                counts[parts[1]] += cnt
        min_count = min(counts.get(b, 0) for b in bins)
        sparse = [b for b in bins if counts.get(b, 0) == min_count]
        return self.rng.choice(sparse)

    def _bin_for_dist(self, dist: int) -> Optional[str]:
        for b in self.config["archive"]["bfs_len_bins"]:
            lo, hi = b["range"]
            if lo <= dist <= hi:
                return b["name"]
        return None

    def _sample_by_bfs_bin(self, grid: Grid, cells: Sequence[Cell], archive: Optional[Any]) -> Tuple[Cell, Cell, Dict[str, Any]]:
        cfg = self.config["start_goal"]
        min_bfs_len = int(cfg.get("min_bfs_len", 3))
        target = None
        cells_list = list(cells)
        max_pair_count = int(cfg.get("max_pair_count", 4096))
        mode = cfg.get("pair_sampling_mode", "all_pairs")
        pairs: List[Tuple[Cell, Cell]] = []
        if len(cells_list) >= 2:
            if mode == "all_pairs" or len(cells_list) * (len(cells_list) - 1) // 2 <= max_pair_count:
                for i in range(len(cells_list)):
                    for j in range(i + 1, len(cells_list)):
                        pairs.append((cells_list[i], cells_list[j]))
            else:
                seen_pairs: set[Tuple[Cell, Cell]] = set()
                attempts = 0
                while len(pairs) < max_pair_count and attempts < max_pair_count * 4:
                    a, b = self._random_pair(cells_list)
                    if a == b:
                        attempts += 1
                        continue
                    key = tuple(sorted([a, b]))  # type: ignore[arg-type]
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        pairs.append((a, b))
                    attempts += 1
        # Shuffle to avoid deterministic upper-triangle bias when choosing from matched buckets.
        self.rng.shuffle(pairs)
        by_bin: Dict[str, List[Tuple[Cell, Cell, int]]] = defaultdict(list)
        best_any: Optional[Tuple[Cell, Cell, int]] = None
        dist_cache: Dict[Cell, Dict[Cell, int]] = {}
        for s, g in pairs:
            if s not in dist_cache:
                dist_cache[s], _p = bfs_dist(grid, s)
            d = dist_cache[s]
            if g not in d or d[g] < min_bfs_len:
                continue
            dist = d[g]
            bname = self._bin_for_dist(dist)
            if bname is not None:
                by_bin[bname].append((s, g, dist))
            best_any = (s, g, dist) if best_any is None or dist > best_any[2] else best_any
        # Choose a target BFS bin after all-pairs/capped enumeration, so sparse-bin preference does not
        # repeatedly request a mathematically absent bin for the current candidate grid.
        available_bins = [b for b, vals in by_bin.items() if vals]
        if available_bins:
            if archive is not None and self.config["start_goal"].get("target_bfs_bin_policy") == "prefer_sparse_bins":
                counts = Counter()
                for key, cnt in archive.counts.items():
                    parts = key.split("|")
                    if len(parts) >= 2:
                        counts[parts[1]] += cnt
                min_count = min(counts.get(b, 0) for b in available_bins)
                sparse_available = [b for b in available_bins if counts.get(b, 0) == min_count]
                target = self.rng.choice(sparse_available)
            else:
                target = self.rng.choice(available_bins)
            self.target_bin_counts[target] += 1
            matched = by_bin[target]
            self.target_bin_hit_counts[target] += 1
            s, g, dist = self.rng.choice(matched)
            return s, g, {"policy": "sample_by_bfs_bin", "pair_sampling_mode": mode, "fallback": False, "target_bfs_bin": target, "sampled_pairs": len(pairs), "target_bin_pair_count": len(matched), "bfs_bin_pair_counts": {k: len(v) for k, v in by_bin.items()}, "selected_bfs_len": dist}
        target = self._choose_target_bfs_bin(archive)
        self.target_bin_counts[target] += 1
        self.fallback_count += 1
        self.fallback_by_target_bin[target] += 1
        if best_any is not None:
            s, g, dist = best_any
            return s, g, {"policy": "sample_by_bfs_bin", "pair_sampling_mode": mode, "fallback": True, "fallback_policy": "best_random_free_pair", "target_bfs_bin": target, "sampled_pairs": len(pairs), "target_bin_pair_count": 0, "bfs_bin_pair_counts": {k: len(v) for k, v in by_bin.items()}, "selected_bfs_len": dist}
        s, g = self._random_pair(cells_list)
        return s, g, {"policy": "sample_by_bfs_bin", "pair_sampling_mode": mode, "fallback": True, "fallback_policy": "random_free_pair", "target_bfs_bin": target, "sampled_pairs": len(pairs), "target_bin_pair_count": 0, "bfs_bin_pair_counts": {k: len(v) for k, v in by_bin.items()}}


class SeedGenerators:
    def __init__(self, config: Dict[str, Any], rng: random.Random):
        self.config = config
        self.rng = rng
        self.n = int(config["grid"]["size"])

    def make_grid(self, name: str) -> Grid:
        return getattr(self, name)()

    def random_wall_generator(self) -> Grid:
        cfg = self.config["seed_generators"]["random_wall_generator"]
        p = self.rng.uniform(cfg["wall_prob_min"], cfg["wall_prob_max"])
        grid = [[1 if self.rng.random() < p else 0 for _ in range(self.n)] for _ in range(self.n)]
        if self.rng.random() < 0.65:
            self._carve_manhattan_path(grid, (0, 0), (self.n - 1, self.n - 1))
        return grid

    def dfs_tree_generator(self) -> Grid:
        cfg = self.config["seed_generators"]["dfs_tree_generator"]
        grid = empty_grid(self.n, 1)
        start = (self.rng.randrange(self.n), self.rng.randrange(self.n))
        grid[start[0]][start[1]] = 0
        stack = [start]
        seen = {start}
        while stack:
            cur = stack[-1]
            nbs = [nb for nb in neighbors(cur, self.n) if nb not in seen]
            if not nbs:
                stack.pop()
                continue
            nb = self.rng.choice(nbs)
            seen.add(nb)
            grid[nb[0]][nb[1]] = 0
            stack.append(nb)
            if len(seen) > self.n * self.n * 0.65 and self.rng.random() < 0.15:
                break
        for cell in list(seen):
            if self.rng.random() < cfg.get("branch_open_prob", 0.2):
                for nb in neighbors(cell, self.n):
                    if self.rng.random() < cfg.get("extra_open_prob", 0.08):
                        grid[nb[0]][nb[1]] = 0
        return grid

    def loose_path_first_generator(self) -> Grid:
        cfg = self.config["seed_generators"]["loose_path_first_generator"]
        grid = empty_grid(self.n, 1)
        a = (self.rng.randrange(self.n), self.rng.randrange(self.n))
        b = (self.rng.randrange(self.n), self.rng.randrange(self.n))
        path = self._random_self_avoiding_path(a, b, max_attempts=int(cfg.get("max_path_attempts", 160))) or self._manhattan_cells(a, b)
        for r, c in path:
            grid[r][c] = 0
        for r in range(self.n):
            for c in range(self.n):
                if grid[r][c] == 1 and self.rng.random() < cfg.get("extra_open_prob", 0.16):
                    if any(grid[x][y] == 0 for x, y in neighbors((r, c), self.n)) or self.rng.random() < 0.2:
                        grid[r][c] = 0
        return grid

    def room_biased_generator(self) -> Grid:
        cfg = self.config["seed_generators"]["room_biased_generator"]
        grid = empty_grid(self.n, 1)
        a = (self.rng.randrange(self.n), self.rng.randrange(self.n))
        b = (self.rng.randrange(self.n), self.rng.randrange(self.n))
        self._carve_manhattan_path(grid, a, b)
        room_count = self.rng.randint(cfg["room_count_min"], cfg["room_count_max"])
        for _ in range(room_count):
            h = self.rng.randint(cfg["room_size_min"], cfg["room_size_max"])
            w = self.rng.randint(cfg["room_size_min"], cfg["room_size_max"])
            r0 = self.rng.randint(0, max(0, self.n - h))
            c0 = self.rng.randint(0, max(0, self.n - w))
            for r in range(r0, r0 + h):
                for c in range(c0, c0 + w):
                    grid[r][c] = 0
            anchor = (r0 + h // 2, c0 + w // 2)
            self._carve_manhattan_path(grid, anchor, self.rng.choice(self._manhattan_cells(a, b)))
        for r in range(self.n):
            for c in range(self.n):
                if grid[r][c] == 1 and self.rng.random() < cfg.get("connector_open_prob", 0.1):
                    grid[r][c] = 0
        return grid

    def template_mutation_generator(self) -> Grid:
        cfg = self.config["seed_generators"].get("template_mutation_generator", {})
        base = cfg.get("base", "loose_path_first_generator")
        grid = self.make_grid(base)
        cells = [(r, c) for r in range(self.n) for c in range(self.n)]
        self.rng.shuffle(cells)
        for cell in cells[: int(cfg.get("initial_mutations", 4))]:
            grid[cell[0]][cell[1]] = 1 - grid[cell[0]][cell[1]]
        return grid

    def _manhattan_cells(self, a: Cell, b: Cell) -> List[Cell]:
        r, c = a
        cells = [(r, c)]
        if self.rng.random() < 0.5:
            while c != b[1]:
                c += 1 if b[1] > c else -1
                cells.append((r, c))
            while r != b[0]:
                r += 1 if b[0] > r else -1
                cells.append((r, c))
        else:
            while r != b[0]:
                r += 1 if b[0] > r else -1
                cells.append((r, c))
            while c != b[1]:
                c += 1 if b[1] > c else -1
                cells.append((r, c))
        return cells

    def _carve_manhattan_path(self, grid: Grid, a: Cell, b: Cell) -> None:
        for r, c in self._manhattan_cells(a, b):
            grid[r][c] = 0

    def _random_self_avoiding_path(self, start: Cell, goal: Cell, max_attempts: int = 160) -> List[Cell]:
        for _ in range(max_attempts):
            cur = start
            path = [cur]
            used = {cur}
            for _step in range(self.n * self.n * 2):
                if cur == goal:
                    return path
                nbs = [nb for nb in neighbors(cur, self.n) if nb not in used]
                if not nbs:
                    break
                nbs.sort(key=lambda x: abs(x[0] - goal[0]) + abs(x[1] - goal[1]) + self.rng.random() * 2.5)
                nb = nbs[0] if self.rng.random() < 0.70 else self.rng.choice(nbs)
                used.add(nb)
                path.append(nb)
                cur = nb
        return []


class Transforms:
    def __init__(self, config: Dict[str, Any], rng: random.Random):
        self.config = config
        self.rng = rng
        self.n = int(config["grid"]["size"])

    def apply(self, op: str, cand: MazeCandidate, before_metrics: Dict[str, Any]) -> Tuple[Optional[MazeCandidate], Dict[str, Any]]:
        return getattr(self, op)(cand, before_metrics)

    def loop_flip(self, cand: MazeCandidate, before_metrics: Dict[str, Any]) -> Tuple[Optional[MazeCandidate], Dict[str, Any]]:
        walls = []
        for r in range(self.n):
            for c in range(self.n):
                if cand.grid[r][c] == 1:
                    free_nb = sum(1 for nb in neighbors((r, c), self.n) if cand.grid[nb[0]][nb[1]] == 0)
                    if free_nb >= 2:
                        walls.append((r, c, free_nb))
        if not walls:
            return None, {"operator": "loop_flip", "status": "failed", "reason": "no_wall_with_two_free_neighbors"}
        walls.sort(key=lambda x: (-x[2], self.rng.random()))
        r, c, _ = walls[0]
        child = cand.clone()
        child.grid[r][c] = 0
        child.transform_history.append({"operator": "loop_flip", "status": "success", "cell": [r, c]})
        return child, child.transform_history[-1]

    def branch_grow(self, cand: MazeCandidate, before_metrics: Dict[str, Any]) -> Tuple[Optional[MazeCandidate], Dict[str, Any]]:
        cfg = self.config["transform"].get("branch_grow", {})
        free = free_cells(cand.grid)
        self.rng.shuffle(free)
        candidates = [(cell, nb) for cell in free for nb in neighbors(cell, self.n) if cand.grid[nb[0]][nb[1]] == 1]
        if not candidates:
            return None, {"operator": "branch_grow", "status": "failed", "reason": "no_adjacent_wall"}
        base, cur = self.rng.choice(candidates)
        child = cand.clone()
        opened = []
        for _ in range(self.rng.randint(1, int(cfg.get("max_new_cells", 3)))):
            if cur in [child.start, child.goal]:
                break
            child.grid[cur[0]][cur[1]] = 0
            opened.append(cur)
            wall_nbs = [nb for nb in neighbors(cur, self.n) if child.grid[nb[0]][nb[1]] == 1]
            if not wall_nbs or self.rng.random() < 0.45:
                break
            cur = self.rng.choice(wall_nbs)
        if not opened:
            return None, {"operator": "branch_grow", "status": "failed", "reason": "opened_zero_cells"}
        child.transform_history.append({"operator": "branch_grow", "status": "success", "opened": [list(x) for x in opened], "base": list(base)})
        return child, child.transform_history[-1]

    def shortcut_block(self, cand: MazeCandidate, before_metrics: Dict[str, Any]) -> Tuple[Optional[MazeCandidate], Dict[str, Any]]:
        path = shortest_path(cand.grid, cand.start, cand.goal)
        candidates = [cell for cell in free_cells(cand.grid) if cell not in [cand.start, cand.goal]]
        path_set = set(path[1:-1])
        candidates.sort(key=lambda x: (0 if x in path_set else 1, self.rng.random()))
        for cell in candidates[:60]:
            child = cand.clone()
            child.grid[cell[0]][cell[1]] = 1
            dist, _ = bfs_dist(child.grid, child.start, child.goal)
            if child.goal in dist:
                child.transform_history.append({"operator": "shortcut_block", "status": "success", "blocked": list(cell)})
                return child, child.transform_history[-1]
        return None, {"operator": "shortcut_block", "status": "failed", "reason": "disconnects_start_goal"}

    def cell_flip_random(self, cand: MazeCandidate, before_metrics: Dict[str, Any]) -> Tuple[Optional[MazeCandidate], Dict[str, Any]]:
        cfg = self.config["transform"].get("cell_flip_random", {})
        child = cand.clone()
        cells = [(r, c) for r in range(self.n) for c in range(self.n) if (r, c) not in [child.start, child.goal]]
        self.rng.shuffle(cells)
        flips = []
        for cell in cells[: self.rng.randint(1, int(cfg.get("max_flips", 2)))]:
            child.grid[cell[0]][cell[1]] = 1 - child.grid[cell[0]][cell[1]]
            flips.append(cell)
        if not flips:
            return None, {"operator": "cell_flip_random", "status": "failed", "reason": "no_flippable_cell"}
        child.transform_history.append({"operator": "cell_flip_random", "status": "success", "flips": [list(x) for x in flips]})
        return child, child.transform_history[-1]


def choose_transform_operator(config: Dict[str, Any], rng: random.Random, metrics: Dict[str, Any], archive: Any, consecutive_failures: int) -> str:
    base = dict(config["transform"]["operator_weights"])
    adapt = config["transform"].get("adaptive_weights", {})
    bin_info = archive.assign_bin(metrics)
    if bin_info["cycle_rank_bin"] == "tree":
        base["loop_flip"] = base.get("loop_flip", 1.0) * adapt.get("cycle_low_loop_flip_multiplier", 2.0)
    if bin_info["macro_endpoint_bin"] == "low_endpoint":
        base["branch_grow"] = base.get("branch_grow", 1.0) * adapt.get("endpoint_low_branch_grow_multiplier", 2.0)
    if bin_info["bfs_len_bin"] == "short":
        base["shortcut_block"] = base.get("shortcut_block", 1.0) * adapt.get("bfs_short_shortcut_block_multiplier", 2.0)
    if consecutive_failures > 0:
        base["cell_flip_random"] = base.get("cell_flip_random", 0.5) * (adapt.get("failure_cell_flip_multiplier", 2.0) ** consecutive_failures)
    return choose_weighted(rng, base)


def metric_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    keys = ["rect_room_compressed_cycle_rank", "corridor_compressed_cycle_rank", "raw_cycle_rank", "bfs_len", "rect_room_macro_choice_count", "rect_room_macro_endpoint_count", "corridor_compressed_endpoint_count", "corridor_compressed_choice_count", "free_ratio", "reachable_free_ratio", "dead_end_count", "room_count"]
    out: Dict[str, Any] = {}
    for k in keys:
        a, b = before.get(k), after.get(k)
        out[k] = b - a if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
    return out


def render_maze(cand: MazeCandidate, metrics: Dict[str, Any], bin_info: Dict[str, Any], out_path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return
    grid = np.array(cand.grid)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(grid, vmin=0, vmax=1)
    sr, sc = cand.start
    gr, gc = cand.goal
    ax.scatter([sc], [sr], marker="o", s=90, label="S")
    ax.scatter([gc], [gr], marker="*", s=120, label="G")
    ax.set_xticks(range(len(cand.grid)))
    ax.set_yticks(range(len(cand.grid)))
    ax.grid(True, linewidth=0.4)
    ax.set_title(title[:100], fontsize=8)
    ax.text(0.02, -0.12, f"bin={bin_info.get('bin_key')} bfs={metrics.get('bfs_len')} rcr={metrics.get('rect_room_compressed_cycle_rank')} mc={metrics.get('rect_room_macro_choice_count')} me={metrics.get('rect_room_macro_endpoint_count')} rawcr={metrics.get('raw_cycle_rank')}", transform=ax.transAxes, fontsize=7)
    ax.legend(loc="upper right", fontsize=6)
    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_default_config_path() -> Path:
    here = Path(__file__).resolve()
    root = here.parents[3] if len(here.parents) >= 4 else Path.cwd()
    for p in [root / DEFAULT_CONFIG_REL, Path("/mnt/data") / DEFAULT_CONFIG_REL, Path(DEFAULT_CONFIG_REL)]:
        if p.exists():
            return p
    return Path(DEFAULT_CONFIG_REL)


def resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_root:
        return Path(args.output_root)
    here = Path(__file__).resolve()
    root = here.parents[3] if len(here.parents) >= 4 else Path.cwd()
    return root / DEFAULT_OUTPUT_ROOT_REL


def apply_config_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    if args.save_visualizations:
        cfg.setdefault("visualization", {})["enabled"] = True
    if args.max_visual_samples is not None:
        cfg.setdefault("visualization", {})["max_samples"] = int(args.max_visual_samples)
        cfg.setdefault("runtime", {})["visual_sample_count"] = int(args.max_visual_samples)
    if getattr(args, "target_total_accepted", None) is not None:
        cfg.setdefault("runtime", {})["target_total_accepted"] = int(args.target_total_accepted)
    if getattr(args, "stop_when_all_bins_reach_quota", False):
        cfg.setdefault("runtime", {})["stop_when_all_bins_reach_quota"] = True
    return cfg


def initialize_summary() -> Dict[str, Any]:
    return {
        "n_candidates_evaluated": 0,
        "n_accepted": 0,
        "generator_seeds": Counter(),
        "generator_accepted": Counter(),
        "generator_bins": defaultdict(Counter),
        "operator_attempts": Counter(),
        "operator_successes": Counter(),
        "operator_failures": Counter(),
        "operator_accepted_after": Counter(),
        "operator_rect_choice_delta": defaultdict(list),
        "operator_rect_choice_delta_sign": defaultdict(Counter),
        "reject_reasons": Counter(),
        "quality_failures": Counter(),
        "transform_failure_reasons": Counter(),
        "accepted_after_transform_count": 0,
        "accepted_after_flipper_count": 0,
        "metrics": defaultdict(list),
        "start_goal_fallback_count": 0,
    }


def update_metric_distribution(summary: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    keys = [
        "raw_cycle_rank", "raw_endpoint_cell_count", "raw_choice_cell_count", "raw_vertex_count", "raw_edge_count",
        "corridor_compressed_cycle_rank", "corridor_compressed_endpoint_count", "corridor_compressed_choice_count", "corridor_compressed_vertex_count", "corridor_compressed_edge_count",
        "rect_room_count", "rect_room_area_sum", "tiny_2x2_room_count", "elongated_room_count",
        "rect_room_compressed_cycle_rank", "rect_room_macro_choice_count", "rect_room_macro_endpoint_count", "rect_room_macro_vertex_count", "rect_room_macro_edge_count",
        "rect_room_compressed_cycle_rank_before", "rect_room_macro_choice_count_before", "rect_room_macro_endpoint_count_before",
        "parallel_macro_edge_count", "room_pair_parallel_edge_count", "macro_cycle_from_parallel_room_edges",
        "internal_rect_room_cycle_rank_sum", "expected_cycle_rank_after_rect_room_internal_compression", "rect_overcompression_residual", "old_overcompression_residual",
        "obstructed_open_component_count", "non_rect_open_component_count", "uncovered_open_area_count", "terminal_inside_room_count",
        "macro_vertex_count", "macro_edge_count", "macro_room_component_count",
        "bfs_len", "free_ratio", "reachable_free_ratio", "room_count", "two_by_two_free_block_count", "dead_end_count",
    ]
    for k in keys:
        v = metrics.get(k)
        if isinstance(v, (int, float)):
            summary["metrics"][k].append(float(v))


def make_accepted_record(cand: MazeCandidate, metrics: Dict[str, Any], bin_info: Dict[str, Any], acceptance: Dict[str, Any], start_goal_debug: Dict[str, Any]) -> Dict[str, Any]:
    return {"maze_id": cand.candidate_id, "grid": cand.grid, "start": list(cand.start), "goal": list(cand.goal), "origin_generator": cand.origin_generator, "origin_seed_id": cand.origin_seed_id, "transform_history": cand.transform_history, "start_goal_assignment": start_goal_debug, "metrics": metrics, "bin": bin_info, "acceptance": acceptance}










def select_visualization_records(config: Dict[str, Any], accepted_records: List[Tuple[MazeCandidate, Dict[str, Any], Dict[str, Any], str]], archive_report: Dict[str, Any]) -> Tuple[List[Tuple[MazeCandidate, Dict[str, Any], Dict[str, Any], str, str]], Dict[str, Any]]:
    vis_cfg = config.get("visualization", {})
    max_samples = int(vis_cfg.get("max_samples", config.get("runtime", {}).get("visual_sample_count", 24)))
    max_per_bin = int(vis_cfg.get("max_per_bin", 2))
    max_per_origin = int(vis_cfg.get("max_per_origin_seed", 1))
    by_bin: Dict[str, List[Tuple[MazeCandidate, Dict[str, Any], Dict[str, Any], str]]] = defaultdict(list)
    by_gen: Dict[str, List[Tuple[MazeCandidate, Dict[str, Any], Dict[str, Any], str]]] = defaultdict(list)
    for rec in accepted_records:
        by_bin[rec[2]["bin_key"]].append(rec)
        by_gen[rec[0].origin_generator].append(rec)
    selected: List[Tuple[MazeCandidate, Dict[str, Any], Dict[str, Any], str, str]] = []
    used_origin: Counter[str] = Counter()
    used_bin: Counter[str] = Counter()
    def try_add(rec: Tuple[MazeCandidate, Dict[str, Any], Dict[str, Any], str], category: str) -> bool:
        if len(selected) >= max_samples:
            return False
        cand, metrics, bin_info, title = rec
        if used_origin[cand.origin_seed_id] >= max_per_origin:
            return False
        if used_bin[bin_info["bin_key"]] >= max_per_bin:
            return False
        selected.append((cand, metrics, bin_info, title, category))
        used_origin[cand.origin_seed_id] += 1
        used_bin[bin_info["bin_key"]] += 1
        return True
    # One per non-empty bin first.
    for key in sorted(by_bin.keys()):
        if len(selected) >= max_samples:
            break
        for rec in by_bin[key]:
            if try_add(rec, "by_bin"):
                break
    # Sparse bins if available.
    if vis_cfg.get("include_sparse_bins", True):
        for row in archive_report.get("sparse_bins", []):
            for rec in by_bin.get(row["bin_key"], []):
                if try_add(rec, "sparse_bins"):
                    break
    # Overfull bins.
    if vis_cfg.get("include_overfull_bins", True):
        for row in archive_report.get("overfull_bins", []):
            for rec in by_bin.get(row["bin_key"], []):
                if try_add(rec, "overfull_bins"):
                    break
    # Generator diversity fill.
    for gen in sorted(by_gen.keys()):
        for rec in by_gen[gen]:
            if len(selected) >= max_samples:
                break
            if try_add(rec, "by_generator"):
                break
    macro_choice_dist = Counter()
    macro_endpoint_dist = Counter()
    for _, _, bin_info, _, _ in selected:
        parts = bin_info["bin_key"].split("|")
        if len(parts) == 4:
            macro_choice_dist[parts[2]] += 1
            macro_endpoint_dist[parts[3]] += 1
    report = {"sampling_policy": vis_cfg.get("sampling_policy", "stratified_by_bin"), "n_visualized": len(selected), "n_bins_visualized": len({x[2]["bin_key"] for x in selected}), "max_same_origin_seed_seen": max(used_origin.values()) if used_origin else 0, "macro_choice_bin_distribution": dict(macro_choice_dist), "macro_endpoint_bin_distribution": dict(macro_endpoint_dist), "selected": []}
    for idx, (cand, metrics, bin_info, title, category) in enumerate(selected, 1):
        report["selected"].append({"rank": idx, "category": category, "maze_id": cand.candidate_id, "origin_generator": cand.origin_generator, "origin_seed_id": cand.origin_seed_id, "bin_key": bin_info["bin_key"], "bfs_len": metrics.get("bfs_len"), "rect_room_compressed_cycle_rank": metrics.get("rect_room_compressed_cycle_rank"), "rect_room_macro_choice_count": metrics.get("rect_room_macro_choice_count"), "rect_room_macro_endpoint_count": metrics.get("rect_room_macro_endpoint_count"), "raw_cycle_rank": metrics.get("raw_cycle_rank")})
    return selected, report


def safe_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s)





def default_config() -> Dict[str, Any]:
    return {
        "grid": {"size": 8, "start": [0, 0], "goal": [7, 7], "start_goal_policy": "sample_by_bfs_bin"},
        "start_goal": {"pair_sampling_mode": "all_pairs", "max_pair_count": 4096, "pair_sample_count": 64, "target_bfs_bin_policy": "prefer_sparse_bins", "fallback_policy": "random_free_pair", "min_bfs_len": 3},
        "seed_generators": {
            "weights": {"random_wall_generator": 1.0, "dfs_tree_generator": 1.0, "loose_path_first_generator": 1.0, "room_biased_generator": 1.0, "template_mutation_generator": 0.25},
            "random_wall_generator": {"wall_prob_min": 0.18, "wall_prob_max": 0.48},
            "dfs_tree_generator": {"branch_open_prob": 0.25, "extra_open_prob": 0.08},
            "loose_path_first_generator": {"max_path_attempts": 160, "extra_open_prob": 0.16},
            "room_biased_generator": {"room_count_min": 1, "room_count_max": 3, "room_size_min": 2, "room_size_max": 4, "connector_open_prob": 0.08},
            "template_mutation_generator": {"base": "loose_path_first_generator", "initial_mutations": 4},
        },
        "transform": {"max_transform_steps": 6, "max_consecutive_transform_failures": 3, "continue_after_reject": True, "operator_weights": {"loop_flip": 1.0, "branch_grow": 1.0, "shortcut_block": 1.0, "cell_flip_random": 0.35}, "adaptive_weights": {"cycle_low_loop_flip_multiplier": 2.0, "endpoint_low_branch_grow_multiplier": 2.0, "bfs_short_shortcut_block_multiplier": 2.0, "failure_cell_flip_multiplier": 2.0}, "branch_grow": {"max_new_cells": 3}, "cell_flip_random": {"max_flips": 2}},
        "archive": {"cycle_rank_bins": [{"name": "tree", "range": [0, 0]}, {"name": "single_loop", "range": [1, 1]}, {"name": "multi_loop", "range": [2, 999]}], "bfs_len_bins": [{"name": "short", "range": [1, 10]}, {"name": "medium", "range": [11, 20]}, {"name": "long", "range": [21, 999]}], "macro_choice_bins": [{"name": "low_choice", "range": [0, 2]}, {"name": "mid_choice", "range": [3, 5]}, {"name": "high_choice", "range": [6, 999]}], "macro_endpoint_bins": [{"name": "low_endpoint", "range": [0, 2]}, {"name": "mid_endpoint", "range": [3, 6]}, {"name": "high_endpoint", "range": [7, 999]}], "bin_quota_default": 30, "bin_quotas": {}},
        "acceptance": {"epsilon": 1.0, "alpha": 1.0, "hard_bin_cap_ratio": 2.0},
        "quality_gate": {"min_reachable_free_ratio": 0.90, "min_bfs_len": 3, "min_start_free_degree": 1, "min_free_ratio": 0.20, "max_free_ratio": 0.85},
        "diversity": {"reject_exact_duplicate": True, "max_same_origin_seed_per_bin": 2, "enable_near_duplicate_hamming_check": False, "near_duplicate_hamming_threshold": 2},
        "room_detection": {"room_min_component_size": 3, "room_high_degree_threshold": 3},
        "macro_extraction": {"enable_room_aware_compression": True, "open_area_by_2x2": True, "open_area_by_degree_density": True, "degree_density_min_degree": 3, "degree_density_min_local_free_count": 4, "degree_density_neighborhood": "8"},
        "rect_room_extraction": {"enabled": True, "min_w": 2, "min_h": 2, "selection": "max_non_overlapping_greedy", "bbox_convention": "inclusive", "elongated_aspect_ratio": 3.0},
        "debug": {"cycle_rank_close_threshold": 2.0, "overcompression_residual_threshold": 2.0, "high_choice_sparse_threshold": 0.05, "generator_dominance_threshold": 0.6, "parallel_edge_loss_raw_cycle_threshold": 5.0},
        "visualization": {"enabled": False, "sampling_policy": "stratified_by_bin", "max_samples": 48, "max_per_bin": 2, "max_per_origin_seed": 1, "include_sparse_bins": True, "include_overfull_bins": True},
        "runtime": {"visual_sample_count": 48, "candidate_debug_limit": 5000, "decision_debug_limit": 500, "rejected_example_limit": 200, "target_total_accepted": None, "stop_when_all_bins_reach_quota": False, "skip_expensive_extractors_on_quality_fail": True},
        "output": {"save_accepted_grid": True},
    }


# -----------------------------------------------------------------------------
# 3.0.4(4) Rect-room macro graph test mode
# -----------------------------------------------------------------------------

class FullRectRoomMacroGraphExtractor:
    """Full rect-room macro graph extractor used by test_rect_room_macro.

    It contracts selected rectangle rooms into room nodes, traces non-room
    corridors, preserves parallel macro edges, and computes multigraph cycle rank.
    It is a test/evaluator component, not an exact maze construction generator.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        local = copy.deepcopy(config)
        local.setdefault("rect_room_extraction", {})["cheap_metrics_only"] = False
        self.room_extractor = RectRoomExtractor(local)

    @staticmethod
    def _norm_edge_pair(a: Any, b: Any) -> Tuple[str, str]:
        return tuple(sorted((str(a), str(b))))  # stable for debug only

    def _build_contracted_graph(self, grid: Grid, start: Cell, goal: Cell, rooms: List[Dict[str, Any]], main_cells: set[Cell]) -> Dict[str, Any]:
        n = len(grid)
        room_cells: set[Cell] = set()
        cell_to_room: Dict[Cell, str] = {}
        room_by_id: Dict[str, Dict[str, Any]] = {}
        for room in rooms:
            rid = room["room_id"]
            room_by_id[rid] = room
            for x in room["covered_cells"]:
                cell = tuple(x)
                room_cells.add(cell)
                cell_to_room[cell] = rid
        def node_of(cell: Cell) -> Any:
            return cell_to_room.get(cell, cell)
        edge_records: List[Dict[str, Any]] = []
        dedup_direct_room: set[Tuple[str, str]] = set()
        dedup_room_cell: set[Tuple[str, Cell]] = set()
        dedup_cell_cell: set[Tuple[Cell, Cell]] = set()
        for cell in sorted(main_cells):
            r, c = cell
            for nb in [(r + 1, c), (r, c + 1)]:
                if not in_grid(grid, nb[0], nb[1]) or nb not in main_cells:
                    continue
                a, b = node_of(cell), node_of(nb)
                if a == b:
                    continue
                if isinstance(a, str) and isinstance(b, str):
                    key = tuple(sorted((a, b)))
                    if key in dedup_direct_room:
                        continue
                    dedup_direct_room.add(key)
                    kind = "room-room-adjacent"
                elif isinstance(a, str) or isinstance(b, str):
                    rid = a if isinstance(a, str) else b
                    outside = b if isinstance(a, str) else a
                    key2 = (rid, outside)
                    if key2 in dedup_room_cell:
                        continue
                    dedup_room_cell.add(key2)
                    kind = "room-corridor-boundary"
                else:
                    key3 = tuple(sorted((a, b)))
                    if key3 in dedup_cell_cell:
                        continue
                    dedup_cell_cell.add(key3)
                    kind = "cell-cell"
                eid = f"he_{len(edge_records):04d}"
                edge_records.append({"edge_id": eid, "a": a, "b": b, "cell_a": cell, "cell_b": nb, "kind": kind})
        adj: Dict[Any, List[Tuple[Any, str]]] = defaultdict(list)
        for e in edge_records:
            adj[e["a"]].append((e["b"], e["edge_id"]))
            adj[e["b"]].append((e["a"], e["edge_id"]))
        return {"room_cells": room_cells, "cell_to_room": cell_to_room, "room_by_id": room_by_id, "half_edges": edge_records, "adj": adj}

    def extract(self, grid: Grid, start: Cell, goal: Cell) -> Dict[str, Any]:
        dist, _ = bfs_dist_grid(grid, start)
        main_cells = set(dist.keys())
        warnings: List[str] = []
        if not main_cells or goal not in main_cells:
            return {"status": "failed", "warnings": ["start_goal_not_connected"], "selected_rect_rooms": [], "macro_nodes_before": [], "macro_edges_before": [], "macro_nodes_after": [], "macro_edges_after": [], "metrics": {}}
        room_info = self.room_extractor.extract(grid, start, goal)
        rooms = room_info.get("selected_rect_rooms", [])
        hg = self._build_contracted_graph(grid, start, goal, rooms, main_cells)
        adj = hg["adj"]
        cell_to_room = hg["cell_to_room"]
        room_ids = {r["room_id"] for r in rooms}
        terminal_inside: Dict[str, Optional[str]] = {"start": cell_to_room.get(start), "goal": cell_to_room.get(goal)}
        terminal_inside_room_count = sum(1 for v in terminal_inside.values() if v is not None)
        # Macro node set: all room nodes, outside terminals, and non-room contracted graph cells with degree != 2.
        macro_nodes: set[Any] = set(room_ids)
        for label, cell in [("start", start), ("goal", goal)]:
            if cell not in cell_to_room:
                macro_nodes.add(cell)
        for node in adj.keys():
            if isinstance(node, tuple) and len(adj[node]) != 2:
                macro_nodes.add(node)
        # Trace macro edges through degree-2 non-room corridor cells. Parallel edges are preserved.
        used_half_edges: set[str] = set()
        macro_edges: List[Dict[str, Any]] = []
        dangling = 0
        loop_without_node = 0
        for src in sorted(macro_nodes, key=lambda x: str(x)):
            for nb, eid0 in list(adj.get(src, [])):
                if eid0 in used_half_edges:
                    continue
                cur = nb
                prev = src
                edge_ids = [eid0]
                path_nodes = [src, cur]
                path_cells: List[List[int]] = []
                if isinstance(src, tuple):
                    path_cells.append(list(src))
                if isinstance(cur, tuple):
                    path_cells.append(list(cur))
                seen_nodes = {src}
                dst = None
                while True:
                    if cur in macro_nodes:
                        dst = cur
                        break
                    if cur in seen_nodes:
                        loop_without_node += 1
                        warnings.append("loop_without_macro_node")
                        break
                    seen_nodes.add(cur)
                    next_items = [(x, e) for x, e in adj.get(cur, []) if x != prev]
                    if len(next_items) != 1:
                        dangling += 1
                        warnings.append("dangling_or_ambiguous_corridor_trace")
                        break
                    nxt, eid = next_items[0]
                    edge_ids.append(eid)
                    prev, cur = cur, nxt
                    path_nodes.append(cur)
                    if isinstance(cur, tuple):
                        path_cells.append(list(cur))
                if dst is None or dst == src:
                    for e in edge_ids:
                        used_half_edges.add(e)
                    continue
                for e in edge_ids:
                    used_half_edges.add(e)
                u, v = src, dst
                kind = "room-room" if isinstance(u, str) and isinstance(v, str) else "room-corridor" if isinstance(u, str) or isinstance(v, str) else "corridor-corridor"
                macro_edges.append({
                    "edge_id": f"rme_{len(macro_edges):04d}",
                    "u": self._node_id(u),
                    "v": self._node_id(v),
                    "kind": kind,
                    "path_cells": path_cells,
                    "length": max(1, len(edge_ids)),
                    "source_h_edges": edge_ids,
                })
        node_rows_before = self._node_rows(macro_nodes, macro_edges, rooms, start, goal, terminal_inside)
        metrics_before = self._metrics(node_rows_before, macro_edges)
        terminal_counts = self._terminal_counts(node_rows_before, terminal_inside)
        # Degree-2 room compression, preserving rooms that contain S/G terminal metadata.
        node_rows_after, edges_after, compression_warnings = self._compress_degree2_rooms(node_rows_before, macro_edges)
        warnings.extend(compression_warnings)
        metrics_after = self._metrics(node_rows_after, edges_after)
        pair_counts = Counter(tuple(sorted((e["u"], e["v"]))) for e in macro_edges)
        parallel_macro_edge_count = sum(cnt for cnt in pair_counts.values() if cnt > 1)
        node_kind = {n["node_id"]: n.get("node_type") for n in node_rows_before}
        room_pair_parallel_edge_count = sum(cnt for pair, cnt in pair_counts.items() if cnt > 1 and all(node_kind.get(x) == "rect_room_anchor" for x in pair))
        selected_rooms = []
        for r in rooms:
            rid = r["room_id"]
            selected_rooms.append({k: v for k, v in r.items() if k != "covered_cells"} | {"covered_cells": r.get("covered_cells", [])})
        metrics = {
            **{k + "_before": v for k, v in metrics_before.items()},
            **metrics_after,
            "rect_room_count": len(rooms),
            "tiny_2x2_room_count": sum(1 for r in rooms if r.get("w") == 2 and r.get("h") == 2),
            "rect_room_area_sum": sum(r.get("area", 0) for r in rooms),
            "parallel_macro_edge_count": parallel_macro_edge_count,
            "room_pair_parallel_edge_count": room_pair_parallel_edge_count,
            "terminal_inside_room_count": terminal_inside_room_count,
            **terminal_counts,
        }
        return {
            "status": "ok" if not warnings else "warning",
            "warnings": sorted(set(warnings)),
            "selected_rect_rooms": selected_rooms,
            "macro_nodes_before": node_rows_before,
            "macro_edges_before": macro_edges,
            "macro_nodes_after": node_rows_after,
            "macro_edges_after": edges_after,
            "metrics": metrics,
            "trace_debug": {"dangling_trace_count": dangling, "loop_without_node_count": loop_without_node, "half_edge_count": len(hg["half_edges"])},
            "terminal_inside": terminal_inside,
        }

    @staticmethod
    def _node_id(x: Any) -> str:
        if isinstance(x, str):
            return x
        r, c = x
        return f"cell_{r}_{c}"

    def _node_rows(self, macro_nodes: set[Any], edges: List[Dict[str, Any]], rooms: List[Dict[str, Any]], start: Cell, goal: Cell, terminal_inside: Dict[str, Optional[str]]) -> List[Dict[str, Any]]:
        room_by_id = {r["room_id"]: r for r in rooms}
        deg = Counter()
        for e in edges:
            deg[e["u"]] += 1
            deg[e["v"]] += 1
        rows = []
        for node in sorted(macro_nodes, key=lambda x: str(x)):
            nid = self._node_id(node)
            is_room = isinstance(node, str)
            is_start = (node == start) or (is_room and terminal_inside.get("start") == node)
            is_goal = (node == goal) or (is_room and terminal_inside.get("goal") == node)
            if is_room:
                room = room_by_id.get(node, {})
                cell = room.get("anchor", [None, None])
                node_type = "rect_room_anchor"
                room_id = node
            else:
                cell = list(node)
                node_type = "terminal" if node in (start, goal) else "macro_cell_node"
                room_id = None
            rows.append({"node_id": nid, "node_type": node_type, "room_id": room_id, "cell": cell, "macro_degree": deg[nid], "is_start": bool(is_start), "is_goal": bool(is_goal)})
        return rows

    @staticmethod
    def _metrics(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, int]:
        v = len(nodes)
        e = len(edges)
        cr = max(0, e - v + (1 if v else 0))
        choice = sum(1 for n in nodes if n.get("macro_degree", 0) >= 3)
        endpoint = sum(1 for n in nodes if n.get("macro_degree", 0) == 1)
        return {
            "rect_room_macro_vertex_count": v,
            "rect_room_macro_edge_count": e,
            "rect_room_compressed_cycle_rank": cr,
            "rect_room_macro_choice_count": choice,
            "rect_room_macro_endpoint_count": endpoint,
        }

    @staticmethod
    def _terminal_counts(nodes: List[Dict[str, Any]], terminal_inside: Dict[str, Optional[str]]) -> Dict[str, int]:
        by_id = {n["node_id"]: n for n in nodes}
        counts = {"terminal_endpoint_count": 0, "terminal_choice_count": 0}
        for label in ["start", "goal"]:
            rid = terminal_inside.get(label)
            if rid:
                nid = rid
            else:
                # outside terminal has cell_* id, found through flag
                row = next((n for n in nodes if (label == "start" and n.get("is_start")) or (label == "goal" and n.get("is_goal"))), None)
                nid = row["node_id"] if row else None
            row = by_id.get(nid) if nid else None
            deg = row.get("macro_degree", 0) if row else 0
            if deg >= 2:
                counts["terminal_choice_count"] += 1
            else:
                counts["terminal_endpoint_count"] += 1
        return counts

    def _compress_degree2_rooms(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
        warnings: List[str] = []
        node_map = {n["node_id"]: dict(n) for n in nodes}
        edge_list = [dict(e) for e in edges]
        changed = True
        while changed:
            changed = False
            deg = Counter()
            inc = defaultdict(list)
            for idx, e in enumerate(edge_list):
                deg[e["u"]] += 1; deg[e["v"]] += 1
                inc[e["u"]].append(idx); inc[e["v"]].append(idx)
            candidate = None
            for nid, row in list(node_map.items()):
                if row.get("node_type") == "rect_room_anchor" and deg[nid] == 2 and not row.get("is_start") and not row.get("is_goal"):
                    candidate = nid; break
            if candidate is None:
                break
            idxs = inc[candidate]
            if len(idxs) != 2:
                warnings.append("degree2_compression_warning_parallel_or_loop_room")
                break
            e1, e2 = edge_list[idxs[0]], edge_list[idxs[1]]
            a = e1["v"] if e1["u"] == candidate else e1["u"]
            b = e2["v"] if e2["u"] == candidate else e2["u"]
            # remove higher index first
            for i in sorted(idxs, reverse=True):
                edge_list.pop(i)
            edge_list.append({"edge_id": f"rme_c_{len(edge_list):04d}", "u": a, "v": b, "kind": "compressed_degree2_room", "path_cells": [], "length": max(1, e1.get("length", 1) + e2.get("length", 1))})
            node_map.pop(candidate, None)
            changed = True
        # recompute degrees
        deg = Counter()
        for e in edge_list:
            deg[e["u"]] += 1; deg[e["v"]] += 1
        rows = []
        for nid, row in node_map.items():
            row = dict(row)
            row["macro_degree"] = deg[nid]
            rows.append(row)
        return sorted(rows, key=lambda x: x["node_id"]), edge_list, warnings


def parse_ascii_maze(lines: List[str]) -> Tuple[Grid, Cell, Cell]:
    width = max(len(x) for x in lines)
    grid: Grid = []
    start: Optional[Cell] = None
    goal: Optional[Cell] = None
    for r, raw in enumerate(lines):
        row = []
        for c, ch in enumerate(raw.ljust(width, "#")):
            if ch == "#":
                row.append(1)
            elif ch in ".SG":
                row.append(0)
                if ch == "S": start = (r, c)
                if ch == "G": goal = (r, c)
            else:
                row.append(1)
        grid.append(row)
    return grid, start, goal


def fixture_expected_rooms(*rooms):
    out=[]
    for bbox in rooms:
        r0,c0,r1,c1=bbox
        w=c1-c0+1; h=r1-r0+1
        out.append({"bbox": list(bbox), "w": w, "h": h, "area": w*h, "anchor": [(r0+r1)//2, (c0+c1)//2]})
    return out


def rect_hard_fixtures() -> List[Dict[str, Any]]:
    F=[]
    def add(test_id, ascii_lines, rooms, before, after, terminals, parallel=(0,0), hints=None, start=None, goal=None):
        F.append({"test_id": test_id, "hard_or_soft": "hard", "ascii_maze": ascii_lines, "start": start, "goal": goal, "expected_rooms": fixture_expected_rooms(*rooms), "expected_metrics_before": before, "expected_metrics_after": after, "expected_parallel_edges": {"parallel_macro_edge_count": parallel[0], "room_pair_parallel_edge_count": parallel[1]}, "terminal_expectations": terminals, "failure_modes": [], "inspect_hints": hints or []})
    add("A1_single_2x2_room", ["########","#S...G##","###..###","########"], [(1,3,2,4)], {"rect_room_macro_vertex_count":3,"rect_room_macro_edge_count":2,"rect_room_compressed_cycle_rank":0,"rect_room_macro_choice_count":0,"rect_room_macro_endpoint_count":2}, {"rect_room_macro_vertex_count":2,"rect_room_macro_edge_count":1,"rect_room_compressed_cycle_rank":0,"rect_room_macro_choice_count":0,"rect_room_macro_endpoint_count":2}, {"terminal_inside_room_count":0,"terminal_endpoint_count":2,"terminal_choice_count":0}, hints=["room extraction","degree-2 room compression","terminal endpoint handling"])
    add("A2_single_2xN_room", ["##########","#S.....G##","###....###","##########"], [(1,3,2,6)], {"rect_room_macro_vertex_count":3,"rect_room_macro_edge_count":2,"rect_room_compressed_cycle_rank":0,"rect_room_macro_choice_count":0,"rect_room_macro_endpoint_count":2}, {"rect_room_macro_vertex_count":2,"rect_room_macro_edge_count":1,"rect_room_compressed_cycle_rank":0,"rect_room_macro_choice_count":0,"rect_room_macro_endpoint_count":2}, {"terminal_inside_room_count":0,"terminal_endpoint_count":2,"terminal_choice_count":0})
    add("A3_single_3x3_room_start_inside", ["########","#...G###","#.S.####","#...####","########"], [(1,1,3,3)], {"rect_room_macro_vertex_count":2,"rect_room_macro_edge_count":1,"rect_room_compressed_cycle_rank":0,"rect_room_macro_choice_count":0,"rect_room_macro_endpoint_count":2}, {"rect_room_macro_vertex_count":2,"rect_room_macro_edge_count":1,"rect_room_compressed_cycle_rank":0,"rect_room_macro_choice_count":0,"rect_room_macro_endpoint_count":2}, {"terminal_inside_room_count":1,"terminal_endpoint_count":2,"terminal_choice_count":0})
    add("A4_two_rooms_single_corridor", ["#########","#S.###.G#","#.......#","#########"], [(1,1,2,2),(1,6,2,7)], {"rect_room_macro_vertex_count":2,"rect_room_macro_edge_count":1,"rect_room_compressed_cycle_rank":0,"rect_room_macro_choice_count":0,"rect_room_macro_endpoint_count":2}, {"rect_room_macro_vertex_count":2,"rect_room_macro_edge_count":1,"rect_room_compressed_cycle_rank":0,"rect_room_macro_choice_count":0,"rect_room_macro_endpoint_count":2}, {"terminal_inside_room_count":2,"terminal_endpoint_count":2,"terminal_choice_count":0})
    for tid in ["A5_two_rooms_two_parallel_corridors","B1_obstructed_open_area_must_not_be_single_room"]:
        add(tid, ["#########","#S.....G#","#..###..#","#.......#","#########"], [(1,1,3,2),(1,6,3,7)], {"rect_room_macro_vertex_count":2,"rect_room_macro_edge_count":2,"rect_room_compressed_cycle_rank":1,"rect_room_macro_choice_count":0,"rect_room_macro_endpoint_count":0}, {"rect_room_macro_vertex_count":2,"rect_room_macro_edge_count":2,"rect_room_compressed_cycle_rank":1,"rect_room_macro_choice_count":0,"rect_room_macro_endpoint_count":0}, {"terminal_inside_room_count":2,"terminal_endpoint_count":0,"terminal_choice_count":2}, parallel=(2,2), hints=["parallel macro edge preservation","multigraph cycle rank"])
    add("B2_top_room_bottom_2xN_room_degree2_compression", ["#########","#.......#","#.......#","####..###","####..###","####..###","#########"], [(1,1,2,7),(3,4,5,5)], {"rect_room_macro_vertex_count":2,"rect_room_macro_edge_count":1,"rect_room_compressed_cycle_rank":0,"rect_room_macro_choice_count":0,"rect_room_macro_endpoint_count":2}, {"rect_room_macro_vertex_count":2,"rect_room_macro_edge_count":1,"rect_room_compressed_cycle_rank":0,"rect_room_macro_choice_count":0,"rect_room_macro_endpoint_count":2}, {"terminal_inside_room_count":2,"terminal_endpoint_count":2,"terminal_choice_count":0}, start=(1,1), goal=(5,5))
    add("B3_three_rooms_and_central_corridor_choice", ["#############","#####..######","#####..######","######.######","#...........#","#..###.###..#","######.######","#####..######","#####..######","#############"], [(1,5,2,6),(4,1,5,2),(4,10,5,11),(7,5,8,6)], {"rect_room_macro_vertex_count":5,"rect_room_macro_edge_count":4,"rect_room_compressed_cycle_rank":0,"rect_room_macro_choice_count":1,"rect_room_macro_endpoint_count":4}, {"rect_room_macro_vertex_count":5,"rect_room_macro_edge_count":4,"rect_room_compressed_cycle_rank":0,"rect_room_macro_choice_count":1,"rect_room_macro_endpoint_count":4}, {"terminal_inside_room_count":2,"terminal_endpoint_count":2,"terminal_choice_count":0}, start=(4,1), goal=(4,11))
    add("B4_start_inside_room_goal_normal_endpoint", ["##########","#...######","#.S....G##","#...######","##########"], [(1,1,3,3)], {"rect_room_macro_vertex_count":2,"rect_room_macro_edge_count":1,"rect_room_compressed_cycle_rank":0,"rect_room_macro_choice_count":0,"rect_room_macro_endpoint_count":2}, {"rect_room_macro_vertex_count":2,"rect_room_macro_edge_count":1,"rect_room_compressed_cycle_rank":0,"rect_room_macro_choice_count":0,"rect_room_macro_endpoint_count":2}, {"terminal_inside_room_count":1,"terminal_endpoint_count":2,"terminal_choice_count":0})
    add("B5_room_room_room_chain_degree2_compression", ["########","#S.#####","#..#####","##.#####","#..#####","#..#####","##.#####","#.G#####","#..#####","########"], [(1,1,2,2),(4,1,5,2),(7,1,8,2)], {"rect_room_macro_vertex_count":3,"rect_room_macro_edge_count":2,"rect_room_compressed_cycle_rank":0,"rect_room_macro_choice_count":0,"rect_room_macro_endpoint_count":2}, {"rect_room_macro_vertex_count":2,"rect_room_macro_edge_count":1,"rect_room_compressed_cycle_rank":0,"rect_room_macro_choice_count":0,"rect_room_macro_endpoint_count":2}, {"terminal_inside_room_count":2,"terminal_endpoint_count":2,"terminal_choice_count":0}, start=(1,1), goal=(7,2))
    add("C4_adversarial_non_rect_open_area_not_single_room", ["########","#.....##","#.....##","###...##","###...##","########"], [(1,3,4,5),(1,1,2,2)], {"rect_room_macro_vertex_count":2,"rect_room_macro_edge_count":1,"rect_room_compressed_cycle_rank":0,"rect_room_macro_choice_count":0,"rect_room_macro_endpoint_count":2}, {"rect_room_macro_vertex_count":2,"rect_room_macro_edge_count":1,"rect_room_compressed_cycle_rank":0,"rect_room_macro_choice_count":0,"rect_room_macro_endpoint_count":2}, {"terminal_inside_room_count":2,"terminal_endpoint_count":2,"terminal_choice_count":0}, start=(1,1), goal=(4,5))
    add("C5_adversarial_room_with_pillar_not_single_room", ["#########","#.......#","#...#...#","#.......#","#########"], [(1,1,3,3),(1,5,3,7)], {"rect_room_macro_vertex_count":2,"rect_room_macro_edge_count":2,"rect_room_compressed_cycle_rank":1,"rect_room_macro_choice_count":0,"rect_room_macro_endpoint_count":0}, {"rect_room_macro_vertex_count":2,"rect_room_macro_edge_count":2,"rect_room_compressed_cycle_rank":1,"rect_room_macro_choice_count":0,"rect_room_macro_endpoint_count":0}, {"terminal_inside_room_count":2,"terminal_endpoint_count":0,"terminal_choice_count":2}, parallel=(2,2), start=(1,1), goal=(3,7))
    add("D1_max_rect_tie_breaker_deterministic", ["#######","#S...##","#....##","#..####","#.G####","#######"], [(1,1,2,4),(3,1,4,2)], {"rect_room_macro_vertex_count":2,"rect_room_macro_edge_count":1,"rect_room_compressed_cycle_rank":0,"rect_room_macro_choice_count":0,"rect_room_macro_endpoint_count":2}, {"rect_room_macro_vertex_count":2,"rect_room_macro_edge_count":1,"rect_room_compressed_cycle_rank":0,"rect_room_macro_choice_count":0,"rect_room_macro_endpoint_count":2}, {"terminal_inside_room_count":2,"terminal_endpoint_count":2,"terminal_choice_count":0}, start=(1,1), goal=(4,2))
    add("D2_overlapping_candidates_greedy_selection", ["#########","#S.....##","#......##","#..######","#.G######","#########"], [(1,1,2,6),(3,1,4,2)], {"rect_room_macro_vertex_count":2,"rect_room_macro_edge_count":1,"rect_room_compressed_cycle_rank":0,"rect_room_macro_choice_count":0,"rect_room_macro_endpoint_count":2}, {"rect_room_macro_vertex_count":2,"rect_room_macro_edge_count":1,"rect_room_compressed_cycle_rank":0,"rect_room_macro_choice_count":0,"rect_room_macro_endpoint_count":2}, {"terminal_inside_room_count":2,"terminal_endpoint_count":2,"terminal_choice_count":0}, start=(1,1), goal=(4,2))
    return F


def rect_soft_fixtures() -> List[Dict[str, Any]]:
    return [
        {"test_id":"C1_random_wall_style_small_rooms_deadends","hard_or_soft":"soft","ascii_maze":["############","#S..#..#...#","#..##..#.#G#","#....#.....#","###..###..##","#....#..#..#","############"],"expected_ranges":{"rect_room_count":[2,5],"parallel_macro_edge_count":[0,2],"room_pair_parallel_edge_count":[0,2],"rect_room_macro_vertex_count_before":[4,10],"rect_room_macro_edge_count_before":[3,12],"rect_room_compressed_cycle_rank_before":[0,3],"rect_room_macro_choice_count_before":[1,4],"rect_room_macro_endpoint_count_before":[2,6],"rect_room_macro_vertex_count":[3,8],"rect_room_macro_edge_count":[2,10],"rect_room_compressed_cycle_rank":[0,3],"rect_room_macro_choice_count":[1,4],"rect_room_macro_endpoint_count":[2,6],"terminal_inside_room_count":[0,1],"terminal_endpoint_count":[1,2],"terminal_choice_count":[0,1]},"inspect_hints":["maximal rectangle extraction","endpoint extraction","macro graph coverage"]},
        {"test_id":"C2_room_biased_multiple_rooms_with_loop","hard_or_soft":"soft","ascii_maze":["###############","#S..###...###G#","#...###...###.#","#.............#","#...###...###.#","#...###...###.#","###############"],"expected_ranges":{"rect_room_count":[4,5],"parallel_macro_edge_count":[0,4],"room_pair_parallel_edge_count":[0,4],"rect_room_macro_vertex_count_before":[5,10],"rect_room_macro_edge_count_before":[5,12],"rect_room_compressed_cycle_rank_before":[1,3],"rect_room_macro_choice_count_before":[1,4],"rect_room_macro_endpoint_count_before":[1,4],"rect_room_macro_vertex_count":[3,8],"rect_room_macro_edge_count":[4,10],"rect_room_compressed_cycle_rank":[1,3],"rect_room_macro_choice_count":[1,4],"rect_room_macro_endpoint_count":[1,4],"terminal_inside_room_count":[1,2],"terminal_endpoint_count":[0,2],"terminal_choice_count":[0,2]},"inspect_hints":["room extraction tie-breaker","macro loop preservation","terminal classification"]},
        {"test_id":"C3_handdrawn_like_main_branch_deadend_loop_room","hard_or_soft":"soft","ascii_maze":["#############","#..S....#...#","#..###..#.#G#","#....#.....##","###..###..###","#....#.....##","#############"],"expected_ranges":{"rect_room_count":[2,5],"rect_room_compressed_cycle_rank_before":[1,999],"rect_room_macro_choice_count_before":[1,999],"rect_room_macro_endpoint_count_before":[1,999],"rect_room_compressed_cycle_rank":[1,999],"rect_room_macro_choice_count":[1,999],"rect_room_macro_endpoint_count":[1,999],"terminal_inside_room_count":[0,1],"terminal_endpoint_count":[0,2],"terminal_choice_count":[0,1]},"inspect_hints":["macro choice extraction","loop preservation","endpoint branch tracing"]},
    ]


def rect_all_fixtures(suite: str) -> List[Dict[str, Any]]:
    hard = rect_hard_fixtures()
    soft = rect_soft_fixtures()
    if suite == "hard": return hard
    if suite == "soft": return soft
    return hard + soft


def compare_rooms(expected: List[Dict[str, Any]], actual: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    failures=[]
    if len(expected) != len(actual):
        failures.append({"failed_field":"rect_room_count","expected":len(expected),"actual":len(actual)})
    for i, exp in enumerate(expected):
        if i >= len(actual):
            failures.append({"failed_field":f"room_{i}","expected":exp,"actual":None}); continue
        act = actual[i]
        for f in ["bbox","w","h","area","anchor"]:
            if act.get(f) != exp.get(f):
                failures.append({"failed_field":f"room_{i}.{f}","expected":exp.get(f),"actual":act.get(f)})
    return failures


def run_rect_fixture(config: Dict[str, Any], fixture: Dict[str, Any], out_dir: Path, save_visualizations: bool) -> Dict[str, Any]:
    grid, parsed_start, parsed_goal = parse_ascii_maze(fixture["ascii_maze"])
    # Explicit coordinates in the algorithm-designer fixture are authoritative.
    start_src = fixture.get("start") if fixture.get("start") is not None else (parsed_start if parsed_start is not None else (-1, -1))
    goal_src = fixture.get("goal") if fixture.get("goal") is not None else (parsed_goal if parsed_goal is not None else (-1, -1))
    start = tuple(start_src)
    goal = tuple(goal_src)
    if start == (-1, -1) or goal == (-1, -1):
        raise ValueError(f"fixture {fixture.get('test_id')} missing S/G and explicit coordinates")
    grid[start[0]][start[1]] = 0
    grid[goal[0]][goal[1]] = 0
    extractor = FullRectRoomMacroGraphExtractor(config)
    actual = extractor.extract(grid, start, goal)
    metrics = actual.get("metrics", {})
    rooms = [{k: v for k, v in r.items() if k != "covered_cells"} for r in actual.get("selected_rect_rooms", [])]
    case = {"test_id": fixture["test_id"], "hard_or_soft": fixture["hard_or_soft"], "start": list(start), "goal": list(goal), "status": None, "failures": [], "warnings": actual.get("warnings", []), "metrics": metrics, "actual_rooms": rooms, "actual_macro_nodes_before": actual.get("macro_nodes_before", []), "actual_macro_edges_before": actual.get("macro_edges_before", []), "actual_macro_nodes_after": actual.get("macro_nodes_after", []), "actual_macro_edges_after": actual.get("macro_edges_after", []), "inspect_hints": fixture.get("inspect_hints", [])}
    if fixture["hard_or_soft"] == "hard":
        failures = compare_rooms(fixture.get("expected_rooms", []), rooms)
        expected = {}
        expected.update({k + "_before": v for k, v in fixture.get("expected_metrics_before", {}).items()})
        expected.update(fixture.get("expected_metrics_after", {}))
        expected.update(fixture.get("expected_parallel_edges", {}))
        expected.update(fixture.get("terminal_expectations", {}))
        for k, exp in expected.items():
            act = metrics.get(k)
            if act != exp:
                failures.append({"failed_field": k, "expected": exp, "actual": act})
        case["failures"] = failures
        case["status"] = "PASS" if not failures and actual.get("status") in ("ok","warning") else "FAIL"
    else:
        failures=[]
        for k, rng in fixture.get("expected_ranges", {}).items():
            lo, hi = rng
            val = metrics.get(k)
            if val is None or not (lo <= val <= hi):
                failures.append({"field": k, "expected_range": rng, "actual": val})
        case["out_of_range_fields"] = failures
        case["status"] = "PASS_RANGE" if not failures else "OUT_OF_RANGE"
    if save_visualizations:
        render_rect_fixture_visuals(grid, start, goal, actual, out_dir / "sample_visualizations" / "rect_room_tests", fixture["test_id"])
    return case


def render_rect_fixture_visuals(grid: Grid, start: Cell, goal: Cell, actual: Dict[str, Any], out_dir: Path, test_id: str) -> None:
    ensure_dir(out_dir)
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
        arr = np.array(grid)
        def base(ax, title):
            ax.imshow(arr, cmap="gray_r", vmin=0, vmax=1)
            ax.scatter([start[1]],[start[0]],marker="o",s=80,label="S")
            ax.scatter([goal[1]],[goal[0]],marker="*",s=100,label="G")
            ax.set_xticks(range(len(grid[0]))); ax.set_yticks(range(len(grid))); ax.grid(True, linewidth=0.3)
            ax.set_title(title); ax.legend(loc="upper right", fontsize=6)
        fig, ax = plt.subplots(figsize=(6,5)); base(ax, test_id); fig.tight_layout(); fig.savefig(out_dir / f"{test_id}__maze.png", dpi=140); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6,5)); base(ax, f"{test_id} rect rooms")
        for room in actual.get("selected_rect_rooms", []):
            r0,c0,r1,c1 = room["bbox"]
            ax.add_patch(patches.Rectangle((c0-0.5,r0-0.5), c1-c0+1, r1-r0+1, fill=False, linewidth=2))
            ar,ac = room["anchor"]; ax.text(ac, ar, room["room_id"], color="red", fontsize=7, ha="center", va="center")
        fig.tight_layout(); fig.savefig(out_dir / f"{test_id}__rect_rooms_overlay.png", dpi=140); plt.close(fig)
        for phase, nodes_key, edges_key in [("before", "macro_nodes_before", "macro_edges_before"),("after", "macro_nodes_after", "macro_edges_after")]:
            fig, ax = plt.subplots(figsize=(6,5)); base(ax, f"{test_id} macro {phase}")
            node_pos={}
            for nrow in actual.get(nodes_key, []):
                cell=nrow.get("cell")
                if isinstance(cell, list) and cell[0] is not None:
                    node_pos[nrow["node_id"]]=(cell[0],cell[1])
                    ax.scatter([cell[1]],[cell[0]],s=120,facecolors="none",edgecolors="blue")
                    ax.text(cell[1],cell[0],nrow["node_id"].replace("cell_","c"),fontsize=6,color="blue")
            for e in actual.get(edges_key, []):
                u,v=e.get("u"),e.get("v")
                if u in node_pos and v in node_pos:
                    r0,c0=node_pos[u]; r1,c1=node_pos[v]
                    ax.plot([c0,c1],[r0,r1],linewidth=1.5)
            fig.tight_layout(); fig.savefig(out_dir / f"{test_id}__macro_graph_{phase}.png", dpi=140); plt.close(fig)
    except Exception:
        return


def run_test_rect_room_macro(args: argparse.Namespace) -> Path:
    config_path = Path(args.config_json) if args.config_json else build_default_config_path()
    config = load_json(config_path) if config_path.exists() else default_config()
    config.setdefault("rect_room_extraction", {})["cheap_metrics_only"] = False
    config["rect_room_extraction"]["enable_full_graph_extraction"] = True
    if args.save_visualizations:
        config.setdefault("visualization", {})["enabled"] = True
    run_name = args.run_name or f"test_rect_room_macro_{args.rect_room_test_suite}_{int(time.time())}"
    output_root = Path(args.output_root) if args.output_root else Path.cwd() / DEFAULT_OUTPUT_ROOT_REL
    out_dir = output_root / run_name
    ensure_dir(out_dir)
    # Avoid stale JSONL append data when rerunning the same run_name.
    for stale_name in [
        "rect_room_macro_test_cases.jsonl",
        "rect_room_macro_hard_failures.jsonl",
        "rect_room_macro_soft_diagnostics.jsonl",
        "rect_room_macro_actual_graphs.jsonl",
    ]:
        stale_path = out_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    fixtures = rect_all_fixtures(args.rect_room_test_suite)
    write_json(out_dir / "resolved_config.json", config)
    write_json(out_dir / "run_metadata.json", {"version": VERSION, "title": TITLE, "mode": "test_rect_room_macro", "suite": args.rect_room_test_suite, "run_name": run_name, "fixture_source": "algorithm_designer_spec_uploaded_as_pasted_text", "cheap_metrics_only": False})
    write_json(out_dir / "rect_room_macro_fixture_dump.json", fixtures)
    cases=[]; hard_failures=[]; soft_diags=[]; actual_graphs=[]
    for fx in iter_progress(fixtures, desc="Rect-room macro tests"):
        case = run_rect_fixture(config, fx, out_dir, args.save_visualizations)
        cases.append(case)
        append_jsonl(out_dir / "rect_room_macro_test_cases.jsonl", case)
        actual_graphs.append({"test_id": case["test_id"], "rooms": case["actual_rooms"], "macro_nodes_before": case["actual_macro_nodes_before"], "macro_edges_before": case["actual_macro_edges_before"], "macro_nodes_after": case["actual_macro_nodes_after"], "macro_edges_after": case["actual_macro_edges_after"], "warnings": case["warnings"]})
        append_jsonl(out_dir / "rect_room_macro_actual_graphs.jsonl", actual_graphs[-1])
        if fx["hard_or_soft"] == "hard" and case["status"] != "PASS":
            hard_failures.append(case); append_jsonl(out_dir / "rect_room_macro_hard_failures.jsonl", case)
        if fx["hard_or_soft"] == "soft":
            soft_diags.append(case); append_jsonl(out_dir / "rect_room_macro_soft_diagnostics.jsonl", case)
    hard = [c for c in cases if c["hard_or_soft"] == "hard"]
    soft = [c for c in cases if c["hard_or_soft"] == "soft"]
    feature_coverage = {"2x2_room": True, "2xN_room": True, "max_rect_room": True, "non_rect_rejection": True, "internal_wall_rejection": True, "parallel_edge": True, "degree2_compression": True, "terminal_inside_room": True, "macro_choice": True, "endpoint": True, "loop": True}
    warning_counts = Counter(w for c in cases for w in c.get("warnings", []))
    summary = {"suite": args.rect_room_test_suite, "n_tests": len(cases), "n_hard_tests": len(hard), "n_soft_tests": len(soft), "hard_pass": sum(1 for c in hard if c["status"] == "PASS"), "hard_fail": len(hard_failures), "soft_pass_range": sum(1 for c in soft if c["status"] == "PASS_RANGE"), "soft_out_of_range": sum(1 for c in soft if c["status"] == "OUT_OF_RANGE"), "error_count": sum(1 for c in cases if c["status"] == "ERROR"), "feature_coverage": feature_coverage, "warning_counts": dict(warning_counts), "hard_failure_ids": [c["test_id"] for c in hard_failures], "route_positioning": "3.0.4(4) test mode validates full rect-room macro graph extraction; generate mode remains non-DL archive data factory."}
    write_json(out_dir / "rect_room_macro_test_summary.json", summary)
    print(TEST_RECT_ROOM_TITLE)
    print("\n=== Test Suite Summary ===")
    for k in ["suite","n_tests","n_hard_tests","n_soft_tests","hard_pass","hard_fail","soft_pass_range","soft_out_of_range","error_count"]:
        print(f"{k:28s} {summary.get(k)}")
    print("\n=== Hard Failures ===")
    if not hard_failures:
        print("none")
    else:
        for c in hard_failures[:12]:
            first = c.get("failures", [{}])[0]
            print(f"{c['test_id']:55s} {first.get('failed_field')} expected={first.get('expected')} actual={first.get('actual')} hints={c.get('inspect_hints')}")
    print("\n=== Soft Diagnostics ===")
    for c in soft_diags:
        print(f"{c['test_id']:55s} {c['status']} out_of_range={len(c.get('out_of_range_fields', []))}")
    print("\n=== Feature Coverage ===")
    for k,v in feature_coverage.items(): print(f"{k:34s} {v}")
    print("\n=== Macro Graph Extraction Warnings ===")
    if warning_counts:
        for k,v in warning_counts.most_common(): print(f"{k:45s} {v}")
    else:
        print("none")
    print("\n=== Debug Hints ===")
    if hard_failures:
        print("[FAIL] rect-room macro graph hard assertions failed.")
        print("Do not trust generate-mode rect_room_compressed_cycle_rank yet.")
    else:
        print("[PASS] rect-room macro graph hard assertions passed.")
    return out_dir















def normalize_grid_from_record(record: Dict[str, Any]) -> Grid:
    raw = record.get("grid") or record.get("maze")
    if raw is None:
        raise ValueError("sample lacks grid/maze field")
    grid: Grid = []
    for row in raw:
        if isinstance(row, str):
            grid.append([0 if ch in ".SG0" else 1 for ch in row])
        else:
            grid.append([0 if int(v) == 0 else 1 for v in row])
    if not grid or any(len(r) != len(grid[0]) for r in grid):
        raise ValueError("invalid grid shape")
    return grid


def full_rect_metrics_for_record(record: Dict[str, Any], config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    grid = normalize_grid_from_record(record)
    if "start" not in record or "goal" not in record:
        raise ValueError("sample lacks start/goal fields; full extraction cannot reproduce sample")
    start = tuple(record["start"])
    goal = tuple(record["goal"])
    extractor = FullRectRoomMacroGraphExtractor(config)
    actual = extractor.extract(grid, start, goal)
    metrics = actual.get("metrics", {})
    if actual.get("status") == "failed":
        raise ValueError("full rect-room extraction failed: " + ",".join(actual.get("warnings", [])))
    full = {
        "solvable": bool(record.get("metrics", {}).get("solvable", True)),
        "bfs_len": record.get("metrics", {}).get("bfs_len"),
        "bfs_len_norm": record.get("metrics", {}).get("bfs_len_norm"),
    }
    # Use full graph extraction metrics as canonical rect metrics.
    for k, v in metrics.items():
        if k.startswith("rect_room_") or k.startswith("parallel_") or k.startswith("room_pair_") or k.startswith("terminal_") or k == "macro_cycle_from_parallel_room_edges":
            full[k] = v
            full["full_" + k] = v
    full["full_rect_room_count"] = metrics.get("rect_room_count", 0)
    full["full_rect_room_area_sum"] = metrics.get("rect_room_area_sum", 0)
    full["full_tiny_2x2_room_count"] = metrics.get("tiny_2x2_room_count", 0)
    full["full_elongated_room_count"] = metrics.get("elongated_room_count", 0)
    full["full_rect_room_macro_vertex_count_before"] = metrics.get("rect_room_macro_vertex_count_before", 0)
    full["full_rect_room_macro_edge_count_before"] = metrics.get("rect_room_macro_edge_count_before", 0)
    full["full_rect_room_compressed_cycle_rank_before"] = metrics.get("rect_room_compressed_cycle_rank_before", 0)
    full["full_rect_room_macro_choice_count_before"] = metrics.get("rect_room_macro_choice_count_before", 0)
    full["full_rect_room_macro_endpoint_count_before"] = metrics.get("rect_room_macro_endpoint_count_before", 0)
    full["full_rect_room_macro_vertex_count"] = metrics.get("rect_room_macro_vertex_count", 0)
    full["full_rect_room_macro_edge_count"] = metrics.get("rect_room_macro_edge_count", 0)
    full["full_rect_room_compressed_cycle_rank"] = metrics.get("rect_room_compressed_cycle_rank", 0)
    full["full_rect_room_macro_choice_count"] = metrics.get("rect_room_macro_choice_count", 0)
    full["full_rect_room_macro_endpoint_count"] = metrics.get("rect_room_macro_endpoint_count", 0)
    full["full_parallel_macro_edge_count"] = metrics.get("parallel_macro_edge_count", 0)
    full["full_room_pair_parallel_edge_count"] = metrics.get("room_pair_parallel_edge_count", 0)
    full["full_terminal_inside_room_count"] = metrics.get("terminal_inside_room_count", 0)
    full["full_terminal_endpoint_count"] = metrics.get("terminal_endpoint_count", 0)
    full["full_terminal_choice_count"] = metrics.get("terminal_choice_count", 0)
    full["full_extraction_warnings"] = actual.get("warnings", [])
    return full, actual


















def hard_quality_filter(cand: MazeCandidate, config: Dict[str, Any], confirmed_hashes: set[str]) -> Dict[str, Any]:
    grid = cand.grid
    n = int(config["grid"].get("size", len(grid)))
    out: Dict[str, Any] = {
        "hard_quality_passed": False,
        "hard_reject_reason": None,
        "sample_hash": None,
        "bfs_len": None,
        "reachable_free_count": 0,
        "reachable_free_ratio": 0.0,
        "reachable_main_component": [],
    }
    if not grid or any(len(row) != len(grid[0]) for row in grid):
        out["hard_reject_reason"] = "invalid_grid_shape"; return out
    if not in_grid(grid, cand.start[0], cand.start[1]) or not in_grid(grid, cand.goal[0], cand.goal[1]):
        out["hard_reject_reason"] = "invalid_start_goal"; return out
    if grid[cand.start[0]][cand.start[1]] != 0 or grid[cand.goal[0]][cand.goal[1]] != 0:
        out["hard_reject_reason"] = "start_or_goal_not_free"; return out
    sample_hash = grid_hash(grid, cand.start, cand.goal)
    out["sample_hash"] = sample_hash
    if sample_hash in confirmed_hashes:
        out["hard_reject_reason"] = "duplicate_hash"; return out
    free = free_cells(grid)
    free_count = len(free)
    free_ratio = free_count / float(max(1, len(grid) * len(grid[0])))
    dist, _ = bfs_dist_grid(grid, cand.start, cand.goal)
    if cand.goal not in dist:
        out["hard_reject_reason"] = "unsolvable"; return out
    bfs_len = int(dist[cand.goal])
    dist_all, _ = bfs_dist_grid(grid, cand.start)
    reachable = set(dist_all.keys())
    reachable_ratio = len(reachable) / float(free_count or 1)
    out.update({
        "bfs_len": bfs_len,
        "reachable_free_count": len(reachable),
        "reachable_free_ratio": reachable_ratio,
        "reachable_main_component": [list(x) for x in sorted(reachable)],
        "free_ratio": free_ratio,
        "start_free_degree": cell_degree_in_set(grid, cand.start, reachable),
        "goal_free_degree": cell_degree_in_set(grid, cand.goal, reachable),
    })
    q = config["quality_gate"]
    if bfs_len < q.get("min_bfs_len", 3):
        out["hard_reject_reason"] = "bfs_len_fail"; return out
    if reachable_ratio < q.get("min_reachable_free_ratio", 0.9):
        out["hard_reject_reason"] = "reachable_free_ratio_fail"; return out
    if out["start_free_degree"] < q.get("min_start_free_degree", 1):
        out["hard_reject_reason"] = "start_free_degree_fail"; return out
    if free_ratio < q.get("min_free_ratio", 0.2) or free_ratio > q.get("max_free_ratio", 0.85):
        out["hard_reject_reason"] = "free_ratio_fail"; return out
    out["hard_quality_passed"] = True
    return out








def full_metrics_for_candidate(cand: MazeCandidate, cheap_metrics: Dict[str, Any], config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], float, Optional[str]]:
    t0 = time.time()
    try:
        tmp_rec = {"grid": cand.grid, "start": list(cand.start), "goal": list(cand.goal), "metrics": cheap_metrics, "maze_id": cand.candidate_id}
        full_metrics, actual = full_rect_metrics_for_record(tmp_rec, config)
        merged = dict(cheap_metrics)
        # Canonical final archive keys are unprefixed rect_room_* fields filled from full_* values.
        for k, v in full_metrics.items():
            merged[k] = v
            if k.startswith("full_rect_room_"):
                merged[k[len("full_"):]] = v
            elif k.startswith("full_parallel_"):
                merged[k[len("full_"):]] = v
            elif k.startswith("full_room_pair_"):
                merged[k[len("full_"):]] = v
            elif k.startswith("full_terminal_"):
                merged[k[len("full_"):]] = v
        runtime_ms = (time.time() - t0) * 1000.0
        return merged, actual, runtime_ms, None
    except Exception as e:
        runtime_ms = (time.time() - t0) * 1000.0
        return dict(cheap_metrics), {"status": "failed", "warnings": [str(e)]}, runtime_ms, str(e)






def runtime_summary_ms(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {"full_extraction_count": 0, "mean_full_extraction_ms": None, "p50_full_extraction_ms": None, "p95_full_extraction_ms": None}
    xs=sorted(vals); idx95=min(len(xs)-1, int(math.ceil(0.95*len(xs)))-1)
    return {"full_extraction_count": len(xs), "mean_full_extraction_ms": sum(xs)/len(xs), "p50_full_extraction_ms": xs[len(xs)//2], "p95_full_extraction_ms": xs[idx95]}



# -----------------------------------------------------------------------------
# 3.0.4(7) Handdraw-weighted target distribution archive
# -----------------------------------------------------------------------------

def _bucket_defs_from_config(config: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    archive_cfg = config.get("archive", {})
    return {
        "cycle_rank_bins": archive_cfg.get("cycle_rank_bins", [
            {"name": "tree", "range": [0, 0]},
            {"name": "single_loop", "range": [1, 1]},
            {"name": "multi_loop", "range": [2, 999]},
        ]),
        "bfs_len_bins": archive_cfg.get("bfs_len_bins", [
            {"name": "short", "range": [1, 10]},
            {"name": "medium", "range": [11, 20]},
            {"name": "long", "range": [21, 999]},
        ]),
        "macro_choice_bins": archive_cfg.get("macro_choice_bins", [
            {"name": "low_choice", "range": [0, 2]},
            {"name": "mid_choice", "range": [3, 5]},
            {"name": "high_choice", "range": [6, 999]},
        ]),
        "macro_endpoint_bins": archive_cfg.get("macro_endpoint_bins", [
            {"name": "low_endpoint", "range": [0, 2]},
            {"name": "mid_endpoint", "range": [3, 6]},
            {"name": "high_endpoint", "range": [7, 999]},
        ]),
    }


def _all_target_bucket_keys(config: Dict[str, Any]) -> List[str]:
    defs = _bucket_defs_from_config(config)
    return [
        f"{a['name']}|{b['name']}|{c['name']}|{d['name']}"
        for a in defs["cycle_rank_bins"]
        for b in defs["bfs_len_bins"]
        for c in defs["macro_choice_bins"]
        for d in defs["macro_endpoint_bins"]
    ]


def _bin_range_map(bins: List[Dict[str, Any]]) -> Dict[str, Tuple[int, int]]:
    out: Dict[str, Tuple[int, int]] = {}
    for b in bins:
        lo, hi = b["range"]
        out[b["name"]] = (int(lo), int(hi))
    return out


def classify_target_bucket_feasibility(bin_key: str, config: Dict[str, Any]) -> Tuple[str, str]:
    parts = bin_key.split("|")
    if len(parts) != 4:
        return "structural_infeasible_or_constrained", "invalid_bin_key"
    cycle_name, _bfs_name, choice_name, endpoint_name = parts
    defs = _bucket_defs_from_config(config)
    cycle_ranges = _bin_range_map(defs["cycle_rank_bins"])
    choice_ranges = _bin_range_map(defs["macro_choice_bins"])
    endpoint_ranges = _bin_range_map(defs["macro_endpoint_bins"])
    if cycle_name not in cycle_ranges or choice_name not in choice_ranges or endpoint_name not in endpoint_ranges:
        return "structural_infeasible_or_constrained", "unknown_bin_component"
    mu_min, mu_max = cycle_ranges[cycle_name]
    c_min, _c_max = choice_ranges[choice_name]
    _e_min, e_max = endpoint_ranges[endpoint_name]
    # Conservative bucket-level graph constraint used by analyze_handdraw_distribution.py.
    if cycle_name in {"tree", "single_loop"}:
        conservative_l_min = max(0, c_min - 2 * mu_max + 2)
        if e_max < conservative_l_min:
            return "structural_infeasible_or_constrained", "endpoint_bin_max < conservative_L_min"
    return "possible", "not_structurally_constrained_by_conservative_rule"


def largest_remainder_quota(probs: Dict[str, float], total: int) -> Dict[str, int]:
    if total < 0:
        raise ValueError("target total must be non-negative")
    raw = {k: max(0.0, float(v)) * total for k, v in probs.items()}
    quotas = {k: int(math.floor(v)) for k, v in raw.items()}
    remaining = total - sum(quotas.values())
    if remaining > 0:
        ordered = sorted(raw.keys(), key=lambda k: (raw[k] - quotas[k], raw[k], k), reverse=True)
        for k in ordered[:remaining]:
            quotas[k] += 1
    elif remaining < 0:
        ordered = sorted(raw.keys(), key=lambda k: (raw[k] - quotas[k], raw[k], k))
        for k in ordered:
            while remaining < 0 and quotas[k] > 0:
                quotas[k] -= 1
                remaining += 1
            if remaining == 0:
                break
    return quotas


def build_fallback_target_distribution(config: Dict[str, Any], target_total: int = 1000) -> Dict[str, Any]:
    keys = _all_target_bucket_keys(config)
    bins: Dict[str, Dict[str, Any]] = {}
    possible: List[str] = []
    structural: List[str] = []
    for key in keys:
        cls, reason = classify_target_bucket_feasibility(key, config)
        if cls == "structural_infeasible_or_constrained":
            structural.append(key)
        else:
            possible.append(key)
    probs = {k: (1.0 / len(possible) if k in possible and possible else 0.0) for k in keys}
    quotas = largest_remainder_quota(probs, target_total)
    for key in keys:
        cls, reason = classify_target_bucket_feasibility(key, config)
        if cls == "structural_infeasible_or_constrained":
            status = "structural_infeasible_or_constrained"
            target_prob = 0.0
            target_quota = 0
        else:
            status = "fallback_uniform_possible"
            target_prob = probs[key]
            target_quota = quotas[key]
        bins[key] = {
            "status": status,
            "handdraw_count": 0,
            "handdraw_prob": 0.0,
            "target_prob": target_prob,
            "raw_quota": target_prob * target_total,
            "target_quota": target_quota,
            "reason": reason if status.startswith("structural") else "fallback uniform over possible buckets",
            "maze_ids": [],
            "observed_structural_conflict": False,
        }
    quota_sum = sum(b["target_quota"] for b in bins.values())
    return {
        "schema_version": "handdraw_target_distribution.v1",
        "created_by": "run_3_0_4_archive_driven_weak_generator_factory.py fallback",
        "source_input": None,
        "target_total": target_total,
        "quota_sum": quota_sum,
        "observed_mass": 0.0,
        "explore_mass": 1.0,
        "structural_mass": 0.0,
        "n_bins_total": len(keys),
        "n_handdraw_observed_bins": 0,
        "n_unobserved_possible_bins": len(possible),
        "n_structural_constrained_bins": len(structural),
        "warnings": ["target_json_missing_used_fallback_uniform_possible_distribution"],
        "bins": bins,
    }


def _recompute_target_quotas_from_probs(target: Dict[str, Any], target_total: int) -> Dict[str, Any]:
    target = copy.deepcopy(target)
    bins = target.get("bins", {})
    probs = {k: float(v.get("target_prob", 0.0)) for k, v in bins.items()}
    prob_sum = sum(probs.values())
    if prob_sum <= 0:
        raise ValueError("target distribution has no positive target_prob")
    probs = {k: v / prob_sum for k, v in probs.items()}
    quotas = largest_remainder_quota(probs, target_total)
    for key, b in bins.items():
        b["target_prob"] = probs[key]
        b["raw_quota"] = probs[key] * target_total
        b["target_quota"] = int(quotas[key])
    target["target_total"] = int(target_total)
    target["quota_sum"] = int(sum(quotas.values()))
    return target


def load_target_distribution(path: Path, target_total_override: Optional[int], config: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        target_total = int(target_total_override or 1000)
        target = build_fallback_target_distribution(config, target_total=target_total)
        ensure_dir(path.parent)
        write_json(path, target)
        print("[WARN] target distribution JSON not found. Fallback target distribution was generated.")
        print("[WARN] fallback target distribution is not handdraw-weighted. Run analyze_handdraw_distribution.py to build the real target JSON.")
        return target
    target = load_json(path)
    if target.get("schema_version") != "handdraw_target_distribution.v1":
        raise SystemExit(f"[ERROR] invalid target distribution schema_version in {path}")
    bins = target.get("bins")
    if not isinstance(bins, dict):
        raise SystemExit(f"[ERROR] target distribution missing bins: {path}")
    expected = set(_all_target_bucket_keys(config))
    if set(bins.keys()) != expected:
        missing = sorted(expected - set(bins.keys()))[:10]
        extra = sorted(set(bins.keys()) - expected)[:10]
        raise SystemExit(f"[ERROR] target distribution bins must match 81 full_4d buckets. missing={missing} extra={extra}")
    required = {"status", "target_prob", "target_quota", "handdraw_count", "handdraw_prob", "reason"}
    for key, b in bins.items():
        miss = required - set(b.keys())
        if miss:
            raise SystemExit(f"[ERROR] target bucket {key} missing fields: {sorted(miss)}")
    if target_total_override is not None:
        target = _recompute_target_quotas_from_probs(target, int(target_total_override))
    quota_sum = sum(int(b.get("target_quota", 0)) for b in target["bins"].values())
    if quota_sum != int(target.get("target_total", quota_sum)):
        raise SystemExit(f"[ERROR] target quota_sum mismatch: quota_sum={quota_sum} target_total={target.get('target_total')}")
    target["quota_sum"] = quota_sum
    structural_nonzero = [k for k, b in target["bins"].items() if b.get("status") == "structural_infeasible_or_constrained" and int(b.get("target_quota", 0)) > 0]
    if structural_nonzero:
        print(f"[WARN] {len(structural_nonzero)} structurally constrained target buckets have nonzero quota.")
    return target


class TargetDistributionArchive:
    def __init__(self, target_distribution: Dict[str, Any], diversity_config: Dict[str, Any], config: Dict[str, Any]):
        self.target_distribution = target_distribution
        self.bins = target_distribution["bins"]
        self.div_cfg = diversity_config or {}
        self.config = config
        self.counts: Counter[str] = Counter()
        self.samples_by_bin: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.accepted_hashes: set[str] = set()
        self.origin_seed_bin_counts: Counter[Tuple[str, str]] = Counter()

    def all_bin_keys(self) -> List[str]:
        return list(self.bins.keys())

    def target_quota(self, bin_key: str) -> int:
        return int(self.bins.get(bin_key, {}).get("target_quota", 0))

    def quota(self, bin_key: str) -> int:
        return self.target_quota(bin_key)

    def target_prob(self, bin_key: str) -> float:
        return float(self.bins.get(bin_key, {}).get("target_prob", 0.0))

    def _assign_one(self, value: int, bins: List[Dict[str, Any]], fallback: str) -> str:
        for b in bins:
            lo, hi = b["range"]
            if lo <= value <= hi:
                return b["name"]
        return fallback

    def assign_bin(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        defs = _bucket_defs_from_config(self.config)
        cr_bin = self._assign_one(int(metrics.get("rect_room_compressed_cycle_rank", 0)), defs["cycle_rank_bins"], "cycle_oob")
        bfs_bin = self._assign_one(int(metrics.get("bfs_len") if metrics.get("bfs_len") is not None else -1), defs["bfs_len_bins"], "bfs_oob")
        choice_bin = self._assign_one(int(metrics.get("rect_room_macro_choice_count", 0)), defs["macro_choice_bins"], "macro_choice_oob")
        endpoint_bin = self._assign_one(int(metrics.get("rect_room_macro_endpoint_count", 0)), defs["macro_endpoint_bins"], "macro_endpoint_oob")
        return {"cycle_rank_bin": cr_bin, "bfs_len_bin": bfs_bin, "macro_choice_bin": choice_bin, "macro_endpoint_bin": endpoint_bin, "bin_key": f"{cr_bin}|{bfs_bin}|{choice_bin}|{endpoint_bin}"}

    def maybe_accept(self, cand: MazeCandidate, metrics: Dict[str, Any], full_bin_info: Dict[str, Any], rng: random.Random) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        key = full_bin_info["bin_key"]
        if key not in self.bins:
            return False, {"accepted": False, "bin": key, "count_before": self.counts[key], "target_quota": 0}, "unknown_target_bin"
        q = self.target_quota(key)
        before = self.counts[key]
        if q <= 0:
            return False, {"accepted": False, "bin": key, "count_before": before, "target_quota": q}, "zero_target_quota"
        if before >= q:
            return False, {"accepted": False, "bin": key, "count_before": before, "target_quota": q}, "target_bin_full"
        sample_hash = metrics.get("duplicate_hash") or grid_hash(cand.grid, cand.start, cand.goal)
        if self.div_cfg.get("reject_exact_duplicate", True) and sample_hash in self.accepted_hashes:
            return False, {"accepted": False, "bin": key, "count_before": before, "target_quota": q}, "duplicate_hash_reject"
        max_same = self.div_cfg.get("max_same_origin_seed_per_bin")
        if max_same is not None and self.origin_seed_bin_counts[(cand.origin_seed_id, key)] >= int(max_same):
            return False, {"accepted": False, "bin": key, "count_before": before, "target_quota": q}, "same_origin_seed_bin_limit"
        self.counts[key] += 1
        self.samples_by_bin[key].append({"candidate_id": cand.candidate_id, "origin_seed_id": cand.origin_seed_id})
        self.accepted_hashes.add(sample_hash)
        self.origin_seed_bin_counts[(cand.origin_seed_id, key)] += 1
        return True, {
            "accepted": True,
            "bin": key,
            "count_before": before,
            "count_after": before + 1,
            "target_quota": q,
            "target_prob": self.target_prob(key),
            "acceptance_policy": "hard_target_quota",
        }, None

    def coverage_report(self) -> Dict[str, Any]:
        rows = []
        for key in self.all_bin_keys():
            q = self.target_quota(key)
            cnt = self.counts[key]
            b = self.bins[key]
            rows.append({
                "bin_key": key,
                "status": b.get("status"),
                "target_prob": self.target_prob(key),
                "target_quota": q,
                "quota": q,
                "count": cnt,
                "confirmed_count": cnt,
                "quota_gap": q - cnt,
                "fill_ratio": cnt / q if q else None,
                "handdraw_count": b.get("handdraw_count", 0),
                "handdraw_prob": b.get("handdraw_prob", 0.0),
            })
        non_empty = sum(1 for r in rows if r["count"] > 0)
        nonzero = [r for r in rows if r["target_quota"] > 0]
        filled_nonzero = sum(1 for r in nonzero if r["count"] > 0)
        structural_confirmed = sum(r["count"] for r in rows if r["status"] == "structural_infeasible_or_constrained")
        return {
            "n_bins_total": len(rows),
            "n_bins_non_empty": non_empty,
            "raw_coverage_all_81": non_empty / len(rows) if rows else 0.0,
            "nonzero_target_bins": len(nonzero),
            "filled_nonzero_target_bins": filled_nonzero,
            "coverage_nonzero_target_bins": filled_nonzero / len(nonzero) if nonzero else 0.0,
            "confirmed_archive_count": sum(r["count"] for r in rows),
            "target_total": int(self.target_distribution.get("target_total", sum(r["target_quota"] for r in rows))),
            "quota_sum": sum(r["target_quota"] for r in rows),
            "structural_confirmed_count": structural_confirmed,
            "bins": rows,
            "sparse_target_bins": sorted([r for r in rows if r["target_quota"] > 0], key=lambda x: (x["fill_ratio"] if x["fill_ratio"] is not None else 999, -x["target_quota"], x["bin_key"]))[:10],
            "overfilled_target_bins": [r for r in rows if r["target_quota"] > 0 and r["count"] > r["target_quota"]],
        }

    def all_bins_reached_quota(self) -> bool:
        return all(self.counts[k] >= self.target_quota(k) for k in self.all_bin_keys() if self.target_quota(k) > 0)


def build_synthetic_vs_target_report(archive: TargetDistributionArchive) -> Dict[str, Any]:
    rows = archive.coverage_report()["bins"]
    confirmed_total = sum(r["confirmed_count"] for r in rows)
    target_total = sum(r["target_quota"] for r in rows)
    eps = 1e-12
    l1 = 0.0
    kl = 0.0
    for r in rows:
        p_t = float(r["target_prob"])
        p_c = r["confirmed_count"] / confirmed_total if confirmed_total else 0.0
        l1 += abs(p_c - p_t)
        if p_t > 0:
            kl += p_t * math.log((p_t + eps) / (p_c + eps))
    handdraw_observed = [r for r in rows if r["status"] == "handdraw_observed"]
    observed_filled = sum(1 for r in handdraw_observed if r["confirmed_count"] > 0)
    report_rows = []
    for r in rows:
        report_rows.append({
            "bin_key": r["bin_key"],
            "status": r["status"],
            "target_prob": r["target_prob"],
            "target_quota": r["target_quota"],
            "confirmed_count": r["confirmed_count"],
            "quota_gap": r["quota_gap"],
            "fill_ratio": r["fill_ratio"],
            "handdraw_count": r["handdraw_count"],
            "handdraw_prob": r["handdraw_prob"],
        })
    return {
        "target_total": target_total,
        "confirmed_total": confirmed_total,
        "quota_sum": target_total,
        "target_fill_ratio": confirmed_total / target_total if target_total else 0.0,
        "filled_target_bins": sum(1 for r in rows if r["confirmed_count"] > 0),
        "target_bins_with_nonzero_quota": sum(1 for r in rows if r["target_quota"] > 0),
        "nonzero_target_bins": sum(1 for r in rows if r["target_quota"] > 0),
        "filled_nonzero_target_bins": sum(1 for r in rows if r["target_quota"] > 0 and r["confirmed_count"] > 0),
        "raw_coverage_all_81": sum(1 for r in rows if r["confirmed_count"] > 0) / len(rows) if rows else 0.0,
        "coverage_nonzero_target_bins": sum(1 for r in rows if r["target_quota"] > 0 and r["confirmed_count"] > 0) / max(1, sum(1 for r in rows if r["target_quota"] > 0)),
        "observed_handdraw_bin_fill_rate": observed_filled / len(handdraw_observed) if handdraw_observed else 0.0,
        "structural_bin_accept_count": sum(r["confirmed_count"] for r in rows if r["status"] == "structural_infeasible_or_constrained"),
        "l1_distance_to_target_distribution": l1,
        "kl_divergence_target_to_synthetic_smooth": kl,
        "bins": report_rows,
    }


def target_debug_hints(archive_report: Dict[str, Any], synthetic_report: Dict[str, Any], target: Dict[str, Any]) -> List[str]:
    hints: List[str] = []
    structural_quota = sum(int(b.get("target_quota", 0)) for b in target.get("bins", {}).values() if b.get("status") == "structural_infeasible_or_constrained")
    if structural_quota > 0:
        hints.append("[WARN] structural constrained buckets have nonzero target quota. Check target distribution JSON.")
    if synthetic_report.get("structural_bin_accept_count", 0) > 0:
        hints.append("[HINT] structurally constrained bins received confirmed samples. Check target JSON and acceptance logic.")
    if synthetic_report.get("observed_handdraw_bin_fill_rate", 0.0) < 0.5:
        hints.append("[HINT] many observed handdraw bins are still unfilled. Increase n_seeds or inspect sparse target bins.")
    if synthetic_report.get("l1_distance_to_target_distribution", 0.0) > 1.0:
        hints.append("[HINT] synthetic distribution is still far from the handdraw target distribution. Inspect synthetic_vs_target_distribution_report.json.")
    if not hints:
        hints.append("No major anomaly detected.")
    return hints


def print_target_summary(run_summary: Dict[str, Any], target: Dict[str, Any], generator_summary: List[Dict[str, Any]], transform_summary: List[Dict[str, Any]], archive_report: Dict[str, Any], synthetic_report: Dict[str, Any], full_runtime: Dict[str, Any], debug_hints: List[str]) -> None:
    print(TITLE)
    print("\n=== Target Distribution ===")
    for k in ["target_distribution_json", "target_total", "quota_sum", "handdraw_observed_bins", "unobserved_possible_bins", "structural_constrained_bins", "nonzero_target_bins"]:
        print(f"{k:36s} {run_summary.get(k)}")
    print("top_target_quotas")
    top = sorted(target.get("bins", {}).items(), key=lambda kv: int(kv[1].get("target_quota", 0)), reverse=True)[:10]
    for key, b in top:
        print(f"  {key:52s} quota={int(b.get('target_quota',0)):5d} status={b.get('status')} handdraw_count={b.get('handdraw_count',0)}")
    print("\n=== Candidate Funnel ===")
    for k in ["n_candidate_total", "n_hard_quality_passed", "n_full_evaluated", "n_full_accepted", "hard_pass_rate", "full_accept_rate_given_hard"]:
        print(f"{k:36s} {run_summary.get(k)}")
    print("\n=== Target Archive Coverage ===")
    for k in ["confirmed_total", "target_total", "target_fill_ratio", "nonzero_target_bins", "filled_nonzero_target_bins", "coverage_nonzero_target_bins", "observed_handdraw_bin_fill_rate", "l1_distance_to_target_distribution"]:
        print(f"{k:36s} {synthetic_report.get(k)}")
    print("\n=== Sparse Target Bins ===")
    for r in archive_report.get("sparse_target_bins", [])[:10]:
        print(f"{r['bin_key']:52s} quota={r['target_quota']:5d} count={r['count']:5d} fill={r['fill_ratio']}")
    print("\n=== Overfilled Target Bins ===")
    over = archive_report.get("overfilled_target_bins", [])
    if not over:
        print("none")
    else:
        for r in over[:10]:
            print(f"{r['bin_key']:52s} quota={r['target_quota']:5d} count={r['count']:5d}")
    print("\n=== Structural Constrained Check ===")
    structural_quota = sum(int(b.get("target_quota", 0)) for b in target.get("bins", {}).values() if b.get("status") == "structural_infeasible_or_constrained")
    print(f"{'structural_target_quota_sum':36s} {structural_quota}")
    print(f"{'structural_confirmed_count':36s} {synthetic_report.get('structural_bin_accept_count')}")
    print("\n=== Full Extraction Runtime ===")
    for k, v in full_runtime.items():
        print(f"{k:36s} {v}")
    print("\n=== Generator Summary ===")
    for row in generator_summary:
        print(f"{row['generator']:32s} seeds={row['seeds']:5d} accepted={row['accepted']:5d} accept_rate={row['accept_rate']:.4f}")
    print("\n=== Transform Summary ===")
    for row in transform_summary:
        print(f"{row['operator']:20s} attempts={row['attempts']:6d} successes={row['successes']:6d} accepted_after={row['accepted_after_operator']:6d} failure_rate={row['failure_rate']:.4f}")
    print("\n=== Debug Hints ===")
    for h in debug_hints:
        print(h)






def run_generate(args: argparse.Namespace) -> Path:
    config_path = Path(args.config_json) if args.config_json else build_default_config_path()
    config = apply_config_overrides(load_json(config_path), args)
    rng = random.Random(args.seed)
    target_path = Path(args.target_distribution_json)
    target = load_target_distribution(target_path, args.target_total_samples, config)
    run_name = args.run_name or f"run_3_0_4_8_handdraw_target_seed{args.seed}_{int(time.time())}"
    out_dir = resolve_output_root(args) / run_name
    ensure_dir(out_dir)
    for filename in [
        "confirmed_samples.jsonl", "accepted_samples.jsonl", "candidate_debug_records.jsonl",
        "rejected_sample_examples.jsonl", "full_extraction_error_samples.jsonl",
    ]:
        p = out_dir / filename
        if p.exists():
            p.unlink()
    write_json(out_dir / "resolved_config.json", config)
    write_json(out_dir / "target_distribution_used.json", target)
    write_json(out_dir / "run_metadata.json", {
        "version": VERSION,
        "title": TITLE,
        "run_name": run_name,
        "seed": args.seed,
        "n_seeds": args.n_seeds,
        "config_path": str(config_path),
        "accepted_samples_semantics": "confirmed_full_bin_samples_matched_to_handdraw_target_distribution",
        "final_archive_bin_source": "full_bin_key",
        "target_distribution_json": str(target_path),
        "target_distribution_schema": target.get("schema_version"),
        "route_positioning": "3.0.4(8) keeps weak candidate generation but final archive acceptance is controlled only by the handdraw target distribution and full rect-room macro graph extraction.",
    })

    quantizer = Quantizer(config)
    archive = TargetDistributionArchive(target, config.get("diversity", {}), config)
    generators = SeedGenerators(config, rng)
    start_goal_assigner = StartGoalAssigner(config, rng)
    transforms = Transforms(config, rng)
    summary = initialize_summary()
    summary.update({
        "n_candidate_total": 0,
        "n_hard_quality_passed": 0,
        "n_full_evaluated": 0,
        "n_full_accepted": 0,
        "full_extraction_runtimes_ms": [],
        "full_extraction_error_count": 0,
        "confirmed_reject_reasons": Counter(),
    })
    accepted_records: List[Tuple[MazeCandidate, Dict[str, Any], Dict[str, Any], str]] = []
    rejected_examples_written = 0
    rejected_limit = int(config["runtime"].get("rejected_example_limit", 200))
    max_steps = int(config["transform"].get("max_transform_steps", 6))
    max_failures = int(config["transform"].get("max_consecutive_transform_failures", 3))
    gen_weights = config["seed_generators"]["weights"]
    debug_limit = config["runtime"].get("candidate_debug_limit", 5000)
    stop_when_target_filled = bool(args.stop_when_target_filled)
    seed_iter = iter_progress(range(int(args.n_seeds)), desc="Generating seeds")
    start_time = time.time()

    for seed_idx in seed_iter:
        gen_name = choose_weighted(rng, gen_weights)
        summary["generator_seeds"][gen_name] += 1
        grid = generators.make_grid(gen_name)
        origin_seed_id = f"seed{seed_idx:06d}_{gen_name}"
        s, g, sg_debug = start_goal_assigner.assign(grid, archive)
        summary["start_goal_fallback_count"] += 1 if sg_debug.get("fallback") else 0
        current = MazeCandidate(grid, s, g, gen_name, origin_seed_id, candidate_id=f"maze_seed{seed_idx:06d}_step0")
        consecutive_failures = 0
        for step in range(max_steps + 1):
            current.candidate_id = f"maze_seed{seed_idx:06d}_step{step}"
            summary["n_candidate_total"] += 1
            summary["n_candidates_evaluated"] += 1
            hard = hard_quality_filter(current, config, archive.accepted_hashes)
            cheap_metrics = quantizer.evaluate(current)
            update_metric_distribution(summary, cheap_metrics)
            if not hard.get("hard_quality_passed"):
                reason = hard.get("hard_reject_reason") or "hard_quality_failed"
                summary["reject_reasons"][reason] += 1
                debug_record = {"candidate_id": current.candidate_id, "origin_seed_id": current.origin_seed_id, "seed_index": seed_idx, "life_step": step, "origin_generator": current.origin_generator, "start": list(current.start), "goal": list(current.goal), "start_goal_assignment": sg_debug, "hard_quality": hard, "metrics": cheap_metrics, "accepted": False, "reject_reason": reason}
                if debug_limit is None or summary["n_candidates_evaluated"] <= int(debug_limit):
                    append_jsonl(out_dir / "candidate_debug_records.jsonl", debug_record)
                if rejected_examples_written < rejected_limit:
                    append_jsonl(out_dir / "rejected_sample_examples.jsonl", debug_record); rejected_examples_written += 1
            else:
                summary["n_hard_quality_passed"] += 1
                summary["n_full_evaluated"] += 1
                full_metrics, full_actual, rt_ms, err = full_metrics_for_candidate(current, cheap_metrics, config)
                summary["full_extraction_runtimes_ms"].append(rt_ms)
                accepted = False
                confirmed_reject_reason = None
                acceptance: Dict[str, Any] = {"accepted": False, "acceptance_policy": "hard_target_quota"}
                full_bin_info = archive.assign_bin(full_metrics)
                if err:
                    summary["full_extraction_error_count"] += 1
                    confirmed_reject_reason = "full_extraction_error"
                    append_jsonl(out_dir / "full_extraction_error_samples.jsonl", {"sample_id": current.candidate_id, "origin_generator": current.origin_generator, "origin_seed_id": current.origin_seed_id, "grid": current.grid, "start": list(current.start), "goal": list(current.goal), "error": err, "warnings": full_actual.get("warnings", []) if isinstance(full_actual, dict) else []})
                else:
                    accepted, acceptance, confirmed_reject_reason = archive.maybe_accept(current, full_metrics, full_bin_info, rng)
                if not accepted and confirmed_reject_reason:
                    summary["confirmed_reject_reasons"][confirmed_reject_reason] += 1
                    summary["reject_reasons"][confirmed_reject_reason] += 1
                if accepted:
                    summary["n_accepted"] += 1
                    summary["n_full_accepted"] += 1
                    summary["generator_accepted"][current.origin_generator] += 1
                    summary["generator_bins"][current.origin_generator][full_bin_info["bin_key"]] += 1
                    if step > 0:
                        summary["accepted_after_transform_count"] += 1
                        if current.transform_history:
                            op = current.transform_history[-1].get("operator", "unknown")
                            summary["operator_accepted_after"][op] += 1
                            if op == "cell_flip_random":
                                summary["accepted_after_flipper_count"] += 1
                    tb = target["bins"][full_bin_info["bin_key"]]
                    target_bin = {
                        "status": tb.get("status"),
                        "target_prob": tb.get("target_prob"),
                        "target_quota": tb.get("target_quota"),
                        "handdraw_count": tb.get("handdraw_count"),
                        "handdraw_prob": tb.get("handdraw_prob"),
                        "count_before": acceptance.get("count_before"),
                        "count_after": acceptance.get("count_after"),
                    }
                    confirmed_record = {
                        "maze_id": current.candidate_id,
                        "sample_id": current.candidate_id,
                        "grid": current.grid,
                        "start": list(current.start),
                        "goal": list(current.goal),
                        "origin_generator": current.origin_generator,
                        "origin_seed_id": current.origin_seed_id,
                        "transform_history": current.transform_history,
                        "start_goal_assignment": sg_debug,
                        "metrics": full_metrics,
                        "full_bin": full_bin_info,
                        "target_bin": target_bin,
                        "acceptance": {"accepted": True, "acceptance_policy": "hard_target_quota", "reject_reason": None},
                    }
                    append_jsonl(out_dir / "confirmed_samples.jsonl", confirmed_record)
                    append_jsonl(out_dir / "accepted_samples.jsonl", confirmed_record)
                    accepted_records.append((current.clone(), full_metrics, full_bin_info, f"{full_bin_info['bin_key']}__{current.origin_generator}__{current.candidate_id}"))
                debug_record = {"candidate_id": current.candidate_id, "origin_seed_id": current.origin_seed_id, "seed_index": seed_idx, "life_step": step, "origin_generator": current.origin_generator, "start": list(current.start), "goal": list(current.goal), "start_goal_assignment": sg_debug, "hard_quality": hard, "metrics": cheap_metrics, "full_metrics": full_metrics if hard.get("hard_quality_passed") else None, "full_bin": full_bin_info if hard.get("hard_quality_passed") else None, "acceptance": acceptance, "accepted": accepted, "confirmed_reject_reason": confirmed_reject_reason}
                if debug_limit is None or summary["n_candidates_evaluated"] <= int(debug_limit):
                    append_jsonl(out_dir / "candidate_debug_records.jsonl", debug_record)
            if stop_when_target_filled and archive.all_bins_reached_quota():
                break
            if step == max_steps:
                break
            op = choose_transform_operator(config, rng, cheap_metrics, archive, consecutive_failures)
            summary["operator_attempts"][op] += 1
            child, op_info = transforms.apply(op, current, cheap_metrics)
            if child is None:
                consecutive_failures += 1
                summary["operator_failures"][op] += 1
                summary["transform_failure_reasons"][op_info.get("reason", "unknown_transform_failure")] += 1
                if consecutive_failures >= max_failures:
                    break
                continue
            after_metrics = quantizer.evaluate(child)
            child.transform_history[-1]["delta"] = metric_delta(cheap_metrics, after_metrics)
            d_choice = child.transform_history[-1]["delta"].get("rect_room_macro_choice_count")
            if isinstance(d_choice, (int, float)):
                summary["operator_rect_choice_delta"][op].append(float(d_choice))
                sign = "positive" if d_choice > 0 else "negative" if d_choice < 0 else "zero"
                summary["operator_rect_choice_delta_sign"][op][sign] += 1
            current = child
            consecutive_failures = 0
            summary["operator_successes"][op] += 1
        if TQDM_AVAILABLE and hasattr(seed_iter, "set_postfix") and seed_idx % max(100, int(getattr(args, "progress_snapshot_interval", 100) or 100)) == 0:
            rep = archive.coverage_report()
            fill = rep["confirmed_archive_count"] / rep["target_total"] if rep.get("target_total") else 0.0
            seed_iter.set_postfix(ok=f"{summary['n_full_accepted']}/{rep.get('target_total', 0)}", fill=f"{fill:.3f}", cov=f"{rep['coverage_nonzero_target_bins']:.2f}", refresh=False)
        if stop_when_target_filled and archive.all_bins_reached_quota():
            break

    elapsed = time.time() - start_time
    archive_report = archive.coverage_report()
    synthetic_report = build_synthetic_vs_target_report(archive)
    generator_summary=[]
    for gen in sorted(gen_weights.keys()):
        seeds = summary["generator_seeds"][gen]; acc = summary["generator_accepted"][gen]
        generator_summary.append({"generator": gen, "seeds": seeds, "accepted": acc, "accept_rate": acc / seeds if seeds else 0.0, "top_bins": summary["generator_bins"][gen].most_common(5)})
    transform_summary=[]
    for op in sorted(config["transform"]["operator_weights"].keys()):
        attempts=summary["operator_attempts"][op]; successes=summary["operator_successes"][op]
        transform_summary.append({"operator": op, "attempts": attempts, "successes": successes, "accepted_after_operator": summary["operator_accepted_after"][op], "failure_rate": (attempts-successes)/attempts if attempts else 0.0})
    metric_distribution={k: metric_stats(v) for k,v in summary["metrics"].items()}
    full_runtime = runtime_summary_ms(summary["full_extraction_runtimes_ms"])
    full_runtime["full_extraction_error_count"] = summary["full_extraction_error_count"]
    run_summary={
        "version": VERSION,
        "n_seeds": args.n_seeds,
        "elapsed_sec": elapsed,
        "target_distribution_json": str(target_path),
        "target_total": target.get("target_total"),
        "quota_sum": target.get("quota_sum"),
        "handdraw_observed_bins": target.get("n_handdraw_observed_bins"),
        "unobserved_possible_bins": target.get("n_unobserved_possible_bins"),
        "structural_constrained_bins": target.get("n_structural_constrained_bins"),
        "nonzero_target_bins": archive_report.get("nonzero_target_bins"),
        "final_archive_bin_source": "full_bin_key",
        "n_candidate_total": summary["n_candidate_total"],
        "n_candidates_evaluated": summary["n_candidate_total"],
        "n_hard_quality_passed": summary["n_hard_quality_passed"],
        "n_full_evaluated": summary["n_full_evaluated"],
        "n_full_accepted": summary["n_full_accepted"],
        "n_accepted": summary["n_full_accepted"],
        "hard_pass_rate": summary["n_hard_quality_passed"] / summary["n_candidate_total"] if summary["n_candidate_total"] else 0.0,
        "full_accept_rate_given_hard": summary["n_full_accepted"] / summary["n_hard_quality_passed"] if summary["n_hard_quality_passed"] else 0.0,
        "start_goal_sampling": {"policy": config["grid"].get("start_goal_policy"), "pair_sampling_mode": config.get("start_goal", {}).get("pair_sampling_mode"), "fallback_count": summary["start_goal_fallback_count"], "fallback_rate": summary["start_goal_fallback_count"] / args.n_seeds if args.n_seeds else 0.0},
    }
    rejection_summary={"reject_reason_count": dict(summary["reject_reasons"]), "confirmed_reject_reason_count": dict(summary["confirmed_reject_reasons"]), "quality_gate_failed_count": dict(summary["quality_failures"]), "transform_failure_count": dict(summary["transform_failure_reasons"]), "accepted_after_transform_count": summary["accepted_after_transform_count"], "accepted_after_flipper_count": summary["accepted_after_flipper_count"]}
    write_json(out_dir / "archive_summary.json", {**run_summary, "accepted_samples_semantics": "confirmed_full_bin_samples_matched_to_handdraw_target_distribution", "generator_summary": generator_summary})
    write_json(out_dir / "target_archive_summary.json", run_summary)
    write_json(out_dir / "synthetic_vs_target_distribution_report.json", synthetic_report)
    write_json(out_dir / "target_bin_coverage_report.json", archive_report)
    write_json(out_dir / "full_extraction_runtime_summary.json", full_runtime)
    write_json(out_dir / "generator_summary.json", generator_summary)
    write_json(out_dir / "transform_summary.json", transform_summary)
    write_json(out_dir / "metric_distribution.json", metric_distribution)
    write_json(out_dir / "rejection_summary.json", rejection_summary)
    write_json(out_dir / "start_goal_sampling_report.json", run_summary["start_goal_sampling"])
    write_json(out_dir / "bin_coverage_report.json", archive_report)
    write_json(out_dir / "full_archive_summary.json", run_summary)
    if config.get("visualization", {}).get("enabled", True):
        selected, vis_report = select_visualization_records(config, accepted_records, archive_report)
        write_json(out_dir / "visualization_selection_report.json", vis_report)
        base_vis = out_dir / "sample_visualizations"
        for idx, (cand, metrics, bin_info, title, category) in enumerate(selected, 1):
            filename=f"{idx:02d}_{safe_filename(bin_info['bin_key'])}__{safe_filename(cand.origin_generator)}__{safe_filename(cand.origin_seed_id)}_step{cand.candidate_id.split('step')[-1]}__bfs{metrics.get('bfs_len')}__rcr{metrics.get('rect_room_compressed_cycle_rank')}__mc{metrics.get('rect_room_macro_choice_count')}__me{metrics.get('rect_room_macro_endpoint_count')}__rawcr{metrics.get('raw_cycle_rank')}.png"
            render_maze(cand, metrics, bin_info, base_vis / category / filename, filename)
    else:
        write_json(out_dir / "visualization_selection_report.json", {"sampling_policy":"disabled", "n_visualized":0})
    debug_hints=target_debug_hints(archive_report, synthetic_report, target)
    write_json(out_dir / "debug_hints.json", debug_hints)
    print_target_summary(run_summary, target, generator_summary, transform_summary, archive_report, synthetic_report, full_runtime, debug_hints)
    return out_dir



class _LocalFactoryNamespace:
    pass

F = _LocalFactoryNamespace()
for _name in [
    "default_config", "load_target_distribution", "write_json",
    "TargetDistributionArchive", "Quantizer", "StartGoalAssigner",
    "SeedGenerators", "Transforms", "MazeCandidate", "hard_quality_filter",
    "full_metrics_for_candidate",
]:
    setattr(F, _name, globals()[_name])
F.__mvp_import_path__ = "self_contained_embedded_core"


def in_bounds(r: int, c: int, n: int) -> bool:
    return 0 <= r < n and 0 <= c < n


def neighbors(cell: Cell, n: int) -> List[Cell]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in ACTIONS if in_bounds(r + dr, c + dc, n)]


def empty_grid(n: int, fill: int = 1) -> Grid:
    return [[fill for _ in range(n)] for _ in range(n)]


def grid_hash(grid: Grid, start: Cell, goal: Cell) -> str:
    s = "".join("".join(str(v) for v in row) for row in grid) + f"|{start}|{goal}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


class BFSResult(dict):
    def __init__(self, dist: Dict[Cell, int], parent: Optional[Dict[Cell, Optional[Cell]]] = None):
        super().__init__(dist)
        self.parent = parent or {}
    def __iter__(self):
        # Backward compatible with older helpers that expect: dist, parent = bfs_dist(...)
        yield self
        yield self.parent


def bfs_dist(grid: Grid, start: Cell, goal: Optional[Cell] = None) -> BFSResult:
    n = len(grid)
    if not in_bounds(start[0], start[1], n) or grid[start[0]][start[1]] != 0:
        return BFSResult({}, {})
    q = deque([start])
    dist = {start: 0}
    parent: Dict[Cell, Optional[Cell]] = {start: None}
    while q:
        cur = q.popleft()
        for nb in neighbors(cur, n):
            if grid[nb[0]][nb[1]] == 0 and nb not in dist:
                dist[nb] = dist[cur] + 1
                parent[nb] = cur
                q.append(nb)
    return BFSResult(dist, parent)


def choose_weighted(rng: random.Random, weights: Dict[str, float]) -> str:
    clean = {k: max(0.0, float(v)) for k, v in weights.items() if float(v) > 0}
    if not clean:
        return rng.choice(list(weights.keys()))
    total = sum(clean.values())
    x = rng.random() * total
    acc = 0.0
    for k, v in clean.items():
        acc += v
        if x <= acc:
            return k
    return next(reversed(clean))


def grid_to_ascii(grid: Grid, start: Cell, goal: Cell) -> List[str]:
    rows = []
    for r, row in enumerate(grid):
        chars = []
        for c, v in enumerate(row):
            if (r, c) == start:
                chars.append("S")
            elif (r, c) == goal:
                chars.append("G")
            else:
                chars.append("." if v == 0 else "#")
        rows.append("".join(chars))
    return rows


def bin_parts(bin_key: str) -> Tuple[str, str, str, str]:
    p = str(bin_key).split("|")
    return (p + [""] * 4)[:4]  # type: ignore[return-value]


def full_bin_key_from_metrics(archive: Any, metrics: Dict[str, Any]) -> str:
    return archive.assign_bin(metrics)["bin_key"]


def compact_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "bfs_len", "rect_room_compressed_cycle_rank", "rect_room_macro_choice_count", "rect_room_macro_endpoint_count",
        "rect_room_macro_vertex_count", "rect_room_macro_edge_count", "parallel_macro_edge_count", "free_ratio", "reachable_free_ratio",
    ]
    return {k: metrics.get(k) for k in keys}


class LongBackboneGenerator:
    def __init__(self, n: int, rng: random.Random):
        self.n = n
        self.rng = rng

    def make_grid(self) -> Grid:
        n = self.n
        # Try long self-avoiding walk. It intentionally controls only backbone length,
        # not endpoint/choice/cycle exact counts.
        best: List[Cell] = []
        for _ in range(120):
            cur = self.rng.choice([(0, self.rng.randrange(n)), (n-1, self.rng.randrange(n)), (self.rng.randrange(n), 0), (self.rng.randrange(n), n-1)])
            path = [cur]
            used = {cur}
            last_dir: Optional[Cell] = None
            target_len = self.rng.randint(max(14, n + 6), min(n*n - 6, 36))
            for _step in range(target_len * 3):
                opts = []
                for dr, dc in ACTIONS:
                    nb = (cur[0] + dr, cur[1] + dc)
                    if not in_bounds(nb[0], nb[1], n) or nb in used:
                        continue
                    # Keep one-cell side walls often, but allow some bends and corridors.
                    touch_used = sum(1 for x in neighbors(nb, n) if x in used and x != cur)
                    if touch_used > 0 and self.rng.random() < 0.85:
                        continue
                    turn_bonus = 0.35 if last_dir is not None and (dr, dc) != last_dir else 0.0
                    edge_bonus = 0.15 if nb[0] in (0, n-1) or nb[1] in (0, n-1) else 0.0
                    opts.append((1.0 + turn_bonus + edge_bonus + self.rng.random() * 0.3, nb, (dr, dc)))
                if not opts:
                    break
                opts.sort(reverse=True, key=lambda x: x[0])
                _, nb, nd = opts[0] if self.rng.random() < 0.7 else self.rng.choice(opts)
                path.append(nb); used.add(nb); cur = nb; last_dir = nd
                if len(path) >= target_len and self.rng.random() < 0.35:
                    break
            if len(path) > len(best):
                best = path
            if len(best) >= 24:
                break
        if len(best) < 10:
            # Snake fallback: still not exact construction; just a long low-loop path.
            best = []
            for r in range(n):
                cols = range(n) if r % 2 == 0 else range(n-1, -1, -1)
                for c in cols:
                    best.append((r, c))
            cut = self.rng.randint(16, min(len(best), 34))
            offset = self.rng.randint(0, max(0, len(best) - cut))
            best = best[offset:offset+cut]
        grid = empty_grid(n, 1)
        for r, c in best:
            grid[r][c] = 0
        # Small stochastic side pockets leave space for TAI branch injection without making room soup.
        for r, c in list(best):
            if self.rng.random() < 0.08:
                walls = [nb for nb in neighbors((r, c), n) if grid[nb[0]][nb[1]] == 1]
                if walls:
                    x, y = self.rng.choice(walls)
                    grid[x][y] = 0
        return grid


class SecondaryBuffer:
    def __init__(self, max_size: int = 2000):
        self.items: List[Dict[str, Any]] = []
        self.max_size = max_size
        self.admit_reasons = Counter()
        self.evict_reasons = Counter()
        self.replay_count = 0
        self.accept_count = 0
        self.sparse_hit_count = 0

    def __len__(self) -> int:
        return len(self.items)

    def admit(self, item: Dict[str, Any]) -> None:
        self.items.append(item)
        for r in item.get("admit_reason", []):
            self.admit_reasons[r] += 1
        while len(self.items) > self.max_size:
            self.items.pop(0)
            self.evict_reasons["buffer_over_capacity"] += 1

    def sample(self, rng: random.Random) -> Optional[Dict[str, Any]]:
        if not self.items:
            return None
        # Prefer lower replay count and long candidates.
        weights = []
        for it in self.items:
            m = it.get("current_full_metrics", {})
            w = 1.0 / (1.0 + it.get("replay_count", 0))
            if bin_parts(it.get("current_full_bin_key", ""))[1] == "long" or (m.get("bfs_len") or 0) >= 13:
                w *= 1.8
            weights.append(w)
        total = sum(weights)
        x = rng.random() * total
        acc = 0.0
        for it, w in zip(self.items, weights):
            acc += w
            if x <= acc:
                it["replay_count"] = it.get("replay_count", 0) + 1
                it["last_replay_step"] = int(time.time())
                self.replay_count += 1
                return copy.deepcopy(it)
        return copy.deepcopy(self.items[-1])

    def prune(self) -> None:
        kept = []
        for it in self.items:
            if it.get("replay_count", 0) >= 3:
                self.evict_reasons["max_replay"] += 1
                continue
            if it.get("total_transform_count", 0) >= 10:
                self.evict_reasons["max_total_transform"] += 1
                continue
            if it.get("no_bin_improvement_count", 0) >= 2:
                self.evict_reasons["no_bin_improvement"] += 1
                continue
            kept.append(it)
        self.items = kept


def compute_deficit_scores(target: Dict[str, Any], counts: Counter) -> Dict[str, float]:
    buckets = target.get("bins", {})
    sums = Counter()
    total_obs_gap = 0.0
    for key, b in buckets.items():
        q = int(b.get("target_quota", 0))
        if q <= 0 or b.get("status") != "handdraw_observed":
            continue
        n = int(counts.get(key, 0))
        gap = max(0, q - n)
        if gap <= 0:
            continue
        total_obs_gap += gap
        cyc, bfs, choice, endpoint = bin_parts(key)
        if bfs == "long": sums["long"] += gap
        if endpoint in ("mid_endpoint", "high_endpoint"): sums["endpoint"] += gap
        if endpoint == "high_endpoint" and bfs == "long": sums["long_high_endpoint"] += gap
        if choice in ("mid_choice", "high_choice"): sums["choice"] += gap
        if cyc == "tree": sums["tree"] += gap
        if cyc == "single_loop": sums["single_loop"] += gap
        if cyc == "multi_loop": sums["multi_loop"] += gap
    denom = max(1.0, total_obs_gap)
    return {
        "long_deficit_score": float(sums["long"] / denom),
        "endpoint_deficit_score": float(sums["endpoint"] / denom),
        "choice_deficit_score": float(sums["choice"] / denom),
        "cycle_deficit_tree": float(sums["tree"] / denom),
        "cycle_deficit_single": float(sums["single_loop"] / denom),
        "cycle_deficit_multi": float(sums["multi_loop"] / denom),
        "long_high_endpoint_deficit_score": float(sums["long_high_endpoint"] / denom),
        "total_observed_gap": float(total_obs_gap),
    }


def generator_weights(policy: str, deficits: Dict[str, float]) -> Dict[str, float]:
    if policy == "fixed_baseline":
        return {
            "random_wall_generator": 1.0,
            "dfs_tree_generator": 1.0,
            "loose_path_first_generator": 1.0,
            "room_biased_generator": 1.0,
            "template_mutation_generator": 0.25,
        }
    long_d = deficits.get("long_deficit_score", 0.0)
    tree_d = deficits.get("cycle_deficit_tree", 0.0)
    # Dominance cap is implemented by limiting random wall base weight.
    return {
        "long_backbone_generator": 1.0 + 4.0 * long_d,
        "random_wall_generator": 1.0,
        "dfs_tree_generator": 0.8 + 2.0 * tree_d,
        "loose_path_first_generator": 0.7 + 1.5 * long_d,
        "room_biased_generator": 0.8,
        "template_mutation_generator": 0.15,
    }


def transform_weights(policy: str, metrics: Dict[str, Any], bin_key: str, deficits: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
    if policy == "fixed_baseline":
        return dict(base)
    bfs = metrics.get("bfs_len") or 0
    cyc = int(metrics.get("rect_room_compressed_cycle_rank") or 0)
    endpoint = int(metrics.get("rect_room_macro_endpoint_count") or 0)
    choice = int(metrics.get("rect_room_macro_choice_count") or 0)
    is_long = bin_parts(bin_key)[1] == "long" or bfs >= 13
    long_d = deficits.get("long_deficit_score", 0.0)
    endpoint_d = deficits.get("endpoint_deficit_score", 0.0)
    choice_d = deficits.get("choice_deficit_score", 0.0)
    multi_d = deficits.get("cycle_deficit_multi", 0.0)
    single_d = deficits.get("cycle_deficit_single", 0.0)
    tree_d = deficits.get("cycle_deficit_tree", 0.0)
    mult = {op: 1.0 for op in base}
    mult["branch_grow"] = 1.0 + 3.0 * endpoint_d + 1.2 * choice_d + (1.0 if is_long and endpoint_d > 0 else 0.0)
    if endpoint >= 7:
        mult["branch_grow"] *= 0.35
    mult["shortcut_block"] = 1.0 + 3.0 * long_d * (0.35 if is_long else 1.0)
    if is_long:
        mult["shortcut_block"] *= 0.45
    mult["loop_flip"] = 1.0 + 2.5 * multi_d + 1.0 * single_d - 1.5 * tree_d
    if endpoint_d > 0.25:
        mult["loop_flip"] *= 0.7
    mult["cell_flip_random"] = 0.5
    out = {}
    for op, w in base.items():
        out[op] = max(0.03, w * min(4.0, max(0.1, mult.get(op, 1.0))))
    return out


def is_adjacent_to_sparse_target(bin_key: str, target: Dict[str, Any], counts: Counter) -> bool:
    c, b, ch, e = bin_parts(bin_key)
    ranks = {"tree": 0, "single_loop": 1, "multi_loop": 2}
    choice_r = {"low_choice": 0, "mid_choice": 1, "high_choice": 2}
    end_r = {"low_endpoint": 0, "mid_endpoint": 1, "high_endpoint": 2}
    for key, tb in target.get("bins", {}).items():
        q = int(tb.get("target_quota", 0))
        if q <= 0 or counts.get(key, 0) >= q:
            continue
        cc, bb, chh, ee = bin_parts(key)
        d = abs(ranks.get(cc, 0) - ranks.get(c, 0)) + (0 if bb == b else 1) + abs(choice_r.get(chh, 0) - choice_r.get(ch, 0)) + abs(end_r.get(ee, 0) - end_r.get(e, 0))
        if d <= 2:
            return True
    return False


def should_admit_buffer(metrics: Dict[str, Any], bin_key: str, previous_metrics: Optional[Dict[str, Any]], previous_bin: Optional[str], deficits: Dict[str, float], target: Dict[str, Any], counts: Counter) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    bfs = int(metrics.get("bfs_len") or 0)
    endpoint = int(metrics.get("rect_room_macro_endpoint_count") or 0)
    choice = int(metrics.get("rect_room_macro_choice_count") or 0)
    cycle = int(metrics.get("rect_room_compressed_cycle_rank") or 0)
    if bin_parts(bin_key)[1] == "long":
        reasons.append("current_long_potential")
    if previous_metrics:
        db = bfs - int(previous_metrics.get("bfs_len") or 0)
        de = endpoint - int(previous_metrics.get("rect_room_macro_endpoint_count") or 0)
        dc = choice - int(previous_metrics.get("rect_room_macro_choice_count") or 0)
        dcy = cycle - int(previous_metrics.get("rect_room_compressed_cycle_rank") or 0)
        if db >= 2: reasons.append("bfs_len_increased")
        if de > 0 and deficits.get("endpoint_deficit_score", 0) > 0.15: reasons.append("endpoint_bin_or_count_up")
        if dc > 0 and deficits.get("choice_deficit_score", 0) > 0.15: reasons.append("choice_bin_or_count_up")
        if dcy > 0 and deficits.get("cycle_deficit_multi", 0) > 0.15: reasons.append("cycle_toward_multi")
    if is_adjacent_to_sparse_target(bin_key, target, counts):
        reasons.append("adjacent_to_sparse_target")
    if bin_parts(bin_key)[1] == "long" and (endpoint < 5 or choice < 5):
        reasons.append("long_with_injection_space")
    return bool(reasons), reasons


def initialize_reports() -> Dict[str, Any]:
    return {
        "candidate_total": 0,
        "hard_passed": 0,
        "full_evaluated": 0,
        "accepted": 0,
        "full_errors": 0,
        "reject_reasons": Counter(),
        "generator_selected": Counter(),
        "generator_full": Counter(),
        "generator_accepted": Counter(),
        "generator_long": Counter(),
        "generator_bins": defaultdict(Counter),
        "operator_attempts": Counter(),
        "operator_successes": Counter(),
        "operator_accepted_after": Counter(),
        "transform_delta": defaultdict(list),
        "full_bin_eval": Counter(),
        "full_bin_accept": Counter(),
        "from_lbs_by_bin": Counter(),
        "from_buffer_by_bin": Counter(),
        "full_by_gen_bin": defaultdict(Counter),
        "accepted_by_gen_bin": defaultdict(Counter),
        "full_by_op_bin": defaultdict(Counter),
        "accepted_by_op_bin": defaultdict(Counter),
        "gen_weight_windows": [],
        "transform_weight_windows": [],
        "lbs_attempts": 0,
        "lbs_hard": 0,
        "lbs_full": 0,
        "lbs_long": 0,
        "lbs_accepted": 0,
        "lbs_bfs_values": [],
        "lbs_bins": Counter(),
    }


def summarize_counter(c: Counter, n: int = 10) -> List[List[Any]]:
    return [[k, v] for k, v in c.most_common(n)]


def summarize_numeric(vals: List[Any]) -> Dict[str, Any]:
    xs = [float(v) for v in vals if v is not None]
    if not xs:
        return {"n": 0, "min": None, "p50": None, "mean": None, "max": None}
    ys = sorted(xs)
    return {"n": len(xs), "min": ys[0], "p50": ys[len(ys)//2], "mean": statistics.mean(xs), "max": ys[-1]}


def l1_to_target(target: Dict[str, Any], counts: Counter) -> float:
    total = sum(counts.values()) or 1
    return sum(abs(counts.get(k, 0) / total - float(b.get("target_prob", 0.0))) for k, b in target.get("bins", {}).items())



def build_target_archive_summary_local(target: Dict[str, Any], counts: Counter) -> Dict[str, Any]:
    bins = {}
    for k, b in target.get("bins", {}).items():
        q = int(b.get("target_quota", 0))
        n = int(counts.get(k, 0))
        bins[k] = {
            "bin_key": k,
            "status": b.get("status"),
            "target_prob": float(b.get("target_prob", 0.0)),
            "target_quota": q,
            "confirmed_count": n,
            "quota_gap": max(0, q - n),
            "fill_ratio": n / q if q else 0.0,
            "handdraw_count": int(b.get("handdraw_count", 0) or 0),
            "handdraw_prob": float(b.get("handdraw_prob", 0.0) or 0.0),
        }
    return {"target_total": int(target.get("target_total", sum(int(b.get("target_quota", 0)) for b in target.get("bins", {}).values()))), "quota_sum": int(target.get("quota_sum", sum(x["target_quota"] for x in bins.values()))), "confirmed_total": int(sum(counts.values())), "bins": bins}


def build_synthetic_vs_target_report_local(target: Dict[str, Any], counts: Counter) -> Dict[str, Any]:
    total = int(sum(counts.values()))
    target_total = int(target.get("target_total", sum(int(b.get("target_quota", 0)) for b in target.get("bins", {}).values())) or 1)
    bins = []
    nonzero = 0
    filled_nonzero = 0
    handdraw_obs = 0
    handdraw_obs_filled = 0
    structural_confirmed = 0
    l1 = 0.0
    eps = 1e-9
    kl = 0.0
    for k, b in target.get("bins", {}).items():
        q = int(b.get("target_quota", 0))
        n = int(counts.get(k, 0))
        p_t = float(b.get("target_prob", 0.0))
        p_s = n / total if total else 0.0
        l1 += abs(p_s - p_t)
        if p_t > 0:
            kl += p_t * math.log((p_t + eps) / (p_s + eps))
        if q > 0:
            nonzero += 1
            if n > 0:
                filled_nonzero += 1
        if b.get("status") == "handdraw_observed":
            handdraw_obs += 1
            if n > 0:
                handdraw_obs_filled += 1
        if b.get("status") == "structural_infeasible_or_constrained" and n > 0:
            structural_confirmed += n
        bins.append({"bin_key": k, "status": b.get("status"), "target_prob": p_t, "target_quota": q, "confirmed_count": n, "quota_gap": max(0, q-n), "fill_ratio": n/q if q else 0.0, "handdraw_count": b.get("handdraw_count", 0), "handdraw_prob": b.get("handdraw_prob", 0.0)})
    return {"target_total": target_total, "confirmed_total": total, "quota_sum": sum(int(b.get("target_quota", 0)) for b in target.get("bins", {}).values()), "target_fill_ratio": total / target_total if target_total else 0.0, "filled_target_bins": sum(1 for v in counts.values() if v > 0), "target_bins_with_nonzero_quota": nonzero, "filled_nonzero_target_bins": filled_nonzero, "raw_coverage_all_81": sum(1 for v in counts.values() if v > 0) / 81.0, "coverage_nonzero_target_bins": filled_nonzero / nonzero if nonzero else 0.0, "observed_handdraw_bin_fill_rate": handdraw_obs_filled / handdraw_obs if handdraw_obs else 0.0, "structural_bin_accept_count": structural_confirmed, "l1_distance_to_target_distribution": l1, "kl_divergence_target_to_synthetic_smooth": kl, "bins": bins}

def target_gap_report(target: Dict[str, Any], counts: Counter, full_eval: Counter, accept: Counter, gen_full: Dict[str, Counter], gen_acc: Dict[str, Counter], op_full: Dict[str, Counter], op_acc: Dict[str, Counter], examples: Dict[str, List[str]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rows = []
    for key, b in target.get("bins", {}).items():
        q = int(b.get("target_quota", 0))
        if q <= 0:
            continue
        confirmed = int(counts.get(key, 0))
        fe = int(full_eval.get(key, 0))
        status = "filled" if confirmed >= q else ("never_full_evaluated" if fe == 0 else "generated_but_underfilled")
        rows.append({
            "bin_key": key,
            "status": b.get("status"),
            "target_quota": q,
            "confirmed_count": confirmed,
            "quota_gap": max(0, q - confirmed),
            "fill_ratio": confirmed / q if q else 0,
            "full_evaluated_count": fe,
            "accepted_count": int(accept.get(key, 0)),
            "generation_status": status,
            "top_generators_by_full_evaluated": summarize_counter(Counter({g: m.get(key, 0) for g, m in gen_full.items()}), 5),
            "top_generators_by_accepted": summarize_counter(Counter({g: m.get(key, 0) for g, m in gen_acc.items()}), 5),
            "top_transforms_by_full_evaluated": summarize_counter(Counter({op: m.get(key, 0) for op, m in op_full.items()}), 5),
            "top_transforms_by_accepted": summarize_counter(Counter({op: m.get(key, 0) for op, m in op_acc.items()}), 5),
            "example_candidate_ids": examples.get(key, [])[:10],
        })
    report = {"bins": rows}
    nonzero = [r for r in rows if r["target_quota"] > 0]
    summary = {
        "target_total": int(target.get("target_total", sum(int(b.get("target_quota", 0)) for b in target.get("bins", {}).values()))),
        "confirmed_total": int(sum(counts.values())),
        "target_fill_ratio": (sum(counts.values()) / float(target.get("target_total", 1) or 1)),
        "nonzero_target_bins": len(nonzero),
        "filled_nonzero_target_bins": sum(1 for r in nonzero if r["confirmed_count"] > 0),
        "underfilled_nonzero_target_bins": sum(1 for r in nonzero if r["confirmed_count"] < r["target_quota"]),
        "handdraw_observed_bins": sum(1 for k,b in target.get("bins",{}).items() if b.get("status") == "handdraw_observed"),
        "filled_handdraw_observed_bins": sum(1 for r in nonzero if target.get("bins",{}).get(r["bin_key"],{}).get("status") == "handdraw_observed" and r["confirmed_count"] > 0),
        "underfilled_handdraw_observed_bins": sum(1 for r in nonzero if target.get("bins",{}).get(r["bin_key"],{}).get("status") == "handdraw_observed" and r["confirmed_count"] < r["target_quota"]),
        "never_full_evaluated_target_bins": [r["bin_key"] for r in nonzero if r["generation_status"] == "never_full_evaluated"],
        "generated_but_underfilled_target_bins": [r["bin_key"] for r in nonzero if r["generation_status"] == "generated_but_underfilled"],
        "top_quota_gap_bins": sorted(nonzero, key=lambda r: r["quota_gap"], reverse=True)[:10],
        "top_low_fill_ratio_bins": sorted(nonzero, key=lambda r: (r["fill_ratio"], -r["target_quota"]))[:10],
    }
    return report, summary




def resolve_output_dir(requested: Path, resume: bool = False, overwrite: bool = False) -> Path:
    """Resolve output directory without silent mixed writes."""
    requested = Path(requested)
    if not requested.exists():
        ensure_dir(requested)
        write_json(requested / "output_dir_resolution_report.json", {"requested_output_dir": str(requested), "actual_output_dir": str(requested), "reason": "created_new"})
        return requested
    if overwrite:
        shutil.rmtree(requested)
        ensure_dir(requested)
        write_json(requested / "output_dir_resolution_report.json", {"requested_output_dir": str(requested), "actual_output_dir": str(requested), "reason": "overwrite_requested"})
        return requested
    if resume:
        raise SystemExit("[ERROR] --resume is declared but full resume is not implemented yet; use --overwrite or choose a new --run-name.")
    parent = requested.parent
    stem = requested.name
    i = 1
    while True:
        cand = parent / f"{stem}({i})"
        if not cand.exists():
            ensure_dir(cand)
            write_json(cand / "output_dir_resolution_report.json", {"requested_output_dir": str(requested), "actual_output_dir": str(cand), "reason": "requested_dir_exists_and_no_resume_or_overwrite"})
            return cand
        i += 1

def save_process_trace(out_dir: Path, bin_key: str, record: Dict[str, Any], max_per_bin: int, counters: Counter) -> None:
    if counters[bin_key] >= max_per_bin:
        return
    sub = out_dir / "sparse_bin_process_traces" / safe_name(bin_key)
    ensure_dir(sub)
    append_jsonl(sub / "examples.jsonl", record)
    counters[bin_key] += 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=TITLE)
    p.add_argument("--mode", choices=["generate_until_target", "multi_seed_until_target", "validate_quality"], default="generate_until_target")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--output-root", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--target-distribution-json", default=DEFAULT_TARGET_DISTRIBUTION_REL)
    p.add_argument("--target-total-samples", type=int, default=None)
    p.add_argument("--save-visualizations", action="store_true")
    p.add_argument("--input-run-dirs", nargs="*", default=None)
    p.add_argument("--handdraw-jsonl", default="feature_maze/handdraw_mazes_ascii.jsonl")
    # Stop criteria. By default max limits are disabled.
    p.add_argument("--target-fill-ratio-threshold", type=float, default=0.85)
    p.add_argument("--coverage-nonzero-target-bins-threshold", type=float, default=0.90)
    p.add_argument("--observed-handdraw-bin-fill-rate-threshold", type=float, default=1.00)
    p.add_argument("--l1-distance-threshold", type=float, default=0.30)
    p.add_argument("--require-critical-bins-nonzero", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--require-full-extraction-error-zero", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-seeds", type=int, default=0, help="0 means no explicit seed cap")
    p.add_argument("--max-wall-clock-seconds", type=float, default=0.0, help="0 means no explicit wall-clock cap")
    p.add_argument("--max-full-evaluations", type=int, default=0, help="0 means no explicit full-evaluation cap")
    p.add_argument("--stop-policy", choices=["quality_threshold", "hard_quota", "both"], default="hard_quota")
    p.add_argument("--quota-scope", choices=["all_nonzero", "exploration_stress_all_nonzero", "handdraw_observed", "hard_observed_plus_critical"], default="handdraw_observed")
    p.add_argument("--progress-snapshot-interval", type=int, default=100)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--per-bin-visualization-limit", type=int, default=5)
    p.add_argument("--visualization-cell-size", type=int, default=32)
    p.add_argument("--closure-l1-threshold", type=float, default=0.05)
    p.add_argument("--export-samples-csv", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--samples-csv-name", default="samples.csv")
    # Internal frozen algorithm knobs retained for reproducibility, not exposed as alternate policies.
    p.add_argument("--max-phase-attempts-per-item", type=int, default=100)
    p.add_argument("--max-lifetime-attempts-per-item", type=int, default=300)
    p.add_argument("--max-accepts-per-item", type=int, default=3)
    p.add_argument("--max-no-progress-streak", type=int, default=20)
    p.add_argument("--max-hard-collapse-count", type=int, default=10)
    p.add_argument("--max-route-collapse-count", type=int, default=30)
    p.add_argument("--max-route-reassignments", type=int, default=2)
    p.add_argument("--max-buffer-size", type=int, default=2000)
    p.add_argument("--max-trace-examples-per-bin", type=int, default=20)
    p.add_argument("--target-route-top-k", type=int, default=8)
    p.add_argument("--critical-route-priority", type=float, default=2.0)
    p.add_argument("--diameter-start-goal-sample-count", type=int, default=64)
    p.add_argument("--pure-tbi-max-branch-len", type=int, default=4)
    p.add_argument("--pure-tbi-min-branch-len", type=int, default=1)
    p.add_argument("--pure-tbi-max-roots-per-item", type=int, default=12)
    p.add_argument("--clo-max-added-path-len", type=int, default=4)
    p.add_argument("--clo-min-cycle-gain", type=int, default=1)
    p.add_argument("--spb-max-bfs-drop", type=int, default=0)
    args = p.parse_args()
    # Frozen formal policy. No fixed/lbs/legacy compatibility modes are exposed.
    args.mvp_policy = "target_class_sfg_rbr"
    args.enable_sfg = True
    args.enable_tbi = True
    args.enable_clo = True
    args.enable_spb = True
    args.enable_rbr = True
    args.enable_pure_tbi = True
    args.enable_stage_clo = True
    args.enable_lbs = False
    args.enable_tai = False
    args.enable_sbr = False
    args.max_total_full_extractions = int(args.max_full_evaluations or 0)
    return args


def run_generate_legacy11(args: argparse.Namespace) -> Path:
    rng = random.Random(args.seed)
    config = F.default_config()
    config.setdefault("rect_room_extraction", {})["cheap_metrics_only"] = False
    output_root = Path(args.output_root or DEFAULT_OUTPUT_ROOT_REL)
    run_name = args.run_name or f"lbs_tai_sbr_mvp_{int(time.time())}"
    out_dir = output_root / run_name
    ensure_dir(out_dir)
    # Clear append-only files for repeatable runs.
    for name in ["confirmed_samples.jsonl", "full_evaluated_samples.jsonl", "rejected_sample_examples.jsonl", "candidate_debug_records.jsonl", "sparse_target_bin_trace_examples.jsonl"]:
        p = out_dir / name
        if p.exists(): p.unlink()
    target_path = Path(args.target_distribution_json)
    target = F.load_target_distribution(target_path, args.target_total_samples, config)
    F.write_json(out_dir / "target_distribution_used.json", target)
    archive = F.TargetDistributionArchive(target, config.get("diversity", {}), config)
    quantizer = F.Quantizer(config)
    start_goal_assigner = F.StartGoalAssigner(config, rng)
    seed_gens = F.SeedGenerators(config, rng)
    transforms = F.Transforms(config, rng)
    lbs_gen = LongBackboneGenerator(int(config["grid"].get("size", 8)), rng)
    base_transform_weights = dict(config.get("transform", {}).get("operator_weights", {"loop_flip":1,"branch_grow":1,"shortcut_block":1,"cell_flip_random":0.4}))
    max_steps = int(config.get("transform", {}).get("max_transform_steps", 6))
    reports = initialize_reports()
    secondary = SecondaryBuffer(max_size=2000)
    full_eval_examples_by_bin: Dict[str, List[str]] = defaultdict(list)
    trace_counters = Counter()
    accepted_records: List[Dict[str, Any]] = []
    source_reference = getattr(F, "__mvp_import_path__", "unknown")
    write_json(out_dir / "run_metadata.json", {
        "version": VERSION,
        "title": TITLE,
        "experiment_role": "minimal_verifiable_algorithm_mvp",
        "does_not_modify_3_0_4_source": True,
        "source_reference": source_reference,
        "mvp_script_path": str(Path(__file__).resolve()),
        "final_archive_bin_source": "full_bin_key",
        "mvp_policy": args.mvp_policy,
        "enable_lbs": bool(args.enable_lbs and args.mvp_policy == "lbs_patch"),
        "enable_tai": bool(args.enable_tai and args.mvp_policy == "lbs_patch"),
        "enable_sbr": bool(args.enable_sbr and args.mvp_policy == "lbs_patch"),
    })
    write_json(out_dir / "resolved_config.json", {"base_config": config, "mvp_args": vars(args), "critical_long_bins": CRITICAL_LONG_BINS})

    for seed_idx in progress(range(int(args.n_seeds)), desc="3.0.4(11) MVP seeds"):
        deficits = compute_deficit_scores(target, archive.counts)
        use_buffer = False
        replay_prob = 0.0
        if args.mvp_policy == "lbs_patch" and args.enable_sbr and len(secondary) > 0:
            replay_prob = min(0.50, 0.15 + 0.35 * deficits.get("long_deficit_score", 0.0))
            use_buffer = rng.random() < replay_prob
        if use_buffer:
            item = secondary.sample(rng)
            if item is None:
                use_buffer = False
            else:
                cand = F.MazeCandidate(
                    grid=copy.deepcopy(item["grid"]),
                    start=tuple(item["start"]),
                    goal=tuple(item["goal"]),
                    origin_generator=item["origin_generator"],
                    origin_seed_id=item["origin_seed_id"],
                    transform_history=copy.deepcopy(item.get("transform_history", [])),
                    candidate_id=f"buffer_replay{seed_idx:06d}_step0",
                )
                cand.transform_history.append({"operator": "secondary_buffer_replay", "status": "success", "replay_count": item.get("replay_count", 0)})
                from_buffer_seed = True
        if not use_buffer:
            gen_weights = generator_weights(args.mvp_policy if args.enable_lbs else "fixed_baseline", deficits)
            gen_name = choose_weighted(rng, gen_weights)
            reports["generator_selected"][gen_name] += 1
            # Periodically capture generator weights.
            if seed_idx % 250 == 0:
                reports["gen_weight_windows"].append({"seed_idx": seed_idx, "weights": gen_weights, "deficits": deficits})
            if gen_name == "long_backbone_generator":
                reports["lbs_attempts"] += 1
                grid = lbs_gen.make_grid()
            else:
                grid = seed_gens.make_grid(gen_name)
            s, g, _sg = start_goal_assigner.assign(grid, archive)
            cand = F.MazeCandidate(grid=grid, start=s, goal=g, origin_generator=gen_name, origin_seed_id=f"seed{seed_idx:06d}_{gen_name}", transform_history=[], candidate_id=f"maze_seed{seed_idx:06d}_step0")
            from_buffer_seed = False
        process_trace: List[Dict[str, Any]] = []
        previous_full_metrics: Optional[Dict[str, Any]] = None
        previous_full_bin: Optional[str] = None
        no_bin_improvement = 0
        for step in range(max_steps + 1):
            cand.candidate_id = f"{cand.origin_seed_id}_step{step}" if not from_buffer_seed else f"buffer_replay{seed_idx:06d}_step{step}"
            reports["candidate_total"] += 1
            cheap = quantizer.evaluate(cand)
            hard = F.hard_quality_filter(cand, config, archive.accepted_hashes)
            if not hard.get("hard_quality_passed"):
                reports["reject_reasons"][hard.get("hard_reject_reason", "hard_quality_failed")] += 1
                append_jsonl(out_dir / "candidate_debug_records.jsonl", {"candidate_id": cand.candidate_id, "hard_quality": hard, "accepted": False})
                current_full_metrics = None
                current_full_bin = None
            else:
                reports["hard_passed"] += 1
                if cand.origin_generator == "long_backbone_generator": reports["lbs_hard"] += 1
                full_metrics, full_actual, rt_ms, err = F.full_metrics_for_candidate(cand, cheap, config)
                if err:
                    reports["full_errors"] += 1
                    reports["reject_reasons"]["full_extraction_error"] += 1
                    append_jsonl(out_dir / "rejected_sample_examples.jsonl", {"candidate_id": cand.candidate_id, "error": err, "grid": cand.grid, "start": list(cand.start), "goal": list(cand.goal)})
                    current_full_metrics = None; current_full_bin = None
                else:
                    reports["full_evaluated"] += 1
                    current_full_metrics = full_metrics
                    current_full_bin_info = archive.assign_bin(full_metrics)
                    current_full_bin = current_full_bin_info["bin_key"]
                    reports["full_bin_eval"][current_full_bin] += 1
                    reports["generator_full"][cand.origin_generator] += 1
                    reports["generator_bins"][cand.origin_generator][current_full_bin] += 1
                    if (full_metrics.get("bfs_len") or 0) >= 13 or bin_parts(current_full_bin)[1] == "long":
                        reports["generator_long"][cand.origin_generator] += 1
                        if cand.origin_generator == "long_backbone_generator": reports["lbs_long"] += 1
                    if cand.origin_generator == "long_backbone_generator":
                        reports["lbs_full"] += 1
                        reports["lbs_bfs_values"].append(full_metrics.get("bfs_len") or 0)
                        reports["lbs_bins"][current_full_bin] += 1
                    last_op = cand.transform_history[-1].get("operator") if cand.transform_history else "seed_step"
                    reports["full_by_gen_bin"][cand.origin_generator][current_full_bin] += 1
                    reports["full_by_op_bin"][last_op][current_full_bin] += 1
                    if current_full_bin in CRITICAL_LONG_BINS and len(full_eval_examples_by_bin[current_full_bin]) < 20:
                        full_eval_examples_by_bin[current_full_bin].append(cand.candidate_id)
                    target_b = target["bins"].get(current_full_bin, {})
                    accepted, acceptance, reject_reason = archive.maybe_accept(cand, full_metrics, current_full_bin_info, rng)
                    if accepted:
                        reports["accepted"] += 1
                        reports["full_bin_accept"][current_full_bin] += 1
                        reports["generator_accepted"][cand.origin_generator] += 1
                        reports["accepted_by_gen_bin"][cand.origin_generator][current_full_bin] += 1
                        reports["accepted_by_op_bin"][last_op][current_full_bin] += 1
                        if last_op != "seed_step": reports["operator_accepted_after"][last_op] += 1
                        if cand.origin_generator == "long_backbone_generator": reports["lbs_accepted"] += 1
                        if from_buffer_seed:
                            secondary.accept_count += 1
                            if current_full_bin in CRITICAL_LONG_BINS or is_adjacent_to_sparse_target(current_full_bin, target, archive.counts):
                                secondary.sparse_hit_count += 1
                        if cand.origin_generator == "long_backbone_generator": reports["from_lbs_by_bin"][current_full_bin] += 1
                        if from_buffer_seed: reports["from_buffer_by_bin"][current_full_bin] += 1
                        rec = {
                            "maze_id": cand.candidate_id,
                            "sample_id": cand.candidate_id,
                            "grid": cand.grid,
                            "ascii_grid": grid_to_ascii(cand.grid, cand.start, cand.goal),
                            "start": list(cand.start),
                            "goal": list(cand.goal),
                            "origin_generator": cand.origin_generator,
                            "origin_seed_id": cand.origin_seed_id,
                            "from_lbs": cand.origin_generator == "long_backbone_generator",
                            "from_secondary_buffer": from_buffer_seed,
                            "transform_history": cand.transform_history,
                            "metrics": full_metrics,
                            "full_bin": current_full_bin_info,
                            "target_bin": {"status": target_b.get("status"), "target_quota": target_b.get("target_quota"), "target_prob": target_b.get("target_prob"), "handdraw_count": target_b.get("handdraw_count"), "count_before": acceptance.get("count_before"), "count_after": acceptance.get("count_after")},
                            "acceptance": {"accepted": True, "reject_reason": None, "policy": "full_bin_key_target_quota"},
                        }
                        append_jsonl(out_dir / "confirmed_samples.jsonl", rec)
                        accepted_records.append(rec)
                    else:
                        reports["reject_reasons"][reject_reason or "unknown_full_reject"] += 1
                        can_admit, admit_reason = should_admit_buffer(full_metrics, current_full_bin, previous_full_metrics, previous_full_bin, deficits, target, archive.counts)
                        if args.mvp_policy == "lbs_patch" and args.enable_sbr and can_admit and reject_reason not in ("duplicate_hash_reject",):
                            secondary.admit({
                                "grid": copy.deepcopy(cand.grid), "start": list(cand.start), "goal": list(cand.goal),
                                "origin_generator": cand.origin_generator, "origin_seed_id": cand.origin_seed_id,
                                "transform_history": copy.deepcopy(cand.transform_history), "current_full_metrics": compact_metrics(full_metrics),
                                "current_full_bin_key": current_full_bin, "last_delta": {}, "admit_reason": admit_reason,
                                "replay_count": 0, "total_transform_count": step, "no_bin_improvement_count": no_bin_improvement,
                                "created_step": seed_idx, "last_replay_step": None,
                            })
                    full_record = {
                        "candidate_id": cand.candidate_id,
                        "origin_generator": cand.origin_generator,
                        "origin_seed_id": cand.origin_seed_id,
                        "last_operator": last_op,
                        "transform_history": cand.transform_history,
                        "full_bin_key": current_full_bin,
                        "full_bin": current_full_bin_info,
                        "full_metrics": compact_metrics(full_metrics),
                        "target_bin": {"status": target_b.get("status"), "target_quota": target_b.get("target_quota"), "target_prob": target_b.get("target_prob"), "handdraw_count": target_b.get("handdraw_count"), "count_before": int(archive.counts.get(current_full_bin, 0))},
                        "accepted": accepted,
                        "reject_reason": None if accepted else reject_reason,
                        "from_lbs": cand.origin_generator == "long_backbone_generator",
                        "from_secondary_buffer": from_buffer_seed,
                    }
                    append_jsonl(out_dir / "full_evaluated_samples.jsonl", full_record)
                    if current_full_bin in CRITICAL_LONG_BINS:
                        trace_record = {
                            "final_bin_key": current_full_bin,
                            "accepted": accepted,
                            "origin_generator": cand.origin_generator,
                            "origin_seed_id": cand.origin_seed_id,
                            "from_lbs": cand.origin_generator == "long_backbone_generator",
                            "entered_secondary_buffer": False,
                            "buffer_replay_count": 1 if from_buffer_seed else 0,
                            "candidate_id": cand.candidate_id,
                            "start": list(cand.start),
                            "goal": list(cand.goal),
                            "process": process_trace + [{"step": step, "operator": last_op, "bin_after": current_full_bin, "metrics_after": compact_metrics(full_metrics), "note": "final_evaluation"}],
                            "acceptance": {"accepted": accepted, "reject_reason": None if accepted else reject_reason, "target_quota": target_b.get("target_quota"), "count_before": archive.counts.get(current_full_bin, 0)},
                            "grid": cand.grid,
                        }
                        save_process_trace(out_dir, current_full_bin, trace_record, args.max_trace_examples_per_bin, trace_counters)
                        append_jsonl(out_dir / "sparse_target_bin_trace_examples.jsonl", trace_record)
                    if previous_full_bin is not None and current_full_bin == previous_full_bin:
                        no_bin_improvement += 1
                    else:
                        no_bin_improvement = 0
                    previous_full_metrics = full_metrics
                    previous_full_bin = current_full_bin
            if step == max_steps:
                break
            deficits = compute_deficit_scores(target, archive.counts)
            if args.mvp_policy == "lbs_patch" and args.enable_tai and current_full_metrics is not None and current_full_bin is not None:
                tw = transform_weights("lbs_patch", current_full_metrics, current_full_bin, deficits, base_transform_weights)
            else:
                tw = transform_weights("fixed_baseline", cheap, "", deficits, base_transform_weights)
            if seed_idx % 250 == 0 and step == 0:
                reports["transform_weight_windows"].append({"seed_idx": seed_idx, "step": step, "weights": tw, "deficits": deficits})
            op = choose_weighted(rng, tw)
            reports["operator_attempts"][op] += 1
            before_metrics = previous_full_metrics if previous_full_metrics is not None else cheap
            before_bin = previous_full_bin
            child, op_info = transforms.apply(op, cand, cheap)
            if child is None:
                reports["reject_reasons"][f"transform_failed:{op_info.get('reason','unknown')}"] += 1
                break
            reports["operator_successes"][op] += 1
            # Trace only uses full metrics when they are available; otherwise null.
            child.candidate_id = f"{cand.origin_seed_id}_step{step+1}"
            after_cheap = quantizer.evaluate(child)
            after_hard = F.hard_quality_filter(child, config, archive.accepted_hashes)
            after_metrics = None; after_bin = None
            if after_hard.get("hard_quality_passed"):
                fm, _act, _rt, ferr = F.full_metrics_for_candidate(child, after_cheap, config)
                if ferr is None:
                    after_metrics = fm
                    after_bin = archive.assign_bin(fm)["bin_key"]
                    delta = {
                        "bfs_len": (fm.get("bfs_len") or 0) - ((before_metrics or {}).get("bfs_len") or 0),
                        "cycle_rank": (fm.get("rect_room_compressed_cycle_rank") or 0) - ((before_metrics or {}).get("rect_room_compressed_cycle_rank") or 0),
                        "choice_count": (fm.get("rect_room_macro_choice_count") or 0) - ((before_metrics or {}).get("rect_room_macro_choice_count") or 0),
                        "endpoint_count": (fm.get("rect_room_macro_endpoint_count") or 0) - ((before_metrics or {}).get("rect_room_macro_endpoint_count") or 0),
                    }
                    for k, v in delta.items(): reports["transform_delta"][f"{op}:{k}"].append(v)
                else:
                    delta = {}
            else:
                delta = {}
            process_trace.append({
                "step": step + 1,
                "operator": op,
                "bin_before": before_bin,
                "bin_after": after_bin,
                "metrics_before": compact_metrics(before_metrics or {}),
                "metrics_after": compact_metrics(after_metrics or {}),
                "delta_bfs": delta.get("bfs_len"),
                "delta_cycle_rank": delta.get("cycle_rank"),
                "delta_choice_count": delta.get("choice_count"),
                "delta_endpoint_count": delta.get("endpoint_count"),
                "scheduler_or_weight_info": {"weights": tw, "deficits": deficits},
                "admit_to_buffer": False,
                "admit_reason": [],
            })
            cand = child
        secondary.prune()

    # Reports
    archive_report = archive.coverage_report()
    write_json(out_dir / "target_archive_summary.json", archive_report)
    total_target = int(target.get("target_total", archive_report.get("target_total", 1) or 1))
    synthetic = {
        "target_total": total_target,
        "confirmed_total": int(sum(archive.counts.values())),
        "target_fill_ratio": int(sum(archive.counts.values())) / float(total_target or 1),
        "coverage_nonzero_target_bins": archive_report.get("coverage_nonzero_target_bins"),
        "observed_handdraw_bin_fill_rate": archive_report.get("observed_handdraw_bin_fill_rate"),
        "l1_distance_to_target_distribution": l1_to_target(target, archive.counts),
        "structural_confirmed_count": sum(archive.counts.get(k, 0) for k,b in target.get("bins",{}).items() if b.get("status") == "structural_infeasible_or_constrained"),
    }
    write_json(out_dir / "synthetic_vs_target_distribution_report.json", synthetic)
    tg_report, tg_summary = target_gap_report(target, archive.counts, reports["full_bin_eval"], reports["full_bin_accept"], reports["full_by_gen_bin"], reports["accepted_by_gen_bin"], reports["full_by_op_bin"], reports["accepted_by_op_bin"], full_eval_examples_by_bin)
    write_json(out_dir / "target_gap_attribution_report.json", tg_report)
    write_json(out_dir / "target_gap_summary.json", tg_summary)
    # Critical bins.
    critical = {}
    for key in CRITICAL_LONG_BINS:
        critical[key] = {
            "target_quota": int(target.get("bins", {}).get(key, {}).get("target_quota", 0)),
            "confirmed_count": int(archive.counts.get(key, 0)),
            "full_evaluated_count": int(reports["full_bin_eval"].get(key, 0)),
            "accepted_count": int(reports["full_bin_accept"].get(key, 0)),
            "fill_ratio": (archive.counts.get(key, 0) / float(target.get("bins", {}).get(key, {}).get("target_quota", 1) or 1)),
            "full_evaluated_by_generator": {g: int(m.get(key, 0)) for g, m in reports["full_by_gen_bin"].items()},
            "accepted_by_generator": {g: int(m.get(key, 0)) for g, m in reports["accepted_by_gen_bin"].items()},
            "full_evaluated_by_operator": {op: int(m.get(key, 0)) for op, m in reports["full_by_op_bin"].items()},
            "accepted_by_operator": {op: int(m.get(key, 0)) for op, m in reports["accepted_by_op_bin"].items()},
            "from_lbs_count": int(reports["from_lbs_by_bin"].get(key, 0)),
            "from_secondary_buffer_count": int(reports["from_buffer_by_bin"].get(key, 0)),
            "top_process_examples": full_eval_examples_by_bin.get(key, [])[:10],
        }
    write_json(out_dir / "critical_long_bins_report.json", critical)
    lbs_bfs = reports["lbs_bfs_values"]
    write_json(out_dir / "long_backbone_generation_report.json", {
        "LBS_attempts": reports["lbs_attempts"],
        "LBS_hard_quality_pass": reports["lbs_hard"],
        "LBS_full_evaluated": reports["lbs_full"],
        "LBS_long_bfs_count": reports["lbs_long"],
        "LBS_accepted_count": reports["lbs_accepted"],
        "LBS_top_bins": summarize_counter(reports["lbs_bins"], 20),
        "LBS_avg_bfs_len": statistics.mean(lbs_bfs) if lbs_bfs else None,
        "LBS_bfs_bin_distribution": Counter(bin_parts(k)[1] for k in reports["lbs_bins"].elements()),
        "LBS_hash_unique_count": None,
    })
    write_json(out_dir / "generator_weight_report.json", {
        "initial_weights": reports["gen_weight_windows"][0] if reports["gen_weight_windows"] else {},
        "windowed_weights": reports["gen_weight_windows"],
        "generator_selected_counts": dict(reports["generator_selected"]),
        "generator_full_evaluated_counts": dict(reports["generator_full"]),
        "generator_accepted_counts": dict(reports["generator_accepted"]),
        "generator_long_bfs_counts": dict(reports["generator_long"]),
        "generator_top_bins": {g: summarize_counter(c, 10) for g, c in reports["generator_bins"].items()},
    })
    avg_delta = {k: (statistics.mean(v) if v else 0.0) for k, v in reports["transform_delta"].items()}
    write_json(out_dir / "transform_weight_report.json", {"windowed_weights": reports["transform_weight_windows"], "operator_attempts": dict(reports["operator_attempts"]), "operator_successes": dict(reports["operator_successes"]), "operator_accepted_after": dict(reports["operator_accepted_after"]), "avg_deltas": avg_delta})
    write_json(out_dir / "secondary_buffer_report.json", {"buffer_admit_count": sum(secondary.admit_reasons.values()), "buffer_size_final": len(secondary), "buffer_replay_count": secondary.replay_count, "buffer_accept_count": secondary.accept_count, "buffer_sparse_hit_count": secondary.sparse_hit_count, "buffer_eviction_count": sum(secondary.evict_reasons.values()), "top_admit_reasons": summarize_counter(secondary.admit_reasons), "top_evict_reasons": summarize_counter(secondary.evict_reasons)})
    write_json(out_dir / "transform_effect_by_context_report.json", {"avg_delta_by_operator_metric": avg_delta})
    write_json(out_dir / "before_after_bucket_shift_report.json", {"note": "See sparse_bin_process_traces/*/examples.jsonl for per-step full-bin before/after shifts."})
    write_json(out_dir / "generator_transform_replay_attribution_report.json", {"full_by_generator_bin": {g: dict(c) for g,c in reports["full_by_gen_bin"].items()}, "accepted_by_generator_bin": {g: dict(c) for g,c in reports["accepted_by_gen_bin"].items()}, "full_by_operator_bin": {op: dict(c) for op,c in reports["full_by_op_bin"].items()}, "accepted_by_operator_bin": {op: dict(c) for op,c in reports["accepted_by_op_bin"].items()}})
    write_json(out_dir / "full_evaluated_bin_distribution.json", {"total_full_evaluated": reports["full_evaluated"], "bins": {k: {"full_evaluated_count": int(reports["full_bin_eval"].get(k,0)), "accepted_count": int(reports["full_bin_accept"].get(k,0)), "target_quota": int(target.get("bins",{}).get(k,{}).get("target_quota",0)), "target_status": target.get("bins",{}).get(k,{}).get("status")} for k in target.get("bins",{})}})
    write_json(out_dir / "sparse_bin_process_trace_summary.json", {"trace_bins": len(trace_counters), "trace_examples_saved": int(sum(trace_counters.values())), "trace_output_dir": str(out_dir / "sparse_bin_process_traces"), "trace_counts_by_bin": dict(trace_counters)})
    # Compatibility summaries requested by MVP.
    write_json(out_dir / "full_evaluated_samples_summary.json", {"path": "full_evaluated_samples.jsonl", "count": reports["full_evaluated"]})
    write_json(out_dir / "rejection_summary.json", dict(reports["reject_reasons"]))
    write_json(out_dir / "confirmed_archive_counts.json", dict(archive.counts))
    # Human summary.
    print(TITLE)
    print("\n=== Protocol Compliance ===")
    print(f"does_not_modify_3_0_4_source      True")
    print(f"source_reference                  {source_reference}")
    print(f"mvp_script_path                   {Path(__file__).resolve()}")
    print("\n=== Target Distribution ===")
    print(f"target_total                      {target.get('target_total')}")
    print(f"quota_sum                         {target.get('quota_sum')}")
    print(f"handdraw_observed_bins            {sum(1 for b in target.get('bins',{}).values() if b.get('status')=='handdraw_observed')}")
    print(f"nonzero_target_bins               {sum(1 for b in target.get('bins',{}).values() if int(b.get('target_quota',0))>0)}")
    print("\n=== Candidate Funnel ===")
    print(f"n_candidate_total                 {reports['candidate_total']}")
    print(f"n_hard_quality_passed             {reports['hard_passed']}")
    print(f"n_full_evaluated                  {reports['full_evaluated']}")
    print(f"n_full_accepted                   {reports['accepted']}")
    print("\n=== MVP Policy ===")
    print(f"mvp_policy                        {args.mvp_policy}")
    print(f"enable_lbs                        {args.enable_lbs and args.mvp_policy=='lbs_patch'}")
    print(f"enable_tai                        {args.enable_tai and args.mvp_policy=='lbs_patch'}")
    print(f"enable_sbr                        {args.enable_sbr and args.mvp_policy=='lbs_patch'}")
    print("\n=== Long Backbone Report ===")
    print(f"LBS_attempts                      {reports['lbs_attempts']}")
    print(f"LBS_full_evaluated                {reports['lbs_full']}")
    print(f"LBS_long_bfs_count                {reports['lbs_long']}")
    print(f"LBS_accepted_count                {reports['lbs_accepted']}")
    print(f"LBS_avg_bfs_len                   {statistics.mean(lbs_bfs) if lbs_bfs else None}")
    print("\n=== Secondary Buffer Report ===")
    print(f"buffer_admit_count                {sum(secondary.admit_reasons.values())}")
    print(f"buffer_replay_count               {secondary.replay_count}")
    print(f"buffer_accept_count               {secondary.accept_count}")
    print(f"buffer_sparse_hit_count           {secondary.sparse_hit_count}")
    print(f"buffer_eviction_count             {sum(secondary.evict_reasons.values())}")
    print(f"top_admit_reasons                 {summarize_counter(secondary.admit_reasons,5)}")
    print(f"top_evict_reasons                 {summarize_counter(secondary.evict_reasons,5)}")
    print("\n=== Diameter Start/Goal Report ===")
    print(f"diameter_pair_attempts            {reports.get('diameter_pair_attempts', 0)}")
    print(f"diameter_pair_success             {reports.get('diameter_pair_success', 0)}")
    print(f"diameter_pair_fallback_count      {reports.get('diameter_pair_fallback_count', 0)}")
    before_sfg = reports.get('sfg_seed_bfs_before_diameter', [])
    after_sfg = reports.get('sfg_seed_bfs_after_diameter', [])
    print(f"sfg_long_bfs_rate_before          {sum(1 for x in before_sfg if x >= 13) / max(1, len(before_sfg)):.3f}")
    print(f"sfg_long_bfs_rate_after           {sum(1 for x in after_sfg if x >= 13) / max(1, len(after_sfg)):.3f}")
    if target_class_mode:
        print("\n=== Target-Class Route Report ===")
        routes_obj = route_report_final.get('routes', {})
        for rid in list(TARGET_CLASS_ROUTE_PRIMARY.keys()) + ['generic_tree_long', 'generic_single_loop_long', 'generic_multi_loop_long']:
            rr = routes_obj.get(rid)
            if rr:
                print(f"{rid:44s} primary={rr.get('primary_bin')} priority={rr.get('priority_score',0):.5f} selected={reports['route_full'].get(rid,0)} full_eval={rr.get('full_evaluated_count',0)} accepted={rr.get('accepted_count',0)} fill={rr.get('fill_ratio',0):.3f}")
    print("\n=== Pure TBI Report ===")
    print(f"pure_tbi_attempts                 {reports['tbi_attempts']}")
    print(f"pure_tbi_successes                {reports.get('pure_tbi_successes', 0)}")
    print(f"pure_tbi_commit_count             {reports['tbi_commit']}")
    print(f"pure_tbi_rollback_count           {reports['tbi_rollback']}")
    print(f"pure_tbi_avg_delta_endpoint       {avg_delta_map(reports['tbi_deltas']).get('endpoint',0):.3f}")
    print(f"pure_tbi_avg_delta_choice         {avg_delta_map(reports['tbi_deltas']).get('choice',0):.3f}")
    print(f"pure_tbi_avg_delta_cycle          {avg_delta_map(reports['tbi_deltas']).get('cycle',0):.3f}")
    print(f"pure_tbi_cycle_violation_count    {reports.get('pure_tbi_cycle_violation_count', 0)}")
    print("\n=== Stage CLO Report ===")
    print(f"stage_clo_attempts                {reports['clo_attempts']}")
    print(f"stage_clo_successes               {reports.get('stage_clo_successes', 0)}")
    print(f"stage_clo_commit_count            {reports['clo_commit']}")
    print(f"stage_clo_rollback_count          {reports['clo_rollback']}")
    print(f"stage_clo_avg_delta_cycle         {avg_delta_map(reports['clo_deltas']).get('cycle',0):.3f}")
    print(f"stage_clo_avg_delta_bfs           {avg_delta_map(reports['clo_deltas']).get('bfs',0):.3f}")
    print(f"stage_clo_avg_delta_endpoint      {avg_delta_map(reports['clo_deltas']).get('endpoint',0):.3f}")
    print(f"stage_clo_no_cycle_gain_count     {reports.get('stage_clo_no_cycle_gain_count', 0)}")
    print(f"stage_clo_bfs_drop_rollback_count {reports.get('stage_clo_bfs_drop_rollback_count', 0)}")
    print(f"stage_clo_endpoint_drop_rollback_count {reports.get('stage_clo_endpoint_drop_rollback_count', 0)}")
    print("\n=== RBR Priority Report ===")
    print(f"critical_route_replay_count       {reports.get('critical_route_replay_count', 0)}")
    print(f"noncritical_route_replay_count    {reports.get('noncritical_route_replay_count', 0)}")
    print(f"top_priority_route_ids            {summarize_counter(Counter([it.target_route for it in rbr.items]), 5)}")
    print("\n=== Critical Long Bins ===")
    for key in CRITICAL_LONG_BINS:
        row = critical[key]
        print(f"{key:45s} quota={row['target_quota']:4d} count={row['confirmed_count']:4d} full_eval={row['full_evaluated_count']:4d} fill={row['fill_ratio']:.3f} from_lbs={row['from_lbs_count']:3d} from_buffer={row['from_secondary_buffer_count']:3d}")
    print("\n=== Target Archive Coverage ===")
    print(f"confirmed_total                   {synthetic['confirmed_total']}")
    print(f"target_fill_ratio                 {synthetic['target_fill_ratio']:.4f}")
    print(f"coverage_nonzero_target_bins      {synthetic.get('coverage_nonzero_target_bins')}")
    print(f"observed_handdraw_bin_fill_rate   {synthetic.get('observed_handdraw_bin_fill_rate')}")
    print(f"l1_distance_to_target_distribution {synthetic['l1_distance_to_target_distribution']:.4f}")
    print("\n=== Generator Summary ===")
    for g in sorted(set(list(reports["generator_selected"].keys()) + list(reports["generator_full"].keys()))):
        full = reports["generator_full"].get(g,0); acc = reports["generator_accepted"].get(g,0)
        print(f"{g:32s} selected={reports['generator_selected'].get(g,0):5d} full_eval={full:5d} accepted={acc:5d} long_bfs={reports['generator_long'].get(g,0):5d} accept_rate={acc/(full or 1):.3f}")
    print("\n=== Transform Summary ===")
    for op in sorted(reports["operator_attempts"].keys()):
        print(f"{op:20s} attempts={reports['operator_attempts'][op]:5d} successes={reports['operator_successes'].get(op,0):5d} accepted_after={reports['operator_accepted_after'].get(op,0):5d} avg_delta_bfs={avg_delta.get(op+':bfs_len',0):.3f} avg_delta_endpoint={avg_delta.get(op+':endpoint_count',0):.3f} avg_delta_choice={avg_delta.get(op+':choice_count',0):.3f} avg_delta_cycle={avg_delta.get(op+':cycle_rank',0):.3f}")
    print("\n=== Debug Hints ===")
    if reports["lbs_attempts"] and reports["lbs_long"] / max(1,reports["lbs_full"]) > 0.5:
        print("[HINT] LBS produced a high long-BFS share; compare critical_long_bins_report.json against fixed_baseline.")
    if secondary.replay_count and secondary.sparse_hit_count == 0:
        print("[HINT] SBR replayed samples but did not hit sparse bins; tighten admission or improve injection transforms.")
    if any(critical[k]["full_evaluated_count"] == 0 for k in CRITICAL_LONG_BINS):
        print("[HINT] Some critical long bins still have zero full_evaluated_count; LBS solves length only partly or injection is insufficient.")
    dominant = reports["generator_accepted"].most_common(1)
    if dominant and dominant[0][1] / max(1, reports["accepted"]) > 0.7:
        print("[HINT] Accepted samples are dominated by one generator; inspect generator_weight_report.json before formal integration.")
    print(f"Output directory: {out_dir}")
    return out_dir



# ---------------------------------------------------------------------------
# 3.0.4(13) target-class route + true skeleton injection helpers
# ---------------------------------------------------------------------------

TARGET_CLASS_ROUTE_PRIMARY = {
    "tree_long_low_choice_mid_endpoint": "tree|long|low_choice|mid_endpoint",
    "tree_long_mid_choice_high_endpoint": "tree|long|mid_choice|high_endpoint",
    "tree_long_high_choice_high_endpoint": "tree|long|high_choice|high_endpoint",
    "single_loop_long_high_choice_high_endpoint": "single_loop|long|high_choice|high_endpoint",
    "multi_loop_long_high_choice_high_endpoint": "multi_loop|long|high_choice|high_endpoint",
}
CRITICAL_ROUTE_IDS = set(TARGET_CLASS_ROUTE_PRIMARY.keys())


def route_id_for_target_bin(bin_key: str) -> str:
    if bin_key in CRITICAL_LONG_BINS:
        for rid, pk in TARGET_CLASS_ROUTE_PRIMARY.items():
            if pk == bin_key:
                return rid
    cyc, bfs, _choice, _endpoint = bin_parts(bin_key)
    if bfs == "long":
        if cyc == "single_loop":
            return "generic_single_loop_long"
        if cyc == "multi_loop":
            return "generic_multi_loop_long"
        return "generic_tree_long"
    return f"generic_{cyc}_{bfs}" if cyc and bfs else "generic_tree_long"


def route_cycle_target(route_id: str) -> str:
    if route_id.startswith("single_loop") or "|single_loop|" in route_id:
        return "single_loop"
    if route_id.startswith("multi_loop"):
        return "multi_loop"
    return "tree"


def range_for_choice_bin(name: str) -> Tuple[int, int]:
    return {"low_choice": (0, 1), "mid_choice": (2, 4), "high_choice": (5, 999)}.get(name, (0, 999))


def range_for_endpoint_bin(name: str) -> Tuple[int, int]:
    return {"low_endpoint": (0, 1), "mid_endpoint": (2, 4), "high_endpoint": (5, 999)}.get(name, (0, 999))


def range_for_bfs_bin(name: str) -> Tuple[int, int]:
    return {"short": (0, 6), "medium": (7, 12), "long": (13, 999)}.get(name, (0, 999))


def metric_range_distance(x: int, lo: int, hi: int) -> int:
    if x < lo:
        return lo - x
    if x > hi:
        return x - hi
    return 0


def build_target_class_routes(target: Dict[str, Any], archive_counts: Counter, full_eval_counts: Counter, accepted_counts: Counter, args: argparse.Namespace) -> Dict[str, Any]:
    route_bins: Dict[str, List[str]] = defaultdict(list)
    for key, b in target.get("bins", {}).items():
        q = int(b.get("target_quota", 0))
        if q <= 0 or b.get("status") == "structural_infeasible_or_constrained":
            continue
        route_bins[route_id_for_target_bin(key)].append(key)
    routes = {}
    scores = {}
    for rid, keys in route_bins.items():
        quota = sum(int(target["bins"][k].get("target_quota", 0)) for k in keys)
        count = sum(int(archive_counts.get(k, 0)) for k in keys)
        full = sum(int(full_eval_counts.get(k, 0)) for k in keys)
        acc = sum(int(accepted_counts.get(k, 0)) for k in keys)
        remaining = max(0, quota - count)
        if quota <= 0:
            continue
        target_prob_sum = sum(float(target["bins"][k].get("target_prob", 0.0)) for k in keys)
        quota_gap_ratio = remaining / max(1, quota)
        critical_boost = float(args.critical_route_priority) if rid in CRITICAL_ROUTE_IDS else (1.0 if any(target["bins"][k].get("status") == "handdraw_observed" for k in keys) else 0.2)
        scarcity_boost = 2.0 if full == 0 else (1.5 if acc == 0 else 1.0)
        priority = max(0.0, target_prob_sum * quota_gap_ratio * critical_boost * scarcity_boost)
        primary = TARGET_CLASS_ROUTE_PRIMARY.get(rid, max(keys, key=lambda k: int(target["bins"][k].get("target_quota", 0))))
        cyc, bfs, choice, endpoint = bin_parts(primary)
        routes[rid] = {
            "route_id": rid,
            "target_bins": keys,
            "primary_bin": primary,
            "cycle_target": cyc,
            "bfs_target": bfs,
            "choice_target": choice,
            "endpoint_target": endpoint,
            "quota_gap": remaining,
            "fill_ratio": count / max(1, quota),
            "full_evaluated_count": full,
            "accepted_count": acc,
            "scarcity_score": scarcity_boost,
            "priority_score": priority,
        }
        scores[rid] = priority
    if not scores:
        scores = {"generic_tree_long": 1.0}
    # Ensure critical routes remain visible even before target bins have samples.
    for rid, pk in TARGET_CLASS_ROUTE_PRIMARY.items():
        if rid not in routes and pk in target.get("bins", {}):
            b = target["bins"][pk]
            q = int(b.get("target_quota", 0))
            n = int(archive_counts.get(pk, 0))
            routes[rid] = {"route_id": rid, "target_bins": [pk], "primary_bin": pk, "cycle_target": bin_parts(pk)[0], "bfs_target": bin_parts(pk)[1], "choice_target": bin_parts(pk)[2], "endpoint_target": bin_parts(pk)[3], "quota_gap": max(0, q-n), "fill_ratio": n/max(1,q), "full_evaluated_count": int(full_eval_counts.get(pk,0)), "accepted_count": int(accepted_counts.get(pk,0)), "scarcity_score": 2.0, "priority_score": 0.0}
    total = sum(scores.values()) or 1.0
    probs = {rid: v / total for rid, v in scores.items()}
    return {"routes": routes, "route_sampling_probabilities": probs}


def choose_target_class_route(rng: random.Random, route_report: Dict[str, Any]) -> str:
    return choose_weighted(rng, route_report.get("route_sampling_probabilities", {"generic_tree_long": 1.0}))


def target_distance_to_bin(metrics: Dict[str, Any], target_bin: str) -> int:
    return metric_distance_to_bin(metrics, target_bin)


def closest_target_class_distance(metrics: Dict[str, Any], route_id: str, route_report: Dict[str, Any]) -> int:
    routes = route_report.get("routes", {})
    targets = routes.get(route_id, {}).get("target_bins", [])
    if not targets:
        primary = routes.get(route_id, {}).get("primary_bin")
        targets = [primary] if primary else []
    if not targets:
        return 999
    return min(target_distance_to_bin(metrics, b) for b in targets)


def free_cells(grid: Grid) -> List[Cell]:
    return [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == 0]


def assign_start_goal_by_diameter_like_pair(grid: Grid, rng: random.Random, fallback_assigner: Any, archive: Any, sample_count: int = 64) -> Tuple[Cell, Cell, Dict[str, Any]]:
    cells = free_cells(grid)
    dbg = {"diameter_pair_attempts": 0, "diameter_pair_success": False, "diameter_pair_bfs_len": None, "diameter_pair_fallback_count": 0}
    if len(cells) < 2:
        s, g, info = fallback_assigner.assign(grid, archive)
        dbg["diameter_pair_fallback_count"] = 1
        dbg["fallback_reason"] = "too_few_free_cells"
        return s, g, dbg
    sampled = cells[:] if len(cells) <= sample_count else rng.sample(cells, sample_count)
    rng.shuffle(sampled)
    best_pairs: List[Tuple[int, Cell, Cell]] = []
    for a in sampled:
        dbg["diameter_pair_attempts"] += 1
        d1 = bfs_dist(grid, a)
        if not d1:
            continue
        maxd = max(d1.values())
        far = [x for x, d in d1.items() if d == maxd]
        rng.shuffle(far)
        for f in far[:3]:
            d2 = bfs_dist(grid, f)
            if not d2:
                continue
            md = max(d2.values())
            far2 = [x for x, d in d2.items() if d == md]
            rng.shuffle(far2)
            best_pairs.append((md, f, far2[0]))
    if best_pairs:
        best_pairs.sort(key=lambda x: x[0], reverse=True)
        topd = best_pairs[0][0]
        top = [p for p in best_pairs if p[0] == topd]
        dist, s, g = rng.choice(top)
        dbg["diameter_pair_success"] = True
        dbg["diameter_pair_bfs_len"] = int(dist)
        return s, g, dbg
    s, g, info = fallback_assigner.assign(grid, archive)
    dbg["diameter_pair_fallback_count"] = 1
    dbg["fallback_reason"] = "no_connected_pair_found"
    return s, g, dbg


def find_shortest_path_cells(grid: Grid, start: Cell, goal: Cell) -> List[Cell]:
    n = len(grid)
    if grid[start[0]][start[1]] != 0 or grid[goal[0]][goal[1]] != 0:
        return []
    q = deque([start])
    prev: Dict[Cell, Optional[Cell]] = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for nb in neighbors(cur, n):
            if grid[nb[0]][nb[1]] == 0 and nb not in prev:
                prev[nb] = cur
                q.append(nb)
    if goal not in prev:
        return []
    path = []
    cur: Optional[Cell] = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def free_degree(grid: Grid, cell: Cell) -> int:
    n = len(grid)
    return sum(1 for nb in neighbors(cell, n) if grid[nb[0]][nb[1]] == 0)


def clone_candidate(cand: Any, grid: Grid, candidate_id: str, operator: str, detail: Dict[str, Any]) -> Any:
    hist = list(getattr(cand, "transform_history", []))
    hist.append({"operator": operator, "detail": detail})
    return F.MazeCandidate(grid=grid, start=cand.start, goal=cand.goal, origin_generator=cand.origin_generator, origin_seed_id=cand.origin_seed_id, transform_history=hist, candidate_id=candidate_id)


def apply_pure_tree_branch_injection(cand: Any, rng: random.Random, route_id: str, args: argparse.Namespace) -> Tuple[Optional[Any], Dict[str, Any]]:
    grid = copy.deepcopy(cand.grid)
    n = len(grid)
    spath = find_shortest_path_cells(grid, cand.start, cand.goal)
    path_set = set(spath)
    candidates = [x for x in spath if x not in (cand.start, cand.goal) and free_degree(grid, x) in (2, 3)]
    others = [x for x in free_cells(grid) if x not in path_set and x not in (cand.start, cand.goal) and free_degree(grid, x) in (2, 3)]
    rng.shuffle(candidates); rng.shuffle(others)
    roots = (candidates + others)[:max(1, int(args.pure_tbi_max_roots_per_item))]
    violation_count = 0
    for root in roots:
        dirs = ACTIONS[:]
        rng.shuffle(dirs)
        for dr, dc in dirs:
            trial = copy.deepcopy(grid)
            cur = root
            branch: List[Cell] = []
            target_len = rng.randint(max(1, int(args.pure_tbi_min_branch_len)), max(1, int(args.pure_tbi_max_branch_len)))
            direction = (dr, dc)
            for step in range(target_len):
                opts = []
                # Mostly continue outward but allow one bend.
                dirs2 = [direction] + [d for d in ACTIONS if d != direction]
                for ddr, ddc in dirs2:
                    nb = (cur[0] + ddr, cur[1] + ddc)
                    if not in_bounds(nb[0], nb[1], n) or trial[nb[0]][nb[1]] != 1:
                        continue
                    free_nbs = [x for x in neighbors(nb, n) if trial[x[0]][x[1]] == 0]
                    if len(free_nbs) != 1 or free_nbs[0] != cur:
                        violation_count += 1
                        continue
                    opts.append((nb, (ddr, ddc)))
                if not opts:
                    break
                nb, direction = opts[0] if rng.random() < 0.75 else rng.choice(opts)
                trial[nb[0]][nb[1]] = 0
                branch.append(nb)
                cur = nb
            if len(branch) >= int(args.pure_tbi_min_branch_len):
                detail = {"root": list(root), "branch_cells": [list(x) for x in branch], "branch_len": len(branch), "free_neighbor_violation_count": violation_count, "cycle_violation_checked": True}
                return clone_candidate(cand, trial, cand.candidate_id + "_pure_tbi", "pure_tbi", detail), detail
    return None, {"failed_reason": "no_valid_tree_branch", "free_neighbor_violation_count": violation_count}


def manhattan_path(a: Cell, b: Cell, order: int) -> List[Cell]:
    r, c = a
    out = [a]
    if order == 0:
        while c != b[1]:
            c += 1 if b[1] > c else -1; out.append((r, c))
        while r != b[0]:
            r += 1 if b[0] > r else -1; out.append((r, c))
    else:
        while r != b[0]:
            r += 1 if b[0] > r else -1; out.append((r, c))
        while c != b[1]:
            c += 1 if b[1] > c else -1; out.append((r, c))
    return out


def apply_stage_controlled_loop_overlay(cand: Any, rng: random.Random, route_id: str, args: argparse.Namespace) -> Tuple[Optional[Any], Dict[str, Any]]:
    grid = copy.deepcopy(cand.grid)
    n = len(grid)
    cells = free_cells(grid)
    rng.shuffle(cells)
    endpoints = cells[:min(len(cells), 48)]
    no_path = 0
    for i, a in enumerate(endpoints):
        for b in endpoints[i+1:i+16]:
            if abs(a[0]-b[0]) + abs(a[1]-b[1]) < 3:
                continue
            for order in [0, 1]:
                path = manhattan_path(a, b, order)
                internal = path[1:-1]
                if len(internal) == 0 or len(internal) > int(args.clo_max_added_path_len):
                    continue
                wall_count = sum(1 for x in internal if grid[x[0]][x[1]] == 1)
                if wall_count != len(internal):
                    no_path += 1
                    continue
                trial = copy.deepcopy(grid)
                for r, c in internal:
                    trial[r][c] = 0
                detail = {"endpoint_a": list(a), "endpoint_b": list(b), "added_path": [list(x) for x in internal], "added_path_len": len(internal)}
                return clone_candidate(cand, trial, cand.candidate_id + "_stage_clo", "stage_clo", detail), detail
    return None, {"failed_reason": "no_short_wall_corridor", "rejected_existing_free_internal": no_path}


def route_target_ranges(route_id: str, route_report: Dict[str, Any]) -> Dict[str, Any]:
    routes = route_report.get("routes", {})
    primary = routes.get(route_id, {}).get("primary_bin") or TARGET_CLASS_ROUTE_PRIMARY.get(route_id) or "tree|long|low_choice|mid_endpoint"
    cyc, bfs, choice, endpoint = bin_parts(primary)
    return {"cycle": cyc, "bfs": bfs, "choice": choice, "endpoint": endpoint, "primary_bin": primary, "choice_range": range_for_choice_bin(choice), "endpoint_range": range_for_endpoint_bin(endpoint), "bfs_range": range_for_bfs_bin(bfs)}


def choose_operator_by_stage(item: Any, route_id: str, stable_metrics: Dict[str, Any], route_report: Dict[str, Any], rng: random.Random, args: argparse.Namespace) -> Tuple[str, Dict[str, Any]]:
    tgt = route_target_ranges(route_id, route_report)
    bfs = int(stable_metrics.get("bfs_len") or 0)
    endpoint = int(stable_metrics.get("rect_room_macro_endpoint_count") or 0)
    choice = int(stable_metrics.get("rect_room_macro_choice_count") or 0)
    cycle = int(stable_metrics.get("rect_room_compressed_cycle_rank") or 0)
    ep_lo, _ = tgt["endpoint_range"]
    ch_lo, _ = tgt["choice_range"]
    cycle_t = tgt["cycle"]
    if bfs < 13:
        stage = "make_long"
        weights = {"spb": 1.0, "pure_tbi": 0.25, "stage_clo": 0.0, "random": 0.03}
    elif endpoint < ep_lo or choice < ch_lo:
        stage = "build_branch_rich_skeleton"
        weights = {"pure_tbi": 1.2, "spb": 0.15, "stage_clo": 0.02 if cycle_t != "tree" else 0.0, "random": 0.03}
    elif cycle_t in ("single_loop", "multi_loop") and ((cycle_t == "single_loop" and cycle < 1) or (cycle_t == "multi_loop" and cycle < 2)):
        stage = "overlay_loop_if_needed"
        weights = {"stage_clo": 1.2, "pure_tbi": 0.2, "spb": 0.1, "random": 0.02}
    else:
        stage = "refine"
        weights = {"pure_tbi": 0.45, "spb": 0.25, "stage_clo": 0.25 if cycle_t != "tree" else 0.0, "random": 0.03}
    if not args.enable_pure_tbi:
        weights["pure_tbi"] = 0.0
    if not args.enable_stage_clo or cycle_t == "tree":
        weights["stage_clo"] = 0.0
    if not args.enable_spb:
        weights["spb"] = 0.0
    op = choose_weighted(rng, weights)
    return op, {"stage": stage, "stage_reason": f"bfs={bfs} endpoint={endpoint}/{ep_lo} choice={choice}/{ch_lo} cycle={cycle}->{cycle_t}", "operator_weights": weights}


def target_class_route_constraints(route_id: str, before: Dict[str, Any], after: Dict[str, Any], route_report: Dict[str, Any], args: argparse.Namespace) -> Tuple[bool, str]:
    tgt = route_target_ranges(route_id, route_report)
    before_bfs = int(before.get("bfs_len") or 0)
    after_bfs = int(after.get("bfs_len") or 0)
    before_ep = int(before.get("rect_room_macro_endpoint_count") or 0)
    after_ep = int(after.get("rect_room_macro_endpoint_count") or 0)
    before_ch = int(before.get("rect_room_macro_choice_count") or 0)
    after_ch = int(after.get("rect_room_macro_choice_count") or 0)
    before_cyc = int(before.get("rect_room_compressed_cycle_rank") or 0)
    after_cyc = int(after.get("rect_room_compressed_cycle_rank") or 0)
    cycle_t = tgt["cycle"]
    if cycle_t == "tree" and after_cyc > 0:
        return False, "route_cycle_violation_tree"
    if cycle_t == "single_loop" and after_cyc > 1:
        return False, "route_over_cycle_single_loop"
    if before_bfs >= 13 and after_bfs < 13:
        return False, "route_bfs_dropped_out_of_long"
    if cycle_t in ("single_loop", "multi_loop") and after_bfs < before_bfs - int(args.spb_max_bfs_drop):
        return False, "route_bfs_drop_after_loop_or_spb"
    if after_ep < before_ep:
        return False, "route_endpoint_regression"
    if after_ch < before_ch - 1:
        return False, "route_choice_regression"
    if cycle_t == "multi_loop" and after_cyc > 8 and after_ch < 1:
        return False, "route_style_collapse"
    return True, "route_ok"


def target_class_progress_events(route_id: str, before: Dict[str, Any], before_bin: str, after: Dict[str, Any], after_bin: str, route_report: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    tgt = route_target_ranges(route_id, route_report)
    before_d = closest_target_class_distance(before, route_id, route_report)
    after_d = closest_target_class_distance(after, route_id, route_report)
    before_bfs = int(before.get("bfs_len") or 0); after_bfs = int(after.get("bfs_len") or 0)
    before_ep = int(before.get("rect_room_macro_endpoint_count") or 0); after_ep = int(after.get("rect_room_macro_endpoint_count") or 0)
    before_ch = int(before.get("rect_room_macro_choice_count") or 0); after_ch = int(after.get("rect_room_macro_choice_count") or 0)
    before_c = int(before.get("rect_room_compressed_cycle_rank") or 0); after_c = int(after.get("rect_room_compressed_cycle_rank") or 0)
    ep_lo, _ = tgt["endpoint_range"]; ch_lo, _ = tgt["choice_range"]
    if before_bfs < 13 <= after_bfs or after_bfs - before_bfs >= 2:
        reasons.append("BFSProgress")
    if after_ep > before_ep and after_ep <= max(ep_lo + 3, after_ep):
        reasons.append("EndpointProgress")
    if after_ch > before_ch and after_ch <= max(ch_lo + 3, after_ch):
        reasons.append("ChoiceProgress")
    if tgt["cycle"] == "tree" and after_c == 0 and (after_ep > before_ep or after_ch > before_ch or after_bfs > before_bfs):
        reasons.append("CycleProgress_tree_preserved")
    if tgt["cycle"] == "single_loop" and before_c == 0 and after_c == 1:
        reasons.append("CycleProgress_single_loop")
    if tgt["cycle"] == "multi_loop" and after_c > before_c:
        reasons.append("CycleProgress_multi_loop")
    if after_d < before_d:
        reasons.append("TargetDistanceProgress")
    return bool(reasons), reasons


def compute_rbr_priority(item: Any, route_report: Dict[str, Any]) -> float:
    routes = route_report.get("routes", {})
    rr = routes.get(item.target_route, {})
    route_priority = float(rr.get("priority_score", 0.0)) + 0.01
    dist = closest_target_class_distance(item.stable_full_metrics, item.target_route, route_report)
    critical_boost = 2.0 if item.target_route in CRITICAL_ROUTE_IDS else 1.0
    recent_progress = 1.0 + 0.2 * min(5, item.lifetime_commit_count) + 0.5 * min(3, item.lifetime_accept_count)
    rollback_streak = item.phase_rollback_count
    resource_penalty = 1.0 / (1.0 + 0.02 * item.lifetime_attempt_count + 0.1 * rollback_streak)
    return max(0.0001, route_priority * (1.0 / (1.0 + dist)) * critical_boost * recent_progress * resource_penalty)
# ---------------------------------------------------------------------------
# 3.0.4(12) SFG + TCI + RBR implementation
# ---------------------------------------------------------------------------

ROUTES = ["tree_long", "single_loop_long", "multi_loop_long"]
ROUTE_TO_CYCLE = {
    "tree_long": "tree",
    "single_loop_long": "single_loop",
    "multi_loop_long": "multi_loop",
}


def cycle_rank_to_bin(v: int) -> str:
    if v <= 0:
        return "tree"
    if v == 1:
        return "single_loop"
    return "multi_loop"


def metric_distance_to_bin(metrics: Dict[str, Any], bin_key: str) -> int:
    cbin, bbin, chbin, ebin = bin_parts(bin_key)
    cycle_rank = int(metrics.get("rect_room_compressed_cycle_rank") or 0)
    bfs = int(metrics.get("bfs_len") or 0)
    choice = int(metrics.get("rect_room_macro_choice_count") or 0)
    endpoint = int(metrics.get("rect_room_macro_endpoint_count") or 0)
    cycle_order = {"tree": 0, "single_loop": 1, "multi_loop": 2}
    d = abs(cycle_order.get(cycle_rank_to_bin(cycle_rank), 0) - cycle_order.get(cbin, 0))
    bfs_ranges = {"short": (0, 6), "medium": (7, 12), "long": (13, 999)}
    choice_ranges = {"low_choice": (0, 1), "mid_choice": (2, 4), "high_choice": (5, 999)}
    endpoint_ranges = {"low_endpoint": (0, 1), "mid_endpoint": (2, 4), "high_endpoint": (5, 999)}
    def range_dist(x: int, lo: int, hi: int) -> int:
        if x < lo: return lo - x
        if x > hi: return x - hi
        return 0
    d += min(4, range_dist(bfs, *bfs_ranges.get(bbin, (0, 999))))
    d += min(4, range_dist(choice, *choice_ranges.get(chbin, (0, 999))))
    d += min(4, range_dist(endpoint, *endpoint_ranges.get(ebin, (0, 999))))
    return int(d)


def route_for_bin(bin_key: str) -> str:
    cycle, bfs, _choice, _endpoint = bin_parts(bin_key)
    if cycle == "single_loop":
        return "single_loop_long" if bfs == "long" else "single_loop"
    if cycle == "multi_loop":
        return "multi_loop_long" if bfs == "long" else "multi_loop"
    return "tree_long" if bfs == "long" else "tree"


def compute_route_demands(target: Dict[str, Any], archive_counts: Counter, full_eval_counts: Optional[Counter] = None, accepted_counts: Optional[Counter] = None) -> Dict[str, Any]:
    full_eval_counts = full_eval_counts or Counter()
    accepted_counts = accepted_counts or Counter()
    demand = Counter()
    long_high_endpoint = 0.0
    for key, b in target.get("bins", {}).items():
        q = int(b.get("target_quota", 0))
        if q <= 0 or b.get("status") != "handdraw_observed":
            continue
        n = int(archive_counts.get(key, 0))
        gap = max(0, q - n)
        if gap <= 0:
            continue
        cycle, bfs, choice, endpoint = bin_parts(key)
        r = route_for_bin(key)
        # Scarcity boost: a target with no full-evaluated samples should demand more skeleton supply.
        scarcity = 1.0
        if full_eval_counts.get(key, 0) == 0:
            scarcity += 0.6
        elif accepted_counts.get(key, 0) == 0:
            scarcity += 0.25
        val = float(b.get("target_prob", 0.0)) * gap * scarcity
        demand[r] += val
        if bfs == "long" and endpoint == "high_endpoint":
            long_high_endpoint += val
    total = sum(demand.values()) or 1.0
    probs = {r: (demand.get(r, 0.0) / total) for r in ROUTES}
    # ensure exploration of all routes
    smoothed = {r: 0.05 + 0.85 * probs.get(r, 0.0) for r in ROUTES}
    z = sum(smoothed.values()) or 1.0
    smoothed = {r: smoothed[r] / z for r in ROUTES}
    return {
        "tree_long_demand": float(demand.get("tree_long", 0.0)),
        "single_loop_long_demand": float(demand.get("single_loop_long", 0.0)),
        "multi_loop_long_demand": float(demand.get("multi_loop_long", 0.0)),
        "long_high_endpoint_demand": float(long_high_endpoint),
        "route_sampling_probabilities": smoothed,
    }


def choose_route(rng: random.Random, route_report: Dict[str, Any]) -> str:
    return choose_weighted(rng, route_report.get("route_sampling_probabilities", {"tree_long": 1.0}))


class SkeletonFirstGenerator(LongBackboneGenerator):
    """SFG: random, long-biased skeleton generator.

    This intentionally controls only long backbone potential, not exact endpoint,
    choice, or cycle targets. It extends the 3.0.4(11) LBS walk with a lower side
    pocket rate so that TBI/CLO/SPB can perform visible injections later.
    """
    def make_grid(self) -> Grid:
        grid = super().make_grid()
        # Lightly close some optional side pockets to preserve a clearer skeleton.
        n = self.n
        for r in range(n):
            for c in range(n):
                if grid[r][c] == 0 and self.rng.random() < 0.015:
                    if sum(1 for nb in neighbors((r, c), n) if grid[nb[0]][nb[1]] == 0) >= 3:
                        grid[r][c] = 1
        return grid


class RBRItem:
    def __init__(self, item_id: str, cand: Any, metrics: Dict[str, Any], bin_key: str, route: str, origin_generator: str):
        self.item_id = item_id
        self.stable_grid = copy.deepcopy(cand.grid)
        self.stable_start = tuple(cand.start)
        self.stable_goal = tuple(cand.goal)
        self.stable_full_metrics = copy.deepcopy(metrics)
        self.stable_full_bin_key = bin_key
        self.target_route = route
        self.target_bins: List[str] = []
        self.nearest_sparse_targets: List[str] = []
        self.origin_generator = origin_generator
        self.origin_seed_id = cand.origin_seed_id
        self.process_trace = copy.deepcopy(cand.transform_history)
        # phase counters
        self.phase_attempt_count = 0
        self.phase_rollback_count = 0
        self.phase_no_progress_count = 0
        # lifetime counters
        self.lifetime_attempt_count = 0
        self.lifetime_commit_count = 0
        self.lifetime_rollback_count = 0
        self.lifetime_accept_count = 0
        self.lineage_transform_count = len(cand.transform_history)
        self.route_reassignment_count = 0
        self.hard_collapse_count = 0
        self.route_collapse_count = 0
        self.full_bin_full_hits = 0
        self.created_step = 0
        self.last_replay_step = None
        self.last_commit_step = None
        self.recent_accepted = False

    def to_candidate(self, candidate_id: str) -> Any:
        return F.MazeCandidate(
            grid=copy.deepcopy(self.stable_grid),
            start=tuple(self.stable_start),
            goal=tuple(self.stable_goal),
            origin_generator=self.origin_generator,
            origin_seed_id=self.origin_seed_id,
            transform_history=copy.deepcopy(self.process_trace),
            candidate_id=candidate_id,
        )

    def commit(self, trial: Any, metrics: Dict[str, Any], bin_key: str, op_record: Dict[str, Any], accepted: bool = False) -> None:
        self.stable_grid = copy.deepcopy(trial.grid)
        self.stable_start = tuple(trial.start)
        self.stable_goal = tuple(trial.goal)
        self.stable_full_metrics = copy.deepcopy(metrics)
        self.stable_full_bin_key = bin_key
        self.process_trace = copy.deepcopy(trial.transform_history)
        self.process_trace.append(op_record)
        self.phase_no_progress_count = 0
        self.phase_rollback_count = 0
        self.phase_attempt_count = 0
        self.lifetime_commit_count += 1
        if accepted:
            self.lifetime_accept_count += 1
            self.recent_accepted = True

    def rollback(self, reason: str) -> None:
        self.phase_rollback_count += 1
        self.lifetime_rollback_count += 1
        if reason.startswith("hard"):
            self.hard_collapse_count += 1
        if reason.startswith("route"):
            self.route_collapse_count += 1
        if reason == "no_progress":
            self.phase_no_progress_count += 1


class RollbackBuffer:
    def __init__(self, max_size: int):
        self.items: List[RBRItem] = []
        self.max_size = max_size
        self.admit_reasons = Counter()
        self.evict_reasons = Counter()
        self.commit_reasons = Counter()
        self.rollback_reasons = Counter()
        self.replay_count = 0
        self.accept_count = 0
        self.sparse_hit_count = 0
        self.accepted_commit_count = 0
        self.phase_counter_refresh_count = 0
        self.route_reassignment_count = 0
        self.attempt_count_total = 0

    def __len__(self) -> int:
        return len(self.items)

    def admit(self, item: RBRItem, reasons: List[str]) -> None:
        self.items.append(item)
        for r in reasons or ["admitted"]:
            self.admit_reasons[r] += 1
        self.prune_over_capacity()

    def prune_over_capacity(self) -> None:
        while len(self.items) > self.max_size:
            # evict oldest no recent commit item first
            idx = 0
            for i, it in enumerate(self.items):
                if not it.recent_accepted and it.lifetime_commit_count == 0:
                    idx = i
                    break
            self.items.pop(idx)
            self.evict_reasons["buffer_over_capacity"] += 1

    def replay_probability(self, route_report: Dict[str, Any]) -> float:
        if not self.items:
            return 0.0
        route_gap = sum(float(route_report.get(k, 0.0)) for k in ["tree_long_demand", "single_loop_long_demand", "multi_loop_long_demand"])
        commit_rate = sum(it.lifetime_commit_count for it in self.items) / max(1, sum(it.lifetime_attempt_count for it in self.items))
        accept_rate = sum(it.lifetime_accept_count for it in self.items) / max(1, sum(it.lifetime_attempt_count for it in self.items))
        p = 0.10 + 0.25 * min(1.0, route_gap / 100.0) + 0.20 * min(1.0, commit_rate * 4.0) + 0.20 * min(1.0, accept_rate * 8.0)
        # If rollbacks dominate, reduce replay pressure.
        rollbacks = sum(it.lifetime_rollback_count for it in self.items)
        attempts = sum(it.lifetime_attempt_count for it in self.items) or 1
        if rollbacks / attempts > 0.7:
            p *= 0.65
        return max(0.0, min(0.65, p))

    def sample(self, rng: random.Random, route_report: Dict[str, Any]) -> Optional[RBRItem]:
        if not self.items:
            return None
        weights = []
        probs = route_report.get("route_sampling_probabilities", {})
        target_class = bool(route_report.get("routes"))
        for it in self.items:
            if target_class:
                w = compute_rbr_priority(it, route_report)
            else:
                w = 1.0 / (1.0 + it.phase_attempt_count + 0.3 * it.lifetime_attempt_count)
                w *= 1.0 + 2.0 * float(probs.get(it.target_route, 0.0))
                if bin_parts(it.stable_full_bin_key)[1] == "long":
                    w *= 1.5
                if it.recent_accepted:
                    w *= 0.75
            weights.append(max(0.01, w))
        x = rng.random() * sum(weights)
        acc = 0.0
        for it, w in zip(self.items, weights):
            acc += w
            if x <= acc:
                self.replay_count += 1
                it.last_replay_step = self.replay_count
                return it
        self.replay_count += 1
        return self.items[-1]

    def retire_if_needed(self, it: RBRItem, args: argparse.Namespace) -> Optional[str]:
        reason = None
        if it.phase_attempt_count >= args.max_phase_attempts_per_item:
            reason = "max_phase_attempts"
        elif it.lifetime_attempt_count >= args.max_lifetime_attempts_per_item:
            reason = "max_lifetime_attempts"
        elif it.lifetime_accept_count >= args.max_accepts_per_item:
            reason = "max_accepts"
        elif it.phase_no_progress_count >= args.max_no_progress_streak:
            reason = "max_no_progress_streak"
        elif it.hard_collapse_count >= args.max_hard_collapse_count:
            reason = "max_hard_collapse_count"
        elif it.route_collapse_count >= args.max_route_collapse_count:
            reason = "max_route_collapse_count"
        elif it.route_reassignment_count >= args.max_route_reassignments:
            reason = "max_route_reassignments"
        if reason:
            self.evict_reasons[reason] += 1
            if it in self.items:
                self.items.remove(it)
            return reason
        return None


def route_constraints(route: str, before: Dict[str, Any], after: Dict[str, Any]) -> Tuple[bool, str]:
    before_bfs = int(before.get("bfs_len") or 0)
    after_bfs = int(after.get("bfs_len") or 0)
    before_ep = int(before.get("rect_room_macro_endpoint_count") or 0)
    after_ep = int(after.get("rect_room_macro_endpoint_count") or 0)
    before_ch = int(before.get("rect_room_macro_choice_count") or 0)
    after_ch = int(after.get("rect_room_macro_choice_count") or 0)
    after_cycle = int(after.get("rect_room_compressed_cycle_rank") or 0)
    if route == "tree_long":
        if after_cycle > 0:
            return False, "route_cycle_violation_tree"
        if before_bfs >= 13 and after_bfs < 13:
            return False, "route_bfs_dropped_out_of_long"
        if after_ep < before_ep - 1:
            return False, "route_endpoint_regression"
        if after_ch < before_ch - 2:
            return False, "route_choice_regression"
    elif route == "single_loop_long":
        if after_cycle > 1:
            return False, "route_over_cycle_single_loop"
        if before_bfs >= 13 and after_bfs < 13:
            return False, "route_bfs_dropped_out_of_long"
        if after_ep < before_ep - 1:
            return False, "route_endpoint_regression"
    elif route == "multi_loop_long":
        if before_bfs >= 13 and after_bfs < 11:
            return False, "route_bfs_regression_multi"
        if after_ep < before_ep - 2:
            return False, "route_endpoint_regression"
        if after_cycle > 8 and after_ch < 1:
            return False, "route_style_collapse"
    return True, "route_ok"


def closest_route_target_distance(metrics: Dict[str, Any], route: str, target: Dict[str, Any], counts: Counter) -> int:
    best = 10**9
    for key, b in target.get("bins", {}).items():
        q = int(b.get("target_quota", 0))
        if q <= 0 or counts.get(key, 0) >= q:
            continue
        if route_for_bin(key) != route:
            continue
        best = min(best, metric_distance_to_bin(metrics, key))
    return int(best if best < 10**9 else 999)


def progress_events(route: str, before: Dict[str, Any], before_bin: str, after: Dict[str, Any], after_bin: str, target: Dict[str, Any], counts: Counter) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    before_bfs = int(before.get("bfs_len") or 0)
    after_bfs = int(after.get("bfs_len") or 0)
    before_ep = int(before.get("rect_room_macro_endpoint_count") or 0)
    after_ep = int(after.get("rect_room_macro_endpoint_count") or 0)
    before_ch = int(before.get("rect_room_macro_choice_count") or 0)
    after_ch = int(after.get("rect_room_macro_choice_count") or 0)
    before_cycle = int(before.get("rect_room_compressed_cycle_rank") or 0)
    after_cycle = int(after.get("rect_room_compressed_cycle_rank") or 0)
    if bin_parts(before_bin)[1] != "long" and bin_parts(after_bin)[1] == "long":
        reasons.append("BFSProgress_entered_long")
    if after_bfs - before_bfs >= 2:
        reasons.append("BFSProgress_len_increased")
    if before_bfs >= 13 and after_bfs >= 13 and (after_ep > before_ep or after_ch > before_ch):
        reasons.append("BFSProgress_kept_long_with_structure_improvement")
    if after_ep > before_ep:
        reasons.append("EndpointProgress_count_up")
    if bin_parts(after_bin)[3] != bin_parts(before_bin)[3] and bin_parts(after_bin)[3] in ("mid_endpoint", "high_endpoint"):
        reasons.append("EndpointProgress_bin_up")
    if after_ch > before_ch:
        reasons.append("ChoiceProgress_count_up")
    if bin_parts(after_bin)[2] != bin_parts(before_bin)[2] and bin_parts(after_bin)[2] in ("mid_choice", "high_choice"):
        reasons.append("ChoiceProgress_bin_up")
    if route == "tree_long" and after_cycle == 0:
        # This is only progress if another dimension improved.
        if after_bfs > before_bfs or after_ep > before_ep or after_ch > before_ch:
            reasons.append("CycleProgress_tree_preserved")
    if route == "single_loop_long" and before_cycle == 0 and after_cycle == 1:
        reasons.append("CycleProgress_tree_to_single_loop")
    if route == "multi_loop_long" and after_cycle > before_cycle:
        reasons.append("CycleProgress_toward_multi_loop")
    if closest_route_target_distance(after, route, target, counts) < closest_route_target_distance(before, route, target, counts):
        reasons.append("TargetDistanceProgress")
    return bool(reasons), reasons


def choose_route_operator(rng: random.Random, route: str, metrics: Dict[str, Any], route_report: Dict[str, Any], args: argparse.Namespace) -> str:
    # TCI policy. Tree route disables CLO/loop_flip. SPB is weakened when already long.
    bfs = int(metrics.get("bfs_len") or 0)
    endpoint = int(metrics.get("rect_room_macro_endpoint_count") or 0)
    choice = int(metrics.get("rect_room_macro_choice_count") or 0)
    cycle = int(metrics.get("rect_room_compressed_cycle_rank") or 0)
    long_gap = float(route_report.get("long_high_endpoint_demand", 0.0))
    weights = {"branch_grow": 0.1, "shortcut_block": 0.1, "loop_flip": 0.01, "cell_flip_random": 0.02}
    if route == "tree_long":
        weights["branch_grow"] = 1.2 + (0.6 if endpoint < 5 else 0.0) + (0.3 if choice < 5 else 0.0)
        weights["shortcut_block"] = 0.8 if bfs < 13 else 0.15
        weights["loop_flip"] = 0.0
    elif route == "single_loop_long":
        weights["branch_grow"] = 0.9 + (0.5 if endpoint < 5 else 0.0)
        weights["loop_flip"] = 1.0 if cycle < 1 else 0.05
        weights["shortcut_block"] = 0.7 if bfs < 13 else 0.12
    else:  # multi_loop_long
        weights["branch_grow"] = 0.8 + (0.5 if endpoint < 5 else 0.0)
        weights["loop_flip"] = 1.1 if cycle < 2 else 0.35
        weights["shortcut_block"] = 0.65 if bfs < 13 else 0.10
    if not args.enable_tbi:
        weights["branch_grow"] = 0.0
    if not args.enable_clo:
        weights["loop_flip"] = 0.0
    if not args.enable_spb:
        weights["shortcut_block"] = 0.0
    if long_gap > 0 and bfs < 13:
        weights["shortcut_block"] *= 1.3
    return choose_weighted(rng, weights)


def evaluate_full(cand: Any, quantizer: Any, config: Dict[str, Any], archive: Any) -> Tuple[bool, Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str], Dict[str, Any], Optional[str]]:
    cheap = quantizer.evaluate(cand)
    hard = F.hard_quality_filter(cand, config, archive.accepted_hashes)
    if not hard.get("hard_quality_passed"):
        return False, None, None, None, hard, hard.get("hard_reject_reason", "hard_quality_failed")
    full_metrics, _actual, _rt, err = F.full_metrics_for_candidate(cand, cheap, config)
    if err:
        return False, None, None, None, hard, "full_extraction_error"
    full_bin = archive.assign_bin(full_metrics)
    return True, full_metrics, full_bin, full_bin["bin_key"], hard, None


def initialize_reports_12() -> Dict[str, Any]:
    r = initialize_reports()
    r.update({
        "sfg_attempts": 0,
        "sfg_hard": 0,
        "sfg_full": 0,
        "sfg_long": 0,
        "sfg_accepted": 0,
        "sfg_bfs_values": [],
        "sfg_bins": Counter(),
        "route_full": Counter(),
        "route_accepted": Counter(),
        "route_transitions": [],
        "injection_attempts": [],
        "rollback_reasons": Counter(),
        "commit_reasons": Counter(),
        "tbi_attempts": 0,
        "tbi_commit": 0,
        "tbi_rollback": 0,
        "clo_attempts": 0,
        "clo_commit": 0,
        "clo_rollback": 0,
        "spb_attempts": 0,
        "spb_commit": 0,
        "spb_rollback": 0,
        "tbi_deltas": defaultdict(list),
        "clo_deltas": defaultdict(list),
        "spb_deltas": defaultdict(list),
        "accepted_commit_triggered": 0,
        "phase_counter_refresh_count": 0,
        "stable_state_updated_after_accept": 0,
        "diameter_pair_attempts": 0,
        "diameter_pair_success": 0,
        "diameter_pair_fallback_count": 0,
        "sfg_seed_bfs_before_diameter": [],
        "sfg_seed_bfs_after_diameter": [],
        "pure_tbi_successes": 0,
        "pure_tbi_cycle_violation_count": 0,
        "pure_tbi_free_neighbor_violation_count": 0,
        "stage_clo_successes": 0,
        "stage_clo_no_cycle_gain_count": 0,
        "stage_clo_bfs_drop_rollback_count": 0,
        "stage_clo_endpoint_drop_rollback_count": 0,
        "spb_successes": 0,
        "spb_bfs_drop_rollback_count": 0,
        "stage_operator_counts": Counter(),
        "stage_operator_by_stage": defaultdict(Counter),
        "rbr_priority_values": [],
        "critical_route_replay_count": 0,
        "noncritical_route_replay_count": 0,
        "buffer_admit_reasons": Counter(),
        "buffer_reject_reasons": Counter(),
        "from_sfg_by_bin": Counter(),
        "from_rbr_by_bin": Counter(),
        "critical_route_examples": defaultdict(list),
    })
    return r


def record_delta(reports: Dict[str, Any], family: str, before: Dict[str, Any], after: Dict[str, Any]) -> None:
    delta = {
        "bfs": (after.get("bfs_len") or 0) - (before.get("bfs_len") or 0),
        "endpoint": (after.get("rect_room_macro_endpoint_count") or 0) - (before.get("rect_room_macro_endpoint_count") or 0),
        "choice": (after.get("rect_room_macro_choice_count") or 0) - (before.get("rect_room_macro_choice_count") or 0),
        "cycle": (after.get("rect_room_compressed_cycle_rank") or 0) - (before.get("rect_room_compressed_cycle_rank") or 0),
    }
    key = family.lower()
    bucket = reports.get(f"{key}_deltas")
    if bucket is None:
        return
    for k, v in delta.items():
        bucket[k].append(float(v))


def avg_delta_map(d: Dict[str, List[float]]) -> Dict[str, float]:
    return {k: (statistics.mean(v) if v else 0.0) for k, v in d.items()}


def maybe_reassign_route(it: RBRItem, route_report: Dict[str, Any], args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    if it.route_reassignment_count >= args.max_route_reassignments:
        return None
    best = max(route_report.get("route_sampling_probabilities", {}).items(), key=lambda kv: kv[1], default=(it.target_route, 0.0))[0]
    if best != it.target_route and route_report.get("route_sampling_probabilities", {}).get(best, 0.0) > 0.45:
        old = it.target_route
        it.target_route = best
        it.route_reassignment_count += 1
        return {"route_reassigned_from": old, "route_reassigned_to": best, "reason": "route_gap_probability_dominant"}
    return None


def run_generate(args: argparse.Namespace) -> Path:
    # fixed_baseline/lbs_patch use 3.0.4(11) logic; sfg_tci_rbr keeps 3.0.4(12)-style coarse routes.
    # target_class_sfg_rbr activates the 3.0.4(13) target-class route + pure injection path.
    if args.mvp_policy in ("fixed_baseline", "lbs_patch"):
        return run_generate_legacy11(args)
    target_class_mode = args.mvp_policy == "target_class_sfg_rbr"
    rng = random.Random(args.seed)
    config = F.default_config()
    config.setdefault("rect_room_extraction", {})["cheap_metrics_only"] = False
    output_root = Path(args.output_root or DEFAULT_OUTPUT_ROOT_REL)
    run_name = args.run_name or f"sfg_tci_rbr_mvp_{int(time.time())}"
    out_dir = resolve_output_dir(output_root / run_name, getattr(args, "resume", False), getattr(args, "overwrite", False))
    for name in [
        "confirmed_samples.jsonl", "full_evaluated_samples.jsonl", "rejected_sample_examples.jsonl", "candidate_debug_records.jsonl",
        "injection_attempt_report.jsonl", "sparse_target_bin_trace_examples.jsonl",
    ]:
        q = out_dir / name
        if q.exists():
            q.unlink()
    target_path = Path(args.target_distribution_json)
    target = F.load_target_distribution(target_path, args.target_total_samples, config)
    F.write_json(out_dir / "target_distribution_used.json", target)
    archive = F.TargetDistributionArchive(target, config.get("diversity", {}), config)
    quantizer = F.Quantizer(config)
    start_goal_assigner = F.StartGoalAssigner(config, rng)
    seed_gens = F.SeedGenerators(config, rng)
    transforms = F.Transforms(config, rng)
    sfg_gen = SkeletonFirstGenerator(int(config["grid"].get("size", 8)), rng)
    rbr = RollbackBuffer(args.max_buffer_size)
    reports = initialize_reports_12()
    source_reference = getattr(F, "__mvp_import_path__", "unknown")
    write_json(out_dir / "run_metadata.json", {
        "version": VERSION,
        "title": TITLE,
        "experiment_role": "clean_distribution_quality_validation_factory",
        "does_not_modify_3_0_4_source": True,
        "source_reference": source_reference,
        "mvp_script_path": str(Path(__file__).resolve()),
        "algorithm_name": "TDA-QD MazeForge",
        "version": VERSION,
        "closure_patch": True,
        "does_not_modify_generator_core": True,
        "default_hard_quota_scope": "handdraw_observed",
        "algorithm_full_name": "Terminal-Diameter Aligned Quality-Diversity MazeForge",
        "final_archive_bin_source": "full_bin_key",
        "generator_policy": "frozen_3_0_4_13_core_algorithm",
        "purpose": "multi_seed_stability_topology_diversity_and_until_target_distribution",
        "focus": "handdraw_quota_closure_csv_export",
        "algorithm": "Terminal-Diameter Alignment + Target-Class Route + Pure TBI + Stage-Based CLO + RBR accepted-as-commit",
        "mvp_policy": "tda_qd_mazeforge",
        "enable_sfg": bool(args.enable_sfg),
        "enable_tbi": bool(args.enable_tbi),
        "enable_clo": bool(args.enable_clo),
        "enable_spb": bool(args.enable_spb),
        "enable_rbr": bool(args.enable_rbr),
        "enable_pure_tbi": bool(args.enable_pure_tbi),
        "enable_stage_clo": bool(args.enable_stage_clo),
    })
    write_json(out_dir / "resolved_config.json", {"base_config": config, "mvp_args": vars(args), "critical_long_bins": CRITICAL_LONG_BINS})
    full_eval_examples_by_bin: Dict[str, List[str]] = defaultdict(list)
    trace_counters = Counter()
    accepted_records = []
    max_steps = int(config.get("transform", {}).get("max_transform_steps", 6))
    base_gen_weights = {"sfg_generator": 2.0, "random_wall_generator": 0.8, "dfs_tree_generator": 0.8, "loose_path_first_generator": 0.6, "room_biased_generator": 0.6, "template_mutation_generator": 0.1}
    start_time_for_stop = time.time()

    _seed_source = range(int(args.n_seeds)) if int(args.n_seeds) > 0 else count()
    _seed_iter = progress(_seed_source, desc="TDA-QD MazeForge")
    args._out_dir = str(out_dir)
    _progress_state: Dict[str, int] = {"last_confirmed": -1, "last_seed_idx": -1}
    for seed_idx in _seed_iter:
        maybe_refresh_mazeforge_progress(_seed_iter, seed_idx, target, archive, reports, args, _progress_state)
        stop_checker = getattr(args, "_stop_checker", None)
        if stop_checker is not None and stop_checker(target, archive, reports, seed_idx, time.time() - start_time_for_stop):
            break
        if args.max_total_full_extractions and reports["full_evaluated"] >= args.max_total_full_extractions:
            break
        route_report = build_target_class_routes(target, archive.counts, reports["full_bin_eval"], reports["full_bin_accept"], args) if target_class_mode else compute_route_demands(target, archive.counts, reports["full_bin_eval"], reports["full_bin_accept"])
        # Choose either RBR replay or a new skeleton/seed.
        use_rbr = args.enable_rbr and len(rbr) > 0 and rng.random() < rbr.replay_probability(route_report)
        item: Optional[RBRItem] = None
        cand = None
        from_rbr = False
        if use_rbr:
            item = rbr.sample(rng, route_report)
            if item is not None:
                from_rbr = True
                if target_class_mode:
                    pr = compute_rbr_priority(item, route_report)
                    reports["rbr_priority_values"].append(float(pr))
                    if item.target_route in CRITICAL_ROUTE_IDS:
                        reports["critical_route_replay_count"] += 1
                    else:
                        reports["noncritical_route_replay_count"] += 1
                cand = item.to_candidate(f"rbr_{item.item_id}_{seed_idx:06d}_stable")
        if cand is None:
            route = choose_target_class_route(rng, route_report) if target_class_mode else choose_route(rng, route_report)
            # SFG gets higher weight when long-route gap exists. Other generators keep baseline diversity.
            weights = dict(base_gen_weights)
            if args.enable_sfg:
                weights["sfg_generator"] *= 1.0 + 3.0 * min(1.0, sum(route_report.get(k, 0.0) for k in ["tree_long_demand", "single_loop_long_demand", "multi_loop_long_demand"]) / 100.0)
            else:
                weights["sfg_generator"] = 0.0
            gen_name = choose_weighted(rng, weights)
            reports["generator_selected"][gen_name] += 1
            if gen_name == "sfg_generator":
                reports["sfg_attempts"] += 1
                grid = sfg_gen.make_grid()
            else:
                mapped = gen_name if gen_name != "sfg_generator" else "loose_path_first_generator"
                grid = seed_gens.make_grid(mapped)
            if target_class_mode and gen_name == "sfg_generator":
                before_tmp_s, before_tmp_g, _ = start_goal_assigner.assign(grid, archive)
                reports["sfg_seed_bfs_before_diameter"].append((bfs_dist(grid, before_tmp_s).get(before_tmp_g) if before_tmp_s and before_tmp_g else None) or 0)
                s, g, sg_dbg = assign_start_goal_by_diameter_like_pair(grid, rng, start_goal_assigner, archive, args.diameter_start_goal_sample_count)
                reports["diameter_pair_attempts"] += int(sg_dbg.get("diameter_pair_attempts", 0))
                reports["diameter_pair_success"] += 1 if sg_dbg.get("diameter_pair_success") else 0
                reports["diameter_pair_fallback_count"] += int(sg_dbg.get("diameter_pair_fallback_count", 0))
                reports["sfg_seed_bfs_after_diameter"].append((bfs_dist(grid, s).get(g) if s and g else None) or 0)
            else:
                s, g, _sg = start_goal_assigner.assign(grid, archive)
            cand = F.MazeCandidate(grid=grid, start=s, goal=g, origin_generator=gen_name, origin_seed_id=f"seed{seed_idx:06d}_{gen_name}", transform_history=[], candidate_id=f"seed{seed_idx:06d}_step0")
            ok, metrics, bin_info, bin_key, hard, err = evaluate_full(cand, quantizer, config, archive)
            reports["candidate_total"] += 1
            if not ok:
                reports["reject_reasons"][err or "seed_reject"] += 1
                append_jsonl(out_dir / "candidate_debug_records.jsonl", {"candidate_id": cand.candidate_id, "stage": "initial_skeleton", "hard_quality": hard, "error": err})
                continue
            reports["hard_passed"] += 1
            reports["full_evaluated"] += 1
            reports["full_bin_eval"][bin_key] += 1
            reports["generator_full"][gen_name] += 1
            reports["generator_bins"][gen_name][bin_key] += 1
            reports["full_by_gen_bin"][gen_name][bin_key] += 1
            reports["full_by_op_bin"]["seed_step"][bin_key] += 1
            if bin_parts(bin_key)[1] == "long" or (metrics.get("bfs_len") or 0) >= 13:
                reports["generator_long"][gen_name] += 1
            if gen_name == "sfg_generator":
                reports["sfg_hard"] += 1; reports["sfg_full"] += 1; reports["sfg_bins"][bin_key] += 1; reports["sfg_bfs_values"].append(metrics.get("bfs_len") or 0)
                if bin_parts(bin_key)[1] == "long": reports["sfg_long"] += 1
            reports["route_full"][route] += 1
            # Initial skeleton can be accepted too.
            target_b = target["bins"].get(bin_key, {})
            accepted, acceptance, reject_reason = archive.maybe_accept(cand, metrics, bin_info, rng)
            if accepted:
                reports["accepted"] += 1; reports["full_bin_accept"][bin_key] += 1; reports["generator_accepted"][gen_name] += 1; reports["accepted_by_gen_bin"][gen_name][bin_key] += 1; reports["accepted_by_op_bin"]["seed_step"][bin_key] += 1; reports["route_accepted"][route] += 1
                if gen_name == "sfg_generator": reports["sfg_accepted"] += 1; reports["from_sfg_by_bin"][bin_key] += 1
                rec = {"maze_id": cand.candidate_id, "sample_id": cand.candidate_id, "grid": cand.grid, "ascii_grid": grid_to_ascii(cand.grid, cand.start, cand.goal), "start": list(cand.start), "goal": list(cand.goal), "origin_generator": gen_name, "origin_seed_id": cand.origin_seed_id, "from_sfg": gen_name == "sfg_generator", "from_rbr": False, "transform_history": cand.transform_history, "metrics": metrics, "full_bin": bin_info, "target_bin": {"status": target_b.get("status"), "target_quota": target_b.get("target_quota"), "target_prob": target_b.get("target_prob"), "handdraw_count": target_b.get("handdraw_count"), "count_before": acceptance.get("count_before"), "count_after": acceptance.get("count_after")}, "acceptance": {"accepted": True, "reject_reason": None, "policy": "full_bin_key_target_quota"}}
                append_jsonl(out_dir / "confirmed_samples.jsonl", rec); accepted_records.append(rec)
            else:
                reports["reject_reasons"][reject_reason or "seed_full_reject"] += 1
            append_jsonl(out_dir / "full_evaluated_samples.jsonl", {"candidate_id": cand.candidate_id, "origin_generator": gen_name, "origin_seed_id": cand.origin_seed_id, "last_operator": "seed_step", "route": route, "full_bin_key": bin_key, "full_bin": bin_info, "full_metrics": compact_metrics(metrics), "target_bin": {"status": target_b.get("status"), "target_quota": target_b.get("target_quota"), "target_prob": target_b.get("target_prob"), "handdraw_count": target_b.get("handdraw_count")}, "accepted": accepted, "reject_reason": None if accepted else reject_reason, "from_sfg": gen_name == "sfg_generator", "from_rbr": False})
            # Admit as RBR item if potential remains.
            can_admit, reasons = should_admit_buffer(metrics, bin_key, None, None, compute_deficit_scores(target, archive.counts), target, archive.counts)
            if args.enable_rbr and can_admit:
                item = RBRItem(f"item_{seed_idx:06d}", cand, metrics, bin_key, route, gen_name)
                rbr.admit(item, ["initial_stable_skeleton"] + reasons)
                for rr in ["initial_stable_skeleton"] + reasons:
                    reports["buffer_admit_reasons"][rr] += 1
            else:
                reports["buffer_reject_reasons"]["not_near_target_or_disabled"] += 1
                item = None
            if bin_key in CRITICAL_LONG_BINS and len(full_eval_examples_by_bin[bin_key]) < 20:
                full_eval_examples_by_bin[bin_key].append(cand.candidate_id)
        # RBR/trial injection phase. For new accepted skeletons that were not admitted, no trial stage.
        if not args.enable_rbr or item is None:
            continue
        # One replay cycle performs up to max_steps trial injections, with stable/trial separation.
        for local_step in range(min(max_steps, max(1, args.max_phase_attempts_per_item))):
            if args.max_total_full_extractions and reports["full_evaluated"] >= args.max_total_full_extractions:
                break
            rbr.attempt_count_total += 1
            item.phase_attempt_count += 1
            item.lifetime_attempt_count += 1
            before_metrics = copy.deepcopy(item.stable_full_metrics)
            before_bin = item.stable_full_bin_key
            trial = item.to_candidate(f"{item.item_id}_trial_{seed_idx:06d}_{local_step:02d}")
            if target_class_mode:
                op, op_sched_debug = choose_operator_by_stage(item, item.target_route, before_metrics, route_report, rng, args)
                reports["stage_operator_counts"][op] += 1
                reports["stage_operator_by_stage"][op_sched_debug.get("stage", "unknown")][op] += 1
            else:
                op = choose_route_operator(rng, item.target_route, before_metrics, route_report, args)
                op_sched_debug = {"stage": "coarse_route", "operator_weights": {op: 1.0}}
            fam = {"pure_tbi": "TBI", "branch_grow": "TBI", "stage_clo": "CLO", "loop_flip": "CLO", "spb": "SPB", "shortcut_block": "SPB", "cell_flip_random": "RND", "random": "RND"}.get(op, "RND")
            if fam == "TBI": reports["tbi_attempts"] += 1
            elif fam == "CLO": reports["clo_attempts"] += 1
            elif fam == "SPB": reports["spb_attempts"] += 1
            reports["operator_attempts"][op] += 1
            cheap = quantizer.evaluate(trial)
            if target_class_mode and op == "pure_tbi":
                child, op_info = apply_pure_tree_branch_injection(trial, rng, item.target_route, args)
                if child is not None:
                    reports["pure_tbi_successes"] += 1
                    reports["pure_tbi_free_neighbor_violation_count"] += int(op_info.get("free_neighbor_violation_count", 0))
            elif target_class_mode and op == "stage_clo":
                child, op_info = apply_stage_controlled_loop_overlay(trial, rng, item.target_route, args)
                if child is not None:
                    reports["stage_clo_successes"] += 1
            elif target_class_mode and op == "spb":
                child, op_info = transforms.apply("shortcut_block", trial, cheap)
                if child is not None:
                    reports["spb_successes"] += 1
            elif target_class_mode and op == "random":
                child, op_info = transforms.apply("cell_flip_random", trial, cheap)
            else:
                child, op_info = transforms.apply(op, trial, cheap)
            if child is None:
                reason = f"hard_transform_failed:{op_info.get('reason','unknown')}"
                item.rollback(reason); rbr.rollback_reasons[reason] += 1; reports["rollback_reasons"][reason] += 1
                if fam == "TBI": reports["tbi_rollback"] += 1
                elif fam == "CLO": reports["clo_rollback"] += 1
                elif fam == "SPB": reports["spb_rollback"] += 1
                continue
            reports["operator_successes"][op] += 1
            child.candidate_id = f"{item.item_id}_trial_{seed_idx:06d}_{local_step:02d}_{op}"
            ok, after_metrics, after_bin_info, after_bin, hard, err = evaluate_full(child, quantizer, config, archive)
            reports["candidate_total"] += 1
            if not ok:
                item.rollback("hard_quality_or_full_extraction_fail")
                reports["reject_reasons"][err or "hard_quality_or_full_extraction_fail"] += 1
                rbr.rollback_reasons[err or "hard_quality_or_full_extraction_fail"] += 1
                append_jsonl(out_dir / "rejected_sample_examples.jsonl", {"candidate_id": child.candidate_id, "route": item.target_route, "operator": op, "error": err, "hard_quality": hard, "grid": child.grid, "start": list(child.start), "goal": list(child.goal)})
                continue
            reports["hard_passed"] += 1; reports["full_evaluated"] += 1
            reports["full_bin_eval"][after_bin] += 1
            reports["generator_full"][child.origin_generator] += 1
            reports["full_by_gen_bin"][child.origin_generator][after_bin] += 1
            reports["full_by_op_bin"][op][after_bin] += 1
            reports["route_full"][item.target_route] += 1
            if bin_parts(after_bin)[1] == "long" or (after_metrics.get("bfs_len") or 0) >= 13:
                reports["generator_long"][child.origin_generator] += 1
            record_delta(reports, fam, before_metrics, after_metrics)
            if target_class_mode:
                route_ok, route_reason = target_class_route_constraints(item.target_route, before_metrics, after_metrics, route_report, args)
                progress_ok, progress_reasons = target_class_progress_events(item.target_route, before_metrics, before_bin, after_metrics, after_bin, route_report)
            else:
                route_ok, route_reason = route_constraints(item.target_route, before_metrics, after_metrics)
                progress_ok, progress_reasons = progress_events(item.target_route, before_metrics, before_bin, after_metrics, after_bin, target, archive.counts)
            target_b = target["bins"].get(after_bin, {})
            accepted, acceptance, reject_reason = archive.maybe_accept(child, after_metrics, after_bin_info, rng)
            decision = "rollback"; decision_reason = route_reason if not route_ok else "no_progress"
            accepted_as_commit = False
            if accepted:
                decision = "accept"; decision_reason = "accepted_as_commit"; accepted_as_commit = True
                reports["accepted"] += 1; reports["full_bin_accept"][after_bin] += 1; reports["generator_accepted"][child.origin_generator] += 1; reports["accepted_by_gen_bin"][child.origin_generator][after_bin] += 1; reports["accepted_by_op_bin"][op][after_bin] += 1; reports["operator_accepted_after"][op] += 1; reports["route_accepted"][item.target_route] += 1
                reports["accepted_commit_triggered"] += 1; reports["stable_state_updated_after_accept"] += 1; reports["phase_counter_refresh_count"] += 1
                rbr.accept_count += 1; rbr.accepted_commit_count += 1; rbr.phase_counter_refresh_count += 1
                if after_bin in CRITICAL_LONG_BINS or is_adjacent_to_sparse_target(after_bin, target, archive.counts):
                    rbr.sparse_hit_count += 1
                if child.origin_generator == "sfg_generator": reports["from_sfg_by_bin"][after_bin] += 1
                reports["from_rbr_by_bin"][after_bin] += 1
                rec = {"maze_id": child.candidate_id, "sample_id": child.candidate_id, "grid": child.grid, "ascii_grid": grid_to_ascii(child.grid, child.start, child.goal), "start": list(child.start), "goal": list(child.goal), "origin_generator": child.origin_generator, "origin_seed_id": child.origin_seed_id, "from_sfg": child.origin_generator == "sfg_generator", "from_rbr": True, "route": item.target_route, "operator": op, "transform_history": child.transform_history, "metrics": after_metrics, "full_bin": after_bin_info, "target_bin": {"status": target_b.get("status"), "target_quota": target_b.get("target_quota"), "target_prob": target_b.get("target_prob"), "handdraw_count": target_b.get("handdraw_count"), "count_before": acceptance.get("count_before"), "count_after": acceptance.get("count_after")}, "acceptance": {"accepted": True, "reject_reason": None, "policy": "full_bin_key_target_quota"}}
                append_jsonl(out_dir / "confirmed_samples.jsonl", rec); accepted_records.append(rec)
            elif not route_ok:
                decision = "rollback"; decision_reason = route_reason
            elif progress_ok:
                decision = "commit"; decision_reason = ";".join(progress_reasons)
            else:
                decision = "rollback"; decision_reason = "no_progress"
            if not route_ok:
                if fam == "TBI" and "cycle" in decision_reason:
                    reports["pure_tbi_cycle_violation_count"] += 1
                if fam == "CLO" and "no_cycle" in decision_reason:
                    reports["stage_clo_no_cycle_gain_count"] += 1
                if fam == "CLO" and "bfs" in decision_reason:
                    reports["stage_clo_bfs_drop_rollback_count"] += 1
                if fam == "CLO" and "endpoint" in decision_reason:
                    reports["stage_clo_endpoint_drop_rollback_count"] += 1
                if fam == "SPB" and "bfs" in decision_reason:
                    reports["spb_bfs_drop_rollback_count"] += 1
            # Injection attempt report before state mutation.
            delta_obj = {
                "delta_bfs": (after_metrics.get("bfs_len") or 0) - (before_metrics.get("bfs_len") or 0),
                "delta_cycle_rank": (after_metrics.get("rect_room_compressed_cycle_rank") or 0) - (before_metrics.get("rect_room_compressed_cycle_rank") or 0),
                "delta_choice_count": (after_metrics.get("rect_room_macro_choice_count") or 0) - (before_metrics.get("rect_room_macro_choice_count") or 0),
                "delta_endpoint_count": (after_metrics.get("rect_room_macro_endpoint_count") or 0) - (before_metrics.get("rect_room_macro_endpoint_count") or 0),
            }
            attempt_record = {"item_id": item.item_id, "route": item.target_route, "operator": op, "stage": op_sched_debug.get("stage"), "stage_reason": op_sched_debug.get("stage_reason"), "operator_weights": op_sched_debug.get("operator_weights"), "attempt_index": item.lifetime_attempt_count, "stable_bin_before": before_bin, "trial_bin_after": after_bin, "stable_metrics_before": compact_metrics(before_metrics), "trial_metrics_after": compact_metrics(after_metrics), "decision": decision, "reason": decision_reason, "accepted_as_commit": accepted_as_commit, "phase_counters_reset_after_accept": accepted_as_commit, "stable_state_updated_after_accept": accepted_as_commit or decision == "commit", **delta_obj}
            append_jsonl(out_dir / "injection_attempt_report.jsonl", attempt_record)
            reports["injection_attempts"].append(attempt_record)
            append_jsonl(out_dir / "full_evaluated_samples.jsonl", {"candidate_id": child.candidate_id, "origin_generator": child.origin_generator, "origin_seed_id": child.origin_seed_id, "last_operator": op, "route": item.target_route, "full_bin_key": after_bin, "full_bin": after_bin_info, "full_metrics": compact_metrics(after_metrics), "target_bin": {"status": target_b.get("status"), "target_quota": target_b.get("target_quota"), "target_prob": target_b.get("target_prob"), "handdraw_count": target_b.get("handdraw_count")}, "accepted": accepted, "reject_reason": None if accepted else reject_reason, "from_sfg": child.origin_generator == "sfg_generator", "from_rbr": True})
            if after_bin in CRITICAL_LONG_BINS:
                trace = {"final_bin_key": after_bin, "accepted": accepted, "origin_generator": child.origin_generator, "origin_seed_id": child.origin_seed_id, "from_sfg": child.origin_generator == "sfg_generator", "entered_rbr": True, "buffer_replay_count": item.lifetime_attempt_count, "candidate_id": child.candidate_id, "start": list(child.start), "goal": list(child.goal), "grid": child.grid, "process": [{"step": local_step, "operator": op, "route": item.target_route, "bin_before": before_bin, "bin_after": after_bin, "metrics_before": compact_metrics(before_metrics), "metrics_after": compact_metrics(after_metrics), **delta_obj, "decision": decision, "decision_reason": decision_reason, "admit_to_buffer": True, "admit_reason": "existing_rbr_item", "accepted_as_commit": accepted_as_commit, "phase_counters_reset": accepted_as_commit, "stable_state_updated": accepted_as_commit or decision == "commit"}]}
                save_process_trace(out_dir, after_bin, trace, args.max_trace_examples_per_bin, trace_counters)
                reports["critical_route_examples"][after_bin].append(child.candidate_id)
            # State transition.
            if decision == "accept":
                item.commit(child, after_metrics, after_bin, attempt_record, accepted=True)
                rbr.commit_reasons["accepted_as_commit"] += 1; reports["commit_reasons"]["accepted_as_commit"] += 1
                if fam == "TBI": reports["tbi_commit"] += 1
                elif fam == "CLO": reports["clo_commit"] += 1
                elif fam == "SPB": reports["spb_commit"] += 1
                reass = maybe_reassign_route(item, route_report, args)
                if reass:
                    reports["route_transitions"].append(reass); rbr.route_reassignment_count += 1
            elif decision == "commit":
                item.commit(child, after_metrics, after_bin, attempt_record, accepted=False)
                for pr in progress_reasons: rbr.commit_reasons[pr] += 1; reports["commit_reasons"][pr] += 1
                if fam == "TBI": reports["tbi_commit"] += 1
                elif fam == "CLO": reports["clo_commit"] += 1
                elif fam == "SPB": reports["spb_commit"] += 1
            else:
                item.rollback(decision_reason)
                rbr.rollback_reasons[decision_reason] += 1; reports["rollback_reasons"][decision_reason] += 1
                if fam == "TBI": reports["tbi_rollback"] += 1
                elif fam == "CLO": reports["clo_rollback"] += 1
                elif fam == "SPB": reports["spb_rollback"] += 1
            if reject_reason == "target_bin_full":
                item.full_bin_full_hits += 1
            if rbr.retire_if_needed(item, args):
                break
    # Write final reports.
    target_archive_summary = build_target_archive_summary_local(target, archive.counts)
    synthetic = build_synthetic_vs_target_report_local(target, archive.counts)
    write_json(out_dir / "target_archive_summary.json", target_archive_summary)
    write_json(out_dir / "synthetic_vs_target_distribution_report.json", synthetic)
    report, gap_summary = target_gap_report(target, archive.counts, reports["full_bin_eval"], reports["full_bin_accept"], reports["full_by_gen_bin"], reports["accepted_by_gen_bin"], reports["full_by_op_bin"], reports["accepted_by_op_bin"], full_eval_examples_by_bin)
    write_json(out_dir / "target_gap_summary.json", gap_summary)
    write_json(out_dir / "target_gap_attribution_report.json", report)
    route_report_final = build_target_class_routes(target, archive.counts, reports["full_bin_eval"], reports["full_bin_accept"], args) if target_class_mode else compute_route_demands(target, archive.counts, reports["full_bin_eval"], reports["full_bin_accept"])
    route_report_final.update({"route_full_evaluated_count": dict(reports["route_full"]), "route_accepted_count": dict(reports["route_accepted"])})
    write_json(out_dir / "route_demand_report.json", route_report_final)
    write_json(out_dir / "target_class_route_report.json", route_report_final)
    sfg_bfs = reports["sfg_bfs_values"]
    write_json(out_dir / "skeleton_generation_report.json", {"SFG_attempts": reports["sfg_attempts"], "SFG_hard_quality_pass": reports["sfg_hard"], "SFG_full_evaluated": reports["sfg_full"], "SFG_long_bfs_count": reports["sfg_long"], "SFG_accepted_count": reports["sfg_accepted"], "SFG_top_bins": summarize_counter(reports["sfg_bins"], 20), "SFG_avg_bfs_len": statistics.mean(sfg_bfs) if sfg_bfs else None, "SFG_bfs_bin_distribution": Counter(bin_parts(k)[1] for k in reports["sfg_bins"].elements()), "SFG_endpoint_distribution": None, "SFG_choice_distribution": None, "SFG_cycle_distribution": None, "SFG_hash_unique_count": None})
    write_json(out_dir / "tree_branch_injection_report.json", {"TBI_attempts": reports["tbi_attempts"], "TBI_commit_count": reports["tbi_commit"], "TBI_rollback_count": reports["tbi_rollback"], "avg_delta_endpoint": statistics.mean(reports["tbi_deltas"].get("endpoint", [0])) if reports["tbi_deltas"].get("endpoint") else 0, "avg_delta_choice": statistics.mean(reports["tbi_deltas"].get("choice", [0])) if reports["tbi_deltas"].get("choice") else 0, "avg_delta_cycle": statistics.mean(reports["tbi_deltas"].get("cycle", [0])) if reports["tbi_deltas"].get("cycle") else 0, "cycle_violation_count": reports["rollback_reasons"].get("route_cycle_violation_tree", 0), "hard_collapse_count": sum(v for k,v in reports["rollback_reasons"].items() if str(k).startswith("hard"))})
    write_json(out_dir / "controlled_loop_overlay_report.json", {"CLO_attempts": reports["clo_attempts"], "CLO_commit_count": reports["clo_commit"], "CLO_rollback_count": reports["clo_rollback"], "avg_delta_cycle": statistics.mean(reports["clo_deltas"].get("cycle", [0])) if reports["clo_deltas"].get("cycle") else 0, "avg_delta_bfs": statistics.mean(reports["clo_deltas"].get("bfs", [0])) if reports["clo_deltas"].get("bfs") else 0, "avg_delta_endpoint": statistics.mean(reports["clo_deltas"].get("endpoint", [0])) if reports["clo_deltas"].get("endpoint") else 0, "over_cycle_rollback_count": reports["rollback_reasons"].get("route_over_cycle_single_loop", 0), "bfs_regression_rollback_count": reports["rollback_reasons"].get("route_bfs_dropped_out_of_long", 0)})
    write_json(out_dir / "shortcut_preserving_block_report.json", {"SPB_attempts": reports["spb_attempts"], "SPB_commit_count": reports["spb_commit"], "SPB_rollback_count": reports["spb_rollback"], "avg_delta_bfs": statistics.mean(reports["spb_deltas"].get("bfs", [0])) if reports["spb_deltas"].get("bfs") else 0, "avg_delta_endpoint": statistics.mean(reports["spb_deltas"].get("endpoint", [0])) if reports["spb_deltas"].get("endpoint") else 0, "avg_delta_choice": statistics.mean(reports["spb_deltas"].get("choice", [0])) if reports["spb_deltas"].get("choice") else 0, "avg_delta_cycle": statistics.mean(reports["spb_deltas"].get("cycle", [0])) if reports["spb_deltas"].get("cycle") else 0})
    write_json(out_dir / "rollback_buffer_report.json", {"buffer_item_count": len(rbr), "attempt_count_total": rbr.attempt_count_total, "commit_count": sum(rbr.commit_reasons.values()), "rollback_count": sum(rbr.rollback_reasons.values()), "rollback_reasons": dict(rbr.rollback_reasons), "commit_reasons": dict(rbr.commit_reasons), "accept_count": rbr.accept_count, "accepted_commit_count": rbr.accepted_commit_count, "accepted_keep_count": max(0, rbr.accept_count - rbr.evict_reasons.get("max_accepts",0)), "accepted_retire_count": rbr.evict_reasons.get("max_accepts",0), "phase_counter_refresh_count": rbr.phase_counter_refresh_count, "max_attempt_reached_count": rbr.evict_reasons.get("max_phase_attempts",0), "avg_attempts_per_item": rbr.attempt_count_total / max(1, len(rbr.items)), "attempts_before_accept": None, "lifetime_attempt_distribution": [it.lifetime_attempt_count for it in rbr.items], "lifetime_accept_distribution": [it.lifetime_accept_count for it in rbr.items], "route_reassignment_count": rbr.route_reassignment_count, "evict_reasons": dict(rbr.evict_reasons), "admit_reasons": dict(rbr.admit_reasons)})
    critical_route = {}
    for key in CRITICAL_LONG_BINS:
        critical_route[key] = {"bin_key": key, "route": (route_id_for_target_bin(key) if target_class_mode else route_for_bin(key)), "target_quota": int(target.get("bins",{}).get(key,{}).get("target_quota",0)), "confirmed_count": int(archive.counts.get(key,0)), "full_evaluated_count": int(reports["full_bin_eval"].get(key,0)), "fill_ratio": int(archive.counts.get(key,0)) / max(1, int(target.get("bins",{}).get(key,{}).get("target_quota",0))), "origin_generator": {g: int(m.get(key,0)) for g,m in reports["full_by_gen_bin"].items()}, "TBI_count": int(reports["full_by_op_bin"].get("branch_grow",Counter()).get(key,0)), "CLO_count": int(reports["full_by_op_bin"].get("loop_flip",Counter()).get(key,0)), "SPB_count": int(reports["full_by_op_bin"].get("shortcut_block",Counter()).get(key,0)), "rollback_count_before_accept": None, "commit_count_before_accept": None, "accepted_count": int(reports["full_bin_accept"].get(key,0)), "from_sfg_count": int(reports["from_sfg_by_bin"].get(key,0)), "from_rbr_count": int(reports["from_rbr_by_bin"].get(key,0)), "top_process_examples": reports["critical_route_examples"].get(key,[])[:10]}
    write_json(out_dir / "critical_bin_route_attribution.json", critical_route)
    diameter_report = {
        "diameter_pair_attempts": reports.get("diameter_pair_attempts", 0),
        "diameter_pair_success": reports.get("diameter_pair_success", 0),
        "diameter_pair_fallback_count": reports.get("diameter_pair_fallback_count", 0),
        "sfg_seed_bfs_before_diameter": summarize_numeric(reports.get("sfg_seed_bfs_before_diameter", [])),
        "sfg_seed_bfs_after_diameter": summarize_numeric(reports.get("sfg_seed_bfs_after_diameter", [])),
        "sfg_long_bfs_rate_before": sum(1 for x in reports.get("sfg_seed_bfs_before_diameter", []) if x >= 13) / max(1, len(reports.get("sfg_seed_bfs_before_diameter", []))),
        "sfg_long_bfs_rate_after": sum(1 for x in reports.get("sfg_seed_bfs_after_diameter", []) if x >= 13) / max(1, len(reports.get("sfg_seed_bfs_after_diameter", []))),
    }
    write_json(out_dir / "diameter_start_goal_report.json", diameter_report)
    write_json(out_dir / "pure_tree_branch_injection_report.json", {"pure_tbi_attempts": reports["tbi_attempts"], "pure_tbi_successes": reports.get("pure_tbi_successes", 0), "pure_tbi_commit_count": reports["tbi_commit"], "pure_tbi_rollback_count": reports["tbi_rollback"], "pure_tbi_avg_delta_endpoint": avg_delta_map(reports["tbi_deltas"]).get("endpoint", 0), "pure_tbi_avg_delta_choice": avg_delta_map(reports["tbi_deltas"]).get("choice", 0), "pure_tbi_avg_delta_cycle": avg_delta_map(reports["tbi_deltas"]).get("cycle", 0), "pure_tbi_cycle_violation_count": reports.get("pure_tbi_cycle_violation_count", 0), "pure_tbi_free_neighbor_violation_count": reports.get("pure_tbi_free_neighbor_violation_count", 0)})
    write_json(out_dir / "stage_controlled_loop_overlay_report.json", {"stage_clo_attempts": reports["clo_attempts"], "stage_clo_successes": reports.get("stage_clo_successes", 0), "stage_clo_commit_count": reports["clo_commit"], "stage_clo_rollback_count": reports["clo_rollback"], "stage_clo_avg_delta_cycle": avg_delta_map(reports["clo_deltas"]).get("cycle", 0), "stage_clo_avg_delta_bfs": avg_delta_map(reports["clo_deltas"]).get("bfs", 0), "stage_clo_avg_delta_endpoint": avg_delta_map(reports["clo_deltas"]).get("endpoint", 0), "stage_clo_no_cycle_gain_count": reports.get("stage_clo_no_cycle_gain_count", 0), "stage_clo_bfs_drop_rollback_count": reports.get("stage_clo_bfs_drop_rollback_count", 0), "stage_clo_endpoint_drop_rollback_count": reports.get("stage_clo_endpoint_drop_rollback_count", 0)})
    write_json(out_dir / "stage_operator_report.json", {"operator_selected_counts": dict(reports.get("stage_operator_counts", {})), "operator_by_stage": {k: dict(v) for k, v in reports.get("stage_operator_by_stage", {}).items()}})
    write_json(out_dir / "rbr_priority_report.json", {"priority_distribution": summarize_numeric(reports.get("rbr_priority_values", [])), "critical_route_replay_count": reports.get("critical_route_replay_count", 0), "noncritical_route_replay_count": reports.get("noncritical_route_replay_count", 0), "top_priority_route_ids": summarize_counter(Counter([it.target_route for it in rbr.items]), 10), "low_priority_evicted_count": int(rbr.evict_reasons.get("low_priority", 0))})
    write_json(out_dir / "buffer_admission_report.json", {"admit_count_by_reason": dict(reports.get("buffer_admit_reasons", {})), "reject_count_by_reason": dict(reports.get("buffer_reject_reasons", {})), "critical_route_admit_count": sum(1 for it in rbr.items if it.target_route in CRITICAL_ROUTE_IDS), "easy_bucket_admit_count": sum(1 for it in rbr.items if it.target_route not in CRITICAL_ROUTE_IDS), "tree_low_mid_over_admit_count": sum(1 for it in rbr.items if it.target_route == "tree_long_low_choice_mid_endpoint")})
    write_json(out_dir / "route_specific_progress_report.json", {"commit_reasons": dict(reports["commit_reasons"]), "rollback_reasons": dict(reports["rollback_reasons"]), "note": "Progress events are route-specific in target_class_sfg_rbr mode."})
    write_json(out_dir / "critical_target_distance_report.json", {key: {"best_distance_seen": min([target_distance_to_bin({"bfs_len": 0}, key)] + [0]) if False else None, "full_evaluated_count": int(reports["full_bin_eval"].get(key, 0)), "accepted_count": int(reports["full_bin_accept"].get(key, 0))} for key in CRITICAL_LONG_BINS})
    write_json(out_dir / "route_transition_report.json", {"transitions": reports["route_transitions"], "route_reassignment_count": len(reports["route_transitions"])})
    write_json(out_dir / "before_after_bucket_shift_report.json", {"note": "Per-attempt before/after full-bin shifts are in injection_attempt_report.jsonl."})
    write_json(out_dir / "sparse_bin_process_trace_summary.json", {"trace_bins": len(trace_counters), "trace_examples_saved": int(sum(trace_counters.values())), "trace_output_dir": str(out_dir / "sparse_bin_process_traces"), "trace_counts_by_bin": dict(trace_counters)})
    write_json(out_dir / "full_evaluated_bin_distribution.json", {"total_full_evaluated": reports["full_evaluated"], "bins": {k: {"full_evaluated_count": int(reports["full_bin_eval"].get(k,0)), "accepted_count": int(reports["full_bin_accept"].get(k,0)), "target_quota": int(target.get("bins",{}).get(k,{}).get("target_quota",0)), "target_status": target.get("bins",{}).get(k,{}).get("status")} for k in target.get("bins",{})}})
    write_json(out_dir / "generator_transform_replay_attribution_report.json", {"full_by_generator_bin": {g: dict(c) for g,c in reports["full_by_gen_bin"].items()}, "accepted_by_generator_bin": {g: dict(c) for g,c in reports["accepted_by_gen_bin"].items()}, "full_by_operator_bin": {op: dict(c) for op,c in reports["full_by_op_bin"].items()}, "accepted_by_operator_bin": {op: dict(c) for op,c in reports["accepted_by_op_bin"].items()}, "from_sfg_by_bin": dict(reports["from_sfg_by_bin"]), "from_rbr_by_bin": dict(reports["from_rbr_by_bin"])})
    write_json(out_dir / "rejection_summary.json", dict(reports["reject_reasons"]))
    # Terminal summary.
    print(TITLE)
    print("\n=== Protocol Compliance ===")
    print("does_not_modify_3_0_4_source      True")
    print(f"source_reference                  {source_reference}")
    print(f"mvp_script_path                   {Path(__file__).resolve()}")
    print("\n=== Target Distribution ===")
    print(f"target_total                      {target.get('target_total')}")
    print(f"quota_sum                         {target.get('quota_sum')}")
    print(f"handdraw_observed_bins            {sum(1 for b in target.get('bins',{}).values() if b.get('status')=='handdraw_observed')}")
    print(f"nonzero_target_bins               {sum(1 for b in target.get('bins',{}).values() if int(b.get('target_quota',0))>0)}")
    print("\n=== Candidate Funnel ===")
    print(f"n_candidate_total                 {reports['candidate_total']}")
    print(f"n_hard_quality_passed             {reports['hard_passed']}")
    print(f"n_full_evaluated                  {reports['full_evaluated']}")
    print(f"n_full_accepted                   {reports['accepted']}")
    print("\n=== MVP Policy ===")
    print(f"mvp_policy                        {args.mvp_policy}")
    print(f"enable_sfg                        {args.enable_sfg}")
    print(f"enable_tbi                        {args.enable_tbi}")
    print(f"enable_clo                        {args.enable_clo}")
    print(f"enable_spb                        {args.enable_spb}")
    print(f"enable_rbr                        {args.enable_rbr}")
    print("\n=== Route Demand ===")
    for k in ["tree_long_demand", "single_loop_long_demand", "multi_loop_long_demand", "long_high_endpoint_demand"]:
        print(f"{k:32s} {route_report_final.get(k,0)}")
    print("\n=== Skeleton Generation Report ===")
    print(f"SFG_attempts                      {reports['sfg_attempts']}")
    print(f"SFG_full_evaluated                {reports['sfg_full']}")
    print(f"SFG_long_bfs_count                {reports['sfg_long']}")
    print(f"SFG_accepted_count                {reports['sfg_accepted']}")
    print(f"SFG_avg_bfs_len                   {statistics.mean(sfg_bfs) if sfg_bfs else None}")
    print("\n=== Injection Report ===")
    print(f"TBI_attempts                      {reports['tbi_attempts']}")
    print(f"TBI_commit_count                  {reports['tbi_commit']}")
    print(f"TBI_rollback_count                {reports['tbi_rollback']}")
    print(f"TBI_avg_delta_endpoint            {avg_delta_map(reports['tbi_deltas']).get('endpoint',0):.3f}")
    print(f"TBI_avg_delta_choice              {avg_delta_map(reports['tbi_deltas']).get('choice',0):.3f}")
    print(f"TBI_avg_delta_cycle               {avg_delta_map(reports['tbi_deltas']).get('cycle',0):.3f}")
    print(f"CLO_attempts                      {reports['clo_attempts']}")
    print(f"CLO_commit_count                  {reports['clo_commit']}")
    print(f"CLO_rollback_count                {reports['clo_rollback']}")
    print(f"CLO_avg_delta_cycle               {avg_delta_map(reports['clo_deltas']).get('cycle',0):.3f}")
    print(f"CLO_avg_delta_bfs                 {avg_delta_map(reports['clo_deltas']).get('bfs',0):.3f}")
    print(f"CLO_avg_delta_endpoint            {avg_delta_map(reports['clo_deltas']).get('endpoint',0):.3f}")
    print(f"SPB_attempts                      {reports['spb_attempts']}")
    print(f"SPB_commit_count                  {reports['spb_commit']}")
    print(f"SPB_rollback_count                {reports['spb_rollback']}")
    print(f"SPB_avg_delta_bfs                 {avg_delta_map(reports['spb_deltas']).get('bfs',0):.3f}")
    print("\n=== Rollback Buffer Report ===")
    print(f"buffer_item_count                 {len(rbr)}")
    print(f"attempt_count_total               {rbr.attempt_count_total}")
    print(f"commit_count                      {sum(rbr.commit_reasons.values())}")
    print(f"rollback_count                    {sum(rbr.rollback_reasons.values())}")
    print(f"accept_count                      {rbr.accept_count}")
    print(f"accepted_commit_count             {rbr.accepted_commit_count}")
    print(f"phase_counter_refresh_count       {rbr.phase_counter_refresh_count}")
    print(f"avg_attempts_per_item             {rbr.attempt_count_total / max(1, len(rbr.items)):.3f}")
    print(f"top_rollback_reasons              {summarize_counter(rbr.rollback_reasons,5)}")
    print(f"top_commit_reasons                {summarize_counter(rbr.commit_reasons,5)}")
    print("\n=== Diameter Start/Goal Report ===")
    print(f"diameter_pair_attempts            {reports.get('diameter_pair_attempts', 0)}")
    print(f"diameter_pair_success             {reports.get('diameter_pair_success', 0)}")
    print(f"diameter_pair_fallback_count      {reports.get('diameter_pair_fallback_count', 0)}")
    before_sfg = reports.get('sfg_seed_bfs_before_diameter', [])
    after_sfg = reports.get('sfg_seed_bfs_after_diameter', [])
    print(f"sfg_long_bfs_rate_before          {sum(1 for x in before_sfg if x >= 13) / max(1, len(before_sfg)):.3f}")
    print(f"sfg_long_bfs_rate_after           {sum(1 for x in after_sfg if x >= 13) / max(1, len(after_sfg)):.3f}")
    if target_class_mode:
        print("\n=== Target-Class Route Report ===")
        routes_obj = route_report_final.get('routes', {})
        for rid in list(TARGET_CLASS_ROUTE_PRIMARY.keys()) + ['generic_tree_long', 'generic_single_loop_long', 'generic_multi_loop_long']:
            rr = routes_obj.get(rid)
            if rr:
                print(f"{rid:44s} primary={rr.get('primary_bin')} priority={rr.get('priority_score',0):.5f} selected={reports['route_full'].get(rid,0)} full_eval={rr.get('full_evaluated_count',0)} accepted={rr.get('accepted_count',0)} fill={rr.get('fill_ratio',0):.3f}")
    print("\n=== Pure TBI Report ===")
    print(f"pure_tbi_attempts                 {reports['tbi_attempts']}")
    print(f"pure_tbi_successes                {reports.get('pure_tbi_successes', 0)}")
    print(f"pure_tbi_commit_count             {reports['tbi_commit']}")
    print(f"pure_tbi_rollback_count           {reports['tbi_rollback']}")
    print(f"pure_tbi_avg_delta_endpoint       {avg_delta_map(reports['tbi_deltas']).get('endpoint',0):.3f}")
    print(f"pure_tbi_avg_delta_choice         {avg_delta_map(reports['tbi_deltas']).get('choice',0):.3f}")
    print(f"pure_tbi_avg_delta_cycle          {avg_delta_map(reports['tbi_deltas']).get('cycle',0):.3f}")
    print(f"pure_tbi_cycle_violation_count    {reports.get('pure_tbi_cycle_violation_count', 0)}")
    print("\n=== Stage CLO Report ===")
    print(f"stage_clo_attempts                {reports['clo_attempts']}")
    print(f"stage_clo_successes               {reports.get('stage_clo_successes', 0)}")
    print(f"stage_clo_commit_count            {reports['clo_commit']}")
    print(f"stage_clo_rollback_count          {reports['clo_rollback']}")
    print(f"stage_clo_avg_delta_cycle         {avg_delta_map(reports['clo_deltas']).get('cycle',0):.3f}")
    print(f"stage_clo_avg_delta_bfs           {avg_delta_map(reports['clo_deltas']).get('bfs',0):.3f}")
    print(f"stage_clo_avg_delta_endpoint      {avg_delta_map(reports['clo_deltas']).get('endpoint',0):.3f}")
    print(f"stage_clo_no_cycle_gain_count     {reports.get('stage_clo_no_cycle_gain_count', 0)}")
    print(f"stage_clo_bfs_drop_rollback_count {reports.get('stage_clo_bfs_drop_rollback_count', 0)}")
    print(f"stage_clo_endpoint_drop_rollback_count {reports.get('stage_clo_endpoint_drop_rollback_count', 0)}")
    print("\n=== RBR Priority Report ===")
    print(f"critical_route_replay_count       {reports.get('critical_route_replay_count', 0)}")
    print(f"noncritical_route_replay_count    {reports.get('noncritical_route_replay_count', 0)}")
    print(f"top_priority_route_ids            {summarize_counter(Counter([it.target_route for it in rbr.items]), 5)}")
    print("\n=== Critical Long Bins ===")
    for key in CRITICAL_LONG_BINS:
        row = critical_route[key]
        print(f"{key:45s} quota={row['target_quota']:4d} count={row['confirmed_count']:4d} full_eval={row['full_evaluated_count']:4d} fill={row['fill_ratio']:.3f} from_sfg={row['from_sfg_count']:3d} from_rbr={row['from_rbr_count']:3d} route={row['route']}")
    print("\n=== Target Archive Coverage ===")
    print(f"confirmed_total                   {synthetic['confirmed_total']}")
    print(f"target_fill_ratio                 {synthetic['target_fill_ratio']:.4f}")
    print(f"coverage_nonzero_target_bins      {synthetic.get('coverage_nonzero_target_bins')}")
    print(f"observed_handdraw_bin_fill_rate   {synthetic.get('observed_handdraw_bin_fill_rate')}")
    print(f"l1_distance_to_target_distribution {synthetic['l1_distance_to_target_distribution']:.4f}")
    print("\n=== Debug Hints ===")
    if reports["sfg_full"] and reports["sfg_long"] / max(1, reports["sfg_full"]) > 0.5:
        print("[HINT] SFG improved long supply; inspect skeleton_generation_report.json and critical_bin_route_attribution.json.")
    if reports["tbi_attempts"] and avg_delta_map(reports["tbi_deltas"]).get("cycle",0) > 0.2:
        print("[HINT] TBI is increasing cycle rank; tree-route injection constraints are too weak.")
    if reports["clo_attempts"] and avg_delta_map(reports["clo_deltas"]).get("cycle",0) <= 0:
        print("[HINT] CLO did not increase cycle rank; controlled loop overlay may be ineffective.")
    if rbr.replay_count and sum(rbr.commit_reasons.values()) == 0:
        print("[HINT] RBR replay happened but no commits; tighten admission or improve TCI operators.")
    if rbr.accepted_commit_count == 0:
        print("[HINT] accepted-as-commit was not triggered in this run; increase n-seeds or inspect rollback reasons.")
    if any(critical_route[k]["full_evaluated_count"] == 0 for k in CRITICAL_LONG_BINS):
        print("[HINT] Some critical long bins still have zero full_evaluated_count; SFG/TCI route coverage remains insufficient.")
    if not any([reports["sfg_full"], reports["tbi_attempts"], rbr.attempt_count_total]):
        print("[HINT] No SFG/TCI/RBR activity recorded; check mvp flags.")
    print(f"Output directory: {out_dir}")
    return out_dir



# ---------------- TDA-QD MazeForge validation helpers ----------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def compute_until_target_metrics(target: Dict[str, Any], counts: Counter, full_error_count: int = 0) -> Dict[str, Any]:
    synthetic = build_synthetic_vs_target_report_local(target, counts)
    crit = {k: int(counts.get(k, 0)) for k in CRITICAL_LONG_BINS}
    crit_full = {k: int(counts.get(k, 0)) for k in CRITICAL_LONG_BINS}
    return {
        "target_fill_ratio": float(synthetic.get("target_fill_ratio", 0.0)),
        "coverage_nonzero_target_bins": float(synthetic.get("coverage_nonzero_target_bins", 0.0)),
        "observed_handdraw_bin_fill_rate": float(synthetic.get("observed_handdraw_bin_fill_rate", 0.0)),
        "l1_distance_to_target_distribution": float(synthetic.get("l1_distance_to_target_distribution", 999.0)),
        "critical_bin_counts": crit,
        "critical_bins_nonzero": all(v > 0 for v in crit.values()),
        "full_extraction_error_count": int(full_error_count),
    }


def criteria_met(args: argparse.Namespace, m: Dict[str, Any]) -> bool:
    if m["target_fill_ratio"] < args.target_fill_ratio_threshold:
        return False
    if m["coverage_nonzero_target_bins"] < args.coverage_nonzero_target_bins_threshold:
        return False
    if m["observed_handdraw_bin_fill_rate"] < args.observed_handdraw_bin_fill_rate_threshold:
        return False
    if m["l1_distance_to_target_distribution"] > args.l1_distance_threshold:
        return False
    if args.require_critical_bins_nonzero and not m["critical_bins_nonzero"]:
        return False
    if args.require_full_extraction_error_zero and m["full_extraction_error_count"] != 0:
        return False
    return True



def normalize_quota_scope(quota_scope: str) -> str:
    if quota_scope == "exploration_stress_all_nonzero":
        return "all_nonzero"
    return quota_scope


def scoped_target_bins(target: Dict[str, Any], quota_scope: str) -> List[str]:
    quota_scope = normalize_quota_scope(quota_scope)
    bins = target.get("bins", {})
    out = []
    for k, b in bins.items():
        q = int(b.get("target_quota", 0))
        if q <= 0:
            continue
        status = b.get("status")
        if quota_scope == "all_nonzero":
            out.append(k)
        elif quota_scope == "handdraw_observed" and status == "handdraw_observed":
            out.append(k)
        elif quota_scope in ("unobserved_possible", "exploration_only") and status == "unobserved_possible":
            out.append(k)
        elif quota_scope == "hard_observed_plus_critical" and (status == "handdraw_observed" or k in CRITICAL_LONG_BINS):
            out.append(k)
    return out


def compute_hard_quota_status(target: Dict[str, Any], archive_counts: Counter, quota_scope: str, full_eval_counts: Optional[Counter] = None) -> Dict[str, Any]:
    full_eval_counts = full_eval_counts or Counter()
    original_quota_scope = quota_scope
    quota_scope = normalize_quota_scope(quota_scope)
    keys = scoped_target_bins(target, quota_scope)
    remaining = []
    required = 0
    confirmed = 0
    for k in keys:
        b = target["bins"].get(k, {})
        q = int(b.get("target_quota", 0))
        c = int(archive_counts.get(k, 0))
        required += q
        confirmed += min(c, q)
        rem = max(0, q - c)
        if rem > 0:
            remaining.append({
                "bin_key": k,
                "status": b.get("status"),
                "quota": q,
                "count": c,
                "remaining": rem,
                "fill_ratio": c / max(1, q),
                "full_evaluated_count": int(full_eval_counts.get(k, 0)),
                "is_critical": k in CRITICAL_LONG_BINS,
            })
    remaining.sort(key=lambda r: (r["fill_ratio"], r["full_evaluated_count"], -r["remaining"], r["bin_key"]))
    return {
        "quota_scope": quota_scope,
        "requested_quota_scope": original_quota_scope,
        "quota_sum_required": required,
        "confirmed_in_scope": confirmed,
        "hard_quota_fill_ratio": confirmed / max(1, required),
        "remaining_quota_total": sum(r["remaining"] for r in remaining),
        "remaining_bin_count": len(remaining),
        "hard_quota_met": len(remaining) == 0 and confirmed >= required,
        "quota_met": len(remaining) == 0 and confirmed >= required,
        "remaining_bins": remaining,
    }


def mazeforge_progress_postfix(
    target: Dict[str, Any],
    archive_counts: Counter,
    reports: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Compact tqdm postfix. Long fields (worst bins, closure detail) stay in progress_snapshot.json."""
    metrics = compute_until_target_metrics(target, archive_counts, int(reports.get("full_errors", 0)))
    full_eval = reports.get("full_bin_eval", Counter())
    hd = compute_hard_quota_status(target, archive_counts, "handdraw_observed", full_eval)
    ex = compute_hard_quota_status(target, archive_counts, "unobserved_possible", full_eval)
    all_scope = compute_hard_quota_status(target, archive_counts, "all_nonzero", full_eval)
    crit_full = all(
        int(archive_counts.get(k, 0)) >= int(target.get("bins", {}).get(k, {}).get("target_quota", 1))
        for k in CRITICAL_LONG_BINS
    )
    return {
        "hd": f"{hd['confirmed_in_scope']}/{hd['quota_sum_required']}",
        "ex": f"{ex['confirmed_in_scope']}/{ex['quota_sum_required']}",
        "all": f"{all_scope['confirmed_in_scope']}/{all_scope['quota_sum_required']}",
        "L1": f"{metrics['l1_distance_to_target_distribution']:.3f}",
        "crit": "ok" if crit_full else "open",
    }


def maybe_refresh_mazeforge_progress(
    bar: Any,
    seed_idx: int,
    target: Dict[str, Any],
    archive: Any,
    reports: Dict[str, Any],
    args: argparse.Namespace,
    state: Dict[str, int],
) -> None:
    if not hasattr(bar, "set_postfix"):
        return
    confirmed_all = sum(archive.counts.values())
    refresh_every = max(100, int(getattr(args, "progress_snapshot_interval", 100) or 100))
    should_refresh = (
        seed_idx == 0
        or confirmed_all != state.get("last_confirmed", -1)
        or seed_idx - state.get("last_seed_idx", -1) >= refresh_every
    )
    if not should_refresh:
        return
    state["last_confirmed"] = confirmed_all
    state["last_seed_idx"] = seed_idx
    bar.set_postfix(**mazeforge_progress_postfix(target, archive.counts, reports, args), refresh=False)


def compute_worst_bins(target: Dict[str, Any], archive_counts: Counter, full_eval_counts: Counter, quota_scope: str, top_k: int = 5) -> List[Dict[str, Any]]:
    rows = []
    for k in scoped_target_bins(target, quota_scope):
        b = target["bins"].get(k, {})
        q = int(b.get("target_quota", 0))
        c = int(archive_counts.get(k, 0))
        fe = int(full_eval_counts.get(k, 0))
        fill = c / max(1, q)
        fe_score = min(1.0, fe / max(1, q))
        score = 0.7 * fill + 0.3 * fe_score
        if c < q:
            rows.append({"bin_key": k, "status": b.get("status"), "quota": q, "count": c, "remaining": max(0, q-c), "fill_ratio": fill, "full_evaluated_count": fe, "score": score})
    rows.sort(key=lambda r: (r["score"], r["fill_ratio"], r["full_evaluated_count"], r["bin_key"]))
    return rows[:top_k]


def quality_threshold_status(args: argparse.Namespace, m: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "quality_threshold_met": criteria_met(args, m),
        "target_fill_ratio": m.get("target_fill_ratio"),
        "coverage_nonzero_target_bins": m.get("coverage_nonzero_target_bins"),
        "observed_handdraw_bin_fill_rate": m.get("observed_handdraw_bin_fill_rate"),
        "l1_distance_to_target_distribution": m.get("l1_distance_to_target_distribution"),
        "critical_bins_nonzero": m.get("critical_bins_nonzero"),
        "full_extraction_error_count": m.get("full_extraction_error_count"),
    }


def make_stop_checker(args: argparse.Namespace):
    def _checker(target: Dict[str, Any], archive: Any, reports: Dict[str, Any], seed_idx: int, elapsed: float) -> bool:
        counts = archive.counts
        full_eval_counts = reports.get("full_bin_eval", Counter())
        m = compute_until_target_metrics(target, counts, int(reports.get("full_errors", reports.get("full_extraction_error_count", 0))))
        qstat = compute_hard_quota_status(target, counts, getattr(args, "quota_scope", "all_nonzero"), full_eval_counts)
        qthr = quality_threshold_status(args, m)
        policy = getattr(args, "stop_policy", "hard_quota")
        quality_met = bool(qthr["quality_threshold_met"])
        hard_met = bool(qstat["hard_quota_met"])
        stop_reason = None
        if policy == "quality_threshold" and quality_met:
            stop_reason = "quality_threshold_met"
        elif policy == "hard_quota" and hard_met:
            scope = normalize_quota_scope(getattr(args, "quota_scope", "handdraw_observed"))
            if scope == "handdraw_observed":
                stop_reason = "handdraw_quota_met"
            elif scope == "all_nonzero":
                stop_reason = "all_nonzero_quota_met"
            elif scope == "hard_observed_plus_critical":
                stop_reason = "hard_observed_plus_critical_quota_met"
            else:
                stop_reason = f"{scope}_quota_met"
        elif policy == "both" and quality_met and hard_met:
            stop_reason = "quality_and_quota_met"
        elif args.max_seeds and seed_idx >= int(args.max_seeds):
            stop_reason = "max_seeds_reached"
        elif args.max_wall_clock_seconds and elapsed >= float(args.max_wall_clock_seconds):
            stop_reason = "max_wall_clock_reached"
        elif args.max_full_evaluations and int(reports.get("full_evaluated", 0)) >= int(args.max_full_evaluations):
            stop_reason = "max_full_evaluations_reached"
        if seed_idx and getattr(args, "_out_dir", None) and getattr(args, "progress_snapshot_interval", 0) and seed_idx % int(args.progress_snapshot_interval) == 0:
            snap = {"seed_count": seed_idx, "confirmed_total": sum(counts.values()), "quota_sum": target.get("quota_sum"), "hard_quota_status": qstat, "quality_threshold_status": qthr, "critical_bins": m.get("critical_bin_counts"), "worst5_bins": compute_worst_bins(target, counts, full_eval_counts, getattr(args, "quota_scope", "all_nonzero"), 5), "elapsed_seconds": elapsed}
            write_json(Path(args._out_dir) / "progress_snapshot.json", snap)
        if stop_reason:
            args._until_target_stop_state = {"stop_reason": stop_reason, "seed_count_to_target": seed_idx, "elapsed_seconds_to_target": elapsed, **m, "quality_threshold_status": qthr, "hard_quota_status": qstat, "worst5_bins": compute_worst_bins(target, counts, full_eval_counts, getattr(args, "quota_scope", "all_nonzero"), 5)}
            return True
        return False
    return _checker

def enrich_until_target_reports(out_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    target = load_json(out_dir / "target_distribution_used.json")
    confirmed = read_jsonl(out_dir / "confirmed_samples.jsonl")
    full_eval = read_jsonl(out_dir / "full_evaluated_samples.jsonl")
    counts = Counter()
    for r in confirmed:
        key = r.get("full_bin", {}).get("bin_key") or r.get("full_bin_key")
        if key:
            counts[key] += 1
    m = compute_until_target_metrics(target, counts, 0)
    state = getattr(args, "_until_target_stop_state", None)
    if state is None:
        if getattr(args, "max_seeds", 0):
            state = {"stop_reason": "max_seeds_reached", "seed_count_to_target": int(args.max_seeds), **m}
        elif getattr(args, "max_full_evaluations", 0):
            state = {"stop_reason": "max_full_evaluations_reached", **m}
        else:
            state = {"stop_reason": "run_completed_without_explicit_stop", **m}
    report = {
        **state,
        "candidate_count_to_target": None,
        "hard_quality_pass_count_to_target": None,
        "full_evaluated_count_to_target": len(full_eval),
        "confirmed_count_to_target": len(confirmed),
        "critical_bin_full_eval_at_stop": {k: sum(1 for r in full_eval if r.get("full_bin_key") == k) for k in CRITICAL_LONG_BINS},
        "critical_bin_fill_at_stop": {k: counts.get(k, 0) / max(1, int(target.get("bins", {}).get(k, {}).get("target_quota", 0))) for k in CRITICAL_LONG_BINS},
        "thresholds": {
            "target_fill_ratio_threshold": args.target_fill_ratio_threshold,
            "coverage_nonzero_target_bins_threshold": args.coverage_nonzero_target_bins_threshold,
            "observed_handdraw_bin_fill_rate_threshold": args.observed_handdraw_bin_fill_rate_threshold,
            "l1_distance_threshold": args.l1_distance_threshold,
            "require_critical_bins_nonzero": args.require_critical_bins_nonzero,
            "require_full_extraction_error_zero": args.require_full_extraction_error_zero,
        },
    }
    full_counts = Counter()
    for r in full_eval:
        key = r.get("full_bin_key") or (r.get("full_bin") or {}).get("bin_key")
        if key: full_counts[key] += 1
    selected_scope = getattr(args, "quota_scope", "handdraw_observed")
    hq = compute_hard_quota_status(target, counts, selected_scope, full_counts)
    three = compute_three_quota_statuses(target, counts, full_counts)
    handdraw_status = three["handdraw_quota_status"]
    exploration_status = three["exploration_quota_status"]
    all_status = three["all_nonzero_quota_status"]
    qthr = quality_threshold_status(args, m)
    report["hard_quota_status"] = hq
    report["quality_threshold_status"] = qthr
    report["handdraw_quota_status"] = handdraw_status
    report["exploration_quota_status"] = exploration_status
    report["all_nonzero_quota_status"] = all_status
    report["worst5_bins"] = compute_worst_bins(target, counts, full_counts, selected_scope, 5)
    interpretation = "selected quota scope still has remaining bins"
    if handdraw_status.get("quota_met") and not exploration_status.get("quota_met"):
        interpretation = "handdraw distribution completed; remaining bins are exploration-only"
    if normalize_quota_scope(selected_scope) == "all_nonzero" and handdraw_status.get("quota_met") and not all_status.get("quota_met"):
        interpretation = "selected scope is exploration stress test; handdraw closure already achieved"
    write_json(out_dir / "handdraw_quota_status.json", handdraw_status)
    write_json(out_dir / "exploration_quota_status.json", exploration_status)
    write_json(out_dir / "all_nonzero_quota_status.json", all_status)
    write_json(out_dir / "hard_quota_status.json", hq)
    write_json(out_dir / "hard_quota_blocking_report.json", {
        "selected_quota_scope": selected_scope,
        "selected_scope_blocking_bins": hq.get("remaining_bins", []),
        "handdraw_remaining_bins": handdraw_status.get("remaining_bins", []),
        "exploration_remaining_bins": exploration_status.get("remaining_bins", []),
        "all_nonzero_remaining_bins": all_status.get("remaining_bins", []),
        "interpretation": interpretation,
        "blocking_reasons": ["remaining_bins_not_filled"] if not hq.get("quota_met") else [],
    })
    def _difficulty_flags(row: Dict[str, Any]) -> List[str]:
        cycle, bfs, choice, endpoint = bin_parts(row.get("bin_key", ""))
        flags = []
        if choice == "low_choice" and endpoint == "high_endpoint": flags.append("low_choice_high_endpoint_conflict")
        if row.get("quota") == 1 and row.get("status") == "unobserved_possible": flags.append("rare_explore_quota")
        if row.get("full_evaluated_count", 0) == 0: flags.append("never_full_evaluated")
        if row.get("full_evaluated_count", 0) > 0 and row.get("count", 0) == 0: flags.append("generated_but_not_accepted")
        if row.get("status") != "handdraw_observed": flags.append("not_handdraw_observed")
        return flags
    rare_rows = []
    for row in all_status.get("remaining_bins", []):
        r = dict(row); r["suspected_difficulty"] = _difficulty_flags(r); rare_rows.append(r)
    write_json(out_dir / "rare_bin_difficulty_report.json", {"remaining_bins": rare_rows})
    write_json(out_dir / "until_target_stop_report.json", report)
    return report


def shortest_path(grid: Grid, start: Cell, goal: Cell) -> List[Cell]:
    n = len(grid)
    q = deque([start])
    parent: Dict[Cell, Optional[Cell]] = {start: None}
    while q:
        u = q.popleft()
        if u == goal:
            break
        for v in neighbors(u, n):
            if v not in parent and grid[v[0]][v[1]] == 0:
                parent[v] = u
                q.append(v)
    if goal not in parent:
        return []
    path=[]; cur=goal
    while cur is not None:
        path.append(cur); cur=parent[cur]
    return list(reversed(path))


def path_shape_stats(grid: Grid, start: Cell, goal: Cell) -> Dict[str, Any]:
    path = shortest_path(grid, start, goal)
    if len(path) < 2:
        return {"path_len": 0, "turn_count": 0, "turn_density": 0.0, "shape_signature": "none"}
    dirs=[]
    for a,b in zip(path,path[1:]):
        dirs.append((b[0]-a[0], b[1]-a[1]))
    turns=sum(1 for i in range(1,len(dirs)) if dirs[i]!=dirs[i-1])
    rs=[x[0] for x in path]; cs=[x[1] for x in path]
    area=(max(rs)-min(rs)+1)*(max(cs)-min(cs)+1)
    runs=[]; cur=dirs[0]; l=1
    for d in dirs[1:]:
        if d==cur: l+=1
        else: runs.append(l); cur=d; l=1
    runs.append(l)
    sig=f"L{len(path)-1}_T{turns}_R{'-'.join(map(str,runs[:8]))}"
    return {"path_len": len(path)-1, "turn_count": turns, "turn_density": turns/max(1,len(path)-1), "straight_run_lengths": runs, "path_bounding_box_area": area, "path_compactness": (len(path)-1)/max(1,area), "shape_signature": sig}


def topology_diversity(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    grid_hashes=Counter(); sg_hashes=Counter(); origins=Counter(); shapes=Counter(); motifs=Counter(); degree_sigs=Counter(); crit={k: [] for k in CRITICAL_LONG_BINS}
    for r in samples:
        grid=r.get("grid") or []
        if grid and isinstance(grid[0], str):
            g=[[0 if ch in ".SG" else 1 for ch in row] for row in grid]
        else:
            g=grid
        start=tuple(r.get("start", [0,0])); goal=tuple(r.get("goal", [0,0]))
        gh=hashlib.sha1(json.dumps(g,separators=(",",":")).encode()).hexdigest()[:16]
        sgh=hashlib.sha1(json.dumps({"g":g,"s":start,"e":goal},default=list,separators=(",",":")).encode()).hexdigest()[:16]
        grid_hashes[gh]+=1; sg_hashes[sgh]+=1; origins[str(r.get("origin_seed_id","unknown"))]+=1
        ps=path_shape_stats(g, start, goal); shapes[ps["shape_signature"]]+=1
        m=r.get("metrics", {})
        bin_key=(r.get("full_bin") or {}).get("bin_key") or r.get("full_bin_key") or "unknown"
        motif=f"{bin_key}|V{m.get('rect_room_macro_vertex_count','?')}|E{m.get('rect_room_macro_edge_count','?')}|R{m.get('rect_room_count','?')}"
        motifs[motif]+=1
        deg=[0,0,0,0,0]
        n=len(g)
        for i in range(n):
            for j in range(len(g[i])):
                if g[i][j]==0:
                    d=sum(1 for nb in neighbors((i,j),n) if g[nb[0]][nb[1]]==0)
                    deg[min(4,d)]+=1
        ds=f"d1={deg[1]}|d2={deg[2]}|d3={deg[3]}|d4={deg[4]}"; degree_sigs[ds]+=1
        if bin_key in crit: crit[bin_key].append({"origin":r.get("origin_seed_id"), "shape":ps["shape_signature"], "motif":motif, "grid_hash":gh})
    def entropy(c: Counter) -> float:
        total=sum(c.values())
        return -sum((v/total)*math.log(v/total+1e-12) for v in c.values()) if total else 0.0
    crit_report={}
    for k, rows in crit.items():
        oc=Counter(x["origin"] for x in rows); sc=Counter(x["shape"] for x in rows); mc=Counter(x["motif"] for x in rows)
        crit_report[k]={"confirmed_count":len(rows),"unique_grid_hash_count":len(set(x['grid_hash'] for x in rows)),"unique_origin_seed_count":len(oc),"origin_entropy":entropy(oc),"main_path_shape_entropy":entropy(sc),"macro_motif_entropy":entropy(mc),"max_origin_share":(max(oc.values())/max(1,len(rows)) if oc else 0),"possible_topology_style_concentration": bool(rows and (entropy(sc)<0.5 or entropy(mc)<0.5)),"top_shape_signatures":sc.most_common(10),"top_macro_motifs":mc.most_common(10)}
    return {"total_confirmed":len(samples),"unique_grid_hash_count":len(grid_hashes),"unique_grid_start_goal_hash_count":len(sg_hashes),"duplicate_grid_count":sum(v-1 for v in grid_hashes.values() if v>1),"duplicate_grid_start_goal_count":sum(v-1 for v in sg_hashes.values() if v>1),"unique_origin_seed_count":len(origins),"origin_seed_entropy":entropy(origins),"max_origin_seed_share":(max(origins.values())/max(1,len(samples)) if origins else 0),"main_path_shape_signature_entropy":entropy(shapes),"top_main_path_shape_signatures":shapes.most_common(20),"degree_signature_entropy":entropy(degree_sigs),"top_degree_signatures":degree_sigs.most_common(20),"macro_motif_signature_entropy":entropy(motifs),"top_macro_motif_signatures":motifs.most_common(20),"critical_bin_topology_diversity":crit_report,"possible_topology_style_concentration_count":sum(1 for v in crit_report.values() if v.get('possible_topology_style_concentration'))}



def render_grid_png(path: Path, grid: Grid, start: Cell, goal: Cell, title: str = "", cell_size: int = 32) -> bool:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except Exception:
        return False
    ensure_dir(path.parent)
    n = len(grid)
    data = []
    for r in range(n):
        row=[]
        for c in range(len(grid[r])):
            if (r,c)==start:
                row.append(2)
            elif (r,c)==goal:
                row.append(3)
            else:
                row.append(0 if grid[r][c]==0 else 1)
        data.append(row)
    fig_w=max(2.0,n*cell_size/96); fig_h=max(2.0,n*cell_size/96)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=96)
    cmap = ListedColormap(["white", "black", "dodgerblue", "orange"])
    ax.imshow(data, cmap=cmap, vmin=0, vmax=3)
    ax.set_xticks([x-0.5 for x in range(1,n)], minor=True)
    ax.set_yticks([y-0.5 for y in range(1,n)], minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7)
    ax.text(start[1], start[0], "S", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    ax.text(goal[1], goal[0], "G", ha="center", va="center", color="black", fontsize=10, fontweight="bold")
    plt.tight_layout(pad=0.2)
    fig.savefig(path)
    plt.close(fig)
    return True


def write_per_bin_visualizations(out_dir: Path, records: List[Dict[str, Any]], target: Dict[str, Any], per_bin_limit: int = 5, cell_size: int = 32) -> Dict[str, Any]:
    index: Dict[str, Any] = {}
    by_bin: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        key = (r.get("full_bin") or {}).get("bin_key") or r.get("full_bin_key")
        if key:
            by_bin[key].append(r)
    root = out_dir / "sample_visualizations" / "by_bin"
    saved_all=[]; saved_critical=[]; saved_sparse=[]
    for key, b in target.get("bins", {}).items():
        rows = by_bin.get(key, [])
        q = int(b.get("target_quota", 0))
        limit = min(len(rows), max(10 if key in CRITICAL_LONG_BINS else per_bin_limit, 1 if q > 0 else 0))
        paths=[]; sample_ids=[]
        for i, r in enumerate(rows[:limit]):
            grid = r.get("grid") or []
            if grid and isinstance(grid[0], str):
                grid = [[0 if ch in ".SG" else 1 for ch in line] for line in grid]
            start = tuple(r.get("start", [0,0])); goal = tuple(r.get("goal", [0,0]))
            m = r.get("metrics", {})
            title = f"{key}\\n{r.get('sample_id') or r.get('maze_id')} bfs={m.get('bfs_len')} cr={m.get('rect_room_compressed_cycle_rank')} ch={m.get('rect_room_macro_choice_count')} ep={m.get('rect_room_macro_endpoint_count')}"
            rel = Path("sample_visualizations") / "by_bin" / safe_name(key) / f"accepted_{i:03d}.png"
            ok = render_grid_png(out_dir / rel, grid, start, goal, title, cell_size)
            if ok:
                paths.append(str(rel)); sample_ids.append(r.get("sample_id") or r.get("maze_id")); saved_all.append(out_dir/rel)
                if key in CRITICAL_LONG_BINS: saved_critical.append(out_dir/rel)
                if q > 0 and len(rows) < q: saved_sparse.append(out_dir/rel)
        index[key] = {"quota": q, "confirmed_count": len(rows), "saved_png_count": len(paths), "paths": paths, "sample_ids": sample_ids, "reason": None if paths or not rows else "visualization_backend_unavailable" if rows else "no_confirmed_sample"}
    write_json(out_dir / "per_bin_visualization_index.json", index)
    ensure_dir(out_dir / "sample_visualizations" / "montage")
    write_json(out_dir / "sample_visualizations" / "montage" / "montage_manifest.json", {"all_bins_montage": "not_implemented_yet", "critical_bins_montage": "not_implemented_yet", "sparse_bins_montage": "not_implemented_yet", "note": "per-bin PNGs are generated; montage composition is deferred."})
    return index


def write_quality_reports(run_dirs: List[Path], handdraw_jsonl: Path, out_dir: Path) -> None:
    ensure_dir(out_dir)
    if not handdraw_jsonl.exists():
        alt = Path.cwd() / handdraw_jsonl.name
        if alt.exists():
            handdraw_jsonl = alt
    generated=[]
    for rd in run_dirs:
        generated.extend(read_jsonl(rd / "confirmed_samples.jsonl"))
    config = F.default_config()
    target_path = Path(DEFAULT_TARGET_DISTRIBUTION_REL)
    try:
        target = F.load_target_distribution(target_path, None, config)
    except Exception:
        target = {"bins": {}}
    archive = F.TargetDistributionArchive(target if target.get("bins") else {"bins": {}}, config.get("diversity", {}), config) if target.get("bins") else None
    quantizer = F.Quantizer(config)
    handdraw=[]; errors=[]
    for row in read_jsonl(handdraw_jsonl):
        maze_id = row.get("maze_id") or row.get("id") or f"handdraw_{len(handdraw)+len(errors):04d}"
        try:
            grid=row.get("grid") or row.get("ascii_grid") or []
            if grid and isinstance(grid[0], str):
                g=[[0 if ch in ".SG" else 1 for ch in line] for line in grid]
            else:
                g=grid
            start=tuple(row.get("start")); goal=tuple(row.get("goal"))
            cand=F.MazeCandidate(grid=g, start=start, goal=goal, origin_generator="handdraw", origin_seed_id=str(maze_id), transform_history=[], candidate_id=str(maze_id))
            cheap=quantizer.evaluate(cand)
            fm, _actual, _rt, err=F.full_metrics_for_candidate(cand, cheap, config)
            if err:
                raise RuntimeError(err)
            bin_info=archive.assign_bin(fm) if archive is not None else {"bin_key": row.get("full_bin_key", "unknown")}
            handdraw.append({"grid":g,"start":list(start),"goal":list(goal),"full_bin":bin_info,"metrics":fm,"maze_id":maze_id})
        except Exception as exc:
            errors.append({"maze_id": maze_id, "error": str(exc), "sample": row})
    for e in errors:
        append_jsonl(out_dir / "handdraw_extraction_errors.jsonl", e)
    gen_div=topology_diversity(generated)
    hand_div=topology_diversity(handdraw)
    def dist(rows, getter):
        c=Counter(getter(r) for r in rows); return dict(c)
    gen_bins=dist(generated, lambda r:(r.get("full_bin") or {}).get("bin_key") or r.get("full_bin_key") or "unknown")
    hand_bins=dist(handdraw, lambda r:(r.get("full_bin") or {}).get("bin_key") or r.get("full_bin_key") or "unknown")
    keys=set(gen_bins)|set(hand_bins)
    gt=sum(gen_bins.values()) or 1; ht=sum(hand_bins.values()) or 1
    full_l1=sum(abs(gen_bins.get(k,0)/gt-hand_bins.get(k,0)/ht) for k in keys)
    validation_valid = bool(handdraw) and "unknown" not in hand_bins and not errors
    report={"validation_valid": validation_valid,"generated_runs":[str(x) for x in run_dirs],"handdraw_samples":len(handdraw),"handdraw_error_count":len(errors),"generated_samples":len(generated),"full_4d_l1":full_l1,"generated_bin_counts":gen_bins,"handdraw_bin_counts":hand_bins,"generated_topology_diversity":gen_div,"handdraw_topology_diversity":hand_div,"not_implemented_yet":["js_divergence_for_all_secondary_histograms","branch_component_exact_matching"]}
    if not validation_valid:
        report["error"]="handdraw_full_extraction_failed_or_unknown_bins_present"
    write_json(out_dir / "generated_vs_handdraw_quality_report.json", report)
    write_json(out_dir / "topology_diversity_report.json", gen_div)
    write_json(out_dir / "critical_bin_topology_diversity.json", gen_div.get("critical_bin_topology_diversity", {}))
    write_json(out_dir / "handdraw_full_extraction_report.json", {"n_handdraw_input": len(handdraw)+len(errors), "n_extracted": len(handdraw), "n_errors": len(errors), "handdraw_bin_counts": hand_bins, "validation_valid": validation_valid})
    write_json(out_dir / "quality_validation_summary.json", {"validation_valid": validation_valid, "full_4d_l1":full_l1,"generated_samples":len(generated),"handdraw_samples":len(handdraw), "handdraw_error_count": len(errors)})
    with (out_dir / "visual_ascii_review_pack.jsonl").open("w",encoding="utf-8") as f:
        for r in generated[:500]:
            g=r.get("ascii_grid") or grid_to_ascii(r.get("grid",[]), tuple(r.get("start",[0,0])), tuple(r.get("goal",[0,0])))
            f.write(json.dumps({"sample_id":r.get("sample_id") or r.get("maze_id"),"bin_key":(r.get("full_bin") or {}).get("bin_key"),"route_id":r.get("route"),"origin_seed_id":r.get("origin_seed_id"),"generator":r.get("origin_generator"),"accepted_as_commit":bool(r.get("from_rbr")),"ascii_grid":g,"metrics":r.get("metrics"),"review_tags":["critical_bin"] if ((r.get("full_bin") or {}).get("bin_key") in CRITICAL_LONG_BINS) else []},ensure_ascii=False)+"\n")

def run_generate_until_target(args: argparse.Namespace) -> Path:
    args.mode = "generate"
    args.n_seeds = int(args.max_seeds or 0)
    args._stop_checker = make_stop_checker(args)
    if normalize_quota_scope(getattr(args, "quota_scope", "handdraw_observed")) == "all_nonzero":
        print("[INFO] quota_scope=all_nonzero is an exploration stress test, not the default handdraw-distribution closure criterion.")
    out = run_generate(args)
    confirmed = read_jsonl(out / "confirmed_samples.jsonl")
    full_eval = read_jsonl(out / "full_evaluated_samples.jsonl")
    target = load_json(out / "target_distribution_used.json")

    # Generate per-bin visualizations before CSV so CSV rows can link to any available PNG.
    vis_index = None
    if getattr(args, "save_visualizations", False):
        vis_index = write_per_bin_visualizations(
            out,
            confirmed,
            target,
            getattr(args, "per_bin_visualization_limit", 5),
            getattr(args, "visualization_cell_size", 32),
        )

    # Export complete sample CSV by default. This is intentionally based on confirmed
    # full-bin samples, not fast/cheap metrics.
    csv_report = None
    if getattr(args, "export_samples_csv", True):
        csv_report = export_samples_csv(out, confirmed, target, vis_index, getattr(args, "samples_csv_name", "samples.csv"))

    # Stop/quota reports.
    stop_report = enrich_until_target_reports(out, args)

    # Topology diversity and closure status. Closure intentionally depends on
    # handdraw quota, critical bins, L1, full-extraction errors, structural count,
    # and topology concentration. Exploration-only quota is *not* required.
    div = topology_diversity(confirmed)
    write_json(out / "topology_diversity_report.json", div)
    write_json(out / "critical_bin_topology_diversity.json", div.get("critical_bin_topology_diversity", {}))
    counts = Counter()
    for r in confirmed:
        key = (r.get("full_bin") or {}).get("bin_key") or r.get("full_bin_key")
        if key:
            counts[key] += 1
    full_counts = Counter()
    for r in full_eval:
        key = r.get("full_bin_key") or (r.get("full_bin") or {}).get("bin_key")
        if key:
            full_counts[key] += 1
    synth_metrics = compute_until_target_metrics(target, counts, 0)
    closure = compute_closure_status(
        target,
        counts,
        synth_metrics,
        full_counts,
        int(stop_report.get("full_extraction_error_count", synth_metrics.get("full_extraction_error_count", 0)) or 0),
        div,
        getattr(args, "closure_l1_threshold", 0.05),
    )
    write_json(out / "closure_status_report.json", closure)
    stop_report["closure_status"] = closure
    stop_report["ready_for_3_0_4_closure"] = closure.get("ready_for_3_0_4_closure")
    write_json(out / "until_target_stop_report.json", stop_report)

    # Required compatibility aliases / clean names.
    for src, dst in [("tree_branch_injection_report.json", "pure_tree_branch_injection_report.json"), ("controlled_loop_overlay_report.json", "stage_controlled_loop_overlay_report.json")]:
        sp = out / src
        if sp.exists():
            write_json(out / dst, load_json(sp))

    print("\n[TDA-QD MazeForge 3.0.4 Final]")
    print("\n=== Quality Threshold Status ===")
    print(json.dumps(stop_report.get("quality_threshold_status", {}), ensure_ascii=False, indent=2))
    print("\n=== Handdraw Quota Status ===")
    hds = stop_report.get("handdraw_quota_status", {})
    for k in ["quota_sum_required", "confirmed_in_scope", "remaining_quota_total", "remaining_bin_count", "quota_met"]:
        print(f"{k:36s} {hds.get(k)}")
    print("\n=== Exploration Quota Status ===")
    exs = stop_report.get("exploration_quota_status", {})
    for k in ["quota_sum_required", "confirmed_in_scope", "remaining_quota_total", "remaining_bin_count", "quota_met"]:
        print(f"{k:36s} {exs.get(k)}")
    print("\n=== All Nonzero Quota Status ===")
    ans = stop_report.get("all_nonzero_quota_status", {})
    for k in ["quota_sum_required", "confirmed_in_scope", "remaining_quota_total", "remaining_bin_count", "quota_met"]:
        print(f"{k:36s} {ans.get(k)}")
    print("\n=== Closure Status ===")
    for k in ["ready_for_3_0_4_closure", "closure_l1_threshold", "observed_handdraw_bin_fill_rate", "critical_long_bins_full", "topology_style_concentration_count", "full_extraction_error_count", "structural_confirmed_count"]:
        print(f"{k:40s} {closure.get(k)}")
    if closure.get("info"):
        for msg in closure["info"]:
            print(f"[INFO] {msg}")
    if closure.get("ready_for_3_0_4_closure"):
        print("[PASS] TDA-QD MazeForge 3.0.4 is ready for closure.")
    else:
        print(f"[INFO] Closure unmet conditions: {closure.get('unmet_conditions')}")
    print("\n=== Final Stop Report ===")
    for k in ["stop_reason", "seed_count_to_target", "full_evaluated_count_to_target", "confirmed_count_to_target", "elapsed_seconds_to_target", "target_fill_ratio", "coverage_nonzero_target_bins", "observed_handdraw_bin_fill_rate", "l1_distance_to_target_distribution"]:
        print(f"{k:40s} {stop_report.get(k)}")
    print("\n=== CSV Export ===")
    if csv_report:
        for k in ["csv_path", "n_rows", "confirmed_samples_count", "row_count_matches_confirmed", "n_handdraw_observed_rows", "n_exploration_rows", "n_critical_long_rows", "missing_visualization_path_count"]:
            print(f"{k:40s} {csv_report.get(k)}")
    else:
        print("CSV export disabled")
    print("\n=== Diversity Quick Check ===")
    for k in ["unique_grid_hash_count", "unique_origin_seed_count", "max_origin_seed_share", "main_path_shape_signature_entropy", "macro_motif_signature_entropy", "possible_topology_style_concentration_count"]:
        print(f"{k:40s} {div.get(k)}")
    return out


def run_multi_seed_until_target(args: argparse.Namespace) -> Path:
    seeds=args.seeds or [42,43,44,45,46]
    root=Path(args.output_root or DEFAULT_OUTPUT_ROOT_REL) / (args.run_name or "tda_qd_mazeforge_multiseed_until_target")
    ensure_dir(root / "per_seed_runs")
    rows=[]
    for sd in seeds:
        sub=copy.deepcopy(args)
        sub.mode="generate_until_target"; sub.seed=sd; sub.run_name=f"seed_{sd}"; sub.output_root=str(root / "per_seed_runs")
        out=run_generate_until_target(sub)
        rep=load_json(out / "until_target_stop_report.json")
        rows.append({"seed":sd, **rep})
    vals=[r.get("seed_count_to_target") for r in rows if isinstance(r.get("seed_count_to_target"), int)]
    def pct(xs,p):
        if not xs: return None
        xs=sorted(xs); return xs[min(len(xs)-1,int(round((len(xs)-1)*p)))]
    summary={"per_seed":rows,"mean_seed_count_to_target":statistics.mean(vals) if vals else None,"p50_seed_count_to_target":pct(vals,0.5),"p95_seed_count_to_target":pct(vals,0.95),"max_seed_count_to_target":max(vals) if vals else None,"min_seed_count_to_target":min(vals) if vals else None,"pass_rate":sum(1 for r in rows if r.get('stop_reason')=='target_criteria_met')/max(1,len(rows)),"stability_pass": False}
    summary["stability_pass"] = summary["pass_rate"] >= 0.8 and all((r.get("l1_distance_to_target_distribution",999)<=0.30) for r in rows if r.get('stop_reason')=='target_criteria_met')
    write_json(root / "multi_seed_until_target_summary.json", summary)
    write_json(root / "multi_seed_stability_report.json", summary)
    print("\n=== Multi Seed Stability ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:4000])
    return root


def main() -> None:
    args = parse_args()
    if args.mode == "generate_until_target":
        run_generate_until_target(args)
    elif args.mode == "multi_seed_until_target":
        run_multi_seed_until_target(args)
    elif args.mode == "validate_quality":
        run_dirs=[Path(p) for p in (args.input_run_dirs or [])]
        out=Path(args.output_dir or (Path(args.output_root or DEFAULT_OUTPUT_ROOT_REL) / (args.run_name or "tda_qd_mazeforge_quality_validation")))
        write_quality_reports(run_dirs, Path(args.handdraw_jsonl), out)
        print("\n[TDA-QD MazeForge 3.0.4]\n=== Quality Validation ===")
        print(f"generated_runs                      {len(run_dirs)}")
        print(f"output_dir                          {out}")
    else:
        raise SystemExit(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
