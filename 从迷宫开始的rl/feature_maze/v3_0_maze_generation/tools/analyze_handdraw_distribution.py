#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hand-drawn maze distribution analyzer.

Standalone tool for:
  handdraw JSONL / JSON / optional ASCII TXT
  -> full rect-room macro graph extraction
  -> full-bin distribution report
  -> 81-bucket feasibility/conflict classification
  -> handdraw target distribution JSON builder for later generators

This tool intentionally does NOT implement generation, archive acceptance, fast
metrics, fast selector, pending/replay, audit-after-generate, or legacy archive
logic. It is a compact distribution profiler for hand-drawn samples.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

Cell = Tuple[int, int]
GridChars = List[str]
ACTIONS: List[Cell] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PASSABLE = {".", "S", "G"}


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def stat(values: Iterable[Optional[float]]) -> Dict[str, Any]:
    xs = [float(x) for x in values if x is not None]
    if not xs:
        return {"n": 0, "min": None, "p50": None, "mean": None, "max": None}
    xs.sort()
    return {
        "n": len(xs),
        "min": xs[0],
        "p50": xs[len(xs) // 2],
        "mean": sum(xs) / len(xs),
        "max": xs[-1],
    }


def in_grid(grid: GridChars, r: int, c: int) -> bool:
    return 0 <= r < len(grid) and 0 <= c < len(grid[r])


def neighbors(grid: GridChars, cell: Cell) -> List[Cell]:
    r, c = cell
    out = []
    for dr, dc in ACTIONS:
        rr, cc = r + dr, c + dc
        if in_grid(grid, rr, cc):
            out.append((rr, cc))
    return out


def is_free(grid: GridChars, cell: Cell) -> bool:
    r, c = cell
    return in_grid(grid, r, c) and grid[r][c] in PASSABLE


def compute_bfs_len(grid: GridChars, start: Cell, goal: Cell) -> Optional[int]:
    dist, _ = bfs_dist(grid, start)
    return dist.get(goal)


def bfs_dist(grid: GridChars, start: Cell) -> Tuple[Dict[Cell, int], Dict[Cell, Cell]]:
    dist: Dict[Cell, int] = {}
    parent: Dict[Cell, Cell] = {}
    if not is_free(grid, start):
        return dist, parent
    q: deque[Cell] = deque([start])
    dist[start] = 0
    while q:
        cur = q.popleft()
        for nb in neighbors(grid, cur):
            if is_free(grid, nb) and nb not in dist:
                dist[nb] = dist[cur] + 1
                parent[nb] = cur
                q.append(nb)
    return dist, parent


def find_start_goal(grid: GridChars) -> Tuple[Optional[Cell], Optional[Cell]]:
    start = goal = None
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == "S":
                start = (r, c)
            elif ch == "G":
                goal = (r, c)
    return start, goal


def normalize_grid(raw_grid: Any) -> GridChars:
    if not isinstance(raw_grid, list) or not raw_grid:
        raise ValueError("grid must be a non-empty list")
    rows: GridChars = []
    for row in raw_grid:
        if isinstance(row, str):
            s = row.rstrip("\n")
        elif isinstance(row, list):
            # Accept ['#','.',...] or [0,1,...] as a convenience.
            chars = []
            for x in row:
                if x in (0, ".", "S", "G"):
                    chars.append("." if x == 0 else str(x))
                elif x in (1, "#"):
                    chars.append("#")
                else:
                    raise ValueError(f"unsupported grid cell: {x!r}")
            s = "".join(chars)
        else:
            raise ValueError("each grid row must be a string or list")
        if any(ch not in "#.SG" for ch in s):
            bad = sorted({ch for ch in s if ch not in "#.SG"})
            raise ValueError(f"grid contains invalid symbols: {bad}")
        rows.append(s)
    width = len(rows[0])
    if width == 0 or any(len(r) != width for r in rows):
        raise ValueError("grid row widths are inconsistent")
    return rows


def validate_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    maze_id = str(sample.get("maze_id", sample.get("id", "unknown")))
    grid = normalize_grid(sample.get("grid"))
    rows, cols = len(grid), len(grid[0])
    start = tuple(sample["start"]) if sample.get("start") is not None else None
    goal = tuple(sample["goal"]) if sample.get("goal") is not None else None
    parsed_start, parsed_goal = find_start_goal(grid)
    if start is None:
        start = parsed_start
    if goal is None:
        goal = parsed_goal
    if start is None or goal is None:
        raise ValueError("missing start or goal")
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    if not in_grid(grid, *start) or not in_grid(grid, *goal):
        raise ValueError("start or goal is out of bounds")
    # Explicit start/goal coordinates are authoritative. Make them passable in a local copy.
    grid_list = [list(r) for r in grid]
    sr, sc = start
    gr, gc = goal
    if grid_list[sr][sc] == "#":
        grid_list[sr][sc] = "S"
    elif grid_list[sr][sc] == ".":
        grid_list[sr][sc] = "S"
    if grid_list[gr][gc] == "#":
        grid_list[gr][gc] = "G"
    elif grid_list[gr][gc] == ".":
        grid_list[gr][gc] = "G"
    if start == goal:
        raise ValueError("start and goal overlap")
    grid = ["".join(r) for r in grid_list]
    bfs_len = compute_bfs_len(grid, start, goal)
    free_total = sum(ch in PASSABLE for row in grid for ch in row)
    dist, _ = bfs_dist(grid, start)
    return {
        "maze_id": maze_id,
        "rows": rows,
        "cols": cols,
        "grid": grid,
        "start": list(start),
        "goal": list(goal),
        "connected": bfs_len is not None,
        "bfs_len": bfs_len,
        "free_cell_count": free_total,
        "reachable_free_count": len(dist),
        "reachable_free_ratio": (len(dist) / free_total) if free_total else 0.0,
    }


# -----------------------------------------------------------------------------
# Input readers
# -----------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"JSONL parse error at line {lineno}: {e}") from e
    return samples


def load_json(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("samples"), list):
        return obj["samples"]
    raise ValueError("JSON input must be a list or {'samples': [...]} object")


def parse_tuple_text(s: str) -> Optional[List[int]]:
    m = re.search(r"\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?", s)
    if not m:
        return None
    return [int(m.group(1)), int(m.group(2))]


def load_txt(path: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    current_header = None
    current_grid: List[str] = []
    header_re = re.compile(r"maze_id=([^\s]+).*?start=\(([^)]*)\).*?goal=\(([^)]*)\)")

    def flush() -> None:
        nonlocal current_header, current_grid
        if not current_grid:
            return
        maze_id = f"txt_{len(samples)+1:06d}"
        start = goal = None
        if current_header:
            m = header_re.search(current_header)
            if m:
                maze_id = m.group(1)
                start = parse_tuple_text(m.group(2))
                goal = parse_tuple_text(m.group(3))
        parsed_start, parsed_goal = find_start_goal(current_grid)
        samples.append({
            "maze_id": maze_id,
            "grid": current_grid[:],
            "start": start if start is not None else (list(parsed_start) if parsed_start else None),
            "goal": goal if goal is not None else (list(parsed_goal) if parsed_goal else None),
        })
        current_grid = []
        current_header = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        if line.startswith("# ---"):
            flush()
            current_header = line
        elif set(line) <= set("#.SG"):
            current_grid.append(line)
    flush()
    return samples


def load_handdraw_samples(args: argparse.Namespace) -> Tuple[Path, List[Dict[str, Any]]]:
    if args.input_jsonl:
        p = Path(args.input_jsonl)
        if p.exists():
            return p, load_jsonl(p)
    if args.input_json:
        p = Path(args.input_json)
        if p.exists():
            return p, load_json(p)
    if args.input_txt:
        p = Path(args.input_txt)
        if p.exists():
            return p, load_txt(p)
    # Optional fallback for user convenience.
    default_txt = Path("feature_maze/handdraw_mazes_ascii.txt")
    if default_txt.exists():
        return default_txt, load_txt(default_txt)
    raise FileNotFoundError("No input file found. Provide --input-jsonl, --input-json, or --input-txt.")


# -----------------------------------------------------------------------------
# Full rect-room macro extraction, minimized from 3.0.4 hard-tested extractor.
# -----------------------------------------------------------------------------
class RectRoomExtractor:
    def __init__(self, elongated_aspect_ratio: float = 3.0):
        self.elongated_aspect_ratio = elongated_aspect_ratio

    @staticmethod
    def rect_cells(r0: int, c0: int, r1: int, c1: int) -> List[Cell]:
        return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]

    def enumerate_candidates(self, grid: GridChars, main_cells: set[Cell]) -> List[Dict[str, Any]]:
        h, w = len(grid), len(grid[0])
        ok = [[1 if (r, c) in main_cells else 0 for c in range(w)] for r in range(h)]
        ps = [[0] * (w + 1) for _ in range(h + 1)]
        for r in range(h):
            row_sum = 0
            for c in range(w):
                row_sum += ok[r][c]
                ps[r + 1][c + 1] = ps[r][c + 1] + row_sum

        def rect_sum(r0: int, c0: int, r1: int, c1: int) -> int:
            return ps[r1 + 1][c1 + 1] - ps[r0][c1 + 1] - ps[r1 + 1][c0] + ps[r0][c0]

        out = []
        for r0 in range(h):
            for c0 in range(w):
                if not ok[r0][c0]:
                    continue
                for r1 in range(r0 + 1, h):
                    for c1 in range(c0 + 1, w):
                        hh, ww = r1 - r0 + 1, c1 - c0 + 1
                        area = hh * ww
                        if rect_sum(r0, c0, r1, c1) != area:
                            continue
                        aspect = max(ww, hh) / float(min(ww, hh))
                        out.append({
                            "bbox": [r0, c0, r1, c1],
                            "bbox_convention": "inclusive",
                            "w": ww,
                            "h": hh,
                            "area": area,
                            "aspect_ratio": aspect,
                            "covered_cells": self.rect_cells(r0, c0, r1, c1),
                            "sort_key": (-area, -min(ww, hh), aspect, r0, c0, r1, c1),
                        })
        out.sort(key=lambda x: x["sort_key"])
        return out

    def extract(self, grid: GridChars, start: Cell, goal: Cell) -> Dict[str, Any]:
        dist, _ = bfs_dist(grid, start)
        main_cells = set(dist)
        if goal not in main_cells:
            return {"status": "failed", "warnings": ["start_goal_not_connected"], "selected_rect_rooms": []}
        candidates = self.enumerate_candidates(grid, main_cells)
        occupied: set[Cell] = set()
        rooms: List[Dict[str, Any]] = []
        cell_to_room: Dict[Cell, str] = {}
        internal_cycle_sum = 0
        for cand in candidates:
            cells = set(cand["covered_cells"])
            if cells & occupied:
                continue
            rid = f"rr_{len(rooms):04d}"
            r0, c0, r1, c1 = cand["bbox"]
            w, h, area = cand["w"], cand["h"], cand["area"]
            anchor = [(r0 + r1) // 2, (c0 + c1) // 2]
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
                "anchor": anchor,
                "covered_cells": [list(x) for x in sorted(cells)],
                "internal_cycle_rank": internal_cycle,
            }
            rooms.append(row)
            occupied.update(cells)
            internal_cycle_sum += internal_cycle
            for cell in cells:
                cell_to_room[cell] = rid
        return {
            "status": "ok",
            "warnings": [],
            "candidate_count": len(candidates),
            "selected_rect_rooms": rooms,
            "cell_to_room": cell_to_room,
            "room_cells": occupied,
            "internal_rect_room_cycle_rank_sum": internal_cycle_sum,
        }


class FullRectRoomMacroGraphExtractor:
    def __init__(self) -> None:
        self.room_extractor = RectRoomExtractor()

    @staticmethod
    def node_id(x: Any) -> str:
        if isinstance(x, str):
            return x
        r, c = x
        return f"cell_{r}_{c}"

    def build_contracted_graph(self, grid: GridChars, rooms: List[Dict[str, Any]], main_cells: set[Cell]) -> Dict[str, Any]:
        cell_to_room: Dict[Cell, str] = {}
        room_cells: set[Cell] = set()
        for room in rooms:
            rid = room["room_id"]
            for xy in room.get("covered_cells", []):
                cell = tuple(xy)
                cell_to_room[cell] = rid
                room_cells.add(cell)

        def node_of(cell: Cell) -> Any:
            return cell_to_room.get(cell, cell)

        half_edges: List[Dict[str, Any]] = []
        dedup_room_room: set[Tuple[str, str]] = set()
        dedup_room_cell: set[Tuple[str, Cell]] = set()
        dedup_cell_cell: set[Tuple[Cell, Cell]] = set()
        for cell in sorted(main_cells):
            r, c = cell
            for nb in [(r + 1, c), (r, c + 1)]:
                if nb not in main_cells:
                    continue
                a, b = node_of(cell), node_of(nb)
                if a == b:
                    continue
                if isinstance(a, str) and isinstance(b, str):
                    key = tuple(sorted((a, b)))
                    if key in dedup_room_room:
                        continue
                    dedup_room_room.add(key)
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
                eid = f"he_{len(half_edges):04d}"
                half_edges.append({"edge_id": eid, "a": a, "b": b, "cell_a": cell, "cell_b": nb, "kind": kind})
        adj: Dict[Any, List[Tuple[Any, str]]] = defaultdict(list)
        for e in half_edges:
            adj[e["a"]].append((e["b"], e["edge_id"]))
            adj[e["b"]].append((e["a"], e["edge_id"]))
        return {"cell_to_room": cell_to_room, "room_cells": room_cells, "adj": adj, "half_edges": half_edges}

    def extract(self, grid: GridChars, start: Cell, goal: Cell) -> Dict[str, Any]:
        dist, _ = bfs_dist(grid, start)
        main_cells = set(dist)
        warnings: List[str] = []
        if goal not in main_cells:
            return {"status": "failed", "warnings": ["start_goal_not_connected"], "metrics": {}}
        room_info = self.room_extractor.extract(grid, start, goal)
        rooms = room_info.get("selected_rect_rooms", [])
        contracted = self.build_contracted_graph(grid, rooms, main_cells)
        adj = contracted["adj"]
        cell_to_room = contracted["cell_to_room"]
        room_ids = {r["room_id"] for r in rooms}
        terminal_inside = {"start": cell_to_room.get(start), "goal": cell_to_room.get(goal)}

        macro_nodes: set[Any] = set(room_ids)
        if start not in cell_to_room:
            macro_nodes.add(start)
        if goal not in cell_to_room:
            macro_nodes.add(goal)
        for node, nbs in adj.items():
            if isinstance(node, tuple) and len(nbs) != 2:
                macro_nodes.add(node)

        used: set[str] = set()
        edges: List[Dict[str, Any]] = []
        dangling = loop_without_node = 0
        for src in sorted(macro_nodes, key=lambda x: str(x)):
            for nb, eid0 in list(adj.get(src, [])):
                if eid0 in used:
                    continue
                cur, prev = nb, src
                edge_ids = [eid0]
                path_cells: List[List[int]] = []
                if isinstance(src, tuple):
                    path_cells.append(list(src))
                if isinstance(cur, tuple):
                    path_cells.append(list(cur))
                seen = {src}
                dst = None
                while True:
                    if cur in macro_nodes:
                        dst = cur
                        break
                    if cur in seen:
                        loop_without_node += 1
                        warnings.append("loop_without_macro_node")
                        break
                    seen.add(cur)
                    nxts = [(x, e) for x, e in adj.get(cur, []) if x != prev]
                    if len(nxts) != 1:
                        dangling += 1
                        warnings.append("dangling_or_ambiguous_corridor_trace")
                        break
                    nxt, eid = nxts[0]
                    edge_ids.append(eid)
                    prev, cur = cur, nxt
                    if isinstance(cur, tuple):
                        path_cells.append(list(cur))
                for e in edge_ids:
                    used.add(e)
                if dst is None or dst == src:
                    continue
                u, v = self.node_id(src), self.node_id(dst)
                kind = "room-room" if isinstance(src, str) and isinstance(dst, str) else "room-corridor" if isinstance(src, str) or isinstance(dst, str) else "corridor-corridor"
                edges.append({"edge_id": f"rme_{len(edges):04d}", "u": u, "v": v, "kind": kind, "path_cells": path_cells, "length": max(1, len(edge_ids)), "source_h_edges": edge_ids})

        nodes_before = self.node_rows(macro_nodes, edges, rooms, start, goal, terminal_inside)
        metrics_before = self.metrics(nodes_before, edges)
        term_counts = self.terminal_counts(nodes_before, terminal_inside)
        nodes_after, edges_after, comp_warn = self.compress_degree2_rooms(nodes_before, edges)
        warnings.extend(comp_warn)
        metrics_after = self.metrics(nodes_after, edges_after)
        pair_counts = Counter(tuple(sorted((e["u"], e["v"]))) for e in edges)
        parallel = sum(cnt for cnt in pair_counts.values() if cnt > 1)
        node_kind = {n["node_id"]: n.get("node_type") for n in nodes_before}
        room_pair_parallel = sum(cnt for pair, cnt in pair_counts.items() if cnt > 1 and all(node_kind.get(x) == "rect_room_anchor" for x in pair))
        metrics = {
            "rect_room_count": len(rooms),
            "rect_room_area_sum": sum(r.get("area", 0) for r in rooms),
            "tiny_2x2_room_count": sum(1 for r in rooms if r.get("w") == 2 and r.get("h") == 2),
            "elongated_room_count": sum(1 for r in rooms if max(r.get("w", 1), r.get("h", 1)) / max(1, min(r.get("w", 1), r.get("h", 1))) >= 3.0),
            **{k + "_before": v for k, v in metrics_before.items()},
            **metrics_after,
            "parallel_macro_edge_count": parallel,
            "room_pair_parallel_edge_count": room_pair_parallel,
            "terminal_inside_room_count": sum(1 for v in terminal_inside.values() if v is not None),
            **term_counts,
            "full_extraction_trace_dangling_count": dangling,
            "full_extraction_loop_without_node_count": loop_without_node,
        }
        return {
            "status": "ok" if not warnings else "warning",
            "warnings": sorted(set(warnings)),
            "selected_rect_rooms": rooms,
            "macro_nodes_before": nodes_before,
            "macro_edges_before": edges,
            "macro_nodes_after": nodes_after,
            "macro_edges_after": edges_after,
            "metrics": metrics,
            "terminal_inside": terminal_inside,
        }

    @staticmethod
    def node_rows(macro_nodes: set[Any], edges: List[Dict[str, Any]], rooms: List[Dict[str, Any]], start: Cell, goal: Cell, terminal_inside: Dict[str, Optional[str]]) -> List[Dict[str, Any]]:
        room_by_id = {r["room_id"]: r for r in rooms}
        deg = Counter()
        for e in edges:
            deg[e["u"]] += 1
            deg[e["v"]] += 1
        rows = []
        for node in sorted(macro_nodes, key=lambda x: str(x)):
            nid = node if isinstance(node, str) else f"cell_{node[0]}_{node[1]}"
            is_room = isinstance(node, str)
            is_start = (node == start) or (is_room and terminal_inside.get("start") == node)
            is_goal = (node == goal) or (is_room and terminal_inside.get("goal") == node)
            if is_room:
                room = room_by_id.get(node, {})
                cell = room.get("anchor", [None, None])
                node_type, room_id = "rect_room_anchor", node
            else:
                cell = list(node)
                node_type, room_id = ("terminal" if node in (start, goal) else "macro_cell_node"), None
            rows.append({"node_id": nid, "node_type": node_type, "room_id": room_id, "cell": cell, "macro_degree": deg[nid], "is_start": bool(is_start), "is_goal": bool(is_goal)})
        return rows

    @staticmethod
    def metrics(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, int]:
        v, e = len(nodes), len(edges)
        return {
            "rect_room_macro_vertex_count": v,
            "rect_room_macro_edge_count": e,
            "rect_room_compressed_cycle_rank": max(0, e - v + (1 if v else 0)),
            "rect_room_macro_choice_count": sum(1 for n in nodes if n.get("macro_degree", 0) >= 3),
            "rect_room_macro_endpoint_count": sum(1 for n in nodes if n.get("macro_degree", 0) == 1),
        }

    @staticmethod
    def terminal_counts(nodes: List[Dict[str, Any]], terminal_inside: Dict[str, Optional[str]]) -> Dict[str, int]:
        counts = {"terminal_endpoint_count": 0, "terminal_choice_count": 0}
        for label in ["start", "goal"]:
            row = next((n for n in nodes if (label == "start" and n.get("is_start")) or (label == "goal" and n.get("is_goal"))), None)
            deg = row.get("macro_degree", 0) if row else 0
            if deg >= 2:
                counts["terminal_choice_count"] += 1
            else:
                counts["terminal_endpoint_count"] += 1
        return counts

    @staticmethod
    def compress_degree2_rooms(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
        warnings: List[str] = []
        node_map = {n["node_id"]: dict(n) for n in nodes}
        edge_list = [dict(e) for e in edges]
        while True:
            deg = Counter()
            inc: Dict[str, List[int]] = defaultdict(list)
            for i, e in enumerate(edge_list):
                deg[e["u"]] += 1
                deg[e["v"]] += 1
                inc[e["u"]].append(i)
                inc[e["v"]].append(i)
            candidate = None
            for nid, row in node_map.items():
                if row.get("node_type") == "rect_room_anchor" and deg[nid] == 2 and not row.get("is_start") and not row.get("is_goal"):
                    candidate = nid
                    break
            if candidate is None:
                break
            idxs = inc[candidate]
            if len(idxs) != 2:
                warnings.append("degree2_compression_warning_parallel_or_loop_room")
                break
            e1, e2 = edge_list[idxs[0]], edge_list[idxs[1]]
            a = e1["v"] if e1["u"] == candidate else e1["u"]
            b = e2["v"] if e2["u"] == candidate else e2["u"]
            for i in sorted(idxs, reverse=True):
                edge_list.pop(i)
            edge_list.append({"edge_id": f"rme_c_{len(edge_list):04d}", "u": a, "v": b, "kind": "compressed_degree2_room", "path_cells": [], "length": max(1, e1.get("length", 1) + e2.get("length", 1))})
            node_map.pop(candidate, None)
        deg = Counter()
        for e in edge_list:
            deg[e["u"]] += 1
            deg[e["v"]] += 1
        rows = []
        for nid, row in node_map.items():
            row = dict(row)
            row["macro_degree"] = deg[nid]
            rows.append(row)
        return sorted(rows, key=lambda x: x["node_id"]), edge_list, warnings


# -----------------------------------------------------------------------------
# Full metrics and bins
# -----------------------------------------------------------------------------
def load_bucket_config(path: Optional[str]) -> Dict[str, List[Dict[str, Any]]]:
    default = {
        "cycle_rank_bins": [
            {"name": "tree", "range": [0, 0]},
            {"name": "single_loop", "range": [1, 1]},
            {"name": "multi_loop", "range": [2, 10**9]},
        ],
        "bfs_len_bins": [
            {"name": "short", "range": [1, 6]},
            {"name": "medium", "range": [7, 12]},
            {"name": "long", "range": [13, 10**9]},
        ],
        "macro_choice_bins": [
            {"name": "low_choice", "range": [0, 1]},
            {"name": "mid_choice", "range": [2, 4]},
            {"name": "high_choice", "range": [5, 10**9]},
        ],
        "macro_endpoint_bins": [
            {"name": "low_endpoint", "range": [0, 1]},
            {"name": "mid_endpoint", "range": [2, 4]},
            {"name": "high_endpoint", "range": [5, 10**9]},
        ],
    }
    if not path:
        return default
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    src = obj.get("archive", obj)
    for key in list(default):
        if key in src:
            default[key] = src[key]
    return default


def find_bin(value: Optional[int], bins: List[Dict[str, Any]]) -> str:
    if value is None:
        return "unknown"
    for b in bins:
        lo, hi = b["range"]
        if lo <= value <= hi:
            return b["name"]
    return "out_of_range"


def compute_bin_key(metrics: Dict[str, Any], bucket_cfg: Dict[str, Any]) -> Dict[str, str]:
    cycle = find_bin(metrics.get("rect_room_compressed_cycle_rank"), bucket_cfg["cycle_rank_bins"])
    bfs = find_bin(metrics.get("bfs_len"), bucket_cfg["bfs_len_bins"])
    choice = find_bin(metrics.get("rect_room_macro_choice_count"), bucket_cfg["macro_choice_bins"])
    endpoint = find_bin(metrics.get("rect_room_macro_endpoint_count"), bucket_cfg["macro_endpoint_bins"])
    return {
        "full_cycle_rank_bin": cycle,
        "full_bfs_len_bin": bfs,
        "full_macro_choice_bin": choice,
        "full_macro_endpoint_bin": endpoint,
        "full_bin_key": f"{cycle}|{bfs}|{choice}|{endpoint}",
    }


def compute_raw_metrics(grid: GridChars, start: Cell, goal: Cell) -> Dict[str, Any]:
    dist, _ = bfs_dist(grid, start)
    main = set(dist)
    v = len(main)
    edge_count = 0
    degs = []
    for cell in main:
        d = sum(1 for nb in neighbors(grid, cell) if nb in main)
        degs.append(d)
        edge_count += d
    edge_count //= 2
    return {
        "raw_vertex_count": v,
        "raw_edge_count": edge_count,
        "raw_cycle_rank": max(0, edge_count - v + (1 if v else 0)),
        "raw_endpoint_cell_count": sum(1 for d in degs if d == 1),
        "raw_choice_cell_count": sum(1 for d in degs if d >= 3),
    }


def run_full_extraction(sample: Dict[str, Any], extractor: FullRectRoomMacroGraphExtractor, bucket_cfg: Dict[str, Any]) -> Dict[str, Any]:
    base = validate_sample(sample)
    grid = base["grid"]
    start = tuple(base["start"])
    goal = tuple(base["goal"])
    record: Dict[str, Any] = dict(base)
    raw = compute_raw_metrics(grid, start, goal)
    record.update(raw)
    if not base["connected"]:
        record.update({
            "extraction_status": "ERROR",
            "error_message": "start_goal_not_connected",
            "extraction_warnings": ["start_goal_not_connected"],
            "full_bin_key": "unknown|unknown|unknown|unknown",
        })
        return record
    try:
        actual = extractor.extract(grid, start, goal)
        metrics = actual.get("metrics", {})
        record.update(metrics)
        record.update(compute_bin_key(record, bucket_cfg))
        record["extraction_status"] = actual.get("status", "ok")
        record["extraction_warnings"] = actual.get("warnings", [])
        record["selected_rect_rooms"] = actual.get("selected_rect_rooms", [])
        record["macro_nodes_after"] = actual.get("macro_nodes_after", [])
        record["macro_edges_after"] = actual.get("macro_edges_after", [])
    except Exception as e:
        record.update({
            "extraction_status": "ERROR",
            "error_message": f"{type(e).__name__}: {e}",
            "extraction_warnings": [],
            "full_bin_key": "unknown|unknown|unknown|unknown",
        })
    return record


# -----------------------------------------------------------------------------
# Bucket feasibility / conflict classification
# -----------------------------------------------------------------------------
def all_bucket_keys(bucket_cfg: Dict[str, Any]) -> List[str]:
    rows = []
    for cy in bucket_cfg["cycle_rank_bins"]:
        for bf in bucket_cfg["bfs_len_bins"]:
            for ch in bucket_cfg["macro_choice_bins"]:
                for ep in bucket_cfg["macro_endpoint_bins"]:
                    rows.append(f"{cy['name']}|{bf['name']}|{ch['name']}|{ep['name']}")
    return rows


def bin_interval(name: str, bins: List[Dict[str, Any]]) -> Tuple[int, float]:
    for b in bins:
        if b["name"] == name:
            lo, hi = b["range"]
            return int(lo), (math.inf if hi >= 10**8 else int(hi))
    return 0, math.inf


def classify_bucket_feasibility(bin_key: str, handdraw_count: int, bucket_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cycle_name, _bfs_name, choice_name, endpoint_name = bin_key.split("|")
    mu_min, mu_max = bin_interval(cycle_name, bucket_cfg["cycle_rank_bins"])
    c_min, c_max = bin_interval(choice_name, bucket_cfg["macro_choice_bins"])
    e_min, e_max = bin_interval(endpoint_name, bucket_cfg["macro_endpoint_bins"])
    note = "conservative bucket-level rule; not a proof for individual mazes"
    structural_conflict = False
    l_min: Optional[int] = None

    # Conservative conflict rule from prompt. For multi_loop, do not mark impossible.
    if cycle_name != "multi_loop":
        l_min = max(0, int(c_min - 2 * mu_max + 2))
        structural_conflict = bool(e_max < l_min)

    # Observed handdraw bins have priority. If the conservative rule marks an
    # observed bin as constrained, keep it covered and flag the conflict for review.
    observed_structural_conflict = bool(handdraw_count > 0 and structural_conflict)
    if handdraw_count > 0:
        status = "covered_by_handdraw"
        reason = "observed in handdraw dataset"
        if observed_structural_conflict:
            reason += "; conservative feasibility rule also flags this bin, inspect extractor/sample"
    elif structural_conflict:
        status = "structural_infeasible_or_constrained"
        reason = f"endpoint_bin_max={e_max} < conservative_L_min={l_min} for C_min={c_min}, mu_max={mu_max}"
        note = "choice / endpoint / cycle dimensions are coupled; uniform 81-bin target includes structurally constrained regions"
    else:
        status = "unobserved_possible"
        reason = "not observed, but not ruled out by conservative bucket rule"

    return {
        "bin_key": bin_key,
        "status": status,
        "reason": reason,
        "handdraw_count": handdraw_count,
        "observed_structural_conflict": observed_structural_conflict,
        "cycle_interval": [mu_min, "inf" if math.isinf(mu_max) else mu_max],
        "choice_interval": [c_min, "inf" if math.isinf(c_max) else c_max],
        "endpoint_interval": [e_min, "inf" if math.isinf(e_max) else e_max],
        "conservative_L_min": l_min,
        "theoretical_note": note,
    }


# -----------------------------------------------------------------------------
# Target distribution JSON builder
# -----------------------------------------------------------------------------
def validate_target_args(args: argparse.Namespace) -> None:
    if args.target_total <= 0:
        raise ValueError("--target-total must be > 0")
    for name in ("observed_mass", "explore_mass", "structural_mass"):
        value = float(getattr(args, name))
        if value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
    total = float(args.observed_mass) + float(args.explore_mass) + float(args.structural_mass)
    if abs(total - 1.0) > 1e-6:
        raise ValueError("--observed-mass + --explore-mass + --structural-mass must equal 1.0")
    if args.min_quota_observed < 0 or args.min_quota_explore < 0:
        raise ValueError("min quota values must be >= 0")


def largest_remainder_quota(raw_by_key: Dict[str, float], total: int) -> Dict[str, int]:
    floors = {k: int(math.floor(v)) for k, v in raw_by_key.items()}
    remaining = total - sum(floors.values())
    if remaining < 0:
        raise ValueError("floor quotas exceed target_total")
    ranked = sorted(raw_by_key, key=lambda k: (raw_by_key[k] - floors[k], raw_by_key[k], k), reverse=True)
    quotas = dict(floors)
    for k in ranked[:remaining]:
        quotas[k] += 1
    return quotas


def rebalance_min_quotas(
    quotas: Dict[str, int],
    raw_by_key: Dict[str, float],
    status_by_key: Dict[str, str],
    total: int,
    min_observed: int,
    min_explore: int,
) -> Dict[str, int]:
    q = dict(quotas)
    for k, status in status_by_key.items():
        if status == "handdraw_observed":
            q[k] = max(q.get(k, 0), min_observed)
        elif status == "unobserved_possible":
            q[k] = max(q.get(k, 0), min_explore)

    def reducible_keys(preferred_status: str) -> List[str]:
        min_floor = min_explore if preferred_status == "unobserved_possible" else min_observed
        return sorted(
            [k for k, status in status_by_key.items() if status == preferred_status and q.get(k, 0) > min_floor],
            key=lambda k: (raw_by_key.get(k, 0.0), q.get(k, 0), k),
        )

    while sum(q.values()) > total:
        changed = False
        for status in ("unobserved_possible", "handdraw_observed"):
            for k in reducible_keys(status):
                q[k] -= 1
                changed = True
                break
            if changed:
                break
        if not changed:
            raise ValueError("target_total is too small for requested min quotas")

    while sum(q.values()) < total:
        candidates = [k for k, status in status_by_key.items() if status != "structural_infeasible_or_constrained" or raw_by_key.get(k, 0.0) > 0]
        if not candidates:
            raise ValueError("no eligible bucket can receive remaining quota")
        k = max(candidates, key=lambda x: (raw_by_key.get(x, 0.0) - math.floor(raw_by_key.get(x, 0.0)), raw_by_key.get(x, 0.0), x))
        q[k] += 1
    return q


def build_target_distribution_json(
    input_path: Path,
    summary: Dict[str, Any],
    bin_map: Dict[str, Any],
    feasibility_report: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    validate_target_args(args)
    all_rows: Dict[str, Dict[str, Any]] = {}
    for group in ("covered_by_handdraw", "unobserved_possible", "structural_infeasible_or_constrained"):
        for row in feasibility_report.get(group, []):
            all_rows[row["bin_key"]] = row

    all_keys = sorted(all_rows)
    observed_keys = [k for k in all_keys if bin_map.get(k, {}).get("count", 0) > 0]
    structural_keys = [k for k in all_keys if k not in observed_keys and all_rows[k].get("status") == "structural_infeasible_or_constrained"]
    explore_keys = [k for k in all_keys if k not in observed_keys and k not in structural_keys]
    observed_count_sum = sum(int(bin_map.get(k, {}).get("count", 0)) for k in observed_keys)

    warnings: List[str] = []
    observed_mass = float(args.observed_mass)
    explore_mass = float(args.explore_mass)
    structural_mass = float(args.structural_mass)
    if structural_mass > 0:
        warnings.append("[WARN] structural_mass > 0: structurally constrained buckets will receive target quota.")
    if not explore_keys and explore_mass > 0:
        warnings.append("explore_mass moved into observed_mass because no unobserved_possible bucket exists")
        observed_mass += explore_mass
        explore_mass = 0.0
    if not observed_keys and observed_mass > 0:
        warnings.append("observed_mass moved into explore_mass because no handdraw_observed bucket exists")
        explore_mass += observed_mass
        observed_mass = 0.0
    if not structural_keys and structural_mass > 0:
        warnings.append("structural_mass ignored because no structural constrained bucket exists")
        structural_mass = 0.0

    raw_by_key: Dict[str, float] = {}
    prob_by_key: Dict[str, float] = {}
    status_by_key: Dict[str, str] = {}
    bins_out: Dict[str, Any] = {}

    for k in all_keys:
        hd_count = int(bin_map.get(k, {}).get("count", 0))
        maze_ids = list(bin_map.get(k, {}).get("maze_ids", []))
        observed_conflict = bool(all_rows[k].get("observed_structural_conflict", False))
        if hd_count > 0:
            status = "handdraw_observed"
            handdraw_prob = hd_count / observed_count_sum if observed_count_sum else 0.0
            target_prob = observed_mass * handdraw_prob
            reason = "observed in handdraw distribution"
        elif k in structural_keys:
            status = "structural_infeasible_or_constrained"
            handdraw_prob = 0.0
            target_prob = structural_mass / len(structural_keys) if structural_keys and structural_mass > 0 else 0.0
            reason = all_rows[k].get("reason", "structural constrained by conservative feasibility rule")
        else:
            status = "unobserved_possible"
            handdraw_prob = 0.0
            target_prob = explore_mass / len(explore_keys) if explore_keys and explore_mass > 0 else 0.0
            reason = "unobserved in handdraw dataset but not ruled out by conservative feasibility rule"
        status_by_key[k] = status
        prob_by_key[k] = target_prob
        raw_by_key[k] = target_prob * int(args.target_total)
        bins_out[k] = {
            "status": status,
            "handdraw_count": hd_count,
            "handdraw_prob": handdraw_prob,
            "target_prob": target_prob,
            "raw_quota": raw_by_key[k],
            "target_quota": 0,
            "reason": reason,
            "maze_ids": maze_ids,
            "observed_structural_conflict": observed_conflict,
        }

    quotas = largest_remainder_quota(raw_by_key, int(args.target_total))
    quotas = rebalance_min_quotas(
        quotas, raw_by_key, status_by_key, int(args.target_total), int(args.min_quota_observed), int(args.min_quota_explore)
    )
    for k, q in quotas.items():
        bins_out[k]["target_quota"] = int(q)

    quota_sum = sum(int(x["target_quota"]) for x in bins_out.values())
    if quota_sum != int(args.target_total):
        raise RuntimeError(f"quota_sum != target_total: {quota_sum} != {args.target_total}")

    return {
        "schema_version": "handdraw_target_distribution.v1",
        "created_by": "analyze_handdraw_distribution.py",
        "source_input": str(input_path),
        "n_handdraw_samples": int(summary.get("n_samples", 0)),
        "n_valid_samples": int(summary.get("n_valid", 0)),
        "n_extraction_success": int(summary.get("n_valid", 0)) - int(summary.get("n_extraction_error", 0)),
        "target_total": int(args.target_total),
        "observed_mass": float(args.observed_mass),
        "explore_mass": float(args.explore_mass),
        "structural_mass": float(args.structural_mass),
        "effective_observed_mass": observed_mass,
        "effective_explore_mass": explore_mass,
        "effective_structural_mass": structural_mass,
        "min_quota_observed": int(args.min_quota_observed),
        "min_quota_explore": int(args.min_quota_explore),
        "n_bins_total": len(all_keys),
        "n_handdraw_observed_bins": len(observed_keys),
        "n_unobserved_possible_bins": len(explore_keys),
        "n_structural_constrained_bins": len(structural_keys),
        "quota_sum": quota_sum,
        "warnings": warnings,
        "top_target_quotas": sorted(
            ({"bin_key": k, **v} for k, v in bins_out.items()),
            key=lambda row: (row["target_quota"], row["target_prob"], row["handdraw_count"]),
            reverse=True,
        )[:15],
        "bins": bins_out,
    }



# -----------------------------------------------------------------------------
# Reporting and visualization
# -----------------------------------------------------------------------------
def summarize_distribution(records: List[Dict[str, Any]], bucket_cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    valid = [r for r in records if r.get("validation_ok", True)]
    connected = [r for r in records if r.get("connected")]
    errors = [r for r in records if r.get("extraction_status") == "ERROR"]
    bin_map: Dict[str, Dict[str, Any]] = {}
    for r in records:
        key = r.get("full_bin_key", "unknown|unknown|unknown|unknown")
        if key.startswith("unknown"):
            continue
        row = bin_map.setdefault(key, {"count": 0, "maze_ids": [], "examples": []})
        row["count"] += 1
        row["maze_ids"].append(r.get("maze_id"))
        if len(row["examples"]) < 5:
            row["examples"].append({"maze_id": r.get("maze_id"), "bfs_len": r.get("bfs_len"), "cycle": r.get("rect_room_compressed_cycle_rank"), "choice": r.get("rect_room_macro_choice_count"), "endpoint": r.get("rect_room_macro_endpoint_count")})
    cycle_counts = Counter(r.get("full_cycle_rank_bin") for r in records if r.get("full_cycle_rank_bin"))
    bfs_counts = Counter(r.get("full_bfs_len_bin") for r in records if r.get("full_bfs_len_bin"))
    choice_counts = Counter(r.get("full_macro_choice_bin") for r in records if r.get("full_macro_choice_bin"))
    endpoint_counts = Counter(r.get("full_macro_endpoint_bin") for r in records if r.get("full_macro_endpoint_bin"))
    all_keys = all_bucket_keys(bucket_cfg)
    feasibility_rows = [classify_bucket_feasibility(k, bin_map.get(k, {}).get("count", 0), bucket_cfg) for k in all_keys]
    structural = [r for r in feasibility_rows if r["status"] == "structural_infeasible_or_constrained"]
    covered = [r for r in feasibility_rows if r["status"] == "covered_by_handdraw"]
    unobs = [r for r in feasibility_rows if r["status"] == "unobserved_possible"]
    feasible_den = len(all_keys) - len(structural)
    summary = {
        "n_samples": len(records),
        "n_valid": len(valid),
        "n_connected": len(connected),
        "n_extraction_error": len(errors),
        "n_bins_total": len(all_keys),
        "n_handdraw_bins_non_empty": len(bin_map),
        "handdraw_bin_coverage_rate": len(bin_map) / len(all_keys) if all_keys else 0.0,
        "cycle_bin_counts": dict(cycle_counts),
        "bfs_bin_counts": dict(bfs_counts),
        "choice_bin_counts": dict(choice_counts),
        "endpoint_bin_counts": dict(endpoint_counts),
        "top_bins": sorted(({"bin_key": k, **v} for k, v in bin_map.items()), key=lambda x: x["count"], reverse=True)[:15],
        "structural_infeasible_or_constrained_count": len(structural),
        "covered_by_handdraw_count": len(covered),
        "unobserved_possible_count": len(unobs),
        "feasibility_adjusted_handdraw_coverage": len(covered) / feasible_den if feasible_den else 0.0,
    }
    metric_dist = {
        "bfs_len": stat(r.get("bfs_len") for r in records),
        "rect_room_count": stat(r.get("rect_room_count") for r in records),
        "rect_room_compressed_cycle_rank": stat(r.get("rect_room_compressed_cycle_rank") for r in records),
        "rect_room_macro_choice_count": stat(r.get("rect_room_macro_choice_count") for r in records),
        "rect_room_macro_endpoint_count": stat(r.get("rect_room_macro_endpoint_count") for r in records),
        "rect_room_macro_vertex_count": stat(r.get("rect_room_macro_vertex_count") for r in records),
        "rect_room_macro_edge_count": stat(r.get("rect_room_macro_edge_count") for r in records),
        "parallel_macro_edge_count": stat(r.get("parallel_macro_edge_count") for r in records),
        "terminal_inside_room_count": stat(r.get("terminal_inside_room_count") for r in records),
        "raw_cycle_rank": stat(r.get("raw_cycle_rank") for r in records),
        "raw_choice_cell_count": stat(r.get("raw_choice_cell_count") for r in records),
        "raw_endpoint_cell_count": stat(r.get("raw_endpoint_cell_count") for r in records),
        "reachable_free_ratio": stat(r.get("reachable_free_ratio") for r in records),
    }
    feasibility_report = {
        "all_bins": len(all_keys),
        "covered_by_handdraw": covered,
        "structural_infeasible_or_constrained": structural,
        "unobserved_possible": unobs,
        "coverage_raw": summary["handdraw_bin_coverage_rate"],
        "coverage_excluding_structural": summary["feasibility_adjusted_handdraw_coverage"],
        "rule_note": "Conservative bucket-level check: for tree/single_loop, endpoint_bin_max must be >= max(0, C_min - 2*mu_max + 2). Multi-loop buckets are not ruled out by this simple rule.",
    }
    return summary, bin_map, feasibility_report, metric_dist


def render_visualizations(records: List[Dict[str, Any]], out_dir: Path, max_count: int) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as e:
        print(f"[WARN] matplotlib unavailable; visualizations skipped: {e}")
        return
    vis_dir = out_dir / "visualizations"
    ensure_dir(vis_dir)
    for r in records[:max_count]:
        grid = r.get("grid", [])
        if not grid:
            continue
        rows, cols = len(grid), len(grid[0])
        fig, ax = plt.subplots(figsize=(max(4, cols * 0.45), max(4, rows * 0.45)))
        ax.set_title(f"{r.get('maze_id')} | {r.get('full_bin_key')}")
        for rr, row in enumerate(grid):
            for cc, ch in enumerate(row):
                color = "black" if ch == "#" else "white"
                ax.add_patch(Rectangle((cc, rows - rr - 1), 1, 1, facecolor=color, edgecolor="gray", linewidth=0.5))
                if ch in "SG":
                    ax.text(cc + 0.5, rows - rr - 0.5, ch, ha="center", va="center", color="blue" if ch == "S" else "orange", fontweight="bold")
        for room in r.get("selected_rect_rooms", []):
            r0, c0, r1, c1 = room["bbox"]
            ax.add_patch(Rectangle((c0, rows - r1 - 1), c1 - c0 + 1, r1 - r0 + 1, fill=False, edgecolor="red", linewidth=2))
            ar, ac = room.get("anchor", [None, None])
            if ar is not None:
                ax.plot([ac + 0.5], [rows - ar - 0.5], marker="o", color="red")
        for e in r.get("macro_edges_after", []):
            pts = e.get("path_cells", [])
            if len(pts) >= 2:
                xs = [p[1] + 0.5 for p in pts]
                ys = [rows - p[0] - 0.5 for p in pts]
                ax.plot(xs, ys, color="green", linewidth=1.5, alpha=0.8)
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(vis_dir / f"{r.get('maze_id')}__maze_macro.png", dpi=150)
        plt.close(fig)


def save_reports(records: List[Dict[str, Any]], output_dir: Path, bucket_cfg: Dict[str, Any], save_vis: bool, max_vis: int) -> Dict[str, Any]:
    ensure_dir(output_dir)
    samples_path = output_dir / "handdraw_samples_full_metrics.jsonl"
    errors_path = output_dir / "handdraw_error_samples.jsonl"
    if samples_path.exists():
        samples_path.unlink()
    if errors_path.exists():
        errors_path.unlink()
    for r in records:
        compact = {k: v for k, v in r.items() if k not in ("macro_nodes_before", "macro_edges_before")}
        append_jsonl(samples_path, compact)
        if r.get("extraction_status") == "ERROR" or r.get("validation_error"):
            append_jsonl(errors_path, r)
    summary, bin_map, feasibility_report, metric_dist = summarize_distribution(records, bucket_cfg)
    write_json(output_dir / "handdraw_distribution_summary.json", summary)
    write_json(output_dir / "handdraw_bin_distribution.json", bin_map)
    write_json(output_dir / "handdraw_bucket_feasibility_report.json", feasibility_report)
    write_json(output_dir / "handdraw_metric_distribution.json", metric_dist)
    write_json(output_dir / "resolved_bucket_config.json", bucket_cfg)
    if save_vis:
        render_visualizations(records, output_dir, max_vis)
    return summary


def print_summary(input_path: Path, summary: Dict[str, Any], feasibility: Dict[str, Any], target_info: Optional[Dict[str, Any]] = None) -> None:
    print("[Handdraw Distribution Analyzer]\n")
    print("=== Input ===")
    print(f"input_path                         {input_path}")
    print(f"n_samples                          {summary['n_samples']}")
    print(f"n_valid                            {summary['n_valid']}")
    print(f"n_connected                        {summary['n_connected']}")
    print(f"n_errors                           {summary['n_extraction_error']}")
    print("\n=== Handdraw Full-Bin Distribution ===")
    print(f"n_bins_total                       {summary['n_bins_total']}")
    print(f"n_bins_non_empty                   {summary['n_handdraw_bins_non_empty']}")
    print(f"coverage_rate                      {summary['handdraw_bin_coverage_rate']:.4f}")
    print("top_bins")
    for b in summary["top_bins"][:10]:
        print(f"  {b['bin_key']:<48} {b['count']}")
    print("\n=== Metric Distribution ===")
    print(f"cycle_bin_counts                   {summary['cycle_bin_counts']}")
    print(f"bfs_bin_counts                     {summary['bfs_bin_counts']}")
    print(f"choice_bin_counts                  {summary['choice_bin_counts']}")
    print(f"endpoint_bin_counts                {summary['endpoint_bin_counts']}")
    print("\n=== Bucket Feasibility / Conflict ===")
    print(f"structural_infeasible_or_constrained_count {summary['structural_infeasible_or_constrained_count']}")
    print(f"unobserved_possible_count          {summary['unobserved_possible_count']}")
    print(f"covered_by_handdraw_count          {summary['covered_by_handdraw_count']}")
    print(f"coverage_excluding_structural      {summary['feasibility_adjusted_handdraw_coverage']:.4f}")
    print("\n=== Potential Conflicts with 81-Bucket Design ===")
    for row in feasibility["structural_infeasible_or_constrained"][:12]:
        print(f"  {row['bin_key']:<48} {row['reason']}")

    if target_info is not None:
        print("\n=== Target Distribution JSON ===")
        print(f"target_json                        {target_info.get('target_json')}")
        print(f"target_total                       {target_info.get('target_total')}")
        print(f"quota_sum                          {target_info.get('quota_sum')}")
        print(f"observed_mass                      {target_info.get('observed_mass')}")
        print(f"explore_mass                       {target_info.get('explore_mass')}")
        print(f"structural_mass                    {target_info.get('structural_mass')}")
        print(f"handdraw_observed_bins             {target_info.get('n_handdraw_observed_bins')}")
        print(f"unobserved_possible_bins           {target_info.get('n_unobserved_possible_bins')}")
        print(f"structural_constrained_bins        {target_info.get('n_structural_constrained_bins')}")
        print("top_target_quotas")
        for row in target_info.get("top_target_quotas", [])[:10]:
            print(f"  {row['bin_key']:<48} quota={row['target_quota']:<4} prob={row['target_prob']:.6f} count={row['handdraw_count']}")

    print("\n=== Debug Hints ===")
    hints = []
    if summary["n_samples"] and summary["n_handdraw_bins_non_empty"] / summary["n_bins_total"] < 0.35:
        hints.append("[HINT] handdraw distribution is concentrated. Likely direction: use this distribution as target weighting instead of uniform 81-bin coverage.")
    if summary["structural_infeasible_or_constrained_count"] > 0:
        hints.append("[HINT] some of the 81 buckets are structurally constrained by graph-theoretic coupling between cycle rank, choice count, and endpoint count.")
    if summary["n_extraction_error"] > 0:
        hints.append("[HINT] some handdraw samples failed full extraction. Inspect handdraw_error_samples.jsonl.")
    if target_info is not None:
        if float(target_info.get("observed_mass", 1.0)) < 0.8:
            hints.append("[HINT] observed_mass is low. Target distribution may drift away from handdraw samples.")
        if float(target_info.get("explore_mass", 0.0)) > 0.2:
            hints.append("[HINT] explore_mass is high. Many unobserved possible bins may receive substantial quota.")
        if float(target_info.get("structural_mass", 0.0)) > 0:
            hints.append("[WARN] structural_mass > 0. Structurally constrained bins will receive nonzero quota.")
        if int(target_info.get("n_handdraw_observed_bins", 0)) < 10:
            hints.append("[HINT] handdraw observed bins are few. Consider collecting more handdraw mazes before fixing target distribution.")
        if int(target_info.get("quota_sum", -1)) != int(target_info.get("target_total", -2)):
            hints.append("[ERROR] quota_sum != target_total")
    if not hints:
        hints.append("No major anomaly detected.")
    for h in hints:
        print(h)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze hand-drawn maze distribution with full rect-room macro extraction.")
    ap.add_argument("--input-jsonl", default="feature_maze/handdraw_mazes_ascii.jsonl")
    ap.add_argument("--input-json", default=None)
    ap.add_argument("--input-txt", default=None)
    ap.add_argument("--output-dir", default="feature_maze/v3_0_maze_generation/outputs/handdraw_distribution")
    ap.add_argument("--bucket-config-json", default=None)
    ap.add_argument("--save-visualizations", action="store_true")
    ap.add_argument("--max-visualizations", type=int, default=100)
    ap.add_argument("--target-json", default="feature_maze/v3_0_maze_generation/configs/handdraw_target_distribution_3_0_4.json")
    ap.add_argument("--target-total", type=int, default=1000)
    ap.add_argument("--observed-mass", type=float, default=0.95)
    ap.add_argument("--explore-mass", type=float, default=0.05)
    ap.add_argument("--structural-mass", type=float, default=0.0)
    ap.add_argument("--min-quota-observed", type=int, default=1)
    ap.add_argument("--min-quota-explore", type=int, default=0)
    ap.add_argument("--no-build-target-json", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    input_path, raw_samples = load_handdraw_samples(args)
    bucket_cfg = load_bucket_config(args.bucket_config_json)
    extractor = FullRectRoomMacroGraphExtractor()
    records: List[Dict[str, Any]] = []
    for idx, sample in enumerate(raw_samples):
        try:
            rec = run_full_extraction(sample, extractor, bucket_cfg)
            rec["validation_ok"] = True
        except Exception as e:
            rec = {
                "maze_id": str(sample.get("maze_id", f"sample_{idx:06d}")) if isinstance(sample, dict) else f"sample_{idx:06d}",
                "validation_ok": False,
                "validation_error": f"{type(e).__name__}: {e}",
                "extraction_status": "ERROR",
                "full_bin_key": "unknown|unknown|unknown|unknown",
            }
        records.append(rec)
    output_dir = Path(args.output_dir)
    summary = save_reports(records, output_dir, bucket_cfg, args.save_visualizations, args.max_visualizations)
    feasibility = json.loads((output_dir / "handdraw_bucket_feasibility_report.json").read_text(encoding="utf-8"))
    bin_map = json.loads((output_dir / "handdraw_bin_distribution.json").read_text(encoding="utf-8"))

    target_info: Optional[Dict[str, Any]] = None
    if not args.no_build_target_json:
        target = build_target_distribution_json(input_path, summary, bin_map, feasibility, args)
        target_path = Path(args.target_json)
        write_json(target_path, target)
        target_info = {
            "target_json": str(target_path),
            "target_total": target["target_total"],
            "quota_sum": target["quota_sum"],
            "observed_mass": target["observed_mass"],
            "explore_mass": target["explore_mass"],
            "structural_mass": target["structural_mass"],
            "n_handdraw_observed_bins": target["n_handdraw_observed_bins"],
            "n_unobserved_possible_bins": target["n_unobserved_possible_bins"],
            "n_structural_constrained_bins": target["n_structural_constrained_bins"],
            "top_target_quotas": target["top_target_quotas"],
        }
        summary.update({
            "target_distribution_json": str(target_path),
            "target_total": target["target_total"],
            "target_quota_sum": target["quota_sum"],
            "observed_mass": target["observed_mass"],
            "explore_mass": target["explore_mass"],
            "structural_mass": target["structural_mass"],
            "n_target_observed_bins": target["n_handdraw_observed_bins"],
            "n_target_explore_bins": target["n_unobserved_possible_bins"],
            "n_target_structural_bins": target["n_structural_constrained_bins"],
        })
        write_json(output_dir / "handdraw_distribution_summary.json", summary)
    print_summary(input_path, summary, feasibility, target_info)


if __name__ == "__main__":
    main()
