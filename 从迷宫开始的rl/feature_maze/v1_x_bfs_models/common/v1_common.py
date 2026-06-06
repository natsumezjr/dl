#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v1_x_bfs_models stable common (v2).

Stable infrastructure only: maze/BFS/dataset/rollout/metrics/audit.
No model classes, training loops, DEFAULTS, or PRESETS.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# =============================================================================
# 1. constants
# =============================================================================

ACTIONS: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES: List[str] = ["UP", "DOWN", "LEFT", "RIGHT"]
INF = 10**9
INPUT_MODES = ("compact_2ch", "standard_3ch")

REQUIRED_REPORTS = (
    "resolved_config.json",
    "run_metadata.json",
    "training_history.json",
    "evaluation_summary.json",
    "rollout_debug.json",
    "protocol_compliance_audit.json",
)

V1_ROOT = Path(__file__).resolve().parents[1]
V1_OUTPUTS_ROOT = V1_ROOT / "outputs"
COMMON_PATH = Path(__file__).resolve()

# =============================================================================
# 2. seed / io / basic utils
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Any) -> None:
    def default(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, set):
            return sorted(o)
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=default)


def mean(xs: Sequence[float]) -> float:
    ys = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(ys)) if ys else float("nan")


def qtile(xs: Sequence[float], q: float) -> float:
    ys = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.quantile(ys, q)) if ys else float("nan")


def maybe_tqdm(iterable, args, desc: str = "", total=None, leave: bool = False):
    if getattr(args, "no_tqdm", False) or tqdm is None:
        return iterable
    kwargs = dict(
        desc=desc,
        total=total,
        leave=leave,
        file=sys.stderr,
        dynamic_ncols=True,
        mininterval=0.3,
        smoothing=0.05,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]",
    )
    if total is not None and total > 20:
        kwargs["miniters"] = max(1, total // 20)
    return tqdm(iterable, **kwargs)


def log_progress(msg: str, args=None) -> None:
    """Print a line without breaking an active tqdm bar."""
    if tqdm is not None and args is not None and not getattr(args, "no_tqdm", False):
        tqdm.write(msg, file=sys.stderr)
    else:
        print(msg)


def fmt_float(x, width: int = 8, prec: int = 3) -> str:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return f"{'nan':>{width}s}"
        return f"{float(x):{width}.{prec}f}"
    except Exception:
        return f"{'nan':>{width}s}"


def fmt_pct2(x, width: int = 8, prec: int = 2) -> str:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return f"{'nan%':>{width}s}"
        return f"{100.0 * float(x):{width - 1}.{prec}f}%"
    except Exception:
        return f"{'nan%':>{width}s}"


def fmt_str(s, width: int = 24) -> str:
    s = "" if s is None else str(s)
    if len(s) <= width:
        return f"{s:<{width}s}"
    if width <= 6:
        return s[:width]
    keep = width - 3
    left = keep // 2
    right = keep - left
    return f"{s[:left]}...{s[-right:]}"


def _fmt_cell(v, width: int, spec: str) -> str:
    if spec == "s":
        return fmt_str(v, width)
    if spec == "d":
        try:
            return f"{int(v):{width}d}"
        except Exception:
            return f"{'':>{width}s}"
    if spec == "pct":
        return fmt_pct2(v, width)
    if spec.startswith(".") and spec.endswith("f"):
        return fmt_float(v, width, int(spec[1:-1]))
    return f"{str(v):>{width}s}"


def print_table(title: str, columns: Sequence[Tuple[str, int, str]], rows: Sequence[Dict[str, Any]]) -> None:
    print(f"\n{title}")
    header = " ".join(fmt_str(c[0], c[1]) if c[2] == "s" else f"{c[0]:>{c[1]}s}" for c in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" ".join(_fmt_cell(row.get(c[0], ""), c[1], c[2]) for c in columns))


# =============================================================================
# 3. maze generation
# =============================================================================

def difficulty_of_len(args, bfs_len: int) -> Optional[str]:
    if args.easy_bfs_min <= bfs_len <= args.easy_bfs_max:
        return "easy"
    if args.medium_bfs_min <= bfs_len <= args.medium_bfs_max:
        return "medium"
    if args.hard_bfs_min <= bfs_len <= args.hard_bfs_max:
        return "hard"
    return None


def choose_target_difficulty(args) -> str:
    r = random.random()
    if r < args.difficulty_easy:
        return "easy"
    if r < args.difficulty_easy + args.difficulty_medium:
        return "medium"
    return "hard"


def generate_maze(args, target_difficulty: str):
    n = args.grid_size
    ranges = {
        "easy": (args.easy_bfs_min, args.easy_bfs_max),
        "medium": (args.medium_bfs_min, args.medium_bfs_max),
        "hard": (args.hard_bfs_min, args.hard_bfs_max),
    }
    lo, hi = ranges[target_difficulty]
    retries = 0
    for attempt in range(500):
        retries = attempt + 1
        wall_p = random.uniform(0.05, 0.30)
        maze = (np.random.rand(n, n) < wall_p).astype(np.int8)
        free = [(r, c) for r in range(n) for c in range(n) if maze[r, c] == 0]
        if len(free) < 2:
            continue
        start, goal = random.sample(free, 2)
        dist = bfs_distances(maze, goal)
        path = shortest_path(maze, start, goal, dist)
        if not path:
            continue
        bfs_len = len(path) - 1
        if lo <= bfs_len <= hi:
            return maze, start, goal, dist, path, dict(
                target=target_difficulty,
                actual=target_difficulty,
                bfs_len=bfs_len,
                retries=retries,
                wall_p=wall_p,
            )
    corridor = []
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
    free = [tuple(x) for x in corridor]
    possible = []
    for _ in range(4000):
        start, goal = random.sample(free, 2)
        dist = bfs_distances(maze, goal)
        path = shortest_path(maze, start, goal, dist)
        if not path:
            continue
        bfs_len = len(path) - 1
        if lo <= bfs_len <= hi:
            possible.append((start, goal, dist, path, bfs_len))
            if len(possible) >= 64:
                break
    if possible:
        start, goal, dist, path, bfs_len = random.choice(possible)
        return maze, start, goal, dist, path, dict(
            target=target_difficulty,
            actual=target_difficulty,
            bfs_len=bfs_len,
            retries=retries,
            wall_p="corridor",
        )
    best = None
    target_mid = (lo + hi) // 2
    for _ in range(4000):
        start, goal = random.sample(free, 2)
        dist = bfs_distances(maze, goal)
        path = shortest_path(maze, start, goal, dist)
        if not path:
            continue
        bfs_len = len(path) - 1
        score = abs(bfs_len - target_mid)
        if best is None or score < best[0]:
            best = (score, start, goal, dist, path, bfs_len)
    if best is None:
        raise RuntimeError("fallback maze generation failed")
    _, start, goal, dist, path, bfs_len = best
    actual = difficulty_of_len(args, bfs_len) or target_difficulty
    return maze, start, goal, dist, path, dict(
        target=target_difficulty,
        actual=actual,
        bfs_len=bfs_len,
        retries=retries,
        wall_p="corridor_closest",
    )


# =============================================================================
# 4. BFS tools
# =============================================================================

def in_bounds(p: Tuple[int, int], n: int) -> bool:
    return 0 <= p[0] < n and 0 <= p[1] < n


def neighbors(p: Tuple[int, int], n: int):
    for a, (dr, dc) in enumerate(ACTIONS):
        q = (p[0] + dr, p[1] + dc)
        if in_bounds(q, n):
            yield a, q


def step_env(maze: np.ndarray, pos: Tuple[int, int], action: int):
    n = maze.shape[0]
    dr, dc = ACTIONS[action]
    q = (pos[0] + dr, pos[1] + dc)
    if not in_bounds(q, n) or maze[q] == 1:
        return pos, True
    return q, False


def bfs_distances(maze: np.ndarray, goal: Tuple[int, int]) -> np.ndarray:
    n = maze.shape[0]
    dist = np.full((n, n), INF, dtype=np.int32)
    if maze[goal] == 1:
        return dist
    dq = deque([goal])
    dist[goal] = 0
    while dq:
        p = dq.popleft()
        for _, q in neighbors(p, n):
            if maze[q] == 0 and dist[q] > dist[p] + 1:
                dist[q] = dist[p] + 1
                dq.append(q)
    return dist


def shortest_path(maze, start, goal, dist) -> Optional[List[Tuple[int, int]]]:
    if dist[start] >= INF:
        return None
    n = maze.shape[0]
    p = start
    path = [p]
    for _ in range(n * n + 10):
        if p == goal:
            return path
        cand = [q for _, q in neighbors(p, n) if maze[q] == 0 and dist[q] == dist[p] - 1]
        if not cand:
            return None
        p = random.choice(cand)
        path.append(p)
    return path if path and path[-1] == goal else None


def action_between(p, q) -> int:
    d = (q[0] - p[0], q[1] - p[1])
    for i, a in enumerate(ACTIONS):
        if a == d:
            return i
    raise ValueError(f"not adjacent {p}->{q}")


def path_to_actions(path) -> List[int]:
    return [action_between(path[i], path[i + 1]) for i in range(len(path) - 1)]


def bfs_best_actions(maze: np.ndarray, pos: Tuple[int, int], dist: np.ndarray) -> List[int]:
    n = maze.shape[0]
    cur = int(dist[pos])
    if cur >= INF:
        return []
    best: List[int] = []
    for a, (dr, dc) in enumerate(ACTIONS):
        q = (pos[0] + dr, pos[1] + dc)
        if in_bounds(q, n) and maze[q] == 0 and int(dist[q]) < cur:
            best.append(a)
    return best


def bfs_label_action(best_actions: Sequence[int]) -> int:
    if not best_actions:
        return 0
    return int(min(best_actions))


def is_valid_action(maze: np.ndarray, pos: Tuple[int, int], action: int) -> bool:
    n = maze.shape[0]
    dr, dc = ACTIONS[action]
    q = (pos[0] + dr, pos[1] + dc)
    return in_bounds(q, n) and maze[q] == 0


# =============================================================================
# 5. state encoding
# =============================================================================

def encode_state(
    maze: np.ndarray,
    agent_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    input_mode: str = "compact_2ch",
) -> np.ndarray:
    n = maze.shape[0]
    if input_mode == "compact_2ch":
        s = np.zeros((2, n, n), dtype=np.float32)
        s[0] = maze.astype(np.float32)
        s[1, agent_pos[0], agent_pos[1]] = 1.0
        s[1, goal_pos[0], goal_pos[1]] = -1.0
        return s
    if input_mode == "standard_3ch":
        s = np.zeros((3, n, n), dtype=np.float32)
        s[0] = maze.astype(np.float32)
        s[1, agent_pos[0], agent_pos[1]] = 1.0
        s[2, goal_pos[0], goal_pos[1]] = 1.0
        return s
    raise ValueError(f"unknown input_mode: {input_mode}")


# =============================================================================
# 6. supervised dataset
# =============================================================================

@dataclass
class BFSSample:
    input_tensor: np.ndarray
    label_action: int
    best_actions: List[int]
    maze: np.ndarray
    agent_pos: Tuple[int, int]
    goal_pos: Tuple[int, int]
    bfs_dist: int
    difficulty: str
    bfs_len: int
    maze_id: int = 0


def samples_from_shortest_path(
    maze: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    dist: np.ndarray,
    path: List[Tuple[int, int]],
    meta: dict,
    input_mode: str,
    maze_id: int = 0,
) -> List[BFSSample]:
    bfs_len = len(path) - 1
    difficulty = meta.get("actual") or meta.get("target") or "unknown"
    out: List[BFSSample] = []
    for pos in path[:-1]:
        best = bfs_best_actions(maze, pos, dist)
        if not best:
            continue
        label = bfs_label_action(best)
        out.append(
            BFSSample(
                input_tensor=encode_state(maze, pos, goal, input_mode),
                label_action=label,
                best_actions=list(best),
                maze=maze,
                agent_pos=pos,
                goal_pos=goal,
                bfs_dist=int(dist[pos]),
                difficulty=str(difficulty),
                bfs_len=int(bfs_len),
                maze_id=maze_id,
            )
        )
    return out


def build_bfs_supervised_dataset(args, n_mazes: int, desc: str = "Build BFS dataset") -> Tuple[List[BFSSample], List[dict]]:
    samples: List[BFSSample] = []
    maze_records: List[dict] = []
    for mid in maybe_tqdm(range(n_mazes), args, desc=desc, total=n_mazes, leave=True):
        target = choose_target_difficulty(args)
        maze, start, goal, dist, path, meta = generate_maze(args, target)
        meta = dict(meta, maze_id=mid, start=start, goal=goal)
        maze_records.append(meta)
        samples.extend(
            samples_from_shortest_path(
                maze, start, goal, dist, path, meta, args.input_mode, maze_id=mid
            )
        )
    return samples, maze_records


class BFSActionDataset(Dataset):
    def __init__(self, samples: Sequence[BFSSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return {
            "x": torch.from_numpy(s.input_tensor),
            "y": torch.tensor(s.label_action, dtype=torch.long),
            "best_actions": s.best_actions,
            "maze_id": s.maze_id,
            "bfs_len": s.bfs_len,
            "difficulty": s.difficulty,
            "agent_pos": s.agent_pos,
            "goal_pos": s.goal_pos,
        }


def split_samples(samples: Sequence[BFSSample], val_ratio: float):
    items = list(samples)
    random.shuffle(items)
    nv = max(1, int(len(items) * val_ratio))
    val = items[:nv]
    train = items[nv:] or items
    return train, val


def collate_batch(batch: List[dict]) -> dict:
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "best_actions": [b["best_actions"] for b in batch],
        "difficulty": [b["difficulty"] for b in batch],
        "bfs_len": [b["bfs_len"] for b in batch],
    }


# =============================================================================
# 7. evaluation / rollout
# =============================================================================

def _bfs_dist_at(dist: np.ndarray, pos: Tuple[int, int]) -> int:
    return int(dist[pos]) if dist[pos] < INF else 999


def classify_failure_type(
    success: bool,
    loop_detected: bool,
    wall_hits: int,
    steps: int,
    start_bfs_dist: int,
    final_bfs_dist: int,
) -> str:
    if success:
        return "success"
    if loop_detected:
        return "loop_timeout"
    if final_bfs_dist > start_bfs_dist:
        return "away_timeout"
    if wall_hits >= max(3, steps // 2):
        return "wall_timeout"
    return "max_step_timeout"


def rollout_policy(
    model: torch.nn.Module,
    maze: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    dist: np.ndarray,
    args,
    device: torch.device,
    input_mode: str,
) -> dict:
    bfs_len = _bfs_dist_at(dist, start)
    start_bfs_dist = bfs_len
    pos = start
    trajectory: List[dict] = []
    wall_hits = 0
    seen = {start: 1}
    loop_detected = False

    model.eval()
    for step_idx in range(args.max_steps):
        pos_before = pos
        bfs_before = _bfs_dist_at(dist, pos_before)
        x = encode_state(maze, pos_before, goal, input_mode)
        with torch.no_grad():
            logits = model(torch.from_numpy(x[None]).to(device))
            action = int(logits.argmax(dim=1).item())
        nxt, hit = step_env(maze, pos_before, action)
        pos_after = pos_before if hit else nxt
        if hit:
            wall_hits += 1
        else:
            pos = nxt
            seen[pos] = seen.get(pos, 0) + 1
            if seen[pos] > 1:
                loop_detected = True
        trajectory.append(
            dict(
                step=step_idx,
                pos_before=tuple(pos_before),
                pos_after=tuple(pos_after),
                action=action,
                action_name=ACTION_NAMES[action],
                is_wall=bool(hit),
                bfs_dist_before=bfs_before,
                bfs_dist_after=_bfs_dist_at(dist, pos_after),
            )
        )
        if pos == goal:
            break

    steps = len(trajectory)
    success = pos == goal
    final_bfs_dist = _bfs_dist_at(dist, pos)
    shortest_path_success = bool(success and steps == bfs_len)
    failure_type = classify_failure_type(
        success, loop_detected, wall_hits, steps, start_bfs_dist, final_bfs_dist
    )
    difficulty = difficulty_of_len(args, bfs_len) or "unknown"

    return dict(
        success=bool(success),
        steps=int(steps),
        bfs_len=int(bfs_len),
        bfs_gap=int(max(0, steps - bfs_len)) if success else int(max(0, final_bfs_dist)),
        shortest_path_success=shortest_path_success,
        wall_hits=int(wall_hits),
        loop_detected=bool(loop_detected),
        final_bfs_dist=int(final_bfs_dist),
        difficulty=str(difficulty),
        failure_type=failure_type,
        trajectory=trajectory,
        start=start,
        goal=goal,
    )


def run_rollout_evaluation(
    model: torch.nn.Module,
    args,
    device: torch.device,
    n_mazes: int,
) -> Tuple[List[dict], dict]:
    rollouts: List[dict] = []
    for mid in maybe_tqdm(range(n_mazes), args, desc="Rollout eval", total=n_mazes):
        target = choose_target_difficulty(args)
        maze, start, goal, dist, path, meta = generate_maze(args, target)
        ro = rollout_policy(model, maze, start, goal, dist, args, device, args.input_mode)
        ro["maze_id"] = mid
        ro["meta"] = meta
        rollouts.append(ro)
    return rollouts, summarize_rollouts(rollouts)


def evaluate_on_samples(model, samples: Sequence[BFSSample], device, args) -> dict:
    model.eval()
    hard_correct = 0
    best_correct = 0
    valid = 0
    wall = 0
    total = len(samples)
    with torch.no_grad():
        for s in samples:
            x = torch.from_numpy(s.input_tensor[None]).to(device)
            pred = int(model(x).argmax(dim=1).item())
            if pred == s.label_action:
                hard_correct += 1
            if pred in s.best_actions:
                best_correct += 1
            if is_valid_action(s.maze, s.agent_pos, pred):
                valid += 1
            else:
                wall += 1
    n = max(1, total)
    return dict(
        hard_action_accuracy=hard_correct / n,
        best_action_accuracy=best_correct / n,
        valid_action_rate=valid / n,
        wall_action_rate=wall / n,
        n_samples=total,
    )


def summarize_rollouts(rollouts: Sequence[dict]) -> dict:
    if not rollouts:
        return dict(
            rollout_success_rate=float("nan"),
            shortest_path_success_rate=float("nan"),
            avg_bfs_gap=float("nan"),
            avg_wall_hits=float("nan"),
            difficulty_breakdown={},
            bfs_len_breakdown={},
            failure_type_breakdown={},
        )
    diff_bd: Dict[str, dict] = defaultdict(lambda: dict(n=0, success=0, shortest_path_success=0, bfs_gap=[]))
    len_bd: Dict[str, dict] = defaultdict(lambda: dict(n=0, success=0, bfs_gap=[]))
    fail_bd: Dict[str, int] = defaultdict(int)
    for ro in rollouts:
        d = ro.get("difficulty", "unknown")
        diff_bd[d]["n"] += 1
        diff_bd[d]["success"] += int(ro["success"])
        diff_bd[d]["shortest_path_success"] += int(ro["shortest_path_success"])
        diff_bd[d]["bfs_gap"].append(ro["bfs_gap"])
        bl = str(ro["bfs_len"])
        len_bd[bl]["n"] += 1
        len_bd[bl]["success"] += int(ro["success"])
        len_bd[bl]["bfs_gap"].append(ro["bfs_gap"])
        fail_bd[ro.get("failure_type", "unknown")] += 1
    difficulty_breakdown = {
        k: dict(
            n=v["n"],
            success_rate=v["success"] / max(1, v["n"]),
            shortest_path_success_rate=v["shortest_path_success"] / max(1, v["n"]),
            avg_bfs_gap=mean(v["bfs_gap"]),
        )
        for k, v in sorted(diff_bd.items())
    }
    bfs_len_breakdown = {
        k: dict(
            n=v["n"],
            success_rate=v["success"] / max(1, v["n"]),
            avg_bfs_gap=mean(v["bfs_gap"]),
        )
        for k, v in sorted(len_bd.items(), key=lambda x: int(x[0]))
    }
    failure_type_breakdown = dict(sorted(fail_bd.items(), key=lambda x: -x[1]))
    return dict(
        rollout_success_rate=mean([float(r["success"]) for r in rollouts]),
        shortest_path_success_rate=mean([float(r["shortest_path_success"]) for r in rollouts]),
        avg_bfs_gap=mean([float(r["bfs_gap"]) for r in rollouts]),
        avg_wall_hits=mean([float(r["wall_hits"]) for r in rollouts]),
        avg_steps=mean([float(r["steps"]) for r in rollouts]),
        difficulty_breakdown=difficulty_breakdown,
        bfs_len_breakdown=bfs_len_breakdown,
        failure_type_breakdown=failure_type_breakdown,
    )


def summarize_maze_generation(args, maze_records: Sequence[dict]) -> dict:
    """Summarize dataset maze generation: difficulty ratios, retries, bfs_len stats."""
    total = max(1, len(maze_records))
    all_bfs = [int(r["bfs_len"]) for r in maze_records]
    all_ret = [int(r.get("retries", 0)) for r in maze_records]
    target_map = {
        "easy": float(args.difficulty_easy),
        "medium": float(args.difficulty_medium),
        "hard": float(args.difficulty_hard),
    }
    out: Dict[str, dict] = {}
    for diff, target_ratio in target_map.items():
        rows = [r for r in maze_records if r.get("actual") == diff]
        bfs = [int(r["bfs_len"]) for r in rows]
        ret = [int(r.get("retries", 0)) for r in rows]
        out[diff] = dict(
            difficulty=diff,
            target_ratio=target_ratio,
            actual_ratio=len(rows) / total,
            n=len(rows),
            bfs_len_min=min(bfs) if bfs else None,
            bfs_len_p50=qtile(bfs, 0.5),
            bfs_len_max=max(bfs) if bfs else None,
            bfs_len_mean=mean(bfs),
            retry_mean=mean(ret),
            retry_min=min(ret) if ret else None,
            retry_p50=qtile(ret, 0.5),
            retry_max=max(ret) if ret else None,
        )
    mismatch = sum(1 for r in maze_records if r.get("target") != r.get("actual"))
    out["overall"] = dict(
        difficulty="overall",
        target_ratio=1.0,
        actual_ratio=1.0,
        n=len(maze_records),
        bfs_len_min=min(all_bfs) if all_bfs else None,
        bfs_len_p50=qtile(all_bfs, 0.5),
        bfs_len_max=max(all_bfs) if all_bfs else None,
        bfs_len_mean=mean(all_bfs),
        retry_mean=mean(all_ret),
        retry_min=min(all_ret) if all_ret else None,
        retry_p50=qtile(all_ret, 0.5),
        retry_max=max(all_ret) if all_ret else None,
        target_actual_mismatch=mismatch,
        target_actual_mismatch_rate=mismatch / total,
    )
    return out


def print_maze_generation_report(report: dict) -> None:
    rows = [report[k] for k in ("easy", "medium", "hard", "overall") if k in report]
    print_table(
        "[Maze Generation Report]",
        [
            ("difficulty", 10, "s"),
            ("target_ratio", 10, ".3f"),
            ("actual_ratio", 10, ".3f"),
            ("n", 6, "d"),
            ("bfs_len_min", 10, "d"),
            ("bfs_len_p50", 10, ".1f"),
            ("bfs_len_max", 10, "d"),
            ("retry_mean", 10, ".2f"),
            ("retry_min", 9, "d"),
            ("retry_p50", 9, ".1f"),
            ("retry_max", 9, "d"),
        ],
        rows,
    )
    overall = report.get("overall", {})
    if overall.get("target_actual_mismatch", 0):
        print(
            f"  [note] target/actual difficulty mismatch: "
            f"{overall['target_actual_mismatch']} ({100 * overall.get('target_actual_mismatch_rate', 0):.2f}%)"
        )


def build_evaluation_summary(
    args,
    val_metrics: dict,
    rollout_summary: dict,
    maze_records: Sequence[dict],
) -> dict:
    return dict(
        hard_action_accuracy=val_metrics.get("hard_action_accuracy"),
        best_action_accuracy=val_metrics.get("best_action_accuracy"),
        valid_action_rate=val_metrics.get("valid_action_rate"),
        wall_action_rate=val_metrics.get("wall_action_rate"),
        rollout_success_rate=rollout_summary.get("rollout_success_rate"),
        shortest_path_success_rate=rollout_summary.get("shortest_path_success_rate"),
        avg_bfs_gap=rollout_summary.get("avg_bfs_gap"),
        difficulty_breakdown=rollout_summary.get("difficulty_breakdown"),
        bfs_len_breakdown=rollout_summary.get("bfs_len_breakdown"),
        failure_type_breakdown=rollout_summary.get("failure_type_breakdown"),
        dataset_val=val_metrics,
        rollout=rollout_summary,
        maze_generation=summarize_maze_generation(args, maze_records),
    )


def print_evaluation_tables(eval_summary: dict, rollout_summary: dict) -> None:
    print_table(
        "[Evaluation Summary]",
        [("metric", 28, "s"), ("value", 12, ".4f")],
        [
            {"metric": "hard_action_accuracy", "value": eval_summary.get("hard_action_accuracy")},
            {"metric": "best_action_accuracy", "value": eval_summary.get("best_action_accuracy")},
            {"metric": "valid_action_rate", "value": eval_summary.get("valid_action_rate")},
            {"metric": "wall_action_rate", "value": eval_summary.get("wall_action_rate")},
            {"metric": "rollout_success_rate", "value": eval_summary.get("rollout_success_rate")},
            {"metric": "shortest_path_success_rate", "value": eval_summary.get("shortest_path_success_rate")},
            {"metric": "avg_bfs_gap", "value": eval_summary.get("avg_bfs_gap")},
        ],
    )
    br = []
    for diff, row in rollout_summary.get("difficulty_breakdown", {}).items():
        br.append(
            dict(
                difficulty=diff,
                n=row.get("n"),
                success=row.get("success_rate"),
                shortest_path_success=row.get("shortest_path_success_rate"),
                avg_bfs_gap=row.get("avg_bfs_gap"),
            )
        )
    print_table(
        "[Difficulty Breakdown]",
        [("difficulty", 12, "s"), ("n", 6, "d"), ("success", 10, ".4f"), ("shortest_path_success", 22, ".4f"), ("avg_bfs_gap", 12, ".3f")],
        br,
    )
    ft = [dict(failure_type=k, count=v) for k, v in rollout_summary.get("failure_type_breakdown", {}).items()]
    print_table(
        "[Failure Type Breakdown]",
        [("failure_type", 18, "s"), ("count", 8, "d")],
        ft,
    )


# =============================================================================
# 8. manual test / survey
# =============================================================================

DEFAULT_TEST_MAZE_TEXT = """
........
##.####.
...#.#..
.#S#G#.#
..#..#..
.##.###.
.#...#..
.#.#....
"""

TEST_STRATEGIES = ("argmax", "bfs_tiebreak")


def parse_manual_maze(text: str):
    """Parse ASCII maze: '.' free, '#' wall, 'S' start, 'G' goal."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    n = len(lines)
    maze = np.zeros((n, n), dtype=np.int8)
    start = goal = None
    for r, line in enumerate(lines):
        if len(line) != n:
            raise ValueError(f"manual maze row {r} width {len(line)} != height {n}")
        for c, ch in enumerate(line):
            if ch == "#":
                maze[r, c] = 1
            elif ch == "S":
                start = (r, c)
            elif ch == "G":
                goal = (r, c)
            elif ch != ".":
                raise ValueError(f"unknown char {ch!r} at ({r},{c})")
    if start is None or goal is None:
        raise ValueError("manual maze needs S (start) and G (goal)")
    return maze, start, goal


def format_manual_maze(maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> str:
    n = maze.shape[0]
    lines = []
    for r in range(n):
        row = []
        for c in range(n):
            if (r, c) == start:
                row.append("S")
            elif (r, c) == goal:
                row.append("G")
            else:
                row.append("#" if maze[r, c] == 1 else ".")
        lines.append("".join(row))
    return "\n".join(lines)


def load_manual_maze_text(args) -> str:
    path = getattr(args, "manual_maze_file", None)
    if path:
        return Path(path).read_text(encoding="utf-8")
    custom = getattr(args, "manual_maze_text", None)
    if custom:
        return custom
    return DEFAULT_TEST_MAZE_TEXT


def action_infos_for_state(
    model: torch.nn.Module,
    maze: np.ndarray,
    goal: Tuple[int, int],
    dist: np.ndarray,
    pos: Tuple[int, int],
    args,
    device: torch.device,
    input_mode: str,
) -> List[dict]:
    """Per-action survey at current state: logits, legality, BFS delta."""
    model.eval()
    cur_bfs = _bfs_dist_at(dist, pos)
    best_set = set(bfs_best_actions(maze, pos, dist))
    x = encode_state(maze, pos, goal, input_mode)
    with torch.no_grad():
        logits_t = model(torch.from_numpy(x[None]).to(device)).squeeze(0)
        logits = logits_t.detach().cpu().numpy().tolist()
    probs = torch.softmax(logits_t, dim=0).detach().cpu().numpy().tolist()
    rows: List[dict] = []
    for a in range(4):
        nxt, hit = step_env(maze, pos, a)
        pos_after = pos if hit else nxt
        nd = _bfs_dist_at(dist, pos_after)
        rows.append(
            dict(
                action_idx=a,
                action=ACTION_NAMES[a],
                logit=float(logits[a]),
                prob=float(probs[a]),
                is_wall=bool(hit),
                is_valid=not hit,
                next_pos=list(pos_after),
                delta_bfs=int(nd - cur_bfs),
                is_bfs_best=a in best_set,
            )
        )
    ranked = sorted(rows, key=lambda z: z["logit"], reverse=True)
    rank_map = {id(r): i + 1 for i, r in enumerate(ranked)}
    for r in rows:
        r["rank"] = rank_map[id(r)]
    return rows


def choose_test_action(rows: Sequence[dict], strategy: str, tie_eps: float = 0.03) -> dict:
    if strategy == "bfs_tiebreak":
        mx = max(r["logit"] for r in rows)
        cand = [r for r in rows if mx - r["logit"] <= tie_eps]
        cand.sort(key=lambda r: (r["delta_bfs"], -r["logit"], r["action_idx"]))
        return cand[0]
    if strategy != "argmax":
        raise ValueError(f"unknown test strategy: {strategy}")
    return max(rows, key=lambda r: r["logit"])


def render_test_frame(maze: np.ndarray, pos: Tuple[int, int], goal: Tuple[int, int], title: str):
    n = maze.shape[0]
    img = np.ones((n, n, 3), dtype=np.float32)
    img[maze == 1] = [0.05, 0.05, 0.05]
    img[goal] = [1.0, 0.85, 0.0]
    img[pos] = [1.0, 0.1, 0.1]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8)
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return arr


def save_test_gif(frames: Sequence[np.ndarray], path: Path, fps: int = 3) -> None:
    if not frames:
        return
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.axis("off")
    im = ax.imshow(frames[0])

    def update(i):
        im.set_data(frames[i])
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=int(1000 / max(1, fps)), blit=True
    )
    ani.save(str(path), writer="pillow")
    plt.close(fig)


def print_manual_test_step_table(step: dict) -> None:
    actions = step.get("actions", {})
    rows = [actions[name] for name in ACTION_NAMES if name in actions]
    print_table(
        f"[Manual Test Step {step.get('t')} pos={step.get('pos_before')} strategy={step.get('strategy')}]",
        [
            ("action", 8, "s"),
            ("logit", 8, ".3f"),
            ("prob", 8, ".3f"),
            ("rank", 5, "d"),
            ("is_wall", 8, "d"),
            ("delta_bfs", 10, "d"),
            ("is_bfs_best", 12, "d"),
        ],
        rows,
    )


def run_manual_test_rollout(
    model: torch.nn.Module,
    args,
    device: torch.device,
    out_dir: Path,
    strategy: str,
    version_tag: str,
    input_mode: str,
    maze_text: Optional[str] = None,
    tie_eps: float = 0.03,
    verbose_steps: bool = True,
) -> dict:
    text = maze_text or DEFAULT_TEST_MAZE_TEXT
    maze, start, goal = parse_manual_maze(text)
    n = maze.shape[0]
    if n != args.grid_size:
        raise ValueError(f"manual maze is {n}x{n} but grid_size={args.grid_size}")
    dist = bfs_distances(maze, goal)
    bfs_len = _bfs_dist_at(dist, start)
    pos = start
    frames: List[np.ndarray] = []
    steps: List[dict] = []
    wall_hits = 0
    success = False

    print(f"\n[Manual Test / {strategy}]")
    print(format_manual_maze(maze, start, goal))
    print(f"  bfs_len={bfs_len} input_mode={input_mode}")

    for t in range(args.max_steps):
        cur = _bfs_dist_at(dist, pos)
        rows = action_infos_for_state(model, maze, goal, dist, pos, args, device, input_mode)
        chosen = choose_test_action(rows, strategy, tie_eps=tie_eps)
        pos_before = pos
        if chosen["is_wall"]:
            wall_hits += 1
            pos_after = pos
        else:
            pos = tuple(chosen["next_pos"])
            pos_after = pos
        step = dict(
            t=t,
            pos_before=list(pos_before),
            pos_after=list(pos_after),
            bfs_dist_before=cur,
            bfs_dist_after=_bfs_dist_at(dist, pos_after),
            chosen_action=chosen["action"],
            chosen_logit=chosen["logit"],
            chosen_prob=chosen["prob"],
            chosen_is_wall=chosen["is_wall"],
            chosen_delta_bfs=chosen["delta_bfs"],
            chosen_is_bfs_best=chosen["is_bfs_best"],
            strategy=strategy,
            actions={
                r["action"]: {k: v for k, v in r.items() if k != "action_idx"}
                for r in rows
            },
        )
        steps.append(step)
        if verbose_steps:
            print_manual_test_step_table(step)
        title = (
            f"step={t} action={chosen['action']} logit={chosen['logit']:.3f} "
            f"is_wall={chosen['is_wall']} delta_bfs={chosen['delta_bfs']} "
            f"bfs_best={chosen['is_bfs_best']} pos={pos_before} strategy={strategy}"
        )
        frames.append(render_test_frame(maze, pos_after, goal, title))
        if pos == goal:
            success = True
            break

    metrics = dict(
        strategy=strategy,
        success=bool(success),
        steps=len(steps),
        wall_hits=wall_hits,
        bfs_len=int(bfs_len),
        bfs_gap=max(0, len(steps) - bfs_len) if success else None,
        final_pos=list(pos),
        final_bfs_dist=_bfs_dist_at(dist, pos),
        shortest_path_success=bool(success and len(steps) == bfs_len),
    )
    ensure_dir(out_dir)
    json_path = out_dir / f"{version_tag}_test_rollout_{strategy}_debug.json"
    gif_path = out_dir / f"{version_tag}_test_rollout_{strategy}.gif"
    save_json(
        json_path,
        dict(
            maze_text=text,
            maze=maze.tolist(),
            start=start,
            goal=goal,
            input_mode=input_mode,
            strategy=strategy,
            tie_eps=tie_eps,
            metrics=metrics,
            steps=steps,
        ),
    )
    save_test_gif(frames, gif_path)
    print(f"[Save] test json: {json_path}")
    print(f"[Save] test gif : {gif_path}")
    return dict(metrics=metrics, json=str(json_path), gif=str(gif_path))


def run_manual_test_mode(
    model: torch.nn.Module,
    args,
    device: torch.device,
    out_dir: Path,
    version_tag: str,
    input_mode: Optional[str] = None,
    maze_text: Optional[str] = None,
) -> dict:
    """Run manual maze survey with one or more rollout strategies."""
    input_mode = input_mode or args.input_mode
    strategy_arg = getattr(args, "test_strategy", "both")
    if strategy_arg == "both":
        strategies = list(TEST_STRATEGIES)
    else:
        strategies = [strategy_arg]
    tie_eps = float(getattr(args, "test_tie_eps", 0.03))
    text = maze_text if maze_text is not None else load_manual_maze_text(args)
    verbose = not getattr(args, "test_quiet_steps", False)

    results = {
        s: run_manual_test_rollout(
            model,
            args,
            device,
            out_dir,
            s,
            version_tag,
            input_mode,
            maze_text=text,
            tie_eps=tie_eps,
            verbose_steps=verbose,
        )
        for s in strategies
    }
    summary_path = out_dir / f"{version_tag}_test_rollout_summary.json"
    save_json(summary_path, results)
    print_table(
        "[Manual Test Summary]",
        [
            ("strategy", 14, "s"),
            ("success", 8, "d"),
            ("steps", 6, "d"),
            ("wall_hits", 10, "d"),
            ("bfs_len", 8, "d"),
            ("bfs_gap", 8, "d"),
            ("shortest_path_success", 22, "d"),
        ],
        [
            dict(
                strategy=s,
                success=int(r["metrics"]["success"]),
                steps=r["metrics"]["steps"],
                wall_hits=r["metrics"]["wall_hits"],
                bfs_len=r["metrics"]["bfs_len"],
                bfs_gap=r["metrics"].get("bfs_gap"),
                shortest_path_success=int(r["metrics"]["shortest_path_success"]),
            )
            for s, r in results.items()
        ],
    )
    print(f"[Save] test summary: {summary_path}")
    return results


# =============================================================================
# 9. reports / audit
# =============================================================================

def _script_version_tag(version: str) -> str:
    return "_".join(version.split("."))


def _common_code_section(common_source: str) -> str:
    """Audit only stable sections, not the audit block itself."""
    marker = "# 9. reports / audit"
    return common_source.split(marker)[0] if marker in common_source else common_source


def _audit_common_compliance(common_source: str) -> List[dict]:
    code = _common_code_section(common_source)
    forbidden = (
        "CNNBFSActionPolicy",
        "def build_model",
        "def train_bfs_cnn",
        "def train_one_epoch",
        "def eval_epoch",
        "DEFAULTS =",
        "PRESETS =",
        "def build_argparser",
        "def apply_preset",
        "def resolved_config_from_args",
        "VERSION =",
        "MODEL_NAME =",
        "class RewardModel",
        "class QNetwork",
        "class ReplayBuffer",
    )
    issues = []
    for pat in forbidden:
        if pat in code:
            issues.append(dict(scope="common", status="FAIL", message=f"forbidden pattern in common: {pat}"))
    return issues


def _audit_experiment_compliance(experiment_source: str) -> List[dict]:
    required = (
        "VERSION =",
        "DEFAULTS",
        "PRESETS",
        "class CNNBFSActionPolicy",
        "def train_one_epoch",
        "def train_model",
    )
    issues = []
    for req in required:
        if req not in experiment_source:
            issues.append(dict(scope="experiment", status="FAIL", message=f"experiment missing required: {req}"))
    return issues


def protocol_compliance_audit(
    script_path: Path,
    version: str,
    output_dir: Path,
    resolved_config: dict,
    required_reports: Sequence[str] = REQUIRED_REPORTS,
    experiment_source: Optional[str] = None,
    common_source: Optional[str] = None,
    loaded_modules: Optional[Sequence[str]] = None,
) -> dict:
    issues: List[dict] = []
    script_s = str(script_path).replace("\\", "/")
    exp_src = experiment_source or script_path.read_text(encoding="utf-8", errors="ignore")
    com_src = common_source or COMMON_PATH.read_text(encoding="utf-8", errors="ignore")

    if "feature_maze/v1_x_bfs_models/experiments" not in script_s:
        issues.append(dict(scope="script_path", status="FAIL", message="script must live under experiments/."))
    if "1_0_0" not in script_path.name:
        issues.append(dict(scope="version", status="FAIL", message="script name must contain 1_0_0."))
    ver_tag = _script_version_tag(version)
    if ver_tag not in script_path.name.replace(".", "_"):
        issues.append(dict(scope="version", status="FAIL", message=f"VERSION {version} must match script name."))
    if not any(x in exp_src for x in ("import v1_common", "from v1_common")):
        issues.append(dict(scope="imports", status="FAIL", message="experiment must import current-gen v1_common."))
    for bad in ("v2_0_reward_model", "v2.0.9", "maze_dqn_v2"):
        if bad in exp_src:
            issues.append(dict(scope="imports", status="FAIL", message=f"forbidden v2 reference: {bad}"))
    for bad in ("RewardModel", "QNetwork", "ReplayBuffer", "enable_qcnn", "train_qcnn", "BTL", "primitive_pressure"):
        if bad in exp_src:
            issues.append(dict(scope="forbidden_stage", status="FAIL", message=f"forbidden RM/DQN pattern: {bad}"))
    issues.extend(_audit_common_compliance(com_src))
    issues.extend(_audit_experiment_compliance(exp_src))
    if resolved_config.get("input_mode") not in INPUT_MODES:
        issues.append(dict(scope="input_mode", status="FAIL", message="input_mode missing or invalid."))
    out_s = str(output_dir).replace("\\", "/")
    if "feature_maze/v1_x_bfs_models/outputs" not in out_s:
        issues.append(dict(scope="output_dir", status="FAIL", message="output must be under v1_x_bfs_models/outputs/."))
    if loaded_modules:
        for mod in loaded_modules:
            ms = mod.replace("\\", "/")
            if any(x in ms for x in ("v2_0_reward_model", "v2.0.9", "v2.0.n")):
                issues.append(dict(scope="runtime_import", status="FAIL", message=f"loaded forbidden module: {mod}"))
    if not required_reports:
        issues.append(dict(scope="reports", status="FAIL", message="required reports not registered."))
    if "protocol_compliance_audit.json" not in required_reports:
        issues.append(dict(scope="reports", status="FAIL", message="protocol_compliance_audit.json not registered."))
    return dict(status="PASS" if not issues else "FAIL", issues=issues, required_reports=list(required_reports))


def print_protocol_compliance_audit(audit: dict) -> None:
    print("\n[Protocol Compliance Audit]")
    print(f"status: {audit.get('status', 'UNKNOWN')}")
    for issue in audit.get("issues", []):
        print(f"[{issue.get('status', 'FAIL')}] {issue.get('scope', '')}: {issue.get('message', '')}")


def save_required_reports(
    output_dir: Path,
    resolved_config: dict,
    run_metadata: dict,
    training_history: dict,
    evaluation_summary: dict,
    rollout_debug: dict,
    protocol_compliance_audit_report: dict,
) -> None:
    ensure_dir(output_dir)
    save_json(output_dir / "resolved_config.json", resolved_config)
    save_json(output_dir / "run_metadata.json", run_metadata)
    save_json(output_dir / "training_history.json", training_history)
    save_json(output_dir / "evaluation_summary.json", evaluation_summary)
    save_json(output_dir / "rollout_debug.json", rollout_debug)
    save_json(output_dir / "protocol_compliance_audit.json", protocol_compliance_audit_report)
