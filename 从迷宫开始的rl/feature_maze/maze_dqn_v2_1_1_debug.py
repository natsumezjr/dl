
import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# ============================================================
# v2.1.1 Debug-only DQN
# Purpose:
#   Test why v2.1.0 / previous BFS demonstration replay still loops.
#   This file does NOT optimize the policy, does NOT train, and does NOT
#   change the model. It only loads an existing trained model and runs
#   diagnostics.
# ============================================================

VERSION = "v2.1.1_debug"

SIZE = 8

ACTIONS = [
    (-1, 0),  # UP
    (1, 0),   # DOWN
    (0, -1),  # LEFT
    (0, 1),   # RIGHT
]

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]


# ============================================================
# Config
# ============================================================

@dataclass
class DebugConfig:
    output_dir: str = "./v2.1.1"
    output_name: str = "v2.1.1_debug"

    model_path: str = "./v2.1.0/v2.1.0_bfs_demo_replay.pt"

    fallback_model_paths: Tuple[str, ...] = ()

    seed: int = 42
    max_steps: int = 64
    gamma: float = 0.99

    goal_reward: float = 10.0
    wall_penalty: float = -10.0
    step_penalty: float = -0.01

    eval_n: int = 30
    manual_n: int = 10
    gif_difficulty: str = "medium"

    # Q diagnostics.
    q_margin_bad_threshold: float = 0.0

    # If true, save one GIF for manual debug.
    save_gif: bool = True


# ============================================================
# Manual topology-only mazes
# Only three classes: easy / medium / hard.
# No fixed S/G in templates.
# ============================================================

MANUAL_MAZES: Dict[str, List[str]] = {
    "easy": [
        """
........
........
........
........
........
........
........
........
""",
        """
........
........
..##....
..##....
........
....##..
........
........
""",
        """
........
.....#..
.....#..
........
..#.....
..#.....
........
........
""",
        """
........
........
...##...
........
........
....##..
........
........
""",
    ],

    "medium": [
        """
....#...
....#...
....#...
....#...
........
....#...
....#...
....#...
""",
        """
........
........
###.####
........
....####
....#...
....#...
........
""",
        """
........
.#####..
.#......
.#.#####
.#.....#
.#####.#
.......#
######..
""",
        """
........
######..
.....#..
.###.#..
.#...#..
.#.###..
.#......
.#......
""",
    ],

    "hard": [
        """
.......#
######.#
#......#
#.######
#......#
######.#
#......#
######..
""",
        """
.......#
######.#
#......#
#.######
#......#
######.#
#......#
#.######
""",
        """
........
.#####.#
.#.....#
.#.###.#
.#.#...#
.#.#.###
.#.....#
.#####..
""",
        """
.......#
.#####.#
.#...#.#
.#.#.#.#
.#.#...#
.#.#####
.#......
.######.
""",
    ],
}


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_output_dir(cfg: DebugConfig) -> Path:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def in_bounds(pos: Tuple[int, int]) -> bool:
    r, c = pos
    return 0 <= r < SIZE and 0 <= c < SIZE


def is_free(grid: np.ndarray, pos: Tuple[int, int]) -> bool:
    r, c = pos
    return in_bounds(pos) and grid[r, c] == 0


def parse_topology_text(text: str) -> np.ndarray:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) != SIZE:
        raise ValueError(f"manual topology must have {SIZE} rows, got {len(lines)}")

    grid = np.zeros((SIZE, SIZE), dtype=np.int64)
    for r, line in enumerate(lines):
        if len(line) != SIZE:
            raise ValueError(f"row {r} must have {SIZE} chars, got {len(line)}: {line}")
        for c, ch in enumerate(line):
            if ch == "#":
                grid[r, c] = 1
            elif ch == ".":
                grid[r, c] = 0
            elif ch in ("S", "G"):
                raise ValueError("manual topology must not contain fixed S/G")
            else:
                raise ValueError(f"invalid char {ch!r}")
    return grid


def transform_grid_randomly(
    grid: np.ndarray,
) -> np.ndarray:
    g = grid.copy()
    if random.random() < 0.5:
        g = np.fliplr(g)
    if random.random() < 0.5:
        g = np.flipud(g)
    return g.copy()


def bfs_shortest_path(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    q = [start]
    parent = {start: None}
    head = 0

    while head < len(q):
        cur = q[head]
        head += 1

        if cur == goal:
            break

        r, c = cur
        for dr, dc in ACTIONS:
            nxt = (r + dr, c + dc)
            if not is_free(grid, nxt):
                continue
            if nxt not in parent:
                parent[nxt] = cur
                q.append(nxt)

    if goal not in parent:
        return None

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def bfs_distance(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[int]:
    path = bfs_shortest_path(grid, start, goal)
    if path is None:
        return None
    return len(path) - 1


def valid_actions(grid: np.ndarray, pos: Tuple[int, int]) -> List[int]:
    acts = []
    r, c = pos
    for a, (dr, dc) in enumerate(ACTIONS):
        nxt = (r + dr, c + dc)
        if is_free(grid, nxt):
            acts.append(a)
    return acts


def bfs_optimal_actions(grid: np.ndarray, pos: Tuple[int, int], goal: Tuple[int, int]) -> List[int]:
    d0 = bfs_distance(grid, pos, goal)
    if d0 is None:
        return []

    acts = []
    r, c = pos
    for a, (dr, dc) in enumerate(ACTIONS):
        nxt = (r + dr, c + dc)
        if not is_free(grid, nxt):
            continue
        d1 = bfs_distance(grid, nxt, goal)
        if d1 is not None and d1 < d0:
            acts.append(a)
    return acts


def action_between_positions(pos: Tuple[int, int], nxt: Tuple[int, int]) -> int:
    dr = nxt[0] - pos[0]
    dc = nxt[1] - pos[1]
    for a, (adr, adc) in enumerate(ACTIONS):
        if (dr, dc) == (adr, adc):
            return a
    raise ValueError(f"positions are not adjacent: {pos} -> {nxt}")


def state_from_grid_pos_goal(
    grid: np.ndarray,
    pos: Tuple[int, int],
    goal: Tuple[int, int],
) -> np.ndarray:
    walls = grid.astype(np.float32)

    agent = np.zeros((SIZE, SIZE), dtype=np.float32)
    agent[pos] = 1.0

    goal_ch = np.zeros((SIZE, SIZE), dtype=np.float32)
    goal_ch[goal] = 1.0

    return np.stack([walls, agent, goal_ch], axis=0)


def sample_random_start_goal_on_grid(
    grid: np.ndarray,
    min_manhattan: int = 6,
    min_shortest_len: int = 6,
    max_tries: int = 1000,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    free = [tuple(map(int, p)) for p in np.argwhere(grid == 0)]
    if len(free) < 2:
        raise RuntimeError("not enough free cells to sample start/goal")

    for _ in range(max_tries):
        start = random.choice(free)
        goal = random.choice(free)
        if start == goal:
            continue
        manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        if manhattan < min_manhattan:
            continue
        path = bfs_shortest_path(grid, start, goal)
        if path is None:
            continue
        if len(path) - 1 < min_shortest_len:
            continue
        return start, goal

    raise RuntimeError("failed to sample random start/goal")


def sample_manual_maze(
    difficulty: str = "medium",
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], str, str]:
    if difficulty not in MANUAL_MAZES:
        raise ValueError(f"unknown manual difficulty: {difficulty}")

    topology = random.choice(MANUAL_MAZES[difficulty])
    grid = parse_topology_text(topology)
    grid = transform_grid_randomly(grid)

    min_len_by_diff = {
        "easy": 6,
        "medium": 8,
        "hard": 10,
    }

    start, goal = sample_random_start_goal_on_grid(
        grid,
        min_manhattan=6,
        min_shortest_len=min_len_by_diff[difficulty],
    )

    return grid, start, goal, f"manual_{difficulty}", difficulty


def random_free_cell(grid: np.ndarray) -> Tuple[int, int]:
    free = np.argwhere(grid == 0)
    idx = random.randrange(len(free))
    r, c = free[idx]
    return int(r), int(c)


def generate_random_maze(
    difficulty: str,
    max_tries: int = 5000,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], str, str]:
    if difficulty == "easy":
        obstacle_min, obstacle_max = 0.05, 0.12
        min_len, max_len = 8, 18
    elif difficulty == "medium":
        obstacle_min, obstacle_max = 0.12, 0.22
        min_len, max_len = 12, 32
    elif difficulty == "hard":
        obstacle_min, obstacle_max = 0.20, 0.30
        min_len, max_len = 18, 48
    else:
        raise ValueError(f"unknown difficulty: {difficulty}")

    for _ in range(max_tries):
        p = random.uniform(obstacle_min, obstacle_max)
        grid = (np.random.rand(SIZE, SIZE) < p).astype(np.int64)

        for _inner in range(200):
            start = random_free_cell(grid)
            goal = random_free_cell(grid)
            if start == goal:
                continue
            manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
            if manhattan < 6:
                continue
            path = bfs_shortest_path(grid, start, goal)
            if path is None:
                continue
            length = len(path) - 1
            if min_len <= length <= max_len:
                return grid, start, goal, f"random_{difficulty}", difficulty

    raise RuntimeError(f"failed to generate random maze: {difficulty}")


# ============================================================
# Model
# ============================================================

class CNN_DQN(nn.Module):
    def __init__(self, in_channels: int = 3, num_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(64 * SIZE * SIZE, 256),
            nn.ReLU(),

            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        return self.net(x)


def resolve_model_path(cfg: DebugConfig) -> Path:
    candidates = [Path(cfg.model_path)] + [Path(p) for p in cfg.fallback_model_paths]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No model found. Tried:\n" + "\n".join(str(p) for p in candidates)
    )


def load_model(cfg: DebugConfig, device: torch.device) -> CNN_DQN:
    model_path = resolve_model_path(cfg)
    model = CNN_DQN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[Load] model: {model_path}")
    return model


def q_values_for_pos(
    model: CNN_DQN,
    grid: np.ndarray,
    pos: Tuple[int, int],
    goal: Tuple[int, int],
    device: torch.device,
) -> np.ndarray:
    state = state_from_grid_pos_goal(grid, pos, goal)
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q = model(s)[0].detach().cpu().numpy()
    return q


def one_step_td_error_for_action(
    model: CNN_DQN,
    grid: np.ndarray,
    pos: Tuple[int, int],
    action: int,
    goal: Tuple[int, int],
    cfg: DebugConfig,
    device: torch.device,
) -> Dict[str, float]:
    q = q_values_for_pos(model, grid, pos, goal, device)
    q_sa = float(q[action])

    r, c = pos
    dr, dc = ACTIONS[action]
    nxt = (r + dr, c + dc)

    if not is_free(grid, nxt):
        reward = cfg.step_penalty + cfg.wall_penalty
        done = False
        nxt = pos
    elif nxt == goal:
        reward = cfg.step_penalty + cfg.goal_reward
        done = True
    else:
        reward = cfg.step_penalty
        done = False

    if done:
        target = reward
    else:
        q_next = q_values_for_pos(model, grid, nxt, goal, device)
        next_action = int(np.argmax(q_next))
        target = reward + cfg.gamma * float(q_next[next_action])

    td_error = target - q_sa

    return {
        "q_sa": q_sa,
        "target": float(target),
        "td_error": float(td_error),
        "abs_td_error": float(abs(td_error)),
    }


# ============================================================
# Diagnostics
# ============================================================

def diagnose_bfs_path_q_ranking(
    model: CNN_DQN,
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    cfg: DebugConfig,
    device: torch.device,
    maze_id: str,
) -> Dict[str, Any]:
    path = bfs_shortest_path(grid, start, goal)
    if path is None or len(path) < 2:
        return {
            "maze_id": maze_id,
            "valid": False,
            "reason": "no_bfs_path",
        }

    rows = []
    agree = 0
    valid_steps = 0
    margins = []
    td_abs = []
    q_bfs_values = []
    q_max_values = []
    bad_examples = []

    for i in range(len(path) - 1):
        pos = path[i]
        nxt = path[i + 1]
        bfs_action = action_between_positions(pos, nxt)

        q = q_values_for_pos(model, grid, pos, goal, device)
        greedy_action = int(np.argmax(q))
        q_bfs = float(q[bfs_action])
        other_actions = [a for a in range(4) if a != bfs_action]
        q_best_other = max(float(q[a]) for a in other_actions)
        margin = q_bfs - q_best_other

        td = one_step_td_error_for_action(
            model=model,
            grid=grid,
            pos=pos,
            action=bfs_action,
            goal=goal,
            cfg=cfg,
            device=device,
        )

        d0 = bfs_distance(grid, pos, goal)
        d1 = bfs_distance(grid, nxt, goal)

        is_agree = greedy_action == bfs_action
        agree += int(is_agree)
        valid_steps += 1
        margins.append(margin)
        td_abs.append(td["abs_td_error"])
        q_bfs_values.append(q_bfs)
        q_max_values.append(float(np.max(q)))

        row = {
            "maze_id": maze_id,
            "path_index": i,
            "pos": str(pos),
            "next": str(nxt),
            "bfs_action": ACTION_NAMES[bfs_action],
            "greedy_action": ACTION_NAMES[greedy_action],
            "agree": bool(is_agree),
            "q_up": float(q[0]),
            "q_down": float(q[1]),
            "q_left": float(q[2]),
            "q_right": float(q[3]),
            "q_bfs": q_bfs,
            "q_best_other": q_best_other,
            "q_margin": margin,
            "td_error": td["td_error"],
            "abs_td_error": td["abs_td_error"],
            "bfs_distance": d0,
            "next_bfs_distance": d1,
        }
        rows.append(row)

        if (not is_agree or margin <= cfg.q_margin_bad_threshold) and len(bad_examples) < 8:
            bad_examples.append(row)

    agreement = agree / max(1, valid_steps)
    negative_margin_ratio = sum(1 for m in margins if m <= 0) / max(1, len(margins))

    return {
        "maze_id": maze_id,
        "valid": True,
        "path_len": len(path) - 1,
        "bfs_path_q_agreement": agreement,
        "negative_margin_ratio": negative_margin_ratio,
        "mean_q_margin": float(np.mean(margins)) if margins else None,
        "min_q_margin": float(np.min(margins)) if margins else None,
        "mean_abs_demo_td_error": float(np.mean(td_abs)) if td_abs else None,
        "mean_q_bfs": float(np.mean(q_bfs_values)) if q_bfs_values else None,
        "mean_q_max": float(np.mean(q_max_values)) if q_max_values else None,
        "rows": rows,
        "bad_examples": bad_examples,
    }


def greedy_rollout(
    model: CNN_DQN,
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    cfg: DebugConfig,
    device: torch.device,
    maze_id: str,
) -> Dict[str, Any]:
    pos = start
    path = [pos]
    rows = []
    total_reward = 0.0
    wall_hits = 0
    progress_count = 0
    regress_count = 0
    same_count = 0
    bfs_agree = 0
    bfs_total = 0

    transition_counter = Counter()
    first_seen = {pos: 0}
    first_cycle = None

    for t in range(cfg.max_steps):
        q = q_values_for_pos(model, grid, pos, goal, device)
        action = int(np.argmax(q))

        valid = valid_actions(grid, pos)
        bfs_best = bfs_optimal_actions(grid, pos, goal)

        if bfs_best:
            bfs_total += 1
            if action in bfs_best:
                bfs_agree += 1

        old_pos = pos
        old_dist = bfs_distance(grid, old_pos, goal)

        dr, dc = ACTIONS[action]
        nxt = (pos[0] + dr, pos[1] + dc)

        hit_wall = False
        reward = cfg.step_penalty

        if not is_free(grid, nxt):
            hit_wall = True
            wall_hits += 1
            reward += cfg.wall_penalty
            nxt = pos
        else:
            pos = nxt

        if pos == goal:
            reward += cfg.goal_reward
            done = True
        else:
            done = False

        new_dist = bfs_distance(grid, pos, goal)

        if old_dist is not None and new_dist is not None:
            if new_dist < old_dist:
                progress_count += 1
                progress_type = "progress"
            elif new_dist > old_dist:
                regress_count += 1
                progress_type = "regress"
            else:
                same_count += 1
                progress_type = "same"
        else:
            progress_type = "unknown"

        total_reward += reward
        path.append(pos)
        transition_counter[(old_pos, pos)] += 1

        if pos in first_seen and first_cycle is None:
            first_cycle = {
                "start_step": first_seen[pos],
                "end_step": t + 1,
                "cycle_len": (t + 1) - first_seen[pos],
                "cycle_entry_pos": str(pos),
            }
        else:
            first_seen[pos] = t + 1

        q_sorted = sorted(
            [(ACTION_NAMES[a], float(q[a])) for a in range(4)],
            key=lambda x: x[1],
            reverse=True,
        )

        rows.append({
            "maze_id": maze_id,
            "step": t,
            "pos": str(old_pos),
            "action": ACTION_NAMES[action],
            "hit_wall": hit_wall,
            "next_pos": str(pos),
            "reward": float(reward),
            "old_bfs_distance": old_dist,
            "new_bfs_distance": new_dist,
            "progress_type": progress_type,
            "valid_actions": [ACTION_NAMES[a] for a in valid],
            "bfs_best_actions": [ACTION_NAMES[a] for a in bfs_best],
            "bfs_agree": bool(action in bfs_best) if bfs_best else False,
            "q_up": float(q[0]),
            "q_down": float(q[1]),
            "q_left": float(q[2]),
            "q_right": float(q[3]),
            "q_rank": str(q_sorted),
        })

        if done:
            break

    steps = len(rows)
    repeat_count = len(path) - len(set(path))
    success = path[-1] == goal
    top_transitions = [
        {
            "transition": f"{k[0]} -> {k[1]}",
            "count": v,
        }
        for k, v in transition_counter.most_common(8)
    ]

    # 2-cycle / bounce detection:
    # A-B-A-B repeats produce many transitions with reversed pairs.
    bounce_count = 0
    for i in range(2, len(path)):
        if path[i] == path[i - 2] and path[i] != path[i - 1]:
            bounce_count += 1
    bounce_ratio = bounce_count / max(1, len(path) - 2)

    return {
        "maze_id": maze_id,
        "success": bool(success),
        "steps": steps,
        "reward": float(total_reward),
        "wall_hits": wall_hits,
        "wall_hit_rate": wall_hits / max(1, steps),
        "repeat_count": repeat_count,
        "unique_cells": len(set(path)),
        "progress_rate": progress_count / max(1, steps),
        "regress_rate": regress_count / max(1, steps),
        "same_rate": same_count / max(1, steps),
        "bfs_action_agreement_rollout": bfs_agree / max(1, bfs_total),
        "first_cycle": first_cycle,
        "bounce_count": bounce_count,
        "bounce_ratio": bounce_ratio,
        "top_repeated_transitions": top_transitions,
        "path": [str(p) for p in path],
        "rows": rows,
    }


def diagnose_one_maze(
    model: CNN_DQN,
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    cfg: DebugConfig,
    device: torch.device,
    maze_id: str,
    source: str,
    difficulty: str,
) -> Dict[str, Any]:
    path = bfs_shortest_path(grid, start, goal)
    shortest_len = None if path is None else len(path) - 1

    q_rank = diagnose_bfs_path_q_ranking(
        model=model,
        grid=grid,
        start=start,
        goal=goal,
        cfg=cfg,
        device=device,
        maze_id=maze_id,
    )

    rollout = greedy_rollout(
        model=model,
        grid=grid,
        start=start,
        goal=goal,
        cfg=cfg,
        device=device,
        maze_id=maze_id,
    )

    return {
        "maze_id": maze_id,
        "source": source,
        "difficulty": difficulty,
        "start": str(start),
        "goal": str(goal),
        "shortest_len": shortest_len,
        "q_ranking": q_rank,
        "rollout": rollout,
    }


def summarize_reports(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    def get(path, obj, default=None):
        cur = obj
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    valid_reports = [r for r in reports if get(["q_ranking", "valid"], r, False)]

    summary = {
        "n_mazes": len(reports),
        "n_valid_q_ranking": len(valid_reports),
        "rollout_success_rate": float(np.mean([get(["rollout", "success"], r, False) for r in reports])) if reports else None,
        "rollout_mean_repeat_count": float(np.mean([get(["rollout", "repeat_count"], r, 0) for r in reports])) if reports else None,
        "rollout_mean_unique_cells": float(np.mean([get(["rollout", "unique_cells"], r, 0) for r in reports])) if reports else None,
        "rollout_mean_bounce_ratio": float(np.mean([get(["rollout", "bounce_ratio"], r, 0) for r in reports])) if reports else None,
        "rollout_mean_progress_rate": float(np.mean([get(["rollout", "progress_rate"], r, 0) for r in reports])) if reports else None,
        "rollout_mean_bfs_agreement": float(np.mean([get(["rollout", "bfs_action_agreement_rollout"], r, 0) for r in reports])) if reports else None,
        "q_path_mean_agreement": float(np.mean([get(["q_ranking", "bfs_path_q_agreement"], r, 0) for r in valid_reports])) if valid_reports else None,
        "q_path_mean_negative_margin_ratio": float(np.mean([get(["q_ranking", "negative_margin_ratio"], r, 0) for r in valid_reports])) if valid_reports else None,
        "q_path_mean_margin": float(np.mean([get(["q_ranking", "mean_q_margin"], r, 0) for r in valid_reports])) if valid_reports else None,
        "q_path_mean_abs_demo_td_error": float(np.mean([get(["q_ranking", "mean_abs_demo_td_error"], r, 0) for r in valid_reports])) if valid_reports else None,
    }

    # Per source/difficulty summary.
    grouped = defaultdict(list)
    for r in reports:
        grouped[(r["source"], r["difficulty"])].append(r)

    by_group = {}
    for (source, diff), group in grouped.items():
        key = f"{source}_{diff}"
        valid_group = [r for r in group if get(["q_ranking", "valid"], r, False)]
        by_group[key] = {
            "n": len(group),
            "success_rate": float(np.mean([get(["rollout", "success"], r, False) for r in group])),
            "mean_repeat_count": float(np.mean([get(["rollout", "repeat_count"], r, 0) for r in group])),
            "mean_unique_cells": float(np.mean([get(["rollout", "unique_cells"], r, 0) for r in group])),
            "mean_bounce_ratio": float(np.mean([get(["rollout", "bounce_ratio"], r, 0) for r in group])),
            "mean_progress_rate": float(np.mean([get(["rollout", "progress_rate"], r, 0) for r in group])),
            "mean_rollout_bfs_agreement": float(np.mean([get(["rollout", "bfs_action_agreement_rollout"], r, 0) for r in group])),
            "mean_bfs_path_q_agreement": float(np.mean([get(["q_ranking", "bfs_path_q_agreement"], r, 0) for r in valid_group])) if valid_group else None,
            "mean_negative_margin_ratio": float(np.mean([get(["q_ranking", "negative_margin_ratio"], r, 0) for r in valid_group])) if valid_group else None,
            "mean_q_margin": float(np.mean([get(["q_ranking", "mean_q_margin"], r, 0) for r in valid_group])) if valid_group else None,
        }
    summary["by_group"] = by_group
    return summary


def flatten_q_path_rows(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for r in reports:
        for row in r.get("q_ranking", {}).get("rows", []):
            out = {
                "source": r["source"],
                "difficulty": r["difficulty"],
                "start": r["start"],
                "goal": r["goal"],
                "shortest_len": r["shortest_len"],
            }
            out.update(row)
            rows.append(out)
    return rows


def flatten_rollout_rows(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for r in reports:
        for row in r.get("rollout", {}).get("rows", []):
            out = {
                "source": r["source"],
                "difficulty": r["difficulty"],
                "start": r["start"],
                "goal": r["goal"],
                "shortest_len": r["shortest_len"],
            }
            out.update(row)
            rows.append(out)
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    keys = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_debug_outputs(
    cfg: DebugConfig,
    reports: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> None:
    out = ensure_output_dir(cfg)

    report_path = out / f"{cfg.output_name}_full_report.json"
    summary_path = out / f"{cfg.output_name}_summary.json"
    q_csv_path = out / f"{cfg.output_name}_bfs_path_q_rows.csv"
    rollout_csv_path = out / f"{cfg.output_name}_rollout_rows.csv"

    report_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(q_csv_path, flatten_q_path_rows(reports))
    write_csv(rollout_csv_path, flatten_rollout_rows(reports))

    print(f"[Save] full report: {report_path}")
    print(f"[Save] summary: {summary_path}")
    print(f"[Save] BFS-path Q rows: {q_csv_path}")
    print(f"[Save] rollout rows: {rollout_csv_path}")


# ============================================================
# Visualization
# ============================================================

def draw_grid_image(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    path: List[Tuple[int, int]],
    frame_idx: int,
) -> np.ndarray:
    img = np.ones((SIZE, SIZE, 3), dtype=np.float32)
    img[grid == 1] = np.array([0.05, 0.05, 0.05])

    for p in path[: frame_idx + 1]:
        if p not in [start, goal]:
            img[p] = np.array([0.45, 0.75, 1.0])

    img[start] = np.array([0.0, 0.85, 0.0])
    img[goal] = np.array([1.0, 0.2, 0.2])

    cur = path[frame_idx]
    img[cur] = np.array([1.0, 0.85, 0.0])
    return img


def save_rollout_gif(
    model: CNN_DQN,
    cfg: DebugConfig,
    device: torch.device,
    difficulty: str,
) -> Path:
    grid, start, goal, maze_type, diff = sample_manual_maze(difficulty)
    result = greedy_rollout(
        model=model,
        grid=grid,
        start=start,
        goal=goal,
        cfg=cfg,
        device=device,
        maze_id=f"gif_{difficulty}",
    )

    path = []
    for s in result["path"]:
        # s is "(r, c)"
        nums = s.strip("()").split(",")
        path.append((int(nums[0]), int(nums[1])))

    out_path = ensure_output_dir(cfg) / f"{cfg.output_name}_manual_{difficulty}_rollout.gif"

    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame_idx: int):
        ax.clear()
        ax.imshow(draw_grid_image(grid, start, goal, path, frame_idx))
        ax.set_xticks(np.arange(-0.5, SIZE, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, SIZE, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.set_title(
            f"{VERSION} {difficulty} | frame={frame_idx+1}/{len(path)} | "
            f"success={result['success']} | repeat={result['repeat_count']} | "
            f"bounce={result['bounce_ratio']:.2f}"
        )
        return []

    ani = FuncAnimation(fig, update, frames=len(path), interval=250, blit=False)
    ani.save(out_path, writer=PillowWriter(fps=4))
    plt.close(fig)

    print(f"[Save] rollout GIF: {out_path}")
    return out_path


# ============================================================
# Main debug runner
# ============================================================

def build_debug_set(cfg: DebugConfig) -> List[Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], str, str, str]]:
    """
    Returns tuples:
      grid, start, goal, maze_id, source, difficulty
    """
    tasks = []

    # Manual debug: carefully designed topology, random S/G.
    for diff in ["easy", "medium", "hard"]:
        for i in range(cfg.manual_n):
            grid, start, goal, maze_type, difficulty = sample_manual_maze(diff)
            tasks.append((grid, start, goal, f"{maze_type}_{i}", "manual", difficulty))

    # Random debug: same distribution family as training.
    for diff in ["easy", "medium", "hard"]:
        for i in range(cfg.eval_n):
            grid, start, goal, maze_type, difficulty = generate_random_maze(diff)
            tasks.append((grid, start, goal, f"{maze_type}_{i}", "random", difficulty))

    return tasks


def print_human_summary(summary: Dict[str, Any]) -> None:
    print("\n=== v2.1.1 Debug Summary ===")
    print(f"n_mazes: {summary['n_mazes']}")
    print(f"rollout_success_rate: {summary['rollout_success_rate']:.4f}")
    print(f"rollout_mean_repeat_count: {summary['rollout_mean_repeat_count']:.2f}")
    print(f"rollout_mean_unique_cells: {summary['rollout_mean_unique_cells']:.2f}")
    print(f"rollout_mean_bounce_ratio: {summary['rollout_mean_bounce_ratio']:.4f}")
    print(f"rollout_mean_progress_rate: {summary['rollout_mean_progress_rate']:.4f}")
    print(f"rollout_mean_bfs_agreement: {summary['rollout_mean_bfs_agreement']:.4f}")
    print(f"q_path_mean_agreement: {summary['q_path_mean_agreement']:.4f}")
    print(f"q_path_mean_negative_margin_ratio: {summary['q_path_mean_negative_margin_ratio']:.4f}")
    print(f"q_path_mean_margin: {summary['q_path_mean_margin']:.4f}")
    print(f"q_path_mean_abs_demo_td_error: {summary['q_path_mean_abs_demo_td_error']:.4f}")

    print("\n--- By group ---")
    for key, val in summary["by_group"].items():
        print(
            f"{key:16s} "
            f"succ={val['success_rate']:.3f} "
            f"repeat={val['mean_repeat_count']:.2f} "
            f"unique={val['mean_unique_cells']:.2f} "
            f"bounce={val['mean_bounce_ratio']:.3f} "
            f"rollAgree={val['mean_rollout_bfs_agreement']:.3f} "
            f"pathAgree={val['mean_bfs_path_q_agreement'] if val['mean_bfs_path_q_agreement'] is not None else None} "
            f"negMargin={val['mean_negative_margin_ratio'] if val['mean_negative_margin_ratio'] is not None else None}"
        )


def parse_args() -> DebugConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="./v2.1.1")
    parser.add_argument("--output-name", type=str, default="v2.1.1_debug")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-n", type=int, default=30)
    parser.add_argument("--manual-n", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--gif-difficulty", choices=["easy", "medium", "hard"], default="medium")
    args = parser.parse_args()

    cfg = DebugConfig(
        output_dir=args.output_dir,
        output_name=args.output_name,
        seed=args.seed,
        eval_n=args.eval_n,
        manual_n=args.manual_n,
        max_steps=args.max_steps,
        save_gif=not args.no_gif,
        gif_difficulty=args.gif_difficulty,
    )

    if args.model:
        cfg.model_path = args.model

    return cfg


def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    ensure_output_dir(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(cfg, device)

    tasks = build_debug_set(cfg)
    reports = []

    print(f"[Debug] number of mazes: {len(tasks)}")

    for idx, (grid, start, goal, maze_id, source, difficulty) in enumerate(tasks):
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"[Debug] {idx + 1}/{len(tasks)}")

        report = diagnose_one_maze(
            model=model,
            grid=grid,
            start=start,
            goal=goal,
            cfg=cfg,
            device=device,
            maze_id=maze_id,
            source=source,
            difficulty=difficulty,
        )
        reports.append(report)

    summary = summarize_reports(reports)
    save_debug_outputs(cfg, reports, summary)
    print_human_summary(summary)

    if cfg.save_gif:
        save_rollout_gif(
            model=model,
            cfg=cfg,
            device=device,
            difficulty=cfg.gif_difficulty,
        )


if __name__ == "__main__":
    main()
