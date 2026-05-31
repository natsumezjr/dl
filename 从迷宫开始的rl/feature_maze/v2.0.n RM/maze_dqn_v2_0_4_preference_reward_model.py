"""
maze_dqn_v2_0_4_preference_reward_model.py

v2.0.4: Trajectory Preference Reward Model -> QCNN

This file is intentionally written as a new program structure instead of a patch
on v2.0.3. It has two separated stages:

1) Offline reward-model learning:
   - Build trajectories from BFS, noisy-BFS, random, and safe-loop policies.
   - Rank trajectories lexicographically:
       success > timeout > wall
     If the primary outcome is the same:
       success: smaller BFS gap is better
       timeout: smaller final BFS distance to goal is better
       wall: smaller BFS distance before/at wall position is better
   - Convert ordered trajectories into preference pairs.
   - Train R_phi(S, a, S') with the Bradley-Terry-Luce loss:
       L_BTL = -log sigmoid(R_phi(tau+) - R_phi(tau-))

2) QCNN training:
   - Freeze R_phi.
   - Use r_t = rm_scale * R_phi(S_t, a_t, S_{t+1}) as the DQN reward.
   - Optional terminal anchors exist for debugging, but default is off.

No demo replay, no BFS margin, no action mask, no progress reward.
BFS is used only to create trajectory preferences and evaluation metrics.
"""

import argparse
import csv
import json
import math
import random
from collections import deque, namedtuple
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VERSION = "v2.0.4_preference_reward_model_debug"
SIZE = 8
MAX_STEPS = 64
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]

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

DQNTransition = namedtuple("DQNTransition", ["state", "action", "reward", "next_state", "done"])


@dataclass
class RMTransition:
    state: np.ndarray
    action: int
    next_state: np.ndarray


@dataclass
class Trajectory:
    transitions: List[RMTransition]
    outcome: str
    steps: int
    bfs_len: int
    final_bfs_dist: int
    gap: int
    wall_hits: int
    quality_key: Tuple[int, int, int]
    source: str


@dataclass
class PreferencePair:
    pos: Trajectory
    neg: Trajectory


@dataclass
class DQNConfig:
    episodes: int = 3000
    max_steps: int = 64
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 128
    replay_size: int = 100_000
    warmup_steps: int = 3000
    updates_per_episode: int = 64
    target_update_interval: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 2500
    eval_n: int = 100
    rm_scale: float = 2.0
    anchor_terminal: bool = False
    anchor_goal: float = 64.0
    anchor_wall: float = -32.0


# -----------------------------
# General utilities
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


def moving_average(values: Sequence[float], window: int = 20) -> List[float]:
    out: List[float] = []
    buf: deque = deque(maxlen=window)
    for v in values:
        buf.append(float(v))
        out.append(float(np.mean(buf)))
    return out


def weighted_choice(weight_map: Dict[str, float]) -> str:
    keys = list(weight_map.keys())
    weights = np.array([weight_map[k] for k in keys], dtype=np.float64)
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


def bfs_path(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    dist = bfs_distances(grid, goal)
    if dist[start] >= 10_000:
        return None
    path = [start]
    cur = start
    while cur != goal:
        best = None
        best_d = dist[cur]
        for dr, dc in ACTIONS:
            nxt = (cur[0] + dr, cur[1] + dc)
            if is_free(grid, nxt) and dist[nxt] < best_d:
                best = nxt
                best_d = int(dist[nxt])
        if best is None:
            return None
        cur = best
        path.append(cur)
    return path


def action_between(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    delta = (b[0] - a[0], b[1] - a[1])
    return ACTIONS.index(delta)


def random_free_cell(grid: np.ndarray) -> Tuple[int, int]:
    cells = [(r, c) for r in range(SIZE) for c in range(SIZE) if grid[r, c] == 0]
    return random.choice(cells)


def generate_maze(difficulty: str) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    # Rejection sampling keeps this simple and deterministic enough for experiments.
    specs = {
        "easy": (0.14, 6, 22),
        "medium": (0.23, 12, 36),
        "hard": (0.31, 18, 64),
    }
    wall_prob, min_len, max_len = specs[difficulty]
    start = (0, 0)
    goal = (SIZE - 1, SIZE - 1)
    for _ in range(10_000):
        grid = (np.random.rand(SIZE, SIZE) < wall_prob).astype(np.int8)
        grid[start] = 0
        grid[goal] = 0
        path = bfs_path(grid, start, goal)
        if path is None:
            continue
        length = len(path) - 1
        if min_len <= length <= max_len:
            return grid, start, goal
    # Fallback: carve a random monotone-ish path, then add walls.
    grid = np.zeros((SIZE, SIZE), dtype=np.int8)
    pos = start
    protected = {start, goal}
    while pos != goal:
        choices = []
        if pos[0] < goal[0]:
            choices.append((pos[0] + 1, pos[1]))
        if pos[1] < goal[1]:
            choices.append((pos[0], pos[1] + 1))
        pos = random.choice(choices)
        protected.add(pos)
    for r in range(SIZE):
        for c in range(SIZE):
            if (r, c) not in protected and random.random() < wall_prob:
                grid[r, c] = 1
    return grid, start, goal


def sample_difficulty(easy: float, medium: float, hard: float) -> str:
    return weighted_choice({"easy": easy, "medium": medium, "hard": hard})


# -----------------------------
# Environment with visited state
# -----------------------------

class MazeEnv:
    def __init__(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], max_steps: int = MAX_STEPS):
        self.grid = grid.copy().astype(np.int8)
        self.start = start
        self.goal = goal
        self.max_steps = max_steps
        self.dist = bfs_distances(self.grid, self.goal)
        self.reset()

    def reset(self) -> np.ndarray:
        self.pos = self.start
        self.steps = 0
        self.done = False
        self.visited = np.zeros((SIZE, SIZE), dtype=np.float32)
        self.visited[self.pos] = 1.0
        return self.state()

    def state(self) -> np.ndarray:
        wall = self.grid.astype(np.float32)
        agent = np.zeros((SIZE, SIZE), dtype=np.float32)
        goal = np.zeros((SIZE, SIZE), dtype=np.float32)
        agent[self.pos] = 1.0
        goal[self.goal] = 1.0
        return np.stack([wall, agent, goal, self.visited.astype(np.float32)], axis=0)

    def step_raw(self, action: int) -> Tuple[np.ndarray, Dict[str, object]]:
        if self.done:
            raise RuntimeError("step after done")
        before = self.pos
        dr, dc = ACTIONS[action]
        nxt = (before[0] + dr, before[1] + dc)
        hit_wall = (not in_bounds(nxt)) or (not is_free(self.grid, nxt))
        success = False
        timeout = False

        # v2.0.4 fix: wall hit is an event, not a terminal state.
        # The agent stays in place, consumes one step, and can continue.
        if not hit_wall:
            self.pos = nxt
            self.visited[self.pos] = 1.0
            success = self.pos == self.goal

        self.steps += 1
        if success:
            self.done = True
            outcome = "success"
        elif self.steps >= self.max_steps:
            self.done = True
            timeout = True
            outcome = "timeout"
        else:
            outcome = "running"

        next_state = self.state()
        info = {
            "before": before,
            "pos": self.pos,
            "hit_wall": hit_wall,
            "success": success,
            "timeout": timeout,
            "outcome": outcome,
            "steps": self.steps,
            "bfs_dist": int(self.dist[self.pos]) if self.dist[self.pos] < 10_000 else 10_000,
            "repeat_count": int(self.visited.sum()) if False else 0,
        }
        return next_state, info


def transition_tensor(state: np.ndarray, action: int, next_state: np.ndarray) -> np.ndarray:
    planes = np.zeros((4, SIZE, SIZE), dtype=np.float32)
    planes[action, :, :] = 1.0
    return np.concatenate([state.astype(np.float32), next_state.astype(np.float32), planes], axis=0)


# -----------------------------
# Trajectory generation
# -----------------------------

def make_trajectory(env: MazeEnv, policy: str, noise_prob: float = 0.2, max_steps: int = MAX_STEPS) -> Trajectory:
    state = env.reset()
    path = bfs_path(env.grid, env.start, env.goal)
    if path is None:
        raise ValueError("trajectory generated on unsolvable maze")
    bfs_len = len(path) - 1
    dist = env.dist
    transitions: List[RMTransition] = []
    outcome = "timeout"
    repeat_visits = 0
    wall_hits = 0

    for t in range(max_steps):
        if policy == "bfs":
            if env.pos == env.goal:
                break
            # Follow the BFS path by local distance descent.
            best_actions = []
            cur_d = int(dist[env.pos])
            for a, (dr, dc) in enumerate(ACTIONS):
                nxt = (env.pos[0] + dr, env.pos[1] + dc)
                if is_free(env.grid, nxt) and int(dist[nxt]) < cur_d:
                    best_actions.append(a)
            action = random.choice(best_actions) if best_actions else random.randrange(4)
        elif policy == "noisy_bfs":
            if random.random() < noise_prob:
                action = random.randrange(4)
            else:
                best_actions = []
                cur_d = int(dist[env.pos])
                for a, (dr, dc) in enumerate(ACTIONS):
                    nxt = (env.pos[0] + dr, env.pos[1] + dc)
                    if is_free(env.grid, nxt) and int(dist[nxt]) < cur_d:
                        best_actions.append(a)
                action = random.choice(best_actions) if best_actions else random.randrange(4)
        elif policy == "random":
            action = random.randrange(4)
        elif policy == "safe_loop":
            # A deliberately non-goal-directed policy. It avoids walls, prefers already visited cells,
            # and therefore produces exactly the safe-loop failure mode observed in DQN.
            legal: List[int] = []
            visited_legal: List[int] = []
            non_goal_legal: List[int] = []
            for a, (dr, dc) in enumerate(ACTIONS):
                nxt = (env.pos[0] + dr, env.pos[1] + dc)
                if is_free(env.grid, nxt):
                    legal.append(a)
                    if env.visited[nxt] > 0.5:
                        visited_legal.append(a)
                    if nxt != env.goal:
                        non_goal_legal.append(a)
            if visited_legal:
                action = random.choice(visited_legal)
            elif non_goal_legal:
                action = random.choice(non_goal_legal)
            elif legal:
                action = random.choice(legal)
            else:
                action = random.randrange(4)
        else:
            raise ValueError(f"unknown policy {policy}")

        s = state.copy()
        old_visited = env.visited.copy()
        ns, info = env.step_raw(action)
        if info["hit_wall"]:
            wall_hits += 1
        if not info["hit_wall"] and old_visited[env.pos] > 0.5:
            repeat_visits += 1
        transitions.append(RMTransition(s, action, ns.copy()))
        state = ns
        if info["outcome"] != "running":
            outcome = str(info["outcome"])
            break
    else:
        outcome = "timeout"

    steps = len(transitions)
    final_dist = int(dist[env.pos]) if int(dist[env.pos]) < 10_000 else 10_000
    if outcome == "success":
        gap = max(0, steps - bfs_len)
        key = (3, -gap, -wall_hits)
        stored_outcome = "success"
        final_dist = 0
    elif wall_hits == 0:
        gap = MAX_STEPS
        key = (2, -final_dist, 0)
        stored_outcome = "timeout"
    else:
        # Wall is non-terminal in the environment. For preference data, a "wall"
        # trajectory means a non-success trajectory that contains at least one wall hit.
        gap = MAX_STEPS
        key = (1, -final_dist, -wall_hits)
        stored_outcome = "wall"

    return Trajectory(
        transitions=transitions,
        outcome=stored_outcome,
        steps=steps,
        bfs_len=bfs_len,
        final_bfs_dist=final_dist,
        gap=gap,
        wall_hits=wall_hits,
        quality_key=key,
        source=policy,
    )


def build_preference_dataset(
    rm_mazes: int,
    trajectories_per_maze: int,
    pairs_per_maze: int,
    easy_ratio: float,
    medium_ratio: float,
    hard_ratio: float,
    max_steps: int,
) -> Tuple[List[PreferencePair], Dict[str, int]]:
    pairs: List[PreferencePair] = []
    stats = {
        "mazes": 0,
        "trajectories": 0,
        "pairs": 0,
        "success": 0,
        "timeout": 0,
        "wall": 0,
        "bfs": 0,
        "noisy_bfs": 0,
        "random": 0,
        "safe_loop": 0,
        "skipped_ties": 0,
    }
    policies = ["bfs", "safe_loop", "random", "noisy_bfs"]
    noise_values = [0.10, 0.20, 0.35]

    for _ in range(rm_mazes):
        difficulty = sample_difficulty(easy_ratio, medium_ratio, hard_ratio)
        grid, start, goal = generate_maze(difficulty)
        env = MazeEnv(grid, start, goal, max_steps=max_steps)
        trajs: List[Trajectory] = []

        # Always include one BFS and one safe-loop trajectory.
        trajs.append(make_trajectory(env, "bfs", max_steps=max_steps))
        trajs.append(make_trajectory(env, "safe_loop", max_steps=max_steps))
        while len(trajs) < trajectories_per_maze:
            pol = random.choice(policies[2:])  # random or noisy_bfs
            noise = random.choice(noise_values)
            trajs.append(make_trajectory(env, pol, noise_prob=noise, max_steps=max_steps))

        stats["mazes"] += 1
        stats["trajectories"] += len(trajs)
        for tr in trajs:
            stats[tr.outcome] += 1
            stats[tr.source] += 1

        candidate_pairs: List[PreferencePair] = []
        for i in range(len(trajs)):
            for j in range(i + 1, len(trajs)):
                a, b = trajs[i], trajs[j]
                if a.quality_key == b.quality_key:
                    stats["skipped_ties"] += 1
                    continue
                if a.quality_key > b.quality_key:
                    candidate_pairs.append(PreferencePair(a, b))
                else:
                    candidate_pairs.append(PreferencePair(b, a))
        if not candidate_pairs:
            continue
        random.shuffle(candidate_pairs)
        pairs.extend(candidate_pairs[:pairs_per_maze])
        stats["pairs"] += min(len(candidate_pairs), pairs_per_maze)

    return pairs, stats


# -----------------------------
# Models
# -----------------------------

class TransitionRewardCNN(nn.Module):
    def __init__(self, in_channels: int = 12):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * SIZE * SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x)).squeeze(-1)


class QNetwork(nn.Module):
    def __init__(self, in_channels: int = 4, n_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * SIZE * SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Reward model training
# -----------------------------

def trajectory_score(model: TransitionRewardCNN, traj: Trajectory, device: torch.device) -> torch.Tensor:
    if not traj.transitions:
        return torch.zeros((), dtype=torch.float32, device=device)
    xs = [transition_tensor(t.state, t.action, t.next_state) for t in traj.transitions]
    x = torch.tensor(np.stack(xs), dtype=torch.float32, device=device)
    return model(x).sum()


def batch_btl_loss(model: TransitionRewardCNN, batch: Sequence[PreferencePair], device: torch.device) -> Tuple[torch.Tensor, float, float, float]:
    diffs: List[torch.Tensor] = []
    pos_scores = []
    neg_scores = []
    correct = 0
    for pair in batch:
        sp = trajectory_score(model, pair.pos, device)
        sn = trajectory_score(model, pair.neg, device)
        diffs.append(sp - sn)
        pos_scores.append(float(sp.detach().cpu()))
        neg_scores.append(float(sn.detach().cpu()))
        if float(sp.detach().cpu()) > float(sn.detach().cpu()):
            correct += 1
    diff = torch.stack(diffs)
    # L_BTL = -log sigmoid(R_phi(tau+) - R_phi(tau-))
    loss = -F.logsigmoid(diff).mean()
    acc = correct / max(1, len(batch))
    return loss, acc, float(np.mean(pos_scores)), float(np.mean(neg_scores))


def evaluate_rm(model: TransitionRewardCNN, pairs: Sequence[PreferencePair], device: torch.device, batch_size: int) -> Dict[str, float]:
    model.eval()
    losses = []
    accs = []
    pos_vals = []
    neg_vals = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            loss, acc, pos_m, neg_m = batch_btl_loss(model, batch, device)
            losses.append(float(loss.cpu()))
            accs.append(acc)
            pos_vals.append(pos_m)
            neg_vals.append(neg_m)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(np.mean(accs)) if accs else 0.0,
        "pos_score": float(np.mean(pos_vals)) if pos_vals else 0.0,
        "neg_score": float(np.mean(neg_vals)) if neg_vals else 0.0,
    }


def train_reward_model(args: argparse.Namespace, device: torch.device, out_dir: Path) -> Path:
    print("\n=== Stage A: Train v2.0.4 Preference Reward Model ===")
    print("Preference primary key: success > clean-timeout > wall-timeout")
    print("Tie-breakers: success uses BFS gap; timeout/wall use BFS distance to goal; wall is non-terminal")
    print("BTL loss: L = -log sigmoid(R_phi(tau+) - R_phi(tau-))")
    print("Safe-loop trajectories are explicitly included.")

    pairs, stats = build_preference_dataset(
        rm_mazes=args.rm_mazes,
        trajectories_per_maze=args.trajectories_per_maze,
        pairs_per_maze=args.pairs_per_maze,
        easy_ratio=args.easy_ratio,
        medium_ratio=args.medium_ratio,
        hard_ratio=args.hard_ratio,
        max_steps=args.max_steps,
    )
    if len(pairs) < 10:
        raise RuntimeError("not enough preference pairs generated")
    random.shuffle(pairs)
    split = int(len(pairs) * (1.0 - args.rm_val_ratio))
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]
    print(f"[RM dataset] stats={stats}")
    print(f"[RM dataset] train_pairs={len(train_pairs)} val_pairs={len(val_pairs)}")

    model = TransitionRewardCNN().to(device)
    opt = optim.Adam(model.parameters(), lr=args.rm_lr)
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.rm_epochs + 1):
        model.train()
        random.shuffle(train_pairs)
        losses = []
        accs = []
        pos_vals = []
        neg_vals = []
        for i in range(0, len(train_pairs), args.rm_batch_size):
            batch = train_pairs[i:i + args.rm_batch_size]
            opt.zero_grad(set_to_none=True)
            loss, acc, pos_m, neg_m = batch_btl_loss(model, batch, device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.rm_grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            accs.append(acc)
            pos_vals.append(pos_m)
            neg_vals.append(neg_m)
        val = evaluate_rm(model, val_pairs, device, args.rm_batch_size) if val_pairs else {"loss": 0.0, "accuracy": 0.0, "pos_score": 0.0, "neg_score": 0.0}
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_accuracy": float(np.mean(accs)),
            "train_pos_score": float(np.mean(pos_vals)),
            "train_neg_score": float(np.mean(neg_vals)),
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_pos_score": val["pos_score"],
            "val_neg_score": val["neg_score"],
        }
        history.append(row)
        print(
            f"rm epoch={epoch:03d}/{args.rm_epochs} "
            f"loss={row['train_loss']:.4f} acc={row['train_accuracy']:.3f} "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_accuracy']:.3f} "
            f"pos={row['val_pos_score']:.2f} neg={row['val_neg_score']:.2f}"
        )

    model_path = out_dir / "v2.0.4_reward_model.pt"
    torch.save({
        "version": VERSION,
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "dataset_stats": stats,
        "history": history,
    }, model_path)
    with open(out_dir / "v2.0.4_reward_model_history.json", "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "history": history}, f, indent=2, ensure_ascii=False)
    plot_rm_history(history, out_dir / "v2.0.4_reward_model_curves.png")
    if not getattr(args, "no_debug", False):
        run_reward_model_debug(model, args, device, out_dir, tag="rm_after_train")
    print(f"[Save] reward model: {model_path}")
    return model_path


def plot_rm_history(history: List[Dict[str, float]], path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [h["train_loss"] for h in history], label="train")
    plt.plot(epochs, [h["val_loss"] for h in history], label="val")
    plt.title("BTL loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [h["train_accuracy"] for h in history], label="train")
    plt.plot(epochs, [h["val_accuracy"] for h in history], label="val")
    plt.title("Preference accuracy")
    plt.xlabel("epoch")
    plt.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)




# -----------------------------
# Reward-model diagnostics
# -----------------------------

def argmax_pos(plane: np.ndarray) -> Tuple[int, int]:
    idx = int(np.argmax(plane))
    return divmod(idx, SIZE)


def infer_transition_features(tr: RMTransition) -> Dict[str, object]:
    s = tr.state
    ns = tr.next_state
    grid = s[0]
    pos = argmax_pos(s[1])
    next_pos = argmax_pos(ns[1])
    goal = argmax_pos(s[2])
    visited = s[3]
    dr, dc = ACTIONS[tr.action]
    target = (pos[0] + dr, pos[1] + dc)
    hit_wall = (not in_bounds(target)) or (grid[target] > 0.5)
    success = next_pos == goal
    repeat = (not hit_wall) and (visited[next_pos] > 0.5)
    dist = bfs_distances(grid.astype(np.int8), goal)
    d0 = int(dist[pos]) if dist[pos] < 10_000 else 10_000
    d1 = int(dist[next_pos]) if dist[next_pos] < 10_000 else 10_000
    delta_d = d0 - d1
    if success:
        category = "goal"
    elif hit_wall:
        category = "wall"
    elif repeat:
        category = "repeat"
    elif delta_d > 0:
        category = "toward_goal"
    elif delta_d < 0:
        category = "away_goal"
    else:
        category = "flat"
    return {
        "pos": pos,
        "next_pos": next_pos,
        "goal": goal,
        "action": tr.action,
        "hit_wall": hit_wall,
        "success": success,
        "repeat": repeat,
        "d0": d0,
        "d1": d1,
        "delta_d": delta_d,
        "category": category,
    }


def rm_step_reward(model: TransitionRewardCNN, tr: RMTransition, device: torch.device, scale: float = 1.0) -> float:
    with torch.no_grad():
        x = torch.tensor(transition_tensor(tr.state, tr.action, tr.next_state)[None, ...], dtype=torch.float32, device=device)
        return float(model(x).item() * scale)


def score_trajectory_raw(model: TransitionRewardCNN, traj: Trajectory, device: torch.device, scale: float = 1.0) -> Tuple[float, List[float]]:
    vals = [rm_step_reward(model, tr, device, scale=scale) for tr in traj.transitions]
    return float(sum(vals)), vals


def summarize_values(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"n": 0, "mean": 0.0, "std": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(vals, dtype=np.float32)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def simulate_transition_from_state(state: np.ndarray, action: int) -> RMTransition:
    grid = state[0].astype(np.int8)
    pos = argmax_pos(state[1])
    goal = argmax_pos(state[2])
    visited = state[3].copy()
    dr, dc = ACTIONS[action]
    target = (pos[0] + dr, pos[1] + dc)
    hit_wall = (not in_bounds(target)) or (not is_free(grid, target))
    next_pos = pos if hit_wall else target
    next_state = state.copy()
    next_state[1, :, :] = 0.0
    next_state[1, next_pos[0], next_pos[1]] = 1.0
    if not hit_wall:
        next_state[3, next_pos[0], next_pos[1]] = 1.0
    next_state[2, :, :] = 0.0
    next_state[2, goal[0], goal[1]] = 1.0
    return RMTransition(state.copy(), action, next_state.copy())


def collect_probe_trajectories(args: argparse.Namespace, probe_mazes: int) -> List[Trajectory]:
    trajectories: List[Trajectory] = []
    weights = {"easy": args.easy_ratio, "medium": args.medium_ratio, "hard": args.hard_ratio}
    for _ in range(probe_mazes):
        diff = weighted_choice(weights)
        grid, start, goal = generate_maze(diff)
        for policy, noise in [("bfs", 0.0), ("noisy_bfs", 0.15), ("noisy_bfs", 0.35), ("random", 0.0), ("safe_loop", 0.0)]:
            env = MazeEnv(grid, start, goal, max_steps=args.max_steps)
            trajectories.append(make_trajectory(env, policy, noise_prob=noise, max_steps=args.max_steps))
    return trajectories


def run_reward_model_debug(
    model: TransitionRewardCNN,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
    tag: str = "rm",
) -> Dict[str, object]:
    model.eval()
    probe_n = max(1, int(getattr(args, "debug_probe_mazes", 30)))
    trajectories = collect_probe_trajectories(args, probe_n)

    # A. Trajectory score diagnostics: detects length bias and safe-loop exploitation.
    traj_rows: List[Dict[str, object]] = []
    step_by_cat: Dict[str, List[float]] = {}
    step_by_source: Dict[str, List[float]] = {}
    for idx, traj in enumerate(trajectories):
        total, vals = score_trajectory_raw(model, traj, device, scale=1.0)
        avg = total / max(1, traj.steps)
        traj_rows.append({
            "idx": idx,
            "source": traj.source,
            "outcome": traj.outcome,
            "steps": traj.steps,
            "bfs_len": traj.bfs_len,
            "gap": traj.gap,
            "final_bfs_dist": traj.final_bfs_dist,
            "wall_hits": traj.wall_hits,
            "score_sum": total,
            "score_avg": avg,
            "quality_key": list(traj.quality_key),
        })
        for tr, r in zip(traj.transitions, vals):
            feat = infer_transition_features(tr)
            step_by_cat.setdefault(str(feat["category"]), []).append(r)
            step_by_source.setdefault(traj.source, []).append(r)

    by_traj_group: Dict[str, List[float]] = {}
    by_traj_group_avg: Dict[str, List[float]] = {}
    for row in traj_rows:
        key = f"{row['source']}|{row['outcome']}"
        by_traj_group.setdefault(key, []).append(float(row["score_sum"]))
        by_traj_group_avg.setdefault(key, []).append(float(row["score_avg"]))

    # B. Counterfactual action sensitivity: fixed state, enumerate four actions.
    action_rows: List[Dict[str, object]] = []
    sample_states: List[np.ndarray] = []
    for traj in trajectories:
        if traj.transitions:
            # Sample the first, middle, and last-ish states when available.
            inds = sorted(set([0, len(traj.transitions) // 2, max(0, len(traj.transitions) - 1)]))
            for i in inds:
                if len(sample_states) < 60:
                    sample_states.append(traj.transitions[i].state)
    for sid, state in enumerate(sample_states):
        for action in range(4):
            tr = simulate_transition_from_state(state, action)
            r = rm_step_reward(model, tr, device, scale=1.0)
            feat = infer_transition_features(tr)
            action_rows.append({
                "state_id": sid,
                "action": action,
                "reward": r,
                "category": feat["category"],
                "hit_wall": bool(feat["hit_wall"]),
                "repeat": bool(feat["repeat"]),
                "success": bool(feat["success"]),
                "d0": int(feat["d0"]),
                "d1": int(feat["d1"]),
                "delta_d": int(feat["delta_d"]),
            })

    # C. Action-shuffle sensitivity: if reward barely changes after replacing action planes,
    # the model is likely ignoring action encoding.
    shuffle_deltas: List[float] = []
    for traj in trajectories[: min(40, len(trajectories))]:
        for tr in traj.transitions[: min(10, len(traj.transitions))]:
            base = rm_step_reward(model, tr, device, scale=1.0)
            alt_action = random.choice([a for a in range(4) if a != tr.action])
            fake = RMTransition(tr.state, alt_action, tr.next_state)
            alt = rm_step_reward(model, fake, device, scale=1.0)
            shuffle_deltas.append(abs(base - alt))

    report: Dict[str, object] = {
        "tag": tag,
        "probe_mazes": probe_n,
        "trajectory_group_score_sum": {k: summarize_values(v) for k, v in sorted(by_traj_group.items())},
        "trajectory_group_score_avg": {k: summarize_values(v) for k, v in sorted(by_traj_group_avg.items())},
        "step_reward_by_category": {k: summarize_values(v) for k, v in sorted(step_by_cat.items())},
        "step_reward_by_source": {k: summarize_values(v) for k, v in sorted(step_by_source.items())},
        "action_sensitivity_abs_delta": summarize_values(shuffle_deltas),
        "trajectory_rows_sample": traj_rows[:100],
        "counterfactual_action_rows_sample": action_rows[:200],
    }

    json_path = out_dir / f"v2.0.4_{tag}_debug_report.json"
    csv_traj = out_dir / f"v2.0.4_{tag}_debug_trajectory_scores.csv"
    csv_action = out_dir / f"v2.0.4_{tag}_debug_counterfactual_actions.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    if traj_rows:
        with open(csv_traj, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(traj_rows[0].keys()))
            writer.writeheader()
            writer.writerows(traj_rows)
    if action_rows:
        with open(csv_action, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(action_rows[0].keys()))
            writer.writeheader()
            writer.writerows(action_rows)

    print(f"[Debug:{tag}] saved report: {json_path}")
    print(f"[Debug:{tag}] step_reward_by_category={report['step_reward_by_category']}")
    print(f"[Debug:{tag}] action_sensitivity_abs_delta={report['action_sensitivity_abs_delta']}")
    return report


def run_q_policy_debug(
    q_net: QNetwork,
    rm: TransitionRewardCNN,
    cfg: DQNConfig,
    device: torch.device,
    out_dir: Path,
    tag: str = "q_policy",
) -> Dict[str, object]:
    q_net.eval()
    rm.eval()
    grid, start, goal = parse_maze_text(TEST_MAZE_TEXT)
    env = MazeEnv(grid, start, goal, max_steps=cfg.max_steps)
    state = env.reset()
    rows: List[Dict[str, object]] = []
    total_r = 0.0
    for t in range(cfg.max_steps):
        with torch.no_grad():
            qs = q_net(torch.tensor(state[None, ...], dtype=torch.float32, device=device))[0].detach().cpu().numpy().astype(float).tolist()
        action_rewards = []
        for a in range(4):
            tr = simulate_transition_from_state(state, a)
            rr = rm_step_reward(rm, tr, device, scale=cfg.rm_scale)
            feat = infer_transition_features(tr)
            action_rewards.append({"a": a, "rm_reward": rr, "category": feat["category"], "delta_d": feat["delta_d"]})
        action = int(np.argmax(qs))
        next_state, info = env.step_raw(action)
        r = learned_reward(rm, state, action, next_state, device, cfg.rm_scale)
        total_r += r
        chosen_feat = infer_transition_features(RMTransition(state, action, next_state))
        rows.append({
            "t": t,
            "pos": list(info["before"]),
            "action": action,
            "q_values": qs,
            "chosen_rm_reward": r,
            "chosen_category": chosen_feat["category"],
            "chosen_delta_d": int(chosen_feat["delta_d"]),
            "hit_wall": bool(info["hit_wall"]),
            "success": bool(info["success"]),
            "timeout": bool(info["timeout"]),
            "bfs_dist": int(info["bfs_dist"]),
            "action_rewards": action_rewards,
        })
        state = next_state
        if info["outcome"] != "running":
            break
    report = {"tag": tag, "total_reward": total_r, "steps": len(rows), "rows": rows}
    path = out_dir / f"v2.0.4_{tag}_debug_rollout.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[Debug:{tag}] saved rollout: {path}")
    return report


# -----------------------------
# DQN training with frozen reward model
# -----------------------------

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data: deque = deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.data.append(DQNTransition(*args))

    def sample(self, batch_size: int) -> DQNTransition:
        batch = random.sample(self.data, batch_size)
        return DQNTransition(*zip(*batch))

    def __len__(self) -> int:
        return len(self.data)


def epsilon_by_episode(ep: int, cfg: DQNConfig) -> float:
    if ep >= cfg.epsilon_decay_episodes:
        return cfg.epsilon_end
    frac = ep / max(1, cfg.epsilon_decay_episodes)
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def select_action(q_net: QNetwork, state: np.ndarray, epsilon: float, device: torch.device) -> int:
    if random.random() < epsilon:
        return random.randrange(4)
    with torch.no_grad():
        x = torch.tensor(state[None, ...], dtype=torch.float32, device=device)
        q = q_net(x)[0]
        return int(torch.argmax(q).item())


def learned_reward(
    rm: TransitionRewardCNN,
    state: np.ndarray,
    action: int,
    next_state: np.ndarray,
    device: torch.device,
    scale: float,
) -> float:
    with torch.no_grad():
        x = torch.tensor(transition_tensor(state, action, next_state)[None, ...], dtype=torch.float32, device=device)
        return float(rm(x).item() * scale)


def optimize_dqn(q_net: QNetwork, target_net: QNetwork, opt: optim.Optimizer, replay: ReplayBuffer, cfg: DQNConfig, device: torch.device) -> Optional[float]:
    if len(replay) < max(cfg.batch_size, cfg.warmup_steps):
        return None
    batch = replay.sample(cfg.batch_size)
    states = torch.tensor(np.stack(batch.state), dtype=torch.float32, device=device)
    actions = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.tensor(np.stack(batch.next_state), dtype=torch.float32, device=device)
    dones = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)

    q = q_net(states).gather(1, actions)
    with torch.no_grad():
        next_actions = torch.argmax(q_net(next_states), dim=1, keepdim=True)
        next_q = target_net(next_states).gather(1, next_actions)
        target = rewards + cfg.gamma * (1.0 - dones) * next_q
    loss = F.smooth_l1_loss(q, target)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
    opt.step()
    return float(loss.detach().cpu())


def rollout_eval(
    q_net: QNetwork,
    rm: TransitionRewardCNN,
    difficulty: str,
    n: int,
    cfg: DQNConfig,
    device: torch.device,
    fixed_test: bool = False,
) -> Dict[str, float]:
    q_net.eval()
    rm.eval()
    totals = {"success": 0, "wall": 0, "timeout": 0, "steps": 0.0, "repeat": 0.0, "bfs_gap": 0.0, "reward": 0.0, "bfs_agree": 0.0}
    for _ in range(n):
        if fixed_test:
            grid, start, goal = parse_maze_text(TEST_MAZE_TEXT)
        else:
            grid, start, goal = generate_maze(difficulty)
        env = MazeEnv(grid, start, goal, max_steps=cfg.max_steps)
        path = bfs_path(grid, start, goal)
        bfs_len = len(path) - 1 if path else cfg.max_steps
        state = env.reset()
        total_r = 0.0
        repeats = 0
        wall_hits = 0
        bfs_agree_count = 0
        outcome = "timeout"
        for _t in range(cfg.max_steps):
            action = select_action(q_net, state, 0.0, device)
            old_visited = env.visited.copy()
            old_pos = env.pos
            cur_d = int(env.dist[old_pos])
            dr, dc = ACTIONS[action]
            cand = (old_pos[0] + dr, old_pos[1] + dc)
            if is_free(env.grid, cand) and int(env.dist[cand]) < cur_d:
                bfs_agree_count += 1
            next_state, info = env.step_raw(action)
            if info["hit_wall"]:
                wall_hits += 1
            if not info["hit_wall"] and old_visited[env.pos] > 0.5:
                repeats += 1
            r = learned_reward(rm, state, action, next_state, device, cfg.rm_scale)
            if cfg.anchor_terminal:
                if info["success"]:
                    r += cfg.anchor_goal
                elif info["hit_wall"]:
                    r += cfg.anchor_wall
            total_r += r
            state = next_state
            if info["outcome"] != "running":
                outcome = str(info["outcome"])
                break

        if outcome == "success":
            totals["success"] += 1
        elif wall_hits > 0:
            totals["wall"] += 1
        else:
            totals["timeout"] += 1
        actual_steps = env.steps
        totals["steps"] += float(actual_steps)
        totals["repeat"] += float(repeats)
        totals["bfs_gap"] += float(max(0, actual_steps - bfs_len) if env.pos == env.goal else cfg.max_steps - bfs_len)
        totals["reward"] += float(total_r)
        totals["bfs_agree"] += float(bfs_agree_count / max(1, actual_steps))
    denom = float(n)
    return {
        "success": totals["success"] / denom,
        "wall": totals["wall"] / denom,
        "timeout": totals["timeout"] / denom,
        "steps": totals["steps"] / denom,
        "repeat": totals["repeat"] / denom,
        "bfs_gap": totals["bfs_gap"] / denom,
        "reward": totals["reward"] / denom,
        "bfs_agree": totals["bfs_agree"] / denom,
    }


def train_qcnn(args: argparse.Namespace, device: torch.device, out_dir: Path, reward_model_path: Path) -> Path:
    print("\n=== Stage B: Train QCNN with frozen preference reward model ===")
    cfg = DQNConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        warmup_steps=args.warmup_steps,
        updates_per_episode=args.updates_per_episode,
        target_update_interval=args.target_update_interval,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
        eval_n=args.eval_n,
        rm_scale=args.rm_scale,
        anchor_terminal=args.anchor_terminal,
        anchor_goal=args.anchor_goal,
        anchor_wall=args.anchor_wall,
    )
    rm = TransitionRewardCNN().to(device)
    ckpt = torch.load(reward_model_path, map_location=device)
    rm.load_state_dict(ckpt["model_state_dict"])
    rm.eval()
    for p in rm.parameters():
        p.requires_grad_(False)

    q_net = QNetwork().to(device)
    target_net = QNetwork().to(device)
    target_net.load_state_dict(q_net.state_dict())
    opt = optim.Adam(q_net.parameters(), lr=cfg.lr)
    replay = ReplayBuffer(cfg.replay_size)

    history: List[Dict[str, float]] = []
    global_step = 0
    difficulty_weights = {"easy": args.easy_ratio, "medium": args.medium_ratio, "hard": args.hard_ratio}

    for ep in range(1, cfg.episodes + 1):
        diff = weighted_choice(difficulty_weights)
        grid, start, goal = generate_maze(diff)
        env = MazeEnv(grid, start, goal, max_steps=cfg.max_steps)
        state = env.reset()
        eps = epsilon_by_episode(ep, cfg)
        ep_reward = 0.0
        repeats = 0
        wall_hits = 0
        bfs_agree_count = 0
        outcome = "timeout"
        losses = []
        path = bfs_path(grid, start, goal)
        bfs_len = len(path) - 1 if path else cfg.max_steps

        for t in range(cfg.max_steps):
            action = select_action(q_net, state, eps, device)
            old_visited = env.visited.copy()
            old_pos = env.pos
            cur_d = int(env.dist[old_pos])
            cand = (old_pos[0] + ACTIONS[action][0], old_pos[1] + ACTIONS[action][1])
            if is_free(env.grid, cand) and int(env.dist[cand]) < cur_d:
                bfs_agree_count += 1
            next_state, info = env.step_raw(action)
            if info["hit_wall"]:
                wall_hits += 1
            if not info["hit_wall"] and old_visited[env.pos] > 0.5:
                repeats += 1
            r = learned_reward(rm, state, action, next_state, device, cfg.rm_scale)
            if cfg.anchor_terminal:
                if info["success"]:
                    r += cfg.anchor_goal
                elif info["hit_wall"]:
                    r += cfg.anchor_wall
            done = bool(info["outcome"] != "running")
            replay.push(state, action, r, next_state, done)
            state = next_state
            ep_reward += r
            global_step += 1
            for _ in range(cfg.updates_per_episode // cfg.max_steps if cfg.max_steps > 0 else 1):
                loss = optimize_dqn(q_net, target_net, opt, replay, cfg, device)
                if loss is not None:
                    losses.append(loss)
            if global_step % cfg.target_update_interval == 0:
                target_net.load_state_dict(q_net.state_dict())
            if done:
                outcome = str(info["outcome"])
                break

        # Episode-end updates to keep total updates stable.
        for _ in range(cfg.updates_per_episode):
            loss = optimize_dqn(q_net, target_net, opt, replay, cfg, device)
            if loss is not None:
                losses.append(loss)
        actual_steps = env.steps if env.steps > 0 else cfg.max_steps
        row = {
            "episode": ep,
            "epsilon": eps,
            "reward": ep_reward,
            "success": 1.0 if outcome == "success" else 0.0,
            "wall": 1.0 if (outcome != "success" and wall_hits > 0) else 0.0,
            "timeout": 1.0 if (outcome != "success" and wall_hits == 0) else 0.0,
            "steps": float(actual_steps),
            "repeat": float(repeats),
            "bfs_gap": float(max(0, actual_steps - bfs_len) if outcome == "success" else cfg.max_steps - bfs_len),
            "bfs_agree": float(bfs_agree_count / max(1, actual_steps)),
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "buffer": len(replay),
        }
        history.append(row)
        if ep % 100 == 0 or ep == 1:
            recent = history[-100:]
            print(
                f"ep={ep:4d}/{cfg.episodes} eps={eps:.3f} "
                f"reward={np.mean([x['reward'] for x in recent]):8.2f} "
                f"success={100*np.mean([x['success'] for x in recent]):6.2f}% "
                f"wall={100*np.mean([x['wall'] for x in recent]):6.2f}% "
                f"steps={np.mean([x['steps'] for x in recent]):6.2f} "
                f"repeat={np.mean([x['repeat'] for x in recent]):6.2f} "
                f"bfsGap={np.mean([x['bfs_gap'] for x in recent]):6.2f} "
                f"loss={row['loss']:.4f} buf={len(replay)}"
            )

    model_path = out_dir / "v2.0.4_qcnn_from_preference_reward.pt"
    torch.save({
        "version": VERSION,
        "model_state_dict": q_net.state_dict(),
        "reward_model": str(reward_model_path),
        "args": vars(args),
        "history": history,
    }, model_path)
    with open(out_dir / "v2.0.4_qcnn_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    plot_q_history(history, out_dir / "v2.0.4_qcnn_curves.png")

    evals = evaluate_all(q_net, rm, cfg, device, args.eval_n)
    with open(out_dir / "v2.0.4_qcnn_eval.json", "w", encoding="utf-8") as f:
        json.dump(evals, f, indent=2, ensure_ascii=False)
    print("\n=== Evaluation Suite ===")
    for name, m in evals.items():
        print(
            f"{name:7s} success={100*m['success']:6.2f}% wall={100*m['wall']:6.2f}% "
            f"timeout={100*m['timeout']:6.2f}% steps={m['steps']:6.2f} "
            f"repeat={m['repeat']:6.2f} bfsGap={m['bfs_gap']:6.2f} reward={m['reward']:7.2f}"
        )
    if not getattr(args, "no_debug", False):
        run_reward_model_debug(rm, args, device, out_dir, tag="rm_after_q_train")
        run_q_policy_debug(q_net, rm, cfg, device, out_dir, tag="q_policy_test")
    print(f"[Save] qcnn model: {model_path}")
    return model_path


def evaluate_all(q_net: QNetwork, rm: TransitionRewardCNN, cfg: DQNConfig, device: torch.device, n: int) -> Dict[str, Dict[str, float]]:
    return {
        "easy": rollout_eval(q_net, rm, "easy", n, cfg, device),
        "medium": rollout_eval(q_net, rm, "medium", n, cfg, device),
        "hard": rollout_eval(q_net, rm, "hard", n, cfg, device),
        "test": rollout_eval(q_net, rm, "easy", 1, cfg, device, fixed_test=True),
    }


def plot_q_history(history: List[Dict[str, float]], path: Path) -> None:
    episodes = [h["episode"] for h in history]
    panels = [
        ("Reward moving average", "reward"),
        ("Success moving average", "success"),
        ("Wall moving average", "wall"),
        ("BFS-gap moving average", "bfs_gap"),
        ("Repeat moving average", "repeat"),
    ]
    fig = plt.figure(figsize=(12, 10))
    for idx, (title, key) in enumerate(panels, start=1):
        plt.subplot(3, 2, idx)
        ys = moving_average([h[key] for h in history], 50)
        plt.plot(episodes, ys)
        plt.title(title)
        plt.xlabel("episode")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# -----------------------------
# Test mode
# -----------------------------

def load_q_model(path: str, device: torch.device) -> QNetwork:
    model = QNetwork().to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_rm_model(path: str, device: torch.device) -> TransitionRewardCNN:
    model = TransitionRewardCNN().to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# -----------------------------
# CLI
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v2.0.4 trajectory preference reward model -> QCNN")
    p.add_argument("--mode", choices=["all", "train-rm", "train-q", "test"], default="all")
    p.add_argument("--output-dir", default="./v2.0.4_preference_reward_model")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")

    # Difficulty distribution defaults are the established v2.0 baseline.
    p.add_argument("--easy-ratio", type=float, default=0.4)
    p.add_argument("--medium-ratio", type=float, default=0.3)
    p.add_argument("--hard-ratio", type=float, default=0.3)
    p.add_argument("--max-steps", type=int, default=64)

    # Recommended reward-model defaults.
    p.add_argument("--rm-mazes", type=int, default=1000)
    p.add_argument("--trajectories-per-maze", type=int, default=10)
    p.add_argument("--pairs-per-maze", type=int, default=16)
    p.add_argument("--rm-epochs", type=int, default=20)
    p.add_argument("--rm-batch-size", type=int, default=16)
    p.add_argument("--rm-lr", type=float, default=1e-4)
    p.add_argument("--rm-grad-clip", type=float, default=5.0)
    p.add_argument("--rm-val-ratio", type=float, default=0.15)

    # Recommended QCNN defaults.
    p.add_argument("--reward-model", default="")
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--replay-size", type=int, default=100000)
    p.add_argument("--warmup-steps", type=int, default=3000)
    p.add_argument("--updates-per-episode", type=int, default=64)
    p.add_argument("--target-update-interval", type=int, default=500)
    p.add_argument("--epsilon-decay-episodes", type=int, default=2500)
    p.add_argument("--rm-scale", type=float, default=2.0)
    p.add_argument("--eval-n", type=int, default=100)

    # Diagnostics are on by default in this debug version.
    p.add_argument("--debug-probe-mazes", type=int, default=30)
    p.add_argument("--no-debug", action="store_true")

    # Debug-only anchors, default off by design.
    p.add_argument("--anchor-terminal", action="store_true")
    p.add_argument("--anchor-goal", type=float, default=64.0)
    p.add_argument("--anchor-wall", type=float, default=-32.0)

    # Test mode. --test is kept as a compatibility alias for --mode test.
    p.add_argument("--model", default="")
    p.add_argument("--test", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if getattr(args, "test", False):
        args.mode = "test"
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir = ensure_dir(args.output_dir)
    print(f"=== {VERSION} ===")
    print(f"Device: {device}")
    print("Default design: offline BTL preference reward model, then frozen-RM QCNN.")
    print("No demo, no BFS margin, no action mask, no handcrafted step/repeat reward in main QCNN reward.")

    reward_model_path: Optional[Path] = None
    if args.mode in ["all", "train-rm"]:
        reward_model_path = train_reward_model(args, device, out_dir)

    if args.mode in ["all", "train-q"]:
        if reward_model_path is None:
            if not args.reward_model:
                raise ValueError("--reward-model is required for --mode train-q")
            reward_model_path = Path(args.reward_model)
        train_qcnn(args, device, out_dir, reward_model_path)

    if args.mode == "test":
        if not args.model:
            args.model = str(out_dir / "v2.0.4_qcnn_from_preference_reward.pt")
        if not args.reward_model:
            args.reward_model = str(out_dir / "v2.0.4_reward_model.pt")
        if not Path(args.model).exists() or not Path(args.reward_model).exists():
            raise ValueError("test mode needs --model and --reward-model, or the default files in --output-dir")
        q = load_q_model(args.model, device)
        rm = load_rm_model(args.reward_model, device)
        cfg = DQNConfig(episodes=args.episodes, eval_n=args.eval_n, rm_scale=args.rm_scale, anchor_terminal=args.anchor_terminal)
        metrics = rollout_eval(q, rm, "easy", 1, cfg, device, fixed_test=True)
        print("=== Fixed canonical test maze ===")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
