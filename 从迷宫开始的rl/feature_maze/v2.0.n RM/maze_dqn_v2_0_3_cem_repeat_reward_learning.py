import argparse
import csv
import json
import math
import random
from collections import deque, namedtuple, Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ============================================================
# v2.0.3 CEM Repeat Reward Learning
# Purpose:
#   Minimal v2.0 baseline focused on reward scaling plus repeat penalty.
#   - Fixed mixed difficulty distribution: easy 0.4 / medium 0.3 / hard 0.3.
#   - Uniform replay only.
#   - Double DQN only.
#   - State has 4 channels: wall, agent, goal, visited.
#   - CEM learns normalized reward ratios (g, w, u): goal, wall, repeat.
#   - No demo, no BFS margin, no action mask.
#   - --test runs only the canonical fixed hand-written maze.
# ============================================================

VERSION = "v2.0.3_cem_repeat_reward_learning"
SIZE = 8

ACTIONS = [
    (-1, 0),  # UP
    (1, 0),   # DOWN
    (0, -1),  # LEFT
    (0, 1),   # RIGHT
]
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]

Transition = namedtuple(
    "Transition",
    [
        "state",
        "action",
        "reward",
        "next_state",
        "done",
        "maze_type",
        "difficulty",
        "hit_wall",
        "success_episode",
        "progress",
        "step_index",
        "episode_id",
        "is_demo",
        "demo_action",
    ],
)

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

@dataclass
class TrainConfig:
    output_dir: str = "./v2.0.3_cem_repeat_reward_learning"
    output_name: str = "v2.0.3_cem_repeat_reward_learning"
    seed: int = 42

    episodes: int = 3000
    max_steps: int = 64
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 128
    replay_size: int = 100_000
    warmup_steps: int = 3000
    train_every: int = 1
    updates_per_episode: int = 64

    # v2.0.1 uses Double DQN only.
    target_mode: str = "double"
    target_update_interval: int = 500

    # Epsilon schedule: linear from epsilon_start to epsilon_end by epsilon_decay_episodes.
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 2500

    # Stable mixed training distribution.
    easy_ratio: float = 0.4
    medium_ratio: float = 0.3
    hard_ratio: float = 0.3

    reward_preset: str = "custom"
    goal_reward: float = 64.0
    wall_penalty: float = -32.0
    step_penalty: float = -1.0
    repeat_penalty: float = -32.0

    # Optional ablations. Default stays clean baseline.
    use_demo: bool = False
    demo_ratio: float = 0.25
    demo_per_episode: int = 1

    use_margin: bool = False
    margin: float = 1.0
    margin_lambda: float = 1.0
    margin_source: str = "demo"  # demo or bfs_online
    use_bfs: bool = False

    eval_n: int = 100
    save_gif: bool = True
    device: str = "auto"


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg: TrainConfig) -> torch.device:
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def apply_reward_preset(cfg: TrainConfig) -> None:
    # v2.0.1 uses directly supplied reward parameters.
    # This function remains for compatibility with train()/main() calls.
    return None


def ensure_output_dir(cfg: TrainConfig) -> Path:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def in_bounds(pos: Tuple[int, int]) -> bool:
    r, c = pos
    return 0 <= r < SIZE and 0 <= c < SIZE


def is_free(grid: np.ndarray, pos: Tuple[int, int]) -> bool:
    r, c = pos
    return in_bounds(pos) and grid[r, c] == 0


def weighted_choice(weight_map: Dict[str, float]) -> str:
    keys = list(weight_map.keys())
    weights = np.asarray([float(weight_map[k]) for k in keys], dtype=np.float64)
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("sum of weights must be positive")
    weights = weights / total
    return str(np.random.choice(keys, p=weights))


def epsilon_by_episode(ep: int, cfg: TrainConfig) -> float:
    if ep >= cfg.epsilon_decay_episodes:
        return cfg.epsilon_end
    frac = max(0.0, min(1.0, ep / max(1, cfg.epsilon_decay_episodes)))
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


# ============================================================
# Maze generation
# ============================================================

def parse_maze_text(text: str) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) != SIZE:
        raise ValueError(f"maze must have {SIZE} rows, got {len(lines)}")
    grid = np.zeros((SIZE, SIZE), dtype=np.int64)
    start = None
    goal = None
    for r, line in enumerate(lines):
        if len(line) != SIZE:
            raise ValueError(f"row {r} must have {SIZE} chars, got {len(line)}")
        for c, ch in enumerate(line):
            if ch == "#":
                grid[r, c] = 1
            elif ch == ".":
                grid[r, c] = 0
            elif ch == "S":
                grid[r, c] = 0
                start = (r, c)
            elif ch == "G":
                grid[r, c] = 0
                goal = (r, c)
            else:
                raise ValueError(f"invalid char {ch!r}")
    if start is None or goal is None:
        raise ValueError("maze must contain S and G")
    return grid, start, goal


def random_free_cell(grid: np.ndarray) -> Tuple[int, int]:
    free = np.argwhere(grid == 0)
    r, c = free[random.randrange(len(free))]
    return int(r), int(c)


def bfs_shortest_path(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    q = deque([start])
    parent = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        r, c = cur
        for dr, dc in ACTIONS:
            nxt = (r + dr, c + dc)
            if is_free(grid, nxt) and nxt not in parent:
                parent[nxt] = cur
                q.append(nxt)
    if goal not in parent:
        return None
    path: List[Tuple[int, int]] = []
    cur: Optional[Tuple[int, int]] = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def bfs_distance(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[int]:
    path = bfs_shortest_path(grid, start, goal)
    return None if path is None else len(path) - 1


def bfs_best_actions(grid: np.ndarray, pos: Tuple[int, int], goal: Tuple[int, int]) -> List[int]:
    d0 = bfs_distance(grid, pos, goal)
    if d0 is None:
        return []
    out = []
    r, c = pos
    for a, (dr, dc) in enumerate(ACTIONS):
        nxt = (r + dr, c + dc)
        if not is_free(grid, nxt):
            continue
        d1 = bfs_distance(grid, nxt, goal)
        if d1 is not None and d1 < d0:
            out.append(a)
    return out


def action_between(pos: Tuple[int, int], nxt: Tuple[int, int]) -> int:
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
    visited: Optional[set] = None,
) -> np.ndarray:
    """Four-channel state: walls, agent, goal, visited-in-current-episode.

    v2.0.3 adds the visited channel because repeat penalty depends on episode
    history. This keeps the state Markov for the new reward definition.
    """
    walls = grid.astype(np.float32)
    agent = np.zeros((SIZE, SIZE), dtype=np.float32)
    agent[pos] = 1.0
    goal_ch = np.zeros((SIZE, SIZE), dtype=np.float32)
    goal_ch[goal] = 1.0
    visited_ch = np.zeros((SIZE, SIZE), dtype=np.float32)
    if visited is not None:
        for vr, vc in visited:
            if 0 <= vr < SIZE and 0 <= vc < SIZE:
                visited_ch[vr, vc] = 1.0
    return np.stack([walls, agent, goal_ch, visited_ch], axis=0)


def generate_random_maze(difficulty: str, max_tries: int = 5000) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], str, str]:
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
            if abs(start[0] - goal[0]) + abs(start[1] - goal[1]) < 6:
                continue
            path = bfs_shortest_path(grid, start, goal)
            if path is None:
                continue
            length = len(path) - 1
            if min_len <= length <= max_len:
                return grid, start, goal, f"random_{difficulty}", difficulty
    raise RuntimeError(f"failed to generate random maze: {difficulty}")


def sample_train_maze(cfg: TrainConfig) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], str, str]:
    difficulty = weighted_choice({"easy": cfg.easy_ratio, "medium": cfg.medium_ratio, "hard": cfg.hard_ratio})
    return generate_random_maze(difficulty)


# ============================================================
# Model and replay
# ============================================================

class CNN_DQN(nn.Module):
    def __init__(self, in_channels: int = 4, num_actions: int = 4):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buffer: deque = deque(maxlen=self.capacity)
        self.demo_buffer: deque = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, tr: Transition) -> None:
        self.buffer.append(tr)
        if tr.is_demo:
            self.demo_buffer.append(tr)

    def push_many(self, transitions: List[Transition]) -> None:
        for tr in transitions:
            self.push(tr)

    def sample_uniform(self, n: int) -> List[Transition]:
        n = min(n, len(self.buffer))
        return random.sample(self.buffer, n)

    def sample_demo(self, n: int) -> List[Transition]:
        if len(self.demo_buffer) == 0 or n <= 0:
            return []
        if len(self.demo_buffer) >= n:
            return random.sample(self.demo_buffer, n)
        return [random.choice(self.demo_buffer) for _ in range(n)]

    def sample(self, batch_size: int, cfg: TrainConfig) -> List[Transition]:
        if cfg.use_demo and cfg.demo_ratio > 0 and len(self.demo_buffer) > 0:
            demo_n = int(round(batch_size * cfg.demo_ratio))
            demo_part = self.sample_demo(demo_n)
            rest_n = batch_size - len(demo_part)
            uniform_part = self.sample_uniform(rest_n)
            batch = demo_part + uniform_part
            if len(batch) < batch_size and len(self.buffer) > 0:
                batch.extend(self.sample_uniform(batch_size - len(batch)))
            random.shuffle(batch)
            return batch
        return self.sample_uniform(batch_size)


# ============================================================
# Transition construction
# ============================================================

def step_env(
    grid: np.ndarray,
    pos: Tuple[int, int],
    action: int,
    goal: Tuple[int, int],
    cfg: TrainConfig,
    visited: Optional[set] = None,
) -> Tuple[Tuple[int, int], float, bool, bool, bool]:
    r, c = pos
    dr, dc = ACTIONS[action]
    nxt = (r + dr, c + dc)
    reward = cfg.step_penalty
    hit_wall = False
    repeated = False
    if not is_free(grid, nxt):
        reward += cfg.wall_penalty
        hit_wall = True
        nxt = pos
        done = False
    elif nxt == goal:
        reward += cfg.goal_reward
        done = True
    else:
        done = False
        if visited is not None and nxt in visited:
            reward += cfg.repeat_penalty
            repeated = True
    return nxt, float(reward), bool(done), bool(hit_wall), bool(repeated)


def make_transition(
    grid: np.ndarray,
    pos: Tuple[int, int],
    action: int,
    reward: float,
    next_pos: Tuple[int, int],
    done: bool,
    goal: Tuple[int, int],
    maze_type: str,
    difficulty: str,
    hit_wall: bool,
    success_episode: bool,
    step_index: int,
    episode_id: int,
    is_demo: bool,
    demo_action: int,
    visited: Optional[set] = None,
    next_visited: Optional[set] = None,
) -> Transition:
    old_dist = bfs_distance(grid, pos, goal)
    new_dist = bfs_distance(grid, next_pos, goal)
    progress = False
    if old_dist is not None and new_dist is not None:
        progress = new_dist < old_dist
    return Transition(
        state=state_from_grid_pos_goal(grid, pos, goal, visited),
        action=int(action),
        reward=float(reward),
        next_state=state_from_grid_pos_goal(grid, next_pos, goal, next_visited),
        done=bool(done),
        maze_type=maze_type,
        difficulty=difficulty,
        hit_wall=bool(hit_wall),
        success_episode=bool(success_episode),
        progress=bool(progress),
        step_index=int(step_index),
        episode_id=int(episode_id),
        is_demo=bool(is_demo),
        demo_action=int(demo_action),
    )


def build_demo_transitions(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    cfg: TrainConfig,
    episode_id: int,
    difficulty: str,
) -> List[Transition]:
    path = bfs_shortest_path(grid, start, goal)
    if path is None or len(path) < 2:
        return []
    out = []
    for i in range(len(path) - 1):
        pos = path[i]
        nxt = path[i + 1]
        action = action_between(pos, nxt)
        reward = cfg.step_penalty + (cfg.goal_reward if nxt == goal else 0.0)
        done = nxt == goal
        out.append(
            make_transition(
                grid=grid,
                pos=pos,
                action=action,
                reward=reward,
                next_pos=nxt,
                done=done,
                goal=goal,
                maze_type=f"random_{difficulty}_demo",
                difficulty=difficulty,
                hit_wall=False,
                success_episode=True,
                step_index=i,
                episode_id=episode_id,
                is_demo=True,
                demo_action=action,
            )
        )
    return out


# ============================================================
# Training
# ============================================================

def select_action(policy_net: CNN_DQN, state: np.ndarray, epsilon: float, device: torch.device) -> int:
    if random.random() < epsilon:
        return random.randrange(len(ACTIONS))
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q = policy_net(s)[0]
        return int(torch.argmax(q).item())


def compute_target(
    policy_net: CNN_DQN,
    target_net: CNN_DQN,
    next_states: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    cfg: TrainConfig,
) -> torch.Tensor:
    with torch.no_grad():
        if cfg.target_mode == "double":
            next_actions = policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = target_net(next_states).gather(1, next_actions).squeeze(1)
        elif cfg.target_mode == "target":
            next_q = target_net(next_states).max(dim=1).values
        elif cfg.target_mode == "none":
            next_q = policy_net(next_states).max(dim=1).values
        else:
            raise ValueError(f"unknown target_mode: {cfg.target_mode}")
        return rewards + cfg.gamma * next_q * (1.0 - dones)


def optimize_model(
    policy_net: CNN_DQN,
    target_net: CNN_DQN,
    replay: ReplayBuffer,
    optimizer: optim.Optimizer,
    cfg: TrainConfig,
    device: torch.device,
) -> Optional[Dict[str, float]]:
    if len(replay) < max(cfg.batch_size, cfg.warmup_steps):
        return None

    batch = replay.sample(cfg.batch_size, cfg)
    if len(batch) < cfg.batch_size:
        return None

    states = torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=device)
    actions = torch.tensor([b.action for b in batch], dtype=torch.long, device=device).unsqueeze(1)
    rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=device)
    next_states = torch.tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32, device=device)
    dones = torch.tensor([float(b.done) for b in batch], dtype=torch.float32, device=device)

    q_all = policy_net(states)
    q_values = q_all.gather(1, actions).squeeze(1)
    target_q = compute_target(policy_net, target_net, next_states, rewards, dones, cfg)
    td_loss = nn.functional.smooth_l1_loss(q_values, target_q)

    margin_loss = torch.tensor(0.0, device=device)
    margin_count = 0
    if cfg.use_margin and cfg.margin_lambda > 0:
        if cfg.margin_source == "demo":
            eligible = torch.tensor([b.is_demo for b in batch], dtype=torch.bool, device=device)
            expert_actions_list = [b.demo_action for b in batch]
        elif cfg.margin_source == "bfs_online":
            eligible_list: List[bool] = []
            expert_actions_list = []
            for b in batch:
                # Reconstruct current position and goal from one-hot channels.
                # state[1] is agent, state[2] is goal.
                agent_idx = int(np.argmax(b.state[1]))
                goal_idx = int(np.argmax(b.state[2]))
                pos = (agent_idx // SIZE, agent_idx % SIZE)
                goal = (goal_idx // SIZE, goal_idx % SIZE)
                grid = b.state[0].astype(np.int64)
                best = bfs_best_actions(grid, pos, goal)
                if best:
                    eligible_list.append(True)
                    expert_actions_list.append(best[0])
                else:
                    eligible_list.append(False)
                    expert_actions_list.append(0)
            eligible = torch.tensor(eligible_list, dtype=torch.bool, device=device)
        else:
            raise ValueError(f"unknown margin_source: {cfg.margin_source}")

        if bool(eligible.any().item()):
            q_m = q_all[eligible]
            expert_actions = torch.tensor(expert_actions_list, dtype=torch.long, device=device)[eligible]
            q_expert = q_m.gather(1, expert_actions.unsqueeze(1)).squeeze(1)
            margin_matrix = torch.full_like(q_m, float(cfg.margin))
            margin_matrix.scatter_(1, expert_actions.unsqueeze(1), 0.0)
            max_margin_q = (q_m + margin_matrix).max(dim=1).values
            margin_loss = torch.clamp(max_margin_q - q_expert, min=0.0).mean()
            margin_count = int(eligible.sum().item())

    loss = td_loss + float(cfg.margin_lambda) * margin_loss
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "td_loss": float(td_loss.item()),
        "margin_loss": float(margin_loss.item()),
        "margin_count": float(margin_count),
    }


def run_episode(
    policy_net: CNN_DQN,
    replay: ReplayBuffer,
    cfg: TrainConfig,
    device: torch.device,
    ep: int,
) -> Dict[str, Any]:
    grid, start, goal, maze_type, difficulty = sample_train_maze(cfg)

    # Demo is disabled in this version, but this branch is kept inert by config.
    if cfg.use_demo and cfg.demo_per_episode > 0:
        for _ in range(cfg.demo_per_episode):
            demo_transitions = build_demo_transitions(grid, start, goal, cfg, episode_id=ep, difficulty=difficulty)
            replay.push_many(demo_transitions)

    pos = start
    visited = {start}
    episode_transitions: List[Transition] = []
    path = [pos]
    total_reward = 0.0
    wall_hits = 0
    repeated_steps = 0
    progress_count = 0
    bfs_agree = 0
    bfs_total = 0
    success = False
    epsilon = epsilon_by_episode(ep, cfg)

    for t in range(cfg.max_steps):
        state = state_from_grid_pos_goal(grid, pos, goal, visited)
        action = select_action(policy_net, state, epsilon, device)
        best = bfs_best_actions(grid, pos, goal)
        if best:
            bfs_total += 1
            if action in best:
                bfs_agree += 1

        old_pos = pos
        old_visited = set(visited)
        nxt, reward, done, hit_wall, repeated = step_env(grid, pos, action, goal, cfg, visited)
        next_visited = set(visited)
        if not hit_wall:
            next_visited.add(nxt)
        tr = make_transition(
            grid=grid,
            pos=old_pos,
            action=action,
            reward=reward,
            next_pos=nxt,
            done=done,
            goal=goal,
            maze_type=maze_type,
            difficulty=difficulty,
            hit_wall=hit_wall,
            success_episode=False,
            step_index=t,
            episode_id=ep,
            is_demo=False,
            demo_action=-1,
            visited=old_visited,
            next_visited=next_visited,
        )
        episode_transitions.append(tr)
        total_reward += reward
        wall_hits += int(hit_wall)
        repeated_steps += int(repeated)
        progress_count += int(tr.progress)
        pos = nxt
        visited = next_visited
        path.append(pos)
        if done:
            success = True
            break

    rewritten: List[Transition] = []
    for tr in episode_transitions:
        rewritten.append(tr._replace(success_episode=success))
    replay.push_many(rewritten)

    steps = len(rewritten)
    return {
        "reward": float(total_reward),
        "success": bool(success),
        "steps": int(steps),
        "wall_hit_rate": wall_hits / max(1, steps),
        "repeat_rate": repeated_steps / max(1, steps),
        "progress_rate": progress_count / max(1, steps),
        "bfs_action_agreement": bfs_agree / max(1, bfs_total),
        "repeat_count": len(path) - len(set(path)),
        "difficulty": difficulty,
        "epsilon": epsilon,
    }

def moving_average(xs: List[float], window: int = 50) -> List[float]:
    if not xs:
        return []
    out = []
    q: deque = deque()
    s = 0.0
    for x in xs:
        q.append(float(x))
        s += float(x)
        if len(q) > window:
            s -= q.popleft()
        out.append(s / len(q))
    return out


def train(cfg: TrainConfig) -> Tuple[CNN_DQN, Dict[str, Any]]:
    apply_reward_preset(cfg)
    set_seed(cfg.seed)
    device = get_device(cfg)
    out_dir = ensure_output_dir(cfg)

    print("\n=== Train v2.0.3 CEM Repeat Reward Learning ===")
    print(f"Device: {device}")
    print("Kept: CNN-DQN, Double DQN, random easy/medium/hard mazes, uniform replay.")
    print("Deleted: stage curriculum, manual training, stratified replay, wall/success/progress/recent buffers.")
    print("No demo, no BFS oracle, no margin. Candidate reward is generated from normalized ratios (g,w,u).")
    print(f"difficulty_mix: easy={cfg.easy_ratio}, medium={cfg.medium_ratio}, hard={cfg.hard_ratio}")
    print(f"target_mode=double, use_demo=False, use_bfs=False, use_margin=False")
    print(f"epsilon: {cfg.epsilon_start} -> {cfg.epsilon_end} by ep {cfg.epsilon_decay_episodes}")
    print(f"reward: goal={cfg.goal_reward}, wall={cfg.wall_penalty}, step={cfg.step_penalty}, repeat={cfg.repeat_penalty}")

    policy_net = CNN_DQN().to(device)
    target_net = CNN_DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=cfg.lr)
    replay = ReplayBuffer(cfg.replay_size)

    history: Dict[str, List[Any]] = {
        "reward": [],
        "success": [],
        "wall_hit_rate": [],
        "repeat_rate": [],
        "progress_rate": [],
        "bfs_action_agreement": [],
        "repeat_count": [],
        "epsilon": [],
        "loss": [],
        "td_loss": [],
        "margin_loss": [],
        "margin_count": [],
        "difficulty": [],
        "buffer_size": [],
        "demo_buffer_size": [],
    }

    update_count = 0
    last_loss: Optional[Dict[str, float]] = None
    for ep in range(1, cfg.episodes + 1):
        metrics = run_episode(policy_net, replay, cfg, device, ep)
        for _ in range(cfg.updates_per_episode):
            update_count += 1
            if update_count % cfg.train_every == 0:
                loss_info = optimize_model(policy_net, target_net, replay, optimizer, cfg, device)
                if loss_info is not None:
                    last_loss = loss_info
            if cfg.target_mode in ("double", "target") and update_count % cfg.target_update_interval == 0:
                target_net.load_state_dict(policy_net.state_dict())

        history["reward"].append(metrics["reward"])
        history["success"].append(float(metrics["success"]))
        history["wall_hit_rate"].append(metrics["wall_hit_rate"])
        history["repeat_rate"].append(metrics.get("repeat_rate", 0.0))
        history["progress_rate"].append(metrics["progress_rate"])
        history["bfs_action_agreement"].append(metrics["bfs_action_agreement"])
        history["repeat_count"].append(metrics["repeat_count"])
        history["epsilon"].append(metrics["epsilon"])
        history["difficulty"].append(metrics["difficulty"])
        history["buffer_size"].append(len(replay.buffer))
        history["demo_buffer_size"].append(len(replay.demo_buffer))
        history["loss"].append(math.nan if last_loss is None else last_loss["loss"])
        history["td_loss"].append(math.nan if last_loss is None else last_loss["td_loss"])
        history["margin_loss"].append(math.nan if last_loss is None else last_loss["margin_loss"])
        history["margin_count"].append(0.0 if last_loss is None else last_loss["margin_count"])

        if ep % 100 == 0:
            w = 100
            print(
                f"ep={ep:4d}/{cfg.episodes} eps={metrics['epsilon']:.3f} "
                f"reward={np.nanmean(history['reward'][-w:]):8.2f} "
                f"success={100*np.mean(history['success'][-w:]):6.2f}% "
                f"wall={100*np.mean(history['wall_hit_rate'][-w:]):6.2f}% "
                f"progress={100*np.mean(history['progress_rate'][-w:]):6.2f}% "
                f"bfsAgree={100*np.mean(history['bfs_action_agreement'][-w:]):6.2f}% "
                f"repeat={np.mean(history['repeat_count'][-w:]):6.2f} "
                f"loss={np.nanmean(history['loss'][-w:]):7.4f} "
                f"buf={len(replay.buffer)} demoBuf={len(replay.demo_buffer)}"
            )

    model_path = out_dir / f"{cfg.output_name}.pt"
    torch.save(policy_net.state_dict(), model_path)
    print(f"[Save] model: {model_path}")

    eval_summary = evaluate_suite(policy_net, cfg, device)
    history_obj: Dict[str, Any] = {"config": asdict(cfg), "history": history, "eval": eval_summary}
    (out_dir / f"{cfg.output_name}_history.json").write_text(json.dumps(history_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"{cfg.output_name}_eval.json").write_text(json.dumps(eval_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    save_curves(cfg, history)

    return policy_net, history_obj


# ============================================================
# Evaluation and visualization
# ============================================================

def greedy_rollout(
    model: CNN_DQN,
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    cfg: TrainConfig,
    device: torch.device,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    max_steps = cfg.max_steps if max_steps is None else max_steps
    pos = start
    visited = {start}
    path = [pos]
    total_reward = 0.0
    wall_hits = 0
    repeated_steps = 0
    progress_count = 0
    bfs_agree = 0
    bfs_total = 0
    rows = []

    for t in range(max_steps):
        state = state_from_grid_pos_goal(grid, pos, goal, visited)
        with torch.no_grad():
            q = model(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))[0]
            action = int(torch.argmax(q).item())
        best = bfs_best_actions(grid, pos, goal)
        if best:
            bfs_total += 1
            bfs_agree += int(action in best)
        old = pos
        nxt, reward, done, hit_wall, repeated = step_env(grid, pos, action, goal, cfg, visited)
        old_d = bfs_distance(grid, old, goal)
        new_d = bfs_distance(grid, nxt, goal)
        progress = old_d is not None and new_d is not None and new_d < old_d
        progress_count += int(progress)
        wall_hits += int(hit_wall)
        repeated_steps += int(repeated)
        total_reward += reward
        if not hit_wall:
            visited.add(nxt)
        pos = nxt
        path.append(pos)
        rows.append({
            "step": t,
            "pos": str(old),
            "action": ACTION_NAMES[action],
            "next_pos": str(nxt),
            "hit_wall": bool(hit_wall),
            "repeated": bool(repeated),
            "reward": float(reward),
            "old_bfs_distance": old_d,
            "new_bfs_distance": new_d,
            "progress": bool(progress),
            "bfs_best_actions": [ACTION_NAMES[a] for a in best],
            "q_values": [float(x) for x in q.detach().cpu().numpy().tolist()],
        })
        if done:
            break

    steps = len(rows)
    return {
        "success": bool(pos == goal),
        "reward": float(total_reward),
        "steps": steps,
        "wall_hit_rate": wall_hits / max(1, steps),
        "repeat_rate": repeated_steps / max(1, steps),
        "progress_rate": progress_count / max(1, steps),
        "bfs_action_agreement": bfs_agree / max(1, bfs_total),
        "repeat_count": len(path) - len(set(path)),
        "path": path,
        "rows": rows,
    }

def evaluate_difficulty(model: CNN_DQN, difficulty: str, cfg: TrainConfig, device: torch.device, n: int) -> Dict[str, float]:
    vals = []
    for _ in range(n):
        grid, start, goal, _, _ = generate_random_maze(difficulty)
        vals.append(greedy_rollout(model, grid, start, goal, cfg, device))
    return {
        "success_rate": float(np.mean([v["success"] for v in vals])),
        "reward": float(np.mean([v["reward"] for v in vals])),
        "steps": float(np.mean([v["steps"] for v in vals])),
        "wall_hit_rate": float(np.mean([v["wall_hit_rate"] for v in vals])),
        "repeat_rate": float(np.mean([v.get("repeat_rate", 0.0) for v in vals])),
        "progress_rate": float(np.mean([v["progress_rate"] for v in vals])),
        "bfs_action_agreement": float(np.mean([v["bfs_action_agreement"] for v in vals])),
        "repeat_count": float(np.mean([v["repeat_count"] for v in vals])),
    }


def evaluate_test_maze(model: CNN_DQN, cfg: TrainConfig, device: torch.device) -> Dict[str, float]:
    grid, start, goal = parse_maze_text(TEST_MAZE_TEXT)
    res = greedy_rollout(model, grid, start, goal, cfg, device)
    return {
        "success_rate": float(res["success"]),
        "reward": float(res["reward"]),
        "steps": float(res["steps"]),
        "wall_hit_rate": float(res["wall_hit_rate"]),
        "repeat_rate": float(res.get("repeat_rate", 0.0)),
        "progress_rate": float(res["progress_rate"]),
        "bfs_action_agreement": float(res["bfs_action_agreement"]),
        "repeat_count": float(res["repeat_count"]),
    }


def evaluate_suite(model: CNN_DQN, cfg: TrainConfig, device: torch.device) -> Dict[str, Any]:
    print("\n=== Evaluation Suite ===")
    summary = {
        "easy": evaluate_difficulty(model, "easy", cfg, device, cfg.eval_n),
        "medium": evaluate_difficulty(model, "medium", cfg, device, cfg.eval_n),
        "hard": evaluate_difficulty(model, "hard", cfg, device, cfg.eval_n),
        "test": evaluate_test_maze(model, cfg, device),
    }
    for k, v in summary.items():
        print(
            f"{k:8s} success={100*v['success_rate']:6.2f}% "
            f"reward={v['reward']:8.2f} steps={v['steps']:7.2f} "
            f"wall={100*v['wall_hit_rate']:6.2f}% progress={100*v['progress_rate']:6.2f}% "
            f"bfsAgree={100*v['bfs_action_agreement']:6.2f}% repeat={v['repeat_count']:7.2f}"
        )
    return summary


def save_curves(cfg: TrainConfig, history: Dict[str, List[Any]]) -> Path:
    out_dir = ensure_output_dir(cfg)
    path = out_dir / f"{cfg.output_name}_curves.png"
    xs = list(range(1, len(history["reward"]) + 1))
    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
    panels = [
        ("Reward moving average", "reward"),
        ("Success-rate moving average", "success"),
        ("Wall-hit-rate moving average", "wall_hit_rate"),
        ("BFS progress-rate moving average", "progress_rate"),
        ("Repeat-count moving average", "repeat_count"),
    ]
    for ax, (title, key) in zip(axes, panels):
        ys = moving_average([float(x) if not (isinstance(x, float) and math.isnan(x)) else np.nan for x in history[key]], 50)
        ax.plot(xs, ys, label=cfg.output_name)
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"[Save] curves: {path}")
    return path


def draw_grid_image(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], path: List[Tuple[int, int]], frame_idx: int) -> np.ndarray:
    img = np.ones((SIZE, SIZE, 3), dtype=np.float32)
    img[grid == 1] = np.array([0.05, 0.05, 0.05])
    for p in path[: frame_idx + 1]:
        if p not in (start, goal):
            img[p] = np.array([0.45, 0.75, 1.0])
    img[start] = np.array([0.0, 0.85, 0.0])
    img[goal] = np.array([1.0, 0.2, 0.2])
    img[path[frame_idx]] = np.array([1.0, 0.85, 0.0])
    return img


def save_test_gif(model: CNN_DQN, cfg: TrainConfig, device: torch.device) -> Path:
    out_dir = ensure_output_dir(cfg)
    grid, start, goal = parse_maze_text(TEST_MAZE_TEXT)
    res = greedy_rollout(model, grid, start, goal, cfg, device)
    path = res["path"]
    gif_path = out_dir / f"{cfg.output_name}_test.gif"
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame_idx: int):
        ax.clear()
        ax.imshow(draw_grid_image(grid, start, goal, path, frame_idx))
        ax.set_xticks(np.arange(-0.5, SIZE, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, SIZE, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.set_title(
            f"{VERSION} test | frame={frame_idx+1}/{len(path)} | "
            f"success={res['success']} | wall={100*res['wall_hit_rate']:.1f}%"
        )
        return []

    ani = FuncAnimation(fig, update, frames=len(path), interval=250, blit=False)
    ani.save(gif_path, writer=PillowWriter(fps=4))
    plt.close(fig)
    print(f"[Save] test gif: {gif_path}")
    return gif_path




# ============================================================
# v2.0.3 CEM reward-ratio learning
# ============================================================

@dataclass
class CEMConfig:
    """Outer-loop CEM settings.

    The learned parameters are normalized reward ratios:
        g = G / c
        w = W / c
        u = U / c
    with normalized inner reward:
        goal = +g, wall = -w, step = -1, repeat = -1-u.

    For actual DQN candidate training we multiply all rewards by step_scale:
        G = step_scale * g
        W = step_scale * w
        U = step_scale * u
        c = step_scale
    so the practical environment reward becomes:
        goal = +G, wall = -W, step = -c, repeat = -c-U.
    """
    generations: int = 4
    population: int = 6
    elite: int = 2
    candidate_episodes: int = 2000
    final_episodes: int = 3000

    init_g: float = 40.0
    init_w: float = 14.0
    init_u: float = 2.0
    init_sigma: float = 0.35
    min_sigma: float = 0.05
    cem_alpha: float = 0.70

    # Actual reward scale used when converting normalized ratios to environment rewards.
    # Example: g=40, w=14, u=2, step_scale=2 -> goal=80, wall=-28, step=-2, repeat=-4.
    step_scale: float = 2.0

    # Candidate parameter clipping in normalized ratio space.
    min_g: float = 4.0
    max_g: float = 128.0
    min_w: float = 2.0
    max_w: float = 128.0
    min_u: float = 0.05
    max_u: float = 128.0

    # Candidate evaluation score. Success is evaluated by BFS step gap;
    # failures are only separated into timeout and wall-hit failures.
    timeout_penalty: float = 0.50
    wall_fail_penalty: float = 1.00

    # Evaluation mazes used for candidate comparison.
    candidate_eval_n: int = 60
    common_eval_seed: int = 20202

    # Keep candidate artifacts? False avoids saving many candidate models/curves.
    keep_candidate_models: bool = False


def clean_number(x: float) -> str:
    return f"{float(x):.4g}".replace("-", "m").replace(".", "p")


def force_clean_v202_config(cfg: TrainConfig) -> TrainConfig:
    """v2.0.3 CEM line is clean: Double DQN, uniform replay, no demo/BFS/margin."""
    cfg.target_mode = "double"
    cfg.use_demo = False
    cfg.use_bfs = False
    cfg.use_margin = False
    cfg.margin_source = "demo"
    cfg.demo_ratio = 0.0
    cfg.demo_per_episode = 0
    return cfg


def build_ratio_config(
    base: TrainConfig,
    cem: CEMConfig,
    g: float,
    w: float,
    u: float,
    generation: int,
    candidate_id: int,
    episodes: int,
    final: bool = False,
) -> TrainConfig:
    cfg = TrainConfig(**asdict(base))
    cfg = force_clean_v202_config(cfg)
    cfg.episodes = int(episodes)
    cfg.goal_reward = float(cem.step_scale * g)
    cfg.wall_penalty = float(-cem.step_scale * w)
    cfg.step_penalty = float(-cem.step_scale)
    cfg.repeat_penalty = float(-cem.step_scale * u)
    cfg.reward_preset = "custom"
    cfg.epsilon_decay_episodes = max(1, int(0.8 * cfg.episodes))
    if final:
        cfg.output_name = (
            f"v2.0.3_cem_final_"
            f"g{clean_number(g)}_w{clean_number(w)}_u{clean_number(u)}_scale{clean_number(cem.step_scale)}"
        )
    else:
        cfg.output_name = (
            f"v2.0.3_cem_gen{generation:02d}_cand{candidate_id:02d}_"
            f"g{clean_number(g)}_w{clean_number(w)}_u{clean_number(u)}"
        )
    return cfg

def sample_cem_candidates(
    mu: np.ndarray,
    sigma: np.ndarray,
    cem: CEMConfig,
    rng: np.random.Generator,
) -> List[Dict[str, float]]:
    candidates: List[Dict[str, float]] = []
    for i in range(cem.population):
        log_g = float(rng.normal(mu[0], sigma[0]))
        log_w = float(rng.normal(mu[1], sigma[1]))
        log_u = float(rng.normal(mu[2], sigma[2]))
        g = float(np.clip(math.exp(log_g), cem.min_g, cem.max_g))
        w = float(np.clip(math.exp(log_w), cem.min_w, cem.max_w))
        u = float(np.clip(math.exp(log_u), cem.min_u, cem.max_u))
        candidates.append({
            "candidate_id": i,
            "g": g,
            "w": w,
            "u": u,
            "log_g": math.log(g),
            "log_w": math.log(w),
            "log_u": math.log(u),
        })
    return candidates

def make_common_eval_mazes(cfg: TrainConfig, n: int, seed: int) -> List[Tuple[str, np.ndarray, Tuple[int, int], Tuple[int, int]]]:
    """Create a shared mixed-difficulty evaluation set for fair candidate comparison."""
    state_random = random.getstate()
    state_numpy = np.random.get_state()
    set_seed(seed)
    weights = {"easy": cfg.easy_ratio, "medium": cfg.medium_ratio, "hard": cfg.hard_ratio}
    mazes: List[Tuple[str, np.ndarray, Tuple[int, int], Tuple[int, int]]] = []
    for _ in range(n):
        difficulty = weighted_choice(weights)
        grid, start, goal, _, _ = generate_random_maze(difficulty)
        mazes.append((difficulty, grid, start, goal))
    random.setstate(state_random)
    np.random.set_state(state_numpy)
    return mazes


def trajectory_quality_score(
    rollout: Dict[str, Any],
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    cfg: TrainConfig,
    cem: CEMConfig,
) -> float:
    """Outer candidate score for one trajectory.

    Primary target: reach goal in as few extra steps as possible relative to BFS.
    If success, score is in roughly [0,1], with 1 = BFS length.
    If failure, do not inspect repeat/progress. Only distinguish timeout from wall-hit failure.
    """
    if bool(rollout["success"]):
        bfs_len = bfs_distance(grid, start, goal)
        if bfs_len is None:
            return -cem.timeout_penalty
        denom = max(1.0, float(cfg.max_steps - bfs_len))
        gap = max(0.0, float(rollout["steps"] - bfs_len))
        normalized_gap = min(1.0, gap / denom)
        return float(1.0 - normalized_gap)
    if float(rollout.get("wall_hit_rate", 0.0)) > 0.0:
        return float(-cem.wall_fail_penalty)
    repeat_tiebreak = 0.10 * float(rollout.get("repeat_count", 0.0)) / max(1.0, float(cfg.max_steps))
    return float(-cem.timeout_penalty - repeat_tiebreak)


def evaluate_candidate_score(
    model: CNN_DQN,
    eval_mazes: List[Tuple[str, np.ndarray, Tuple[int, int], Tuple[int, int]]],
    cfg: TrainConfig,
    cem: CEMConfig,
    device: torch.device,
) -> Dict[str, Any]:
    scores: List[float] = []
    by_diff: Dict[str, List[float]] = {"easy": [], "medium": [], "hard": []}
    raw: List[Dict[str, Any]] = []
    for difficulty, grid, start, goal in eval_mazes:
        rollout = greedy_rollout(model, grid, start, goal, cfg, device)
        score = trajectory_quality_score(rollout, grid, start, goal, cfg, cem)
        scores.append(score)
        by_diff[difficulty].append(score)
        raw.append({
            "difficulty": difficulty,
            "score": score,
            "success": bool(rollout["success"]),
            "steps": float(rollout["steps"]),
            "wall_hit_rate": float(rollout["wall_hit_rate"]),
            "repeat_count": float(rollout["repeat_count"]),
            "repeat_rate": float(rollout.get("repeat_rate", 0.0)),
            "progress_rate": float(rollout["progress_rate"]),
            "bfs_action_agreement": float(rollout["bfs_action_agreement"]),
        })
    return {
        "score": float(np.mean(scores)) if scores else float("nan"),
        "score_std": float(np.std(scores)) if scores else float("nan"),
        "score_by_difficulty": {k: (float(np.mean(v)) if v else float("nan")) for k, v in by_diff.items()},
        "success_rate": float(np.mean([r["success"] for r in raw])) if raw else 0.0,
        "wall_hit_rate": float(np.mean([r["wall_hit_rate"] for r in raw])) if raw else 0.0,
        "steps": float(np.mean([r["steps"] for r in raw])) if raw else 0.0,
        "repeat_count": float(np.mean([r["repeat_count"] for r in raw])) if raw else 0.0,
        "repeat_rate": float(np.mean([r.get("repeat_rate", 0.0) for r in raw])) if raw else 0.0,
        "progress_rate": float(np.mean([r["progress_rate"] for r in raw])) if raw else 0.0,
        "bfs_action_agreement": float(np.mean([r["bfs_action_agreement"] for r in raw])) if raw else 0.0,
    }


def update_cem_distribution(
    mu: np.ndarray,
    sigma: np.ndarray,
    elites: List[Dict[str, Any]],
    cem: CEMConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    elite_logs = np.asarray([[float(e["log_g"]), float(e["log_w"]), float(e["log_u"])] for e in elites], dtype=np.float64)
    elite_mu = elite_logs.mean(axis=0)
    elite_sigma = elite_logs.std(axis=0)
    elite_sigma = np.maximum(elite_sigma, cem.min_sigma)
    new_mu = (1.0 - cem.cem_alpha) * mu + cem.cem_alpha * elite_mu
    new_sigma = (1.0 - cem.cem_alpha) * sigma + cem.cem_alpha * elite_sigma
    new_sigma = np.maximum(new_sigma, cem.min_sigma)
    return new_mu, new_sigma


def write_cem_summary(out_dir: Path, summary: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "v2.0.3_cem_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    rows: List[Dict[str, Any]] = []
    for gen in summary.get("generations", []):
        for cand in gen.get("candidates", []):
            row = {
                "generation": gen["generation"],
                "candidate_id": cand["candidate_id"],
                "g": cand["g"],
                "w": cand["w"],
                "u": cand.get("u", 0.0),
                "goal_reward": cand["goal_reward"],
                "wall_penalty": cand["wall_penalty"],
                "step_penalty": cand["step_penalty"],
                "repeat_penalty": cand.get("repeat_penalty", 0.0),
                "score": cand["score"],
                "is_elite": cand["is_elite"],
                "success_rate": cand["candidate_eval"]["success_rate"],
                "wall_hit_rate": cand["candidate_eval"]["wall_hit_rate"],
                "steps": cand["candidate_eval"]["steps"],
                "repeat_count": cand["candidate_eval"]["repeat_count"],
                "repeat_rate": cand["candidate_eval"].get("repeat_rate", 0.0),
                "progress_rate": cand["candidate_eval"]["progress_rate"],
                "bfs_action_agreement": cand["candidate_eval"]["bfs_action_agreement"],
            }
            rows.append(row)
    csv_path = out_dir / "v2.0.3_cem_summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    print(f"[Save] CEM summary json: {json_path}")
    if rows:
        print(f"[Save] CEM summary csv : {csv_path}")


def run_cem_reward_learning(base_cfg: TrainConfig, cem: CEMConfig) -> Dict[str, Any]:
    base_cfg = force_clean_v202_config(base_cfg)
    out_dir = ensure_output_dir(base_cfg)
    device = get_device(base_cfg)
    rng = np.random.default_rng(base_cfg.seed)

    mu = np.asarray([math.log(cem.init_g), math.log(cem.init_w), math.log(cem.init_u)], dtype=np.float64)
    sigma = np.asarray([cem.init_sigma, cem.init_sigma, cem.init_sigma], dtype=np.float64)
    summary: Dict[str, Any] = {
        "version": VERSION,
        "base_config": asdict(base_cfg),
        "cem_config": asdict(cem),
        "generations": [],
    }

    print("\n=== v2.0.3 CEM Repeat Reward Learning ===")
    print("Clean baseline: Double DQN, uniform replay, mixed difficulty, no demo/BFS/margin.")
    print("BFS is used only in the outer evaluation score as shortest-path reference, not in DQN training.")
    print(f"initial normalized ratios: g={cem.init_g}, w={cem.init_w}, u={cem.init_u}, sigma={cem.init_sigma}")
    print(f"actual scale: goal=step_scale*g, wall=-step_scale*w, step=-step_scale, repeat=-step_scale*u; step_scale={cem.step_scale}")

    global_best: Optional[Dict[str, Any]] = None
    for generation in range(cem.generations):
        print(f"\n=== CEM Generation {generation+1}/{cem.generations} ===")
        print(f"distribution center: g={math.exp(mu[0]):.4f}, w={math.exp(mu[1]):.4f}, u={math.exp(mu[2]):.4f}, sigma_log=({sigma[0]:.4f},{sigma[1]:.4f},{sigma[2]:.4f})")
        candidates = sample_cem_candidates(mu, sigma, cem, rng)
        eval_mazes = make_common_eval_mazes(base_cfg, cem.candidate_eval_n, cem.common_eval_seed + generation)
        cand_records: List[Dict[str, Any]] = []

        for cand in candidates:
            cid = int(cand["candidate_id"])
            g = float(cand["g"])
            w = float(cand["w"])
            u = float(cand["u"])
            cfg = build_ratio_config(base_cfg, cem, g, w, u, generation, cid, cem.candidate_episodes, final=False)
            cfg.seed = int(base_cfg.seed + generation * 1000 + cid)
            cfg.save_gif = False
            print(f"\n--- candidate gen={generation} id={cid}: g={g:.4f}, w={w:.4f}, u={u:.4f}, actual=(G={cfg.goal_reward:.4f}, W={-cfg.wall_penalty:.4f}, c={-cfg.step_penalty:.4f}, U={-cfg.repeat_penalty:.4f}) ---")
            model, obj = train(cfg)
            candidate_eval = evaluate_candidate_score(model, eval_mazes, cfg, cem, device)
            score = float(candidate_eval["score"])
            record = {
                **cand,
                "generation": generation,
                "goal_reward": cfg.goal_reward,
                "wall_penalty": cfg.wall_penalty,
                "step_penalty": cfg.step_penalty,
                "repeat_penalty": cfg.repeat_penalty,
                "score": score,
                "candidate_eval": candidate_eval,
                "train_eval": obj.get("eval", {}),
                "output_name": cfg.output_name,
                "is_elite": False,
            }
            cand_records.append(record)
            if global_best is None or score > float(global_best["score"]):
                global_best = record
            print(f"[Candidate score] J={score:.4f}, success={candidate_eval['success_rate']:.3f}, wall={candidate_eval['wall_hit_rate']:.3f}, steps={candidate_eval['steps']:.2f}, repeat={candidate_eval['repeat_count']:.2f}")

        cand_records.sort(key=lambda r: float(r["score"]), reverse=True)
        elites = cand_records[: max(1, cem.elite)]
        for e in elites:
            e["is_elite"] = True
        old_mu, old_sigma = mu.copy(), sigma.copy()
        mu, sigma = update_cem_distribution(mu, sigma, elites, cem)

        gen_obj = {
            "generation": generation,
            "old_mu_log": old_mu.tolist(),
            "old_sigma_log": old_sigma.tolist(),
            "old_center": {"g": math.exp(old_mu[0]), "w": math.exp(old_mu[1]), "u": math.exp(old_mu[2])},
            "new_mu_log": mu.tolist(),
            "new_sigma_log": sigma.tolist(),
            "new_center": {"g": math.exp(mu[0]), "w": math.exp(mu[1]), "u": math.exp(mu[2])},
            "elite_ids": [int(e["candidate_id"]) for e in elites],
            "candidates": cand_records,
        }
        summary["generations"].append(gen_obj)
        summary["current_best"] = global_best
        summary["current_distribution"] = {
            "mu_log": mu.tolist(),
            "sigma_log": sigma.tolist(),
            "center_g": math.exp(mu[0]),
            "center_w": math.exp(mu[1]),
            "center_u": math.exp(mu[2]),
        }
        write_cem_summary(out_dir, summary)
        print(f"[CEM update] center -> g={math.exp(mu[0]):.4f}, w={math.exp(mu[1]):.4f}, u={math.exp(mu[2]):.4f}; sigma=({sigma[0]:.4f},{sigma[1]:.4f},{sigma[2]:.4f})")

    learned_g = float(math.exp(mu[0]))
    learned_w = float(math.exp(mu[1]))
    learned_u = float(math.exp(mu[2]))
    print("\n=== Final Training with Learned Reward Ratios ===")
    print(f"learned normalized ratios: g={learned_g:.4f}, w={learned_w:.4f}, u={learned_u:.4f}")
    final_cfg = build_ratio_config(base_cfg, cem, learned_g, learned_w, learned_u, cem.generations, 0, cem.final_episodes, final=True)
    final_cfg.seed = base_cfg.seed + 999999
    final_model, final_obj = train(final_cfg)
    if final_cfg.save_gif:
        save_test_gif(final_model, final_cfg, device)
    summary["final"] = {
        "learned_g": learned_g,
        "learned_w": learned_w,
        "learned_u": learned_u,
        "actual_goal_reward": final_cfg.goal_reward,
        "actual_wall_penalty": final_cfg.wall_penalty,
        "actual_step_penalty": final_cfg.step_penalty,
        "actual_repeat_penalty": final_cfg.repeat_penalty,
        "output_name": final_cfg.output_name,
        "eval": final_obj.get("eval", {}),
    }
    write_cem_summary(out_dir, summary)
    return summary


def build_base_config_from_args(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig()
    for field in cfg.__dataclass_fields__:
        if hasattr(args, field):
            val = getattr(args, field)
            if val is not None:
                setattr(cfg, field, val)
    cfg.output_dir = args.output_dir or "./v2.0.3_cem_repeat_reward_learning"
    cfg.output_name = args.output_name or "v2.0.3_cem_repeat_reward_learning"
    return force_clean_v202_config(cfg)


def build_cem_config_from_args(args: argparse.Namespace) -> CEMConfig:
    cem = CEMConfig()
    for field in cem.__dataclass_fields__:
        if hasattr(args, field):
            val = getattr(args, field)
            if val is not None:
                setattr(cem, field, val)
    if cem.elite < 1 or cem.elite > cem.population:
        raise ValueError("--elite must be between 1 and --population")
    return cem


def load_model(model_path: str, device: torch.device) -> CNN_DQN:
    model = CNN_DQN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["cem", "final"], default="cem", help="cem learns reward ratios; final trains once from given final-g/final-w/final-u.")
    p.add_argument("--test", action="store_true", help="Run only the fixed canonical test maze.")
    p.add_argument("--model", type=str, default="")

    # Base DQN controls.
    p.add_argument("--output-dir", type=str, default="./v2.0.3_cem_repeat_reward_learning")
    p.add_argument("--output-name", type=str, default="")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--eval-n", type=int, dest="eval_n", default=None)

    # CEM controls.
    p.add_argument("--generations", type=int, default=None)
    p.add_argument("--population", type=int, default=None)
    p.add_argument("--elite", type=int, default=None)
    p.add_argument("--candidate-episodes", type=int, default=None)
    p.add_argument("--final-episodes", type=int, default=None)
    p.add_argument("--init-g", type=float, default=None)
    p.add_argument("--init-w", type=float, default=None)
    p.add_argument("--init-u", type=float, default=None)
    p.add_argument("--init-sigma", type=float, default=None)
    p.add_argument("--min-sigma", type=float, default=None)
    p.add_argument("--cem-alpha", type=float, default=None)
    p.add_argument("--step-scale", type=float, default=None)
    p.add_argument("--candidate-eval-n", type=int, default=None)
    p.add_argument("--common-eval-seed", type=int, default=None)
    p.add_argument("--timeout-penalty", type=float, default=None)
    p.add_argument("--wall-fail-penalty", type=float, default=None)

    # Manual final mode ratios.
    p.add_argument("--final-g", type=float, default=None)
    p.add_argument("--final-w", type=float, default=None)
    p.add_argument("--final-u", type=float, default=None)
    args = p.parse_args()

    base_cfg = build_base_config_from_args(args)
    cem_cfg = build_cem_config_from_args(args)
    set_seed(base_cfg.seed)
    device = get_device(base_cfg)

    if args.test:
        if not args.model:
            raise ValueError("--test requires --model")
        model = load_model(args.model, device)
        print("\n=== Fixed Canonical Test Maze ===")
        test_summary = evaluate_test_maze(model, base_cfg, device)
        print(json.dumps(test_summary, ensure_ascii=False, indent=2))
        if base_cfg.save_gif:
            save_test_gif(model, base_cfg, device)
        return

    if args.mode == "final":
        if args.final_g is None or args.final_w is None or args.final_u is None:
            raise ValueError("--mode final requires --final-g, --final-w, and --final-u")
        cfg = build_ratio_config(base_cfg, cem_cfg, float(args.final_g), float(args.final_w), float(args.final_u), 0, 0, cem_cfg.final_episodes, final=True)
        model, _ = train(cfg)
        save_test_gif(model, cfg, device)
        return

    run_cem_reward_learning(base_cfg, cem_cfg)


if __name__ == "__main__":
    main()
