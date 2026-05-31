
import argparse
import random
import math
from collections import deque, namedtuple, Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ============================================================
# 0. Global constants
# ============================================================

SIZE = 8

ACTIONS = [
    (-1, 0),  # UP
    (1, 0),   # DOWN
    (0, -1),  # LEFT
    (0, 1),   # RIGHT
]

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]

# Extended transition:
# Metadata is not fed into the model. It is only used for stratified replay
# and diagnostics.
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
    ],
)


REWARD_PRESETS = {
    # name: (goal_reward, wall_penalty, step_penalty)
    "g10_w10_s001": (10.0, -10.0, -0.01),
    "g10_w5_s001": (10.0, -5.0, -0.01),
    "g10_w2_s001": (10.0, -2.0, -0.01),
    "g10_w1_s001": (10.0, -1.0, -0.01),
}

EPSILON_PRESETS = {
    # These are interpreted by epsilon_by_episode.
    "normal": {
        "type": "linear",
        "start": 1.0,
        "end": 0.05,
        "decay_ratio": 0.75,
    },
    "slow": {
        "type": "linear",
        "start": 1.0,
        "end": 0.05,
        "decay_ratio": 1.25,
    },
    "high_floor": {
        "type": "linear",
        "start": 1.0,
        "end": 0.10,
        "decay_ratio": 1.00,
    },
    "two_phase": {
        "type": "two_phase",
        "start": 1.0,
        "mid": 0.20,
        "end": 0.05,
        "phase1_ratio": 0.40,
        "phase2_ratio": 1.00,
    },
}


@dataclass
class TrainConfig:
    size: int = 8

    episodes: int = 3000
    max_steps: int = 64

    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 128
    replay_size: int = 100_000
    warmup_steps: int = 2000

    reward_preset: str = "g10_w5_s001"
    epsilon_preset: str = "two_phase"
    maze_schedule: str = "curriculum"
    replay_type: str = "stratified"

    target_update_interval: int = 500
    train_every: int = 1

    goal_reward: float = 10.0
    wall_penalty: float = -5.0
    step_penalty: float = -0.01

    # Maze schedule config
    manual_ratio: float = 0.15

    # Model saving
    save_model: str = "diagnose_cnn_dqn.pt"


# ============================================================
# 1. Hand-made mazes
# ============================================================

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
# 2. Basic utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def apply_reward_preset(cfg: TrainConfig) -> None:
    if cfg.reward_preset not in REWARD_PRESETS:
        raise ValueError(f"unknown reward preset: {cfg.reward_preset}")
    goal_reward, wall_penalty, step_penalty = REWARD_PRESETS[cfg.reward_preset]
    cfg.goal_reward = goal_reward
    cfg.wall_penalty = wall_penalty
    cfg.step_penalty = step_penalty


def in_bounds(pos: Tuple[int, int]) -> bool:
    r, c = pos
    return 0 <= r < SIZE and 0 <= c < SIZE


def is_free(grid: np.ndarray, pos: Tuple[int, int]) -> bool:
    r, c = pos
    return in_bounds(pos) and grid[r, c] == 0


def parse_maze_text(text: str) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]

    if len(lines) != SIZE:
        raise ValueError(f"maze must have {SIZE} rows, got {len(lines)}")

    grid = np.zeros((SIZE, SIZE), dtype=np.int64)
    start = None
    goal = None

    for r, line in enumerate(lines):
        if len(line) != SIZE:
            raise ValueError(f"row {r} must have {SIZE} chars, got {len(line)}: {line}")

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
                raise ValueError(f"invalid char {ch!r} at row={r}, col={c}")

    if start is None:
        raise ValueError("maze missing S")
    if goal is None:
        raise ValueError("maze missing G")

    return grid, start, goal



def parse_topology_text(text: str) -> np.ndarray:
    """
    Parse a topology-only 8x8 map.
    Allowed chars:
      "." free cell
      "#" wall
    No S/G are allowed here. Manual start/goal must be sampled randomly.
    """
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
                raise ValueError(f"invalid char {ch!r} at row={r}, col={c}")

    return grid


def bfs_shortest_path(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    q = deque([start])
    parent = {start: None}

    while q:
        cur = q.popleft()
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


def bfs_distance(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[int]:
    path = bfs_shortest_path(grid, start, goal)
    if path is None:
        return None
    return len(path) - 1


def bfs_optimal_actions(grid: np.ndarray, pos: Tuple[int, int], goal: Tuple[int, int]) -> List[int]:
    """
    Diagnostic only. This is never fed into the model or reward.
    Returns actions that reduce BFS distance to goal.
    """
    cur_dist = bfs_distance(grid, pos, goal)
    if cur_dist is None:
        return []

    acts = []
    r, c = pos
    for a, (dr, dc) in enumerate(ACTIONS):
        nxt = (r + dr, c + dc)
        if not is_free(grid, nxt):
            continue
        nd = bfs_distance(grid, nxt, goal)
        if nd is not None and nd < cur_dist:
            acts.append(a)
    return acts


def valid_actions(grid: np.ndarray, pos: Tuple[int, int]) -> List[int]:
    acts = []
    r, c = pos
    for a, (dr, dc) in enumerate(ACTIONS):
        nxt = (r + dr, c + dc)
        if is_free(grid, nxt):
            acts.append(a)
    return acts


def is_action_valid(grid: np.ndarray, pos: Tuple[int, int], action: int) -> bool:
    r, c = pos
    dr, dc = ACTIONS[action]
    return is_free(grid, (r + dr, c + dc))


# ============================================================
# 3. Maze generation
# ============================================================

def transform_grid_randomly(
    grid: np.ndarray,
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    Randomly flip a topology. If start/goal are provided, transform them too.
    Manual maps normally call this with start=None, goal=None.
    """
    g = grid.copy()
    s = start
    t = goal

    if random.random() < 0.5:
        g = np.fliplr(g)
        if s is not None:
            s = (s[0], SIZE - 1 - s[1])
        if t is not None:
            t = (t[0], SIZE - 1 - t[1])

    if random.random() < 0.5:
        g = np.flipud(g)
        if s is not None:
            s = (SIZE - 1 - s[0], s[1])
        if t is not None:
            t = (SIZE - 1 - t[0], t[1])

    return g.copy(), s, t


def sample_random_start_goal_on_grid(
    grid: np.ndarray,
    min_manhattan: int = 6,
    min_shortest_len: int = 6,
    max_tries: int = 500,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Randomly sample start and goal from free cells.
    This is used by manual mazes and random mazes; no fixed S/G is used.
    """
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

    raise RuntimeError("failed to sample random start/goal on manual topology")


def sample_manual_difficulty(ep: Optional[int] = None, total_episodes: Optional[int] = None) -> str:
    """
    Manual mazes have only three categories: easy, medium, hard.
    If episode information is provided, use curriculum sampling.
    Otherwise sample uniformly for evaluation.
    """
    if ep is None or total_episodes is None:
        return random.choice(["easy", "medium", "hard"])

    progress = ep / max(1, total_episodes)
    if progress < 0.30:
        return weighted_choice({"easy": 0.60, "medium": 0.30, "hard": 0.10})
    if progress < 0.70:
        return weighted_choice({"easy": 0.25, "medium": 0.50, "hard": 0.25})
    return weighted_choice({"easy": 0.15, "medium": 0.45, "hard": 0.40})


def sample_manual_maze(
    manual_difficulty: Optional[str] = None,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], str, str]:
    """
    Sample a hand-designed topology from easy/medium/hard.
    Manual topology contains no fixed S/G. Start and goal are always sampled
    randomly from free cells after random flips.
    """
    if manual_difficulty is None:
        manual_difficulty = sample_manual_difficulty()

    if manual_difficulty not in MANUAL_MAZES:
        raise ValueError(f"unknown manual difficulty: {manual_difficulty}")

    topology_text = random.choice(MANUAL_MAZES[manual_difficulty])
    grid = parse_topology_text(topology_text)
    grid, _, _ = transform_grid_randomly(grid)

    min_len_by_diff = {
        "easy": 6,
        "medium": 8,
        "hard": 10,
    }

    start, goal = sample_random_start_goal_on_grid(
        grid,
        min_manhattan=6,
        min_shortest_len=min_len_by_diff[manual_difficulty],
    )

    return grid, start, goal, f"manual_{manual_difficulty}", manual_difficulty


def random_free_cell(grid: np.ndarray) -> Tuple[int, int]:

    free = np.argwhere(grid == 0)
    idx = random.randrange(len(free))
    r, c = free[idx]
    return int(r), int(c)


def generate_random_maze(
    difficulty: str,
    fixed_sg_prob: float,
    max_tries: int = 5000,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], str, str]:
    """
    Random maze generator with controlled difficulty.
    This replaces the previous split between random_fixed/random_var as the
    default training generator, while still supporting fixed/random start-goal
    mixtures internally.
    """
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

    # fixed_sg_prob is kept only for backward-compatible function calls.
    # Training no longer uses fixed start/goal.
    fixed_sg = False

    for _ in range(max_tries):
        p = random.uniform(obstacle_min, obstacle_max)
        grid = (np.random.rand(SIZE, SIZE) < p).astype(np.int64)

        # Choose start/goal after generating walls.
        # Force them to be reasonably far apart and connected.
        for _inner in range(100):
            start = random_free_cell(grid)
            goal = random_free_cell(grid)
            if start == goal:
                continue
            manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
            if manhattan >= 6:
                break
        else:
            continue

        grid[start] = 0
        grid[goal] = 0

        path = bfs_shortest_path(grid, start, goal)
        if path is None:
            continue

        length = len(path) - 1
        if length < min_len:
            continue
        if length > max_len:
            continue

        sg_tag = "random_sg"
        return grid, start, goal, f"random_{difficulty}_{sg_tag}", difficulty

    raise RuntimeError(f"failed to generate random maze: difficulty={difficulty}")


def curriculum_weights(ep: int, total_episodes: int) -> Dict[str, float]:
    """
    Returns difficulty weights for easy/medium/hard/manual.
    Manual is a structural supplement, not the main training distribution.
    """
    progress = ep / max(1, total_episodes)

    if progress < 0.30:
        return {
            "easy": 0.70,
            "medium": 0.20,
            "hard": 0.05,
            "manual": 0.05,
        }
    if progress < 0.70:
        return {
            "easy": 0.35,
            "medium": 0.40,
            "hard": 0.15,
            "manual": 0.10,
        }
    return {
        "easy": 0.20,
        "medium": 0.40,
        "hard": 0.25,
        "manual": 0.15,
    }


def fixed_sg_prob_by_phase(ep: int, total_episodes: int) -> float:
    """
    Deprecated compatibility hook. Start/goal are no longer fixed.
    """
    return 0.0



def weighted_choice(weights: Dict[str, float]) -> str:
    items = list(weights.items())
    names = [x[0] for x in items]
    probs = np.array([x[1] for x in items], dtype=np.float64)
    probs = probs / probs.sum()
    return str(np.random.choice(names, p=probs))


def sample_train_maze(cfg: TrainConfig, ep: int) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], str, str]:
    if cfg.maze_schedule == "curriculum":
        weights = curriculum_weights(ep, cfg.episodes)
        kind = weighted_choice(weights)
        if kind == "manual":
            manual_diff = sample_manual_difficulty(ep, cfg.episodes)
            return sample_manual_maze(manual_diff)
        fixed_prob = fixed_sg_prob_by_phase(ep, cfg.episodes)
        return generate_random_maze(kind, fixed_sg_prob=fixed_prob)

    if cfg.maze_schedule == "easy":
        return generate_random_maze("easy", fixed_sg_prob=0.50)

    if cfg.maze_schedule == "medium":
        return generate_random_maze("medium", fixed_sg_prob=0.30)

    if cfg.maze_schedule == "hard":
        return generate_random_maze("hard", fixed_sg_prob=0.10)

    if cfg.maze_schedule == "manual":
        return sample_manual_maze()

    raise ValueError(f"unknown maze_schedule: {cfg.maze_schedule}")


def sample_eval_maze(eval_type: str):
    if eval_type == "easy":
        return generate_random_maze("easy", fixed_sg_prob=0.20)

    if eval_type == "medium":
        return generate_random_maze("medium", fixed_sg_prob=0.10)

    if eval_type == "hard":
        return generate_random_maze("hard", fixed_sg_prob=0.00)

    if eval_type == "manual":
        return sample_manual_maze()

    if eval_type == "heldout":
        grid, start, goal = parse_maze_text(TEST_MAZE_TEXT)
        return grid, start, goal, "heldout", "heldout"

    raise ValueError(f"unknown eval_type: {eval_type}")


# ============================================================
# 4. Environment
# ============================================================

class MazeEnv:
    def __init__(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        cfg: TrainConfig,
        maze_type: str = "unknown",
        difficulty: str = "unknown",
    ):
        self.grid = grid.astype(np.int64)
        self.start = start
        self.goal = goal
        self.cfg = cfg
        self.maze_type = maze_type
        self.difficulty = difficulty

        path = bfs_shortest_path(self.grid, self.start, self.goal)
        if path is None:
            raise ValueError("unsolvable maze")

        self.shortest_len = len(path) - 1
        self.reset()

    def reset(self):
        self.pos = self.start
        self.steps = 0
        self.done = False

        self.path = [self.pos]

        self.wall_hits = 0
        self.valid_moves = 0
        self.invalid_moves = 0
        self.action_hist = Counter()
        self.bfs_progress_count = 0
        self.bfs_nonprogress_count = 0

        return self._state()

    def _state(self) -> np.ndarray:
        walls = self.grid.astype(np.float32)

        agent = np.zeros((SIZE, SIZE), dtype=np.float32)
        agent[self.pos] = 1.0

        goal = np.zeros((SIZE, SIZE), dtype=np.float32)
        goal[self.goal] = 1.0

        return np.stack([walls, agent, goal], axis=0)

    def step(self, action: int):
        if self.done:
            raise RuntimeError("step after done")

        self.steps += 1
        self.action_hist[action] += 1

        old_pos = self.pos
        old_dist = bfs_distance(self.grid, old_pos, self.goal)

        dr, dc = ACTIONS[action]
        r, c = self.pos
        nxt = (r + dr, c + dc)

        reward = self.cfg.step_penalty
        hit_wall = False
        moved = False

        if not is_free(self.grid, nxt):
            reward += self.cfg.wall_penalty
            nxt = self.pos
            hit_wall = True
            self.wall_hits += 1
            self.invalid_moves += 1
        else:
            self.pos = nxt
            moved = True
            self.valid_moves += 1

        self.path.append(self.pos)
        new_dist = bfs_distance(self.grid, self.pos, self.goal)

        progress = False
        if old_dist is not None and new_dist is not None and new_dist < old_dist:
            progress = True
            self.bfs_progress_count += 1
        else:
            self.bfs_nonprogress_count += 1

        if self.pos == self.goal:
            reward += self.cfg.goal_reward
            self.done = True
            return self._state(), reward, True, self._info(
                success=True,
                hit_wall=hit_wall,
                moved=moved,
                old_pos=old_pos,
                action=action,
                progress=progress,
                old_dist=old_dist,
                new_dist=new_dist,
            )

        if self.steps >= self.cfg.max_steps:
            self.done = True
            return self._state(), reward, True, self._info(
                success=False,
                hit_wall=hit_wall,
                moved=moved,
                old_pos=old_pos,
                action=action,
                progress=progress,
                old_dist=old_dist,
                new_dist=new_dist,
            )

        return self._state(), reward, False, self._info(
            success=False,
            hit_wall=hit_wall,
            moved=moved,
            old_pos=old_pos,
            action=action,
            progress=progress,
            old_dist=old_dist,
            new_dist=new_dist,
        )

    def _info(self, success, hit_wall, moved, old_pos, action, progress, old_dist, new_dist):
        return {
            "success": success,
            "steps": self.steps,
            "shortest_len": self.shortest_len,
            "maze_type": self.maze_type,
            "difficulty": self.difficulty,
            "hit_wall": hit_wall,
            "moved": moved,
            "wall_hits": self.wall_hits,
            "valid_moves": self.valid_moves,
            "invalid_moves": self.invalid_moves,
            "old_pos": old_pos,
            "pos": self.pos,
            "action": action,
            "valid_actions": valid_actions(self.grid, old_pos),
            "progress": progress,
            "old_dist": old_dist,
            "new_dist": new_dist,
            "bfs_progress_count": self.bfs_progress_count,
            "bfs_nonprogress_count": self.bfs_nonprogress_count,
        }


# ============================================================
# 5. CNN DQN
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


# ============================================================
# 6. Replay buffers
# ============================================================

class StratifiedReplayBuffer:
    """
    Stratified replay changes only the batch composition.
    It does not change the DQN target or the state input.

    Sub-buffers:
      - all: all transitions
      - success: transitions from successful episodes
      - wall: transitions that hit walls
      - progress: transitions that reduce BFS distance to goal
      - recent: recent transitions
    """

    def __init__(self, capacity: int, recent_capacity: int = 10_000):
        self.capacity = capacity

        self.all_buffer = deque(maxlen=capacity)
        self.success_buffer = deque(maxlen=capacity)
        self.wall_buffer = deque(maxlen=capacity)
        self.progress_buffer = deque(maxlen=capacity)
        self.recent_buffer = deque(maxlen=recent_capacity)

        self.type_counts = Counter()
        self.difficulty_counts = Counter()

    def push(self, transition: Transition):
        self.all_buffer.append(transition)
        self.recent_buffer.append(transition)

        self.type_counts[transition.maze_type] += 1
        self.difficulty_counts[transition.difficulty] += 1

        if transition.success_episode:
            self.success_buffer.append(transition)
        if transition.hit_wall:
            self.wall_buffer.append(transition)
        if transition.progress:
            self.progress_buffer.append(transition)

    def push_episode(self, transitions: List[Transition]):
        for tr in transitions:
            self.push(tr)

    def __len__(self):
        return len(self.all_buffer)

    def _sample_from(self, buf: deque, n: int) -> List[Transition]:
        if n <= 0 or len(buf) == 0:
            return []
        n = min(n, len(buf))
        return random.sample(list(buf), n)

    def sample_uniform(self, batch_size: int) -> List[Transition]:
        return self._sample_from(self.all_buffer, batch_size)

    def sample_stratified(self, batch_size: int) -> List[Transition]:
        # Proposed composition for batch_size=128:
        # uniform 48, success 32, wall 16, progress 16, recent 16.
        # Scale if batch_size differs.
        ratios = {
            "uniform": 48 / 128,
            "success": 32 / 128,
            "wall": 16 / 128,
            "progress": 16 / 128,
            "recent": 16 / 128,
        }

        counts = {k: int(round(batch_size * v)) for k, v in ratios.items()}

        # Correct rounding mismatch.
        diff = batch_size - sum(counts.values())
        counts["uniform"] += diff

        samples: List[Transition] = []
        samples += self._sample_from(self.all_buffer, counts["uniform"])
        samples += self._sample_from(self.success_buffer, counts["success"])
        samples += self._sample_from(self.wall_buffer, counts["wall"])
        samples += self._sample_from(self.progress_buffer, counts["progress"])
        samples += self._sample_from(self.recent_buffer, counts["recent"])

        # Fill shortage from uniform buffer.
        if len(samples) < batch_size:
            need = batch_size - len(samples)
            samples += self._sample_from(self.all_buffer, need)

        # If still short, it means total buffer is smaller than batch_size.
        # optimize_model checks warmup, so this should rarely happen.
        if len(samples) > batch_size:
            samples = random.sample(samples, batch_size)

        random.shuffle(samples)
        return samples

    def sample(self, batch_size: int, replay_type: str) -> List[Transition]:
        if replay_type == "uniform":
            return self.sample_uniform(batch_size)
        if replay_type == "stratified":
            return self.sample_stratified(batch_size)
        raise ValueError(f"unknown replay_type: {replay_type}")

    def stats(self) -> Dict[str, float]:
        total = max(1, len(self.all_buffer))

        return {
            "buffer_size": len(self.all_buffer),
            "success_ratio": len(self.success_buffer) / total,
            "wall_ratio": len(self.wall_buffer) / total,
            "progress_ratio": len(self.progress_buffer) / total,
            "recent_size": len(self.recent_buffer),
        }


# ============================================================
# 7. DQN tools
# ============================================================

def epsilon_by_episode(ep: int, cfg: TrainConfig) -> float:
    if cfg.epsilon_preset not in EPSILON_PRESETS:
        raise ValueError(f"unknown epsilon preset: {cfg.epsilon_preset}")

    preset = EPSILON_PRESETS[cfg.epsilon_preset]
    typ = preset["type"]

    if typ == "linear":
        start = float(preset["start"])
        end = float(preset["end"])
        decay_ratio = float(preset["decay_ratio"])
        decay_episodes = max(1, int(cfg.episodes * decay_ratio))
        t = min(1.0, ep / decay_episodes)
        return start + t * (end - start)

    if typ == "two_phase":
        start = float(preset["start"])
        mid = float(preset["mid"])
        end = float(preset["end"])
        phase1_ratio = float(preset["phase1_ratio"])
        phase2_ratio = float(preset["phase2_ratio"])

        phase1_end = max(1, int(cfg.episodes * phase1_ratio))
        phase2_end = max(phase1_end + 1, int(cfg.episodes * phase2_ratio))

        if ep <= phase1_end:
            t = ep / phase1_end
            return start + t * (mid - start)

        t = min(1.0, (ep - phase1_end) / (phase2_end - phase1_end))
        return mid + t * (end - mid)

    raise ValueError(f"unknown epsilon type: {typ}")


def select_action(
    model: CNN_DQN,
    state: np.ndarray,
    epsilon: float,
    device: torch.device,
) -> int:
    if random.random() < epsilon:
        return random.randrange(4)

    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = model(s)
        return int(q_values.argmax(dim=1).item())


def q_values_for_state(
    model: CNN_DQN,
    state: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q = model(s)[0].detach().cpu().numpy()
    return q


def optimize_model(
    policy_net: CNN_DQN,
    target_net: CNN_DQN,
    replay: StratifiedReplayBuffer,
    optimizer: optim.Optimizer,
    cfg: TrainConfig,
    device: torch.device,
) -> Optional[float]:
    if len(replay) < max(cfg.batch_size, cfg.warmup_steps):
        return None

    transitions = replay.sample(cfg.batch_size, cfg.replay_type)
    if len(transitions) < cfg.batch_size:
        return None

    batch = Transition(*zip(*transitions))

    states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=device)
    actions = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=device)
    dones = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)

    q_values = policy_net(states).gather(1, actions)

    # Keep the original Double-DQN style already present in the user's code:
    # policy_net selects next action, target_net evaluates it.
    with torch.no_grad():
        next_actions = policy_net(next_states).argmax(dim=1, keepdim=True)
        next_q_values = target_net(next_states).gather(1, next_actions)
        target_q_values = rewards + cfg.gamma * next_q_values * (1.0 - dones)

    loss = nn.functional.smooth_l1_loss(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
    optimizer.step()

    return float(loss.item())


# ============================================================
# 8. Training
# ============================================================

def make_transition(
    state: np.ndarray,
    action: int,
    reward: float,
    next_state: np.ndarray,
    done: bool,
    maze_type: str,
    difficulty: str,
    hit_wall: bool,
    success_episode: bool,
    progress: bool,
    step_index: int,
    episode_id: int,
) -> Transition:
    return Transition(
        state,
        action,
        reward,
        next_state,
        done,
        maze_type,
        difficulty,
        hit_wall,
        success_episode,
        progress,
        step_index,
        episode_id,
    )


def relabel_episode_success(
    episode_transitions: List[Transition],
    success: bool,
) -> List[Transition]:
    relabeled = []
    for tr in episode_transitions:
        relabeled.append(
            Transition(
                tr.state,
                tr.action,
                tr.reward,
                tr.next_state,
                tr.done,
                tr.maze_type,
                tr.difficulty,
                tr.hit_wall,
                success,
                tr.progress,
                tr.step_index,
                tr.episode_id,
            )
        )
    return relabeled


def train_one_experiment(
    cfg: TrainConfig,
    device: torch.device,
    save_path: Optional[str] = None,
):
    apply_reward_preset(cfg)

    policy_net = CNN_DQN().to(device)
    target_net = CNN_DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=cfg.lr)
    replay = StratifiedReplayBuffer(cfg.replay_size)

    episode_rewards = []
    losses = []
    wall_hit_rates = []
    success_flags = []
    invalid_action_rates = []
    repeat_counts = []
    progress_rates = []
    path_ratios = []

    global_step = 0
    global_opt_step = 0

    print("\n=== Train Experiment ===")
    print("No action mask. No visited. No BFS reward shaping.")
    print(f"reward_preset={cfg.reward_preset}: goal={cfg.goal_reward}, wall={cfg.wall_penalty}, step={cfg.step_penalty}")
    print(f"epsilon_preset={cfg.epsilon_preset}")
    print(f"maze_schedule={cfg.maze_schedule}")
    print(f"replay_type={cfg.replay_type}")
    print(f"episodes={cfg.episodes}, max_steps={cfg.max_steps}, lr={cfg.lr}, gamma={cfg.gamma}")

    for ep in range(1, cfg.episodes + 1):
        grid, start, goal, maze_type, difficulty = sample_train_maze(cfg, ep)
        env = MazeEnv(grid, start, goal, cfg, maze_type=maze_type, difficulty=difficulty)

        state = env.reset()
        epsilon = epsilon_by_episode(ep, cfg)
        total_reward = 0.0
        final_info = None
        episode_transitions: List[Transition] = []

        for step_index in range(cfg.max_steps):
            action = select_action(policy_net, state, epsilon, device)
            next_state, reward, done, info = env.step(action)

            tr = make_transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                maze_type=maze_type,
                difficulty=difficulty,
                hit_wall=info["hit_wall"],
                success_episode=False,  # relabeled after episode ends
                progress=info["progress"],
                step_index=step_index,
                episode_id=ep,
            )
            episode_transitions.append(tr)

            state = next_state
            total_reward += reward
            global_step += 1
            final_info = info

            if done:
                break

        success = bool(final_info and final_info["success"])

        # Relabel transitions with episode-level success and then insert into replay.
        relabeled = relabel_episode_success(episode_transitions, success)
        replay.push_episode(relabeled)

        # Preserve the original spirit of training roughly once per env step,
        # but do it after the episode so success_episode metadata is available.
        for _ in range(len(relabeled)):
            if global_opt_step % cfg.train_every == 0:
                loss = optimize_model(
                    policy_net,
                    target_net,
                    replay,
                    optimizer,
                    cfg,
                    device,
                )
                if loss is not None:
                    losses.append(loss)

            global_opt_step += 1

            if global_opt_step % cfg.target_update_interval == 0:
                target_net.load_state_dict(policy_net.state_dict())

        episode_rewards.append(total_reward)
        success_flags.append(1 if success else 0)

        wall_rate = env.wall_hits / max(1, env.steps)
        invalid_rate = env.invalid_moves / max(1, env.steps)
        repeat_count = len(env.path) - len(set(env.path))
        progress_rate = env.bfs_progress_count / max(1, env.steps)

        if success:
            path_ratio = env.steps / max(1, env.shortest_len)
        else:
            path_ratio = np.nan

        wall_hit_rates.append(wall_rate)
        invalid_action_rates.append(invalid_rate)
        repeat_counts.append(repeat_count)
        progress_rates.append(progress_rate)
        path_ratios.append(path_ratio)

        if ep % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_success = np.mean(success_flags[-100:]) * 100.0
            avg_wall = np.mean(wall_hit_rates[-100:]) * 100.0
            avg_invalid = np.mean(invalid_action_rates[-100:]) * 100.0
            avg_repeat = np.mean(repeat_counts[-100:])
            avg_progress = np.mean(progress_rates[-100:]) * 100.0
            recent_path = np.array(path_ratios[-100:], dtype=np.float32)
            avg_path_ratio = np.nanmean(recent_path) if np.any(~np.isnan(recent_path)) else float("nan")
            avg_loss = np.mean(losses[-200:]) if losses else float("nan")
            rstats = replay.stats()

            print(
                f"ep={ep:4d}/{cfg.episodes} "
                f"eps={epsilon:.3f} "
                f"reward={avg_reward:8.2f} "
                f"success={avg_success:6.2f}% "
                f"wall={avg_wall:6.2f}% "
                f"invalid={avg_invalid:6.2f}% "
                f"progress={avg_progress:6.2f}% "
                f"repeat={avg_repeat:6.2f} "
                f"pathR={avg_path_ratio:5.2f} "
                f"loss={avg_loss:.4f} "
                f"buf={rstats['buffer_size']} "
                f"succBuf={rstats['success_ratio']:.2f} "
                f"wallBuf={rstats['wall_ratio']:.2f} "
                f"progBuf={rstats['progress_ratio']:.2f}"
            )

    if save_path:
        torch.save(policy_net.state_dict(), save_path)
        print(f"[Save] {save_path}")

    history = {
        "rewards": episode_rewards,
        "losses": losses,
        "wall_hit_rates": wall_hit_rates,
        "success_flags": success_flags,
        "invalid_action_rates": invalid_action_rates,
        "repeat_counts": repeat_counts,
        "progress_rates": progress_rates,
        "path_ratios": path_ratios,
    }

    return policy_net, history


# ============================================================
# 9. Evaluation and diagnostics
# ============================================================

def run_policy_once(
    model: CNN_DQN,
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    cfg: TrainConfig,
    device: torch.device,
    maze_type: str = "eval",
    difficulty: str = "eval",
    greedy: bool = True,
    debug_stuck: bool = False,
):
    env = MazeEnv(grid, start, goal, cfg, maze_type=maze_type, difficulty=difficulty)
    state = env.reset()

    total_reward = 0.0
    actions = []
    q_trace = []
    invalid_chosen = 0
    bfs_agree_count = 0
    bfs_agree_total = 0
    final_info = None

    while True:
        epsilon = 0.0 if greedy else 0.05

        q = q_values_for_state(model, state, device)
        action = select_action(model, state, epsilon, device)

        old_pos = env.pos
        vacts = valid_actions(grid, old_pos)
        optimal_acts = bfs_optimal_actions(grid, old_pos, goal)

        if action not in vacts:
            invalid_chosen += 1

        if optimal_acts:
            bfs_agree_total += 1
            if action in optimal_acts:
                bfs_agree_count += 1

        next_state, reward, done, info = env.step(action)

        q_trace.append((old_pos, q.copy(), action, vacts, optimal_acts, info["hit_wall"], info["progress"]))
        actions.append(action)
        total_reward += reward
        state = next_state
        final_info = info

        if done:
            break

    path = env.path
    unique_cells = len(set(path))
    repeat_count = len(path) - unique_cells

    success = final_info["success"]
    shortest_len = env.shortest_len
    steps = final_info["steps"]

    if success:
        overrun = steps - shortest_len
        ratio = steps / max(1, shortest_len)
    else:
        overrun = None
        ratio = None

    result = {
        "success": success,
        "reward": total_reward,
        "steps": steps,
        "shortest_len": shortest_len,
        "overrun": overrun,
        "ratio": ratio,
        "repeat_count": repeat_count,
        "wall_hits": env.wall_hits,
        "wall_hit_rate": env.wall_hits / max(1, steps),
        "invalid_chosen": invalid_chosen,
        "invalid_action_rate": invalid_chosen / max(1, steps),
        "progress_rate": env.bfs_progress_count / max(1, steps),
        "bfs_action_agreement": bfs_agree_count / max(1, bfs_agree_total),
        "path": path,
        "actions": actions,
        "q_trace": q_trace,
        "action_hist": dict(env.action_hist),
        "maze_type": maze_type,
        "difficulty": difficulty,
    }

    if debug_stuck:
        print_debug_trace(result, grid, start, goal)

    return result


def print_debug_trace(result, grid, start, goal, last_n=12):
    print("\n--- Debug Trace ---")
    print(f"maze_type={result['maze_type']} difficulty={result['difficulty']}")
    print(f"success={result['success']}")
    print(f"reward={result['reward']:.2f}")
    print(f"steps={result['steps']}")
    print(f"shortest_len={result['shortest_len']}")
    print(f"wall_hits={result['wall_hits']}")
    print(f"wall_hit_rate={result['wall_hit_rate']:.2%}")
    print(f"invalid_action_rate={result['invalid_action_rate']:.2%}")
    print(f"progress_rate={result['progress_rate']:.2%}")
    print(f"bfs_action_agreement={result['bfs_action_agreement']:.2%}")
    print(f"action_hist={ {ACTION_NAMES[k]: v for k, v in result['action_hist'].items()} }")
    print(f"path_tail={result['path'][-last_n:]}")
    print("q_trace_tail:")

    for pos, q, action, vacts, optimal_acts, hit_wall, progress in result["q_trace"][-last_n:]:
        valid_names = [ACTION_NAMES[a] for a in vacts]
        opt_names = [ACTION_NAMES[a] for a in optimal_acts]
        print(
            f"  pos={pos} "
            f"Q={np.round(q, 2)} "
            f"chosen={ACTION_NAMES[action]} "
            f"valid={valid_names} "
            f"bfs_best={opt_names} "
            f"hit_wall={hit_wall} "
            f"progress={progress}"
        )


def eval_model(
    model: CNN_DQN,
    cfg: TrainConfig,
    device: torch.device,
    eval_type: str,
    n: int = 50,
    debug_first_failure: bool = False,
):
    results = []

    for i in range(n):
        grid, start, goal, maze_type, difficulty = sample_eval_maze(eval_type)

        res = run_policy_once(
            model,
            grid,
            start,
            goal,
            cfg,
            device,
            maze_type=maze_type,
            difficulty=difficulty,
            greedy=True,
            debug_stuck=False,
        )

        results.append(res)

        if debug_first_failure and not res["success"]:
            print(f"\nFirst failure in eval_type={eval_type}, i={i}")
            run_policy_once(
                model,
                grid,
                start,
                goal,
                cfg,
                device,
                maze_type=maze_type,
                difficulty=difficulty,
                greedy=True,
                debug_stuck=True,
            )
            break

    success = np.mean([r["success"] for r in results]) * 100.0
    reward = np.mean([r["reward"] for r in results])
    steps = np.mean([r["steps"] for r in results])
    wall_rate = np.mean([r["wall_hit_rate"] for r in results]) * 100.0
    invalid_rate = np.mean([r["invalid_action_rate"] for r in results]) * 100.0
    progress_rate = np.mean([r["progress_rate"] for r in results]) * 100.0
    bfs_agreement = np.mean([r["bfs_action_agreement"] for r in results]) * 100.0
    repeat = np.mean([r["repeat_count"] for r in results])

    ratios = [r["ratio"] for r in results if r["ratio"] is not None]
    path_ratio = np.mean(ratios) if ratios else float("nan")

    print(
        f"{eval_type:10s} "
        f"success={success:6.2f}% "
        f"reward={reward:8.2f} "
        f"steps={steps:7.2f} "
        f"wall={wall_rate:6.2f}% "
        f"invalid={invalid_rate:6.2f}% "
        f"progress={progress_rate:6.2f}% "
        f"bfsAgree={bfs_agreement:6.2f}% "
        f"repeat={repeat:7.2f} "
        f"pathR={path_ratio:5.2f}"
    )

    return results


def run_evaluation_suite(model, cfg, device, debug=False, eval_n=50):
    print("\n=== Evaluation Suite ===")
    for typ in ["easy", "medium", "hard", "manual", "heldout"]:
        eval_model(
            model=model,
            cfg=cfg,
            device=device,
            eval_type=typ,
            n=eval_n if typ != "heldout" else 1,
            debug_first_failure=debug,
        )


# ============================================================
# 10. Visualization
# ============================================================

def draw_path(grid, start, goal, path, title):
    img = np.ones((SIZE, SIZE, 3), dtype=np.float32)
    img[grid == 1] = np.array([0.05, 0.05, 0.05])

    for p in path:
        r, c = p
        if p not in [start, goal]:
            img[r, c] = np.array([0.4, 0.7, 1.0])

    sr, sc = start
    gr, gc = goal
    img[sr, sc] = np.array([0.0, 0.8, 0.0])
    img[gr, gc] = np.array([1.0, 0.2, 0.2])

    if path:
        ar, ac = path[-1]
        img[ar, ac] = np.array([1.0, 0.85, 0.0])

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.xticks(np.arange(-0.5, SIZE, 1), minor=True)
    plt.yticks(np.arange(-0.5, SIZE, 1), minor=True)
    plt.grid(which="minor", color="black", linestyle="-", linewidth=1)
    plt.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.title(title)
    plt.show()


def moving_average(x: List[float], window: int) -> Optional[np.ndarray]:
    arr = np.array(x, dtype=np.float32)
    if len(arr) < window:
        return None
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def plot_curves(histories: Dict[str, dict]):
    plt.figure(figsize=(12, 10))

    plt.subplot(3, 1, 1)
    for name, hist in histories.items():
        ma = moving_average(hist["rewards"], 50)
        if ma is not None:
            plt.plot(ma, label=name)
    plt.title("Reward moving average")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    for name, hist in histories.items():
        ma = moving_average(hist["wall_hit_rates"], 50)
        if ma is not None:
            plt.plot(ma, label=name)
    plt.title("Wall-hit-rate moving average")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    for name, hist in histories.items():
        ma = moving_average(hist["progress_rates"], 50)
        if ma is not None:
            plt.plot(ma, label=name)
    plt.title("BFS progress-rate moving average")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ============================================================
# 11. Main
# ============================================================

def make_save_name(args) -> str:
    return (
        f"dqn_{args.reward_preset}"
        f"_eps_{args.epsilon_preset}"
        f"_maze_{args.maze_schedule}"
        f"_replay_{args.replay_type}"
        f"_seed{args.seed}.pt"
    )


def build_cfg_from_args(args) -> TrainConfig:
    cfg = TrainConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        reward_preset=args.reward_preset,
        epsilon_preset=args.epsilon_preset,
        maze_schedule=args.maze_schedule,
        replay_type=args.replay_type,
        save_model=args.model,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        warmup_steps=args.warmup_steps,
        target_update_interval=args.target_update_interval,
    )
    apply_reward_preset(cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["single", "reward_ablate", "eval", "test"],
        default="single",
    )

    parser.add_argument(
        "--reward-preset",
        choices=list(REWARD_PRESETS.keys()),
        default="g10_w5_s001",
    )

    parser.add_argument(
        "--epsilon-preset",
        choices=list(EPSILON_PRESETS.keys()),
        default="two_phase",
    )

    parser.add_argument(
        "--maze-schedule",
        choices=["curriculum", "easy", "medium", "hard", "manual"],
        default="curriculum",
    )

    parser.add_argument(
        "--replay-type",
        choices=["uniform", "stratified"],
        default="stratified",
    )

    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--eval-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=100_000)
    parser.add_argument("--warmup-steps", type=int, default=3000)
    parser.add_argument("--target-update-interval", type=int, default=500)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = build_cfg_from_args(args)

    if args.model:
        model_path = args.model
    else:
        model_path = make_save_name(args)

    if args.mode == "single":
        model, hist = train_one_experiment(
            cfg=cfg,
            device=device,
            save_path=model_path,
        )
        run_evaluation_suite(model, cfg, device, debug=args.debug, eval_n=args.eval_n)
        plot_curves({cfg.reward_preset: hist})

    elif args.mode == "reward_ablate":
        histories = {}
        models = {}

        for reward_preset in REWARD_PRESETS.keys():
            cfg.reward_preset = reward_preset
            apply_reward_preset(cfg)

            save_path = (
                f"dqn_{reward_preset}"
                f"_eps_{cfg.epsilon_preset}"
                f"_maze_{cfg.maze_schedule}"
                f"_replay_{cfg.replay_type}"
                f"_seed{args.seed}.pt"
            )

            model, hist = train_one_experiment(
                cfg=cfg,
                device=device,
                save_path=save_path,
            )
            histories[reward_preset] = hist
            models[reward_preset] = model

            print(f"\n>>> Eval model trained with reward_preset={reward_preset}")
            run_evaluation_suite(model, cfg, device, debug=False, eval_n=args.eval_n)

        plot_curves(histories)

    elif args.mode == "eval":
        if not model_path:
            raise ValueError("please provide --model for eval mode")

        model = CNN_DQN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        run_evaluation_suite(model, cfg, device, debug=args.debug, eval_n=args.eval_n)

    elif args.mode == "test":
        if not model_path:
            raise ValueError("please provide --model for test mode")

        model = CNN_DQN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        grid, start, goal = parse_maze_text(TEST_MAZE_TEXT)
        res = run_policy_once(
            model,
            grid,
            start,
            goal,
            cfg,
            device,
            maze_type="heldout",
            difficulty="heldout",
            greedy=True,
            debug_stuck=True,
        )
        draw_path(
            grid,
            start,
            goal,
            res["path"],
            title=(
                f"heldout success={res['success']} "
                f"wall={res['wall_hit_rate']:.2%} "
                f"progress={res['progress_rate']:.2%}"
            ),
        )


if __name__ == "__main__":
    main()
