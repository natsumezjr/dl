import argparse
import random
import math
from collections import deque, namedtuple, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ============================================================
# 0. 单独测试用手绘接口
# ============================================================
# 规则：
#   S = start
#   G = goal
#   # = wall
#   . = road
#
# 必须是 8 行，每行 8 个字符。
# 这个图不会参与训练，只用于 --mode test。
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


# ============================================================
# 1. 基础配置
# ============================================================

SIZE = 8

ACTIONS = [
    (-1, 0),  # up
    (1, 0),   # down
    (0, -1),  # left
    (0, 1),   # right
]

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]

Transition = namedtuple(
    "Transition",
    ["state", "action", "reward", "next_state", "done"]
)


@dataclass
class TrainConfig:
    size: int = 8

    episodes: int = 3000
    max_steps: int = 160

    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 128
    replay_size: int = 100_000
    warmup_steps: int = 3000

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 2200

    target_update_interval: int = 500
    train_every: int = 1

    random_ratio: float = 0.5
    manual_ratio: float = 0.5

    # 基础 reward，不含 BFS shaping，不含 visited/repeat penalty
    step_penalty: float = -0.05
    wall_penalty: float = -1.0
    goal_reward: float = 100.0

    save_model: str = "cnn_dqn_maze_8x8_basic.pt"


# ============================================================
# 2. Maze 工具函数
# ============================================================

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


# ============================================================
# 3. 手绘训练集
# ============================================================
episode_action_counts = np.zeros(4, dtype=np.int64)
episode_wall_hits = 0

MANUAL_MAZES: Dict[str, List[str]] = {
    "open_shortest": [
        """
S.......
........
........
........
........
........
........
.......G
""",
        """
S.......
..##....
..##....
........
....##..
....##..
........
.......G
""",
    ],

    "multi_open_paths": [
        """
S.......
.##..##.
....#...
##..#..#
...##...
.#......
.##..##.
.......G
""",
        """
S.......
...##...
.#....#.
.#.##.#.
.#....#.
...##...
.#......
.......G
""",
    ],

    "long_detour": [
        """
S......#
######.#
#......#
#.######
#......#
######.#
#......#
######.G
""",
        """
S.......
######.#
.......#
#.######
#......#
#.####.#
#....#.#
####.#.G
""",
    ],

    "bottleneck": [
        """
S...#...
....#...
....#...
....#...
........
....#...
....#...
....#..G
""",
        """
S.......
........
###.####
........
....####
....#...
....#...
.......G
""",
    ],

    "deadends": [
        """
S.......
.#####..
.#......
.#.#####
.#.....#
.#####.#
.......#
######.G
""",
        """
S.......
.###.###
...#...#
##.#.#.#
...#.#..
.###.##.
.#......
.#.####G
""",
    ],

    "snake": [
        """
S......#
######.#
#......#
#.######
#......#
######.#
#......#
#G######
""",
        """
S......#
######.#
#......#
#.######
#......#
######.#
#......#
######.G
""",
    ],
}


def print_manual_maze_stats():
    print("\n=== Manual Maze Stats ===")
    for category, mazes in MANUAL_MAZES.items():
        for i, text in enumerate(mazes):
            grid, start, goal = parse_maze_text(text)
            path = bfs_shortest_path(grid, start, goal)

            if path is None:
                print(f"{category}[{i}] unsolvable")
            else:
                print(
                    f"{category}[{i}] "
                    f"start={start} goal={goal} "
                    f"shortest_len={len(path) - 1}"
                )


def transform_grid_randomly(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    g = grid.copy()
    s = start
    t = goal

    if random.random() < 0.5:
        g = np.fliplr(g)
        s = (s[0], SIZE - 1 - s[1])
        t = (t[0], SIZE - 1 - t[1])

    if random.random() < 0.5:
        g = np.flipud(g)
        s = (SIZE - 1 - s[0], s[1])
        t = (SIZE - 1 - t[0], t[1])

    return g.copy(), s, t


def sample_manual_maze() -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], str]:
    category = random.choice(list(MANUAL_MAZES.keys()))
    text = random.choice(MANUAL_MAZES[category])

    grid, start, goal = parse_maze_text(text)
    grid, start, goal = transform_grid_randomly(grid, start, goal)

    path = bfs_shortest_path(grid, start, goal)
    if path is None:
        raise RuntimeError(f"manual maze became unsolvable: {category}")

    return grid, start, goal, category


# ============================================================
# 4. 随机迷宫生成
# ============================================================

def generate_random_maze(
    obstacle_prob_min: float = 0.15,
    obstacle_prob_max: float = 0.32,
    min_shortest_len: int = 8,
    max_tries: int = 3000,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], str]:
    for _ in range(max_tries):
        p = random.uniform(obstacle_prob_min, obstacle_prob_max)
        grid = (np.random.rand(SIZE, SIZE) < p).astype(np.int64)

        start = (random.randint(0, SIZE - 1), random.randint(0, SIZE - 1))
        goal = (random.randint(0, SIZE - 1), random.randint(0, SIZE - 1))

        if start == goal:
            continue

        manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        if manhattan < 8:
            continue

        grid[start] = 0
        grid[goal] = 0

        path = bfs_shortest_path(grid, start, goal)
        if path is None:
            continue

        if len(path) - 1 < min_shortest_len:
            continue

        return grid, start, goal, "random"

    raise RuntimeError("failed to generate random maze")


def sample_train_maze(cfg: TrainConfig):
    if random.random() < cfg.random_ratio:
        return generate_random_maze()
    return sample_manual_maze()


# ============================================================
# 5. MazeEnv
# ============================================================

class MazeEnv:
    def __init__(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        cfg: TrainConfig,
        maze_type: str = "unknown",
    ):
        self.grid = grid.astype(np.int64)
        self.start = start
        self.goal = goal
        self.cfg = cfg
        self.maze_type = maze_type

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
        return self._state()

    def _state(self) -> np.ndarray:
        """
        3 通道：
          0: wall
          1: agent
          2: goal
        """
        walls = self.grid.astype(np.float32)

        agent = np.zeros((SIZE, SIZE), dtype=np.float32)
        agent[self.pos] = 1.0

        goal = np.zeros((SIZE, SIZE), dtype=np.float32)
        goal[self.goal] = 1.0

        return np.stack([walls, agent, goal], axis=0)

    def step(self, action: int):
        episode_action_counts[action] += 1

        if self.done:
            raise RuntimeError("step after done")

        self.steps += 1

        dr, dc = ACTIONS[action]
        r, c = self.pos
        nxt = (r + dr, c + dc)

        reward = self.cfg.step_penalty

        if not is_free(self.grid, nxt):
            reward += self.cfg.wall_penalty
            nxt = self.pos
        else:
            self.pos = nxt
            self.path.append(self.pos)

        if self.pos == self.goal:
            reward += self.cfg.goal_reward
            self.done = True

            return self._state(), reward, True, {
                "success": True,
                "steps": self.steps,
                "shortest_len": self.shortest_len,
                "maze_type": self.maze_type,
            }

        if self.steps >= self.cfg.max_steps:
            self.done = True

            return self._state(), reward, True, {
                "success": False,
                "steps": self.steps,
                "shortest_len": self.shortest_len,
                "maze_type": self.maze_type,
            }

        return self._state(), reward, False, {
            "success": False,
            "steps": self.steps,
            "shortest_len": self.shortest_len,
            "maze_type": self.maze_type,
        }


# ============================================================
# 6. CNN DQN
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
# 7. Replay Buffer
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ============================================================
# 8. DQN 训练工具
# ============================================================

def epsilon_by_episode(ep: int, cfg: TrainConfig) -> float:
    t = min(1.0, ep / cfg.epsilon_decay_episodes)
    return cfg.epsilon_start + t * (cfg.epsilon_end - cfg.epsilon_start)


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


def optimize_model(
    policy_net: CNN_DQN,
    target_net: CNN_DQN,
    replay: ReplayBuffer,
    optimizer: optim.Optimizer,
    cfg: TrainConfig,
    device: torch.device,
) -> Optional[float]:
    if len(replay) < max(cfg.batch_size, cfg.warmup_steps):
        return None

    transitions = replay.sample(cfg.batch_size)
    batch = Transition(*zip(*transitions))

    states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=device)
    actions = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=device)
    dones = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)

    q_values = policy_net(states).gather(1, actions)

    # Double DQN:
    #   policy_net 选择 next action
    #   target_net 评估该 action 的 Q 值
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
# 9. 训练
# ============================================================

def train(
    cfg: TrainConfig,
    device: torch.device,
    load_model: Optional[str] = None,
):
    policy_net = CNN_DQN().to(device)
    target_net = CNN_DQN().to(device)

    if load_model:
        print(f"[Load] continue training from: {load_model}")
        policy_net.load_state_dict(torch.load(load_model, map_location=device))

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=cfg.lr)
    replay = ReplayBuffer(cfg.replay_size)

    episode_rewards = []
    losses = []
    type_counter = defaultdict(int)
    global_step = 0

    print("\n=== Training CNN Double-DQN ===")
    print("Distribution: 50% random + 50% manual")
    print("State channels: wall / agent / goal")
    print("Reward: basic only, no BFS shaping, no visited penalty")
    print(f"Device: {device}")

    for ep in range(1, cfg.episodes + 1):
        grid, start, goal, maze_type = sample_train_maze(cfg)
        env = MazeEnv(grid, start, goal, cfg, maze_type=maze_type)

        state = env.reset()
        epsilon = epsilon_by_episode(ep, cfg)

        total_reward = 0.0
        type_counter[maze_type] += 1

        for _ in range(cfg.max_steps):
            action = select_action(policy_net, state, epsilon, device)
            next_state, reward, done, info = env.step(action)

            replay.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            global_step += 1

            if global_step % cfg.train_every == 0:
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

            if global_step % cfg.target_update_interval == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        episode_rewards.append(total_reward)

        if ep % 100 == 0:
            recent_rewards = episode_rewards[-100:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            avg_loss = sum(losses[-200:]) / max(1, len(losses[-200:]))

            print(
                f"Episode {ep}/{cfg.episodes} "
                f"epsilon={epsilon:.3f} "
                f"avg_reward={avg_reward:.2f} "
                f"loss={avg_loss:.4f} "
                f"buffer={len(replay)} "
                f"type_count={dict(type_counter)}"
            )
            type_counter.clear()

    torch.save(policy_net.state_dict(), cfg.save_model)
    print(f"\n[Save] Model saved to: {cfg.save_model}")

    return policy_net


# ============================================================
# 10. 评估与测试
# ============================================================

def valid_action_flags(grid, pos):
    flags = []
    r, c = pos
    for dr, dc in ACTIONS:
        nxt = (r + dr, c + dc)
        flags.append(is_free(grid, nxt))
    return flags


def debug_state_q(model, grid, pos, goal, device, title="debug"):
    # 构造一个临时 env-like state，不走环境
    walls = grid.astype(np.float32)

    agent = np.zeros((SIZE, SIZE), dtype=np.float32)
    agent[pos] = 1.0

    goal_ch = np.zeros((SIZE, SIZE), dtype=np.float32)
    goal_ch[goal] = 1.0

    state = np.stack([walls, agent, goal_ch], axis=0)

    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q = model(s)[0].detach().cpu().numpy()

    valid = valid_action_flags(grid, pos)

    print(f"\n=== Q Debug: {title} pos={pos}, goal={goal} ===")
    for i, name in enumerate(ACTION_NAMES):
        print(f"{name:5s} Q={q[i]:8.3f} valid={valid[i]}")

def run_policy_once(
    model: CNN_DQN,
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    cfg: TrainConfig,
    device: torch.device,
    maze_type: str = "eval",
    greedy: bool = True,
):
    env = MazeEnv(grid, start, goal, cfg, maze_type=maze_type)
    state = env.reset()

    total_reward = 0.0
    done = False
    info = None
    actions = []

    while not done:
        epsilon = 0.0 if greedy else 0.05
        action = select_action(model, state, epsilon, device)

        next_state, reward, done, info = env.step(action)

        actions.append(action)
        total_reward += reward
        state = next_state

    path = env.path
    unique_cells = len(set(path))
    repeat_count = len(path) - unique_cells

    shortest_len = env.shortest_len
    steps = info["steps"]
    success = info["success"]

    overrun = steps - shortest_len if success else None
    ratio = steps / max(1, shortest_len) if success else None

    return {
        "success": success,
        "reward": total_reward,
        "steps": steps,
        "shortest_len": shortest_len,
        "overrun": overrun,
        "ratio": ratio,
        "repeat_count": repeat_count,
        "path": path,
        "actions": actions,
    }


def eval_distribution(
    model: CNN_DQN,
    cfg: TrainConfig,
    device: torch.device,
    n_each: int = 100,
):
    model.eval()

    eval_types = ["random"] + list(MANUAL_MAZES.keys())

    print("\n=== Evaluation by maze type ===")

    for typ in eval_types:
        results = []

        for _ in range(n_each):
            if typ == "random":
                grid, start, goal, maze_type = generate_random_maze()
            else:
                text = random.choice(MANUAL_MAZES[typ])
                grid, start, goal = parse_maze_text(text)
                grid, start, goal = transform_grid_randomly(grid, start, goal)
                maze_type = typ

            res = run_policy_once(
                model,
                grid,
                start,
                goal,
                cfg,
                device,
                maze_type=maze_type,
                greedy=True,
            )
            results.append(res)

        success_rate = np.mean([r["success"] for r in results]) * 100
        avg_reward = np.mean([r["reward"] for r in results])
        avg_steps = np.mean([r["steps"] for r in results])
        avg_shortest = np.mean([r["shortest_len"] for r in results])
        avg_repeat = np.mean([r["repeat_count"] for r in results])

        success_results = [r for r in results if r["success"]]
        if success_results:
            avg_overrun = np.mean([r["overrun"] for r in success_results])
            avg_ratio = np.mean([r["ratio"] for r in success_results])
        else:
            avg_overrun = math.inf
            avg_ratio = math.inf

        print(
            f"{typ:16s} "
            f"success={success_rate:6.2f}% "
            f"avg_reward={avg_reward:8.2f} "
            f"avg_steps={avg_steps:6.2f} "
            f"shortest={avg_shortest:5.2f} "
            f"overrun={avg_overrun:6.2f} "
            f"ratio={avg_ratio:5.2f} "
            f"repeat={avg_repeat:5.2f}"
        )


# ============================================================
# 11. 动态路径显示
# ============================================================

def build_display_grid(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    current: Optional[Tuple[int, int]] = None,
    path_so_far: Optional[List[Tuple[int, int]]] = None,
    bfs_path: Optional[List[Tuple[int, int]]] = None,
) -> np.ndarray:
    img = np.ones((SIZE, SIZE, 3), dtype=np.float32)

    # wall
    img[grid == 1] = np.array([0.05, 0.05, 0.05])

    # BFS shortest path: light green
    if bfs_path is not None:
        for r, c in bfs_path:
            if (r, c) not in [start, goal]:
                img[r, c] = np.array([0.72, 1.00, 0.72])

    # model path: light blue
    if path_so_far is not None:
        counts = defaultdict(int)
        for pos in path_so_far:
            counts[pos] += 1

        for r, c in path_so_far:
            if (r, c) not in [start, goal]:
                k = min(counts[(r, c)], 5)
                img[r, c] = np.array([0.45, 0.70, max(0.25, 1.0 - 0.10 * k)])

    # start
    sr, sc = start
    img[sr, sc] = np.array([0.0, 0.8, 0.0])

    # goal
    gr, gc = goal
    img[gr, gc] = np.array([1.0, 0.2, 0.2])

    # current agent
    if current is not None:
        ar, ac = current
        img[ar, ac] = np.array([1.0, 0.85, 0.0])

    return img


def animate_test_path(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    model_path: List[Tuple[int, int]],
    bfs_path: Optional[List[Tuple[int, int]]],
    title: str,
    delay: float = 0.25,
):
    plt.ion()

    fig, ax = plt.subplots(figsize=(7, 7))

    for i in range(len(model_path)):
        ax.clear()

        current = model_path[i]
        path_so_far = model_path[: i + 1]

        img = build_display_grid(
            grid=grid,
            start=start,
            goal=goal,
            current=current,
            path_so_far=path_so_far,
            bfs_path=bfs_path,
        )

        ax.imshow(img)

        ax.set_xticks(np.arange(-0.5, SIZE, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, SIZE, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=1)

        ax.tick_params(
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )

        ax.set_title(f"{title}\nstep {i}/{len(model_path) - 1}")

        plt.pause(delay)

    plt.ioff()
    plt.show()


def test_drawn_maze(
    model: CNN_DQN,
    cfg: TrainConfig,
    device: torch.device,
    maze_text: str = TEST_MAZE_TEXT,
    delay: float = 0.25,
):
    grid, start, goal = parse_maze_text(maze_text)
    bfs_path = bfs_shortest_path(grid, start, goal)

    if bfs_path is None:
        print("[Test] This hand-drawn maze is unsolvable.")
        return

    res = run_policy_once(
        model,
        grid,
        start,
        goal,
        cfg,
        device,
        maze_type="hand_drawn_test",
        greedy=True,
    )

    print("\n=== Hand-drawn Test Maze ===")
    print(f"success      = {res['success']}")
    print(f"reward       = {res['reward']:.2f}")
    print(f"steps        = {res['steps']}")
    print(f"shortest_len = {res['shortest_len']}")
    print(f"overrun      = {res['overrun']}")
    print(f"ratio        = {res['ratio']}")
    print(f"repeat_count = {res['repeat_count']}")
    print(f"actions      = {[ACTION_NAMES[a] for a in res['actions']]}")
    print(f"path         = {res['path']}")

    title = (
        f"Hand-drawn Test | success={res['success']} | "
        f"steps={res['steps']} | shortest={res['shortest_len']}"
    )

    for p in res["path"]:
        debug_state_q(model, grid, p, goal, device, title="hand_drawn_path")

    animate_test_path(
        grid=grid,
        start=start,
        goal=goal,
        model_path=res["path"],
        bfs_path=bfs_path,
        title=title,
        delay=delay,
    )


# ============================================================
# 12. 加载模型
# ============================================================

def load_policy_model(path: str, device: torch.device) -> CNN_DQN:
    model = CNN_DQN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"[Load] Model loaded from: {path}")
    return model


# ============================================================
# 13. main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["train", "eval", "test", "all", "stats"],
        default="all",
        help=(
            "train: train only; "
            "eval: load and evaluate; "
            "test: load and dynamically show TEST_MAZE_TEXT; "
            "all: train + eval + test; "
            "stats: print manual maze shortest lengths"
        ),
    )

    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--save-model", type=str, default="cnn_dqn_maze_8x8_basic.pt")
    parser.add_argument("--load-model", type=str, default=None)

    parser.add_argument("--eval-n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=0.25)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = TrainConfig(
        episodes=args.episodes,
        save_model=args.save_model,
        epsilon_decay_episodes=max(100, int(args.episodes * 0.75)),
    )

    if args.mode == "stats":
        print_manual_maze_stats()
        return

    model = None

    if args.mode in ["train", "all"]:
        model = train(
            cfg=cfg,
            device=device,
            load_model=args.load_model,
        )

    if args.mode == "eval":
        if args.load_model is None:
            raise ValueError("--mode eval requires --load-model")
        model = load_policy_model(args.load_model, device)

    if args.mode == "test":
        if args.load_model is None:
            raise ValueError("--mode test requires --load-model")
        model = load_policy_model(args.load_model, device)

    if args.mode in ["eval", "all"]:
        if model is None:
            raise RuntimeError("model is None before eval")
        eval_distribution(
            model=model,
            cfg=cfg,
            device=device,
            n_each=args.eval_n,
        )

    if args.mode in ["test", "all"]:
        if model is None:
            raise RuntimeError("model is None before test")
        test_drawn_maze(
            model=model,
            cfg=cfg,
            device=device,
            maze_text=TEST_MAZE_TEXT,
            delay=args.delay,
        )


if __name__ == "__main__":
    main()