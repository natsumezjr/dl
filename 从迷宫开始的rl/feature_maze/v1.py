import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ============================================================
# 0. 全局设置
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIONS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}

ACTION_NAMES = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
}


# ============================================================
# 1. 随机迷宫生成
# ============================================================

def bfs_path_exists(grid, start, goal):
    rows, cols = grid.shape
    q = deque([start])
    visited = {start}

    while q:
        x, y = q.popleft()

        if (x, y) == goal:
            return True

        for dx, dy in ACTIONS.values():
            nx, ny = x + dx, y + dy

            if not (0 <= nx < rows and 0 <= ny < cols):
                continue

            if grid[nx, ny] == 1:
                continue

            if (nx, ny) in visited:
                continue

            visited.add((nx, ny))
            q.append((nx, ny))

    return False


def generate_random_maze(size=8, wall_prob=0.22):
    start = (0, 0)
    goal = (size - 1, size - 1)

    while True:
        grid = (np.random.rand(size, size) < wall_prob).astype(np.int32)

        grid[start] = 0
        grid[goal] = 0

        if bfs_path_exists(grid, start, goal):
            return grid


# ============================================================
# 2. 支持人工手画迷宫
# ============================================================

def parse_manual_maze(maze_text):
    """
    支持字符：

    S = 起点
    E = 终点
    # = 墙
    . = 可走
    空格会被忽略

    示例：

    manual_maze = '''
    S . . # .
    # # . # .
    . . . . .
    . # # # .
    . . . . E
    '''
    """

    lines = [
        line.strip()
        for line in maze_text.strip().splitlines()
        if line.strip()
    ]

    rows = []
    start = None
    goal = None

    for i, line in enumerate(lines):
        # 允许用空格分隔，也允许直接写成 S..#.
        if " " in line:
            tokens = line.split()
        else:
            tokens = list(line)

        row = []

        for j, ch in enumerate(tokens):
            if ch == "S":
                start = (i, j)
                row.append(0)
            elif ch == "E":
                goal = (i, j)
                row.append(0)
            elif ch == "#":
                row.append(1)
            elif ch == ".":
                row.append(0)
            else:
                raise ValueError(f"Unsupported maze char: {ch}")

        rows.append(row)

    width = len(rows[0])
    for row in rows:
        if len(row) != width:
            raise ValueError("Manual maze must be rectangular.")

    if start is None:
        raise ValueError("Manual maze must contain S.")

    if goal is None:
        raise ValueError("Manual maze must contain E.")

    grid = np.array(rows, dtype=np.int32)

    if not bfs_path_exists(grid, start, goal):
        print("[Warning] Manual maze has no valid path from S to E.")

    return grid, start, goal


# ============================================================
# 3. 迷宫环境
# ============================================================

class MazeEnv:
    def __init__(self, grid, start=(0, 0), goal=None, max_steps=None):
        self.grid = grid.astype(np.int32)
        self.rows, self.cols = self.grid.shape

        self.start = start
        self.goal = goal if goal is not None else (self.rows - 1, self.cols - 1)
        self.max_steps = max_steps if max_steps is not None else self.rows * self.cols * 3

        self.state = None
        self.steps = 0

    def reset(self):
        self.state = self.start
        self.steps = 0
        return self.get_obs()

    def in_bounds(self, x, y):
        return 0 <= x < self.rows and 0 <= y < self.cols

    def is_wall(self, x, y):
        return self.grid[x, y] == 1

    def get_obs(self):
        """
        CNN 输入：
        obs.shape = (3, H, W)

        channel 0: wall map
        channel 1: agent position
        channel 2: goal position
        """
        obs = np.zeros((3, self.rows, self.cols), dtype=np.float32)

        obs[0] = self.grid

        ax, ay = self.state
        gx, gy = self.goal

        obs[1, ax, ay] = 1.0
        obs[2, gx, gy] = 1.0

        return obs

    def step(self, action):
        self.steps += 1

        x, y = self.state
        dx, dy = ACTIONS[action]

        nx, ny = x + dx, y + dy

        if not self.in_bounds(nx, ny):
            reward = -5
            done = False

        elif self.is_wall(nx, ny):
            reward = -5
            done = False

        else:
            self.state = (nx, ny)

            if self.state == self.goal:
                reward = 100
                done = True
            else:
                reward = -1
                done = False

        if self.steps >= self.max_steps:
            done = True

        return self.get_obs(), reward, done


# ============================================================
# 4. CNN Q 网络
# ============================================================

class CNNQNetwork(nn.Module):
    def __init__(self, maze_size, in_channels=3, num_actions=4):
        super().__init__()

        self.maze_size = maze_size

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(32 * maze_size * maze_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        return self.head(x)


# ============================================================
# 5. DQN Agent
# ============================================================

class DQNAgent:
    def __init__(
        self,
        maze_size,
        replay_capacity=30000,
        batch_size=64,
        gamma=0.9,
        lr=1e-3,
        target_update_steps=500,
    ):
        self.maze_size = maze_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_steps = target_update_steps

        self.online_net = CNNQNetwork(maze_size).to(DEVICE)
        self.target_net = CNNQNetwork(maze_size).to(DEVICE)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = deque(maxlen=replay_capacity)
        self.global_step = 0

    def choose_action(self, obs, epsilon=0.0):
        if random.random() < epsilon:
            return random.choice(list(ACTIONS.keys()))

        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            q_values = self.online_net(obs_tensor)

        return int(torch.argmax(q_values, dim=1).item())

    def q_values(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            q = self.online_net(obs_tensor)[0]

        return q.cpu().numpy()

    def store_transition(self, obs, action, reward, next_obs, done):
        self.replay_buffer.append((obs, action, reward, next_obs, done))

    def sample_batch(self):
        batch = random.sample(self.replay_buffer, self.batch_size)

        obs_batch = np.stack([item[0] for item in batch])
        action_batch = np.array([item[1] for item in batch], dtype=np.int64)
        reward_batch = np.array([item[2] for item in batch], dtype=np.float32)
        next_obs_batch = np.stack([item[3] for item in batch])
        done_batch = np.array([item[4] for item in batch], dtype=np.float32)

        obs_batch = torch.tensor(obs_batch, dtype=torch.float32).to(DEVICE)
        action_batch = torch.tensor(action_batch, dtype=torch.long).to(DEVICE)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).to(DEVICE)
        next_obs_batch = torch.tensor(next_obs_batch, dtype=torch.float32).to(DEVICE)
        done_batch = torch.tensor(done_batch, dtype=torch.float32).to(DEVICE)

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = self.sample_batch()

        q_values = self.online_net(obs_batch)

        q_sa = q_values.gather(
            1,
            action_batch.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_obs_batch)
            max_next_q = torch.max(next_q_values, dim=1).values

            target = reward_batch + self.gamma * max_next_q * (1.0 - done_batch)

        loss = self.loss_fn(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.global_step += 1

        if self.global_step % self.target_update_steps == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def save(self, path):
        checkpoint = {
            "maze_size": self.maze_size,
            "online_net_state_dict": self.online_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "target_update_steps": self.target_update_steps,
        }

        torch.save(checkpoint, path)
        print(f"[Save] Model saved to: {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=DEVICE)

        if checkpoint["maze_size"] != self.maze_size:
            raise ValueError(
                f"Checkpoint maze_size={checkpoint['maze_size']} "
                f"but current maze_size={self.maze_size}."
            )

        self.online_net.load_state_dict(checkpoint["online_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]

        print(f"[Load] Model loaded from: {path}")


# ============================================================
# 6. epsilon 衰减
# ============================================================

def get_epsilon(episode, num_episodes):
    epsilon_start = 1.0
    epsilon_end = 0.05
    decay_ratio = 0.7

    decay_episodes = int(num_episodes * decay_ratio)

    if episode >= decay_episodes:
        return epsilon_end

    progress = episode / decay_episodes
    return epsilon_start + progress * (epsilon_end - epsilon_start)


# ============================================================
# 7. 训练
# ============================================================

def train_agent(
    agent,
    maze_size=8,
    wall_prob=0.22,
    num_episodes=1200,
    max_steps=None,
    live_every=100,
):
    rewards = []
    losses = []
    buffer_sizes = []
    epsilons = []

    if max_steps is None:
        max_steps = maze_size * maze_size * 3

    for episode in range(num_episodes):
        grid = generate_random_maze(size=maze_size, wall_prob=wall_prob)
        env = MazeEnv(grid, max_steps=max_steps)

        obs = env.reset()
        total_reward = 0

        epsilon = get_epsilon(episode, num_episodes)

        for _ in range(max_steps):
            action = agent.choose_action(obs, epsilon=epsilon)
            next_obs, reward, done = env.step(action)

            agent.store_transition(obs, action, reward, next_obs, done)

            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            total_reward += reward
            obs = next_obs

            if done:
                break

        rewards.append(total_reward)
        buffer_sizes.append(len(agent.replay_buffer))
        epsilons.append(epsilon)

        if (episode + 1) % live_every == 0:
            avg_reward = np.mean(rewards[-live_every:])
            print(
                f"Episode {episode + 1}/{num_episodes} "
                f"epsilon={epsilon:.3f} "
                f"avg_reward={avg_reward:.2f} "
                f"buffer={len(agent.replay_buffer)}"
            )

    return rewards, losses, buffer_sizes, epsilons


# ============================================================
# 8. 评估和运行
# ============================================================

def run_policy_once(agent, grid, start=(0, 0), goal=None, max_steps=None, epsilon=0.0):
    maze_size = grid.shape[0]

    if goal is None:
        goal = (maze_size - 1, maze_size - 1)

    if max_steps is None:
        max_steps = maze_size * maze_size * 3

    env = MazeEnv(grid, start=start, goal=goal, max_steps=max_steps)

    obs = env.reset()
    path = [env.state]
    actions_taken = []
    total_reward = 0
    done = False

    for _ in range(max_steps):
        action = agent.choose_action(obs, epsilon=epsilon)
        next_obs, reward, done = env.step(action)

        actions_taken.append(ACTION_NAMES[action])
        total_reward += reward
        obs = next_obs
        path.append(env.state)

        if done:
            break

    success = env.state == env.goal

    return path, actions_taken, total_reward, success


def evaluate(agent, num_mazes=100, maze_size=8, wall_prob=0.22, trained=True):
    successes = 0
    rewards = []

    for _ in range(num_mazes):
        grid = generate_random_maze(size=maze_size, wall_prob=wall_prob)

        epsilon = 0.0 if trained else 1.0

        path, actions_taken, total_reward, success = run_policy_once(
            agent,
            grid,
            max_steps=maze_size * maze_size * 3,
            epsilon=epsilon,
        )

        successes += int(success)
        rewards.append(total_reward)

    return successes / num_mazes, float(np.mean(rewards))


# ============================================================
# 9. 可视化
# ============================================================

def draw_maze(ax, grid, path=None, start=(0, 0), goal=None, title=None):
    ax.clear()

    rows, cols = grid.shape
    if goal is None:
        goal = (rows - 1, cols - 1)

    for x in range(rows):
        for y in range(cols):
            color = "black" if grid[x, y] == 1 else "white"

            rect = plt.Rectangle(
                (y, rows - 1 - x),
                1,
                1,
                facecolor=color,
                edgecolor="gray",
            )
            ax.add_patch(rect)

    sx, sy = start
    gx, gy = goal

    ax.text(
        sy + 0.5,
        rows - 1 - sx + 0.5,
        "S",
        ha="center",
        va="center",
        fontsize=14,
        color="green",
        fontweight="bold",
    )

    ax.text(
        gy + 0.5,
        rows - 1 - gx + 0.5,
        "E",
        ha="center",
        va="center",
        fontsize=14,
        color="red",
        fontweight="bold",
    )

    if path is not None and len(path) > 0:
        xs = [y + 0.5 for x, y in path]
        ys = [rows - 1 - x + 0.5 for x, y in path]

        ax.plot(xs, ys, marker="o", linewidth=2)

        ax.text(
            path[-1][1] + 0.5,
            rows - 1 - path[-1][0] + 0.5,
            "A",
            ha="center",
            va="center",
            fontsize=12,
            color="blue",
            fontweight="bold",
        )

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    if title:
        ax.set_title(title)


def animate_policy(agent, grid, start=(0, 0), goal=None, epsilon=0.0, delay=0.18, title="Policy Run"):
    maze_size = grid.shape[0]
    if goal is None:
        goal = (maze_size - 1, maze_size - 1)

    env = MazeEnv(
        grid,
        start=start,
        goal=goal,
        max_steps=maze_size * maze_size * 3,
    )

    obs = env.reset()
    path = [env.state]
    total_reward = 0
    actions_taken = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))

    done = False
    step_idx = 0

    while not done and step_idx < env.max_steps:
        q_values = agent.q_values(obs)
        best_action = int(np.argmax(q_values))

        draw_maze(
            ax,
            grid,
            path,
            start=start,
            goal=goal,
            title=(
                f"{title}\n"
                f"step={step_idx}, reward={total_reward}\n"
                f"Q={np.round(q_values, 2)}, best={ACTION_NAMES[best_action]}"
            ),
        )

        plt.pause(delay)

        action = agent.choose_action(obs, epsilon=epsilon)
        next_obs, reward, done = env.step(action)

        total_reward += reward
        actions_taken.append(ACTION_NAMES[action])
        obs = next_obs
        path.append(env.state)

        step_idx += 1

    draw_maze(
        ax,
        grid,
        path,
        start=start,
        goal=goal,
        title=f"{title}\nDONE success={env.state == env.goal}, reward={total_reward}",
    )

    plt.pause(delay)
    plt.ioff()
    plt.show()

    success = env.state == env.goal
    return path, actions_taken, total_reward, success


def moving_average(values, window=100):
    if len(values) < window:
        return np.array(values)

    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_training_curves(rewards, losses, buffer_sizes, epsilons):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(rewards, alpha=0.4, label="raw reward")
    if len(rewards) >= 100:
        axes[0, 0].plot(
            range(99, len(rewards)),
            moving_average(rewards, 100),
            label="moving avg 100",
            linewidth=2,
        )
    axes[0, 0].set_title("Episode Reward")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    if losses:
        axes[0, 1].plot(losses, alpha=0.4, label="raw loss")
        if len(losses) >= 200:
            axes[0, 1].plot(
                range(199, len(losses)),
                moving_average(losses, 200),
                label="moving avg 200",
                linewidth=2,
            )
        axes[0, 1].set_title("DQN Loss")
        axes[0, 1].set_xlabel("Train Step")
        axes[0, 1].set_ylabel("MSE Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

    axes[1, 0].plot(buffer_sizes)
    axes[1, 0].set_title("Replay Buffer Size")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Buffer Size")
    axes[1, 0].grid(True)

    axes[1, 1].plot(epsilons)
    axes[1, 1].set_title("Epsilon Decay")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Epsilon")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


# ============================================================
# 10. 多次跑随机迷宫
# ============================================================

def run_many_random_mazes(agent, num_runs=10, maze_size=8, wall_prob=0.22, visualize=False):
    results = []

    for i in range(num_runs):
        grid = generate_random_maze(size=maze_size, wall_prob=wall_prob)

        if visualize:
            path, actions_taken, total_reward, success = animate_policy(
                agent,
                grid,
                epsilon=0.0,
                delay=0.12,
                title=f"Random Maze {i + 1}/{num_runs}",
            )
        else:
            path, actions_taken, total_reward, success = run_policy_once(
                agent,
                grid,
                epsilon=0.0,
            )

        results.append({
            "run": i + 1,
            "success": success,
            "reward": total_reward,
            "steps": len(path) - 1,
            "actions": actions_taken,
        })

        print(
            f"[Run {i + 1}] "
            f"success={success}, "
            f"reward={total_reward}, "
            f"steps={len(path) - 1}"
        )

    success_rate = sum(1 for r in results if r["success"]) / num_runs
    avg_reward = np.mean([r["reward"] for r in results])
    avg_steps = np.mean([r["steps"] for r in results])

    print("\n=== Multi-random-maze summary ===")
    print(f"Runs: {num_runs}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average steps: {avg_steps:.2f}")

    return results


# ============================================================
# 11. 主流程
# ============================================================

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    print("Device:", DEVICE)

    # ------------------------------------------------------------
    # 配置区
    # ------------------------------------------------------------

    maze_size = 8
    wall_prob = 0.22

    MODEL_PATH = "cnn_dqn_maze_8x8.pt"

    TRAIN = True
    LOAD_MODEL = False

    # 如果你已经训练并保存过模型：
    # TRAIN = False
    # LOAD_MODEL = True

    agent = DQNAgent(
        maze_size=maze_size,
        replay_capacity=100000,
        batch_size=64,
        gamma=0.9,
        lr=1e-3,
        target_update_steps=500,
    )

    if LOAD_MODEL:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        agent.load(MODEL_PATH)

    # ------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------

    if TRAIN:
        print("\n=== Training CNN-DQN ===")

        rewards, losses, buffer_sizes, epsilons = train_agent(
            agent,
            maze_size=maze_size,
            wall_prob=wall_prob,
            num_episodes=1200,
            max_steps=maze_size * maze_size * 3,
            live_every=100,
        )

        print("Training finished.")

        agent.save(MODEL_PATH)

        plot_training_curves(rewards, losses, buffer_sizes, epsilons)

    # ------------------------------------------------------------
    # 随机迷宫评估
    # ------------------------------------------------------------

    print("\n=== Transfer evaluation on unseen random mazes ===")

    random_success, random_avg_reward = evaluate(
        agent,
        num_mazes=100,
        maze_size=maze_size,
        wall_prob=wall_prob,
        trained=False,
    )

    trained_success, trained_avg_reward = evaluate(
        agent,
        num_mazes=100,
        maze_size=maze_size,
        wall_prob=wall_prob,
        trained=True,
    )

    print(f"Random policy success rate:  {random_success:.2%}")
    print(f"Random policy avg reward:    {random_avg_reward:.2f}")
    print(f"Trained policy success rate: {trained_success:.2%}")
    print(f"Trained policy avg reward:   {trained_avg_reward:.2f}")

    # ------------------------------------------------------------
    # 人工手画迷宫测试
    # ------------------------------------------------------------

    print("\n=== Manual maze demo ===")

    manual_maze = """
    S . . . . . . .
    # # # # . # # .
    . . . . . . # .
    . # # # # # # #
    . # . . . . . .
    . # . # # # # .
    . . . . . # . .
    # # # # # # . E
    """

    manual_grid, manual_start, manual_goal = parse_manual_maze(manual_maze)

    if manual_grid.shape[0] != maze_size or manual_grid.shape[1] != maze_size:
        raise ValueError(
            f"Manual maze shape={manual_grid.shape}, "
            f"but model expects {maze_size}x{maze_size}."
        )

    manual_path, manual_actions, manual_reward, manual_success = animate_policy(
        agent,
        manual_grid,
        start=manual_start,
        goal=manual_goal,
        epsilon=0.0,
        delay=0.18,
        title="Trained Policy on Manual Maze",
    )

    print("Manual maze success:", manual_success)
    print("Manual maze reward:", manual_reward)
    print("Manual maze actions:", manual_actions)

    # ------------------------------------------------------------
    # 多次随机迷宫测试
    # ------------------------------------------------------------

    print("\n=== Run many unseen random mazes ===")

    run_many_random_mazes(
        agent,
        num_runs=10,
        maze_size=maze_size,
        wall_prob=wall_prob,
        visualize=False,
    )