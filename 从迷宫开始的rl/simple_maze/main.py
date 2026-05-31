import random
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. 定义更一般的迷宫
# =========================

maze = np.array([
    ["S", ".", ".", "."],
    ["#", "#", ".", "#"],
    [".", "K", ".", "R"],
    [".", "#", "T", "E"],
])

start_pos = (0, 0)
end_pos = (3, 3)

actions = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}

action_names = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
}

# 状态 = (x, y, has_key, got_reward)
# has_key: 0/1
# got_reward: 0/1
start_state = (start_pos[0], start_pos[1], 0, 0)

success_prob = 0.85  # 动作成功概率；设成 1.0 就是确定转移


# =========================
# 2. 概率转移动作
# =========================

def sample_actual_action(intended_action):
    """
    intended_action 是智能体想做的动作。
    但环境可能让它滑向别的方向。
    """
    if random.random() < success_prob:
        return intended_action

    other_actions = [a for a in actions.keys() if a != intended_action]
    return random.choice(other_actions)


# =========================
# 3. 环境 step
# =========================

def step(state, intended_action):
    x, y, has_key, got_reward = state

    actual_action = sample_actual_action(intended_action)
    dx, dy = actions[actual_action]

    nx = x + dx
    ny = y + dy

    # 越界
    if nx < 0 or nx >= maze.shape[0] or ny < 0 or ny >= maze.shape[1]:
        return state, -5, False, actual_action

    # 撞墙
    if maze[nx, ny] == "#":
        return state, -5, False, actual_action

    cell = maze[nx, ny]

    new_has_key = has_key
    new_got_reward = got_reward
    reward = -1
    done = False

    # 拿钥匙
    if cell == "K" and has_key == 0:
        new_has_key = 1
        reward = 2

    # 奖励点，只能拿一次
    elif cell == "R" and got_reward == 0:
        new_got_reward = 1
        reward = 5

    # 陷阱
    elif cell == "T":
        reward = -10
        done = True

    # 终点
    elif cell == "E":
        if has_key == 1:
            reward = 20
            done = True
        else:
            reward = -5
            done = False

    next_state = (nx, ny, new_has_key, new_got_reward)
    return next_state, reward, done, actual_action


# =========================
# 4. Q 表
# =========================

rows, cols = maze.shape
num_key_states = 2
num_reward_states = 2
num_actions = len(actions)

q_table = np.zeros((rows, cols, num_key_states, num_reward_states, num_actions))


def choose_action(state, epsilon):
    x, y, has_key, got_reward = state

    if random.random() < epsilon:
        return random.choice(list(actions.keys()))

    return int(np.argmax(q_table[x, y, has_key, got_reward]))


def update_q_table(state, action, reward, next_state, alpha, gamma):
    x, y, has_key, got_reward = state
    nx, ny, n_has_key, n_got_reward = next_state

    old_q = q_table[x, y, has_key, got_reward, action]
    best_next_q = np.max(q_table[nx, ny, n_has_key, n_got_reward])

    target = reward + gamma * best_next_q
    q_table[x, y, has_key, got_reward, action] = old_q + alpha * (target - old_q)


# =========================
# 5. 训练
# =========================

def train(
    num_episodes=3000,
    max_steps=80,
    alpha=0.4,
    gamma=0.9,
    epsilon=0.25,
):
    rewards = []

    for episode in range(num_episodes):
        state = start_state
        total_reward = 0

        for _ in range(max_steps):
            action = choose_action(state, epsilon)
            next_state, reward, done, actual_action = step(state, action)

            update_q_table(state, action, reward, next_state, alpha, gamma)

            total_reward += reward
            state = next_state

            if done:
                break

        rewards.append(total_reward)

    return rewards


# =========================
# 6. 评估：贪心策略
# =========================

def run_greedy_episode(max_steps=80):
    state = start_state
    path = [state]
    intended_actions = []
    actual_actions = []
    total_reward = 0

    for _ in range(max_steps):
        x, y, has_key, got_reward = state
        action = int(np.argmax(q_table[x, y, has_key, got_reward]))

        next_state, reward, done, actual_action = step(state, action)

        intended_actions.append(action_names[action])
        actual_actions.append(action_names[actual_action])
        path.append(next_state)

        total_reward += reward
        state = next_state

        if done:
            break

    return path, intended_actions, actual_actions, total_reward


# =========================
# 7. 随机 baseline
# =========================

def run_random_episode(max_steps=80):
    state = start_state
    path = [state]
    intended_actions = []
    actual_actions = []
    total_reward = 0

    for _ in range(max_steps):
        action = random.choice(list(actions.keys()))
        next_state, reward, done, actual_action = step(state, action)

        intended_actions.append(action_names[action])
        actual_actions.append(action_names[actual_action])
        path.append(next_state)

        total_reward += reward
        state = next_state

        if done:
            break

    return path, intended_actions, actual_actions, total_reward


# =========================
# 8. 可视化
# =========================

def draw_maze_with_path(path, title="Maze Path"):
    fig, ax = plt.subplots(figsize=(6, 6))

    rows, cols = maze.shape

    color_map = {
        "#": "black",
        ".": "white",
        "S": "lightgreen",
        "E": "lightcoral",
        "K": "gold",
        "R": "lightskyblue",
        "T": "orange",
    }

    for x in range(rows):
        for y in range(cols):
            cell = maze[x, y]
            color = color_map.get(cell, "white")

            rect = plt.Rectangle(
                (y, rows - 1 - x),
                1,
                1,
                facecolor=color,
                edgecolor="gray"
            )
            ax.add_patch(rect)

            if cell != ".":
                ax.text(
                    y + 0.5,
                    rows - 1 - x + 0.5,
                    cell,
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold"
                )

    # 画路径，只看位置，不画 has_key / got_reward
    if len(path) > 1:
        xs = [state[1] + 0.5 for state in path]
        ys = [rows - 1 - state[0] + 0.5 for state in path]

        ax.plot(xs, ys, marker="o", linewidth=2)

        for i, state in enumerate(path):
            x, y, has_key, got_reward = state
            ax.text(
                y + 0.5,
                rows - 1 - x + 0.25,
                str(i),
                ha="center",
                va="center",
                fontsize=9,
            )

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_aspect("equal")
    plt.show()


def plot_rewards(rewards):
    plt.figure(figsize=(8, 4))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards")
    plt.grid(True)
    plt.show()


# =========================
# 9. 主流程
# =========================

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    print("=== Before training: random policy ===")
    random_path, random_intended, random_actual, random_reward = run_random_episode()
    print("Random path:", random_path)
    print("Random intended actions:", random_intended)
    print("Random actual actions:", random_actual)
    print("Random reward:", random_reward)
    draw_maze_with_path(random_path, title="Before Training: Random Path")

    print("\n=== Training ===")
    rewards = train(
        num_episodes=3000,
        max_steps=80,
        alpha=0.4,
        gamma=0.9,
        epsilon=0.25,
    )
    print("Training finished.")
    plot_rewards(rewards)

    print("\n=== After training: greedy policy ===")
    learned_path, intended_actions, actual_actions, learned_reward = run_greedy_episode()
    print("Learned path:", learned_path)
    print("Intended actions:", intended_actions)
    print("Actual actions:", actual_actions)
    print("Learned reward:", learned_reward)
    draw_maze_with_path(learned_path, title="After Training: Learned Path")

    print("\n=== Useful Q values ===")
    for x in range(rows):
        for y in range(cols):
            if maze[x, y] != "#":
                for has_key in [0, 1]:
                    for got_reward in [0, 1]:
                        q_values = q_table[x, y, has_key, got_reward]
                        best_action = action_names[int(np.argmax(q_values))]
                        print(
                            f"State ({x},{y}, key={has_key}, reward={got_reward}) "
                            f"Q={q_values.round(2)} best={best_action}"
                        )