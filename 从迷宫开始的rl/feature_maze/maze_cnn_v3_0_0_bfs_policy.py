import argparse
import json
import math
import random
from collections import deque, Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

VERSION = "v3.0.0"
SIZE = 8
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

class MazeSample(NamedTuple):
    grid: np.ndarray
    start: Tuple[int, int]
    goal: Tuple[int, int]
    path: List[Tuple[int, int]]
    maze_type: str
    difficulty: str


@dataclass
class Config:
    size: int = 8
    episodes: int = 300
    steps_per_episode: int = 16
    batch_size: int = 64
    maze_pool_size: int = 48
    lr: float = 1e-4
    eval_n: int = 50
    seed: int = 42
    output_dir: str = "./v3.0.0"
    output_name: str = "v3.0.0_bfs_policy_cnn"
    model: str = ""
    maze_schedule: str = "mixed"
    max_steps: int = 64


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
            raise ValueError(f"row {r} must have {SIZE} chars, got {len(line)}")
        for c, ch in enumerate(line):
            if ch == "#":
                grid[r, c] = 1
            elif ch == ".":
                grid[r, c] = 0
            elif ch == "S":
                start = (r, c)
            elif ch == "G":
                goal = (r, c)
            else:
                raise ValueError(f"invalid char {ch!r}")
    if start is None or goal is None:
        raise ValueError("custom maze must include S and G")
    return grid, start, goal


def parse_topology_text(text: str) -> np.ndarray:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) != SIZE:
        raise ValueError(f"manual topology must have {SIZE} rows")
    grid = np.zeros((SIZE, SIZE), dtype=np.int64)
    for r, line in enumerate(lines):
        if len(line) != SIZE:
            raise ValueError(f"row {r} must have {SIZE} chars")
        for c, ch in enumerate(line):
            if ch == "#":
                grid[r, c] = 1
            elif ch == ".":
                grid[r, c] = 0
            else:
                raise ValueError("topology must use only . and #")
    return grid


def transform_grid_randomly(grid: np.ndarray) -> np.ndarray:
    g = grid.copy()
    if random.random() < 0.5:
        g = np.fliplr(g)
    if random.random() < 0.5:
        g = np.flipud(g)
    return g.copy()


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
    path = []
    cur = goal
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


def random_free_cell(grid: np.ndarray) -> Tuple[int, int]:
    free = np.argwhere(grid == 0)
    r, c = free[random.randrange(len(free))]
    return int(r), int(c)


def sample_random_start_goal_on_grid(
    grid: np.ndarray,
    min_manhattan: int = 6,
    min_shortest_len: int = 6,
    max_tries: int = 1000,
) -> Tuple[Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]:
    free = [tuple(map(int, p)) for p in np.argwhere(grid == 0)]
    for _ in range(max_tries):
        start, goal = random.choice(free), random.choice(free)
        if start == goal:
            continue
        if abs(start[0] - goal[0]) + abs(start[1] - goal[1]) < min_manhattan:
            continue
        path = bfs_shortest_path(grid, start, goal)
        if path is not None and len(path) - 1 >= min_shortest_len:
            return start, goal, path
    raise RuntimeError("failed to sample start/goal")


def sample_manual_maze(difficulty: Optional[str] = None) -> MazeSample:
    difficulty = difficulty or random.choice(["easy", "medium", "hard"])
    grid = parse_topology_text(random.choice(MANUAL_MAZES[difficulty]))
    grid = transform_grid_randomly(grid)
    min_len = {"easy": 6, "medium": 8, "hard": 10}[difficulty]
    start, goal, path = sample_random_start_goal_on_grid(grid, min_manhattan=6, min_shortest_len=min_len)
    return MazeSample(grid, start, goal, path, f"manual_{difficulty}", difficulty)


def generate_random_maze(difficulty: str, max_tries: int = 5000) -> MazeSample:
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
            start, goal = random_free_cell(grid), random_free_cell(grid)
            if start == goal:
                continue
            if abs(start[0] - goal[0]) + abs(start[1] - goal[1]) < 6:
                continue
            path = bfs_shortest_path(grid, start, goal)
            if path is not None and min_len <= len(path) - 1 <= max_len:
                return MazeSample(grid, start, goal, path, f"random_{difficulty}", difficulty)
    raise RuntimeError(f"failed to generate {difficulty}")


TRAIN_DIFFICULTY_WEIGHTS = {"easy": 0.4, "medium": 0.3, "hard": 0.3}


def sample_weighted_train_difficulty() -> str:
    names = list(TRAIN_DIFFICULTY_WEIGHTS.keys())
    probs = np.array([TRAIN_DIFFICULTY_WEIGHTS[k] for k in names], dtype=np.float64)
    probs = probs / probs.sum()
    return str(np.random.choice(names, p=probs))


def sample_train_maze(cfg: Config) -> MazeSample:
    if cfg.maze_schedule in ["easy", "medium", "hard"]:
        return generate_random_maze(cfg.maze_schedule)
    if cfg.maze_schedule == "manual":
        return sample_manual_maze()
    # mixed default: easy 40%, medium 30%, hard 30%
    return generate_random_maze(sample_weighted_train_difficulty())


def sample_eval_maze(eval_type: str) -> MazeSample:
    if eval_type in ["easy", "medium", "hard"]:
        return generate_random_maze(eval_type)
    if eval_type == "manual":
        return sample_manual_maze()
    if eval_type == "heldout":
        grid, start, goal = parse_maze_text(TEST_MAZE_TEXT)
        path = bfs_shortest_path(grid, start, goal)
        if path is None:
            raise ValueError("heldout maze is unsolvable")
        return MazeSample(grid, start, goal, path, "heldout", "heldout")
    raise ValueError(eval_type)


class MazePool:
    """Cache a small set of mazes + BFS paths; training batches sample from the pool."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.samples: List[MazeSample] = []

    def refill(self, desc: str = "maze pool") -> None:
        self.samples = []
        for _ in tqdm(range(self.cfg.maze_pool_size), desc=desc, leave=False):
            self.samples.append(sample_train_maze(self.cfg))

    def sample_state_label(self) -> Tuple[np.ndarray, int]:
        maze = random.choice(self.samples)
        i = random.randrange(len(maze.path) - 1)
        pos, nxt = maze.path[i], maze.path[i + 1]
        delta = (nxt[0] - pos[0], nxt[1] - pos[1])
        action = ACTIONS.index(delta)
        state = state_from_grid_pos_goal(maze.grid, pos, maze.goal)
        return state, action


def state_from_grid_pos_goal(grid: np.ndarray, pos: Tuple[int, int], goal: Tuple[int, int]) -> np.ndarray:
    walls = grid.astype(np.float32)
    agent = np.zeros((SIZE, SIZE), dtype=np.float32)
    agent[pos] = 1.0
    goal_ch = np.zeros((SIZE, SIZE), dtype=np.float32)
    goal_ch[goal] = 1.0
    return np.stack([walls, agent, goal_ch], axis=0)


def sample_supervised_batch(pool: MazePool, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    states = np.empty((batch_size, 3, SIZE, SIZE), dtype=np.float32)
    labels = np.empty((batch_size,), dtype=np.int64)
    for i in range(batch_size):
        state, action = pool.sample_state_label()
        states[i] = state
        labels[i] = action
    return states, labels


class CNNPolicy(nn.Module):
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


def select_action(model: CNNPolicy, state: np.ndarray, device: torch.device) -> int:
    with torch.no_grad():
        x = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        return int(model(x).argmax(dim=1).item())


def train(cfg: Config, device: torch.device):
    model = CNNPolicy().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    losses, accs = [], []
    print("\n=== Train v3.0.0 BFS Policy CNN ===")
    print("Supervised BFS action prediction. No DQN, no replay, no reward target.")
    updates_per_ep = cfg.steps_per_episode * cfg.batch_size
    print(
        f"episodes={cfg.episodes}, steps/ep={cfg.steps_per_episode}, "
        f"batch={cfg.batch_size}, pool={cfg.maze_pool_size}, "
        f"updates/ep={updates_per_ep}, lr={cfg.lr}"
    )
    print(f"maze_schedule={cfg.maze_schedule} (mixed => easy/medium/hard = 0.4/0.3/0.3)")
    pool = MazePool(cfg)
    ep_bar = tqdm(range(1, cfg.episodes + 1), desc="Train", unit="ep", position=0)
    for ep in ep_bar:
        pool.refill(desc=f"ep {ep} pool")
        ep_losses, ep_accs = [], []
        step_bar = tqdm(
            range(cfg.steps_per_episode),
            desc=f"ep {ep} steps",
            unit="step",
            leave=False,
            position=1,
        )
        for _ in step_bar:
            states, labels = sample_supervised_batch(pool, cfg.batch_size)
            x = torch.tensor(states, dtype=torch.float32, device=device)
            y = torch.tensor(labels, dtype=torch.long, device=device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            with torch.no_grad():
                acc = (logits.argmax(dim=1) == y).float().mean().item()
            step_loss = float(loss.item())
            ep_losses.append(step_loss)
            ep_accs.append(acc)
            step_bar.set_postfix(loss=f"{step_loss:.4f}", acc=f"{acc * 100:.1f}%", refresh=False)
        losses.append(float(np.mean(ep_losses)))
        accs.append(float(np.mean(ep_accs)))
        recent_loss = float(np.mean(losses[-100:]))
        recent_acc = float(np.mean(accs[-100:]) * 100.0)
        ep_bar.set_postfix(loss=f"{recent_loss:.4f}", acc=f"{recent_acc:.2f}%", refresh=False)
    return model, {"losses": losses, "accuracies": accs}


def rollout(model: CNNPolicy, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], cfg: Config, device: torch.device, debug: bool = False):
    pos = start
    path = [pos]
    wall_hits = 0
    bfs_agree = 0
    bfs_total = 0
    action_hist = Counter()
    rows = []
    for step in range(cfg.max_steps):
        state = state_from_grid_pos_goal(grid, pos, goal)
        action = select_action(model, state, device)
        action_hist[action] += 1
        best = bfs_best_actions(grid, pos, goal)
        if best:
            bfs_total += 1
            bfs_agree += int(action in best)
        old_pos = pos
        dr, dc = ACTIONS[action]
        nxt = (pos[0] + dr, pos[1] + dc)
        hit_wall = False
        if not is_free(grid, nxt):
            wall_hits += 1
            hit_wall = True
            nxt = pos
        pos = nxt
        path.append(pos)
        rows.append({"step": step, "pos": str(old_pos), "action": ACTION_NAMES[action], "next_pos": str(pos), "hit_wall": hit_wall, "bfs_best": [ACTION_NAMES[a] for a in best]})
        if pos == goal:
            break
    success = path[-1] == goal
    result = {
        "success": success,
        "steps": len(path) - 1,
        "wall_hits": wall_hits,
        "wall_hit_rate": wall_hits / max(1, len(path) - 1),
        "bfs_action_agreement": bfs_agree / max(1, bfs_total),
        "repeat_count": len(path) - len(set(path)),
        "path": path,
        "rows": rows,
        "action_hist": {ACTION_NAMES[k]: v for k, v in action_hist.items()},
    }
    if debug:
        print("\n--- Policy rollout debug ---")
        print({k: v for k, v in result.items() if k not in ["path", "rows"]})
        print("path_tail=", result["path"][-16:])
        for row in result["rows"][-12:]:
            print(row)
    return result


def evaluate(model: CNNPolicy, cfg: Config, device: torch.device):
    summary = {}
    print("\n=== Evaluation Suite ===")
    eval_types = ["easy", "medium", "hard", "manual", "heldout"]
    for typ in tqdm(eval_types, desc="Eval", unit="suite"):
        n = cfg.eval_n if typ != "heldout" else 1
        results = []
        for _ in tqdm(range(n), desc=typ, unit="maze", leave=False):
            maze = sample_eval_maze(typ)
            results.append(rollout(model, maze.grid, maze.start, maze.goal, cfg, device))
        summary[typ] = {
            "success_rate": float(np.mean([r["success"] for r in results])),
            "steps": float(np.mean([r["steps"] for r in results])),
            "wall_hit_rate": float(np.mean([r["wall_hit_rate"] for r in results])),
            "bfs_action_agreement": float(np.mean([r["bfs_action_agreement"] for r in results])),
            "repeat_count": float(np.mean([r["repeat_count"] for r in results])),
        }
        print(f"{typ:10s} success={summary[typ]['success_rate']*100:6.2f}% wall={summary[typ]['wall_hit_rate']*100:6.2f}% bfsAgree={summary[typ]['bfs_action_agreement']*100:6.2f}% repeat={summary[typ]['repeat_count']:6.2f}")
    return summary


def ensure_output_dir(cfg: Config) -> Path:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def model_path(cfg: Config) -> Path:
    return Path(cfg.model) if cfg.model else ensure_output_dir(cfg) / f"{cfg.output_name}.pt"


def save_history(history: dict, cfg: Config):
    p = ensure_output_dir(cfg) / f"{cfg.output_name}_history.json"
    p.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Save] history: {p}")


def save_eval(summary: dict, cfg: Config):
    p = ensure_output_dir(cfg) / f"{cfg.output_name}_eval.json"
    p.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Save] eval summary: {p}")


def save_curves(history: dict, cfg: Config):
    p = ensure_output_dir(cfg) / f"{cfg.output_name}_curves.png"
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(history["losses"], label="loss")
    plt.legend(); plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(history["accuracies"], label="accuracy")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(p, dpi=160)
    plt.close()
    print(f"[Save] curves: {p}")


def render_rollout_gif(model: CNNPolicy, cfg: Config, device: torch.device, grid, start, goal, name: str):
    result = rollout(model, grid, start, goal, cfg, device, debug=True)
    path_cells = result["path"]
    out_path = ensure_output_dir(cfg) / f"{cfg.output_name}_test_{name}.gif"
    fig, ax = plt.subplots(figsize=(6, 6))
    def make_img(frame_idx):
        img = np.ones((SIZE, SIZE, 3), dtype=np.float32)
        img[grid == 1] = np.array([0.05, 0.05, 0.05])
        for p in path_cells[: frame_idx + 1]:
            if p not in [start, goal]:
                img[p] = np.array([0.45, 0.75, 1.0])
        img[start] = np.array([0.0, 0.85, 0.0])
        img[goal] = np.array([1.0, 0.2, 0.2])
        img[path_cells[frame_idx]] = np.array([1.0, 0.85, 0.0])
        return img
    def update(frame_idx):
        ax.clear(); ax.imshow(make_img(frame_idx))
        ax.set_xticks(np.arange(-0.5, SIZE, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, SIZE, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.set_title(f"{VERSION} {name} | frame={frame_idx+1}/{len(path_cells)} | success={result['success']} | wall={result['wall_hit_rate']:.1%}")
        return []
    ani = FuncAnimation(fig, update, frames=len(path_cells), interval=250, blit=False)
    ani.save(out_path, writer=PillowWriter(fps=4))
    plt.close(fig)
    print(f"[Save] rollout gif: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "eval", "test"], default="single")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test-maze-file", type=str, default="")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--steps-per-episode", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--maze-pool-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./v3.0.0")
    parser.add_argument("--output-name", type=str, default="v3.0.0_bfs_policy_cnn")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--maze-schedule", choices=["mixed", "easy", "medium", "hard", "manual"], default="mixed")
    parser.add_argument("--max-steps", type=int, default=64)
    args = parser.parse_args()
    if args.test:
        args.mode = "test"
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    cfg = Config(
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        batch_size=args.batch_size,
        maze_pool_size=args.maze_pool_size,
        lr=args.lr,
        eval_n=args.eval_n,
        seed=args.seed,
        output_dir=args.output_dir,
        output_name=args.output_name,
        model=args.model,
        maze_schedule=args.maze_schedule,
        max_steps=args.max_steps,
    )
    ensure_output_dir(cfg)
    p = model_path(cfg)
    if args.mode == "single":
        model, history = train(cfg, device)
        torch.save(model.state_dict(), p)
        print(f"[Save] model: {p}")
        summary = evaluate(model, cfg, device)
        save_history(history, cfg); save_eval(summary, cfg); save_curves(history, cfg)
    else:
        if not p.exists():
            raise FileNotFoundError(f"model not found: {p}")
        model = CNNPolicy().to(device)
        model.load_state_dict(torch.load(p, map_location=device))
        model.eval()
        print(f"[Load] model: {p}")
        if args.mode == "eval":
            save_eval(evaluate(model, cfg, device), cfg)
        else:
            if args.test_maze_file:
                grid, start, goal = parse_maze_text(Path(args.test_maze_file).read_text(encoding="utf-8"))
                render_rollout_gif(model, cfg, device, grid, start, goal, Path(args.test_maze_file).stem.replace(" ", "_"))
            else:
                for diff in ["easy", "medium", "hard"]:
                    maze = sample_manual_maze(diff)
                    render_rollout_gif(model, cfg, device, maze.grid, maze.start, maze.goal, diff)

if __name__ == "__main__":
    main()
