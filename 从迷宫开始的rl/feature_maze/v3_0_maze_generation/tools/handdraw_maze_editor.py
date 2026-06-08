#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive hand-drawn maze editor for collecting ASCII maze samples.

This is a standalone tool. It does not import or modify the 3.0.4 generator.

Controls:
  Left click / drag  : draw road '.'
  Right click / drag : draw wall '#'
  Double left click  : set S or G using a popup input
  Ctrl+S             : save / append current maze
  Ctrl+N             : new maze
  Ctrl+L             : clear all
  V                  : validate
  Esc                : quit

Output:
  Appends all saved mazes to one ASCII text file and additionally writes a
  sibling JSONL file for easier downstream parsing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import deque
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import tkinter as tk
    from tkinter import messagebox, simpledialog
except Exception as exc:  # pragma: no cover - GUI import depends on runtime
    tk = None
    messagebox = None
    simpledialog = None
    _TK_IMPORT_ERROR = exc
else:
    _TK_IMPORT_ERROR = None

Coord = Tuple[int, int]
Grid = List[List[str]]

SCRIPT_PATH = Path(__file__).resolve()
LEARNING_ROOT = SCRIPT_PATH.parent.parent.parent.parent.parent  # .../tools -> .../learning
DEFAULT_OUTPUT_FILE = str(LEARNING_ROOT / "从迷宫开始的rl/feature_maze/handdraw_mazes_ascii.txt")

PASSABLE = {".", "S", "G"}
VALID_CHARS = {"#", ".", "S", "G"}


def ensure_output_dir(path: str | Path) -> None:
    """Create parent directory for an output file if it does not exist."""
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def validate_positive_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{name} must be positive")
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive tkinter editor for hand-drawn maze ASCII samples."
    )
    parser.add_argument("--rows", type=lambda v: validate_positive_int(v, "rows"), default=8)
    parser.add_argument("--cols", type=lambda v: validate_positive_int(v, "cols"), default=8)
    parser.add_argument(
        "--cell-size", type=lambda v: validate_positive_int(v, "cell-size"), default=48
    )
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    return parser.parse_args(argv)


def find_start_goal(grid: Sequence[Sequence[str]]) -> Tuple[Optional[Coord], Optional[Coord]]:
    start: Optional[Coord] = None
    goal: Optional[Coord] = None
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == "S":
                if start is not None:
                    raise ValueError("Multiple S cells found")
                start = (r, c)
            elif value == "G":
                if goal is not None:
                    raise ValueError("Multiple G cells found")
                goal = (r, c)
    return start, goal


def validate_grid_shape_and_chars(grid: Sequence[Sequence[str]]) -> None:
    if not grid:
        raise ValueError("Grid is empty")
    cols = len(grid[0])
    if cols == 0:
        raise ValueError("Grid has zero columns")
    for r, row in enumerate(grid):
        if len(row) != cols:
            raise ValueError(f"Grid row {r} width mismatch: expected {cols}, got {len(row)}")
        for c, value in enumerate(row):
            if value not in VALID_CHARS:
                raise ValueError(f"Invalid grid character at ({r}, {c}): {value!r}")


def compute_bfs_len(grid: Sequence[Sequence[str]], start: Coord, goal: Coord) -> Optional[int]:
    """Return shortest four-neighbor path length, or None if unreachable."""
    validate_grid_shape_and_chars(grid)
    rows = len(grid)
    cols = len(grid[0])
    sr, sc = start
    gr, gc = goal
    if not (0 <= sr < rows and 0 <= sc < cols):
        raise ValueError(f"Start out of bounds: {start}")
    if not (0 <= gr < rows and 0 <= gc < cols):
        raise ValueError(f"Goal out of bounds: {goal}")
    if grid[sr][sc] not in PASSABLE:
        return None
    if grid[gr][gc] not in PASSABLE:
        return None
    if start == goal:
        return 0

    q = deque([(sr, sc, 0)])
    seen = {(sr, sc)}
    while q:
        r, c, d = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if (nr, nc) in seen or grid[nr][nc] not in PASSABLE:
                continue
            if (nr, nc) == goal:
                return d + 1
            seen.add((nr, nc))
            q.append((nr, nc, d + 1))
    return None


def count_free_cells(grid: Sequence[Sequence[str]]) -> int:
    return sum(1 for row in grid for value in row if value in PASSABLE)


def next_jsonl_path(ascii_path: str | Path) -> Path:
    path = Path(ascii_path)
    return path.with_suffix(".jsonl")


def load_next_maze_id(output_file: str | Path) -> int:
    """Scan existing ASCII file headers and return the next integer maze id."""
    path = Path(output_file)
    if not path.exists():
        return 1
    max_id = 0
    pattern = re.compile(r"^#\s*---\s*maze_id=(\d+)")
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                match = pattern.match(line.strip())
                if match:
                    max_id = max(max_id, int(match.group(1)))
    except OSError:
        # Let the GUI save path report a detailed write error later.
        return 1
    return max_id + 1


class HanddrawMazeEditor:
    def __init__(self, rows: int, cols: int, cell_size: int, output_file: str) -> None:
        if tk is None:
            raise RuntimeError(f"tkinter is unavailable: {_TK_IMPORT_ERROR}")
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.output_file = str(Path(output_file))
        self.jsonl_file = str(next_jsonl_path(self.output_file))

        self.grid: Grid = [["#" for _ in range(cols)] for _ in range(rows)]
        self.start: Optional[Coord] = None
        self.goal: Optional[Coord] = None
        self.next_maze_id = load_next_maze_id(self.output_file)
        self.saved_count = max(0, self.next_maze_id - 1)
        self.dirty = False
        self._last_drawn: Optional[Coord] = None

        self.root = tk.Tk()
        self.root.title("Handdraw Maze Editor")
        self.canvas: tk.Canvas
        self.status_var = tk.StringVar()
        self.build_ui()
        self.draw_grid()
        self.update_status()

    # ------------------------- UI construction -------------------------
    def build_ui(self) -> None:
        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        canvas_width = self.cols * self.cell_size
        canvas_height = self.rows * self.cell_size
        self.canvas = tk.Canvas(
            main,
            width=canvas_width,
            height=canvas_height,
            background="white",
            highlightthickness=0,
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=8, pady=8)

        panel = tk.Frame(main)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)

        button_specs = [
            ("Save / Append Current Maze", self.save_current_maze),
            ("New Maze", self.new_maze),
            ("Clear All", self.clear_all),
            ("Clear S", self.clear_start),
            ("Clear G", self.clear_goal),
            ("Validate", self.validate_maze),
            ("Quit", self.quit_editor),
        ]
        for text, command in button_specs:
            btn = tk.Button(panel, text=text, command=command, width=26)
            btn.pack(fill=tk.X, pady=3)

        help_text = (
            "Left drag: road '.'\n"
            "Right drag: wall '#'\n"
            "Double-click: set S/G\n"
            "Ctrl+S: save\nCtrl+N: new\nCtrl+L: clear\nV: validate\nEsc: quit"
        )
        tk.Label(panel, text=help_text, justify=tk.LEFT, anchor="w").pack(fill=tk.X, pady=12)

        status = tk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            relief=tk.SUNKEN,
            padx=6,
        )
        status.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.bind("<Button-1>", self.on_left_button)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.canvas.bind("<Button-3>", self.on_right_button)
        self.canvas.bind("<B3-Motion>", self.on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_mouse_release)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)

        self.root.bind("<Control-s>", lambda _e: self.save_current_maze())
        self.root.bind("<Control-S>", lambda _e: self.save_current_maze())
        self.root.bind("<Control-n>", lambda _e: self.new_maze())
        self.root.bind("<Control-N>", lambda _e: self.new_maze())
        self.root.bind("<Control-l>", lambda _e: self.clear_all())
        self.root.bind("<Control-L>", lambda _e: self.clear_all())
        self.root.bind("v", lambda _e: self.validate_maze())
        self.root.bind("V", lambda _e: self.validate_maze())
        self.root.bind("<Escape>", lambda _e: self.quit_editor())
        self.root.protocol("WM_DELETE_WINDOW", self.quit_editor)

    # ------------------------- Drawing -------------------------
    def draw_grid(self) -> None:
        self.canvas.delete("all")
        for r in range(self.rows):
            for c in range(self.cols):
                self.draw_cell(r, c)

    def draw_cell(self, row: int, col: int) -> None:
        x0 = col * self.cell_size
        y0 = row * self.cell_size
        x1 = x0 + self.cell_size
        y1 = y0 + self.cell_size
        value = self.grid[row][col]
        if value == "#":
            fill = "black"
            text = ""
            text_fill = "white"
        elif value == ".":
            fill = "white"
            text = ""
            text_fill = "black"
        elif value == "S":
            fill = "#2f80ed"
            text = "S"
            text_fill = "white"
        elif value == "G":
            fill = "#f2994a"
            text = "G"
            text_fill = "black"
        else:
            fill = "red"
            text = "?"
            text_fill = "white"

        tag = f"cell_{row}_{col}"
        self.canvas.delete(tag)
        self.canvas.create_rectangle(
            x0,
            y0,
            x1,
            y1,
            fill=fill,
            outline="gray",
            tags=tag,
        )
        if value in {"S", "G"}:
            radius = max(5, self.cell_size // 4)
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            if value == "S":
                self.canvas.create_oval(
                    cx - radius,
                    cy - radius,
                    cx + radius,
                    cy + radius,
                    fill="#1c5db8",
                    outline="white",
                    width=2,
                    tags=tag,
                )
            else:
                # A simple star-like marker using text keeps dependencies minimal.
                self.canvas.create_text(
                    cx,
                    cy,
                    text="★",
                    fill="black",
                    font=("Arial", max(12, self.cell_size // 2), "bold"),
                    tags=tag,
                )
        if text:
            self.canvas.create_text(
                (x0 + x1) / 2,
                (y0 + y1) / 2,
                text=text,
                fill=text_fill,
                font=("Arial", max(10, self.cell_size // 3), "bold"),
                tags=tag,
            )

    def update_status(self) -> None:
        self.status_var.set(
            f"{self.rows} x {self.cols} | start={self.start} | goal={self.goal} | "
            f"output_file={self.output_file} | saved_count={self.saved_count} | "
            f"next_maze_id={self.next_maze_id:06d}"
        )

    # ------------------------- Mouse events -------------------------
    def cell_from_event(self, event: tk.Event) -> Optional[Coord]:
        col = int(event.x // self.cell_size)
        row = int(event.y // self.cell_size)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return row, col
        return None

    def on_left_button(self, event: tk.Event) -> None:
        self._last_drawn = None
        self._draw_from_event(event, ".")

    def on_left_drag(self, event: tk.Event) -> None:
        self._draw_from_event(event, ".")

    def on_right_button(self, event: tk.Event) -> None:
        self._last_drawn = None
        self._draw_from_event(event, "#")

    def on_right_drag(self, event: tk.Event) -> None:
        self._draw_from_event(event, "#")

    def on_mouse_release(self, _event: tk.Event) -> None:
        self._last_drawn = None

    def _draw_from_event(self, event: tk.Event, value: str) -> None:
        coord = self.cell_from_event(event)
        if coord is None or coord == self._last_drawn:
            return
        self._last_drawn = coord
        self.set_cell(coord[0], coord[1], value)

    def on_double_click(self, event: tk.Event) -> None:
        coord = self.cell_from_event(event)
        if coord is None:
            return
        value = simpledialog.askstring(
            "Set Start / Goal",
            "请输入 S 或 G / Enter S or G:",
            parent=self.root,
        )
        if value is None:
            return
        value = value.strip().upper()
        if value not in {"S", "G"}:
            messagebox.showerror("Invalid input", "Please enter only S or G.")
            return
        self.set_start_or_goal(coord[0], coord[1], value)

    # ------------------------- Grid operations -------------------------
    def set_cell(self, row: int, col: int, value: str) -> None:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return
        if value not in {"#", "."}:
            raise ValueError("set_cell only accepts '#' or '.'")
        current = self.grid[row][col]
        if current in {"S", "G"}:
            # S/G are protected from draw overwrite. Use Clear S/G buttons.
            return
        if current == value:
            return
        self.grid[row][col] = value
        self.dirty = True
        self.draw_cell(row, col)
        self.update_status()

    def set_start_or_goal(self, row: int, col: int, marker: str) -> None:
        if marker not in {"S", "G"}:
            raise ValueError("marker must be S or G")
        coord = (row, col)
        if marker == "S" and self.goal == coord:
            messagebox.showerror("Invalid start", "S and G cannot be on the same cell.")
            return
        if marker == "G" and self.start == coord:
            messagebox.showerror("Invalid goal", "S and G cannot be on the same cell.")
            return

        if marker == "S":
            if self.start is not None:
                sr, sc = self.start
                self.grid[sr][sc] = "."
                self.draw_cell(sr, sc)
            self.start = coord
        else:
            if self.goal is not None:
                gr, gc = self.goal
                self.grid[gr][gc] = "."
                self.draw_cell(gr, gc)
            self.goal = coord

        self.grid[row][col] = marker
        self.dirty = True
        self.draw_cell(row, col)
        self.update_status()

    def clear_start(self) -> None:
        if self.start is None:
            return
        r, c = self.start
        self.grid[r][c] = "."
        self.start = None
        self.dirty = True
        self.draw_cell(r, c)
        self.update_status()

    def clear_goal(self) -> None:
        if self.goal is None:
            return
        r, c = self.goal
        self.grid[r][c] = "."
        self.goal = None
        self.dirty = True
        self.draw_cell(r, c)
        self.update_status()

    def clear_all(self) -> None:
        if self.dirty and not messagebox.askyesno(
            "Clear all", "Current maze has unsaved changes. Clear all anyway?"
        ):
            return
        self._reset_grid()

    def new_maze(self) -> None:
        if self.dirty and not messagebox.askyesno(
            "New maze", "Current maze has unsaved changes. Start a new maze anyway?"
        ):
            return
        self._reset_grid()

    def _reset_grid(self) -> None:
        self.grid = [["#" for _ in range(self.cols)] for _ in range(self.rows)]
        self.start = None
        self.goal = None
        self.dirty = False
        self._last_drawn = None
        self.draw_grid()
        self.update_status()

    # ------------------------- Validation and persistence -------------------------
    def grid_to_ascii(self) -> List[str]:
        lines = ["".join(row) for row in self.grid]
        if len(lines) != self.rows:
            raise ValueError("ASCII row count mismatch")
        for i, line in enumerate(lines):
            if len(line) != self.cols:
                raise ValueError(f"ASCII line {i} width mismatch")
            if any(ch not in VALID_CHARS for ch in line):
                raise ValueError(f"ASCII line {i} contains invalid characters")
        return lines

    def _validate_start_goal_present(self) -> bool:
        if self.start is None or self.goal is None:
            messagebox.showerror("Missing S/G", "Both S and G must be set before saving.")
            return False
        if self.start == self.goal:
            messagebox.showerror("Invalid S/G", "S and G cannot be the same cell.")
            return False
        return True

    def validate_maze(self) -> None:
        try:
            lines = self.grid_to_ascii()
            if self.start is None or self.goal is None:
                messagebox.showwarning(
                    "Validation",
                    f"Rows: {self.rows}\nCols: {self.cols}\nFree cells: {count_free_cells(self.grid)}\n"
                    "Missing S or G.",
                )
                return
            bfs_len = compute_bfs_len(self.grid, self.start, self.goal)
            connected = bfs_len is not None
            message = (
                f"Rows: {self.rows}\n"
                f"Cols: {self.cols}\n"
                f"Start: {self.start}\n"
                f"Goal: {self.goal}\n"
                f"Free cells: {count_free_cells(self.grid)}\n"
                f"Connected: {connected}\n"
                f"BFS length: {bfs_len if bfs_len is not None else 'unreachable'}"
            )
            messagebox.showinfo("Validation", message)
        except Exception as exc:
            messagebox.showerror("Validation error", str(exc))

    def save_current_maze(self) -> None:
        try:
            if not self._validate_start_goal_present():
                return
            assert self.start is not None and self.goal is not None
            bfs_len = compute_bfs_len(self.grid, self.start, self.goal)
            connected = bfs_len is not None
            if not connected:
                should_save = messagebox.askyesno(
                    "Maze is not connected",
                    "Start and goal are not connected. Save anyway?",
                )
                if not should_save:
                    return

            lines = self.grid_to_ascii()
            ensure_output_dir(self.output_file)
            ensure_output_dir(self.jsonl_file)
            maze_id_int = self.next_maze_id
            maze_id = f"{maze_id_int:06d}"
            header = (
                f"# --- maze_id={maze_id} rows={self.rows} cols={self.cols} "
                f"start={self.start} goal={self.goal} ---"
            )
            with open(self.output_file, "a", encoding="utf-8", newline="\n") as f:
                f.write(header + "\n")
                for line in lines:
                    f.write(line + "\n")
                f.write("\n")

            record = {
                "maze_id": maze_id,
                "rows": self.rows,
                "cols": self.cols,
                "grid": lines,
                "start": list(self.start),
                "goal": list(self.goal),
                "bfs_len": bfs_len,
                "connected": connected,
            }
            with open(self.jsonl_file, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            self.saved_count += 1
            self.next_maze_id += 1
            self.dirty = False
            self.update_status()
            messagebox.showinfo(
                "Saved",
                f"Maze {maze_id} appended to:\n{self.output_file}\n\nJSONL also appended to:\n{self.jsonl_file}",
            )
        except OSError as exc:
            messagebox.showerror("Save error", f"Cannot write output file:\n{exc}")
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))

    def quit_editor(self) -> None:
        if self.dirty and not messagebox.askyesno(
            "Quit", "Current maze has unsaved changes. Quit anyway?"
        ):
            return
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if tk is None:
        print(f"ERROR: tkinter is unavailable: {_TK_IMPORT_ERROR}", file=sys.stderr)
        return 1
    try:
        ensure_output_dir(args.output_file)
        editor = HanddrawMazeEditor(
            rows=args.rows,
            cols=args.cols,
            cell_size=args.cell_size,
            output_file=args.output_file,
        )
        editor.run()
    except Exception as exc:
        # If GUI has not started yet, stderr is safer than messagebox.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
