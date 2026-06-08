#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
External 3.0.4 maze dataset reader for v1.x BFS supervised tests.

Scope:
- Read-only parsing of 3.0.4 CSV / JSONL outputs.
- No v3 code import.
- No maze repair, no default start/goal, no fallback corridor.
- Converts accepted samples into plain maze/start/goal records for v1 scripts.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

MAZE_FIELD_CANDIDATES = ("json_grid", "maze", "grid", "ascii_grid")
ASCII_MAZE_FIELD_CANDIDATES = ("ascii_grid", "maze_ascii", "grid_ascii")
START_PAIR_CANDIDATES = (("start_r", "start_c"), ("start_row", "start_col"), ("start_y", "start_x"))
GOAL_PAIR_CANDIDATES = (("goal_r", "goal_c"), ("goal_row", "goal_col"), ("goal_y", "goal_x"))
START_SINGLE_CANDIDATES = ("start", "start_pos", "start_position")
GOAL_SINGLE_CANDIDATES = ("goal", "goal_pos", "goal_position")
BUCKET_FIELD_CANDIDATES = ("bin_key", "target_bucket", "target_bin", "full_bin", "bucket", "bin")
SOURCE_FIELD_CANDIDATES = ("origin_generator", "source", "generator", "route_id")
SPLIT_FIELD_CANDIDATES = ("split", "dataset_split", "train_val_test")
ID_FIELD_CANDIDATES = ("sample_id", "id", "lineage_id", "grid_start_goal_hash", "grid_hash")
METADATA_PREFIXES = ("rect_room_", "quota_", "target_", "from_", "origin_", "route", "lineage", "grid_", "visualization")

class ExternalDatasetError(RuntimeError):
    pass

def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def default(o: Any) -> Any:
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, Path): return str(o)
        if isinstance(o, set): return sorted(o)
        return str(o)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=default), encoding="utf-8")

def detect_format(path: Path, requested: str = "auto") -> str:
    if requested and requested != "auto":
        return requested.lower()
    suffix = path.suffix.lower()
    if suffix == ".jsonl": return "jsonl"
    if suffix == ".csv": return "csv"
    raise ExternalDatasetError(f"cannot infer external dataset format from suffix: {path}")

def read_records(path: Path, fmt: str, max_records: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise ExternalDatasetError(f"external dataset path does not exist: {path}")
    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    if fmt == "csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                records.append(dict(row, __row_index=i))
                if max_records is not None and len(records) >= max_records:
                    break
        fields = reader.fieldnames or []
    elif fmt == "jsonl":
        fields_set = set()
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        failures.append({"row_index": i, "reason": "jsonl row is not an object"})
                        continue
                    obj["__row_index"] = i
                    records.append(obj)
                    fields_set.update(obj.keys())
                except Exception as e:
                    failures.append({"row_index": i, "reason": f"json parse failed: {e}"})
                if max_records is not None and len(records) >= max_records:
                    break
        fields = sorted(fields_set)
    else:
        raise ExternalDatasetError(f"unsupported external dataset format: {fmt}")
    return records, {"path": str(path), "format": fmt, "loaded_records": len(records), "read_failures": failures[:20], "fields": fields}

def _nonempty(v: Any) -> bool:
    if v is None: return False
    if isinstance(v, float) and math.isnan(v): return False
    s = str(v)
    return bool(s.strip()) and s.strip().lower() not in {"nan", "none", "null"}

def detect_schema(records: Sequence[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, Any]:
    fields = list(fields)
    field_set = set(fields)
    maze_candidates = [f for f in MAZE_FIELD_CANDIDATES if f in field_set]
    start_pair_candidates = [pair for pair in START_PAIR_CANDIDATES if pair[0] in field_set and pair[1] in field_set]
    goal_pair_candidates = [pair for pair in GOAL_PAIR_CANDIDATES if pair[0] in field_set and pair[1] in field_set]
    start_single_candidates = [f for f in START_SINGLE_CANDIDATES if f in field_set]
    goal_single_candidates = [f for f in GOAL_SINGLE_CANDIDATES if f in field_set]
    bucket_candidates = [f for f in BUCKET_FIELD_CANDIDATES if f in field_set]
    source_candidates = [f for f in SOURCE_FIELD_CANDIDATES if f in field_set]
    split_candidates = [f for f in SPLIT_FIELD_CANDIDATES if f in field_set]
    id_candidates = [f for f in ID_FIELD_CANDIDATES if f in field_set]
    metadata_candidates = [f for f in fields if any(f.startswith(p) for p in METADATA_PREFIXES)]
    unknown = [f for f in fields if f not in set(maze_candidates + bucket_candidates + source_candidates + split_candidates + id_candidates + metadata_candidates) and all(f not in pair for pair in start_pair_candidates + goal_pair_candidates) and f not in start_single_candidates and f not in goal_single_candidates]

    selected_maze_field = None
    # Prefer json_grid over ascii_grid when both exist, because json_grid is lossless for wall/free.
    for f in ("json_grid", "maze", "grid", "ascii_grid"):
        if f in maze_candidates:
            selected_maze_field = f
            break
    selected_start_pair = start_pair_candidates[0] if len(start_pair_candidates) == 1 else None
    selected_goal_pair = goal_pair_candidates[0] if len(goal_pair_candidates) == 1 else None
    selected_start_single = start_single_candidates[0] if not selected_start_pair and len(start_single_candidates) == 1 else None
    selected_goal_single = goal_single_candidates[0] if not selected_goal_pair and len(goal_single_candidates) == 1 else None
    selected_bucket_field = bucket_candidates[0] if bucket_candidates else None
    selected_source_field = source_candidates[0] if source_candidates else None
    selected_split_field = split_candidates[0] if split_candidates else None
    selected_id_field = id_candidates[0] if id_candidates else None

    blocking = []
    if not selected_maze_field:
        blocking.append("maze field is not uniquely detected")
    if not (selected_start_pair or selected_start_single):
        blocking.append("start field is not uniquely detected")
    if not (selected_goal_pair or selected_goal_single):
        blocking.append("goal field is not uniquely detected")

    return {
        "fields": fields,
        "candidate_maze_fields": maze_candidates,
        "selected_maze_field": selected_maze_field,
        "candidate_start_fields": {"pairs": start_pair_candidates, "single": start_single_candidates},
        "selected_start_pair": selected_start_pair,
        "selected_start_single": selected_start_single,
        "candidate_goal_fields": {"pairs": goal_pair_candidates, "single": goal_single_candidates},
        "selected_goal_pair": selected_goal_pair,
        "selected_goal_single": selected_goal_single,
        "candidate_bucket_fields": bucket_candidates,
        "selected_bucket_field": selected_bucket_field,
        "candidate_source_fields": source_candidates,
        "selected_source_field": selected_source_field,
        "candidate_split_fields": split_candidates,
        "selected_split_field": selected_split_field,
        "candidate_id_fields": id_candidates,
        "selected_id_field": selected_id_field,
        "candidate_metadata_fields": metadata_candidates,
        "unrecognized_fields": unknown,
        "blocking_reasons": blocking,
        "status": "BLOCKED" if blocking else "READY",
    }

def parse_coord_pair(row: Dict[str, Any], pair: Optional[Tuple[str, str]], single: Optional[str], name: str) -> Tuple[int, int]:
    if pair:
        r, c = row.get(pair[0]), row.get(pair[1])
        if not (_nonempty(r) and _nonempty(c)):
            raise ValueError(f"missing {name} pair fields {pair}")
        return int(float(r)), int(float(c))
    if single:
        v = row.get(single)
        if not _nonempty(v):
            raise ValueError(f"missing {name} field {single}")
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return int(v[0]), int(v[1])
        s = str(v).strip()
        if s.startswith("[") or s.startswith("("):
            obj = json.loads(s.replace("(", "[").replace(")", "]"))
            return int(obj[0]), int(obj[1])
        parts = [p.strip() for p in s.replace(";", ",").split(",")]
        if len(parts) == 2:
            return int(float(parts[0])), int(float(parts[1]))
    raise ValueError(f"cannot parse {name}")

def _parse_ascii_lines_to_maze(lines: Sequence[str]) -> np.ndarray:
    lines = [str(ln).strip() for ln in lines if str(ln).strip()]
    if not lines:
        raise ValueError("ascii maze has no lines")
    width = len(lines[0])
    if any(len(ln) != width for ln in lines):
        raise ValueError("ascii maze rows have inconsistent widths")
    arr = np.zeros((len(lines), width), dtype=np.int8)
    for r, ln in enumerate(lines):
        for c, ch in enumerate(ln):
            if ch == "#":
                arr[r, c] = 1
            elif ch in ".SG":
                arr[r, c] = 0
            else:
                raise ValueError(f"unsupported ascii maze char {ch!r}")
    return arr


def parse_maze_grid(row: Dict[str, Any], field: str) -> np.ndarray:
    raw = row.get(field)
    if not _nonempty(raw):
        raise ValueError(f"empty maze field: {field}")
    # Handdraw JSONL often stores grid as ["..#", "#S.", "..G"].
    # This is accepted as read-only maze geometry. S/G are treated as free cells;
    # start/goal coordinates remain the explicit source of endpoint truth.
    if isinstance(raw, (list, tuple)) and raw and all(isinstance(x, str) for x in raw):
        return _parse_ascii_lines_to_maze(raw)
    if field == "ascii_grid" or (isinstance(raw, str) and ("#" in raw or "." in raw or "S" in raw or "G" in raw) and not str(raw).strip().startswith("[")):
        return _parse_ascii_lines_to_maze(str(raw).splitlines())
    obj = raw
    if isinstance(raw, str):
        obj = json.loads(raw)
        if isinstance(obj, list) and obj and all(isinstance(x, str) for x in obj):
            return _parse_ascii_lines_to_maze(obj)
    arr = np.asarray(obj, dtype=np.int8)
    if arr.ndim != 2:
        raise ValueError(f"maze grid must be 2-D; got shape {arr.shape}")
    values = set(int(x) for x in np.unique(arr).tolist())
    if not values.issubset({0, 1}):
        raise ValueError(f"maze grid must contain only 0/1; got {sorted(values)}")
    return arr

def stable_sample_id(row: Dict[str, Any], schema: Dict[str, Any]) -> str:
    id_field = schema.get("selected_id_field")
    if id_field and _nonempty(row.get(id_field)):
        return str(row.get(id_field))
    payload = json.dumps({k: v for k, v in row.items() if not k.startswith("__")}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

def make_bfs_len_bin(bfs_len: Optional[int]) -> str:
    if bfs_len is None: return "unavailable"
    if bfs_len <= 10: return "short"
    if bfs_len <= 18: return "medium"
    return "long"

def convert_records(records: Sequence[Dict[str, Any]], schema: Dict[str, Any], max_samples: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if schema.get("status") != "READY":
        raise ExternalDatasetError("schema is blocked; conversion is not allowed")
    converted: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for row in records:
        if max_samples is not None and len(converted) >= max_samples:
            break
        rid = row.get("__row_index")
        try:
            maze = parse_maze_grid(row, schema["selected_maze_field"])
            start = parse_coord_pair(row, tuple(schema["selected_start_pair"]) if schema.get("selected_start_pair") else None, schema.get("selected_start_single"), "start")
            goal = parse_coord_pair(row, tuple(schema["selected_goal_pair"]) if schema.get("selected_goal_pair") else None, schema.get("selected_goal_single"), "goal")
            n, m = maze.shape
            if not (0 <= start[0] < n and 0 <= start[1] < m):
                raise ValueError(f"start out of bounds: {start}")
            if not (0 <= goal[0] < n and 0 <= goal[1] < m):
                raise ValueError(f"goal out of bounds: {goal}")
            if maze[start] == 1:
                raise ValueError(f"start is wall: {start}")
            if maze[goal] == 1:
                raise ValueError(f"goal is wall: {goal}")
            sample_id = stable_sample_id(row, schema)
            bucket_field = schema.get("selected_bucket_field")
            source_field = schema.get("selected_source_field")
            split_field = schema.get("selected_split_field")
            metadata = {k: row.get(k) for k in schema.get("candidate_metadata_fields", []) if k in row}
            # Preserve common useful fields even if not classified as metadata.
            for k in ["bin_key", "target_bucket", "target_bin", "full_bin", "bucket", "bin", "cycle_bin", "bfs_bin", "choice_bin", "endpoint_bin", "bfs_len", "origin_generator", "source", "generator", "route_id", "grid_hash", "grid_start_goal_hash", "visualization_path"]:
                if k in row:
                    metadata[k] = row.get(k)
            try:
                bfs_len = int(float(row["bfs_len"])) if _nonempty(row.get("bfs_len")) else None
            except Exception:
                bfs_len = None
            converted.append({
                "sample_id": sample_id,
                "row_index": rid,
                "maze": maze,
                "start": tuple(start),
                "goal": tuple(goal),
                "target_bucket": str(row.get(bucket_field)) if bucket_field else "unavailable",
                "source": str(row.get(source_field)) if source_field else "unavailable",
                "split": str(row.get(split_field)) if split_field and _nonempty(row.get(split_field)) else None,
                "bfs_len_source": bfs_len,
                "bfs_len_bin_source": make_bfs_len_bin(bfs_len),
                "metadata": metadata,
                "raw_row": {k: v for k, v in row.items() if not k.startswith("__")},
            })
        except Exception as e:
            skipped.append({"row_index": rid, "reason": str(e), "sample_preview": {k: row.get(k) for k in list(row.keys())[:12]}})
    report = {
        "input_records": len(records),
        "converted_records": len(converted),
        "skipped_records": len(skipped),
        "skip_reasons": summarize_reasons(skipped),
        "skipped_examples": skipped[:20],
        "mutation_policy": "read_only_no_repair_no_default_start_goal_no_fallback_corridor",
    }
    return converted, report

def summarize_reasons(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        reason = str(r.get("reason", "unknown")).split(":")[0]
        out[reason] = out.get(reason, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

def deterministic_split(records: Sequence[Dict[str, Any]], seed: int = 42, ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15)) -> Tuple[Dict[str, str], Dict[str, Any]]:
    existing = [r.get("split") for r in records if r.get("split")]
    if existing:
        mapping = {r["sample_id"]: r.get("split") or "unavailable" for r in records}
        counts: Dict[str, int] = {}
        for s in mapping.values(): counts[s] = counts.get(s, 0) + 1
        return mapping, {"mode": "source_split", "split_counts": counts, "seed": None, "ratios": None}
    mapping: Dict[str, str] = {}
    counts = {"train": 0, "validation": 0, "test": 0}
    train_r, val_r, _ = ratios
    for r in records:
        key = str(r["sample_id"])
        h = hashlib.sha1((str(seed) + "|" + key).encode("utf-8")).hexdigest()
        x = int(h[:12], 16) / float(16**12)
        if x < train_r: split = "train"
        elif x < train_r + val_r: split = "validation"
        else: split = "test"
        mapping[key] = split
        counts[split] += 1
        r["split"] = split
    return mapping, {"mode": "deterministic_hash_split", "seed": seed, "ratios": {"train": ratios[0], "validation": ratios[1], "test": ratios[2]}, "split_counts": counts, "hash_key": "sample_id"}

def build_dataset_source_manifest(dataset_path: Path, fmt: str, schema: Dict[str, Any], read_info: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset_path": str(dataset_path),
        "external_dataset_format": fmt,
        "loaded_records": read_info.get("loaded_records"),
        "field_count": len(schema.get("fields", [])),
        "selected_fields": {
            "maze": schema.get("selected_maze_field"),
            "start_pair": schema.get("selected_start_pair"),
            "start_single": schema.get("selected_start_single"),
            "goal_pair": schema.get("selected_goal_pair"),
            "goal_single": schema.get("selected_goal_single"),
            "bucket": schema.get("selected_bucket_field"),
            "source": schema.get("selected_source_field"),
            "split": schema.get("selected_split_field"),
            "id": schema.get("selected_id_field"),
        },
        "read_only_policy": "input files are never modified",
        "code_dependency_policy": "no v3 code import; file input only",
    }

def load_external_dataset(dataset_path: str, requested_format: str = "auto", max_records: Optional[int] = None, split_seed: int = 42) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    path = Path(dataset_path)
    fmt = detect_format(path, requested_format)
    records, read_info = read_records(path, fmt, max_records=max_records)
    schema = detect_schema(records, read_info.get("fields", []))
    if schema.get("status") != "READY":
        manifest = build_dataset_source_manifest(path, fmt, schema, read_info)
        return [], schema, manifest, {"status": "BLOCKED", "blocking_reasons": schema.get("blocking_reasons", [])}
    converted, conversion_report = convert_records(records, schema, max_samples=max_records)
    split_manifest, split_report = deterministic_split(converted, seed=split_seed)
    manifest = build_dataset_source_manifest(path, fmt, schema, read_info)
    conversion_report["split_report"] = split_report
    return converted, schema, manifest, conversion_report


# =============================================================================
# Rollout GIF utilities for extdata test/evaluation
# =============================================================================

def _as_pair(x, fallback=(0, 0)):
    if x is None:
        return tuple(fallback)
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return int(x[0]), int(x[1])
    return tuple(fallback)


def rollout_maze_array(rollout: Dict[str, Any]) -> Optional[np.ndarray]:
    """Return maze array from a rollout dict, if present.

    Expected keys are `maze_grid` or `maze`. Values may be a nested list or ndarray.
    The function never repairs or infers a missing maze; missing maze means GIF cannot be rendered.
    """
    raw = rollout.get("maze_grid", rollout.get("maze"))
    if raw is None:
        return None
    try:
        arr = np.asarray(raw, dtype=np.int8)
        if arr.ndim != 2:
            return None
        return arr
    except Exception:
        return None


def render_rollout_frame(maze: np.ndarray, agent_pos, goal_pos, title: str = "") -> np.ndarray:
    """Render one rollout frame as RGB image using matplotlib Agg backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n, m = maze.shape
    img = np.ones((n, m, 3), dtype=np.float32)
    img[maze == 1] = [0.05, 0.05, 0.05]
    g = _as_pair(goal_pos)
    a = _as_pair(agent_pos)
    if 0 <= g[0] < n and 0 <= g[1] < m:
        img[g] = [1.0, 0.85, 0.0]
    if 0 <= a[0] < n and 0 <= a[1] < m:
        img[a] = [1.0, 0.1, 0.1]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, m, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return arr


def save_rollout_gif(rollout: Dict[str, Any], path: Path, fps: int = 3, max_frames: int = 96) -> Dict[str, Any]:
    """Save one rollout trajectory as GIF.

    This is a visualization utility only. It does not change rollout metrics or labels.
    Returns a small status dict so callers can audit missing/failed GIFs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    maze = rollout_maze_array(rollout)
    if maze is None:
        return {"status": "SKIPPED", "reason": "rollout missing maze_grid/maze", "path": str(path)}
    trajectory = list(rollout.get("trajectory", []))
    start = _as_pair(rollout.get("start"), fallback=(0, 0))
    goal = _as_pair(rollout.get("goal"), fallback=start)
    sample_id = str(rollout.get("sample_id", rollout.get("maze_id", "unknown")))
    failure_type = str(rollout.get("failure_type", "unknown"))
    success = bool(rollout.get("success", False))

    frames: List[np.ndarray] = []
    title0 = f"sample={sample_id} start success={success} failure={failure_type}"
    frames.append(render_rollout_frame(maze, start, goal, title0))
    for st in trajectory[:max(0, int(max_frames) - 1)]:
        pos_after = st.get("pos_after", st.get("pos", start))
        title = (
            f"sample={sample_id} step={st.get('step', st.get('t', ''))} "
            f"action={st.get('action_name', st.get('action', ''))} "
            f"wall={st.get('is_wall', st.get('chosen_is_wall', ''))} "
            f"bfs={st.get('bfs_dist_after', '')}"
        )
        frames.append(render_rollout_frame(maze, pos_after, goal, title))
    if not frames:
        return {"status": "SKIPPED", "reason": "no frames", "path": str(path)}

    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.axis("off")
    im = ax.imshow(frames[0])

    def update(i):
        im.set_data(frames[i])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=int(1000 / max(1, fps)), blit=True)
    ani.save(str(path), writer="pillow")
    plt.close(fig)
    return {"status": "SAVED", "path": str(path), "n_frames": len(frames), "success": success, "failure_type": failure_type, "sample_id": sample_id}


def save_rollout_gif_examples(
    output_dir: Path,
    rollouts: Sequence[Dict[str, Any]],
    success_count: int = 2,
    failure_count: int = 2,
    fps: int = 3,
    max_frames: int = 96,
    subdir: str = "gif_examples",
) -> Dict[str, Any]:
    """Save up to N success and N failure rollout GIFs.

    The function guarantees selection intent, not artificial success/failure availability:
    if fewer than requested successes/failures exist, it saves what exists and records the shortage.
    """
    out = Path(output_dir) / subdir
    out.mkdir(parents=True, exist_ok=True)
    successes = [r for r in rollouts if bool(r.get("success", False))]
    failures = [r for r in rollouts if not bool(r.get("success", False))]
    selected: List[Tuple[str, int, Dict[str, Any]]] = []
    selected += [("success", i, r) for i, r in enumerate(successes[:max(0, int(success_count))])]
    selected += [("failure", i, r) for i, r in enumerate(failures[:max(0, int(failure_count))])]
    saved = []
    for kind, idx, ro in selected:
        sid = str(ro.get("sample_id", ro.get("maze_id", idx))).replace("/", "_").replace("\\", "_")
        ftype = str(ro.get("failure_type", "success" if kind == "success" else "failure")).replace("/", "_").replace("\\", "_")
        path = out / f"{kind}_{idx+1:02d}_{sid}_{ftype}.gif"
        saved.append(save_rollout_gif(ro, path, fps=fps, max_frames=max_frames))
    report = {
        "requested_success_gifs": int(success_count),
        "requested_failure_gifs": int(failure_count),
        "available_success_rollouts": len(successes),
        "available_failure_rollouts": len(failures),
        "saved": saved,
        "output_dir": str(out),
        "note": "GIFs are visualization artifacts only; they do not change metrics or labels.",
    }
    save_json(Path(output_dir) / "gif_examples_report.json", report)
    return report
