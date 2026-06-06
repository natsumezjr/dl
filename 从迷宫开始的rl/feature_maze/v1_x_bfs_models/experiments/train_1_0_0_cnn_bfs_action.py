#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1.0.0 — CNN BFS action supervised learning (experiment v2).

All frequently-changed experiment code lives here: model, training, CLI, presets.
Stable maze/BFS/dataset/rollout utilities are imported from v1_common.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_PATH = Path(__file__).resolve()
V1_ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(V1_ROOT / "common"))
import v1_common as vc

# =============================================================================
# experiment identity (frequently edited)
# =============================================================================

VERSION = "1.0.0"
EXPERIMENT_NAME = "cnn_bfs_action"
MODEL_NAME = "CNNBFSActionPolicy"

DEFAULTS: Dict[str, Any] = dict(
    run_name="default",
    seed=42,
    preset="default",
    input_mode="compact_2ch",
    grid_size=8,
    max_steps=64,
    train_mazes=1000,
    val_ratio=0.15,
    epochs=16,
    batch_size=64,
    lr=3e-4,
    eval_mazes=100,
    difficulty_easy=0.2,
    difficulty_medium=0.4,
    difficulty_hard=0.4,
    easy_bfs_min=6,
    easy_bfs_max=10,
    medium_bfs_min=11,
    medium_bfs_max=18,
    hard_bfs_min=19,
    hard_bfs_max=48,
    conv_channels="32,64",
    hidden_dim=128,
    dropout=0.0,
    no_tqdm=False,
    test=False,
    test_strategy="both",
    test_tie_eps=0.03,
    test_quiet_steps=False,
    manual_maze_file=None,
)

PRESETS: Dict[str, Dict[str, Any]] = {
    "quick": dict(train_mazes=50, epochs=2, eval_mazes=20, batch_size=32),
    "default": dict(train_mazes=1000, epochs=16, eval_mazes=100, batch_size=64),
    "large": dict(train_mazes=2000, epochs=24, eval_mazes=200, batch_size=64),
}


# =============================================================================
# CLI / config helpers
# =============================================================================

def parse_conv_channels(s: str) -> List[int]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if not parts:
        raise ValueError("conv_channels must not be empty")
    return [int(p) for p in parts]


def in_channels_for_mode(input_mode: str) -> int:
    if input_mode == "compact_2ch":
        return 2
    if input_mode == "standard_3ch":
        return 3
    raise ValueError(f"unknown input_mode: {input_mode}")


def nondefault_run_name(args: argparse.Namespace) -> str:
    if args.run_name != "default":
        return args.run_name
    watch = ["train_mazes", "epochs", "batch_size", "lr", "input_mode", "seed", "conv_channels"]
    parts = []
    for k in watch:
        if hasattr(args, k) and getattr(args, k) != DEFAULTS.get(k):
            v = str(getattr(args, k)).replace(".", "p").replace(",", "-")
            parts.append(f"{k}-{v}")
    return "default" if not parts else "__".join(parts)


def resolve_output_dir(args: argparse.Namespace) -> Path:
    return vc.V1_OUTPUTS_ROOT / VERSION / nondefault_run_name(args)


def standard_checkpoint_paths(out_dir: Path) -> Tuple[Path, Path]:
    tag = VERSION.replace(".", "_")
    best = out_dir / f"{tag}_{EXPERIMENT_NAME}_best.pt"
    last = out_dir / f"{tag}_{EXPERIMENT_NAME}_last.pt"
    return best, last


def resolve_checkpoint(
    args: argparse.Namespace,
    out_dir: Path,
    *,
    allow_auto: bool = False,
    required: bool = False,
) -> Optional[Path]:
    """Resolve checkpoint: explicit --checkpoint, or best/last under output_dir."""
    if args.checkpoint:
        p = Path(args.checkpoint).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"[ERROR] checkpoint not found: {p}")
        return p
    if allow_auto:
        for p in standard_checkpoint_paths(out_dir):
            if p.exists():
                return p
    if required:
        best, last = standard_checkpoint_paths(out_dir)
        raise SystemExit(
            "[ERROR] no checkpoint found for current run config.\n"
            f"  output_dir: {out_dir}\n"
            f"  tried: {best}\n"
            f"         {last}\n"
            "  Train first with the same config, or pass --checkpoint explicitly."
        )
    return None


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"{VERSION} {EXPERIMENT_NAME}")
    for k, v in DEFAULTS.items():
        name = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            p.add_argument(name, action="store_true" if not v else "store_false", default=v)
        elif isinstance(v, int):
            p.add_argument(name, type=int, default=v)
        elif isinstance(v, float):
            p.add_argument(name, type=float, default=v)
        else:
            p.add_argument(name, default=v)
    p.add_argument(
        "--checkpoint",
        default=None,
        help="checkpoint .pt path; if omitted, --test auto-loads best/last from output_dir",
    )
    return p


def apply_preset(args: argparse.Namespace, explicit: set) -> argparse.Namespace:
    if args.preset not in PRESETS:
        raise SystemExit(f"unknown --preset {args.preset}; choose {sorted(PRESETS)}")
    for k, v in PRESETS[args.preset].items():
        if ("--" + k.replace("_", "-")) not in explicit:
            setattr(args, k, v)
    args.conv_channels_list = parse_conv_channels(args.conv_channels)
    return args


def resolved_config_from_args(args: argparse.Namespace) -> dict:
    cfg = {k: getattr(args, k) for k in DEFAULTS if hasattr(args, k)}
    cfg["version"] = VERSION
    cfg["experiment_name"] = EXPERIMENT_NAME
    cfg["model_name"] = MODEL_NAME
    cfg["conv_channels_list"] = getattr(args, "conv_channels_list", parse_conv_channels(args.conv_channels))
    cfg["output_dir"] = str(resolve_output_dir(args))
    cfg["test"] = getattr(args, "test", False)
    cfg["test_strategy"] = getattr(args, "test_strategy", "both")
    return cfg


# =============================================================================
# model (frequently edited)
# =============================================================================

class CNNBFSActionPolicy(nn.Module):
    """CNN for BFS optimal action classification."""

    def __init__(
        self,
        grid_size: int,
        in_channels: int,
        conv_channels: Sequence[int],
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.model_name = MODEL_NAME
        self.grid_size = grid_size
        self.in_channels = in_channels
        self.conv_channels = list(conv_channels)
        self.hidden_dim = hidden_dim
        self.dropout_p = float(dropout)

        layers: List[nn.Module] = []
        ch_in = in_channels
        for ch_out in self.conv_channels:
            layers.extend([nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1), nn.ReLU()])
            ch_in = ch_out
        self.conv = nn.Sequential(*layers)
        flat_dim = ch_in * grid_size * grid_size
        drop = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            drop,
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))

    def summary(self) -> dict:
        return dict(
            model_name=self.model_name,
            grid_size=self.grid_size,
            in_channels=self.in_channels,
            conv_channels=self.conv_channels,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout_p,
            out_dim=4,
            n_params=sum(p.numel() for p in self.parameters()),
        )


def build_model(args) -> CNNBFSActionPolicy:
    in_ch = in_channels_for_mode(args.input_mode)
    conv_ch = getattr(args, "conv_channels_list", None) or parse_conv_channels(args.conv_channels)
    return CNNBFSActionPolicy(
        grid_size=args.grid_size,
        in_channels=in_ch,
        conv_channels=conv_ch,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )


def print_model_summary(model: CNNBFSActionPolicy) -> None:
    print("\n[Model Summary]")
    for k, v in model.summary().items():
        print(f"  {k}: {v}")


# =============================================================================
# training (frequently edited)
# =============================================================================

def _batch_best_action_hits(pred: torch.Tensor, best_actions: Sequence[Sequence[int]]) -> float:
    hits = [int(pred[i].item() in best_actions[i]) for i in range(len(best_actions))]
    return vc.mean(hits) if hits else float("nan")


def eval_epoch(model, loader, device, args) -> Tuple[float, float, float]:
    model.eval()
    losses: List[float] = []
    hard_accs: List[float] = []
    best_accs: List[float] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            pred = logits.argmax(dim=1)
            losses.append(float(loss.item()))
            hard_accs.append(float((pred == y).float().mean().item()))
            best_accs.append(_batch_best_action_hits(pred, batch["best_actions"]))
    return vc.mean(losses), vc.mean(hard_accs), vc.mean(best_accs)


def train_one_epoch(model, loader, device, args, optimizer) -> Tuple[float, float]:
    model.train()
    losses: List[float] = []
    accs: List[float] = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = logits.argmax(dim=1)
        losses.append(float(loss.item()))
        accs.append(float((pred == y).float().mean().item()))
    return vc.mean(losses), vc.mean(accs)


def train_model(
    model: CNNBFSActionPolicy,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args,
    device: torch.device,
    out_dir: Path,
) -> Tuple[CNNBFSActionPolicy, dict]:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    hist: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_best_action_acc": [],
    }
    best_val_loss = 1e9
    best_path = out_dir / f"{VERSION.replace('.', '_')}_{EXPERIMENT_NAME}_best.pt"
    last_path = out_dir / f"{VERSION.replace('.', '_')}_{EXPERIMENT_NAME}_last.pt"

    for ep in range(1, args.epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, device, args, optimizer)
        vl, va, vba = eval_epoch(model, val_loader, device, args)
        hist["train_loss"].append(tl)
        hist["train_acc"].append(ta)
        hist["val_loss"].append(vl)
        hist["val_acc"].append(va)
        hist["val_best_action_acc"].append(vba)
        print(
            f"[Epoch {ep}/{args.epochs}] "
            f"train_loss={tl:.4f} val_loss={vl:.4f} "
            f"train_acc={ta:.4f} val_acc={va:.4f} val_best_action_acc={vba:.4f}"
        )
        if vl < best_val_loss:
            best_val_loss = vl
            torch.save(model.state_dict(), best_path)
    torch.save(model.state_dict(), last_path)
    hist["best_val_loss"] = best_val_loss
    hist["checkpoint_best"] = str(best_path)
    hist["checkpoint_last"] = str(last_path)
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    return model, hist


# =============================================================================
# main
# =============================================================================

def main() -> None:
    explicit = {a.split("=")[0] for a in sys.argv[1:]}
    args = build_argparser().parse_args()
    args = apply_preset(args, explicit)

    if args.input_mode not in vc.INPUT_MODES:
        raise SystemExit(f"unknown input_mode {args.input_mode}; choose {vc.INPUT_MODES}")

    vc.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = resolve_output_dir(args)
    resolved_config = resolved_config_from_args(args)

    audit = vc.protocol_compliance_audit(
        script_path=SCRIPT_PATH,
        version=VERSION,
        output_dir=out_dir,
        resolved_config=resolved_config,
        required_reports=vc.REQUIRED_REPORTS,
        loaded_modules=list(sys.modules.keys()),
    )
    vc.print_protocol_compliance_audit(audit)
    if audit["status"] != "PASS":
        raise SystemExit("[ERROR] Protocol compliance audit failed; aborting.")

    vc.ensure_dir(out_dir)
    vc.save_json(out_dir / "resolved_config.json", resolved_config)

    if args.test:
        ckpt = resolve_checkpoint(args, out_dir, allow_auto=True, required=True)
        model = build_model(args).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print_model_summary(model)
        print(f"\n[Manual Test Mode] output_dir={out_dir}")
        print(f"[Manual Test Mode] checkpoint={ckpt}")
        test_results = vc.run_manual_test_mode(
            model,
            args,
            device,
            out_dir,
            version_tag=VERSION.replace(".", "_"),
        )
        vc.save_json(
            out_dir / "run_metadata.json",
            dict(
                version=VERSION,
                mode="test",
                script=str(SCRIPT_PATH),
                output_dir=str(out_dir),
                checkpoint=str(ckpt),
                model_summary=model.summary(),
                test_results=test_results,
            ),
        )
        print(f"\n[Done] test artifacts saved to: {out_dir}")
        return

    print("\n==== CONFIGURATION SUMMARY ====")
    vc.print_table(
        "[Effective Configuration]",
        [("key", 28, "s"), ("value", 60, "s")],
        [
            {"key": "version", "value": VERSION},
            {"key": "experiment", "value": EXPERIMENT_NAME},
            {"key": "model_name", "value": MODEL_NAME},
            {"key": "input_mode", "value": args.input_mode},
            {"key": "conv_channels", "value": args.conv_channels},
            {"key": "hidden_dim", "value": args.hidden_dim},
            {"key": "dropout", "value": args.dropout},
            {"key": "output_dir", "value": str(out_dir)},
            {"key": "preset", "value": args.preset},
            {"key": "device", "value": str(device)},
            {"key": "seed", "value": args.seed},
            {"key": "train_mazes", "value": args.train_mazes},
            {"key": "epochs", "value": args.epochs},
            {"key": "batch_size", "value": args.batch_size},
            {"key": "lr", "value": args.lr},
            {"key": "eval_mazes", "value": args.eval_mazes},
        ],
    )

    print("\n[Build Dataset]")
    all_samples, maze_records = vc.build_bfs_supervised_dataset(args, args.train_mazes)
    train_samples, val_samples = vc.split_samples(all_samples, args.val_ratio)
    maze_gen_report = vc.summarize_maze_generation(args, maze_records)
    print(f"  total_samples={len(all_samples)} train={len(train_samples)} val={len(val_samples)}")
    vc.print_maze_generation_report(maze_gen_report)

    train_loader = DataLoader(
        vc.BFSActionDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=vc.collate_batch,
    )
    val_loader = DataLoader(
        vc.BFSActionDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=vc.collate_batch,
    )

    model = build_model(args).to(device)
    print_model_summary(model)

    ckpt = resolve_checkpoint(args, out_dir) if args.checkpoint else None
    if ckpt:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"[INFO] loaded checkpoint: {ckpt}")

    print("\n[Train]")
    model, training_history = train_model(model, train_loader, val_loader, args, device, out_dir)

    print("\n[Evaluate]")
    val_metrics = vc.evaluate_on_samples(model, val_samples, device, args)
    rollouts, rollout_summary = vc.run_rollout_evaluation(model, args, device, args.eval_mazes)
    evaluation_summary = vc.build_evaluation_summary(args, val_metrics, rollout_summary, maze_records)
    vc.print_evaluation_tables(evaluation_summary, rollout_summary)

    run_metadata = dict(
        version=VERSION,
        experiment_name=EXPERIMENT_NAME,
        script=str(SCRIPT_PATH),
        model_name=MODEL_NAME,
        model_summary=model.summary(),
        started_at=datetime.now(timezone.utc).isoformat(),
        device=str(device),
        n_train_samples=len(train_samples),
        n_val_samples=len(val_samples),
        n_rollout_mazes=args.eval_mazes,
        maze_generation_report=maze_gen_report,
        checkpoint_best=training_history.get("checkpoint_best"),
        checkpoint_last=training_history.get("checkpoint_last"),
    )
    rollout_debug = dict(
        version=VERSION,
        n_rollouts=len(rollouts),
        rollouts=rollouts,
        summary=rollout_summary,
    )

    vc.save_required_reports(
        out_dir,
        resolved_config=resolved_config,
        run_metadata=run_metadata,
        training_history=training_history,
        evaluation_summary=evaluation_summary,
        rollout_debug=rollout_debug,
        protocol_compliance_audit_report=audit,
    )

    print(f"\n[Done] artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
