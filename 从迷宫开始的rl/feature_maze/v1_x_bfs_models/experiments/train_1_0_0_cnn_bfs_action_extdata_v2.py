#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1.0.0 external-data replacement test for CNN BFS action classifier.

MVP scope:
- Replace old generator dataset with read-only 3.0.4 exported samples.
- Recompute BFS labels using current v1_common BFS logic.
- Keep model architecture identical to train_1_0_0_cnn_bfs_action.py.
- Produce auditable conversion, label, rollout, baseline and protocol reports.
"""
from __future__ import annotations
import argparse
import copy
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_PATH = Path(__file__).resolve()
V1_ROOT = SCRIPT_PATH.parents[1]
LEARNING_ROOT = V1_ROOT.parent.parent.parent  # .../从迷宫开始的rl/feature_maze/v1_x_bfs_models -> .../learning
sys.path.insert(0, str(V1_ROOT / "common"))
import v1_common as vc
import external_dataset_3_0_4_v1 as ext

VERSION = "1.0.0_data_swap_3_0_4_extdata_v2"
EXPERIMENT_NAME = "cnn_bfs_action_extdata"
MODEL_NAME = "CNNBFSActionPolicy"
OLD_BASELINE_PATH = V1_ROOT / "outputs" / "1.0.0" / "default"

DEFAULTS: Dict[str, Any] = dict(
    run_name="default",
    seed=42,
    preset="default",
    input_mode="compact_2ch",
    grid_size=8,
    max_steps=64,
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
    external_dataset_path=str(LEARNING_ROOT / "从迷宫开始的rl/feature_maze/v3_0_maze_generation/outputs/3.0.4/tda_qd_final_handdraw_quota_seed43/confirmed_samples.jsonl"),
    external_dataset_format="auto",
    split_manifest_path=None,
    max_external_samples=10000,
    dry_run_dataset_only=False,
    baseline_old_generator_eval=False,
    mode="train",
    checkpoint_path=None,
    manual_maze_file=None,
    manual_maze_text=None,
    split="test",
    sample_policy="sequential",
    difficulty_filter="all",
    bucket_filter="all",
    max_test_samples=100,
    test_seed=42,
    save_rollout_traces=False,
    save_failure_examples=False,
    save_gif_examples=True,
    gif_success_examples=2,
    gif_failure_examples=2,
    gif_fps=3,
    gif_max_frames=96,
    auto_final_evaluation=True,
    handdraw_dataset_path=str(LEARNING_ROOT / "从迷宫开始的rl/feature_maze/handdraw_mazes_ascii.jsonl"),
    sample_check_dataset_path=str(LEARNING_ROOT / "从迷宫开始的rl/feature_maze/v3_0_maze_generation/outputs/3.0.4/tda_qd_final_handdraw_quota_seed43/confirmed_samples.jsonl"),
    sample_check_run_name="default_sample_check_extdata_v2",
    handdraw_check_run_name="default_handdraw_check_extdata_v2",
    sample_check_max_test_samples=100,
    handdraw_check_max_test_samples=10000,
    gif_per_difficulty_success=5,
    gif_per_difficulty_failure=5,
    gif_all_examples=False,
)
PRESETS = {
    "quick": dict(epochs=2, batch_size=64, eval_mazes=100),
    "default": dict(epochs=16, batch_size=64, eval_mazes=200),
}

def parse_conv_channels(s: str) -> List[int]:
    return [int(p.strip()) for p in str(s).split(",") if p.strip()]

def in_channels_for_mode(input_mode: str) -> int:
    if input_mode == "compact_2ch": return 2
    if input_mode == "standard_3ch": return 3
    raise ValueError(f"unknown input_mode: {input_mode}")

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=VERSION)
    for k, v in DEFAULTS.items():
        opt1 = "--" + k.replace("_", "-")
        opt2 = "--" + k
        opts = [opt1] if opt1 == opt2 else [opt1, opt2]
        if isinstance(v, bool):
            # Important: bool options with default=True must not be flipped off
            # when the user passes the positive flag.  Previous generic logic used
            # store_false for default=True, so passing --save_gif_examples made
            # args.save_gif_examples False and silently skipped GIF generation.
            p.add_argument(*opts, dest=k, action="store_true", default=v)
            if v is True:
                no_opt1 = "--no-" + k.replace("_", "-")
                no_opt2 = "--no_" + k
                no_opts = [no_opt1] if no_opt1 == no_opt2 else [no_opt1, no_opt2]
                p.add_argument(*no_opts, dest=k, action="store_false")
        elif isinstance(v, int): p.add_argument(*opts, dest=k, type=int, default=v)
        elif isinstance(v, float): p.add_argument(*opts, dest=k, type=float, default=v)
        else: p.add_argument(*opts, dest=k, default=v)
    return p

def apply_preset(args, explicit: set):
    if args.preset not in PRESETS:
        raise SystemExit(f"unknown --preset {args.preset}; choose {sorted(PRESETS)}")
    for k, v in PRESETS[args.preset].items():
        if "--" + k.replace("_", "-") not in explicit:
            setattr(args, k, v)
    args.conv_channels_list = parse_conv_channels(args.conv_channels)
    if args.max_external_samples is not None:
        args.max_external_samples = int(args.max_external_samples)
    return args

def resolve_output_dir(args) -> Path:
    rn = args.run_name if args.run_name != "default" else "default_extdata_v2"
    return V1_ROOT / "outputs" / "1.0.0_data_swap_3_0_4" / rn

class CNNBFSActionPolicy(nn.Module):
    def __init__(self, grid_size: int, in_channels: int, conv_channels: Sequence[int], hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.model_name = MODEL_NAME
        layers: List[nn.Module] = []
        ch_in = in_channels
        for ch_out in conv_channels:
            layers.extend([nn.Conv2d(ch_in, ch_out, 3, padding=1), nn.ReLU()])
            ch_in = ch_out
        self.conv = nn.Sequential(*layers)
        drop = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(ch_in * grid_size * grid_size, hidden_dim), nn.ReLU(), drop, nn.Linear(hidden_dim, 4))
        self._summary = dict(model_name=MODEL_NAME, grid_size=grid_size, in_channels=in_channels, conv_channels=list(conv_channels), hidden_dim=hidden_dim, dropout=float(dropout), out_dim=4, n_params=sum(p.numel() for p in self.parameters()))
    def forward(self, x): return self.head(self.conv(x))
    def summary(self): return dict(self._summary)

def build_model(args):
    return CNNBFSActionPolicy(args.grid_size, in_channels_for_mode(args.input_mode), args.conv_channels_list, args.hidden_dim, args.dropout)

def external_records_to_bfs_samples(records: Sequence[Dict[str, Any]], args) -> Tuple[List[vc.BFSSample], List[dict], dict]:
    samples: List[vc.BFSSample] = []
    maze_records: List[dict] = []
    skipped: List[dict] = []
    unsolvable: List[dict] = []
    for mid, rec in enumerate(records):
        maze = rec["maze"]
        start = tuple(rec["start"]); goal = tuple(rec["goal"])
        try:
            dist = vc.bfs_distances(maze, goal)
            path = vc.shortest_path(maze, start, goal, dist)
            if not path:
                unsolvable.append({"sample_id": rec["sample_id"], "row_index": rec.get("row_index"), "reason": "unsolvable"})
                continue
            bfs_len = len(path) - 1
            meta = dict(rec.get("metadata", {}), maze_id=mid, sample_id=rec["sample_id"], row_index=rec.get("row_index"), start=start, goal=goal, bfs_len=bfs_len, actual=rec.get("bfs_len_bin_source") or "external", target_bucket=rec.get("target_bucket"), source=rec.get("source"), split=rec.get("split"))
            maze_records.append(meta)
            ss = vc.samples_from_shortest_path(maze, start, goal, dist, path, meta, args.input_mode, maze_id=mid)
            for s in ss:
                # Attach external metadata dynamically for breakdown; BFSSample is dataclass but mutable.
                s.target_bucket = rec.get("target_bucket")
                s.source = rec.get("source")
                s.split = rec.get("split")
                s.sample_id = rec.get("sample_id")
            samples.extend(ss)
        except Exception as e:
            skipped.append({"sample_id": rec.get("sample_id"), "row_index": rec.get("row_index"), "reason": str(e)})
    report = {
        "input_external_mazes": len(records),
        "labelable_mazes": len(maze_records),
        "generated_action_samples": len(samples),
        "unsolvable_mazes": len(unsolvable),
        "skipped_mazes": len(skipped),
        "unsolvable_examples": unsolvable[:20],
        "skipped_examples": skipped[:20],
        "label_source": "v1_common.bfs_distances + v1_common.shortest_path + v1_common.bfs_best_actions",
        "no_maze_repair": True,
        "no_default_start_goal": True,
    }
    return samples, maze_records, report

def split_samples_by_maze(records, samples, split_manifest_path=None):
    if split_manifest_path:
        mapping = json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
    else:
        mapping = {r["sample_id"]: r.get("split") for r in records}
    train, val, test = [], [], []
    for s in samples:
        sp = mapping.get(getattr(s, "sample_id", ""), getattr(s, "split", None)) or "train"
        if sp in {"val", "valid", "validation"}: val.append(s)
        elif sp == "test": test.append(s)
        else: train.append(s)
    if not val and train:
        # This is not a sample skip; it is deterministic split already created by external common.
        val = train[:max(1, int(0.15 * len(train)))]
        train = train[len(val):] or val
    return train, val, test or val

def _batch_best_action_hits(pred, best_actions):
    hits = [int(pred[i].item() in best_actions[i]) for i in range(len(best_actions))]
    return vc.mean(hits) if hits else float("nan")

def train_one_epoch(model, loader, device, optimizer):
    model.train(); losses=[]; acc=[]
    for batch in loader:
        x=batch["x"].to(device); y=batch["y"].to(device)
        logits=model(x); loss=F.cross_entropy(logits,y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        pred=logits.argmax(1); losses.append(float(loss.item())); acc.append(float((pred==y).float().mean().item()))
    return vc.mean(losses), vc.mean(acc)

def eval_epoch(model, loader, device):
    model.eval(); losses=[]; hard=[]; best=[]
    with torch.no_grad():
        for batch in loader:
            x=batch["x"].to(device); y=batch["y"].to(device)
            logits=model(x); loss=F.cross_entropy(logits,y); pred=logits.argmax(1)
            losses.append(float(loss.item())); hard.append(float((pred==y).float().mean().item())); best.append(_batch_best_action_hits(pred,batch["best_actions"]))
    return vc.mean(losses), vc.mean(hard), vc.mean(best)

def train_model(model, train_loader, val_loader, args, device, out_dir):
    opt=torch.optim.Adam(model.parameters(), lr=args.lr)
    hist={"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[],"val_best_action_acc":[]}
    best_loss=1e9; best_path=out_dir/"1_0_0_extdata_v2_cnn_bfs_action_best.pt"; last_path=out_dir/"1_0_0_extdata_v2_cnn_bfs_action_last.pt"
    for ep in range(1,args.epochs+1):
        tl,ta=train_one_epoch(model,train_loader,device,opt); vl,va,vba=eval_epoch(model,val_loader,device)
        hist["train_loss"].append(tl); hist["train_acc"].append(ta); hist["val_loss"].append(vl); hist["val_acc"].append(va); hist["val_best_action_acc"].append(vba)
        print(f"[Epoch {ep}/{args.epochs}] train_loss={tl:.4f} val_loss={vl:.4f} train_acc={ta:.4f} val_acc={va:.4f} val_best_action_acc={vba:.4f}")
        if vl < best_loss:
            best_loss=vl; torch.save(model.state_dict(), best_path)
    torch.save(model.state_dict(), last_path)
    hist.update(best_val_loss=best_loss, checkpoint_best=str(best_path), checkpoint_last=str(last_path))
    if best_path.exists(): model.load_state_dict(torch.load(best_path, map_location=device))
    return model, hist

def rollout_external(model, records, args, device):
    rollouts=[]
    for mid, rec in enumerate(records[:args.eval_mazes]):
        dist=vc.bfs_distances(rec["maze"], tuple(rec["goal"]))
        ro=vc.rollout_policy(model, rec["maze"], tuple(rec["start"]), tuple(rec["goal"]), dist, args, device, args.input_mode)
        ro.update(maze_id=mid, sample_id=rec["sample_id"], target_bucket=rec.get("target_bucket"), source=rec.get("source"), split=rec.get("split"), metadata=rec.get("metadata", {}))
        rollouts.append(ro)
    return rollouts, vc.summarize_rollouts(rollouts)

def breakdown_samples(samples, model, device, args):
    base = vc.evaluate_on_samples(model, samples, device, args)
    dims = {"split": defaultdict(list), "bfs_len_bin": defaultdict(list), "target_bucket": defaultdict(list), "source": defaultdict(list)}
    for s in samples:
        dims["split"][getattr(s,"split","unavailable") or "unavailable"].append(s)
        dims["bfs_len_bin"][ext.make_bfs_len_bin(getattr(s,"bfs_len",None))].append(s)
        dims["target_bucket"][getattr(s,"target_bucket","unavailable") or "unavailable"].append(s)
        dims["source"][getattr(s,"source","unavailable") or "unavailable"].append(s)
    bd={}
    for dim, groups in dims.items():
        bd[dim]={k: vc.evaluate_on_samples(model,v,device,args) for k,v in groups.items()}
    base["breakdown"] = bd
    return base

def write_baseline_comparison(out_dir: Path):
    comp={"old_baseline_path": str(OLD_BASELINE_PATH), "new_data_run_path": str(out_dir), "comparable_metrics": ["action_accuracy_hard_label","action_accuracy_best_action_set","valid_action_rate","wall_action_rate","rollout_success_rate","shortest_path_success_rate","avg_bfs_gap","avg_wall_hits","avg_steps"], "non_comparable_metrics": [], "non_comparable_reasons": {}, "note": "This file records paths and metric names only. It does not claim improvement without an independent review."}
    if not OLD_BASELINE_PATH.exists():
        comp["non_comparable_metrics"].append("old baseline values")
        comp["non_comparable_reasons"]["old baseline values"] = "baseline path not found in this runtime"
    ext.save_json(out_dir/"baseline_comparison.json", comp)

def protocol_audit(args, out_dir: Path):
    issues=[]
    if not str(SCRIPT_PATH).replace('\\','/').endswith('train_1_0_0_cnn_bfs_action_extdata_v2.py'):
        issues.append({"scope":"script_name","status":"FAIL","message":"script must use extdata_v2 name"})
    if "v1_x_bfs_models" not in str(SCRIPT_PATH).replace('\\','/'):
        issues.append({"scope":"script_path","status":"FAIL","message":"script must live under v1_x_bfs_models"})
    if "1.0.0_data_swap_3_0_4" not in str(out_dir):
        issues.append({"scope":"output_dir","status":"FAIL","message":"output must be under new data_swap directory"})
    if args.mode != "test" and not args.external_dataset_path:
        issues.append({"scope":"external_dataset_path","status":"FAIL","message":"external dataset path is required for train/dry-run"})
    if args.mode == "test" and not (args.external_dataset_path or args.manual_maze_file or args.manual_maze_text):
        issues.append({"scope":"test_input","status":"FAIL","message":"test mode requires external dataset or manual maze"})
    return {"status":"PASS" if not issues else "FAIL", "issues":issues, "required_reports":["dataset_schema_report.json","dataset_source_manifest.json","conversion_report.json","split_manifest.json","bfs_label_generation_report.json","training_history.json","evaluation_summary.json","rollout_summary.json","rollout_debug.json","failure_examples.jsonl","baseline_comparison.json","protocol_compliance_audit.json"]}

def write_failure_examples(out_dir: Path, rollouts):
    with (out_dir/"failure_examples.jsonl").open("w", encoding="utf-8") as f:
        for ro in rollouts:
            if not ro.get("success"):
                f.write(json.dumps({k:v for k,v in ro.items() if k != "trajectory"} | {"trajectory_tail": ro.get("trajectory", [])[-8:]}, ensure_ascii=False, default=str) + "\n")


def _difficulty_from_bfs_len(args, bfs_len: int) -> str:
    if args.easy_bfs_min <= bfs_len <= args.easy_bfs_max:
        return "easy"
    if args.medium_bfs_min <= bfs_len <= args.medium_bfs_max:
        return "medium"
    if args.hard_bfs_min <= bfs_len <= args.hard_bfs_max:
        return "hard"
    return "out_of_range"

def _load_manual_text(args) -> str:
    if args.manual_maze_text:
        return str(args.manual_maze_text)
    if args.manual_maze_file:
        return Path(args.manual_maze_file).read_text(encoding="utf-8")
    raise ValueError("manual maze input is missing")

def _checkpoint_or_block(args, out_dir: Path) -> Path:
    if not args.checkpoint_path:
        bp = out_dir / "blocking_report.json"
        ext.save_json(bp, {"found":"missing checkpoint_path", "affected_goal":"v1 test cannot load trained model", "risk_if_continue":"test would use untrained/random weights", "needs_upper_confirmation":"provide checkpoint_path", "suggested_next_agent":"code generation agent"})
        raise SystemExit(f"[BLOCKED] missing --checkpoint_path; see {bp}")
    p = Path(args.checkpoint_path)
    if not p.exists():
        bp = out_dir / "blocking_report.json"
        ext.save_json(bp, {"found":f"checkpoint not found: {p}", "affected_goal":"v1 test cannot load trained model", "risk_if_continue":"would silently test wrong model", "needs_upper_confirmation":"correct checkpoint path", "suggested_next_agent":"code generation agent"})
        raise SystemExit(f"[BLOCKED] checkpoint not found; see {bp}")
    return p

def _make_sampling_contract(args, schema, manifest, conv, records, selected, mode: str, diagnostic_only: bool = False):
    split_report = conv.get("split_report", {}) if isinstance(conv, dict) else {}
    split_counts = split_report.get("split_counts", {})
    return {
        "report_name": "sampling_contract_report",
        "diagnostic_only": True,
        "mode": mode,
        "data_source_contract": {
            "dataset_path": args.external_dataset_path,
            "external_dataset_format": args.external_dataset_format,
            "actual_format": manifest.get("external_dataset_format"),
            "maze_field": schema.get("selected_maze_field"),
            "start_fields": schema.get("selected_start_pair") or schema.get("selected_start_single"),
            "goal_fields": schema.get("selected_goal_pair") or schema.get("selected_goal_single"),
            "bfs_len_read": "bfs_len" in schema.get("fields", []),
            "bucket_field": schema.get("selected_bucket_field"),
            "candidate_bucket_fields": schema.get("candidate_bucket_fields"),
            "split_field": schema.get("selected_split_field"),
            "source_field": schema.get("selected_source_field"),
            "original_data_modified": False,
            "maze_repair": False,
            "default_start_goal": False,
            "fallback_corridor": False,
            "v3_code_imported": False,
        },
        "split_contract": {
            "mode": split_report.get("mode"),
            "hash_key": split_report.get("hash_key"),
            "seed": split_report.get("seed"),
            "ratios": split_report.get("ratios"),
            "split_counts": split_counts,
            "requested_split": args.split,
            "train_split_test_is_diagnostic_only": bool(args.split == "train"),
        },
        "difficulty_contract": {
            "primary_fact_source": "recomputed_bfs_shortest_path_length_after_import",
            "external_bfs_len_is_auxiliary_only": True,
            "difficulty_filter": args.difficulty_filter,
            "thresholds": {"easy":[args.easy_bfs_min,args.easy_bfs_max], "medium":[args.medium_bfs_min,args.medium_bfs_max], "hard":[args.hard_bfs_min,args.hard_bfs_max]},
            "easy_medium_hard_not_3_0_4_bucket": True,
        },
        "sample_policy_contract": {
            "sample_policy": args.sample_policy,
            "seed": args.test_seed,
            "bucket_filter": args.bucket_filter,
            "max_test_samples": args.max_test_samples,
            "selected_sample_ids": [r.get("sample_id") for r in selected],
        },
        "v2_previous_adapter_note": "not applicable to v1; v2 extdata_v1 shuffled records then cycled provider calls. This v1 test selects records explicitly by split/filter.",
        "counts": {"candidate_records": len(records), "selected_records": len(selected)},
    }

def _select_records_for_test(records, args):
    candidates=[]; skipped=[]
    for r in records:
        split_ok = (args.split == "all" or r.get("split") == args.split)
        if not split_ok:
            continue
        try:
            dist=vc.bfs_distances(r["maze"], tuple(r["goal"]))
            path=vc.shortest_path(r["maze"], tuple(r["start"]), tuple(r["goal"]), dist)
            if not path:
                skipped.append({"sample_id":r.get("sample_id"),"reason":"unsolvable"}); continue
            bfs_len=len(path)-1; diff=_difficulty_from_bfs_len(args,bfs_len)
            r["recomputed_bfs_len"]=bfs_len; r["recomputed_difficulty"]=diff
        except Exception as e:
            skipped.append({"sample_id":r.get("sample_id"),"reason":str(e)}); continue
        if args.difficulty_filter != "all" and r.get("recomputed_difficulty") != args.difficulty_filter:
            continue
        if args.bucket_filter != "all" and str(r.get("target_bucket")) != str(args.bucket_filter):
            continue
        candidates.append(r)
    if args.sample_policy == "random_seeded":
        import random
        rng=random.Random(args.test_seed); rng.shuffle(candidates)
    elif args.sample_policy == "stratified_by_difficulty":
        groups=defaultdict(list)
        for r in candidates: groups[r.get("recomputed_difficulty","unknown")].append(r)
        ordered=[]
        while any(groups.values()):
            for k in sorted(groups):
                if groups[k]: ordered.append(groups[k].pop(0))
        candidates=ordered
    elif args.sample_policy == "stratified_by_bucket":
        groups=defaultdict(list)
        for r in candidates: groups[str(r.get("target_bucket","unavailable"))].append(r)
        ordered=[]
        while any(groups.values()):
            for k in sorted(groups):
                if groups[k]: ordered.append(groups[k].pop(0))
        candidates=ordered
    elif args.sample_policy != "sequential":
        raise SystemExit(f"unsupported sample_policy: {args.sample_policy}")
    selected=candidates[:int(args.max_test_samples)]
    return selected, {"num_candidate_records": len(records), "num_filter_pass_records": len(candidates), "num_selected_records": len(selected), "num_skipped_records": len(skipped), "skip_reasons": ext.summarize_reasons(skipped), "skipped_examples": skipped[:20], "selected_records": [{"sample_id":r.get("sample_id"),"split":r.get("split"),"difficulty":r.get("recomputed_difficulty"),"bfs_len":r.get("recomputed_bfs_len"),"target_bucket":r.get("target_bucket"),"source":r.get("source")} for r in selected]}

def _rollout_records_test(model, records, args, device):
    rollouts=[]
    for i,r in enumerate(records):
        dist=vc.bfs_distances(r["maze"], tuple(r["goal"]))
        ro=vc.rollout_policy(model, r["maze"], tuple(r["start"]), tuple(r["goal"]), dist, args, device, args.input_mode)
        ro.update(sample_id=r.get("sample_id"), row_index=r.get("row_index"), split=r.get("split"), target_bucket=r.get("target_bucket"), source=r.get("source"), recomputed_difficulty=r.get("recomputed_difficulty"), recomputed_bfs_len=r.get("recomputed_bfs_len"), maze_grid=r["maze"].tolist())
        rollouts.append(ro)
    return rollouts

def _summarize_test_rollouts(rollouts):
    if not rollouts:
        return {"num_tested_records":0,"rollout_success_rate":float("nan"),"shortest_path_success_rate":float("nan"),"avg_bfs_gap":float("nan"),"median_bfs_gap":float("nan"),"wall_action_rate":float("nan"),"avg_wall_hits":float("nan"),"loop_timeout_count":0,"failure_type_breakdown":{}}
    total_steps=sum(len(r.get("trajectory",[])) for r in rollouts)
    wall_steps=sum(1 for r in rollouts for st in r.get("trajectory",[]) if st.get("is_wall"))
    fail=defaultdict(int)
    for r in rollouts: fail[r.get("failure_type","unknown")]+=1
    return {"num_tested_records":len(rollouts),"rollout_success_rate":vc.mean([float(r.get("success")) for r in rollouts]),"shortest_path_success_rate":vc.mean([float(r.get("shortest_path_success")) for r in rollouts]),"avg_bfs_gap":vc.mean([float(r.get("bfs_gap")) for r in rollouts]),"median_bfs_gap":float(np.median([float(r.get("bfs_gap",0)) for r in rollouts])),"wall_action_rate":wall_steps/max(1,total_steps),"avg_wall_hits":vc.mean([float(r.get("wall_hits",0)) for r in rollouts]),"avg_steps":vc.mean([float(r.get("steps",0)) for r in rollouts]),"loop_timeout_count":sum(1 for r in rollouts if r.get("failure_type")=="loop_timeout"),"failure_type_breakdown":dict(fail)}

def _breakdown_rollouts(rollouts):
    out={}
    for dim in ["recomputed_difficulty","split","target_bucket"]:
        groups=defaultdict(list)
        for r in rollouts: groups[str(r.get(dim,"unavailable"))].append(r)
        out[dim]={k:_summarize_test_rollouts(v) for k,v in groups.items()}
    return out

def _write_jsonl(path: Path, rows):
    with path.open("w",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r,ensure_ascii=False,default=str)+"\n")


def _safe_name(x: Any) -> str:
    return str(x).replace("/", "_").replace("\\", "_").replace(" ", "_")[:120]


def _local_maze_array(rollout: Dict[str, Any]):
    maze = rollout.get("maze_grid", rollout.get("maze"))
    if maze is None:
        return None
    arr = np.array(maze, dtype=np.int32)
    return arr


def _local_pair(x, fallback=(0, 0)):
    if x is None:
        return tuple(fallback)
    try:
        return (int(x[0]), int(x[1]))
    except Exception:
        return tuple(fallback)


def _local_render_frame(maze, agent, goal, title=""):
    arr = np.zeros((maze.shape[0], maze.shape[1], 3), dtype=np.uint8)
    arr[maze == 1] = np.array([30, 30, 30], dtype=np.uint8)
    arr[maze == 0] = np.array([245, 245, 245], dtype=np.uint8)
    gr, gc = _local_pair(goal)
    ar, ac = _local_pair(agent)
    if 0 <= gr < maze.shape[0] and 0 <= gc < maze.shape[1]:
        arr[gr, gc] = np.array([60, 180, 75], dtype=np.uint8)
    if 0 <= ar < maze.shape[0] and 0 <= ac < maze.shape[1]:
        arr[ar, ac] = np.array([220, 50, 47], dtype=np.uint8)
    return arr


def _local_save_rollout_gif(rollout: Dict[str, Any], path: Path, fps: int = 3, max_frames: int = 96) -> Dict[str, Any]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import animation
    except Exception as e:
        return {"status": "ERROR", "reason": "matplotlib/pillow animation dependency unavailable", "error": str(e), "path": str(path)}
    maze = _local_maze_array(rollout)
    if maze is None:
        return {"status": "SKIPPED", "reason": "rollout missing maze_grid/maze", "path": str(path)}
    start = _local_pair(rollout.get("start"), fallback=(0, 0))
    goal = _local_pair(rollout.get("goal"), fallback=start)
    trajectory = list(rollout.get("trajectory", []))
    frames = [_local_render_frame(maze, start, goal)]
    for st in trajectory[:max(0, int(max_frames) - 1)]:
        pos = st.get("pos_after", st.get("pos", st.get("agent", start)))
        frames.append(_local_render_frame(maze, _local_pair(pos, fallback=start), goal))
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(5, 5)); ax = plt.gca(); ax.axis("off"); im = ax.imshow(frames[0])
    def update(i):
        im.set_data(frames[i]); return [im]
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=int(1000 / max(1, fps)), blit=True)
    try:
        ani.save(str(path), writer="pillow")
        status = {"status": "SAVED", "path": str(path), "n_frames": len(frames), "success": bool(rollout.get("success", False)), "failure_type": rollout.get("failure_type")}
    except Exception as e:
        status = {"status": "ERROR", "path": str(path), "error": str(e)}
    plt.close(fig)
    return status


def _unique_rollouts(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for i, r in enumerate(rows):
        key = (r.get("sample_id"), r.get("row_index"), r.get("strategy"), r.get("failure_type"), i)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _select_balanced_gif_rollouts(rollouts: Sequence[Dict[str, Any]], per_success: int, per_failure: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select GIF examples by difficulty.

    Target for random/external checks: for each easy/medium/hard, save N success and N fail.
    If failures are insufficient, fill with successes; if a difficulty has too few examples,
    use all available examples in that difficulty and backfill from other difficulties.
    """
    all_rows = list(rollouts)
    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    report = {"policy": "balanced_by_difficulty_success_failure_with_backfill", "per_difficulty_success_target": int(per_success), "per_difficulty_failure_target": int(per_failure), "difficulty": {}}

    def add_rows(rows):
        for r in rows:
            rid = id(r)
            if rid not in selected_ids:
                selected_ids.add(rid)
                selected.append(r)

    for diff in ["easy", "medium", "hard"]:
        group = [r for r in all_rows if str(r.get("recomputed_difficulty", r.get("difficulty", "unavailable"))) == diff]
        succ = [r for r in group if bool(r.get("success", False))]
        fail = [r for r in group if not bool(r.get("success", False))]
        target_total = int(per_success) + int(per_failure)
        chosen = []
        chosen += succ[:int(per_success)]
        chosen += fail[:int(per_failure)]
        if len(chosen) < target_total:
            # First fill fail shortage with extra successes, then success shortage with extra failures.
            extras = [r for r in succ[int(per_success):] + fail[int(per_failure):] if r not in chosen]
            chosen += extras[: max(0, target_total - len(chosen))]
        add_rows(chosen)
        report["difficulty"][diff] = {"available_total": len(group), "available_success": len(succ), "available_failure": len(fail), "selected": len(chosen), "selected_success": sum(1 for r in chosen if r.get("success")), "selected_failure": sum(1 for r in chosen if not r.get("success"))}

    requested_total = 3 * (int(per_success) + int(per_failure))
    if len(selected) < min(requested_total, len(all_rows)):
        extras = [r for r in all_rows if id(r) not in selected_ids]
        add_rows(extras[: max(0, min(requested_total, len(all_rows)) - len(selected))])
        report["backfilled_from_other_examples"] = True
    else:
        report["backfilled_from_other_examples"] = False
    report["selected_total"] = len(selected)
    report["available_total"] = len(all_rows)
    return selected, report


def _save_selected_gifs(out_dir: Path, selected: Sequence[Dict[str, Any]], fps: int, max_frames: int, subdir: str = "gif_examples") -> Dict[str, Any]:
    gif_dir = Path(out_dir) / subdir
    gif_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, ro in enumerate(selected, 1):
        kind = "success" if ro.get("success") else "failure"
        diff = _safe_name(ro.get("recomputed_difficulty", ro.get("difficulty", "unknown")))
        sid = _safe_name(ro.get("sample_id", ro.get("maze_id", i)))
        ftype = _safe_name(ro.get("failure_type", kind))
        path = gif_dir / f"{i:03d}_{diff}_{kind}_{sid}_{ftype}.gif"
        try:
            save_fn = getattr(ext, "save_rollout_gif", _local_save_rollout_gif)
            saved.append(save_fn(ro, path, fps=fps, max_frames=max_frames))
        except Exception as e:
            saved.append({"status": "ERROR", "path": str(path), "error": str(e), "sample_id": sid})
    return {"output_dir": str(gif_dir), "saved": saved, "n_saved": sum(1 for x in saved if isinstance(x, dict) and x.get("status") == "SAVED")}


def _save_policy_gifs(out_dir: Path, rollouts: Sequence[Dict[str, Any]], args, *, handdraw: bool = False) -> Dict[str, Any]:
    if handdraw or getattr(args, "gif_all_examples", False):
        selected = list(rollouts)
        selection_report = {"policy": "all_examples", "reason": "handdraw test outputs every rollout GIF" if handdraw else "gif_all_examples enabled", "selected_total": len(selected), "available_total": len(rollouts)}
    else:
        selected, selection_report = _select_balanced_gif_rollouts(
            rollouts,
            per_success=getattr(args, "gif_per_difficulty_success", 5),
            per_failure=getattr(args, "gif_per_difficulty_failure", 5),
        )
    gif_report = _save_selected_gifs(out_dir, selected, fps=getattr(args, "gif_fps", 3), max_frames=getattr(args, "gif_max_frames", 96))
    gif_report.update(selection_report=selection_report, note="GIFs are diagnostic visualization artifacts only; they do not change metrics or labels.")
    ext.save_json(Path(out_dir) / "gif_examples_report.json", gif_report)
    ext.save_json(Path(out_dir) / "test_gif_examples_report.json", gif_report)
    return gif_report


def _is_handdraw_path(args) -> bool:
    p = str(getattr(args, "external_dataset_path", "") or "").lower()
    return "handdraw" in p or "hand" in p

def run_test_mode(args, out_dir: Path, device):
    audit={"status":"PASS","issues":[],"mode":"test","version":VERSION,"script":str(SCRIPT_PATH),"old_generator_called":False,"model_structure_changed":False}
    ext.save_json(out_dir/"test_protocol_compliance_audit.json", audit)
    ckpt=_checkpoint_or_block(args,out_dir)
    model=build_model(args).to(device)
    try:
        model.load_state_dict(torch.load(ckpt,map_location=device))
    except Exception as e:
        bp=out_dir/"blocking_report.json"
        ext.save_json(bp,{"found":f"checkpoint load failed: {e}","affected_goal":"v1 test cannot evaluate trained checkpoint","risk_if_continue":"checkpoint/model mismatch","needs_upper_confirmation":"matching checkpoint or architecture config","suggested_next_agent":"code generation agent"})
        raise SystemExit(f"[BLOCKED] checkpoint load failed; see {bp}")
    model.eval()
    test_config={k:getattr(args,k) for k in DEFAULTS if hasattr(args,k)}; test_config.update({"version":VERSION,"checkpoint_path":str(ckpt),"output_dir":str(out_dir)})
    ext.save_json(out_dir/"test_config.json", test_config)
    if args.manual_maze_file or args.manual_maze_text:
        text=_load_manual_text(args)
        maze,start,goal=vc.parse_manual_maze(text)
        dist=vc.bfs_distances(maze,goal); path=vc.shortest_path(maze,start,goal,dist)
        if not path:
            bp=out_dir/"blocking_report.json"; ext.save_json(bp,{"found":"manual maze is unsolvable","affected_goal":"manual test","risk_if_continue":"would test invalid maze","needs_upper_confirmation":"valid manual maze","suggested_next_agent":"code generation agent"}); return
        rec={"sample_id":"manual_maze","row_index":None,"maze":maze,"start":start,"goal":goal,"split":"manual","target_bucket":"manual","source":"manual","recomputed_bfs_len":len(path)-1,"recomputed_difficulty":_difficulty_from_bfs_len(args,len(path)-1)}
        selected=[rec]; schema={"status":"MANUAL"}; manifest={"external_dataset_format":"manual"}; conv={"split_report":{"mode":"manual","split_counts":{"manual":1}}}
        sampling_contract=_make_sampling_contract(args,schema,manifest,conv,selected,selected,"manual",diagnostic_only=True)
        selection_report={"num_candidate_records":1,"num_selected_records":1,"num_tested_records":1,"selected_records":[{"sample_id":"manual_maze","difficulty":rec["recomputed_difficulty"],"bfs_len":rec["recomputed_bfs_len"]}],"manual_maze_text":text}
    else:
        converted,schema,manifest,conv=ext.load_external_dataset(args.external_dataset_path,args.external_dataset_format,max_records=None,split_seed=args.seed)
        ext.save_json(out_dir/"test_dataset_schema_report.json", schema)
        if schema.get("status")!="READY":
            bp=out_dir/"blocking_report.json"; ext.save_json(bp,{"found":schema.get("blocking_reasons",[]),"affected_goal":"v1 external test","risk_if_continue":"would guess schema","needs_upper_confirmation":"explicit field mapping","suggested_next_agent":"code generation agent"}); return
        ext.save_json(out_dir/"test_split_manifest.json", {r["sample_id"]:r.get("split") for r in converted})
        selected, selection_report=_select_records_for_test(converted,args)
        sampling_contract=_make_sampling_contract(args,schema,manifest,conv,converted,selected,"external",diagnostic_only=(args.split=="train"))
    ext.save_json(out_dir/"sampling_contract_report.json", sampling_contract)
    ext.save_json(out_dir/"test_sampling_contract_report.json", sampling_contract)
    ext.save_json(out_dir/"test_sample_selection_report.json", selection_report)
    rollouts=_rollout_records_test(model,selected,args,device)
    summary=_summarize_test_rollouts(rollouts)
    metric_breakdown=_breakdown_rollouts(rollouts)
    ext.save_json(out_dir/"test_rollout_summary.json", summary)
    ext.save_json(out_dir/"test_metric_breakdown.json", metric_breakdown)
    _write_jsonl(out_dir/"test_rollout_traces.jsonl", rollouts if args.save_rollout_traces else [{k:v for k,v in r.items() if k!="trajectory"} for r in rollouts])
    _write_jsonl(out_dir/"test_failure_examples.jsonl", [{k:v for k,v in r.items() if k!="trajectory"} | {"trajectory_tail":r.get("trajectory",[])[-8:]} for r in rollouts if not r.get("success")])
    if getattr(args, "save_gif_examples", True):
        gif_report = _save_policy_gifs(out_dir, rollouts, args, handdraw=_is_handdraw_path(args))
        print(f"[GIF] policy={gif_report.get('selection_report',{}).get('policy')} saved={gif_report.get('n_saved', 0)} dir={gif_report.get('output_dir')}")
    else:
        ext.save_json(out_dir/"test_gif_examples_report.json", {"status":"SKIPPED", "reason":"--no-save-gif-examples set", "note":"GIF generation explicitly disabled."})
    print("[TEST DONE]")
    print(f"  output_dir={out_dir}")
    print(f"  selected={len(selected)} tested={len(rollouts)} success={summary.get('rollout_success_rate')}")


def _run_default_post_train_evaluations(args, train_out_dir: Path, hist: Dict[str, Any], device) -> None:
    """After default training, run sample split check and handdraw check automatically."""
    if not getattr(args, "auto_final_evaluation", True):
        return
    ckpt = hist.get("checkpoint_best") or str(train_out_dir / "1_0_0_extdata_v2_cnn_bfs_action_best.pt")
    plan = {"status": "STARTED", "checkpoint_path": ckpt, "sample_check": None, "handdraw_check": None}
    ext.save_json(train_out_dir / "default_post_train_evaluation_plan.json", plan)
    tests = []
    if getattr(args, "sample_check_dataset_path", None):
        t = copy.deepcopy(args)
        t.mode = "test"; t.checkpoint_path = ckpt
        t.external_dataset_path = args.sample_check_dataset_path
        t.external_dataset_format = "auto"
        t.split = "test"; t.sample_policy = "sequential"; t.difficulty_filter = "all"; t.bucket_filter = "all"
        t.max_test_samples = int(getattr(args, "sample_check_max_test_samples", 100))
        t.run_name = getattr(args, "sample_check_run_name", "default_sample_check_extdata_v2")
        t.save_gif_examples = True; t.gif_all_examples = False
        tests.append((t, V1_ROOT / "outputs" / "1.0.0_data_swap_3_0_4" / t.run_name, "sample_check"))
    if getattr(args, "handdraw_dataset_path", None) and Path(args.handdraw_dataset_path).exists():
        t = copy.deepcopy(args)
        t.mode = "test"; t.checkpoint_path = ckpt
        t.external_dataset_path = args.handdraw_dataset_path
        t.external_dataset_format = "jsonl"
        t.split = "test"; t.sample_policy = "sequential"; t.difficulty_filter = "all"; t.bucket_filter = "all"
        t.max_test_samples = int(getattr(args, "handdraw_check_max_test_samples", 10000))
        t.run_name = getattr(args, "handdraw_check_run_name", "default_handdraw_check_extdata_v2")
        t.save_gif_examples = True; t.gif_all_examples = True
        tests.append((t, V1_ROOT / "outputs" / "1.0.0_data_swap_3_0_4" / t.run_name, "handdraw_check"))
    else:
        plan["handdraw_check"] = {"status": "SKIPPED", "reason": "handdraw_dataset_path not found", "path": getattr(args, "handdraw_dataset_path", None)}
    for targs, tout, key in tests:
        tout.mkdir(parents=True, exist_ok=True)
        print(f"[DEFAULT POST EVAL] {key} -> {tout}")
        run_test_mode(targs, tout, device)
        plan[key] = {"status": "DONE", "output_dir": str(tout)}
    plan["status"] = "DONE"
    ext.save_json(train_out_dir / "default_post_train_evaluation_plan.json", plan)

def main():
    explicit={a.split('=')[0] for a in sys.argv[1:]}
    args=apply_preset(build_argparser().parse_args(), explicit)
    vc.set_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir=resolve_output_dir(args); out_dir.mkdir(parents=True, exist_ok=True)
    audit=protocol_audit(args,out_dir); ext.save_json(out_dir/"protocol_compliance_audit.json", audit)
    if audit["status"] != "PASS":
        raise SystemExit("[ERROR] protocol audit failed; see protocol_compliance_audit.json")
    if args.mode == "test":
        run_test_mode(args,out_dir,device)
        return
    if args.mode not in {"train","all","dry-run"}:
        raise SystemExit(f"unsupported --mode {args.mode}; use train or test")
    converted, schema, manifest, conv = ext.load_external_dataset(args.external_dataset_path, args.external_dataset_format, max_records=args.max_external_samples, split_seed=args.seed)
    ext.save_json(out_dir/"dataset_schema_report.json", schema)
    ext.save_json(out_dir/"dataset_source_manifest.json", manifest)
    if schema.get("status") != "READY":
        ext.save_json(out_dir/"blocking_report.json", {"status":"BLOCKED", "found":schema.get("blocking_reasons",[]), "affects_goal":"v1 external data replacement cannot start", "risk_if_continue":"would guess schema", "needs_upper_confirmation":"explicit field mapping", "suggested_next_agent":"code generation agent"})
        return
    ext.save_json(out_dir/"conversion_report.json", conv)
    split_manifest={r["sample_id"]: r["split"] for r in converted}; ext.save_json(out_dir/"split_manifest.json", split_manifest)
    samples, maze_records, label_report = external_records_to_bfs_samples(converted,args)
    ext.save_json(out_dir/"bfs_label_generation_report.json", label_report)
    if args.dry_run_dataset_only:
        ext.save_json(out_dir/"training_history.json", {"status":"DRY_RUN_SKIPPED"})
        ext.save_json(out_dir/"evaluation_summary.json", {"status":"DRY_RUN_SKIPPED"})
        ext.save_json(out_dir/"rollout_summary.json", {"status":"DRY_RUN_SKIPPED"})
        ext.save_json(out_dir/"rollout_debug.json", {"status":"DRY_RUN_SKIPPED"})
        write_baseline_comparison(out_dir)
        print(f"[DRY RUN DONE] output_dir={out_dir}")
        return
    train_samples,val_samples,test_samples=split_samples_by_maze(converted,samples,args.split_manifest_path)
    train_loader=DataLoader(vc.BFSActionDataset(train_samples), batch_size=args.batch_size, shuffle=True, collate_fn=vc.collate_batch)
    val_loader=DataLoader(vc.BFSActionDataset(val_samples), batch_size=args.batch_size, shuffle=False, collate_fn=vc.collate_batch)
    model=build_model(args).to(device)
    model, hist=train_model(model,train_loader,val_loader,args,device,out_dir)
    test_metrics=breakdown_samples(test_samples,model,device,args)
    rollouts, rollout_summary=rollout_external(model,converted,args,device)
    evaluation_summary={
        "action_accuracy_hard_label": test_metrics.get("hard_action_accuracy"),
        "action_accuracy_best_action_set": test_metrics.get("best_action_accuracy"),
        "valid_action_rate": test_metrics.get("valid_action_rate"),
        "wall_action_rate": test_metrics.get("wall_action_rate"),
        "rollout_success_rate": rollout_summary.get("rollout_success_rate"),
        "shortest_path_success_rate": rollout_summary.get("shortest_path_success_rate"),
        "avg_bfs_gap": rollout_summary.get("avg_bfs_gap"),
        "median_bfs_gap": float(np.median([r.get("bfs_gap",0) for r in rollouts])) if rollouts else float("nan"),
        "avg_wall_hits": rollout_summary.get("avg_wall_hits"),
        "avg_steps": rollout_summary.get("avg_steps"),
        "failure_type_breakdown": rollout_summary.get("failure_type_breakdown"),
        "breakdown": test_metrics.get("breakdown"),
        "diagnostic_only_not_stage_closure": True,
    }
    ext.save_json(out_dir/"training_history.json",hist)
    ext.save_json(out_dir/"evaluation_summary.json",evaluation_summary)
    ext.save_json(out_dir/"rollout_summary.json",rollout_summary)
    ext.save_json(out_dir/"rollout_debug.json", {"n_rollouts":len(rollouts), "rollouts":rollouts})
    write_failure_examples(out_dir,rollouts)
    write_baseline_comparison(out_dir)
    ext.save_json(out_dir/"run_metadata.json", {"version":VERSION,"script":str(SCRIPT_PATH),"model_summary":model.summary(),"started_at":datetime.now(timezone.utc).isoformat(),"external_dataset_path":args.external_dataset_path,"output_dir":str(out_dir),"n_train_samples":len(train_samples),"n_val_samples":len(val_samples),"n_test_samples":len(test_samples),"default_auto_final_evaluation":bool(getattr(args,"auto_final_evaluation",True))})
    _run_default_post_train_evaluations(args, out_dir, hist, device)
    print(f"[DONE] output_dir={out_dir}")

if __name__ == "__main__":
    main()
