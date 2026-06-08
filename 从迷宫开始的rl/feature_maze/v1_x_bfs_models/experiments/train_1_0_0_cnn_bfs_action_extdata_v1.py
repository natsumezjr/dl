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
sys.path.insert(0, str(V1_ROOT / "common"))
import v1_common as vc
import external_dataset_3_0_4_v1 as ext

VERSION = "1.0.0_data_swap_3_0_4_extdata_v1"
EXPERIMENT_NAME = "cnn_bfs_action_extdata"
MODEL_NAME = "CNNBFSActionPolicy"
OLD_BASELINE_PATH = V1_ROOT / "outputs" / "1.0.0" / "default"

DEFAULTS: Dict[str, Any] = dict(
    run_name="default",
    seed=42,
    preset="quick",
    input_mode="compact_2ch",
    grid_size=8,
    max_steps=64,
    val_ratio=0.15,
    epochs=2,
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
    external_dataset_path=None,
    external_dataset_format="auto",
    split_manifest_path=None,
    max_external_samples=None,
    dry_run_dataset_only=False,
    baseline_old_generator_eval=False,
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
        if isinstance(v, bool): p.add_argument(*opts, dest=k, action="store_true" if not v else "store_false", default=v)
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
    rn = args.run_name if args.run_name != "default" else "default_extdata_v1"
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
    best_loss=1e9; best_path=out_dir/"1_0_0_extdata_v1_cnn_bfs_action_best.pt"; last_path=out_dir/"1_0_0_extdata_v1_cnn_bfs_action_last.pt"
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
    if not str(SCRIPT_PATH).replace('\\','/').endswith('train_1_0_0_cnn_bfs_action_extdata_v1.py'):
        issues.append({"scope":"script_name","status":"FAIL","message":"script must use extdata_v1 name"})
    if "v1_x_bfs_models" not in str(SCRIPT_PATH).replace('\\','/'):
        issues.append({"scope":"script_path","status":"FAIL","message":"script must live under v1_x_bfs_models"})
    if "1.0.0_data_swap_3_0_4" not in str(out_dir):
        issues.append({"scope":"output_dir","status":"FAIL","message":"output must be under new data_swap directory"})
    if not args.external_dataset_path:
        issues.append({"scope":"external_dataset_path","status":"FAIL","message":"external dataset path is required"})
    return {"status":"PASS" if not issues else "FAIL", "issues":issues, "required_reports":["dataset_schema_report.json","dataset_source_manifest.json","conversion_report.json","split_manifest.json","bfs_label_generation_report.json","training_history.json","evaluation_summary.json","rollout_summary.json","rollout_debug.json","failure_examples.jsonl","baseline_comparison.json","protocol_compliance_audit.json"]}

def write_failure_examples(out_dir: Path, rollouts):
    with (out_dir/"failure_examples.jsonl").open("w", encoding="utf-8") as f:
        for ro in rollouts:
            if not ro.get("success"):
                f.write(json.dumps({k:v for k,v in ro.items() if k != "trajectory"} | {"trajectory_tail": ro.get("trajectory", [])[-8:]}, ensure_ascii=False, default=str) + "\n")

def main():
    explicit={a.split('=')[0] for a in sys.argv[1:]}
    args=apply_preset(build_argparser().parse_args(), explicit)
    vc.set_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir=resolve_output_dir(args); out_dir.mkdir(parents=True, exist_ok=True)
    audit=protocol_audit(args,out_dir); ext.save_json(out_dir/"protocol_compliance_audit.json", audit)
    if audit["status"] != "PASS":
        raise SystemExit("[ERROR] protocol audit failed; see protocol_compliance_audit.json")
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
    ext.save_json(out_dir/"run_metadata.json", {"version":VERSION,"script":str(SCRIPT_PATH),"model_summary":model.summary(),"started_at":datetime.now(timezone.utc).isoformat(),"external_dataset_path":args.external_dataset_path,"output_dir":str(out_dir),"n_train_samples":len(train_samples),"n_val_samples":len(val_samples),"n_test_samples":len(test_samples)})
    print(f"[DONE] output_dir={out_dir}")

if __name__ == "__main__":
    main()
