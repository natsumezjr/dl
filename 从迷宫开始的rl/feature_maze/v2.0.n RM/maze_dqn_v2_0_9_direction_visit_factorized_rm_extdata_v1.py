#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2.0.9 external-data replacement test for RM/probe/greedy rollout.

MVP scope:
- Keep original v2.0.9 RM semantics and implementation.
- Replace generate_maze with a read-only 3.0.4 external dataset source.
- Do not import v1 or v3 code.
- Do not enable Q model unless --enable-qcnn is explicitly provided.

Implementation note:
This is an adapter script around the existing v2.0.9 script, not a new RM algorithm.
It patches the old generate_maze symbol at runtime so all RM dataset/probe/greedy calls use
external records instead of the old generator.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

SCRIPT_PATH = Path(__file__).resolve()
V2_ROOT = SCRIPT_PATH.parent
sys.path.insert(0, str(V2_ROOT / "common"))
import external_dataset_3_0_4_v1 as ext

VERSION = "v2.0.9_data_swap_3_0_4_extdata_v2"
ORIGINAL_SCRIPT = V2_ROOT / "maze_dqn_v2_0_9_direction_visit_factorized_rm.py"
OLD_BASELINE_PATH = V2_ROOT / "v2.0.9" / "default"

def load_original_module():
    if not ORIGINAL_SCRIPT.exists():
        raise SystemExit(f"[ERROR] original v2.0.9 script not found: {ORIGINAL_SCRIPT}")
    spec = importlib.util.spec_from_file_location("v2_0_9_original_extdata_source", ORIGINAL_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

def build_parser(mod) -> argparse.ArgumentParser:
    p = mod.build_argparser()
    p.add_argument("--external-dataset-path", "--external_dataset_path", dest="external_dataset_path", required=True)
    p.add_argument("--external-dataset-format", "--external_dataset_format", dest="external_dataset_format", default="auto", choices=["auto","csv","jsonl"])
    p.add_argument("--split-manifest-path", "--split_manifest_path", dest="split_manifest_path", default=None)
    p.add_argument("--max-external-mazes", "--max_external_mazes", dest="max_external_mazes", type=int, default=None)
    p.add_argument("--dry-run-dataset-only", "--dry_run_dataset_only", dest="dry_run_dataset_only", action="store_true", default=False)
    return p

def output_dir(args) -> Path:
    rn = args.run_name if args.run_name != "default" else "default_extdata_v2"
    return V2_ROOT / "v2.0.9_data_swap_3_0_4" / rn

def protocol_audit(args, out_dir: Path) -> Dict[str, Any]:
    issues=[]
    s=str(SCRIPT_PATH).replace('\\','/')
    if not s.endswith("maze_dqn_v2_0_9_direction_visit_factorized_rm_extdata_v1.py"):
        issues.append({"scope":"script_name","status":"FAIL","message":"script must use extdata_v1 filename"})
    if "v2.0.n RM" not in s:
        issues.append({"scope":"script_path","status":"WARN","message":"expected v2.0.n RM path; check copied location"})
    if args.mode in {"train-q","all-q"} and not args.enable_qcnn:
        issues.append({"scope":"qcnn","status":"FAIL","message":"Q model is not allowed unless --enable-qcnn is explicit"})
    if not args.external_dataset_path:
        issues.append({"scope":"external_dataset_path","status":"FAIL","message":"external dataset path is required"})
    if "v2.0.9_data_swap_3_0_4" not in str(out_dir):
        issues.append({"scope":"output_dir","status":"FAIL","message":"must write into new data_swap directory"})
    return {"status":"PASS" if not [x for x in issues if x["status"]=="FAIL"] else "FAIL", "issues": issues, "required_reports": ["rm_dataset_build_report.json","sample_profile_distribution.json","pair_bucket_report.json","primitive_pressure_report.json","rm_training_history.json","primitive_reward_score_report.json","contextual_probe.json","greedy_rollout_summary.json","greedy_difficulty_breakdown.json","stuck_timeout_analysis.json","failure_tail_action_rank_report.json","baseline_comparison.json","protocol_compliance_audit.json"], "adapter_scope":"MVP external dataset adapter; original RM primitive semantics preserved"}

class ExternalMazeProvider:
    def __init__(self, records: List[Dict[str, Any]], mod, args):
        self.records = list(records)
        self.mod = mod
        self.args = args
        self.index = 0
        self.calls = 0
        self.reuse_count = 0
        self.unsolvable = []
        self.skipped = []
        self.served = []
        if not self.records:
            raise RuntimeError("no external records available for v2.0.9 RM")
        random.shuffle(self.records)
    def next_maze(self, args_or_requested_target=None, requested_target: str = None):
        # Compatible with both direct provider calls and the original v2.0.9 signature:
        #   provider.next_maze("dry_run")
        #   generate_maze(args, target_difficulty)
        # The first form is used by this adapter's dry-run probe; the second form is
        # used inside the original v2.0.9 build_dataset / probe / rollout code.
        if requested_target is None:
            requested_target = str(args_or_requested_target) if args_or_requested_target is not None else "external"
        if self.index >= len(self.records):
            # Explicit cycling: still external data only; no old generator fallback.
            self.index = 0
            self.reuse_count += 1
        rec = self.records[self.index]
        self.index += 1
        self.calls += 1
        maze = rec["maze"]
        start = tuple(rec["start"]); goal = tuple(rec["goal"])
        dist = self.mod.bfs_distances(maze, goal)
        path = self.mod.shortest_path(maze, start, goal, dist)
        if not path:
            self.unsolvable.append({"sample_id": rec.get("sample_id"), "row_index": rec.get("row_index"), "reason": "unsolvable"})
            # Do not repair. Move to next external record. If all are bad, fail.
            if len(self.unsolvable) >= len(self.records):
                raise RuntimeError("all external records are unsolvable; no fallback allowed")
            return self.next_maze(requested_target)
        bfs_len = len(path) - 1
        actual = self.mod.difficulty_of_len(self.args, bfs_len) or rec.get("bfs_len_bin_source") or "external"
        meta = dict(rec.get("metadata", {}), target=requested_target, actual=actual, bfs_len=bfs_len, retries=0, wall_p="external_3_0_4_no_generator", sample_id=rec.get("sample_id"), row_index=rec.get("row_index"), target_bucket=rec.get("target_bucket"), source=rec.get("source"), split=rec.get("split"), external_dataset=True)
        self.served.append({"sample_id": rec.get("sample_id"), "bfs_len": bfs_len, "actual": actual, "target_bucket": rec.get("target_bucket"), "source": rec.get("source"), "split": rec.get("split")})
        return maze, start, goal, dist, path, meta
    def report(self):
        counts=defaultdict(int)
        for r in self.served:
            counts[r.get("actual","unknown")] += 1
        return {"external_unique_records_loaded": len(self.records), "generate_maze_calls_served_from_external": self.calls, "external_dataset_reuse_cycles": self.reuse_count, "unsolvable_records": len(self.unsolvable), "unsolvable_examples": self.unsolvable[:20], "served_difficulty_counts": dict(counts), "old_generator_called": False, "no_maze_repair": True, "no_default_start_goal": True, "no_fallback_corridor": True, "served_examples": self.served[:20]}

def write_baseline_comparison(out_dir: Path):
    comp={"old_baseline_path": str(OLD_BASELINE_PATH), "new_data_run_path": str(out_dir), "comparable_metrics": ["pair_bucket_report","primitive_pressure_report","primitive_reward_score_report","contextual_probe","greedy_rollout_summary","greedy_difficulty_breakdown","stuck_timeout_analysis","failure_tail_action_rank_report"], "non_comparable_metrics": [], "non_comparable_reasons": {}, "note": "Diagnostic path comparison only; not an improvement claim."}
    if not OLD_BASELINE_PATH.exists():
        comp["non_comparable_metrics"].append("old baseline values")
        comp["non_comparable_reasons"]["old baseline values"] = "baseline path not found in this runtime"
    ext.save_json(out_dir/"baseline_comparison.json", comp)

def copy_or_alias_reports(mod, out_dir: Path, report: Dict[str, Any], hist_path: Optional[Path] = None):
    # Unprefixed names requested by the task.
    if report:
        ext.save_json(out_dir/"sample_profile_distribution.json", report.get("sample_profile_distribution", {}))
        ext.save_json(out_dir/"pair_bucket_report.json", report.get("pair_bucket_report", {}))
        ext.save_json(out_dir/"primitive_pressure_report.json", report.get("observed_primitive_pressure", {}))
        ext.save_json(out_dir/"bucket_pressure_contribution.json", report.get("bucket_pressure_contribution", {}))
    if hist_path and hist_path.exists():
        try:
            obj=json.loads(hist_path.read_text(encoding="utf-8"))
        except Exception:
            obj={"source_path": str(hist_path)}
        ext.save_json(out_dir/"rm_training_history.json", obj)

def run_dry(args, out_dir, converted, schema, manifest, conv, provider_report):
    ext.save_json(out_dir/"dataset_schema_report.json", schema)
    ext.save_json(out_dir/"dataset_source_manifest.json", manifest)
    ext.save_json(out_dir/"conversion_report.json", conv)
    ext.save_json(out_dir/"split_manifest.json", {r["sample_id"]: r.get("split") for r in converted})
    ext.save_json(out_dir/"rm_dataset_build_report.json", provider_report | {"status":"DRY_RUN_DATASET_ONLY"})
    for name in ["sample_profile_distribution.json","pair_bucket_report.json","primitive_pressure_report.json","rm_training_history.json","primitive_reward_score_report.json","contextual_probe.json","greedy_rollout_summary.json","greedy_difficulty_breakdown.json","stuck_timeout_analysis.json","failure_tail_action_rank_report.json"]:
        ext.save_json(out_dir/name, {"status":"DRY_RUN_SKIPPED"})
    write_baseline_comparison(out_dir)
    print(f"[DRY RUN DONE] output_dir={out_dir}")

def main():
    mod=load_original_module()
    # Accept both project-style underscore flags and argparse-style hyphen flags.
    sys.argv = [sys.argv[0]] + [(a.replace("_", "-") if a.startswith("--") else a) for a in sys.argv[1:]]
    explicit=set(a.split('=')[0] for a in sys.argv[1:])
    parser=build_parser(mod)
    args=parser.parse_args()
    args=mod.apply_preset(args, explicit)
    if args.mode == "all":
        print("[INFO] all maps to all-rm for extdata MVP; QCNN remains gated.")
        args.mode = "all-rm"
    allowed={"train-rm","debug-rm","all-rm","test","train-q","all-q"}
    if args.mode not in allowed:
        raise SystemExit(f"unsupported --mode {args.mode}; allowed={sorted(allowed)}")
    if args.mode in {"train-q","all-q"} and not args.enable_qcnn:
        raise SystemExit("[ERROR] Q model is disabled unless --enable-qcnn is explicitly set")
    out_dir=output_dir(args); out_dir.mkdir(parents=True, exist_ok=True)
    audit=protocol_audit(args,out_dir); ext.save_json(out_dir/"protocol_compliance_audit.json", audit)
    if audit["status"] != "PASS":
        raise SystemExit("[ERROR] protocol audit failed; see protocol_compliance_audit.json")
    converted, schema, manifest, conv = ext.load_external_dataset(args.external_dataset_path, args.external_dataset_format, max_records=args.max_external_mazes, split_seed=args.seed)
    ext.save_json(out_dir/"dataset_schema_report.json", schema)
    ext.save_json(out_dir/"dataset_source_manifest.json", manifest)
    if schema.get("status") != "READY":
        ext.save_json(out_dir/"blocking_report.json", {"status":"BLOCKED", "found":schema.get("blocking_reasons",[]), "affects_goal":"v2.0.9 RM external data replacement cannot start", "risk_if_continue":"would guess schema or endpoints", "needs_upper_confirmation":"explicit field mapping", "suggested_next_agent":"code generation agent"})
        return
    ext.save_json(out_dir/"conversion_report.json", conv)
    ext.save_json(out_dir/"split_manifest.json", {r["sample_id"]: r.get("split") for r in converted})

    mod.set_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    provider=ExternalMazeProvider(converted, mod, args)
    mod.generate_maze = provider.next_maze
    # Force old script output to our new directory by replacing nondefault_run_name behavior only inside this run.
    mod.nondefault_run_name = lambda _args: args.run_name if args.run_name != "default" else "default_extdata_v2"

    if args.dry_run_dataset_only:
        # Probe only a few provider calls to verify BFS reconstruction; no RM pairs/training.
        for _ in range(min(10, len(converted))):
            provider.next_maze("dry_run")
        run_dry(args, out_dir, converted, schema, manifest, conv, provider.report())
        return

    if args.mode in {"train-q","all-q"}:
        raise SystemExit("[BLOCKED] Q model stage is intentionally not executed in first extdata MVP unless separately reviewed; use RM mode first.")
    if args.mode == "test":
        raise SystemExit("[BLOCKED] test mode requires a trained extdata RM checkpoint; run all-rm first and pass --reward-model in a reviewed second step.")

    print("[Build External RM Dataset]")
    pairs, report = mod.build_dataset(args)
    mod.print_reports(args, report)
    mod.save_reports(out_dir, report)
    ext.save_json(out_dir/"rm_dataset_build_report.json", provider.report() | {"pairs": len(pairs), "status":"BUILT"})
    copy_or_alias_reports(mod, out_dir, report)
    if args.mode in {"train-rm","all-rm"}:
        model=mod.train_rm(args,pairs,out_dir,device)
        hist_path=out_dir/f"{mod.VERSION}_reward_model_history.json"
        copy_or_alias_reports(mod, out_dir, report, hist_path)
        chk=mod.collapse_check(model,args,device)
        prim=mod.primitive_reward_report(model,args,device)
        ctx=mod.contextual_probe(model,args,device)
        greedy=mod.greedy_comparison(model,args,device)
        stuck=mod.stuck_timeout_analysis(greedy,args)
        tail=mod.failure_tail_action_rank_report(greedy,args)
        mod.print_probe_summary(ctx,greedy,prim,stuck,None,tail)
        ext.save_json(out_dir/"primitive_reward_score_report.json", prim)
        ext.save_json(out_dir/"contextual_probe.json", ctx)
        ext.save_json(out_dir/"greedy_rollout_summary.json", {k:v for k,v in greedy.items() if not k.startswith('_')})
        ext.save_json(out_dir/"greedy_difficulty_breakdown.json", greedy.get("difficulty_breakdown",{}))
        ext.save_json(out_dir/"stuck_timeout_analysis.json", stuck)
        ext.save_json(out_dir/"failure_tail_action_rank_report.json", tail)
        ext.save_json(out_dir/"rm_diagnostic.json", {"collapse_check": chk})
        ext.save_json(out_dir/"run_metadata.json", {"version": VERSION, "script": str(SCRIPT_PATH), "original_script": str(ORIGINAL_SCRIPT), "output_dir": str(out_dir), "external_dataset_path": args.external_dataset_path, "mode": args.mode, "device": str(device), "started_at": datetime.now(timezone.utc).isoformat(), "rm_semantics_preserved": "toward > away > wall", "adapter_mvp": True})
        write_baseline_comparison(out_dir)
    print(f"[DONE] output_dir={out_dir}")

if __name__ == "__main__":
    main()
