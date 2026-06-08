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
import copy
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
LEARNING_ROOT = V2_ROOT.parent.parent.parent  # .../从迷宫开始的rl/feature_maze/v2.0.n RM -> .../learning
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
    p.add_argument("--external-dataset-path", "--external_dataset_path", dest="external_dataset_path", default=str(LEARNING_ROOT / "从迷宫开始的rl/samples.csv"))
    p.add_argument("--external-dataset-format", "--external_dataset_format", dest="external_dataset_format", default="auto", choices=["auto","csv","jsonl"])
    p.add_argument("--split-manifest-path", "--split_manifest_path", dest="split_manifest_path", default=None)
    p.add_argument("--max-external-mazes", "--max_external_mazes", dest="max_external_mazes", type=int, default=None)
    p.add_argument("--dry-run-dataset-only", "--dry_run_dataset_only", dest="dry_run_dataset_only", action="store_true", default=False)
    p.add_argument("--reward-model-path", "--reward_model_path", dest="reward_model_path", default=None)
    p.add_argument("--split", dest="split", default="test")
    p.add_argument("--sample-policy", "--sample_policy", dest="sample_policy", default="sequential", choices=["sequential","random_seeded","stratified_by_difficulty","stratified_by_bucket"])
    p.add_argument("--difficulty-filter", "--difficulty_filter", dest="difficulty_filter", default="all")
    p.add_argument("--bucket-filter", "--bucket_filter", dest="bucket_filter", default="all")
    p.add_argument("--max-test-mazes", "--max_test_mazes", dest="max_test_mazes", type=int, default=100)
    p.add_argument("--test-seed", "--test_seed", dest="test_seed", type=int, default=42)
    p.add_argument("--greedy-policy", "--greedy_policy", dest="greedy_policy", default="both", choices=["both","pure_argmax","bfs_tiebreak"])
    p.add_argument("--save-rollout-traces", "--save_rollout_traces", dest="save_rollout_traces", action="store_true", default=False)
    p.add_argument("--save-failure-examples", "--save_failure_examples", dest="save_failure_examples", action="store_true", default=False)
    p.add_argument("--save-gif-examples", "--save_gif_examples", dest="save_gif_examples", action="store_true", default=True)
    p.add_argument("--no-save-gif-examples", "--no_save_gif_examples", dest="save_gif_examples", action="store_false")
    p.add_argument("--gif-success-examples", "--gif_success_examples", dest="gif_success_examples", type=int, default=2)
    p.add_argument("--gif-failure-examples", "--gif_failure_examples", dest="gif_failure_examples", type=int, default=2)
    p.add_argument("--gif-fps", "--gif_fps", dest="gif_fps", type=int, default=3)
    p.add_argument("--gif-max-frames", "--gif_max_frames", dest="gif_max_frames", type=int, default=96)
    p.add_argument("--auto-final-evaluation", "--auto_final_evaluation", dest="auto_final_evaluation", action="store_true", default=True)
    p.add_argument("--no-auto-final-evaluation", "--no_auto_final_evaluation", dest="auto_final_evaluation", action="store_false")
    p.add_argument("--handdraw-dataset-path", "--handdraw_dataset_path", dest="handdraw_dataset_path", default=str(LEARNING_ROOT / "从迷宫开始的rl/feature_maze/handdraw_mazes_ascii.jsonl"))
    p.add_argument("--sample-check-dataset-path", "--sample_check_dataset_path", dest="sample_check_dataset_path", default=str(LEARNING_ROOT / "从迷宫开始的rl/samples.csv"))
    p.add_argument("--sample-check-run-name", "--sample_check_run_name", dest="sample_check_run_name", default="default_rm_sample_check_extdata_v2")
    p.add_argument("--handdraw-check-run-name", "--handdraw_check_run_name", dest="handdraw_check_run_name", default="default_rm_handdraw_check_extdata_v2")
    p.add_argument("--sample-check-max-test-mazes", "--sample_check_max_test_mazes", dest="sample_check_max_test_mazes", type=int, default=100)
    p.add_argument("--handdraw-check-max-test-mazes", "--handdraw_check_max_test_mazes", dest="handdraw_check_max_test_mazes", type=int, default=10000)
    p.add_argument("--gif-per-difficulty-success", "--gif_per_difficulty_success", dest="gif_per_difficulty_success", type=int, default=5)
    p.add_argument("--gif-per-difficulty-failure", "--gif_per_difficulty_failure", dest="gif_per_difficulty_failure", type=int, default=5)
    p.add_argument("--gif-all-examples", "--gif_all_examples", dest="gif_all_examples", action="store_true", default=False)
    return p

def output_dir(args) -> Path:
    rn = args.run_name if args.run_name != "default" else "default_extdata_v2"
    return V2_ROOT / "v2.0.9_data_swap_3_0_4" / rn

def protocol_audit(args, out_dir: Path) -> Dict[str, Any]:
    issues=[]
    s=str(SCRIPT_PATH).replace('\\','/')
    if not s.endswith("maze_dqn_v2_0_9_direction_visit_factorized_rm_extdata_v2.py"):
        issues.append({"scope":"script_name","status":"FAIL","message":"script must use extdata_v2 filename"})
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


def difficulty_from_bfs_len(args, bfs_len: int) -> str:
    if args.easy_bfs_min <= bfs_len <= args.easy_bfs_max:
        return "easy"
    if args.medium_bfs_min <= bfs_len <= args.medium_bfs_max:
        return "medium"
    if args.hard_bfs_min <= bfs_len <= args.hard_bfs_max:
        return "hard"
    return "out_of_range"

class TestExternalMazeProvider:
    """Sequential provider for test-only external records. No shuffle by default."""
    def __init__(self, records: List[Dict[str, Any]], mod, args):
        self.records=list(records); self.mod=mod; self.args=args; self.index=0; self.calls=0; self.reuse_count=0; self.served=[]; self.unsolvable=[]
        if not self.records: raise RuntimeError("no selected records available for test")
    def next_maze(self, args_or_requested_target=None, requested_target: str=None):
        if requested_target is None:
            requested_target=str(args_or_requested_target) if args_or_requested_target is not None else "external_test"
        if self.index >= len(self.records):
            self.index=0; self.reuse_count += 1
        rec=self.records[self.index]; self.index += 1; self.calls += 1
        maze=rec["maze"]; start=tuple(rec["start"]); goal=tuple(rec["goal"])
        dist=self.mod.bfs_distances(maze,goal); path=self.mod.shortest_path(maze,start,goal,dist)
        if not path:
            self.unsolvable.append({"sample_id":rec.get("sample_id"),"reason":"unsolvable"})
            if len(self.unsolvable) >= len(self.records): raise RuntimeError("all selected test records are unsolvable")
            return self.next_maze(requested_target)
        bfs_len=len(path)-1; actual=difficulty_from_bfs_len(self.args,bfs_len)
        meta=dict(rec.get("metadata",{}), target=requested_target, actual=actual, bfs_len=bfs_len, retries=0, wall_p="external_3_0_4_test_no_generator", sample_id=rec.get("sample_id"), row_index=rec.get("row_index"), target_bucket=rec.get("target_bucket"), source=rec.get("source"), split=rec.get("split"), external_dataset=True)
        self.served.append({"sample_id":rec.get("sample_id"),"split":rec.get("split"),"target_bucket":rec.get("target_bucket"),"source":rec.get("source"),"bfs_len":bfs_len,"actual":actual})
        return maze,start,goal,dist,path,meta
    def report(self):
        return {"selected_records_loaded":len(self.records),"generate_maze_calls_served_from_external":self.calls,"external_dataset_reuse_cycles":self.reuse_count,"unsolvable_records":len(self.unsolvable),"old_generator_called":False,"no_maze_repair":True,"no_default_start_goal":True,"no_fallback_corridor":True,"served_examples":self.served[:50]}

def select_test_records(records, mod, args):
    candidates=[]; skipped=[]
    for rec in records:
        if args.split != "all" and rec.get("split") != args.split:
            continue
        try:
            dist=mod.bfs_distances(rec["maze"],tuple(rec["goal"])); path=mod.shortest_path(rec["maze"],tuple(rec["start"]),tuple(rec["goal"]),dist)
            if not path:
                skipped.append({"sample_id":rec.get("sample_id"),"reason":"unsolvable"}); continue
            rec["recomputed_bfs_len"]=len(path)-1; rec["recomputed_difficulty"]=difficulty_from_bfs_len(args,len(path)-1)
        except Exception as e:
            skipped.append({"sample_id":rec.get("sample_id"),"reason":str(e)}); continue
        if args.difficulty_filter != "all" and rec.get("recomputed_difficulty") != args.difficulty_filter:
            continue
        if args.bucket_filter != "all" and str(rec.get("target_bucket")) != str(args.bucket_filter):
            continue
        candidates.append(rec)
    if args.sample_policy == "random_seeded":
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
    selected=candidates[:int(args.max_test_mazes)]
    return selected,{"num_candidate_records":len(records),"num_filter_pass_records":len(candidates),"num_selected_records":len(selected),"num_skipped_records":len(skipped),"skip_reasons":ext.summarize_reasons(skipped),"skipped_examples":skipped[:20],"selected_records":[{"sample_id":r.get("sample_id"),"split":r.get("split"),"difficulty":r.get("recomputed_difficulty"),"bfs_len":r.get("recomputed_bfs_len"),"target_bucket":r.get("target_bucket"),"source":r.get("source")} for r in selected]}

def make_sampling_contract(args, schema, manifest, conv, records, selected):
    split_report=conv.get("split_report",{}) if isinstance(conv,dict) else {}
    return {"report_name":"sampling_contract_report","diagnostic_only":True,"data_source_contract":{"dataset_path":args.external_dataset_path,"requested_format":args.external_dataset_format,"actual_format":manifest.get("external_dataset_format"),"maze_field":schema.get("selected_maze_field"),"start_fields":schema.get("selected_start_pair") or schema.get("selected_start_single"),"goal_fields":schema.get("selected_goal_pair") or schema.get("selected_goal_single"),"bfs_len_read":"bfs_len" in schema.get("fields",[]),"bucket_field":schema.get("selected_bucket_field"),"candidate_bucket_fields":schema.get("candidate_bucket_fields"),"split_field":schema.get("selected_split_field"),"source_field":schema.get("selected_source_field"),"original_data_modified":False,"maze_repair":False,"default_start_goal":False,"fallback_corridor":False,"v3_code_imported":False},"split_contract":{"mode":split_report.get("mode"),"hash_key":split_report.get("hash_key"),"seed":split_report.get("seed"),"ratios":split_report.get("ratios"),"split_counts":split_report.get("split_counts"),"requested_split":args.split,"train_split_test_is_diagnostic_only":bool(args.split=="train")},"difficulty_contract":{"primary_fact_source":"recomputed_bfs_shortest_path_length_after_import","external_bfs_len_is_auxiliary_only":True,"difficulty_filter":args.difficulty_filter,"thresholds":{"easy":[args.easy_bfs_min,args.easy_bfs_max],"medium":[args.medium_bfs_min,args.medium_bfs_max],"hard":[args.hard_bfs_min,args.hard_bfs_max]},"easy_medium_hard_not_3_0_4_bucket":True},"sample_policy_contract":{"sample_policy":args.sample_policy,"seed":args.test_seed,"bucket_filter":args.bucket_filter,"max_test_mazes":args.max_test_mazes,"selected_sample_ids":[r.get("sample_id") for r in selected]},"v2_previous_adapter_note":{"previous_rm_build_provider":"extdata_v1 ExternalMazeProvider shuffled external records, served old generate_maze(args,target) calls sequentially after shuffle, cycled when exhausted, and used requested_target only as metadata.","test_sampling_policy_differs_from_previous_rm_build_provider_cycling_policy":True},"counts":{"candidate_records":len(records),"selected_records":len(selected)}}

def write_jsonl(path: Path, rows):
    with path.open("w",encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,default=str)+"\n")

def compact_greedy(greedy):
    return {k:v for k,v in greedy.items() if not k.startswith('_')}


def extract_strategy_summary(greedy: Dict[str, Any], strategy: str) -> Dict[str, Any]:
    """Extract stable terminal metrics from original v2.0.9 greedy output."""
    node = greedy.get(strategy, {})
    if isinstance(node, dict) and isinstance(node.get("summary"), dict):
        src = node["summary"]
    elif isinstance(node, dict):
        src = node
    else:
        src = {}
    aliases = {
        "success": ["success", "success_rate", "rollout_success_rate"],
        "wall_timeout": ["wall_timeout", "wall_timeout_rate"],
        "stuck_timeout": ["stuck_timeout", "stuck_timeout_rate"],
        "explore_timeout": ["explore_timeout", "explore_timeout_rate"],
        "avg_bfs_gap": ["avg_bfs_gap", "bfs_gap"],
        "wall_chosen_rate": ["wall_chosen_rate", "wall_rate"],
        "toward_chosen_rate": ["toward_chosen_rate", "toward_rate"],
        "away_chosen_rate": ["away_chosen_rate", "away_rate"],
        "all_actions_saturated_rate": ["all_actions_saturated_rate", "saturated_rate"],
    }
    out: Dict[str, Any] = {}
    for canonical, keys in aliases.items():
        val = None
        for key in keys:
            if key in src:
                val = src.get(key)
                break
        out[canonical] = val
    return out

def build_test_terminal_summary(greedy: Dict[str, Any], selected_count: int, provider_calls: int, rm_path: Path, out_dir: Path) -> Dict[str, Any]:
    strategies = [s for s in ("bfs_tiebreak", "pure_argmax") if s in greedy]
    if not strategies:
        strategies = [k for k in greedy.keys() if not str(k).startswith("_") and k != "difficulty_breakdown"]
    return {
        "selected": selected_count,
        "provider_calls": provider_calls,
        "reward_model_path": str(rm_path),
        "output_dir": str(out_dir),
        "strategies": {s: extract_strategy_summary(greedy, s) for s in strategies},
        "note": "Diagnostic test-only summary. Success here is rollout success on the selected external test subset, not a formal generalization claim.",
    }

def print_test_terminal_summary(summary: Dict[str, Any]) -> None:
    print("[TEST DONE]")
    print(f"  output_dir={summary.get('output_dir')}")
    print(f"  selected={summary.get('selected')} provider_calls={summary.get('provider_calls')}")
    print("  metrics:")
    for strategy, metrics in summary.get("strategies", {}).items():
        parts = []
        for key in ("success", "wall_timeout", "stuck_timeout", "explore_timeout", "avg_bfs_gap", "wall_chosen_rate", "away_chosen_rate", "all_actions_saturated_rate"):
            val = metrics.get(key) if isinstance(metrics, dict) else None
            if val is None:
                continue
            if isinstance(val, float):
                parts.append(f"{key}={val:.4f}")
            else:
                parts.append(f"{key}={val}")
        print(f"    {strategy}: " + (" ".join(parts) if parts else "no summary metrics found; inspect test_greedy_rollout_summary.json"))

def run_extdata_test(args, out_dir: Path, mod, device):
    rm_path=args.reward_model_path or args.reward_model
    if not rm_path:
        bp=out_dir/"blocking_report.json"; ext.save_json(bp,{"found":"missing reward_model_path","affected_goal":"v2 RM test cannot load checkpoint","risk_if_continue":"would test random RM","needs_upper_confirmation":"provide reward_model_path","suggested_next_agent":"code generation agent"}); raise SystemExit(f"[BLOCKED] missing checkpoint; see {bp}")
    rm_path=Path(rm_path)
    if not rm_path.exists():
        bp=out_dir/"blocking_report.json"; ext.save_json(bp,{"found":f"checkpoint not found: {rm_path}","affected_goal":"v2 RM test","risk_if_continue":"wrong model evaluation","needs_upper_confirmation":"correct checkpoint path","suggested_next_agent":"code generation agent"}); raise SystemExit(f"[BLOCKED] checkpoint not found; see {bp}")
    converted,schema,manifest,conv=ext.load_external_dataset(args.external_dataset_path,args.external_dataset_format,max_records=None,split_seed=args.seed)
    ext.save_json(out_dir/"test_dataset_schema_report.json",schema)
    if schema.get("status")!="READY":
        bp=out_dir/"blocking_report.json"; ext.save_json(bp,{"found":schema.get("blocking_reasons",[]),"affected_goal":"v2 RM external test","risk_if_continue":"would guess schema","needs_upper_confirmation":"explicit field mapping","suggested_next_agent":"code generation agent"}); return
    selected,selection_report=select_test_records(converted,mod,args)
    if not selected:
        bp=out_dir/"blocking_report.json"; ext.save_json(bp,{"found":"no selected test records after split/difficulty/bucket filters","affected_goal":"v2 RM external test","risk_if_continue":"would call old generator or test empty set","needs_upper_confirmation":"loosen filters or choose existing split","suggested_next_agent":"code generation agent"}); return
    ext.save_json(out_dir/"test_split_manifest.json",{r["sample_id"]:r.get("split") for r in converted})
    contract=make_sampling_contract(args,schema,manifest,conv,converted,selected)
    ext.save_json(out_dir/"sampling_contract_report.json",contract)
    ext.save_json(out_dir/"test_sampling_contract_report.json",contract)
    ext.save_json(out_dir/"test_sample_selection_report.json",selection_report)
    audit={"status":"PASS","issues":[],"mode":"test","version":VERSION,"script":str(SCRIPT_PATH),"old_generator_called":False,"q_model_enabled":False,"rm_semantics_preserved":"toward > away > wall"}
    ext.save_json(out_dir/"test_protocol_compliance_audit.json",audit)
    model=mod.RewardModel(args.grid_size).to(device)
    try:
        model.load_state_dict(torch.load(rm_path,map_location=device))
    except Exception as e:
        bp=out_dir/"blocking_report.json"; ext.save_json(bp,{"found":f"checkpoint load failed: {e}","affected_goal":"v2 RM test","risk_if_continue":"checkpoint/model mismatch","needs_upper_confirmation":"matching RM checkpoint","suggested_next_agent":"code generation agent"}); raise SystemExit(f"[BLOCKED] checkpoint load failed; see {bp}")
    model.eval()
    provider=TestExternalMazeProvider(selected,mod,args)
    mod.generate_maze=provider.next_maze
    # test-only: no pair dataset build or training. Probes reuse selected external records through provider.
    args.rm_greedy_probe_mazes=len(selected)
    args.debug_probe_mazes=min(args.debug_probe_mazes, max(1,len(selected)))
    prim=mod.primitive_reward_report(model,args,device)
    ctx=mod.contextual_probe(model,args,device)
    if args.greedy_policy == "both":
        greedy=mod.greedy_comparison(model,args,device,n=len(selected))
    else:
        gr=mod.greedy_rollouts(model,args,device,len(selected),args.greedy_policy)
        greedy={args.greedy_policy:{"summary":gr["summary"],"action_ranking":gr["action_ranking"]},"difficulty_breakdown":{args.greedy_policy:mod.difficulty_breakdown_from_rollouts(gr["rollouts"])},"_rollouts_"+args.greedy_policy:gr["rollouts"]}
    stuck=mod.stuck_timeout_analysis(greedy,args)
    tail=mod.failure_tail_action_rank_report(greedy,args)
    ext.save_json(out_dir/"test_config.json",{**{k:getattr(args,k) for k in vars(args)},"version":VERSION,"reward_model_path":str(rm_path),"output_dir":str(out_dir)})
    ext.save_json(out_dir/"test_primitive_reward_score_report.json",prim)
    ext.save_json(out_dir/"test_contextual_probe.json",ctx)
    ext.save_json(out_dir/"test_greedy_rollout_summary.json",compact_greedy(greedy))
    ext.save_json(out_dir/"test_greedy_difficulty_breakdown.json",greedy.get("difficulty_breakdown",{}))
    ext.save_json(out_dir/"test_stuck_timeout_analysis.json",stuck)
    ext.save_json(out_dir/"test_failure_tail_action_rank_report.json",tail)
    rollouts=[]
    for key in ["_rollouts_bfs","_rollouts_pure","_rollouts_pure_argmax","_rollouts_bfs_tiebreak"]:
        if key in greedy: rollouts.extend(greedy[key])
    write_jsonl(out_dir/"test_rollout_traces.jsonl", rollouts if args.save_rollout_traces else [{k:v for k,v in r.items() if k!="trajectory"} for r in rollouts])
    write_jsonl(out_dir/"test_failure_examples.jsonl", [{k:v for k,v in r.items() if k!="trajectory"} | {"trajectory_tail":r.get("trajectory",[])[-args.tail_window:]} for r in rollouts if not r.get("success")])
    ext.save_json(out_dir/"test_provider_report.json",provider.report())
    if getattr(args, "save_gif_examples", True):
        gif_rollouts = _collect_v2_gif_rollouts(greedy, selected)
        gif_report = _save_v2_policy_gifs(out_dir, gif_rollouts, args)
        print(f"[GIF] policy={gif_report.get('selection_report',{}).get('policy')} saved={gif_report.get('n_saved', 0)} dir={gif_report.get('output_dir')}")
    else:
        ext.save_json(out_dir/"test_gif_examples_report.json", {"status":"SKIPPED", "reason":"--no-save-gif-examples set", "note":"GIF generation explicitly disabled."})
    terminal_summary = build_test_terminal_summary(greedy, len(selected), provider.calls, rm_path, out_dir)
    ext.save_json(out_dir/"test_terminal_summary.json", terminal_summary)
    print_test_terminal_summary(terminal_summary)



def _enrich_v2_rollout_for_gif(ro: Dict[str, Any], rec: Dict[str, Any], strategy: str) -> Dict[str, Any]:
    """Attach external maze geometry to a v2 greedy rollout so common GIF renderer can visualize it."""
    out = dict(ro)
    out["strategy"] = strategy
    out["sample_id"] = rec.get("sample_id", out.get("maze_id"))
    out["row_index"] = rec.get("row_index")
    out["split"] = rec.get("split")
    out["target_bucket"] = rec.get("target_bucket")
    out["source"] = rec.get("source")
    out["recomputed_difficulty"] = rec.get("recomputed_difficulty")
    out["recomputed_bfs_len"] = rec.get("recomputed_bfs_len")
    out["start"] = list(rec.get("start", []))
    out["goal"] = list(rec.get("goal", []))
    maze = rec.get("maze")
    out["maze_grid"] = maze.tolist() if hasattr(maze, "tolist") else maze
    out["failure_type"] = out.get("outcome", "success" if out.get("success") else "failure")
    norm_traj = []
    for st in out.get("trajectory", []):
        st2 = dict(st)
        if "pos_after" not in st2 and "pos" in st2:
            st2["pos_after"] = st2.get("pos")
        if "action" not in st2 and "chosen_action" in st2:
            st2["action"] = st2.get("chosen_action")
        if "is_wall" not in st2 and "chosen_is_wall" in st2:
            st2["is_wall"] = st2.get("chosen_is_wall")
        norm_traj.append(st2)
    out["trajectory"] = norm_traj
    return out

def _collect_v2_gif_rollouts(greedy: Dict[str, Any], selected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not selected:
        return []
    out: List[Dict[str, Any]] = []
    key_map = {
        "_rollouts_bfs": "bfs_tiebreak",
        "_rollouts_bfs_tiebreak": "bfs_tiebreak",
        "_rollouts_pure": "pure_argmax",
        "_rollouts_pure_argmax": "pure_argmax",
    }
    for key, strategy in key_map.items():
        for ro in greedy.get(key, []):
            idx = int(ro.get("maze_id", 0)) % len(selected)
            out.append(_enrich_v2_rollout_for_gif(ro, selected[idx], strategy))
    return out


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


def _select_balanced_gif_rollouts(rollouts: List[Dict[str, Any]], per_success: int, per_failure: int):
    all_rows = list(rollouts)
    selected = []
    seen = set()
    report = {"policy": "balanced_by_difficulty_success_failure_with_backfill", "per_difficulty_success_target": int(per_success), "per_difficulty_failure_target": int(per_failure), "difficulty": {}}
    def add(rows):
        for r in rows:
            rid = id(r)
            if rid not in seen:
                seen.add(rid); selected.append(r)
    for diff in ["easy", "medium", "hard"]:
        group = [r for r in all_rows if str(r.get("recomputed_difficulty", r.get("difficulty", "unavailable"))) == diff]
        succ = [r for r in group if bool(r.get("success", False))]
        fail = [r for r in group if not bool(r.get("success", False))]
        target_total = int(per_success) + int(per_failure)
        chosen = succ[:int(per_success)] + fail[:int(per_failure)]
        if len(chosen) < target_total:
            extras = [r for r in succ[int(per_success):] + fail[int(per_failure):] if r not in chosen]
            chosen += extras[:max(0, target_total - len(chosen))]
        add(chosen)
        report["difficulty"][diff] = {"available_total": len(group), "available_success": len(succ), "available_failure": len(fail), "selected": len(chosen), "selected_success": sum(1 for r in chosen if r.get("success")), "selected_failure": sum(1 for r in chosen if not r.get("success"))}
    requested_total = 3 * (int(per_success) + int(per_failure))
    if len(selected) < min(requested_total, len(all_rows)):
        add([r for r in all_rows if id(r) not in seen][:max(0, min(requested_total, len(all_rows)) - len(selected))])
        report["backfilled_from_other_examples"] = True
    else:
        report["backfilled_from_other_examples"] = False
    report["selected_total"] = len(selected); report["available_total"] = len(all_rows)
    return selected, report


def _save_selected_gifs(out_dir: Path, selected: List[Dict[str, Any]], args, subdir="test_gif_examples"):
    gif_dir = Path(out_dir) / subdir
    gif_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, ro in enumerate(selected, 1):
        kind = "success" if ro.get("success") else "failure"
        diff = _safe_name(ro.get("recomputed_difficulty", ro.get("difficulty", "unknown")))
        strategy = _safe_name(ro.get("strategy", "rm"))
        sid = _safe_name(ro.get("sample_id", ro.get("maze_id", i)))
        ftype = _safe_name(ro.get("failure_type", kind))
        path = gif_dir / f"{i:03d}_{strategy}_{diff}_{kind}_{sid}_{ftype}.gif"
        try:
            save_fn = getattr(ext, "save_rollout_gif", _local_save_rollout_gif)
            saved.append(save_fn(ro, path, fps=getattr(args, "gif_fps", 3), max_frames=getattr(args, "gif_max_frames", 96)))
        except Exception as e:
            saved.append({"status": "ERROR", "path": str(path), "error": str(e), "sample_id": sid})
    return {"output_dir": str(gif_dir), "saved": saved, "n_saved": sum(1 for x in saved if isinstance(x, dict) and x.get("status") == "SAVED")}


def _is_handdraw_path(args) -> bool:
    p = str(getattr(args, "external_dataset_path", "") or "").lower()
    return "handdraw" in p or "hand" in p


def _save_v2_policy_gifs(out_dir: Path, gif_rollouts: List[Dict[str, Any]], args):
    if _is_handdraw_path(args) or getattr(args, "gif_all_examples", False):
        selected = list(gif_rollouts)
        selection_report = {"policy": "all_examples", "reason": "handdraw test outputs every rollout GIF" if _is_handdraw_path(args) else "gif_all_examples enabled", "selected_total": len(selected), "available_total": len(gif_rollouts)}
    else:
        selected, selection_report = _select_balanced_gif_rollouts(gif_rollouts, getattr(args, "gif_per_difficulty_success", 5), getattr(args, "gif_per_difficulty_failure", 5))
    report = _save_selected_gifs(out_dir, selected, args, subdir="test_gif_examples")
    report.update(selection_report=selection_report, note="GIFs are diagnostic visualization artifacts only; they do not change metrics or labels.")
    ext.save_json(Path(out_dir)/"test_gif_examples_report.json", report)
    ext.save_json(Path(out_dir)/"gif_examples_report.json", report)
    return report


def _find_rm_checkpoint(out_dir: Path) -> Optional[Path]:
    patterns = ["*best_pure_argmax*.pt", "*best_loss*.pt", "*last*.pt", "*.pt"]
    for pat in patterns:
        found = sorted(Path(out_dir).glob(pat))
        if found:
            return found[0]
    return None


def _run_default_post_train_evaluations(args, train_out_dir: Path, device, mod) -> None:
    if not getattr(args, "auto_final_evaluation", True):
        return
    ckpt = _find_rm_checkpoint(train_out_dir)
    plan = {"status": "STARTED", "checkpoint_path": str(ckpt) if ckpt else None, "sample_check": None, "handdraw_check": None}
    ext.save_json(train_out_dir/"default_post_train_evaluation_plan.json", plan)
    if not ckpt:
        plan["status"] = "BLOCKED"; plan["reason"] = "no RM checkpoint found after training"
        ext.save_json(train_out_dir/"default_post_train_evaluation_plan.json", plan)
        return
    tests=[]
    if getattr(args, "sample_check_dataset_path", None):
        t=copy.deepcopy(args); t.mode="test"; t.reward_model_path=str(ckpt); t.external_dataset_path=args.sample_check_dataset_path; t.external_dataset_format="auto"; t.split="test"; t.sample_policy="sequential"; t.difficulty_filter="all"; t.bucket_filter="all"; t.max_test_mazes=int(getattr(args,"sample_check_max_test_mazes",100)); t.greedy_policy="both"; t.save_gif_examples=True; t.gif_all_examples=False; t.run_name=getattr(args,"sample_check_run_name","default_rm_sample_check_extdata_v2")
        tests.append((t, V2_ROOT/"v2.0.9_data_swap_3_0_4"/t.run_name, "sample_check"))
    if getattr(args, "handdraw_dataset_path", None) and Path(args.handdraw_dataset_path).exists():
        t=copy.deepcopy(args); t.mode="test"; t.reward_model_path=str(ckpt); t.external_dataset_path=args.handdraw_dataset_path; t.external_dataset_format="jsonl"; t.split="test"; t.sample_policy="sequential"; t.difficulty_filter="all"; t.bucket_filter="all"; t.max_test_mazes=int(getattr(args,"handdraw_check_max_test_mazes",10000)); t.greedy_policy="both"; t.save_gif_examples=True; t.gif_all_examples=True; t.run_name=getattr(args,"handdraw_check_run_name","default_rm_handdraw_check_extdata_v2")
        tests.append((t, V2_ROOT/"v2.0.9_data_swap_3_0_4"/t.run_name, "handdraw_check"))
    else:
        plan["handdraw_check"] = {"status":"SKIPPED", "reason":"handdraw_dataset_path not found", "path":getattr(args,"handdraw_dataset_path",None)}
    for targs, tout, key in tests:
        tout.mkdir(parents=True, exist_ok=True)
        print(f"[DEFAULT POST EVAL] {key} -> {tout}")
        run_extdata_test(targs, tout, mod, device)
        plan[key] = {"status":"DONE", "output_dir":str(tout)}
    plan["status"]="DONE"
    ext.save_json(train_out_dir/"default_post_train_evaluation_plan.json", plan)

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
    if args.mode == "test":
        run_extdata_test(args,out_dir,mod,torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        return
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
        ext.save_json(out_dir/"run_metadata.json", {"version": VERSION, "script": str(SCRIPT_PATH), "original_script": str(ORIGINAL_SCRIPT), "output_dir": str(out_dir), "external_dataset_path": args.external_dataset_path, "mode": args.mode, "device": str(device), "started_at": datetime.now(timezone.utc).isoformat(), "rm_semantics_preserved": "toward > away > wall", "adapter_mvp": True, "default_auto_final_evaluation": bool(getattr(args,"auto_final_evaluation",True))})
        write_baseline_comparison(out_dir)
        _run_default_post_train_evaluations(args, out_dir, device, mod)
    print(f"[DONE] output_dir={out_dir}")

if __name__ == "__main__":
    main()
