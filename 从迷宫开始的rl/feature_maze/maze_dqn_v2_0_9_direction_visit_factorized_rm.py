#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2.0.9_direction_visit_factorized_rm_stuck_away_rebalanced_qtest
RM-only reward-model training/debug for an 8x8 maze.

This file is intentionally organized around the engineering refactor requested for v2.0.9(6):
  1. data generation
  2. sample profiles
  3. pair buckets
  4. B values and margins
  5. normalized primitive pressure
  6. terminal/debug reports
  7. primitive reward/contextual/greedy probes

Default is RM-only. Optional QCNN modes are gated behind --enable-qcnn.
"""

import argparse
import json
import math
import os
import sys
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

VERSION = "v2.0.9_direction_visit_factorized_rm_stuck_away_rebalanced_qtest"
ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]
ACTION_NAMES = ["UP","DOWN","LEFT","RIGHT"]
PRIMITIVES = ["toward","away","wall","visit"]
EPS = 1e-8
INF = 10**9

TARGET_PRESSURE = {
    "toward": (0.24, 0.42),
    "away": (-0.09, 0.09),
    "wall": (-0.42, -0.24),
    "visit": (-0.21, -0.09),
    "toward-away": (0.24, None),
    "away-wall": (0.24, None),
    "absWall-absVisit": (0.12, None),
}
EXPECTED_PRESSURE = {
    "toward": 0.33,
    "away": 0.00,
    "wall": -0.33,
    "visit": -0.15,
    "toward-away": 0.33,
    "away-wall": 0.33,
    "absWall-absVisit": 0.18,
}

DEFAULTS = dict(
    mode="all-rm", run_name="default", seed=42, score_normalizer="episode_len",
    grid_size=8, max_steps=64,
    rm_mazes=1000, rm_epochs=16, rm_batch_size=64, lr=3e-4,
    difficulty_easy=0.2, difficulty_medium=0.4, difficulty_hard=0.4,
    easy_bfs_min=6, easy_bfs_max=10,
    medium_bfs_min=11, medium_bfs_max=18,
    hard_bfs_min=19, hard_bfs_max=48,
    visit_count_cap=8,
    tail_window=8, tail_unique_min=4, tail_visit_severity_max=0.75, tail_oscillation_max=0.45,
    explore_wall_severity_min=0.00, explore_wall_severity_max=0.18,
    explore_visit_severity_min=0.20, explore_visit_severity_max=0.85,
    stuck_unique_max=8, stuck_area_ratio_bfs=0.90, stuck_min_visit_severity=0.60, stuck_wall_severity_max=0.18,
    wall_timeout_min_severity=0.25,
    wall_recovery_success_min_wall=2, wall_recovery_success_max_wall=6,
    high_visit_success_min_visit_severity=0.55, high_visit_success_tail_visit_max=0.45, high_visit_success_tail_unique_min=4,
    rm_greedy_probe_mazes=100, debug_probe_mazes=50, select_probe_mazes=20, rm_greedy_tie_eps=0.03,
    success_timeout_min=0.28, success_timeout_max=0.52, success_timeout_scale=1.20,
    explore_timeout_min=0.08, explore_timeout_max=0.20, explore_timeout_scale=0.90,
    global_quant_min=0.00, global_quant_max=0.04, global_quant_scale=0.80,
    timeout_quant_min=0.00, timeout_quant_max=0.03, timeout_quant_scale=0.80,
    progress_min=0.14, progress_max=0.38, progress_scale=0.90,
    visit_min=0.08, visit_max=0.24, visit_scale=0.55,
    wall_min=0.22, wall_max=0.40, wall_scale=1.05,
    stuck_escape_min=0.12, stuck_escape_max=0.30, stuck_escape_scale=0.80,
    pairs_success_explore=3, pairs_success_stuck=3, pairs_success_wall=2,
    pairs_explore_stuck=2, pairs_explore_wall=1,
    pairs_success_internal=2, pairs_explore_internal=2, pairs_stuck_internal=2, pairs_wall_internal=1,
    pairs_timeout_quant=1,
    pairs_progress=28, pairs_visit=24, pairs_wall=12, pairs_stuck_escape=16,
    success_profile_optimal_ratio=0.10,
    success_profile_detour_ratio=0.15,
    success_profile_visit_recovery_ratio=0.25,
    success_profile_high_visit_recovery_ratio=0.30,
    success_profile_wall_recovery_ratio=0.20,
    progress_context_clean_ratio=0.20,
    progress_context_visited_ratio=0.25,
    progress_context_high_visit_ratio=0.30,
    progress_context_wall_adjacent_ratio=0.25,
    wall_context_clean_ratio=0.20,
    wall_context_visited_ratio=0.25,
    wall_context_high_visit_ratio=0.25,
    wall_context_wall_adjacent_ratio=0.30,
    wall_positive_toward_ratio=0.75,
    wall_positive_away_ratio=0.25,
    wall_repeated_context_ratio=0.35,
    visit_delta_v_max=0.35,
    preset="default", no_tqdm=False, enable_qcnn=False, test=False, test_strategy="both",
    q_episodes=50, q_batch_size=64, q_lr=1e-3, gamma=0.95, replay_size=5000, q_warmup_steps=100, target_update_interval=100,
    eps_start=0.20, eps_end=0.02, eps_decay_episodes=100,
)
# ---------------- basic utils ----------------

def set_seed(seed:int)->None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def ensure_dir(p:Path)->None: p.mkdir(parents=True, exist_ok=True)

def save_json(path:Path, obj:Any)->None:
    def default(o):
        if isinstance(o,(np.integer,)): return int(o)
        if isinstance(o,(np.floating,)): return float(o)
        if isinstance(o,np.ndarray): return o.tolist()
        if isinstance(o,Path): return str(o)
        return str(o)
    with open(path,"w",encoding="utf-8") as f: json.dump(obj,f,ensure_ascii=False,indent=2,default=default)

def mean(xs:List[float])->float:
    ys=[float(x) for x in xs if x is not None and not (isinstance(x,float) and math.isnan(x))]
    return float(np.mean(ys)) if ys else float("nan")

def qtile(xs:List[float], q:float)->float:
    ys=[float(x) for x in xs if x is not None and not (isinstance(x,float) and math.isnan(x))]
    return float(np.quantile(ys,q)) if ys else float("nan")

def pct(x:float)->str:
    if x is None or (isinstance(x,float) and math.isnan(x)): return "   nan%"
    return f"{100*x:6.2f}%"

def clip01(x:float)->float: return float(np.clip(x,0.0,1.0))

def weighted_choice(items):
    total=sum(max(0.0,float(w)) for _,w in items)
    if total<=0: return random.choice([x for x,_ in items])
    r=random.random()*total; acc=0.0
    for x,w in items:
        acc += max(0.0,float(w))
        if r <= acc: return x
    return items[-1][0]

def nondefault_run_name(args)->str:
    if args.run_name != "default": return args.run_name
    watch=[
        "rm_mazes","rm_epochs","rm_batch_size","pairs_wall","pairs_progress","pairs_visit",
        "wall_max","progress_max","visit_max","wall_min","progress_min","visit_min",
        "success_profile_optimal_ratio","success_profile_detour_ratio","success_profile_visit_recovery_ratio",
        "success_profile_high_visit_recovery_ratio","success_profile_wall_recovery_ratio",
        "wall_positive_toward_ratio","wall_positive_away_ratio"
    ]
    parts=[]
    for k in watch:
        if hasattr(args,k) and getattr(args,k) != DEFAULTS.get(k):
            parts.append(f"{k}-{str(getattr(args,k)).replace('.','p')}")
    return "default" if not parts else "__".join(parts)

# ---------------- maze / BFS tools ----------------

def in_bounds(p:Tuple[int,int], n:int)->bool: return 0 <= p[0] < n and 0 <= p[1] < n

def neighbors(p:Tuple[int,int], n:int):
    for a,(dr,dc) in enumerate(ACTIONS):
        q=(p[0]+dr,p[1]+dc)
        if in_bounds(q,n): yield a,q

def bfs_distances(maze:np.ndarray, goal:Tuple[int,int])->np.ndarray:
    n=maze.shape[0]
    dist=np.full((n,n),INF,dtype=np.int32)
    if maze[goal]==1: return dist
    dq=deque([goal]); dist[goal]=0
    while dq:
        p=dq.popleft()
        for _,q in neighbors(p,n):
            if maze[q]==0 and dist[q] > dist[p]+1:
                dist[q]=dist[p]+1; dq.append(q)
    return dist

def shortest_path(maze,start,goal,dist)->Optional[List[Tuple[int,int]]]:
    if dist[start] >= INF: return None
    n=maze.shape[0]; p=start; path=[p]
    for _ in range(n*n+10):
        if p==goal: return path
        cand=[q for _,q in neighbors(p,n) if maze[q]==0 and dist[q]==dist[p]-1]
        if not cand: return None
        p=random.choice(cand); path.append(p)
    return path if path and path[-1]==goal else None

def action_between(p,q)->int:
    d=(q[0]-p[0],q[1]-p[1])
    for i,a in enumerate(ACTIONS):
        if a==d: return i
    raise ValueError(f"not adjacent {p}->{q}")

def path_to_actions(path): return [action_between(path[i],path[i+1]) for i in range(len(path)-1)]

def difficulty_of_len(args,bfs_len:int)->Optional[str]:
    if args.easy_bfs_min <= bfs_len <= args.easy_bfs_max: return "easy"
    if args.medium_bfs_min <= bfs_len <= args.medium_bfs_max: return "medium"
    if args.hard_bfs_min <= bfs_len <= args.hard_bfs_max: return "hard"
    return None

def choose_target_difficulty(args)->str:
    r=random.random()
    if r < args.difficulty_easy: return "easy"
    if r < args.difficulty_easy + args.difficulty_medium: return "medium"
    return "hard"

def generate_maze(args, target_difficulty:str):
    n=args.grid_size
    ranges={"easy":(args.easy_bfs_min,args.easy_bfs_max),"medium":(args.medium_bfs_min,args.medium_bfs_max),"hard":(args.hard_bfs_min,args.hard_bfs_max)}
    lo,hi=ranges[target_difficulty]
    retries=0
    for attempt in range(500):
        retries=attempt+1
        wall_p=random.uniform(0.05,0.30)
        maze=(np.random.rand(n,n)<wall_p).astype(np.int8)
        free=[(r,c) for r in range(n) for c in range(n) if maze[r,c]==0]
        if len(free)<2: continue
        start,goal=random.sample(free,2)
        dist=bfs_distances(maze,goal)
        path=shortest_path(maze,start,goal,dist)
        if not path: continue
        bfs_len=len(path)-1
        if lo <= bfs_len <= hi:
            return maze,start,goal,dist,path,dict(target=target_difficulty,actual=target_difficulty,bfs_len=bfs_len,retries=retries,wall_p=wall_p)
    corridor=[]
    for r in range(n):
        if r % 2 == 0:
            cols = range(n) if (r//2) % 2 == 0 else range(n-1, -1, -1)
            for c in cols: corridor.append((r, c))
            if r + 1 < n:
                corridor.append((r + 1, n-1 if (r//2) % 2 == 0 else 0))
    maze=np.ones((n,n),dtype=np.int8)
    for cell in corridor: maze[cell]=0
    free=[tuple(x) for x in corridor]
    possible=[]
    for _ in range(4000):
        start,goal=random.sample(free,2); dist=bfs_distances(maze,goal); path=shortest_path(maze,start,goal,dist)
        if not path: continue
        bfs_len=len(path)-1
        if lo <= bfs_len <= hi:
            possible.append((start,goal,dist,path,bfs_len))
            if len(possible)>=64: break
    if possible:
        start,goal,dist,path,bfs_len=random.choice(possible)
        return maze,start,goal,dist,path,dict(target=target_difficulty,actual=target_difficulty,bfs_len=bfs_len,retries=retries,wall_p="corridor")
    best=None; target_mid=(lo+hi)//2
    for _ in range(4000):
        start,goal=random.sample(free,2); dist=bfs_distances(maze,goal); path=shortest_path(maze,start,goal,dist)
        if not path: continue
        bfs_len=len(path)-1; score=abs(bfs_len-target_mid)
        if best is None or score<best[0]: best=(score,start,goal,dist,path,bfs_len)
    if best is None: raise RuntimeError("fallback maze generation failed")
    _,start,goal,dist,path,bfs_len=best; actual=difficulty_of_len(args,bfs_len) or target_difficulty
    return maze,start,goal,dist,path,dict(target=target_difficulty,actual=actual,bfs_len=bfs_len,retries=retries,wall_p="corridor_closest")

# ---------------- transition simulation ----------------

def step_env(maze,pos,action):
    n=maze.shape[0]; dr,dc=ACTIONS[action]; q=(pos[0]+dr,pos[1]+dc)
    if not in_bounds(q,n) or maze[q]==1: return pos, True
    return q, False

def action_candidates(maze,p,dist,n):
    cur=int(dist[p]) if dist[p]<INF else 999
    toward=[]; away=[]; walls=[]
    for a,(dr,dc) in enumerate(ACTIONS):
        q=(p[0]+dr,p[1]+dc)
        if (not in_bounds(q,n)) or maze[q]==1:
            walls.append(a)
        else:
            dd=int(dist[q])-cur
            if dd<0: toward.append(a)
            else: away.append(a)
    return toward,away,walls

def make_state(maze,pos,goal,visits,cap):
    n=maze.shape[0]
    s=np.zeros((4,n,n),dtype=np.float32)
    s[0]=maze.astype(np.float32); s[1,pos[0],pos[1]]=1.0; s[2,goal[0],goal[1]]=1.0
    s[3]=np.minimum(visits,cap).astype(np.float32)/float(cap)
    return s

def action_tensor(action:int,n:int):
    x=np.zeros((4,n,n),dtype=np.float32); x[action,:,:]=1.0; return x

def transition_tensor(tr): return torch.from_numpy(tr["x"])

def primitive_phi(transitions):
    T=max(1,len(transitions)); t=a=w=0.0; visit_sum=0.0
    for tr in transitions:
        if tr["is_wall"]: w += 1.0
        else:
            if tr["delta_bfs"] < 0: t += 1.0
            else: a += 1.0
        visit_sum += max(0.0, (tr.get("visit_after",1)-1))
    return {"toward":t/T,"away":a/T,"wall":w/T,"visit":visit_sum/T}

def compute_traj_attrs(transitions, maze, start, goal, dist, success, args):
    T=max(1,len(transitions)); positions=[start]+[tr["next_pos"] for tr in transitions]
    counts=defaultdict(int)
    for p in positions: counts[tuple(p)] += 1
    unique_visited=len(counts)
    max_visit=int(max(counts.values())) if counts else 1
    reachable_cells=int(np.sum((maze==0)&(dist<INF)))
    bfs_len=int(dist[start]) if dist[start] < INF else 999
    final_pos=positions[-1]
    final_bfs_dist=int(dist[final_pos]) if dist[final_pos] < INF else 999
    wall_hits=sum(1 for tr in transitions if tr["is_wall"])
    wall_seen=defaultdict(int); repeated_wall_hits=0; max_consec=0; cur_consec=0; unique_wall_positions=set()
    for tr in transitions:
        if tr["is_wall"]:
            unique_wall_positions.add(tuple(tr["pos"])); wall_seen[tuple(tr["pos"])] += 1
            if wall_seen[tuple(tr["pos"])] > 1: repeated_wall_hits += 1
            cur_consec += 1; max_consec=max(max_consec,cur_consec)
        else:
            cur_consec=0
    wall_rate=wall_hits/(T+EPS); repeated_wall_rate=repeated_wall_hits/(T+EPS); max_consec_rate=max_consec/(T+EPS)
    wall_severity=0.50*wall_rate+0.35*repeated_wall_rate+0.15*max_consec_rate
    visit_severity=sum(max(0,(c-1)) for c in counts.values())/(T+EPS)
    coverage_reachable=unique_visited/(reachable_cells+EPS); coverage_bfs=unique_visited/(bfs_len+EPS)
    oscillations=sum(1 for i in range(2,len(positions)) if positions[i]==positions[i-2])
    oscillation_rate=oscillations/(T+EPS)
    tail_positions=positions[-args.tail_window:]; tail_unique=len(set(tail_positions)); tail_counts=defaultdict(int)
    for p in tail_positions: tail_counts[tuple(p)] += 1
    tail_len=max(1,len(tail_positions)-1)
    tail_visit_severity=sum(max(0,(c-1)) for c in tail_counts.values())/(tail_len+EPS)
    tail_osc=sum(1 for i in range(2,len(tail_positions)) if tail_positions[i]==tail_positions[i-2])
    tail_oscillation_rate=tail_osc/(tail_len+EPS)
    gap_norm=max(0,T-bfs_len)/max(1,args.max_steps-bfs_len)
    final_distance_badness=final_bfs_dist/(bfs_len+EPS)
    phi=primitive_phi(transitions)
    phi["visit"] = max(phi["visit"], visit_severity)
    return dict(
        success=bool(success), steps=len(transitions), wall_hits=int(wall_hits), repeated_wall_hits=int(repeated_wall_hits),
        max_consecutive_wall=int(max_consec), unique_wall_positions=int(len(unique_wall_positions)), wall_rate=wall_rate,
        repeated_wall_rate=repeated_wall_rate, max_consecutive_wall_rate=max_consec_rate, wall_severity=wall_severity,
        visit_severity=visit_severity, max_visit=max_visit, unique_visited=int(unique_visited), reachable_cells=int(reachable_cells),
        coverage_reachable=coverage_reachable, coverage_bfs=coverage_bfs, tail_unique=int(tail_unique),
        tail_visit_severity=tail_visit_severity, tail_oscillation_rate=tail_oscillation_rate, oscillation_rate=oscillation_rate,
        final_bfs_dist=int(final_bfs_dist), start_bfs_dist=int(bfs_len), bfs_len=int(bfs_len), gap_norm=gap_norm,
        final_distance_badness=final_distance_badness, phi=phi,
    )

def simulate_actions(maze,start,goal,dist,actions,args,init_pos=None,init_visits=None,sample_type="sample",profile=""):
    n=args.grid_size; pos=start if init_pos is None else init_pos
    visits=np.zeros((n,n),dtype=np.int32) if init_visits is None else init_visits.copy()
    if visits[pos]==0: visits[pos]=1
    transitions=[]; success=False
    for a in actions[:args.max_steps]:
        s=make_state(maze,pos,goal,visits,args.visit_count_cap)
        cur_d=int(dist[pos]) if dist[pos] < INF else 999
        current_visit=int(visits[pos])
        nxt,is_wall=step_env(maze,pos,a)
        nv=visits.copy(); nv[nxt if not is_wall else pos] += 1
        ns=make_state(maze,nxt,goal,nv,args.visit_count_cap)
        nd=int(dist[nxt]) if dist[nxt] < INF else 999
        x=np.concatenate([s,ns,action_tensor(a,n)],axis=0).astype(np.float32)
        tr=dict(state=s,next_state=ns,x=x,action=a,pos=pos,next_pos=nxt,is_wall=is_wall,delta_bfs=nd-cur_d,
                cur_bfs=cur_d,next_bfs=nd,visit_after=int(nv[nxt if not is_wall else pos]),current_visit=current_visit,
                reached_goal=(nxt==goal and not is_wall))
        transitions.append(tr); visits=nv; pos=nxt
        if pos==goal:
            success=True; break
    attrs=compute_traj_attrs(transitions,maze,start,goal,dist,success,args)
    return dict(sample_type=sample_type,profile=profile,kind="episode",transitions=transitions,**attrs)

def make_segment(maze,goal,dist,pos,visits,first_action,args,sample_type="segment",bias="toward",length=1,context=""):
    actions=[first_action]
    p,_=step_env(maze,pos,first_action)
    for _ in range(max(0,length-1)):
        toward,away,walls=action_candidates(maze,p,dist,args.grid_size); nonwall=toward+away
        if not nonwall: break
        if bias=="toward" and toward and random.random()<0.8: aa=random.choice(toward)
        elif bias=="away" and away and random.random()<0.8: aa=random.choice(away)
        else: aa=random.choice(nonwall)
        actions.append(aa); p,_=step_env(maze,p,aa)
    seq=simulate_actions(maze,pos,goal,dist,actions,args,init_pos=pos,init_visits=visits,sample_type=sample_type,profile=sample_type)
    seq["kind"]="local"; seq["context"]=context
    context_visit_severity=max(0.0,(float(visits[pos])-1.0))/max(1,len(seq["transitions"]))
    if context_visit_severity > seq.get("visit_severity",0.0):
        seq["visit_severity"] = float(context_visit_severity)
        seq["phi"]["visit"] = float(max(seq["phi"].get("visit",0.0), context_visit_severity))
    return seq

# ---------------- sample profiles ----------------

def validate_success_sample(x,args,profile=None):
    if x is None or x.get("sample_type")!="success" or not x.get("success",False):
        return False
    if profile=="high_visit_recovery_success":
        x["exit_high_visit"] = bool(
            x.get("visit_severity",0.0) >= args.high_visit_success_min_visit_severity and
            x.get("tail_visit_severity",999.0) <= args.high_visit_success_tail_visit_max and
            x.get("tail_unique",0) >= args.high_visit_success_tail_unique_min
        )
        return x["exit_high_visit"]
    return True

def validate_explore_timeout_sample(x,args):
    if x is None or x.get("sample_type")!="explore_timeout" or x.get("success",False): return False
    low=math.ceil(2.0*max(1,x["bfs_len"])); high=min(int(3.0*max(1,x["bfs_len"])), x.get("reachable_cells",INF))
    return (low <= x["unique_visited"] <= high and
            args.explore_wall_severity_min <= x["wall_severity"] <= args.explore_wall_severity_max and
            args.explore_visit_severity_min <= x["visit_severity"] <= args.explore_visit_severity_max and
            x["tail_unique"] >= args.tail_unique_min and
            x["tail_visit_severity"] <= args.tail_visit_severity_max and
            x["tail_oscillation_rate"] <= args.tail_oscillation_max)

def validate_stuck_timeout_sample(x,args):
    if x is None or x.get("sample_type")!="stuck_timeout" or x.get("success",False): return False
    return (x["unique_visited"] <= args.stuck_unique_max and
            x["unique_visited"] <= args.stuck_area_ratio_bfs*max(1,x["bfs_len"]) and
            x["visit_severity"] >= args.stuck_min_visit_severity and
            x["wall_severity"] <= args.stuck_wall_severity_max)

def validate_wall_timeout_sample(x,args):
    return x is not None and x.get("sample_type")=="wall_timeout" and not x.get("success",False) and x["wall_severity"] >= args.wall_timeout_min_severity

def gen_success_profile(maze,start,goal,dist,path,args,profile):
    bfs_actions=path_to_actions(path)
    if profile=="optimal_success":
        return simulate_actions(maze,start,goal,dist,bfs_actions,args,sample_type="success",profile=profile)
    if profile=="visit_recovery_success" and len(path)>=4:
        idx=random.randint(1,len(path)-2); pprev,p=path[idx-1],path[idx]
        loop=[action_between(p,pprev),action_between(pprev,p)]*random.randint(8,12)
        acts=path_to_actions(path[:idx+1])+loop+path_to_actions(path[idx:])
        return simulate_actions(maze,start,goal,dist,acts,args,sample_type="success",profile=profile)
    if profile=="high_visit_recovery_success" and len(path)>=4:
        idx=random.randint(1,len(path)-2); pprev,p=path[idx-1],path[idx]
        loop=[action_between(p,pprev),action_between(pprev,p)]*random.randint(14,18)
        acts=path_to_actions(path[:idx+1])+loop+path_to_actions(path[idx:])
        return simulate_actions(maze,start,goal,dist,acts,args,sample_type="success",profile=profile)
    if profile=="detour_success":
        for _ in range(80):
            idx=random.randint(0,len(path)-2); p=path[idx]
            toward,away,walls=action_candidates(maze,p,dist,args.grid_size)
            away=[a for a in away if not step_env(maze,p,a)[1]]
            if away:
                a=random.choice(away); q,_=step_env(maze,p,a)
                try: back=action_between(q,p)
                except Exception: continue
                acts=path_to_actions(path[:idx+1])+[a,back]*random.randint(3,6)+path_to_actions(path[idx:])
                return simulate_actions(maze,start,goal,dist,acts,args,sample_type="success",profile=profile)
    if profile=="wall_recovery_success":
        for _ in range(100):
            idx=random.randint(0,len(path)-2); p=path[idx]
            _,_,walls=action_candidates(maze,p,dist,args.grid_size)
            if walls:
                wh=random.randint(max(4,args.wall_recovery_success_min_wall),max(8,args.wall_recovery_success_max_wall))
                acts=path_to_actions(path[:idx+1])+[random.choice(walls)]*wh+path_to_actions(path[idx:])
                return simulate_actions(maze,start,goal,dist,acts,args,sample_type="success",profile=profile)
    return None

def gen_success_profiles(maze,start,goal,dist,path,args):
    profiles=[
        ("optimal_success",args.success_profile_optimal_ratio),
        ("detour_success",args.success_profile_detour_ratio),
        ("visit_recovery_success",args.success_profile_visit_recovery_ratio),
        ("high_visit_recovery_success",args.success_profile_high_visit_recovery_ratio),
        ("wall_recovery_success",args.success_profile_wall_recovery_ratio),
    ]
    out=[]
    for prof,_ in profiles:
        x=gen_success_profile(maze,start,goal,dist,path,args,prof)
        if validate_success_sample(x,args,prof): out.append(x)
    for _ in range(4):
        prof=weighted_choice(profiles); x=gen_success_profile(maze,start,goal,dist,path,args,prof)
        if validate_success_sample(x,args,prof): out.append(x)
    return out

def gen_explore_timeout(maze,start,goal,dist,path,args):
    """Generate broad exploration failures.

    Strict prompt audit note:
    - explore_timeout must not collapse to wall-free paths.
    - The generator therefore has an explicit mild-wall profile that injects
      one or two non-repeated wall contacts after the trajectory has already
      covered enough new cells.
    - It still validates against the requested wall/visit/tail bounds, so wall
      remains present but non-dominant.
    """
    bfs_len=max(1,len(path)-1)
    low=math.ceil(2*bfs_len)
    high=min(int(3*bfs_len), int(np.sum((maze==0)&(dist<INF))))
    profiles=[("mild_wall_explore",0.75),("clean_explore",0.25)]
    for _ in range(240):
        profile=weighted_choice(profiles)
        pos=start; actions=[]; visited={pos}; injected_walls=0; wall_keys=set()
        for t in range(args.max_steps):
            toward,away,walls=action_candidates(maze,pos,dist,args.grid_size)
            # Once the path is broad enough, force a small number of wall contacts
            # for the mild-wall profile. Avoid repeated same-position wall spam.
            if profile=="mild_wall_explore" and len(visited)>=low and injected_walls < random.choice([1,2]) and walls:
                candidates=[a for a in walls if (pos,a) not in wall_keys]
                if candidates:
                    a=random.choice(candidates)
                    actions.append(a); injected_walls += 1; wall_keys.add((pos,a))
                    # wall leaves pos unchanged; continue exploration next step
                    if len(actions)>=args.max_steps: break
                    continue
            choices=away+toward
            if not choices: break
            random.shuffle(choices); best=None
            # Prefer genuinely new non-goal cells to keep this an exploration failure.
            for a in choices:
                q,isw=step_env(maze,pos,a)
                if (not isw) and q not in visited and q!=goal:
                    best=a; break
            if best is None:
                nonwall=[a for a in choices if not step_env(maze,pos,a)[1]]
                best=random.choice(nonwall or choices)
            q,isw=step_env(maze,pos,best); actions.append(best)
            if not isw:
                pos=q; visited.add(pos)
            if pos==goal: break
            if len(visited)>=high and len(actions)>args.tail_window:
                # Add one last mild wall if this profile has not yet expressed wall.
                if profile=="mild_wall_explore" and injected_walls==0:
                    _,_,ww=action_candidates(maze,pos,dist,args.grid_size)
                    if ww:
                        actions.append(random.choice(ww)); injected_walls += 1
                break
        seq=simulate_actions(maze,start,goal,dist,actions,args,sample_type="explore_timeout",profile=profile)
        seq["success"]=False
        if profile=="mild_wall_explore" and seq.get("wall_hits",0)==0:
            continue
        if validate_explore_timeout_sample(seq,args): return seq
    return None

def gen_stuck_timeout(maze,start,goal,dist,path,args):
    """Generate small-area high-repeat failures.

    The original attempt used arbitrary path prefixes; for hard mazes that can
    exceed uniqueVisited constraints before the actual stuck loop begins.
    This version intentionally sticks near the start/early path cells so the
    path semantics match the prompt definition.
    """
    if len(path)<2: return None
    for _ in range(160):
        max_idx=min(len(path)-2, max(0,args.stuck_unique_max-2), 4)
        idx=random.randint(0,max_idx)
        p=path[idx]; q=path[idx+1]
        acts=path_to_actions(path[:idx+1])
        forward=action_between(p,q); back=action_between(q,p)
        loop=[]
        for _j in range(args.max_steps):
            loop += [forward,back]
        acts += loop
        # Allow a rare single wall, but keep wall non-dominant.
        _,_,walls=action_candidates(maze,p,dist,args.grid_size)
        if walls and random.random()<0.30 and len(acts)>4:
            insert=random.randint(len(path_to_actions(path[:idx+1])), min(len(acts)-1, len(path_to_actions(path[:idx+1]))+8))
            acts[insert]=random.choice(walls)
        seq=simulate_actions(maze,start,goal,dist,acts[:args.max_steps],args,sample_type="stuck_timeout",profile="stuck_timeout")
        seq["success"]=False
        if validate_stuck_timeout_sample(seq,args): return seq
    return None

def gen_wall_timeout(maze,start,goal,dist,path,args,profile=None):
    profiles=["same_position_repeated_wall","multi_position_wall","mixed_wall_loop","wall_after_explore","wall_after_high_visit"]
    profile=profile or random.choice(profiles)
    for _ in range(100):
        idx=random.randint(0,len(path)-2); p=path[idx]
        _,_,walls=action_candidates(maze,p,dist,args.grid_size)
        if not walls: continue
        acts=path_to_actions(path[:idx+1])
        if profile=="same_position_repeated_wall":
            acts += [random.choice(walls)]*args.max_steps
        elif profile=="mixed_wall_loop" and idx+1 < len(path):
            q=path[idx+1]; forward=action_between(p,q); back=action_between(q,p); wa=random.choice(walls)
            for _ in range(args.max_steps//3): acts += [wa,forward,back]
        elif profile=="wall_after_explore":
            pos=p
            for _e in range(12):
                toward,away,ww=action_candidates(maze,pos,dist,args.grid_size); nonwall=away+toward
                if not nonwall: break
                a=random.choice(nonwall); acts.append(a); pos,_=step_env(maze,pos,a)
            _,_,ww=action_candidates(maze,pos,dist,args.grid_size)
            acts += [random.choice(ww or walls)]*args.max_steps
        elif profile=="wall_after_high_visit" and idx+1 < len(path):
            q=path[idx+1]; forward=action_between(p,q); back=action_between(q,p)
            acts += [forward,back]*random.randint(4,8)+[random.choice(walls)]*args.max_steps
        else:
            cur_idx=idx
            while len(acts)<args.max_steps and cur_idx < len(path)-1:
                pp=path[cur_idx]; _,_,ww=action_candidates(maze,pp,dist,args.grid_size)
                if ww: acts += [random.choice(ww)]*random.randint(1,3)
                acts.append(action_between(path[cur_idx],path[cur_idx+1])); cur_idx += 1
            if len(acts)<args.max_steps: acts += [random.choice(walls)]*(args.max_steps-len(acts))
        seq=simulate_actions(maze,start,goal,dist,acts[:args.max_steps],args,sample_type="wall_timeout",profile=profile)
        seq["success"]=False
        if validate_wall_timeout_sample(seq,args): return seq
    return None

# ---------------- local pair generation ----------------

def visits_for_context(args,pos,context):
    visits=np.zeros((args.grid_size,args.grid_size),dtype=np.int32)
    if context in ("clean_progress","clean_wall_context"): visits[pos]=1
    elif context in ("visited_progress","visited_wall_context","wall_adjacent_progress","wall_adjacent_context"): visits[pos]=random.choice([2,2,3])
    elif context in ("high_visit_progress","high_visit_wall_context"): visits[pos]=random.choice([3,4,5])
    else: visits[pos]=1
    return visits

def cell_has_wall_adjacent(maze,p,n):
    return bool(action_candidates(maze,p,bfs_distances(maze,p),n)[2])

def gen_local_progress_pair(maze,start,goal,dist,path,args,context=None):
    contexts=[("clean_progress",args.progress_context_clean_ratio),("visited_progress",args.progress_context_visited_ratio),("high_visit_progress",args.progress_context_high_visit_ratio),("wall_adjacent_progress",args.progress_context_wall_adjacent_ratio)]
    context=context or weighted_choice(contexts)
    cells=[(r,c) for r in range(args.grid_size) for c in range(args.grid_size) if maze[r,c]==0 and dist[r,c]<INF]
    random.shuffle(cells)
    for p in cells:
        toward,away,walls=action_candidates(maze,p,dist,args.grid_size)
        if context=="wall_adjacent_progress" and not walls: continue
        if toward and away:
            visits=visits_for_context(args,p,context)
            pos=make_segment(maze,goal,dist,p,visits,random.choice(toward),args,"toward_step","toward",1,context)
            neg=make_segment(maze,goal,dist,p,visits,random.choice(away),args,"away_step","away",1,context)
            return pos,neg,context
    return None

VISIT_BANDS=[("lowV",0.20,0.50,2), ("midV",0.50,0.85,3), ("highV",0.85,1.20,4), ("extremeV",1.20,1.60,5)]
VISIT_BAND_REP={"lowV":0.35,"midV":0.65,"highV":0.95,"extremeV":1.25}

def visit_band_count(name):
    for n,lo,hi,cnt in VISIT_BANDS:
        if n==name: return cnt
    return 2

def visit_band_mid(name):
    # Representative values are deliberately adjacent and bounded so local visit
    # comparisons are continuous calibrations, not low-vs-extreme classifiers.
    return VISIT_BAND_REP.get(name,0.5)

def gen_local_visit_pair(maze,start,goal,dist,path,args,context=None):
    band_pairs=[("lowV","midV"),("midV","highV"),("highV","extremeV")]
    lo_band,hi_band=random.choice(band_pairs)
    direction=random.choice(["toward","away"])
    context=f"{direction}_{lo_band}_gt_{hi_band}"
    cells=[(r,c) for r in range(args.grid_size) for c in range(args.grid_size) if maze[r,c]==0 and dist[r,c]<INF]
    random.shuffle(cells)
    for p in cells:
        toward,away,walls=action_candidates(maze,p,dist,args.grid_size)
        choices=toward if direction=="toward" else away
        if not choices: continue
        a=random.choice(choices)
        vlow=np.zeros((args.grid_size,args.grid_size),dtype=np.int32); vlow[p]=visit_band_count(lo_band)
        vhi=np.zeros((args.grid_size,args.grid_size),dtype=np.int32); vhi[p]=visit_band_count(hi_band)
        pos=make_segment(maze,goal,dist,p,vlow,a,args,"low_visit_step",direction,1,context)
        neg=make_segment(maze,goal,dist,p,vhi,a,args,"high_visit_step",direction,1,context)
        # Override to calibrated adjacent-band means so Delta B_V cannot become extreme.
        pos["visit_severity"]=visit_band_mid(lo_band); neg["visit_severity"]=visit_band_mid(hi_band)
        pos["phi"]["visit"]=pos["visit_severity"]; neg["phi"]["visit"]=neg["visit_severity"]
        return pos,neg,context
    return None

def gen_local_wall_pair(maze,start,goal,dist,path,args,context=None):
    contexts=[("clean_wall_context",args.wall_context_clean_ratio),("visited_wall_context",args.wall_context_visited_ratio),("high_visit_wall_context",args.wall_context_high_visit_ratio),("wall_adjacent_context",args.wall_context_wall_adjacent_ratio)]
    context=context or weighted_choice(contexts)
    cells=[(r,c) for r in range(args.grid_size) for c in range(args.grid_size) if maze[r,c]==0 and dist[r,c]<INF]
    random.shuffle(cells)
    for p in cells:
        toward,away,walls=action_candidates(maze,p,dist,args.grid_size)
        if context=="wall_adjacent_context" and not walls: continue
        safe=[]
        if random.random()<args.wall_positive_toward_ratio and toward:
            safe=toward
        elif away:
            safe=away
        elif toward:
            safe=toward
        if safe and walls:
            visits=visits_for_context(args,p,context)
            if random.random()<args.wall_repeated_context_ratio: visits[p]=max(int(visits[p]),random.choice([2,3,4]))
            a=random.choice(safe); bias="toward" if a in toward else "away"
            pos=make_segment(maze,goal,dist,p,visits,a,args,"safe_step",bias,1,context)
            neg=make_segment(maze,goal,dist,p,visits,random.choice(walls),args,"wall_step","away",1,context)
            # Wall bucket is meant to isolate nonWallStep >> wallStep under the
            # same visit context. Do not let the bookkeeping increase from a
            # failed wall step leak into a separate visit penalty.
            neg["visit_severity"]=pos.get("visit_severity",0.0)
            neg["phi"]["visit"]=pos["phi"].get("visit",pos.get("visit_severity",0.0))
            return pos,neg,context
    return None


def escape_score_for_action(maze,dist,pos,visits,action,recent_tail,args):
    nxt,is_wall=step_env(maze,pos,action)
    if is_wall:
        return -999.0, dict(next_pos=nxt,is_wall=True,delta_bfs=0,visit_drop=0.0,area_exit=0.0)
    cur_d=int(dist[pos]) if dist[pos]<INF else 999
    nd=int(dist[nxt]) if dist[nxt]<INF else 999
    delta=nd-cur_d
    progress_score=float(-delta)
    visit_drop=float(visits[pos]-visits[nxt])
    area_exit=1.0 if tuple(nxt) not in set(recent_tail) else 0.0
    score=0.45*progress_score+0.35*visit_drop+0.20*area_exit
    return score, dict(next_pos=nxt,is_wall=False,delta_bfs=delta,visit_drop=visit_drop,area_exit=area_exit,progress_score=progress_score)

def gen_local_stuck_escape_pair(maze,start,goal,dist,path,args,context=None):
    # Build a high-visit, small-tail context and prefer an escape action over continuing the loop.
    cells=[(r,c) for r in range(args.grid_size) for c in range(args.grid_size) if maze[r,c]==0 and dist[r,c]<INF]
    random.shuffle(cells)
    for p in cells:
        toward,away,walls=action_candidates(maze,p,dist,args.grid_size)
        nonwall=toward+away
        if len(nonwall)<2: continue
        visits=np.zeros((args.grid_size,args.grid_size),dtype=np.int32)
        visits[p]=random.choice([4,5,6])
        recent_tail=[p]
        # Prefer a neighbor as recent loop cell, so returning to it is negative.
        loop_candidates=[]
        for a in nonwall:
            q,_=step_env(maze,p,a)
            visits[q]=random.choice([2,3,4])
            loop_candidates.append(q)
        if loop_candidates:
            recent_tail += random.sample(loop_candidates,min(2,len(loop_candidates)))
        scored=[]
        for a in nonwall:
            sc,meta=escape_score_for_action(maze,dist,p,visits,a,recent_tail,args)
            scored.append((sc,a,meta))
        scored.sort(key=lambda z:z[0], reverse=True)
        # If any toward escape is competitive, use it as positive; otherwise best escape can be away if it leaves high visit.
        toward_scored=[z for z in scored if z[1] in toward]
        pos_item=toward_scored[0] if toward_scored and toward_scored[0][0]>=scored[0][0]-0.25 else scored[0]
        neg_candidates=[z for z in scored[::-1] if z[1]!=pos_item[1] and z[0] < pos_item[0]-1e-6]
        if not neg_candidates: continue
        neg_item=neg_candidates[0]
        pos=make_segment(maze,goal,dist,p,visits,pos_item[1],args,"stuck_escape_step","toward" if pos_item[1] in toward else "away",1,"stuck_escape")
        neg=make_segment(maze,goal,dist,p,visits,neg_item[1],args,"stuck_loop_step","away",1,"stuck_escape")
        pos["escape_score"]=float(pos_item[0]); neg["escape_score"]=float(neg_item[0])
        pos["escape_meta"]=pos_item[2]; neg["escape_meta"]=neg_item[2]
        pos["phi"]["visit"]=pos.get("visit_severity",0.0); neg["phi"]["visit"]=neg.get("visit_severity",0.0)
        if pos["escape_score"] > neg["escape_score"] and not first_wall(pos) and not first_wall(neg):
            return pos,neg,"stuck_escape"
    return None

# ---------------- B values / margins / pairs ----------------

def area_badness(x,args):
    bfs=max(1,x["bfs_len"])
    return float(np.clip(1.0 - ((x["unique_visited"]-2*bfs)/(bfs+EPS)),0,1))

def area_smallness(x,args):
    return float(np.clip(1.0 - x["unique_visited"]/(x["bfs_len"]+EPS),0,1))

def b_success(x,args): return 0.35*x["wall_severity"] + 0.35*x["visit_severity"] + 0.30*x["gap_norm"]
def b_explore(x,args): return 0.30*x["final_distance_badness"] + 0.25*x["wall_severity"] + 0.25*x["visit_severity"] + 0.20*area_badness(x,args)
def b_stuck(x,args): return 0.45*x["visit_severity"] + 0.25*area_smallness(x,args) + 0.20*x["oscillation_rate"] + 0.10*x["wall_severity"]
def b_wall(x,args): return 0.60*x["wall_severity"] + 0.20*x["visit_severity"] + 0.10*area_smallness(x,args) + 0.10*x["final_distance_badness"]
def b_timeout(x,args): return 0.30*x["wall_severity"] + 0.30*x["visit_severity"] + 0.20*area_smallness(x,args) + 0.20*x["final_distance_badness"]

def internal_b(x,args):
    st=x["sample_type"]
    if st=="success": return b_success(x,args)
    if st=="explore_timeout": return b_explore(x,args)
    if st=="stuck_timeout": return b_stuck(x,args)
    if st=="wall_timeout": return b_wall(x,args)
    return b_timeout(x,args)

def margin_params(args,bucket):
    return {
        "success_timeout":(args.success_timeout_min,args.success_timeout_max,args.success_timeout_scale),
        "explore_timeout":(args.explore_timeout_min,args.explore_timeout_max,args.explore_timeout_scale),
        "global_quant":(args.global_quant_min,args.global_quant_max,args.global_quant_scale),
        "timeout_quant":(args.timeout_quant_min,args.timeout_quant_max,args.timeout_quant_scale),
        "progress":(args.progress_min,args.progress_max,args.progress_scale),
        "visit":(args.visit_min,args.visit_max,args.visit_scale),
        "wall":(args.wall_min,args.wall_max,args.wall_scale),
        "stuck_escape":(args.stuck_escape_min,args.stuck_escape_max,args.stuck_escape_scale),
    }[bucket]

def delta_b(pos,neg,bucket,args):
    if bucket=="success_timeout": return 1.0
    if bucket=="explore_timeout": return max(0.0,b_timeout(neg,args)-b_explore(pos,args)+0.20)
    if bucket=="timeout_quant": return max(0.0,b_timeout(neg,args)-b_timeout(pos,args))
    if bucket=="global_quant": return max(0.0,internal_b(neg,args)-internal_b(pos,args))
    if bucket=="progress": return max(0.0,pos["phi"]["toward"]-neg["phi"]["toward"] + neg["phi"]["away"]-pos["phi"]["away"])
    if bucket=="visit":
        bv_pos=clip01((pos["visit_severity"]-0.20)/(2.00-0.20+EPS)); bv_neg=clip01((neg["visit_severity"]-0.20)/(2.00-0.20+EPS))
        return min(float(args.visit_delta_v_max),max(0.0,bv_neg-bv_pos))
    if bucket=="wall": return max(0.0,neg["phi"]["wall"]-pos["phi"]["wall"])
    if bucket=="stuck_escape": return max(0.0, pos.get("escape_score",0.0)-neg.get("escape_score",0.0))
    return 0.0

def first_delta(x): return int(x["transitions"][0]["delta_bfs"]) if x["transitions"] else 0
def first_wall(x): return bool(x["transitions"][0]["is_wall"]) if x["transitions"] else False

def validate_pair(p,args):
    pos,neg,b=p["pos"],p["neg"],p["bucket"]
    if b=="success_timeout": return pos["success"] and not neg["success"]
    if b=="explore_timeout": return pos["sample_type"]=="explore_timeout" and neg["sample_type"] in ("stuck_timeout","wall_timeout")
    if b=="timeout_quant": return pos["sample_type"] in ("stuck_timeout","wall_timeout") and neg["sample_type"] in ("stuck_timeout","wall_timeout") and b_timeout(pos,args)<=b_timeout(neg,args)+1e-6
    if b=="global_quant": return pos["sample_type"]==neg["sample_type"] and internal_b(pos,args)<=internal_b(neg,args)+1e-6
    if b=="progress": return first_delta(pos)<0 and first_delta(neg)>0 and not first_wall(pos) and not first_wall(neg)
    if b=="visit": return not first_wall(pos) and not first_wall(neg) and first_delta(pos)==first_delta(neg) and pos["visit_severity"] < neg["visit_severity"] and delta_b(pos,neg,"visit",args)<=args.visit_delta_v_max+1e-6
    if b=="wall": return not first_wall(pos) and first_wall(neg)
    if b=="stuck_escape": return (not first_wall(pos)) and (not first_wall(neg)) and pos.get("escape_score",-999)>neg.get("escape_score",999)
    return True

def make_pair(args,pos,neg,bucket,context=""):
    db=delta_b(pos,neg,bucket,args); mn,mx,sc=margin_params(args,bucket)
    m=mn+(mx-mn)*math.tanh(max(0.0,db)/(sc+EPS))
    p=dict(pos=pos,neg=neg,bucket=bucket,pair_bucket=bucket,context=context,
           pos_sample_type=pos["sample_type"],neg_sample_type=neg["sample_type"],
           pair_name=f"{bucket}|{pos['sample_type']}|{neg['sample_type']}",delta_b=float(db),margin=float(m),
           margin_min=mn,margin_max=mx,margin_scale=sc,near_min=bool(m<=mn+0.02),near_max=bool(m>=mx-0.02))
    p["valid"]=validate_pair(p,args)
    return p

def add_pair(pairs,args,pos,neg,bucket,context=""):
    if pos is None or neg is None: return
    pairs.append(make_pair(args,pos,neg,bucket,context))

def lower_b_first(xs,args):
    xs=[x for x in xs if x]
    return sorted(xs,key=lambda x: internal_b(x,args)) if xs else []

# ---------------- engineering utilities ----------------

def maybe_tqdm(iterable, args, desc="", total=None, leave=False):
    if getattr(args, "no_tqdm", False) or tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=leave, ncols=100)

def fmt_float(x, width=8, prec=3):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return f"{'nan':>{width}s}"
        return f"{float(x):{width}.{prec}f}"
    except Exception:
        return f"{'nan':>{width}s}"

def fmt_pct2(x, width=8, prec=2):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return f"{'nan%':>{width}s}"
        return f"{100.0*float(x):{width-1}.{prec}f}%"
    except Exception:
        return f"{'nan%':>{width}s}"

def fmt_str(s, width=24):
    s = "" if s is None else str(s)
    if len(s) <= width:
        return f"{s:<{width}s}"
    if width <= 6:
        return s[:width]
    keep = width - 3
    left = keep // 2
    right = keep - left
    return f"{s[:left]}...{s[-right:]}"

def _fmt_cell(v, width, spec):
    if spec == "s": return fmt_str(v, width)
    if spec == "d":
        try: return f"{int(v):{width}d}"
        except Exception: return f"{'':>{width}s}"
    if spec == "pct": return fmt_pct2(v, width)
    if spec.startswith(".") and spec.endswith("f"):
        return fmt_float(v, width, int(spec[1:-1]))
    return f"{str(v):>{width}s}"

def print_table(title, columns, rows):
    print(f"\n{title}")
    header = " ".join(fmt_str(c[0], c[1]) if c[2] == "s" else f"{c[0]:>{c[1]}s}" for c in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" ".join(_fmt_cell(row.get(c[0], ""), c[1], c[2]) for c in columns))

def margin_configuration_summary(args):
    buckets = [
        ("success_timeout", args.success_timeout_min, args.success_timeout_max, args.success_timeout_scale, args.pairs_success_explore + args.pairs_success_stuck + args.pairs_success_wall),
        ("explore_timeout", args.explore_timeout_min, args.explore_timeout_max, args.explore_timeout_scale, args.pairs_explore_stuck + args.pairs_explore_wall),
        ("global_quant", args.global_quant_min, args.global_quant_max, args.global_quant_scale, args.pairs_success_internal + args.pairs_explore_internal + args.pairs_stuck_internal + args.pairs_wall_internal),
        ("timeout_quant", args.timeout_quant_min, args.timeout_quant_max, args.timeout_quant_scale, args.pairs_timeout_quant),
        ("progress", args.progress_min, args.progress_max, args.progress_scale, args.pairs_progress),
        ("visit", args.visit_min, args.visit_max, args.visit_scale, args.pairs_visit),
        ("wall", args.wall_min, args.wall_max, args.wall_scale, args.pairs_wall),
        ("stuck_escape", args.stuck_escape_min, args.stuck_escape_max, args.stuck_escape_scale, args.pairs_stuck_escape),
    ]
    return {b: {"bucket": b, "min": mn, "max": mx, "scale": sc, "pair_count": pc} for b, mn, mx, sc, pc in buckets}

# ---------------- data generation ----------------

def build_dataset(args):
    pairs=[]; elements=defaultdict(list); maze_records=[]
    for _ in maybe_tqdm(range(args.rm_mazes), args, desc="Build RM dataset", total=args.rm_mazes):
        target=choose_target_difficulty(args)
        maze,start,goal,dist,path,meta=generate_maze(args,target); maze_records.append(meta)
        success=gen_success_profiles(maze,start,goal,dist,path,args)
        explore=[]; stuck=[]; wall=[]
        for _k in range(5):
            e=gen_explore_timeout(maze,start,goal,dist,path,args)
            if e: explore.append(e)
        for _k in range(4):
            s=gen_stuck_timeout(maze,start,goal,dist,path,args)
            if s: stuck.append(s)
        for prof in ["same_position_repeated_wall","multi_position_wall","mixed_wall_loop","wall_after_explore","wall_after_high_visit"]:
            w=gen_wall_timeout(maze,start,goal,dist,path,args,prof)
            if w: wall.append(w)
        for _k in range(2):
            w=gen_wall_timeout(maze,start,goal,dist,path,args)
            if w: wall.append(w)
        for x in success+explore+stuck+wall: elements[x["sample_type"]].append(x)
        for i in range(args.pairs_success_explore):
            if success and explore: add_pair(pairs,args,success[i%len(success)],explore[i%len(explore)],"success_timeout")
        for i in range(args.pairs_success_stuck):
            if success and stuck: add_pair(pairs,args,success[i%len(success)],stuck[i%len(stuck)],"success_timeout")
        for i in range(args.pairs_success_wall):
            if success and wall: add_pair(pairs,args,success[i%len(success)],wall[i%len(wall)],"success_timeout")
        for i in range(args.pairs_explore_stuck):
            if explore and stuck: add_pair(pairs,args,explore[i%len(explore)],stuck[i%len(stuck)],"explore_timeout")
        for i in range(args.pairs_explore_wall):
            if explore and wall: add_pair(pairs,args,explore[i%len(explore)],wall[i%len(wall)],"explore_timeout")
        for pool,n in [(success,args.pairs_success_internal),(explore,args.pairs_explore_internal),(stuck,args.pairs_stuck_internal),(wall,args.pairs_wall_internal)]:
            sx=lower_b_first(pool,args)
            for i in range(min(n,max(0,len(sx)//2))): add_pair(pairs,args,sx[i],sx[-1-i],"global_quant")
        tw=lower_b_first(stuck+wall,args)
        for i in range(min(args.pairs_timeout_quant,max(0,len(tw)//2))): add_pair(pairs,args,tw[i],tw[-1-i],"timeout_quant")
        progress_contexts=["clean_progress","visited_progress","high_visit_progress","wall_adjacent_progress"]
        for i in maybe_tqdm(range(args.pairs_progress), args, desc="local progress", leave=False):
            got=gen_local_progress_pair(maze,start,goal,dist,path,args,progress_contexts[i%len(progress_contexts)])
            if got: add_pair(pairs,args,got[0],got[1],"progress",got[2])
        for _i in maybe_tqdm(range(args.pairs_visit), args, desc="local visit", leave=False):
            got=gen_local_visit_pair(maze,start,goal,dist,path,args)
            if got: add_pair(pairs,args,got[0],got[1],"visit",got[2])
        wall_contexts=["clean_wall_context","visited_wall_context","high_visit_wall_context","wall_adjacent_context"]
        for i in maybe_tqdm(range(args.pairs_wall), args, desc="local wall", leave=False):
            got=gen_local_wall_pair(maze,start,goal,dist,path,args,wall_contexts[i%len(wall_contexts)])
            if got: add_pair(pairs,args,got[0],got[1],"wall",got[2])
        for i in maybe_tqdm(range(args.pairs_stuck_escape), args, desc="local stuck_escape", leave=False):
            got=gen_local_stuck_escape_pair(maze,start,goal,dist,path,args)
            if got: add_pair(pairs,args,got[0],got[1],"stuck_escape",got[2])
    report=build_reports(args,pairs,elements,maze_records)
    return pairs,report

# ---------------- reports and pressure ----------------

def summarize_maze_report(args,records):
    out={}
    for d,target_ratio in [("easy",args.difficulty_easy),("medium",args.difficulty_medium),("hard",args.difficulty_hard)]:
        rs=[r for r in records if r["actual"]==d]
        bfs=[r["bfs_len"] for r in rs]; ret=[r["retries"] for r in rs]
        out[d]=dict(difficulty=d,target_ratio=target_ratio,actual_ratio=len(rs)/max(1,len(records)),n=len(rs),bfs_len_mean=mean(bfs),bfs_len_min=min(bfs) if bfs else None,bfs_len_p50=qtile(bfs,.5),bfs_len_max=max(bfs) if bfs else None,retry_mean=mean(ret))
    return out

def summarize_sample_profile_distribution(elements):
    grouped=defaultdict(list)
    for xs in elements.values():
        for x in xs: grouped[(x.get("sample_type",""),x.get("profile",""))].append(x)
    total=max(1,sum(len(v) for v in grouped.values()))
    out={}
    for (st,prof),xs in sorted(grouped.items()):
        key=f"{st}|{prof}"
        wall=[x.get("wall_severity",0.0) for x in xs]; visit=[x.get("visit_severity",0.0) for x in xs]
        out[key]=dict(sample_type=st,profile=prof,n=len(xs),ratio=len(xs)/total,succ_rate=mean([float(x.get("success",False)) for x in xs]),
            wall_sev_mean=mean(wall),wall_sev_min=min(wall) if wall else None,wall_sev_p50=qtile(wall,.5),wall_sev_max=max(wall) if wall else None,
            visit_sev_mean=mean(visit),visit_sev_min=min(visit) if visit else None,visit_sev_p50=qtile(visit,.5),visit_sev_max=max(visit) if visit else None,
            unique_mean=mean([x.get("unique_visited",0.0) for x in xs]),coverage_bfs_mean=mean([x.get("coverage_bfs",0.0) for x in xs]),
            gap_norm_mean=mean([x.get("gap_norm",0.0) for x in xs]),final_dist_mean=mean([x.get("final_bfs_dist",0.0) for x in xs]))
    return out

def summarize_primitive_distribution_by_profile(elements):
    grouped=defaultdict(list)
    for xs in elements.values():
        for x in xs: grouped[(x.get("sample_type",""),x.get("profile",""))].append(x)
    out={}
    for (st,prof),xs in sorted(grouped.items()):
        key=f"{st}|{prof}"
        out[key]=dict(sample_type=st,profile=prof,n=len(xs),toward_rate=mean([x["phi"].get("toward",0.0) for x in xs]),
                      away_rate=mean([x["phi"].get("away",0.0) for x in xs]),wall_rate=mean([x["phi"].get("wall",0.0) for x in xs]),
                      visit_severity=mean([x.get("visit_severity",0.0) for x in xs]))
    return out

def pair_dphi(p,k):
    return p["pos"]["phi"].get(k,0.0)-p["neg"]["phi"].get(k,0.0) if k!="visit" else p["pos"].get("visit_severity",0.0)-p["neg"].get("visit_severity",0.0)

def bucket_pressure_contribution(pairs):
    total=max(1,len(pairs)); by=defaultdict(list); out={}
    for p in pairs: by[p["bucket"]].append(p)
    for b,ps in sorted(by.items()):
        pi=len(ps)/total; valid=mean([float(p.get("valid",False)) for p in ps]); m=mean([p["margin"] for p in ps])
        d={k:mean([pair_dphi(p,k) for p in ps]) for k in PRIMITIVES}
        contrib={"P"+k[0].upper():pi*valid*m*d[k] for k in PRIMITIVES}
        out[b]=dict(bucket=b,n=len(ps),bucket_ratio=pi,valid_rate=valid,m_mean=m,dT=d["toward"],dA=d["away"],dW=d["wall"],dV=d["visit"],**contrib)
    return out

PRESSURE_NOTES = {
    "toward":"desired positive",
    "away":"desired near-neutral or mildly negative",
    "wall":"desired negative",
    "visit":"desired mildly negative",
    "toward-away":"desired positive",
    "away-wall":"desired positive",
    "absWall-absVisit":"desired positive",
}

def observed_pressure_from_contrib(contrib):
    obs={"global":{k:0.0 for k in PRIMITIVES},"local":{k:0.0 for k in PRIMITIVES},"total":{k:0.0 for k in PRIMITIVES}}
    for b,r in contrib.items():
        dest="local" if b in ("progress","visit","wall","stuck_escape") else "global"
        for k,pk in [("toward","PT"),("away","PA"),("wall","PW"),("visit","PV")]:
            obs[dest][k] += r.get(pk,0.0); obs["total"][k] += r.get(pk,0.0)
    derived={
        "toward-away": obs["total"]["toward"]-obs["total"]["away"],
        "away-wall": obs["total"]["away"]-obs["total"]["wall"],
        "absWall-absVisit": abs(obs["total"]["wall"])-abs(obs["total"]["visit"]),
    }
    rows={}
    for k in PRIMITIVES:
        rows[k]=dict(primitive=k,global_value=obs["global"][k],local_value=obs["local"][k],total=obs["total"][k],target_note=PRESSURE_NOTES[k])
    for k,v in derived.items():
        rows[k]=dict(primitive=k,global_value=None,local_value=None,total=v,target_note=PRESSURE_NOTES[k])
    return dict(by_scope=obs,rows=rows,notes=pressure_notes(rows))

def pressure_notes(rows):
    notes=[]
    if rows["toward-away"]["total"] < 0.24:
        notes.append("toward-away is below desired semantic separation; check reward probe before tuning.")
    if rows["away-wall"]["total"] < 0.24:
        notes.append("away-wall is below desired margin-like separation in semantic pressure; check reward probe before tuning.")
    if rows["absWall-absVisit"]["total"] < 0.12:
        notes.append("absWall-absVisit is small; visit and wall pressures may be close.")
    if rows["wall"]["total"] >= 0:
        notes.append("wall semantic pressure is not negative; inspect wall reward and greedy wall tail.")
    if rows["visit"]["total"] >= 0:
        notes.append("visit semantic pressure is not negative; inspect visit_mean after training.")
    return notes

def summarize_local_bucket_context_distribution(pairs):
    grouped=defaultdict(list)
    for p in pairs:
        if p["bucket"] in ("progress","visit","wall","stuck_escape"):
            grouped[(p["bucket"],p.get("context","") or "none")].append(p)
    out={}
    for (b,ctx),ps in sorted(grouped.items()):
        total_b=max(1,sum(1 for p in pairs if p["bucket"]==b)); ms=[p["margin"] for p in ps]
        out[f"{b}|{ctx}"]=dict(bucket=b,context=ctx,n=len(ps),ratio=len(ps)/total_b,
            pos_toward_rate=mean([p["pos"]["phi"].get("toward",0.0) for p in ps]),pos_away_rate=mean([p["pos"]["phi"].get("away",0.0) for p in ps]),pos_wall_rate=mean([p["pos"]["phi"].get("wall",0.0) for p in ps]),
            neg_toward_rate=mean([p["neg"]["phi"].get("toward",0.0) for p in ps]),neg_away_rate=mean([p["neg"]["phi"].get("away",0.0) for p in ps]),neg_wall_rate=mean([p["neg"]["phi"].get("wall",0.0) for p in ps]),
            pos_visit_sev_mean=mean([p["pos"].get("visit_severity",0.0) for p in ps]),neg_visit_sev_mean=mean([p["neg"].get("visit_severity",0.0) for p in ps]),
            pos_wall_sev_mean=mean([p["pos"].get("wall_severity",0.0) for p in ps]),neg_wall_sev_mean=mean([p["neg"].get("wall_severity",0.0) for p in ps]),
            m_mean=mean(ms),dT=mean([pair_dphi(p,"toward") for p in ps]),dA=mean([pair_dphi(p,"away") for p in ps]),dW=mean([pair_dphi(p,"wall") for p in ps]),dV=mean([pair_dphi(p,"visit") for p in ps]),valid_rate=mean([float(p.get("valid",False)) for p in ps]))
    return out

def margin_tuning_advice(obs):
    return [dict(condition="pre-training pressure diagnosis", suggestion="Use pressure to inspect semantic push direction, then use post-training reward probes and greedy behavior for tuning.")]

def protocol_compliance_audit(report):
    issues=[]
    sp=report.get("sample_profile_distribution",{})
    exp=[r for r in sp.values() if r.get("sample_type")=="explore_timeout"]
    if not exp:
        issues.append(dict(scope="explore_timeout",status="FAIL",message="no explore_timeout samples generated."))
    else:
        wall_max=max(r.get("wall_sev_max",0.0) or 0.0 for r in exp)
        if wall_max <= EPS:
            issues.append(dict(scope="explore_timeout",status="FAIL",message="explore_timeout is wall-free; prompt requires mild wall coverage."))
    stuck=[r for r in sp.values() if r.get("sample_type")=="stuck_timeout"]
    if not stuck or sum(r["n"] for r in stuck)==0:
        issues.append(dict(scope="stuck_timeout",status="FAIL",message="no stuck_timeout samples generated."))
    local=report.get("local_bucket_context_distribution",{})
    for key,r in local.items():
        if r.get("bucket")=="visit" and abs(r.get("dV",0.0)) > 0.35 + 1e-6:
            issues.append(dict(scope=key,status="FAIL",message=f"visit local dV={r.get('dV')} exceeds adjacent calibration bound 0.35."))
        if r.get("bucket")=="wall" and abs(r.get("dV",0.0)) > 1e-6:
            issues.append(dict(scope=key,status="FAIL",message="wall bucket leaks visit pressure."))
    return {"status":"PASS" if not issues else "FAIL", "issues":issues}

def build_reports(args,pairs,elements,maze_records):
    bucket_contrib=bucket_pressure_contribution(pairs)
    observed=observed_pressure_from_contrib(bucket_contrib)
    sample_profile=summarize_sample_profile_distribution(elements)
    primitive_profile=summarize_primitive_distribution_by_profile(elements)
    local_ctx=summarize_local_bucket_context_distribution(pairs)
    margin_summary=margin_configuration_summary(args)
    by=defaultdict(list)
    for p in pairs: by[p["bucket"]].append(p)
    pair_bucket_report={}
    for b,ps in sorted(by.items()):
        ms=[p["margin"] for p in ps]
        pair_bucket_report[b]=dict(n=len(ps),m_mean=mean(ms),m_p10=qtile(ms,.1),m_p50=qtile(ms,.5),m_p90=qtile(ms,.9),m_min=min(ms) if ms else None,m_max=max(ms) if ms else None,valid_rate=mean([float(p["valid"]) for p in ps]),nearMin=mean([float(p["near_min"]) for p in ps]),nearMax=mean([float(p["near_max"]) for p in ps]))
    report=dict(
        maze_generation_report=summarize_maze_report(args,maze_records),
        sample_profile_distribution=sample_profile,
        primitive_distribution_by_profile=primitive_profile,
        local_bucket_context_distribution=local_ctx,
        bucket_pressure_contribution=bucket_contrib,
        margin_configuration_summary=margin_summary,
        primitive_pressure_design_note=dict(note="Primitive pressure is computed after dataset construction from actual bucket ratios, valid rates, margins and semantic deltas. It is not the same quantity as margin m. Use pressure to inspect semantic push direction, then use post-training reward probes to decide tuning."),
        observed_primitive_pressure=observed,
        pair_bucket_report=pair_bucket_report,
        margin_tuning_advice=margin_tuning_advice(observed),
    )
    report["protocol_compliance_audit"]=protocol_compliance_audit(report)
    return report

def print_reports(args,report):
    print_table("[Maze Generation Report]", [("difficulty",10,"s"),("target_ratio",10,".3f"),("actual_ratio",10,".3f"),("n",6,"d"),("bfs_len_mean",12,".3f"),("bfs_len_min",10,"d"),("bfs_len_p50",10,".3f"),("bfs_len_max",10,"d"),("retry_mean",10,".2f")], list(report["maze_generation_report"].values()))
    print_table("[Margin Configuration Summary]", [("bucket",18,"s"),("min",8,".3f"),("max",8,".3f"),("scale",8,".3f"),("pair_count",10,"d")], list(report["margin_configuration_summary"].values()))
    print("\n[Primitive Pressure Design Note]")
    print("Primitive pressure is computed after dataset construction from actual bucket ratios, valid rates, margins and semantic deltas.")
    print("It is not the same quantity as margin m.")
    print("Use pressure to inspect semantic push direction, then use post-training reward probes to decide tuning.")
    print_table("[Sample Profile Distribution]", [("sample_type",16,"s"),("profile",30,"s"),("n",6,"d"),("ratio",8,".3f"),("succ_rate",8,"pct"),("wall_sev_mean",10,".3f"),("wall_sev_min",10,".3f"),("wall_sev_p50",10,".3f"),("wall_sev_max",10,".3f"),("visit_sev_mean",10,".3f"),("visit_sev_min",10,".3f"),("visit_sev_p50",10,".3f"),("visit_sev_max",10,".3f"),("unique_mean",9,".2f"),("coverage_bfs_mean",9,".2f"),("gap_norm_mean",9,".3f"),("final_dist_mean",9,".2f")], list(report["sample_profile_distribution"].values()))
    print_table("[Primitive Distribution by Sample Profile]", [("sample_type",16,"s"),("profile",30,"s"),("toward_rate",10,".3f"),("away_rate",10,".3f"),("wall_rate",10,".3f"),("visit_severity",12,".3f")], list(report["primitive_distribution_by_profile"].values()))
    print_table("[Local Bucket Context Distribution]", [("bucket",14,"s"),("context",28,"s"),("n",7,"d"),("ratio",8,".3f"),("pos_toward_rate",8,".3f"),("pos_away_rate",8,".3f"),("pos_wall_rate",8,".3f"),("neg_toward_rate",8,".3f"),("neg_away_rate",8,".3f"),("neg_wall_rate",8,".3f"),("pos_visit_sev_mean",8,".3f"),("neg_visit_sev_mean",8,".3f"),("m_mean",8,".3f"),("dT",8,".3f"),("dA",8,".3f"),("dW",8,".3f"),("dV",8,".3f"),("valid_rate",8,"pct")], list(report["local_bucket_context_distribution"].values()))
    print_table("[Bucket Primitive Pressure Contribution]", [("bucket",18,"s"),("n",7,"d"),("bucket_ratio",12,".4f"),("valid_rate",10,".3f"),("m_mean",8,".3f"),("dT",8,".3f"),("dA",8,".3f"),("dW",8,".3f"),("dV",8,".3f"),("PT",8,".3f"),("PA",8,".3f"),("PW",8,".3f"),("PV",8,".3f")], list(report["bucket_pressure_contribution"].values()))
    rows=[]
    for k in ["toward","away","wall","visit","toward-away","away-wall","absWall-absVisit"]:
        r=report["observed_primitive_pressure"]["rows"][k]
        rows.append({"primitive":k,"global":r.get("global_value"),"local":r.get("local_value"),"total":r.get("total"),"target_note":r.get("target_note")})
    print_table("[Observed Primitive Pressure]", [("primitive",18,"s"),("global",10,".3f"),("local",10,".3f"),("total",10,".3f"),("target_note",48,"s")], rows)
    for w in report["observed_primitive_pressure"].get("notes",[]): print(f"[PRIMITIVE PRESSURE NOTE] {w}")
    audit=report.get("protocol_compliance_audit",{})
    print("\n[Protocol Compliance Audit]")
    print(f"status: {audit.get('status','UNKNOWN')}")
    for issue in audit.get("issues",[]): print(f"[{issue.get('status','FAIL')}] {issue.get('scope','')}: {issue.get('message','')}")

# ---------------- model / train / probes ----------------

class RewardModel(nn.Module):
    def __init__(self,n=8):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(12,32,kernel_size=3,padding=1), nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3,padding=1), nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64*n*n,128), nn.ReLU(), nn.Linear(128,1), nn.Tanh())
    def forward(self,x): return self.net(x).squeeze(-1)

class PairDataset(Dataset):
    def __init__(self,pairs): self.pairs=pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self,i): return self.pairs[i]

def collate_pairs(batch): return batch

def score_sequences(model,seqs,device,normalizer):
    tensors=[]; lengths=[]; denoms=[]
    for s in seqs:
        lengths.append(len(s['transitions'])); denoms.append(max(1,s['bfs_len']) if normalizer=='bfs_len' else max(1,len(s['transitions'])))
        for tr in s['transitions']: tensors.append(transition_tensor(tr))
    if not tensors: return torch.zeros(len(seqs),device=device)
    r=model(torch.stack(tensors).to(device)); out=[]; idx=0
    for L,D in zip(lengths,denoms):
        out.append(r[idx:idx+L].sum()/float(D) if L else torch.tensor(0.0,device=device)); idx+=L
    return torch.stack(out)

def eval_epoch(model,loader,device,args):
    model.eval(); losses=[]; acc=[]
    with torch.no_grad():
        for batch in maybe_tqdm(loader,args,desc="Validate RM",leave=False):
            sp=score_sequences(model,[p['pos'] for p in batch],device,args.score_normalizer); sn=score_sequences(model,[p['neg'] for p in batch],device,args.score_normalizer); m=torch.tensor([p['margin'] for p in batch],dtype=torch.float32,device=device)
            diff=sp-sn; loss=F.softplus(-(diff-m)).mean(); losses.append(loss.item()); acc.append((diff>0).float().mean().item())
    return mean(losses),mean(acc)

def plot_training(hist,path):
    plt.figure(figsize=(12,4)); plt.subplot(1,2,1); plt.plot(hist['train_loss'],label='train'); plt.plot(hist['val_loss'],label='val'); plt.title('BTL loss'); plt.legend(); plt.subplot(1,2,2); plt.plot(hist['train_acc'],label='train'); plt.plot(hist['val_acc'],label='val'); plt.title('Preference accuracy'); plt.legend(); plt.tight_layout(); plt.savefig(path,dpi=150); plt.close()

def train_rm(args,pairs,out_dir,device):
    if not pairs: raise RuntimeError("No pairs were generated; cannot train RM")
    random.shuffle(pairs); nv=max(1,int(0.15*len(pairs))); val=pairs[:nv]; train=pairs[nv:] or pairs
    train_loader=DataLoader(PairDataset(train),batch_size=args.rm_batch_size,shuffle=True,collate_fn=collate_pairs)
    val_loader=DataLoader(PairDataset(val),batch_size=args.rm_batch_size,shuffle=False,collate_fn=collate_pairs)
    model=RewardModel(args.grid_size).to(device); opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    hist={"train_loss":[],"val_loss":[],"train_acc":[],"val_acc":[]}; best_loss=1e9; best_acc=-1
    paths={k:out_dir/f"{VERSION}_reward_model_{k}.pt" for k in ['best_loss','best_acc','best_contextual_gap','best_pure_argmax','best_low_saturation','last']}
    for ep in maybe_tqdm(range(1,args.rm_epochs+1),args,desc="RM epochs",total=args.rm_epochs):
        model.train(); ls=[]; ac=[]
        for batch in maybe_tqdm(train_loader,args,desc=f"RM train ep {ep}",leave=False):
            sp=score_sequences(model,[p['pos'] for p in batch],device,args.score_normalizer); sn=score_sequences(model,[p['neg'] for p in batch],device,args.score_normalizer); m=torch.tensor([p['margin'] for p in batch],dtype=torch.float32,device=device)
            diff=sp-sn; loss=F.softplus(-(diff-m)).mean(); opt.zero_grad(); loss.backward(); opt.step(); ls.append(loss.item()); ac.append((diff.detach()>0).float().mean().item())
        tl,ta=mean(ls),mean(ac); vl,va=eval_epoch(model,val_loader,device,args)
        hist['train_loss'].append(tl); hist['train_acc'].append(ta); hist['val_loss'].append(vl); hist['val_acc'].append(va)
        print(f"[RM ep {ep}/{args.rm_epochs}] train_loss={tl:.4f} val_loss={vl:.4f} train_acc={ta:.4f} val_acc={va:.4f}")
        if vl<best_loss: best_loss=vl; torch.save(model.state_dict(),paths['best_loss'])
        if va>best_acc: best_acc=va; torch.save(model.state_dict(),paths['best_acc'])
    torch.save(model.state_dict(),paths['last'])
    for k in ['best_contextual_gap','best_pure_argmax','best_low_saturation']: torch.save(model.state_dict(),paths[k])
    plot_training(hist,out_dir/f"{VERSION}_training_curves.png")
    save_json(out_dir/f"{VERSION}_reward_model_history.json",dict(history=hist,best_val_loss=best_loss,best_val_acc=best_acc,paths={k:str(v) for k,v in paths.items()}))
    return model

def reward_tr(model,tr,device):
    with torch.no_grad(): return float(model(transition_tensor(tr).unsqueeze(0).to(device)).cpu().item())

def collect_primitive_rewards(model,args,device,primitive,n=100):
    vals=[]
    for _ in maybe_tqdm(range(n),args,desc=f"Primitive probe {primitive}",leave=False):
        maze,start,goal,dist,path,meta=generate_maze(args,choose_target_difficulty(args))
        cells=[(r,c) for r in range(args.grid_size) for c in range(args.grid_size) if maze[r,c]==0 and dist[r,c]<INF]
        random.shuffle(cells)
        if primitive in ("stuck_escape","stuck_loop"):
            got=gen_local_stuck_escape_pair(maze,start,goal,dist,path,args)
            if got:
                seq=got[0] if primitive=="stuck_escape" else got[1]
                vals.append(reward_tr(model,seq['transitions'][0],device))
            continue
        for p in cells[:40]:
            toward,away,walls=action_candidates(maze,p,dist,args.grid_size)
            visits=np.zeros((args.grid_size,args.grid_size),dtype=np.int32); visits[p]=1
            if primitive=='toward' and toward: a=random.choice(toward); bias='toward'
            elif primitive=='away' and away: a=random.choice(away); bias='away'
            elif primitive=='wall' and walls: a=random.choice(walls); bias='away'
            elif primitive=='visit' and (toward or away): visits[p]=random.choice([3,4,5]); a=random.choice(toward+away); bias='toward' if a in toward else 'away'
            else: continue
            seq=make_segment(maze,goal,dist,p,visits,a,args,primitive,bias,1,primitive); vals.append(reward_tr(model,seq['transitions'][0],device)); break
    return vals

def reward_stats(vals):
    return dict(n=len(vals),min=min(vals) if vals else None,p10=qtile(vals,.1),p50=qtile(vals,.5),p90=qtile(vals,.9),max=max(vals) if vals else None,mean=mean(vals),std=float(np.std(vals)) if vals else float('nan'),pct_neg_one=mean([float(v<=-0.98) for v in vals]) if vals else float('nan'),pct_pos_one=mean([float(v>=0.98) for v in vals]) if vals else float('nan'))

def primitive_reward_report(model,args,device):
    names=['toward','away','wall','visit','stuck_escape','stuck_loop']
    out={p:reward_stats(collect_primitive_rewards(model,args,device,p,args.debug_probe_mazes*4)) for p in names}
    out['gaps']=dict(toward_mean_minus_away_mean=out['toward']['mean']-out['away']['mean'],away_mean_minus_wall_mean=out['away']['mean']-out['wall']['mean'],stuck_escape_mean_minus_stuck_loop_mean=out['stuck_escape']['mean']-out['stuck_loop']['mean'],toward_p50_minus_away_p50=out['toward']['p50']-out['away']['p50'],away_p50_minus_wall_p50=out['away']['p50']-out['wall']['p50'],wall_pct_pos_one=out['wall']['pct_pos_one'],visit_mean=out['visit']['mean'])
    warns=[]
    if out['away']['mean']>0.25: warns.append('away_mean > 0.25')
    if out['away']['p50']>0.50: warns.append('away_p50 > 0.50')
    if out['visit']['mean']>0: warns.append('visit_mean > 0')
    if out['stuck_escape']['mean']<=out['stuck_loop']['mean']: warns.append('stuck_escape_mean <= stuck_loop_mean')
    if out['wall']['pct_pos_one']>0.03: warns.append('wall_pct_pos_one > 0.03')
    if out['toward']['mean']<=out['away']['mean']: warns.append('toward_mean <= away_mean')
    if out['away']['mean']<=out['wall']['mean']: warns.append('away_mean <= wall_mean')
    out['warnings']=warns
    return out

def contextual_probe(model,args,device,n=None):
    old=args.debug_probe_mazes
    if n is not None: args.debug_probe_mazes=n
    cats={'toward_no_visit':('toward',1),'away_no_visit':('away',1),'toward_visited':('toward',2),'away_visited':('away',2),'toward_high_visit':('toward',4),'away_high_visit':('away',4),'wall_first':('wall',1),'wall_repeated':('wall',4)}
    out={}
    for name,(direction,vis) in maybe_tqdm(list(cats.items()),args,desc="Contextual probe"):
        vals=[]
        for _ in range(args.debug_probe_mazes):
            maze,start,goal,dist,path,meta=generate_maze(args,choose_target_difficulty(args)); cells=[(r,c) for r in range(args.grid_size) for c in range(args.grid_size) if maze[r,c]==0 and dist[r,c]<INF]; random.shuffle(cells)
            for p in cells[:40]:
                toward,away,walls=action_candidates(maze,p,dist,args.grid_size)
                if direction=='toward' and toward: a=random.choice(toward); bias='toward'
                elif direction=='away' and away: a=random.choice(away); bias='away'
                elif direction=='wall' and walls: a=random.choice(walls); bias='away'
                else: continue
                visits=np.zeros((args.grid_size,args.grid_size),dtype=np.int32); visits[p]=vis
                seq=make_segment(maze,goal,dist,p,visits,a,args,name,bias,1,name); vals.append(reward_tr(model,seq['transitions'][0],device)); break
        out[name]=reward_stats(vals)
    for name,side in [('stuck_escape',0),('stuck_loop',1)]:
        vals=[]
        for _ in maybe_tqdm(range(args.debug_probe_mazes),args,desc=f"Context {name}",leave=False):
            maze,start,goal,dist,path,meta=generate_maze(args,choose_target_difficulty(args))
            got=gen_local_stuck_escape_pair(maze,start,goal,dist,path,args)
            if got:
                seq=got[side]
                vals.append(reward_tr(model,seq['transitions'][0],device))
        out[name]=reward_stats(vals)
    args.debug_probe_mazes=old
    out['gaps']=dict(toward_no_visit_minus_away_no_visit=out['toward_no_visit']['mean']-out['away_no_visit']['mean'],toward_visited_minus_away_visited=out['toward_visited']['mean']-out['away_visited']['mean'],toward_high_visit_minus_away_high_visit=out['toward_high_visit']['mean']-out['away_high_visit']['mean'],stuck_escape_minus_stuck_loop=out['stuck_escape']['mean']-out['stuck_loop']['mean'],wall_first_minus_wall_repeated=out['wall_first']['mean']-out['wall_repeated']['mean'])
    return out

def action_infos_for_state(model,maze,goal,dist,pos,visits,args,device):
    infos=[]; xs=[]
    for a in range(4):
        seq=make_segment(maze,goal,dist,pos,visits,a,args,'step','toward',1,'greedy')
        tr=seq['transitions'][0]; infos.append((a,tr,seq)); xs.append(transition_tensor(tr))
    with torch.no_grad(): rs=model(torch.stack(xs).to(device)).detach().cpu().numpy().tolist()
    rows=[]
    for (a,tr,seq),r in zip(infos,rs):
        rows.append(dict(action_idx=a,action=ACTION_NAMES[a],reward=float(r),is_wall=bool(tr['is_wall']),next_pos=list(tr['next_pos']),delta_bfs=int(tr['delta_bfs']),visit_count_after=int(tr['visit_after']),transition=tr))
    rows_sorted=sorted(rows,key=lambda z:z['reward'],reverse=True)
    rank={id(r):i+1 for i,r in enumerate(rows_sorted)}
    for r in rows: r['rank']=rank[id(r)]
    return rows

def choose_greedy_action(rows,args,mode):
    if mode=='bfs_tiebreak':
        mx=max(r['reward'] for r in rows); cand=[r for r in rows if mx-r['reward']<=args.rm_greedy_tie_eps]
        cand.sort(key=lambda r:(r['delta_bfs'],-r['reward']))
        return cand[0]
    return max(rows,key=lambda r:r['reward'])

def greedy_rollouts(model,args,device,n_mazes,mode):
    roll=[]; counts=defaultdict(float)
    for mi in maybe_tqdm(range(n_mazes),args,desc=f"Greedy probe {mode}"):
        target=choose_target_difficulty(args)
        maze,start,goal,dist,path,meta=generate_maze(args,target); pos=start; visits=np.zeros((args.grid_size,args.grid_size),dtype=np.int32); visits[pos]=1; traj=[]; success=False; wall_hits=0
        for t in range(args.max_steps):
            cur=int(dist[pos]) if dist[pos]<INF else 999
            rows=action_infos_for_state(model,maze,goal,dist,pos,visits,args,device)
            chosen=choose_greedy_action(rows,args,mode)
            if chosen['is_wall']: wall_hits+=1; counts['wall']+=1
            elif chosen['delta_bfs']<0: counts['toward']+=1
            else: counts['away']+=1
            rewards=[r['reward'] for r in rows]
            if max(rewards)-min(rewards)<0.05: counts['saturated']+=1
            counts['steps']+=1
            traj.append(dict(t=t,pos=list(pos),bfs_dist=cur,chosen_action=chosen['action'],chosen_reward=chosen['reward'],chosen_is_wall=chosen['is_wall'],chosen_delta_bfs=chosen['delta_bfs'],visit_count_after=chosen['visit_count_after'],all_actions_saturated=(max(rewards)-min(rewards)<0.05),actions={r['action']:{k:v for k,v in r.items() if k not in ('transition','action_idx')} for r in rows}))
            pos=tuple(chosen['next_pos']); visits[pos]+=1
            if pos==goal: success=True; break
        final=int(dist[pos]) if dist[pos]<INF else 999; outcome='success' if success else ('wall_timeout' if wall_hits>0 else 'stuck_timeout')
        roll.append(dict(maze_id=mi,difficulty=meta.get('actual',target),bfs_len=meta.get('bfs_len',int(dist[start])),success=success,outcome=outcome,steps=len(traj),wall_hits=wall_hits,bfs_gap=(len(traj)-int(dist[start]) if success else final+len(traj)),trajectory=traj))
    return summarize_rollouts(roll, counts)

def summarize_rollouts(roll, counts=None):
    if counts is None:
        counts=defaultdict(float)
        for ro in roll:
            for st in ro.get('trajectory',[]):
                if st.get('chosen_is_wall'): counts['wall']+=1
                elif st.get('chosen_delta_bfs',0)<0: counts['toward']+=1
                else: counts['away']+=1
                if st.get('all_actions_saturated'): counts['saturated']+=1
                counts['steps']+=1
    total=max(1.0,counts['steps'])
    summary=dict(n=len(roll),success=mean([float(r['success']) for r in roll]),wall_timeout=mean([float(r['outcome']=='wall_timeout') for r in roll]),stuck_timeout=mean([float(r['outcome']=='stuck_timeout') for r in roll]),explore_timeout=mean([float(r['outcome']=='explore_timeout') for r in roll]),avg_bfs_gap=mean([r['bfs_gap'] for r in roll]),avg_steps=mean([r['steps'] for r in roll]),avg_wall_hits=mean([r['wall_hits'] for r in roll]))
    action=dict(toward_chosen_rate=counts['toward']/total,away_chosen_rate=counts['away']/total,wall_chosen_rate=counts['wall']/total,all_actions_saturated_rate=counts['saturated']/total)
    return dict(summary=summary,action_ranking=action,rollouts=roll)

def difficulty_breakdown_from_rollouts(rollouts):
    out={}
    for diff in ['easy','medium','hard']:
        arr=[r for r in rollouts if r.get('difficulty')==diff]
        c=defaultdict(float)
        for ro in arr:
            for st in ro.get('trajectory',[]):
                if st.get('chosen_is_wall'): c['wall']+=1
                elif st.get('chosen_delta_bfs',0)<0: c['toward']+=1
                else: c['away']+=1
                if st.get('all_actions_saturated'): c['saturated']+=1
                c['steps']+=1
        denom=max(1.0,c['steps'])
        out[diff]=dict(difficulty=diff,n=len(arr),success=mean([float(r['success']) for r in arr]),wall_timeout=mean([float(r['outcome']=='wall_timeout') for r in arr]),stuck_timeout=mean([float(r['outcome']=='stuck_timeout') for r in arr]),explore_timeout=mean([float(r['outcome']=='explore_timeout') for r in arr]),avg_bfs_gap=mean([r['bfs_gap'] for r in arr]),wall_rate=c['wall']/denom,away_rate=c['away']/denom,saturated_rate=c['saturated']/denom)
    return out

def greedy_comparison(model,args,device,n=None):
    nmazes=args.rm_greedy_probe_mazes if n is None else n
    bfs=greedy_rollouts(model,args,device,nmazes,'bfs_tiebreak'); pure=greedy_rollouts(model,args,device,nmazes,'pure_argmax')
    breakdown={'bfs_tiebreak':difficulty_breakdown_from_rollouts(bfs['rollouts']),'pure_argmax':difficulty_breakdown_from_rollouts(pure['rollouts'])}
    return {'bfs_tiebreak':{'summary':bfs['summary'],'action_ranking':bfs['action_ranking']},'pure_argmax':{'summary':pure['summary'],'action_ranking':pure['action_ranking']},'difficulty_breakdown':breakdown,'_rollouts_bfs':bfs['rollouts'],'_rollouts_pure':pure['rollouts']}

def stuck_timeout_analysis(greedy,args):
    rollouts=greedy.get('_rollouts_pure',[])
    stuck=[r for r in rollouts if r.get('outcome')=='stuck_timeout']
    rows=[]
    for ro in maybe_tqdm(stuck,args,desc="Stuck timeout analysis"):
        tail=ro.get('trajectory',[])[-args.tail_window:]
        if not tail: continue
        positions=[tuple(st.get('pos',[0,0])) for st in tail]
        unique_tail=len(set(positions))
        visit_vals=[max(0.0,(st.get('visit_count_after',1)-1.0)) for st in tail]
        away_chosen=[]; toward_avail=[]; escape_avail=[]; escape_chosen=[]; sat=[]
        for st in tail:
            acts=st.get('actions',{})
            chosen_delta=st.get('chosen_delta_bfs',0); chosen_wall=st.get('chosen_is_wall',False)
            away_chosen.append(float((not chosen_wall) and chosen_delta>=0))
            available=[a for a in acts.values() if not a.get('is_wall')]
            tw=[a for a in available if a.get('delta_bfs',0)<0]
            toward_avail.append(float(len(tw)>0))
            cur_v=st.get('visit_count_after',1)
            esc=[a for a in available if a.get('delta_bfs',0)<0 or a.get('visit_count_after',cur_v) < cur_v]
            escape_avail.append(float(len(esc)>0))
            escape_chosen.append(float((not chosen_wall) and (chosen_delta<0)))
            sat.append(float(st.get('all_actions_saturated',False)))
        rows.append(dict(loop_length=len(tail),unique_tail=unique_tail,tail_visit_severity=mean(visit_vals),tail_away_chosen_rate=mean(away_chosen),tail_toward_available_rate=mean(toward_avail),tail_escape_available_rate=mean(escape_avail),tail_escape_chosen_rate=mean(escape_chosen),tail_all_actions_saturated_rate=mean(sat)))
    out=dict(n_stuck=len(stuck),avg_loop_length=mean([r['loop_length'] for r in rows]),avg_unique_tail=mean([r['unique_tail'] for r in rows]),avg_tail_visit_severity=mean([r['tail_visit_severity'] for r in rows]),avg_tail_away_chosen_rate=mean([r['tail_away_chosen_rate'] for r in rows]),avg_tail_toward_available_rate=mean([r['tail_toward_available_rate'] for r in rows]),avg_tail_escape_available_rate=mean([r['tail_escape_available_rate'] for r in rows]),avg_tail_escape_chosen_rate=mean([r['tail_escape_chosen_rate'] for r in rows]),avg_tail_all_actions_saturated_rate=mean([r['tail_all_actions_saturated_rate'] for r in rows]),rows=rows[:50],interpretation=[])
    if out['n_stuck']>0 and out['avg_tail_toward_available_rate']>0.4 and out['avg_tail_escape_chosen_rate']<0.3: out['interpretation'].append('tail_toward_available_rate is high but escape_chosen_rate is low: RM still does not lift escape/toward enough.')
    if out['n_stuck']>0 and out['avg_tail_all_actions_saturated_rate']>0.2: out['interpretation'].append('tail_all_actions_saturated_rate is high: high-visit action separation is insufficient.')
    if out['n_stuck']>0 and out['avg_tail_away_chosen_rate']>0.25: out['interpretation'].append('tail_away_chosen_rate is high: away reward is likely overestimated.')
    return out

def failure_tail_action_rank_report(greedy,args):
    rows=[]
    for ro in greedy.get('_rollouts_pure',[]):
        if ro.get('success'): continue
        tail=ro.get('trajectory',[])[-args.tail_window:]
        for st in tail:
            acts=list(st.get('actions',{}).values())
            nonwall=[a for a in acts if not a.get('is_wall')]
            toward=[a for a in nonwall if a.get('delta_bfs',0)<0]
            escape=[a for a in nonwall if a.get('delta_bfs',0)<0 or a.get('visit_count_after',999) < st.get('visit_count_after',999)]
            walls=[a for a in acts if a.get('is_wall')]
            away=[a for a in nonwall if a.get('delta_bfs',0)>=0]
            rows.append(dict(failure_type=ro.get('outcome'),chosen_reward=st.get('chosen_reward'),best_toward_reward=max([a['reward'] for a in toward],default=float('nan')),best_escape_reward=max([a['reward'] for a in escape],default=float('nan')),wall_reward_max=max([a['reward'] for a in walls],default=float('nan')),away_reward_max=max([a['reward'] for a in away],default=float('nan')),toward_available=float(bool(toward)),escape_available=float(bool(escape)),chosen_is_wall=float(st.get('chosen_is_wall',False)),chosen_is_away=float((not st.get('chosen_is_wall',False)) and st.get('chosen_delta_bfs',0)>=0),chosen_is_escape=float((not st.get('chosen_is_wall',False)) and st.get('chosen_delta_bfs',0)<0),saturated=float(st.get('all_actions_saturated',False))))
    grouped={}
    for ft in sorted(set(r['failure_type'] for r in rows)):
        arr=[r for r in rows if r['failure_type']==ft]
        grouped[ft]=dict(failure_type=ft,n_steps=len(arr),chosen_reward_mean=mean([r['chosen_reward'] for r in arr]),best_toward_reward_mean=mean([r['best_toward_reward'] for r in arr]),best_escape_reward_mean=mean([r['best_escape_reward'] for r in arr]),wall_reward_max_mean=mean([r['wall_reward_max'] for r in arr]),away_reward_max_mean=mean([r['away_reward_max'] for r in arr]),toward_available_rate=mean([r['toward_available'] for r in arr]),escape_available_rate=mean([r['escape_available'] for r in arr]),chosen_is_wall_rate=mean([r['chosen_is_wall'] for r in arr]),chosen_is_away_rate=mean([r['chosen_is_away'] for r in arr]),chosen_is_escape_rate=mean([r['chosen_is_escape'] for r in arr]),saturated_rate=mean([r['saturated'] for r in arr]))
    return {"by_failure_type": grouped, "rows": rows[:200]}

def dual_gate_tuning_advice(report,prim,greedy,stuck):
    pure=greedy['pure_argmax']; pure_summary=pure['summary']; pure_action=pure['action_ranking']
    out=[]
    wall_avg_solved = prim['wall']['mean'] <= -0.90 and prim['wall']['pct_pos_one'] <= 0.03
    wall_solved = wall_avg_solved and pure_action.get('wall_chosen_rate',1.0) <= 0.05
    if wall_solved:
        out.append(dict(condition='wall is solved on average',suggestion='avoid increasing wall margin.'))
    elif wall_avg_solved and pure_action.get('wall_chosen_rate',0.0)>0.10:
        out.append(dict(condition='pure_argmax_wall_rate > 0.10 while average wall reward is solved',suggestion='wall failure likely comes from tail states / saturation, not average wall reward. Add hard-tail counterfactuals or enlarge data, not wall margin.'))
    if pure_summary.get('stuck_timeout',0.0) >= 0.10:
        out.append(dict(condition='pure_argmax_stuck_timeout >= 0.10',suggestion='increase pairs_stuck_escape or add failure-mined stuck_escape pairs.'))
    if pure_action.get('all_actions_saturated_rate',0.0) > 0.20:
        out.append(dict(condition='all_actions_saturated_rate > 0.20',suggestion='increase diverse progress/stuck_escape samples; avoid pushing all rewards to tanh extremes.'))
    if prim['visit']['mean'] > 0:
        out.append(dict(condition='visit_mean > 0',suggestion='visit mean is positive; increase adjacent-band visit pairs slightly, but do not compare clean vs extreme visit directly.'))
    if not out:
        out.append(dict(condition='dual gate passed',suggestion='keep wall stable; inspect stuck tail metrics before further tuning.'))
    return out

def collapse_check(model,args,device):
    vals=[]
    for p in ['toward','away','wall','visit']: vals+=collect_primitive_rewards(model,args,device,p,20)
    raw=float(np.std(vals)) if vals else 0.0
    return dict(raw_rm_std=raw,rm_scale=1.0/max(raw,EPS),status='PASS' if raw>=0.05 else 'FAIL')

def print_probe_summary(ctx,greedy,prim,stuck=None,advice=None,tail=None):
    print_table("[RM Primitive Reward Score Report]", [("primitive",14,"s"),("n",6,"d"),("min",8,".3f"),("p10",8,".3f"),("p50",8,".3f"),("p90",8,".3f"),("max",8,".3f"),("mean",8,".3f"),("std",8,".3f"),("pct_neg_one",10,"pct"),("pct_pos_one",10,"pct")], [{"primitive":p,**prim[p]} for p in ['toward','away','wall','visit','stuck_escape','stuck_loop']])
    for w in prim.get('warnings',[]): print(f"[PRIMITIVE REWARD WARNING] {w}")
    print_table("[Contextual Probe]", [("category",28,"s"),("n",6,"d"),("mean",9,".4f"),("p50",9,".4f"),("p10",9,".4f"),("p90",9,".4f")], [{"category":k,**ctx[k]} for k in ['toward_no_visit','away_no_visit','toward_visited','away_visited','toward_high_visit','away_high_visit','wall_first','wall_repeated','stuck_escape','stuck_loop']])
    gc_rows=[]
    for k in ['success','wall_timeout','stuck_timeout','explore_timeout','avg_bfs_gap']:
        gc_rows.append({'metric':k,'bfs_tiebreak':greedy['bfs_tiebreak']['summary'].get(k),'pure_argmax':greedy['pure_argmax']['summary'].get(k)})
    for k in ['wall_chosen_rate','toward_chosen_rate','away_chosen_rate','all_actions_saturated_rate']:
        gc_rows.append({'metric':k,'bfs_tiebreak':greedy['bfs_tiebreak']['action_ranking'].get(k),'pure_argmax':greedy['pure_argmax']['action_ranking'].get(k)})
    print_table("[Greedy Probe Comparison]", [("metric",32,"s"),("bfs_tiebreak",14,".4f"),("pure_argmax",14,".4f")], gc_rows)
    br=[]
    for strategy in ['pure_argmax','bfs_tiebreak']:
        for diff,row in greedy.get('difficulty_breakdown',{}).get(strategy,{}).items():
            br.append({'strategy':strategy, **row})
    print_table("[Greedy Probe Difficulty Breakdown]", [("strategy",14,"s"),("difficulty",10,"s"),("n",6,"d"),("success",9,".4f"),("wall_timeout",12,".4f"),("stuck_timeout",13,".4f"),("explore_timeout",15,".4f"),("avg_bfs_gap",12,".3f"),("wall_rate",10,".4f"),("away_rate",10,".4f"),("saturated_rate",14,".4f")], br)
    if stuck is not None:
        print_table("[Stuck Timeout Analysis]", [("metric",40,"s"),("value",12,".4f")], [{'metric':k,'value':stuck.get(k)} for k in ['n_stuck','avg_loop_length','avg_unique_tail','avg_tail_visit_severity','avg_tail_away_chosen_rate','avg_tail_toward_available_rate','avg_tail_escape_available_rate','avg_tail_escape_chosen_rate','avg_tail_all_actions_saturated_rate']])
        for msg in stuck.get('interpretation',[]): print(f"[STUCK ANALYSIS] {msg}")
    if tail is not None:
        print_table("[Failure Tail Action Rank Report]", [("failure_type",16,"s"),("n_steps",8,"d"),("chosen_reward_mean",18,".4f"),("best_toward_reward_mean",22,".4f"),("best_escape_reward_mean",22,".4f"),("wall_reward_max_mean",20,".4f"),("away_reward_max_mean",20,".4f"),("toward_available_rate",20,".4f"),("escape_available_rate",20,".4f"),("chosen_is_wall_rate",20,".4f"),("chosen_is_away_rate",20,".4f"),("chosen_is_escape_rate",22,".4f"),("saturated_rate",15,".4f")], list(tail.get('by_failure_type',{}).values()))
    if advice is not None:
        print("\n[Dual-Gate Tuning Advice]")
        for a in advice: print(f"{a['condition']}: {a['suggestion']}")

# ---------------- manual test and GIF ----------------

TEST_MAZE_TEXT = \
"""
    ........
    ##.####.
    ...#.#..
    .#S#G..#
    ..#..#..
    .##.###.
    .#...#..
    .#.#....
"""

def parse_manual_maze(text):
    lines=[ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    n=len(lines); maze=np.zeros((n,n),dtype=np.int8); start=goal=None
    for r,line in enumerate(lines):
        for c,ch in enumerate(line):
            if ch=='#': maze[r,c]=1
            elif ch=='S': start=(r,c)
            elif ch=='G': goal=(r,c)
    if start is None or goal is None: raise ValueError('manual maze needs S and G')
    return maze,start,goal

def render_frame(maze,pos,goal,visits,title):
    n=maze.shape[0]
    img=np.ones((n,n,3),dtype=np.float32)
    img[maze==1]=[0.05,0.05,0.05]
    vc=np.minimum(visits,8)/8.0
    img[:,:,1]=np.minimum(img[:,:,1],1.0-0.35*vc)
    img[goal]=[1.0,0.85,0.0]
    img[pos]=[1.0,0.1,0.1]
    fig,ax=plt.subplots(figsize=(5,5))
    ax.imshow(img,interpolation='nearest'); ax.set_xticks(np.arange(-.5,n,1),minor=True); ax.set_yticks(np.arange(-.5,n,1),minor=True); ax.grid(which='minor',color='black',linewidth=1); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title,fontsize=8)
    fig.canvas.draw(); w,h=fig.canvas.get_width_height(); arr=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy(); plt.close(fig); return arr

def save_gif(frames,path,fps=3):
    if not frames: return
    fig=plt.figure(figsize=(5,5)); ax=plt.gca(); ax.axis('off'); im=ax.imshow(frames[0])
    def update(i): im.set_data(frames[i]); return [im]
    ani=animation.FuncAnimation(fig,update,frames=len(frames),interval=int(1000/max(1,fps)),blit=True)
    ani.save(path,writer='pillow'); plt.close(fig)

def run_manual_test_strategy(model,args,device,out_dir,strategy):
    maze,start,goal=parse_manual_maze(TEST_MAZE_TEXT); dist=bfs_distances(maze,goal); pos=start; visits=np.zeros((args.grid_size,args.grid_size),dtype=np.int32); visits[pos]=1
    frames=[]; steps=[]; success=False; wall_hits=0
    for t in maybe_tqdm(range(args.max_steps),args,desc=f"--test {strategy}"):
        cur=int(dist[pos]) if dist[pos]<INF else 999
        rows=action_infos_for_state(model,maze,goal,dist,pos,visits,args,device)
        chosen=choose_greedy_action(rows,args,strategy)
        if chosen['is_wall']: wall_hits+=1
        frames.append(render_frame(maze,pos,goal,visits,f"step={t} action={chosen['action']} reward={chosen['reward']:.3f} is_wall={chosen['is_wall']} delta_bfs={chosen['delta_bfs']} visit={chosen['visit_count_after']} pos={pos} strategy={strategy}"))
        steps.append(dict(t=t,pos=list(pos),bfs_dist=cur,chosen_action=chosen['action'],chosen_reward=chosen['reward'],chosen_is_wall=chosen['is_wall'],chosen_delta_bfs=chosen['delta_bfs'],visit_count_after=chosen['visit_count_after'],actions={r['action']:{k:v for k,v in r.items() if k not in ('transition','action_idx')} for r in rows}))
        pos=tuple(chosen['next_pos']); visits[pos]+=1
        if pos==goal: success=True; break
    metrics=dict(strategy=strategy,success=success,steps=len(steps),wall_hits=wall_hits,final_pos=list(pos),final_bfs_dist=int(dist[pos]) if dist[pos]<INF else 999)
    json_path=out_dir/f"{VERSION}_test_rollout_{strategy}_debug.json"
    gif_path=out_dir/f"{VERSION}_test_rollout_{strategy}.gif"
    save_json(json_path,dict(metrics=metrics,steps=steps))
    save_gif(frames,gif_path)
    print(f"[Save] test json: {json_path}")
    print(f"[Save] test gif : {gif_path}")
    return dict(metrics=metrics,json=str(json_path),gif=str(gif_path))

def run_test_mode(args,device,out_dir):
    rm_path=args.reward_model or str(out_dir/f"{VERSION}_reward_model_best_loss.pt")
    if not Path(rm_path).exists():
        raise SystemExit('[ERROR] --test requires a reward model. Provide --reward-model or train RM first.')
    model=RewardModel(args.grid_size).to(device); model.load_state_dict(torch.load(rm_path,map_location=device)); model.eval()
    strategies=['pure_argmax','bfs_tiebreak'] if args.test_strategy=='both' else [args.test_strategy]
    results={s:run_manual_test_strategy(model,args,device,out_dir,s) for s in strategies}
    save_json(out_dir/f"{VERSION}_test_rollout_summary.json",results)

# ---------------- preset / CLI / main ----------------

def build_argparser():
    p=argparse.ArgumentParser(description=VERSION)
    for k,v in DEFAULTS.items():
        name='--'+k.replace('_','-')
        if isinstance(v,bool): p.add_argument(name,action='store_true' if not v else 'store_false',default=v)
        elif isinstance(v,int): p.add_argument(name,type=int,default=v)
        elif isinstance(v,float): p.add_argument(name,type=float,default=v)
        else: p.add_argument(name,default=v)
    p.add_argument('--reward-model',default=None)
    return p

def apply_preset(args, explicit):
    presets={
        'quick': dict(rm_mazes=50,rm_epochs=2,rm_greedy_probe_mazes=20,debug_probe_mazes=20),
        'default': dict(rm_mazes=1000,rm_epochs=16,rm_greedy_probe_mazes=100,debug_probe_mazes=50),
        'large': dict(rm_mazes=800,rm_epochs=12,rm_greedy_probe_mazes=200,debug_probe_mazes=100),
        'xl': dict(rm_mazes=1200,rm_epochs=14,rm_greedy_probe_mazes=300,debug_probe_mazes=150),
    }
    if args.preset not in presets: raise SystemExit(f"unknown --preset {args.preset}; choose {sorted(presets)}")
    for k,v in presets[args.preset].items():
        if ('--'+k.replace('_','-')) not in explicit:
            setattr(args,k,v)
    return args

def save_reports(out_dir,report):
    for key,filename in [
        ('sample_profile_distribution',f'{VERSION}_sample_profile_distribution.json'),
        ('primitive_distribution_by_profile',f'{VERSION}_primitive_distribution_by_profile.json'),
        ('local_bucket_context_distribution',f'{VERSION}_local_bucket_context_distribution.json'),
        ('bucket_pressure_contribution',f'{VERSION}_bucket_pressure_contribution.json'),
        ('margin_configuration_summary',f'{VERSION}_margin_configuration_summary.json'),
        ('observed_primitive_pressure',f'{VERSION}_observed_primitive_pressure.json'),
        ('pair_bucket_report',f'{VERSION}_pair_bucket_report.json'),
        ('maze_generation_report',f'{VERSION}_maze_generation_report.json'),
        ('protocol_compliance_audit',f'{VERSION}_protocol_compliance_audit.json'),
    ]: save_json(out_dir/filename,report[key])
    save_json(out_dir/f"{VERSION}_dataset_debug.json",report)


# ---------------- optional QCNN / DQN stage, gated by --enable-qcnn ----------------

class QNetwork(nn.Module):
    def __init__(self,n=8):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(4,32,3,padding=1), nn.ReLU(),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64*n*n,128), nn.ReLU(), nn.Linear(128,4))
    def forward(self,x): return self.net(x)

class ReplayBuffer:
    def __init__(self,cap): self.cap=cap; self.buf=deque(maxlen=cap)
    def push(self,*items): self.buf.append(items)
    def sample(self,n): return random.sample(self.buf,min(n,len(self.buf)))
    def __len__(self): return len(self.buf)

class MazeEnvLite:
    def __init__(self,args):
        target=choose_target_difficulty(args)
        self.maze,self.start,self.goal,self.dist,self.path,self.meta=generate_maze(args,target)
        self.args=args; self.pos=self.start; self.steps=0; self.wall_hits=0
        self.visits=np.zeros((args.grid_size,args.grid_size),dtype=np.int32); self.visits[self.pos]=1
    def state(self): return make_state(self.maze,self.pos,self.goal,self.visits,self.args.visit_count_cap)
    def step(self,a):
        before=self.pos; cur=int(self.dist[before]) if self.dist[before]<INF else 999
        nxt,is_wall=step_env(self.maze,self.pos,int(a)); self.wall_hits += int(is_wall)
        self.visits[nxt]+=1; self.pos=nxt; self.steps += 1
        nd=int(self.dist[nxt]) if self.dist[nxt]<INF else 999
        done=(self.pos==self.goal) or (self.steps>=self.args.max_steps)
        return self.state(), done, dict(is_wall=is_wall,delta_bfs=nd-cur,visit_count_after=int(self.visits[nxt]),success=self.pos==self.goal)

def rm_reward_for_q(rm,state,action,next_state,args,device):
    x=np.concatenate([state,next_state,action_tensor(int(action),args.grid_size)],axis=0).astype(np.float32)
    with torch.no_grad(): return float(rm(torch.from_numpy(x[None]).to(device)).item())

def epsilon_by_ep(ep,args):
    frac=min(1.0, ep/max(1,args.eps_decay_episodes))
    return args.eps_start + frac*(args.eps_end-args.eps_start)

def train_qcnn_optional(args,rm,device,out_dir):
    print('[INFO] QCNN enabled by --enable-qcnn.')
    q=QNetwork(args.grid_size).to(device); target=QNetwork(args.grid_size).to(device); target.load_state_dict(q.state_dict())
    opt=torch.optim.Adam(q.parameters(),lr=args.q_lr); replay=ReplayBuffer(args.replay_size); updates=0
    best_success=-1.0; hist=[]
    for ep in maybe_tqdm(range(1,args.q_episodes+1),args,desc='QCNN episodes'):
        env=MazeEnvLite(args); state=env.state(); eps=epsilon_by_ep(ep,args); total_r=0.0
        for _ in range(args.max_steps):
            if random.random()<eps: a=random.randrange(4)
            else:
                with torch.no_grad(): a=int(q(torch.from_numpy(state[None].astype(np.float32)).to(device)).argmax(dim=1).item())
            ns,done,info=env.step(a); r=rm_reward_for_q(rm,state,a,ns,args,device); total_r+=r
            replay.push(state,a,r,ns,float(done)); state=ns
            if len(replay)>=args.q_warmup_steps:
                batch=replay.sample(args.q_batch_size)
                S=torch.from_numpy(np.stack([b[0] for b in batch]).astype(np.float32)).to(device)
                A=torch.tensor([b[1] for b in batch],dtype=torch.long,device=device)
                R=torch.tensor([b[2] for b in batch],dtype=torch.float32,device=device)
                NS=torch.from_numpy(np.stack([b[3] for b in batch]).astype(np.float32)).to(device)
                D=torch.tensor([b[4] for b in batch],dtype=torch.float32,device=device)
                qv=q(S).gather(1,A.view(-1,1)).squeeze(1)
                with torch.no_grad(): y=R+args.gamma*(1-D)*target(NS).max(dim=1).values
                loss=F.smooth_l1_loss(qv,y); opt.zero_grad(); loss.backward(); opt.step(); updates+=1
                if updates%args.target_update_interval==0: target.load_state_dict(q.state_dict())
            if done: break
        success=float(env.pos==env.goal); hist.append(dict(ep=ep,success=success,reward=total_r,steps=env.steps,wall_hits=env.wall_hits,epsilon=eps))
        if success>best_success:
            best_success=success; torch.save({'model':q.state_dict(),'args':vars(args),'best_success':best_success},out_dir/f'{VERSION}_qcnn_best_success.pt')
    torch.save({'model':q.state_dict(),'args':vars(args)},out_dir/f'{VERSION}_qcnn_last.pt')
    save_json(out_dir/f'{VERSION}_qcnn_history.json',hist)
    print(f'[Done] optional QCNN artifacts saved to: {out_dir}')

def maybe_run_qcnn(args,device,out_dir):
    if not args.enable_qcnn:
        raise SystemExit('[ERROR] QCNN is disabled by default. Use --enable-qcnn to enable QCNN training.')
    if not args.reward_model or not Path(args.reward_model).exists():
        raise SystemExit('[ERROR] --mode train-q/all-q requires --reward-model for the gated QCNN stage.')
    rm=RewardModel(args.grid_size).to(device); rm.load_state_dict(torch.load(args.reward_model,map_location=device)); rm.eval()
    train_qcnn_optional(args,rm,device,out_dir)

def main():
    parser=build_argparser(); explicit=set(a.split('=')[0] for a in sys.argv[1:]); args=parser.parse_args(); args=apply_preset(args, explicit)
    if args.test: args.mode='test'
    if args.mode=='all':
        print('[INFO] v2.0.9(6) defaults to RM-only. QCNN training is disabled unless --enable-qcnn is set.'); args.mode='all-rm'
    allowed={'train-rm','debug-rm','all-rm','test','train-q','all-q'}
    if args.mode not in allowed: raise SystemExit(f"unsupported --mode {args.mode}; allowed: {sorted(allowed)}")
    if args.mode in {'train-q','all-q'} and not args.enable_qcnn:
        raise SystemExit('[ERROR] QCNN is disabled by default. Use --enable-qcnn to enable QCNN training.')
    set_seed(args.seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_name=nondefault_run_name(args); out_dir=Path('./v2.0.9')/run_name; ensure_dir(out_dir)
    print("\n==== CONFIGURATION SUMMARY ====")
    rows=[{'key':'version','value':VERSION},{'key':'output_dir','value':str(out_dir)},{'key':'mode','value':args.mode},{'key':'preset','value':args.preset},{'key':'device','value':str(device)},{'key':'seed','value':args.seed},{'key':'grid_size','value':args.grid_size},{'key':'max_steps','value':args.max_steps},{'key':'rm_mazes','value':args.rm_mazes},{'key':'rm_epochs','value':args.rm_epochs},{'key':'rm_batch_size','value':args.rm_batch_size},{'key':'rm_greedy_probe_mazes','value':args.rm_greedy_probe_mazes},{'key':'debug_probe_mazes','value':args.debug_probe_mazes},{'key':'score_normalizer','value':args.score_normalizer}]
    print_table('[Effective Configuration]', [('key',28,'s'),('value',60,'s')], rows)
    print('[INFO] QCNN enabled by --enable-qcnn.' if args.enable_qcnn else '[INFO] RM-only mode. QCNN disabled.')
    model=None; report=None
    if args.mode=='test':
        return run_test_mode(args,device,out_dir)
    if args.mode in {'train-q','all-q'}:
        return maybe_run_qcnn(args,device,out_dir)
    if args.mode in ('train-rm','all-rm'):
        pairs,report=build_dataset(args); print_reports(args,report); save_reports(out_dir,report)
        model=train_rm(args,pairs,out_dir,device)
    elif args.mode=='debug-rm':
        if not args.reward_model: raise SystemExit('--reward-model is required for --mode debug-rm')
        model=RewardModel(args.grid_size).to(device); model.load_state_dict(torch.load(args.reward_model,map_location=device)); model.eval()
        report={'observed_primitive_pressure':{'rows':{}}}
    chk=collapse_check(model,args,device); print(f"\n[RM Collapse Check] raw_rm_std={chk['raw_rm_std']:.4f} rm_scale={chk['rm_scale']:.4f} status={chk['status']}")
    prim=primitive_reward_report(model,args,device); ctx=contextual_probe(model,args,device); greedy=greedy_comparison(model,args,device)
    stuck=stuck_timeout_analysis(greedy,args); advice=dual_gate_tuning_advice(report or {'observed_primitive_pressure':{'rows':{}}},prim,greedy,stuck); tail=failure_tail_action_rank_report(greedy,args)
    print_probe_summary(ctx,greedy,prim,stuck,advice,tail)
    save_json(out_dir/f"{VERSION}_rm_diagnostic.json",{'collapse_check':chk})
    save_json(out_dir/f"{VERSION}_primitive_reward_score_report.json",prim)
    save_json(out_dir/f"{VERSION}_contextual_probe.json",ctx)
    save_json(out_dir/f"{VERSION}_greedy_comparison.json",{k:v for k,v in greedy.items() if not k.startswith('_')})
    save_json(out_dir/f"{VERSION}_greedy_difficulty_breakdown.json",greedy.get('difficulty_breakdown',{}))
    save_json(out_dir/f"{VERSION}_stuck_timeout_analysis.json",stuck)
    save_json(out_dir/f"{VERSION}_dual_gate_tuning_advice.json",advice)
    save_json(out_dir/f"{VERSION}_failure_tail_action_rank_report.json",tail)
    save_json(out_dir/f"{VERSION}_greedy_rollout_debug_pure_argmax.json",greedy['_rollouts_pure'])
    save_json(out_dir/f"{VERSION}_greedy_rollout_debug_bfs_tiebreak.json",greedy['_rollouts_bfs'])
    save_json(out_dir/f"{VERSION}_probe_summary.json",dict(contextual_gaps=ctx['gaps'],primitive_gaps=prim['gaps'],pure_argmax_success=greedy['pure_argmax']['summary']['success'],pure_argmax_wall_rate=greedy['pure_argmax']['action_ranking']['wall_chosen_rate'],all_actions_saturated_rate=greedy['pure_argmax']['action_ranking']['all_actions_saturated_rate']))

if __name__=='__main__': main()
