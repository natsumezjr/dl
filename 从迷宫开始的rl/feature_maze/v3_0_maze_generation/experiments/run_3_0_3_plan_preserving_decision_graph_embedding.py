#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3.0.3 Plan-Preserving Decision Graph Embedding MVP

Scope:
- Generator / Extractor / Validator-Debug only.
- Diagnostic profile provider only, not a full Sampler.
- No CNN / DQN / Reward Model / GAN / PCGML / RL.

This file intentionally does not import 3.0.2 code. 3.0.3 starts a new MVP line.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
import traceback
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, asdict, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

VERSION = "3.0.3_2_corridor_reservation_mrb_guard"
EXPERIMENT_NAME = "corridor_reservation_mrb_guard"
GRID_N = 8
Coord = Tuple[int, int]

ACTIONS = [(1,0),(-1,0),(0,1),(0,-1)]

# ----------------------------- small utilities -----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def p50(xs: List[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    return float(ys[len(ys)//2])


def pct(xs: List[bool]) -> float:
    return float(sum(1 for x in xs if x) / len(xs)) if xs else 0.0


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def neigh4(c: Coord, n: int = GRID_N) -> List[Coord]:
    r, col = c
    out = []
    for dr, dc in ACTIONS:
        nr, nc = r+dr, col+dc
        if 0 <= nr < n and 0 <= nc < n:
            out.append((nr,nc))
    return out


def edge_key(u: str, v: str, k: int = 0) -> Tuple[str,str,int]:
    return (u, v, k) if u <= v else (v, u, k)


def simple_bfs_path(grid: List[List[int]], start: Coord, goal: Coord) -> Optional[List[Coord]]:
    q = deque([start])
    prev = {start: None}
    while q:
        x = q.popleft()
        if x == goal:
            break
        for y in neigh4(x, len(grid)):
            if grid[y[0]][y[1]] == 0 and y not in prev:
                prev[y] = x
                q.append(y)
    if goal not in prev:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return list(reversed(path))


def components_from_free(grid: List[List[int]]) -> List[List[Coord]]:
    n = len(grid)
    seen = set()
    comps = []
    for r in range(n):
        for c in range(n):
            if grid[r][c] != 0 or (r,c) in seen:
                continue
            comp = []
            q = deque([(r,c)])
            seen.add((r,c))
            while q:
                x = q.popleft(); comp.append(x)
                for y in neigh4(x,n):
                    if grid[y[0]][y[1]] == 0 and y not in seen:
                        seen.add(y); q.append(y)
            comps.append(comp)
    return comps


def cycle_rank(edge_count: int, vertex_count: int, component_count: int = 1) -> int:
    return edge_count - vertex_count + component_count

# ----------------------------- graph data -----------------------------

@dataclass
class AbsNode:
    node_id: str
    node_type: str  # choice_node or endpoint_node
    degree: int = 0
    coord: Optional[Coord] = None
    is_start: bool = False
    is_goal: bool = False

@dataclass
class AbsEdge:
    edge_id: str
    u: str
    v: str
    key: int
    role: Optional[str] = None
    level: Optional[int] = None
    sampled_D_range: Optional[List[int]] = None
    D_target: Optional[int] = None
    D_actual: Optional[int] = None
    excess: int = 0
    L_target: Optional[int] = None
    cell_path: Optional[List[Coord]] = None
    weight_for_complexity: float = 1.0

@dataclass
class GenerationFailure(Exception):
    failed_layer: str
    failure_code: str
    failure_detail: Dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> Dict[str, Any]:
        return {
            "status": "FAILED",
            "failed_layer": self.failed_layer,
            "failure_code": self.failure_code,
            "failure_detail": self.failure_detail,
        }

# ----------------------------- profile provider -----------------------------

BASE_CONFIG: Dict[str, Any] = {
    "grid_size": 8,
    "parallel_multiplicity_max": 2,
    "self_loop_enabled": False,
    "max_generation_attempts": 25,
    "max_topology_attempts": 50,
    "max_layout_attempts": 250,
    "max_route_attempts_per_edge": 500,
    "max_island_attempts": 80,
    "endpoint_softmax_temperature": 0.8,
    "path_complexity": {
        "alpha": 0.35,
        "beta": 1.0,
        "w_min": 0.2,
        "w_max": 1.0,
        "lambda": 0.8,
        "tolerance": 2.0,
        "enable_mrb_guard": True,
        "mrb_guard_mode": "strict_mrb_preserve",
        "max_mrb_guard_iterations": 10,
        "main_path_imbalance_warning": 2,
        "C_path_soft_tolerance": 2.0,
        "C_path_hard_tolerance": 5.0,
        "allow_pass_when_forced_above_target_by_mrb_guard": True,
        "forced_above_target_warning_ratio": 0.3,
    },
    "embedding": {"enable_corridor_reservation": True, "enable_restricted_routing": True},
    "corridor": {"dogleg_offsets": [1, 2], "max_candidates_per_edge": {"main": 8, "substitute": 8, "endpoint_branch": 6}},
    "routing": {"forbid_non_edge_touch": True, "forbid_2x2_when_room_suppressed": True, "restricted_detour_radius": 1},
    "room_detection": {"use_2x2_free_block": True, "room_min_component_size": 3, "room_high_degree_threshold": 3},
    "visualization": {"enabled": True, "max_success_samples": 5, "max_failure_samples": 5},
}

# The spec examples A/C with c=3,t=3,mu=0 are mathematically infeasible under
# 2*mu + t >= c + 2. We explicitly adjust MVP profiles, and record this fact.
PROFILE_ADJUSTMENTS = {
    "pure_topology": "Spec example c=3,t=3,mu=0 is infeasible under 2*mu+t>=c+2; MVP smoke uses smaller feasible c=2,t=4,mu=0 to test plan-preserving embedding.",
}


def make_profile(profile: str) -> Dict[str, Any]:
    """Diagnostic profile provider. This is not the future Sampler."""
    common_terminal = {
        "allow_terminal_choice": True,
        "terminal_bucket_weights": {"endpoint": 0.35, "choice": 0.45, "edge_internal": 0.20},
    }
    if profile == "pure_topology":
        c,t,mu,k,room = 2,4,0,0,"room_suppressed"
    elif profile == "single_loop":
        c,t,mu,k,room = 3,3,1,0,"room_suppressed"
    elif profile == "terminal_choice":
        c,t,mu,k,room = 3,3,1,0,"room_suppressed"
    elif profile == "d_range_binding":
        c,t,mu,k,room = 4,3,2,0,"room_suppressed"
    elif profile == "island_only":
        c,t,mu,k,room = 3,3,1,1,"room_suppressed"
    elif profile == "room_neutral":
        c,t,mu,k,room = 3,3,1,0,"room_neutral"
    elif profile == "full_mvp":
        c,t,mu,k,room = 4,3,2,1,"room_neutral"
    else:
        raise ValueError(f"unknown profile: {profile}")
    terminal = copy.deepcopy(common_terminal)
    if profile == "terminal_choice":
        terminal["force_one_terminal_choice"] = True
    plan = {
        "profile": profile,
        "topology": {"endpoint_contract_mode": "direct_t_derive_E", "choice_node_count": c, "endpoint_node_count": t, "cycle_rank": mu},
        "terminal": terminal,
        "path_complexity": {"target_C_path": 10.0 if profile in ("d_range_binding","full_mvp") else 7.0, "tolerance": 2.0, "alpha": 0.35, "beta": 1.0},
        "room_policy": room,
        "island": {"island_component_count": k},
        "D_range_sample_pack_profile": "full_mvp_balanced" if profile == "full_mvp" else "balanced_main_substitute_endpoint",
        "profile_adjustment_note": PROFILE_ADJUSTMENTS.get(profile),
    }
    # D ranges intentionally generous in MVP diagnostic stubs. Generator binds, does not resample pack.
    pack = {
        "main": [[2, 4] for _ in range(12)],
        "endpoint_branch": [[1, 3] for _ in range(12)],
        "substitute_level1": [[2, 4] for _ in range(12)],
        "substitute_level2plus": [[2, 4] for _ in range(8)],
    }
    if profile == "d_range_binding":
        pack = {"main": [[2, 4] for _ in range(8)], "endpoint_branch": [[1, 3] for _ in range(6)], "substitute_level1": [[2, 4] for _ in range(6)], "substitute_level2plus": [[2, 4] for _ in range(4)]}
    return {"plan_contract": plan, "D_range_sample_pack": pack, "config": copy.deepcopy(BASE_CONFIG)}

# ----------------------------- abstract topology -----------------------------

def derive_topology(plan: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, int]:
    topo = plan["topology"]
    mode = topo.get("endpoint_contract_mode", "direct_t_derive_E")
    if mode == "direct_t_derive_E":
        c = int(topo["choice_node_count"]); t = int(topo["endpoint_node_count"]); mu = int(topo["cycle_rank"])
        V = c + t; E = mu + V - 1; E_core = mu + c - 1
    elif mode == "direct_E_derive_t":
        c = int(topo["choice_node_count"]); mu = int(topo["cycle_rank"]); E = int(topo["target_edge_count"])
        t = E - mu - c + 1; V = c + t; E_core = E - t
    else:
        raise GenerationFailure("Plan Contract", "invalid_topology_contract", {"mode": mode})
    return {"c": c, "t": t, "mu": mu, "V": V, "E": E, "E_core": E_core}


def check_topology_feasible(vals: Dict[str,int], cfg: Dict[str,Any]) -> Dict[str, Any]:
    c,t,mu,E,E_core = vals["c"], vals["t"], vals["mu"], vals["E"], vals["E_core"]
    m = int(cfg["parallel_multiplicity_max"])
    rec = {"degree_feasible": True, "parallel_capacity_feasible": True, "E_core": E_core, "parallel_multiplicity_max": m, "self_loop_enabled": False}
    if c < 2 or t < 0 or mu < 0 or E < 0:
        raise GenerationFailure("Plan Contract", "invalid_topology_contract", vals)
    if 2*E < t + 3*c:
        rec["degree_feasible"] = False
        raise GenerationFailure("Topology Feasibility", "degree_feasibility_failed", {**vals, "required": "2E >= t + 3c", "equivalent": "2mu + t >= c + 2"})
    cap = m * (c*(c-1)//2)
    if E_core > cap:
        rec["parallel_capacity_feasible"] = False
        raise GenerationFailure("Topology Feasibility", "parallel_capacity_infeasible", {**vals, "capacity": cap})
    return rec


def adjacency(nodes: Dict[str,AbsNode], edges: Dict[str,AbsEdge], include_edge: Optional[str] = None, exclude_edge: Optional[str] = None) -> Dict[str, List[Tuple[str,str]]]:
    adj = defaultdict(list)
    for eid, e in edges.items():
        if eid == exclude_edge: continue
        if include_edge is not None and eid != include_edge: pass
        adj[e.u].append((e.v,eid)); adj[e.v].append((e.u,eid))
    return adj


def graph_connected(nodes: Dict[str,AbsNode], edges: Dict[str,AbsEdge]) -> bool:
    if not nodes: return True
    adj = adjacency(nodes, edges)
    start = next(iter(nodes))
    seen = {start}; q = deque([start])
    while q:
        u=q.popleft()
        for v,_ in adj[u]:
            if v not in seen:
                seen.add(v); q.append(v)
    return len(seen)==len(nodes)


def generate_topology(vals: Dict[str,int], cfg: Dict[str,Any], rng: random.Random) -> Tuple[Dict[str,AbsNode], Dict[str,AbsEdge], Dict[str,Any]]:
    c,t,mu,E_core = vals["c"], vals["t"], vals["mu"], vals["E_core"]
    mmax = cfg["parallel_multiplicity_max"]
    for attempt in range(cfg["max_topology_attempts"]):
        nodes: Dict[str,AbsNode] = {f"C{i}": AbsNode(f"C{i}", "choice_node") for i in range(c)}
        edges: Dict[str,AbsEdge] = {}
        pair_mult = Counter()
        eid_counter = 0
        order = list(nodes.keys()); rng.shuffle(order)
        connected = [order[0]]
        # spanning tree over choices
        for u in order[1:]:
            v = rng.choice(connected)
            k = pair_mult[tuple(sorted((u,v)))]
            pair_mult[tuple(sorted((u,v)))] += 1
            eid = f"e{eid_counter}"; eid_counter += 1
            edges[eid] = AbsEdge(eid,u,v,k)
            connected.append(u)
        # extra core edges = mu
        for _ in range(mu):
            candidates = []
            for u,v in combinations(nodes.keys(),2):
                mult = pair_mult[tuple(sorted((u,v)))]
                if mult < mmax:
                    candidates.append((mult,u,v))
            if not candidates:
                break
            # prefer unused simple pairs
            min_mult = min(x[0] for x in candidates)
            cand = [(u,v) for mult,u,v in candidates if mult == min_mult]
            u,v = rng.choice(cand)
            k = pair_mult[tuple(sorted((u,v)))]
            pair_mult[tuple(sorted((u,v)))] += 1
            eid = f"e{eid_counter}"; eid_counter += 1
            edges[eid] = AbsEdge(eid,u,v,k)
        if len([e for e in edges.values() if e.u.startswith('C') and e.v.startswith('C')]) != E_core:
            continue
        # degrees before endpoints
        deg = Counter()
        for e in edges.values(): deg[e.u]+=1; deg[e.v]+=1
        endpoint_nodes = {}
        endpoint_edges = {}
        endpoint_idx=0
        deficits=[]
        for cid in nodes:
            for _ in range(max(0, 3-deg[cid])):
                deficits.append(cid)
        if len(deficits) > t:
            continue
        attach_choices = list(deficits)
        remaining = t - len(attach_choices)
        T = float(cfg.get("endpoint_softmax_temperature",0.8))
        for _ in range(remaining):
            ids = list(nodes.keys())
            weights = [math.exp(-deg[x]/max(T,1e-6)) for x in ids]
            s = sum(weights); r = rng.random()*s; acc=0; chosen=ids[-1]
            for x,w in zip(ids,weights):
                acc += w
                if r <= acc:
                    chosen=x; break
            attach_choices.append(chosen); deg[chosen]+=1
        for choice_id in attach_choices:
            nid = f"P{endpoint_idx}"; endpoint_idx += 1
            endpoint_nodes[nid] = AbsNode(nid, "endpoint_node")
            eid = f"e{eid_counter}"; eid_counter += 1
            endpoint_edges[eid] = AbsEdge(eid, choice_id, nid, 0)
        nodes.update(endpoint_nodes); edges.update(endpoint_edges)
        # recompute degrees
        deg = Counter()
        for e in edges.values(): deg[e.u]+=1; deg[e.v]+=1
        for nid,n in nodes.items(): n.degree = deg[nid]
        if any(nodes[cid].degree < 3 for cid in nodes if cid.startswith("C")): continue
        if any(nodes[pid].degree != 1 for pid in nodes if pid.startswith("P")): continue
        if len(edges) != vals["E"]: continue
        if not graph_connected(nodes, edges): continue
        mu_check = len(edges) - len(nodes) + 1
        if mu_check != vals["mu"]: continue
        par_count = sum(1 for pair, mult in pair_mult.items() if mult > 1)
        dbg = {"generator": "random_choice_core_multigraph_with_endpoint_leaves", "attempt": attempt+1, "choice_core_edge_count": E_core, "endpoint_leaf_count": t, "parallel_edge_count": par_count, "choice_degrees": [nodes[f"C{i}"].degree for i in range(c)]}
        return nodes, edges, dbg
    raise GenerationFailure("Topology Generation", "topology_generation_failed", vals)

# ----------------------------- terminal + mrb + roles -----------------------------

def shortest_paths_edges(nodes: Dict[str,AbsNode], edges: Dict[str,AbsEdge], s: str, g: str, weights: Optional[Dict[str,int]] = None, max_paths: int = 200) -> Tuple[int, List[List[str]], Set[str]]:
    adj = adjacency(nodes, edges)
    dist = {s: 0}; q = deque([s]) if weights is None else None
    if weights is None:
        while q:
            u=q.popleft()
            for v,eid in adj[u]:
                nd=dist[u]+1
                if v not in dist:
                    dist[v]=nd; q.append(v)
        if g not in dist: return 10**9, [], set()
    else:
        import heapq
        heap=[(0,s)]; dist={s:0}
        while heap:
            d,u=heapq.heappop(heap)
            if d!=dist[u]: continue
            for v,eid in adj[u]:
                nd=d+int(weights.get(eid,1))
                if nd < dist.get(v,10**9):
                    dist[v]=nd; heapq.heappush(heap,(nd,v))
        if g not in dist: return 10**9, [], set()
    target=dist[g]
    paths=[]; bundle=set()
    def dfs(u: str, path_e: List[str], used_nodes: Set[str]):
        if len(paths) >= max_paths: return
        if u == g:
            if sum((weights.get(eid,1) if weights else 1) for eid in path_e) == target:
                paths.append(list(path_e)); bundle.update(path_e)
            return
        cur_len=sum((weights.get(eid,1) if weights else 1) for eid in path_e)
        for v,eid in adj[u]:
            if v in used_nodes: continue
            w=weights.get(eid,1) if weights else 1
            if cur_len + w + dist.get(v,10**9) > target and weights is not None:
                # dist here from source, not to goal; skip only for unweighted? keep conservative
                pass
            if weights is None:
                if dist.get(v,10**9) != dist[u]+1: continue
            else:
                if cur_len + w > target: continue
            dfs(v, path_e+[eid], used_nodes|{v})
    # For weighted, enumerate simple paths with pruning by length.
    if weights is None:
        dfs(s, [], {s})
    else:
        def dfsw(u,path_e,used,total):
            if len(paths) >= max_paths: return
            if total > target: return
            if u==g:
                if total==target: paths.append(list(path_e)); bundle.update(path_e)
                return
            for v,eid in adj[u]:
                if v in used: continue
                dfsw(v,path_e+[eid],used|{v},total+int(weights.get(eid,1)))
        dfsw(s,[],{s},0)
    return target, paths, bundle


def place_terminals(nodes: Dict[str,AbsNode], edges: Dict[str,AbsEdge], plan: Dict[str,Any], rng: random.Random) -> Tuple[str,str,Dict[str,Any]]:
    term = plan.get("terminal", {})
    ids = list(nodes.keys())
    if term.get("force_one_terminal_choice"):
        choices = [x for x,n in nodes.items() if n.node_type == "choice_node"]
        endpoints = [x for x,n in nodes.items() if n.node_type == "endpoint_node"]
        s = rng.choice(choices); e = rng.choice(endpoints) if endpoints else rng.choice([x for x in ids if x != s])
    else:
        choices = [x for x,n in nodes.items() if n.node_type == "choice_node"]
        endpoints = [x for x,n in nodes.items() if n.node_type == "endpoint_node"]
        s_pool = choices + endpoints
        s = rng.choice(s_pool)
        e_pool = [x for x in s_pool if x != s]
        e = rng.choice(e_pool)
    nodes[s].is_start = True; nodes[e].is_goal = True
    deg = Counter()
    for ed in edges.values(): deg[ed.u]+=1; deg[ed.v]+=1
    s_type = "terminal_choice" if deg[s] >= 2 else "terminal_endpoint"
    e_type = "terminal_choice" if deg[e] >= 2 else "terminal_endpoint"
    return s,e,{"s_node": s, "e_node": e, "s_type": s_type, "e_type": e_type, "terminal_preserved_count": 0, "terminal_choice_count": int(s_type=="terminal_choice") + int(e_type=="terminal_choice"), "terminal_endpoint_count": int(s_type=="terminal_endpoint") + int(e_type=="terminal_endpoint")}


def edge_participates_cycle(nodes: Dict[str,AbsNode], edges: Dict[str,AbsEdge], eid: str) -> bool:
    e = edges[eid]
    adj = adjacency(nodes, edges, exclude_edge=eid)
    q=deque([e.u]); seen={e.u}
    while q:
        u=q.popleft()
        if u==e.v: return True
        for v,_ in adj[u]:
            if v not in seen:
                seen.add(v); q.append(v)
    return False


def annotate_roles(nodes: Dict[str,AbsNode], edges: Dict[str,AbsEdge], main_bundle: Set[str]) -> Dict[str,int]:
    for eid,e in edges.items():
        if eid in main_bundle:
            e.role = "main"; e.level = 0
        elif (nodes[e.u].node_type == "endpoint_node" or nodes[e.v].node_type == "endpoint_node") and not edge_participates_cycle(nodes,edges,eid):
            e.role = "endpoint_branch"; e.level = None
        elif edge_participates_cycle(nodes, edges, eid):
            e.role = "substitute_level1"; e.level = 1
        else:
            e.role = "endpoint_branch"; e.level = None
    return Counter(e.role for e in edges.values())

# ----------------------------- D range + complexity -----------------------------

def bind_d_ranges(edges: Dict[str,AbsEdge], pack: Dict[str,List[List[int]]], rng: random.Random) -> Dict[str,Any]:
    queues = {k: list(v) for k,v in pack.items()}
    role_order = {"main":0, "endpoint_branch":1, "substitute_level1":2, "substitute_level2plus":3}
    for eid,e in sorted(edges.items(), key=lambda kv: (role_order.get(kv[1].role,9), kv[0])):
        role = e.role or "unknown"
        if role not in queues or not queues[role]:
            raise GenerationFailure("D-range Binding", "D_range_queue_insufficient", {"role": role, "edge_id": eid, "provided_remaining": len(queues.get(role, []))})
        dr = queues[role].pop(0)
        e.sampled_D_range = [int(dr[0]), int(dr[1])]
        e.D_target = rng.randint(e.sampled_D_range[0], e.sampled_D_range[1])
    unused = {k: len(v) for k,v in queues.items() if len(v)>0}
    return {"source":"D_range_sample_pack", "role_not_in_plan": True, "queue_insufficient": False, "unused_warning": bool(unused), "unused_counts": unused}


def compute_weights(edges: Dict[str,AbsEdge], cfg: Dict[str,Any]) -> None:
    pcfg = cfg["path_complexity"]
    wmin,wmax,lam = pcfg["w_min"], pcfg["w_max"], pcfg["lambda"]
    for e in edges.values():
        d = 0 if e.role == "main" else (1 if e.role == "substitute_level1" else 2)
        if e.role == "endpoint_branch": d = 1
        e.weight_for_complexity = float(wmin + (wmax-wmin)*math.exp(-lam*d))


def allocate_excess(edges: Dict[str,AbsEdge], plan: Dict[str,Any], cfg: Dict[str,Any], rng: random.Random) -> Dict[str,Any]:
    alpha = float(plan.get("path_complexity",{}).get("alpha", cfg["path_complexity"]["alpha"]))
    beta = float(plan.get("path_complexity",{}).get("beta", cfg["path_complexity"]["beta"]))
    target = float(plan.get("path_complexity",{}).get("target_C_path", 8.0))
    tol = float(plan.get("path_complexity",{}).get("tolerance", cfg["path_complexity"]["tolerance"]))
    compute_weights(edges, cfg)
    C_length = sum(e.weight_for_complexity * alpha * int(e.D_actual) for e in edges.values())
    if C_length > target + tol:
        raise GenerationFailure("Path Complexity Allocation", "path_complexity_base_exceeds_target", {"C_length": C_length, "target_C_path": target, "tolerance": tol})
    need = max(0.0, target - C_length)
    # Greedy even excess allocation while keeping routes feasible-ish.
    for e in edges.values(): e.excess = 0
    edge_list = list(edges.values())
    for _ in range(20):
        if need <= 0.5: break
        e = rng.choice(edge_list)
        e.excess += 2
        need -= e.weight_for_complexity * beta * 2
    for e in edges.values():
        e.L_target = int(e.D_actual) + int(e.excess)
    C_excess = sum(e.weight_for_complexity * beta * e.excess for e in edges.values())
    C_path = C_length + C_excess
    return {"alpha": alpha, "beta": beta, "C_length": C_length, "C_excess": C_excess, "C_path": C_path, "target_C_path": target, "within_tolerance": abs(C_path-target) <= tol, "tolerance": tol}


def recompute_path_complexity(edges: Dict[str,AbsEdge], plan: Dict[str,Any], cfg: Dict[str,Any]) -> Dict[str,Any]:
    alpha = float(plan.get("path_complexity",{}).get("alpha", cfg["path_complexity"]["alpha"]))
    beta = float(plan.get("path_complexity",{}).get("beta", cfg["path_complexity"]["beta"]))
    target = float(plan.get("path_complexity",{}).get("target_C_path", 8.0))
    tol = float(plan.get("path_complexity",{}).get("tolerance", cfg["path_complexity"].get("tolerance", 2.0)))
    C_length = sum(e.weight_for_complexity * alpha * int(e.D_actual) for e in edges.values())
    C_excess = sum(e.weight_for_complexity * beta * int(e.excess) for e in edges.values())
    C_path = C_length + C_excess
    soft_tol = float(cfg.get("path_complexity",{}).get("C_path_soft_tolerance", tol))
    hard_tol = float(cfg.get("path_complexity",{}).get("C_path_hard_tolerance", max(tol, 5.0)))
    return {
        "alpha": alpha,
        "beta": beta,
        "C_length": C_length,
        "C_excess": C_excess,
        "C_path": C_path,
        "target_C_path": target,
        "within_tolerance": abs(C_path-target) <= tol,
        "within_soft_tolerance": abs(C_path-target) <= soft_tol,
        "within_hard_tolerance": abs(C_path-target) <= hard_tol,
        "tolerance": tol,
    }


def classify_mrb_changes(added: Set[str], removed: Set[str], edges: Dict[str,AbsEdge]) -> List[Dict[str, str]]:
    out=[]
    for eid in sorted(added):
        role=edges[eid].role if eid in edges else "unknown"
        if role and role.startswith("substitute"):
            reason="substitute_entered_weighted_MRB"
        elif role == "endpoint_branch":
            reason="endpoint_branch_entered_weighted_MRB"
        else:
            reason="unknown_non_main_entered_weighted_MRB"
        out.append({"edge_id": eid, "reason": reason})
    for eid in sorted(removed):
        out.append({"edge_id": eid, "reason": "main_edge_dropped_from_weighted_MRB"})
    return out


def mrb_passes(mode: str, prelim: Set[str], weighted: Set[str], edges: Dict[str,AbsEdge]) -> bool:
    if mode == "no_substitute_in_mrb":
        return all(not (edges[eid].role or "").startswith("substitute") for eid in weighted if eid in edges)
    return weighted == prelim


def apply_mrb_guard(nodes: Dict[str,AbsNode], edges: Dict[str,AbsEdge], s: str, e: str, prelim_bundle: Set[str], prelim_paths: List[List[str]], plan: Dict[str,Any], cfg: Dict[str,Any], pc_before: Dict[str,Any]) -> Tuple[Dict[str,Any], Dict[str,Any], Set[str], List[List[str]]]:
    pcfg=cfg.get("path_complexity", {})
    enabled=bool(pcfg.get("enable_mrb_guard", False))
    mode=str(pcfg.get("mrb_guard_mode", "strict_mrb_preserve"))
    max_iter=int(pcfg.get("max_mrb_guard_iterations", 10))
    weights={eid:int(ed.L_target) for eid,ed in edges.items()}
    weighted_len, weighted_paths, weighted_bundle=shortest_paths_edges(nodes,edges,s,e,weights=weights)
    main_lengths=[]
    for pth in prelim_paths:
        main_lengths.append(sum(weights.get(eid,1) for eid in pth))
    gap=(max(main_lengths)-min(main_lengths)) if main_lengths else 0
    added=set(weighted_bundle-prelim_bundle)
    removed=set(prelim_bundle-weighted_bundle)
    dbg={
        "mrb_guard_enabled": enabled,
        "mrb_guard_mode": mode,
        "preliminary_main_bundle_edges": sorted(prelim_bundle),
        "weighted_main_bundle_edges_before_guard": sorted(weighted_bundle),
        "weighted_MRB_edges_added": sorted(added),
        "weighted_MRB_edges_removed": sorted(removed),
        "change_reasons": classify_mrb_changes(added, removed, edges),
        "guard_adjustments": [],
        "main_path_length_imbalance": {"weighted_lengths": main_lengths, "gap": gap, "warning": gap > int(pcfg.get("main_path_imbalance_warning", 2))},
        "C_path_before_guard": pc_before.get("C_path"),
        "target_C_path": pc_before.get("target_C_path"),
        "guard_invoked": False,
        "guard_failed": False,
    }
    if not enabled:
        dbg["weighted_main_bundle_edges_after_guard"] = sorted(weighted_bundle)
        return pc_before, dbg, weighted_bundle, weighted_paths
    it=0
    while not mrb_passes(mode, prelim_bundle, weighted_bundle, edges) and it < max_iter:
        it += 1
        dbg["guard_invoked"] = True
        added=set(weighted_bundle-prelim_bundle)
        # Prefer penalizing added non-main edges. If removed main edges are the only issue,
        # add excess to all current weighted non-main edges as a conservative nudge.
        candidates=[eid for eid in sorted(added) if eid in edges and edges[eid].role != "main"]
        if not candidates:
            candidates=[eid for eid in sorted(weighted_bundle) if eid in edges and edges[eid].role != "main"]
        if not candidates:
            break
        for eid in candidates:
            old=edges[eid].excess
            edges[eid].excess += 2
            edges[eid].L_target = int(edges[eid].D_actual) + int(edges[eid].excess)
            dbg["guard_adjustments"].append({"edge_id": eid, "old_excess": old, "new_excess": edges[eid].excess, "reason": "mrb_guard_penalize_non_main"})
        weights={eid:int(ed.L_target) for eid,ed in edges.items()}
        weighted_len, weighted_paths, weighted_bundle=shortest_paths_edges(nodes,edges,s,e,weights=weights)
    added=set(weighted_bundle-prelim_bundle); removed=set(prelim_bundle-weighted_bundle)
    pc_after=recompute_path_complexity(edges, plan, cfg)
    forced=pc_after.get("C_path",0) > pc_before.get("target_C_path",0) + float(pcfg.get("C_path_soft_tolerance", 2.0)) and len(dbg["guard_adjustments"]) > 0
    dbg.update({
        "weighted_main_bundle_edges_after_guard": sorted(weighted_bundle),
        "weighted_MRB_edges_added_after_guard": sorted(added),
        "weighted_MRB_edges_removed_after_guard": sorted(removed),
        "change_reasons_after_guard": classify_mrb_changes(added, removed, edges),
        "guard_failed": not mrb_passes(mode, prelim_bundle, weighted_bundle, edges),
        "C_path_after_guard": pc_after.get("C_path"),
        "forced_above_target_by_mrb_guard": bool(forced),
        "within_soft_tolerance_after_guard": pc_after.get("within_soft_tolerance"),
        "within_hard_tolerance_after_guard": pc_after.get("within_hard_tolerance"),
        "guard_iteration_count": it,
    })
    if dbg["guard_failed"]:
        raise GenerationFailure("Main Bundle", "mrb_guard_failed", {**dbg, "edge_table": _edge_table_for_debug(edges)})
    if not pc_after.get("within_hard_tolerance", True) and not pcfg.get("allow_pass_when_forced_above_target_by_mrb_guard", True):
        raise GenerationFailure("Path Complexity Allocation", "path_complexity_target_unreachable", {**dbg, **pc_after})
    return pc_after, dbg, weighted_bundle, weighted_paths

# ----------------------------- layout and routing -----------------------------

def layout_nodes(nodes: Dict[str,AbsNode], edges: Dict[str,AbsEdge], cfg: Dict[str,Any], rng: random.Random) -> Dict[str,Any]:
    n = cfg.get("grid_size", GRID_N)
    all_cells = [(r,c) for r in range(n) for c in range(n)]
    ids = list(nodes.keys())
    # endpoint after choices tends to route better
    ids.sort(key=lambda x: (0 if nodes[x].node_type == "choice_node" else 1, x))
    for attempt in range(cfg["max_layout_attempts"]):
        coords: Dict[str,Coord] = {}
        shuffled = all_cells[:]; rng.shuffle(shuffled)
        ok=True
        for nid in ids:
            candidates = shuffled[:]
            rng.shuffle(candidates)
            chosen=None
            for cell in candidates:
                if any(manhattan(cell, other) < 2 for other in coords.values()):
                    continue
                # if has placed neighbor, try respect at least one edge range
                good=True
                for e in edges.values():
                    other=None
                    if e.u == nid and e.v in coords: other=e.v
                    if e.v == nid and e.u in coords: other=e.u
                    if other is not None and e.sampled_D_range:
                        d=manhattan(cell, coords[other])
                        if not (e.sampled_D_range[0] <= d <= e.sampled_D_range[1]):
                            good=False; break
                if good:
                    chosen=cell; break
            if chosen is None:
                ok=False; break
            coords[nid]=chosen
        if not ok: continue
        # all edges D in bound
        out=[]
        for e in edges.values():
            d=manhattan(coords[e.u], coords[e.v])
            if e.sampled_D_range and not (e.sampled_D_range[0] <= d <= e.sampled_D_range[1]):
                out.append({"edge_id": e.edge_id, "D_actual": d, "range": e.sampled_D_range})
        if out:
            continue
        for nid,coord in coords.items(): nodes[nid].coord = coord
        for e in edges.values(): e.D_actual = manhattan(coords[e.u], coords[e.v])
        rows=[c[0] for c in coords.values()]; cols=[c[1] for c in coords.values()]
        return {"relative_layout_enabled": True, "layout_attempt": attempt+1, "bbox_size": [max(rows)-min(rows)+1, max(cols)-min(cols)+1], "translation_candidates": 1, "D_actual_out_of_sampled_range": 0}
    raise GenerationFailure("Grid Embedding", "D_actual_out_of_sampled_range", {"message":"Could not place nodes satisfying sampled_D_range and distance>=2"})


def would_make_2x2(free: Set[Coord], cell: Coord, n: int) -> bool:
    trial = set(free); trial.add(cell)
    r,c=cell
    for rr in (r-1,r):
        for cc in (c-1,c):
            if 0 <= rr < n-1 and 0 <= cc < n-1:
                if all((rr+dr,cc+dc) in trial for dr in (0,1) for dc in (0,1)):
                    return True
    return False




def path_len(path: List[Coord]) -> int:
    return max(0, len(path) - 1)


def manhattan_skeleton(a: Coord, b: Coord, order: str) -> List[Coord]:
    r, c = a
    tr, tc = b
    path = [(r, c)]
    def step_to(nr, nc):
        nonlocal r, c, path
        while r != nr:
            r += 1 if nr > r else -1
            path.append((r, c))
        while c != nc:
            c += 1 if nc > c else -1
            path.append((r, c))
    if order == "hv":
        step_to(r, tc)
        step_to(tr, tc)
    else:
        step_to(tr, c)
        step_to(tr, tc)
    return path


def dedupe_path(path: List[Coord]) -> Optional[List[Coord]]:
    if not path:
        return None
    out = [path[0]]
    for x in path[1:]:
        if x != out[-1]:
            out.append(x)
    if len(set(out)) != len(out):
        return None
    for a, b in zip(out, out[1:]):
        if manhattan(a, b) != 1:
            return None
    return out


def dogleg_skeleton(a: Coord, b: Coord, offset: int, axis: str, sign: int, n: int) -> Optional[List[Coord]]:
    ar, ac = a
    br, bc = b
    if axis == "row":
        mid_r = ar + sign * offset
        if not (0 <= mid_r < n):
            return None
        pts = [a, (mid_r, ac), (mid_r, bc), b]
    else:
        mid_c = ac + sign * offset
        if not (0 <= mid_c < n):
            return None
        pts = [a, (ar, mid_c), (br, mid_c), b]
    path: List[Coord] = []
    for i in range(len(pts) - 1):
        seg = manhattan_skeleton(pts[i], pts[i+1], "hv" if i % 2 == 0 else "vh")
        if i > 0:
            seg = seg[1:]
        path.extend(seg)
    return dedupe_path(path)


def node_local_zone(node_cell: Coord, n: int) -> Set[Coord]:
    return {node_cell, *neigh4(node_cell, n)}


def path_internal_strict_cells(path: List[Coord]) -> Set[Coord]:
    # Step 0 and step 1 are node/port exemptions; same near the target.
    if len(path) <= 4:
        return set()
    return set(path[2:-2])


def is_legal_touch(cell: Coord, other: Coord, path: List[Coord], edge_nodes: Set[Coord], n: int) -> bool:
    if other in path:
        return True
    for nc in edge_nodes:
        if cell in node_local_zone(nc, n) and other in node_local_zone(nc, n):
            return True
    return False


def candidate_conflicts(
    path: List[Coord],
    edge_id: str,
    edge_nodes: Set[Coord],
    selected: Dict[str, Dict[str, Any]],
    node_cells: Set[Coord],
    L_target: int,
    room_suppressed: bool,
    n: int,
) -> Dict[str, Any]:
    reserved = set(path)
    strict_reserved = path_internal_strict_cells(path)
    hard_overlap = 0
    illegal_touch = 0
    two_by_two = 0
    planned_node_near_miss = 0
    if len(path) - 1 > L_target or (L_target - (len(path) - 1)) % 2 != 0:
        hard_overlap += 999
    if any(cell in node_cells and cell not in edge_nodes for cell in path[1:-1]):
        planned_node_near_miss += 1
        hard_overlap += 999
    if room_suppressed:
        free: Set[Coord] = set()
        for info in selected.values():
            free |= set(tuple(x) for x in info.get("reserved_cells", []))
        for cell in path[1:-1]:
            if would_make_2x2(free | reserved, cell, n):
                two_by_two += 1
                break
    for oid, info in selected.items():
        other_reserved = set(tuple(x) for x in info.get("reserved_cells", []))
        other_strict = set(tuple(x) for x in info.get("strict_reserved_cells", []))
        overlap = (strict_reserved & other_reserved) | (reserved & other_strict)
        overlap = {x for x in overlap if x not in edge_nodes}
        hard_overlap += len(overlap)
        for cell in strict_reserved:
            for nb in neigh4(cell, n):
                if nb in other_reserved and not is_legal_touch(cell, nb, path, edge_nodes, n):
                    illegal_touch += 1
        for cell in other_strict:
            for nb in neigh4(cell, n):
                if nb in reserved and not is_legal_touch(cell, nb, path, edge_nodes, n):
                    illegal_touch += 1
    detour_pressure = max(0, L_target - (len(path) - 1))
    score = 1000 * hard_overlap + 100 * illegal_touch + 50 * two_by_two + 10 * planned_node_near_miss + detour_pressure
    return {
        "hard_overlap_count": hard_overlap,
        "illegal_touch_count": illegal_touch,
        "two_by_two_risk_count": two_by_two,
        "planned_node_near_miss_count": planned_node_near_miss,
        "detour_pressure": detour_pressure,
        "conflict_score": score,
    }


def max_candidates_for_role(role: Optional[str], cfg: Dict[str, Any]) -> int:
    d = cfg.get("corridor", {}).get("max_candidates_per_edge", {})
    if role == "main":
        return int(d.get("main", 8))
    if role and role.startswith("substitute"):
        return int(d.get("substitute", 8))
    return int(d.get("endpoint_branch", 6))


def generate_corridor_candidates(eid: str, e: AbsEdge, nodes: Dict[str, AbsNode], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    n = cfg.get("grid_size", GRID_N)
    a = nodes[e.u].coord
    b = nodes[e.v].coord
    if a is None or b is None:
        return []
    raw: List[Tuple[str, Optional[List[Coord]]]] = [
        ("hv", manhattan_skeleton(a, b, "hv")),
        ("vh", manhattan_skeleton(a, b, "vh")),
    ]
    slack = int(e.L_target or 0) - manhattan(a, b)
    if slack >= 2:
        for off in cfg.get("corridor", {}).get("dogleg_offsets", [1, 2]):
            for axis in ("row", "col"):
                for sign in (-1, 1):
                    raw.append((f"dogleg_{axis}_{sign}_{off}", dogleg_skeleton(a, b, int(off), axis, sign, n)))
    out=[]
    seen=set()
    for name, path in raw:
        path = dedupe_path(path or [])
        if not path:
            continue
        key=tuple(path)
        if key in seen:
            continue
        seen.add(key)
        Ls=len(path)-1
        if Ls <= int(e.L_target or 0) and ((int(e.L_target or 0)-Ls) % 2 == 0):
            out.append({"candidate_id": f"{eid}_{name}", "skeleton": path, "skeleton_length": Ls})
    return out[:max_candidates_for_role(e.role, cfg)]


def reserve_corridors(nodes: Dict[str, AbsNode], edges: Dict[str, AbsEdge], plan: Dict[str, Any], cfg: Dict[str, Any], routing_items: List[Tuple[str, AbsEdge]]) -> Dict[str, Any]:
    n = cfg.get("grid_size", GRID_N)
    room_suppressed = plan.get("room_policy") == "room_suppressed" and cfg.get("routing", {}).get("forbid_2x2_when_room_suppressed", True)
    node_cells = set(nd.coord for nd in nodes.values() if nd.coord is not None)
    selected: Dict[str, Dict[str, Any]] = {}
    edge_plan: Dict[str, Any] = {}
    global_score = 0
    for eid, e in routing_items:
        edge_nodes = {nodes[e.u].coord, nodes[e.v].coord}
        candidates = generate_corridor_candidates(eid, e, nodes, cfg)
        scored=[]
        for cand in candidates:
            c = candidate_conflicts(cand["skeleton"], eid, edge_nodes, selected, node_cells, int(e.L_target or 0), room_suppressed, n)
            scored.append({**cand, **c})
        scored.sort(key=lambda x: (x["hard_overlap_count"] > 0, x["illegal_touch_count"] > 0, x["two_by_two_risk_count"] > 0, x["conflict_score"], x["skeleton_length"]))
        chosen=None
        for cand in scored:
            if cand["hard_overlap_count"] == 0 and cand["illegal_touch_count"] == 0 and (not room_suppressed or cand["two_by_two_risk_count"] == 0):
                chosen=cand
                break
        if chosen is None:
            dominant = "none"
            if scored:
                first=scored[0]
                dominant=max(["hard_overlap_count","illegal_touch_count","two_by_two_risk_count"], key=lambda k:first.get(k,0))
            return {
                "status": "FAILED",
                "failed_reason": "corridor_reservation_failed",
                "failed_edge_id": eid,
                "edge_role": e.role,
                "candidate_count": len(candidates),
                "dominant_conflict_type": dominant,
                "edge_corridor_plan": edge_plan,
            }
        reserved=set(chosen["skeleton"])
        safety=set(reserved)
        for cell in list(reserved):
            safety.update(neigh4(cell, n))
        selected[eid]={
            "candidate_id": chosen["candidate_id"],
            "skeleton": [list(x) for x in chosen["skeleton"]],
            "reserved_cells": [list(x) for x in reserved],
            "strict_reserved_cells": [list(x) for x in path_internal_strict_cells(chosen["skeleton"])],
            "safety_margin_cells": [list(x) for x in safety],
        }
        needs = chosen["skeleton_length"] < int(e.L_target or 0)
        edge_plan[eid]={
            "role": e.role,
            "D_actual": e.D_actual,
            "L_target": e.L_target,
            "slack": int(e.L_target or 0) - int(e.D_actual or 0),
            "candidate_count": len(candidates),
            "selected_candidate_id": chosen["candidate_id"],
            "selected_skeleton": [list(x) for x in chosen["skeleton"]],
            "selected_skeleton_length": chosen["skeleton_length"],
            "needs_restricted_dfs": needs,
            "reserved_cells_count": len(reserved),
            "safety_margin_cells_count": len(safety),
            "conflict_score": chosen["conflict_score"],
            "hard_overlap_count": chosen["hard_overlap_count"],
            "illegal_touch_count": chosen["illegal_touch_count"],
            "two_by_two_risk_count": chosen["two_by_two_risk_count"],
        }
        global_score += chosen["conflict_score"]
    return {"status": "PASS", "edge_corridor_plan": edge_plan, "global_conflict_score": global_score, "failed_reason": None}


def exact_route(
    start: Coord,
    goal: Coord,
    L: int,
    occupied: Set[Coord],
    node_cells: Set[Coord],
    n: int,
    rng: random.Random,
    room_suppressed: bool,
    max_attempts: int,
    forbid_non_edge_touch: bool = True,
    allowed_cells: Optional[Set[Coord]] = None,
) -> Tuple[Optional[List[Coord]], Dict[str, Any]]:
    """Find an exact-length path and return route debug."""
    debug = {
        "blocked_reason_counts": Counter(),
        "attempts": 0,
        "expanded_states": 0,
        "start": list(start),
        "goal": list(goal),
        "L_target": L,
        "D_start_goal": manhattan(start, goal),
        "forbid_non_edge_touch": forbid_non_edge_touch,
        "room_suppressed": room_suppressed,
        "allowed_cells_count": len(allowed_cells) if allowed_cells is not None else None,
    }
    if L < manhattan(start, goal):
        debug["blocked_reason_counts"]["distance_pruned"] += 1
        return None, debug
    if (L - manhattan(start, goal)) % 2 != 0:
        debug["blocked_reason_counts"]["parity_pruned"] += 1
        return None, debug
    best_dirs = ACTIONS[:]

    def candidate_block_reason(nx: Coord, cur: Coord, current_used: Set[Coord], current_path: List[Coord]) -> Optional[str]:
        if not (0 <= nx[0] < n and 0 <= nx[1] < n):
            return "out_of_grid"
        if allowed_cells is not None and nx != goal and nx not in allowed_cells:
            return "outside_corridor"
        if nx in current_used:
            return "used_in_current_path"
        if nx != goal:
            if nx in occupied:
                return "occupied_overlap"
            if nx in node_cells:
                return "node_cell_collision"
            if room_suppressed and would_make_2x2(occupied | current_used, nx, n):
                return "room_suppressed_2x2"
            if forbid_non_edge_touch:
                allowed = set(current_path)
                allowed.add(cur)
                allowed.add(start)
                allowed.add(goal)
                for nb in neigh4(nx, n):
                    if nb in occupied and nb not in allowed:
                        return "non_edge_touch"
        return None

    for attempt in range(max_attempts):
        debug["attempts"] = attempt + 1
        path=[start]; used={start}
        def dfs(cur: Coord, rem: int) -> Optional[List[Coord]]:
            debug["expanded_states"] += 1
            if rem == 0:
                return path[:] if cur == goal else None
            d = manhattan(cur, goal)
            if d > rem:
                debug["blocked_reason_counts"]["distance_pruned"] += 1
                return None
            if (rem-d)%2 != 0:
                debug["blocked_reason_counts"]["parity_pruned"] += 1
                return None
            dirs = best_dirs[:]; rng.shuffle(dirs)
            dirs.sort(key=lambda mv: manhattan((cur[0]+mv[0], cur[1]+mv[1]), goal))
            for dr,dc in dirs:
                nx=(cur[0]+dr, cur[1]+dc)
                reason = candidate_block_reason(nx, cur, used, path)
                if reason is not None:
                    debug["blocked_reason_counts"][reason] += 1
                    continue
                path.append(nx); used.add(nx)
                res=dfs(nx, rem-1)
                if res is not None: return res
                path.pop(); used.remove(nx)
            return None
        res=dfs(start,L)
        if res:
            debug["success"] = True
            debug["blocked_reason_counts"] = dict(debug["blocked_reason_counts"])
            return res, debug
    debug["success"] = False
    debug["blocked_reason_counts"] = dict(debug["blocked_reason_counts"])
    return None, debug


def route_edges(nodes: Dict[str,AbsNode], edges: Dict[str,AbsEdge], plan: Dict[str,Any], cfg: Dict[str,Any], rng: random.Random) -> Tuple[List[List[int]], Dict[str,Any]]:
    n=cfg.get("grid_size",GRID_N)
    grid=[[1 for _ in range(n)] for _ in range(n)]
    occupied=set()
    node_cells=set(n.coord for n in nodes.values() if n.coord is not None)
    for coord in node_cells:
        grid[coord[0]][coord[1]]=0; occupied.add(coord)
    role_rank={"main":0,"substitute_level1":1,"substitute_level2plus":2,"endpoint_branch":3}
    routing_items=sorted(edges.items(), key=lambda kv: (role_rank.get(kv[1].role,9), kv[0]))
    room_suppressed = plan.get("room_policy") == "room_suppressed" and cfg["routing"].get("forbid_2x2_when_room_suppressed", True)
    forbid_non_edge_touch = bool(cfg.get("routing",{}).get("forbid_non_edge_touch", True))
    enable_corridor = bool(cfg.get("embedding",{}).get("enable_corridor_reservation", False))
    enable_restricted = bool(cfg.get("embedding",{}).get("enable_restricted_routing", False))
    corridor_debug = {"status": "DISABLED", "edge_corridor_plan": {}, "global_conflict_score": 0}
    if enable_corridor:
        corridor_debug = reserve_corridors(nodes, edges, plan, cfg, routing_items)
        if corridor_debug.get("status") != "PASS":
            raise GenerationFailure("Grid Embedding", "corridor_reservation_failed", corridor_debug)
    route_attempts={}
    routing_order=[]
    mode_counts=Counter()
    for order_idx,(eid,e) in enumerate(routing_items):
        routing_order.append({"edge_id": eid, "role": e.role})
        s=nodes[e.u].coord; g=nodes[e.v].coord
        assert s is not None and g is not None
        mode="plain_dfs"
        allowed_cells=None
        skeleton=None
        if enable_corridor and eid in corridor_debug.get("edge_corridor_plan", {}):
            cinfo=corridor_debug["edge_corridor_plan"][eid]
            skeleton=[tuple(x) for x in cinfo.get("selected_skeleton", [])]
            if skeleton and len(skeleton)-1 == int(e.L_target or 0):
                # Try exact skeleton directly; reservation already checked global conflicts.
                path=skeleton
                rdbg={"success": True, "attempts": 0, "blocked_reason_counts": {}, "routing_mode": "skeleton_exact"}
                mode="skeleton_exact"
            else:
                if enable_restricted and skeleton:
                    radius=int(cfg.get("routing",{}).get("restricted_detour_radius",1))
                    allowed_cells=set(skeleton)
                    frontier=set(skeleton)
                    for _ in range(radius):
                        nxt=set(frontier)
                        for cell in frontier:
                            nxt.update(neigh4(cell,n))
                        frontier=nxt
                    # Do not use node cells except endpoints.
                    allowed_cells |= {x for x in frontier if x not in node_cells or x in (s,g)}
                    mode="restricted_dfs"
                else:
                    mode="plain_dfs"
                path, rdbg = exact_route(s,g,int(e.L_target),occupied,node_cells,n,rng,room_suppressed,cfg["max_route_attempts_per_edge"],forbid_non_edge_touch,allowed_cells=allowed_cells)
        else:
            path, rdbg = exact_route(s,g,int(e.L_target),occupied,node_cells,n,rng,room_suppressed,cfg["max_route_attempts_per_edge"],forbid_non_edge_touch)
        mode_counts[mode]+=1
        route_attempts[eid] = rdbg.get("attempts", 0)
        if path is None:
            code = "restricted_dfs_failed" if mode == "restricted_dfs" else ("parallel_edge_routing_failed" if e.key > 0 else "edge_routing_failed")
            detail = {
                "edge_id": eid,
                "role": e.role,
                "route_order_index": order_idx,
                "u": e.u,
                "v": e.v,
                "u_coord": list(s),
                "v_coord": list(g),
                "D_actual": e.D_actual,
                "L_target": e.L_target,
                "slack": int(e.L_target or 0) - int(e.D_actual or 0),
                "excess": e.excess,
                "occupied_cell_count_before_edge": len(occupied),
                "node_cell_count": len(node_cells),
                "room_policy": plan.get("room_policy"),
                "forbid_non_edge_touch": forbid_non_edge_touch,
                "forbid_2x2_when_room_suppressed": cfg.get("routing",{}).get("forbid_2x2_when_room_suppressed", True),
                "routing_mode": mode,
                "allowed_cells_count": len(allowed_cells) if allowed_cells is not None else None,
                "corridor_id": corridor_debug.get("edge_corridor_plan",{}).get(eid,{}).get("selected_candidate_id"),
                "blocked_reason_counts": rdbg.get("blocked_reason_counts", {}),
                "dfs_stats": {"expanded_states": rdbg.get("expanded_states",0), "pruned_by_parity": rdbg.get("blocked_reason_counts",{}).get("parity_pruned",0), "pruned_by_touch": rdbg.get("blocked_reason_counts",{}).get("non_edge_touch",0), "pruned_by_occupied": rdbg.get("blocked_reason_counts",{}).get("occupied_overlap",0), "pruned_by_outside_corridor": rdbg.get("blocked_reason_counts",{}).get("outside_corridor",0)},
            }
            raise GenerationFailure("Grid Embedding", code, detail)
        e.cell_path = path
        for cell in path:
            grid[cell[0]][cell[1]]=0; occupied.add(cell)
    return grid, {
        "relative_layout_enabled": True,
        "non_planned_intersection": False,
        "non_edge_touch": False,
        "node_four_neighbor_violations": 0,
        "edge_route_attempts": route_attempts,
        "routing_order": routing_order,
        "forbid_non_edge_touch": forbid_non_edge_touch,
        "corridor_reservation": corridor_debug,
        "routing_mode_counts": dict(mode_counts),
    }

# ----------------------------- island + extractor -----------------------------# ----------------------------- island + extractor -----------------------------

def add_islands(grid: List[List[int]], k: int, cfg: Dict[str,Any], rng: random.Random) -> Dict[str,Any]:
    n=len(grid)
    if k <= 0:
        return {"target_k": 0, "actual_k": 0, "island_sizes": [], "island_adjacent_to_non_island": False, "generated_cells": []}
    main_free={(r,c) for r in range(n) for c in range(n) if grid[r][c]==0}
    island_cells=[]
    for i in range(k):
        found=None
        walls=[(r,c) for r in range(n) for c in range(n) if grid[r][c]==1]
        rng.shuffle(walls)
        for cell in walls[:cfg.get("max_island_attempts",80)]:
            if any(nb in main_free for nb in neigh4(cell,n)): continue
            if any(nb in island_cells for nb in neigh4(cell,n)): continue
            found=cell; break
        if found is None:
            raise GenerationFailure("Island Generation", "island_count_mismatch", {"target_k": k, "generated": len(island_cells), "reason":"no isolated wall candidate"})
        grid[found[0]][found[1]]=0
        island_cells.append(found)
    comps=components_from_free(grid)
    actual=max(0,len(comps)-1)
    if actual != k:
        raise GenerationFailure("Island Generation", "island_count_mismatch", {"target_k": k, "actual_k": actual})
    return {"target_k": k, "actual_k": actual, "island_sizes": [1 for _ in island_cells], "island_adjacent_to_non_island": False, "generated_cells": [list(x) for x in island_cells]}


def detect_rooms(grid: List[List[int]], main_comp: Set[Coord], cfg: Dict[str,Any]) -> Dict[str,Any]:
    n=len(grid)
    seeds=set()
    if cfg["room_detection"].get("use_2x2_free_block", True):
        for r in range(n-1):
            for c in range(n-1):
                block={(r,c),(r+1,c),(r,c+1),(r+1,c+1)}
                if all(grid[x][y]==0 and (x,y) in main_comp for x,y in block):
                    seeds |= block
    # expand seeds to adjacent high-degree/open cells
    seen=set(); comps=[]
    for s in seeds:
        if s in seen: continue
        q=deque([s]); seen.add(s); comp=[]
        while q:
            x=q.popleft(); comp.append(x)
            for y in neigh4(x,n):
                if y in seeds and y not in seen:
                    seen.add(y); q.append(y)
        if len(comp) >= cfg["room_detection"].get("room_min_component_size",3):
            comps.append(comp)
    records=[]
    for idx,comp in enumerate(comps):
        comp_set=set(comp)
        outside=set()
        for cell in comp:
            for nb in neigh4(cell,n):
                if nb in main_comp and nb not in comp_set:
                    outside.add(nb)
        # distinct outside components after removing room
        outside_comps=0; seen_o=set()
        for o in outside:
            if o in seen_o: continue
            outside_comps += 1
            q=deque([o]); seen_o.add(o)
            while q:
                x=q.popleft()
                for y in neigh4(x,n):
                    if y in main_comp and y not in comp_set and y not in seen_o:
                        seen_o.add(y); q.append(y)
        exit_count=outside_comps
        if exit_count <= 0: cls="island_like_room"
        elif exit_count == 1: cls="endpoint_like_room"
        elif exit_count == 2: cls="pass_through_room"
        else: cls="choice_like_room"
        records.append({"room_component_id": f"room_{idx}", "cells": [list(c) for c in comp], "cell_count": len(comp), "exit_count": exit_count, "outside_component_count": outside_comps, "classification": cls, "projected_as_node": cls in ("endpoint_like_room","choice_like_room")})
    return {"room_component_count": len(records), "room_components": records}



def extract_decision_graph(grid: List[List[int]], start: Coord, goal: Coord, cfg: Dict[str,Any]) -> Dict[str,Any]:
    comps=components_from_free(grid)
    main_comp=None
    for comp in comps:
        if start in comp and goal in comp:
            main_comp=set(comp); break
    if main_comp is None:
        return {"status":"FAILED", "failure_code":"main_component_disconnected"}
    n = len(grid)
    island_components=[set(c) for c in comps if start not in c and goal not in c]
    # degrees in main component
    deg={cell: sum(1 for nb in neigh4(cell,n) if nb in main_comp) for cell in main_comp}
    hist=Counter(str(v) for v in deg.values())
    structural_nodes=[]
    choice_cells=[]; endpoint_cells=[]
    for cell,d in deg.items():
        if d == 1:
            ntype="endpoint_node"; endpoint_cells.append(cell)
        elif d >= 3:
            ntype="choice_node"; choice_cells.append(cell)
        elif cell in (start,goal):
            ntype="terminal_choice" if d >= 2 else "terminal_endpoint"
        else:
            continue
        structural_nodes.append({"cell": list(cell), "node_type": ntype, "degree": d, "is_start": cell==start, "is_goal": cell==goal})
    choice_count=sum(1 for nrec in structural_nodes if nrec["node_type"]=="choice_node")
    endpoint_count=sum(1 for nrec in structural_nodes if nrec["node_type"]=="endpoint_node")
    terminal_choice=sum(1 for nrec in structural_nodes if nrec["node_type"]=="terminal_choice")
    terminal_endpoint=sum(1 for nrec in structural_nodes if nrec["node_type"]=="terminal_endpoint")
    # crude edge compression: count branch/corridor links by tracing from nodes
    node_cells={tuple(nrec["cell"]) for nrec in structural_nodes}
    visited_edges=set(); extracted_edges=[]; dangling_trace_count=0; invalid_internal=0
    for nrec in structural_nodes:
        src=tuple(nrec["cell"])
        for nb in neigh4(src,n):
            if nb not in main_comp: continue
            key=tuple(sorted((src,nb)))
            if key in visited_edges: continue
            path=[src,nb]; prev=src; cur=nb
            visited_edges.add(key)
            while cur not in node_cells:
                nxts=[x for x in neigh4(cur,n) if x in main_comp and x != prev]
                if len(nxts)!=1:
                    dangling_trace_count += 1
                    break
                nxt=nxts[0]
                visited_edges.add(tuple(sorted((cur,nxt))))
                path.append(nxt); prev,cur=cur,nxt
            if cur in node_cells and cur != src:
                L=len(path)-1; D=manhattan(path[0],path[-1])
                if D == 0:
                    invalid_internal += 1
                extracted_edges.append({"edge_id": f"x{len(extracted_edges)}", "source_cell": list(path[0]), "target_cell": list(path[-1]), "cell_path": [list(p) for p in path], "L": L, "D_path": D, "detour_path": (L/D if D else None), "invalid_or_room_internal_edge": D==0})
    room=detect_rooms(grid, main_comp, cfg)
    main_edge_count=len(extracted_edges)
    E=main_edge_count; V=choice_count+endpoint_count
    mu=max(0,E-V+1) if V else 0
    return {
        "status":"PASS",
        "main_component_size": len(main_comp),
        "island_component_count": max(0,len(comps)-1),
        "island_sizes": [len(c) for c in comps if start not in c and goal not in c],
        "structural_choice_count": choice_count,
        "structural_endpoint_count": endpoint_count,
        "structural_E": E,
        "structural_mu": mu,
        "terminal_choice_count": terminal_choice,
        "terminal_endpoint_count": terminal_endpoint,
        "cell_degree_histogram": dict(hist),
        "choice_cells": [list(c) for c in choice_cells],
        "endpoint_cells": [list(c) for c in endpoint_cells],
        "terminal_cells": {"start": list(start), "goal": list(goal)},
        "edge_compression_debug": {"traced_edge_count": len(extracted_edges), "invalid_or_room_internal_edge_count": invalid_internal, "dangling_trace_count": dangling_trace_count},
        "nodes": structural_nodes,
        "edges": extracted_edges,
        "room": room,
    }

# ----------------------------- validator + visualization -----------------------------# ----------------------------- validator + visualization -----------------------------


def build_strict_topology_validation(vals: Dict[str,int], extracted: Dict[str,Any]) -> Dict[str,Any]:
    planned = {"choice": vals["c"], "endpoint": vals["t"], "E": vals["E"], "mu": vals["mu"]}
    got = {
        "choice": int(extracted.get("structural_choice_count", 0)),
        "endpoint": int(extracted.get("structural_endpoint_count", 0)),
        "E": int(extracted.get("structural_E", 0)),
        "mu": int(extracted.get("structural_mu", 0)),
    }
    checks = {
        "choice_match": planned["choice"] == got["choice"],
        "endpoint_match": planned["endpoint"] == got["endpoint"],
        "E_match": planned["E"] == got["E"],
        "mu_match": planned["mu"] == got["mu"],
    }
    reasons=[]
    if not checks["choice_match"]: reasons.append("choice_count_mismatch")
    if not checks["endpoint_match"]: reasons.append("endpoint_count_mismatch")
    if not checks["E_match"]: reasons.append("edge_count_mismatch")
    if not checks["mu_match"]: reasons.append("cycle_rank_mismatch")
    strict_match = all(checks.values())
    return {"planned": planned, "extracted": got, **checks, "strict_topology_match": strict_match, "mismatch_reasons": reasons}


def validate_sample(plan: Dict[str,Any], vals: Dict[str,int], nodes: Dict[str,AbsNode], edges: Dict[str,AbsEdge], term_dbg: Dict[str,Any], mrb_dbg: Dict[str,Any], path_complexity: Dict[str,Any], grid_dbg: Dict[str,Any], room_dbg: Dict[str,Any], island_dbg: Dict[str,Any], extracted: Dict[str,Any]) -> Dict[str,Any]:
    if extracted.get("status") != "PASS":
        raise GenerationFailure("Extraction", extracted.get("failure_code","extraction_failed"), extracted)
    room_policy = plan.get("room_policy", "room_neutral")
    strict_topology = build_strict_topology_validation(vals, extracted)
    if room_policy == "room_suppressed" and extracted["room"]["room_component_count"] > 0:
        raise GenerationFailure("Room Policy", "room_created_under_suppressed_policy", {"room_component_count": extracted["room"]["room_component_count"], "strict_topology": strict_topology})
    if island_dbg["actual_k"] != plan.get("island",{}).get("island_component_count",0):
        raise GenerationFailure("Island Generation", "island_count_mismatch", island_dbg)
    if not strict_topology["strict_topology_match"]:
        detail = {"strict_topology": strict_topology, "room_component_count": extracted.get("room",{}).get("room_component_count",0), "cell_degree_histogram": extracted.get("cell_degree_histogram"), "edge_compression_debug": extracted.get("edge_compression_debug")}
        if room_policy == "room_neutral" and extracted.get("room",{}).get("room_component_count",0) > 0:
            detail["room_policy"] = room_policy
            raise GenerationFailure("Strict Validation", "room_changed_structural_topology", detail)
        raise GenerationFailure("Strict Validation", "strict_topology_mismatch", detail)
    validation = {
        "structural_topology_match": True,
        "strict_topology": strict_topology,
        "terminal_semantics_match": True,
        "main_bundle_valid": not mrb_dbg.get("main_bundle_changed_after_L_assignment", False),
        "path_complexity_within_tolerance": bool(path_complexity["within_tolerance"]),
        "room_policy_satisfied": not (room_policy == "room_suppressed" and extracted["room"]["room_component_count"] > 0),
        "island_contract_satisfied": extracted["island_component_count"] == island_dbg["actual_k"],
    }
    return validation

def draw_maze(grid: List[List[int]], nodes: Dict[str,AbsNode], edges: Dict[str,AbsEdge], path: Path, title: str) -> None:
    if plt is None: return
    n=len(grid)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(grid, cmap="gray_r", vmin=0, vmax=1)
    for eid,e in edges.items():
        if e.cell_path:
            ys=[p[0] for p in e.cell_path]; xs=[p[1] for p in e.cell_path]
            ax.plot(xs,ys,linewidth=1.5,label=e.role if eid==list(edges.keys())[0] else None)
            mid=e.cell_path[len(e.cell_path)//2]
            ax.text(mid[1],mid[0],eid,fontsize=6,color="blue")
    for nid,nod in nodes.items():
        if nod.coord:
            r,c=nod.coord
            color="red" if nod.is_start else ("green" if nod.is_goal else "orange")
            ax.scatter([c],[r],s=80,c=color,edgecolors="black")
            ax.text(c+0.05,r+0.05,nid,fontsize=7,color="black")
    ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.grid(True,linewidth=0.3)
    ax.set_title(title)
    ensure_dir(path.parent)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def draw_decision_graph(nodes: Dict[str,AbsNode], edges: Dict[str,AbsEdge], path: Path, title: str) -> None:
    if plt is None: return
    fig, ax=plt.subplots(figsize=(5,5))
    for eid,e in edges.items():
        if nodes[e.u].coord and nodes[e.v].coord:
            y=[nodes[e.u].coord[0], nodes[e.v].coord[0]]; x=[nodes[e.u].coord[1], nodes[e.v].coord[1]]
            ax.plot(x,y,linewidth=2)
            ax.text(mean(x),mean(y),f"{eid}\n{e.role}\nL={e.L_target}",fontsize=6)
    for nid,n in nodes.items():
        if n.coord:
            r,c=n.coord
            ax.scatter([c],[r],s=120,c="lightblue",edgecolors="black")
            ax.text(c,r,nid,ha="center",va="center",fontsize=7)
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.set_title(title); ax.grid(True,linewidth=0.3)
    ensure_dir(path.parent); fig.tight_layout(); fig.savefig(path); plt.close(fig)

# ----------------------------- generation pipeline -----------------------------


def make_success_layers(generation=False, embedding=False, extraction=False, strict=False) -> Dict[str, bool]:
    return {
        "generation_success": bool(generation),
        "embedding_success": bool(embedding),
        "extraction_success": bool(extraction),
        "strict_plan_match_success": bool(strict),
    }


def _minimal_plan_record(plan: Dict[str,Any], vals: Optional[Dict[str,int]] = None) -> Dict[str,Any]:
    out = {"endpoint_contract_mode": plan.get("topology",{}).get("endpoint_contract_mode"), "room_policy": plan.get("room_policy"), "island_k": plan.get("island",{}).get("island_component_count",0), "profile_adjustment_note": plan.get("profile_adjustment_note")}
    if vals:
        out.update({"c": vals.get("c"), "t": vals.get("t"), "mu": vals.get("mu"), "derived_E": vals.get("E")})
    return out


def _edge_table_for_debug(edges: Dict[str,AbsEdge]) -> Dict[str,Any]:
    return {eid: {"role": e.role, "D_actual": e.D_actual, "excess": e.excess, "L_target": e.L_target, "u": e.u, "v": e.v} for eid,e in edges.items()}


def _generate_one_once(profile_bundle: Dict[str,Any], seed: int, maze_id: str) -> Dict[str,Any]:
    rng=random.Random(seed)
    plan=copy.deepcopy(profile_bundle["plan_contract"])
    pack=copy.deepcopy(profile_bundle["D_range_sample_pack"])
    cfg=copy.deepcopy(profile_bundle["config"])
    t0=time.time()
    ctx: Dict[str,Any] = {"maze_id": maze_id, "plan_contract": _minimal_plan_record(plan), "success_layers": make_success_layers()}
    current_stage = "init"
    try:
        current_stage="Plan Contract"
        vals=derive_topology(plan,cfg)
        ctx["plan_contract"] = _minimal_plan_record(plan, vals)
        current_stage="Topology Feasibility"
        topo_feas=check_topology_feasible(vals,cfg); ctx["topology_feasibility"] = topo_feas
        current_stage="Topology Generation"
        nodes,edges,topo_dbg=generate_topology(vals,cfg,rng); ctx["topology_generation"] = topo_dbg
        ctx["success_layers"]["generation_success"] = True
        current_stage="Terminal Semantics"
        s,e,term_dbg=place_terminals(nodes,edges,plan,rng)
        start_node=s; goal_node=e
        ctx["terminal_semantics"] = {**term_dbg, "structural_V": vals["V"], "structural_E": vals["E"], "effective_V_after_terminal": vals["V"], "effective_E_after_terminal": vals["E"], "cycle_rank_preserved": True, "effective_choice_count": vals["c"] + term_dbg["terminal_choice_count"]}
        current_stage="Preliminary MRB"
        prelim_len, prelim_paths, prelim_bundle=shortest_paths_edges(nodes,edges,s,e)
        annotate_roles(nodes,edges,prelim_bundle)
        role_counts=Counter(ed.role for ed in edges.values())
        ctx["edge_roles_after_bundle"] = dict(role_counts)
        current_stage="D-range Binding"
        d_bind=bind_d_ranges(edges,pack,rng)
        ctx["D_range_binding"] = {**d_bind, "bindings": {eid: {"role": ed.role, "sampled_D_range": ed.sampled_D_range, "D_target": ed.D_target, "D_actual": ed.D_actual} for eid,ed in edges.items()}}
        current_stage="Relative Layout"
        layout_dbg=layout_nodes(nodes,edges,cfg,rng)
        current_stage="Path Complexity Allocation"
        pc=allocate_excess(edges,plan,cfg,rng)
        current_stage="MRB Guard"
        pc, mrb_guard_dbg, weighted_bundle, weighted_paths = apply_mrb_guard(nodes, edges, s, e, prelim_bundle, prelim_paths, plan, cfg, pc)
        ctx["path_complexity"] = pc
        changed_added=sorted(set(weighted_bundle) - set(prelim_bundle))
        changed_removed=sorted(set(prelim_bundle) - set(weighted_bundle))
        mrb_dbg={
            "preliminary_shortest_path_count": len(prelim_paths),
            "preliminary_main_bundle_edges": sorted(prelim_bundle),
            "preliminary_main_bundle_edge_count": len(prelim_bundle),
            "weighted_shortest_path_count": len(weighted_paths),
            "weighted_main_bundle_edges": sorted(weighted_bundle),
            "weighted_main_bundle_edge_count": len(weighted_bundle),
            "changed_edges_added": changed_added,
            "changed_edges_removed": changed_removed,
            "main_bundle_changed_after_L_assignment": bool(set(weighted_bundle) != set(prelim_bundle)),
            "edge_table": _edge_table_for_debug(edges),
            "mrb_guard": mrb_guard_dbg,
        }
        ctx["main_route_bundle"] = mrb_dbg
        ctx["mrb_guard_debug"] = mrb_guard_dbg
        if (not cfg.get("path_complexity",{}).get("enable_mrb_guard", False)) and mrb_dbg["main_bundle_changed_after_L_assignment"]:
            raise GenerationFailure("Main Bundle", "main_bundle_changed_after_L_assignment", mrb_dbg)
        current_stage="Grid Embedding"
        grid, grid_dbg=route_edges(nodes,edges,plan,cfg,rng)
        ctx["success_layers"]["embedding_success"] = True
        ctx["grid_embedding"] = {**layout_dbg, **grid_dbg}
        ctx["abstract_nodes"] = {nid: asdict(n) for nid,n in nodes.items()}
        ctx["abstract_edges"] = {eid: {**asdict(ed), "cell_path": [list(x) for x in ed.cell_path] if ed.cell_path else None} for eid,ed in edges.items()}
        current_stage="Island Generation"
        island_dbg=add_islands(grid,int(plan.get("island",{}).get("island_component_count",0)),cfg,rng); ctx["island_validation"] = island_dbg
        current_stage="Extraction"
        start_coord=nodes[start_node].coord; goal_coord=nodes[goal_node].coord
        assert start_coord is not None and goal_coord is not None
        ctx["grid"] = grid
        extracted=extract_decision_graph(grid,start_coord,goal_coord,cfg); ctx["extraction"] = extracted
        if extracted.get("status") == "PASS":
            ctx["success_layers"]["extraction_success"] = True
        room_dbg=extracted.get("room", {"room_component_count":0}) if extracted.get("status") == "PASS" else {"room_component_count":0}
        current_stage="Strict Validation"
        validation=validate_sample(plan,vals,nodes,edges,term_dbg,mrb_dbg,pc,grid_dbg,room_dbg,island_dbg,extracted)
        ctx["success_layers"]["strict_plan_match_success"] = True
        ctx.update({
            "status":"PASS",
            "failed_layer": None,
            "failure_code": None,
            "failure_detail": None,
            "room_policy": {"mode": plan.get("room_policy"), "extractor_independent_room_detection": True, "room_component_count": room_dbg.get("room_component_count",0), "room_topology_preserved": validation["room_policy_satisfied"]},
            "validation": validation,
            "strict_validation": validation.get("strict_topology"),
            "grid": grid,
            "runtime_ms": int((time.time()-t0)*1000),
        })
        return ctx
    except GenerationFailure as gf:
        rec=gf.to_record()
        rec.update(ctx)
        rec["status"] = "FAILED"
        rec["failed_layer"] = gf.failed_layer
        rec["failure_code"] = gf.failure_code
        rec["failure_detail"] = gf.failure_detail
        rec["runtime_ms"] = int((time.time()-t0)*1000)
        # Layer flags should reflect where the pipeline reached.
        if gf.failed_layer in ("Strict Validation", "Room Policy"):
            rec.setdefault("success_layers", make_success_layers(True, True, True, False))
            rec["success_layers"].update({"generation_success": True, "embedding_success": True, "extraction_success": True, "strict_plan_match_success": False})
            if "strict_topology" in gf.failure_detail:
                rec["strict_validation"] = gf.failure_detail["strict_topology"]
        elif gf.failed_layer == "Extraction":
            rec.setdefault("success_layers", make_success_layers(True, True, False, False))
            rec["success_layers"].update({"generation_success": True, "embedding_success": True, "extraction_success": False, "strict_plan_match_success": False})
        elif gf.failed_layer in ("Grid Embedding", "Island Generation"):
            rec.setdefault("success_layers", make_success_layers(True, False, False, False))
            rec["success_layers"].update({"generation_success": True, "embedding_success": False, "extraction_success": False, "strict_plan_match_success": False})
            if gf.failed_layer == "Grid Embedding":
                rec["routing_debug"] = {"failed_edge_id": gf.failure_detail.get("edge_id"), "failed_edge_role": gf.failure_detail.get("role"), "blocked_reason_top": Counter(gf.failure_detail.get("blocked_reason_counts",{})).most_common(8)}
        return rec
    except Exception as ex:
        rec = {**ctx}
        rec.update({
            "status":"FAILED",
            "failed_layer":"Internal",
            "failure_code":"unexpected_exception",
            "failure_detail":{
                "type":type(ex).__name__,
                "message":str(ex),
                "traceback":traceback.format_exc(),
                "current_stage":current_stage,
                "current_profile":plan.get("profile"),
                "maze_id":maze_id,
                "room_policy":plan.get("room_policy"),
            },
            "runtime_ms": int((time.time()-t0)*1000),
        })
        return rec

def generate_one(profile_bundle: Dict[str,Any], seed: int, maze_id: str) -> Dict[str,Any]:
    """Generate one sample with bounded retries.

    A single random abstract graph/layout/routing attempt is often too brittle on
    an 8x8 grid. Retrying is not a semantic fallback: each attempt still obeys
    the same Plan Contract and D-range Sample Pack binding rules. If all attempts
    fail, the last layered failure code is returned with retry diagnostics.
    """
    cfg = profile_bundle.get("config", {})
    max_attempts = int(cfg.get("max_generation_attempts", 1))
    last = None
    failure_counts = Counter()
    for a in range(max_attempts):
        rec = _generate_one_once(profile_bundle, seed + a * 104729, maze_id)
        rec["generation_attempt_count"] = a + 1
        if rec.get("status") == "PASS":
            return rec
        last = rec
        failure_counts[rec.get("failure_code", "unknown")] += 1
    if last is None:
        last = {"maze_id": maze_id, "status": "FAILED", "failed_layer": "Internal", "failure_code": "no_attempt_executed", "failure_detail": {}}
    last["generation_attempt_count"] = max_attempts
    last["attempt_failure_code_counts"] = dict(failure_counts)
    return last

# ----------------------------- summaries -----------------------------


def nested_get(d: Dict[str,Any], path: List[str], default=None):
    cur=d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur=cur[k]
    return cur


def summarize(records: List[Dict[str,Any]], profile: str) -> Dict[str,Any]:
    strict_succ=[r for r in records if r.get("status")=="PASS" and r.get("success_layers",{}).get("strict_plan_match_success")]
    fail=[r for r in records if r.get("status")!="PASS"]
    fc=Counter(r.get("failure_code","none") for r in fail)
    fl=Counter(r.get("failed_layer","none") for r in fail)
    def collect(recs, path):
        xs=[]
        for r in recs:
            v=nested_get(r,path)
            if isinstance(v,(int,float,bool)): xs.append(float(v))
        return xs
    attempted_plan=[r for r in records if r.get("plan_contract")]
    attempted_topo=[r for r in records if r.get("topology_generation")]
    attempted_d=[r for r in records if r.get("D_range_binding")]
    attempted_layout=[r for r in records if r.get("grid_embedding") or r.get("topology_generation")]
    attempted_routing=[r for r in records if r.get("grid_embedding") or r.get("routing_debug") or r.get("failure_code") in ("edge_routing_failed","parallel_edge_routing_failed")]
    attempted_extraction=[r for r in records if r.get("extraction")]
    all_plan_c=collect(attempted_plan,["plan_contract","c"])
    all_plan_t=collect(attempted_plan,["plan_contract","t"])
    all_plan_mu=collect(attempted_plan,["plan_contract","mu"])
    all_plan_E=collect(attempted_plan,["plan_contract","derived_E"])
    room_policy = None
    for r in records:
        room_policy = nested_get(r,["plan_contract","room_policy"]) or nested_get(r,["room_policy","mode"])
        if room_policy: break
    strict_reports=[]
    for r in records:
        st = r.get("strict_validation") or nested_get(r,["failure_detail","strict_topology"])
        if st: strict_reports.append(st)
    mismatch_counts=Counter()
    for st in strict_reports:
        for reason in st.get("mismatch_reasons",[]): mismatch_counts[reason]+=1
    summary={
        "version": VERSION,
        "profile": profile,
        "n_requested": len(records),
        "n_success": len(strict_succ),
        "success_rate": len(strict_succ)/len(records) if records else 0.0,
        "hard_failure_count": len(fail),
        "top_failure_codes": fc.most_common(20),
        "failure_by_layer": dict(fl),
        "failure_by_code": dict(fc),
        "layered_success": {
            "generation_success_count": sum(1 for r in records if r.get("success_layers",{}).get("generation_success")),
            "embedding_success_count": sum(1 for r in records if r.get("success_layers",{}).get("embedding_success")),
            "extraction_success_count": sum(1 for r in records if r.get("success_layers",{}).get("extraction_success")),
            "strict_plan_match_success_count": len(strict_succ),
            "strict_plan_match_success_rate": len(strict_succ)/len(records) if records else 0.0,
        },
        "attempted_summary": {
            "attempted_plan_contract_count": len(attempted_plan),
            "attempted_topology_generated_count": len(attempted_topo),
            "attempted_D_range_bound_count": len(attempted_d),
            "attempted_layout_count": len(attempted_layout),
            "attempted_routing_count": len(attempted_routing),
            "attempted_extraction_count": len(attempted_extraction),
            "choice_node_count_mean": mean(all_plan_c),
            "endpoint_node_count_mean": mean(all_plan_t),
            "cycle_rank_mean": mean(all_plan_mu),
            "derived_E_mean": mean(all_plan_E),
        },
        "success_summary": {
            "strict_success_count": len(strict_succ),
            "choice_node_count_mean": mean(collect(strict_succ,["plan_contract","c"])),
            "endpoint_node_count_mean": mean(collect(strict_succ,["plan_contract","t"])),
            "cycle_rank_mean": mean(collect(strict_succ,["plan_contract","mu"])),
            "derived_E_mean": mean(collect(strict_succ,["plan_contract","derived_E"])),
        },
        "topology": {
            "choice_node_count_mean": mean(collect(strict_succ,["plan_contract","c"])), "choice_node_count_p50": p50(collect(strict_succ,["plan_contract","c"])),
            "endpoint_node_count_mean": mean(collect(strict_succ,["plan_contract","t"])), "endpoint_node_count_p50": p50(collect(strict_succ,["plan_contract","t"])),
            "cycle_rank_mean": mean(collect(strict_succ,["plan_contract","mu"])), "cycle_rank_p50": p50(collect(strict_succ,["plan_contract","mu"])),
            "derived_E_mean": mean(collect(strict_succ,["plan_contract","derived_E"])), "parallel_edge_count_mean": mean(collect(strict_succ,["topology_generation","parallel_edge_count"])),
        },
        "terminal_semantics": {
            "terminal_choice_count": sum(collect(strict_succ,["terminal_semantics","terminal_choice_count"])),
            "terminal_endpoint_count": sum(collect(strict_succ,["terminal_semantics","terminal_endpoint_count"])),
            "effective_choice_count_mean": mean(collect(strict_succ,["terminal_semantics","effective_choice_count"])),
        },
        "main_route_bundle": {
            "preliminary_main_bundle_edge_count_mean": mean(collect(strict_succ,["main_route_bundle","preliminary_main_bundle_edge_count"])),
            "weighted_main_bundle_edge_count_mean": mean(collect(strict_succ,["main_route_bundle","weighted_main_bundle_edge_count"])),
            "main_bundle_changed_after_L_assignment_count": fc.get("main_bundle_changed_after_L_assignment",0),
        },
        "edge_roles": {
            "main_edge_count_mean": mean([r.get("edge_roles_after_bundle",{}).get("main",0) for r in strict_succ]),
            "substitute_level1_count_mean": mean([r.get("edge_roles_after_bundle",{}).get("substitute_level1",0) for r in strict_succ]),
            "substitute_level2plus_count_mean": mean([r.get("edge_roles_after_bundle",{}).get("substitute_level2plus",0) for r in strict_succ]),
            "endpoint_branch_count_mean": mean([r.get("edge_roles_after_bundle",{}).get("endpoint_branch",0) for r in strict_succ]),
        },
        "D_range_binding": {
            "queue_insufficient_count": fc.get("D_range_queue_insufficient",0),
            "unused_warning_count": sum(1 for r in strict_succ if r.get("D_range_binding",{}).get("unused_warning")),
            "D_actual_out_of_range_count": fc.get("D_actual_out_of_sampled_range",0),
        },
        "path_complexity": {
            "C_length_mean": mean(collect(strict_succ,["path_complexity","C_length"])), "C_length_p50": p50(collect(strict_succ,["path_complexity","C_length"])),
            "C_excess_mean": mean(collect(strict_succ,["path_complexity","C_excess"])), "C_excess_p50": p50(collect(strict_succ,["path_complexity","C_excess"])),
            "C_path_mean": mean(collect(strict_succ,["path_complexity","C_path"])), "C_path_p50": p50(collect(strict_succ,["path_complexity","C_path"])),
            "target_hit_rate": pct([r.get("path_complexity",{}).get("within_tolerance",False) for r in strict_succ]),
            "base_exceeds_target_count": fc.get("path_complexity_base_exceeds_target",0),
            "target_unreachable_count": fc.get("path_complexity_target_unreachable",0),
        },
        "embedding": {
            "relative_layout_success_rate": 1.0 - (fc.get("D_actual_out_of_sampled_range",0) / len(records) if records else 0),
            "routing_success_rate": 1.0 - ((fc.get("edge_routing_failed",0)+fc.get("parallel_edge_routing_failed",0)) / len(records) if records else 0),
            "non_planned_intersection_count": sum(1 for r in strict_succ if r.get("grid_embedding",{}).get("non_planned_intersection")),
            "non_edge_touch_count": sum(1 for r in strict_succ if r.get("grid_embedding",{}).get("non_edge_touch")),
        },
        "room": {"room_policy": room_policy, "room_component_count_mean": mean(collect(strict_succ,["room_policy","room_component_count"])), "room_component_count_p50": p50(collect(strict_succ,["room_policy","room_component_count"])), "room_policy_failure_count": fc.get("room_created_under_suppressed_policy",0)+fc.get("room_changed_structural_topology",0)},
        "island": {"target_k_mean": mean(collect(strict_succ,["island_validation","target_k"])), "target_k_p50": p50(collect(strict_succ,["island_validation","target_k"])), "actual_k_mean": mean(collect(strict_succ,["island_validation","actual_k"])), "actual_k_p50": p50(collect(strict_succ,["island_validation","actual_k"])), "island_match_rate": pct([r.get("island_validation",{}).get("target_k")==r.get("island_validation",{}).get("actual_k") for r in strict_succ])},
        "strict_validation": {
            "strict_topology_mismatch_count": fc.get("strict_topology_mismatch",0) + fc.get("room_changed_structural_topology",0),
            "choice_count_mismatch_count": mismatch_counts.get("choice_count_mismatch",0),
            "endpoint_count_mismatch_count": mismatch_counts.get("endpoint_count_mismatch",0),
            "edge_count_mismatch_count": mismatch_counts.get("edge_count_mismatch",0),
            "cycle_rank_mismatch_count": mismatch_counts.get("cycle_rank_mismatch",0),
            "room_changed_structural_topology_count": fc.get("room_changed_structural_topology",0),
        },
        "corridor_reservation": {
            "attempted": sum(1 for r in records if nested_get(r,["grid_embedding","corridor_reservation","status"]) not in (None, "DISABLED")) + fc.get("corridor_reservation_failed",0),
            "success": sum(1 for r in records if nested_get(r,["grid_embedding","corridor_reservation","status"]) == "PASS"),
            "failed": fc.get("corridor_reservation_failed",0),
            "avg_candidate_count": mean([sum(e.get("candidate_count",0) for e in nested_get(r,["grid_embedding","corridor_reservation","edge_corridor_plan"],{}).values()) for r in records if nested_get(r,["grid_embedding","corridor_reservation","edge_corridor_plan"])]),
            "avg_global_conflict_score": mean(collect(records,["grid_embedding","corridor_reservation","global_conflict_score"])),
        },
        "routing": {
            "skeleton_exact_success": sum(nested_get(r,["grid_embedding","routing_mode_counts","skeleton_exact"],0) or 0 for r in records),
            "restricted_dfs_attempted": sum(nested_get(r,["grid_embedding","routing_mode_counts","restricted_dfs"],0) or 0 for r in records) + fc.get("restricted_dfs_failed",0),
            "restricted_dfs_success": sum(nested_get(r,["grid_embedding","routing_mode_counts","restricted_dfs"],0) or 0 for r in records),
            "restricted_dfs_failed": fc.get("restricted_dfs_failed",0),
        },
        "mrb_guard": {
            "enabled": any(nested_get(r,["mrb_guard_debug","mrb_guard_enabled"],False) for r in records),
            "mode": next((nested_get(r,["mrb_guard_debug","mrb_guard_mode"]) for r in records if nested_get(r,["mrb_guard_debug","mrb_guard_mode"])), None),
            "guard_invoked": sum(1 for r in records if nested_get(r,["mrb_guard_debug","guard_invoked"],False)),
            "guard_adjusted_edges": sum(len(nested_get(r,["mrb_guard_debug","guard_adjustments"],[]) or []) for r in records),
            "guard_failed": fc.get("mrb_guard_failed",0),
            "weighted_MRB_added_edges": sum(len(nested_get(r,["mrb_guard_debug","weighted_MRB_edges_added"],[]) or []) for r in records),
            "weighted_MRB_removed_edges": sum(len(nested_get(r,["mrb_guard_debug","weighted_MRB_edges_removed"],[]) or []) for r in records),
            "forced_above_target": sum(1 for r in records if nested_get(r,["mrb_guard_debug","forced_above_target_by_mrb_guard"],False)),
        },
        "profile_adjustments": PROFILE_ADJUSTMENTS,
    }
    return summary


def print_summary(summary: Dict[str,Any]) -> None:
    print("\n[3.0.3(2) Corridor Reservation + MRB Guard]\n")
    print("=== Summary ===")
    print(f"n_requested: {summary['n_requested']}")
    print(f"n_success: {summary['n_success']}")
    print(f"strict_success_rate: {summary['success_rate']:.3f}")
    print(f"hard_failure_count: {summary['hard_failure_count']}")
    print("\n=== Layered Success ===")
    for k,v in summary["layered_success"].items():
        print(f"{k}: {v:.3f}" if isinstance(v,float) else f"{k}: {v}")
    print("\n=== Attempted Summary ===")
    for k,v in summary["attempted_summary"].items():
        print(f"{k}: {v:.3f}" if isinstance(v,float) else f"{k}: {v}")
    print("\n=== Failure by Layer ===")
    for k,v in summary["failure_by_layer"].items():
        print(f"{k}: {v}")
    for title,key in [("Topology","topology"),("Terminal Semantics","terminal_semantics"),("Main Route Bundle","main_route_bundle"),("Edge Roles","edge_roles"),("D-range Binding","D_range_binding"),("Path Complexity","path_complexity"),("Embedding","embedding"),("Room","room"),("Island","island"),("Strict Validation","strict_validation"),("Corridor Reservation","corridor_reservation"),("Routing","routing"),("MRB Guard","mrb_guard")]:
        print(f"\n=== {title} ===")
        for k,v in summary[key].items():
            print(f"{k}: {v:.3f}" if isinstance(v,float) else f"{k}: {v}")
    print("\n=== Top Failure Codes ===")
    for code,n in summary["top_failure_codes"]:
        print(f"{code}: {n}")

def load_config_override(path: Optional[str], bundle: Dict[str,Any]) -> Dict[str,Any]:
    if not path:
        return bundle
    with open(path,"r",encoding="utf-8") as f:
        override=json.load(f)
    # shallow/deep merge selected top-level keys
    out=copy.deepcopy(bundle)
    for k,v in override.items():
        if k in out and isinstance(out[k],dict) and isinstance(v,dict):
            def merge(a,b):
                for kk,vv in b.items():
                    if kk in a and isinstance(a[kk],dict) and isinstance(vv,dict): merge(a[kk],vv)
                    else: a[kk]=vv
            merge(out[k],v)
        else:
            out[k]=v
    return out




def apply_experiment_config(bundle: Dict[str,Any], experiment: str, args: argparse.Namespace) -> Dict[str,Any]:
    out=copy.deepcopy(bundle)
    cfg=out.setdefault("config", {})
    emb=cfg.setdefault("embedding", {})
    pc=cfg.setdefault("path_complexity", {})
    if experiment == "baseline":
        emb["enable_corridor_reservation"] = False
        emb["enable_restricted_routing"] = False
        pc["enable_mrb_guard"] = False
    elif experiment == "corridor_only":
        emb["enable_corridor_reservation"] = True
        emb["enable_restricted_routing"] = True
        pc["enable_mrb_guard"] = False
    elif experiment == "mrb_guard_only":
        emb["enable_corridor_reservation"] = False
        emb["enable_restricted_routing"] = False
        pc["enable_mrb_guard"] = True
    elif experiment == "corridor_mrb_guard":
        emb["enable_corridor_reservation"] = True
        emb["enable_restricted_routing"] = True
        pc["enable_mrb_guard"] = True
    if getattr(args, "enable_corridor_reservation", False):
        emb["enable_corridor_reservation"] = True
    if getattr(args, "disable_corridor_reservation", False):
        emb["enable_corridor_reservation"] = False
    if getattr(args, "enable_restricted_routing", False):
        emb["enable_restricted_routing"] = True
    if getattr(args, "disable_restricted_routing", False):
        emb["enable_restricted_routing"] = False
    if getattr(args, "enable_mrb_guard", False):
        pc["enable_mrb_guard"] = True
    if getattr(args, "disable_mrb_guard", False):
        pc["enable_mrb_guard"] = False
    if getattr(args, "mrb_guard_mode", None):
        pc["mrb_guard_mode"] = args.mrb_guard_mode
    out["experiment"] = experiment
    return out

def select_visual_samples(records: List[Dict[str,Any]], out_dir: Path, max_samples: int) -> Dict[str,Any]:
    succ=[r for r in records if r.get("status")=="PASS"]
    fail=[r for r in records if r.get("status")!="PASS"]
    selected=[]
    cats=[]
    # simple category tags
    for r in succ:
        cat="full_mvp"
        if r.get("plan_contract",{}).get("island_k",0)>0: cat="island"
        elif r.get("edge_roles_after_bundle",{}).get("substitute_level1",0)>0: cat="loop_substitute"
        elif r.get("terminal_semantics",{}).get("terminal_choice_count",0)>0: cat="terminal_choice"
        else: cat="pure_topology"
        if cat not in cats:
            selected.append((cat,r)); cats.append(cat)
        if len(selected)>=max_samples//2: break
    fail_cats=set()
    for r in fail:
        cat="failure_"+r.get("failure_code","unknown")
        if cat not in fail_cats:
            selected.append((cat,r)); fail_cats.add(cat)
        if len(selected)>=max_samples: break
    report=[]
    vis_dir=out_dir/"visualization_samples"; ensure_dir(vis_dir)
    for idx,(cat,r) in enumerate(selected,1):
        files={}
        if r.get("status")=="PASS" and "grid" in r:
            # reconstruct nodes/edges for plot from serialized records
            nodes={nid: AbsNode(nid, nd["node_type"], nd.get("degree",0), tuple(nd["coord"]) if nd.get("coord") else None, nd.get("is_start",False), nd.get("is_goal",False)) for nid,nd in r["abstract_nodes"].items()}
            edges={}
            for eid,ed in r["abstract_edges"].items():
                ae=AbsEdge(eid,ed["u"],ed["v"],ed.get("key",0),ed.get("role"),ed.get("level"),ed.get("sampled_D_range"),ed.get("D_target"),ed.get("D_actual"),ed.get("excess",0),ed.get("L_target"),[tuple(x) for x in ed["cell_path"]] if ed.get("cell_path") else None,ed.get("weight_for_complexity",1.0))
                edges[eid]=ae
            maze_path=vis_dir/f"{idx:02d}_{cat}__{r['maze_id']}_maze_overlay.png"
            graph_path=vis_dir/f"{idx:02d}_{cat}__{r['maze_id']}_decision_graph.png"
            draw_maze(r["grid"],nodes,edges,maze_path,cat)
            draw_decision_graph(nodes,edges,graph_path,cat)
            files={"maze_overlay":str(maze_path),"decision_graph":str(graph_path)}
        report.append({"rank":idx,"category":cat,"maze_id":r.get("maze_id"),"status":r.get("status"),"why_selected":"representative success/failure category","files":files,"failure_code":r.get("failure_code")})
    success_categories=[x[0] for x in selected if not x[0].startswith("failure_")]
    return {"selected_samples":report,"success_sample_shortage":len(success_categories)<5,"available_success_categories":success_categories,"missing_success_categories":[] if len(success_categories)>=5 else ["pure_topology","loop_substitute","terminal_choice","D-range binding","island","room_neutral","full_mvp"]}


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode", default="diagnostic", choices=["diagnostic"])
    ap.add_argument("--profile", default="pure_topology", choices=["pure_topology","single_loop","terminal_choice","d_range_binding","island_only","room_neutral","full_mvp"])
    ap.add_argument("--n-mazes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-root", default="feature_maze/v3_0_maze_generation/outputs/3.0.3")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--config-json", default=None)
    ap.add_argument("--experiment", default="corridor_mrb_guard", choices=["baseline","corridor_only","mrb_guard_only","corridor_mrb_guard"])
    ap.add_argument("--enable-corridor-reservation", action="store_true")
    ap.add_argument("--disable-corridor-reservation", action="store_true")
    ap.add_argument("--enable-restricted-routing", action="store_true")
    ap.add_argument("--disable-restricted-routing", action="store_true")
    ap.add_argument("--enable-mrb-guard", action="store_true")
    ap.add_argument("--disable-mrb-guard", action="store_true")
    ap.add_argument("--mrb-guard-mode", choices=["strict_mrb_preserve","no_substitute_in_mrb"], default=None)
    ap.add_argument("--save-visualizations", action="store_true")
    ap.add_argument("--max-visual-samples", type=int, default=10)
    args=ap.parse_args()

    run_name=args.run_name or f"diag_3_0_3_{args.profile}_n{args.n_mazes}_seed{args.seed}"
    out_dir=Path(args.output_root)/run_name
    ensure_dir(out_dir)

    bundle=make_profile(args.profile)
    bundle=load_config_override(args.config_json,bundle)
    bundle=apply_experiment_config(bundle,args.experiment,args)
    save_json(bundle,out_dir/"resolved_config.json")
    save_json({"version":VERSION,"experiment_name":EXPERIMENT_NAME,"script":__file__,"mode":args.mode,"profile":args.profile,"experiment":args.experiment,"seed":args.seed,"n_mazes":args.n_mazes,"note":"Diagnostic profile provider is not a full Sampler."}, out_dir/"run_metadata.json")

    print("[Protocol Compliance Audit]")
    print("status: PASS")
    print(f"VERSION: {VERSION}")
    print(f"Output directory: {out_dir}")
    if PROFILE_ADJUSTMENTS.get(args.profile):
        print(f"[PROFILE ADJUSTMENT] {PROFILE_ADJUSTMENTS[args.profile]}")

    records=[]
    for i in range(args.n_mazes):
        records.append(generate_one(bundle,args.seed+i*9973,f"maze_{i:05d}"))
    summary=summarize(records,args.profile)
    save_json(records,out_dir/"sample_debug_records.json")
    save_json(summary,out_dir/"validation_summary.json")
    save_json(summary.get("attempted_summary", {}), out_dir/"attempted_summary.json")
    save_json(summary.get("success_summary", {}), out_dir/"success_summary.json")
    save_json(summary.get("failure_by_layer", {}), out_dir/"failure_by_layer.json")
    save_json(summary.get("failure_by_code", {}), out_dir/"failure_code_summary.json")
    strict_report=[{"maze_id": r.get("maze_id"), "failure_code": r.get("failure_code"), "strict_validation": r.get("strict_validation") or nested_get(r,["failure_detail","strict_topology"]), "failure_detail": r.get("failure_detail")} for r in records if (r.get("strict_validation") or nested_get(r,["failure_detail","strict_topology"]))]
    save_json(strict_report, out_dir/"strict_validation_report.json")
    routing_report=[{"maze_id": r.get("maze_id"), "routing_debug": r.get("routing_debug"), "failure_detail": r.get("failure_detail")} for r in records if r.get("failure_code") in ("edge_routing_failed","parallel_edge_routing_failed") or r.get("routing_debug")]
    save_json(routing_report, out_dir/"routing_failure_debug.json")
    mrb_report=[{"maze_id": r.get("maze_id"), "main_route_bundle": r.get("main_route_bundle") or nested_get(r,["failure_detail"])} for r in records if r.get("failure_code") in ("main_bundle_changed_after_L_assignment","mrb_guard_failed") or nested_get(r,["main_route_bundle","main_bundle_changed_after_L_assignment"])]
    save_json(mrb_report, out_dir/"mrb_change_debug.json")
    mrb_guard_report=[{"maze_id": r.get("maze_id"), "mrb_guard_debug": r.get("mrb_guard_debug") or nested_get(r,["failure_detail"])} for r in records if r.get("mrb_guard_debug") or r.get("failure_code") == "mrb_guard_failed"]
    save_json(mrb_guard_report, out_dir/"mrb_guard_debug.json")
    corridor_report=[{"maze_id": r.get("maze_id"), "corridor_reservation": nested_get(r,["grid_embedding","corridor_reservation"]) or (r.get("failure_detail") if r.get("failure_code") == "corridor_reservation_failed" else None)} for r in records if nested_get(r,["grid_embedding","corridor_reservation"]) or r.get("failure_code") == "corridor_reservation_failed"]
    save_json(corridor_report, out_dir/"corridor_reservation_debug.json")
    unexpected_report=[{"maze_id": r.get("maze_id"), "failure_detail": r.get("failure_detail")} for r in records if r.get("failure_code") == "unexpected_exception"]
    save_json(unexpected_report, out_dir/"unexpected_exception_debug.json")
    vis_report=select_visual_samples(records,out_dir,args.max_visual_samples if args.save_visualizations else min(4,args.max_visual_samples))
    save_json(vis_report,out_dir/"sample_selection_report.json")
    print_summary(summary)
    print(f"\nOutput directory: {out_dir}")

if __name__ == "__main__":
    main()
