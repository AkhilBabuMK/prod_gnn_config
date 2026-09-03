# -*- coding: utf-8 -*-
"""
Operational scenario/event extraction for the scenario-aware GNN.

This module is deliberately deterministic and auditable. It finds candidate
operational interactions from the raw snapshot, route, and topology before the
model sees the graph. The model still learns delay impact; this layer only says
"these trains are operationally related right now."
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


EVENT_FEAT_DIM = 8
CONFLICT_EDGE_DIM = 6


@dataclass(frozen=True)
class TrainEvent:
    next_junction: int = -1
    prev_branch: int = -1
    junction_conflicts: int = 0
    cross_branch_conflicts: int = 0
    higher_priority_conflicts: int = 0
    queue_ahead: int = 0
    min_eta_gap: float = 999.0
    max_priority_delta: float = 0.0


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if math.isnan(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def route_map(day: dict) -> Dict[str, List[int]]:
    routes: Dict[str, List[int]] = {}
    for inst in day.get("train_instances", []):
        route = inst.get("route") or []
        if route:
            routes[str(inst.get("instance_id"))] = [int(x) for x in route]
    return routes


def next_junction_for_node(
    node: dict,
    route: List[int],
    junction_ids: set[int],
) -> Tuple[int, int]:
    """Return (next_junction, station_before_junction_on_route), or (-1, -1)."""
    if not route:
        return -1, -1

    current = node.get("current_station")
    nxt = node.get("next_station")
    start_idx = 0
    if nxt in route:
        start_idx = route.index(nxt)
    elif current in route:
        start_idx = min(route.index(current) + 1, len(route) - 1)

    for i in range(start_idx, len(route)):
        sid = int(route[i])
        if sid in junction_ids:
            prev_sid = int(route[i - 1]) if i > 0 else -1
            return sid, prev_sid
    return -1, -1


def compute_events_and_conflict_edges(
    snapshot: dict,
    routes: Dict[str, List[int]],
    junction_ids: set[int],
    window_min: float = 60.0,
) -> Tuple[List[TrainEvent], torch.Tensor, torch.Tensor]:
    """
    Returns per-train events plus direct conflict edges.

    Edge direction is bidirectional so each train can attend to the other.
    Edge attrs:
      0 proximity       1/(eta_gap + 1)
      1 eta_gap_norm    clipped eta gap / window
      2 priority_delta  other_priority - self_priority, [-1, 1]
      3 other_first     1 if other train reaches junction earlier
      4 cross_branch    1 if branch before junction differs
      5 stored_overlap  target train's stored junction_overlap_count / 6
    """
    nodes = snapshot.get("train_nodes", [])
    events = [TrainEvent() for _ in nodes]
    groups: Dict[int, List[dict]] = defaultdict(list)

    for idx, node in enumerate(nodes):
        jct, prev_branch = next_junction_for_node(
            node,
            routes.get(str(node.get("instance_id")), []),
            junction_ids,
        )
        if jct < 0:
            continue
        eta = safe_float(node.get("eta_nearest_jct_min"), 999.0)
        item = {
            "idx": idx,
            "node": node,
            "jct": jct,
            "prev_branch": prev_branch,
            "eta": eta,
            "priority": safe_float(node.get("priority"), 0.5),
        }
        groups[jct].append(item)
        events[idx] = TrainEvent(
            next_junction=jct,
            prev_branch=prev_branch,
            queue_ahead=int(round(safe_float(node.get("queue_ahead_count"), 0.0))),
        )

    mutable = [event.__dict__.copy() for event in events]
    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_attr: List[List[float]] = []

    for _, trains in groups.items():
        if len(trains) < 2:
            continue
        for i, a in enumerate(trains):
            for b in trains[i + 1:]:
                gap = abs(a["eta"] - b["eta"])
                if gap > window_min:
                    continue
                cross_branch = (
                    a["prev_branch"] >= 0
                    and b["prev_branch"] >= 0
                    and a["prev_branch"] != b["prev_branch"]
                )
                for self_item, other_item in ((a, b), (b, a)):
                    si = self_item["idx"]
                    oi = other_item["idx"]
                    priority_delta = other_item["priority"] - self_item["priority"]
                    mutable[si]["junction_conflicts"] += 1
                    mutable[si]["cross_branch_conflicts"] += int(cross_branch)
                    mutable[si]["higher_priority_conflicts"] += int(priority_delta > 1e-6)
                    mutable[si]["min_eta_gap"] = min(mutable[si]["min_eta_gap"], gap)
                    mutable[si]["max_priority_delta"] = max(
                        mutable[si]["max_priority_delta"], priority_delta
                    )

                    stored_overlap = safe_float(
                        self_item["node"].get("junction_overlap_count"), 0.0
                    ) / 6.0
                    edge_src.append(si)
                    edge_dst.append(oi)
                    edge_attr.append([
                        1.0 / (gap + 1.0),
                        min(gap / max(window_min, 1e-6), 1.0),
                        max(-1.0, min(1.0, priority_delta)),
                        1.0 if other_item["eta"] < self_item["eta"] else 0.0,
                        1.0 if cross_branch else 0.0,
                        min(stored_overlap, 1.5),
                    ])

    final_events = [TrainEvent(**m) for m in mutable]

    if edge_src:
        edges = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        attrs = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edges = torch.zeros(2, 0, dtype=torch.long)
        attrs = torch.zeros(0, CONFLICT_EDGE_DIM, dtype=torch.float)
    return final_events, edges, attrs


def event_features(events: List[TrainEvent]) -> torch.Tensor:
    feats = []
    for e in events:
        min_gap = e.min_eta_gap if e.min_eta_gap < 999.0 else 60.0
        feats.append([
            min(e.junction_conflicts / 6.0, 1.5),
            min(e.cross_branch_conflicts / 6.0, 1.5),
            min(e.higher_priority_conflicts / 6.0, 1.5),
            min(e.queue_ahead / 6.0, 1.5),
            1.0 - min(min_gap / 60.0, 1.0),
            max(0.0, min(1.0, e.max_priority_delta)),
            1.0 if e.next_junction >= 0 else 0.0,
            1.0 if e.cross_branch_conflicts > 0 else 0.0,
        ])
    return torch.tensor(feats, dtype=torch.float)


def scenario_labels(events: List[TrainEvent]) -> torch.Tensor:
    labels = []
    for e in events:
        labels.append([
            float(e.junction_conflicts > 0),
            float(e.cross_branch_conflicts > 0),
            float(e.higher_priority_conflicts > 0),
            float(e.queue_ahead > 0),
            float(e.min_eta_gap <= 5.0),
        ])
    return torch.tensor(labels, dtype=torch.float)
