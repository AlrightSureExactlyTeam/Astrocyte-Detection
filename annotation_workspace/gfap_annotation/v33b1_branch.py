"""V3.3-B.1 branch-level association using fixed V3.3-B path evidence.

Only state/ownership labels change. Approved geometry, centerlines, Soma IDs,
GFAP preprocessing, and all model outputs remain immutable.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np
from scipy import ndimage

from gfap_annotation.v33a_tracing import (
    AMBIGUOUS,
    ASSIGNED,
    REJECTED,
    UNASSIGNED,
    extract_graph_nodes,
)
from gfap_annotation.v33b_association import AssociationConfig, STATE_NAMES, decide_segment


def build_structural_units(
    centerline: np.ndarray,
    soma_candidates: np.ndarray,
    fixed_segments: np.ndarray | None = None,
    segment_records: list[dict] | None = None,
) -> dict:
    """Merge continuous branches, split at crossings and reachable-Soma changes."""
    skeleton = np.asarray(centerline, dtype=bool)
    if skeleton.ndim != 2 or skeleton.shape != soma_candidates.shape:
        raise ValueError("Centerline and Soma candidates must share coordinates")
    graph_nodes = extract_graph_nodes(skeleton, soma_candidates)
    crossing = skeleton & ((graph_nodes & 2) > 0)
    signature_map = np.zeros(skeleton.shape, dtype=np.int32)
    if fixed_segments is not None or segment_records is not None:
        if fixed_segments is None or segment_records is None:
            raise ValueError("Fixed segments and their evidence records must be provided together")
        signatures = sorted({tuple(sorted(_parse_costs(record))) for record in segment_records})
        signature_codes = {signature: index + 1 for index, signature in enumerate(signatures)}
        for record in segment_records:
            code = signature_codes[tuple(sorted(_parse_costs(record)))]
            signature_map[fixed_segments == int(record["segment_id"])] = code
    unit_map = np.zeros(skeleton.shape, dtype=np.int32)
    unit_types: dict[int, str] = {}
    next_id = 1
    branch = skeleton & ~crossing
    count, base_labels = cv2.connectedComponents(branch.astype(np.uint8), connectivity=8)
    topology_only = fixed_segments is None and segment_records is None
    for label in range(1, count):
        base = base_labels == label
        if topology_only:
            unit_map[base] = next_id
            unit_types[next_id] = "branch"
            next_id += 1
            continue
        signatures = [int(value) for value in np.unique(signature_map[base])]
        for signature in signatures:
            mask = base if signature == 0 else base & (signature_map == signature)
            subcount, sublabels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
            for sublabel in range(1, subcount):
                unit_map[sublabels == sublabel] = next_id
                unit_types[next_id] = "branch"
                next_id += 1
    count, labels = cv2.connectedComponents(crossing.astype(np.uint8), connectivity=8)
    for label in range(1, count):
        unit_map[labels == label] = next_id
        unit_types[next_id] = "crossing"
        next_id += 1
    if not np.array_equal(unit_map > 0, skeleton):
        raise RuntimeError("Structural units do not preserve fixed centerlines")
    adjacency: dict[int, set[int]] = {unit_id: set() for unit_id in unit_types}
    for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        y_dst = slice(max(0, dy), min(skeleton.shape[0], skeleton.shape[0] + dy))
        x_dst = slice(max(0, dx), min(skeleton.shape[1], skeleton.shape[1] + dx))
        y_src = slice(max(0, -dy), min(skeleton.shape[0], skeleton.shape[0] - dy))
        x_src = slice(max(0, -dx), min(skeleton.shape[1], skeleton.shape[1] - dx))
        first = unit_map[y_dst, x_dst]
        second = unit_map[y_src, x_src]
        valid = (first > 0) & (second > 0) & (first != second)
        for a, b in np.unique(np.stack([first[valid], second[valid]], axis=1), axis=0):
            adjacency[int(a)].add(int(b))
            adjacency[int(b)].add(int(a))
    return {"unit_map": unit_map, "unit_types": unit_types, "adjacency": adjacency, "graph_nodes": graph_nodes}


def _parse_costs(record: dict) -> dict[int, float]:
    text = record.get("reachable_soma_costs", "")
    costs = {}
    for item in text.split(";"):
        if not item:
            continue
        instance_id, cost = item.split(":", 1)
        costs[int(instance_id)] = float(cost)
    return costs


def _weighted_mean(records: list[dict], key: str, local_weights: list[int] | None = None) -> float:
    weights = np.asarray(
        local_weights if local_weights is not None else [int(record["centerline_pixels"]) for record in records],
        dtype=np.float64,
    )
    values = np.asarray([float(record[key]) for record in records], dtype=np.float64)
    return float(np.sum(values * weights) / max(weights.sum(), 1.0))


def _unit_tangent(unit_map: np.ndarray, unit_id: int) -> tuple[np.ndarray | None, float]:
    ys, xs = np.nonzero(unit_map == unit_id)
    if len(ys) < 3:
        return None, 0.0
    coordinates = np.stack([xs - xs.mean(), ys - ys.mean()], axis=1)
    covariance = coordinates.T @ coordinates / max(len(ys), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    anisotropy = float((eigenvalues[1] - eigenvalues[0]) / max(eigenvalues.sum(), 1e-8))
    if anisotropy < 0.20:
        return None, anisotropy
    return eigenvectors[:, 1], anisotropy


def _directional_continuity(unit_map: np.ndarray, first: int, second: int) -> float | None:
    tangent_a, _ = _unit_tangent(unit_map, first)
    tangent_b, _ = _unit_tangent(unit_map, second)
    if tangent_a is None or tangent_b is None:
        return None
    return float(abs(np.dot(tangent_a, tangent_b)))


def _cost_margin(cost: float, config: AssociationConfig) -> float:
    return max(config.ambiguity_absolute_gap, config.ambiguity_relative_gap * cost)


def associate_branches(
    centerline: np.ndarray,
    soma_candidates: np.ndarray,
    approved_mask: np.ndarray,
    fixed_segments: np.ndarray,
    approved_pixel_segments: np.ndarray,
    segment_records: list[dict],
    association_config: AssociationConfig = AssociationConfig(),
    minimum_propagation_direction: float = 0.25,
) -> dict:
    """Aggregate B evidence over meaningful branches with limited crossing propagation."""
    if not np.array_equal(fixed_segments > 0, centerline):
        raise ValueError("Fixed B segments do not match the fixed centerline")
    if not np.array_equal(approved_pixel_segments > 0, approved_mask):
        raise ValueError("B pixel-to-segment map does not match approved geometry")
    structural = build_structural_units(centerline, soma_candidates, fixed_segments, segment_records)
    unit_map = structural["unit_map"]
    unit_types = structural["unit_types"]
    adjacency = structural["adjacency"]
    records_by_segment = {int(record["segment_id"]): record for record in segment_records}
    pairs, pair_counts = np.unique(
        np.stack([fixed_segments[centerline], unit_map[centerline]], axis=1),
        axis=0,
        return_counts=True,
    )
    unit_sources: dict[int, list[int]] = {unit_id: [] for unit_id in unit_types}
    pair_weight: dict[tuple[int, int], int] = {}
    for (segment_id, unit_id), count in zip(pairs, pair_counts):
        segment_id, unit_id = int(segment_id), int(unit_id)
        unit_sources[unit_id].append(segment_id)
        pair_weight[(segment_id, unit_id)] = int(count)

    unit_records: dict[int, dict] = {}
    for unit_id, kind in unit_types.items():
        source_ids = sorted(unit_sources[unit_id])
        source_records = [records_by_segment[segment_id] for segment_id in source_ids]
        if not source_records:
            raise RuntimeError(f"Structural unit {unit_id} has no B evidence")
        costs_by_segment = [_parse_costs(record) for record in source_records]
        common_ids = set(costs_by_segment[0])
        for costs in costs_by_segment[1:]:
            common_ids &= set(costs)
        weights = {segment_id: pair_weight[(segment_id, unit_id)] for segment_id in source_ids}
        total_weight = sum(weights.values())
        local_weights = [weights[segment_id] for segment_id in source_ids]
        costs = {
            instance_id: sum(costs_by_segment[index][instance_id] * weights[segment_id] for index, segment_id in enumerate(source_ids)) / total_weight
            for instance_id in sorted(common_ids)
        }
        mean_gfap = _weighted_mean(source_records, "mean_gfap", local_weights)
        mean_ridge = _weighted_mean(source_records, "mean_ridge", local_weights)
        mean_contrast = _weighted_mean(source_records, "mean_local_contrast", local_weights)
        decision = decide_segment(mean_gfap, mean_ridge, mean_contrast, costs, association_config)
        state, owner, reason, second_id, best_cost, second_cost, gap, margin = decision
        best_id = min(costs, key=costs.get) if costs else None
        competitors = []
        if costs:
            best = costs[best_id]
            threshold = best + _cost_margin(best, association_config)
            competitors = [instance_id for instance_id in sorted(costs) if costs[instance_id] <= threshold]
        unit_records[unit_id] = {
            "unit_id": unit_id,
            "unit_type": kind,
            "source_segment_ids": ";".join(str(value) for value in source_ids),
            "source_segment_count": len(source_ids),
            "centerline_pixels": int((unit_map == unit_id).sum()),
            "mean_gfap": mean_gfap,
            "mean_local_contrast": mean_contrast,
            "mean_ridge": mean_ridge,
            "mean_v31_process_probability": _weighted_mean(source_records, "mean_v31_process_probability", local_weights),
            "reachable_soma_ids": ";".join(str(value) for value in sorted(costs)),
            "reachable_soma_costs": ";".join(f"{key}:{costs[key]:.8g}" for key in sorted(costs)),
            "reachable_soma_count": len(costs),
            "best_soma_id": best_id,
            "second_soma_id": second_id,
            "best_path_cost": best_cost,
            "second_path_cost": second_cost,
            "path_cost_gap": gap,
            "ambiguity_margin": margin,
            "competing_soma_ids": ";".join(str(value) for value in competitors),
            "initial_state": state,
            "state": state,
            "state_name": STATE_NAMES[state],
            "assigned_soma_instance_id": owner,
            "decision_reason": reason,
            "propagated_from_unit_id": None,
            "propagation_direction_continuity": None,
        }

    # Propagate only downstream from ambiguous crossings, only along shared
    # competing Soma paths, and never through a reliable cost/direction separation.
    propagation_queue = deque(
        unit_id for unit_id, record in unit_records.items()
        if record["unit_type"] == "crossing" and record["state"] == AMBIGUOUS
    )
    visited: set[tuple[int, int]] = set()
    while propagation_queue:
        current = propagation_queue.popleft()
        current_record = unit_records[current]
        current_costs = _parse_costs(current_record)
        current_competitors = {int(value) for value in current_record["competing_soma_ids"].split(";") if value}
        for neighbor in sorted(adjacency[current]):
            if (current, neighbor) in visited:
                continue
            visited.add((current, neighbor))
            neighbor_record = unit_records[neighbor]
            if neighbor_record["state"] == REJECTED:
                continue
            neighbor_costs = _parse_costs(neighbor_record)
            shared = sorted(current_competitors & set(current_costs) & set(neighbor_costs))
            if len(shared) < 2:
                continue
            downstream_votes = sum(neighbor_costs[instance_id] > current_costs[instance_id] for instance_id in shared)
            if downstream_votes < 2:
                continue
            ordered = sorted(neighbor_costs[instance_id] for instance_id in shared)
            gap = ordered[1] - ordered[0]
            if gap > _cost_margin(ordered[0], association_config):
                continue  # Reliable local separation; do not spread uncertainty.
            continuity = _directional_continuity(unit_map, current, neighbor)
            if continuity is not None and continuity < minimum_propagation_direction:
                continue
            if neighbor_record["state"] != AMBIGUOUS:
                neighbor_record["state"] = AMBIGUOUS
                neighbor_record["state_name"] = STATE_NAMES[AMBIGUOUS]
                neighbor_record["assigned_soma_instance_id"] = 0
                neighbor_record["decision_reason"] = "downstream_of_ambiguous_crossing_without_reliable_separation"
                neighbor_record["propagated_from_unit_id"] = current
                neighbor_record["propagation_direction_continuity"] = continuity
            propagation_queue.append(neighbor)

    _, nearest = ndimage.distance_transform_edt(~centerline, return_indices=True)
    nearest_unit = unit_map[tuple(nearest)]
    pixel_unit = np.where(approved_mask, nearest_unit, 0).astype(np.int32)
    max_unit = max(unit_records, default=0)
    state_lookup = np.zeros(max_unit + 1, dtype=np.uint8)
    owner_lookup = np.zeros(max_unit + 1, dtype=np.int32)
    reach_lookup = np.zeros(max_unit + 1, dtype=np.uint16)
    for unit_id, record in unit_records.items():
        state_lookup[unit_id] = record["state"]
        owner_lookup[unit_id] = record["assigned_soma_instance_id"]
        reach_lookup[unit_id] = record["reachable_soma_count"]
    state = state_lookup[pixel_unit]
    instance = soma_candidates.astype(np.int32).copy()
    assigned = state == ASSIGNED
    instance[assigned] = owner_lookup[pixel_unit[assigned]]
    reachable_map = reach_lookup[pixel_unit]

    # Detect equal-distance attribution to different fixed centerline units.
    # This is a local width conflict, not a pixel-level association decision.
    distance = ndimage.distance_transform_edt(~centerline)
    minimum_squared_distance = np.rint(distance * distance).astype(np.int16)
    nearest_owner = owner_lookup[nearest_unit]
    nearest_state = state_lookup[nearest_unit]
    width_conflict = np.zeros(approved_mask.shape, dtype=bool)
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            squared_distance = dy * dy + dx * dx
            if squared_distance == 0 or squared_distance > 16:
                continue
            y_dst = slice(max(0, dy), min(centerline.shape[0], centerline.shape[0] + dy))
            x_dst = slice(max(0, dx), min(centerline.shape[1], centerline.shape[1] + dx))
            y_src = slice(max(0, -dy), min(centerline.shape[0], centerline.shape[0] - dy))
            x_src = slice(max(0, -dx), min(centerline.shape[1], centerline.shape[1] - dx))
            neighbor_unit = np.zeros_like(unit_map)
            neighbor_unit[y_dst, x_dst] = unit_map[y_src, x_src]
            tied = (
                approved_mask
                & (minimum_squared_distance == squared_distance)
                & (neighbor_unit > 0)
                & (neighbor_unit != nearest_unit)
            )
            if not tied.any():
                continue
            neighbor_owner = owner_lookup[neighbor_unit]
            neighbor_state = state_lookup[neighbor_unit]
            same_assigned_owner = (
                (nearest_state == ASSIGNED) & (neighbor_state == ASSIGNED)
                & (nearest_owner == neighbor_owner) & (nearest_owner > 0)
            )
            same_nonassigned_state = (nearest_state != ASSIGNED) & (nearest_state == neighbor_state)
            width_conflict |= tied & ~same_assigned_owner & ~same_nonassigned_state
    state[width_conflict] = AMBIGUOUS
    instance[width_conflict] = 0

    if not np.array_equal(state > 0, approved_mask):
        raise RuntimeError("B.1 Process-state union differs from approved geometry")
    if np.any(approved_mask & (state != ASSIGNED) & (instance > 0)):
        raise RuntimeError("A non-ASSIGNED B.1 Process pixel received an owner")
    for unit_id, record in unit_records.items():
        own = pixel_unit == unit_id
        record["approved_process_pixels_before_width_conflicts"] = int(own.sum())
        record["width_conflict_ambiguous_pixels"] = int((own & width_conflict).sum())
        record["final_process_pixels"] = int((own & (state == record["state"])).sum())
    ambiguous_units = [record for record in unit_records.values() if record["state"] == AMBIGUOUS]
    ambiguous_lengths = [record["centerline_pixels"] for record in ambiguous_units]
    isolated = [
        record["unit_id"] for record in ambiguous_units
        if not any(unit_records[neighbor]["state"] == AMBIGUOUS for neighbor in adjacency[record["unit_id"]])
    ]
    stats = {
        "structural_units": len(unit_records),
        "branch_units": sum(kind == "branch" for kind in unit_types.values()),
        "crossing_units": sum(kind == "crossing" for kind in unit_types.values()),
        "approved_process_pixels": int(approved_mask.sum()),
        "process_state_union_pixels": int((state > 0).sum()),
        "ambiguous_structural_units": len(ambiguous_units),
        "ambiguous_centerline_length_median": float(np.median(ambiguous_lengths)) if ambiguous_lengths else None,
        "ambiguous_centerline_length_p90": float(np.percentile(ambiguous_lengths, 90)) if ambiguous_lengths else None,
        "ambiguous_centerline_length_max": max(ambiguous_lengths, default=0),
        "isolated_ambiguous_structural_units": len(isolated),
        "isolated_single_pixel_ambiguous_units": sum(unit_records[unit_id]["centerline_pixels"] == 1 for unit_id in isolated),
        "propagated_ambiguous_units": sum(record["propagated_from_unit_id"] is not None for record in unit_records.values()),
        "width_conflict_ambiguous_pixels": int(width_conflict.sum()),
        "final_cross_instance_conflicts": 0,
    }
    for label, name in STATE_NAMES.items():
        stats[f"{name.lower()}_pixels"] = int((state == label).sum())
        stats[f"{name.lower()}_units"] = sum(record["state"] == label for record in unit_records.values())
    ordered_records = [unit_records[unit_id] for unit_id in sorted(unit_records)]
    return {
        "state": state,
        "instance": instance,
        "unit_map": unit_map,
        "pixel_unit": pixel_unit,
        "reachable_map": reachable_map,
        "width_conflict": width_conflict,
        "unit_records": ordered_records,
        "adjacency": adjacency,
        "graph_nodes": structural["graph_nodes"],
        "stats": stats,
    }
