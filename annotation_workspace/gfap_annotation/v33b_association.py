"""V3.3-B conservative association on immutable V3.3-A adaptive geometry.

Ground truth and Uncertain masks are deliberately absent from every inference API.
This module changes only Process state/ownership labels, never candidate geometry.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from gfap_annotation.v33a_tracing import (
    AMBIGUOUS,
    ASSIGNED,
    REJECTED,
    UNASSIGNED,
    ThinTraceConfig,
    extract_graph_nodes,
    trace_soma_graph,
)

STATE_NAMES = {
    ASSIGNED: "ASSIGNED",
    AMBIGUOUS: "AMBIGUOUS",
    UNASSIGNED: "UNASSIGNED",
    REJECTED: "REJECTED",
}


@dataclass(frozen=True)
class AssociationConfig:
    ambiguity_absolute_gap: float = 8.0
    ambiguity_relative_gap: float = 0.15
    minimum_gfap: float = 0.07
    minimum_ridge: float = 0.035
    strong_gfap: float = 0.16
    strong_minimum_ridge: float = 0.015
    minimum_mean_contrast: float = 0.03


def extract_fixed_segments(centerline: np.ndarray, soma_candidates: np.ndarray) -> np.ndarray:
    """Split fixed centerlines at existing endpoint/branch/Soma-connection nodes.

    Node clusters become their own segments so every centerline pixel has exactly
    one deterministic segment ID and no node is silently discarded.
    """
    skeleton = np.asarray(centerline, dtype=bool)
    soma = np.asarray(soma_candidates)
    if skeleton.ndim != 2 or skeleton.shape != soma.shape:
        raise ValueError("Centerline and Soma candidates must share a 2D coordinate system")
    nodes = extract_graph_nodes(skeleton, soma) > 0
    segments = np.zeros(skeleton.shape, dtype=np.int32)
    next_id = 1
    for mask in (skeleton & ~nodes, skeleton & nodes):
        count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        for label in range(1, count):
            segments[labels == label] = next_id
            next_id += 1
    if not np.array_equal(segments > 0, skeleton):
        raise RuntimeError("Segment extraction did not preserve every fixed centerline pixel")
    return segments


def map_approved_pixels_to_segments(
    approved_mask: np.ndarray,
    centerline: np.ndarray,
    segments: np.ndarray,
) -> np.ndarray:
    """Index width pixels by their nearest fixed centerline segment, not a Soma."""
    approved = np.asarray(approved_mask, dtype=bool)
    skeleton = np.asarray(centerline, dtype=bool)
    if approved.shape != skeleton.shape or approved.shape != segments.shape:
        raise ValueError("Approved mask, centerline, and segments must share coordinates")
    if not skeleton.any() or np.any(skeleton & ~approved):
        raise ValueError("Every fixed centerline pixel must belong to the approved geometry")
    _, nearest = ndimage.distance_transform_edt(~skeleton, return_indices=True)
    nearest_segment = segments[tuple(nearest)]
    pixel_segment = np.where(approved, nearest_segment, 0).astype(np.int32)
    if np.any(approved & (pixel_segment <= 0)):
        raise RuntimeError("An approved pixel could not be indexed to a fixed segment")
    return pixel_segment


def decide_segment(
    mean_gfap: float,
    mean_ridge: float,
    mean_contrast: float,
    costs: dict[int, float],
    config: AssociationConfig,
) -> tuple[int, int, str, int | None, float | None, float | None, float | None, float | None]:
    """Return state, owner, reason, competitor, best/second costs, gap and margin."""
    ordinary = (
        mean_gfap >= config.minimum_gfap
        and mean_ridge >= config.minimum_ridge
        and mean_contrast >= config.minimum_mean_contrast
    )
    strong = mean_gfap >= config.strong_gfap and mean_ridge >= config.strong_minimum_ridge
    if not (ordinary or strong):
        return REJECTED, 0, "insufficient_gfap_or_local_structural_evidence", None, None, None, None, None
    if not costs:
        return UNASSIGNED, 0, "plausible_structure_without_complete_soma_path", None, None, None, None, None
    ordered = sorted((float(cost), int(instance_id)) for instance_id, cost in costs.items())
    best_cost, best_id = ordered[0]
    if len(ordered) == 1:
        return ASSIGNED, best_id, "unique_complete_soma_path", None, best_cost, None, None, None
    second_cost, second_id = ordered[1]
    gap = second_cost - best_cost
    margin = max(config.ambiguity_absolute_gap, config.ambiguity_relative_gap * best_cost)
    if gap <= margin:
        return AMBIGUOUS, 0, "competing_soma_paths_within_conservative_margin", second_id, best_cost, second_cost, gap, margin
    return ASSIGNED, best_id, "unique_best_path_exceeds_conservative_margin", second_id, best_cost, second_cost, gap, margin


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def associate_approved_process(
    gfap: np.ndarray,
    process_probability: np.ndarray,
    ridge: np.ndarray,
    local_contrast: np.ndarray,
    approved_mask: np.ndarray,
    centerline: np.ndarray,
    soma_candidates: np.ndarray,
    candidate_records: list[dict],
    roi_size: int,
    trace_config: ThinTraceConfig,
    association_config: AssociationConfig = AssociationConfig(),
) -> dict:
    """Recompute path evidence only; preserve all approved pixels and fixed IDs."""
    arrays = [gfap, process_probability, ridge, local_contrast, approved_mask, centerline, soma_candidates]
    if any(array.shape != gfap.shape for array in arrays) or gfap.ndim != 2:
        raise ValueError("All association inputs must share one 2D coordinate system")
    if not np.isfinite(gfap).all() or not np.isfinite(process_probability).all() or not np.isfinite(ridge).all():
        raise ValueError("GFAP, V3.1 Process probability, and ridge evidence must be finite")
    if not np.isfinite(local_contrast).all():
        raise ValueError("Local contrast must be finite")
    if np.any(centerline & ~approved_mask) or np.any(approved_mask & (soma_candidates > 0)):
        raise ValueError("Approved Process geometry is inconsistent with centerlines or Soma masks")

    segments = extract_fixed_segments(centerline, soma_candidates)
    pixel_segment = map_approved_pixels_to_segments(approved_mask, centerline, segments)
    segment_ids = [int(value) for value in np.unique(segments) if value > 0]
    complete_costs: dict[int, dict[int, float]] = {segment_id: {} for segment_id in segment_ids}
    partial_paths: dict[int, list[int]] = {segment_id: [] for segment_id in segment_ids}
    centerline_reach = np.zeros(gfap.shape, dtype=np.uint16)
    trace_records = []

    for record in candidate_records:
        if not record["selected"]:
            continue
        instance_id = int(record["cellpose_instance_id"])
        traced = trace_soma_graph(
            gfap,
            process_probability,
            ridge,
            centerline,
            soma_candidates,
            instance_id,
            float(record["center_x"]),
            float(record["center_y"]),
            roi_size,
            trace_config,
        )
        x0, y0, x1, y1 = traced["bounds"]
        local_segments = segments[y0:y1, x0:x1]
        local_cost = traced["cost"]
        local_traced = traced["traced"] & (local_segments > 0)
        centerline_reach[y0:y1, x0:x1] += local_traced.astype(np.uint16)
        trace_records.append({
            "cellpose_instance_id": instance_id,
            "roi_size": roi_size,
            "roi_x0": x0,
            "roi_y0": y0,
            "roi_x1": x1,
            "roi_y1": y1,
            "seed_pixels": traced["seed_pixels"],
            "centerline_pixels": traced["centerline_pixels"],
            "stop_reason": traced["stop_reason"],
        })
        for segment_id in [int(value) for value in np.unique(local_segments[local_segments > 0])]:
            in_segment = local_segments == segment_id
            finite = in_segment & np.isfinite(local_cost)
            finite_count = int(finite.sum())
            total = int(in_segment.sum())
            if finite_count == total:
                complete_costs[segment_id][instance_id] = float(np.median(local_cost[finite]))
            elif finite_count:
                partial_paths[segment_id].append(instance_id)

    state = np.zeros(gfap.shape, dtype=np.uint8)
    instance = soma_candidates.astype(np.int32).copy()
    reachable_map = np.zeros(gfap.shape, dtype=np.uint16)
    best_cost_map = np.full(gfap.shape, np.nan, dtype=np.float32)
    second_cost_map = np.full(gfap.shape, np.nan, dtype=np.float32)
    segment_records = []

    for segment_id in segment_ids:
        center_mask = segments == segment_id
        process_mask = pixel_segment == segment_id
        if not process_mask.any():
            raise RuntimeError(f"Fixed segment {segment_id} has no approved pixels")
        mean_gfap = _mean(gfap[center_mask])
        mean_ridge = _mean(ridge[center_mask])
        mean_contrast = _mean(local_contrast[center_mask])
        mean_process = _mean(process_probability[center_mask])
        costs = complete_costs[segment_id]
        decision = decide_segment(mean_gfap, mean_ridge, mean_contrast, costs, association_config)
        segment_state, owner, reason, second_id, best_cost, second_cost, gap, margin = decision
        state[process_mask] = segment_state
        if segment_state == ASSIGNED:
            if owner <= 0:
                raise RuntimeError("ASSIGNED segment has no Soma owner")
            instance[process_mask] = owner
        reachable_map[process_mask] = len(costs)
        if best_cost is not None:
            best_cost_map[process_mask] = best_cost
        if second_cost is not None:
            second_cost_map[process_mask] = second_cost
        segment_records.append({
            "segment_id": segment_id,
            "centerline_pixels": int(center_mask.sum()),
            "approved_process_pixels": int(process_mask.sum()),
            "mean_gfap": mean_gfap,
            "mean_local_contrast": mean_contrast,
            "mean_ridge": mean_ridge,
            "mean_v31_process_probability": mean_process,
            "reachable_soma_ids": ";".join(str(value) for value in sorted(costs)),
            "partial_path_soma_ids": ";".join(str(value) for value in sorted(partial_paths[segment_id])),
            "reachable_soma_costs": ";".join(f"{key}:{costs[key]:.8g}" for key in sorted(costs)),
            "reachable_soma_count": len(costs),
            "best_soma_id": owner if segment_state == ASSIGNED else (min(costs, key=costs.get) if costs else None),
            "second_soma_id": second_id,
            "best_path_cost": best_cost,
            "second_path_cost": second_cost,
            "path_cost_gap": gap,
            "ambiguity_margin": margin,
            "state": segment_state,
            "state_name": STATE_NAMES[segment_state],
            "assigned_soma_instance_id": owner,
            "decision_reason": reason,
        })

    union = state > 0
    if not np.array_equal(union, approved_mask):
        raise RuntimeError("Output Process-state union differs from immutable approved geometry")
    if np.any((state != ASSIGNED) & approved_mask & (instance > 0)):
        raise RuntimeError("A non-ASSIGNED Process pixel received a Soma instance ID")
    if not np.array_equal(instance[soma_candidates > 0], soma_candidates[soma_candidates > 0]):
        raise RuntimeError("Soma candidate shapes or IDs changed")
    if not np.array_equal(segments > 0, centerline):
        raise RuntimeError("Fixed centerlines changed")

    stats = {
        "selected_soma_candidates": sum(bool(record["selected"]) for record in candidate_records),
        "segments": len(segment_ids),
        "approved_process_pixels": int(approved_mask.sum()),
        "process_state_union_pixels": int(union.sum()),
        "multi_soma_competing_segments": sum(record["reachable_soma_count"] > 1 for record in segment_records),
        "multi_soma_competing_pixels": int((reachable_map > 1).sum()),
        "ambiguous_segments": sum(record["state"] == AMBIGUOUS for record in segment_records),
        "final_cross_instance_conflicts": 0,
    }
    for label, name in STATE_NAMES.items():
        stats[f"{name.lower()}_pixels"] = int((state == label).sum())
        stats[f"{name.lower()}_segments"] = sum(record["state"] == label for record in segment_records)
    return {
        "state": state,
        "instance": instance,
        "segments": segments,
        "pixel_segment": pixel_segment,
        "reachable_map": reachable_map,
        "best_cost": best_cost_map,
        "second_cost": second_cost_map,
        "centerline_reach": centerline_reach,
        "segment_records": segment_records,
        "trace_records": trace_records,
        "stats": stats,
    }
