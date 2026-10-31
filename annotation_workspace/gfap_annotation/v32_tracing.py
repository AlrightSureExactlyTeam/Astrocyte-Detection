"""Experimental V3.2 Soma-guided Process tracing with explicit abstention.

This is an inference-only, candidate-centered heuristic. It does not perform
whole-image detection, calibrated classification, training, or GT-guided tracing.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ASSIGNED = 1
AMBIGUOUS = 2
UNASSIGNED = 3
VARIANTS = ("D1", "D2", "D3", "E1")
STATE_NAMES = {0: "NOT_A_PROCESS_CANDIDATE", ASSIGNED: "ASSIGNED", AMBIGUOUS: "AMBIGUOUS", UNASSIGNED: "UNASSIGNED"}


@dataclass(frozen=True)
class TraceConfig:
    minimum_soma_area: int = 20
    maximum_soma_area: int = 12000
    minimum_candidate_gfap_mean: float = 0.05
    minimum_candidate_soma_response: float = 0.30
    maximum_candidates_per_tile: int = 24
    gfap_trace_threshold: float = 0.08
    gfap_process_bridge_threshold: float = 0.04
    process_trace_threshold: float = 0.65
    gfap_candidate_threshold: float = 0.18
    gfap_candidate_bridge_threshold: float = 0.07
    process_candidate_threshold: float = 0.70
    gfap_cost_weight: float = 5.0
    process_cost_weight: float = 3.0
    direction_cost_weight: float = 4.0
    maximum_path_cost: float = 600.0
    maximum_traced_pixels_per_soma: int = 25000
    ambiguity_relative_gap: float = 0.10
    ambiguity_absolute_gap: float = 4.0


def uses_process_evidence(variant: str) -> bool:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown V3.2 variant: {variant}")
    return variant != "D1"


def uses_directional_continuity(variant: str) -> bool:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown V3.2 variant: {variant}")
    return variant in {"D3", "E1"}


def roi_bounds(center_x: float, center_y: float, size: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Return clipped YX ROI bounds without changing full-image coordinates."""
    height, width = shape
    if size <= 0 or not (0 <= center_x < width and 0 <= center_y < height):
        raise ValueError("ROI size or center is invalid")
    x0 = max(0, int(round(center_x - size / 2)))
    y0 = max(0, int(round(center_y - size / 2)))
    x1 = min(width, x0 + size)
    y1 = min(height, y0 + size)
    x0 = max(0, x1 - size)
    y0 = max(0, y1 - size)
    return x0, y0, x1, y1


def select_soma_candidates(
    cellpose_mask: np.ndarray,
    gfap: np.ndarray,
    soma_response: np.ndarray,
    config: TraceConfig,
) -> tuple[np.ndarray, list[dict]]:
    """Select credible candidate ROIs while preserving original Cellpose IDs."""
    if cellpose_mask.shape != gfap.shape or gfap.shape != soma_response.shape:
        raise ValueError("Candidate inputs must share original image coordinates")
    if not np.issubdtype(cellpose_mask.dtype, np.integer) or cellpose_mask.min() < 0:
        raise ValueError("Cellpose mask must contain nonnegative integer IDs")
    selected = np.zeros(cellpose_mask.shape, dtype=np.int32)
    records = []
    for instance_id in [int(v) for v in np.unique(cellpose_mask) if v > 0]:
        mask = cellpose_mask == instance_id
        area = int(mask.sum())
        ys, xs = np.nonzero(mask)
        record = {
            "cellpose_instance_id": instance_id,
            "area_pixels": area,
            "center_x": float(xs.mean()),
            "center_y": float(ys.mean()),
            "bbox_x0": int(xs.min()), "bbox_y0": int(ys.min()),
            "bbox_x1": int(xs.max()) + 1, "bbox_y1": int(ys.max()) + 1,
            "mean_gfap": float(gfap[mask].mean()),
            "mean_v3_soma_response": float(soma_response[mask].mean()),
            "selected": True,
            "exclusion_reason": "",
        }
        if area < config.minimum_soma_area:
            record.update(selected=False, exclusion_reason="area_below_minimum")
        elif area > config.maximum_soma_area:
            record.update(selected=False, exclusion_reason="area_above_maximum")
        elif record["mean_gfap"] < config.minimum_candidate_gfap_mean:
            record.update(selected=False, exclusion_reason="insufficient_local_gfap")
        elif record["mean_v3_soma_response"] < config.minimum_candidate_soma_response:
            record.update(selected=False, exclusion_reason="weak_supplementary_soma_response")
        record["credibility_ranking_score"] = record["mean_v3_soma_response"] * (0.5 + record["mean_gfap"])
        records.append(record)
    eligible = sorted((r for r in records if r["selected"]), key=lambda r: (-r["credibility_ranking_score"], r["cellpose_instance_id"]))
    for record in eligible[config.maximum_candidates_per_tile:]:
        record.update(selected=False, exclusion_reason="beyond_pilot_candidate_cap")
    for record in records:
        if record["selected"]:
            selected[cellpose_mask == record["cellpose_instance_id"]] = record["cellpose_instance_id"]
    records.sort(key=lambda r: r["cellpose_instance_id"])
    return selected, records


def evidence_masks(gfap: np.ndarray, process: np.ndarray, variant: str, config: TraceConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return traceable support and broader Process-candidate support."""
    trace = gfap >= config.gfap_trace_threshold
    candidate = gfap >= config.gfap_candidate_threshold
    if uses_process_evidence(variant):
        trace |= (process >= config.process_trace_threshold) & (gfap >= config.gfap_process_bridge_threshold)
        candidate |= (process >= config.process_candidate_threshold) & (gfap >= config.gfap_candidate_bridge_threshold)
    return trace, candidate


def _pixel_cost(gfap: float, process: float, step_length: float, variant: str, config: TraceConfig) -> float:
    evidence_cost = 1.0 + config.gfap_cost_weight * (1.0 - gfap)
    if uses_process_evidence(variant):
        evidence_cost += config.process_cost_weight * (1.0 - process)
    return step_length * evidence_cost


def trace_soma_roi(
    gfap: np.ndarray,
    process: np.ndarray,
    soma_candidates: np.ndarray,
    instance_id: int,
    center_x: float,
    center_y: float,
    roi_size: int,
    variant: str,
    config: TraceConfig,
) -> dict:
    """Bounded 8-connected least-cost expansion from the Soma boundary.

    D3/E1 add a deterministic best-predecessor direction-change penalty. This
    single-state approximation is intentional for the small technical pilot.
    """
    if instance_id <= 0 or not np.any(soma_candidates == instance_id):
        raise ValueError("Tracing requires a visible positive Soma candidate ID")
    x0, y0, x1, y1 = roi_bounds(center_x, center_y, roi_size, gfap.shape)
    local_soma = soma_candidates[y0:y1, x0:x1]
    own = local_soma == instance_id
    if not own.any():
        raise ValueError("Soma candidate is absent from its centered ROI")
    local_gfap = gfap[y0:y1, x0:x1]
    local_process = process[y0:y1, x0:x1]
    allowed, _ = evidence_masks(local_gfap, local_process, variant, config)
    blocked = local_soma > 0
    allowed &= ~blocked
    boundary = own & ~cv2.erode(own.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if not boundary.any():
        raise RuntimeError("Could not construct Soma boundary")
    height, width = own.shape
    distance = np.full((height, width), np.inf, dtype=np.float64)
    previous_direction = np.full((height, width), -1, dtype=np.int8)
    queue = []
    for y, x in zip(*np.nonzero(boundary)):
        distance[y, x] = 0.0
        heapq.heappush(queue, (0.0, int(y), int(x)))
    directions = [
        (-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1),
    ]
    units = [(dy / math.hypot(dy, dx), dx / math.hypot(dy, dx)) for dy, dx in directions]
    finalized = np.zeros((height, width), dtype=bool)
    reached = np.zeros((height, width), dtype=bool)
    reached_count = 0
    stop_reason = "path_cost_or_frontier_exhausted"
    while queue:
        cost, y, x = heapq.heappop(queue)
        if cost != distance[y, x] or finalized[y, x]:
            continue
        if cost > config.maximum_path_cost:
            stop_reason = "maximum_path_cost"
            break
        finalized[y, x] = True
        if allowed[y, x] and not reached[y, x]:
            reached_count += 1
            if reached_count > config.maximum_traced_pixels_per_soma:
                stop_reason = "maximum_traced_pixels"
                break
            reached[y, x] = True
        prior = int(previous_direction[y, x])
        for direction_index, ((dy, dx), (uy, ux)) in enumerate(zip(directions, units)):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < height and 0 <= nx < width) or finalized[ny, nx] or not allowed[ny, nx]:
                continue
            step = _pixel_cost(
                float(local_gfap[ny, nx]), float(local_process[ny, nx]),
                math.hypot(dy, dx), variant, config,
            )
            if uses_directional_continuity(variant) and prior >= 0:
                py, px = units[prior]
                alignment = float(np.clip(py * uy + px * ux, -1, 1))
                step += config.direction_cost_weight * (1.0 - alignment) / 2.0
            updated = cost + step
            if updated < distance[ny, nx] and updated <= config.maximum_path_cost:
                distance[ny, nx] = updated
                previous_direction[ny, nx] = direction_index
                heapq.heappush(queue, (updated, ny, nx))
    traced = np.isfinite(distance) & ~blocked & (distance <= config.maximum_path_cost)
    distance[~traced] = np.inf
    return {
        "bounds": (x0, y0, x1, y1),
        "cost": distance.astype(np.float32),
        "traced": traced,
        "stop_reason": stop_reason,
        "traced_pixels": int(traced.sum()),
        "boundary_seed_pixels": int(boundary.sum()),
    }


def trace_tile(
    gfap: np.ndarray,
    process: np.ndarray,
    soma_candidates: np.ndarray,
    candidate_records: list[dict],
    roi_size: int,
    variant: str,
    config: TraceConfig,
) -> dict:
    """Trace all selected candidates and associate without forcing residual Process."""
    shape = gfap.shape
    best_cost = np.full(shape, np.inf, dtype=np.float32)
    second_cost = np.full(shape, np.inf, dtype=np.float32)
    best_owner = np.zeros(shape, dtype=np.int32)
    second_owner = np.zeros(shape, dtype=np.int32)
    roi_union = np.zeros(shape, dtype=bool)
    candidate_union = np.zeros(shape, dtype=bool)
    trace_records = []
    selected = [r for r in candidate_records if r["selected"]]
    for record in selected:
        instance_id = record["cellpose_instance_id"]
        result = trace_soma_roi(
            gfap, process, soma_candidates, instance_id,
            record["center_x"], record["center_y"], roi_size, variant, config,
        )
        x0, y0, x1, y1 = result["bounds"]
        roi_union[y0:y1, x0:x1] = True
        local = result["traced"]
        cost = result["cost"]
        local_best = best_cost[y0:y1, x0:x1]
        local_second = second_cost[y0:y1, x0:x1]
        local_owner = best_owner[y0:y1, x0:x1]
        local_second_owner = second_owner[y0:y1, x0:x1]
        better = local & (cost < local_best)
        not_better = local & ~better & (cost < local_second)
        local_second[better] = local_best[better]
        local_second_owner[better] = local_owner[better]
        local_best[better] = cost[better]
        local_owner[better] = instance_id
        local_second[not_better] = cost[not_better]
        local_second_owner[not_better] = instance_id
        candidate_union[y0:y1, x0:x1][local] = True
        trace_records.append({
            "cellpose_instance_id": instance_id,
            "roi_size": roi_size,
            "variant": variant,
            "roi_x0": x0, "roi_y0": y0, "roi_x1": x1, "roi_y1": y1,
            "traced_pixels": result["traced_pixels"],
            "boundary_seed_pixels": result["boundary_seed_pixels"],
            "stop_reason": result["stop_reason"],
        })
    _, process_candidate = evidence_masks(gfap, process, variant, config)
    process_candidate &= roi_union & (soma_candidates == 0)
    traced = candidate_union & (soma_candidates == 0) & (best_owner > 0)
    competing = traced & (second_owner > 0)
    ambiguous = np.zeros(shape, dtype=bool)
    if variant == "E1":
        gap = np.full(shape, np.inf, dtype=np.float32)
        np.subtract(second_cost, best_cost, out=gap, where=competing)
        tolerance = np.maximum(config.ambiguity_absolute_gap, config.ambiguity_relative_gap * best_cost)
        ambiguous = competing & (gap <= tolerance)
    assigned = traced & ~ambiguous
    unassigned = process_candidate & ~assigned & ~ambiguous
    state = np.zeros(shape, dtype=np.uint8)
    state[assigned] = ASSIGNED
    state[ambiguous] = AMBIGUOUS
    state[unassigned] = UNASSIGNED
    instance = soma_candidates.copy()
    instance[assigned] = best_owner[assigned]
    uncertainty = ambiguous | unassigned
    per_soma_components = []
    for record in selected:
        instance_id = record["cellpose_instance_id"]
        pixels = int(np.count_nonzero(assigned & (best_owner == instance_id)))
        if pixels:
            count, _ = cv2.connectedComponents((assigned & (best_owner == instance_id)).astype(np.uint8), connectivity=8)
            components = count - 1
        else:
            components = 0
        per_soma_components.append({
            "cellpose_instance_id": instance_id,
            "assigned_process_pixels": pixels,
            "assigned_connected_components": components,
        })
    stats = {
        "selected_soma_candidates": len(selected),
        "process_candidate_pixels": int(process_candidate.sum()),
        "traced_process_pixels": int(traced.sum()),
        "assigned_pixels": int(assigned.sum()),
        "ambiguous_pixels": int(ambiguous.sum()),
        "unassigned_pixels": int(unassigned.sum()),
        "uncertainty_pixels": int(uncertainty.sum()),
        "competing_trace_pixels": int(competing.sum()),
        "assigned_components": int(sum(r["assigned_connected_components"] for r in per_soma_components)),
        "candidates_with_assigned_process": int(sum(r["assigned_process_pixels"] > 0 for r in per_soma_components)),
        "maximum_components_per_candidate": int(max((r["assigned_connected_components"] for r in per_soma_components), default=0)),
    }
    return {
        "state": state,
        "instance": instance,
        "uncertainty": uncertainty,
        "traced": traced,
        "process_candidate": process_candidate,
        "soma_candidates": soma_candidates,
        "best_cost": best_cost,
        "second_cost": second_cost,
        "best_owner": best_owner,
        "second_owner": second_owner,
        "stats": stats,
        "trace_records": trace_records,
        "fragmentation_records": per_soma_components,
    }


def competition_cases(result: dict) -> list[dict]:
    """Describe connected E1 Ambiguous regions and their principal Soma pair."""
    state = result["state"]
    count, labels, _, centroids = cv2.connectedComponentsWithStats((state == AMBIGUOUS).astype(np.uint8), connectivity=8)
    cases = []
    for label in range(1, count):
        mask = labels == label
        owners = np.stack([result["best_owner"][mask], result["second_owner"][mask]], axis=1)
        pairs, counts = np.unique(owners, axis=0, return_counts=True)
        pair = pairs[int(np.argmax(counts))]
        ys, xs = np.nonzero(mask)
        cases.append({
            "ambiguous_region_pixels": int(mask.sum()),
            "soma_candidate_a": int(pair[0]), "soma_candidate_b": int(pair[1]),
            "center_x": float(centroids[label][0]), "center_y": float(centroids[label][1]),
            "bbox_x0": int(xs.min()), "bbox_y0": int(ys.min()),
            "bbox_x1": int(xs.max()) + 1, "bbox_y1": int(ys.max()) + 1,
            "mean_path_cost_gap": float((result["second_cost"][mask] - result["best_cost"][mask]).mean()),
        })
    return cases


def match_candidates_to_ground_truth(soma_candidates: np.ndarray, supervision: dict) -> list[dict]:
    """Evaluation-only one-to-one Soma matching; never used by tracing."""
    complete_ids = [int(v) for v in supervision["complete_cell_ids"]]
    candidate_ids = [int(v) for v in np.unique(soma_candidates) if v > 0]
    matches = []
    if not complete_ids or not candidate_ids:
        return [{"gt_cell_id": cell_id, "cellpose_instance_id": None, "soma_recall": 0.0, "intersection_pixels": 0} for cell_id in complete_ids]
    gt_soma = [(supervision["soma_owner"] == cell_id) & (supervision["morphology"] == 0) for cell_id in complete_ids]
    scores = np.zeros((len(complete_ids), len(candidate_ids)), dtype=np.float64)
    intersections = np.zeros_like(scores, dtype=np.int32)
    for row, mask in enumerate(gt_soma):
        area = int(mask.sum())
        for column, candidate_id in enumerate(candidate_ids):
            intersection = int(np.count_nonzero(mask & (soma_candidates == candidate_id)))
            intersections[row, column] = intersection
            scores[row, column] = intersection / max(area, 1)
    rows, columns = linear_sum_assignment(-scores)
    matched = {}
    for row, column in zip(rows, columns):
        if scores[row, column] >= 0.10 and intersections[row, column] >= 10:
            matched[complete_ids[row]] = {
                "gt_cell_id": complete_ids[row],
                "cellpose_instance_id": candidate_ids[column],
                "soma_recall": float(scores[row, column]),
                "intersection_pixels": int(intersections[row, column]),
            }
    return [matched.get(cell_id, {"gt_cell_id": cell_id, "cellpose_instance_id": None, "soma_recall": 0.0, "intersection_pixels": 0}) for cell_id in complete_ids]


def evaluate_association(result: dict, supervision: dict) -> list[dict]:
    """Evaluate only confirmed Process pixels; unlabeled pixels are never negatives."""
    soma_candidates = np.where(result["instance"] > 0, result["instance"], 0)
    # Candidate Soma matching must use the original candidate mask, not traced instances.
    candidate_mask = result["soma_candidates"]
    matches = match_candidates_to_ground_truth(candidate_mask, supervision)
    rows = []
    confirmed_positive = supervision["morphology"] != 255
    for match in matches:
        gt_id = match["gt_cell_id"]
        candidate_id = match["cellpose_instance_id"]
        gt_process = (supervision["process_owner"] == gt_id) & (supervision["morphology"] == 1)
        total = int(gt_process.sum())
        state = result["state"]
        correct = int(np.count_nonzero(gt_process & (result["instance"] == candidate_id) & (state == ASSIGNED))) if candidate_id else 0
        ambiguous = int(np.count_nonzero(gt_process & (state == AMBIGUOUS)))
        unassigned = int(np.count_nonzero(gt_process & (state == UNASSIGNED)))
        not_candidate = total - int(np.count_nonzero(gt_process & (state > 0)))
        extension_review = 0
        if candidate_id:
            extension_review = int(np.count_nonzero((result["instance"] == candidate_id) & (state == ASSIGNED) & ~confirmed_positive))
        rows.append({
            **match,
            "gt_process_pixels": total,
            "confirmed_process_assigned_correct_pixels": correct,
            "confirmed_process_coverage": correct / total if total else None,
            "confirmed_process_traced_any_state_pixels": int(np.count_nonzero(gt_process & ((state == ASSIGNED) | (state == AMBIGUOUS)))),
            "gt_process_ambiguous_pixels": ambiguous,
            "gt_process_unassigned_pixels": unassigned,
            "gt_process_outside_candidate_state_pixels": not_candidate,
            "extension_review_pixels_not_confirmed_false_positive": extension_review,
            "unlabeled_pixels_treated_as_background": False,
        })
    return rows
