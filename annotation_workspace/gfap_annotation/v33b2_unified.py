"""V3.3-B.2 unified Soma-centered Process recognition.

Additional GFAP/V3-supported structures are merged into one graph before tracing.
Every structural unit then uses the same evidence, competition, and ownership
rules. Ground truth is deliberately absent from every inference API.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from gfap_annotation.v33a_tracing import (
    AMBIGUOUS,
    ASSIGNED,
    REJECTED,
    UNASSIGNED,
    ThinTraceConfig,
)
from gfap_annotation.v33a_width import reconstruct_adaptive
from gfap_annotation.v33b1_branch import build_structural_units
from gfap_annotation.v33b_association import (
    AssociationConfig,
    STATE_NAMES,
    associate_approved_process,
)


@dataclass(frozen=True)
class UnifiedConfig:
    """Fixed pilot parameters; evaluation annotations are not inference inputs."""

    background_sigma: float = 8.0
    minimum_new_gfap: float = 0.12
    minimum_new_ridge: float = 0.02
    minimum_new_contrast: float = 0.04
    assisted_minimum_process: float = 0.35
    assisted_minimum_gfap: float = 0.09
    assisted_minimum_ridge: float = 0.025
    assisted_minimum_contrast: float = 0.03
    minimum_component_pixels: int = 4
    ambiguity_absolute_gap: float = 4.0
    ambiguity_relative_gap: float = 0.08
    maximum_competitive_path_cost: float = 240.0
    radius_cap: int = 4


def _finite_2d(name: str, array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    result = np.asarray(array, dtype=np.float32)
    if result.shape != shape or result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 2D array in source coordinates")
    return result


def discover_unified_candidates(
    gfap: np.ndarray,
    process_probability: np.ndarray,
    ridge: np.ndarray,
    soma_candidates: np.ndarray,
    existing_structural: np.ndarray,
    existing_centerline: np.ndarray,
    config: UnifiedConfig = UnifiedConfig(),
    candidate_pool_centerline: np.ndarray | None = None,
) -> dict:
    """Merge selected additional structural evidence before ownership.

    The fixed approved centerlines are always retained. A broader historical
    candidate pool may contribute only pixels that pass the same local intensity,
    contrast, ridge/V3 evidence and minimum-component guards as new detections.
    """
    shape = soma_candidates.shape
    image = _finite_2d("GFAP", gfap, shape)
    process = _finite_2d("V3 Process probability", process_probability, shape)
    ridge = _finite_2d("Ridge", ridge, shape)
    old_structural = np.asarray(existing_structural, dtype=bool)
    old_centerline = np.asarray(existing_centerline, dtype=bool)
    pool = old_centerline if candidate_pool_centerline is None else np.asarray(candidate_pool_centerline, dtype=bool)
    if old_structural.shape != shape or old_centerline.shape != shape or pool.shape != shape or len(shape) != 2:
        raise ValueError("Existing graph and Soma candidates must share coordinates")
    background = ndimage.gaussian_filter(image, config.background_sigma, mode="reflect")
    contrast = np.clip(image - background, 0, None).astype(np.float32)
    outside_soma = np.asarray(soma_candidates) == 0
    bright = (
        (image >= config.minimum_new_gfap)
        & (ridge >= config.minimum_new_ridge)
        & (contrast >= config.minimum_new_contrast)
    )
    assisted = (
        (process >= config.assisted_minimum_process)
        & (image >= config.assisted_minimum_gfap)
        & (ridge >= config.assisted_minimum_ridge)
        & (contrast >= config.assisted_minimum_contrast)
    )
    supported = (bright | assisted) & outside_soma
    proposed = supported & ~old_structural
    labels, _ = ndimage.label(proposed, structure=ndimage.generate_binary_structure(2, 1))
    keep = np.bincount(labels.ravel()) > config.minimum_component_pixels
    keep[0] = False
    new_structural = keep[labels]
    proposed_centerline = (skeletonize(new_structural) | (pool & supported)) & outside_soma & ~old_centerline
    count, components = cv2.connectedComponents(proposed_centerline.astype(np.uint8), connectivity=8)
    sizes = np.bincount(components.ravel(), minlength=count)
    retained = sizes > config.minimum_component_pixels
    retained[0] = False
    new_centerline = retained[components]
    unified_structural = old_structural | new_structural
    unified_centerline = old_centerline | new_centerline
    if np.any(unified_centerline & ~outside_soma):
        raise RuntimeError("Unified centerlines entered a Soma candidate")
    return {
        "local_contrast": contrast,
        "bright_candidate": bright & outside_soma,
        "v3_assisted_candidate": assisted & outside_soma,
        "new_structural": new_structural,
        "new_centerline": new_centerline,
        "unified_structural": unified_structural,
        "unified_centerline": unified_centerline,
        "stats": {
            "new_structural_candidate_pixels": int(new_structural.sum()),
            "new_centerline_pixels": int(new_centerline.sum()),
            "unified_structural_candidate_pixels": int(unified_structural.sum()),
            "unified_centerline_pixels": int(unified_centerline.sum()),
            "bright_candidate_pixels": int((bright & outside_soma).sum()),
            "v3_assisted_candidate_pixels": int((assisted & outside_soma).sum()),
            "candidate_pool_centerline_pixels": int(pool.sum()),
            "supported_pool_centerline_pixels": int((pool & supported & ~old_centerline).sum()),
        },
    }


def decide_branch(
    mean_gfap: float,
    mean_ridge: float,
    mean_contrast: float,
    costs: dict[int, float],
    evidence_config: AssociationConfig = AssociationConfig(),
    config: UnifiedConfig = UnifiedConfig(),
) -> dict:
    """Require two strong, comparable complete paths for AMBIGUOUS.

    Weak secondary reachability never triggers uncertainty. There is no nearest-
    Soma fallback and no propagation from neighboring branches.
    """
    result = {
        "best_soma_id": None,
        "best_path_cost": None,
        "second_soma_id": None,
        "second_path_cost": None,
        "path_cost_gap": None,
        "ambiguity_margin": None,
        "strong_competition": False,
    }
    ordinary = (
        mean_gfap >= evidence_config.minimum_gfap
        and mean_ridge >= evidence_config.minimum_ridge
        and mean_contrast >= evidence_config.minimum_mean_contrast
    )
    strong = mean_gfap >= evidence_config.strong_gfap and mean_ridge >= evidence_config.strong_minimum_ridge
    if not (ordinary or strong):
        return {**result, "state": REJECTED, "owner": 0, "reason": "insufficient_unified_structural_evidence"}
    if not costs:
        return {**result, "state": UNASSIGNED, "owner": 0, "reason": "no_complete_soma_path"}
    ordered = sorted((float(cost), int(instance_id)) for instance_id, cost in costs.items())
    best_cost, best_id = ordered[0]
    result.update({"best_soma_id": best_id, "best_path_cost": best_cost})
    if len(ordered) == 1:
        return {**result, "state": ASSIGNED, "owner": best_id, "reason": "unique_complete_soma_path"}
    second_cost, second_id = ordered[1]
    gap = second_cost - best_cost
    margin = max(config.ambiguity_absolute_gap, config.ambiguity_relative_gap * best_cost)
    strong_competition = second_cost <= config.maximum_competitive_path_cost
    comparable = gap <= margin
    result.update({
        "second_soma_id": second_id,
        "second_path_cost": second_cost,
        "path_cost_gap": gap,
        "ambiguity_margin": margin,
        "strong_competition": strong_competition,
    })
    if strong_competition and comparable:
        return {**result, "state": AMBIGUOUS, "owner": 0, "reason": "two_strong_comparable_soma_paths"}
    reason = "weak_secondary_path_not_competitive" if not strong_competition else "downstream_cost_clearly_separates_somas"
    return {**result, "state": ASSIGNED, "owner": best_id, "reason": reason}


def _parse_costs(record: dict) -> dict[int, float]:
    costs = {}
    for item in record.get("reachable_soma_costs", "").split(";"):
        if item:
            instance_id, cost = item.split(":", 1)
            costs[int(instance_id)] = float(cost)
    return costs


def _weighted_mean(records: list[dict], weights: list[int], key: str) -> float:
    values = np.asarray([float(record[key]) for record in records], dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    return float(np.sum(values * weight) / max(weight.sum(), 1.0))


def associate_unit_evidence(
    gfap: np.ndarray,
    structural_candidate: np.ndarray,
    existing_approved: np.ndarray,
    centerline: np.ndarray,
    soma_candidates: np.ndarray,
    segments: np.ndarray,
    segment_records: list[dict],
    config: UnifiedConfig = UnifiedConfig(),
    evidence_config: AssociationConfig = AssociationConfig(),
) -> dict:
    """Make one branch-level decision per unit and reconstruct bounded width."""
    shape = gfap.shape
    image = _finite_2d("GFAP", gfap, shape)
    structural = np.asarray(structural_candidate, dtype=bool)
    old_approved = np.asarray(existing_approved, dtype=bool)
    skeleton = np.asarray(centerline, dtype=bool)
    soma = np.asarray(soma_candidates, dtype=np.int32)
    if any(array.shape != shape for array in (structural, old_approved, skeleton, soma, segments)):
        raise ValueError("All unified association inputs must share coordinates")
    if np.any(old_approved & (soma > 0)) or np.any(skeleton & (soma > 0)):
        raise ValueError("Process geometry overlaps a Soma candidate")
    if not np.array_equal(segments > 0, skeleton):
        raise ValueError("Unified segments do not match the unified centerline")
    # B.2 uses continuous topological branches as the sole decision units.
    # Reachability changes remain in source segment records, but do not create
    # additional one-pixel boundaries inside an otherwise continuous branch.
    graph = build_structural_units(skeleton, soma)
    unit_map = graph["unit_map"]
    records_by_segment = {int(record["segment_id"]): record for record in segment_records}
    pairs, pair_counts = np.unique(
        np.stack([segments[skeleton], unit_map[skeleton]], axis=1), axis=0, return_counts=True,
    )
    sources: dict[int, list[int]] = {unit_id: [] for unit_id in graph["unit_types"]}
    weights: dict[tuple[int, int], int] = {}
    for (segment_id, unit_id), count in zip(pairs, pair_counts):
        key = (int(segment_id), int(unit_id))
        sources[key[1]].append(key[0])
        weights[key] = int(count)

    centerline_counts = np.bincount(unit_map.ravel(), minlength=max(graph["unit_types"], default=0) + 1)
    unit_records: dict[int, dict] = {}
    for unit_id, kind in graph["unit_types"].items():
        source_ids = sorted(sources[unit_id])
        source_records = [records_by_segment[segment_id] for segment_id in source_ids]
        local_weights = [weights[(segment_id, unit_id)] for segment_id in source_ids]
        costs_by_segment = [_parse_costs(record) for record in source_records]
        common_ids = set(costs_by_segment[0])
        for costs in costs_by_segment[1:]:
            common_ids &= set(costs)
        total_weight = max(sum(local_weights), 1)
        costs = {
            instance_id: sum(
                costs_by_segment[index][instance_id] * local_weights[index]
                for index in range(len(source_ids))
            ) / total_weight
            for instance_id in sorted(common_ids)
        }
        mean_gfap = _weighted_mean(source_records, local_weights, "mean_gfap")
        mean_ridge = _weighted_mean(source_records, local_weights, "mean_ridge")
        mean_contrast = _weighted_mean(source_records, local_weights, "mean_local_contrast")
        decision = decide_branch(mean_gfap, mean_ridge, mean_contrast, costs, evidence_config, config)
        unit_records[unit_id] = {
            "unit_id": unit_id,
            "unit_type": kind,
            "source_segment_ids": ";".join(str(value) for value in source_ids),
            "source_segment_count": len(source_ids),
            "centerline_pixels": int(centerline_counts[unit_id]),
            "mean_gfap": mean_gfap,
            "mean_local_contrast": mean_contrast,
            "mean_ridge": mean_ridge,
            "mean_v31_process_probability": _weighted_mean(source_records, local_weights, "mean_v31_process_probability"),
            "reachable_soma_ids": ";".join(str(value) for value in sorted(costs)),
            "reachable_soma_costs": ";".join(f"{key}:{costs[key]:.8g}" for key in sorted(costs)),
            "reachable_soma_count": len(costs),
            **decision,
            "state_name": STATE_NAMES[decision["state"]],
            "assigned_soma_instance_id": decision["owner"],
            "uncertainty_propagated": False,
        }

    max_unit = max(unit_records, default=0)
    state_lookup = np.zeros(max_unit + 1, dtype=np.uint8)
    owner_lookup = np.zeros(max_unit + 1, dtype=np.int32)
    reach_lookup = np.zeros(max_unit + 1, dtype=np.uint16)
    for unit_id, record in unit_records.items():
        state_lookup[unit_id] = record["state"]
        owner_lookup[unit_id] = record["owner"]
        reach_lookup[unit_id] = record["reachable_soma_count"]
    center_state = state_lookup[unit_map]
    center_owner = owner_lookup[unit_map]
    _, nearest = ndimage.distance_transform_edt(~skeleton, return_indices=True)
    nearest_unit = unit_map[tuple(nearest)]
    pixel_state = state_lookup[nearest_unit]
    pixel_owner = owner_lookup[nearest_unit]

    # Existing approved pixels are immutable. New width is reconstructed by the
    # same adaptive 4 px method, then indexed to one structural unit. No local
    # pixel-level uncertainty exceptions are introduced.
    geometry = old_approved | skeleton
    added_by_state = np.zeros(shape, dtype=np.uint8)
    for label in (ASSIGNED, AMBIGUOUS, UNASSIGNED, REJECTED):
        state_centerline = skeleton & (center_state == label)
        if not state_centerline.any():
            continue
        owners = center_owner if label == ASSIGNED else state_centerline.astype(np.int32)
        forbidden = (skeleton & ~state_centerline) | (soma > 0)
        reconstructed = reconstruct_adaptive(
            image, structural, state_centerline, owners, soma, forbidden, config.radius_cap,
        )["reconstructed_mask"]
        accepted = reconstructed & ~geometry & (pixel_state == label)
        geometry |= accepted
        added_by_state[accepted] = label
    geometry &= soma == 0
    state = np.where(geometry, pixel_state, 0).astype(np.uint8)
    instance = soma.copy()
    assigned = geometry & (state == ASSIGNED)
    instance[assigned] = pixel_owner[assigned]
    pixel_unit = np.where(geometry, nearest_unit, 0).astype(np.int32)
    added = geometry & ~old_approved

    if np.any(old_approved & ~geometry) or np.any(skeleton & ~geometry):
        raise RuntimeError("Existing geometry or unified centerlines were lost")
    if not np.array_equal(instance[soma > 0], soma[soma > 0]):
        raise RuntimeError("Soma IDs or shapes changed")
    if np.any(geometry & (state != ASSIGNED) & (instance > 0)):
        raise RuntimeError("Non-ASSIGNED Process received an owner")
    final_counts = np.bincount(pixel_unit.ravel(), minlength=max_unit + 1)
    added_counts = np.bincount(pixel_unit[added], minlength=max_unit + 1)
    for unit_id, record in unit_records.items():
        record["final_process_pixels"] = int(final_counts[unit_id])
        record["added_process_pixels"] = int(added_counts[unit_id])
        record["new_centerline_unit"] = False  # Annotated by the outer unified call.

    distance = ndimage.distance_transform_edt(~skeleton)
    stats = {
        "structural_units": len(unit_records),
        "branch_units": sum(kind == "branch" for kind in graph["unit_types"].values()),
        "crossing_units": sum(kind == "crossing" for kind in graph["unit_types"].values()),
        "existing_approved_pixels": int(old_approved.sum()),
        "unified_process_pixels": int(geometry.sum()),
        "added_process_pixels": int(added.sum()),
        "maximum_geometry_distance_to_centerline": float(distance[geometry].max()) if geometry.any() else 0.0,
        "strong_competing_units": sum(record.get("strong_competition", False) for record in unit_records.values()),
        "weak_secondary_assigned_units": sum(record["reason"] == "weak_secondary_path_not_competitive" for record in unit_records.values()),
        "clearly_separated_assigned_units": sum(record["reason"] == "downstream_cost_clearly_separates_somas" for record in unit_records.values()),
        "uncertainty_propagated_units": 0,
        "branch_state_consistency_violations": 0,
        "final_cross_instance_conflicts": 0,
    }
    for label, name in STATE_NAMES.items():
        stats[f"{name.lower()}_pixels"] = int((state == label).sum())
        stats[f"{name.lower()}_units"] = sum(record["state"] == label for record in unit_records.values())
        stats[f"added_{name.lower()}_pixels"] = int((added & (state == label)).sum())
    return {
        "state": state,
        "instance": instance,
        "geometry": geometry,
        "added": added,
        "unit_map": unit_map,
        "pixel_unit": pixel_unit,
        "reachable_map": np.where(geometry, reach_lookup[nearest_unit], 0).astype(np.uint16),
        "unit_records": [unit_records[unit_id] for unit_id in sorted(unit_records)],
        "adjacency": graph["adjacency"],
        "stats": stats,
    }


def recognize_unified_process(
    gfap: np.ndarray,
    process_probability: np.ndarray,
    ridge: np.ndarray,
    soma_candidates: np.ndarray,
    existing_structural: np.ndarray,
    existing_centerline: np.ndarray,
    existing_approved: np.ndarray,
    candidate_records: list[dict],
    roi_size: int,
    trace_config: ThinTraceConfig,
    config: UnifiedConfig = UnifiedConfig(),
    evidence_config: AssociationConfig = AssociationConfig(),
    candidate_pool_centerline: np.ndarray | None = None,
) -> dict:
    """Execute the single unified discovery-tracing-association pipeline."""
    discovered = discover_unified_candidates(
        gfap, process_probability, ridge, soma_candidates,
        existing_structural, existing_centerline, config, candidate_pool_centerline,
    )
    from gfap_annotation.v32_tracing import roi_bounds

    roi_union = np.zeros(gfap.shape, dtype=bool)
    for record in candidate_records:
        if not record["selected"]:
            continue
        x0, y0, x1, y1 = roi_bounds(float(record["center_x"]), float(record["center_y"]), roi_size, gfap.shape)
        roi_union[y0:y1, x0:x1] = True
    outside_roi = discovered["new_centerline"] & ~roi_union
    discovered["new_centerline"] = discovered["new_centerline"] & roi_union
    discovered["unified_centerline"] = np.asarray(existing_centerline, dtype=bool) | discovered["new_centerline"]
    discovered["stats"]["roi_excluded_new_centerline_pixels"] = int(outside_roi.sum())
    discovered["stats"]["new_centerline_pixels"] = int(discovered["new_centerline"].sum())
    discovered["stats"]["unified_centerline_pixels"] = int(discovered["unified_centerline"].sum())
    skeleton = discovered["unified_centerline"]
    # This first pass computes only shared path evidence on centerline pixels.
    # Its preliminary states are discarded; all final states come from the same
    # branch-level decision below, including both old and newly discovered units.
    evidence = associate_approved_process(
        gfap,
        process_probability,
        ridge,
        discovered["local_contrast"],
        skeleton,
        skeleton,
        soma_candidates,
        candidate_records,
        roi_size,
        trace_config,
        evidence_config,
    )
    result = associate_unit_evidence(
        gfap,
        discovered["unified_structural"],
        existing_approved,
        skeleton,
        soma_candidates,
        evidence["segments"],
        evidence["segment_records"],
        config,
        evidence_config,
    )
    old_centerline = np.asarray(existing_centerline, dtype=bool)
    new_unit_ids = set(int(value) for value in np.unique(result["unit_map"][skeleton & ~old_centerline]))
    for record in result["unit_records"]:
        record["new_centerline_unit"] = record["unit_id"] in new_unit_ids
    association_stats = result["stats"]
    result.update(discovered)
    result["segments"] = evidence["segments"]
    result["trace_records"] = evidence["trace_records"]
    result["stats"] = {**discovered["stats"], **association_stats}
    result["stats"]["new_centerline_units"] = sum(record["new_centerline_unit"] for record in result["unit_records"])
    return result
