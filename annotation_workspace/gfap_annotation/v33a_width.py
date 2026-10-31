"""Conservative adaptive width reconstruction from fixed V3.3-A centerlines.

This stage never retraces, creates branches, changes Soma IDs, or implements V3.3-B.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def _local_tangents(centerline: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate centerline tangents from a 5x5 neighborhood using vectorized PCA."""
    skeleton = np.asarray(centerline, dtype=bool)
    yy, xx = np.mgrid[-2:3, -2:3]
    sxx = np.zeros(skeleton.shape, np.float32)
    sxy = np.zeros(skeleton.shape, np.float32)
    syy = np.zeros(skeleton.shape, np.float32)
    count = np.zeros(skeleton.shape, np.int16)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(skeleton)
            y_src = slice(max(0, -dy), min(skeleton.shape[0], skeleton.shape[0] - dy))
            x_src = slice(max(0, -dx), min(skeleton.shape[1], skeleton.shape[1] - dx))
            y_dst = slice(max(0, dy), min(skeleton.shape[0], skeleton.shape[0] + dy))
            x_dst = slice(max(0, dx), min(skeleton.shape[1], skeleton.shape[1] + dx))
            shifted[y_dst, x_dst] = skeleton[y_src, x_src]
            count += shifted
            sxx += shifted * float(dx * dx)
            sxy += shifted * float(dx * dy)
            syy += shifted * float(dy * dy)
    ys, xs = np.nonzero(skeleton)
    covariance = np.stack([
        np.stack([sxx[ys, xs], sxy[ys, xs]], axis=-1),
        np.stack([sxy[ys, xs], syy[ys, xs]], axis=-1),
    ], axis=-2)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tangent = eigenvectors[..., :, 1]
    anisotropy = (eigenvalues[:, 1] - eigenvalues[:, 0]) / np.maximum(eigenvalues.sum(axis=-1), 1e-6)
    return ys, xs, tangent.astype(np.float32), anisotropy.astype(np.float32)


def estimate_profile_widths(
    gfap: np.ndarray,
    structural_candidate: np.ndarray,
    assigned_centerline: np.ndarray,
    maximum_radius: int = 5,
    minimum_gfap: float = 0.07,
    minimum_contrast: float = 0.03,
) -> dict:
    """Estimate radius from intensity profiles perpendicular to fixed centerlines."""
    if maximum_radius <= 0:
        raise ValueError("Profile radius must be positive")
    image = np.asarray(gfap, dtype=np.float32)
    structural = np.asarray(structural_candidate, dtype=bool)
    centerline = np.asarray(assigned_centerline, dtype=bool)
    if image.shape != structural.shape or image.shape != centerline.shape:
        raise ValueError("Width inputs must share coordinates")
    ys, xs, tangent, anisotropy = _local_tangents(centerline)
    count = len(ys)
    shape = image.shape
    radius = np.zeros(shape, np.float32)
    left_radius = np.zeros(shape, np.float32)
    right_radius = np.zeros(shape, np.float32)
    contrast_map = np.zeros(shape, np.float32)
    reliable_map = np.zeros(shape, bool)
    if not count:
        return {"radius": radius, "left_radius": left_radius, "right_radius": right_radius, "contrast": contrast_map, "reliable": reliable_map}
    normal_x = -tangent[:, 1]
    normal_y = tangent[:, 0]
    step = 0.25
    sample_radius = maximum_radius + 3  # Background estimation only; never a reconstruction cap.
    offsets = np.arange(-sample_radius, sample_radius + step, step, dtype=np.float32)
    sample_y = ys[:, None] + normal_y[:, None] * offsets[None, :]
    sample_x = xs[:, None] + normal_x[:, None] * offsets[None, :]
    profile = ndimage.map_coordinates(image, [sample_y, sample_x], order=1, mode="constant", cval=0.0)
    support = ndimage.map_coordinates(structural.astype(np.uint8), [sample_y, sample_x], order=0, mode="constant", cval=0).astype(bool)
    center = sample_radius * 4
    peak = profile[:, center - 1:center + 2].max(axis=1)
    background = np.median(np.concatenate([profile[:, :3], profile[:, -3:]], axis=1), axis=1)
    contrast = np.maximum(peak - background, 0)
    threshold = np.maximum.reduce([
        np.full(count, minimum_gfap, np.float32),
        background + 0.25 * contrast,
        0.50 * peak,
    ])
    left = np.zeros(count, np.float32)
    right = np.zeros(count, np.float32)
    for direction, output in ((-1, left), (1, right)):
        active = np.ones(count, dtype=bool)
        for distance_index in range(1, maximum_radius * 4 + 1):
            index = center + direction * distance_index
            unsupported = (~support[:, index]) | (profile[:, index] < threshold)
            newly_stopped = active & unsupported
            output[newly_stopped] = (distance_index - 1) * step
            active &= ~unsupported
        output[active] = maximum_radius
    reliable = (anisotropy >= 0.20) & (peak >= 0.08) & (contrast >= minimum_contrast)
    left[~reliable] = 0
    right[~reliable] = 0
    estimated = 0.5 * (left + right)
    radius[ys, xs] = estimated
    left_radius[ys, xs] = left
    right_radius[ys, xs] = right
    contrast_map[ys, xs] = contrast
    reliable_map[ys, xs] = reliable
    return {"radius": radius, "left_radius": left_radius, "right_radius": right_radius, "contrast": contrast_map, "reliable": reliable_map}


def reconstruct_adaptive(
    gfap: np.ndarray,
    structural_candidate: np.ndarray,
    assigned_centerline: np.ndarray,
    centerline_instance: np.ndarray,
    soma_candidates: np.ndarray,
    unassigned_or_ambiguous: np.ndarray,
    radius_cap: int,
    maximum_profile_radius: int = 5,
) -> dict:
    """Apply evidence-estimated widths with strict ownership and connectivity guards."""
    if radius_cap not in (3, 4, 5):
        raise ValueError("Controlled caps are 3, 4, and 5 pixels")
    profiles = estimate_profile_widths(gfap, structural_candidate, assigned_centerline, maximum_profile_radius)
    capped_radius = np.minimum(profiles["radius"], radius_cap).astype(np.float32)
    integer_radius = np.rint(capped_radius).astype(np.int16)
    forbidden = np.asarray(unassigned_or_ambiguous, dtype=bool) | (soma_candidates > 0)
    support = np.asarray(structural_candidate, dtype=bool) & (gfap >= 0.07) & ~forbidden
    assigned = np.asarray(assigned_centerline, dtype=bool)
    owner_count = np.zeros(assigned.shape, np.uint16)
    first_owner = np.zeros(assigned.shape, np.int32)
    for instance_id in [int(v) for v in np.unique(centerline_instance[assigned]) if v > 0]:
        own_centerline = assigned & (centerline_instance == instance_id)
        other_centerline = assigned & ~own_centerline
        owner_mask = own_centerline.copy()
        for radius in range(1, radius_cap + 1):
            seeds = own_centerline & (integer_radius >= radius)
            if not seeds.any():
                continue
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            expanded = cv2.dilate(seeds.astype(np.uint8), kernel).astype(bool)
            owner_mask |= expanded & support & ~other_centerline
        # Retain only components physically connected to this owner's fixed centerline.
        count, labels = cv2.connectedComponents(owner_mask.astype(np.uint8), connectivity=8)
        connected_labels = np.unique(labels[own_centerline])
        keep = np.zeros(count, dtype=bool)
        keep[connected_labels[connected_labels > 0]] = True
        owner_mask = keep[labels] & ~other_centerline & ~forbidden
        owner_mask &= ndimage.distance_transform_edt(~own_centerline) <= radius_cap
        owner_count[owner_mask] += 1
        available = owner_mask & (first_owner == 0)
        first_owner[available] = instance_id
    conflicts = (owner_count > 1) & ~assigned
    # Fixed centerlines are immutable even where another model marked uncertainty;
    # the prohibition applies only to newly added width pixels.
    reconstructed = ((owner_count == 1) & ~forbidden) | assigned
    # Conflict removal can cut a widened lobe off its centerline. Remove only
    # such orphaned fragments; never remove a component containing fixed centerline.
    connected_reconstruction = np.zeros_like(reconstructed)
    for instance_id in [int(v) for v in np.unique(centerline_instance[assigned]) if v > 0]:
        own_centerline = assigned & (centerline_instance == instance_id)
        own = (reconstructed & (first_owner == instance_id)) | own_centerline
        count, labels = cv2.connectedComponents(own.astype(np.uint8), connectivity=8)
        connected_labels = np.unique(labels[own_centerline])
        keep = np.zeros(count, dtype=bool)
        keep[connected_labels[connected_labels > 0]] = True
        connected_reconstruction |= keep[labels]
    reconstructed = connected_reconstruction
    instance = soma_candidates.copy().astype(np.int32)
    instance[reconstructed & ~assigned] = first_owner[reconstructed & ~assigned]
    instance[assigned] = centerline_instance[assigned]
    stats = {
        "radius_cap": radius_cap,
        "reconstructed_pixels": int(reconstructed.sum()),
        "cross_instance_overlap_attempt_pixels": int(conflicts.sum()),
        "profile_reliable_centerline_pixels": int((profiles["reliable"] & assigned).sum()),
        "profile_unreliable_centerline_pixels": int((~profiles["reliable"] & assigned).sum()),
        "conflict_pixels_excluded_from_instances": int(conflicts.sum()),
        "unassigned_or_ambiguous_pixels_absorbed": int((reconstructed & unassigned_or_ambiguous & ~assigned).sum()),
        "centerline_pixels_lost": int((assigned & ~reconstructed).sum()),
    }
    return {
        "reconstructed_mask": reconstructed,
        "reconstructed_instance": instance,
        "conflict_mask": conflicts,
        "radius_map": capped_radius,
        "left_radius_map": np.minimum(profiles["left_radius"], radius_cap).astype(np.float32),
        "right_radius_map": np.minimum(profiles["right_radius"], radius_cap).astype(np.float32),
        "contrast_map": profiles["contrast"],
        "profile_reliable": profiles["reliable"],
        "stats": stats,
    }
