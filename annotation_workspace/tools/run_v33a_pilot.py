#!/usr/bin/env python
"""Run only V3.3-A on the five existing V3.2 tiles; V3.3-B is not implemented here."""
from __future__ import annotations

import argparse
import colorsys
import csv
from dataclasses import asdict
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import tifffile
from skimage.morphology import skeletonize

WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WORKSPACE))

from annotation_workspace.tools.compare_gfap_models import (  # noqa: E402
    display_image, sha256, write_csv, write_json, write_npy, write_png, write_tiff,
)
from gfap_annotation.v3_data import reconstruct_positive_supervision  # noqa: E402
from gfap_annotation.v32_tracing import evaluate_association  # noqa: E402
from gfap_annotation.v33a_tracing import (  # noqa: E402
    ASSIGNED, UNASSIGNED, ThinTraceConfig, build_thin_candidates, trace_thin_tile,
)
from gfap_annotation.workflow import normalize_for_cellpose  # noqa: E402
from src.preprocessing import load_image  # noqa: E402

V32_RUN = ROOT / "results/gfap_v32/pilot_20261031T010500_soma_guided"
ROI_SIZE = 256


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def integer(value: str) -> int:
    return int(float(value))


def instance_color(instance_id: int) -> np.ndarray:
    rgb = colorsys.hsv_to_rgb((instance_id * 0.61803398875) % 1.0, 0.78, 1.0)
    return np.asarray(rgb, dtype=np.float32)


def colorize(base: np.ndarray, soma: np.ndarray, owner: np.ndarray, process: np.ndarray) -> np.ndarray:
    result = base.astype(np.float32) / 255.0
    for instance_id in [int(v) for v in np.unique(soma) if v > 0]:
        color = instance_color(instance_id)
        own_process = process & (owner == instance_id)
        own_soma = soma == instance_id
        result[own_process] = 0.25 * result[own_process] + 0.75 * color
        result[own_soma] = 0.20 * result[own_soma] + 0.80 * color
    return np.rint(np.clip(result, 0, 1) * 255).astype(np.uint8)


def baseline_overlay(base: np.ndarray, state: np.ndarray, instance: np.ndarray, soma: np.ndarray) -> np.ndarray:
    result = colorize(base, soma, instance, state == ASSIGNED).astype(np.float32) / 255.0
    for label, color in ((2, (1.0, 0.85, 0.0)), (3, (1.0, 0.0, 0.75))):
        mask = state == label
        result[mask] = 0.40 * result[mask] + 0.60 * np.asarray(color, dtype=np.float32)
    return np.rint(np.clip(result, 0, 1) * 255).astype(np.uint8)


def uncertainty_overlay(base: np.ndarray, result: dict) -> np.ndarray:
    output = base.astype(np.float32) / 255.0
    unassigned = result["state"] == UNASSIGNED
    competing = result["reach_count"] > 1
    output[unassigned] = 0.40 * output[unassigned] + 0.60 * np.asarray([1.0, 0.0, 0.75], np.float32)
    output[competing] = 0.35 * output[competing] + 0.65 * np.asarray([1.0, 0.85, 0.0], np.float32)
    soma = result["instance"] * (result["instance"] > 0)
    for instance_id in [int(v) for v in np.unique(result["instance"]) if v > 0]:
        mask = (result["instance"] == instance_id) & ~unassigned & ~competing
        output[mask] = 0.25 * output[mask] + 0.75 * instance_color(instance_id)
    return np.rint(np.clip(output, 0, 1) * 255).astype(np.uint8)


def title_panel(image: np.ndarray, title: str, size: int = 640) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    panel = np.full((size + 52, size, 3), 255, dtype=np.uint8)
    panel[52:] = resized
    cv2.putText(panel, title, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    return panel


def comparison(sample: str, panels: list[np.ndarray]) -> np.ndarray:
    titles = [
        "Original GFAP", "Cellpose Soma candidates", "V3.2 E1 baseline; ROI 256",
        "V3.3-A assigned centerline", "V3.3-A bounded-width reconstruction",
        "A: Unassigned magenta; multi-Soma reach yellow",
    ]
    top = np.concatenate([title_panel(im, title) for im, title in zip(panels[:3], titles[:3])], axis=1)
    bottom = np.concatenate([title_panel(im, title) for im, title in zip(panels[3:], titles[3:])], axis=1)
    header = np.full((84, top.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(header, f"{sample} | V3.3-A only | unverified; V3.3-B not run", (14, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (0, 0, 0), 2, cv2.LINE_AA)
    return np.concatenate([header, top, bottom], axis=0)


def contact_sheet(sample_ids: list[str], comparisons: dict[str, np.ndarray]) -> np.ndarray:
    rows = []
    for sample in sample_ids:
        rows.append(cv2.resize(comparisons[sample], (1920, 1470), interpolation=cv2.INTER_AREA))
    return np.concatenate(rows, axis=0)


def add_ground_truth_contours(image: np.ndarray, supervision: dict) -> np.ndarray:
    result = image.copy()
    for cell_id in supervision["complete_cell_ids"]:
        for kind, color in (("soma", (255, 255, 255)), ("process", (255, 80, 0))):
            owner = supervision[f"{kind}_owner"] == cell_id
            valid = supervision["morphology"] == (0 if kind == "soma" else 1)
            contours, _ = cv2.findContours((owner & valid).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, color, 1)
    return result


def width_statistics(mask: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {"width_sample_pixels": 0, "width_p50": None, "width_p90": None, "width_p95": None, "width_max": None}
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    values = 2.0 * distance[skeletonize(mask)]
    values = values[values > 0]
    if not values.size:
        return {"width_sample_pixels": 0, "width_p50": None, "width_p90": None, "width_p95": None, "width_max": None}
    return {
        "width_sample_pixels": int(values.size),
        "width_p50": float(np.percentile(values, 50)), "width_p90": float(np.percentile(values, 90)),
        "width_p95": float(np.percentile(values, 95)), "width_max": float(values.max()),
    }


def component_statistics(state: np.ndarray, instance: np.ndarray, soma: np.ndarray) -> dict:
    components = 0
    candidates_with_process = 0
    for instance_id in [int(v) for v in np.unique(soma) if v > 0]:
        mask = (state == ASSIGNED) & (instance == instance_id)
        if mask.any():
            candidates_with_process += 1
            count, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
            components += count - 1
    return {
        "assigned_components": components,
        "candidates_with_assigned_process": candidates_with_process,
        "components_per_candidate_with_process": components / candidates_with_process if candidates_with_process else 0.0,
    }


def coverage_summary(rows: list[dict]) -> dict:
    total = sum(r["gt_process_pixels"] for r in rows)
    correct = sum(r["confirmed_process_assigned_correct_pixels"] for r in rows)
    matched = [r for r in rows if r["cellpose_instance_id"] is not None]
    matched_total = sum(r["gt_process_pixels"] for r in matched)
    return {
        "gt_complete_pairs": len(rows), "matched_soma_pairs": len(matched),
        "confirmed_process_coverage_micro": correct / total if total else None,
        "confirmed_process_coverage_matched_pairs_micro": correct / matched_total if matched_total else None,
        "confirmed_process_traced_any_state_micro": sum(r["confirmed_process_traced_any_state_pixels"] for r in rows) / total if total else None,
        "extension_review_pixels_not_false_positive_rate": sum(r["extension_review_pixels_not_confirmed_false_positive"] for r in rows),
        "unlabeled_pixels_used_as_background": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    config = ThinTraceConfig()
    sources = read_csv(V32_RUN / "pilot_source_manifest.csv")
    candidates = read_csv(V32_RUN / "soma_candidate_manifest.csv")
    write_csv(output / "pilot_source_manifest.csv", sources)
    write_json(output / "experiment_config.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "V3.3-A thin-structure graph tracing only",
        "status": "inference_only_pending_owner_visual_review",
        "v32_source_run": str(V32_RUN.relative_to(ROOT)).replace("\\", "/"),
        "v32_baseline": "E1 ROI 256",
        "roi_size": ROI_SIZE, "ridge_method": "sato", "tracing_config": asdict(config),
        "tuning_tiles": [r["image_id"] for r in sources if r["role"] == "tuning"],
        "evaluation_tiles": [r["image_id"] for r in sources if r["role"] == "evaluation"],
        "ground_truth_used_during_inference": False,
        "uncertain_masks_used_during_inference": False,
        "v33b_executed": False, "training_performed": False,
        "unlabeled_pixels_are_background": False,
    })

    tile_rows, evaluation_rows, trace_rows, fragmentation_rows, ridge_rows = [], [], [], [], []
    comparisons = {}
    for source in sources:
        sample = source["sample_id"]
        role = source["role"]
        raw = load_image(ROOT / source["source_image"])
        if sha256(ROOT / source["source_image"]) != source["source_sha256"] or raw.shape != (1024, 1024, 3):
            raise ValueError(f"Source integrity/shape failure: {sample}")
        gfap, normalization = normalize_for_cellpose(raw[..., 1])
        shared32 = V32_RUN / "shared" / sample
        soma = tifffile.imread(shared32 / "selected_soma_candidates.tif")
        probability = np.load(shared32 / "v31_morphology_probabilities.npy", allow_pickle=False)
        if probability.shape != (2, 1024, 1024) or not np.isfinite(probability).all():
            raise ValueError(f"Invalid shared V3.1 probabilities: {sample}")
        records = [{
            "cellpose_instance_id": integer(r["cellpose_instance_id"]),
            "center_x": float(r["center_x"]), "center_y": float(r["center_y"]),
            "selected": r["selected"].lower() == "true",
        } for r in candidates if r["sample_id"] == sample]
        tick = time.perf_counter()
        thin = build_thin_candidates(gfap, soma, config, method="sato")
        result = trace_thin_tile(gfap, probability[1], soma, records, thin, ROI_SIZE, config)
        elapsed = time.perf_counter() - tick
        shared = output / "shared" / sample
        write_tiff(shared / "sato_ridge_response.tif", thin["ridge"], "Normalized multiscale Sato response from GFAP channel 1")
        write_tiff(shared / "structural_candidate_mask.tif", thin["structural_candidate"].astype(np.uint8), "GFAP structural candidate before skeletonization")
        write_tiff(shared / "candidate_centerline.tif", thin["candidate_centerline"].astype(np.uint8), "Skeleton of credible GFAP structural candidate")
        write_tiff(shared / "graph_nodes.tif", thin["graph_nodes"], "Bitmask: 1 endpoint; 2 branch point; 4 Soma connection")
        write_npy(shared / "gfap_normalization.npy", np.asarray([normalization["low"], normalization["high"]], dtype=np.float32))

        if role == "tuning":
            methods = {"sato": thin["ridge"]}
            for method in ("meijering", "frangi"):
                methods[method] = build_thin_candidates(gfap, soma, config, method=method)["ridge"]
            ridge_panels = []
            for method, ridge in methods.items():
                mask = (gfap >= config.minimum_gfap) & (ridge >= config.minimum_ridge) & (soma == 0)
                ridge_rows.append({
                    "sample_id": sample, "role": role, "method": method,
                    "ridge_response_p50": float(np.percentile(ridge, 50)),
                    "ridge_response_p95": float(np.percentile(ridge, 95)),
                    "ordinary_threshold_pixels": int(mask.sum()),
                    "note": "Tuning-only structural diagnostic; Sato fixed for pilot; no held-out parameter selection",
                })
                ridge_panels.append(title_panel(np.rint(np.clip(ridge, 0, 1) * 255).astype(np.uint8), method))
            write_png(output / "figures" / "ridge_diagnostics" / f"{sample}.png", np.concatenate(ridge_panels, axis=1))

        target = output / "predictions" / sample / "v33a_roi_256"
        write_tiff(target / "association_state.tif", result["state"], "A only: 1 Assigned; 3 Unassigned; no B Ambiguous/Rejected")
        write_tiff(target / "assigned_centerline.tif", result["assigned_centerline"].astype(np.uint8), "Primary V3.3-A Soma-associated graph centerline")
        write_tiff(target / "centerline_instance_mask.tif", result["instance"], "Soma IDs and Assigned centerline owner IDs")
        write_tiff(target / "reconstructed_process_mask.tif", result["reconstructed_mask"].astype(np.uint8), "Bounded local-width reconstruction; maximum radius 3")
        write_tiff(target / "reconstructed_instance_mask.tif", result["reconstructed_instance"], "Width pixels inherit associated centerline owner")
        write_tiff(target / "estimated_width_map.tif", result["width_map"], "Local structural full-width estimates at assigned centerline")
        write_tiff(target / "multi_soma_reach_count.tif", result["reach_count"], "Number of independently reaching Soma graph traces; no B arbitration")
        write_tiff(target / "unassigned_centerline.tif", (result["state"] == UNASSIGNED).astype(np.uint8), "Candidate graph centerline without a reaching Soma")
        trace_rows.extend({"sample_id": sample, "role": role, **r} for r in result["trace_records"])
        fragmentation_rows.extend({"sample_id": sample, "role": role, **r} for r in result["fragmentation_records"])

        baseline_dir = V32_RUN / "predictions" / sample / "roi_256" / "E1"
        baseline_state = tifffile.imread(baseline_dir / "association_state.tif")
        baseline_instance = tifffile.imread(baseline_dir / "assigned_instance_mask.tif")
        base = np.rint(display_image(raw) * 255).astype(np.uint8)
        candidate_image = colorize(base, soma, soma, np.zeros_like(soma, dtype=bool))
        baseline_image = baseline_overlay(base, baseline_state, baseline_instance, soma)
        centerline_image = colorize(base, soma, result["instance"], result["assigned_centerline"])
        reconstructed_image = colorize(base, soma, result["reconstructed_instance"], result["reconstructed_mask"])
        uncertain_image = uncertainty_overlay(base, result)
        candidate_centerline_image = base.copy()
        candidate_centerline_image[thin["candidate_centerline"]] = (0, 220, 255)
        write_png(target / "candidate_graph_centerline.png", candidate_centerline_image)
        write_png(target / "assigned_centerline_overlay.png", centerline_image)
        write_png(target / "reconstructed_process_overlay.png", reconstructed_image)
        write_png(target / "unassigned_and_multisoma_overlay.png", uncertain_image)
        figure = comparison(sample, [base, candidate_image, baseline_image, centerline_image, reconstructed_image, uncertain_image])
        comparisons[sample] = figure
        write_png(target / "v32_vs_v33a_comparison.png", figure)

        row = {
            "sample_id": sample, "role": role, **result["stats"],
            "baseline_assigned_pixels": int((baseline_state == ASSIGNED).sum()),
            "baseline_ambiguous_pixels": int((baseline_state == 2).sum()),
            "baseline_unassigned_pixels": int((baseline_state == 3).sum()),
            "uncertain_overlap_centerline_evaluation_only": 0,
            "uncertain_overlap_reconstruction_evaluation_only": 0,
            "v33a_tracing_seconds": round(elapsed, 3),
            **{f"v33a_centerline_{k}": v for k, v in component_statistics(result["state"], result["instance"], soma).items()},
            **{f"v33a_reconstructed_{k}": v for k, v in width_statistics(result["reconstructed_mask"]).items()},
            **{f"baseline_{k}": v for k, v in width_statistics((baseline_state == ASSIGNED)).items()},
        }
        if role == "evaluation":
            sample_dir = ROOT / source["sample_dir"]
            metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
            supervision = reconstruct_positive_supervision(sample_dir, metadata, raw.shape[:2])
            uncertain = tifffile.imread(sample_dir / "uncertain_mask.tif") > 0
            row["uncertain_overlap_centerline_evaluation_only"] = int((result["assigned_centerline"] & uncertain).sum())
            row["uncertain_overlap_reconstruction_evaluation_only"] = int((result["reconstructed_mask"] & uncertain).sum())
            reconstruction_state = np.zeros_like(result["state"])
            reconstruction_state[result["reconstructed_mask"]] = ASSIGNED
            reconstruction_state[result["state"] == UNASSIGNED] = UNASSIGNED
            for model, eval_result in (
                ("v32_e1", {"state": baseline_state, "instance": baseline_instance, "soma_candidates": soma}),
                ("v33a_centerline", result),
                ("v33a_reconstructed", {"state": reconstruction_state, "instance": result["reconstructed_instance"], "soma_candidates": soma}),
            ):
                for item in evaluate_association(eval_result, supervision):
                    evaluation_rows.append({"sample_id": sample, "role": role, "model": model, **item})
            review = add_ground_truth_contours(reconstructed_image, supervision)
            write_png(target / "reconstruction_with_ground_truth_review.png", review)
        tile_rows.append(row)
        print(f"{sample}: Assigned centerline {result['stats']['assigned_centerline_pixels']}, reconstructed {result['stats']['reconstructed_assigned_pixels']}, Unassigned {result['stats']['unassigned_centerline_pixels']}", flush=True)

    write_csv(output / "tile_statistics.csv", tile_rows)
    write_csv(output / "soma_trace_records.csv", trace_rows)
    write_csv(output / "centerline_fragmentation.csv", fragmentation_rows)
    write_csv(output / "ridge_method_diagnostics_tuning_only.csv", ridge_rows)
    write_csv(output / "confirmed_process_evaluation.csv", evaluation_rows)
    aggregate = []
    for model in ("v32_e1", "v33a_centerline", "v33a_reconstructed"):
        rows = [r for r in evaluation_rows if r["model"] == model]
        aggregate.append({"model": model, **coverage_summary(rows)})
    write_csv(output / "aggregate_metrics.csv", aggregate)
    ordered = [r["sample_id"] for r in sources]
    write_png(output / "figures/v32_vs_v33a_contact_sheet.png", contact_sheet(ordered, comparisons))
    write_json(output / "pilot_summary.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed_pending_owner_visual_review",
        "tiles": len(sources), "roi_size": ROI_SIZE, "v33b_executed": False,
        "aggregate": aggregate, "total_seconds": round(time.perf_counter() - started, 3),
        "training_performed": False, "new_checkpoint_created": False,
        "next_step_requires_explicit_owner_approval": True,
    })
    print(f"Output: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
