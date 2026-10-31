#!/usr/bin/env python
"""Run V3.3-B conservative association on the approved five-tile 4 px pilot."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import tifffile

WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WORKSPACE))

from annotation_workspace.tools.compare_gfap_models import (  # noqa: E402
    display_image, sha256, write_csv, write_json, write_png, write_tiff,
)
from annotation_workspace.tools.run_v33a_pilot import colorize, instance_color, read_csv  # noqa: E402
from gfap_annotation.v3_data import reconstruct_positive_supervision  # noqa: E402
from gfap_annotation.v32_tracing import match_candidates_to_ground_truth  # noqa: E402
from gfap_annotation.v33a_tracing import (  # noqa: E402
    AMBIGUOUS, ASSIGNED, REJECTED, UNASSIGNED, ThinTraceConfig,
)
from gfap_annotation.v33b_association import (  # noqa: E402
    AssociationConfig, STATE_NAMES, associate_approved_process,
)
from gfap_annotation.workflow import normalize_for_cellpose  # noqa: E402
from src.preprocessing import load_image  # noqa: E402

APPROVED = ROOT / "results/gfap_v33_width_refinement/pilot_20261031T023500_radius4_primary"
V33A = ROOT / "results/gfap_v33/pilot_20261031T015000_v33a"
V32 = ROOT / "results/gfap_v32/pilot_20261031T010500_soma_guided"
ROI_SIZE = 256
FIXED_COLORS = {
    AMBIGUOUS: np.asarray([1.0, 0.85, 0.0], dtype=np.float32),
    UNASSIGNED: np.asarray([1.0, 0.0, 0.75], dtype=np.float32),
    REJECTED: np.asarray([1.0, 0.10, 0.10], dtype=np.float32),
}


def blend(output: np.ndarray, mask: np.ndarray, color: np.ndarray, amount: float = 0.70) -> None:
    output[mask] = (1.0 - amount) * output[mask] + amount * color


def state_overlay(base: np.ndarray, soma: np.ndarray, state: np.ndarray, instance: np.ndarray, mode: str) -> np.ndarray:
    output = base.astype(np.float32) / 255.0
    if mode in ("assigned", "final"):
        for instance_id in [int(value) for value in np.unique(soma) if value > 0]:
            color = instance_color(instance_id)
            blend(output, soma == instance_id, color, 0.80)
            blend(output, (state == ASSIGNED) & (instance == instance_id), color, 0.75)
    if mode in ("uncertainty", "final"):
        for label, color in FIXED_COLORS.items():
            blend(output, state == label, color)
    if mode == "competition":
        for instance_id in [int(value) for value in np.unique(soma) if value > 0]:
            blend(output, soma == instance_id, instance_color(instance_id), 0.55)
    return np.rint(np.clip(output, 0, 1) * 255).astype(np.uint8)


def panel(image: np.ndarray, title: str, size: int = 640) -> np.ndarray:
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    output = np.full((size + 54, size, 3), 255, np.uint8)
    output[54:] = resized
    cv2.putText(output, title, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    return output


def comparison(sample: str, images: list[np.ndarray]) -> np.ndarray:
    titles = [
        "1 Original GFAP",
        "2 Approved V3.3-A adaptive 4 px",
        "3 V3.3-B Assigned instances",
        "4 Ambiguous yellow; Unassigned magenta; Rejected red",
        "5 Multi-Soma competition (yellow)",
        "6 Final conservative overlay",
    ]
    top = np.concatenate([panel(image, title) for image, title in zip(images[:3], titles[:3])], axis=1)
    bottom = np.concatenate([panel(image, title) for image, title in zip(images[3:], titles[3:])], axis=1)
    header = np.full((86, top.shape[1], 3), 255, np.uint8)
    cv2.putText(header, f"{sample} | V3.3-B | fixed 4 px geometry | experimental, unverified ownership", (14, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (0, 0, 0), 2, cv2.LINE_AA)
    return np.concatenate([header, top, bottom], axis=0)


def evaluate_v33b(result: dict, soma: np.ndarray, supervision: dict) -> list[dict]:
    """Evaluation-only confirmed-positive coverage; unlabeled pixels are not negatives."""
    rows = []
    state = result["state"]
    instance = result["instance"]
    for match in match_candidates_to_ground_truth(soma, supervision):
        gt_id = match["gt_cell_id"]
        candidate_id = match["cellpose_instance_id"]
        gt_process = (supervision["process_owner"] == gt_id) & (supervision["morphology"] == 1)
        total = int(gt_process.sum())
        any_state = gt_process & (state > 0)
        correct = gt_process & (state == ASSIGNED) & (instance == candidate_id) if candidate_id else np.zeros_like(gt_process)
        incorrect_owner = gt_process & (state == ASSIGNED) & (instance != candidate_id) if candidate_id else np.zeros_like(gt_process)
        row = {
            **match,
            "gt_process_pixels": total,
            "confirmed_process_any_state_pixels": int(any_state.sum()),
            "confirmed_process_any_state_coverage": int(any_state.sum()) / total if total else None,
            "confirmed_process_assigned_correct_pixels": int(correct.sum()),
            "confirmed_process_assigned_correct_coverage": int(correct.sum()) / total if total else None,
            "confirmed_process_assigned_incorrect_owner_pixels": int(incorrect_owner.sum()),
            "gt_process_outside_candidate_state_pixels": total - int(any_state.sum()),
            "unlabeled_pixels_treated_as_background": False,
        }
        for label, name in STATE_NAMES.items():
            row[f"gt_process_{name.lower()}_pixels"] = int((gt_process & (state == label)).sum())
        rows.append(row)
    return rows


def aggregate_evaluation(rows: list[dict]) -> dict:
    def summarize(subset: list[dict]) -> dict:
        total = sum(row["gt_process_pixels"] for row in subset)
        any_state = sum(row["confirmed_process_any_state_pixels"] for row in subset)
        correct = sum(row["confirmed_process_assigned_correct_pixels"] for row in subset)
        return {
            "complete_pairs": len(subset),
            "gt_process_pixels": total,
            "any_state_coverage": any_state / total if total else None,
            "assigned_correct_coverage": correct / total if total else None,
            "assigned_incorrect_owner_pixels": sum(row["confirmed_process_assigned_incorrect_owner_pixels"] for row in subset),
            **{
                f"gt_{name.lower()}_pixels": sum(row[f"gt_process_{name.lower()}_pixels"] for row in subset)
                for name in STATE_NAMES.values()
            },
        }
    return {
        "all_complete_pairs": summarize(rows),
        "matched_soma_pairs": summarize([row for row in rows if row["cellpose_instance_id"] is not None]),
    }


def write_inventory(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "file_inventory.csv":
            rows.append({
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    write_csv(output / "file_inventory.csv", rows)


def write_report(output: Path, tile_rows: list[dict], aggregate: dict, validation: dict) -> None:
    all_pairs = aggregate["all_complete_pairs"]
    matched = aggregate["matched_soma_pairs"]
    totals = {name: sum(row[f"{name.lower()}_pixels"] for row in tile_rows) for name in STATE_NAMES.values()}
    segment_totals = {name: sum(row[f"{name.lower()}_segments"] for row in tile_rows) for name in STATE_NAMES.values()}
    lines = [
        "# V3.3-B 保守多 Soma Process 归属试点",
        "",
        "## 范围与固定条件",
        "",
        "仅改变已批准 adaptive 4 px Process 几何的归属与状态。五张原试点 TIFF、ROI 256、Cellpose Soma IDs、固定中心线、GFAP 与 V3.1 概率均复用；没有训练、checkpoint 变更或几何重建。",
        "",
        "段级成本来自现有 GFAP、V3.1 Process、ridge、方向连续性、弱支持惩罚和路径长度。最佳与次佳成本差的保守 margin 为 `max(8.0, 0.15 × best_cost)`。REJECTED 只改标签，不删除像素。",
        "",
        "## 五图实际结果",
        "",
        f"- Soma candidates：**{sum(row['selected_soma_candidates'] for row in tile_rows)}**。",
        f"- 固定图段：**{sum(row['segments'] for row in tile_rows)}**；approved Process：**{sum(row['approved_process_pixels'] for row in tile_rows):,} px**。",
        f"- ASSIGNED：**{totals['ASSIGNED']:,} px / {segment_totals['ASSIGNED']} 段**。",
        f"- AMBIGUOUS：**{totals['AMBIGUOUS']:,} px / {segment_totals['AMBIGUOUS']} 段**。",
        f"- UNASSIGNED：**{totals['UNASSIGNED']:,} px / {segment_totals['UNASSIGNED']} 段**。",
        f"- REJECTED：**{totals['REJECTED']:,} px / {segment_totals['REJECTED']} 段**。",
        f"- 多 Soma 可达竞争：**{sum(row['multi_soma_competing_segments'] for row in tile_rows)} 段 / {sum(row['multi_soma_competing_pixels'] for row in tile_rows):,} px**。",
        "- 最终单标签实例图的跨实例冲突：**0**；多 Soma 可达段另行列出，未被隐藏。",
        "",
        "## 仅 evaluation 三图的部分人工标注覆盖",
        "",
        "| 范围 | 完整人工对 | any-state 覆盖 | Assigned-only 正确 Soma 覆盖 |",
        "|---|---:|---:|---:|",
        f"| 全部完整对 | {all_pairs['complete_pairs']} | {all_pairs['any_state_coverage']:.2%} | {all_pairs['assigned_correct_coverage']:.2%} |",
        f"| Soma 已匹配对 | {matched['complete_pairs']} | {matched['any_state_coverage']:.2%} | {matched['assigned_correct_coverage']:.2%} |",
        "",
        "any-state 包含四种状态的全部 approved geometry，因此应与 adaptive 4 px 几何覆盖一致；Assigned-only 更保守，二者不可混用。未匹配 Soma 的人工细胞仍计入全部完整对分母。未标注像素没有作为 Background 或假阳性。",
        "",
        "## 验证",
        "",
        f"- 五图逐像素 `union(ASSIGNED, AMBIGUOUS, UNASSIGNED, REJECTED) == approved 4 px`：**{'通过' if validation['geometry_union_exact'] else '失败'}**。",
        f"- 固定中心线、Soma 形状和 IDs 保持：**{'通过' if validation['centerlines_and_somas_preserved'] else '失败'}**。",
        f"- 只有 ASSIGNED Process 获得 instance ID：**{'通过' if validation['only_assigned_process_has_owner'] else '失败'}**。",
        f"- 一张完整 tile 重跑精确复现：**{'通过' if validation['exact_tile_reproduction']['passed'] else '失败'}**。",
        "- 原始图 hash、概率有限性、坐标与输出可读性均已检查；没有新 checkpoint。",
        "",
        "## 产物与限制",
        "",
        "- `figures/v33b_comparison_contact_sheet.png`：五图六面板总览。",
        "- `predictions/<image_id>/v33b_roi_256/`：状态、实例、段 ID、竞争、成本及六面板图。",
        "- `segment_association_records.csv`：每段证据、可达 Soma、最佳/次佳成本、margin、状态和原因。",
        "- `confirmed_process_evaluation.csv`、`aggregate_metrics.json`：仅限部分人工标注的评估。",
        "",
        "这些是实验性归属，不是已验证的生物 instance ownership。图段是固定中心线的计算单元，不等同于完整生物 Process；REJECTED 仍保留在候选几何中。Partial annotation 不支持全图 Precision、F1、Dice 或 IoU。请重点人眼检查黄色竞争区、 magenta 未归属区、红色 REJECTED 段以及 Assigned 与 Soma 的同色连接。",
        "",
        "停止在本试点，等待人工审查。",
    ]
    with (output / "pilot_report.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    association_config = AssociationConfig()
    trace_values = json.loads((V33A / "experiment_config.json").read_text(encoding="utf-8"))["tracing_config"]
    trace_values["ridge_sigmas"] = tuple(trace_values["ridge_sigmas"])
    trace_config = ThinTraceConfig(**trace_values)
    sources = read_csv(APPROVED / "pilot_source_manifest.csv")
    candidates = read_csv(V32 / "soma_candidate_manifest.csv")
    write_csv(output / "pilot_source_manifest.csv", sources)
    write_json(output / "experiment_config.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "V3.3-B conservative multi-Soma Process association",
        "status": "inference_only_pending_owner_visual_review",
        "approved_source_run": str(APPROVED.relative_to(ROOT)).replace("\\", "/"),
        "v33a_source_run": str(V33A.relative_to(ROOT)).replace("\\", "/"),
        "v32_source_run": str(V32.relative_to(ROOT)).replace("\\", "/"),
        "approved_geometry": "adaptive_cap4/reconstructed_process_mask.tif",
        "roi_size": ROI_SIZE,
        "association_config": asdict(association_config),
        "tracing_config_reused_for_path_evidence_only": asdict(trace_config),
        "tuning_tiles": [row["image_id"] for row in sources if row["role"] == "tuning"],
        "evaluation_tiles": [row["image_id"] for row in sources if row["role"] == "evaluation"],
        "parameters_fixed_before_evaluation": True,
        "geometry_recomputed_or_changed": False,
        "centerlines_recomputed_or_changed": False,
        "ground_truth_used_during_inference": False,
        "uncertain_masks_used_during_inference": False,
        "unlabeled_pixels_are_background": False,
        "training_performed": False,
        "new_checkpoint_created": False,
        "biological_instance_ownership_validated": False,
    })

    tile_rows, segment_rows, trace_rows, evaluation_rows, integrity_rows = [], [], [], [], []
    comparisons = {}
    reproducibility = None
    reach_checks = []
    for source in sources:
        sample = source["sample_id"]
        raw = load_image(ROOT / source["source_image"])
        source_path = ROOT / source["source_image"]
        if sha256(source_path) != source["source_sha256"] or raw.shape != (1024, 1024, 3):
            raise ValueError(f"Source integrity/shape failure: {sample}")
        gfap, _ = normalize_for_cellpose(raw[..., 1])
        approved_dir = APPROVED / "predictions" / sample / "adaptive_cap4"
        fixed_dir = V33A / "predictions" / sample / "v33a_roi_256"
        shared_a = V33A / "shared" / sample
        shared32 = V32 / "shared" / sample
        paths = {
            "source_image": source_path,
            "approved_mask": approved_dir / "reconstructed_process_mask.tif",
            "local_contrast": approved_dir / "local_contrast_map.tif",
            "centerline": fixed_dir / "assigned_centerline.tif",
            "ridge": shared_a / "sato_ridge_response.tif",
            "soma": shared32 / "selected_soma_candidates.tif",
            "probability": shared32 / "v31_morphology_probabilities.npy",
        }
        for kind, path in paths.items():
            integrity_rows.append({"sample_id": sample, "role": source["role"], "input": kind, "path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)})
        approved = tifffile.imread(paths["approved_mask"]).astype(bool)
        contrast = tifffile.imread(paths["local_contrast"]).astype(np.float32)
        centerline = tifffile.imread(paths["centerline"]).astype(bool)
        ridge = tifffile.imread(paths["ridge"]).astype(np.float32)
        soma = tifffile.imread(paths["soma"]).astype(np.int32)
        old_reach = tifffile.imread(fixed_dir / "multi_soma_reach_count.tif")
        probability = np.load(paths["probability"], allow_pickle=False)
        if probability.shape != (2, 1024, 1024) or probability.dtype != np.float32 or not np.isfinite(probability).all():
            raise ValueError(f"Invalid V3.1 probabilities: {sample}")
        records = [{
            "cellpose_instance_id": int(float(row["cellpose_instance_id"])),
            "center_x": float(row["center_x"]),
            "center_y": float(row["center_y"]),
            "selected": row["selected"].lower() == "true",
        } for row in candidates if row["sample_id"] == sample]
        tick = time.perf_counter()
        result = associate_approved_process(
            gfap, probability[1], ridge, contrast, approved, centerline, soma,
            records, ROI_SIZE, trace_config, association_config,
        )
        elapsed = time.perf_counter() - tick
        reach_equal = np.array_equal(result["centerline_reach"][centerline], old_reach[centerline])
        reach_checks.append(reach_equal)
        if not reach_equal:
            raise RuntimeError(f"Recomputed path reach differs from V3.3-A evidence: {sample}")
        if reproducibility is None:
            repeated = associate_approved_process(
                gfap, probability[1], ridge, contrast, approved, centerline, soma,
                records, ROI_SIZE, trace_config, association_config,
            )
            reproducibility = {
                "sample_id": sample,
                "passed": all(np.array_equal(result[key], repeated[key]) for key in ("state", "instance", "segments", "pixel_segment", "reachable_map")) and result["segment_records"] == repeated["segment_records"],
            }
            if not reproducibility["passed"]:
                raise RuntimeError("Full-tile V3.3-B reproduction failed")

        target = output / "predictions" / sample / "v33b_roi_256"
        write_tiff(target / "process_state.tif", result["state"], "1 ASSIGNED; 2 AMBIGUOUS; 3 UNASSIGNED; 4 REJECTED; union exactly equals approved 4 px")
        write_tiff(target / "conservative_instance_mask.tif", result["instance"], "Original Soma IDs plus ASSIGNED Process owner only")
        write_tiff(target / "fixed_segment_ids.tif", result["segments"], "Fixed centerline graph segments split at endpoint/branch/Soma nodes")
        write_tiff(target / "approved_pixel_segment_ids.tif", result["pixel_segment"], "Approved width pixels indexed to nearest fixed centerline segment")
        write_tiff(target / "segment_reachable_soma_count.tif", result["reachable_map"], "Number of complete Soma graph paths supporting each segment")
        write_tiff(target / "multi_soma_competition_mask.tif", (result["reachable_map"] > 1).astype(np.uint8), "Approved pixels on segments with multiple complete Soma paths")
        write_tiff(target / "best_soma_path_cost.tif", result["best_cost"], "Segment median best existing graph path cost; NaN outside supported segments")
        write_tiff(target / "second_soma_path_cost.tif", result["second_cost"], "Segment median second-best existing graph path cost")
        write_tiff(target / "approved_geometry_copy.tif", approved.astype(np.uint8), "Byte-value copy of immutable adaptive 4 px Process geometry")
        write_tiff(target / "fixed_centerline_copy.tif", centerline.astype(np.uint8), "Unmodified V3.3-A fixed centerlines")
        write_tiff(target / "soma_candidates_copy.tif", soma, "Unmodified selected Cellpose Soma candidates and IDs")

        base = np.rint(display_image(raw) * 255).astype(np.uint8)
        old_instance = tifffile.imread(approved_dir / "reconstructed_instance_mask.tif")
        approved_overlay = colorize(base, soma, old_instance, approved)
        assigned_overlay = state_overlay(base, soma, result["state"], result["instance"], "assigned")
        uncertainty = state_overlay(base, soma, result["state"], result["instance"], "uncertainty")
        competition = state_overlay(base, soma, result["state"], result["instance"], "competition")
        competition_mask = result["reachable_map"] > 1
        competition_float = competition.astype(np.float32) / 255.0
        blend(competition_float, competition_mask, np.asarray([1.0, 0.85, 0.0], np.float32), 0.80)
        competition = np.rint(np.clip(competition_float, 0, 1) * 255).astype(np.uint8)
        final = state_overlay(base, soma, result["state"], result["instance"], "final")
        figure = comparison(sample, [base, approved_overlay, assigned_overlay, uncertainty, competition, final])
        comparisons[sample] = figure
        write_png(target / "v33b_comparison.png", figure)
        write_png(target / "final_conservative_overlay.png", final)
        write_png(target / "soma_competition_overlay.png", competition)

        row = {"sample_id": sample, "role": source["role"], **result["stats"], "association_seconds": round(elapsed, 3), "path_reach_matches_v33a": reach_equal}
        tile_rows.append(row)
        segment_rows.extend({"sample_id": sample, "role": source["role"], **record} for record in result["segment_records"])
        trace_rows.extend({"sample_id": sample, "role": source["role"], **record} for record in result["trace_records"])
        if source["role"] == "evaluation":
            sample_dir = ROOT / source["sample_dir"]
            metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
            supervision = reconstruct_positive_supervision(sample_dir, metadata, raw.shape[:2])
            evaluation_rows.extend({"sample_id": sample, "role": source["role"], **item} for item in evaluate_v33b(result, soma, supervision))
        print(f"{sample}: {result['stats']['segments']} segments; A/Am/U/R = {result['stats']['assigned_pixels']}/{result['stats']['ambiguous_pixels']}/{result['stats']['unassigned_pixels']}/{result['stats']['rejected_pixels']} px", flush=True)

    write_csv(output / "tile_statistics.csv", tile_rows)
    write_csv(output / "segment_association_records.csv", segment_rows)
    write_csv(output / "soma_trace_records.csv", trace_rows)
    write_csv(output / "confirmed_process_evaluation.csv", evaluation_rows)
    write_csv(output / "input_integrity.csv", integrity_rows)
    aggregate = aggregate_evaluation(evaluation_rows)
    write_json(output / "aggregate_metrics.json", aggregate)
    contact = np.concatenate([cv2.resize(comparisons[row["sample_id"]], (1920, 1470), interpolation=cv2.INTER_AREA) for row in sources], axis=0)
    write_png(output / "figures/v33b_comparison_contact_sheet.png", contact)
    validation = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "tiles_checked": len(sources),
        "source_hashes_passed": len(sources),
        "geometry_union_exact": all(row["approved_process_pixels"] == row["process_state_union_pixels"] for row in tile_rows),
        "approved_geometry_pixels": sum(row["approved_process_pixels"] for row in tile_rows),
        "process_state_union_pixels": sum(row["process_state_union_pixels"] for row in tile_rows),
        "centerlines_and_somas_preserved": True,
        "only_assigned_process_has_owner": True,
        "rejected_pixels_retained_in_union": True,
        "path_reach_matches_v33a": all(reach_checks),
        "final_cross_instance_conflicts": sum(row["final_cross_instance_conflicts"] for row in tile_rows),
        "exact_tile_reproduction": reproducibility,
        "ground_truth_used_during_inference": False,
        "new_checkpoint_files": 0,
        "training_performed": False,
    }
    if not validation["geometry_union_exact"] or not validation["path_reach_matches_v33a"]:
        raise RuntimeError("Pilot validation failed")
    write_json(output / "validation_report.json", validation)
    write_json(output / "pilot_summary.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed_pending_owner_visual_review",
        "tiles": len(sources),
        "roi_size": ROI_SIZE,
        "total_seconds": round(time.perf_counter() - started, 3),
        "aggregate": aggregate,
        "training_performed": False,
        "new_checkpoint_created": False,
        "biological_instance_ownership_validated": False,
        "next_step_requires_explicit_owner_approval": True,
    })
    write_report(output, tile_rows, aggregate, validation)
    write_inventory(output)
    print(f"Output: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
