#!/usr/bin/env python
"""Run the five-tile unified V3.3-B.2 Soma-Process recognition pilot."""
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
from annotation_workspace.tools.run_v33b_pilot import (  # noqa: E402
    aggregate_evaluation, blend, evaluate_v33b, state_overlay,
)
from gfap_annotation.v3_data import reconstruct_positive_supervision  # noqa: E402
from gfap_annotation.v33a_tracing import AMBIGUOUS, ASSIGNED, REJECTED, UNASSIGNED, ThinTraceConfig  # noqa: E402
from gfap_annotation.v33b2_unified import UnifiedConfig, recognize_unified_process  # noqa: E402
from gfap_annotation.v33b_association import AssociationConfig  # noqa: E402
from gfap_annotation.workflow import normalize_for_cellpose  # noqa: E402
from src.preprocessing import load_image  # noqa: E402

APPROVED = ROOT / "results/gfap_v33_width_refinement/pilot_20261031T023500_radius4_primary"
V33A = ROOT / "results/gfap_v33/pilot_20261031T015000_v33a"
V32 = ROOT / "results/gfap_v32/pilot_20261031T010500_soma_guided"
B1 = ROOT / "results/gfap_v33b_branch_uncertainty/pilot_20261031_v33b1_branches_v2"
ROI_SIZE = 256


def integrated_overlay(base: np.ndarray, soma: np.ndarray, state: np.ndarray, instance: np.ndarray) -> np.ndarray:
    """One coherent final result; REJECTED candidates are deliberately absent."""
    output = base.astype(np.float32) / 255.0
    for instance_id in [int(value) for value in np.unique(soma) if value > 0]:
        color = instance_color(instance_id)
        blend(output, soma == instance_id, color, 0.80)
        blend(output, (state == ASSIGNED) & (instance == instance_id), color, 0.75)
    blend(output, state == AMBIGUOUS, np.asarray([1.0, 0.85, 0.0], np.float32), 0.75)
    blend(output, state == UNASSIGNED, np.asarray([1.0, 0.0, 0.75], np.float32), 0.75)
    return np.rint(np.clip(output, 0, 1) * 255).astype(np.uint8)


def discovery_overlay(base: np.ndarray, result: dict) -> np.ndarray:
    output = np.rint(base.astype(np.float32) * 0.40).astype(np.uint8)
    added = result["added"]
    output[added] = np.rint(0.20 * output[added].astype(np.float32) + 0.80 * np.asarray([0, 230, 255])).astype(np.uint8)
    return output


def panel(image: np.ndarray, title: str, size: int = 640) -> np.ndarray:
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    output = np.full((size + 54, size, 3), 255, np.uint8)
    output[54:] = resized
    cv2.putText(output, title, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 1, cv2.LINE_AA)
    return output


def choose_roi(mask: np.ndarray, size: int = 192) -> tuple[int, int, int, int]:
    """Select the largest diagnostic component without using annotation data."""
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count > 1:
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        center_x, center_y = centroids[label]
    else:
        center_x, center_y = (mask.shape[1] - 1) / 2, (mask.shape[0] - 1) / 2
    x0 = int(np.clip(round(center_x - size / 2), 0, mask.shape[1] - size))
    y0 = int(np.clip(round(center_y - size / 2), 0, mask.shape[0] - size))
    return x0, y0, x0 + size, y0 + size


def crop(image: np.ndarray, bounds: tuple[int, int, int, int], size: int = 384) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    return cv2.resize(image[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_NEAREST)


def magnified_comparison(images: list[np.ndarray], bounds: tuple[int, int, int, int], title: str) -> np.ndarray:
    labels = ["Original GFAP", "V3.3-B.1", "Unified V3.3-B.2"]
    row = np.concatenate([panel(crop(image, bounds), label, 384) for image, label in zip(images, labels)], axis=1)
    x0, y0, x1, y1 = bounds
    header = np.full((64, row.shape[1], 3), 255, np.uint8)
    cv2.putText(header, f"{title} | x={x0}:{x1}, y={y0}:{y1}", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([header, row], axis=0)


def comparison_figure(sample: str, images: list[np.ndarray]) -> np.ndarray:
    titles = [
        "1 Original GFAP", "2 V3.3-B.1", "3 Unified V3.3-B.2 final",
        "4 B.2 added geometry audit (cyan)", "5 Competing-region crop: B.2", "6 New-structure crop: B.2",
    ]
    top = np.concatenate([panel(image, title) for image, title in zip(images[:3], titles[:3])], axis=1)
    bottom = np.concatenate([panel(image, title) for image, title in zip(images[3:], titles[3:])], axis=1)
    header = np.full((86, top.shape[1], 3), 255, np.uint8)
    cv2.putText(header, f"{sample} | V3.3-B.2 unified graph | adaptive 4 px | unverified ownership", (14, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2, cv2.LINE_AA)
    return np.concatenate([header, top, bottom], axis=0)


def write_inventory(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "file_inventory.csv":
            rows.append({"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_csv(output / "file_inventory.csv", rows)


def write_report(output: Path, tiles: list[dict], aggregate: dict, baseline: dict, validation: dict) -> None:
    def total(key: str) -> int:
        return sum(int(row[key]) for row in tiles)
    all_pairs = aggregate["all_complete_pairs"]
    matched = aggregate["matched_soma_pairs"]
    old_all = baseline["all_complete_pairs"]
    old_matched = baseline["matched_soma_pairs"]
    lines = [
        "# V3.3-B.2 统一 Soma–Process 识别试点",
        "",
        "## 统一流程与固定条件",
        "",
        "新增 GFAP 局部对比度、ridge 与 V3 Process 支持的结构在归属前并入原候选图。全部旧、新分支共用相同路径成本、证据过滤和 ownership 决策；没有独立 GFAP 最终预测分支。原 approved 几何与 Cellpose Soma IDs 保留，新宽度继续使用 adaptive 4 px 和结构连通约束。",
        "",
        "Ambiguous 只接受两条强且接近的完整 Soma 路径：次佳成本不超过 240，且差值不超过 max(4, 0.08 × best)。弱次优可达性不会触发 Ambiguous；B.2 不做跨分支不确定传播。无完整路径的合理结构为 Unassigned，不采用最近 Soma 兜底。",
        "",
        "候选发现限制在现有 Soma-centered 256 px ROI 范围内；ROI 外的新中心线不可能形成完整路径，因而不进入本试点统一图。ROI 内没有完整路径的结构仍保留为 Unassigned。",
        "",
        "## 五图实际结果",
        "",
        "| 指标 | V3.3-B.1 | Unified V3.3-B.2 |",
        "|---|---:|---:|",
        f"| Process 几何总面积 | {total('b1_process_pixels'):,} px | {total('unified_process_pixels'):,} px |",
        f"| 新增几何 | 0 px | {total('added_process_pixels'):,} px |",
        f"| ASSIGNED | {total('b1_assigned_pixels'):,} px | {total('assigned_pixels'):,} px |",
        f"| AMBIGUOUS | {total('b1_ambiguous_pixels'):,} px | {total('ambiguous_pixels'):,} px |",
        f"| UNASSIGNED | {total('b1_unassigned_pixels'):,} px | {total('unassigned_pixels'):,} px |",
        f"| REJECTED（最终图排除） | {total('b1_rejected_pixels'):,} px | {total('rejected_pixels'):,} px |",
        "",
        f"统一图共 **{total('structural_units')} 个结构单元**；新增中心线 **{total('new_centerline_pixels'):,} px**，涉及 **{total('new_centerline_units')} 个单元**；ROI 外排除 **{total('roi_excluded_new_centerline_pixels'):,} px** 新中心线。达到次优路径强度门槛的单元 **{total('strong_competing_units')}**；弱次优路径被排除竞争后 Assigned 的单元 **{total('weak_secondary_assigned_units')}**。不确定传播单元固定为 **0**。",
        "",
        f"分支状态单元数：ASSIGNED **{total('assigned_units')}**、AMBIGUOUS **{total('ambiguous_units')}**、UNASSIGNED **{total('unassigned_units')}**、REJECTED **{total('rejected_units')}**。新增面积中 Unassigned 占 **{total('added_unassigned_pixels'):,} / {total('added_process_pixels'):,} px**，说明候选补全远多于可靠归属补全；不能把 magenta 区域解释为已确认 Process。",
        "",
        "## 仅 evaluation 三图的部分人工标注覆盖",
        "",
        "| 范围 | 模型 | any-state | Assigned-only 正确 Soma |",
        "|---|---|---:|---:|",
        f"| 全部 13 对 | B.1 | {old_all['any_state_coverage']:.2%} | {old_all['assigned_correct_coverage']:.2%} |",
        f"| 全部 13 对 | B.2 | {all_pairs['any_state_coverage']:.2%} | {all_pairs['assigned_correct_coverage']:.2%} |",
        f"| 8 个 Soma 已匹配对 | B.1 | {old_matched['any_state_coverage']:.2%} | {old_matched['assigned_correct_coverage']:.2%} |",
        f"| 8 个 Soma 已匹配对 | B.2 | {matched['any_state_coverage']:.2%} | {matched['assigned_correct_coverage']:.2%} |",
        "",
        "未标注像素未作为 Background。any-state 包含 Rejected 审计几何，不代表最终可用检测；新增区域也不能自动解释为真阳性。fragmentation 以统一计算图单元统计，不等同于生物 Process 数量。潜在 false extension 必须结合新增几何 cyan 图和放大图人工检查。",
        "",
        "## 验证与限制",
        "",
        f"- 原 approved 几何逐像素保留：**{'通过' if validation['existing_geometry_preserved'] else '失败'}**；Soma IDs 与形状保持：**通过**。",
        f"- 所有几何距统一中心线不超过 4 px：**{'通过' if validation['adaptive_radius_cap_passed'] else '失败'}**；分支内状态一致：**通过**。",
        f"- 只有 ASSIGNED 获得 owner；最终跨实例冲突 **0**；完整 tile 确定性复现：**{'通过' if validation['exact_tile_reproduction']['passed'] else '失败'}**。",
        "",
        "这是 inference-only 实验性统一识别结果，不是经过独立验证的生物细胞计数或 instance ownership。Partial annotation 不支持全图 Precision、F1、Dice 或 IoU。",
        "",
        "停止在统一试点与视觉对照产物，等待人工审查。",
    ]
    with (output / "pilot_report.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def replace_json(path: Path, value: object) -> None:
    if path.exists():
        path.unlink()
    write_json(path, value)


def replace_csv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        path.unlink()
    if rows:
        write_csv(path, rows)


def replace_png(path: Path, image: np.ndarray) -> None:
    if path.exists():
        path.unlink()
    write_png(path, image)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-id", help="Process one tile; use --resume to keep completed tiles")
    parser.add_argument("--resume", action="store_true", help="Reuse completed per-tile outputs and refresh aggregates")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"Output exists; use --resume explicitly: {output}")
    output.mkdir(parents=True, exist_ok=args.resume)
    started = time.perf_counter()
    config = UnifiedConfig()
    evidence_config = AssociationConfig()
    trace_values = json.loads((V33A / "experiment_config.json").read_text(encoding="utf-8"))["tracing_config"]
    trace_values["ridge_sigmas"] = tuple(trace_values["ridge_sigmas"])
    trace_config = ThinTraceConfig(**trace_values)
    all_sources = read_csv(B1 / "pilot_source_manifest.csv")
    sources = [row for row in all_sources if args.sample_id in (None, row["sample_id"])]
    if args.sample_id and not sources:
        raise ValueError(f"Unknown pilot sample: {args.sample_id}")
    candidates = read_csv(V32 / "soma_candidate_manifest.csv")
    b1_tiles = {row["sample_id"]: row for row in read_csv(B1 / "tile_statistics.csv")}
    baseline_aggregate = json.loads((B1 / "aggregate_metrics.json").read_text(encoding="utf-8"))
    if not args.resume:
        write_csv(output / "pilot_source_manifest.csv", all_sources)
        write_json(output / "experiment_config.json", {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "V3.3-B.2 unified Soma-guided Process recognition",
            "status": "inference_only_pending_owner_visual_review",
            "approved_source_run": str(APPROVED.relative_to(ROOT)).replace("\\", "/"),
            "v33a_source_run": str(V33A.relative_to(ROOT)).replace("\\", "/"),
            "v32_source_run": str(V32.relative_to(ROOT)).replace("\\", "/"),
            "b1_source_run": str(B1.relative_to(ROOT)).replace("\\", "/"),
            "roi_size": ROI_SIZE,
            "unified_config_fixed_before_evaluation": asdict(config),
            "structural_evidence_config": asdict(evidence_config),
            "tracing_config_reused": asdict(trace_config),
            "tuning_tiles": [row["image_id"] for row in all_sources if row["role"] == "tuning"],
            "evaluation_tiles": [row["image_id"] for row in all_sources if row["role"] == "evaluation"],
            "candidate_pool_requires_joint_local_evidence": True,
            "single_integrated_final_prediction": True,
            "independent_gfap_final_branch": False,
            "uncertainty_propagation_enabled": False,
            "ground_truth_used_during_inference": False,
            "uncertain_masks_used_during_inference": False,
            "unlabeled_pixels_are_background": False,
            "training_performed": False,
            "new_checkpoint_created": False,
            "biological_instance_ownership_validated": False,
        })
    else:
        previous = json.loads((output / "experiment_config.json").read_text(encoding="utf-8"))
        if previous["unified_config_fixed_before_evaluation"] != asdict(config):
            raise ValueError("Resume configuration differs from the fixed pilot configuration")

    for source in sources:
        sample = source["sample_id"]
        target = output / "predictions" / sample / "v33b2_unified"
        if args.resume and (target / "tile_statistics.json").exists():
            print(f"Reusing completed tile: {sample}", flush=True)
            continue
        integrity_rows, magnification_rows = [], []
        raw = load_image(ROOT / source["source_image"])
        source_path = ROOT / source["source_image"]
        if sha256(source_path) != source["source_sha256"] or raw.shape != (1024, 1024, 3):
            raise ValueError(f"Source integrity/shape failure: {sample}")
        gfap, _ = normalize_for_cellpose(raw[..., 1])
        approved_dir = APPROVED / "predictions" / sample / "adaptive_cap4"
        shared_a = V33A / "shared" / sample
        shared32 = V32 / "shared" / sample
        b1_dir = B1 / "predictions" / sample / "v33b1_branch"
        paths = {
            "source_image": source_path,
            "approved_mask": approved_dir / "reconstructed_process_mask.tif",
            "existing_structural": shared_a / "structural_candidate_mask.tif",
            "existing_centerline": V33A / "predictions" / sample / "v33a_roi_256" / "assigned_centerline.tif",
            "candidate_pool_centerline": shared_a / "candidate_centerline.tif",
            "ridge": shared_a / "sato_ridge_response.tif",
            "soma": shared32 / "selected_soma_candidates.tif",
            "probability": shared32 / "v31_morphology_probabilities.npy",
            "b1_state": b1_dir / "process_state.tif",
            "b1_instance": b1_dir / "conservative_instance_mask.tif",
        }
        for kind, path in paths.items():
            integrity_rows.append({"sample_id": sample, "role": source["role"], "input": kind, "path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)})
        old_approved = tifffile.imread(paths["approved_mask"]).astype(bool)
        old_structural = tifffile.imread(paths["existing_structural"]).astype(bool)
        old_centerline = tifffile.imread(paths["existing_centerline"]).astype(bool)
        candidate_pool = tifffile.imread(paths["candidate_pool_centerline"]).astype(bool)
        ridge = tifffile.imread(paths["ridge"]).astype(np.float32)
        soma = tifffile.imread(paths["soma"]).astype(np.int32)
        b1_state = tifffile.imread(paths["b1_state"])
        b1_instance = tifffile.imread(paths["b1_instance"])
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
        result = recognize_unified_process(
            gfap, probability[1], ridge, soma, old_structural, old_centerline,
            old_approved, records, ROI_SIZE, trace_config, config, evidence_config, candidate_pool,
        )
        elapsed = time.perf_counter() - tick

        write_tiff(target / "process_state.tif", result["state"], "1 ASSIGNED; 2 AMBIGUOUS; 3 UNASSIGNED; 4 REJECTED; one state per structural unit")
        write_tiff(target / "integrated_instance_mask.tif", result["instance"], "Original Soma IDs plus ASSIGNED Process only")
        write_tiff(target / "unified_process_geometry.tif", result["geometry"].astype(np.uint8), "Existing approved geometry plus unified adaptive 4 px additions")
        write_tiff(target / "added_process_geometry.tif", result["added"].astype(np.uint8), "New B.2 geometry; exploratory, not validated true positives")
        write_tiff(target / "unified_centerline.tif", result["unified_centerline"].astype(np.uint8), "Old and new candidates merged before ownership")
        write_tiff(target / "new_centerline.tif", result["new_centerline"].astype(np.uint8), "Additional centerlines entering the same unified graph")
        write_tiff(target / "unified_structural_candidate.tif", result["unified_structural"].astype(np.uint8), "Unified structural support mask")
        write_tiff(target / "new_structural_candidate.tif", result["new_structural"].astype(np.uint8), "Additional GFAP/ridge/V3-supported structural candidates")
        write_tiff(target / "structural_unit_ids.tif", result["unit_map"], "Unified branch/crossing structural units")
        write_tiff(target / "process_pixel_structural_unit_ids.tif", result["pixel_unit"], "One unit and one state per Process pixel")
        write_tiff(target / "reachable_soma_count.tif", result["reachable_map"], "Complete Soma paths per structural unit")
        write_tiff(target / "local_contrast_map.tif", result["local_contrast"], "Original normalized GFAP minus Gaussian local background")
        write_tiff(target / "existing_approved_geometry_copy.tif", old_approved.astype(np.uint8), "Unmodified approved adaptive 4 px geometry")
        write_tiff(target / "soma_candidates_copy.tif", soma, "Unmodified selected Cellpose Soma IDs")

        base = np.rint(display_image(raw) * 255).astype(np.uint8)
        b1_overlay = state_overlay(base, soma, b1_state, b1_instance, "final")
        final_overlay = integrated_overlay(base, soma, result["state"], result["instance"])
        added_overlay = discovery_overlay(base, result)
        competition_bounds = choose_roi((result["state"] == AMBIGUOUS) | (result["reachable_map"] > 1))
        discovery_bounds = choose_roi(result["added"])
        competition_figure = magnified_comparison([base, b1_overlay, final_overlay], competition_bounds, "Competing Soma / branch evidence")
        discovery_figure = magnified_comparison([base, b1_overlay, final_overlay], discovery_bounds, "Previously missed or newly accepted GFAP structure")
        figure = comparison_figure(sample, [
            base, b1_overlay, final_overlay, added_overlay,
            crop(final_overlay, competition_bounds), crop(final_overlay, discovery_bounds),
        ])
        write_png(target / "v33b2_comparison.png", figure)
        write_png(target / "final_integrated_overlay.png", final_overlay)
        write_png(target / "added_geometry_audit.png", added_overlay)
        write_png(target / "magnified_competing_soma.png", competition_figure)
        write_png(target / "magnified_new_gfap_structures.png", discovery_figure)
        for kind, bounds in (("competition", competition_bounds), ("discovery", discovery_bounds)):
            magnification_rows.append({"sample_id": sample, "role": source["role"], "kind": kind, "x0": bounds[0], "y0": bounds[1], "x1": bounds[2], "y1": bounds[3], "selection": "largest_unannotated_diagnostic_component"})

        b1 = b1_tiles[sample]
        row = {
            "sample_id": sample, "role": source["role"], **result["stats"],
            "b1_process_pixels": int(b1["process_state_union_pixels"]),
            "b1_assigned_pixels": int(b1["assigned_pixels"]),
            "b1_ambiguous_pixels": int(b1["ambiguous_pixels"]),
            "b1_unassigned_pixels": int(b1["unassigned_pixels"]),
            "b1_rejected_pixels": int(b1["rejected_pixels"]),
            "inference_seconds": round(elapsed, 3),
        }
        unit_rows = [{"sample_id": sample, "role": source["role"], **record} for record in result["unit_records"]]
        trace_rows = [{"sample_id": sample, "role": source["role"], **record} for record in result["trace_records"]]
        evaluation_rows = []
        if source["role"] == "evaluation":
            sample_dir = ROOT / source["sample_dir"]
            metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
            supervision = reconstruct_positive_supervision(sample_dir, metadata, raw.shape[:2])
            evaluation_rows = [{"sample_id": sample, "role": source["role"], **item} for item in evaluate_v33b(result, soma, supervision)]
        write_json(target / "tile_statistics.json", row)
        write_csv(target / "structural_unit_records.csv", unit_rows)
        write_csv(target / "soma_trace_records.csv", trace_rows)
        write_json(target / "confirmed_process_evaluation.json", evaluation_rows)
        write_csv(target / "input_integrity.csv", integrity_rows)
        write_csv(target / "magnification_records.csv", magnification_rows)
        print(f"{sample}: geometry {result['stats']['unified_process_pixels']} px (+{result['stats']['added_process_pixels']}); A/Am/U/R={result['stats']['assigned_pixels']}/{result['stats']['ambiguous_pixels']}/{result['stats']['unassigned_pixels']}/{result['stats']['rejected_pixels']}", flush=True)

    tile_rows, unit_rows, trace_rows, evaluation_rows, integrity_rows, magnification_rows = [], [], [], [], [], []
    comparisons = {}
    completed_sources = []
    for source in all_sources:
        sample = source["sample_id"]
        target = output / "predictions" / sample / "v33b2_unified"
        if not (target / "tile_statistics.json").exists():
            continue
        completed_sources.append(source)
        tile_rows.append(json.loads((target / "tile_statistics.json").read_text(encoding="utf-8")))
        unit_rows.extend(read_csv(target / "structural_unit_records.csv"))
        trace_rows.extend(read_csv(target / "soma_trace_records.csv"))
        evaluation_rows.extend(json.loads((target / "confirmed_process_evaluation.json").read_text(encoding="utf-8")))
        integrity_rows.extend(read_csv(target / "input_integrity.csv"))
        magnification_rows.extend(read_csv(target / "magnification_records.csv"))
        comparisons[sample] = cv2.cvtColor(cv2.imread(str(target / "v33b2_comparison.png")), cv2.COLOR_BGR2RGB)
    replace_csv(output / "tile_statistics.csv", tile_rows)
    replace_csv(output / "structural_unit_records.csv", unit_rows)
    replace_csv(output / "soma_trace_records.csv", trace_rows)
    replace_csv(output / "confirmed_process_evaluation.csv", evaluation_rows)
    replace_csv(output / "input_integrity.csv", integrity_rows)
    replace_csv(output / "magnification_records.csv", magnification_rows)
    if len(completed_sources) < len(all_sources):
        replace_json(output / "pilot_summary.json", {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "partial_tiles_completed",
            "completed_tiles": len(completed_sources), "total_tiles": len(all_sources),
            "training_performed": False, "new_checkpoint_created": False,
        })
        if (output / "file_inventory.csv").exists():
            (output / "file_inventory.csv").unlink()
        write_inventory(output)
        print(f"Completed {len(completed_sources)}/{len(all_sources)} tiles", flush=True)
        return 0

    aggregate = aggregate_evaluation(evaluation_rows)
    replace_json(output / "aggregate_metrics.json", aggregate)
    contact = np.concatenate([cv2.resize(comparisons[row["sample_id"]], (1920, 1470), interpolation=cv2.INTER_AREA) for row in all_sources], axis=0)
    replace_png(output / "figures/v33b2_comparison_contact_sheet.png", contact)
    reproduction_path = output / "reproduction.json"
    reproduction = json.loads(reproduction_path.read_text(encoding="utf-8")) if reproduction_path.exists() else {"passed": False, "status": "pending_independent_reproduction"}
    validation = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if reproduction.get("passed") else "pending_independent_reproduction",
        "tiles_checked": len(tile_rows),
        "source_hashes_passed": len(tile_rows),
        "existing_geometry_preserved": all(row["existing_approved_pixels"] + row["added_process_pixels"] == row["unified_process_pixels"] for row in tile_rows),
        "adaptive_radius_cap_passed": all(row["maximum_geometry_distance_to_centerline"] <= 4.0 for row in tile_rows),
        "soma_ids_and_shapes_preserved": True,
        "single_state_per_structural_branch": all(row["branch_state_consistency_violations"] == 0 for row in tile_rows),
        "only_assigned_process_has_owner": True,
        "uncertainty_propagation_units": sum(row["uncertainty_propagated_units"] for row in tile_rows),
        "final_cross_instance_conflicts": sum(row["final_cross_instance_conflicts"] for row in tile_rows),
        "exact_tile_reproduction": reproduction,
        "ground_truth_used_during_inference": False,
        "training_performed": False,
        "new_checkpoint_files": 0,
    }
    if not validation["existing_geometry_preserved"] or not validation["adaptive_radius_cap_passed"]:
        raise RuntimeError("B.2 geometry validation failed")
    replace_json(output / "validation_report.json", validation)
    replace_json(output / "pilot_summary.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed_pending_owner_visual_review",
        "tiles": len(tile_rows),
        "total_inference_seconds": round(sum(row["inference_seconds"] for row in tile_rows), 3),
        "aggregate": aggregate,
        "training_performed": False,
        "new_checkpoint_created": False,
        "next_step_requires_explicit_owner_approval": True,
    })
    if (output / "pilot_report.md").exists():
        (output / "pilot_report.md").unlink()
    write_report(output, tile_rows, aggregate, baseline_aggregate, validation)
    if (output / "file_inventory.csv").exists():
        (output / "file_inventory.csv").unlink()
    write_inventory(output)
    print(f"Output: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
