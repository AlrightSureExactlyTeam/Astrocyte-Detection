# Experiment Log — Astrocyte Detection

## Experiment 001: BBBC042 Dataset Integrity Check

**Date:** 2026-10-20
**Status:** Passed (filename-level checks only)

| Check | Result |
| --- | ---: |
| Total images | 1117 |
| Total annotation files | 1117 |
| Duplicate image filenames (stems) | 0 |
| Duplicate annotation filenames (stems) | 0 |
| Images without matching annotations | 0 |
| Annotations without matching images | 0 |
| Script exit code | 0 |

All 1117 microscopy images had matching annotation files, with no duplicate filenames or missing matches. This confirms filename-level dataset integrity, but does not validate annotation coordinates, image readability, or biological correctness.

---

## Experiment 002: Initial Evaluation and Cellpose Baseline Unit Tests

**Date:** 2026-10-27
**Environment:** Conda `astrocyte`
**Status:** Passed — 11/11 tests

| Test module | Passed | Failed / Errors | Result |
| --- | ---: | ---: | --- |
| `test_evaluate.py` | 6 | 0 | OK |
| `test_baseline_cellpose.py` | 5 | 0 | OK |
| **Total** | **11** | **0** | **Passed** |

The initial tests covered point matching, detection and counting metrics, report export, Cellpose center extraction, result saving, and `uint32` TIFF output (`test_evaluate.py`: 6; `test_baseline_cellpose.py`: 5). These tests used synthetic inputs and mocked Cellpose inference, so they did not measure real-image accuracy.

---

## Experiment 002.v2: Updated Evaluation and Cellpose Baseline Unit Tests

**Date:** 2026-10-28
**Environment:** Conda `astrocyte`
**Status:** Passed — 25/25 tests

| Test module | Passed | Failed / Errors |
| --- | ---: | ---: |
| `test_evaluate.py` | 16 | 0 |
| `test_baseline_cellpose.py` | 9 | 0 |
| **Total** | **25** | **0** |

The updated tests cover current Cellpose parameter forwarding and output validation, exact center coordinates, CSV/overlay/TIFF output, point-matching objectives and thresholds, `pixel_size_um`, metric validation, and JSON/CSV report contents. All 25 tests passed, but Cellpose inference remained mocked; real-model inference and real-image accuracy still require separate validation.

---

## Experiment 003: train.py / preprocessing.py Rebuild and End-to-End Verification

**Date:** 2026-10-28
**Environment:** Conda `astrocyte` (Python 3.12, torch 2.14.1+cpu, cellpose 4.2.1.1), CPU only
**Status:** Passed

**Background:** `src/train.py` had been overwritten by a copy of preprocessing code, and `src/preprocessing.py` contained a full duplicated copy nested inside `iter_qptiff_tiles`. Both files were rebuilt; preprocessing TODOs (overlap tiling, stable tile IDs, channel/series/level selection) and the full training pipeline were implemented.

| Check | Result |
| --- | ---: |
| preprocessing functional checks (loading, dtype/range contract, channel selection, tiled iteration with overlap, tile IDs, error handling) | 27/27 passed |
| `train.py --self-test` (synthetic data, 3 epochs, CPU) | train loss 2.1739 -> 1.8486; checkpoint save/reload OK |
| Existing unit tests (tool/) | 25/25 passed |
| Real Cellpose (cpsam) inference, 256x256 synthetic TIFF | 7 instances, ran on CPU |
| Real Cellpose (cpsam) inference, 512x512 synthetic TIFF | 18 instances, ran on CPU |
| Evaluation of Cellpose output vs known synthetic ground truth (match radius 8 px) | P/R/F1 = 1.0 on both crops, count MAE = 0 |

**Notes:**

- Fixes applied while rebuilding `preprocessing.py`: `prepare_image` now returns true float32 (percentile scaling silently produced float64); `load_image` validates the extension before file existence and transposes planar (C,H,W) TIFFs to HWC so `--channel` works on multi-channel fluorescence TIFFs; `iter_qptiff_tiles` accepts one non-spatial axis (C/Q/S) as the channel axis.
- `train.py` implements config loading, seeding, the composite loss (3-class semantic CE + center BCE-with-logits, masked by `valid_mask`), train/validation epochs, early stopping, and best/last checkpoints under `checkpoints/`. Real-data training still requires `dataset.build_dataloaders()` (dataset.py stub, owner Zhang).
- Synthetic test images (`datasets/test_images/`, single-channel uint16 TIFF, 1024x1024 original + 256/512 crops) follow the team convention: TIFF only, no JPG/screenshots. Cellpose results on synthetic blobs only prove the pipeline runs end to end; accuracy on real BBBC042/FFPE images still requires real data.
- Cellpose cpsam weights (1.15 GB) were downloaded on first run and are cached locally.

---

## Experiment 003.v2: Code-Review Fixes and Re-Verification

**Date:** 2026-10-28
**Environment:** Conda `astrocyte` (Python 3.12, torch 2.14.1+cpu, cellpose 4.2.1.1), CPU only
**Status:** Passed

**Background:** An external review of the Experiment 003 rebuild flagged 15 points. Eight were confirmed and fixed; five were rejected as convention/scope differences (documented RGB->gray behavior, valid_mask instead of 255-ignore labels, torch>=2.0 baseline, per-pass tile_id scope, CPU-only target); two low-priority items deferred.

| Fix | Change | Verification |
| --- | --- | --- |
| A1/A5 | `load_image` now decides TIFF axis layout from `series.axes` metadata: Z/T stacks rejected, (1,H,W) squeezed, planar (C,H,W) transposed; the `2<=shape[0]<=8` shape heuristic was removed | Z-stack raises ValueError; (1,H,W) prepares correctly |
| A2 | `prepare_image` accepts explicit `channel=` for ANY channel count (e.g. 2-channel GFAP+DAPI); without `channel=`, only 1/3/4 channels are handled | 2/5-channel selection verified plane-exact |
| A4 | Sparse images (<0.2% foreground) fall back to image max when the 99.8th percentile equals background | 2 bright pixels in 100x100 keep signal; constant images still map to 0 |
| B1 | `train()` uses the same package/script import branch as `model`; `python -m src.train` now reaches the friendly "dataset stub" message | verified both entry modes |
| B2-note | NOT changed: 255-ignore rejected — repo contract marks unlabeled pixels via `valid_mask`, labels are 0/1/2 only | n/a |
| B3 | `--self-test --epochs 1` rejected with a clear ValueError | verified |
| B5 | `build_model` rejects a stale `model.output_channels` (config updated 1 -> 4: 3 semantic + 1 center) | mismatch raises ValueError |
| B6 | epoch metrics weighted by batch size | code review |
| B7 | cuDNN deterministic flags set when CUDA exists (no-op on CPU) | code review |
| extra | history.json written after every epoch; self-test checkpoint moved to a temp dir (source tree stays clean) | verified |

| Re-verification | Result |
| --- | ---: |
| preprocessing checks (27 original + 8 regression) | 35/35 passed |
| `train.py --self-test` (3 epochs, CPU) | train loss 2.1739 -> 1.8486; checkpoint round-trip OK |
| Existing unit tests (tool/) | 25/25 passed |
| Real Cellpose re-run on 256/512 crops | identical detections (7 / 18) |
| Evaluation vs synthetic ground truth (8 px) | P/R/F1 = 1.0, count MAE = 0 on both crops |

---

## Experiment 003.v3: Review Round 2 — Cross-File Consistency Fixes

**Date:** 2026-10-28
**Environment:** Conda `astrocyte` (Python 3.12, torch 2.14.1+cpu, cellpose 4.2.1.1), CPU only
**Status:** Passed

| Fix | Change | Verification |
| --- | --- | --- |
| check_dataset extensions | `IMAGE_EXTENSIONS` reduced to {.tif, .tiff, .png} to match `load_image`; unsupported files (e.g. stray .jpg) are now listed under `unsupported_files_ignored` instead of passing silently | synthetic tree: .jpg ignored AND reported; counts correct |
| baseline multi-channel guard | `baseline_cellpose.py` CLI now refuses multi-channel input without `--channel` (enforcement point for "no silent RGB mixing" is the caller, not preprocessing; `prepare_image`'s documented RGB->gray behavior is unchanged) | 3-channel TIFF without `--channel` exits with a clear message; with `--channel 0` it runs (7 instances, matching the single-channel crop) |
| QYX ambiguity documented | `_load_tiff` docstring states the known limitation: a 3-plane stack without axes metadata is indistinguishable from planar 3-channel and will be treated as (H,W,3); exporters should write axes metadata. No `--assume-zstack` switch added (no caller could plumb it through; documentation is the proportionate fix) | docstring review |

| Re-verification | Result |
| --- | ---: |
| preprocessing checks | 35/35 passed |
| Existing unit tests (tool/) | 25/25 passed |

---

## Experiment 004.v1: Complete Tool Unit-Test Suite

**Date:** 2026-10-28
**Environment:** Conda `astrocyte`
**Command:** `python -m unittest discover -s tool -p "test_*.py" -v`
**Status:** Passed — 49/49 tests in 2.355 seconds

| Test module | Passed | Failed / Errors | Result |
| --- | ---: | ---: | --- |
| `test_baseline_cellpose.py` | 9 | 0 | OK |
| `test_check_dataset.py` | 2 | 0 | OK |
| `test_dataset.py` | 8 | 0 | OK |
| `test_evaluate.py` | 16 | 0 | OK |
| `test_postprocessing.py` | 7 | 0 | OK |
| `test_predict.py` | 5 | 0 | OK |
| `test_train.py` | 2 | 0 | OK |
| **Total** | **49** | **0** | **Passed** |

### Full Test Results

| Test module | Test case | Result |
| --- | --- | --- |
| `test_baseline_cellpose.py` | `test_main_forwards_cli_parameters_and_saves_uint32_mask` | OK |
| `test_baseline_cellpose.py` | `test_mask_centers_accepts_empty_label_map` | OK |
| `test_baseline_cellpose.py` | `test_mask_centers_rejects_malformed_labels` | OK |
| `test_baseline_cellpose.py` | `test_mask_centers_returns_exact_id_xy_rows` | OK |
| `test_baseline_cellpose.py` | `test_run_cellpose_passes_current_parameters` | OK |
| `test_baseline_cellpose.py` | `test_run_cellpose_rejects_invalid_model_output` | OK |
| `test_baseline_cellpose.py` | `test_run_cellpose_selects_requested_channel` | OK |
| `test_baseline_cellpose.py` | `test_save_results_rejects_invalid_centers_before_creating_output` | OK |
| `test_baseline_cellpose.py` | `test_save_results_writes_exact_csv_and_overlay` | OK |
| `test_check_dataset.py` | `test_accepts_in_bounds_centers` | OK |
| `test_check_dataset.py` | `test_rejects_out_of_bounds_centers` | OK |
| `test_dataset.py` | `test_heatmap_has_gaussian_peak_and_handles_empty_points` | OK |
| `test_dataset.py` | `test_reads_bbb042_boxes_as_centers` | OK |
| `test_dataset.py` | `test_reads_xy_csv_and_explicit_empty_annotation` | OK |
| `test_dataset.py` | `test_rejects_malformed_annotation` | OK |
| `test_dataset.py` | `test_build_dataloaders_splits_by_source_image` | OK |
| `test_dataset.py` | `test_point_only_sample_disables_semantic_supervision` | OK |
| `test_dataset.py` | `test_rejects_unpaired_files` | OK |
| `test_dataset.py` | `test_resize_transforms_image_points_and_masks_consistently` | OK |
| `test_evaluate.py` | `test_exports_csv_content` | OK |
| `test_evaluate.py` | `test_exports_json_content_and_creates_parent_directory` | OK |
| `test_evaluate.py` | `test_rejects_invalid_report_arguments` | OK |
| `test_evaluate.py` | `test_distance_threshold_is_inclusive` | OK |
| `test_evaluate.py` | `test_empty_point_sets` | OK |
| `test_evaluate.py` | `test_matches_indices_distances_and_counts` | OK |
| `test_evaluate.py` | `test_maximizes_match_count_before_minimizing_distance` | OK |
| `test_evaluate.py` | `test_minimizes_distance_among_maximum_matchings` | OK |
| `test_evaluate.py` | `test_one_to_one_matching_selects_nearest_prediction` | OK |
| `test_evaluate.py` | `test_pixel_size_converts_threshold_but_reports_pixel_distance` | OK |
| `test_evaluate.py` | `test_rejects_invalid_points_thresholds_and_pixel_size` | OK |
| `test_evaluate.py` | `test_counting_metrics` | OK |
| `test_evaluate.py` | `test_counting_metrics_rejects_invalid_inputs` | OK |
| `test_evaluate.py` | `test_detection_metrics` | OK |
| `test_evaluate.py` | `test_detection_metrics_empty_conventions` | OK |
| `test_evaluate.py` | `test_detection_metrics_rejects_invalid_counts` | OK |
| `test_postprocessing.py` | `test_extract_centers_rejects_invalid_input` | OK |
| `test_postprocessing.py` | `test_extracts_xy_centers_and_empty_result` | OK |
| `test_postprocessing.py` | `test_global_coordinate_conversion` | OK |
| `test_postprocessing.py` | `test_probability_conversion` | OK |
| `test_postprocessing.py` | `test_decodes_semantic_labels_and_uncertainty` | OK |
| `test_postprocessing.py` | `test_separates_two_foreground_instances` | OK |
| `test_postprocessing.py` | `test_unassigned_centers_do_not_create_instances` | OK |
| `test_predict.py` | `test_checkpoint_schema_and_strict_loading` | OK |
| `test_predict.py` | `test_predict_heatmap_uses_correct_activations` | OK |
| `test_predict.py` | `test_save_predictions_accepts_verified_empty_points` | OK |
| `test_predict.py` | `test_single_image_prediction_and_saved_outputs` | OK |
| `test_predict.py` | `test_whole_slide_deduplicates_overlap_and_saves_tiles` | OK |
| `test_train.py` | `test_fully_ignored_batch_has_finite_zero_loss` | OK |
| `test_train.py` | `test_point_only_targets_exclude_semantic_loss` | OK |

All 49 unit tests passed, covering the Cellpose baseline, coordinate validation, dataset loading, evaluation, postprocessing, prediction, and point-only training loss. These tests used synthetic inputs and mocked model inference; they do not establish real-data training convergence or real-image detection accuracy.

---

## Experiment 005.v1: Comprehensive Robustness and Interface Regression Tests

**Date:** 2026-10-28
**Environment:** Conda `astrocyte`
**Command:** `python -m unittest discover -s tool -p "test_*.py" -v`
**Status:** Passed — 74/74 tests in 0.971 seconds

| Test module | Passed | Failed / Errors | Result |
| --- | ---: | ---: | --- |
| `test_baseline_cellpose.py` | 11 | 0 | OK |
| `test_check_dataset.py` | 2 | 0 | OK |
| `test_dataset.py` | 12 | 0 | OK |
| `test_evaluate.py` | 18 | 0 | OK |
| `test_model.py` | 2 | 0 | OK |
| `test_postprocessing.py` | 7 | 0 | OK |
| `test_predict.py` | 10 | 0 | OK |
| `test_preprocessing.py` | 6 | 0 | OK |
| `test_train.py` | 6 | 0 | OK |
| **Total** | **74** | **0** | **Passed** |

### Full Test Results

| Test module | Test case | Result |
| --- | --- | --- |
| `test_baseline_cellpose.py` | `test_main_forwards_cli_parameters_and_saves_uint32_mask` | OK |
| `test_baseline_cellpose.py` | `test_mask_centers_accepts_empty_label_map` | OK |
| `test_baseline_cellpose.py` | `test_mask_centers_preserves_large_uint32_ids` | OK |
| `test_baseline_cellpose.py` | `test_mask_centers_rejects_malformed_labels` | OK |
| `test_baseline_cellpose.py` | `test_mask_centers_returns_exact_id_xy_rows` | OK |
| `test_baseline_cellpose.py` | `test_rejects_invalid_cellpose_parameters` | OK |
| `test_baseline_cellpose.py` | `test_run_cellpose_passes_current_parameters` | OK |
| `test_baseline_cellpose.py` | `test_run_cellpose_rejects_invalid_model_output` | OK |
| `test_baseline_cellpose.py` | `test_run_cellpose_selects_requested_channel` | OK |
| `test_baseline_cellpose.py` | `test_save_results_rejects_invalid_centers_before_creating_output` | OK |
| `test_baseline_cellpose.py` | `test_save_results_writes_exact_csv_and_overlay` | OK |
| `test_check_dataset.py` | `test_accepts_in_bounds_centers` | OK |
| `test_check_dataset.py` | `test_rejects_out_of_bounds_centers` | OK |
| `test_dataset.py` | `test_heatmap_has_gaussian_peak_and_handles_empty_points` | OK |
| `test_dataset.py` | `test_reads_bbb042_boxes_as_centers` | OK |
| `test_dataset.py` | `test_reads_xy_csv_and_explicit_empty_annotation` | OK |
| `test_dataset.py` | `test_rejects_malformed_annotation` | OK |
| `test_dataset.py` | `test_build_dataloaders_splits_by_source_image` | OK |
| `test_dataset.py` | `test_explicit_group_split_is_reproducible_and_leakage_free` | OK |
| `test_dataset.py` | `test_legacy_transform_is_rejected_when_masks_exist` | OK |
| `test_dataset.py` | `test_mask_aware_geometric_transforms_keep_all_targets_aligned` | OK |
| `test_dataset.py` | `test_point_only_sample_disables_semantic_supervision` | OK |
| `test_dataset.py` | `test_rejects_out_of_bounds_source_annotations` | OK |
| `test_dataset.py` | `test_rejects_unpaired_files` | OK |
| `test_dataset.py` | `test_resize_transforms_image_points_and_masks_consistently` | OK |
| `test_evaluate.py` | `test_exports_csv_content` | OK |
| `test_evaluate.py` | `test_exports_json_content_and_creates_parent_directory` | OK |
| `test_evaluate.py` | `test_rejects_invalid_report_arguments` | OK |
| `test_evaluate.py` | `test_distance_threshold_is_inclusive` | OK |
| `test_evaluate.py` | `test_empty_point_sets` | OK |
| `test_evaluate.py` | `test_matches_indices_distances_and_counts` | OK |
| `test_evaluate.py` | `test_maximizes_match_count_before_minimizing_distance` | OK |
| `test_evaluate.py` | `test_minimizes_distance_among_maximum_matchings` | OK |
| `test_evaluate.py` | `test_one_to_one_matching_selects_nearest_prediction` | OK |
| `test_evaluate.py` | `test_pixel_size_converts_threshold_but_reports_pixel_distance` | OK |
| `test_evaluate.py` | `test_rejects_invalid_points_thresholds_and_pixel_size` | OK |
| `test_evaluate.py` | `test_counting_metrics` | OK |
| `test_evaluate.py` | `test_counting_metrics_rejects_invalid_inputs` | OK |
| `test_evaluate.py` | `test_counting_rmse_avoids_overflow` | OK |
| `test_evaluate.py` | `test_detection_metrics` | OK |
| `test_evaluate.py` | `test_detection_metrics_empty_conventions` | OK |
| `test_evaluate.py` | `test_detection_metrics_rejects_invalid_counts` | OK |
| `test_evaluate.py` | `test_excessive_dense_matching_is_rejected_before_allocation` | OK |
| `test_model.py` | `test_invalid_image_tensor_is_rejected` | OK |
| `test_model.py` | `test_two_head_forward_contract_without_pretrained_weights` | OK |
| `test_postprocessing.py` | `test_extract_centers_rejects_invalid_input` | OK |
| `test_postprocessing.py` | `test_extracts_xy_centers_and_empty_result` | OK |
| `test_postprocessing.py` | `test_global_coordinate_conversion` | OK |
| `test_postprocessing.py` | `test_probability_conversion` | OK |
| `test_postprocessing.py` | `test_decodes_semantic_labels_and_uncertainty` | OK |
| `test_postprocessing.py` | `test_separates_two_foreground_instances` | OK |
| `test_postprocessing.py` | `test_unassigned_centers_do_not_create_instances` | OK |
| `test_predict.py` | `test_background_semantics_do_not_remove_center` | OK |
| `test_predict.py` | `test_checkpoint_schema_and_strict_loading` | OK |
| `test_predict.py` | `test_global_dedup_uses_overlap_provenance_and_deterministic_representatives` | OK |
| `test_predict.py` | `test_invalid_prediction_outputs_do_not_create_final_files` | OK |
| `test_predict.py` | `test_predict_heatmap_uses_correct_activations` | OK |
| `test_predict.py` | `test_save_predictions_accepts_verified_empty_points` | OK |
| `test_predict.py` | `test_single_image_prediction_and_saved_outputs` | OK |
| `test_predict.py` | `test_whole_slide_deduplicates_only_exact_within_tile_points` | OK |
| `test_predict.py` | `test_whole_slide_deduplicates_overlap_and_saves_tiles` | OK |
| `test_predict.py` | `test_whole_slide_keeps_raw_centers_when_assignment_fails` | OK |
| `test_preprocessing.py` | `test_corrupted_tiff_is_not_an_empty_image` | OK |
| `test_preprocessing.py` | `test_explicit_planar_channels_and_ambiguous_tiff` | OK |
| `test_preprocessing.py` | `test_grayscale_and_rgb_png` | OK |
| `test_preprocessing.py` | `test_empty_constant_and_single_pixel_images` | OK |
| `test_preprocessing.py` | `test_small_tiled_image_preserves_coordinates_and_edge_shapes` | OK |
| `test_preprocessing.py` | `test_stack_axes_are_not_channels` | OK |
| `test_train.py` | `test_criterion_uses_configured_loss_weights` | OK |
| `test_train.py` | `test_fully_ignored_batch_has_finite_zero_loss` | OK |
| `test_train.py` | `test_invalid_supervision_is_rejected` | OK |
| `test_train.py` | `test_point_only_targets_exclude_semantic_loss` | OK |
| `test_train.py` | `test_shared_valid_mask_excludes_semantic_pixels` | OK |
| `test_train.py` | `test_rejects_fractional_and_nonfinite_configuration` | OK |

All 74 unit tests passed. New coverage includes mask-aware geometric transforms, explicit group-based splitting, strict training configuration, ambiguous TIFF rejection, tiled-image edge shapes, cross-tile center deduplication and provenance, prediction-output prevalidation, large instance IDs, and numerical overflow guards. The run emitted an expected warning for center-only supervision and a `tifffile` deprecation warning; neither was a test failure. Tests used synthetic inputs and mocked external model inference, so this result does not establish real-data training convergence, real QPTIFF performance, or real-image biological accuracy.

## Experiment 006: GFAP Cellpose Pilot

**Date:** 2026-10-29

结论：本次试验结果非常不理想，没有任何细胞突起被成功画出。下一步进行人工描边。
