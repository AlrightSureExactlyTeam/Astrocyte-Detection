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
