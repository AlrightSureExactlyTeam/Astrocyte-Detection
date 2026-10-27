# Experiment Log — Astrocyte Detection

## Experiment 001: BBBC042 Dataset Integrity Check

**Date:** 2026-10-20
**Dataset:** BBBC042
**Script:** `tool/check_dataset.py`
**Status:** Passed (filename-level checks only)

### 1. Objective

Verify the integrity of the BBBC042 dataset before model training.

Check whether the microscopy images and annotation files can be matched correctly.

### 2. Results

| Check                                  | Result |
| -------------------------------------- | ------ |
| Total images                           | 1117   |
| Total annotation files                 | 1117   |
| Duplicate image filenames (stems)      | 0      |
| Duplicate annotation filenames (stems) | 0      |
| Images without matching annotations    | 0      |
| Annotations without matching images    | 0      |
| Script exit code                       | 0      |

### 3. Observations

All 1117 image files have corresponding annotation files based on their filenames.

No duplicate filenames or missing matches were detected.

The initial dataset integrity check was successful.

### 4. Limitations

This check only verifies file counts and filename matching.

The following have not yet been verified:

- Whether annotation coordinates can be parsed correctly.
- Whether coordinates fall within image boundaries.
- Whether annotation points visually correspond to astrocyte centers.
- Whether the dataset contains unreadable or corrupted images.

### 5. Next Steps

- [ ] Inspect the contents of `positions/1.txt`.
- [ ] Confirm the annotation coordinate format.
- [ ] Visualize annotation points on the corresponding image.
- [ ] Implement coordinate validation in `check_dataset.py`.
- [ ] Implement Gaussian Heatmap generation in `dataset.py`.

### 6. Conclusion

The BBBC042 dataset passed the initial filename-level integrity check and is ready for further annotation validation.


---

## Experiment 002: Evaluation and Cellpose Baseline Unit Tests

**Date:** 2026-10-27  
**Environment:** Conda `astrocyte` (Python 3.11, Windows, PyCharm)  
**Branch:** `LIU`  
**Scripts:** `src/evaluate.py`, `src/baseline_cellpose.py`  
**Tests:** `tool/test_evaluate.py`, `tool/test_baseline_cellpose.py`  
**Status:** Passed (unit tests only)

### 1. Objective

Check the basic correctness of point matching, detection and counting metrics, report export, Cellpose instance-mask processing, and lossless TIFF output before running on real microscopy images.

### 2. Test Commands

```powershell
python -m unittest discover -s tool -p "test_evaluate.py" -v
python -m unittest discover -s tool -p "test_baseline_cellpose.py" -v
```

### 3. Results

| Test module | Passed | Failed / Errors | Result |
| --- | ---: | ---: | --- |
| `test_evaluate.py` | 6 | 0 | OK |
| `test_baseline_cellpose.py` | 5 | 0 | OK |
| **Total** | **11** | **0** | **Passed** |

Evaluation tests covered point matching (including empty inputs and one-to-one matching), precision/recall/F1, MAE/RMSE, and JSON report export.

Cellpose baseline tests covered instance-center extraction, empty and invalid masks, result saving, and lossless `uint32` instance-mask TIFF export. A synthetic test produced two instances and saved `cellpose_instance_mask.tif` to a temporary directory.

### 4. Environment Notes

Cellpose was installed successfully in the `astrocyte` environment. An earlier baseline test could not be imported because `torch` was missing; the dependency issue was resolved before the successful test run.

### 5. Limitations

- Tests use synthetic inputs and mocked Cellpose inference; they do not establish real-image segmentation accuracy.
- GPU inference and CUDA availability were not verified by these test results.
- The full `evaluate_dataset()` workflow has not yet been tested.
- No precision/recall/F1, Dice/IoU, or cell-count accuracy on annotated microscopy images is claimed here.

### 6. Next Steps

- [ ] Run the Cellpose baseline on representative real images.
- [ ] Validate image/annotation coordinates and inspect overlays.
- [ ] Complete and test dataset-level evaluation where required.
- [ ] Compare predictions against genuine ground-truth annotations.

### 7. Conclusion

All 11 current evaluation and Cellpose baseline unit tests passed in the `astrocyte` environment. The tested functions are ready for the next integration and real-data validation steps.
