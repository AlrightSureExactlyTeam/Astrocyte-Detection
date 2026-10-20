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
