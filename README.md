# Astrocyte Detection and Counting

## 1. Project Overview

This project aims to develop an automated astrocyte detection and counting pipeline for fluorescence microscopy images.

We use the BBBC042 dataset for model training and evaluate the model on FFPE tissue images.

## 2. Objectives

- Detect astrocyte cell centers.
- Count astrocytes automatically.
- Support PNG, TIFF, and QPTIFF images.
- Compare our model with Cellpose as a baseline.
- Evaluate model performance on FFPE images.

## 3. Methods

**Main Model:** U-Net with Heatmap Regression (proposed)

**Baseline:** Cellpose (pre-trained)

**Dataset:** BBBC042 and FFPE microscopy images

**Evaluation Metrics:** Precision, Recall, F1-score, and Counting MAE

## 4. Team Members

| Member | Responsibility |
|---|---|
| Liu | Model development, training, evaluation |
| Zhang | Data processing, inference, integration |
| Zhu | Imaging analysis, quality control, annotation review |

## 5. Project Structure

- `configs/`: Configuration files
- `src/`: Model and processing code
- `tools/`: Dataset checking tools
- `docs/`: Project documentation
- `datasets/`: Local datasets (not uploaded)
- `checkpoints/`: Saved model weights (not uploaded)
- `results/`: Experimental outputs

## 6. Project Status

Under development.

## 7. Third-Party Software

This project uses [Cellpose](https://github.com/MouseLand/cellpose) as a baseline. Cellpose is developed by the MouseLand team and distributed under the BSD 3-Clause License.
