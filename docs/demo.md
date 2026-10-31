# Astrocyte Detection Demo

## Launch

Use the project Conda environment from the repository root:

```bash
conda activate astrocyte
streamlit run demo_app.py
```

For the Kimi Work preview launcher, `npm run dev -- --host 0.0.0.0 --port 7100` forwards to Streamlit through `demo_launch.py`. No Node packages are required.

The default **Existing results** mode reads five real, matched GFAP predictions and requires no GPU or SciencePal credential. **Live inference** loads only the selected existing Cellpose/U-Net model. Uploaded images must be PNG/TIFF, have at least two channels for the current models, and have dimensions divisible by 256 for U-Net. The demo does not resize uploads or modify inference intensities.

## Outputs

Use **Save this demo run** to create a unique directory under `results/sciencepal_demo/`. Cached runs copy prediction artifacts and record their original run paths; live runs record model/checkpoint provenance. Reports and SciencePal responses can also be downloaded directly. Existing experiment outputs and checkpoints are never overwritten.

For AI analysis, set `SCIENCEPAL_API_KEY` in the environment. No `.env` file is used. SciencePal receives numeric/de-identified result statistics, not original images. Rotate and remove any previously exposed key before sharing or deployment.

## Add a dataset

Add an entry to `configs/demo_datasets.yaml`, or call:

```python
from demo_core.datasets import register_dataset
register_dataset("path/to/dataset_config.yaml")
```

Required metadata: `name`, `image_location`, `annotation_location`, `annotation_type`, `label_mapping`, `source_description`, and `version`. Types are `manual_partial`, `imagej_weak`, and `fully_verified`. Manual partial mapping must be exactly Soma=1, Process=2, Ignore=255. ImageJ labels remain weak foreground and cannot be registered as Soma/Process. Registration validates and records metadata only; it never launches training.

## Add or replace a model

Add a unique entry to `configs/demo_models.yaml` with model name/type, checkpoint path, inference adapter, expected input, output label mapping, version, and optional cached artifact patterns. Current live adapter names are `cellpose` and `unet`. Missing files or unimplemented adapters disable live inference with a reason; cached mode remains independent. Future improved checkpoints can therefore be registered without modifying the UI.

## Interpretation limits

Predictions are unverified. Manual labels are partial and unlabeled pixels are not confirmed background. ImageJ masks are weak GFAP foreground labels, not semantic ground truth. The five comparison images are in-sample and from one tissue section. High U-Net coverage is accompanied by extensive overprediction and does not establish superiority over Cellpose. No full-image Precision, F1, Dice, or IoU is reported against incomplete annotations.

Run focused tests with:

```bash
python -m pytest tests/test_demo.py -q
```
