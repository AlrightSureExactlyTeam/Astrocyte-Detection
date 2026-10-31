"""Opt-in real local CPU tests for the unchanged U-Net inference pipeline."""
from __future__ import annotations

import json
import os

import pytest
import tifffile

from demo_core.inference import run_live_inference
from demo_core.registry import get_model
from demo_core.results import available_samples, load_cached_result
from src.preprocessing import load_image

pytestmark = pytest.mark.skipif(
    os.environ.get("ASTROCYTE_RUN_UNET_INFERENCE_TESTS") != "1",
    reason="Set ASTROCYTE_RUN_UNET_INFERENCE_TESTS=1 to run real local U-Net CPU inference",
)


@pytest.mark.parametrize("model_id", ["unet_v1", "unet_v2"])
def test_real_unet_cpu_live_inference(model_id, tmp_path):
    model = get_model(model_id)
    cached = load_cached_result(available_samples()[0], model)
    image = load_image(cached.image_path)[:256, :256].copy()

    result, run_dir = run_live_inference(
        image, "local-cpu-test", model, output_root=tmp_path / "runs", device_name="cpu",
    )

    assert result.mode == "live" and result.review_status == "unverified"
    assert result.statistics["finite_probabilities"] is True
    assert tifffile.imread(result.mask_path).shape == (256, 256)
    assert result.overlay_path.is_file()
    assert result.artifacts["process_probability"].is_file()
    metadata = json.loads((run_dir / "input_metadata.json").read_text(encoding="utf-8"))
    assert metadata["device"] == "cpu"
    assert metadata["input_shape"] == [256, 256, image.shape[2]]
