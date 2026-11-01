"""Arbitrary-size U-Net inference tests with a lightweight deterministic model."""
from __future__ import annotations

import builtins
import io
import json
from dataclasses import replace

import cv2
import numpy as np
import pytest
import tifffile
import torch

from demo_core.images import read_upload, validate_for_model
from demo_core.inference import _run_unet
from demo_core.registry import get_model
from src.preprocessing import prepare_image


class PatchModel:
    """Assign each Patch a known class and retain its actual normalized input."""

    def __init__(self):
        self.call_index = 0
        self.batch_sizes = []
        self.inputs = []

    def __call__(self, batch):
        self.batch_sizes.append(int(batch.shape[0]))
        logits = torch.full((batch.shape[0], 3, 256, 256), -20.0)
        for index in range(batch.shape[0]):
            logits[index, self.call_index % 3] = 20.0
            self.inputs.append(batch[index, 0].cpu().numpy().copy())
            self.call_index += 1
        return {"semantic_logits": logits}


@pytest.mark.parametrize("shape", [(256, 256), (1024, 1024), (1000, 850), (1024, 832)])
def test_unet_arbitrary_shape_padding_alignment_and_outputs(shape, tmp_path, monkeypatch):
    height, width = shape
    rng = np.random.default_rng(height * 10000 + width)
    image = np.zeros((height, width, 2), dtype=np.uint16)
    image[..., 0] = 65535
    image[..., 1] = rng.integers(0, 4096, size=shape, dtype=np.uint16)
    unet = PatchModel()
    checkpoint = tmp_path / "weights" / "checkpoints" / "final.pt"
    model = replace(get_model("unet_v1"), checkpoint=checkpoint)
    monkeypatch.setattr(
        "demo_core.unet_runtime.load_checkpoint",
        lambda run_dir, device: (unet, {"limitations": []}, checkpoint),
    )
    real_import = builtins.__import__

    def reject_annotation_workspace(name, *args, **kwargs):
        if name.split(".")[0] == "annotation_workspace":
            raise AssertionError("Cloud U-Net inference must not import annotation_workspace")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_annotation_workspace)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = _run_unet(image, "arbitrary-size", model, run_dir, "cpu")

    coordinates = [
        (x, y)
        for y in range(0, height, 256)
        for x in range(0, width, 256)
    ]
    assert unet.batch_sizes == [1] * len(coordinates)
    mask = tifffile.imread(result.mask_path)
    assert mask.shape == shape
    for index, (x, y) in enumerate(coordinates):
        valid_height = min(256, height - y)
        valid_width = min(256, width - x)
        patch = image[y:y + valid_height, x:x + valid_width]
        if patch.shape[:2] != (256, 256):
            patch = np.pad(
                patch,
                [(0, 256 - valid_height), (0, 256 - valid_width), (0, 0)],
                mode="edge",
            )
        np.testing.assert_allclose(unet.inputs[index], prepare_image(patch, channel=1)[0].numpy())
        np.testing.assert_array_equal(
            mask[y:y + valid_height, x:x + valid_width],
            np.full((valid_height, valid_width), index % 3, dtype=np.uint8),
        )

    assert cv2.imread(str(result.overlay_path), cv2.IMREAD_UNCHANGED).shape == (*shape, 3)
    assert cv2.imread(str(result.artifacts["process_probability"]), cv2.IMREAD_UNCHANGED).shape == shape
    counts = result.statistics["predicted_pixel_counts"]
    assert sum(counts.values()) == height * width
    assert result.statistics["finite_probabilities"] is True
    metadata = json.loads((run_dir / "input_metadata.json").read_text(encoding="utf-8"))
    assert metadata["input_shape"] == [height, width, 2]
    assert metadata["patch_size"] == 256
    assert metadata["patch_batch_size"] == 1
    assert metadata["edge_padding"] == "edge"
    assert metadata["output_cropped_to_input"] is True


def test_png_and_tiff_uploads_preserve_arbitrary_dimensions():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 4096, size=(1000, 850, 3), dtype=np.uint16)
    success, png = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert success
    tiff = io.BytesIO()
    tifffile.imwrite(tiff, image, photometric="rgb", metadata={"axes": "YXS"})

    for filename, content in (("upload.png", png.tobytes()), ("upload.tiff", tiff.getvalue())):
        decoded, metadata = read_upload(filename, content)
        np.testing.assert_array_equal(decoded, image)
        validate_for_model(decoded, {"min_channels": 2, "patch_size": 256})
        assert metadata["shape"] == [1000, 850, 3]
        assert metadata["resized"] is False
        assert metadata["content_modified"] is False
