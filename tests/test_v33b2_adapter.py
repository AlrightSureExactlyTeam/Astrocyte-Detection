"""V3.3-B.2 Demo integration tests; heavy inference is explicitly opt-in."""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile

from demo_core.images import association_state_display, mask_display
from demo_core.inference import run_live_inference
from demo_core.registry import PROJECT_ROOT, get_model
from demo_core.results import available_samples, load_cached_result
from demo_core.v33b2_adapter import validate_v33b2_dependencies
from src.preprocessing import load_image


def test_cached_v33b2_result_and_structured_artifacts():
    sample_id = available_samples()[0]
    result = load_cached_result(sample_id, get_model("v33b2"))
    image = load_image(result.image_path)
    state = tifffile.imread(result.artifacts["process_state"])
    instance = tifffile.imread(result.mask_path)
    soma = tifffile.imread(result.artifacts["soma_candidates"])
    geometry = tifffile.imread(result.artifacts["unified_process_geometry"])
    assert result.review_status == "unverified" and result.ground_truth is False
    assert state.shape == instance.shape == soma.shape == geometry.shape == image.shape[:2]
    assert set(np.unique(state)).issubset({0, 1, 2, 3, 4})
    assert np.array_equal(instance[soma > 0], soma[soma > 0])
    assert not np.any((state > 0) & (state != 1) & (instance > 0))
    assert result.artifacts["structural_unit_records"].is_file()
    assert result.statistics["maximum_geometry_distance_to_centerline"] <= 4.0


def test_b2_instance_and_state_visualization_preserve_arrays():
    result = load_cached_result(available_samples()[0], get_model("v33b2"))
    instance = tifffile.imread(result.mask_path)
    soma = tifffile.imread(result.artifacts["soma_candidates"])
    state = tifffile.imread(result.artifacts["process_state"])
    originals = (instance.copy(), soma.copy(), state.copy())
    assigned = np.where((instance > 0) & (soma == 0), instance, 0)
    soma_display = mask_display(soma, instance=True)
    assigned_display = mask_display(assigned, instance=True)
    state_display = association_state_display(state)
    assert soma_display.shape == assigned_display.shape == state_display.shape == (*soma.shape, 3)
    assert np.array_equal(soma_display, mask_display(soma, instance=True))
    assert len(np.unique(soma_display.reshape(-1, 3), axis=0)) >= 25
    assert np.max(assigned_display) > 0.9
    assert np.all(soma_display[soma == 0] == 0)
    assert np.all(assigned_display[assigned == 0] == 0)
    colors = {label: state_display[state == label][0] for label in (1, 2, 3, 4)}
    assert len({tuple(color) for color in colors.values()}) == 4
    assert np.array_equal(instance, originals[0])
    assert np.array_equal(soma, originals[1])
    assert np.array_equal(state, originals[2])


def test_v33b2_missing_cellpose_checkpoint_is_explicit(tmp_path):
    model = get_model("v33b2")
    model = replace(model, expected_input={
        **model.expected_input,
        "cellpose_checkpoint": str(tmp_path / "missing_cpsam"),
    })
    with pytest.raises(FileNotFoundError, match="Cellpose CPSAM"):
        validate_v33b2_dependencies(model)


@pytest.mark.skipif(
    os.environ.get("ASTROCYTE_RUN_INFERENCE_TESTS") != "1",
    reason="Set ASTROCYTE_RUN_INFERENCE_TESTS=1 for the real one-tile reproduction test",
)
def test_live_v33b2_exactly_reproduces_accepted_pilot(tmp_path):
    sample_id = available_samples()[0]
    model = get_model("v33b2")
    cached = load_cached_result(sample_id, model)
    image = load_image(cached.image_path)
    result, run_dir = run_live_inference(
        image, sample_id, model, output_root=tmp_path, device_name="auto",
    )
    expected_dir = Path(cached.mask_path).parent
    comparisons = {
        "Soma IDs": (result.artifacts["soma_candidates"], expected_dir / "soma_candidates_copy.tif"),
        "complete association state": (result.artifacts["process_state"], expected_dir / "process_state.tif"),
        "adaptive 4 px geometry": (result.artifacts["unified_process_geometry"], expected_dir / "unified_process_geometry.tif"),
        "integrated instances": (result.mask_path, expected_dir / "integrated_instance_mask.tif"),
        "added geometry": (result.artifacts["added_process_geometry"], expected_dir / "added_process_geometry.tif"),
    }
    for name, (actual_path, expected_path) in comparisons.items():
        actual = tifffile.imread(actual_path)
        expected = tifffile.imread(expected_path)
        assert actual.shape == expected.shape == image.shape[:2], name
        assert np.array_equal(actual, expected), f"{name} differs from accepted pilot"
    actual_overlay = cv2.imread(str(result.overlay_path), cv2.IMREAD_UNCHANGED)
    expected_overlay = cv2.imread(str(expected_dir / "final_integrated_overlay.png"), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(actual_overlay, expected_overlay), "Final overlay differs from accepted pilot"
    assert result.statistics["adaptive_reconstruction_radius_cap_px"] == 4
    assert result.statistics["maximum_geometry_distance_to_centerline"] <= 4.0
    assert result.review_status == "unverified" and result.ground_truth is False
    assert result.provenance["training_performed"] is False
    assert result.provenance["external_api_called"] is False
    assert run_dir.parent == tmp_path
