"""Verify packaged cached assets for all five validated GFAP samples."""
from pathlib import Path

from demo_core.registry import PROJECT_ROOT, load_model_registry
from demo_core.results import available_samples, load_cached_result

MISSING_V33B2_SAMPLE = "GBM664_S5_Scan1_x043008_y111616_RGB"


def test_all_packaged_sample_paths_and_cached_model_assets():
    samples = available_samples()
    assert len(samples) == len(set(samples)) == 5
    asset_root = (PROJECT_ROOT / "demo_assets").resolve()
    missing = []

    for sample_id in samples:
        for model in load_model_registry():
            try:
                result = load_cached_result(sample_id, model)
            except (FileNotFoundError, ValueError) as exc:
                missing.append((sample_id, model.id, str(exc)))
                continue
            paths = [result.image_path, result.mask_path, result.overlay_path, *result.artifacts.values()]
            for path in paths:
                assert path is not None and path.is_file(), (sample_id, model.id, path)
                assert Path(path).resolve().is_relative_to(asset_root), path
            assert result.mode == "cached"
            assert result.review_status == "unverified"
            assert result.ground_truth is False

    assert {(sample, model) for sample, model, _ in missing} == {
        (MISSING_V33B2_SAMPLE, "v33b2")
    }, missing
