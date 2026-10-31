import io
from dataclasses import replace
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

from demo_core.registry import get_model
from demo_core.results import available_samples, load_cached_result

APP_PATH = Path(__file__).resolve().parents[1] / "demo_app.py"


class FakeUpload:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


def upload_bytes(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_upload_ui_uses_verified_crop_for_live_inference(monkeypatch, tmp_path):
    monkeypatch.setattr(
        st, "file_uploader", lambda *args, **kwargs: FakeUpload("probe.png", upload_bytes(300, 300)),
    )
    cached = load_cached_result(available_samples()[0], get_model("cellpose"))
    fake_result = replace(cached, sample_id="probe_crop256", mode="live")
    calls = []

    def fake_inference(image, sample_id, model, **kwargs):
        calls.append((image.copy(), sample_id, model.id))
        return fake_result, tmp_path

    monkeypatch.setattr("demo_core.inference.run_live_inference", fake_inference)
    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()
    app.radio[0].set_value("Upload PNG/TIFF")
    app.radio[2].set_value("Live inference").run()
    assert not app.exception
    assert any("300×300 → 256×256" in item.value for item in app.success)

    next(item for item in app.button if item.label == "Run live inference").click().run()
    assert not app.exception
    assert calls[0][0].shape == (256, 256, 3)
    assert calls[0][1:] == ("probe_crop256", "cellpose")
    assert app.session_state.demo_image.shape == (256, 256, 3)
    metadata = app.session_state.demo_input_metadata
    assert metadata["original_filename"] == "probe.png"
    assert metadata["content_modified"] is True and metadata["resized"] is False
    assert metadata["crop256"]["verified"] is True
    assert metadata["crop256"]["removed_right"] == 44
    assert metadata["crop256"]["removed_bottom"] == 44


def test_invalid_upload_is_visible_and_cannot_run(monkeypatch):
    monkeypatch.setattr(
        st, "file_uploader", lambda *args, **kwargs: FakeUpload("small.png", upload_bytes(255, 256)),
    )
    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()
    app.radio[0].set_value("Upload PNG/TIFF")
    app.radio[2].set_value("Live inference").run()
    assert not app.exception
    assert any("Upload rejected" in item.value and "至少为 256" in item.value for item in app.error)
    run = next(item for item in app.button if item.label == "Run live inference")
    assert run.disabled
    assert "demo_result" not in app.session_state
