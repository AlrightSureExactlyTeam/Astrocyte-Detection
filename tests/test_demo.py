"""Focused demo integration tests; no training, GPU inference, or real API calls."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
import yaml

from demo_core.datasets import load_dataset_registry, register_dataset, validate_dataset_config
from demo_core.images import mask_display, read_upload, semantic_overlay, validate_for_model
from demo_core.registry import get_model, load_model_registry
from demo_core.reports import report_markdown, save_analysis, save_demo_result
from demo_core.results import aggregate_metrics, available_samples, load_cached_result
from demo_core.sciencepal_client import SciencePalClient, SciencePalError, extract_final_response
from src.preprocessing import load_image


def dataset_config(tmp_path, annotation_type="manual_partial"):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir(exist_ok=True)
    labels.mkdir(exist_ok=True)
    mapping = {"soma": 1, "process": 2, "ignore": 255} if annotation_type == "manual_partial" else {"gfap_foreground": 1}
    return {
        "name": "test dataset", "image_location": str(images), "annotation_location": str(labels),
        "annotation_type": annotation_type, "label_mapping": mapping,
        "source_description": "unit test", "version": "1",
    }


def test_dataset_registration_and_validation(tmp_path):
    config = dataset_config(tmp_path)
    record = register_dataset(config, tmp_path / "registry.json")
    assert record["training_started"] is False
    assert record["registration_only"] is True
    with pytest.raises(ValueError, match="already registered"):
        register_dataset(config, tmp_path / "registry.json")
    invalid = dict(config, label_mapping={"soma": 1, "process": 2, "background": 0, "ignore": 255})
    with pytest.raises(ValueError, match="exactly"):
        validate_dataset_config(invalid)
    weak = dataset_config(tmp_path, "imagej_weak")
    weak["label_mapping"] = {"soma": 1}
    with pytest.raises(ValueError, match="must not"):
        validate_dataset_config(weak)


def test_dataset_and_model_registries_load():
    assert len(load_dataset_registry()) == 2
    specs = load_model_registry()
    assert {spec.id for spec in specs} == {"cellpose", "unet_v1", "unet_v2", "v33b2"}
    # Source-only checkouts intentionally omit weights; unavailable live modes must explain why.
    assert all(spec.live_available or spec.unavailable_reason for spec in specs)


def test_missing_checkpoint_is_explicit(tmp_path):
    source = yaml.safe_load(open("configs/demo_models.yaml", encoding="utf-8"))
    source["models"][0]["checkpoint"] = str(tmp_path / "missing.ckpt")
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    model = load_model_registry(path)[0]
    assert not model.live_available
    assert "not found" in model.unavailable_reason


def test_real_cached_result_end_to_end(tmp_path):
    sample_id = available_samples()[0]
    model = get_model("unet_v1")
    result = load_cached_result(sample_id, model)
    image = load_image(result.image_path)
    mask = tifffile.imread(result.mask_path)
    assert result.mode == "cached" and result.review_status == "unverified"
    assert result.ground_truth is False
    assert mask.shape == image.shape[:2]
    assert result.statistics["comparison_metrics"]["V1 Process"]["manual_process_coverage"] > 0
    run_dir = save_demo_result(result, image, {"source": "test"}, tmp_path)
    assert (run_dir / "statistics.json").is_file()
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "prediction_mask.tif").is_file()
    analysis = "Measured: high predicted area. Interpretation: overprediction remains possible."
    response, report = save_analysis(run_dir, analysis, result)
    assert response.read_text(encoding="utf-8").startswith("Measured:")
    assert analysis in report.read_text(encoding="utf-8")


def test_cached_cellpose_statistics_and_aggregate():
    sample_id = available_samples()[0]
    result = load_cached_result(sample_id, get_model("cellpose"))
    assert result.statistics["candidate_cell_count"] == 28
    assert result.statistics["candidate_nucleus_count"] == 97
    assert result.artifacts["nucleus_mask"].is_file()
    assert len(aggregate_metrics()) == 5


def test_visualization_preserves_shape_and_labels():
    base = np.zeros((8, 8, 3), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:4, 2:4] = 1
    mask[5:7, 5:7] = 2
    overlay = semantic_overlay(base, mask)
    display = mask_display(mask)
    assert overlay.shape == display.shape == (8, 8, 3)
    assert np.all(display[2, 2] == (0, 1, 0))
    assert np.allclose(display[5, 5], (1, 0.1, 0.1))
    with pytest.raises(ValueError, match="does not match"):
        semantic_overlay(base, np.zeros((4, 4), dtype=np.uint8))


def test_upload_validation_and_model_contract():
    with pytest.raises(ValueError, match="Unsupported image format"):
        read_upload("image.jpg", b"not an image")
    with pytest.raises(ValueError, match="Could not validate"):
        read_upload("broken.png", b"not an image")
    validate_for_model(np.zeros((100, 100, 2), dtype=np.uint8), {"min_channels": 2, "patch_size": 256})
    with pytest.raises(ValueError, match="at least 2 channels"):
        validate_for_model(np.zeros((256, 256), dtype=np.uint8), {"min_channels": 2})


def test_sciencepal_final_response_extraction():
    events = [
        {"type": "assistant_response_delta", "delta": "Preliminary "},
        {"type": "assistant_response_delta", "delta": "analysis."},
        {"type": "assistant_response_end", "response": "Final assistant response."},
        {"type": "thread_run_end", "status": "completed"},
    ]
    assert extract_final_response(events) == "Final assistant response."
    assert extract_final_response(events[:2]) == "Preliminary analysis."
    wrapped = '{"choices":[{"message":{"role":"assistant","content":"Connection successful."}}]}'
    assert extract_final_response([{"type": "assistant_response_end", "response": wrapped}]) == "Connection successful."
    with pytest.raises(SciencePalError, match="without extractable"):
        extract_final_response([{"type": "thread_run_end", "status": "completed"}])


class MockRuns:
    def __init__(self, fail=None, empty=False):
        self.fail = fail
        self.empty = empty
        self.prompts = []
        self.initiation_options = []
        self.continued_messages = []
        self.start_count = 0

    async def initiate(self, agent_id, prompt, **kwargs):
        self.prompts.append(prompt)
        self.initiation_options.append(kwargs)
        if self.fail:
            raise self.fail
        return SimpleNamespace(agent_run_id="mock-run", thread_id="mock-thread")

    async def add_message(self, thread_id, content):
        self.continued_messages.append((thread_id, content))

    async def start(self, thread_id, request):
        self.start_count += 1
        assert request.agent_id == "mock-agent"
        return SimpleNamespace(agent_run_id="continued-run")

    async def stream(self, run_id):
        if not self.empty:
            yield {"type": "assistant_response_end", "content": "Mocked final report"}


class MockClient:
    def __init__(self, fail=None, empty=False):
        self.runs = MockRuns(fail, empty)


def configured_mock_client(monkeypatch, mock=None):
    monkeypatch.setenv("SCIENCEPAL_API_KEY", "mock-not-a-secret")
    monkeypatch.setenv("SCIENCEPAL_AGENT_ID", "mock-agent")
    return SciencePalClient(client_factory=lambda **kwargs: mock or MockClient())


def test_sciencepal_chat_context_analysis_and_deidentification(monkeypatch):
    mock = MockClient()
    client = configured_mock_client(monkeypatch, mock)
    assert client.ask("hello") == "Mocked final report"
    response, thread_id = client.chat("follow up", thread_id="mock-thread")
    assert response == "Mocked final report" and thread_id == "mock-thread"
    assert mock.runs.continued_messages == [("mock-thread", "follow up")]
    assert mock.runs.start_count == 1
    # Delivery-only correction has dedicated tests; this test covers context and de-identification.
    monkeypatch.setattr(client, "_is_delivery_status", lambda _text: False)
    assert client.analyze_results(
        {
            "sample_id": "private-id",
            "patient_id": "private-patient",
            "image_path": "private-path",
            "candidate_cell_count": 3,
        },
        model_id="v33b2",
        mode="cached",
    ) == "Mocked final report"
    prompt = mock.runs.prompts[-1]
    assert all(value not in prompt for value in ("private-id", "private-patient", "private-path"))
    assert all(value in prompt for value in ("candidate_cell_count", "v33b2", "cached"))
    attached_names = [name for name, _content in mock.runs.initiation_options[-1]["files"]]
    assert attached_names == sorted(attached_names) and len(attached_names) == 5


def test_sciencepal_sdk_uses_bounded_timeout_without_paid_retries(monkeypatch):
    monkeypatch.setenv("SCIENCEPAL_API_KEY", "mock-not-a-secret")
    monkeypatch.setenv("SCIENCEPAL_AGENT_ID", "mock-agent")
    options = {}
    mock = MockClient()

    def factory(**kwargs):
        options.update(kwargs)
        return mock

    assert SciencePalClient(client_factory=factory).ask("hello") == "Mocked final report"
    assert options["timeout"] == 30 and options["max_retries"] == 0
    assert options["base_url"] == "https://sciencepal.ai/api"


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (RuntimeError("sensitive service payload"), "service request failed"),
        (TimeoutError("sensitive timeout payload"), "timed out"),
        (AttributeError("sensitive SDK payload"), "incompatibility"),
    ],
)
def test_sciencepal_safe_request_errors(monkeypatch, failure, message):
    client = configured_mock_client(monkeypatch, MockClient(fail=failure))
    with pytest.raises(SciencePalError, match=message) as caught:
        client.ask("hello")
    assert "sensitive" not in str(caught.value)


@pytest.mark.parametrize("status", [401, 403, 429])
def test_sciencepal_status_errors(monkeypatch, status):
    failure = RuntimeError("sensitive status payload")
    failure.status_code = status
    client = configured_mock_client(monkeypatch, MockClient(fail=failure))
    expected = "rate limit" if status == 429 else "authentication failed"
    with pytest.raises(SciencePalError, match=expected) as caught:
        client.ask("hello")
    assert "sensitive" not in str(caught.value)


def test_sciencepal_empty_response_is_explicit(monkeypatch):
    client = configured_mock_client(monkeypatch, MockClient(empty=True))
    with pytest.raises(SciencePalError, match="without extractable"):
        client.ask("hello")


def test_sciencepal_without_credentials(monkeypatch):
    monkeypatch.delenv("SCIENCEPAL_API_KEY", raising=False)
    monkeypatch.delenv("SCIENCEPAL_AGENT_ID", raising=False)
    with pytest.raises(SciencePalError, match="SCIENCEPAL_API_KEY"):
        SciencePalClient()
    monkeypatch.setenv("SCIENCEPAL_API_KEY", "mock-not-a-secret")
    with pytest.raises(SciencePalError, match="SCIENCEPAL_AGENT_ID"):
        SciencePalClient()


def test_sciencepal_analysis_corrects_delivery_only_response(monkeypatch):
    client = configured_mock_client(monkeypatch)
    calls = []

    def fake_chat(prompt, thread_id=None, attachments=None):
        calls.append((prompt, thread_id, attachments))
        if thread_id is None:
            return "The task is complete. The report was written to a file.", "mock-analysis-thread"
        return (
            "## Measured results\nAssigned pixels: 3.\n"
            "## Interpretation\nUnverified candidates only.\n"
            "## Uncertainties and missing evidence\nNo confirmed negatives."
        ), thread_id

    monkeypatch.setattr(client, "chat", fake_chat)
    answer = client.analyze_results({"candidate_cell_count": 3}, model_id="v33b2", mode="cached")
    assert answer.startswith("## Measured results")
    assert len(calls) == 2
    assert calls[1][1] == "mock-analysis-thread"
    assert "Do not call any tools" in calls[1][0]
    assert "candidate_cell_count" in calls[1][0]


def test_sciencepal_analysis_rejects_repeated_delivery_status(monkeypatch):
    client = configured_mock_client(monkeypatch)
    monkeypatch.setattr(
        client, "chat",
        lambda *args, **kwargs: ("The task is complete; no action is needed.", "mock-thread"),
    )
    with pytest.raises(SciencePalError, match="only a delivery status"):
        client.analyze_results({"candidate_cell_count": 3}, model_id="v33b2", mode="cached")


def test_report_has_scientific_limits():
    result = load_cached_result(available_samples()[0], get_model("unet_v2"))
    report = report_markdown(result)
    assert "unverified" in report
    assert "not confirmed background" in report
    assert "overprediction" in report
    assert "Dice" in report  # named only as an explicitly invalid metric
