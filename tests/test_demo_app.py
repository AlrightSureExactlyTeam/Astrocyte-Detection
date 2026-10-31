"""UI-level tests using cached results and mocked SciencePal calls only."""
from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from demo_core.sciencepal_client import SciencePalClient, SciencePalError

APP_PATH = Path(__file__).resolve().parents[1] / "demo_app.py"


class FakeSciencePalClient:
    deidentified_statistics = staticmethod(SciencePalClient.deidentified_statistics)
    instances = 0
    analysis_calls = []
    chat_calls = []
    fail = False

    def __init__(self):
        type(self).instances += 1

    def analyze_results(self, statistics, *, model_id, mode):
        type(self).analysis_calls.append((statistics, model_id, mode))
        if self.fail:
            raise SciencePalError("mock analysis failure")
        return "Mocked SciencePal analysis"

    def chat(self, prompt, thread_id=None):
        type(self).chat_calls.append((prompt, thread_id))
        if self.fail:
            raise SciencePalError("mock chat failure")
        return "Mocked SciencePal chat", "mock-thread"


def new_app():
    return AppTest.from_file(APP_PATH, default_timeout=30)


def configure_sciencepal(monkeypatch, fail=False):
    st.cache_resource.clear()
    FakeSciencePalClient.instances = 0
    FakeSciencePalClient.analysis_calls = []
    FakeSciencePalClient.chat_calls = []
    FakeSciencePalClient.fail = fail
    monkeypatch.setenv("SCIENCEPAL_API_KEY", "mock-not-a-secret")
    monkeypatch.setenv("SCIENCEPAL_AGENT_ID", "mock-agent")
    monkeypatch.setattr("demo_core.sciencepal_client.SciencePalClient", FakeSciencePalClient)


def load_cached_result(app):
    app.run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state.demo_result.mode == "cached"


def test_cached_results_ui_path():
    app = new_app()
    load_cached_result(app)
    assert app.session_state.demo_result.review_status == "unverified"
    assert app.session_state.demo_result.sample_id


def test_cached_v33b2_ui_path_and_uncertainty_toggle():
    app = new_app()
    app.run()
    app.radio[1].set_value("v33b2").run()
    app.button[0].click().run()
    assert not app.exception
    result = app.session_state.demo_result
    assert result.model_id == "v33b2" and result.mode == "cached"
    assert result.artifacts["process_state"].is_file()
    toggle = next(item for item in app.checkbox if item.label == "Show Ambiguous and Unassigned structures")
    toggle.uncheck().run()
    assert not app.exception
    result.artifacts.pop("soma_candidates")  # Simulate an older, incomplete session result.
    app.run()
    assert not app.exception
    assert any("Soma candidate data unavailable" in item.value for item in app.warning)


def test_sciencepal_explicit_calls_context_and_no_duplicate_reruns(monkeypatch):
    configure_sciencepal(monkeypatch)
    app = new_app()
    load_cached_result(app)
    app.run()
    assert FakeSciencePalClient.instances == 0

    next(button for button in app.button if button.label == "Analyze current statistics").click().run()
    assert not app.exception
    assert len(FakeSciencePalClient.analysis_calls) == 1
    _, model_id, mode = FakeSciencePalClient.analysis_calls[0]
    assert model_id == app.session_state.demo_result.model_id == "cellpose" and mode == "cached"
    assert app.session_state.analysis == "Mocked SciencePal analysis"
    app.run()
    assert len(FakeSciencePalClient.analysis_calls) == 1

    result = app.session_state.demo_result
    app.chat_input[0].set_value("First question").run()
    assert not app.exception
    app.chat_input[0].set_value("Second question").run()
    assert not app.exception
    first_prompt, first_thread = FakeSciencePalClient.chat_calls[0]
    assert "Astrocyte Detection preliminary report" in first_prompt
    assert "First question" in first_prompt and "Ground truth:** False" in first_prompt
    assert result.sample_id not in first_prompt and str(result.image_path) not in first_prompt
    assert first_thread is None and not app.session_state.sciencepal_report_pending
    assert FakeSciencePalClient.chat_calls[1] == ("Second question", "mock-thread")
    assert app.session_state.sciencepal_thread_id == "mock-thread"
    assert len(app.session_state.sciencepal_messages) == 4
    app.run()
    assert len(FakeSciencePalClient.chat_calls) == 2
    assert FakeSciencePalClient.instances == 1

    app.radio[1].set_value("v33b2").run()
    next(button for button in app.button if button.label == "Load existing result").click().run()
    assert not app.exception
    assert app.session_state.demo_result.model_id == "v33b2"
    assert app.session_state.sciencepal_report_pending
    assert app.session_state.sciencepal_thread_id is None
    assert app.session_state.sciencepal_messages == []


def test_sciencepal_failure_preserves_local_result_and_reruns(monkeypatch):
    configure_sciencepal(monkeypatch, fail=True)
    app = new_app()
    load_cached_result(app)
    original_result = app.session_state.demo_result
    app.chat_input[0].set_value("This request fails").run()
    assert not app.exception
    assert any("mock chat failure" in item.value for item in app.error)
    assert app.session_state.demo_result is original_result
    app.run()
    assert not app.exception
    assert len(FakeSciencePalClient.chat_calls) == 1
    assert app.session_state.demo_result is original_result

    next(button for button in app.button if button.label == "Analyze current statistics").click().run()
    assert not app.exception
    assert any("mock analysis failure" in item.value for item in app.error)
    assert app.session_state.demo_result is original_result


def test_missing_agent_id_disables_api_without_affecting_cached_results(monkeypatch):
    configure_sciencepal(monkeypatch)
    monkeypatch.delenv("SCIENCEPAL_AGENT_ID")
    app = new_app()
    load_cached_result(app)
    analyze = next(button for button in app.button if button.label == "Analyze current statistics")
    assert analyze.disabled and app.chat_input[0].disabled
    assert any("SCIENCEPAL_AGENT_ID" in item.value for item in app.warning)
    assert FakeSciencePalClient.instances == 0
