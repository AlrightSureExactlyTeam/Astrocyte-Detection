"""Streamlit MVP for cached and live astrocyte segmentation results."""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tifffile

from demo_core.datasets import load_dataset_registry
from demo_core.images import (
    association_state_display, display_image, mask_display, read_upload, validate_for_model,
)
from demo_core.inference import run_live_inference
from demo_core.registry import load_model_registry
from demo_core.reports import OUTPUT_ROOT, report_markdown, save_analysis, save_demo_result
from demo_core.results import aggregate_metrics, available_samples, comparison_visualization, load_cached_result
from demo_core.sciencepal_client import SciencePalClient, SciencePalError
from src.preprocessing import load_image

st.set_page_config(page_title="Astrocyte Detection Demo", page_icon="🔬", layout="wide")


@st.cache_resource(max_entries=4)
def get_sciencepal_client() -> SciencePalClient:
    """Reuse only the lightweight wrapper; each request owns its async SDK client."""
    return SciencePalClient()


st.title("Astrocyte Detection — Research Demo")
st.caption("Unverified candidate predictions · No training is performed · Preliminary in-sample comparison only")

try:
    models = load_model_registry()
    samples = available_samples()
except Exception as exc:
    st.error(f"Demo configuration could not be loaded: {exc}")
    st.stop()

model_by_id = {model.id: model for model in models}

with st.sidebar:
    st.header("A · Image input")
    input_kind = st.radio("Input source", ["Validated GFAP sample", "Upload PNG/TIFF"])
    uploaded = None
    if input_kind == "Validated GFAP sample":
        sample_id = st.selectbox("Image identifier", samples)
    else:
        uploaded = st.file_uploader("Original-resolution image", type=["png", "tif", "tiff"])
        sample_id = Path(uploaded.name).stem if uploaded else "upload"
        st.caption("Files are decoded and validated without resizing or changing intensities.")

    st.header("B · Model and mode")
    model_id = st.radio("Model", list(model_by_id), format_func=lambda key: model_by_id[key].name)
    model = model_by_id[model_id]
    mode = st.radio("Prediction source", ["Existing results (recommended)", "Live inference"])
    cached_mode = mode.startswith("Existing")
    if not cached_mode and not model.live_available:
        st.error(model.unavailable_reason)
    if cached_mode:
        st.info("Previously generated real predictions; not ground truth and not a new inference run.")
    else:
        st.caption(f"Model: {model.checkpoint}")
        st.caption("Only the selected model will be loaded. Live inference does not train or update it.")
    execute = st.button(
        "Load existing result" if cached_mode else "Run live inference",
        type="primary",
        width="stretch",
        disabled=(not cached_mode and not model.live_available) or (input_kind != "Validated GFAP sample" and uploaded is None),
    )

if cached_mode and input_kind != "Validated GFAP sample":
    st.warning("Cached results exist only for the five validated samples. Select Live inference for an upload.")
    execute = False

if execute:
    try:
        if input_kind == "Validated GFAP sample":
            from demo_core.results import comparison_manifest
            comparison, _, _ = comparison_manifest()
            image_path = Path(comparison["source_pattern"].format(source_stem=sample_id.removesuffix("_RGB")))
            if not image_path.is_absolute():
                image_path = Path.cwd() / image_path
            image = load_image(image_path)
            input_metadata = {"source": "validated_project_sample", "image_path": str(image_path)}
        else:
            image, input_metadata = read_upload(uploaded.name, uploaded.getvalue())
            input_metadata["source"] = "user_upload"
        validate_for_model(image, model.expected_input)
        if cached_mode:
            result = load_cached_result(sample_id, model)
            if result.image_path != image_path:
                image = load_image(result.image_path)
                input_metadata["image_path"] = str(result.image_path)
        else:
            with st.spinner(f"Running {model.name}; no training is occurring…"):
                result, live_dir = run_live_inference(image, sample_id, model)
            st.success(f"Live inference completed. Outputs: {live_dir}")
        st.session_state.demo_result = result
        st.session_state.demo_image = image
        st.session_state.demo_input_metadata = input_metadata
        st.session_state.saved_run_dir = None
        st.session_state.analysis = None
        report_key = (result.sample_id, result.model_id, result.mode, str(result.mask_path))
        if st.session_state.get("sciencepal_report_key") != report_key:
            safe_result = replace(
                result,
                sample_id="deidentified",
                statistics=SciencePalClient.deidentified_statistics(result.statistics),
            )
            st.session_state.sciencepal_report = report_markdown(safe_result)
            st.session_state.sciencepal_report_key = report_key
            st.session_state.sciencepal_report_pending = True
            st.session_state.sciencepal_messages = []
            st.session_state.sciencepal_thread_id = None
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")

result = st.session_state.get("demo_result")
image = st.session_state.get("demo_image")

st.header("C · Visualization and statistics")
if result is None:
    st.info("Choose a validated sample and model, then load a cached result. This mode needs no GPU or SciencePal credentials.")
else:
    st.subheader(f"{result.model_name} · {result.sample_id}")
    st.write(f"**Source:** {'Previously generated cached prediction' if result.mode == 'cached' else 'New live inference'} · **Review:** unverified · **Ground truth:** no")
    mask = tifffile.imread(result.mask_path)
    model_type = model_by_id[result.model_id].model_type
    overlay_display = str(result.overlay_path)
    soma = None
    state = None
    assigned = None
    if model_type == "v33b2":
        soma_path = result.artifacts.get("soma_candidates")
        state_path = result.artifacts.get("process_state")
        if soma_path and Path(soma_path).is_file():
            soma = tifffile.imread(soma_path)
            if soma.shape != image.shape[:2] or soma.shape != mask.shape:
                raise ValueError("Soma candidate mask dimensions do not match the original image")
            assigned = np.where((mask > 0) & (soma == 0), mask, 0)
        else:
            st.warning("Soma candidate data unavailable")
        if state_path and Path(state_path).is_file():
            state = tifffile.imread(state_path)
            if state.shape != image.shape[:2]:
                raise ValueError("Association-state mask dimensions do not match the original image")
        else:
            st.warning("Association-state data unavailable")
        show_uncertainty = st.checkbox("Show Ambiguous and Unassigned structures", value=True)
        if not show_uncertainty:
            if "assigned_overlay" in result.artifacts:
                overlay_display = str(result.artifacts["assigned_overlay"])
            elif soma is not None and state is not None:
                from annotation_workspace.tools.run_v33b_pilot import state_overlay
                base = np.rint(display_image(image) * 255).astype(np.uint8)
                overlay_display = state_overlay(base, soma, state, mask, "assigned")
        columns = st.columns(3)
        columns[0].image(display_image(image), caption="Original GFAP display (contrast-adjusted only)", width="stretch")
        if soma is None:
            columns[1].info("Soma candidate data unavailable")
            columns[2].info("Assigned Process instance data unavailable without the Soma mask")
        else:
            columns[1].image(mask_display(soma, instance=True), caption="Soma candidate instance IDs", width="stretch")
            columns[2].image(mask_display(assigned, instance=True), caption="Assigned Process instance IDs", width="stretch")
        status_columns = st.columns(2)
        if state is None:
            status_columns[0].info("Association-state data unavailable")
        else:
            status_columns[0].image(association_state_display(state), caption="Assigned cyan · Ambiguous yellow · Unassigned magenta · Rejected red", width="stretch")
        status_columns[1].image(overlay_display, caption="Final colored prediction overlay", width="stretch")
    else:
        columns = st.columns(3)
        columns[0].image(display_image(image), caption="Original GFAP display (contrast-adjusted only)", width="stretch")
        columns[1].image(mask_display(mask, instance=model_type == "cellpose"), caption="Predicted mask / instance IDs", width="stretch")
        columns[2].image(overlay_display, caption="Prediction overlay", width="stretch")
    if model_type == "unet":
        st.caption("Semantic colors: green = predicted Soma; red = predicted Process. Background output is not reliable foreground/background supervision.")
    elif model_type == "v33b2":
        st.caption("Experimental inference: categorical instance colors preserve IDs; association states remain separate. Ownership is unverified; counts are not validated biological cell counts.")
    else:
        st.caption("Cellpose counts are candidate instance counts, not confirmed cells or nuclei.")

    stats = result.statistics
    if "predicted_pixel_counts" in stats:
        counts = stats["predicted_pixel_counts"]
        metric_columns = st.columns(4)
        for column, name in zip(metric_columns, ("background", "soma", "process")):
            column.metric(f"Predicted {name} pixels", f"{counts.get(name, 0):,}")
        fractions = stats.get("predicted_pixel_fractions", {})
        metric_columns[3].metric("Predicted Process area", f"{fractions.get('process', 0):.2%}")
    if "candidate_cell_count" in stats:
        metric_columns = st.columns(4)
        metric_columns[0].metric("Candidate cell regions", stats["candidate_cell_count"])
        metric_columns[1].metric("Candidate nuclei", stats["candidate_nucleus_count"])
        if "possible_one_to_one_pairs" in stats:
            metric_columns[2].metric("Possible 1:1 pairs", stats["possible_one_to_one_pairs"])
            metric_columns[3].metric("Matching issues", stats["matching_issue_count"])
        else:
            metric_columns[2].metric("Possible 1:1 pairs", "Not saved")
            metric_columns[3].metric("Matching issues", "Not saved")
    if model_type == "v33b2":
        if soma is not None:
            soma_count = int(np.count_nonzero(np.unique(soma[soma > 0])))
        else:
            soma_count = stats.get("selected_soma_candidates", "Unavailable")
        metric_columns = st.columns(5)
        metric_columns[0].metric("Candidate Somas", soma_count)
        metric_columns[1].metric("Assigned pixels", f"{stats.get('assigned_pixels', 0):,}")
        metric_columns[2].metric("Ambiguous pixels", f"{stats.get('ambiguous_pixels', 0):,}")
        metric_columns[3].metric("Unassigned pixels", f"{stats.get('unassigned_pixels', 0):,}")
        metric_columns[4].metric("Structural units", stats.get("structural_units", 0))
        st.caption("Adaptive reconstruction cap: 4 px · ROI: 256 px · Rejected candidates are excluded from the final overlay.")
        with st.expander("Branch association records"):
            records_path = result.artifacts.get("structural_unit_records")
            if records_path and Path(records_path).is_file():
                st.dataframe(pd.read_csv(records_path), width="stretch")
            else:
                st.info("Branch association records unavailable")

    comparison_metrics = stats.get("comparison_metrics", {})
    if comparison_metrics:
        st.subheader("Partial-annotation comparison measurements")
        metric_names = {
            "manual_process_coverage": "Manual Process coverage",
            "process_skeleton_recall_exact": "Skeleton recall (exact)",
            "process_skeleton_recall_tolerance_2px": "Skeleton recall (≤2 px)",
            "manual_soma_coverage": "Manual Soma coverage",
            "predicted_foreground_area_ratio": "Predicted foreground area",
            "process_enrichment": "Process enrichment",
        }
        rows = []
        for variant, values in comparison_metrics.items():
            rows.append({"Prediction": variant, **{label: values.get(key) for key, label in metric_names.items()}})
        st.dataframe(pd.DataFrame(rows).set_index("Prediction"), width="stretch")
        st.caption("Coverage metrics use partial manual labels; no full-image Precision, F1, Dice, or IoU is calculated.")

    if result.mode == "cached" and result.sample_id in samples:
        if st.checkbox("Show Cellpose / V1 / V2 side-by-side comparison", value=True):
            st.image(str(comparison_visualization(result.sample_id)), caption="Same five-sample preliminary comparison; previously generated outputs", width="stretch")
        with st.expander("Pooled five-sample measurements"):
            pooled = pd.DataFrame(aggregate_metrics()).set_index("prediction")
            display_columns = [
                "manual_process_coverage", "process_skeleton_recall_exact",
                "process_skeleton_recall_tolerance_2px", "manual_soma_coverage",
                "predicted_foreground_area_ratio", "process_enrichment",
            ]
            st.dataframe(pooled[display_columns], width="stretch")
            st.warning("U-Net coverage is higher, but extreme foreground overprediction undermines that gain. Superiority over Cellpose is not established.")

    input_metadata = st.session_state.demo_input_metadata
    report = report_markdown(result, st.session_state.get("analysis"))
    export_columns = st.columns(3)
    export_columns[0].download_button("Download Markdown report", report, file_name="astrocyte_report.md", mime="text/markdown")
    export_columns[1].download_button("Download text report", report, file_name="astrocyte_report.txt", mime="text/plain")
    if export_columns[2].button("Save this demo run"):
        saved_dir = save_demo_result(result, image, input_metadata, OUTPUT_ROOT)
        if st.session_state.get("analysis"):
            save_analysis(saved_dir, st.session_state.analysis, result)
        st.session_state.saved_run_dir = str(saved_dir)
    if st.session_state.get("saved_run_dir"):
        st.success(f"Saved unique output run: {st.session_state.saved_run_dir}")
    if model_type == "v33b2":
        b2_exports = st.columns(3)
        b2_exports[0].download_button(
            "Download final visualization", result.overlay_path.read_bytes(),
            file_name=f"{result.sample_id}_v33b2_overlay.png", mime="image/png",
        )
        b2_exports[1].download_button(
            "Download branch association CSV", result.artifacts["structural_unit_records"].read_bytes(),
            file_name=f"{result.sample_id}_v33b2_branches.csv", mime="text/csv",
        )
        b2_exports[2].download_button(
            "Download Soma and Assigned instance mask", result.mask_path.read_bytes(),
            file_name=f"{result.sample_id}_v33b2_instances.tif", mime="image/tiff",
        )

st.header("D · SciencePal preliminary quality analysis")
st.caption("Only structured, de-identified statistics are sent. Original images and source metadata are not uploaded.")
if result is None:
    st.info("Load or generate a prediction before requesting analysis.")
else:
    missing_credentials = [
        name for name in ("SCIENCEPAL_API_KEY", "SCIENCEPAL_AGENT_ID") if not os.environ.get(name)
    ]
    sciencepal_ready = not missing_credentials
    if missing_credentials:
        st.warning(
            f"{', '.join(missing_credentials)} not set. Detection, visualization, and local report export still work."
        )
    if st.button("Analyze current statistics", disabled=not sciencepal_ready):
        try:
            with st.spinner("Requesting preliminary analysis from SciencePal…"):
                st.session_state.analysis = get_sciencepal_client().analyze_results(
                    result.statistics, model_id=result.model_id, mode=result.mode,
                )
        except SciencePalError as exc:
            st.error(str(exc))
        except Exception:
            st.error("SciencePal analysis failed safely; local results and inference remain unchanged")
    analysis = st.session_state.get("analysis")
    if analysis:
        st.markdown(analysis)
        st.download_button("Download SciencePal response", analysis, file_name="sciencepal_analysis.md", mime="text/markdown")
    with st.expander("Ask the existing Astrocyte Research Assistant"):
        st.session_state.setdefault("sciencepal_messages", [])
        st.session_state.setdefault("sciencepal_thread_id", None)
        for message in st.session_state.sciencepal_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        prompt = st.chat_input(
            "Message the Astrocyte Research Assistant",
            disabled=not sciencepal_ready,

        )
        if prompt:
            st.session_state.sciencepal_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            try:
                outgoing_prompt = prompt
                if st.session_state.get("sciencepal_report_pending"):
                    outgoing_prompt = (
                        "Use this de-identified report as context for the current prediction. "
                        "Predictions are unverified candidates, not ground truth.\n\n"
                        + st.session_state.sciencepal_report
                        + "\nUser question:\n" + prompt
                    )
                with st.spinner("Requesting a response from SciencePal…"):
                    response, thread_id = get_sciencepal_client().chat(
                        outgoing_prompt, thread_id=st.session_state.sciencepal_thread_id,
                    )
                st.session_state.sciencepal_report_pending = False
                st.session_state.sciencepal_thread_id = thread_id
                st.session_state.sciencepal_messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
            except SciencePalError as exc:
                st.error(str(exc))
            except Exception:
                st.error("SciencePal chat failed safely; local results and inference remain unchanged")

with st.expander("Future dataset and model registration"):
    st.write("Dataset registration validates metadata only and never starts training. Manual partial, ImageJ weak, and fully verified annotations remain distinct.")
    try:
        datasets = load_dataset_registry()
        st.dataframe(pd.DataFrame([{
            "Name": item["name"], "Type": item["annotation_type"],
            "Version": item["version"], "Description": item["source_description"],
        } for item in datasets]), width="stretch")
    except Exception as exc:
        st.warning(f"Dataset registry could not be displayed: {exc}")
    st.code("from demo_core.datasets import register_dataset\nregister_dataset('path/to/dataset_config.yaml')", language="python")
    st.write("Add future models to `configs/demo_models.yaml` with a unique ID, checkpoint, adapter, input contract, output mapping, and version. Existing adapters are `cellpose`, `unet`, and `v33b2`; other adapters are explicitly unavailable until implemented.")
