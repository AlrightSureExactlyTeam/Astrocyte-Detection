"""Reusable SciencePal client with de-identified statistics and safe stream parsing."""
from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import AsyncIterable, Iterable
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

BASE_URL = "https://sciencepal.ai/api"
RESEARCH_PDF_DIR = Path(__file__).resolve().parents[1] / "research_pdfs"
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


def research_attachments() -> list[tuple[str, bytes]]:
    """Bundled, de-identified research PDFs sent as run attachments.

    The SciencePal knowledge-base retrieval chain is broken platform-side
    (verified 2026-11-01: no retrieval tool is injected even when a KB is
    attached in the web UI). Uploading the project documents as run files
    is the validated alternative: the agent converts them with MinerU and
    cites document names and page numbers.
    """
    if not RESEARCH_PDF_DIR.is_dir():
        return []
    return [
        (path.name, path.read_bytes())
        for path in sorted(RESEARCH_PDF_DIR.glob("*.pdf"))
    ]


class SciencePalError(RuntimeError):
    """A safe, user-displayable SciencePal integration failure."""


def _as_mapping(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        return event
    if hasattr(event, "model_dump"):
        try:
            value = event.model_dump()
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}
    fields = (
        "type", "event", "event_type", "name", "status", "stream_status",
        "response", "content", "text", "delta", "message", "data", "output",
        "final_response", "assistant_response",
    )
    return {key: getattr(event, key) for key in fields if hasattr(event, key)}


def _text_value(value: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{"):
            try:
                decoded = json.loads(stripped)
            except (json.JSONDecodeError, TypeError):
                decoded = None
            if isinstance(decoded, dict) and "choices" in decoded:
                text = _text_value(decoded["choices"])
                if text:
                    return text
        return value
    if isinstance(value, dict):
        if "choices" in value:
            text = _text_value(value["choices"])
            if text:
                return text
        for key in ("text", "content", "value", "response", "message"):
            if key in value:
                text = _text_value(value[key])
                if text:
                    return text
    if isinstance(value, list):
        return "".join(_text_value(item) for item in value)
    return ""


def extract_final_response(events: Iterable[Any]) -> str:
    """Extract final assistant text without exposing tools or raw event objects."""
    final = ""
    deltas: list[str] = []
    for event in events:
        payload = _as_mapping(event)
        event_name = str(
            payload.get("event_type") or payload.get("event") or payload.get("type") or payload.get("name") or ""
        ).lower()
        if "tool" in event_name:
            continue
        is_final = any(token in event_name for token in ("assistant_response_end", "thread_run_end", "response.completed"))
        for key in ("final_response", "assistant_response", "response", "output"):
            text = _text_value(payload.get(key))
            if text and (is_final or key in {"final_response", "assistant_response"}):
                final = text
        for key in ("content", "text", "message"):
            text = _text_value(payload.get(key))
            if text and is_final:
                final = text
        nested = payload.get("data")
        if isinstance(nested, dict) and is_final:
            for key in ("final_response", "assistant_response", "response", "content", "text"):
                text = _text_value(nested.get(key))
                if text:
                    final = text
        if not is_final and any(token in event_name for token in ("delta", "assistant_response", "message")):
            text = _text_value(payload.get("delta")) or _text_value(payload.get("content"))
            if text:
                deltas.append(text)
    answer = final.strip() or "".join(deltas).strip()
    if not answer:
        raise SciencePalError("SciencePal stream completed without extractable assistant text")
    return answer


def _run_async(factory: Callable[[], Any]) -> Any:
    """Run async SDK work from synchronous Streamlit code, including active loops."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())
    result: list[Any] = []
    errors: list[BaseException] = []

    def runner() -> None:
        try:
            result.append(asyncio.run(factory()))
        except BaseException as exc:  # propagated in caller thread
            errors.append(exc)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    return result[0]


def _request_error(exc: Exception) -> SciencePalError:
    """Map SDK failures to stable messages without exposing exception payloads."""
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
    name = type(exc).__name__.lower()
    if status in {401, 403} or "authentication" in name or "permission" in name:
        return SciencePalError("SciencePal authentication failed; check the rotated API key and Agent ID")
    if status == 429 or "ratelimit" in name or "rate_limit" in name:
        return SciencePalError("SciencePal rate limit reached; wait before submitting another request")
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or "timeout" in name:
        return SciencePalError("SciencePal request timed out; local inference and visualization remain available")
    if isinstance(exc, (AttributeError, TypeError)) or "schema" in name or "validation" in name:
        return SciencePalError("SciencePal SDK incompatibility; verify the installed sciencepal-sdk version")
    if "connect" in name or "network" in name:
        return SciencePalError("Could not connect to SciencePal; local inference and visualization remain available")
    return SciencePalError("SciencePal service request failed; no response was generated and local results are unchanged")


class SciencePalClient:
    def __init__(
        self,
        api_key: str | None = None,
        agent_id: str | None = None,
        base_url: str = BASE_URL,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("SCIENCEPAL_API_KEY")
        self.agent_id = agent_id if agent_id is not None else os.environ.get("SCIENCEPAL_AGENT_ID")
        if not self.api_key:
            raise SciencePalError("SCIENCEPAL_API_KEY is not set; detection and visualization remain available")
        if not self.agent_id:
            raise SciencePalError("SCIENCEPAL_AGENT_ID is not set; detection and visualization remain available")
        self.base_url = base_url
        self.client_factory = client_factory

    def _new_sdk_client(self) -> Any:
        options = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "timeout": 30,
            "max_retries": 0,
        }
        if self.client_factory is not None:
            return self.client_factory(**options)
        try:
            from sciencepal import SciencePal
            from sciencepal.agent import AgentResponse
            if "agentpress_tools" in AgentResponse.model_fields:
                from typing import Any as TypingAny
                field = AgentResponse.model_fields["agentpress_tools"]
                if field.annotation is not (dict[str, TypingAny] | None):
                    field.annotation = dict[str, TypingAny] | None
                    AgentResponse.model_rebuild(force=True)
            return SciencePal(**options)
        except Exception as exc:
            raise SciencePalError("SciencePal SDK initialization failed; verify sciencepal-sdk compatibility") from exc

    @staticmethod
    def deidentified_statistics(statistics: dict[str, Any]) -> dict[str, Any]:
        """Retain measurements and fixed flags, never source identifiers or paths."""
        forbidden = {
            "path", "source", "sample", "image_id", "filename", "sha256", "metadata",
            "provenance", "patient", "file", "directory",
        }

        def clean(value: Any, key: str = "") -> Any:
            if any(token in key.lower() for token in forbidden):
                return None
            if isinstance(value, dict):
                return {str(k): cleaned for k, v in value.items() if (cleaned := clean(v, str(k))) is not None}
            if isinstance(value, (list, tuple)):
                return [cleaned for item in value if (cleaned := clean(item, key)) is not None]
            if isinstance(value, bool) or isinstance(value, (int, float)) or value is None:
                return value
            if isinstance(value, str) and len(value) <= 160:
                lowered = value.lower().strip()
                path_like = "/" in value or "\\" in value or lowered.endswith(
                    (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".psd", ".npy", ".pt", ".pth")
                )
                return None if path_like else value
            return None

        return clean(statistics) or {}

    def ask(self, prompt: str) -> str:
        """Send a single-turn prompt and return only the assistant text."""
        return self.chat(prompt)[0]

    def chat(self, prompt: str, thread_id: str | None = None,
             attachments: list[tuple[str, bytes]] | None = None) -> tuple[str, str | None]:
        """Send a prompt, continuing the supplied SciencePal thread when available."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must not be empty")

        async def operation() -> tuple[str, str | None]:
            try:
                client = self._new_sdk_client()
                if hasattr(client, "__aenter__"):
                    async with client as active:
                        return await self._stream_request(active, prompt, thread_id, attachments)
                return await self._stream_request(client, prompt, thread_id, attachments)
            except SciencePalError:
                raise
            except Exception as exc:
                raise _request_error(exc) from exc

        return _run_async(operation)

    def _kb_options(self) -> dict[str, Any]:
        """Optional knowledge-base binding; values come from env, never from code."""
        dataset_ids = os.environ.get("SCIENCEPAL_KB_DATASET_IDS", "").strip()
        if not dataset_ids:
            return {}
        options: dict[str, Any] = {
            "dataset_ids": [item.strip() for item in dataset_ids.split(",") if item.strip()],
        }
        kb_url = os.environ.get("SCIENCEPAL_KB_URL", "").strip()
        kb_api_key = os.environ.get("SCIENCEPAL_KB_API_KEY", "").strip()
        if kb_url:
            options["kb_url"] = kb_url
        if kb_api_key:
            options["kb_api_key"] = kb_api_key
        return options

    async def _stream_request(
        self, client: Any, prompt: str, thread_id: str | None = None,
        attachments: list[tuple[str, bytes]] | None = None,
    ) -> tuple[str, str | None]:
        if thread_id:
            await client.runs.add_message(thread_id=thread_id, content=prompt)
            try:
                from sciencepal.models import AgentStartRequest
                request = AgentStartRequest(agent_id=self.agent_id, stream=True)
            except Exception as exc:
                raise SciencePalError("Installed SciencePal SDK does not support contextual chat") from exc
            run = await client.runs.start(thread_id, request)
        else:
            kb_options = self._kb_options()
            if attachments:
                kb_options["files"] = attachments
            try:
                run = await client.runs.initiate(agent_id=self.agent_id, prompt=prompt, **kb_options)
            except TypeError as exc:
                raise SciencePalError(
                    "Installed SciencePal SDK does not support knowledge-base options"
                ) from exc
        run_id = getattr(run, "agent_run_id", None) or (run.get("agent_run_id") if isinstance(run, dict) else None)
        if not run_id:
            raise SciencePalError("SciencePal did not return an agent_run_id")
        returned_thread_id = getattr(run, "thread_id", None) or (
            run.get("thread_id") if isinstance(run, dict) else None
        )
        stream = client.runs.stream(run_id)
        if not isinstance(stream, AsyncIterable):
            raise SciencePalError("SciencePal stream is not asynchronous")
        events = []
        async for event in stream:
            events.append(event)
        return extract_final_response(events), returned_thread_id or thread_id

    def analyze_results(self, statistics: dict[str, Any], *, model_id: str, mode: str) -> str:
        if not model_id or mode not in {"cached", "live"}:
            raise ValueError("A model ID and cached/live result mode are required")
        safe = self.deidentified_statistics({
            "model_id": model_id,
            "result_mode": mode,
            "statistics": statistics,
        })
        attachments = research_attachments()
        evidence_note = (
            "The attached project research PDFs are the evidence base: read them first, and cite "
            "document names and page numbers for any methodology, metric-definition, or QC-rule claims. "
            if attachments else
            "No research documents are attached; explicitly state when evidence is missing. "
        )
        prompt = (
            "Prepare a concise preliminary quality-control analysis for an astrocyte GFAP segmentation demo. "
            + evidence_note
            + "Use only these de-identified measurements for the current run:\n"
            + json.dumps(safe, indent=2, ensure_ascii=False)
            + "\nSeparate measured facts from interpretation and hypotheses. Do not claim diagnostic validity, "
            "confirmed cell counts, or superiority between models. Manual labels are partial: unlabeled pixels "
            "are not confirmed background, and ImageJ masks are weak foreground labels. Explicitly consider "
            "foreground overprediction and the preliminary in-sample scope. Do not invent missing metrics."
        )
        return self.chat(prompt, attachments=attachments or None)[0]
