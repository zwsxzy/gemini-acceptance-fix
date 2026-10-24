"""
Unified Executor engine for running Antigravity & Gemini Flash worker agents.
Supports priority dispatch:
1. Local Antigravity IDE language server (active authenticated session, no API key).
2. Google Gemini REST API (requires GEMINI_API_KEY or GOOGLE_API_KEY).
If neither is available, the job fails loudly with an explicit error rather
than inventing a result.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Optional
from urllib.parse import urlencode

from .antigravity_local import (
    AntigravityLocalExecutor,
    ModelInfo,
    collect_file_contexts,
    describe_available_models,
    discover_language_server,
    resolve_model_id,
)
from .job_manager import Job

# How long a single agent job may run before it is abandoned.
#
# The old default was 300s, which is shorter than plenty of legitimate work: a
# `pip install` on a slow link, a multi-file refactor, or anything that waits on
# a subprocess. A real run was observed finishing at 17:55:0x after the bridge
# had already given up at 17:54:57 — the agent had done the work correctly and
# the result was thrown away three seconds short.
#
# A generous ceiling costs nothing when jobs finish early (completion is
# detected, not waited out); it only bounds a genuinely stuck agent. Override
# with ANTIGRAVITY_JOB_TIMEOUT (seconds).
DEFAULT_JOB_TIMEOUT = float(os.environ.get("ANTIGRAVITY_JOB_TIMEOUT", "1800"))


class AgentExecutor:
    """
    Unified executor dispatching agent jobs in priority order:
    local Antigravity language server, then the Gemini REST API.
    """

    def __init__(self, api_key: Optional[str] = None, timeout: Optional[float] = None):
        self.api_key = (
            api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
        self.timeout = DEFAULT_JOB_TIMEOUT if timeout is None else timeout

    def run_job_sync(
        self,
        job: Job,
        context_files: Optional[list[str]] = None,
        system_instruction: Optional[str] = None,
    ) -> None:
        if job.is_cancelled():
            return
        try:
            # Priority 1: Local Antigravity Language Server
            if os.environ.get("ANTIGRAVITY_DISABLE_LOCAL", "").lower() not in ("1", "true"):
                endpoint = discover_language_server()
                if endpoint:
                    local_exec = AntigravityLocalExecutor(endpoint=endpoint, timeout=self.timeout)
                    local_exec.run_job_sync(
                        job, context_files=context_files, system_instruction=system_instruction
                    )
                    return

            # Priority 2: Gemini REST API (if GEMINI_API_KEY is set)
            if self.api_key:
                self._run_gemini_rest(
                    job, context_files=context_files, system_instruction=system_instruction
                )
                return

            if job.start():
                job.fail(
                    "No execution backend available. "
                    "Ensure the Antigravity IDE is running (with its language server active), "
                    "or set GEMINI_API_KEY in the environment for cloud execution."
                )
                job.log("Execution failed: No available backend.")
        except Exception as exc:
            job.fail(f"{type(exc).__name__}: {exc}")
            job.log(f"Execution failed: {exc}")

    def describe_available_models(self) -> str:
        if (
            os.environ.get("ANTIGRAVITY_DISABLE_LOCAL", "").lower() not in ("1", "true")
            and discover_language_server()
        ):
            return describe_available_models()
        if not self.api_key:
            raise RuntimeError(
                "No execution backend available. Open Antigravity or set GEMINI_API_KEY."
            )
        models = self._list_rest_models(time.monotonic() + self.timeout)
        lines = [f"{len(models)} models available (Gemini REST API):"]
        lines.extend(
            f'  - "{m.model_id}" ({m.label})' for m in sorted(models, key=lambda m: m.model_id)
        )
        lines.append(
            "\nPass an exact model ID or a model family such as 'flash' to dispatch_gemini_agent."
        )
        return "\n".join(lines)

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Gemini REST execution timed out.")
        return remaining

    def _rest_request(self, path: str, deadline: float, payload=None) -> dict:
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/{path}",
            data=json.dumps(payload).encode("utf-8") if payload is not None else None,
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
        )
        with urllib.request.urlopen(req, timeout=self._remaining(deadline)) as response:
            result = json.loads(response.read().decode("utf-8"))
        self._remaining(deadline)
        if not isinstance(result, dict):
            raise ValueError("Gemini REST returned a non-object response.")
        return result

    def _list_rest_models(self, deadline: float) -> list[ModelInfo]:
        models = {}
        page_token = None
        seen_tokens = set()
        while True:
            params = {"pageSize": 1000}
            if page_token:
                params["pageToken"] = page_token
            data = self._rest_request(f"models?{urlencode(params)}", deadline)
            for model in data.get("models", []):
                name = model.get("name", "")
                if (
                    "generateContent" in model.get("supportedGenerationMethods", [])
                    and isinstance(name, str)
                    and re.fullmatch(r"models/[A-Za-z0-9._-]+", name)
                ):
                    model_id = name.removeprefix("models/")
                    models[model_id] = ModelInfo(
                        label=model.get("displayName") or model_id,
                        model_id=model_id,
                        raw=model,
                    )
            page_token = data.get("nextPageToken")
            if not page_token:
                break
            if page_token in seen_tokens:
                raise ValueError("Gemini REST returned a repeated models page token.")
            seen_tokens.add(page_token)
        if not models:
            raise ValueError("Gemini REST returned no models supporting generateContent.")
        return list(models.values())

    def _run_gemini_rest(
        self,
        job: Job,
        context_files: Optional[list[str]] = None,
        system_instruction: Optional[str] = None,
    ) -> None:
        if not job.start():
            return
        job.log(f"Started job on Gemini REST API with model {job.model}")
        deadline = time.monotonic() + self.timeout
        try:
            context_str = ""
            if context_files:
                job.log(f"Reading {len(context_files)} context files...")
                context_str = collect_file_contexts(context_files)
                job.log(f"Context gathered ({len(context_str)} characters).")
            if job.is_cancelled():
                return
            models = self._list_rest_models(deadline)
            # The API's resource prefix is accepted as well as the short ID.
            model_id = resolve_model_id(job.model.removeprefix("models/"), models)
            job.metadata["resolved_model"] = model_id
            job.log(f"Resolved REST model '{job.model}' -> '{model_id}'")
            if job.is_cancelled():
                return

            system_prompt = system_instruction or (
                "You are an expert AI software engineer and background subagent worker assisting a lead architect. "
                "Provide direct, high-precision, actionable code and analysis."
            )

            parts = []
            if context_str:
                parts.append({"text": f"Context files:\n{context_str}\n\n"})
            parts.append({"text": f"Task:\n{job.task}"})

            payload = {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192},
            }

            job.log("Dispatching request to Gemini REST endpoint...")
            resp_data = self._rest_request(f"models/{model_id}:generateContent", deadline, payload)
            if job.is_cancelled():
                return
            candidates = resp_data.get("candidates", [])
            if not candidates:
                reason = resp_data.get("promptFeedback", {}).get("blockReason", "no candidate")
                raise ValueError(f"Gemini returned no candidate: {reason}")
            candidate = candidates[0]
            finish_reason = candidate.get("finishReason")
            if finish_reason != "STOP":
                raise ValueError(f"Gemini generation did not finish completely: {finish_reason}")
            parts = candidate.get("content", {}).get("parts", [])
            result = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            if not result.strip():
                raise ValueError("Gemini returned an empty text response.")
            job.metadata["model_usage"] = resp_data.get("usageMetadata", {})
            if job.complete(result):
                job.log("Response received and processed.")

        except Exception as e:
            job.fail(str(e))
            job.log(f"Error during execution: {e}")


# Backwards compatibility alias
GeminiExecutor = AgentExecutor
