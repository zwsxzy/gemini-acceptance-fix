"""Hermetic REST fallback acceptance; no real key or network is used."""

import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from antigravity_mcp.executor import AgentExecutor
from antigravity_mcp.job_manager import Job


def response(value):
    return io.BytesIO(json.dumps(value).encode())


def catalog(*names, next_page=None):
    data = {
        "models": [
            {
                "name": f"models/{name}",
                "displayName": name,
                "supportedGenerationMethods": ["generateContent"],
            }
            for name in names
        ]
    }
    if next_page:
        data["nextPageToken"] = next_page
    return data


def answer(parts=None, finish="STOP"):
    return {
        "candidates": [
            {
                "finishReason": finish,
                "content": {"parts": [{"text": "REST_OK"}] if parts is None else parts},
            }
        ],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2},
    }


@pytest.fixture(autouse=True)
def disable_local(monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_DISABLE_LOCAL", "1")


def job(model="flash"):
    return Job(
        id="rest-acceptance", task="public acceptance fixture", model=model, status="pending"
    )


def test_default_flash_resolves_to_a_live_rest_model_and_keeps_key_out_of_url():
    target = job()
    calls = []

    def request(req, timeout):
        calls.append(req)
        assert 0 < timeout <= 5
        assert "test-key" not in req.full_url
        assert req.get_header("X-goog-api-key") == "test-key"
        if req.get_method() == "GET":
            return response(catalog("gemini-3.8-flash"))
        assert req.full_url.endswith("/models/gemini-3.8-flash:generateContent")
        return response(answer([{"thought": True, "text": "internal"}, {"text": "REST_OK"}]))

    with patch("antigravity_mcp.executor.urllib.request.urlopen", side_effect=request):
        AgentExecutor(api_key="test-key", timeout=5).run_job_sync(target)
    assert target.status == "completed", target.error
    assert target.result == "REST_OK"
    assert len(calls) == 2
    assert target.metadata["model_usage"]["promptTokenCount"] == 3


def test_rest_catalog_is_paginated_and_filters_non_generation_models():
    executor = AgentExecutor(api_key="test-key")
    first = catalog("gemini-3.8-flash", next_page="next token")
    first["models"].append(
        {"name": "models/embed-only", "supportedGenerationMethods": ["embedContent"]}
    )
    with patch(
        "antigravity_mcp.executor.urllib.request.urlopen",
        side_effect=[response(first), response(catalog("gemini-4-pro"))],
    ) as request:
        listing = executor.describe_available_models()
    assert "gemini-3.8-flash" in listing
    assert "gemini-4-pro" in listing
    assert "embed-only" not in listing
    assert "pageToken=next+token" in request.call_args.args[0].full_url


def test_unknown_rest_model_fails_before_generation():
    target = job("not-a-live-model")
    with patch(
        "antigravity_mcp.executor.urllib.request.urlopen",
        return_value=response(catalog("gemini-3.8-flash")),
    ) as request:
        AgentExecutor(api_key="test-key").run_job_sync(target)
    assert target.status == "failed"
    assert "Unknown model" in target.error
    assert request.call_count == 1


@pytest.mark.parametrize(
    "payload, expected",
    [
        (answer(parts=[]), "empty"),
        (answer(finish="MAX_TOKENS"), "MAX_TOKENS"),
        ({"promptFeedback": {"blockReason": "SAFETY"}}, "SAFETY"),
    ],
)
def test_empty_blocked_or_truncated_rest_output_is_not_success(payload, expected):
    target = job("gemini-3.8-flash")
    with patch(
        "antigravity_mcp.executor.urllib.request.urlopen",
        side_effect=[response(catalog("gemini-3.8-flash")), response(payload)],
    ):
        AgentExecutor(api_key="test-key").run_job_sync(target)
    assert target.status == "failed"
    assert expected.lower() in target.error.lower()


def test_request_error_after_cancellation_cannot_turn_job_into_failure():
    target = job("gemini-3.8-flash")

    def request(req, timeout):
        target.cancel()
        raise urllib.error.URLError("cancelled fixture connection")

    with patch("antigravity_mcp.executor.urllib.request.urlopen", side_effect=request):
        AgentExecutor(api_key="test-key").run_job_sync(target)
    assert target.status == "cancelled"
    assert target.error is None
