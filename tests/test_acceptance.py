"""Regression coverage for failures found during Gemini bridge acceptance."""

import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from antigravity_mcp.antigravity_local import (
    AntigravityLocalExecutor,
    LanguageServerEndpoint,
    collect_file_contexts,
)
from antigravity_mcp.executor import AgentExecutor
from antigravity_mcp.job_manager import Job, JobManager
from antigravity_mcp.server import INVALID_PARAMS, MCPServer


class ExplodingExecutor:
    def run_job_sync(self, job, context_files=None, system_instruction=None):
        raise RuntimeError("acceptance worker failure")


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_STATE_DIR", str(tmp_path))
    instance = MCPServer(executor=ExplodingExecutor())
    yield instance
    instance.shutdown(wait_for_jobs=True)


def call(server, name, arguments):
    return server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    )


@pytest.mark.parametrize(
    "name, arguments",
    [
        ("dispatch_gemini_agent", {"task": "x", "model": 42}),
        ("dispatch_gemini_agent", {"task": "x", "system_instruction": 42}),
        ("dispatch_gemini_agent", {"task": "x", "context_files": ""}),
        ("dispatch_gemini_agent", {"task": "x", "context_files": [""]}),
        ("dispatch_gemini_agent", ["not an object"]),
        ("list_active_jobs", {"limit": True}),
        ({"not": "a tool name"}, {}),
    ],
)
def test_malformed_tool_arguments_are_protocol_errors(server, name, arguments):
    response = call(server, name, arguments)
    assert response.get("error", {}).get("code") == INVALID_PARAMS
    assert server.job_manager.list_jobs() == []


def test_malformed_request_params_always_receive_a_response(server):
    response = server.handle_request(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": ["bad"]}
    )
    assert response["error"]["code"] == INVALID_PARAMS


def test_unhandled_worker_exception_finishes_and_persists_job(server):
    call(server, "dispatch_gemini_agent", {"task": "raise a worker exception"})
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        snapshot = server.job_manager.list_jobs()[0]
        if snapshot["status"] == "failed":
            break
        time.sleep(0.01)
    assert snapshot["status"] == "failed"
    assert "acceptance worker failure" in snapshot["error"]
    restored = JobManager(state_path=server.job_manager._state_path).get_job(snapshot["id"])
    assert restored.status == "failed"
    assert restored.error == snapshot["error"]


def test_cancelled_queued_job_never_enters_a_backend(monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_DISABLE_LOCAL", "1")
    job = Job(id="cancelled", task="must not run", model="flash", status="pending")
    job.cancel()
    with (
        patch("antigravity_mcp.executor.discover_language_server") as discover,
        patch.object(AgentExecutor, "_run_gemini_rest") as rest,
    ):
        AgentExecutor(api_key="test-key").run_job_sync(job)
    discover.assert_not_called()
    rest.assert_not_called()
    assert job.status == "cancelled"
    assert job.started_at is None


def test_cancelled_local_job_does_not_resolve_models():
    endpoint = LanguageServerEndpoint("127.0.0.1", 1, "test")
    job = Job(id="cancelled-local", task="must not run", model="flash", status="pending")
    job.cancel()
    with patch("antigravity_mcp.antigravity_local._rpc_call") as rpc:
        AntigravityLocalExecutor(endpoint).run_job_sync(job)
    rpc.assert_not_called()
    assert job.status == "cancelled"


def test_cancellation_during_final_poll_cannot_become_completed():
    endpoint = LanguageServerEndpoint("127.0.0.1", 1, "test")
    job = Job(id="cancel-race", task="x", model="flash", status="pending")
    now = [0.0]
    polls = [0]
    stops = []

    def rpc(endpoint, method, payload, timeout=10):
        if method == "CancelCascadeInvocation":
            stops.append(payload)
        if method == "GetUserStatus":
            return {
                "userStatus": {
                    "cascadeModelConfigData": {
                        "clientModelConfigs": [
                            {"label": "Gemini Flash", "modelOrAlias": {"model": "FLASH"}}
                        ]
                    }
                }
            }
        if method == "GetCascadeTrajectory":
            polls[0] += 1
            if polls[0] == 2:
                job.cancel()
            return {
                "trajectory": {
                    "steps": [
                        {
                            "type": "CORTEX_STEP_TYPE_PLANNER_RESPONSE",
                            "status": "CORTEX_STEP_STATUS_DONE",
                            "plannerResponse": {"response": "final output"},
                        }
                    ]
                }
            }
        return {}

    def sleep(seconds):
        now[0] += 5

    with (
        patch("antigravity_mcp.antigravity_local._rpc_call", side_effect=rpc),
        patch("antigravity_mcp.antigravity_local.time.time", side_effect=lambda: now[0]),
        patch("antigravity_mcp.antigravity_local.time.sleep", side_effect=sleep),
    ):
        AntigravityLocalExecutor(endpoint, timeout=30).run_job_sync(job)
    assert job.status == "cancelled"
    assert job.result is None
    assert stops == [{"cascadeId": job.metadata["cascade_id"]}]
    assert job.metadata["engine_stop_requested"] is True


def test_cancelled_queued_task_is_skipped_when_pool_slot_opens(server):
    entered = threading.Barrier(4)
    release = threading.Event()
    executed = []

    class BlockingExecutor:
        def run_job_sync(self, job, context_files=None, system_instruction=None):
            executed.append(job.task)
            job.start()
            entered.wait(timeout=5)
            release.wait(timeout=5)
            job.complete("done")

    server.executor = BlockingExecutor()
    try:
        for n in range(3):
            call(server, "dispatch_gemini_agent", {"task": f"blocker-{n}"})
        entered.wait(timeout=5)
        call(server, "dispatch_gemini_agent", {"task": "queued-must-not-run"})
        queued = server.job_manager.list_jobs()[0]
        response = call(server, "cancel_agent_job", {"job_id": queued["id"]})
        assert not response["result"]["isError"]
    finally:
        release.set()
        server.shutdown(wait_for_jobs=True)
    assert sorted(executed) == ["blocker-0", "blocker-1", "blocker-2"]
    assert server.job_manager.get_job(queued["id"]).status == "cancelled"


def test_code_search_uses_requested_model(server, tmp_path):
    source = tmp_path / "public-fixture.py"
    source.write_text("def add(a, b): return a + b\n")

    class ModelEchoExecutor:
        def run_job_sync(self, job, context_files=None, system_instruction=None):
            job.start()
            job.complete(f"selected model: {job.model}")

    server.executor = ModelEchoExecutor()
    response = call(
        server,
        "gemini_code_search",
        {"query": "what does add do?", "context_files": [str(source)], "model": "explicit-model"},
    )
    assert "selected model: explicit-model" in response["result"]["content"][0]["text"]
    from antigravity_mcp.server import TOOLS

    search_tool = next(tool for tool in TOOLS if tool["name"] == "gemini_code_search")
    assert "model" in search_tool["inputSchema"]["properties"]


def test_code_search_worker_failure_is_terminal(server, tmp_path):
    source = tmp_path / "public-fixture.py"
    source.write_text("ACCEPTANCE = True\n")
    response = call(
        server,
        "gemini_code_search",
        {"query": "what is ACCEPTANCE?", "context_files": [str(source)]},
    )
    assert response["result"]["isError"]
    assert "acceptance worker failure" in response["result"]["content"][0]["text"]
    assert server.job_manager.list_jobs()[0]["status"] == "failed"


def test_concurrent_history_writes_are_serialized(tmp_path):
    manager = JobManager(state_path=str(tmp_path / "jobs.json"))
    job = manager.create_job("concurrent logging")
    original_dump = json.dump
    writing = 0
    peak_writers = 0
    counter_lock = threading.Lock()

    def slow_dump(*args, **kwargs):
        nonlocal writing, peak_writers
        with counter_lock:
            writing += 1
            peak_writers = max(peak_writers, writing)
        try:
            time.sleep(0.01)
            return original_dump(*args, **kwargs)
        finally:
            with counter_lock:
                writing -= 1

    with (
        patch("antigravity_mcp.job_manager.json.dump", side_effect=slow_dump),
        ThreadPoolExecutor(max_workers=8) as pool,
    ):
        list(pool.map(lambda n: job.log(f"message {n}"), range(16)))
    assert peak_writers == 1
    state = json.loads((tmp_path / "jobs.json").read_text())
    assert len(state["jobs"][0]["progress_logs"]) == 16


def test_context_size_limit_prevents_unbounded_reads(tmp_path):
    path = tmp_path / "large.py"
    path.touch()

    class BoundedReadOnly(io.StringIO):
        def read(self, size=-1):
            assert size >= 0, "context reader must not read an entire arbitrarily large file"
            return super().read(size)

    with patch("builtins.open", return_value=BoundedReadOnly("x" * 10000)):
        result = collect_file_contexts([str(path)], max_total_chars=100)
    assert "TRUNCATED" in result
    assert "x" * 101 not in result
