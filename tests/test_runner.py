"""Runner + ScriptedModel: lifecycle, cleanup and resume (fake API, real local commands)."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

import pytest
from agents import Runner
from agents.run import RunConfig
from agents.sandbox import LocalSnapshotSpec, Manifest, SandboxAgent, SandboxRunConfig
from agents.sandbox.capabilities import Filesystem, Shell
from agents.testing import ModelStep, ScriptedModel, assistant_message, function_call

from openai_agents_wbxsandbox import WBXSandboxClient, WBXSandboxClientOptions


def _agent(model: ScriptedModel, root: str) -> SandboxAgent:
    return SandboxAgent(
        name="tester",
        instructions="Use the shell.",
        model=model,
        default_manifest=Manifest(root=root),
        capabilities=[Shell(), Filesystem()],
    )


def _exec(cmd: str, call_id: str, **extra) -> object:
    return function_call("exec_command", {"cmd": cmd, **extra}, call_id=call_id)


def _config(client: WBXSandboxClient, **kw) -> RunConfig:
    return RunConfig(sandbox=SandboxRunConfig(client=client, **kw))


async def test_success_runs_commands_and_deletes_sandbox(api, ws_root):
    model = ScriptedModel(
        [
            [_exec("echo hello > out.txt && cat out.txt", "c1")],
            [assistant_message("done")],
        ]
    )
    result = await Runner.run(
        _agent(model, ws_root),
        "write a file",
        run_config=_config(WBXSandboxClient(), options=WBXSandboxClientOptions()),
    )
    assert result.final_output == "done"
    model.assert_complete()
    assert (Path(ws_root) / "out.txt").read_text() == "hello\n"
    tool_output = json.dumps([i.to_input_item() for i in result.new_items], default=str)
    assert "hello" in tool_output
    assert len(api.sandboxes) == 1
    assert api.alive() == []  # deleted after the run


async def test_failing_command_is_a_tool_result(api, ws_root):
    model = ScriptedModel(
        [
            [_exec("echo oops >&2; exit 7", "c1")],
            [assistant_message("recovered")],
        ]
    )
    result = await Runner.run(
        _agent(model, ws_root),
        "go",
        run_config=_config(WBXSandboxClient(), options=WBXSandboxClientOptions()),
    )
    assert result.final_output == "recovered"
    second_input = json.dumps(model.calls[1].input, default=str)
    assert "oops" in second_input
    assert api.alive() == []


async def test_transport_error_fails_the_run_and_cleans_up(api, ws_root):
    from agents.exceptions import UserError

    api.fail_exec_matching = "fail-me"
    model = ScriptedModel([[_exec("echo fail-me", "c1")], [assistant_message("unreachable")]])
    with pytest.raises(UserError, match="exec transport error"):
        await Runner.run(
            _agent(model, ws_root),
            "go",
            run_config=_config(WBXSandboxClient(), options=WBXSandboxClientOptions()),
        )
    assert len(api.sandboxes) == 1
    assert api.alive() == []


async def test_model_error_still_cleans_up(api, ws_root):
    model = ScriptedModel(
        [[_exec("true", "c1")], ModelStep.raise_error(RuntimeError("model down"))]
    )
    with pytest.raises(RuntimeError, match="model down"):
        await Runner.run(
            _agent(model, ws_root),
            "go",
            run_config=_config(WBXSandboxClient(), options=WBXSandboxClientOptions()),
        )
    assert len(api.sandboxes) == 1
    assert api.alive() == []


async def test_command_timeout_is_a_tool_result_and_cleans_up(api, ws_root):
    model = ScriptedModel(
        [
            [_exec("sleep 5", "c1", yield_time_ms=300)],
            [assistant_message("gave up")],
        ]
    )
    result = await Runner.run(
        _agent(model, ws_root),
        "go",
        run_config=_config(WBXSandboxClient(), options=WBXSandboxClientOptions()),
    )
    assert result.final_output == "gave up"
    assert api.alive() == []


async def test_cancellation_cleans_up(api, ws_root):
    api.exec_delay_s = 0.0
    started = asyncio.Event()

    def slow(call):
        started.set()
        return [_exec("sleep 30", "c1", yield_time_ms=60000)]

    model = ScriptedModel([ModelStep.respond(slow), [assistant_message("never")]])
    task = asyncio.create_task(
        Runner.run(
            _agent(model, ws_root),
            "go",
            run_config=_config(WBXSandboxClient(), options=WBXSandboxClientOptions()),
        )
    )
    await asyncio.wait_for(started.wait(), 10)
    await asyncio.sleep(1.0)  # the sleep command is running in the sandbox
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for _ in range(100):  # cleanup runs in the cancelled task's finally blocks
        if not api.alive():
            break
        await asyncio.sleep(0.1)
    assert len(api.sandboxes) == 1
    assert api.alive() == []


async def test_partial_startup_failure_deletes_sandbox(api, ws_root):
    api.fail_exec_matching = "mkdir -p"  # workspace root setup fails after the VM exists
    api.fail_exec_status = 500
    model = ScriptedModel([[assistant_message("unreachable")]])
    with pytest.raises(Exception):  # noqa: B017
        await Runner.run(
            _agent(model, ws_root),
            "go",
            run_config=_config(WBXSandboxClient(), options=WBXSandboxClientOptions()),
        )
    assert len(api.sandboxes) == 1
    assert api.alive() == []


async def test_create_failure_leaves_nothing(api, ws_root):
    api.fail_create = (429, "concurrent sandbox limit reached")
    model = ScriptedModel([[assistant_message("unreachable")]])
    with pytest.raises(Exception, match="concurrent sandbox limit"):
        await Runner.run(
            _agent(model, ws_root),
            "go",
            run_config=_config(WBXSandboxClient(), options=WBXSandboxClientOptions()),
        )
    assert api.sandboxes == {}


async def test_default_options_are_allowed(api, ws_root):
    model = ScriptedModel([[_exec("true", "c1")], [assistant_message("ok")]])
    result = await Runner.run(_agent(model, ws_root), "go", run_config=_config(WBXSandboxClient()))
    assert result.final_output == "ok"
    assert api.alive() == []


async def test_live_session_is_owned_by_the_application(api, ws_root):
    client = WBXSandboxClient()
    session = await client.create(manifest=Manifest(root=ws_root), options=None)
    async with session:
        model = ScriptedModel([[_exec("echo shared > s.txt", "c1")], [assistant_message("ok")]])
        result = await Runner.run(
            _agent(model, ws_root),
            "go",
            run_config=RunConfig(sandbox=SandboxRunConfig(session=session)),
        )
        assert result.final_output == "ok"
        assert api.alive() == [session.state.sandbox_id]  # the runner did not delete it
        assert (await session.read(Path("s.txt"))).read() == b"shared\n"
    await client.delete(session)
    assert api.alive() == []


async def test_snapshot_restore_into_new_sandbox(api, ws_root, tmp_path):
    """Workspace snapshot from run 1 restores into a replacement sandbox on resume."""
    client = WBXSandboxClient()
    snapshot = LocalSnapshotSpec(base_path=tmp_path / "snapshots")
    session = await client.create(
        snapshot=snapshot, manifest=Manifest(root=ws_root), options=WBXSandboxClientOptions()
    )
    await session.start()
    await session.write(Path("notes/todo.txt"), io.BytesIO(b"keep me"))
    await session.aclose()  # persists the snapshot, then shuts the VM down
    await client.delete(session)
    payload = client.serialize_session_state(session.state)
    first_id = session.state.sandbox_id
    assert api.alive() == []

    Path(ws_root, "notes/todo.txt").unlink()  # the old VM's disk is gone
    restored = await client.resume(client.deserialize_session_state(payload))
    await restored.start()
    assert restored.state.sandbox_id != first_id
    assert (await restored.read(Path("notes/todo.txt"))).read() == b"keep me"
    await restored.aclose()
    await client.delete(restored)
    assert api.alive() == []


async def test_resume_reconnects_to_sleeping_sandbox_without_rewriting(api, ws_root):
    client = WBXSandboxClient()
    manifest = Manifest(root=ws_root, entries={})
    session = await client.create(
        manifest=manifest, options=WBXSandboxClientOptions(on_exit="sleep")
    )
    await session.start()
    await session.write(Path("state.txt"), io.BytesIO(b"v1"))
    await session.aclose()
    await client.delete(session)
    sid = session.state.sandbox_id
    assert api.slept == [sid] and api.sandboxes[sid].state == "sleeping"

    payload = client.serialize_session_state(session.state)
    resumed = await client.resume(client.deserialize_session_state(payload))
    assert resumed.state.sandbox_id == sid
    assert api.sandboxes[sid].state == "running"  # woken up
    await resumed.start()
    assert (await resumed.read(Path("state.txt"))).read() == b"v1"
    await resumed.aclose()
    await client.delete(resumed)
    assert len(api.sandboxes) == 1


async def test_named_sandbox_is_reused_and_kept(api, ws_root):
    from agents.sandbox.entries import File

    client = WBXSandboxClient()
    opts = WBXSandboxClientOptions(name="proj", on_exit="sleep")
    m1 = Manifest(root=ws_root, entries={"seed.txt": File(content=b"first")})
    s1 = await client.create(manifest=m1, options=opts)
    await s1.start()
    await s1.exec("echo edited > seed.txt")
    await s1.aclose()
    await client.delete(s1)

    m2 = Manifest(root=ws_root, entries={"seed.txt": File(content=b"second")})
    s2 = await client.create(manifest=m2, options=opts)
    assert s2.state.sandbox_id == s1.state.sandbox_id
    await s2.start()
    # the reused workspace is not overwritten by fresh creation inputs
    assert (await s2.read(Path("seed.txt"))).read() == b"edited\n"
    await s2.aclose()
    await client.delete(s2)
