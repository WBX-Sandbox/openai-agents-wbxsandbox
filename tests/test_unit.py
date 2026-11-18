"""Options, state and error handling without a runner."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
import textwrap
from pathlib import Path

import pytest
from agents.sandbox import Manifest
from agents.sandbox.errors import (
    ConfigurationError,
    ExecTimeoutError,
    ExecTransportError,
    WorkspaceArchiveWriteError,
    WorkspaceReadNotFoundError,
    WorkspaceStartError,
)
from agents.sandbox.session import BaseSandboxClientOptions, SandboxSessionState
from pydantic import ValidationError

from openai_agents_wbxsandbox import (
    WBXSandboxClient,
    WBXSandboxClientOptions,
    WBXSandboxSessionState,
)

# ------------------------------------------------------------------ options


def test_options_defaults_and_type():
    o = WBXSandboxClientOptions()
    assert o.type == "wbx"
    assert o.template == "base"
    assert o.on_exit == "kill"
    assert o.timeout_s == 3600


def test_options_round_trip_through_registry():
    o = WBXSandboxClientOptions(
        template="node",
        name="proj-1",
        on_exit="sleep",
        idle_timeout_s=600,
        compute_class="performance",
        allow=["pypi.org", "files.pythonhosted.org"],
        ip="dual",
        env={"MODE": "test"},
        metadata={"team": "a"},
        exposed_ports=[8000, 3000],
    )
    payload = json.loads(o.model_dump_json())
    assert payload["type"] == "wbx"
    parsed = BaseSandboxClientOptions.parse(payload)
    assert isinstance(parsed, WBXSandboxClientOptions)
    assert parsed == o


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "Has Space"},
        {"name": "-starts-with-dash"},
        {"name": "x" * 64},
        {"internet": False, "allow": ["pypi.org"]},
        {"internet": False, "ip": "ipv6"},
        {"idle_timeout_s": 300},  # needs on_exit="sleep"
        {"timeout_s": 10},
        {"vcpu": -1},
        {"exposed_ports": [0]},
        {"exposed_ports": [8000, 8000]},
        {"compute_class": "turbo"},
        {"on_exit": "pause"},
        {"ip": "ipv5"},
        {"unknown_option": 1},
        {"type": "docker"},
        {"env": {"": "x"}},
    ],
)
def test_options_reject_invalid(kwargs):
    with pytest.raises((ValidationError, TypeError)):
        WBXSandboxClientOptions(**kwargs)


def test_options_parse_rejects_unknown_fields_in_payload():
    with pytest.raises(ValidationError):
        BaseSandboxClientOptions.parse({"type": "wbx", "template": "base", "image": "x"})


def test_options_are_frozen():
    o = WBXSandboxClientOptions()
    with pytest.raises(ValidationError):
        o.template = "node"  # type: ignore[misc]


# ------------------------------------------------------------------ state


def _state(**kw) -> WBXSandboxSessionState:
    from agents.sandbox import resolve_snapshot

    return WBXSandboxSessionState(
        manifest=Manifest(root="/workspace"),
        snapshot=resolve_snapshot(None, "s1"),
        sandbox_id="sb_abc",
        **kw,
    )


def test_state_round_trip_via_client(api):
    client = WBXSandboxClient()
    state = _state(sandbox_name="proj", on_exit="sleep", base_env={"A": "1"}, exposed_ports=(80,))
    payload = client.serialize_session_state(state)
    assert payload["type"] == "wbx"
    assert "test-key-not-real" not in json.dumps(payload)
    restored = client.deserialize_session_state(json.loads(json.dumps(payload)))
    assert isinstance(restored, WBXSandboxSessionState)
    assert restored.sandbox_id == "sb_abc"
    assert restored.sandbox_name == "proj"
    assert restored.on_exit == "sleep"
    assert restored.exposed_ports == (80,)
    # the generic parser finds the type through the registry
    assert isinstance(
        SandboxSessionState.parse(json.loads(json.dumps(payload))), WBXSandboxSessionState
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.pop("sandbox_id"),
        lambda p: p.update(sandbox_id=None),
        lambda p: p.update(on_exit="explode"),
        lambda p: p.update(manifest="not a manifest"),
    ],
)
def test_state_rejects_malformed_payload(api, mutate):
    client = WBXSandboxClient()
    payload = client.serialize_session_state(_state())
    mutate(payload)
    with pytest.raises(Exception):  # noqa: B017 – the SDK redacts the concrete error
        client.deserialize_session_state(payload)


def test_unknown_state_type_is_rejected():
    with pytest.raises(Exception):  # noqa: B017
        SandboxSessionState.parse({"type": "wbx-unknown", "sandbox_id": "x"})


def test_round_trip_in_fresh_interpreter(api, tmp_path):
    """Serialized options and state parse in a new process that only imports the package."""
    client = WBXSandboxClient()
    state_payload = client.serialize_session_state(_state(sandbox_name="fresh"))
    opts_payload = WBXSandboxClientOptions(template="go", allow=["proxy.golang.org"]).model_dump(
        mode="json"
    )
    (tmp_path / "state.json").write_text(json.dumps(state_payload))
    (tmp_path / "opts.json").write_text(json.dumps(opts_payload))
    code = textwrap.dedent(
        """
        import json, sys
        import openai_agents_wbxsandbox  # registers the types
        from agents.sandbox.session import BaseSandboxClientOptions, SandboxSessionState
        s = SandboxSessionState.parse(json.load(open(sys.argv[1])))
        o = BaseSandboxClientOptions.parse(json.load(open(sys.argv[2])))
        assert type(s).__name__ == "WBXSandboxSessionState", type(s)
        assert s.sandbox_name == "fresh" and s.sandbox_id == "sb_abc"
        assert type(o).__name__ == "WBXSandboxClientOptions" and o.template == "go"
        try:
            SandboxSessionState.parse({"type": "wbx"})
        except Exception:
            print("malformed-rejected")
        print("ok")
        """
    )
    out = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            code,
            str(tmp_path / "state.json"),
            str(tmp_path / "opts.json"),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["malformed-rejected", "ok"]


# ------------------------------------------------------------------ client


def test_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("WBXSANDBOX_API_KEY", raising=False)
    monkeypatch.delenv("WBX_API_KEY", raising=False)
    with pytest.raises(ConfigurationError):
        WBXSandboxClient()


def test_client_reads_wbx_api_key_fallback(monkeypatch):
    monkeypatch.delenv("WBXSANDBOX_API_KEY", raising=False)
    monkeypatch.setenv("WBX_API_KEY", "fallback-key")
    assert WBXSandboxClient()._api_key == "fallback-key"


def test_client_repr_hides_key(api):
    assert "test-key-not-real" not in repr(WBXSandboxClient())


async def test_mounts_are_rejected_before_creating(api):
    from agents.sandbox.entries import InContainerMountStrategy, RcloneMountPattern, S3Mount

    mount = S3Mount(
        bucket="b", mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern())
    )
    manifest = Manifest(entries={"data": mount})
    with pytest.raises(ConfigurationError, match="mounts"):
        await WBXSandboxClient().create(manifest=manifest, options=WBXSandboxClientOptions())
    assert api.sandboxes == {}


async def test_wrong_options_type_is_rejected(api):
    with pytest.raises(ConfigurationError):
        await WBXSandboxClient().create(options=object())  # type: ignore[arg-type]
    assert api.sandboxes == {}


async def test_create_failure_is_translated(api):
    api.fail_create = (402, "not enough credit")
    with pytest.raises(WorkspaceStartError) as ei:
        await WBXSandboxClient().create(options=WBXSandboxClientOptions())
    assert ei.value.context["status"] == 402
    assert ei.value.retryable is False
    assert "not enough credit" in str(ei.value)


async def test_create_sends_options_and_secret_env(api):
    client = WBXSandboxClient(env={"TOKEN": "secret"})
    opts = WBXSandboxClientOptions(
        template="node",
        allow=["pypi.org"],
        ip="dual",
        env={"MODE": "x"},
        on_exit="sleep",
        idle_timeout_s=900,
        compute_class="economy",
    )
    session = await client.create(options=opts)
    req = api.sandboxes[session.state.sandbox_id].request
    assert req["template"] == "node"
    assert req["network"] == {"mode": "allowlist", "allow": ["pypi.org"], "block": [], "ip": "dual"}
    assert req["env"] == {"MODE": "x", "TOKEN": "secret"}
    assert req["on_timeout"] == "pause"
    assert req["idle_timeout_s"] == 900
    assert req["compute_class"] == "economy"
    assert req["metadata"]["integration"] == "openai-agents"
    # the secret never reaches serialized state
    assert "secret" not in json.dumps(client.serialize_session_state(session.state))
    await client.delete(session)


async def test_default_options_when_none(api):
    client = WBXSandboxClient()
    session = await client.create(options=None)
    assert session.state.template == "base"
    await client.delete(session)
    assert api.killed == [session.state.sandbox_id]


# ------------------------------------------------------------------ session


async def test_exec_files_and_errors(api, ws_root):
    client = WBXSandboxClient()
    session = await client.create(manifest=Manifest(root=ws_root), options=None)
    async with session:
        r = await session.exec("echo", "hello world")
        assert r.ok() and r.stdout == b"hello world\n"
        r = await session.exec("pwd")
        assert r.stdout.decode().strip() == ws_root
        r = await session.exec("exit 3")
        assert r.exit_code == 3

        await session.write(Path("a/b.txt"), io.BytesIO(b"\x00\x01binary"))
        assert (await session.read(Path("a/b.txt"))).read() == b"\x00\x01binary"
        with pytest.raises(WorkspaceReadNotFoundError):
            await session.read(Path("missing.txt"))

        with pytest.raises(ExecTimeoutError):
            await session.exec("sleep 5", timeout=0.5)

        api.fail_exec_matching = "boom-marker"
        with pytest.raises(ExecTransportError) as ei:
            await session.exec("echo boom-marker")
        assert ei.value.retryable is True
        assert ei.value.context["status"] == 502
        api.fail_exec_matching = None
    await client.delete(session)
    assert api.alive() == []


async def test_paths_cannot_escape_workspace(api, ws_root):
    client = WBXSandboxClient()
    session = await client.create(manifest=Manifest(root=ws_root), options=None)
    async with session:
        await session.exec(f"ln -s /etc {ws_root}/etc-link")
        with pytest.raises(Exception):  # noqa: B017 – escape through a symlink
            await session.read(Path("etc-link/hostname"))
        with pytest.raises(Exception):  # noqa: B017 – escape with ..
            await session.read(Path("../../etc/hostname"))
        with pytest.raises(Exception):  # noqa: B017 – absolute path outside the root
            await session.write(Path("/etc/wbx-should-not-exist"), io.BytesIO(b"x"))
    await client.delete(session)
    assert not Path("/etc/wbx-should-not-exist").exists()


async def test_persist_and_hydrate_workspace(api, ws_root):
    client = WBXSandboxClient()
    session = await client.create(manifest=Manifest(root=ws_root), options=None)
    async with session:
        await session.write(Path("src/main.py"), io.BytesIO(b"print(1)\n"))
        archive = (await session.persist_workspace()).read()
        names = tarfile.open(fileobj=io.BytesIO(archive)).getnames()
        assert "./src/main.py" in names
        assert not any(n.startswith(ws_root.lstrip("/")) for n in names)  # relative members
        await session.exec("rm -rf src")
        await session.hydrate_workspace(io.BytesIO(archive))
        assert (await session.read(Path("src/main.py"))).read() == b"print(1)\n"

        evil = io.BytesIO()
        with tarfile.open(fileobj=evil, mode="w") as tf:
            info = tarfile.TarInfo("../escape.txt")
            info.size = 1
            tf.addfile(info, io.BytesIO(b"x"))
        with pytest.raises(WorkspaceArchiveWriteError):
            await session.hydrate_workspace(io.BytesIO(evil.getvalue()))
    await client.delete(session)


async def test_exposed_port_endpoint(api):
    client = WBXSandboxClient()
    session = await client.create(options=WBXSandboxClientOptions(exposed_ports=(8000,)))
    ep = await session.resolve_exposed_port(8000)
    assert ep.tls and ep.port == 443
    assert ep.host.endswith(".wbxbox.net")
    assert ep.query == "wbx_token=tok123"
    assert ep.url_for("http").startswith("https://8000-")
    from agents.sandbox.errors import ExposedPortUnavailableError

    with pytest.raises(ExposedPortUnavailableError):
        await session.resolve_exposed_port(9000)  # not configured
    await client.delete(session)


async def test_shutdown_retries_and_is_idempotent(api):
    client = WBXSandboxClient()
    session = await client.create(options=None)
    api.fail_kill_times = 1
    await client.delete(session)
    await client.delete(session)
    assert api.killed == [session.state.sandbox_id]
    assert [c for c in api.calls if c[0] == "DELETE"].__len__() == 2  # one failure, one success


async def test_on_exit_sleep_keeps_sandbox(api):
    client = WBXSandboxClient()
    session = await client.create(options=WBXSandboxClientOptions(on_exit="sleep"))
    await client.delete(session)
    assert api.killed == []
    assert api.slept == [session.state.sandbox_id]
