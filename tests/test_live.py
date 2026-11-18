"""Tests against the real WBX Sandbox service.

    WBXSANDBOX_API_KEY=... pytest -m live

Every test deletes the sandboxes it creates (also when it fails).
"""

from __future__ import annotations

import http.cookiejar
import io
import json
import os
import time
import urllib.request
import uuid
from pathlib import Path

import pytest
import wbxsandbox
from agents import Runner
from agents.run import RunConfig
from agents.sandbox import LocalSnapshotSpec, SandboxAgent, SandboxRunConfig
from agents.sandbox.capabilities import Filesystem, Shell
from agents.sandbox.errors import ExecTimeoutError
from agents.testing import ScriptedModel, assistant_message, function_call

from openai_agents_wbxsandbox import WBXSandboxClient, WBXSandboxClientOptions

pytestmark = pytest.mark.live


@pytest.fixture
async def client():
    c = WBXSandboxClient()
    created: list = []
    c._test_created = created  # sessions to clean up
    yield c
    for session in created:
        try:
            await session.aclose()
        except Exception:  # noqa: BLE001
            pass
        try:
            sb = wbxsandbox.Sandbox.connect(session.state.sandbox_id)
            sb.kill()
        except wbxsandbox.SandboxError:
            pass


async def _start(client, **opts):
    snapshot = opts.pop("snapshot", None)
    manifest = opts.pop("manifest", None)
    session = await client.create(
        snapshot=snapshot, manifest=manifest, options=WBXSandboxClientOptions(**opts)
    )
    client._test_created.append(session)
    await session.start()
    return session


async def _sh(session, cmd: str, timeout: float = 60):
    r = await session.exec(cmd, timeout=timeout)
    return r.exit_code, r.stdout.decode(), r.stderr.decode()


PROBE = """
import socket, sys, urllib.request
kind, target = sys.argv[1], sys.argv[2]
try:
    if kind == "tcp":
        host, port = target.rsplit(":", 1)
        socket.create_connection((host, int(port)), timeout=5).close()
    elif kind == "tcp6":
        s = socket.create_connection((target, 443), timeout=10)
        print(s.getpeername()[0])
    else:
        urllib.request.urlopen(target, timeout=10)
    print("OPEN")
except Exception as e:
    print("CLOSED", type(e).__name__)
"""


async def _probe(session, kind: str, target: str) -> str:
    """OPEN or CLOSED: can the sandbox reach `target` (tcp host:port, tcp6 host, or http URL)?"""
    await session.write(Path(".probe.py"), io.BytesIO(PROBE.encode()))
    _, out, _ = await _sh(session, f"python3 .probe.py {kind} {target}")
    return out


# ------------------------------------------------------------------ isolation


async def test_isolation_own_kernel_and_no_private_networks(client):
    s = await _start(client)
    code, out, _ = await _sh(s, "id -u; uname -r; test -e /.dockerenv && echo docker || echo vm")
    uid, kernel, kind = out.split()
    assert uid == "0" and kind == "vm" and kernel
    # private networks, the cloud metadata address and the hosts' VLAN are unreachable
    for target in ("169.254.169.254:80", "10.77.0.1:22", "192.168.1.1:80", "172.16.0.1:80"):
        assert "CLOSED" in await _probe(s, "tcp", target), f"{target} reachable"
    assert "CLOSED" in await _probe(s, "tcp", "smtp.gmail.com:25"), "SMTP (25) must be blocked"
    assert "OPEN" in await _probe(s, "http", "https://pypi.org/simple/")


async def test_two_sandboxes_do_not_share_files(client):
    a = await _start(client)
    b = await _start(client)
    await a.write(Path("secret.txt"), io.BytesIO(b"only in a"))
    code, out, _ = await _sh(
        b, "ls /workspace; test -e /workspace/secret.txt && echo LEAK || echo none"
    )
    assert "none" in out and "LEAK" not in out
    assert a.state.sandbox_id != b.state.sandbox_id


# ------------------------------------------------------------------ filesystem


async def test_filesystem_binary_round_trip_and_confinement(client):
    s = await _start(client)
    blob = os.urandom(1024 * 1024)
    await s.write(Path("data/blob.bin"), io.BytesIO(blob))
    assert (await s.read(Path("data/blob.bin"))).read() == blob
    code, out, _ = await _sh(s, "sha256sum data/blob.bin")
    import hashlib

    assert out.split()[0] == hashlib.sha256(blob).hexdigest()
    await s.exec("ln -s /etc /workspace/etc-link")
    with pytest.raises(Exception):  # noqa: B017
        await s.read(Path("etc-link/passwd"))
    with pytest.raises(Exception):  # noqa: B017
        await s.read(Path("../etc/passwd"))
    with pytest.raises(Exception):  # noqa: B017
        await s.write(Path("/etc/wbx-escape"), io.BytesIO(b"x"))
    code, _, _ = await _sh(s, "test ! -e /etc/wbx-escape")
    assert code == 0


async def test_workspace_archive_round_trip(client):
    s = await _start(client)
    await s.exec("mkdir -p src && echo 'print(42)' > src/app.py && ln -s app.py src/link.py")
    archive = (await s.persist_workspace()).read()
    await s.exec("rm -rf src")
    await s.hydrate_workspace(io.BytesIO(archive))
    code, out, _ = await _sh(s, "python3 src/link.py")
    assert out.strip() == "42"


# ------------------------------------------------------------------ network


async def test_network_none(client):
    s = await _start(client, internet=False)
    assert "CLOSED" in await _probe(s, "http", "https://pypi.org/simple/")
    assert "CLOSED" in await _probe(s, "tcp", "1.1.1.1:443")


async def test_network_allowlist(client):
    s = await _start(client, allow=("pypi.org",))
    assert "OPEN" in await _probe(s, "http", "https://pypi.org/simple/")
    assert "CLOSED" in await _probe(s, "http", "https://example.com/")


async def test_network_ipv6(client):
    s = await _start(client, ip="dual")
    # a global IPv6 address on eth0 (the base template has no `ip` command)
    _, out, _ = await _sh(s, "grep -v -e '^fe80' -e '^0000' /proc/net/if_inet6 | grep -c eth0")
    assert int(out.strip() or 0) >= 1
    out = await _probe(s, "tcp6", "ipv6.google.com")
    assert "OPEN" in out and ":" in out


# ------------------------------------------------------------------ processes


async def test_processes_timeout_and_background(client):
    s = await _start(client)
    started = time.monotonic()
    with pytest.raises(ExecTimeoutError):
        await s.exec("sleep 60", timeout=2)
    assert time.monotonic() - started < 30
    _, out, _ = await _sh(s, "pgrep -x sleep >/dev/null && echo alive || echo killed")
    assert "killed" in out  # the timed-out command was killed
    await _sh(s, "nohup sleep 300 >/dev/null 2>&1 &")
    _, out, _ = await _sh(s, "pgrep -x sleep >/dev/null && echo alive || echo gone")
    assert "alive" in out


async def test_delete_removes_the_sandbox(client):
    s = await _start(client)
    sid = s.state.sandbox_id
    await s.aclose()
    await client.delete(s)
    with pytest.raises(wbxsandbox.SandboxError) as ei:
        wbxsandbox.Sandbox.connect(sid)
    assert ei.value.status == 404


# ------------------------------------------------------------------ sleep / resume


async def test_sleep_and_resume_keeps_files(client):
    s = await _start(client, on_exit="sleep")
    await s.write(Path("progress.json"), io.BytesIO(b'{"step": 3}'))
    await _sh(s, "nohup sleep 600 >/dev/null 2>&1 &")
    await s.aclose()
    await client.delete(s)  # puts it to sleep
    sid = s.state.sandbox_id
    state = wbxsandbox.Sandbox.connect(sid).state
    assert state in ("sleeping", "falling_asleep"), state

    payload = json.loads(json.dumps(client.serialize_session_state(s.state)))
    resumed = await client.resume(client.deserialize_session_state(payload))
    client._test_created.append(resumed)
    assert resumed.state.sandbox_id == sid
    await resumed.start()
    assert (await resumed.read(Path("progress.json"))).read() == b'{"step": 3}'
    _, out, _ = await _sh(resumed, "pgrep -x sleep >/dev/null && echo running || echo stopped")
    assert "stopped" in out  # programs do not survive sleep, files do
    resumed.state.on_exit = "kill"


async def test_resume_after_sandbox_is_gone_restores_snapshot(client, tmp_path):
    s = await _start(client, snapshot=LocalSnapshotSpec(base_path=tmp_path))
    await s.write(Path("keep.txt"), io.BytesIO(b"from snapshot"))
    await s.aclose()  # persists the workspace snapshot locally
    await client.delete(s)
    payload = json.loads(json.dumps(client.serialize_session_state(s.state)))
    resumed = await client.resume(client.deserialize_session_state(payload))
    client._test_created.append(resumed)
    assert resumed.state.sandbox_id != s.state.sandbox_id
    await resumed.start()
    assert (await resumed.read(Path("keep.txt"))).read() == b"from snapshot"


async def test_named_sandbox_reuse(client):
    name = f"oa-test-{uuid.uuid4().hex[:8]}"
    a = await _start(client, name=name, on_exit="sleep")
    await a.write(Path("mark.txt"), io.BytesIO(b"first run"))
    await a.aclose()
    await client.delete(a)
    b = await _start(client, name=name)
    assert b.state.sandbox_id == a.state.sandbox_id
    assert (await b.read(Path("mark.txt"))).read() == b"first run"


# ------------------------------------------------------------------ ports


async def test_exposed_port_url(client):
    s = await _start(client, exposed_ports=(8000,))
    await _sh(s, "mkdir -p site && echo wbx-port-ok > site/index.html")
    await _sh(s, "cd site && nohup python3 -m http.server 8000 --bind 0.0.0.0 >/dev/null 2>&1 &")
    time.sleep(2)
    ep = await s.resolve_exposed_port(8000)
    url = ep.url_for("http")
    assert url.startswith("https://") and ep.query
    # the token moves to a cookie on the first request (redirect), like in a browser
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )
    body = ""
    for _ in range(10):
        try:
            body = opener.open(url, timeout=15).read().decode()
            break
        except Exception:  # noqa: BLE001
            time.sleep(2)
    assert "wbx-port-ok" in body


# ------------------------------------------------------------------ runner


async def test_runner_end_to_end_with_scripted_model(client):
    model = ScriptedModel(
        [
            [
                function_call(
                    "exec_command", {"cmd": "python3 --version > v.txt; cat v.txt"}, call_id="c1"
                )
            ],
            [assistant_message("done")],
        ]
    )
    agent = SandboxAgent(
        name="live", instructions="x", model=model, capabilities=[Shell(), Filesystem()]
    )
    before = {s.id for s in wbxsandbox.Sandbox.list()}
    result = await Runner.run(
        agent, "go", run_config=RunConfig(sandbox=SandboxRunConfig(client=client))
    )
    assert result.final_output == "done"
    assert "Python 3" in json.dumps(model.calls[1].input, default=str)
    after = {s.id for s in wbxsandbox.Sandbox.list()}
    assert after <= before  # the run's sandbox was deleted
