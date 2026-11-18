"""In-memory stand-in for the WBX Sandbox REST API, installed below the ``wbxsandbox`` SDK.

Commands really run (``sh -c``) on the test machine and files are real local files, so the SDK's
runtime helpers, path checks and tar archives work exactly as in a VM. Only orchestration is
tested this way; isolation, network and process properties are covered by the live tests.
"""

from __future__ import annotations

import itertools
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import wbxsandbox


@dataclass
class FakeSandbox:
    id: str
    template: str
    name: str | None
    state: str = "running"
    on_timeout: str = "kill"
    env: dict[str, str] = field(default_factory=dict)
    request: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "template": self.template, "state": self.state}
        if self.name:
            d["name"] = self.name
        return d


class FakeAPI:
    def __init__(self) -> None:
        self.sandboxes: dict[str, FakeSandbox] = {}
        self.calls: list[tuple[str, str]] = []
        self.killed: list[str] = []
        self.slept: list[str] = []
        self.exec_log: list[str] = []
        self.lock = threading.Lock()
        self._seq = itertools.count(1)
        # fault injection
        self.fail_create: tuple[int, str] | None = None
        self.fail_exec_matching: str | None = None
        self.fail_exec_status: int = 502
        self.fail_kill_times: int = 0
        self.exec_delay_s: float = 0.0

    # -------------------------------------------------------------- helpers

    def alive(self) -> list[str]:
        return [i for i, s in self.sandboxes.items() if s.state != "deleted"]

    def _get(self, sid: str) -> FakeSandbox:
        sb = self.sandboxes.get(sid)
        if sb is None or sb.state == "deleted":
            raise wbxsandbox.SandboxError(404, "sandbox not found")
        return sb

    # -------------------------------------------------------------- dispatcher

    def request(self, method: str, path: str, body: Any = None, query: Any = None) -> Any:
        with self.lock:
            self.calls.append((method, path))
        parts = path.strip("/").split("/")  # v1 sandboxes {id} {op}
        if parts[:2] != ["v1", "sandboxes"]:
            raise wbxsandbox.SandboxError(404, "not found")
        if len(parts) == 2:
            if method == "POST":
                return self._create(body)
            if method == "GET":
                return [s.as_dict() for s in self.sandboxes.values() if s.state != "deleted"]
        sid = parts[2]
        op = "/".join(parts[3:])
        if op == "" and method == "GET":
            return self._get(sid).as_dict()
        if op == "" and method == "DELETE":
            if self.fail_kill_times > 0:
                self.fail_kill_times -= 1
                raise wbxsandbox.SandboxError(503, "temporarily unavailable")
            sb = self._get(sid)
            sb.state = "deleted"
            self.killed.append(sid)
            return None
        sb = self._get(sid)
        if op == "exec":
            return self._exec(sb, body)
        if op == "files" and method == "GET":
            p = query["path"]
            if not os.path.isfile(p):
                raise wbxsandbox.SandboxError(404, "no such file")
            with open(p, "rb") as f:
                return f.read()
        if op == "files" and method == "PUT":
            p = query["path"]
            os.makedirs(os.path.dirname(p) or "/", exist_ok=True)
            with open(p, "wb") as f:
                f.write(body)
            return {"ok": True}
        if op == "sleep":
            sb.state = "sleeping"
            self.slept.append(sid)
            return {"state": "sleeping"}
        if op == "resume":
            sb.state = "running"
            return {"state": "running"}
        if op == "url":
            port = query["port"]
            token = "tok123"
            base = f"https://{port}-n1-{sid}.wbxbox.net/"
            return {"url": base, "url_with_token": f"{base}?wbx_token={token}", "token": token}
        raise wbxsandbox.SandboxError(404, f"unknown {method} {path}")

    def _create(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.fail_create:
            status, msg = self.fail_create
            raise wbxsandbox.SandboxError(status, msg)
        name = body.get("name")
        if name:
            for s in self.sandboxes.values():
                if s.name == name and s.state != "deleted":
                    s.state = "running"
                    return s.as_dict()
        sid = f"sb_fake_{next(self._seq):04d}"
        sb = FakeSandbox(
            id=sid,
            template=body.get("template", "base"),
            name=name,
            on_timeout=body.get("on_timeout", "kill"),
            env=dict(body.get("env") or {}),
            request=dict(body),
        )
        self.sandboxes[sid] = sb
        return sb.as_dict()

    def _exec(self, sb: FakeSandbox, body: dict[str, Any]) -> dict[str, Any]:
        cmd = body["cmd"]
        with self.lock:
            self.exec_log.append(cmd)
        if self.fail_exec_matching and self.fail_exec_matching in cmd:
            raise wbxsandbox.SandboxError(self.fail_exec_status, "node unavailable")
        if sb.state == "sleeping":
            sb.state = "running"  # like the real service: a command wakes the sandbox
        if self.exec_delay_s:
            time.sleep(self.exec_delay_s)
        timeout_ms = body.get("timeout_ms") or 0
        env = {**os.environ, **sb.env, **(body.get("env") or {})}
        cwd = body.get("cwd")
        if cwd and not os.path.isdir(cwd):
            raise wbxsandbox.SandboxError(400, f"cannot start the command: no such directory {cwd}")
        started = time.monotonic()
        try:
            p = subprocess.run(
                ["/bin/sh", "-c", cmd],
                cwd=cwd or "/",
                env=env,
                capture_output=True,
                timeout=(timeout_ms / 1000) if timeout_ms else None,
                input=(body.get("stdin") or "").encode(),
            )
            code, out, err, timed_out = p.returncode, p.stdout, p.stderr, False
        except subprocess.TimeoutExpired as e:
            code, out, err, timed_out = -1, e.stdout or b"", e.stderr or b"", True
        return {
            "exit_code": code,
            "stdout": out.decode("utf-8", "replace"),
            "stderr": err.decode("utf-8", "replace"),
            "duration_ms": int((time.monotonic() - started) * 1000),
            "timed_out": timed_out,
            "truncated": False,
        }


def install(monkeypatch: Any) -> FakeAPI:
    api = FakeAPI()

    def request(self, method, path, body=None, query=None, timeout=None, stream=False):  # noqa: ARG001
        assert self.api_key, "the SDK client must carry an API key"
        return api.request(method, path, body, query)

    monkeypatch.setattr(wbxsandbox.Client, "request", request)
    return api
