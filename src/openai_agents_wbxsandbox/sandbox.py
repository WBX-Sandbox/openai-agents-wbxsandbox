"""WBX Sandbox backend for OpenAI Agents SDK sandbox agents.

Every sandbox is an isolated Linux microVM on WBX Sandbox (https://sandbox.wbx.cz). All API
calls go through the ``wbxsandbox`` Python SDK. The SDK is synchronous, so its calls run in a
worker thread and never block the event loop.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import shlex
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, TypeVar
from urllib.parse import urlsplit

import wbxsandbox
from agents.sandbox.errors import (
    ConfigurationError,
    ErrorCode,
    ExecTimeoutError,
    ExecTransportError,
    ExposedPortUnavailableError,
    WorkspaceArchiveReadError,
    WorkspaceArchiveWriteError,
    WorkspaceReadNotFoundError,
    WorkspaceStartError,
    WorkspaceWriteTypeError,
)
from agents.sandbox.manifest import Manifest
from agents.sandbox.session.base_sandbox_session import BaseSandboxSession
from agents.sandbox.session.dependencies import Dependencies
from agents.sandbox.session.manager import Instrumentation
from agents.sandbox.session.runtime_helpers import (
    RESOLVE_WORKSPACE_PATH_HELPER,
    RuntimeHelperScript,
)
from agents.sandbox.session.sandbox_client import BaseSandboxClient, BaseSandboxClientOptions
from agents.sandbox.session.sandbox_session import SandboxSession
from agents.sandbox.session.sandbox_session_state import SandboxSessionState
from agents.sandbox.session.tar_workspace import shell_tar_exclude_args
from agents.sandbox.snapshot import SnapshotBase, SnapshotSpec, resolve_snapshot
from agents.sandbox.types import ExecResult, ExposedPortEndpoint, User
from agents.sandbox.util.tar_utils import UnsafeTarMemberError, validate_tar_bytes
from agents.sandbox.workspace_paths import coerce_posix_path, posix_path_as_path, sandbox_path_str
from pydantic import ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger("openai_agents_wbxsandbox")

BACKEND_ID = "wbx"
DEFAULT_WBX_URL = "https://sandbox.wbx.cz"

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")
_RETRYABLE_STATUS = frozenset({0, 408, 425, 429, 500, 502, 503, 504})
# Sandbox states reported by the API in which the VM is not running but can be woken up.
_WAKEABLE_STATES = frozenset({"paused", "sleeping"})
_SHUTDOWN_ATTEMPTS = 3
# Extra time on top of a command's own timeout for the HTTP round trip.
_EXEC_TRANSPORT_MARGIN_S = 60.0
_FAST_OP_TIMEOUT_S = 60.0
_ARCHIVE_TIMEOUT_S = 600.0

T = TypeVar("T")


def _status(error: BaseException) -> int | None:
    status = getattr(error, "status", None)
    return status if isinstance(status, int) else None


def _retryable(error: BaseException) -> bool | None:
    status = _status(error)
    if status is None:
        return None
    return status in _RETRYABLE_STATUS


def _detail(error: BaseException) -> str:
    """Short provider message for error context. The API never echoes the API key."""
    message = getattr(error, "message", None)
    text = message if isinstance(message, str) else type(error).__name__
    return text[:500]


def _context(error: BaseException, **extra: object) -> dict[str, object]:
    context: dict[str, object] = {"backend": BACKEND_ID, **extra}
    status = _status(error)
    if status is not None:
        context["status"] = status
    context["detail"] = _detail(error)
    return context


def _is_not_found(error: BaseException) -> bool:
    return isinstance(error, wbxsandbox.SandboxError) and error.status == 404


class WBXSandboxClientOptions(BaseSandboxClientOptions):
    """Options for creating a WBX sandbox.

    These values may be serialized into run state, so do not put secrets here. Secrets belong
    in ``WBXSandboxClient(env=...)``, which is never serialized.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["wbx"] = "wbx"
    template: str = Field(default="base", min_length=1, max_length=128)
    """Base image of the sandbox, e.g. ``base`` (Debian, Python 3.12), ``node``, ``browser``."""
    name: str | None = None
    """Named sandbox. If your account already has a sandbox with this name, it is reused (and
    woken up if it sleeps) instead of creating a new one; its workspace is kept."""
    timeout_s: int = Field(default=3600, ge=60, le=7 * 24 * 3600)
    """Lifetime of the sandbox in seconds; the plan's maximum applies."""
    on_exit: Literal["kill", "sleep"] = "kill"
    """What happens when the run ends: ``kill`` deletes the sandbox with its data, ``sleep``
    keeps its files so the session can be resumed later (also used when the lifetime ends).
    Default: ``sleep`` for a named sandbox, otherwise ``kill``."""
    idle_timeout_s: int | None = Field(default=None, ge=60)
    """With ``on_exit="sleep"``: fall asleep after this many seconds without commands."""
    compute_class: Literal["economy", "balanced", "performance", "memory", "dedicated"] | None = (
        None
    )
    vcpu: int = Field(default=0, ge=0, le=64)
    """0 = the template's default."""
    mem_mb: int = Field(default=0, ge=0, le=262144)
    """0 = the template's default."""
    internet: bool = True
    """``False`` = no network at all."""
    allow: tuple[str, ...] = ()
    """Allowlist: only these domains or addresses are reachable."""
    block: tuple[str, ...] = ()
    """Additional blocked destinations."""
    ip: Literal["ipv4", "dual", "ipv6"] | None = None
    """IP version; ``dual`` and ``ipv6`` give the sandbox its own public IPv6 address."""
    env: dict[str, str] = Field(default_factory=dict)
    """Non-secret environment variables for every command."""
    metadata: dict[str, str] = Field(default_factory=dict)
    """Your own labels, returned when listing sandboxes."""
    exposed_ports: tuple[int, ...] = ()
    """Ports that may be resolved to a public HTTPS URL with an access token."""

    def __init__(
        self,
        template: str = "base",
        name: str | None = None,
        timeout_s: int = 3600,
        on_exit: Literal["kill", "sleep"] | None = None,
        idle_timeout_s: int | None = None,
        compute_class: Literal["economy", "balanced", "performance", "memory", "dedicated"]
        | None = None,
        vcpu: int = 0,
        mem_mb: int = 0,
        internet: bool = True,
        allow: tuple[str, ...] | list[str] = (),
        block: tuple[str, ...] | list[str] = (),
        ip: Literal["ipv4", "dual", "ipv6"] | None = None,
        env: Mapping[str, str] | None = None,
        metadata: Mapping[str, str] | None = None,
        exposed_ports: tuple[int, ...] | list[int] = (),
        *,
        type: Literal["wbx"] = "wbx",
        **unknown: Any,
    ) -> None:
        super().__init__(
            type=type,
            template=template,
            name=name,
            timeout_s=timeout_s,
            on_exit=on_exit,
            idle_timeout_s=idle_timeout_s,
            compute_class=compute_class,
            vcpu=vcpu,
            mem_mb=mem_mb,
            internet=internet,
            allow=tuple(allow),
            block=tuple(block),
            ip=ip,
            env=dict(env or {}),
            metadata=dict(metadata or {}),
            exposed_ports=tuple(exposed_ports),
            **unknown,  # rejected by extra="forbid"
        )

    @model_validator(mode="before")
    @classmethod
    def _default_on_exit(cls, data: Any) -> Any:
        # A named sandbox is meant to be kept: never delete a reused workspace by default.
        if isinstance(data, dict) and data.get("on_exit") is None:
            data = {**data, "on_exit": "sleep" if data.get("name") else "kill"}
        return data

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str | None) -> str | None:
        if value is not None and not _NAME_RE.match(value):
            raise ValueError(
                "name: 1-63 characters: lowercase letters, digits, '.', '_' and '-', "
                "starting with a letter or digit"
            )
        return value

    @field_validator("exposed_ports")
    @classmethod
    def _check_ports(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        for port in value:
            if not 1 <= port <= 65535:
                raise ValueError(f"exposed port {port} is outside 1-65535")
        if len(set(value)) != len(value):
            raise ValueError("exposed_ports contains duplicates")
        return value

    @field_validator("env", "metadata")
    @classmethod
    def _check_str_map(cls, value: dict[str, str]) -> dict[str, str]:
        for key in value:
            if not key:
                raise ValueError("keys must not be empty")
        return value

    @model_validator(mode="after")
    def _check_combinations(self) -> WBXSandboxClientOptions:
        if not self.internet and (self.allow or self.block):
            raise ValueError("allow/block cannot be combined with internet=False")
        if self.idle_timeout_s is not None and self.on_exit != "sleep":
            raise ValueError('idle_timeout_s requires on_exit="sleep"')
        if not self.internet and self.ip == "ipv6":
            raise ValueError('ip="ipv6" cannot be combined with internet=False')
        return self


class WBXSandboxSessionState(SandboxSessionState):
    """Serializable state of a WBX-backed session (identity and creation parameters).

    Fields that survive upgrades of this package: ``sandbox_id``, ``sandbox_name``,
    ``template``, ``on_exit`` and the creation parameters below. New fields are always
    added with defaults, so saved states from 0.1.x keep working.
    """

    type: Literal["wbx"] = "wbx"
    sandbox_id: str
    sandbox_name: str | None = None
    template: str = "base"
    on_exit: Literal["kill", "sleep"] = "kill"
    timeout_s: int = 3600
    idle_timeout_s: int | None = None
    compute_class: str | None = None
    vcpu: int = 0
    mem_mb: int = 0
    internet: bool = True
    allow: tuple[str, ...] = ()
    block: tuple[str, ...] = ()
    ip: str | None = None
    base_env: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)


class WBXSandboxSession(BaseSandboxSession):
    """A live session in one WBX sandbox."""

    state: WBXSandboxSessionState

    def __init__(
        self,
        *,
        state: WBXSandboxSessionState,
        sandbox: wbxsandbox.Sandbox,
        http: wbxsandbox.Client,
        secret_env: Mapping[str, str] | None = None,
    ) -> None:
        self.state = state
        self._sandbox = sandbox
        self._http = http
        self._secret_env = dict(secret_env or {})
        self._backend_released = False
        self._release_lock = asyncio.Lock()

    @property
    def sandbox_id(self) -> str:
        return self.state.sandbox_id

    async def _call(self, fn: Callable[..., T], *args: Any, wait: float, **kwargs: Any) -> T:
        """Run a blocking SDK call in a worker thread, waiting at most ``wait`` seconds."""
        return await asyncio.wait_for(asyncio.to_thread(fn, *args, **kwargs), timeout=wait)

    # ------------------------------------------------------------------ lifecycle

    async def _prepare_backend_workspace(self) -> None:
        root = sandbox_path_str(self.state.manifest.root)
        try:
            result = await self._run_raw(f"mkdir -p -- {shlex.quote(root)}", cwd=None, timeout=60)
        except Exception as e:
            raise WorkspaceStartError(
                path=Path(root),
                context=_context(e, reason="workspace_root_setup_failed"),
                cause=e,
                retryable=_retryable(e),
            ) from e
        if result.exit_code != 0:
            raise WorkspaceStartError(
                path=Path(root),
                context={
                    "backend": BACKEND_ID,
                    "reason": "workspace_root_nonzero_exit",
                    "exit_code": result.exit_code,
                    "stderr": result.stderr[:2000],
                },
            )

    async def _shutdown_backend(self) -> None:
        async with self._release_lock:
            if self._backend_released:
                return
            action = self._sandbox.sleep if self.state.on_exit == "sleep" else self._sandbox.kill
            last: BaseException | None = None
            for attempt in range(_SHUTDOWN_ATTEMPTS):
                try:
                    await self._call(action, wait=_FAST_OP_TIMEOUT_S)
                    last = None
                    break
                except Exception as e:
                    if _is_not_found(e):
                        last = None
                        break
                    last = e
                    if _retryable(e) is False:
                        break
                    await asyncio.sleep(1.0 * (attempt + 1))
            self._backend_released = True
            if last is not None:
                # Best effort: the sandbox lifetime (timeout_s) still ends it on the server.
                logger.warning(
                    "WBX sandbox %s could not be %s (%s); it ends when its lifetime expires",
                    self.state.sandbox_id,
                    "put to sleep" if self.state.on_exit == "sleep" else "deleted",
                    _detail(last),
                )

    async def running(self) -> bool:
        if self._backend_released:
            return False
        try:
            data = await self._call(
                self._http.request,
                "GET",
                f"/v1/sandboxes/{self.state.sandbox_id}",
                wait=_FAST_OP_TIMEOUT_S,
            )
        except Exception:
            return False
        return isinstance(data, dict) and data.get("state") == "running"

    # ------------------------------------------------------------------ commands

    def _env(self, manifest_env: Mapping[str, str]) -> dict[str, str]:
        return {**self.state.base_env, **manifest_env, **self._secret_env}

    async def _run_raw(
        self, cmd: str, *, cwd: str | None, timeout: float | None
    ) -> wbxsandbox.CommandResult:
        manifest_env = await self.state.manifest.environment.resolve()
        wait = (timeout + _EXEC_TRANSPORT_MARGIN_S) if timeout else 24 * 3600.0
        return await self._call(
            self._sandbox.run,
            cmd,
            wait=wait,
            env=self._env(manifest_env) or None,
            cwd=cwd,
            timeout=timeout,
        )

    async def _exec_internal(
        self,
        *command: str | Path,
        timeout: float | None = None,
    ) -> ExecResult:
        cmd = shlex.join(str(c) for c in command)
        if timeout is not None and timeout <= 0:
            timeout = 0.001
        try:
            result = await self._run_raw(
                cmd, cwd=sandbox_path_str(self.state.manifest.root), timeout=timeout
            )
        except asyncio.TimeoutError as e:
            raise ExecTimeoutError(command=command, timeout_s=timeout, cause=e) from e
        except Exception as e:
            raise ExecTransportError(
                command=command,
                context=_context(e),
                cause=e,
                retryable=_retryable(e),
            ) from e
        if result.timed_out:
            raise ExecTimeoutError(
                command=command, timeout_s=timeout, context={"backend": BACKEND_ID}
            )
        return ExecResult(
            stdout=result.stdout.encode("utf-8", errors="replace"),
            stderr=result.stderr.encode("utf-8", errors="replace"),
            exit_code=int(result.exit_code),
        )

    # ------------------------------------------------------------------ files

    def _runtime_helpers(self) -> tuple[RuntimeHelperScript, ...]:
        return (RESOLVE_WORKSPACE_PATH_HELPER,)

    async def _validate_path_access(self, path: Path | str, *, for_write: bool = False) -> Path:
        # Resolves symlinks inside the VM, so paths cannot escape the workspace.
        return await self._validate_remote_path_access(path, for_write=for_write)

    async def read(self, path: Path | str, *, user: str | User | None = None) -> io.IOBase:
        error_path = posix_path_as_path(coerce_posix_path(path))
        if user is not None:
            workspace_path = await self._check_read_with_exec(path, user=user)
        else:
            workspace_path = await self._validate_path_access(path)
        try:
            data = await self._call(
                self._sandbox.files.read,
                sandbox_path_str(workspace_path),
                wait=_ARCHIVE_TIMEOUT_S,
            )
        except Exception as e:
            if _is_not_found(e):
                raise WorkspaceReadNotFoundError(path=error_path, cause=e) from e
            raise WorkspaceArchiveReadError(
                path=error_path, context=_context(e), cause=e, retryable=_retryable(e)
            ) from e
        if isinstance(data, str):
            data = data.encode("utf-8")
        return io.BytesIO(bytes(data or b""))

    async def write(
        self,
        path: Path | str,
        data: io.IOBase,
        *,
        user: str | User | None = None,
    ) -> None:
        error_path = posix_path_as_path(coerce_posix_path(path))
        if user is not None:
            await self._check_write_with_exec(path, user=user)
        payload = data.read()
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        if not isinstance(payload, bytes | bytearray):
            raise WorkspaceWriteTypeError(path=error_path, actual_type=type(payload).__name__)
        workspace_path = await self._validate_path_access(path, for_write=True)
        try:
            await self._call(
                self._sandbox.files.write,
                sandbox_path_str(workspace_path),
                bytes(payload),
                wait=_ARCHIVE_TIMEOUT_S,
            )
        except Exception as e:
            raise WorkspaceArchiveWriteError(
                path=workspace_path, context=_context(e), cause=e, retryable=_retryable(e)
            ) from e
        if user is not None:
            # Files are uploaded by root; hand them over to the user who asked for the write.
            name = user.name if isinstance(user, User) else user
            target = sandbox_path_str(workspace_path)
            result = await self.exec("chown", f"{name}:", target, shell=False)
            if not result.ok():
                raise WorkspaceArchiveWriteError(
                    path=workspace_path,
                    context={
                        "backend": BACKEND_ID,
                        "reason": "chown_failed",
                        "stderr": result.stderr.decode("utf-8", errors="replace")[:2000],
                    },
                )

    # ------------------------------------------------------------------ workspace archive

    async def persist_workspace(self) -> io.IOBase:
        root = self._workspace_root_path()
        tar_path = f"/tmp/wbx-persist-{self.state.session_id.hex}-{uuid.uuid4().hex[:8]}.tar"
        excludes = " ".join(shell_tar_exclude_args(self._persist_workspace_skip_relpaths()))
        tar_cmd = f"tar {excludes} -C {shlex.quote(root.as_posix())} -cf {shlex.quote(tar_path)} ."
        try:
            result = await self._run_raw(tar_cmd, cwd=None, timeout=_ARCHIVE_TIMEOUT_S)
            if result.timed_out or result.exit_code != 0:
                raise WorkspaceArchiveReadError(
                    path=root,
                    context={
                        "backend": BACKEND_ID,
                        "reason": "tar_failed",
                        "exit_code": result.exit_code,
                        "stderr": result.stderr[:2000],
                    },
                    retryable=False,
                )
            raw = await self._call(self._sandbox.files.read, tar_path, wait=_ARCHIVE_TIMEOUT_S)
        except WorkspaceArchiveReadError:
            raise
        except Exception as e:
            raise WorkspaceArchiveReadError(
                path=root, context=_context(e), cause=e, retryable=_retryable(e)
            ) from e
        finally:
            await self._remove_tmp(tar_path)
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        return io.BytesIO(bytes(raw or b""))

    async def hydrate_workspace(self, data: io.IOBase) -> None:
        root = self._workspace_root_path()
        tar_path = f"/tmp/wbx-hydrate-{self.state.session_id.hex}-{uuid.uuid4().hex[:8]}.tar"
        payload = data.read()
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        if not isinstance(payload, bytes | bytearray):
            raise WorkspaceWriteTypeError(path=Path(tar_path), actual_type=type(payload).__name__)
        try:
            validate_tar_bytes(bytes(payload), allow_external_symlink_targets=False)
        except UnsafeTarMemberError as e:
            raise WorkspaceArchiveWriteError(
                path=root,
                context={
                    "backend": BACKEND_ID,
                    "reason": "unsafe_or_invalid_tar",
                    "member": e.member,
                    "detail": str(e),
                },
                cause=e,
            ) from e
        try:
            await self._call(
                self._sandbox.files.write, tar_path, bytes(payload), wait=_ARCHIVE_TIMEOUT_S
            )
            q_root = shlex.quote(root.as_posix())
            result = await self._run_raw(
                f"mkdir -p -- {q_root} && tar -C {q_root} -xf {shlex.quote(tar_path)}",
                cwd=None,
                timeout=_ARCHIVE_TIMEOUT_S,
            )
            if result.timed_out or result.exit_code != 0:
                raise WorkspaceArchiveWriteError(
                    path=root,
                    context={
                        "backend": BACKEND_ID,
                        "reason": "tar_extract_failed",
                        "exit_code": result.exit_code,
                        "stderr": result.stderr[:2000],
                    },
                )
        except WorkspaceArchiveWriteError:
            raise
        except Exception as e:
            raise WorkspaceArchiveWriteError(
                path=root, context=_context(e), cause=e, retryable=_retryable(e)
            ) from e
        finally:
            await self._remove_tmp(tar_path)

    async def _remove_tmp(self, path: str) -> None:
        try:
            await self._run_raw(f"rm -f -- {shlex.quote(path)}", cwd=None, timeout=60)
        except Exception:
            pass

    # ------------------------------------------------------------------ ports

    async def _resolve_exposed_port(self, port: int) -> ExposedPortEndpoint:
        try:
            info = await self._call(self._sandbox.get_url, port, wait=_FAST_OP_TIMEOUT_S)
        except Exception as e:
            raise ExposedPortUnavailableError(
                port=port,
                exposed_ports=self.state.exposed_ports,
                reason="backend_unavailable",
                context=_context(e),
                cause=e,
            ) from e
        url = info.get("url_with_token") or info.get("url") if isinstance(info, dict) else None
        split = urlsplit(url) if isinstance(url, str) else None
        if split is None or not split.hostname or split.scheme not in ("https", "http"):
            raise ExposedPortUnavailableError(
                port=port,
                exposed_ports=self.state.exposed_ports,
                reason="backend_unavailable",
                context={"backend": BACKEND_ID, "detail": "invalid_url"},
            )
        tls = split.scheme == "https"
        return ExposedPortEndpoint(
            host=split.hostname,
            port=split.port or (443 if tls else 80),
            tls=tls,
            query=split.query,
        )


class WBXSandboxClient(BaseSandboxClient[WBXSandboxClientOptions]):
    """Creates, resumes and deletes WBX sandboxes for the OpenAI Agents SDK runner.

    Credentials stay on this live object and are never serialized:

    * ``api_key`` – defaults to the ``WBXSANDBOX_API_KEY`` (or ``WBX_API_KEY``) variable;
    * ``url`` – defaults to ``WBXSANDBOX_URL`` or https://sandbox.wbx.cz;
    * ``env`` – secret environment variables for every command (e.g. API tokens).
    """

    backend_id = BACKEND_ID
    supports_default_options = True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        url: str | None = None,
        env: Mapping[str, str] | None = None,
        http_timeout: float = 120.0,
        instrumentation: Instrumentation | None = None,
        dependencies: Dependencies | None = None,
    ) -> None:
        key = api_key or os.environ.get("WBXSANDBOX_API_KEY") or os.environ.get("WBX_API_KEY", "")
        if not key:
            raise ConfigurationError(
                message=(
                    "WBX Sandbox API key is missing: set WBXSANDBOX_API_KEY "
                    "or pass WBXSandboxClient(api_key=...)"
                ),
                error_code=ErrorCode.SANDBOX_CONFIG_INVALID,
                op="start",
                context={"backend": BACKEND_ID},
            )
        self._api_key = key
        self._url = (url or os.environ.get("WBXSANDBOX_URL") or DEFAULT_WBX_URL).rstrip("/")
        self._http_timeout = http_timeout
        self._secret_env = dict(env or {})
        self._instrumentation = (
            instrumentation if instrumentation is not None else Instrumentation()
        )
        self._dependencies = dependencies

    def __repr__(self) -> str:  # never show the key
        return f"WBXSandboxClient(url={self._url!r})"

    def _http(self) -> wbxsandbox.Client:
        return wbxsandbox.Client(self._url, self._api_key, timeout=self._http_timeout)

    # ------------------------------------------------------------------ validation

    def _reject_unsupported_manifest(self, manifest: Manifest) -> None:
        mounts = manifest.mount_targets()
        if mounts:
            raise ConfigurationError(
                message=(
                    "WBX Sandbox does not support manifest mounts (S3, GCS, R2, ...) in this "
                    "version; copy the data into the workspace instead"
                ),
                error_code=ErrorCode.SANDBOX_CONFIG_INVALID,
                op="start",
                context={
                    "backend": BACKEND_ID,
                    "mount_paths": [path.as_posix() for _, path in mounts],
                },
            )
        host_grants = [g.path for g in manifest.extra_path_grants if g.host_path is not None]
        if host_grants:
            raise ConfigurationError(
                message="WBX Sandbox runs in a remote VM and cannot grant host filesystem paths",
                error_code=ErrorCode.SANDBOX_CONFIG_INVALID,
                op="start",
                context={"backend": BACKEND_ID, "paths": host_grants},
            )

    @staticmethod
    def _coerce_options(options: object) -> WBXSandboxClientOptions:
        if options is None:
            return WBXSandboxClientOptions()
        if isinstance(options, WBXSandboxClientOptions):
            return options
        if isinstance(options, dict):
            parsed = BaseSandboxClientOptions.parse(options)
            if isinstance(parsed, WBXSandboxClientOptions):
                return parsed
        raise ConfigurationError(
            message="WBXSandboxClient expects WBXSandboxClientOptions",
            error_code=ErrorCode.SANDBOX_CONFIG_INVALID,
            op="start",
            context={"backend": BACKEND_ID, "options_type": type(options).__name__},
        )

    # ------------------------------------------------------------------ backend calls

    async def _call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def _find_named(self, name: str) -> dict[str, Any] | None:
        listing = await self._call(self._http().request, "GET", "/v1/sandboxes")
        for item in listing or []:
            if isinstance(item, dict) and item.get("name") == name:
                return item
        return None

    async def _create_backend(
        self, params: WBXSandboxSessionState | WBXSandboxClientOptions
    ) -> wbxsandbox.Sandbox:
        name = params.sandbox_name if isinstance(params, WBXSandboxSessionState) else params.name
        base_env = params.base_env if isinstance(params, WBXSandboxSessionState) else params.env
        metadata = {"integration": "openai-agents", **params.metadata}
        try:
            return await self._call(
                wbxsandbox.Sandbox.create,
                template=params.template,
                timeout=params.timeout_s,
                metadata=metadata,
                vcpu=params.vcpu,
                mem_mb=params.mem_mb,
                url=self._url,
                api_key=self._api_key,
                internet=params.internet,
                allow=list(params.allow) or None,
                block=list(params.block) or None,
                env={**base_env, **self._secret_env} or None,
                compute_class=params.compute_class,
                ip=params.ip,
                name=name,
                on_timeout="pause" if params.on_exit == "sleep" else "kill",
                idle_timeout=params.idle_timeout_s,
            )
        except Exception as e:
            raise WorkspaceStartError(
                path=Path("/"),
                context=_context(e, reason="create_failed"),
                cause=e,
                retryable=_retryable(e),
                message=f"failed to create WBX sandbox: {_detail(e)}",
            ) from e

    async def _connect(self, sandbox_id: str) -> wbxsandbox.Sandbox:
        return await self._call(
            wbxsandbox.Sandbox.connect, sandbox_id, url=self._url, api_key=self._api_key
        )

    async def _wake(self, sandbox: wbxsandbox.Sandbox) -> None:
        if sandbox.state in _WAKEABLE_STATES:
            await self._call(sandbox.resume)

    async def _kill_quietly(self, sandbox: wbxsandbox.Sandbox) -> None:
        try:
            await self._call(sandbox.kill)
        except Exception as e:
            logger.warning("WBX sandbox %s could not be deleted: %s", sandbox.id, _detail(e))

    # ------------------------------------------------------------------ BaseSandboxClient

    async def create(
        self,
        *,
        snapshot: SnapshotSpec | SnapshotBase | None = None,
        manifest: Manifest | None = None,
        options: WBXSandboxClientOptions | None = None,
    ) -> SandboxSession:
        opts = self._coerce_options(options)
        if manifest is None:
            manifest = Manifest()
        self._validate_manifest_for_create(manifest)
        self._reject_unsupported_manifest(manifest)

        reused = False
        if opts.name:
            try:
                reused = await self._find_named(opts.name) is not None
            except Exception as e:
                raise WorkspaceStartError(
                    path=Path("/"),
                    context=_context(e, reason="list_failed"),
                    cause=e,
                    retryable=_retryable(e),
                    message=f"failed to look up WBX sandbox {opts.name!r}: {_detail(e)}",
                ) from e

        sandbox = await self._create_backend(opts)
        try:
            session_id = uuid.uuid4()
            state = WBXSandboxSessionState(
                session_id=session_id,
                manifest=manifest,
                snapshot=resolve_snapshot(snapshot, str(session_id)),
                exposed_ports=opts.exposed_ports,
                sandbox_id=sandbox.id,
                sandbox_name=opts.name,
                template=sandbox.template or opts.template,
                on_exit=opts.on_exit,
                timeout_s=opts.timeout_s,
                idle_timeout_s=opts.idle_timeout_s,
                compute_class=opts.compute_class,
                vcpu=opts.vcpu,
                mem_mb=opts.mem_mb,
                internet=opts.internet,
                allow=opts.allow,
                block=opts.block,
                ip=opts.ip,
                base_env=dict(opts.env),
                metadata=dict(opts.metadata),
            )
            if reused:
                await self._wake(sandbox)
            inner = WBXSandboxSession(
                state=state, sandbox=sandbox, http=self._http(), secret_env=self._secret_env
            )
            # A reused named sandbox keeps its workspace: never overwrite it with creation inputs.
            inner._set_start_state_preserved(reused, system=reused)
            return self._wrap_session(inner, instrumentation=self._instrumentation)
        except BaseException:
            if not reused:
                await self._kill_quietly(sandbox)
            raise

    async def delete(self, session: SandboxSession) -> SandboxSession:
        inner = session._inner
        if not isinstance(inner, WBXSandboxSession):
            raise TypeError("WBXSandboxClient.delete expects a WBX sandbox session")
        await inner._shutdown_backend()
        return session

    async def resume(self, state: SandboxSessionState) -> SandboxSession:
        if not isinstance(state, WBXSandboxSessionState):
            raise TypeError("WBXSandboxClient.resume expects a WBXSandboxSessionState")
        state.assert_path_grants_rebound()
        self._reject_unsupported_manifest(state.manifest)

        sandbox: wbxsandbox.Sandbox | None = None
        try:
            # The API only returns sandboxes of the account behind this client's API key.
            sandbox = await self._connect(state.sandbox_id)
        except Exception as e:
            if not _is_not_found(e):
                raise WorkspaceStartError(
                    path=Path("/"),
                    context=_context(e, reason="reconnect_failed"),
                    cause=e,
                    retryable=_retryable(e),
                    message=f"failed to reconnect to WBX sandbox: {_detail(e)}",
                ) from e

        preserved = sandbox is not None
        if sandbox is not None:
            try:
                await self._wake(sandbox)
            except Exception as e:
                raise WorkspaceStartError(
                    path=Path("/"),
                    context=_context(e, reason="wake_failed"),
                    cause=e,
                    retryable=_retryable(e),
                    message=f"failed to wake WBX sandbox: {_detail(e)}",
                ) from e
        else:
            # The sandbox is gone; a replacement is hydrated from state.snapshot on start().
            if state.sandbox_name:
                preserved = await self._find_named(state.sandbox_name) is not None
            sandbox = await self._create_backend(state)
            if preserved:
                await self._wake(sandbox)
            state.sandbox_id = sandbox.id
            if not preserved:
                state.workspace_root_ready = False

        inner = WBXSandboxSession(
            state=state, sandbox=sandbox, http=self._http(), secret_env=self._secret_env
        )
        inner._set_start_state_preserved(preserved, system=preserved)
        return self._wrap_session(inner, instrumentation=self._instrumentation)

    def deserialize_session_state(self, payload: dict[str, object]) -> SandboxSessionState:
        return self._deserialize_session_state_payload(payload, WBXSandboxSessionState)


__all__ = [
    "BACKEND_ID",
    "WBXSandboxClient",
    "WBXSandboxClientOptions",
    "WBXSandboxSession",
    "WBXSandboxSessionState",
]
