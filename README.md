# openai-agents-wbxsandbox

[WBX Sandbox](https://sandbox.wbx.cz) as a hosted sandbox for
[OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) sandbox agents.
Each sandbox is an isolated Linux microVM with its own kernel, disk and network policy. The
agent's shell and file tools run there, not on your machine.

```bash
pip install openai-agents-wbxsandbox
export WBXSANDBOX_API_KEY=...        # API key from https://sandbox.wbx.cz (Settings → API keys)
export OPENAI_API_KEY=...
```

## Quick start

```python
import asyncio

from agents import Runner
from agents.run import RunConfig
from agents.sandbox import SandboxAgent, SandboxRunConfig
from openai_agents_wbxsandbox import WBXSandboxClient, WBXSandboxClientOptions

agent = SandboxAgent(
    name="Workspace assistant",
    instructions="Use the shell to answer. Work in the workspace.",
)

run_config = RunConfig(
    sandbox=SandboxRunConfig(
        client=WBXSandboxClient(),          # API key from WBXSANDBOX_API_KEY
        options=WBXSandboxClientOptions(template="base"),
    ),
)


async def main() -> None:
    result = await Runner.run(
        agent,
        "Which Python version is installed? Write it to version.txt.",
        run_config=run_config,
    )
    print(result.final_output)


asyncio.run(main())
```

The runner creates the sandbox, gives the agent its shell and file tools, and deletes the
sandbox when the run ends.

## Supported versions

| Package | Tested range |
| --- | --- |
| `openai-agents` | `>=0.25.0,<0.26` |
| `wbxsandbox` | `>=0.15.7,<0.16` |
| Python | 3.10 – 3.13 |

The integration uses the SDK's sandbox provider interface, which is still evolving, so the
dependency on `openai-agents` is pinned to the tested minor version. A new release of this
package follows each tested SDK minor version.

## Configuration

`WBXSandboxClient` holds the credentials. They stay on the live object and are never written
into run state:

| Argument | Default | Meaning |
| --- | --- | --- |
| `api_key` | `WBXSANDBOX_API_KEY`, then `WBX_API_KEY` | API key of your WBX Sandbox account |
| `url` | `WBXSANDBOX_URL` or `https://sandbox.wbx.cz` | API address |
| `env` | – | **Secret** environment variables for every command (API tokens) |

`WBXSandboxClientOptions` describes the sandbox. Options may be saved in run state, so do not
put secrets here:

| Option | Default | Meaning |
| --- | --- | --- |
| `template` | `"base"` | Base image: `base` (Debian, Python 3.12), `code-interpreter`, `node`, `go`, `browser`, … |
| `name` | – | Named sandbox: an existing sandbox with this name is reused and its workspace kept |
| `timeout_s` | `3600` | Lifetime in seconds (your plan's maximum applies) |
| `on_exit` | `"kill"` | `"kill"` deletes the sandbox after the run; `"sleep"` keeps its files for a later resume |
| `idle_timeout_s` | – | With `on_exit="sleep"`: fall asleep after this many idle seconds |
| `compute_class` | account default | `economy`, `balanced`, `performance`, `memory`, `dedicated` |
| `vcpu`, `mem_mb` | template default | Size of the VM |
| `internet` | `True` | `False` = no network at all |
| `allow` | – | Allowlist of reachable domains or addresses |
| `block` | – | Additional blocked destinations |
| `ip` | team default | `"ipv4"`, `"dual"` or `"ipv6"` (own public IPv6 address) |
| `env` | – | Non-secret environment variables |
| `metadata` | – | Your labels, shown when listing sandboxes |
| `exposed_ports` | – | Ports the agent or your code may resolve to a public HTTPS URL |

Network settings can only make your account's policy stricter, never looser.

### Keep the workspace between runs

```python
options = WBXSandboxClientOptions(name="report-builder", on_exit="sleep")
```

After the run the sandbox falls asleep instead of being deleted: files are kept, running
programs stop and compute is not billed. The next run with the same `name` (or a resume from
saved run state) wakes it up in a few seconds and continues in the same workspace. Sleeping
sandboxes are kept for the retention time of your plan.

### Exposed ports

```python
options = WBXSandboxClientOptions(exposed_ports=(8000,))
endpoint = await session.resolve_exposed_port(8000)
print(endpoint.url_for("http"))     # https://8000-….wbxbox.net/?…token…
```

The URL contains an access token. Anyone with the URL can reach the port, so treat it like a
password. The app must listen on `0.0.0.0`.

## Lifecycle and resume

* With `SandboxRunConfig(client=...)` the runner owns the session: it starts it, snapshots the
  workspace when a snapshot is configured, and then calls `delete`, which deletes the sandbox
  (`on_exit="kill"`) or puts it to sleep (`on_exit="sleep"`).
* With a live `session=` your code owns the session: close it with `await session.aclose()` and
  then `await client.delete(session)`.
* `resume` reconnects to the sandbox recorded in the saved state (and wakes it up if it
  sleeps). If the sandbox no longer exists, a new one is created and its workspace is restored
  from the configured snapshot. The API only returns sandboxes of the account behind the
  client's API key, so a saved state cannot attach to someone else's sandbox.
* Import `openai_agents_wbxsandbox` in every process that restores saved run state; the import
  registers the `wbx` state type.

Saved state contains the sandbox ID and name, the template and the creation options (including
non-secret `env`), never the API key or `WBXSandboxClient(env=...)`. Fields are only added with
defaults, so states saved by 0.1.x keep working in later 0.x releases.

## How it works

| SDK operation | WBX Sandbox |
| --- | --- |
| `create` | create a sandbox (template, class, size, network, name, lifetime) |
| `resume` | reconnect by ID and wake it up; recreate from the snapshot if it is gone |
| `delete` | delete the sandbox, or put it to sleep with `on_exit="sleep"` |
| `exec` | run the command with `sh -c` in the workspace (`/workspace` by default) |
| `read` / `write` | file download / upload; paths are resolved inside the VM so they cannot leave the workspace |
| `persist_workspace` / `hydrate_workspace` | tar of the workspace, validated before extraction |
| exposed ports | public HTTPS URL with an access token |

Commands run as `root` inside the VM. The VM is the isolation boundary; nothing runs on the
machine of the application.

## Limitations

* Manifest mounts (S3, GCS, R2, Azure Blob, Box) are not supported yet; creating a sandbox with
  a mount fails with a clear `ConfigurationError`. Copy the data into the workspace instead.
* Host path grants (`extra_path_grants` with a host path) are rejected: the sandbox is a remote
  VM.
* No PTY sessions: the shell tool runs each command to completion (or until its timeout).
* Command output is captured as UTF-8 text (invalid bytes are replaced) and limited to 16 MB per
  stream. Use file reads for binary data.
* Background processes stop when the sandbox falls asleep; files are kept.

## Examples

* [`examples/fix_tests`](examples/fix_tests) – the agent fixes failing tests in a small repository.
* [`examples/csv_chart`](examples/csv_chart) – the agent analyzes a CSV file and returns a chart.

## Development

```bash
pip install -e ".[test]"
pytest                                   # unit and runner tests, no network
WBXSANDBOX_API_KEY=... pytest -m live    # tests against the real service
```

## License

MIT
