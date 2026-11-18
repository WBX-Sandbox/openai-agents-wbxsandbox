"""WBX Sandbox for OpenAI Agents SDK sandbox agents.

    from agents.run import RunConfig
    from agents.sandbox import SandboxRunConfig
    from openai_agents_wbxsandbox import WBXSandboxClient, WBXSandboxClientOptions

    run_config = RunConfig(
        sandbox=SandboxRunConfig(
            client=WBXSandboxClient(),                  # API key from WBXSANDBOX_API_KEY
            options=WBXSandboxClientOptions(template="base"),
        ),
    )

Importing this package registers the ``wbx`` options and session-state types, so import it in
every process that restores saved run state.
"""

from .sandbox import (
    BACKEND_ID,
    WBXSandboxClient,
    WBXSandboxClientOptions,
    WBXSandboxSession,
    WBXSandboxSessionState,
)

__version__ = "0.1.0"

__all__ = [
    "BACKEND_ID",
    "WBXSandboxClient",
    "WBXSandboxClientOptions",
    "WBXSandboxSession",
    "WBXSandboxSessionState",
    "__version__",
]
