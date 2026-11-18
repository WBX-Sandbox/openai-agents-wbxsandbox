"""The agent runs the tests of a small repository in a WBX sandbox and fixes the code.

    pip install openai-agents-wbxsandbox
    export WBXSANDBOX_API_KEY=... OPENAI_API_KEY=...
    python examples/fix_tests/main.py

The repository is copied into the sandbox workspace. The runner creates the sandbox, the agent
uses its shell and file tools there, and the sandbox is deleted when the run ends. The fixed
files are read back through a session that this script owns, so they can be printed locally.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from agents import Runner
from agents.run import RunConfig
from agents.sandbox import Manifest, SandboxAgent, SandboxRunConfig
from agents.sandbox.entries import LocalFile

from openai_agents_wbxsandbox import WBXSandboxClient, WBXSandboxClientOptions

REPO = Path(__file__).parent / "repo"

INSTRUCTIONS = """\
You maintain the Python repository in the workspace.
1. Install pytest if it is missing (`python3 -m pip install -q pytest`).
2. Run `python3 -m pytest -q` and read the failures.
3. Fix the code in pricing.py. Never change the tests.
4. Run the tests again until they all pass.
Finish with a short summary: what was wrong and what you changed."""


async def main() -> None:
    manifest = Manifest(
        entries={p.name: LocalFile(src=p) for p in sorted(REPO.glob("*.py"))},
    )
    agent = SandboxAgent(name="Test fixer", instructions=INSTRUCTIONS, default_manifest=manifest)

    client = WBXSandboxClient()
    options = WBXSandboxClientOptions(
        template="base",
        # only the package index is reachable from the sandbox
        allow=("pypi.org", "files.pythonhosted.org"),
        timeout_s=1800,
    )
    # This script owns the session, so it can read the fixed files after the run.
    session = await client.create(manifest=manifest, options=options)
    try:
        async with session:
            result = await Runner.run(
                agent,
                "The tests fail. Please fix the code.",
                run_config=RunConfig(sandbox=SandboxRunConfig(session=session)),
                max_turns=30,
            )
            print("=== agent summary ===")
            print(result.final_output)
            check = await session.exec("python3 -m pytest -q", timeout=120)
            print("=== pytest in the sandbox after the run ===")
            print(check.stdout.decode(), check.stderr.decode())
            fixed = (await session.read(Path("pricing.py"))).read().decode()
            print("=== pricing.py after the fix ===")
            print(fixed)
    finally:
        await client.delete(session)


if __name__ == "__main__":
    asyncio.run(main())
