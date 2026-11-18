"""The agent analyzes a CSV file in a WBX sandbox and returns a chart.

    pip install openai-agents-wbxsandbox
    export WBXSANDBOX_API_KEY=... OPENAI_API_KEY=...
    python examples/csv_chart/main.py          # writes chart.png next to this script

The CSV is copied into the sandbox, the agent writes and runs Python there (the
``code-interpreter`` template has pandas and matplotlib), and this script downloads the
finished chart from the sandbox before deleting it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from agents import Runner
from agents.run import RunConfig
from agents.sandbox import Manifest, SandboxAgent, SandboxRunConfig
from agents.sandbox.entries import LocalFile

from openai_agents_wbxsandbox import WBXSandboxClient, WBXSandboxClientOptions

HERE = Path(__file__).parent

INSTRUCTIONS = """\
You are a data analyst. The workspace contains sales.csv.
Use Python in the shell (pandas, matplotlib; install them with pip only if they are missing).
Save a line chart of monthly revenue per region to chart.png (1200x700 px, labeled axes,
title, legend). Then answer with three short findings backed by numbers from the data."""


async def main() -> None:
    manifest = Manifest(entries={"sales.csv": LocalFile(src=HERE / "sales.csv")})
    agent = SandboxAgent(name="Analyst", instructions=INSTRUCTIONS, default_manifest=manifest)

    client = WBXSandboxClient()
    options = WBXSandboxClientOptions(
        template="code-interpreter",
        allow=("pypi.org", "files.pythonhosted.org"),
        timeout_s=1800,
    )
    session = await client.create(manifest=manifest, options=options)
    try:
        async with session:
            result = await Runner.run(
                agent,
                "Analyze the sales and make the chart.",
                run_config=RunConfig(sandbox=SandboxRunConfig(session=session)),
                max_turns=30,
            )
            print(result.final_output)
            chart = (await session.read(Path("chart.png"))).read()
            (HERE / "chart.png").write_bytes(chart)
            print(f"chart.png saved ({len(chart)} bytes)")
    finally:
        await client.delete(session)


if __name__ == "__main__":
    asyncio.run(main())
