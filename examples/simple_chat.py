"""Minimal example: a simple chat agent with no tools.

Usage:
    export OPENAI_API_KEY="sk-..."
    python -m examples.simple_chat
"""

import asyncio

from harness import Agent


async def main() -> None:
    async with Agent(
        name="simple-assistant",
        model="gpt-4o",
        system_prompt="You are a helpful assistant. Be concise.",
    ) as agent:
        result = await agent.run("What are the 3 most important principles of software engineering?")
        print(result.output)
        print()
        print(result.trace.summary())


if __name__ == "__main__":
    asyncio.run(main())
