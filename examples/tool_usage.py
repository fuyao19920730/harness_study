"""Example: an agent that uses tools to answer questions.

Usage:
    export OPENAI_API_KEY="sk-..."
    python -m examples.tool_usage
"""

import asyncio
import random

from harness import Agent, tool


@tool(description="Generate a random integer between min_val and max_val (inclusive).")
async def random_number(min_val: int = 1, max_val: int = 100) -> str:
    num = random.randint(min_val, max_val)
    return f"Random number: {num}"


@tool(description="Calculate a mathematical expression. Supports basic arithmetic.")
async def calculate(expression: str) -> str:
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "Error: expression contains disallowed characters"
    try:
        result = eval(expression)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


async def main() -> None:
    async with Agent(
        name="math-assistant",
        model="gpt-4o",
        tools=[random_number, calculate],
        system_prompt="You are a math assistant. Use tools to help with calculations.",
    ) as agent:
        result = await agent.run(
            "Generate two random numbers between 1 and 50, then calculate their product."
        )
        print(result.output)
        print()
        print(result.trace.summary())


if __name__ == "__main__":
    asyncio.run(main())
