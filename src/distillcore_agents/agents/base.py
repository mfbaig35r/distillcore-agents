"""AgentEvent streaming utilities for the distillcore agent pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent

from ..client import DistillcoreClient

OutputT = TypeVar("OutputT", bound=BaseModel)


@dataclass
class AgentEvent:
    """A typed event emitted during agent execution."""

    event_type: str  # started | tool_call | tool_result | completed | error
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


async def iter_agent_events(
    agent: Agent[DistillcoreClient, OutputT],
    prompt: str,
    *,
    deps: DistillcoreClient,
) -> AsyncIterator[tuple[AgentEvent, OutputT | None]]:
    """Drive an agent and yield typed events as it executes.

    Yields (AgentEvent, None) for intermediate events and
    (AgentEvent, output) for the final completion event.
    """
    yield AgentEvent(event_type="started"), None

    try:
        async with agent.iter(prompt, deps=deps) as run:
            async for node in run:
                node_name = type(node).__name__

                if node_name == "ToolCallPart":
                    yield AgentEvent(
                        event_type="tool_call",
                        data={
                            "tool_name": getattr(node, "tool_name", ""),
                            "args": getattr(node, "args_as_dict", lambda: {})(),
                        },
                    ), None

                elif node_name == "ToolReturn":
                    yield AgentEvent(
                        event_type="tool_result",
                        data={
                            "tool_name": getattr(node, "tool_name", ""),
                            "content": str(getattr(node, "content", ""))[:500],
                        },
                    ), None

            # Run complete — get output
            output = run.result.output
            yield AgentEvent(
                event_type="completed",
                data={"output": output.model_dump()},
            ), output

    except Exception as exc:
        yield AgentEvent(
            event_type="error",
            data={"error": str(exc), "error_type": type(exc).__name__},
        ), None
