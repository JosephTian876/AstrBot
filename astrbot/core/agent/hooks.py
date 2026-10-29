from typing import Generic

import mcp

from astrbot.core.agent.tool import FunctionTool
from astrbot.core.provider.entities import LLMResponse

from .run_context import ContextWrapper, TContext


class BaseAgentRunHooks(Generic[TContext]):
    async def on_agent_begin(self, run_context: ContextWrapper[TContext]) -> None: ...
    async def on_tool_start(
        self,
        run_context: ContextWrapper[TContext],
        tool: FunctionTool,
        tool_args: dict | None,
    ) -> None: ...
    async def on_tool_end(
        self,
        run_context: ContextWrapper[TContext],
        tool: FunctionTool,
        tool_args: dict | None,
        tool_result: mcp.types.CallToolResult | None,
    ) -> None: ...
    async def on_agent_done(
        self,
        run_context: ContextWrapper[TContext],
        llm_response: LLMResponse,
    ) -> None: ...
    async def on_agent_response(
        self,
        run_context: ContextWrapper[TContext],
        llm_response: LLMResponse,
    ) -> None:
        """Called once for every provider round that requests a tool call.

        This runs after the round's text has been produced and before the tool
        calls are executed. It never runs for the final, tool-free round; that
        one is reported through `on_agent_done`.

        Args:
            run_context: The run context of the current agent execution.
            llm_response: The completed response of the round. Only its text
                fields are meant to be modified; the tool call fields are
                restored afterwards so a hook cannot change what the agent has
                already decided to execute.

        Returns:
            None.
        """
        ...
