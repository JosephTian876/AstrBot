import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from astrbot.core.agent.hooks import BaseAgentRunHooks
from astrbot.core.agent.response import AgentResponse
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.astr_agent_run_util import _simulated_stream_tts, run_agent
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from astrbot.core.provider.provider import Provider


class _FakeEvent:
    """Minimal event surface used by the agent stream bridge."""

    def is_stopped(self) -> bool:
        return False

    def get_extra(self, key: str):
        del key
        return None

    def get_platform_name(self) -> str:
        return "test"


class _StreamingErrorRunner:
    """Agent runner that finishes with one provider error response."""

    streaming = True
    req = None

    def __init__(self, error_text: str) -> None:
        self.error_text = error_text
        self.finished = False
        self.run_context = SimpleNamespace(context=SimpleNamespace(event=_FakeEvent()))

    async def step(self):
        self.finished = True
        yield AgentResponse(
            type="err",
            data={"chain": MessageChain().message(self.error_text)},
        )

    def done(self) -> bool:
        return self.finished


class _MalformedStreamingErrorRunner(_StreamingErrorRunner):
    """Agent runner that returns an invalid provider error payload."""

    async def step(self):
        self.finished = True
        yield AgentResponse(type="err", data={})


@pytest.mark.asyncio
async def test_run_agent_forwards_streaming_provider_error():
    error_text = (
        "LLM 响应错误: Not found the model k2.7-code-highspeed or Permission denied"
    )
    runner = _StreamingErrorRunner(error_text)

    chains = [chain async for chain in run_agent(runner)]

    assert len(chains) == 1
    assert chains[0].get_plain_text() == error_text


@pytest.mark.asyncio
async def test_run_agent_replaces_malformed_streaming_provider_error():
    runner = _MalformedStreamingErrorRunner("unused")

    chains = [chain async for chain in run_agent(runner)]

    assert len(chains) == 1
    assert chains[0].get_plain_text() == "Error occurred during AI execution."


@pytest.mark.asyncio
async def test_simulated_stream_tts_leaves_audio_for_deferred_cleanup(tmp_path):
    audio_path = tmp_path / "speech.wav"
    audio_path.write_bytes(b"audio")

    class _TTSProvider:
        async def get_audio(self, text: str) -> str:
            assert text == "hello"
            return str(audio_path)

    text_queue: asyncio.Queue[str | None] = asyncio.Queue()
    audio_queue: asyncio.Queue[bytes | tuple[str, bytes] | None] = asyncio.Queue()
    await text_queue.put("hello")
    await text_queue.put(None)

    await _simulated_stream_tts(_TTSProvider(), text_queue, audio_queue)

    assert await audio_queue.get() == ("hello", b"audio")
    assert await audio_queue.get() is None
    assert audio_path.exists()


class _RecordingEvent:
    """Event surface used by the agent stream bridge, backed by real extras."""

    def __init__(self) -> None:
        self.extras: dict = {}
        self.trace = SimpleNamespace(record=lambda *args, **kwargs: None)

    def is_stopped(self) -> bool:
        return False

    def get_extra(self, key: str, default=None):
        return self.extras.get(key, default)

    def set_extra(self, key: str, value) -> None:
        self.extras[key] = value

    def get_platform_name(self) -> str:
        return "lark"

    def get_platform_id(self) -> str:
        return "lark"

    async def send(self, chain) -> None:
        del chain

    def set_result(self, result) -> None:
        self.result = result

    def clear_result(self) -> None:
        self.result = None


class _NarratingProvider(Provider):
    """Round 1 answers with text plus a tool call, round 2 with final text."""

    def __init__(self) -> None:
        super().__init__({}, {})
        self.round = 0

    def get_current_key(self) -> str:
        return "test"

    def set_key(self, key: str) -> None:
        pass

    async def get_models(self) -> list[str]:
        return ["test"]

    async def text_chat(self, **kwargs) -> LLMResponse:
        self.round += 1
        if self.round == 1:
            return LLMResponse(
                role="assistant",
                completion_text="narration",
                tools_call_name=["test_tool"],
                tools_call_args=[{"query": "q"}],
                tools_call_ids=["call_1"],
            )
        return LLMResponse(role="assistant", completion_text="final-answer")

    async def text_chat_stream(self, **kwargs):
        aggregate = await self.text_chat(**kwargs)
        yield LLMResponse(
            role="assistant",
            completion_text=aggregate.completion_text,
            is_chunk=True,
        )
        yield aggregate


class _RecordingHooks(BaseAgentRunHooks):
    def __init__(self) -> None:
        self.intermediate = []
        self.done = []

    async def on_agent_response(self, run_context, llm_response) -> None:
        self.intermediate.append(llm_response.completion_text)

    async def on_agent_done(self, run_context, llm_response) -> None:
        self.done.append(llm_response.completion_text)


class _PassThroughToolExecutor:
    def execute(self, tool, run_context, **tool_args):
        async def generator():
            from mcp.types import CallToolResult, TextContent

            yield CallToolResult(content=[TextContent(type="text", text="ok")])

        return generator()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_to_general", [False, True])
async def test_tool_round_text_is_delivered_once_through_stream_bridge(
    stream_to_general,
):
    provider = _NarratingProvider()
    hooks = _RecordingHooks()
    runner = ToolLoopAgentRunner()
    event = _RecordingEvent()
    tool_set = ToolSet(
        tools=[
            FunctionTool(
                name="test_tool",
                description="test tool",
                parameters={"type": "object", "properties": {}},
                handler=AsyncMock(),
            )
        ]
    )
    await runner.reset(
        provider=provider,
        request=ProviderRequest(prompt="hi", func_tool=tool_set, contexts=[]),
        run_context=ContextWrapper(context=SimpleNamespace(event=event)),
        tool_executor=_PassThroughToolExecutor(),
        agent_hooks=hooks,
        streaming=True,
    )

    delivered = []
    async for chain in run_agent(runner, stream_to_general=stream_to_general):
        text = chain.get_plain_text()
        if text and text.strip():
            delivered.append(text)

    assert hooks.intermediate == ["narration"]
    assert hooks.done == ["final-answer"]
    assert delivered.count("narration") == 1
    assert delivered.count("final-answer") == 1
