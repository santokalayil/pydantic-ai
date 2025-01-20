from __future__ import annotations as _annotations

import asyncio
import dataclasses
import inspect
from collections.abc import Awaitable, Iterator, Sequence
from contextlib import contextmanager
from types import FrameType
from typing import Any, Callable, Generic, Literal, cast, final, overload

import logfire_api
from typing_extensions import TypeVar, assert_never, deprecated

from pydantic_graph import BaseNode, Graph, GraphRunContext
from pydantic_graph.nodes import End, NodeRunEndT

from . import (
    _result,
    _system_prompt,
    _utils,
    exceptions,
    messages as _messages,
    models,
    result,
    usage as _usage,
)
from .agent import get_captured_run_messages
from .messages import ModelRequest
from .result import ResultData, ResultData_co
from .settings import ModelSettings, merge_model_settings
from .tools import (
    AgentDeps,
    DocstringFormat,
    RunContext,
    Tool,
    ToolDefinition,
    ToolFuncContext,
    ToolFuncEither,
    ToolFuncPlain,
    ToolParams,
    ToolPrepareFunc,
)

__all__ = 'GraphAgent', 'EndStrategy'

_logfire = logfire_api.Logfire(otel_scope='pydantic-ai')

# while waiting for https://github.com/pydantic/logfire/issues/745
try:
    import logfire._internal.stack_info
except ImportError:
    pass
else:
    from pathlib import Path

    logfire._internal.stack_info.NON_USER_CODE_PREFIXES += (str(Path(__file__).parent.absolute()),)


@dataclasses.dataclass
class MarkFinalResult(Generic[ResultData_co]):
    """Marker class to indicate that the result is the final result.

    This allows us to use `isinstance`, which wouldn't be possible if we were returning `ResultData` directly.

    It also avoids problems in the case where the result type is itself `None`, but is set.
    """

    data: ResultData_co
    """The final result data."""
    tool_name: str | None
    """Name of the final result tool, None if the result is a string."""


@dataclasses.dataclass
class GraphAgentState:
    """State kept across the execution of the agent graph."""

    message_history: list[_messages.ModelMessage]
    usage: _usage.Usage
    retries: int
    run_step: int

    def increment_retries(self, max_result_retries: int) -> None:
        self.retries += 1
        if self.retries > max_result_retries:
            raise exceptions.UnexpectedModelBehavior(
                f'Exceeded maximum retries ({max_result_retries}) for result validation'
            )


T = TypeVar('T')
DepsT = TypeVar('DepsT', default=object)
"""Type variable for the dependencies of a graph and node."""


@dataclasses.dataclass
class GraphAgentDeps(Generic[DepsT, ResultData]):
    """Dependencies/config passed to the agent graph."""

    user_deps: DepsT

    prompt: str
    """The original user prompt passed to the run."""

    function_tools: dict[str, Tool[DepsT]] = dataclasses.field(repr=False)
    result_tools: list[ToolDefinition]

    run_span: logfire_api.LogfireSpan
    # TODO: Add step_span?

    result_schema: _result.ResultSchema[ResultData] | None
    result_validators: list[_result.ResultValidator[DepsT, ResultData]]

    model: models.Model
    model_settings: ModelSettings | None
    usage_limits: _usage.UsageLimits

    max_result_retries: int
    end_strategy: EndStrategy


@dataclasses.dataclass
class UserPromptNode(BaseNode[GraphAgentState, GraphAgentDeps[DepsT, Any], NodeRunEndT]):
    """First node: incorporate the system prompts and user prompt into state.message_history, then go to ModelRequestNode."""

    user_prompt: str

    system_prompts: tuple[str, ...]
    system_prompt_functions: list[_system_prompt.SystemPromptRunner[DepsT]]
    system_prompt_dynamic_functions: dict[str, _system_prompt.SystemPromptRunner[DepsT]]

    async def run(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, Any]]
    ) -> ModelRequestNode[DepsT, NodeRunEndT]:
        run_context = _build_run_context(ctx)
        history, next_message = await self._prepare_messages(self.user_prompt, ctx.state.message_history, run_context)
        ctx.state.message_history = history
        run_context.messages = history

        # TODO: We need to make it so that function_tools are not shared between runs
        #   See comment on the current_retry field of `Tool` for more details.
        for tool in ctx.deps.function_tools.values():
            tool.current_retry = 0

        return ModelRequestNode(request=next_message)

    async def _prepare_messages(
        self, user_prompt: str, message_history: list[_messages.ModelMessage] | None, run_context: RunContext[DepsT]
    ) -> tuple[list[_messages.ModelMessage], _messages.ModelRequest]:
        try:
            ctx_messages = get_captured_run_messages()
        except LookupError:
            messages: list[_messages.ModelMessage] = []
        else:
            if ctx_messages.used:
                messages = []
            else:
                messages = ctx_messages.messages
                ctx_messages.used = True

        if message_history:
            # Shallow copy messages
            messages.extend(message_history)
            # Reevaluate any dynamic system prompt parts
            await self._reevaluate_dynamic_prompts(messages, run_context)
            return messages, _messages.ModelRequest([_messages.UserPromptPart(user_prompt)])
        else:
            parts = await self._sys_parts(run_context)
            parts.append(_messages.UserPromptPart(user_prompt))
            return messages, _messages.ModelRequest(parts)

    async def _reevaluate_dynamic_prompts(
        self, messages: list[_messages.ModelMessage], run_context: RunContext[DepsT]
    ) -> None:
        """Reevaluate any `SystemPromptPart` with dynamic_ref in the provided messages by running the associated runner function."""
        # Only proceed if there's at least one dynamic runner.
        if self.system_prompt_dynamic_functions:
            for msg in messages:
                if isinstance(msg, _messages.ModelRequest):
                    for i, part in enumerate(msg.parts):
                        if isinstance(part, _messages.SystemPromptPart) and part.dynamic_ref:
                            # Look up the runner by its ref
                            if runner := self.system_prompt_dynamic_functions.get(part.dynamic_ref):
                                updated_part_content = await runner.run(run_context)
                                msg.parts[i] = _messages.SystemPromptPart(
                                    updated_part_content, dynamic_ref=part.dynamic_ref
                                )

    async def _sys_parts(self, run_context: RunContext[DepsT]) -> list[_messages.ModelRequestPart]:
        """Build the initial messages for the conversation."""
        messages: list[_messages.ModelRequestPart] = [_messages.SystemPromptPart(p) for p in self.system_prompts]
        for sys_prompt_runner in self.system_prompt_functions:
            prompt = await sys_prompt_runner.run(run_context)
            if sys_prompt_runner.dynamic:
                messages.append(_messages.SystemPromptPart(prompt, dynamic_ref=sys_prompt_runner.function.__qualname__))
            else:
                messages.append(_messages.SystemPromptPart(prompt))
        return messages


@dataclasses.dataclass
class ModelRequestNode(BaseNode[GraphAgentState, GraphAgentDeps[DepsT, Any], NodeRunEndT]):
    """Make a request to the model using the last message in state.message_history (or a specified request)."""

    request: _messages.ModelRequest

    async def run(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> ModelResponseNode[DepsT, NodeRunEndT]:
        ctx.state.message_history.append(self.request)

        # Check usage
        if ctx.deps.usage_limits:
            ctx.deps.usage_limits.check_before_request(ctx.state.usage)

        # Increment run_step
        ctx.state.run_step += 1

        with _logfire.span('preparing model and tools {run_step=}', run_step=ctx.state.run_step):
            agent_model = await self._prepare_model(ctx)

        # Actually make the model request
        model_settings = merge_model_settings(ctx.deps.model_settings, None)
        with _logfire.span('model request') as span:
            model_response, request_usage = await agent_model.request(ctx.state.message_history, model_settings)
            span.set_attribute('response', model_response)
            span.set_attribute('usage', request_usage)

        # Update usage
        ctx.state.usage.incr(request_usage, requests=1)
        if ctx.deps.usage_limits:
            ctx.deps.usage_limits.check_tokens(ctx.state.usage)

        # Append the model response to state.message_history
        ctx.state.message_history.append(model_response)

        # Next node is ModelResponseNode
        return ModelResponseNode[DepsT, NodeRunEndT](response=model_response)

    async def _prepare_model(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> models.AgentModel:
        """Build tools and create an agent model."""
        function_tool_defs: list[ToolDefinition] = []

        run_context = _build_run_context(ctx)

        async def add_tool(tool: Tool[DepsT]) -> None:
            ctx = run_context.replace_with(retry=tool.current_retry, tool_name=tool.name)
            if tool_def := await tool.prepare_tool_def(ctx):
                function_tool_defs.append(tool_def)

        await asyncio.gather(*map(add_tool, ctx.deps.function_tools.values()))

        result_schema = ctx.deps.result_schema
        return await run_context.model.agent_model(
            function_tools=function_tool_defs,
            allow_text_result=_allow_text_result(result_schema),
            result_tools=result_schema.tool_defs() if result_schema is not None else [],
        )


@dataclasses.dataclass
class ModelResponseNode(BaseNode[GraphAgentState, GraphAgentDeps[DepsT, Any], NodeRunEndT]):
    """Analyze the model response. Possibly run tools, or finalize the result, or continue."""

    response: _messages.ModelResponse

    async def run(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> ModelRequestNode[DepsT, NodeRunEndT] | FinalResultNode[DepsT, NodeRunEndT]:
        texts: list[str] = []
        tool_calls: list[_messages.ToolCallPart] = []
        for part in self.response.parts:
            if isinstance(part, _messages.TextPart):
                # ignore empty content for text parts, see #437
                if part.content:
                    texts.append(part.content)
            elif isinstance(part, _messages.ToolCallPart):
                tool_calls.append(part)
            else:
                assert_never(part)

        result_schema = ctx.deps.result_schema

        # At the moment, we prioritize at least executing tool calls if they are present.
        # In the future, we'd consider making this configurable at the agent or run level.
        # This accounts for cases like anthropic returns that might contain a text response
        # and a tool call response, where the text response just indicates the tool call will happen.
        if tool_calls:
            # first look for the result tool call
            final_result: MarkFinalResult[NodeRunEndT] | None = None

            parts: list[_messages.ModelRequestPart] = []
            if result_schema is not None:
                if match := result_schema.find_tool(tool_calls):
                    call, result_tool = match
                    try:
                        result_data = result_tool.validate(call)
                        result_data = await self._validate_result(result_data, ctx, call)
                    except _result.ToolRetryError as e:
                        # TODO: Should only increment retry stuff once per node execution, not for each tool call
                        #   Also, should increment the tool-specific retry count rather than the run retry count
                        ctx.state.increment_retries(ctx.deps.max_result_retries)
                        parts.append(e.tool_retry)
                    else:
                        final_result = MarkFinalResult(result_data, call.tool_name)

            # Then build the other request parts based on end strategy
            extra_parts = await self._process_function_tools(tool_calls, final_result and final_result.tool_name, ctx)

            if final_result:
                return FinalResultNode[DepsT, NodeRunEndT](final_result, extra_parts)
            else:
                parts.extend(extra_parts)
                return ModelRequestNode[DepsT, NodeRunEndT](ModelRequest(parts=parts))

        elif texts:
            text = '\n\n'.join(texts)
            if _allow_text_result(result_schema):
                result_data_input = cast(NodeRunEndT, text)
                try:
                    result_data = await self._validate_result(result_data_input, ctx, None)
                except _result.ToolRetryError as e:
                    ctx.state.increment_retries(ctx.deps.max_result_retries)
                    return ModelRequestNode[DepsT, NodeRunEndT](ModelRequest(parts=[e.tool_retry]))
                else:
                    return FinalResultNode[DepsT, NodeRunEndT](MarkFinalResult(result_data, None))
            else:
                ctx.state.increment_retries(ctx.deps.max_result_retries)
                return ModelRequestNode[DepsT, NodeRunEndT](
                    ModelRequest(
                        parts=[
                            _messages.RetryPromptPart(
                                content='Plain text responses are not permitted, please call one of the functions instead.',
                            )
                        ]
                    )
                )
        else:
            raise exceptions.UnexpectedModelBehavior('Received empty model response')

    async def _process_function_tools(
        self,
        tool_calls: list[_messages.ToolCallPart],
        result_tool_name: str | None,
        # run_context: RunContext[AgentDeps],
        # result_schema: _result.ResultSchema[RunResultData] | None,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
    ) -> list[_messages.ModelRequestPart]:
        """Process function (non-result) tool calls in parallel.

        Also add stub return parts for any other tools that need it.
        """
        parts: list[_messages.ModelRequestPart] = []
        tasks: list[asyncio.Task[_messages.ToolReturnPart | _messages.RetryPromptPart]] = []

        stub_function_tools = bool(result_tool_name) and ctx.deps.end_strategy == 'early'
        result_schema = ctx.deps.result_schema

        # we rely on the fact that if we found a result, it's the first result tool in the last
        found_used_result_tool = False
        run_context = _build_run_context(ctx)

        for call in tool_calls:
            if call.tool_name == result_tool_name and not found_used_result_tool:
                found_used_result_tool = True
                parts.append(
                    _messages.ToolReturnPart(
                        tool_name=call.tool_name,
                        content='Final result processed.',
                        tool_call_id=call.tool_call_id,
                    )
                )
            elif tool := ctx.deps.function_tools.get(call.tool_name):
                if stub_function_tools:
                    parts.append(
                        _messages.ToolReturnPart(
                            tool_name=call.tool_name,
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id=call.tool_call_id,
                        )
                    )
                else:
                    tasks.append(asyncio.create_task(tool.run(call, run_context), name=call.tool_name))
            elif result_schema is not None and call.tool_name in result_schema.tools:
                # if tool_name is in _result_schema, it means we found a result tool but an error occurred in
                # validation, we don't add another part here
                if result_tool_name is not None:
                    parts.append(
                        _messages.ToolReturnPart(
                            tool_name=call.tool_name,
                            content='Result tool not used - a final result was already processed.',
                            tool_call_id=call.tool_call_id,
                        )
                    )
            else:
                parts.append(self._unknown_tool(call.tool_name, ctx))

        # Run all tool tasks in parallel
        if tasks:
            with _logfire.span('running {tools=}', tools=[t.get_name() for t in tasks]):
                task_results: Sequence[_messages.ToolReturnPart | _messages.RetryPromptPart] = await asyncio.gather(
                    *tasks
                )
                for result in task_results:
                    if isinstance(result, _messages.ToolReturnPart):
                        parts.append(result)
                    elif isinstance(result, _messages.RetryPromptPart):
                        parts.append(result)
                    else:
                        assert_never(result)
        return parts

    def _unknown_tool(
        self,
        tool_name: str,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
    ) -> _messages.RetryPromptPart:
        ctx.state.increment_retries(ctx.deps.max_result_retries)
        tool_names = list(ctx.deps.function_tools.keys())
        if result_schema := ctx.deps.result_schema:
            tool_names.extend(result_schema.tool_names())

        if tool_names:
            msg = f'Available tools: {", ".join(tool_names)}'
        else:
            msg = 'No tools available.'

        return _messages.RetryPromptPart(content=f'Unknown tool name: {tool_name!r}. {msg}')

    async def _validate_result(
        self: ModelResponseNode[Any, T],
        result_data: T,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, T]],
        tool_call: _messages.ToolCallPart | None,
    ) -> T:
        for validator in ctx.deps.result_validators:
            run_context = _build_run_context(ctx)
            result_data = await validator.validate(result_data, tool_call, run_context)
        return result_data


@dataclasses.dataclass
class FinalResultNode(BaseNode[GraphAgentState, GraphAgentDeps[DepsT, Any], MarkFinalResult[NodeRunEndT]]):
    """Produce the final result of the run."""

    data: MarkFinalResult[NodeRunEndT]
    """The final result data."""
    extra_parts: list[_messages.ModelRequestPart] = dataclasses.field(default_factory=list)

    async def run(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> End[MarkFinalResult[NodeRunEndT]]:
        run_span = ctx.deps.run_span
        usage = ctx.state.usage
        messages = ctx.state.message_history

        # TODO: For backwards compatibility, append a new ModelRequest using the tool returns and retries
        if self.extra_parts:
            messages.append(ModelRequest(parts=self.extra_parts))

        # TODO: Set this attribute somewhere
        # handle_span = self.handle_model_response_span
        # handle_span.set_attribute('final_data', self.data)
        run_span.set_attribute('usage', usage)
        run_span.set_attribute('all_messages', messages)

        # End the run with self.data
        return End(self.data)


NoneType = type(None)
EndStrategy = Literal['early', 'exhaustive']
"""The strategy for handling multiple tool calls when a final result is found.

- `'early'`: Stop processing other tool calls once a final result is found
- `'exhaustive'`: Process all tool calls even after finding a final result
"""
RunResultData = TypeVar('RunResultData')
"""Type variable for the result data of a run where `result_type` was customized on the run call."""


@final
@dataclasses.dataclass(init=False)
class GraphAgent(Generic[AgentDeps, ResultData]):
    """Class for defining "agents" - a way to have a specific type of "conversation" with an LLM.

    Agents are generic in the dependency type they take [`AgentDeps`][pydantic_ai.tools.AgentDeps]
    and the result data type they return, [`ResultData`][pydantic_ai.result.ResultData].

    By default, if neither generic parameter is customised, agents have type `Agent[None, str]`.

    Minimal usage example:

    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')
    result = agent.run_sync('What is the capital of France?')
    print(result.data)
    #> Paris
    ```
    """

    # we use dataclass fields in order to conveniently know what attributes are available
    model: models.Model | models.KnownModelName | None
    """The default model configured for this agent."""

    name: str | None
    """The name of the agent, used for logging.

    If `None`, we try to infer the agent name from the call frame when the agent is first run.
    """
    end_strategy: EndStrategy
    """Strategy for handling tool calls when a final result is found."""

    model_settings: ModelSettings | None
    """Optional model request settings to use for this agents's runs, by default.

    Note, if `model_settings` is provided by `run`, `run_sync`, or `run_stream`, those settings will
    be merged with this value, with the runtime argument taking priority.
    """

    result_type: type[ResultData]
    """
    The type of the result data, used to validate the result data, defaults to `str`.
    """

    # _result_tool_name: str
    # _result_tool_description: str | None
    # _result_schema: _result.ResultSchema[ResultData] | None
    # _system_prompts: tuple[str, ...]
    # _function_tools: dict[str, Tool[AgentDeps]]
    # _default_retries: int
    # _system_prompt_functions: list[_system_prompt.SystemPromptRunner[AgentDeps]]
    # _system_prompt_dynamic_functions: dict[str, _system_prompt.SystemPromptRunner[AgentDeps]]
    # _deps_type: type[AgentDeps]
    # _max_result_retries: int
    # _result_validators: list[_result.ResultValidator[AgentDeps, ResultData]]
    #
    _override_deps: _utils.Option[AgentDeps]
    _override_model: _utils.Option[models.Model]

    def __init__(
        self,
        model: models.Model | models.KnownModelName | None = None,
        *,
        result_type: type[ResultData] = str,
        system_prompt: str | Sequence[str] = (),
        deps_type: type[AgentDeps] = NoneType,
        name: str | None = None,
        model_settings: ModelSettings | None = None,
        retries: int = 1,
        result_tool_name: str = 'final_result',
        result_tool_description: str | None = None,
        result_retries: int | None = None,
        tools: Sequence[Tool[AgentDeps] | ToolFuncEither[AgentDeps, ...]] = (),
        defer_model_check: bool = False,
        end_strategy: EndStrategy = 'early',
    ):
        """Create an agent.

        Args:
            model: The default model to use for this agent, if not provide,
                you must provide the model when calling it.
            result_type: The type of the result data, used to validate the result data, defaults to `str`.
            system_prompt: Static system prompts to use for this agent, you can also register system
                prompts via a function with [`system_prompt`][pydantic_ai.Agent.system_prompt].
            deps_type: The type used for dependency injection, this parameter exists solely to allow you to fully
                parameterize the agent, and therefore get the best out of static type checking.
                If you're not using deps, but want type checking to pass, you can set `deps=None` to satisfy Pyright
                or add a type hint `: Agent[None, <return type>]`.
            name: The name of the agent, used for logging. If `None`, we try to infer the agent name from the call frame
                when the agent is first run.
            model_settings: Optional model request settings to use for this agent's runs, by default.
            retries: The default number of retries to allow before raising an error.
            result_tool_name: The name of the tool to use for the final result.
            result_tool_description: The description of the final result tool.
            result_retries: The maximum number of retries to allow for result validation, defaults to `retries`.
            tools: Tools to register with the agent, you can also register tools via the decorators
                [`@agent.tool`][pydantic_ai.Agent.tool] and [`@agent.tool_plain`][pydantic_ai.Agent.tool_plain].
            defer_model_check: by default, if you provide a [named][pydantic_ai.models.KnownModelName] model,
                it's evaluated to create a [`Model`][pydantic_ai.models.Model] instance immediately,
                which checks for the necessary environment variables. Set this to `false`
                to defer the evaluation until the first run. Useful if you want to
                [override the model][pydantic_ai.Agent.override] for testing.
            end_strategy: Strategy for handling tool calls that are requested alongside a final result.
                See [`EndStrategy`][pydantic_ai.agent.EndStrategy] for more information.
        """
        if model is None or defer_model_check:
            self.model = model
        else:
            self.model = models.infer_model(model)

        self.end_strategy = end_strategy
        self.name = name
        self.model_settings = model_settings
        self.result_type = result_type

        self._result_tool_name = result_tool_name
        self._result_tool_description = result_tool_description
        self._result_schema: _result.ResultSchema[ResultData] | None = _result.ResultSchema[result_type].build(
            result_type, result_tool_name, result_tool_description
        )

        self._system_prompts = (system_prompt,) if isinstance(system_prompt, str) else tuple(system_prompt)
        self._function_tools: dict[str, Tool[AgentDeps]] = {}
        self._default_retries = retries
        for tool in tools:
            if isinstance(tool, Tool):
                self._register_tool(tool)
            else:
                self._register_tool(Tool(tool))
        self._deps_type = deps_type
        self._system_prompt_functions: list[_system_prompt.SystemPromptRunner[AgentDeps]] = []
        self._system_prompt_dynamic_functions: dict[str, _system_prompt.SystemPromptRunner[AgentDeps]] = {}
        self._max_result_retries = result_retries if result_retries is not None else retries
        self._result_validators: list[_result.ResultValidator[AgentDeps, ResultData]] = []

        self._override_deps = None
        self._override_model = None

    @overload
    async def run(
        self,
        user_prompt: str,
        *,
        result_type: None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        deps: AgentDeps = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
    ) -> result.RunResult[ResultData]: ...

    @overload
    async def run(
        self,
        user_prompt: str,
        *,
        result_type: type[RunResultData],
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        deps: AgentDeps = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
    ) -> result.RunResult[RunResultData]: ...

    async def run(
        self,
        user_prompt: str,
        *,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        deps: AgentDeps = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        result_type: type[RunResultData] | None = None,
        infer_name: bool = True,
    ) -> result.RunResult[Any]:
        """Run the agent with a user prompt in async mode.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            result = await agent.run('What is the capital of France?')
            print(result.data)
            #> Paris
        ```

        Args:
            result_type: Custom result type to use for this run, `result_type` may only be used if the agent has no
                result validators since result validators would expect an argument that matches the agent's result type.
            user_prompt: User input to start/continue the conversation.
            message_history: History of the conversation so far.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.

        Returns:
            The result of the run.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())
        model_used = await self._get_model(model)

        deps = self._get_deps(deps)
        new_message_index = len(message_history) if message_history else 0
        # The following cast is to deal with the fact that ResultSchema is functionally covariant, but the typevar isn't
        result_schema: _result.ResultSchema[RunResultData] | None = self._prepare_result_schema(result_type)

        # Build the graph
        g = self._build_graph(result_type)

        # Build the initial state
        s = GraphAgentState(
            message_history=message_history[:] if message_history else [],
            usage=usage or _usage.Usage(),
            retries=0,
            run_step=0,
        )

        # We consider it a user error if a user tries to restrict the result type while having a result validator that
        # may change the result type from the restricted type to something else. Therefore, we consider the following
        # typecast reasonable, even though it is possible to violate it with otherwise-type-checked code.
        result_validators = cast(list[_result.ResultValidator[AgentDeps, RunResultData]], self._result_validators)

        # TODO: Instead of this, copy the function tools to ensure they don't share current_retry state between agent
        #  runs. Requires some changes to `Tool` to make them copyable though.
        for v in self._function_tools.values():
            v.current_retry = 0

        model_settings = merge_model_settings(self.model_settings, model_settings)
        usage_limits = usage_limits or _usage.UsageLimits()

        with _logfire.span(
            '{agent_name} run {prompt=}',
            prompt=user_prompt,
            agent=self,
            model_name=model_used.name() if model_used else 'no-model',
            agent_name=self.name or 'agent',
        ) as run_span:
            # Build the deps object for the graph
            d = GraphAgentDeps[AgentDeps, RunResultData](
                user_deps=deps,
                prompt=user_prompt,
                function_tools=self._function_tools,
                result_tools=self._result_schema.tool_defs() if self._result_schema else [],
                run_span=run_span,
                result_schema=result_schema,
                result_validators=result_validators,
                model=model_used,
                model_settings=model_settings,
                usage_limits=usage_limits,
                max_result_retries=self._max_result_retries,
                end_strategy=self.end_strategy,
            )

            start_node = UserPromptNode[AgentDeps](
                user_prompt=user_prompt,
                system_prompts=self._system_prompts,
                system_prompt_functions=self._system_prompt_functions,
                system_prompt_dynamic_functions=self._system_prompt_dynamic_functions,
            )

            # Actually run
            end_result, history = await g.run(
                start_node,
                state=s,
                deps=d,
                infer_name=False,
            )
            run_span.set_attribute('history', history)

        usage_out = s.usage
        # Build final runresult
        # We don't do any advanced checking if the data is actually from a final result or not
        rr = result.RunResult(
            s.message_history,
            new_message_index,
            end_result.data,
            end_result.tool_name,
            usage_out,
        )
        return rr

    @overload
    def run_sync(
        self,
        user_prompt: str,
        *,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        deps: AgentDeps = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
    ) -> result.RunResult[ResultData]: ...

    @overload
    def run_sync(
        self,
        user_prompt: str,
        *,
        result_type: type[RunResultData] | None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        deps: AgentDeps = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
    ) -> result.RunResult[RunResultData]: ...

    def run_sync(
        self,
        user_prompt: str,
        *,
        result_type: type[RunResultData] | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        deps: AgentDeps = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
    ) -> result.RunResult[Any]:
        """Run the agent with a user prompt synchronously.

        This is a convenience method that wraps [`self.run`][pydantic_ai.Agent.run] with `loop.run_until_complete(...)`.
        You therefore can't use this method inside async code or if there's an active event loop.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        result_sync = agent.run_sync('What is the capital of Italy?')
        print(result_sync.data)
        #> Rome
        ```

        Args:
            result_type: Custom result type to use for this run, `result_type` may only be used if the agent has no
                result validators since result validators would expect an argument that matches the agent's result type.
            user_prompt: User input to start/continue the conversation.
            message_history: History of the conversation so far.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.

        Returns:
            The result of the run.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())
        return asyncio.get_event_loop().run_until_complete(
            self.run(
                user_prompt,
                result_type=result_type,
                message_history=message_history,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=False,
            )
        )

    @contextmanager
    def override(
        self,
        *,
        deps: AgentDeps | _utils.Unset = _utils.UNSET,
        model: models.Model | models.KnownModelName | _utils.Unset = _utils.UNSET,
    ) -> Iterator[None]:
        """Context manager to temporarily override agent dependencies and model.

        This is particularly useful when testing.
        You can find an example of this [here](../testing-evals.md#overriding-model-via-pytest-fixtures).

        Args:
            deps: The dependencies to use instead of the dependencies passed to the agent run.
            model: The model to use instead of the model passed to the agent run.
        """
        if _utils.is_set(deps):
            override_deps_before = self._override_deps
            self._override_deps = _utils.Some(deps)
        else:
            override_deps_before = _utils.UNSET

        # noinspection PyTypeChecker
        if _utils.is_set(model):
            override_model_before = self._override_model
            # noinspection PyTypeChecker
            self._override_model = _utils.Some(models.infer_model(model))  # pyright: ignore[reportArgumentType]
        else:
            override_model_before = _utils.UNSET

        try:
            yield
        finally:
            if _utils.is_set(override_deps_before):
                self._override_deps = override_deps_before
            if _utils.is_set(override_model_before):
                self._override_model = override_model_before

    @overload
    def system_prompt(
        self, func: Callable[[RunContext[AgentDeps]], str], /
    ) -> Callable[[RunContext[AgentDeps]], str]: ...

    @overload
    def system_prompt(
        self, func: Callable[[RunContext[AgentDeps]], Awaitable[str]], /
    ) -> Callable[[RunContext[AgentDeps]], Awaitable[str]]: ...

    @overload
    def system_prompt(self, func: Callable[[], str], /) -> Callable[[], str]: ...

    @overload
    def system_prompt(self, func: Callable[[], Awaitable[str]], /) -> Callable[[], Awaitable[str]]: ...

    @overload
    def system_prompt(
        self, /, *, dynamic: bool = False
    ) -> Callable[[_system_prompt.SystemPromptFunc[AgentDeps]], _system_prompt.SystemPromptFunc[AgentDeps]]: ...

    def system_prompt(
        self,
        func: _system_prompt.SystemPromptFunc[AgentDeps] | None = None,
        /,
        *,
        dynamic: bool = False,
    ) -> (
        Callable[[_system_prompt.SystemPromptFunc[AgentDeps]], _system_prompt.SystemPromptFunc[AgentDeps]]
        | _system_prompt.SystemPromptFunc[AgentDeps]
    ):
        """Decorator to register a system prompt function.

        Optionally takes [`RunContext`][pydantic_ai.tools.RunContext] as its only argument.
        Can decorate a sync or async functions.

        The decorator can be used either bare (`agent.system_prompt`) or as a function call
        (`agent.system_prompt(...)`), see the examples below.

        Overloads for every possible signature of `system_prompt` are included so the decorator doesn't obscure
        the type of the function, see `tests/typed_agent.py` for tests.

        Args:
            func: The function to decorate
            dynamic: If True, the system prompt will be reevaluated even when `messages_history` is provided,
                see [`SystemPromptPart.dynamic_ref`][pydantic_ai.messages.SystemPromptPart.dynamic_ref]

        Example:
        ```python
        from pydantic_ai import Agent, RunContext

        agent = Agent('test', deps_type=str)

        @agent.system_prompt
        def simple_system_prompt() -> str:
            return 'foobar'

        @agent.system_prompt(dynamic=True)
        async def async_system_prompt(ctx: RunContext[str]) -> str:
            return f'{ctx.deps} is the best'
        ```
        """
        if func is None:

            def decorator(
                func_: _system_prompt.SystemPromptFunc[AgentDeps],
            ) -> _system_prompt.SystemPromptFunc[AgentDeps]:
                runner = _system_prompt.SystemPromptRunner(func_, dynamic=dynamic)
                self._system_prompt_functions.append(runner)
                if dynamic:
                    self._system_prompt_dynamic_functions[func_.__qualname__] = runner
                return func_

            return decorator
        else:
            assert not dynamic, "dynamic can't be True in this case"
            self._system_prompt_functions.append(_system_prompt.SystemPromptRunner(func, dynamic=dynamic))
            return func

    @overload
    def result_validator(
        self, func: Callable[[RunContext[AgentDeps], ResultData], ResultData], /
    ) -> Callable[[RunContext[AgentDeps], ResultData], ResultData]: ...

    @overload
    def result_validator(
        self, func: Callable[[RunContext[AgentDeps], ResultData], Awaitable[ResultData]], /
    ) -> Callable[[RunContext[AgentDeps], ResultData], Awaitable[ResultData]]: ...

    @overload
    def result_validator(self, func: Callable[[ResultData], ResultData], /) -> Callable[[ResultData], ResultData]: ...

    @overload
    def result_validator(
        self, func: Callable[[ResultData], Awaitable[ResultData]], /
    ) -> Callable[[ResultData], Awaitable[ResultData]]: ...

    def result_validator(
        self, func: _result.ResultValidatorFunc[AgentDeps, ResultData], /
    ) -> _result.ResultValidatorFunc[AgentDeps, ResultData]:
        """Decorator to register a result validator function.

        Optionally takes [`RunContext`][pydantic_ai.tools.RunContext] as its first argument.
        Can decorate a sync or async functions.

        Overloads for every possible signature of `result_validator` are included so the decorator doesn't obscure
        the type of the function, see `tests/typed_agent.py` for tests.

        Example:
        ```python
        from pydantic_ai import Agent, ModelRetry, RunContext

        agent = Agent('test', deps_type=str)

        @agent.result_validator
        def result_validator_simple(data: str) -> str:
            if 'wrong' in data:
                raise ModelRetry('wrong response')
            return data

        @agent.result_validator
        async def result_validator_deps(ctx: RunContext[str], data: str) -> str:
            if ctx.deps in data:
                raise ModelRetry('wrong response')
            return data

        result = agent.run_sync('foobar', deps='spam')
        print(result.data)
        #> success (no tool calls)
        ```
        """
        self._result_validators.append(_result.ResultValidator[AgentDeps, Any](func))
        return func

    @overload
    def tool(self, func: ToolFuncContext[AgentDeps, ToolParams], /) -> ToolFuncContext[AgentDeps, ToolParams]: ...

    @overload
    def tool(
        self,
        /,
        *,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDeps] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
    ) -> Callable[[ToolFuncContext[AgentDeps, ToolParams]], ToolFuncContext[AgentDeps, ToolParams]]: ...

    def tool(
        self,
        func: ToolFuncContext[AgentDeps, ToolParams] | None = None,
        /,
        *,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDeps] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
    ) -> Any:
        """Decorator to register a tool function which takes [`RunContext`][pydantic_ai.tools.RunContext] as its first argument.

        Can decorate a sync or async functions.

        The docstring is inspected to extract both the tool description and description of each parameter,
        [learn more](../tools.md#function-tools-and-schema).

        We can't add overloads for every possible signature of tool, since the return type is a recursive union
        so the signature of functions decorated with `@agent.tool` is obscured.

        Example:
        ```python
        from pydantic_ai import Agent, RunContext

        agent = Agent('test', deps_type=int)

        @agent.tool
        def foobar(ctx: RunContext[int], x: int) -> int:
            return ctx.deps + x

        @agent.tool(retries=2)
        async def spam(ctx: RunContext[str], y: float) -> float:
            return ctx.deps + y

        result = agent.run_sync('foobar', deps=1)
        print(result.data)
        #> {"foobar":1,"spam":1.0}
        ```

        Args:
            func: The tool function to register.
            retries: The number of retries to allow for this tool, defaults to the agent's default retries,
                which defaults to 1.
            prepare: custom method to prepare the tool definition for each step, return `None` to omit this
                tool from a given step. This is useful if you want to customise a tool at call time,
                or omit it completely from a step. See [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc].
            docstring_format: The format of the docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
                Defaults to `'auto'`, such that the format is inferred from the structure of the docstring.
            require_parameter_descriptions: If True, raise an error if a parameter description is missing. Defaults to False.
        """
        if func is None:

            def tool_decorator(
                func_: ToolFuncContext[AgentDeps, ToolParams],
            ) -> ToolFuncContext[AgentDeps, ToolParams]:
                # noinspection PyTypeChecker
                self._register_function(func_, True, retries, prepare, docstring_format, require_parameter_descriptions)
                return func_

            return tool_decorator
        else:
            # noinspection PyTypeChecker
            self._register_function(func, True, retries, prepare, docstring_format, require_parameter_descriptions)
            return func

    @overload
    def tool_plain(self, func: ToolFuncPlain[ToolParams], /) -> ToolFuncPlain[ToolParams]: ...

    @overload
    def tool_plain(
        self,
        /,
        *,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDeps] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
    ) -> Callable[[ToolFuncPlain[ToolParams]], ToolFuncPlain[ToolParams]]: ...

    def tool_plain(
        self,
        func: ToolFuncPlain[ToolParams] | None = None,
        /,
        *,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDeps] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
    ) -> Any:
        """Decorator to register a tool function which DOES NOT take `RunContext` as an argument.

        Can decorate a sync or async functions.

        The docstring is inspected to extract both the tool description and description of each parameter,
        [learn more](../tools.md#function-tools-and-schema).

        We can't add overloads for every possible signature of tool, since the return type is a recursive union
        so the signature of functions decorated with `@agent.tool` is obscured.

        Example:
        ```python
        from pydantic_ai import Agent, RunContext

        agent = Agent('test')

        @agent.tool
        def foobar(ctx: RunContext[int]) -> int:
            return 123

        @agent.tool(retries=2)
        async def spam(ctx: RunContext[str]) -> float:
            return 3.14

        result = agent.run_sync('foobar', deps=1)
        print(result.data)
        #> {"foobar":123,"spam":3.14}
        ```

        Args:
            func: The tool function to register.
            retries: The number of retries to allow for this tool, defaults to the agent's default retries,
                which defaults to 1.
            prepare: custom method to prepare the tool definition for each step, return `None` to omit this
                tool from a given step. This is useful if you want to customise a tool at call time,
                or omit it completely from a step. See [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc].
            docstring_format: The format of the docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
                Defaults to `'auto'`, such that the format is inferred from the structure of the docstring.
            require_parameter_descriptions: If True, raise an error if a parameter description is missing. Defaults to False.
        """
        if func is None:

            def tool_decorator(func_: ToolFuncPlain[ToolParams]) -> ToolFuncPlain[ToolParams]:
                # noinspection PyTypeChecker
                self._register_function(
                    func_, False, retries, prepare, docstring_format, require_parameter_descriptions
                )
                return func_

            return tool_decorator
        else:
            self._register_function(func, False, retries, prepare, docstring_format, require_parameter_descriptions)
            return func

    def _register_function(
        self,
        func: ToolFuncEither[AgentDeps, ToolParams],
        takes_ctx: bool,
        retries: int | None,
        prepare: ToolPrepareFunc[AgentDeps] | None,
        docstring_format: DocstringFormat,
        require_parameter_descriptions: bool,
    ) -> None:
        """Private utility to register a function as a tool."""
        retries_ = retries if retries is not None else self._default_retries
        tool = Tool(
            func,
            takes_ctx=takes_ctx,
            max_retries=retries_,
            prepare=prepare,
            docstring_format=docstring_format,
            require_parameter_descriptions=require_parameter_descriptions,
        )
        self._register_tool(tool)

    def _register_tool(self, tool: Tool[AgentDeps]) -> None:
        """Private utility to register a tool instance."""
        if tool.max_retries is None:
            # noinspection PyTypeChecker
            tool = dataclasses.replace(tool, max_retries=self._default_retries)

        if tool.name in self._function_tools:
            raise exceptions.UserError(f'Tool name conflicts with existing tool: {tool.name!r}')

        if self._result_schema and tool.name in self._result_schema.tools:
            raise exceptions.UserError(f'Tool name conflicts with result schema name: {tool.name!r}')

        self._function_tools[tool.name] = tool

    async def _get_model(self, model: models.Model | models.KnownModelName | None) -> models.Model:
        """Create a model configured for this agent.

        Args:
            model: model to use for this run, required if `model` was not set when creating the agent.

        Returns:
            The model used
        """
        model_: models.Model
        if some_model := self._override_model:
            # we don't want `override()` to cover up errors from the model not being defined, hence this check
            if model is None and self.model is None:
                raise exceptions.UserError(
                    '`model` must be set either when creating the agent or when calling it. '
                    '(Even when `override(model=...)` is customizing the model that will actually be called)'
                )
            model_ = some_model.value
        elif model is not None:
            model_ = models.infer_model(model)
        elif self.model is not None:
            # noinspection PyTypeChecker
            model_ = self.model = models.infer_model(self.model)
        else:
            raise exceptions.UserError('`model` must be set either when creating the agent or when calling it.')

        return model_

    def _get_deps(self, deps: AgentDeps) -> AgentDeps:
        """Get deps for a run.

        If we've overridden deps via `_override_deps`, use that, otherwise use the deps passed to the call.

        We could do runtime type checking of deps against `self._deps_type`, but that's a slippery slope.
        """
        if some_deps := self._override_deps:
            return some_deps.value
        else:
            return deps

    def _infer_name(self, function_frame: FrameType | None) -> None:
        """Infer the agent name from the call frame.

        Usage should be `self._infer_name(inspect.currentframe())`.
        """
        assert self.name is None, 'Name already set'
        if function_frame is not None:  # pragma: no branch
            if parent_frame := function_frame.f_back:  # pragma: no branch
                for name, item in parent_frame.f_locals.items():
                    if item is self:
                        self.name = name
                        return
                if parent_frame.f_locals != parent_frame.f_globals:
                    # if we couldn't find the agent in locals and globals are a different dict, try globals
                    for name, item in parent_frame.f_globals.items():
                        if item is self:
                            self.name = name
                            return

    @property
    @deprecated(
        'The `last_run_messages` attribute has been removed, use `capture_run_messages` instead.', category=None
    )
    def last_run_messages(self) -> list[_messages.ModelMessage]:
        raise AttributeError('The `last_run_messages` attribute has been removed; use `capture_run_messages` instead.')

    def _build_graph(
        self, result_type: type[RunResultData] | None
    ) -> Graph[GraphAgentState, GraphAgentDeps[AgentDeps, Any], Any]:
        # We'll define the known node classes:
        nodes = [
            UserPromptNode[AgentDeps],
            ModelRequestNode[AgentDeps],
            ModelResponseNode[AgentDeps],
            FinalResultNode[AgentDeps, MarkFinalResult[ResultData]],
        ]
        graph: Graph[GraphAgentState, GraphAgentDeps[AgentDeps, Any], MarkFinalResult[Any]] = Graph(
            nodes=nodes,
            name=self.name or 'Agent',
            state_type=GraphAgentState,
            run_end_type=MarkFinalResult[result_type or self.result_type],
        )
        return graph

    def _prepare_result_schema(
        self, result_type: type[RunResultData] | None
    ) -> _result.ResultSchema[RunResultData] | None:
        if result_type is not None:
            if self._result_validators:
                raise exceptions.UserError('Cannot set a custom run `result_type` when the agent has result validators')
            return _result.ResultSchema[result_type].build(
                result_type, self._result_tool_name, self._result_tool_description
            )
        else:
            return self._result_schema  # pyright: ignore[reportReturnType]


def _build_run_context(ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, Any]]) -> RunContext[DepsT]:
    return RunContext[DepsT](
        deps=ctx.deps.user_deps,
        model=ctx.deps.model,
        usage=ctx.state.usage,
        prompt=ctx.deps.prompt,
        messages=ctx.state.message_history,
        run_step=ctx.state.run_step,
    )


def _allow_text_result(result_schema: _result.ResultSchema[RunResultData] | None) -> bool:
    return result_schema is None or result_schema.allow_text_result
