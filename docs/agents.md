from typing import Any

from pydantic import BaseModel

from agents import RunContextWrapper, FunctionTool



def do_some_work(data: str) -> str:
    return "done"


class FunctionArgs(BaseModel):
    username: str
    age: int


async def run_function(ctx: RunContextWrapper[Any], args: str) -> str:
    parsed = FunctionArgs.model_validate_json(args)
    return do_some_work(data=f"{parsed.username} is {parsed.age} years old")


tool = FunctionTool(
    name="process_user",
    description="Processes extracted user data",
    params_json_schema=FunctionArgs.model_json_schema(),
    on_invoke_tool=run_function,
)

-------------


# Agents

Agents are the core building block in your apps. An agent is a large language model (LLM), configured with instructions and tools.

## Table of Contents
- [Basic configuration](#basic-configuration)
- [Context](#context)
- [Output types](#output-types)
- [Handoffs](#handoffs)
- [Dynamic instructions](#dynamic-instructions)
- [Lifecycle events (hooks)](#lifecycle-events-hooks)
- [Guardrails](#guardrails)
- [Cloning/copying agents](#cloningcopying-agents)
- [Forcing tool use](#forcing-tool-use)

## Basic configuration

The most common properties of an agent you'll configure are:

- `instructions`: also known as a developer message or system prompt.
- `model`: which LLM to use, and optional `model_settings` to configure model tuning parameters like temperature, top_p, etc.
- `tools`: Tools that the agent can use to achieve its tasks.

```python
from agents import Agent, ModelSettings, function_tool

@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny"

agent = Agent(
    name="Haiku agent",
    instructions="Always respond in haiku form",
    model="o3-mini",
    tools=[get_weather],
)
```

## Context

Agents are generic on their context type. Context is a dependency-injection tool: it's an object you create and pass to `Runner.run()`, that is passed to every agent, tool, handoff etc, and it serves as a grab bag of dependencies and state for the agent run.

You can provide any Python object as the context.

```python
@dataclass
class UserContext:
    uid: str
    is_pro_user: bool
    
    async def fetch_purchases() -> list[Purchase]:
        return ...

agent = Agent[UserContext](
    ...,
)
```

## Output types

By default, agents produce plain text (i.e. `str`) outputs. If you want the agent to produce a particular type of output, you can use the `output_type` parameter. A common choice is to use Pydantic objects, but we support any type that can be wrapped in a Pydantic TypeAdapter - dataclasses, lists, TypedDict, etc.

```python
from pydantic import BaseModel
from agents import Agent

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

agent = Agent(
    name="Calendar extractor",
    instructions="Extract calendar events from text",
    output_type=CalendarEvent,
)
```

> **Note**
> When you pass an `output_type`, that tells the model to use structured outputs instead of regular plain text responses.

## Handoffs

Handoffs are sub-agents that the agent can delegate to. You provide a list of handoffs, and the agent can choose to delegate to them if relevant. This is a powerful pattern that allows orchestrating modular, specialized agents that excel at a single task.

Read more in the handoffs documentation.

```python
from agents import Agent

booking_agent = Agent(...)
refund_agent = Agent(...)

triage_agent = Agent(
    name="Triage agent",
    instructions=(
        "Help the user with their questions."
        "If they ask about booking, handoff to the booking agent."
        "If they ask about refunds, handoff to the refund agent."
    ),
    handoffs=[booking_agent, refund_agent],
)
```

## Dynamic instructions

In most cases, you can provide instructions when you create the agent. However, you can also provide dynamic instructions via a function. The function will receive the agent and context, and must return the prompt. Both regular and async functions are accepted.

```python
def dynamic_instructions(
    context: RunContextWrapper[UserContext],
    agent: Agent[UserContext]
) -> str:
    return f"The user's name is {context.context.name}. Help them with their questions."

agent = Agent[UserContext](
    name="Triage agent",
    instructions=dynamic_instructions,
)
```

## Lifecycle events (hooks)

Sometimes, you want to observe the lifecycle of an agent. For example, you may want to log events, or pre-fetch data when certain events occur. You can hook into the agent lifecycle with the `hooks` property. Subclass the `AgentHooks` class, and override the methods you're interested in.

## Guardrails

Guardrails allow you to run checks/validations on user input, in parallel to the agent running. For example, you could screen the user's input for relevance.

Read more in the guardrails documentation.

## Cloning/copying agents

By using the `clone()` method on an agent, you can duplicate an Agent, and optionally change any properties you like.

```python
pirate_agent = Agent(
    name="Pirate",
    instructions="Write like a pirate",
    model="o3-mini",
)

robot_agent = pirate_agent.clone(
    name="Robot",
    instructions="Write like a robot",
)
```

## Forcing tool use

Supplying a list of tools doesn't always mean the LLM will use a tool. You can force tool use by setting `ModelSettings.tool_choice`. Valid values are:

- `auto`, which allows the LLM to decide whether or not to use a tool.
- `required`, which requires the LLM to use a tool (but it can intelligently decide which tool).
- `none`, which requires the LLM to not use a tool.
- Setting a specific string e.g. `my_tool`, which requires the LLM to use that specific tool.

> **Note**
> To prevent infinite loops, the framework automatically resets `tool_choice` to "auto" after a tool call. This behavior is configurable via `agent.reset_tool_choice`. The infinite loop is because tool results are sent to the LLM, which then generates another tool call because of `tool_choice`, ad infinitum.
> 
> If you want the Agent to completely stop after a tool call (rather than continuing with auto mode), you can set `Agent.tool_use_behavior="stop_on_first_tool"` which will directly use the tool output as the final response without further LLM processing.


# Tracing in the OpenAI Agents SDK

Based on the OpenAI Agents SDK documentation, here's a detailed explanation of how to implement tracing in your applications:

## Overview of Tracing

Tracing in the OpenAI Agents SDK allows you to track and debug the execution of your agent workflows. The SDK automatically traces your agent runs, making it easy to observe the behavior of your agents, tool calls, and handoffs.

Traces represent a single end-to-end operation of a "workflow" and are composed of spans. A trace has properties like workflow_name (the logical workflow or app), trace_id (a unique ID), and group_id (to link multiple traces from the same conversation).

Spans represent operations that have a start and end time. They contain information about various activities like agent execution, LLM generation, tool calls, etc.

## Basic Tracing Usage

Here's how to use tracing in your applications:

### 1. Using Trace as a Context Manager

```python
from agents import Agent, Runner, trace

async def main():
    agent = Agent(name="Joke generator", instructions="Tell funny jokes.")
    
    # Create a trace for the entire workflow
    with trace("Joke workflow") as workflow_trace:
        # First agent run
        first_result = await Runner.run(agent, "Tell me a joke")
        
        # Second agent run within the same trace
        second_result = await Runner.run(agent, f"Rate this joke: {first_result.final_output}")
        
        print(f"Joke: {first_result.final_output}")
        print(f"Rating: {second_result.final_output}")
```

By wrapping multiple `Runner.run` calls in a `with trace()` block, the individual runs become part of the overall trace rather than creating separate traces.

### 2. Manual Trace Control

You can also manually start and finish traces:

```python
from agents import trace

# Create a trace
my_trace = trace("My Workflow")

# Start the trace
my_trace.start(mark_as_current=True)

# ... perform operations ...

# Finish the trace
my_trace.finish(reset_current=True)
```

### 3. Trace for Conversation Threads

For multi-turn conversations, you can use a consistent group_id to link traces:

```python
from agents import Agent, Runner, trace
import uuid

# Create a unique ID for this conversation
thread_id = f"thread_{uuid.uuid4().hex[:8]}"

async def conversation_turn(agent, user_message):
    # Create a trace for this turn in the conversation
    with trace(
        workflow_name="Conversation",
        group_id=thread_id,  # Link all turns in the conversation
        trace_metadata={"user_type": "customer"}
    ):
        result = await Runner.run(agent, user_message)
        return result.final_output
```

## Creating Custom Spans

You can create custom spans to track specific operations:

```python
from agents import trace, Agent, Runner

async def main():
    agent = Agent(name="Assistant", instructions="...")
    
    with trace("Main Workflow") as main_trace:
        # Create a custom span for a specific operation
        with main_trace.custom_span("data_preparation") as prep_span:
            # Add custom data to the span
            prep_span.data.update({"records_processed": 100})
            
            # ... perform data preparation ...
        
        # Run the agent after data preparation
        result = await Runner.run(agent, "Process the prepared data")
```

## Span Types

The SDK provides several specialized span types:

```python
from agents import trace

with trace("Workflow") as workflow_trace:
    # Agent span
    with workflow_trace.agent_span(
        name="Agent Execution",
        handoffs=["agent_1", "agent_2"],
        tools=["tool_1", "tool_2"]
    ):
        # Agent operations
        pass
        
    # Function span
    with workflow_trace.function_span(
        name="Function Call",
        input="Input data",
        output="Output data"
    ):
        # Function operations
        pass
        
    # Guardrail span
    with workflow_trace.guardrail_span(
        name="Input Validation",
        triggered=False
    ):
        # Guardrail operations
        pass
        
    # Handoff span
    with workflow_trace.handoff_span(
        from_agent="Source Agent",
        to_agent="Target Agent"
    ):
        # Handoff operations
        pass
```

## Customizing Trace Processors

You can customize how traces are processed using trace processors. The SDK provides two methods: `add_trace_processor()` lets you add additional processors, and `set_trace_processors()` lets you replace the default processors entirely.

```python
from agents.tracing import add_trace_processor, set_trace_processors
from agents.tracing.processor_interface import TraceProcessor
from agents.tracing.traces import Trace

class CustomTraceProcessor(TraceProcessor):
    async def process_trace(self, trace: Trace) -> None:
        print(f"Processing trace: {trace.workflow_name}")
        # Custom processing logic
    
    async def process_span(self, span) -> None:
        print(f"Processing span: {span.name}")
        # Custom processing logic

# Add a processor alongside the default ones
add_trace_processor(CustomTraceProcessor())

# Or replace all default processors
set_trace_processors([CustomTraceProcessor()])
```

## Disabling Tracing

For certain scenarios, you may want to disable tracing:

```python
from agents import Runner

# Disable tracing for a specific run
result = await Runner.run(
    agent,
    "Query",
    tracing_disabled=True
)
```

In cases where you don't have an OpenAI API key from platform.openai.com, you can disable tracing using `set_tracing_disabled()` or set up a different tracing processor.

## Viewing Traces

By default, traces are sent to the OpenAI backend, where you can view them in the OpenAI Dashboard. If you're using a custom model provider or want to avoid sending traces to OpenAI, you can either disable tracing or set up custom trace processors to handle the data differently.

## Using Traces for Conversation State

Traces can be particularly useful for maintaining conversation state across multiple turns. You can use the `group_id` parameter to link related traces, such as all turns in a single conversation:

```python
async def main():
    agent = Agent(name="Assistant", instructions="Reply very concisely.")
    thread_id = "conversation_123"
    
    with trace(workflow_name="Conversation", group_id=thread_id):
        # First turn
        result1 = await Runner.run(agent, "What city is the Golden Gate Bridge in?")
        print(result1.final_output)  # San Francisco
        
        # Second turn - same conversation
        new_input = result1.to_input_list() + [
            {"role": "user", "content": "What state is it in?"}
        ]
        result2 = await Runner.run(agent, new_input)
        print(result2.final_output)  # California
```

This detailed information should help you implement tracing effectively in your OpenAI Agents SDK applications. The tracing system provides powerful debugging and monitoring capabilities while being flexible enough to accommodate custom processing needs.



Context management
Context is an overloaded term. There are two main classes of context you might care about:

Context available locally to your code: this is data and dependencies you might need when tool functions run, during callbacks like on_handoff, in lifecycle hooks, etc.
Context available to LLMs: this is data the LLM sees when generating a response.
Local context
This is represented via the RunContextWrapper class and the context property within it. The way this works is:

You create any Python object you want. A common pattern is to use a dataclass or a Pydantic object.
You pass that object to the various run methods (e.g. Runner.run(..., **context=whatever**)).
All your tool calls, lifecycle hooks etc will be passed a wrapper object, RunContextWrapper[T], where T represents your context object type which you can access via wrapper.context.
The most important thing to be aware of: every agent, tool function, lifecycle etc for a given agent run must use the same type of context.

You can use the context for things like:

Contextual data for your run (e.g. things like a username/uid or other information about the user)
Dependencies (e.g. logger objects, data fetchers, etc)
Helper functions
Note

The context object is not sent to the LLM. It is purely a local object that you can read from, write to and call methods on it.


import asyncio
from dataclasses import dataclass

from agents import Agent, RunContextWrapper, Runner, function_tool

@dataclass
class UserInfo:  
    name: str
    uid: int

@function_tool
async def fetch_user_age(wrapper: RunContextWrapper[UserInfo]) -> str:  
    return f"User {wrapper.context.name} is 47 years old"

async def main():
    user_info = UserInfo(name="John", uid=123)

    agent = Agent[UserInfo](  
        name="Assistant",
        tools=[fetch_user_age],
    )

    result = await Runner.run(  
        starting_agent=agent,
        input="What is the age of the user?",
        context=user_info,
    )

    print(result.final_output)  
    # The user John is 47 years old.

if __name__ == "__main__":
    asyncio.run(main())


Agents
ToolsToFinalOutputFunction module-attribute

ToolsToFinalOutputFunction: TypeAlias = Callable[
    [RunContextWrapper[TContext], list[FunctionToolResult]],
    MaybeAwaitable[ToolsToFinalOutputResult],
]
A function that takes a run context and a list of tool results, and returns a ToolToFinalOutputResult.

ToolsToFinalOutputResult dataclass
Source code in src/agents/agent.py

@dataclass
class ToolsToFinalOutputResult:
    is_final_output: bool
    """Whether this is the final output. If False, the LLM will run again and receive the tool call
    output.
    """

    final_output: Any | None = None
    """The final output. Can be None if `is_final_output` is False, otherwise must match the
    `output_type` of the agent.
    """
is_final_output instance-attribute

is_final_output: bool
Whether this is the final output. If False, the LLM will run again and receive the tool call output.

final_output class-attribute instance-attribute

final_output: Any | None = None
The final output. Can be None if is_final_output is False, otherwise must match the output_type of the agent.

StopAtTools
Bases: TypedDict

Source code in src/agents/agent.py
stop_at_tool_names instance-attribute

stop_at_tool_names: list[str]
A list of tool names, any of which will stop the agent from running further.

Agent dataclass
Bases: Generic[TContext]

An agent is an AI model configured with instructions, tools, guardrails, handoffs and more.

We strongly recommend passing instructions, which is the "system prompt" for the agent. In addition, you can pass handoff_description, which is a human-readable description of the agent, used when the agent is used inside tools/handoffs.

Agents are generic on the context type. The context is a (mutable) object you create. It is passed to tool functions, handoffs, guardrails, etc.

Source code in src/agents/agent.py

@dataclass
class Agent(Generic[TContext]):
    """An agent is an AI model configured with instructions, tools, guardrails, handoffs and more.

    We strongly recommend passing `instructions`, which is the "system prompt" for the agent. In
    addition, you can pass `handoff_description`, which is a human-readable description of the
    agent, used when the agent is used inside tools/handoffs.

    Agents are generic on the context type. The context is a (mutable) object you create. It is
    passed to tool functions, handoffs, guardrails, etc.
    """

    name: str
    """The name of the agent."""

    instructions: (
        str
        | Callable[
            [RunContextWrapper[TContext], Agent[TContext]],
            MaybeAwaitable[str],
        ]
        | None
    ) = None
    """The instructions for the agent. Will be used as the "system prompt" when this agent is
    invoked. Describes what the agent should do, and how it responds.

    Can either be a string, or a function that dynamically generates instructions for the agent. If
    you provide a function, it will be called with the context and the agent instance. It must
    return a string.
    """

    handoff_description: str | None = None
    """A description of the agent. This is used when the agent is used as a handoff, so that an
    LLM knows what it does and when to invoke it.
    """

    handoffs: list[Agent[Any] | Handoff[TContext]] = field(default_factory=list)
    """Handoffs are sub-agents that the agent can delegate to. You can provide a list of handoffs,
    and the agent can choose to delegate to them if relevant. Allows for separation of concerns and
    modularity.
    """

    model: str | Model | None = None
    """The model implementation to use when invoking the LLM.

    By default, if not set, the agent will use the default model configured in
    `model_settings.DEFAULT_MODEL`.
    """

    model_settings: ModelSettings = field(default_factory=ModelSettings)
    """Configures model-specific tuning parameters (e.g. temperature, top_p).
    """

    tools: list[Tool] = field(default_factory=list)
    """A list of tools that the agent can use."""

    mcp_servers: list[MCPServer] = field(default_factory=list)
    """A list of [Model Context Protocol](https://modelcontextprotocol.io/) servers that
    the agent can use. Every time the agent runs, it will include tools from these servers in the
    list of available tools.

    NOTE: You are expected to manage the lifecycle of these servers. Specifically, you must call
    `server.connect()` before passing it to the agent, and `server.cleanup()` when the server is no
    longer needed.
    """

    input_guardrails: list[InputGuardrail[TContext]] = field(default_factory=list)
    """A list of checks that run in parallel to the agent's execution, before generating a
    response. Runs only if the agent is the first agent in the chain.
    """

    output_guardrails: list[OutputGuardrail[TContext]] = field(default_factory=list)
    """A list of checks that run on the final output of the agent, after generating a response.
    Runs only if the agent produces a final output.
    """

    output_type: type[Any] | None = None
    """The type of the output object. If not provided, the output will be `str`."""

    hooks: AgentHooks[TContext] | None = None
    """A class that receives callbacks on various lifecycle events for this agent.
    """

    tool_use_behavior: (
        Literal["run_llm_again", "stop_on_first_tool"] | StopAtTools | ToolsToFinalOutputFunction
    ) = "run_llm_again"
    """This lets you configure how tool use is handled.
    - "run_llm_again": The default behavior. Tools are run, and then the LLM receives the results
        and gets to respond.
    - "stop_on_first_tool": The output of the first tool call is used as the final output. This
        means that the LLM does not process the result of the tool call.
    - A list of tool names: The agent will stop running if any of the tools in the list are called.
        The final output will be the output of the first matching tool call. The LLM does not
        process the result of the tool call.
    - A function: If you pass a function, it will be called with the run context and the list of
      tool results. It must return a `ToolToFinalOutputResult`, which determines whether the tool
      calls result in a final output.

      NOTE: This configuration is specific to FunctionTools. Hosted tools, such as file search,
      web search, etc are always processed by the LLM.
    """

    reset_tool_choice: bool = True
    """Whether to reset the tool choice to the default value after a tool has been called. Defaults
    to True. This ensures that the agent doesn't enter an infinite loop of tool usage."""

    def clone(self, **kwargs: Any) -> Agent[TContext]:
        """Make a copy of the agent, with the given arguments changed. For example, you could do:
        ```
        new_agent = agent.clone(instructions="New instructions")
        ```
        """
        return dataclasses.replace(self, **kwargs)

    def as_tool(
        self,
        tool_name: str | None,
        tool_description: str | None,
        custom_output_extractor: Callable[[RunResult], Awaitable[str]] | None = None,
    ) -> Tool:
        """Transform this agent into a tool, callable by other agents.

        This is different from handoffs in two ways:
        1. In handoffs, the new agent receives the conversation history. In this tool, the new agent
           receives generated input.
        2. In handoffs, the new agent takes over the conversation. In this tool, the new agent is
           called as a tool, and the conversation is continued by the original agent.

        Args:
            tool_name: The name of the tool. If not provided, the agent's name will be used.
            tool_description: The description of the tool, which should indicate what it does and
                when to use it.
            custom_output_extractor: A function that extracts the output from the agent. If not
                provided, the last message from the agent will be used.
        """

        @function_tool(
            name_override=tool_name or _transforms.transform_string_function_style(self.name),
            description_override=tool_description or "",
        )
        async def run_agent(context: RunContextWrapper, input: str) -> str:
            from .run import Runner

            output = await Runner.run(
                starting_agent=self,
                input=input,
                context=context.context,
            )
            if custom_output_extractor:
                return await custom_output_extractor(output)

            return ItemHelpers.text_message_outputs(output.new_items)

        return run_agent

    async def get_system_prompt(self, run_context: RunContextWrapper[TContext]) -> str | None:
        """Get the system prompt for the agent."""
        if isinstance(self.instructions, str):
            return self.instructions
        elif callable(self.instructions):
            if inspect.iscoroutinefunction(self.instructions):
                return await cast(Awaitable[str], self.instructions(run_context, self))
            else:
                return cast(str, self.instructions(run_context, self))
        elif self.instructions is not None:
            logger.error(f"Instructions must be a string or a function, got {self.instructions}")

        return None

    async def get_mcp_tools(self) -> list[Tool]:
        """Fetches the available tools from the MCP servers."""
        return await MCPUtil.get_all_function_tools(self.mcp_servers)

    async def get_all_tools(self) -> list[Tool]:
        """All agent tools, including MCP tools and function tools."""
        mcp_tools = await self.get_mcp_tools()
        return mcp_tools + self.tools
name instance-attribute

name: str
The name of the agent.

instructions class-attribute instance-attribute

instructions: (
    str
    | Callable[
        [RunContextWrapper[TContext], Agent[TContext]],
        MaybeAwaitable[str],
    ]
    | None
) = None
The instructions for the agent. Will be used as the "system prompt" when this agent is invoked. Describes what the agent should do, and how it responds.

Can either be a string, or a function that dynamically generates instructions for the agent. If you provide a function, it will be called with the context and the agent instance. It must return a string.

handoff_description class-attribute instance-attribute

handoff_description: str | None = None
A description of the agent. This is used when the agent is used as a handoff, so that an LLM knows what it does and when to invoke it.

handoffs class-attribute instance-attribute

handoffs: list[Agent[Any] | Handoff[TContext]] = field(
    default_factory=list
)
Handoffs are sub-agents that the agent can delegate to. You can provide a list of handoffs, and the agent can choose to delegate to them if relevant. Allows for separation of concerns and modularity.

model class-attribute instance-attribute

model: str | Model | None = None
The model implementation to use when invoking the LLM.

By default, if not set, the agent will use the default model configured in model_settings.DEFAULT_MODEL.

model_settings class-attribute instance-attribute

model_settings: ModelSettings = field(
    default_factory=ModelSettings
)
Configures model-specific tuning parameters (e.g. temperature, top_p).

tools class-attribute instance-attribute

tools: list[Tool] = field(default_factory=list)
A list of tools that the agent can use.

mcp_servers class-attribute instance-attribute

mcp_servers: list[MCPServer] = field(default_factory=list)
A list of Model Context Protocol servers that the agent can use. Every time the agent runs, it will include tools from these servers in the list of available tools.

NOTE: You are expected to manage the lifecycle of these servers. Specifically, you must call server.connect() before passing it to the agent, and server.cleanup() when the server is no longer needed.

input_guardrails class-attribute instance-attribute

input_guardrails: list[InputGuardrail[TContext]] = field(
    default_factory=list
)
A list of checks that run in parallel to the agent's execution, before generating a response. Runs only if the agent is the first agent in the chain.

output_guardrails class-attribute instance-attribute

output_guardrails: list[OutputGuardrail[TContext]] = field(
    default_factory=list
)
A list of checks that run on the final output of the agent, after generating a response. Runs only if the agent produces a final output.

output_type class-attribute instance-attribute

output_type: type[Any] | None = None
The type of the output object. If not provided, the output will be str.

hooks class-attribute instance-attribute

hooks: AgentHooks[TContext] | None = None
A class that receives callbacks on various lifecycle events for this agent.

tool_use_behavior class-attribute instance-attribute

tool_use_behavior: (
    Literal["run_llm_again", "stop_on_first_tool"]
    | StopAtTools
    | ToolsToFinalOutputFunction
) = "run_llm_again"
This lets you configure how tool use is handled. - "run_llm_again": The default behavior. Tools are run, and then the LLM receives the results and gets to respond. - "stop_on_first_tool": The output of the first tool call is used as the final output. This means that the LLM does not process the result of the tool call. - A list of tool names: The agent will stop running if any of the tools in the list are called. The final output will be the output of the first matching tool call. The LLM does not process the result of the tool call. - A function: If you pass a function, it will be called with the run context and the list of tool results. It must return a ToolToFinalOutputResult, which determines whether the tool calls result in a final output.

NOTE: This configuration is specific to FunctionTools. Hosted tools, such as file search, web search, etc are always processed by the LLM.

reset_tool_choice class-attribute instance-attribute

reset_tool_choice: bool = True
Whether to reset the tool choice to the default value after a tool has been called. Defaults to True. This ensures that the agent doesn't enter an infinite loop of tool usage.

clone

clone(**kwargs: Any) -> Agent[TContext]
Make a copy of the agent, with the given arguments changed. For example, you could do:


new_agent = agent.clone(instructions="New instructions")
Source code in src/agents/agent.py

def clone(self, **kwargs: Any) -> Agent[TContext]:
    """Make a copy of the agent, with the given arguments changed. For example, you could do:
    ```
    new_agent = agent.clone(instructions="New instructions")
    ```
    """
    return dataclasses.replace(self, **kwargs)
as_tool

as_tool(
    tool_name: str | None,
    tool_description: str | None,
    custom_output_extractor: Callable[
        [RunResult], Awaitable[str]
    ]
    | None = None,
) -> Tool
Transform this agent into a tool, callable by other agents.

This is different from handoffs in two ways: 1. In handoffs, the new agent receives the conversation history. In this tool, the new agent receives generated input. 2. In handoffs, the new agent takes over the conversation. In this tool, the new agent is called as a tool, and the conversation is continued by the original agent.

Parameters:

Name	Type	Description	Default
tool_name	str | None	The name of the tool. If not provided, the agent's name will be used.	required
tool_description	str | None	The description of the tool, which should indicate what it does and when to use it.	required
custom_output_extractor	Callable[[RunResult], Awaitable[str]] | None	A function that extracts the output from the agent. If not provided, the last message from the agent will be used.	None
Source code in src/agents/agent.py

def as_tool(
    self,
    tool_name: str | None,
    tool_description: str | None,
    custom_output_extractor: Callable[[RunResult], Awaitable[str]] | None = None,
) -> Tool:
    """Transform this agent into a tool, callable by other agents.

    This is different from handoffs in two ways:
    1. In handoffs, the new agent receives the conversation history. In this tool, the new agent
       receives generated input.
    2. In handoffs, the new agent takes over the conversation. In this tool, the new agent is
       called as a tool, and the conversation is continued by the original agent.

    Args:
        tool_name: The name of the tool. If not provided, the agent's name will be used.
        tool_description: The description of the tool, which should indicate what it does and
            when to use it.
        custom_output_extractor: A function that extracts the output from the agent. If not
            provided, the last message from the agent will be used.
    """

    @function_tool(
        name_override=tool_name or _transforms.transform_string_function_style(self.name),
        description_override=tool_description or "",
    )
    async def run_agent(context: RunContextWrapper, input: str) -> str:
        from .run import Runner

        output = await Runner.run(
            starting_agent=self,
            input=input,
            context=context.context,
        )
        if custom_output_extractor:
            return await custom_output_extractor(output)

        return ItemHelpers.text_message_outputs(output.new_items)

    return run_agent
get_system_prompt async

get_system_prompt(
    run_context: RunContextWrapper[TContext],
) -> str | None
Get the system prompt for the agent.

Source code in src/agents/agent.py

async def get_system_prompt(self, run_context: RunContextWrapper[TContext]) -> str | None:
    """Get the system prompt for the agent."""
    if isinstance(self.instructions, str):
        return self.instructions
    elif callable(self.instructions):
        if inspect.iscoroutinefunction(self.instructions):
            return await cast(Awaitable[str], self.instructions(run_context, self))
        else:
            return cast(str, self.instructions(run_context, self))
    elif self.instructions is not None:
        logger.error(f"Instructions must be a string or a function, got {self.instructions}")

    return None
get_mcp_tools async

get_mcp_tools() -> list[Tool]
Fetches the available tools from the MCP servers.

Source code in src/agents/agent.py

async def get_mcp_tools(self) -> list[Tool]:
    """Fetches the available tools from the MCP servers."""
    return await MCPUtil.get_all_function_tools(self.mcp_servers)
get_all_tools async

get_all_tools() -> list[Tool]
All agent tools, including MCP tools and function tools.

Source code in src/agents/agent.py

async def get_all_tools(self) -> list[Tool]:
    """All agent tools, including MCP tools and function tools."""
    mcp_tools = await self.get_mcp_tools()
    return mcp_tools + self.tools