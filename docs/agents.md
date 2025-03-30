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