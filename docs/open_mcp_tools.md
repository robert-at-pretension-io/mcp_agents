# Comprehensive Guide to Writing OpenAI Agents with MCP Tool Usage

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started with OpenAI Agents SDK](#getting-started)
3. [Creating Agents](#creating-agents)
   - [Basic Configuration](#basic-configuration)
   - [Context Management](#context-management)
   - [Output Types](#output-types)
   - [Dynamic Instructions](#dynamic-instructions)
   - [Lifecycle Events (Hooks)](#lifecycle-events)
   - [Cloning/Copying Agents](#cloning-copying-agents)
   - [Forcing Tool Use](#forcing-tool-use)
4. [Tools](#tools)
   - [Hosted Tools](#hosted-tools)
   - [Function Tools](#function-tools)
   - [Custom Function Tools](#custom-function-tools)
   - [Automatic Argument and Docstring Parsing](#automatic-argument-parsing)
   - [Agents as Tools](#agents-as-tools)
   - [Handling Errors in Function Tools](#handling-errors)
5. [Model Context Protocol (MCP)](#model-context-protocol)
   - [Understanding MCP](#understanding-mcp)
   - [MCP Servers](#mcp-servers)
   - [Using MCP Servers with Agents](#using-mcp-servers-with-agents)
   - [Caching MCP Tools](#caching-mcp-tools)
6. [Running Agents](#running-agents)
   - [The Agent Loop](#agent-loop)
   - [Streaming](#streaming)
   - [Run Configuration](#run-configuration)
   - [Conversations/Chat Threads](#conversations-chat-threads)
   - [Handling Exceptions](#handling-exceptions)
7. [Advanced Features](#advanced-features)
   - [Handoffs](#handoffs)
   - [Guardrails](#guardrails)
   - [Orchestrating Multiple Agents](#orchestrating-multiple-agents)
8. [Best Practices and Tips](#best-practices)
9. [Example Applications](#example-applications)

## Introduction <a name="introduction"></a>

The OpenAI Agents SDK provides a powerful framework for building and deploying AI agents with capabilities beyond standard LLM interactions. These agents can use tools, make decisions, delegate to specialized sub-agents, and maintain contextual awareness. The Model Context Protocol (MCP) extends these capabilities by providing a standardized way to connect AI models to different data sources and tools.

This guide will walk you through creating, configuring, and deploying agents with a special focus on integrating MCP tools.

## Getting Started with OpenAI Agents SDK <a name="getting-started"></a>

First, install the OpenAI Agents SDK:

```bash
pip install openai-agents
```

The SDK requires Python 3.8 or later. The core components you'll be working with include:

- **Agents**: The main building blocks configured with instructions and tools
- **Tools**: Functions that agents can use to perform actions
- **Runner**: The component that executes agents
- **MCP**: An open protocol for providing context and tools to language models

## Creating Agents <a name="creating-agents"></a>

### Basic Configuration <a name="basic-configuration"></a>

An agent is essentially a large language model (LLM) configured with instructions and tools. Here's a basic example:

```python
from agents import Agent, ModelSettings, function_tool

@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny"

agent = Agent(
    name="Weather Agent",
    instructions="You help users with weather information.",
    model="o3-mini",  # OpenAI model identifier
    tools=[get_weather],
)
```

The most important properties to configure are:
- `instructions`: The system prompt or developer message
- `model`: Which LLM to use
- `tools`: Tools that the agent can use
- `model_settings`: Optional configuration for model parameters (temperature, top_p, etc.)

### Context Management <a name="context-management"></a>

Agents are generic on their context type. Context is a dependency-injection tool that's passed to the agent, tools, and handoffs during execution.

```python
from dataclasses import dataclass
from typing import List

@dataclass
class UserContext:
    uid: str
    is_pro_user: bool
    
    async def fetch_purchases(self) -> list:
        # Implementation for fetching user purchases
        return []

# Type-annotate the agent with the context type
agent = Agent[UserContext](
    name="Context-aware Agent",
    instructions="You help users based on their purchase history.",
    # other configuration...
)
```

### Output Types <a name="output-types"></a>

By default, agents produce plain text (string) outputs. For structured outputs, use the `output_type` parameter:

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

This tells the model to use structured outputs instead of plain text.

### Dynamic Instructions <a name="dynamic-instructions"></a>

You can provide dynamic instructions via a function that receives the agent and context:

```python
def dynamic_instructions(
    context: RunContextWrapper[UserContext], 
    agent: Agent[UserContext]
) -> str:
    return f"The user's name is {context.context.name}. Help them with their questions."

agent = Agent[UserContext](
    name="Dynamic Agent",
    instructions=dynamic_instructions,
    # other configuration...
)
```

The function can be either synchronous or asynchronous.

### Lifecycle Events (Hooks) <a name="lifecycle-events"></a>

You can observe the lifecycle of an agent by using hooks:

```python
from agents import AgentHooks

class MyAgentHooks(AgentHooks):
    async def on_tool_start(self, context, agent, tool_name, tool_input):
        print(f"Starting tool: {tool_name}")
    
    async def on_tool_end(self, context, agent, tool_name, tool_input, tool_output):
        print(f"Tool completed: {tool_name}")

agent = Agent(
    name="Hooked Agent",
    instructions="You help users with various tasks.",
    hooks=MyAgentHooks(),
    # other configuration...
)
```

### Cloning/Copying Agents <a name="cloning-copying-agents"></a>

You can duplicate an agent and modify properties using the `clone()` method:

```python
base_agent = Agent(
    name="Base Agent",
    instructions="You are a helpful assistant",
    model="o3-mini",
)

specialized_agent = base_agent.clone(
    name="Specialized Agent",
    instructions="You are an expert in Python programming",
)
```

### Forcing Tool Use <a name="forcing-tool-use"></a>

You can control when and how an agent uses tools by setting `ModelSettings.tool_choice`:

```python
from agents import ModelSettings

agent = Agent(
    name="Tool User",
    instructions="You help users by using tools.",
    model="o3-mini",
    tools=[get_weather],
    model_settings=ModelSettings(
        tool_choice="required"  # Forces the agent to use a tool
    )
)
```

Valid values for `tool_choice` are:
- `"auto"`: Let the LLM decide whether to use a tool
- `"required"`: Make the LLM use a tool, but it can choose which one
- `"none"`: Don't use any tools
- A specific tool name (e.g., `"get_weather"`): Use that specific tool

To prevent infinite loops, the framework automatically resets `tool_choice` to `"auto"` after a tool call. You can modify this behavior with `agent.reset_tool_choice`.

If you want the agent to stop after the first tool call:

```python
agent = Agent(
    # other configuration...
    tool_use_behavior="stop_on_first_tool"  # Use tool output as final response
)
```

## Tools <a name="tools"></a>

Tools allow agents to take actions—fetching data, calling APIs, running code, etc. The Agents SDK supports several types of tools.

### Hosted Tools <a name="hosted-tools"></a>

OpenAI offers built-in tools when using the `OpenAIResponsesModel`:

```python
from agents import Agent, FileSearchTool, Runner, WebSearchTool

agent = Agent(
    name="Assistant",
    tools=[
        WebSearchTool(),  # Search the web
        FileSearchTool(
            max_num_results=3,
            vector_store_ids=["YOUR_VECTOR_STORE_ID"],
        ),  # Retrieve from OpenAI Vector Stores
    ],
)
```

Available hosted tools include:
- `WebSearchTool`: Search the web
- `FileSearchTool`: Retrieve information from OpenAI Vector Stores
- `ComputerTool`: Automate computer use tasks

### Function Tools <a name="function-tools"></a>

You can turn any Python function into a tool using the `@function_tool` decorator:

```python
import json
from typing_extensions import TypedDict, Any
from agents import Agent, FunctionTool, RunContextWrapper, function_tool

class Location(TypedDict):
    lat: float
    long: float

@function_tool
async def fetch_weather(location: Location) -> str:
    """Fetch the weather for a given location.
    
    Args:
        location: The location to fetch the weather for.
    """
    # In a real implementation, you'd call a weather API
    return "sunny"

@function_tool(name_override="fetch_data")
def read_file(ctx: RunContextWrapper[Any], path: str, directory: str | None = None) -> str:
    """Read the contents of a file.
    
    Args:
        path: The path to the file to read.
        directory: The directory to read the file from.
    """
    # Implementation for reading a file
    return "file contents"
```

The SDK automatically extracts:
- Tool name from the function name (or override)
- Description from the function's docstring
- Parameter schema from type annotations
- Parameter descriptions from docstring

### Custom Function Tools <a name="custom-function-tools"></a>

You can create tools directly without using Python functions:

```python
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
```

### Automatic Argument and Docstring Parsing <a name="automatic-argument-parsing"></a>

The SDK automatically parses function signatures and docstrings to create tool schemas:

- Signature parsing uses Python's `inspect` module
- Docstring parsing uses `griffe` and supports multiple formats (Google, Sphinx, NumPy)
- You can disable docstring parsing with `use_docstring_info=False`
- You can specify the docstring format when calling `function_tool`

### Agents as Tools <a name="agents-as-tools"></a>

You can use agents as tools, enabling orchestration of specialized agents:

```python
from agents import Agent, Runner
import asyncio

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You translate the user's message to Spanish",
)

french_agent = Agent(
    name="French agent",
    instructions="You translate the user's message to French",
)

orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "You are a translation agent. You use the tools given to you to translate."
        "If asked for multiple translations, you call the relevant tools."
    ),
    tools=[
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translate the user's message to Spanish",
        ),
        french_agent.as_tool(
            tool_name="translate_to_french",
            tool_description="Translate the user's message to French",
        ),
    ],
)
```

This allows central agents to orchestrate a network of specialized agents without handing off control.

### Handling Errors in Function Tools <a name="handling-errors"></a>

When creating function tools, you can handle errors by providing a `failure_error_function`:

```python
def custom_error_handler(error: Exception) -> str:
    return f"The tool encountered an error: {str(error)}"

@function_tool(failure_error_function=custom_error_handler)
def risky_function(data: str) -> str:
    # This might raise an exception
    if data == "error":
        raise ValueError("Invalid input")
    return "success"
```

Options for error handling:
- Default: Uses `default_tool_error_function` to inform the LLM of the error
- Custom function: Pass your own error handler
- `None`: Re-raise the exception to be handled by your code

## Model Context Protocol (MCP) <a name="model-context-protocol"></a>

### Understanding MCP <a name="understanding-mcp"></a>

The Model Context Protocol (MCP) is an open standard for providing context and tools to language models. Think of it as a "USB-C port" for AI applications, providing a standardized way to connect models to different data sources and tools.

The Agents SDK has built-in support for MCP, enabling you to use a wide range of MCP servers with your agents.

### MCP Servers <a name="mcp-servers"></a>

MCP defines two types of servers, based on their transport mechanism:

1. **stdio servers**: Run as subprocesses of your application (locally)
2. **HTTP over SSE servers**: Run remotely and are connected to via a URL

You can use the `MCPServerStdio` and `MCPServerSse` classes to connect to these servers.

Here's an example using the official MCP filesystem server:

```python
from agents import MCPServerStdio

async with MCPServerStdio(
    params={
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/files"],
    }
) as server:
    tools = await server.list_tools()
```

### Using MCP Servers with Agents <a name="using-mcp-servers-with-agents"></a>

You can add MCP servers to agents through the `mcp_servers` parameter:

```python
from agents import Agent, MCPServerStdio, Runner

async def main():
    # Create MCP server
    mcp_server = MCPServerStdio(
        params={
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/files"],
        }
    )
    
    # Create agent with MCP server
    agent = Agent(
        name="Assistant",
        instructions="Use the tools to achieve the task",
        mcp_servers=[mcp_server],
    )
    
    # Run the agent
    result = await Runner.run(
        agent,
        "Find all PDF files in the directory and summarize their contents"
    )
    print(result.final_output)
```

When an agent runs:
1. The SDK calls `list_tools()` on the MCP servers
2. This makes the LLM aware of the available MCP tools
3. When the LLM calls a tool from an MCP server, the SDK calls `call_tool()` on that server

### Caching MCP Tools <a name="caching-mcp-tools"></a>

To improve performance, you can cache the list of tools from MCP servers:

```python
mcp_server = MCPServerStdio(
    params={
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/files"],
    },
    cache_tools_list=True  # Enable caching
)
```

Only do this if you're certain the tool list won't change. To invalidate the cache:

```python
await mcp_server.invalidate_tools_cache()
```

## Running Agents <a name="running-agents"></a>

You run agents using the `Runner` class, which provides three main methods:

```python
from agents import Agent, Runner

async def main():
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant",
    )
    
    # Async run (recommended)
    result = await Runner.run(agent, "Write a haiku about recursion in programming.")
    print(result.final_output)
    
    # Sync run (convenience wrapper)
    result = Runner.run_sync(agent, "Tell me a joke.")
    print(result.final_output)
    
    # Streaming run (for real-time updates)
    result = await Runner.run_streamed(agent, "Explain quantum computing.")
    for event in result.stream_events():
        # Process streaming events
        pass
    print(result.final_output)
```

### The Agent Loop <a name="agent-loop"></a>

When you run an agent, the Runner executes a loop:

1. Call the LLM for the current agent with the current input
2. Process the LLM's output:
   - If it's a final output (text with no tool calls), end the loop and return the result
   - If it's a handoff, update the current agent and input, then re-run the loop
   - If it contains tool calls, run those tools, append the results, and re-run the loop
3. If the loop exceeds `max_turns`, raise a `MaxTurnsExceeded` exception

### Streaming <a name="streaming"></a>

Streaming allows you to receive real-time events as the LLM generates output:

```python
async def main():
    agent = Agent(name="Assistant", instructions="Be concise.")
    
    result = await Runner.run_streamed(agent, "Write a story about a robot.")
    
    # Process streaming events
    for event in result.stream_events():
        if hasattr(event, "delta") and event.delta:
            print(event.delta, end="", flush=True)
```

### Run Configuration <a name="run-configuration"></a>

You can configure global settings for an agent run using the `run_config` parameter:

```python
from agents import RunConfig, ModelSettings

run_config = RunConfig(
    model="gpt-4-turbo",  # Override agent-specific model
    model_settings=ModelSettings(temperature=0.7),  # Global model settings
    input_guardrails=[],  # Input guardrails for all agents
    output_guardrails=[],  # Output guardrails for all agents
    tracing_disabled=False,  # Enable/disable tracing
    workflow_name="Customer Support",  # Name for tracing
)

result = await Runner.run(agent, "Help me troubleshoot my printer.", run_config=run_config)
```

### Conversations/Chat Threads <a name="conversations-chat-threads"></a>

To maintain conversational context across multiple turns:

```python
from agents import Agent, Runner, trace

async def main():
    agent = Agent(name="Assistant", instructions="Reply very concisely.")
    thread_id = "user-123-thread-456"  # Unique identifier for the conversation
    
    with trace(workflow_name="Conversation", group_id=thread_id):
        # First turn
        result = await Runner.run(agent, "What city is the Golden Gate Bridge in?")
        print(result.final_output)  # "San Francisco"
        
        # Second turn, using the conversation history
        new_input = result.to_input_list() + [{
            "role": "user",
            "content": "What state is it in?"
        }]
        result = await Runner.run(agent, new_input)
        print(result.final_output)  # "California"
```

The `to_input_list()` method converts the previous result into input items that include the conversation history.

### Handling Exceptions <a name="handling-exceptions"></a>

The SDK raises several exceptions that you should handle:

```python
from agents import (
    AgentsException, MaxTurnsExceeded, ModelBehaviorError,
    UserError, InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered
)

try:
    result = await Runner.run(agent, user_input, max_turns=10)
    print(result.final_output)
except MaxTurnsExceeded:
    print("The agent ran too many turns without producing a final output.")
except ModelBehaviorError as e:
    print(f"The model produced invalid output: {e}")
except InputGuardrailTripwireTriggered as e:
    print(f"Input guardrail triggered: {e}")
except OutputGuardrailTripwireTriggered as e:
    print(f"Output guardrail triggered: {e}")
except UserError as e:
    print(f"Incorrect SDK usage: {e}")
except AgentsException as e:
    print(f"General agent error: {e}")
```

## Advanced Features <a name="advanced-features"></a>

### Handoffs <a name="handoffs"></a>

Handoffs allow one agent to delegate to specialized agents:

```python
booking_agent = Agent(
    name="Booking Agent",
    instructions="You help users book flights and hotels.",
)

refund_agent = Agent(
    name="Refund Agent",
    instructions="You help users process refunds.",
)

triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "Help the user with their questions. "
        "If they ask about booking, handoff to the booking agent. "
        "If they ask about refunds, handoff to the refund agent."
    ),
    handoffs=[booking_agent, refund_agent],
)
```

The triage agent can decide whether to handle the query itself or delegate to a specialized agent.

### Guardrails <a name="guardrails"></a>

Guardrails allow you to validate inputs and outputs:

```python
from agents import Guardrail, RunConfig

class ProfanityGuardrail(Guardrail):
    async def check(self, content: str) -> tuple[bool, str]:
        if "profanity" in content.lower():
            return False, "Content contains profanity"
        return True, ""

run_config = RunConfig(
    input_guardrails=[ProfanityGuardrail()],
)

result = await Runner.run(agent, user_input, run_config=run_config)
```

### Orchestrating Multiple Agents <a name="orchestrating-multiple-agents"></a>

You can create complex workflows by orchestrating multiple agents:

```python
# Using agents as tools (as shown earlier)
orchestrator_agent = Agent(
    name="Orchestrator",
    tools=[
        specialist_agent_1.as_tool(
            tool_name="specialist_1",
            tool_description="A specialist for task 1",
        ),
        specialist_agent_2.as_tool(
            tool_name="specialist_2",
            tool_description="A specialist for task 2",
        ),
    ],
)

# Or using handoffs for more independent delegation
coordinator_agent = Agent(
    name="Coordinator",
    handoffs=[specialist_agent_1, specialist_agent_2],
)
```

## Best Practices and Tips <a name="best-practices"></a>

1. **Clear Instructions**: Write clear, specific instructions for your agents.

2. **Tool Design**: 
   - Keep tool names descriptive and consistent
   - Write comprehensive docstrings for automatic tool descriptions
   - Return structured data when possible

3. **Context Management**:
   - Use typed contexts for dependency injection
   - Keep context objects focused and minimal

4. **Error Handling**:
   - Implement custom error handlers for important tools
   - Use try/except blocks to gracefully handle agent failures

5. **Tracing and Debugging**:
   - Enable tracing in development to understand agent behavior
   - Set meaningful workflow names for better trace organization

6. **MCP Tools**:
   - Cache tool lists for performance when they won't change
   - Use MCP for complex interactions with external systems

7. **Handoffs vs. Tools**:
   - Use handoffs when agents need independent decision-making
   - Use tools (agent-as-tool) for tighter orchestration

## Example Applications <a name="example-applications"></a>

Here's a complete example combining many of the concepts we've covered:

```python
from agents import (
    Agent, Runner, ModelSettings, function_tool, trace,
    MCPServerStdio, RunConfig
)
from dataclasses import dataclass
from typing import List

# Define context type
@dataclass
class AppContext:
    user_id: str
    preferences: dict

# Define tools
@function_tool
async def search_database(query: str) -> list:
    """Search the database for information.
    
    Args:
        query: The search query string.
    """
    # Simulated database search
    return [{"id": 1, "title": "Result 1"}, {"id": 2, "title": "Result 2"}]

@function_tool
async def send_notification(user_id: str, message: str) -> bool:
    """Send a notification to a user.
    
    Args:
        user_id: The ID of the user to notify.
        message: The notification message.
    """
    # Simulated notification
    print(f"Notification to {user_id}: {message}")
    return True

# Create specialized agents
researcher_agent = Agent[AppContext](
    name="Researcher",
    instructions="You help find information by searching the database.",
    tools=[search_database],
    model="o3-mini",
)

notifier_agent = Agent[AppContext](
    name="Notifier",
    instructions="You help send notifications to users.",
    tools=[send_notification],
    model="o3-mini",
)

# Create MCP server for file access
file_server = MCPServerStdio(
    params={
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data/files"],
    },
    cache_tools_list=True,
)

# Create main agent with orchestration capabilities
main_agent = Agent[AppContext](
    name="Assistant",
    instructions="""
    You are a helpful assistant that helps users find information and manage notifications.
    Use the database search tool to find information.
    Use the notification tool to send messages to users.
    Use file tools to access and read files when necessary.
    """,
    model="o3-turbo",
    model_settings=ModelSettings(
        temperature=0.7,
    ),
    tools=[search_database, send_notification],
    mcp_servers=[file_server],
    handoffs=[researcher_agent, notifier_agent],
)

# Run the agent
async def main():
    # Create context
    context = AppContext(
        user_id="user123",
        preferences={"language": "English", "notifications": True},
    )
    
    # Configure run
    run_config = RunConfig(
        workflow_name="UserAssistant",
        trace_metadata={"user_id": context.user_id},
    )
    
    # Run with tracing
    with trace(workflow_name="UserAssistant"):
        result = await Runner.run(
            main_agent,
            "Find information about project XYZ and notify me when you're done.",
            context=context,
            run_config=run_config,
        )
        
        print(result.final_output)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

This example showcases:
- Type-annotated context
- Multiple tools and specialized agents
- MCP integration for file access
- Handoffs for specialized tasks
- Tracing and run configuration

By combining these elements, you can build sophisticated agent systems that leverage the full power of the OpenAI Agents SDK and Model Context Protocol.