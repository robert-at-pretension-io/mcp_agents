I'll look up and document the `ToolCallItem` in the OpenAI Agents Python SDK.



Let's examine the `ToolCallItem` class definition:









Let's get more context by looking at the related types and see where `ToolCallItemTypes` is defined:










Let's look at the beginning of the file to understand the `RunItemBase` class and the imports:











Now let's check how the `ToolCallItem` is used in practice:












Let's see how tool calls are processed in the `_run_impl.py` file:













Let's also look at how the tools are called:











Let's try another approach:























Let's find a more comprehensive view of how the tool calls are processed:












Let's look for functions that handle executing tools:













Let's try looking at a broader section of the implementation code:














Let's check the main runner class to see how it processes tools:









Let's search for tool execution in a different way:














Let's look at the implementation file more comprehensively:






























Let's look at an example file that uses tool calls:













Let's try looking at a simpler example:















Let's find a basic example:













Based on my examination of the OpenAI Agents Python SDK codebase, I can now provide comprehensive documentation on the `ToolCallItem` class.

# ToolCallItem Documentation

## Overview

The `ToolCallItem` class is a key component in the OpenAI Agents Python SDK that represents a tool call made by an agent during its execution. It encapsulates different types of tool calls (function calls, computer actions, file search, and web search) in a unified interface, allowing for consistent handling of various tool interactions.

## Class Definition

```python
@dataclass
class ToolCallItem(RunItemBase[ToolCallItemTypes]):
    """Represents a tool call e.g. a function call or computer action call."""

    raw_item: ToolCallItemTypes
    """The raw tool call item."""

    type: Literal["tool_call_item"] = "tool_call_item"
```

## Inheritance Hierarchy

- Inherits from `RunItemBase[ToolCallItemTypes]`, which provides the base functionality for all run items
- Used alongside other run items like `MessageOutputItem`, `HandoffCallItem`, and `ToolCallOutputItem`

## Associated Types

```python
ToolCallItemTypes: TypeAlias = Union[
    ResponseFunctionToolCall,
    ResponseComputerToolCall,
    ResponseFileSearchToolCall,
    ResponseFunctionWebSearch,
]
```

These types come from the OpenAI SDK and represent different kinds of tool calls:

1. **ResponseFunctionToolCall**: Standard function calls to Python functions
2. **ResponseComputerToolCall**: Calls to computer-control actions (clicking, typing, etc.)
3. **ResponseFileSearchToolCall**: File search operations
4. **ResponseFunctionWebSearch**: Web search operations

## Fields

- **agent**: The agent that made this tool call (inherited from `RunItemBase`)
- **raw_item**: The raw tool call item from the OpenAI API
- **type**: A literal string identifier set to `"tool_call_item"`

## Lifecycle

1. **Creation**: A `ToolCallItem` is created when the agent invokes a tool during its execution. It's generated in response to the agent deciding to call a tool based on user input or context.

2. **Processing**: After creation, the Runner processes the tool call by:
   - Locating the appropriate tool implementation
   - Executing the tool with provided arguments
   - Creating a corresponding `ToolCallOutputItem` with the results

3. **Usage in Results**: Both the `ToolCallItem` and its corresponding `ToolCallOutputItem` are stored in the `new_items` list of the `RunResult` object returned by `Runner.run()`.

## Tool Call Types

### Function Tool Calls

Most common type, representing invocations of Python functions decorated with `@function_tool`:

```python
@function_tool
def get_weather(city: str) -> str:
    # Implementation
    return f"The weather in {city} is sunny."
```

When the agent calls this tool, a `ToolCallItem` with a `ResponseFunctionToolCall` is created.

### Computer Tool Calls

Represents actions on a computer, like mouse clicks, typing, or drag operations:

```python
# Computer tool calls include operations like:
# - ActionClick
# - ActionDoubleClick
# - ActionDrag
# - ActionType
# - ActionHover
```

### File Search Tool Calls

Used when the agent performs file search operations using the `FileSearchTool`:

```python
from agents import Agent, FileSearchTool

agent = Agent(
    tools=[
        FileSearchTool(
            max_num_results=3,
            vector_store_ids=["my_vector_store_id"],
        )
    ]
)
```

### Web Search Tool Calls

Used when the agent performs web search operations using the `WebSearchTool`:

```python
from agents import Agent, WebSearchTool

agent = Agent(
    tools=[WebSearchTool()]
)
```

## Usage in Code

### Accessing Tool Call Information

```python
result = await Runner.run(agent, "What's the weather in Tokyo?")

# Iterate through new items looking for tool calls
for item in result.new_items:
    if isinstance(item, ToolCallItem):
        if isinstance(item.raw_item, ResponseFunctionToolCall):
            print(f"Function called: {item.raw_item.name}")
            print(f"Arguments: {item.raw_item.arguments}")
        elif isinstance(item.raw_item, ResponseComputerToolCall):
            print(f"Computer action: {item.raw_item.type}")
            # Handle specific action types
            if isinstance(item.raw_item.action, ActionClick):
                print(f"Clicked at: {item.raw_item.action.position}")
```

### Pairing Tool Calls with Outputs

Tools calls are typically followed by their outputs in the `new_items` list. You can pair them as follows:

```python
tool_calls_with_outputs = []
current_tool_call = None

for item in result.new_items:
    if isinstance(item, ToolCallItem):
        current_tool_call = item
    elif isinstance(item, ToolCallOutputItem) and current_tool_call:
        tool_calls_with_outputs.append((current_tool_call, item))
        current_tool_call = None

# Process paired tool calls and outputs
for tool_call, output in tool_calls_with_outputs:
    print(f"Tool {tool_call.raw_item.name} returned: {output.output}")
```

## Internal Implementation Details

In the internal implementation of the SDK:

1. `ToolCallItem` instances are created in the agent loop when processing model responses.

2. The `Agent` or `Runner` classes identify tool calls in the model's output and create the corresponding `ToolCallItem`.

3. The internal `_run_impl.py` file handles the execution of tools and creates the corresponding `ToolCallOutputItem` with the result.

4. Both items are added to the list of generated items and included in the final `RunResult`.

## Best Practices

1. **Type Checking**: Always check the type of `raw_item` before accessing specific properties as each tool call type has different properties.

2. **Error Handling**: Handle potential errors in tool calls, as tools might fail to execute properly.

3. **Contextual Processing**: Consider the context in which the tool call was made to better understand and process the results.

4. **Function Tool Design**: When designing function tools, consider the information the agent needs and provide meaningful error handling and return values.

5. **Output Parsing**: Parse tool outputs carefully, especially when using complex tools that might return structured data.

## Related Classes

- **ToolCallOutputItem**: Represents the output of a tool call
- **FunctionTool**: Class for function-based tools
- **ComputerTool**: Class for computer action tools
- **WebSearchTool**: Class for web search operations
- **FileSearchTool**: Class for file search operations
- **RunResult**: Contains all items (including tool calls) generated during a run

## Examples

### Using Function Tools

```python
from agents import Agent, Runner, function_tool

@function_tool
def get_weather(city: str) -> str:
    # In a real application, you would call a weather API
    return f"The weather in {city} is sunny."

agent = Agent(
    name="Weather Assistant",
    instructions="You help users with weather information.",
    tools=[get_weather],
)

result = await Runner.run(agent, "What's the weather in Tokyo?")
print(result.final_output)  # Will likely contain the weather in Tokyo
```

### Analyzing Tool Usage

```python
from agents import ToolCallItem, ToolCallOutputItem

result = await Runner.run(agent, "What's the weather in London and Tokyo?")

# Count tool calls by type
function_calls = 0
computer_actions = 0
file_searches = 0
web_searches = 0

for item in result.new_items:
    if isinstance(item, ToolCallItem):
        if isinstance(item.raw_item, ResponseFunctionToolCall):
            function_calls += 1
        elif isinstance(item.raw_item, ResponseComputerToolCall):
            computer_actions += 1
        elif isinstance(item.raw_item, ResponseFileSearchToolCall):
            file_searches += 1 
        elif isinstance(item.raw_item, ResponseFunctionWebSearch):
            web_searches += 1

print(f"Function calls: {function_calls}")
print(f"Computer actions: {computer_actions}")
print(f"File searches: {file_searches}")
print(f"Web searches: {web_searches}")
```

By understanding and properly using the `ToolCallItem` class, developers can build more sophisticated agent applications that effectively leverage various types of tools and process their results appropriately.