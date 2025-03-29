I'll look up and document the `RunResult` class from the OpenAI Agents Python SDK.

Now let's examine the `RunResult` class definition:



Let's also see how `RunResult` is used in the runner:





Let's also look at some examples of how `RunResult` is used in practice:













Let's look at one more example that might show more usage of the `RunResult` properties:















Let's try to find a more complex example that uses more features of the `RunResult` class:













Based on my examination of the OpenAI Agents Python SDK codebase, here is comprehensive documentation on the `RunResult` class:

# RunResult Documentation

## Overview

The `RunResult` class is a central component of the OpenAI Agents Python SDK that encapsulates the results of an agent run. It contains all inputs, outputs, and intermediate items generated during execution, providing a complete record of the agent's operations and final results.

## Class Hierarchy

The class hierarchy consists of:

- `RunResultBase` (Abstract Base Class): Defines the core properties and methods that all run result types share
  - `RunResult`: A concrete implementation for standard (non-streaming) agent runs
  - `RunResultStreaming`: A specialized implementation with additional properties and methods for streaming agent runs

## Key Properties

### Common Properties (from RunResultBase)

- **`input`**: The original input to the agent run, may be either a string or a list of input items. This can be mutated by handoff input filters.

- **`new_items`**: A list of `RunItem` objects generated during the run. These include messages, tool calls, tool outputs, and other items created by the agent.

- **`raw_responses`**: A list of `ModelResponse` objects containing the raw responses from the LLM during each turn.

- **`final_output`**: The output from the agent, which could be a string or a structured output depending on the agent's configuration.

- **`input_guardrail_results`**: Results from guardrails applied to the input.

- **`output_guardrail_results`**: Results from guardrails applied to the output.

- **`last_agent`**: The last agent that was run during the execution. If there were handoffs, this will be the final agent in the chain.

### Specific to RunResultStreaming

- **`current_agent`**: The agent currently running (updated during streaming).

- **`current_turn`**: The current turn number in the agent loop.

- **`max_turns`**: Maximum number of turns the agent can execute.

- **`is_complete`**: Boolean indicator of whether the agent has finished running.

## Methods

### Core Methods (RunResultBase)

- **`final_output_as(cls, raise_if_incorrect_type=False)`**: Casts the final output to a specific type, optionally raising an exception if the type doesn't match.

- **`to_input_list()`**: Creates a new input list by merging the original input with all new items generated, useful for creating continuous conversations.

### Streaming-specific Methods

- **`stream_events()`**: An async iterator that yields `StreamEvent` objects as they're generated, allowing for real-time processing of agent outputs.

## Usage Patterns

### Basic Usage

```python
from agents import Agent, Runner

agent = Agent(
    name="Assistant",
    instructions="You help users with weather information."
)

# Run the agent
result = await Runner.run(agent, "What's the weather in Tokyo?")

# Access the final output
print(result.final_output)  # The agent's final response
```

### Accessing Generated Items

```python
# Get all items created during the run
for item in result.new_items:
    if item.type == "tool_call":
        print(f"Tool called: {item.name}")
    elif item.type == "text":
        print(f"Text generated: {item.text}")
```

### Continuing Conversations

```python
# First turn
result1 = await Runner.run(agent, "What's the weather in Tokyo?")

# Second turn, using history
next_input = result1.to_input_list() + [{"role": "user", "content": "How about New York?"}]
result2 = await Runner.run(agent, next_input)
```

### Working with Structured Outputs

```python
from pydantic import BaseModel

class WeatherData(BaseModel):
    temperature: float
    condition: str
    city: str

# Agent with structured output
agent = Agent(
    name="Weather",
    instructions="Return structured weather data",
    output_type=WeatherData
)

result = await Runner.run(agent, "What's the weather in Tokyo?")
weather_data = result.final_output_as(WeatherData, raise_if_incorrect_type=True)
print(f"It's {weather_data.condition} and {weather_data.temperature}°C in {weather_data.city}")
```

### Streaming Results

```python
result = await Runner.run_streamed(agent, "Generate a long response")
async for event in result.stream_events():
    if event.type == "content_block_delta":
        print(event.delta, end="", flush=True)
```

### Agent Chains with Evaluation

```python
generator_agent = Agent(name="Generator", instructions="Generate content")
evaluator_agent = Agent(name="Evaluator", instructions="Evaluate content", output_type=Evaluation)

input_items = [{"role": "user", "content": "Write a story about robots"}]

while True:
    # Generator produces content
    gen_result = await Runner.run(generator_agent, input_items)
    
    # Update input items with generated content
    input_items = gen_result.to_input_list()
    
    # Evaluator judges the content
    eval_result = await Runner.run(evaluator_agent, input_items)
    evaluation = eval_result.final_output
    
    if evaluation.score == "pass":
        break
        
    # Add feedback for next iteration
    input_items.append({"role": "user", "content": f"Feedback: {evaluation.feedback}"})
```

## Implementation Details

- `RunResult` is created at the end of the agent's run and returned by the `Runner.run()` method.

- The runner gathers all inputs, outputs, and intermediate items throughout the agent execution cycle.

- For streaming, `RunResultStreaming` maintains internal queues and tasks to handle asynchronous event generation.

- The base class provides common functionality while specialized implementations handle different run modes (standard vs. streaming).

## Best Practices

- Use `to_input_list()` to create proper input for follow-up questions in conversations.

- For structured outputs, use `final_output_as()` with appropriate type hints.

- Check guardrail results to handle potential policy violations.

- For streamed results, always await the complete stream to ensure proper cleanup of resources.

- Use the `last_agent` property to get the final agent if handoffs occurred during execution.