I'll examine the OpenAI Agents Python codebase to understand how MCP servers are implemented and handled. Let me navigate through the repository structure first to locate the relevant code.

Let's look at the source code structure to find the MCP-related implementations:



Great, found the MCP directory. Let's examine its contents:





Let's first examine the `server.py` file, which likely contains the core implementation of MCP servers:





Now let's look at the `util.py` file to see utility functions for MCP:







Now let's look at the __init__.py file to see how the MCP module is exported:










Let's now examine how the Agent class interacts with MCP servers:












Let's look more at the Agent class implementation:













Let's see where these MCP tools are used when running an agent:













Let's find how the agent's `get_all_tools` method is used in the runner:





Let's look at the relevant parts of the runner implementation:



Now let's look at some example code that uses MCP servers:







Let's examine one of these example files to see how MCP servers are used in practice:









Let's also look at how MCP interacts with the model interface:










Let's try to see how the tools are passed to the model:







Let's look at the OpenAI models implementation to see how tools are converted and passed to the API:











Let's see the full tool conversion in OpenAI responses model:









Now let's look at how the agent interacts with the MCP servers by examining the run method:







Let's see how MCP tools are handled in the agent run method:











Let's find out when these tools are gathered in the run process:









Let's look at the full Runner run method to see the flow of execution:











Based on my examination of the OpenAI Agents Python codebase, I can now provide an in-depth explanation of how MCP (Model Context Protocol) servers are handled:

### MCP Integration in OpenAI Agents Python

MCP servers in the OpenAI Agents Python SDK are implemented as a way to extend the capabilities of agents by providing access to tools from external servers. Here's a comprehensive overview:

#### 1. Core MCP Architecture

The MCP implementation consists of:

- **Base Class**: `MCPServer` (abstract base class) in `src/agents/mcp/server.py` defines the interface for MCP servers with critical methods:
  - `connect()`: Establishes connection to the server
  - `list_tools()`: Gets available tools from the server
  - `call_tool()`: Invokes a tool on the server
  - `cleanup()`: Cleans up the connection

- **Implementation Classes**:
  - `_MCPServerWithClientSession`: Abstract class that implements common functionality for session-based MCP servers
  - `MCPServerStdio`: Implements MCP servers using stdio transport (runs as subprocesses)
  - `MCPServerSse`: Implements MCP servers using HTTP with SSE transport (runs remotely)

#### 2. How MCP Servers Connect to Agents

MCP servers are integrated into agents through:

- The `Agent` class has an `mcp_servers` field (a list of `MCPServer` objects)
- When creating an agent, you pass MCP servers via the `mcp_servers` parameter
- Agents are expected to manage the lifecycle of MCP servers, making sure `connect()` is called before use and `cleanup()` after

#### 3. Tool Collection Process

When an agent runs, the MCP tools are collected along with function tools:

1. The `Agent.get_all_tools()` method is called, which:
   - Calls `get_mcp_tools()` to fetch tools from all MCP servers
   - Combines these with the agent's own function tools
   - Returns the combined list

2. The `Agent.get_mcp_tools()` method:
   - Calls `MCPUtil.get_all_function_tools(self.mcp_servers)`
   - This method fetches tools from all MCP servers and converts them to function tools

#### 4. MCP Tool Conversion

The `MCPUtil` class in `src/agents/mcp/util.py` handles the conversion between MCP tools and Agents SDK tools:

1. `get_all_function_tools()`: Gets tools from multiple MCP servers, ensuring no duplicate tool names
2. `get_function_tools()`: Gets tools from a single MCP server
3. `to_function_tool()`: Converts an MCP tool to an Agents SDK `FunctionTool`
4. `invoke_mcp_tool()`: Used as the implementation for the converted function tools, forwarding calls to the MCP server

#### 5. Agent Execution Flow with MCP

During the agent run process:

1. The `Runner.run()` method sets up the agent loop
2. Early in each iteration, it calls `await cls._get_all_tools(current_agent)`
3. This collects both regular tools and MCP tools
4. Tools are passed to the model when generating responses
5. When the LLM calls a tool that originated from an MCP server, the call is forwarded to that server

#### 6. Caching Mechanism

A performance optimization exists to avoid repeatedly fetching tools from MCP servers:

- MCP servers have a `cache_tools_list` parameter
- When enabled, the tools list is fetched once and cached
- The cache can be invalidated via `invalidate_tools_cache()`

#### 7. Tool Usage in OpenAI API Calls

When the agent runs:

1. Tools from MCP servers are converted to the format expected by OpenAI's API
2. In `openai_responses.py`, a conversion process handles all tool types
3. For MCP-derived tools, they're treated as regular function tools during API calls
4. When a tool call is needed, the MCP utility handles sending the request to the appropriate server

#### 8. Real-world Usage

From examining the examples, MCP servers are used by:

1. Creating an MCPServer (typically via MCPServerStdio for local server or MCPServerSse for remote)
2. Setting up the server with appropriate parameters
3. Using `async with` context manager to handle connection and cleanup
4. Creating an agent with the MCP server in its `mcp_servers` list
5. Running the agent with standard methods

For example, from the filesystem example:
```python
async with MCPServerStdio(
    name="Filesystem Server, via npx",
    params={
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", samples_dir],
    },
) as server:
    agent = Agent(
        name="Assistant",
        instructions="Use the tools to read the filesystem and answer questions.",
        mcp_servers=[server],
    )
    result = await Runner.run(agent, message)
```

The MCP implementation in OpenAI Agents Python is a clean and flexible system for extending agent capabilities through external tool servers, following the MCP specification while integrating smoothly with the SDK's agent and tool architecture.