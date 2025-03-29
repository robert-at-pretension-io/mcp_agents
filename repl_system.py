import asyncio
import os
import sys
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from agents import (
    Agent,
    FunctionTool,
    
    ModelSettings,
    Runner,
    RunConfig,
    trace
)

from agents.mcp import MCPServerStdio

# --- Configuration ---
# Adjust paths if your server scripts are elsewhere
SERVER_SCRIPTS = {
    "aider": "./aider_server.py",
    "brave": "./brave_search_server.py",
    "long_runner": "./long_running_server.py",
    "quick_bash": "./quick_bash_server.py",
    "regex_replace": "./regex_replace_server.py",
    "scraper": "./scrape_url_server.py",
}

# Choose primary model for agents (can be overridden per agent)
DEFAULT_MODEL = "gpt-4o" # Or "o3-turbo", "o3-mini", etc.
ORCHESTRATOR_MODEL = "gpt-4o" # Often benefits from a more powerful model

# --- MCP Server Management ---

def create_mcp_server_instances(script_paths: Dict[str, str]) -> Dict[str, MCPServerStdio]:
    """Creates MCPServerStdio instances for each script path."""
    servers = {}
    print("Creating MCP Server instances...")
    for name, script_path in script_paths.items():
        if not os.path.exists(script_path):
            print(f"Warning: Script not found for server '{name}': {script_path}. Skipping.")
            continue

        print(f"  Creating instance for {name} server ({script_path})...")
        server = MCPServerStdio(
            params={
                "command": sys.executable, # Use the current Python interpreter
                "args": [script_path],
            },
            cache_tools_list=True # Cache tools for performance
        )
        servers[name] = server
    print("MCP Server instances created.")
    return servers

# Note: Stopping servers is handled by AsyncExitStack in main()

async def stop_mcp_servers(servers: Dict[str, MCPServerStdio]):
    """Stops all MCP servers (Deprecated - Use AsyncExitStack)."""
    print("Stopping MCP Servers (using explicit stop - prefer AsyncExitStack)...")
    tasks = [asyncio.create_task(server.close(), name=f"close_{name}") for name, server in servers.items()]
    if tasks:
        await asyncio.wait(tasks)
    print("MCP Servers stopped.")

# --- Specialist Agents Definition ---

def create_specialist_agents(mcp_servers: Dict[str, MCPServerStdio]) -> Dict[str, Agent]:
    """Creates specialist agents, assigning MCP servers."""
    agents = {}

    # --- Coding Agent ---
    coding_servers = []
    if "aider" in mcp_servers: coding_servers.append(mcp_servers["aider"])
    if "regex_replace" in mcp_servers: coding_servers.append(mcp_servers["regex_replace"])

    if coding_servers:
        agents["coding_agent"] = Agent(
            name="CodingAgent",
            instructions="""
            You are an expert AI programming assistant specializing in code modification and generation.
            Your goal is to fulfill user requests related to writing, debugging, refactoring, and understanding code.
            You MUST use the available tools to interact with the codebase:
            - Use the 'aider' tool for complex changes, implementing features, fixing bugs across files, or adding tests. Provide detailed context and requirements in the message.
            - Use the 'regex_replace' tool for precise, targeted multi-line text replacements within specific files using regular expressions.
            Explain your plan before using a tool. Analyze the tool's output and report the results clearly to the user.
            """,
            model=DEFAULT_MODEL,
            mcp_servers=coding_servers,
        )

    # --- Search Agent ---
    search_servers = []
    if "brave" in mcp_servers: search_servers.append(mcp_servers["brave"])
    if "scraper" in mcp_servers: search_servers.append(mcp_servers["scraper"])

    if search_servers:
        agents["search_agent"] = Agent(
            name="SearchAgent",
            instructions="""
            You are an AI research assistant specializing in finding and extracting information from the web.
            Your goal is to answer user questions using up-to-date information.
            You MUST use the available tools:
            - Use the 'brave_search' tool to perform web searches for information, current events, or specific resources. Formulate concise search queries.
            - Use the 'scrape_url' tool to extract the main textual content from a specific webpage URL. Use this when a search result URL looks promising or when the user provides a direct URL.
            Synthesize information from the tools and present a comprehensive answer to the user. Cite your sources (URLs).
            """,
            model=DEFAULT_MODEL,
            mcp_servers=search_servers,
        )

    # --- System Agent ---
    system_servers = []
    if "quick_bash" in mcp_servers: system_servers.append(mcp_servers["quick_bash"])
    if "long_runner" in mcp_servers: system_servers.append(mcp_servers["long_runner"])
    # Decided to put regex_replace here too for general file ops, but also kept in coding
    if "regex_replace" in mcp_servers and mcp_servers["regex_replace"] not in system_servers:
         system_servers.append(mcp_servers["regex_replace"])

    if system_servers:
        agents["system_agent"] = Agent(
            name="SystemAgent",
            instructions="""
            You are an AI assistant specializing in interacting with the operating system and managing tasks.
            Your goal is to help users execute commands, manage files, and run background processes.
            You MUST use the available tools:
            - Use the 'quick_bash' tool for executing simple, short-running shell commands (like ls, cat, grep, mkdir, rm). Check output and status carefully.
            - Use the 'long_running_tool' to start ('start_task'), monitor ('get_status'), or list ('list_tasks') background shell commands that might take a long time (e.g., builds, data processing). Provide a clear reason when starting tasks.
            - Use the 'regex_replace' tool for targeted multi-line text replacements in files using regular expressions (useful for config files, scripts etc.). Specify patterns and replacement clearly.
            Confirm actions, especially destructive ones (like deleting files), before proceeding if appropriate. Report outcomes clearly.
            """,
            model=DEFAULT_MODEL,
            mcp_servers=system_servers,
        )

    # --- File Edit Agent (Example of more granular agent) ---
    # You could also merge regex_replace into coding/system agents if preferred
    # file_edit_servers = []
    # if "regex_replace" in mcp_servers: file_edit_servers.append(mcp_servers["regex_replace"])
    # if file_edit_servers:
    #     agents["file_edit_agent"] = Agent(
    #         name="FileEditAgent",
    #         instructions="You are a specialist in performing precise multi-line text replacements in files using regular expressions. Use the 'regex_replace' tool.",
    #         model=DEFAULT_MODEL,
    #         mcp_servers=file_edit_servers,
    #     )

    if not agents:
        raise ValueError("No specialist agents could be created. Check MCP server availability.")

    return agents

# --- Orchestrator Agent Definition ---

def create_orchestrator_agent(specialist_agents: Dict[str, Agent]) -> Agent:
    """Creates the orchestrator agent using specialists as tools."""

    orchestrator_tools: List[FunctionTool] = []
    for agent_name, agent in specialist_agents.items():
        description = ""
        if agent_name == "coding_agent":
            description = "Delegates tasks related to writing, debugging, refactoring, or modifying code to the Coding Assistant. Use for requests like 'implement feature X', 'fix bug Y', 'add tests for Z', 'replace code block'."
        elif agent_name == "search_agent":
            description = "Delegates tasks related to searching the web or scraping specific URLs to the Search Assistant. Use for requests like 'what is X?', 'find recent news on Y', 'summarize website Z'."
        elif agent_name == "system_agent":
            description = "Delegates tasks related to running shell commands, managing background processes, or performing file operations (including regex replace) to the System Assistant. Use for requests like 'list files in /tmp', 'start a long build process', 'check status of task 123', 'replace text in config.txt'."
        # elif agent_name == "file_edit_agent":
        #     description = "Delegates precise multi-line text replacement tasks using regex to the File Edit Assistant."
        else:
            description = f"Delegates tasks to the {agent.name}."

        orchestrator_tools.append(
            agent.as_tool(
                tool_name=agent_name, # Use agent name as tool name
                tool_description=description,
            )
        )

    orchestrator = Agent(
        name="OrchestratorAgent",
        instructions=f"""
        You are the central orchestrator for a team of specialized AI assistants.
        Your primary role is to analyze the user's request and delegate it to the MOST appropriate specialist assistant by using the corresponding tool.
        Available specialists (represented as tools):
        {[f'- {tool.name}: {tool.description}' for tool in orchestrator_tools]}

        Follow these steps:
        1. Understand the user's intent and the core task they want to perform.
        2. Determine which specialist assistant (Coding, Search, System) is best equipped to handle the task based on its description.
        3. Call the corresponding tool ONCE to delegate the entire user request to that specialist. Do NOT try to break down the task or call multiple tools sequentially unless the user asks for a multi-step process explicitly involving different domains (which is rare).
        4. Do NOT attempt to answer the user directly or perform the task yourself. Your ONLY job is delegation via the available tools.
        5. If the user asks a clarifying question about a previous turn, route it to the specialist who handled the original task if possible, otherwise handle it briefly yourself if it's about the orchestration process.
        6. If the request is ambiguous or doesn't fit any specialist, ask the user for clarification.
        """,
        model=ORCHESTRATOR_MODEL,
        model_settings=ModelSettings(temperature=0.2), # Lower temp for more predictable delegation
        tools=orchestrator_tools,
    )
    return orchestrator

# --- REPL Function ---

async def run_repl(orchestrator: Agent, run_config: RunConfig):
    """Runs the Read-Eval-Print Loop."""
    print("\n--- Agent REPL System ---")
    print("Type 'quit' or 'exit' to end.")
    print("Enter your requests for the agent system.")

    conversation_history: List[Dict[str, Any]] = []

    while True:
        try:
            user_input = await asyncio.to_thread(input, "\nYou> ")
        except EOFError: # Handle Ctrl+D
             print("\nExiting...")
             break

        if user_input.lower() in ["quit", "exit"]:
            print("Exiting...")
            break

        if not user_input.strip():
            continue

        # Append user message to history
        conversation_history.append({"role": "user", "content": user_input})

        print("Orchestrator thinking...")
        try:
            # Use trace context for better observability (optional but recommended)
            with trace(
                workflow_name="AgentREPL",
                group_id=run_config.trace_metadata.get("session_id", "default_session") # Example group_id
                # trace_metadata is handled via RunConfig passed to Runner.run
             ):
                result = await Runner.run(
                    orchestrator,
                    conversation_history, # Pass the whole history
                    run_config=run_config,
                    max_turns=15, # Prevent runaway loops within a single turn
                    # context=None # No global context needed for this example
                )

            # --- Print Tool Interactions ---
            print("\n--- Agent Turn Details ---")
            # Use result.new_items to access the generated items during the run
            for i, item in enumerate(result.new_items):
                # Check the type of the RunItem
                if item.type == "tool_call_item":
                    # Access attributes specific to ToolCallItem
                    # Note: raw_item holds the actual tool call details (like ResponseFunctionToolCall)
                    tool_name = getattr(item.raw_item, 'name', 'unknown_tool')
                    tool_args = getattr(item.raw_item, 'arguments', {})
                    print(f"  Tool Call: {tool_name}({tool_args})")
                elif item.type == "tool_call_output_item":
                    # Access attributes specific to ToolCallOutputItem
                    tool_name = getattr(item, 'tool_name', 'unknown_tool')
                    output = getattr(item, 'output', '')
                    # Truncate long outputs for readability
                    output_str = str(output)
                    max_len = 200
                    if len(output_str) > max_len:
                        output_str = output_str[:max_len] + "..."
                    print(f"  Tool Output ({tool_name}): {output_str}")
                elif item.type == "message_output_item":
                    # Access attributes specific to MessageOutputItem
                    role = getattr(item.message, 'role', 'unknown')
                    content = getattr(item.message, 'content', '')
                    if role == "assistant":
                         # You might want to print assistant messages during the turn too,
                         # or just use the final_output at the end.
                         # print(f"  Assistant Message: {content}")
                         pass # We print the final consolidated output later
                    elif role == "user":
                         # This shouldn't typically happen in new_items unless it's complex
                         # print(f"  User Message (in turn): {content}")
                         pass
                # Add more checks here if other RunItem types are expected (e.g., HandoffCallItem)

            print("--- End Agent Turn Details ---\n")

            # --- Extract Final Assistant Response ---
            # The final output might be slightly different from the last assistant message
            # if post-processing occurs, but usually it's the content of the last one.
            assistant_response = result.final_output
            if isinstance(assistant_response, dict) and 'content' in assistant_response:
                 assistant_response_content = assistant_response['content']
            elif isinstance(assistant_response, str):
                assistant_response_content = assistant_response
            else:
                 assistant_response_content = str(assistant_response) # Fallback

            print(f"\nAssistant> {assistant_response_content}")

            # Append assistant response to history
            # Use result.to_input_list() to get the full turn history including tool calls
            # for the next iteration, but only display final_output to user.
            # We reconstruct the history manually here for clarity, but to_input_list is robust.
            conversation_history.append({"role": "assistant", "content": assistant_response_content})

            # Optional: Limit history length
            # MAX_HISTORY = 10 # Keep last 5 user/assistant pairs
            # if len(conversation_history) > MAX_HISTORY * 2:
            #     conversation_history = conversation_history[-(MAX_HISTORY * 2):]

        except Exception as e:
            print(f"\nError during agent execution: {e}")
            # Optionally remove the last user message if the turn failed badly
            # if conversation_history and conversation_history[-1]["role"] == "user":
            #     conversation_history.pop()


# --- Main Execution ---

async def main():
    # Use AsyncExitStack to ensure servers are cleaned up
    async with AsyncExitStack() as stack:
        started_mcp_servers: Dict[str, MCPServerStdio] = {}
        try:
            # 1. Create MCP Server Instances
            potential_mcp_servers = create_mcp_server_instances(SERVER_SCRIPTS)

            # 2. Start MCP Servers using AsyncExitStack
            print("Attempting to start MCP Servers...")
            for name, server in potential_mcp_servers.items():
                try:
                    print(f"  Starting {name} server...")
                    # Enter the server's async context (this starts it)
                    # The stack will call server.__aexit__ automatically on exit
                    await stack.enter_async_context(server)
                    started_mcp_servers[name] = server
                    print(f"  Successfully started {name} server.")
                    # Optional: List tools after start
                    # try:
                    #     tools = await server.list_tools()
                    #     print(f"    Tools for {name}: {[t.name for t in tools]}")
                    # except Exception as list_e:
                    #     print(f"    Warning: Could not list tools for {name}: {list_e}")
                except Exception as e:
                    print(f"Error starting server '{name}': {e}")
                    # Server failed to start, AsyncExitStack won't manage it,
                    # and it shouldn't be passed to agents.

            if not started_mcp_servers:
                 print("Error: No MCP servers could be started successfully. Exiting.")
                 return # Or raise an exception

            print(f"Successfully started {len(started_mcp_servers)} MCP server(s).")

            # 3. Create Specialist Agents with successfully started servers
            specialist_agents = create_specialist_agents(started_mcp_servers)
            print(f"Created Specialist Agents: {list(specialist_agents.keys())}")

            # 4. Create Orchestrator Agent
            orchestrator_agent = create_orchestrator_agent(specialist_agents)
            print("Created Orchestrator Agent.")

            # Configure Run
            run_config = RunConfig(
                # model="gpt-4-turbo", # Can override agent models globally here
                # model_settings=ModelSettings(temperature=0.5), # Global settings override
                tracing_disabled=False, # Enable tracing (requires LangSmith/Helicone setup)
                workflow_name="AgentREPLSystem",
                trace_metadata={"session_id": f"repl_{os.getpid()}"} # Example metadata
                # max_turns is passed directly to Runner.run
            )

            # Run the REPL
            await run_repl(orchestrator_agent, run_config)

        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
        finally:
             # AsyncExitStack handles cleanup registered with stack.enter_async_context
             print("\nMain execution finished. Resources should be cleaned up.")
             # Explicit stop call (redundant with AsyncExitStack but safe)
             # await stop_mcp_servers(mcp_servers)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
