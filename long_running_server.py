import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from mcp.server.fastmcp import Context, FastMCP

# Import the manager class
from long_running_task_manager import LongRunningTaskManager, TaskState, TaskStatus

# --- Server State and Lifespan ---
@dataclass
class AppContext:
    """Context holding the task manager."""
    long_running_manager: LongRunningTaskManager

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Initialize and cleanup the LongRunningTaskManager."""
    print("Long Running Task Server Lifespan: Startup")
    manager = LongRunningTaskManager() # Uses default filename 'mcp_tasks.json'
    await manager.load_persistent_tasks()
    try:
        yield AppContext(long_running_manager=manager)
    finally:
        print("Long Running Task Server Lifespan: Shutdown")
        await manager.save() # Ensure final state is saved
        print("Long Running Task Server Shutdown Complete.")

# MCP Server Initialization with Lifespan
mcp = FastMCP(
    "Long Running Task Tool",
    lifespan=app_lifespan,
    description="Manages long-running background shell commands.",
)

@mcp.tool()
async def long_running_tool(
    ctx: Context,
    command: str, # Literal["start_task", "get_status", "list_tasks"]
    commandString: Optional[str] = None,
    taskId: Optional[str] = None,
    reason: Optional[str] = "No reason given",
    status: Optional[str] = None, # Optional filter for list_tasks
    lines: Optional[int] = 100, # Trailing lines for get_status
) -> str:
    """
    Task management system for running commands that may take minutes or hours to complete. This tool handles:

    1. **Task creation** with `start_task`: Launch shell commands that will continue running after your conversation ends
    2. **Status monitoring** with `get_status`: Check if tasks are still running and view their real-time output
    3. **Output inspection** with `get_status`: Review both standard output and error streams from running or completed tasks
    4. **Task organization** with `list_tasks`: View all active and completed tasks with filtering options

    Key benefits:
    - Runs asynchronously in the background, independent of API timeouts
    - Persists between sessions (tasks continue even if you close this conversation)
    - Captures all stdout and stderr output for inspection
    - Provides detailed task status and progress information
    - Maintains a record of completed tasks with their full output

    Common use cases:
    - Long-running data processing jobs
    - Compilation of large codebases
    - Extended test suites that take significant time
    - Scheduled maintenance tasks
    - Monitoring system operations
    """
    app_context: AppContext = ctx.lifespan
    manager = app_context.long_running_manager
    lines_to_return = max(1, lines or 100) # Ensure at least 1 line

    if command == "start_task":
        if not commandString:
            return "Error: 'commandString' is required for 'start_task'."
        effective_reason = reason or "No reason given"
        try:
            task_id = await manager.spawn_task(commandString, effective_reason)
            return f"Task started with id: {task_id}\nReason: {effective_reason}"
        except Exception as e:
            return f"Error starting task: {e}"

    elif command == "get_status":
        if not taskId:
            return "Error: 'taskId' is required for 'get_status'."
        try:
            state = await manager.get_task_status(taskId)
            if not state:
                return f"Error: Task not found: {taskId}"

            stdout_short = manager.last_n_lines(state.stdout, lines_to_return)
            stderr_short = manager.last_n_lines(state.stderr, lines_to_return)

            return (
                f"Task ID: {state.task_id}\n"
                f"Status: {state.status.value}\n"
                f"Reason: {state.reason}\n"
                f"Command: {state.command}\n\n"
                f"(Showing last {lines_to_return} lines) STDOUT:\n{stdout_short}\n\n"
                f"(Showing last {lines_to_return} lines) STDERR:\n{stderr_short}"
            )
        except Exception as e:
            return f"Error getting task status for {taskId}: {e}"

    elif command == "list_tasks":
        try:
            filter_status: Optional[TaskStatus] = None
            if status:
                try:
                    filter_status = TaskStatus(status.lower())
                except ValueError:
                    return f"Error: Invalid status filter '{status}'. Valid values are: created, running, ended, error."

            tasks = await manager.list_tasks(filter_status)
            if not tasks:
                 return f"No tasks found{f' with status {status}' if status else ''}."

            # Prepare a summary list for the text response
            summary_list = []
            for task in tasks:
                summary_list.append({
                    "taskId": task.task_id,
                    "status": task.status.value,
                    "reason": task.reason,
                    "command": task.command[:80] + ('...' if len(task.command) > 80 else ''), # Truncate long commands
                    "outputLen": len(task.stdout) + len(task.stderr)
                })

            # Return as pretty-printed JSON string
            return json.dumps(summary_list, indent=2)

        except Exception as e:
            return f"Error listing tasks: {e}"

    else:
        return f"Error: Invalid command '{command}'. Use 'start_task', 'get_status', or 'list_tasks'."


if __name__ == "__main__":
    print("Starting Long Running Task MCP Server...")
    mcp.run()