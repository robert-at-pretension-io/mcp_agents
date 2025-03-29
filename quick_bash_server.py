import asyncio
import os
import shlex
from typing import List, Optional, Tuple

from mcp.server.fastmcp import Context, FastMCP

# Helper function to run subprocess
async def run_subprocess(command_args: List[str], cwd: Optional[str] = None) -> Tuple[int, str, str]:
    """Runs a subprocess asynchronously and returns status, stdout, stderr."""
    try:
        process = await asyncio.create_subprocess_exec(
            *command_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout_bytes, stderr_bytes = await process.communicate()
        return_code = process.returncode if process.returncode is not None else -1
        stdout = stdout_bytes.decode('utf-8', errors='replace')
        stderr = stderr_bytes.decode('utf-8', errors='replace')
        return return_code, stdout, stderr
    except FileNotFoundError:
         raise FileNotFoundError(f"Command not found: {command_args[0]}. Ensure it's installed and in PATH.")
    except Exception as e:
        raise RuntimeError(f"Subprocess execution failed for {command_args}: {e}")

# MCP Server Initialization
mcp = FastMCP("Quick Bash Tool", description="Runs simple one-line shell commands.")

@mcp.tool()
async def quick_bash(ctx: Context, cmd: str) -> str:
    """
    Fast shell command tool for simple one-liners. Use this to:

    1. Run quick system checks (ls, ps, grep, find, etc.)
    2. View file contents (cat, head, tail, less)
    3. Create, move, or delete files and directories
    4. Process text with utilities like grep, sed, awk

    Advantages over regular bash tool:
    - Streamlined interface for common commands
    - Optimized for one-line operations
    - Focuses on readable command output
    - Perfect for file system operations and text processing

    Example commands:
    - `ls -la /path/to/dir`
    - `grep -r \"pattern\" /search/path`
    - `find . -name \"*.txt\" -mtime -7`
    - `cat file.txt | grep pattern | wc -l`
    - `du -sh /path/to/dir`

    Note: Commands run with your current user permissions in the default working directory. Uses 'sh -c'.
    """
    if not cmd or not cmd.strip():
        return "Error: Command cannot be empty."

    try:
        # Use sh -c for shell interpretation in the default working directory
        return_code, stdout, stderr = await run_subprocess(["sh", "-c", cmd], cwd=None)
        success = return_code == 0

        # Check for permission denied specifically in stderr if failed
        if not success and "permission denied" in stderr.lower():
             return f"Error: Permission denied.\nSTDERR:\n{stderr.strip()}"

        return (
            f"Command completed with status {return_code}\n\n"
            f"STDOUT:\n{stdout.strip()}\n\n"
            f"STDERR:\n{stderr.strip()}"
        )
    except FileNotFoundError as e:
        return f"Error: 'sh' command not found. Ensure a shell is available in the environment. {e}"
    except Exception as e:
        return f"Error executing quick_bash command: {e}"

if __name__ == "__main__":
    print("Starting Quick Bash MCP Server...")
    mcp.run()