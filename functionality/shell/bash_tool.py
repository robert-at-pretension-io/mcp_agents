from typing import Any, Dict, Optional
import subprocess
import os
from pathlib import Path

from agents import function_tool, RunContextWrapper

@function_tool
async def bash_tool(ctx: RunContextWrapper[Any], command: str, cwd: Optional[str] = None) -> Dict[str, Any]:
    """Executes bash shell commands on the host system.

    Use this tool to:
                
    1. Run system commands and utilities
    2. Check file/directory status
    3. Process text/data with command line tools
    4. Manage files and directories
    
    Important notes:
    - Always provide the full command including any required flags
    - Use absolute paths or specify working directory (cwd)
    - Commands run with the same permissions as the host process
    - Output is limited to stdout/stderr (no interactive prompts)
    - Commands run in a non-interactive shell (sh)
    
    Args:
        command: The bash command to execute
        cwd: The working directory for the command. Defaults to current directory if not specified.
    
    Returns:
        A dictionary with execution results including stdout and stderr
    """
    cwd = cwd or os.getcwd()
    
    # Create working directory if it doesn't exist
    cwd_path = Path(cwd)
    if not cwd_path.exists():
        os.makedirs(cwd, exist_ok=True)
    
    # Execute the command
    try:
        process = subprocess.run(
            ["sh", "-c", command],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False
        )
        
        return {
            "success": process.returncode == 0,
            "status": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr
        }
    
    except Exception as e:
        return {
            "success": False,
            "status": 1,
            "stdout": "",
            "stderr": str(e)
        }