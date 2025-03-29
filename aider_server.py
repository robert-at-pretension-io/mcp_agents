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
mcp = FastMCP("Aider Tool", description="Runs the Aider AI pair programming tool.")

@mcp.tool()
async def aider(
    ctx: Context,
    directory: str,
    message: str,
    options: Optional[List[str]] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    thinking_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
) -> str:
    """
    AI pair programming tool for making targeted code changes. Use this tool to:

    1. Implement new features or functionality in existing code
    2. Add tests to an existing codebase
    3. Fix bugs in code
    4. Refactor or improve existing code
    5. Make structural changes across multiple files

    When using aider, make sure to pass ALL of the context into the message needed for a particular issue. don't just provide the solution.

    The tool requires:
    - A directory path where the code exists
    - A detailed message describing what changes to make. Please only describe one change per message. If you need to make multiple changes, please submit multiple requests. You must include all context required because this tool doesn't have any memory of previous requests.

    Best practices for messages:
    - Clearly describe the problem we're seeing in the tests
    - Show the relevant code that's failing
    - Explain why it's failing
    - Provide the specific error messages
    - Outline the approach to fix it
    - Include any related code that might be affected by the changes
    - Specify the file paths that include relevant context for the problem


    Note: This tool runs aider with the --yes-always flag which automatically accepts all proposed changes.

    MODEL AND PROVIDER OPTIONS:
    This tool supports both Anthropic (Claude) and OpenAI models. You can specify which provider and model to use:

    - Default provider: 'anthropic' with model 'anthropic/claude-3-7-sonnet-20250219'
    - Alternative provider: 'openai' with default model 'openai/o3-mini'

    Examples of provider/model usage:
    - Basic usage (uses default Anthropic model): {\"directory\": \"/path/to/code\", \"message\": \"Fix the bug\"}
    - Specify provider: {\"directory\": \"/path/to/code\", \"message\": \"Fix the bug\", \"provider\": \"openai\"}
    - Specify provider and model: {\"directory\": \"/path/to/code\", \"message\": \"Fix the bug\", \"provider\": \"anthropic\", \"model\": \"claude-3-opus-20240229\"}

    ADVANCED FEATURES:
    - For Anthropic models (Claude), the default 'thinking_tokens' is set to 32000 for optimal performance, but you can override it:
      Example: {\"directory\": \"/path/to/code\", \"message\": \"Fix the bug\", \"provider\": \"anthropic\", \"thinking_tokens\": 16000}

    - For OpenAI models, the default 'reasoning_effort' is set to 'high' for optimal performance, but you can override it:
      Example: {\"directory\": \"/path/to/code\", \"message\": \"Fix the bug\", \"provider\": \"openai\", \"reasoning_effort\": \"medium\"}
      Valid values: 'auto', 'low', 'medium', 'high'

    Note: The tool will look for API keys in environment variables. It first checks for provider-specific keys
    (ANTHROPIC_API_KEY or OPENAI_API_KEY) and then falls back to AIDER_API_KEY if needed.
    """
    # --- Input Validation ---
    if not directory or not os.path.exists(directory):
        return f"Error: Directory '{directory}' does not exist."
    if not os.path.isdir(directory):
        return f"Error: Path '{directory}' is not a directory."
    if not message or not message.strip():
        return "Error: Message cannot be empty."

    # --- Determine Provider and API Key ---
    has_anthropic_key = os.getenv("ANTHROPIC_API_KEY") is not None
    has_openai_key = os.getenv("OPENAI_API_KEY") is not None
    aider_fallback_key = os.getenv("AIDER_API_KEY")

    if provider:
        prov = provider.lower()
        if prov not in ["anthropic", "openai"]:
            print(f"Warning: Unsupported provider '{provider}'. Defaulting to 'anthropic'.")
            prov = "anthropic"
    elif has_anthropic_key:
        prov = "anthropic"
    elif has_openai_key:
        prov = "openai"
    else:
        print("Warning: No ANTHROPIC_API_KEY or OPENAI_API_KEY found. Defaulting provider to 'anthropic'. Will rely on AIDER_API_KEY or aider's internal config.")
        prov = "anthropic" # Default if no specific key found

    # Get the correct API key
    provider_env_key = f"{prov.upper()}_API_KEY"
    api_key = os.getenv(provider_env_key) or aider_fallback_key

    if not api_key:
         print(f"Warning: No API key found for provider '{prov}'. Checked {provider_env_key} and AIDER_API_KEY. Aider might use its own configuration.")
         # Aider might still work if configured internally, so don't error out yet.

    # --- Determine Model ---
    final_model = model or os.getenv("AIDER_MODEL")
    if not final_model:
        if prov == "anthropic":
            final_model = "anthropic/claude-3-opus-20240229" # Or use latest like claude-3-haiku-20240307
            print(f"Using default Anthropic model: {final_model}")
        elif prov == "openai":
             final_model = "openai/gpt-4-turbo" # Or "oai/gpt-4o" if available
             print(f"Using default OpenAI model: {final_model}")
        # else: stay None if provider is unknown and no default

    # --- Build Aider Command ---
    command_args = ["aider"] # Assuming 'aider' is in the system PATH
    command_args.extend(["--message", message])
    command_args.extend(["--yes-always"])
    command_args.extend(["--no-detect-urls"])

    if api_key:
        # Pass API key with provider prefix if available
        command_args.extend(["--api-key", f"{prov}={api_key}"])
    # else: If no key, don't pass the --api-key flag, let aider handle it

    if final_model:
        command_args.extend(["--model", final_model])

    # Provider-specific options
    if prov == "anthropic" and thinking_tokens is not None:
        # Ensure thinking_tokens is positive
        thinking_tokens = max(0, thinking_tokens)
        command_args.extend(["--thinking-tokens", str(thinking_tokens)])
    elif prov == "openai" and reasoning_effort is not None:
        valid_efforts = ["auto", "low", "medium", "high"]
        validated_effort = reasoning_effort.lower()
        if validated_effort not in valid_efforts:
            print(f"Warning: Invalid reasoning_effort '{reasoning_effort}'. Defaulting to 'high'.")
            validated_effort = "high"
        command_args.extend(["--reasoning-effort", validated_effort])

    if options:
        command_args.extend(options)

    # --- Execute Aider ---
    try:
        print(f"Running aider in '{directory}' with args: {' '.join(shlex.quote(a) for a in command_args)}") # Log command safely
        return_code, stdout, stderr = await run_subprocess(command_args, cwd=directory)
        success = return_code == 0

        model_info = f"Provider: {prov}"
        if final_model:
            model_info += f" | Model: {final_model}"

        result_status = "succeeded" if success else "failed"

        return (
            f"Aider execution {result_status} [{model_info}]\n\n"
            f"Directory: {directory}\n"
            f"Exit status: {return_code}\n\n"
            f"STDOUT:\n{stdout.strip()}\n\n"
            f"STDERR:\n{stderr.strip()}"
        )

    except FileNotFoundError:
        return "Error: 'aider' command not found. Please ensure aider is installed and in your system's PATH."
    except Exception as e:
        return f"Error executing aider: {e}"

if __name__ == "__main__":
    print("Starting Aider MCP Server...")
    mcp.run()