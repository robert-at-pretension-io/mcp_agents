import asyncio
import os
import re
import shutil
from typing import List, Optional, Tuple

import aiofiles
import aiofiles.os

from mcp.server.fastmcp import Context, FastMCP

# MCP Server Initialization
mcp = FastMCP("Regex Replace Tool", description="Performs multi-line regex replacements in files.")

def get_context_lines(all_lines: List[str], line_idx: int, num_context: int) -> List[str]:
    """Helper to extract context lines around a specific index."""
    start = max(0, line_idx - num_context)
    end = min(len(all_lines), line_idx + num_context + 1)
    context = []
    for i in range(start, end):
        prefix = ">> " if i == line_idx else "   "
        context.append(f"{prefix}{i + 1}: {all_lines[i]}")
    return context

@mcp.tool()
async def regex_replace(
    ctx: Context,
    file_path: str,
    start_pattern: str,
    end_pattern: str,
    replacement: str,
    dry_run: bool = False,
    match_occurrence: Optional[int] = None, # 1-based index
    match_all: bool = False,
    create_backup: bool = False,
    context_lines: int = 2,
) -> str:
    """
    Precision multi-line text replacement tool using regular expressions to safely modify files. Use this to:

    1. Replace sections of code or content between specified regex patterns
    2. Update multi-line blocks in configuration files
    3. Modify sections of text while preserving surrounding content
    4. Make targeted multi-line modifications in various file types

    Safety features:
    - Reports the number of matches when more than one is found
    - Preserves original file line endings and encoding (best effort)
    - Provides dry-run option to preview changes
    - Can create backup files automatically
    - Shows context around replacements
    - Never modifies files unless explicitly instructed

    Pattern syntax guide (Python regex):
    - Character classes: [a-z], [0-9], \\w (word), \\d (digit), \\s (whitespace)
    - Anchors: ^ (start of line), $ (end of line), \\b (word boundary)
    - Quantifiers: * (0+), + (1+), ? (0-1), {n} (exactly n), {n,m} (n to m)
    - Groups: (pattern) creates a capture group, (?:pattern) non-capturing

    Example use cases:
    - Replace a function: '^def my_func\\(self\\):$' and '^\\s+return' with a new implementation
    - Update config blocks: '^config: {$' and '^}$' with new configuration
    - Replace specific sections: '^# START SECTION$' and '^# END SECTION$' with new content
    """
    context_lines = max(0, min(context_lines, 10)) # Clamp context lines

    # --- Input Validation ---
    if not file_path:
         return "Error: file_path parameter is missing."
    # Use aiofiles.os for async path checks
    try:
        if not await aiofiles.os.path.exists(file_path):
            return f"Error: File not found at '{file_path}'"
        if not await aiofiles.os.path.isfile(file_path):
            return f"Error: Path '{file_path}' is not a file."
    except Exception as path_e:
         return f"Error checking file path '{file_path}': {path_e}"

    if not start_pattern or not start_pattern.strip():
        return "Error: Start pattern cannot be empty."
    if not end_pattern or not end_pattern.strip():
        return "Error: End pattern cannot be empty."
    if match_occurrence is not None and match_occurrence < 1:
         return "Error: match_occurrence must be 1 or greater."

    try:
        start_re = re.compile(start_pattern)
        end_re = re.compile(end_pattern)
    except re.error as e:
        return f"Error: Invalid regex pattern provided: {e}"

    # --- File Reading and Line Ending Detection ---
    try:
        async with aiofiles.open(file_path, mode='r', encoding='utf-8', errors='replace') as f:
            content = await f.read()

        # Detect line endings (simple check)
        if '\r\n' in content:
            line_ending = '\r\n'
        elif '\n' in content:
            line_ending = '\n'
        else:
            line_ending = os.linesep # Fallback to OS default

        lines = content.split(line_ending)
    except Exception as e:
        return f"Error reading file '{file_path}': {e}"

    if not lines or (len(lines) == 1 and not lines[0].strip()):
         return f"File '{file_path}' is empty, nothing to replace."

    # --- Find Matches ---
    start_match_indices = [i for i, line in enumerate(lines) if start_re.search(line)]

    if not start_match_indices:
        return f"No matches found for start pattern. No changes made."

    target_replacements: List[Tuple[int, int]] = [] # Stores (start_idx, end_idx) for replacement

    if match_all:
        indices_to_process = start_match_indices
    elif match_occurrence is not None:
        if match_occurrence > len(start_match_indices):
            return f"Error: Invalid occurrence {match_occurrence}. Found {len(start_match_indices)} matches for start pattern."
        indices_to_process = [start_match_indices[match_occurrence - 1]] # Get the specific index
    elif len(start_match_indices) == 1:
        indices_to_process = start_match_indices
    else:
         return (f"Error: Found {len(start_match_indices)} matches for start pattern. "
                 f"Specify which to replace using 'match_occurrence' (1-{len(start_match_indices)}) "
                 f"or set 'match_all=True'.")

    # Find corresponding end matches for the targeted start matches
    processed_indices = set() # Keep track of lines already part of a replacement block
    for start_idx in indices_to_process:
        if start_idx in processed_indices:
            continue # Skip if this start line is already part of a found block

        # Find the *first* end match at or after the start line
        found_end_idx = -1
        for end_idx in range(start_idx, len(lines)):
            if end_idx in processed_indices: # Don't let end match overlap previous block
                 continue
            if end_re.search(lines[end_idx]):
                found_end_idx = end_idx
                break

        if found_end_idx == -1:
            # Check if the error message should mention the specific start line
            # Since match_all processes multiple, a general error might be better
            return (f"Error: Could not find a corresponding end match for pattern '{end_pattern}' "
                    f"after at least one of the targeted start matches (e.g., near line {start_idx + 1}).")


        # Add the valid block to our target replacements
        target_replacements.append((start_idx, found_end_idx))
        # Mark these lines as processed
        for i in range(start_idx, found_end_idx + 1):
            processed_indices.add(i)


    # Sort replacements by start index to process them correctly if match_all is used
    target_replacements.sort(key=lambda x: x[0])

    # --- Create Backup (if requested and not dry run) ---
    backup_path = ""
    if create_backup and not dry_run:
        backup_path = f"{file_path}.bak"
        try:
            if await aiofiles.os.path.exists(backup_path):
                 await aiofiles.os.remove(backup_path)
            # Use shutil.copyfile via asyncio.to_thread for robustness
            await asyncio.to_thread(shutil.copyfile, file_path, backup_path)
            # Verify backup
            if not await aiofiles.os.path.exists(backup_path):
                 raise OSError("Backup file creation failed verification.")
            print(f"Created backup file: {backup_path}")
        except Exception as e:
            return f"Error creating backup file '{backup_path}': {e}"

    # --- Perform Replacements (or simulate for dry run) ---
    new_lines = []
    last_processed_end_idx = -1
    replacements_made_info = []

    for start_idx, end_idx in target_replacements:
        # Add lines before the current replacement block
        new_lines.extend(lines[last_processed_end_idx + 1 : start_idx])

        # Get context before modification (for reporting)
        start_context = get_context_lines(lines, start_idx, context_lines)
        end_context = get_context_lines(lines, end_idx, context_lines)
        replaced_lines_count = (end_idx - start_idx) + 1

        replacements_made_info.append({
            "start_line": start_idx + 1,
            "end_line": end_idx + 1,
            "lines_replaced": replaced_lines_count,
            "start_context": start_context,
            "end_context": end_context,
        })

        # Add the replacement text (split into lines if it contains newlines)
        # Correctly handle line endings within the replacement text itself
        replacement_lines = replacement.splitlines()
        new_lines.extend(replacement_lines)

        last_processed_end_idx = end_idx

    # Add any remaining lines after the last replacement block
    new_lines.extend(lines[last_processed_end_idx + 1 :])

    # --- Write Changes (if not dry run) ---
    if not dry_run:
        try:
            # Join using the detected line ending
            new_content = line_ending.join(new_lines)
            async with aiofiles.open(file_path, mode='w', encoding='utf-8') as f:
                await f.write(new_content)

            # Verify write (optional but recommended)
            async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
                written_content = await f.read()
            if written_content != new_content:
                 # Attempt to restore from backup if possible
                 if backup_path and await aiofiles.os.path.exists(backup_path):
                    try:
                        await asyncio.to_thread(shutil.copyfile, backup_path, file_path)
                        return (f"Error: File content verification failed after write. "
                                f"Restored from backup '{backup_path}'. Please check the file.")
                    except Exception as restore_e:
                        return (f"Error: File content verification failed after write. "
                                f"Attempted to restore from backup '{backup_path}' but failed: {restore_e}. "
                                f"Manual check required.")
                 else:
                    return ("Error: File content verification failed after write. "
                            "No backup was created or available. Manual check required.")

        except Exception as e:
            return f"Error writing changes to file '{file_path}': {e}"

    # --- Build Response ---
    response_text = ""
    if dry_run:
        response_text += "DRY RUN - No changes made to file.\n\n"
    if create_backup and not dry_run and backup_path:
        response_text += f"Backup created at: {backup_path}\n\n"

    if not replacements_made_info:
         # This case should ideally not be reached if start matches were found and end matches were required
         response_text += "No valid replacement blocks were identified or processed."
    else:
        for i, info in enumerate(replacements_made_info):
             response_text += (
                f"Replacement #{i + 1}: Replaced {info['lines_replaced']} lines "
                f"(from line {info['start_line']} to {info['end_line']})\n"
             )
             if context_lines > 0:
                 response_text += "Context Before (start):\n"
                 response_text += "\n".join(info['start_context']) + "\n"
                 response_text += "Context Before (end):\n"
                 response_text += "\n".join(info['end_context']) + "\n\n"

        summary_action = "potential changes identified" if dry_run else "replacements made"
        response_text += f"\nSummary: {len(replacements_made_info)} {summary_action} in '{file_path}'."

    return response_text.strip()


if __name__ == "__main__":
    print("Starting Regex Replace MCP Server...")
    mcp.run()