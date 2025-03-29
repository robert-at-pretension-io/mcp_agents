import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles
import aiofiles.os


class TaskStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    ENDED = "ended"
    ERROR = "error"


@dataclass
class TaskState:
    task_id: str
    command: str
    reason: str
    status: TaskStatus = TaskStatus.CREATED
    stdout: str = ""
    stderr: str = ""


class LongRunningTaskManager:
    def __init__(self, filename: str = "mcp_tasks.json"):
        # Determine persistence path robustly
        home_dir = Path.home()
        default_store_dir = home_dir / ".mcp_tool_data"
        store_dir_str = os.getenv("MCP_DATA_DIR", str(default_store_dir))
        self.persistence_path = Path(store_dir_str) / filename
        self.tasks_in_memory: Dict[str, TaskState] = {}
        self._lock = asyncio.Lock()
        self._save_task: Optional[asyncio.Task] = None
        self._save_pending = False
        print(f"LongRunningTaskManager storing tasks at: {self.persistence_path}")

    async def _ensure_dir_exists(self):
        """Ensures the persistence directory exists."""
        if not await aiofiles.os.path.exists(self.persistence_path.parent):
            try:
                await aiofiles.os.makedirs(self.persistence_path.parent, exist_ok=True)
            except OSError as e:
                print(f"Error creating directory {self.persistence_path.parent}: {e}")
                # Decide if this is fatal or if we can continue in-memory only

    async def load_persistent_tasks(self) -> None:
        await self._ensure_dir_exists()
        async with self._lock:
            if await aiofiles.os.path.exists(self.persistence_path):
                try:
                    async with aiofiles.open(self.persistence_path, mode='r') as f:
                        content = await f.read()
                        if content.strip(): # Avoid parsing empty file
                            tasks_data = json.loads(content)
                            self.tasks_in_memory = {
                                task_id: TaskState(**data)
                                for task_id, data in tasks_data.items()
                            }
                            print(f"Loaded {len(self.tasks_in_memory)} tasks from {self.persistence_path}")
                        else:
                            print(f"Task file {self.persistence_path} is empty, starting fresh.")
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from {self.persistence_path}: {e}. Starting with empty state.")
                    self.tasks_in_memory = {}
                except Exception as e:
                    print(f"Error loading tasks from {self.persistence_path}: {e}. Starting with empty state.")
                    self.tasks_in_memory = {}
            else:
                 print(f"Task file {self.persistence_path} not found, starting fresh.")


    async def _schedule_save(self):
        """Schedules a save operation, debouncing multiple requests."""
        self._save_pending = True
        if self._save_task is None or self._save_task.done():
            # Schedule a new save task after a short delay (e.g., 1 second)
             self._save_task = asyncio.create_task(self._delayed_save())


    async def _delayed_save(self):
        await asyncio.sleep(1) # Debounce saves
        if self._save_pending:
            await self.save()
            self._save_pending = False


    async def save(self) -> None:
        await self._ensure_dir_exists()
        async with self._lock:
            try:
                tasks_to_save = {
                    task_id: task.__dict__ for task_id, task in self.tasks_in_memory.items()
                }
                async with aiofiles.open(self.persistence_path, mode='w') as f:
                    await f.write(json.dumps(tasks_to_save, indent=2))
                # print(f"Saved {len(tasks_to_save)} tasks to {self.persistence_path}")
            except Exception as e:
                print(f"Error saving tasks to {self.persistence_path}: {e}")

    async def _run_task_background(self, task_id: str):
        """The actual background process runner."""
        state = None
        async with self._lock:
            state = self.tasks_in_memory.get(task_id)
            if state:
                state.status = TaskStatus.RUNNING
            else:
                print(f"Error: Task {task_id} not found for background run.")
                return # Should not happen

        await self._schedule_save()

        process = None
        try:
            # Use asyncio's subprocess handling
            process = await asyncio.create_subprocess_shell(
                state.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Process stdout and stderr concurrently
            async def read_stream(stream, stream_name):
                if stream:
                    while True:
                        line_bytes = await stream.readline()
                        if not line_bytes:
                            break
                        line = line_bytes.decode('utf-8', errors='replace').rstrip()
                        async with self._lock:
                            current_state = self.tasks_in_memory.get(task_id)
                            if current_state:
                                if stream_name == 'stdout':
                                    current_state.stdout += line + "\n"
                                else:
                                    current_state.stderr += line + "\n"
                        # Schedule save after processing each line (debounced)
                        await self._schedule_save()


            stdout_task = asyncio.create_task(read_stream(process.stdout, 'stdout'))
            stderr_task = asyncio.create_task(read_stream(process.stderr, 'stderr'))

            # Wait for the process to complete and streams to be fully read
            await asyncio.gather(stdout_task, stderr_task)
            return_code = await process.wait()

            async with self._lock:
                final_state = self.tasks_in_memory.get(task_id)
                if final_state:
                    final_state.status = TaskStatus.ENDED if return_code == 0 else TaskStatus.ERROR
                    if return_code != 0:
                         final_state.stderr += f"\n[Process exited with code {return_code}]"

        except Exception as e:
            print(f"Error running task {task_id} command '{state.command}': {e}")
            async with self._lock:
                 error_state = self.tasks_in_memory.get(task_id)
                 if error_state:
                    error_state.status = TaskStatus.ERROR
                    error_state.stderr += f"\n[Failed to execute command: {e}]"
        finally:
            # Final save ensures the end state is persisted
            await self.save()


    async def spawn_task(self, command: str, reason: str) -> str:
        task_id = f"task-{uuid.uuid4()}"
        state = TaskState(task_id=task_id, command=command, reason=reason, status=TaskStatus.CREATED)

        async with self._lock:
            self.tasks_in_memory[task_id] = state
        await self._schedule_save() # Save the initial created state

        # Start the background execution
        asyncio.create_task(self._run_task_background(task_id))

        return task_id

    async def get_task_status(self, task_id: str) -> Optional[TaskState]:
        async with self._lock:
            # Return a copy to avoid external modification
            task = self.tasks_in_memory.get(task_id)
            return TaskState(**task.__dict__) if task else None

    async def list_tasks(self, filter_status: Optional[TaskStatus] = None) -> List[TaskState]:
        async with self._lock:
            tasks = []
            for task in self.tasks_in_memory.values():
                 if filter_status is None or task.status == filter_status:
                    # Return copies
                    tasks.append(TaskState(**task.__dict__))
            return tasks

    @staticmethod
    def last_n_lines(text: str, n: int) -> str:
        """Helper to get the last N lines of a string."""
        lines = text.splitlines()
        return "\n".join(lines[-n:])