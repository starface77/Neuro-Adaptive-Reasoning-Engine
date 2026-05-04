"""CLI commands for NARE daemon control.

Usage:
    nare daemon start
    nare daemon stop
    nare daemon status
    nare task add "description"
    nare task list
    nare task cancel <task_id>
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

from nare.daemon import NareDaemon, TaskQueue, Task, TaskPriority

def daemon_start(repo_path: str = "."):
    """Start daemon in background."""
    daemon = NareDaemon(repo_path=repo_path)

    if daemon.is_running():
        print("❌ Daemon already running")
        return 1

    print("🚀 Starting NARE daemon...")

    if sys.platform != "win32":
        pid = os.fork()
        if pid > 0:

            print(f"✓ Daemon started (PID: {pid})")
            return 0

    daemon.start()
    return 0

def daemon_stop():
    """Stop daemon."""
    daemon = NareDaemon()

    if not daemon.is_running():
        print("❌ Daemon is not running")
        return 1

    print("🛑 Stopping daemon...")

    import signal
    pid = int(Path(".nare_daemon/daemon.pid").read_text())

    try:
        import os
        os.kill(pid, signal.SIGTERM)

        for _ in range(10):
            time.sleep(0.5)
            if not daemon.is_running():
                print("✓ Daemon stopped")
                return 0

        print("⚠ Daemon did not stop gracefully, forcing...")
        os.kill(pid, signal.SIGKILL)
        print("✓ Daemon killed")
        return 0

    except ProcessLookupError:
        print("✓ Daemon already stopped")
        return 0

def daemon_status():
    """Show daemon status."""
    daemon = NareDaemon()

    if daemon.is_running():
        pid = int(Path(".nare_daemon/daemon.pid").read_text())
        print(f"✓ Daemon is running (PID: {pid})")

        queue = TaskQueue()
        pending = len(queue.list_tasks(status="pending"))
        running = len(queue.list_tasks(status="running"))
        completed = len(queue.list_tasks(status="completed"))
        failed = len(queue.list_tasks(status="failed"))

        print(f"\n📊 Queue statistics:")
        print(f"  Pending:   {pending}")
        print(f"  Running:   {running}")
        print(f"  Completed: {completed}")
        print(f"  Failed:    {failed}")

        return 0
    else:
        print("❌ Daemon is not running")
        return 1

def task_add(description: str, priority: str = "normal", **kwargs):
    """Add task to queue."""
    queue = TaskQueue()

    priority_map = {
        "low": TaskPriority.LOW.value,
        "normal": TaskPriority.NORMAL.value,
        "high": TaskPriority.HIGH.value,
    }
    priority_value = priority_map.get(priority.lower(), TaskPriority.NORMAL.value)

    task = Task(
        description=description,
        priority=priority_value,
        **kwargs
    )

    task_id = queue.add_task(task)
    print(f"✓ Task added: {task_id}")
    print(f"  Description: {description}")
    print(f"  Priority: {priority}")

    return 0

def task_list(status: Optional[str] = None, limit: int = 20):
    """List tasks."""
    queue = TaskQueue()
    tasks = queue.list_tasks(status=status, limit=limit)

    if not tasks:
        print("No tasks found")
        return 0

    print(f"\n📋 Tasks ({len(tasks)}):\n")

    for task in tasks:
        status_icon = {
            "pending": "⏳",
            "running": "▶",
            "completed": "✓",
            "failed": "✗",
            "cancelled": "⊘"
        }.get(task.status, "?")

        print(f"{status_icon} {task.id}")
        print(f"  {task.description[:80]}")
        print(f"  Status: {task.status} | Priority: {task.priority}")

        if task.tokens_used:
            print(f"  Tokens: {task.tokens_used:,}")

        if task.error:
            print(f"  Error: {task.error[:100]}")

        print()

    return 0

def task_cancel(task_id: str):
    """Cancel a pending task."""
    queue = TaskQueue()

    if queue.cancel_task(task_id):
        print(f"✓ Task {task_id} cancelled")
        return 0
    else:
        print(f"❌ Could not cancel task {task_id} (not found or already running)")
        return 1

def task_logs(task_id: str):
    """Show task logs."""
    queue = TaskQueue()
    task = queue.get_task(task_id)

    if not task:
        print(f"❌ Task {task_id} not found")
        return 1

    print(f"\n📄 Task {task_id}\n")
    print(f"Description: {task.description}")
    print(f"Status: {task.status}")
    print(f"Priority: {task.priority}")
    print(f"Created: {time.ctime(task.created_at)}")

    if task.started_at:
        print(f"Started: {time.ctime(task.started_at)}")

    if task.completed_at:
        print(f"Completed: {time.ctime(task.completed_at)}")
        duration = task.completed_at - task.started_at
        print(f"Duration: {duration:.1f}s")

    if task.tokens_used:
        print(f"Tokens: {task.tokens_used:,}")

    if task.result:
        print(f"\n📝 Result:\n{task.result}")

    if task.error:
        print(f"\n❌ Error:\n{task.error}")

    return 0
