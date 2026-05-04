"""NARE Daemon - Background worker for autonomous task execution.

Runs in background, processes tasks from queue, integrates with NARE core.
"""

import os
import sys
import time
import signal
import logging
import traceback
from pathlib import Path
from typing import Optional

from nare.daemon.task_queue import TaskQueue, Task, TaskStatus
from nare.cli.session import NareSession

class NareDaemon:
    """Background daemon for autonomous task execution."""

    def __init__(
        self,
        repo_path: str = ".",
        db_path: str = ".nare_daemon/tasks.db",
        log_path: str = ".nare_daemon/daemon.log",
        pid_path: str = ".nare_daemon/daemon.pid"
    ):
        self.repo_path = Path(repo_path).resolve()
        self.db_path = db_path
        self.log_path = Path(log_path)
        self.pid_path = Path(pid_path)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(self.log_path),
                logging.StreamHandler()
            ]
        )
        self.log = logging.getLogger("nare.daemon")

        self.queue = TaskQueue(db_path)

        self.session = NareSession(repo_path=str(self.repo_path))

        self.running = False
        self.current_task: Optional[Task] = None

    def start(self):
        """Start daemon."""

        if self.is_running():
            self.log.error("Daemon already running")
            return False

        self.pid_path.write_text(str(os.getpid()))

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.log.info(f"NARE Daemon started (PID: {os.getpid()})")
        self.log.info(f"Repository: {self.repo_path}")
        self.log.info(f"Database: {self.db_path}")

        self.running = True
        self._main_loop()

        return True

    def stop(self):
        """Stop daemon gracefully."""
        self.log.info("Stopping daemon...")
        self.running = False

        if self.current_task:
            self.log.info(f"Waiting for task {self.current_task.id} to finish...")

        if self.pid_path.exists():
            self.pid_path.unlink()

        self.log.info("Daemon stopped")

    def is_running(self) -> bool:
        """Check if daemon is running."""
        if not self.pid_path.exists():
            return False

        try:
            pid = int(self.pid_path.read_text())

            os.kill(pid, 0)
            return True
        except (ProcessLookupError, ValueError):

            self.pid_path.unlink()
            return False

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.log.info(f"Received signal {signum}")
        self.stop()

    def _main_loop(self):
        """Main worker loop."""
        idle_count = 0
        idle_threshold = 60

        while self.running:
            try:

                task = self.queue.get_next_task()

                if task is None:

                    idle_count += 1

                    if idle_count >= idle_threshold:

                        self._background_crystallization()
                        idle_count = 0

                    time.sleep(5)
                    continue

                idle_count = 0

                self._execute_task(task)

            except Exception as e:
                self.log.error(f"Error in main loop: {e}")
                self.log.error(traceback.format_exc())
                time.sleep(10)

    def _execute_task(self, task: Task):
        """Execute a single task."""
        self.current_task = task
        self.log.info(f"Starting task {task.id}: {task.description}")

        task.status = TaskStatus.RUNNING.value
        task.started_at = time.time()
        self.queue.update_task(task)

        try:

            result = self.session.solve(
                query=task.description,
                thinking_display=None
            )

            if result.get("_stop_reason") == "success" or result.get("final_answer"):
                task.status = TaskStatus.COMPLETED.value
                task.result = result.get("final_answer", "")
                task.tokens_used = result.get("_tokens", 0)
                self.log.info(f"Task {task.id} completed successfully")
            else:
                raise Exception(f"Task failed: {result.get('_stop_reason', 'unknown')}")

        except Exception as e:
            self.log.error(f"Task {task.id} failed: {e}")
            task.error = str(e)

            if task.current_retry < task.retry_count:
                task.current_retry += 1
                task.status = TaskStatus.PENDING.value
                self.log.info(f"Task {task.id} will retry ({task.current_retry}/{task.retry_count})")
            else:
                task.status = TaskStatus.FAILED.value
                self.log.error(f"Task {task.id} failed after {task.retry_count} retries")

        finally:

            task.completed_at = time.time()
            self.queue.update_task(task)

            if task.force_crystallization and task.status == TaskStatus.COMPLETED.value:
                self._trigger_crystallization()

            self.current_task = None

    def _background_crystallization(self):
        """Trigger background crystallization during idle time."""
        try:
            self.log.info("Starting background crystallization...")

            self.session.init_agent()

            if self.session.agent.config.sleep.enabled:
                if self.session.agent.evolution.check_compilation_trigger():

                    self.session.agent.evolution.run_compilation_cycle()
                    self.log.info("Background crystallization completed")
                else:
                    self.log.info("Crystallization not needed yet")

        except Exception as e:
            self.log.error(f"Background crystallization failed: {e}")

    def _trigger_crystallization(self):
        """Force crystallization after task completion."""
        try:
            self.log.info("Forcing crystallization...")

            self.session.agent.evolution.run_compilation_cycle()
            self.log.info("Forced crystallization completed")
        except Exception as e:
            self.log.error(f"Forced crystallization failed: {e}")

def main():
    """CLI entry point for daemon control."""
    import argparse

    parser = argparse.ArgumentParser(description="NARE Daemon")
    parser.add_argument("command", choices=["start", "stop", "status"], help="Daemon command")
    parser.add_argument("--repo", default=".", help="Repository path")
    parser.add_argument("--db", default=".nare_daemon/tasks.db", help="Database path")
    parser.add_argument("--log", default=".nare_daemon/daemon.log", help="Log file path")

    args = parser.parse_args()

    daemon = NareDaemon(
        repo_path=args.repo,
        db_path=args.db,
        log_path=args.log
    )

    if args.command == "start":
        daemon.start()
    elif args.command == "stop":
        if daemon.is_running():

            pid = int(Path(".nare_daemon/daemon.pid").read_text())
            os.kill(pid, signal.SIGTERM)
            print(f"Sent stop signal to daemon (PID: {pid})")
        else:
            print("Daemon is not running")
    elif args.command == "status":
        if daemon.is_running():
            pid = int(Path(".nare_daemon/daemon.pid").read_text())
            print(f"Daemon is running (PID: {pid})")
        else:
            print("Daemon is not running")

if __name__ == "__main__":
    main()
