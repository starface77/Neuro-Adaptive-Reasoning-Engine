"""Task queue database schema and operations.

SQLite-based task queue for NARE daemon.
"""

import sqlite3
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum
from dataclasses import dataclass, asdict

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 3
    HIGH = 5

@dataclass
class Task:
    """Task model."""
    id: Optional[str] = None
    description: str = ""
    priority: int = TaskPriority.NORMAL.value
    status: str = TaskStatus.PENDING.value
    dependencies: List[str] = None
    context: Dict[str, Any] = None
    max_tokens: int = 200000
    timeout: int = 3600
    retry_count: int = 3
    current_retry: int = 0

    can_use_cache: bool = True
    force_crystallization: bool = False

    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None
    tokens_used: int = 0

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.context is None:
            self.context = {}
        if self.created_at == 0.0:
            self.created_at = time.time()

class TaskQueue:
    """SQLite-based task queue."""

    def __init__(self, db_path: str = ".nare_daemon/tasks.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    priority INTEGER DEFAULT 3,
                    status TEXT DEFAULT 'pending',
                    dependencies TEXT,
                    context TEXT,
                    max_tokens INTEGER DEFAULT 200000,
                    timeout INTEGER DEFAULT 3600,
                    retry_count INTEGER DEFAULT 3,
                    current_retry INTEGER DEFAULT 0,
                    can_use_cache INTEGER DEFAULT 1,
                    force_crystallization INTEGER DEFAULT 0,
                    created_at REAL,
                    started_at REAL,
                    completed_at REAL,
                    result TEXT,
                    error TEXT,
                    tokens_used INTEGER DEFAULT 0
                )
            """)
                CREATE INDEX IF NOT EXISTS idx_status_priority
                ON tasks(status, priority DESC, created_at)
            """)
            conn.commit()

    def add_task(self, task: Task) -> str:
        """Add task to queue."""
        if task.id is None:
            task.id = f"task_{int(time.time() * 1000)}"

        with sqlite3.connect(self.db_path) as conn:
                INSERT INTO tasks (
                    id, description, priority, status, dependencies, context,
                    max_tokens, timeout, retry_count, current_retry,
                    can_use_cache, force_crystallization, created_at,
                    started_at, completed_at, result, error, tokens_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task.id, task.description, task.priority, task.status,
                json.dumps(task.dependencies), json.dumps(task.context),
                task.max_tokens, task.timeout, task.retry_count, task.current_retry,
                int(task.can_use_cache), int(task.force_crystallization),
                task.created_at, task.started_at, task.completed_at,
                task.result, task.error, task.tokens_used
            ))
            conn.commit()

        return task.id

    def get_next_task(self) -> Optional[Task]:
        """Get next pending task with highest priority."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
                SELECT * FROM tasks
                WHERE status = 'pending'
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
            """)
            row = cursor.fetchone()

            if row is None:
                return None

            return self._row_to_task(row)

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
            row = cursor.fetchone()

            if row is None:
                return None

            return self._row_to_task(row)

    def update_task(self, task: Task):
        """Update task in database."""
        with sqlite3.connect(self.db_path) as conn:
                UPDATE tasks SET
                    description = ?, priority = ?, status = ?,
                    dependencies = ?, context = ?, max_tokens = ?,
                    timeout = ?, retry_count = ?, current_retry = ?,
                    can_use_cache = ?, force_crystallization = ?,
                    started_at = ?, completed_at = ?, result = ?,
                    error = ?, tokens_used = ?
                WHERE id = ?
            """, (
                task.description, task.priority, task.status,
                json.dumps(task.dependencies), json.dumps(task.context),
                task.max_tokens, task.timeout, task.retry_count, task.current_retry,
                int(task.can_use_cache), int(task.force_crystallization),
                task.started_at, task.completed_at, task.result,
                task.error, task.tokens_used, task.id
            ))
            conn.commit()

    def list_tasks(self, status: Optional[str] = None, limit: int = 100) -> List[Task]:
        """List tasks, optionally filtered by status."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if status:
                cursor = conn.execute(
                    "SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status, limit)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                )

            return [self._row_to_task(row) for row in cursor.fetchall()]

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE tasks SET status = 'cancelled' WHERE id = ? AND status = 'pending'",
                (task_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def _row_to_task(self, row: sqlite3.Row) -> Task:
        """Convert database row to Task object."""
        return Task(
            id=row['id'],
            description=row['description'],
            priority=row['priority'],
            status=row['status'],
            dependencies=json.loads(row['dependencies']),
            context=json.loads(row['context']),
            max_tokens=row['max_tokens'],
            timeout=row['timeout'],
            retry_count=row['retry_count'],
            current_retry=row['current_retry'],
            can_use_cache=bool(row['can_use_cache']),
            force_crystallization=bool(row['force_crystallization']),
            created_at=row['created_at'],
            started_at=row['started_at'],
            completed_at=row['completed_at'],
            result=row['result'],
            error=row['error'],
            tokens_used=row['tokens_used']
        )
