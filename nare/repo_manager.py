"""Repository manager for real SWE-bench testing.

Manages git repositories: cloning, checkout, applying patches, running tests.
"""

import os
import subprocess
import shutil
import logging
import tempfile
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class RepoManager:
    """Manages SWE-bench repositories for real testing."""

    def __init__(self, repos_dir: str = "swe_bench_repos"):
        """Initialize repository manager.

        Args:
            repos_dir: Directory to store cloned repositories
        """
        self.repos_dir = Path(repos_dir)
        self.repos_dir.mkdir(exist_ok=True)
        self.active_tasks = {}  # task_id -> repo_path

    def prepare_task(self, task: dict) -> str:
        """Prepare repository for a task.

        Args:
            task: Task dict with 'id', 'repo', 'base_commit'

        Returns:
            Path to prepared repository
        """
        task_id = task['id']
        repo_name = task['repo']  # e.g., "astropy/astropy"
        base_commit = task.get('base_commit')

        logging.info(f"[RepoManager] Preparing task {task_id}")

        # Ensure repo is cloned
        repo_path = self._ensure_repo(repo_name)

        # Checkout base commit if specified
        if base_commit:
            self._checkout(repo_path, base_commit)

        # Create working branch
        branch_name = f"swe-bench-{task_id}"
        self._create_branch(repo_path, branch_name)

        self.active_tasks[task_id] = str(repo_path)
        logging.info(f"[RepoManager] Task {task_id} ready at {repo_path}")

        return str(repo_path)

    def _ensure_repo(self, repo_name: str) -> Path:
        """Ensure repository is cloned.

        Args:
            repo_name: Repository name (e.g., "astropy/astropy")

        Returns:
            Path to repository
        """
        # Extract org and repo
        parts = repo_name.split('/')
        if len(parts) != 2:
            raise ValueError(f"Invalid repo name: {repo_name}")

        org, repo = parts
        repo_path = self.repos_dir / org / repo

        if repo_path.exists():
            logging.info(f"[RepoManager] Repository {repo_name} already exists")
            return repo_path

        # Clone repository
        logging.info(f"[RepoManager] Cloning {repo_name}...")
        repo_path.parent.mkdir(parents=True, exist_ok=True)

        clone_url = f"https://github.com/{repo_name}.git"
        result = subprocess.run(
            ["git", "clone", clone_url, str(repo_path)],
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to clone {repo_name}: {result.stderr}")

        logging.info(f"[RepoManager] Cloned {repo_name}")
        return repo_path

    def _checkout(self, repo_path: Path, commit: str):
        """Checkout specific commit.

        Args:
            repo_path: Path to repository
            commit: Commit hash or branch name
        """
        logging.info(f"[RepoManager] Checking out {commit}")

        result = subprocess.run(
            ["git", "checkout", commit],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to checkout {commit}: {result.stderr}")

    def _create_branch(self, repo_path: Path, branch_name: str):
        """Create and checkout new branch.

        Args:
            repo_path: Path to repository
            branch_name: Name of branch to create
        """
        # First, checkout main/master to avoid issues
        subprocess.run(
            ["git", "checkout", "main"],
            cwd=repo_path,
            capture_output=True,
            timeout=10
        )

        # Delete branch if exists (force delete)
        subprocess.run(
            ["git", "branch", "-D", branch_name],
            cwd=repo_path,
            capture_output=True,
            timeout=10
        )

        # Create new branch
        result = subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to create branch {branch_name}: {result.stderr}")

        logging.info(f"[RepoManager] Created branch {branch_name}")

    def apply_solution(self, task_id: str, solution: str) -> Tuple[bool, str]:
        """Apply NARA solution to repository.

        Args:
            task_id: Task ID
            solution: Solution text from NARA

        Returns:
            (success, error_message)
        """
        if task_id not in self.active_tasks:
            return False, f"Task {task_id} not prepared"

        repo_path = Path(self.active_tasks[task_id])

        # Parse solution for file changes
        changes = self._parse_solution(solution)

        if not changes:
            return False, "No file changes found in solution"

        # Apply changes
        try:
            for file_path, content in changes.items():
                full_path = repo_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)

                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                logging.info(f"[RepoManager] Modified {file_path}")

            return True, ""
        except Exception as e:
            return False, str(e)

    def _parse_solution(self, solution: str) -> Dict[str, str]:
        """Parse solution text to extract file changes.

        Supports multiple formats:
        1. File: path/to/file.py followed by code block
        2. Plain file paths with code blocks
        3. Markdown headers with file paths

        Args:
            solution: Solution text

        Returns:
            Dict of file_path -> content
        """
        changes = {}
        import re

        # Pattern 1: File: <path> followed by code block
        pattern1 = r'File:\s*([^\n]+)\s*```(?:python|py)?\s*\n(.*?)\n```'
        matches = re.findall(pattern1, solution, re.DOTALL | re.IGNORECASE)

        for file_path, content in matches:
            file_path = file_path.strip()
            changes[file_path] = content

        # Pattern 2: File path in text followed by code block
        # e.g., "The file astropy/modeling/separable.py should be modified:"
        pattern2 = r'([a-zA-Z0-9_/\-\.]+\.py)[:\s]*```(?:python|py)?\s*\n(.*?)\n```'
        matches2 = re.findall(pattern2, solution, re.DOTALL | re.IGNORECASE)

        for file_path, content in matches2:
            if file_path not in changes:
                changes[file_path] = content

        # Pattern 3: Just extract all code blocks and try to infer file from context
        # Look for file paths mentioned before code blocks
        lines = solution.split('\n')
        current_file = None
        in_code_block = False
        code_lines = []

        for i, line in enumerate(lines):
            # Check for file path mentions
            file_match = re.search(r'([a-zA-Z0-9_/\-]+/[a-zA-Z0-9_/\-]+\.py)', line)
            if file_match and not in_code_block:
                current_file = file_match.group(1)

            # Check for code block start
            if line.strip().startswith('```'):
                if not in_code_block:
                    in_code_block = True
                    code_lines = []
                else:
                    # End of code block
                    in_code_block = False
                    if current_file and current_file not in changes:
                        changes[current_file] = '\n'.join(code_lines)
                    current_file = None
            elif in_code_block:
                code_lines.append(line)

        logging.info(f"[RepoManager] Parsed {len(changes)} file changes from solution")
        return changes

    def run_tests(self, task_id: str, test_command: str, timeout: int = 300) -> Tuple[bool, str]:
        """Run tests in repository.

        Args:
            task_id: Task ID
            test_command: Test command to run
            timeout: Timeout in seconds

        Returns:
            (passed, output)
        """
        if task_id not in self.active_tasks:
            return False, f"Task {task_id} not prepared"

        repo_path = self.active_tasks[task_id]

        logging.info(f"[RepoManager] Running tests: {test_command}")

        try:
            result = subprocess.run(
                test_command,
                cwd=repo_path,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            passed = result.returncode == 0
            output = result.stdout + result.stderr

            if passed:
                logging.info(f"[RepoManager] Tests PASSED")
            else:
                logging.warning(f"[RepoManager] Tests FAILED")

            return passed, output

        except subprocess.TimeoutExpired:
            return False, f"Tests timed out after {timeout}s"
        except Exception as e:
            return False, str(e)

    def cleanup_task(self, task_id: str, keep_changes: bool = False):
        """Cleanup task repository.

        Args:
            task_id: Task ID
            keep_changes: If True, keep changes; if False, reset to clean state
        """
        if task_id not in self.active_tasks:
            return

        repo_path = Path(self.active_tasks[task_id])

        if not keep_changes:
            # Reset to clean state
            subprocess.run(
                ["git", "reset", "--hard", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                timeout=10
            )
            subprocess.run(
                ["git", "clean", "-fd"],
                cwd=repo_path,
                capture_output=True,
                timeout=10
            )

        del self.active_tasks[task_id]
        logging.info(f"[RepoManager] Cleaned up task {task_id}")

    def get_file_content(self, task_id: str, file_path: str) -> Optional[str]:
        """Get content of a file in repository.

        Args:
            task_id: Task ID
            file_path: Relative path to file

        Returns:
            File content or None if not found
        """
        if task_id not in self.active_tasks:
            return None

        repo_path = Path(self.active_tasks[task_id])
        full_path = repo_path / file_path

        if not full_path.exists():
            return None

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logging.warning(f"[RepoManager] Failed to read {file_path}: {e}")
            return None

    def list_files(self, task_id: str, pattern: str = "*") -> List[str]:
        """List files in repository matching pattern.

        Args:
            task_id: Task ID
            pattern: Glob pattern

        Returns:
            List of relative file paths
        """
        if task_id not in self.active_tasks:
            return []

        repo_path = Path(self.active_tasks[task_id])

        try:
            files = list(repo_path.rglob(pattern))
            return [str(f.relative_to(repo_path)) for f in files if f.is_file()]
        except Exception as e:
            logging.warning(f"[RepoManager] Failed to list files: {e}")
            return []
