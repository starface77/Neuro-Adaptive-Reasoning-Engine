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

        # Install dependencies for testing
        self._install_dependencies(repo_path, repo_name)

        self.active_tasks[task_id] = str(repo_path)
        logging.info(f"[RepoManager] Task {task_id} ready at {repo_path}")

        return str(repo_path)

    def _install_dependencies(self, repo_path: Path, repo_name: str):
        """Install test dependencies for repository.

        Args:
            repo_path: Path to repository
            repo_name: Repository name (e.g., "astropy/astropy")
        """
        # CRITICAL: Always install dependencies for each task
        # Don't use marker file - each task may need different commit's dependencies
        logging.info(f"[RepoManager] Installing dependencies for {repo_name}")

        try:
            # First, install the project itself in editable mode
            # This ensures all project dependencies are available
            install_result = subprocess.run(
                ["pip", "install", "-q", "-e", "."],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=300
            )

            if install_result.returncode == 0:
                logging.info(f"[RepoManager] Project installed successfully")
            else:
                logging.warning(f"[RepoManager] Failed to install project: {install_result.stderr[:200]}")

            # Then install common test dependencies
            common_deps = ["pytest", "pytest-cov", "hypothesis"]

            result = subprocess.run(
                ["pip", "install", "-q"] + common_deps,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                logging.info(f"[RepoManager] Test dependencies installed successfully")
            else:
                logging.warning(f"[RepoManager] Failed to install test dependencies: {result.stderr[:200]}")

        except Exception as e:
            logging.warning(f"[RepoManager] Error installing dependencies: {e}")

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

        # CRITICAL: Reset any local changes before checkout
        # This prevents "would be overwritten by checkout" errors
        reset_result = subprocess.run(
            ["git", "reset", "--hard"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30
        )

        if reset_result.returncode != 0:
            logging.warning(f"[RepoManager] Failed to reset: {reset_result.stderr}")

        # CRITICAL: Remove untracked files left by test_patch
        # Without this, checkout fails with "untracked working tree files would be overwritten"
        clean_result = subprocess.run(
            ["git", "clean", "-fd"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30
        )

        if clean_result.returncode != 0:
            logging.warning(f"[RepoManager] Failed to clean: {clean_result.stderr}")

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

        Supports:
        1. File sections with context (preferred)
        2. Complete file content in code blocks

        Args:
            solution: Solution text

        Returns:
            Dict of file_path -> content (complete file after merging changes)
        """
        changes = {}
        import re

        # Pattern: File: <path> followed by ```python block
        pattern = r'File:\s*([^\n]+?)(?:\s*\n+\s*)?```(?:python|py)?\s*\n(.*?)```'
        matches = re.findall(pattern, solution, re.DOTALL | re.IGNORECASE)

        for file_path, content in matches:
            file_path = file_path.strip()
            # Skip if no valid path (e.g., just "File:" with no path)
            if not file_path or file_path == ':':
                continue
            # Extract first word as path (remove any trailing text)
            file_path = file_path.split()[0] if file_path.split() else ''
            if not file_path or not file_path.endswith('.py'):
                continue
            content = content.strip()

            # Try to merge with original file if this looks like a section
            task_id = self._get_task_for_path(file_path)
            if task_id:
                original = self.get_file_content(task_id, file_path)
                if original:
                    merged = self._merge_section(original, content)
                    if merged:
                        changes[file_path] = merged
                        logging.info(f"[RepoManager] Merged section into {file_path}")
                        continue

            # Fallback: treat as complete file
            changes[file_path] = content

        logging.info(f"[RepoManager] Parsed {len(changes)} file changes from solution")

        if not changes:
            logging.warning(f"[RepoManager] Parsing failed. Solution preview:\n{solution[:500]}")

        return changes

    def _get_task_for_path(self, file_path: str) -> Optional[str]:
        """Find task_id that contains this file path."""
        for task_id, repo_path in self.active_tasks.items():
            full_path = Path(repo_path) / file_path
            if full_path.exists():
                return task_id
        return None

    def _merge_section(self, original: str, section: str) -> Optional[str]:
        """Merge a modified section back into the original file.

        Finds the section in the original by matching context lines,
        then replaces it with the modified section.

        Returns:
            Complete file with merged changes, or None if merge failed
        """
        orig_lines = original.split('\n')
        section_lines = section.split('\n')

        if len(section_lines) < 3:
            # Too short to be a section with context, treat as complete file
            return None

        # Try to find where this section belongs by matching first/last lines
        first_line = section_lines[0].strip()
        last_line = section_lines[-1].strip()

        # Find matching range in original
        start_idx = None
        end_idx = None

        for i, line in enumerate(orig_lines):
            if line.strip() == first_line:
                # Check if last line also matches at expected distance
                expected_end = i + len(section_lines) - 1
                if expected_end < len(orig_lines):
                    if orig_lines[expected_end].strip() == last_line:
                        start_idx = i
                        end_idx = expected_end
                        break

        if start_idx is not None and end_idx is not None:
            # Replace the section
            result_lines = orig_lines[:start_idx] + section_lines + orig_lines[end_idx+1:]
            return '\n'.join(result_lines)

        # Fallback: couldn't find matching section
        logging.warning(f"[RepoManager] Could not locate section in original file")
        return None

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
