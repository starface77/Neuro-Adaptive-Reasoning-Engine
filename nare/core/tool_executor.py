"""Tool Executor - Parse and execute XML tool calls from LLM responses.

Aider-inspired approach: after LLM generates response with tool calls,
parse the XML tags and execute real actions.
"""

import re
import os
import subprocess
import logging
from typing import List, Tuple, Optional


class ToolExecutor:
    """Parse and execute XML tool calls from LLM responses."""

    def __init__(self, working_dir: str = "."):
        self.working_dir = working_dir
        self.logger = logging.getLogger("nare.tool_executor")

    def parse_and_execute(self, response: str) -> Tuple[str, List[str]]:
        """Parse XML tags from response and execute actions.

        Args:
            response: LLM response containing XML tool calls

        Returns:
            Tuple of (cleaned_response, list of modified files)
        """
        modified_files = []

        # Parse <read_file> tags
        read_pattern = re.compile(r'<read_file>\s*<path>(.*?)</path>\s*</read_file>', re.DOTALL)
        for match in read_pattern.finditer(response):
            path = match.group(1).strip()
            content = self._read_file(path)
            if content:
                self.logger.info(f"[ToolExecutor] Read file: {path}")

        # Parse <edit_file> tags
        edit_pattern = re.compile(
            r'<edit_file>\s*<path>(.*?)</path>\s*<diff>(.*?)</diff>\s*</edit_file>',
            re.DOTALL
        )
        for match in edit_pattern.finditer(response):
            path = match.group(1).strip()
            diff = match.group(2).strip()
            if self._apply_diff(path, diff):
                modified_files.append(path)
                self.logger.info(f"[ToolExecutor] Edited file: {path}")

        # Parse <write_file> tags
        write_pattern = re.compile(
            r'<write_file>\s*<path>(.*?)</path>\s*<content>(.*?)</content>\s*</write_file>',
            re.DOTALL
        )
        for match in write_pattern.finditer(response):
            path = match.group(1).strip()
            content = match.group(2).strip()
            if self._write_file(path, content):
                modified_files.append(path)
                self.logger.info(f"[ToolExecutor] Wrote file: {path}")

        # Parse <bash_command> tags
        bash_pattern = re.compile(
            r'<bash_command>\s*<command>(.*?)</command>\s*</bash_command>',
            re.DOTALL
        )
        for match in bash_pattern.finditer(response):
            command = match.group(1).strip()
            self._run_command(command)
            self.logger.info(f"[ToolExecutor] Ran command: {command}")

        # Clean response - remove XML tags
        cleaned = response
        cleaned = read_pattern.sub('', cleaned)
        cleaned = edit_pattern.sub('', cleaned)
        cleaned = write_pattern.sub('', cleaned)
        cleaned = bash_pattern.sub('', cleaned)

        return cleaned.strip(), modified_files

    def _read_file(self, path: str) -> Optional[str]:
        """Read file content."""
        full_path = os.path.join(self.working_dir, path)
        try:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        except Exception as e:
            self.logger.warning(f"[ToolExecutor] Failed to read {path}: {e}")
            return None

    def _write_file(self, path: str, content: str) -> bool:
        """Write file content."""
        full_path = os.path.join(self.working_dir, path)
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            self.logger.error(f"[ToolExecutor] Failed to write {path}: {e}")
            return False

    def _apply_diff(self, path: str, diff: str) -> bool:
        """Apply unified diff to file."""
        full_path = os.path.join(self.working_dir, path)
        try:
            # Read current content
            if os.path.exists(full_path):
                with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                    current = f.read()
            else:
                current = ""

            # Parse diff and apply changes
            # Simple implementation - just extract additions
            new_content = self._parse_diff(current, diff)
            if new_content is not None:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                return True
            return False
        except Exception as e:
            self.logger.error(f"[ToolExecutor] Failed to apply diff to {path}: {e}")
            return False

    def _parse_diff(self, current: str, diff: str) -> Optional[str]:
        """Parse unified diff and return new content."""
        # Simple implementation - extract lines starting with +
        lines = current.split('\n')
        diff_lines = diff.split('\n')

        # Find context and apply additions
        for i, line in enumerate(diff_lines):
            if line.startswith('+') and not line.startswith('+++'):
                # Add new line
                new_line = line[1:]  # Remove +
                lines.append(new_line)

        return '\n'.join(lines)

    def _run_command(self, command: str) -> Optional[str]:
        """Run shell command."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return result.stdout
            else:
                self.logger.warning(f"[ToolExecutor] Command failed: {result.stderr}")
                return None
        except Exception as e:
            self.logger.error(f"[ToolExecutor] Failed to run command: {e}")
            return None
