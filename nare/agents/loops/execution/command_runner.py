"""Command execution utilities."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Optional

from nare.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CommandResult:
    """Result of command execution."""
    stdout: str
    stderr: str
    returncode: int
    success: bool
    
    @property
    def output(self) -> str:
        """Combined output."""
        return self.stdout + self.stderr


def run_command(
    command: str,
    timeout: int = 30,
    cwd: Optional[str] = None,
    shell: bool = True
) -> CommandResult:
    """
    Run a shell command and return the result.
    
    Args:
        command: Command to execute
        timeout: Timeout in seconds
        cwd: Working directory
        shell: Whether to use shell
    
    Returns:
        CommandResult with execution details
    """
    try:
        logger.debug(f"Executing command: {command}")
        
        process = subprocess.run(
            command,
            shell=shell,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd
        )
        
        result = CommandResult(
            stdout=process.stdout,
            stderr=process.stderr,
            returncode=process.returncode,
            success=process.returncode == 0
        )
        
        if not result.success:
            logger.warning(f"Command failed with code {result.returncode}: {command}")
        
        return result
    
    except subprocess.TimeoutExpired:
        logger.error(f"Command timeout after {timeout}s: {command}")
        return CommandResult(
            stdout="",
            stderr=f"Command timed out after {timeout} seconds",
            returncode=-1,
            success=False
        )
    
    except Exception as e:
        logger.error(f"Command execution error: {command}", exc_info=True)
        return CommandResult(
            stdout="",
            stderr=str(e),
            returncode=-1,
            success=False
        )