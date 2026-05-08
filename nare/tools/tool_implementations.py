"""Concrete tool implementations for file and system operations."""

import os
import subprocess
from pathlib import Path
from typing import Dict, Any


def read_file_tool(filepath: str) -> Dict[str, Any]:
    """Read contents of a file."""
    try:
        path = Path(filepath)
        if not path.exists():
            return {"success": False, "error": f"File not found: {filepath}"}
        
        if not path.is_file():
            return {"success": False, "error": f"Not a file: {filepath}"}
        
        content = path.read_text(encoding='utf-8')
        return {
            "success": True,
            "content": content,
            "size": len(content),
            "lines": content.count('\n') + 1
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def write_file_tool(filepath: str, content: str, mode: str = "w") -> Dict[str, Any]:
    """Write content to a file."""
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if mode == "a":
            path.write_text(path.read_text() + content, encoding='utf-8')
        else:
            path.write_text(content, encoding='utf-8')
        
        return {
            "success": True,
            "filepath": str(path),
            "size": len(content)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def list_directory_tool(directory: str = ".", pattern: str = "*") -> Dict[str, Any]:
    """List contents of a directory."""
    try:
        path = Path(directory)
        if not path.exists():
            return {"success": False, "error": f"Directory not found: {directory}"}
        
        if not path.is_dir():
            return {"success": False, "error": f"Not a directory: {directory}"}
        
        items = list(path.glob(pattern))
        files = [str(f.relative_to(path)) for f in items if f.is_file()]
        dirs = [str(d.relative_to(path)) for d in items if d.is_dir()]
        
        return {
            "success": True,
            "files": files,
            "directories": dirs,
            "total": len(items)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def search_files_tool(
    directory: str = ".",
    pattern: str = "*",
    content: str = None
) -> Dict[str, Any]:
    """Search for files by name or content."""
    try:
        path = Path(directory)
        if not path.exists():
            return {"success": False, "error": f"Directory not found: {directory}"}
        
        matches = []
        for file_path in path.rglob(pattern):
            if file_path.is_file():
                if content:
                    try:
                        file_content = file_path.read_text(encoding='utf-8')
                        if content in file_content:
                            matches.append(str(file_path.relative_to(path)))
                    except:
                        continue
                else:
                    matches.append(str(file_path.relative_to(path)))
        
        return {
            "success": True,
            "matches": matches,
            "count": len(matches)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def execute_command_tool(
    command: str,
    cwd: str = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """Execute a shell command."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Command timed out after {timeout}s"}
    except Exception as e:
        return {"success": False, "error": str(e)}