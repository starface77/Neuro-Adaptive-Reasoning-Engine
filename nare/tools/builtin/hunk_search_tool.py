"""Hunk search tool - integrated search for efficient hunk application."""

from typing import Optional
from ..hunk_search import HunkSearch, format_context_for_display, format_context_for_hunk
from .base import ToolResult


def _search_for_hunk_sync(
    pattern: str,
    file_path: str,
    context_lines: int = 3,
    is_regex: bool = False,
    working_dir: Optional[str] = None
) -> ToolResult:
    """Search for a pattern and return context for hunk application.

    This tool helps the agent find the exact location and context needed
    to apply hunks without reading entire files.

    Args:
        pattern: Text or regex pattern to search for
        file_path: Path to file (relative to working directory)
        context_lines: Number of context lines before/after match (default: 3)
        is_regex: Whether pattern is a regex (default: False)
        working_dir: Working directory for relative paths

    Returns:
        ToolResult with context window and hunk template
    """
    searcher = HunkSearch(working_dir)

    # Find pattern and get context
    result = searcher.prepare_hunk_context(
        file_path=file_path,
        search_pattern=pattern,
        context_lines=context_lines,
        is_regex=is_regex
    )

    if not result:
        return ToolResult(
            ok=False,
            error=f"Pattern '{pattern}' not found in {file_path}",
            summary=f"Not found in {file_path}"
        )

    context, match = result

    # Format for display
    display = format_context_for_display(context)
    hunk_template = format_context_for_hunk(context)

    summary = (
        f"Found at line {match.line_number} in {file_path} "
        f"(showing lines {context.start_line}-{context.end_line})"
    )

    body = f"""Match found at line {match.line_number}:
{match.line_content}

Context window:
{display}

Hunk template (modify -/+ lines as needed):
{hunk_template}
"""

    return ToolResult(
        ok=True,
        summary=summary,
        body=body,
        meta={
            "file_path": file_path,
            "line_number": match.line_number,
            "start_line": context.start_line,
            "end_line": context.end_line,
            "context_lines": context.lines
        }
    )


async def search_for_hunk(
    pattern: str,
    file_path: str,
    context_lines: int = 3,
    is_regex: bool = False,
    working_dir: Optional[str] = None
) -> ToolResult:
    import asyncio
    return await asyncio.to_thread(_search_for_hunk_sync, pattern, file_path, context_lines, is_regex, working_dir)


def _find_function_for_hunk_sync(
    function_name: str,
    file_path: str,
    context_lines: int = 5,
    working_dir: Optional[str] = None
) -> ToolResult:
    """Find a function definition and return context for hunk application.

    Args:
        function_name: Name of the function to find
        file_path: Path to file (relative to working directory)
        context_lines: Number of context lines (default: 5 for functions)
        working_dir: Working directory for relative paths

    Returns:
        ToolResult with function context and hunk template
    """
    searcher = HunkSearch(working_dir)

    # Find function
    match = searcher.find_function(function_name, file_path)
    if not match:
        return ToolResult(
            ok=False,
            error=f"Function '{function_name}' not found in {file_path}",
            summary=f"Function not found in {file_path}"
        )

    # Get context
    context = searcher.get_context_window(
        file_path=file_path,
        line_number=match.line_number,
        context_lines=context_lines
    )

    if not context:
        return ToolResult(
            ok=False,
            error=f"Could not read context around line {match.line_number}",
            summary="Context read failed"
        )

    # Format for display
    display = format_context_for_display(context)
    hunk_template = format_context_for_hunk(context)

    summary = (
        f"Found function '{function_name}' at line {match.line_number} "
        f"(showing lines {context.start_line}-{context.end_line})"
    )

    body = f"""Function definition at line {match.line_number}:
{match.line_content}

Context window:
{display}

Hunk template (modify -/+ lines as needed):
{hunk_template}
"""

    return ToolResult(
        ok=True,
        summary=summary,
        body=body,
        meta={
            "file_path": file_path,
            "function_name": function_name,
            "line_number": match.line_number,
            "start_line": context.start_line,
            "end_line": context.end_line,
            "context_lines": context.lines
        }
    )


async def find_function_for_hunk(
    function_name: str,
    file_path: str,
    context_lines: int = 5,
    working_dir: Optional[str] = None
) -> ToolResult:
    import asyncio
    return await asyncio.to_thread(_find_function_for_hunk_sync, function_name, file_path, context_lines, working_dir)


def _find_class_for_hunk_sync(
    class_name: str,
    file_path: str,
    context_lines: int = 5,
    working_dir: Optional[str] = None
) -> ToolResult:
    """Find a class definition and return context for hunk application.

    Args:
        class_name: Name of the class to find
        file_path: Path to file (relative to working directory)
        context_lines: Number of context lines (default: 5 for classes)
        working_dir: Working directory for relative paths

    Returns:
        ToolResult with class context and hunk template
    """
    searcher = HunkSearch(working_dir)

    # Find class
    match = searcher.find_class(class_name, file_path)
    if not match:
        return ToolResult(
            ok=False,
            error=f"Class '{class_name}' not found in {file_path}",
            summary=f"Class not found in {file_path}"
        )

    # Get context
    context = searcher.get_context_window(
        file_path=file_path,
        line_number=match.line_number,
        context_lines=context_lines
    )

    if not context:
        return ToolResult(
            ok=False,
            error=f"Could not read context around line {match.line_number}",
            summary="Context read failed"
        )

    # Format for display
    display = format_context_for_display(context)
    hunk_template = format_context_for_hunk(context)

    summary = (
        f"Found class '{class_name}' at line {match.line_number} "
        f"(showing lines {context.start_line}-{context.end_line})"
    )

    body = f"""Class definition at line {match.line_number}:
{match.line_content}

Context window:
{display}

Hunk template (modify -/+ lines as needed):
{hunk_template}
"""

    return ToolResult(
        ok=True,
        summary=summary,
        body=body,
        meta={
            "file_path": file_path,
            "class_name": class_name,
            "line_number": match.line_number,
            "start_line": context.start_line,
            "end_line": context.end_line,
            "context_lines": context.lines
        }
    )

async def find_class_for_hunk(
    class_name: str,
    file_path: str,
    context_lines: int = 5,
    working_dir: Optional[str] = None
) -> ToolResult:
    import asyncio
    return await asyncio.to_thread(_find_class_for_hunk_sync, class_name, file_path, context_lines, working_dir)
