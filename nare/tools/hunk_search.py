"""Hunk search module - finds context for efficient hunks application.

This module helps the agent find the exact line numbers needed for hunks
without reading entire files, saving 90%+ tokens.
"""

import re
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Result of searching for a pattern in a file."""
    file_path: str
    line_number: int
    line_content: str
    match_start: int
    match_end: int


@dataclass
class ContextWindow:
    """Context window around a match for hunk application."""
    file_path: str
    start_line: int
    end_line: int
    lines: List[str]
    target_line: int
    target_content: str


class HunkSearch:
    """Search engine for finding context needed for hunk application."""

    def __init__(self, working_dir: Optional[str] = None):
        """Initialize hunk search.

        Args:
            working_dir: Working directory for relative paths (default: cwd)
        """
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()

    def find_pattern(
        self,
        pattern: str,
        file_path: str,
        is_regex: bool = False,
        case_sensitive: bool = True
    ) -> List[SearchResult]:
        """Find all occurrences of a pattern in a file.

        Args:
            pattern: Text or regex pattern to search for
            file_path: Path to file (relative to working_dir)
            is_regex: Whether pattern is a regex (default: False)
            case_sensitive: Whether search is case-sensitive (default: True)

        Returns:
            List of SearchResult objects with line numbers and content
        """
        full_path = self.working_dir / file_path
        if not full_path.exists():
            return []

        results = []
        flags = 0 if case_sensitive else re.IGNORECASE

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, start=1):
                    if is_regex:
                        match = re.search(pattern, line, flags)
                        if match:
                            results.append(SearchResult(
                                file_path=file_path,
                                line_number=line_num,
                                line_content=line.rstrip('\n'),
                                match_start=match.start(),
                                match_end=match.end()
                            ))
                    else:
                        # Simple text search
                        search_line = line if case_sensitive else line.lower()
                        search_pattern = pattern if case_sensitive else pattern.lower()

                        pos = search_line.find(search_pattern)
                        if pos != -1:
                            results.append(SearchResult(
                                file_path=file_path,
                                line_number=line_num,
                                line_content=line.rstrip('\n'),
                                match_start=pos,
                                match_end=pos + len(pattern)
                            ))
        except (IOError, UnicodeDecodeError):
            return []

        return results

    def get_context_window(
        self,
        file_path: str,
        line_number: int,
        context_lines: int = 3
    ) -> Optional[ContextWindow]:
        """Get context window around a specific line.

        Args:
            file_path: Path to file (relative to working_dir)
            line_number: Target line number (1-indexed)
            context_lines: Number of context lines before/after (default: 3)

        Returns:
            ContextWindow with lines around target, or None if file not found
        """
        full_path = self.working_dir / file_path
        if not full_path.exists():
            return None

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()

            if line_number < 1 or line_number > len(all_lines):
                return None

            # Calculate window bounds
            start_line = max(1, line_number - context_lines)
            end_line = min(len(all_lines), line_number + context_lines)

            # Extract lines (convert to 0-indexed for slicing)
            window_lines = [
                line.rstrip('\n')
                for line in all_lines[start_line - 1:end_line]
            ]

            return ContextWindow(
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                lines=window_lines,
                target_line=line_number,
                target_content=all_lines[line_number - 1].rstrip('\n')
            )
        except (IOError, UnicodeDecodeError):
            return None

    def find_function(
        self,
        function_name: str,
        file_path: str
    ) -> Optional[SearchResult]:
        """Find a function definition in a file.

        Args:
            function_name: Name of the function to find
            file_path: Path to file (relative to working_dir)

        Returns:
            SearchResult for the function definition, or None if not found
        """
        # Try common function definition patterns
        patterns = [
            rf'^\s*def\s+{re.escape(function_name)}\s*\(',  # Python
            rf'^\s*function\s+{re.escape(function_name)}\s*\(',  # JavaScript
            rf'^\s*async\s+def\s+{re.escape(function_name)}\s*\(',  # Python async
            rf'^\s*const\s+{re.escape(function_name)}\s*=',  # JS const
            rf'^\s*let\s+{re.escape(function_name)}\s*=',  # JS let
        ]

        for pattern in patterns:
            results = self.find_pattern(pattern, file_path, is_regex=True)
            if results:
                return results[0]  # Return first match

        return None

    def find_class(
        self,
        class_name: str,
        file_path: str
    ) -> Optional[SearchResult]:
        """Find a class definition in a file.

        Args:
            class_name: Name of the class to find
            file_path: Path to file (relative to working_dir)

        Returns:
            SearchResult for the class definition, or None if not found
        """
        # Try common class definition patterns
        patterns = [
            rf'^\s*class\s+{re.escape(class_name)}\s*[:\(]',  # Python/JS
            rf'^\s*export\s+class\s+{re.escape(class_name)}\s*[:\(]',  # JS export
        ]

        for pattern in patterns:
            results = self.find_pattern(pattern, file_path, is_regex=True)
            if results:
                return results[0]

        return None

    def prepare_hunk_context(
        self,
        file_path: str,
        search_pattern: str,
        context_lines: int = 3,
        is_regex: bool = False
    ) -> Optional[Tuple[ContextWindow, SearchResult]]:
        """Find pattern and prepare context for hunk application.

        This is the main method agents should use - it combines search
        and context extraction in one call.

        Args:
            file_path: Path to file (relative to working_dir)
            search_pattern: Pattern to search for
            context_lines: Number of context lines (default: 3)
            is_regex: Whether pattern is regex (default: False)

        Returns:
            Tuple of (ContextWindow, SearchResult) or None if not found
        """
        # Find the pattern
        results = self.find_pattern(search_pattern, file_path, is_regex)
        if not results:
            return None

        # Get context around first match
        first_match = results[0]
        context = self.get_context_window(
            file_path,
            first_match.line_number,
            context_lines
        )

        if not context:
            return None

        return (context, first_match)


def format_context_for_display(context: ContextWindow) -> str:
    """Format context window for display to user or agent.

    Args:
        context: ContextWindow to format

    Returns:
        Formatted string with line numbers
    """
    lines = []
    for i, line in enumerate(context.lines):
        line_num = context.start_line + i
        marker = "→" if line_num == context.target_line else " "
        lines.append(f"{marker} {line_num:4d} | {line}")

    return "\n".join(lines)


def format_context_for_hunk(context: ContextWindow) -> str:
    """Format context window as hunk template.

    Args:
        context: ContextWindow to format

    Returns:
        Hunk template string that agent can fill in
    """
    lines = []
    lines.append(f"<<<<<<< {context.file_path}")

    # Calculate hunk header
    old_count = len(context.lines)
    new_count = old_count  # Agent will adjust this
    lines.append(f"@@ -{context.start_line},{old_count} +{context.start_line},{new_count} @@")

    # Add context lines (agent will mark changes with -/+)
    for line in context.lines:
        lines.append(f" {line}")

    lines.append(">>>>>>>")

    return "\n".join(lines)
