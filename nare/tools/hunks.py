"""Hunk-based file editing system for efficient code modifications.

Hunks allow the AI to modify specific lines of code instead of rewriting
entire files, reducing token consumption by 70-90% and providing better
change control.

Format:
    <<<<<<< path/to/file.py
    @@ -10,3 +10,4 @@
     def example():
    -    old_line
    +    new_line
    +    added_line
         context_line
    >>>>>>>

The format is inspired by unified diff but simplified for LLM generation.
"""

from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Avoid circular import - ToolResult is only used for type hints in apply_hunkset
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .builtin.base import ToolResult


@dataclass
class HunkLine:
    """A single line in a hunk."""
    type: str  # ' ' (context), '-' (remove), '+' (add)
    content: str
    line_num: Optional[int] = None  # Original line number for context/remove


@dataclass
class Hunk:
    """A single hunk representing a localized change."""
    file_path: str
    start_line: int  # 1-indexed
    old_count: int
    new_count: int
    lines: List[HunkLine]

    def __str__(self) -> str:
        """Render hunk in unified diff format."""
        result = [f"@@ -{self.start_line},{self.old_count} +{self.start_line},{self.new_count} @@"]
        for line in self.lines:
            result.append(f"{line.type}{line.content}")
        return "\n".join(result)


@dataclass
class HunkSet:
    """A collection of hunks for one or more files."""
    hunks: List[Hunk]

    def group_by_file(self) -> dict[str, List[Hunk]]:
        """Group hunks by file path."""
        result = {}
        for hunk in self.hunks:
            if hunk.file_path not in result:
                result[hunk.file_path] = []
            result[hunk.file_path].append(hunk)
        return result


class HunkParser:
    """Parse hunk format from LLM output."""

    HUNK_START = re.compile(r'^<{7}\s+(.+)$')
    HUNK_END = re.compile(r'^>{7}$')
    HUNK_HEADER = re.compile(r'^@@\s+-(\d+),(\d+)\s+\+(\d+),(\d+)\s+@@$')

    @classmethod
    def parse(cls, text: str) -> HunkSet:
        """Parse hunks from text.

        Format:
            <<<<<<< path/to/file.py
            @@ -10,3 +10,4 @@
             context
            -old
            +new
            >>>>>>>
        """
        hunks = []
        lines = text.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i].rstrip()

            # Look for hunk start
            match = cls.HUNK_START.match(line)
            if not match:
                i += 1
                continue

            file_path = match.group(1).strip()
            i += 1

            # Parse hunks for this file until we hit >>>>>>>
            while i < len(lines):
                line = lines[i].rstrip()

                if cls.HUNK_END.match(line):
                    i += 1
                    break

                # Parse hunk header
                header_match = cls.HUNK_HEADER.match(line)
                if not header_match:
                    i += 1
                    continue

                start_line = int(header_match.group(1))
                old_count = int(header_match.group(2))
                new_start = int(header_match.group(3))
                new_count = int(header_match.group(4))

                i += 1

                # Parse hunk lines
                hunk_lines = []
                while i < len(lines):
                    line = lines[i]

                    # Check for next hunk or end
                    if cls.HUNK_HEADER.match(line.rstrip()) or cls.HUNK_END.match(line.rstrip()):
                        break

                    if len(line) == 0:
                        i += 1
                        continue

                    line_type = line[0]
                    if line_type in (' ', '-', '+'):
                        # Content is everything after the first character
                        content = line[1:].rstrip('\n\r')
                        hunk_lines.append(HunkLine(type=line_type, content=content))

                    i += 1

                hunks.append(Hunk(
                    file_path=file_path,
                    start_line=start_line,
                    old_count=old_count,
                    new_count=new_count,
                    lines=hunk_lines
                ))

        return HunkSet(hunks=hunks)


class HunkApplier:
    """Apply hunks to files."""

    @staticmethod
    def apply_hunk(file_path: str, hunk: Hunk, working_dir: Optional[str] = None) -> Tuple[bool, str]:
        """Apply a single hunk to a file.

        Returns:
            (success, message)
        """
        if working_dir:
            full_path = os.path.join(working_dir, file_path)
        else:
            full_path = file_path

        if not os.path.exists(full_path):
            return False, f"File not found: {file_path}"

        # Read original file
        with open(full_path, 'r', encoding='utf-8') as f:
            original_lines = f.readlines()

        # Validate hunk can be applied
        valid, msg = HunkApplier._validate_hunk(original_lines, hunk)
        if not valid:
            return False, msg

        # Apply hunk
        new_lines = HunkApplier._apply_hunk_to_lines(original_lines, hunk)

        # Write back
        with open(full_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

        return True, f"Applied hunk to {file_path}"

    @staticmethod
    def _validate_hunk(lines: List[str], hunk: Hunk) -> Tuple[bool, str]:
        """Validate that hunk can be applied to lines."""
        # Check bounds
        if hunk.start_line < 1 or hunk.start_line > len(lines) + 1:
            return False, f"Hunk start line {hunk.start_line} out of bounds (file has {len(lines)} lines)"

        # Check context matches
        line_idx = hunk.start_line - 1  # Convert to 0-indexed
        for hunk_line in hunk.lines:
            if hunk_line.type == ' ':  # Context line
                if line_idx >= len(lines):
                    return False, f"Context line extends beyond file end"

                expected = hunk_line.content.rstrip()
                actual = lines[line_idx].rstrip()

                if expected != actual:
                    return False, f"Context mismatch at line {line_idx + 1}: expected '{expected}', got '{actual}'"

                line_idx += 1

            elif hunk_line.type == '-':  # Remove line
                if line_idx >= len(lines):
                    return False, f"Remove line extends beyond file end"

                expected = hunk_line.content.rstrip()
                actual = lines[line_idx].rstrip()

                if expected != actual:
                    return False, f"Remove line mismatch at line {line_idx + 1}: expected '{expected}', got '{actual}'"

                line_idx += 1

            # '+' lines don't consume original lines

        return True, "OK"

    @staticmethod
    def _apply_hunk_to_lines(lines: List[str], hunk: Hunk) -> List[str]:
        """Apply hunk to lines and return new lines."""
        result = []

        # Copy lines before hunk
        result.extend(lines[:hunk.start_line - 1])

        # Apply hunk
        line_idx = hunk.start_line - 1
        for hunk_line in hunk.lines:
            if hunk_line.type == ' ':  # Context - keep
                result.append(lines[line_idx])
                line_idx += 1
            elif hunk_line.type == '-':  # Remove - skip
                line_idx += 1
            elif hunk_line.type == '+':  # Add
                # Preserve original line ending style
                content = hunk_line.content
                if line_idx < len(lines) and not content.endswith('\n'):
                    if lines[line_idx].endswith('\n'):
                        content += '\n'
                result.append(content)

        # Copy remaining lines
        result.extend(lines[line_idx:])

        return result

    @staticmethod
    def apply_hunkset(hunkset: HunkSet, working_dir: Optional[str] = None):
        """Apply all hunks in a hunkset.

        Returns:
            dict with ok, summary, body, error keys (ToolResult-compatible)
        """
        from .builtin.base import ToolResult
        grouped = hunkset.group_by_file()
        results = []
        errors = []

        for file_path, hunks in grouped.items():
            # Sort hunks by start line (apply from bottom to top to avoid offset issues)
            hunks_sorted = sorted(hunks, key=lambda h: h.start_line, reverse=True)

            for hunk in hunks_sorted:
                success, msg = HunkApplier.apply_hunk(file_path, hunk, working_dir)
                if success:
                    results.append(msg)
                else:
                    errors.append(msg)

        if errors:
            return ToolResult(
                ok=False,
                error="\n".join(errors),
                summary=f"Failed to apply {len(errors)} hunk(s)"
            )

        return ToolResult(
            ok=True,
            summary=f"Applied {len(results)} hunk(s) to {len(grouped)} file(s)",
            body="\n".join(results)
        )


def generate_hunk_from_edit(file_path: str, old_text: str, new_text: str,
                            working_dir: Optional[str] = None, context_lines: int = 3) -> str:
    """Generate a hunk from old/new text comparison.

    This is useful for converting traditional edit operations to hunk format.
    """
    if working_dir:
        full_path = os.path.join(working_dir, file_path)
    else:
        full_path = file_path

    if not os.path.exists(full_path):
        return f"<<<<<<< {file_path}\n@@ -1,0 +1,{len(new_text.splitlines())} @@\n" + \
               "\n".join(f"+{line}" for line in new_text.splitlines()) + "\n>>>>>>>"

    with open(full_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the old text in the file
    if old_text not in content:
        return None

    lines = content.splitlines(keepends=True)
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)

    # Find start position
    start_idx = None
    for i in range(len(lines) - len(old_lines) + 1):
        if lines[i:i+len(old_lines)] == old_lines:
            start_idx = i
            break

    if start_idx is None:
        return None

    # Build hunk
    hunk_lines = []

    # Add context before
    ctx_start = max(0, start_idx - context_lines)
    for i in range(ctx_start, start_idx):
        hunk_lines.append(f" {lines[i]}")

    # Add removals
    for line in old_lines:
        hunk_lines.append(f"-{line}")

    # Add additions
    for line in new_lines:
        hunk_lines.append(f"+{line}")

    # Add context after
    ctx_end = min(len(lines), start_idx + len(old_lines) + context_lines)
    for i in range(start_idx + len(old_lines), ctx_end):
        hunk_lines.append(f" {lines[i]}")

    start_line = ctx_start + 1  # 1-indexed
    old_count = (start_idx - ctx_start) + len(old_lines) + (ctx_end - start_idx - len(old_lines))
    new_count = (start_idx - ctx_start) + len(new_lines) + (ctx_end - start_idx - len(old_lines))

    result = f"<<<<<<< {file_path}\n"
    result += f"@@ -{start_line},{old_count} +{start_line},{new_count} @@\n"
    result += "".join(hunk_lines)
    result += ">>>>>>>\n"

    return result
