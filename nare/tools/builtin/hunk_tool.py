"""Hunk-based editing tool for the agent loop."""

from typing import Optional
from .base import Tool, ToolParam, ToolResult
from ..hunks import HunkParser, HunkApplier


def _apply_hunks_sync(
    hunks: str,
    *,
    working_dir: Optional[str] = None,
) -> ToolResult:
    """Apply hunks to files.

    Hunks allow precise line-by-line edits without rewriting entire files.
    This dramatically reduces token consumption (70-90% savings).

    Format:
        <<<<<<< path/to/file.py
        @@ -10,3 +10,4 @@
         def example():
        -    old_line
        +    new_line
        +    added_line
             context_line
        >>>>>>>

    Rules:
    - Lines starting with ' ' are context (unchanged)
    - Lines starting with '-' are removed
    - Lines starting with '+' are added
    - Context lines must match exactly for validation
    - Multiple hunks can be in one call

    Example:
        <<<<<<< web/src/App.vue
        @@ -15,2 +15,3 @@
         <template>
        -  <div class="old">
        +  <div class="new">
        +    <p>Added content</p>
         </template>
        >>>>>>>

    This is much more efficient than read_file + write_file for small changes.
    """
    try:
        hunkset = HunkParser.parse(hunks)

        if not hunkset.hunks:
            return ToolResult(
                ok=False,
                error="No valid hunks found in input. Check format:\n"
                      "<<<<<<< path/to/file\n"
                      "@@ -line,count +line,count @@\n"
                      " context\n"
                      "-removed\n"
                      "+added\n"
                      ">>>>>>>"
            )

        result = HunkApplier.apply_hunkset(hunkset, working_dir)
        return result

    except Exception as e:
        return ToolResult(
            ok=False,
            error=f"Failed to parse or apply hunks: {str(e)}"
        )

async def apply_hunks(
    hunks: str,
    *,
    working_dir: Optional[str] = None,
) -> ToolResult:
    import asyncio
    return await asyncio.to_thread(_apply_hunks_sync, hunks, working_dir=working_dir)
