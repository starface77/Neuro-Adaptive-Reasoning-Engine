"""Smart code truncation for CLI output."""

import re
from typing import Tuple

def truncate_code_blocks(text: str, max_lines: int = 40) -> Tuple[str, bool]:
    """Truncate long code blocks intelligently.

    Args:
        text: Text containing code blocks
        max_lines: Maximum lines to show per code block

    Returns:
        (truncated_text, was_truncated)
    """

    code_block_pattern = r'```(\w+)?\n(.*?)```'

    was_truncated = False
    result = []
    last_end = 0

    for match in re.finditer(code_block_pattern, text, re.DOTALL):

        result.append(text[last_end:match.start()])

        lang = match.group(1) or ''
        code = match.group(2)
        lines = code.split('\n')

        if len(lines) > max_lines:

            truncated_lines = lines[:max_lines]
            result.append(f'```{lang}\n')
            result.append('\n'.join(truncated_lines))
            result.append(f'\n... ({len(lines) - max_lines} more lines)\n```')
            was_truncated = True
        else:

            result.append(match.group(0))

        last_end = match.end()

    result.append(text[last_end:])

    return ''.join(result), was_truncated

def smart_truncate_answer(answer: str, max_lines: int = 50) -> Tuple[str, str]:
    """Smart truncation of answer with hint.

    Args:
        answer: The answer text
        max_lines: Maximum lines to show

    Returns:
        (truncated_answer, hint_message)
    """

    answer, code_truncated = truncate_code_blocks(answer, max_lines=40)

    lines = answer.split('\n')

    if len(lines) <= max_lines and not code_truncated:
        return answer, ""

    if len(lines) > max_lines:
        truncated = '\n'.join(lines[:max_lines])
        remaining = len(lines) - max_lines
        hint = f"\n[dim]... {remaining} more lines. Files saved to disk.[/dim]"
        return truncated, hint

    if code_truncated:
        hint = "\n[dim]Long code blocks truncated. Files saved to disk.[/dim]"
        return answer, hint

    return answer, ""
