"""
Tool executor — Parses and executes tool calls from LLM responses.

The LLM generates tool calls in its response, and this module
extracts and executes them.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from nare.tools.file_tools import read_file, create_file, edit_file, list_files

KNOWN_TOOLS = ('create_file', 'edit_file', 'read_file', 'list_files', 'list_dir', 'write_file')


def _parse_json_tool(data: dict) -> Optional[Dict[str, Any]]:
    """Convert a JSON tool call dict into standardized {tool, args} format."""
    tool_name = data.get('name') or data.get('tool', '')
    if tool_name not in KNOWN_TOOLS:
        return None

    args = data.get('args', {})
    if not isinstance(args, dict):
        return None

    if tool_name == 'read_file':
        return {'tool': 'read_file', 'args': [args.get('path', args.get('filepath', ''))]}
    elif tool_name in ('create_file', 'write_file'):
        return {'tool': 'create_file', 'args': [args.get('path', args.get('filepath', '')), args.get('content', '')]}
    elif tool_name == 'edit_file':
        return {'tool': 'edit_file', 'args': [args.get('path', args.get('filepath', '')), args.get('old', ''), args.get('new', '')]}
    elif tool_name in ('list_files', 'list_dir'):
        return {'tool': 'list_files', 'args': [args.get('path', args.get('directory', '.'))]}
    return None


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Parse tool calls from LLM response.

    Supports formats:
    1. XML tags: <create_file>path/content</create_file>
    2. Function calls: create_file("path", "content")
    3. JSON in <tool_call> tags: <tool_call>{"name": "read_file", ...}</tool_call>
    4. Bare JSON: {"name": "read_file", "args": {"path": "..."}}

    Returns list of tool call dicts with 'tool' and 'args' keys.
    """
    tool_calls = []

    # Format 3: <tool_call> JSON </tool_call>
    for match in re.finditer(r'<tool_call\s*>(.*?)</tool_call\s*>', text, re.DOTALL):
        try:
            parsed = _parse_json_tool(json.loads(match.group(1).strip()))
            if parsed:
                tool_calls.append(parsed)
        except (json.JSONDecodeError, AttributeError):
            pass

    if tool_calls:
        return tool_calls

    # Format 4: Bare JSON tool calls
    json_pat = r'\{\s*"name"\s*:\s*"(' + '|'.join(KNOWN_TOOLS) + r')"\s*,\s*"args"\s*:\s*\{[^}]*\}\s*\}'
    for match in re.finditer(json_pat, text, re.DOTALL):
        try:
            parsed = _parse_json_tool(json.loads(match.group(0)))
            if parsed:
                tool_calls.append(parsed)
        except (json.JSONDecodeError, AttributeError):
            pass

    if tool_calls:
        return tool_calls

    # Format 1: XML tags
    xml_pattern = r'<(create_file|edit_file|read_file|list_files)>\s*(.*?)\s*</\1>'
    xml_matches = re.finditer(xml_pattern, text, re.DOTALL)

    for match in xml_matches:
        tool_name = match.group(1)
        content = match.group(2).strip()

        if tool_name == "create_file":
            path_match = re.search(r'<path>\s*(.*?)\s*</path>', content, re.DOTALL)
            content_match = re.search(r'<content>\s*(.*?)\s*</content>', content, re.DOTALL)

            if path_match and content_match:
                filepath = path_match.group(1).strip()
                file_content = content_match.group(1).strip()
            else:
                lines = content.split('\n', 1)
                if len(lines) >= 2:
                    filepath = lines[0].strip()
                    filepath = re.sub(r'</?path>', '', filepath)
                    file_content = lines[1].strip()
                    file_content = re.sub(r'^<content>\s*', '', file_content)
                    file_content = re.sub(r'\s*</content>$', '', file_content)
                else:
                    continue

            tool_calls.append({'tool': tool_name, 'args': [filepath, file_content]})

        elif tool_name == "edit_file":
            path_match = re.search(r'<path>\s*(.*?)\s*</path>', content, re.DOTALL)
            old_match = re.search(r'<old>\s*(.*?)\s*</old>', content, re.DOTALL)
            new_match = re.search(r'<new>\s*(.*?)\s*</new>', content, re.DOTALL)

            if path_match and old_match and new_match:
                filepath = path_match.group(1).strip()
                target = old_match.group(1)
                replacement = new_match.group(1)
            else:
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    filepath = parts[0].strip()
                    target = parts[1].strip()
                    replacement = parts[2].strip()
                else:
                    lines = content.split('\n', 2)
                    if len(lines) >= 3:
                        filepath = lines[0].strip()
                        target = lines[1].strip()
                        replacement = lines[2].strip()
                    else:
                        continue

            filepath = re.sub(r'</?path>', '', filepath)
            tool_calls.append({'tool': tool_name, 'args': [filepath, target, replacement]})

        elif tool_name == "read_file":
            filepath = content.strip()
            filepath = re.sub(r'</?path>', '', filepath)
            tool_calls.append({'tool': tool_name, 'args': [filepath]})

        elif tool_name == "list_files":
            lines = content.split('\n', 1)
            args = [lines[0].strip()]
            if len(lines) > 1:
                args.append(lines[1].strip())
            tool_calls.append({'tool': tool_name, 'args': args})

    if tool_calls:
        return tool_calls

    # Format 2: Function call syntax
    pattern = r'(create_file|edit_file|read_file|list_files)\s*\((.*?)\)(?=\s*(?:create_file|edit_file|read_file|list_files|\n\n|$))'
    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        tool_name = match.group(1)
        args_str = match.group(2)

        try:
            import ast
            expr = f"{tool_name}({args_str})"
            tree = ast.parse(expr)
            if isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Call):
                call = tree.body[0].value
                args = []
                for arg in call.args:
                    if isinstance(arg, ast.Constant):
                        args.append(arg.value)
                    elif isinstance(arg, ast.Str):
                        args.append(arg.s)
                    else:
                        args.append(ast.literal_eval(arg))
                tool_calls.append({'tool': tool_name, 'args': args})
                continue
        except Exception as e:
            logging.warning(f"Failed to ast.parse tool call {tool_name}: {e}")

        tool_calls.append({'tool': tool_name, 'args': []})

    return tool_calls


def execute_tool_call(tool_name: str, args: List[str], stream_callback: Optional[Callable] = None, working_dir: str = ".") -> Optional[str]:
    """Execute a single tool call."""
    try:
        if tool_name == "create_file":
            if len(args) < 2:
                return "Error: create_file requires 2 arguments (filepath, content)"
            filepath = args[0]
            content = args[1]
            if not os.path.isabs(filepath):
                filepath = os.path.join(working_dir, filepath)
            if stream_callback:
                stream_callback('start', filepath, None)
            create_file(filepath, content,
                       stream_callback=lambda chunk: stream_callback('chunk', filepath, chunk) if stream_callback else None)
            if stream_callback:
                stream_callback('finish', filepath, None)
            return f"Created {filepath}"

        elif tool_name == "edit_file":
            if len(args) < 3:
                return "Error: edit_file requires 3 arguments (filepath, target, replacement)"
            filepath, target, replacement = args[0], args[1], args[2]
            if not os.path.isabs(filepath):
                filepath = os.path.join(working_dir, filepath)
            if stream_callback:
                stream_callback('start', filepath, None)
            edit_file(filepath, target, replacement,
                     stream_callback=lambda chunk: stream_callback('chunk', filepath, chunk) if stream_callback else None)
            if stream_callback:
                stream_callback('finish', filepath, None)
            return f"Edited {filepath}"

        elif tool_name == "read_file":
            if len(args) < 1:
                return "Error: read_file requires at least 1 argument (filepath)"
            filepath = args[0]
            if not os.path.isabs(filepath):
                filepath = os.path.join(working_dir, filepath)
            start_line = int(args[1]) if len(args) > 1 else None
            end_line = int(args[2]) if len(args) > 2 else None
            content = read_file(filepath, start_line, end_line)

            # Show more content for read operations (up to 3000 chars)
            if len(content) > 3000:
                return f"Read {filepath} ({len(content)} chars, showing first 3000):\n```\n{content[:3000]}\n...\n```"
            else:
                return f"Read {filepath}:\n```\n{content}\n```"

        elif tool_name == "list_files":
            directory = args[0] if len(args) > 0 else "."
            if not os.path.isabs(directory):
                directory = os.path.join(working_dir, directory)
            pattern = args[1] if len(args) > 1 else "*"
            files = list_files(directory, pattern)
            return f"Found {len(files)} files:\n" + "\n".join(files[:20])

        else:
            return f"Error: Unknown tool {tool_name}"

    except Exception as e:
        logging.error(f"Tool execution failed: {e}")
        return f"Error: {e}"


def execute_tools_from_response(response: str, stream_callback: Optional[Callable] = None, working_dir: str = ".") -> List[str]:
    """Parse and execute all tool calls from LLM response."""
    tool_calls = parse_tool_calls(response)
    results = []
    for call in tool_calls:
        result = execute_tool_call(call['tool'], call['args'], stream_callback, working_dir)
        if result:
            results.append(result)
            logging.info(f"Executed {call['tool']}: {result}")
    return results


def clean_tool_calls_from_text(text: str) -> str:
    """Remove all forms of tool calls from text for clean display."""
    cleaned = text
    cleaned = re.sub(r'<(create_file|edit_file|read_file|list_files)>.*?</\1>', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'<tool_call\s*>.*?</tool_call\s*>', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(
        r'\{\s*"name"\s*:\s*"(?:' + '|'.join(KNOWN_TOOLS) + r')"\s*,\s*"args"\s*:\s*\{[^}]*\}\s*\}',
        '', cleaned, flags=re.DOTALL
    )
    cleaned = re.sub(r'(create_file|edit_file|read_file|list_files)\s*\([^)]*\)', '', cleaned, flags=re.DOTALL)
    return cleaned


class ToolExecutor:
    """Parse and execute tool calls from LLM responses."""

    def __init__(self, working_dir: str = "."):
        self.working_dir = working_dir
        self.logger = logging.getLogger("nare.tools.parsing.executor")

    def parse_and_execute(self, response: str) -> Tuple[str, List[str], List[str]]:
        """Parse tool calls from response and execute actions.

        Returns:
            Tuple of (cleaned_response, list of modified files, list of tool results)
        """
        tool_calls = parse_tool_calls(response)
        results = []
        modified_files = []

        for call in tool_calls:
            result = execute_tool_call(call['tool'], call['args'], working_dir=self.working_dir)
            if result:
                results.append(result)
                if call['tool'] in ["create_file", "edit_file"] and len(call['args']) > 0:
                    modified_files.append(call['args'][0])

        cleaned = clean_tool_calls_from_text(response)
        return cleaned.strip(), modified_files, results
