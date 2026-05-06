"""
Tool executor — Parses and executes tool calls from LLM responses.

The LLM generates tool calls in its response, and this module
extracts and executes them.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Callable
from nare.tools.file_tools import read_file, create_file, edit_file, list_files

def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Parse tool calls from LLM response.

    Supports two formats:
    1. Function calls: create_file("path", "content")
    2. XML tags: <create_file>path\ncontent</create_file>

    Returns list of tool call dicts with 'tool' and 'args' keys.
    """
    tool_calls = []

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

            tool_calls.append({
                'tool': tool_name,
                'args': [filepath, file_content]
            })
        elif tool_name == "edit_file":
            # Try XML tags first
            path_match = re.search(r'<path>\s*(.*?)\s*</path>', content, re.DOTALL)
            old_match = re.search(r'<old>\s*(.*?)\s*</old>', content, re.DOTALL)
            new_match = re.search(r'<new>\s*(.*?)\s*</new>', content, re.DOTALL)

            if path_match and old_match and new_match:
                filepath = path_match.group(1).strip()
                target = old_match.group(1)
                replacement = new_match.group(1)
            else:
                # Try separator format
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    filepath = parts[0].strip()
                    target = parts[1].strip()
                    replacement = parts[2].strip()
                else:
                    # Fallback: old single-line format
                    lines = content.split('\n', 2)
                    if len(lines) >= 3:
                        filepath = lines[0].strip()
                        target = lines[1].strip()
                        replacement = lines[2].strip()
                    else:
                        continue

            filepath = re.sub(r'</?path>', '', filepath)
            tool_calls.append({
                'tool': tool_name,
                'args': [filepath, target, replacement]
            })
        elif tool_name == "read_file":

            filepath = content.strip()

            filepath = re.sub(r'</?path>', '', filepath)
            tool_calls.append({
                'tool': tool_name,
                'args': [filepath]
            })
        elif tool_name == "list_files":

            lines = content.split('\n', 1)
            args = [lines[0].strip()]
            if len(lines) > 1:
                args.append(lines[1].strip())
            tool_calls.append({
                'tool': tool_name,
                'args': args
            })

    if tool_calls:
        return tool_calls

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

                tool_calls.append({
                    'tool': tool_name,
                    'args': args
                })
                continue
        except Exception as e:
            logging.warning(f"Failed to ast.parse tool call {tool_name}: {e}")
            pass

        args = []

        tool_calls.append({
            'tool': tool_name,
            'args': args
        })

    return tool_calls

def execute_tool_call(tool_name: str, args: List[str], stream_callback: Optional[Callable] = None, working_dir: str = ".") -> Optional[str]:
    """Execute a single tool call.

    Args:
        tool_name: Name of the tool (create_file, edit_file, etc.)
        args: List of string arguments
        stream_callback: Optional callback for streaming file content display
        working_dir: Working directory for relative paths

    Returns:
        Result message or None if failed
    """
    try:
        if tool_name == "create_file":
            if len(args) < 2:
                return f"Error: create_file requires 2 arguments (filepath, content)"
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
                return f"Error: edit_file requires 3 arguments (filepath, target, replacement)"
            filepath = args[0]
            target = args[1]
            replacement = args[2]

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
                return f"Error: read_file requires at least 1 argument (filepath)"
            filepath = args[0]

            if not os.path.isabs(filepath):
                filepath = os.path.join(working_dir, filepath)

            start_line = int(args[1]) if len(args) > 1 else None
            end_line = int(args[2]) if len(args) > 2 else None
            content = read_file(filepath, start_line, end_line)
            return f"Read {filepath}:\n{content[:500]}..."

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
    """Parse and execute all tool calls from LLM response.

    Args:
        response: Full LLM response text
        stream_callback: Optional callback for streaming file content display
        working_dir: Working directory for relative paths

    Returns:
        List of execution results
    """
    tool_calls = parse_tool_calls(response)
    results = []

    for call in tool_calls:
        result = execute_tool_call(call['tool'], call['args'], stream_callback, working_dir)
        if result:
            results.append(result)
            logging.info(f"Executed {call['tool']}: {result}")

    return results

class ToolExecutor:
    """Parse and execute XML tool calls from LLM responses."""

    def __init__(self, working_dir: str = "."):
        self.working_dir = working_dir
        self.logger = get_logger("nare.tools.parsing.executor")

    def parse_and_execute(self, response: str) -> Tuple[str, List[str]]:
        """Parse XML tags from response and execute actions.

        Returns:
            Tuple of (cleaned_response, list of modified files)
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

        cleaned = response

        cleaned = re.sub(r'<(create_file|edit_file|read_file|list_files)>.*?</\1>', '', cleaned, flags=re.DOTALL)

        cleaned = re.sub(r'(create_file|edit_file|read_file|list_files)\s*\([^)]*\)', '', cleaned, flags=re.DOTALL)

        return cleaned.strip(), modified_files
