"""Prompt-based tools loader.

Loads specialized tools with minimal prompts from nare/tools/prompts/.
Each tool is a separate file with TOOL_NAME, TOOL_DESCRIPTION, TOOL_PROMPT, and execute().
"""

import os
import importlib.util
from typing import Dict, List, Callable

class PromptTool:
    """Wrapper for a prompt-based tool."""

    def __init__(self, name: str, description: str, prompt: str, execute_fn: Callable):
        self.name = name
        self.description = description
        self.prompt = prompt
        self.execute = execute_fn

    def __repr__(self):
        return f"PromptTool({self.name})"

def load_prompt_tools() -> Dict[str, PromptTool]:
    """Load all prompt tools from nare/tools/prompts/."""
    tools = {}

    prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")

    if not os.path.exists(prompts_dir):
        return tools

    for filename in os.listdir(prompts_dir):
        if not filename.endswith(".py") or filename.startswith("_"):
            continue

        filepath = os.path.join(prompts_dir, filename)

        try:
            # Load module dynamically
            spec = importlib.util.spec_from_file_location(filename[:-3], filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Extract tool metadata
            tool_name = getattr(module, "TOOL_NAME", None)
            tool_desc = getattr(module, "TOOL_DESCRIPTION", None)
            tool_prompt = getattr(module, "TOOL_PROMPT", None)
            execute_fn = getattr(module, "execute", None)

            if all([tool_name, tool_desc, tool_prompt, execute_fn]):
                tools[tool_name] = PromptTool(
                    name=tool_name,
                    description=tool_desc,
                    prompt=tool_prompt,
                    execute_fn=execute_fn
                )
        except Exception as e:
            print(f"[PromptTools] Failed to load {filename}: {e}")

    return tools

def get_tools_schema() -> List[Dict]:
    """Get tool schemas for LLM function calling."""
    tools = load_prompt_tools()

    schemas = []
    for tool in tools.values():
        # Parse prompt to extract parameters (simple heuristic)
        schema = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }

        # Extract Args section from prompt
        if "Args:" in tool.prompt:
            args_section = tool.prompt.split("Args:")[1].split("Example:")[0]
            for line in args_section.strip().split("\n"):
                if line.strip().startswith("-"):
                    # Parse "- param: description"
                    parts = line.strip()[2:].split(":", 1)
                    if len(parts) == 2:
                        param_name = parts[0].strip()
                        param_desc = parts[1].strip()

                        schema["input_schema"]["properties"][param_name] = {
                            "type": "string",
                            "description": param_desc
                        }

                        # Mark as required if no default value
                        if "default:" not in param_desc.lower():
                            schema["input_schema"]["required"].append(param_name)

        schemas.append(schema)

    return schemas

def execute_tool(tool_name: str, **kwargs) -> str:
    """Execute a prompt tool by name."""
    tools = load_prompt_tools()

    if tool_name not in tools:
        return f"Error: tool '{tool_name}' not found"

    try:
        return tools[tool_name].execute(**kwargs)
    except Exception as e:
        return f"Error executing {tool_name}: {e}"

# Export
__all__ = ["load_prompt_tools", "get_tools_schema", "execute_tool", "PromptTool"]
