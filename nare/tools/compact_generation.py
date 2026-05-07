"""Compact tool-based generation mode.

Uses specialized prompt tools instead of large system prompts.
Reduces token usage by 80-90% for simple tasks.
"""

import json
import re
from typing import List, Dict, Optional
from .prompt_loader import load_prompt_tools, execute_tool

def _select_tool_locally(query: str) -> Optional[str]:
    """Select appropriate tool based on query keywords WITHOUT calling LLM.

    This is the prompt orchestrator - it chooses the right tool locally
    so we don't send all tool schemas to the LLM.
    """
    query_lower = query.lower()

    # Bash commands (check first - higher priority)
    if any(kw in query_lower for kw in ["выполни команду", "запусти команду", "run command", "execute command"]):
        return "bash_command"

    # File operations
    if any(kw in query_lower for kw in ["прочитай файл", "покажи содержимое", "read file", "show file", "cat "]):
        return "read_file"

    if any(kw in query_lower for kw in ["создай файл", "напиши файл", "create file", "write file"]):
        return "write_file"

    if any(kw in query_lower for kw in ["добавь в файл", "append", "дописать в файл"]):
        return "append_file"

    if any(kw in query_lower for kw in ["измени в файл", "замени в файл", "edit file", "replace in file", "поменяй в файл"]):
        return "edit_file"

    # Directory operations
    if any(kw in query_lower for kw in ["список файлов", "покажи файлы", "list files"]):
        return "list_files"

    return None

def generate_with_tools(query: str, thinking_display=None) -> Dict:
    """Generate response using compact tool-based approach.

    Instead of sending huge system prompts, we:
    1. Locally select the right tool (prompt orchestrator)
    2. Send ONLY that tool's minimal prompt to LLM
    3. Execute tool and return result

    This reduces token usage from ~6k to ~200-500 tokens for simple tasks.
    """
    from ..reasoning.generation import engine as llm

    # Load available tools
    tools = load_prompt_tools()

    # Prompt orchestrator: select tool locally WITHOUT LLM call
    selected_tool_name = _select_tool_locally(query)

    if not selected_tool_name or selected_tool_name not in tools:
        # Fallback: let LLM decide (rare case)
        return {
            "solution": "Could not determine tool",
            "reasoning": "Orchestrator failed to select tool"
        }

    selected_tool = tools[selected_tool_name]

    # Build minimal prompt with ONLY the selected tool
    system_prompt = f"""You are a coding assistant. Use the {selected_tool_name} tool.
Output JSON: {{"params": {{"param1": "value"}}}}"""

    user_prompt = f"""Task: {query}

Tool: {selected_tool_name}
{selected_tool.prompt}

Respond with JSON parameters only."""

    # Call LLM with minimal prompt (only 1 tool, not all tools)
    payload = {
        "model": llm.ANTHROPIC_MODEL,
        "max_tokens": 512,
        "temperature": 0.3,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}]
    }

    # Setup streaming callback if display available
    stream_callback = None
    if thinking_display:
        # Switch to solution mode for streaming
        if hasattr(thinking_display, 'switch_to_solution'):
            thinking_display.switch_to_solution()

        def callback(token: str):
            thinking_display.stream_token(token)
        stream_callback = callback

    try:
        response = llm._post_anthropic("messages", payload, stream_callback=stream_callback)

        # Parse parameters
        params = json.loads(response)
        if "params" in params:
            params = params["params"]

        # Execute tool
        result = execute_tool(selected_tool_name, **params)

        return {
            "solution": result,
            "reasoning": f"Used tool: {selected_tool_name}",
            "tool_used": selected_tool_name
        }
    except Exception as e:
        # Fallback to direct response
        return {
            "solution": f"Error: {e}",
            "reasoning": "Tool execution failed"
        }

def should_use_tools(query: str, intent: str) -> bool:
    """Decide if we should use compact tools instead of full prompts.

    Use tools for:
    - Simple file operations (read, write, edit, append)
    - List/search operations
    - Bash commands

    Don't use tools for:
    - Complex reasoning tasks
    - Multi-step workflows
    - Questions requiring deep analysis
    """
    query_lower = query.lower()

    # Check if orchestrator can select a tool
    selected_tool = _select_tool_locally(query)
    if selected_tool:
        # Additional check: query should be simple (not multi-step)
        multi_step_indicators = ["затем", "потом", "после этого", "then", "after", "next", "и также", "а также"]
        if any(ind in query_lower for ind in multi_step_indicators):
            return False  # Multi-step, use full prompts

        # Check query length - very long queries likely need full reasoning
        if len(query.split()) > 20:
            return False

        return True

    return False
