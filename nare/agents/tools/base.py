"""Tool primitives for the NARE agent loop.

A `Tool` is a name + JSON-shaped parameter schema + a callable. The
agent loop renders the schema into the LLM's system prompt, the LLM
emits a tool call, and we dispatch to the matching `Tool.run`.

The schema is intentionally tiny — just enough to drive the LLM
prompt and validate inputs at runtime. We don't depend on Pydantic
to keep startup fast and the dependency surface small.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class ToolError(Exception):
    """Raised when a tool's input is invalid or execution fails fatally."""


@dataclass
class ToolParam:
    """One parameter of a tool's schema."""

    name: str
    type: str  # 'string' | 'integer' | 'number' | 'boolean' | 'array'
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolResult:
    """Outcome of a single tool call.

    `summary` is a one-line human-readable result (rendered in the
    `└` summary slot of the UI block). `body` is multi-line content
    (file body, command output, diff). `meta` carries structured data
    (line counts, exit codes, paths) the renderer or the agent loop
    consumes.
    """

    ok: bool
    summary: Optional[str] = None
    body: Optional[str] = None
    body_lang: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_llm_observation(self, max_chars: int = 4_000) -> str:
        """Pack the result into a string that gets fed back to the LLM."""
        parts: List[str] = []
        if self.summary:
            parts.append(self.summary)
        if self.body:
            body = self.body
            if len(body) > max_chars:
                body = body[:max_chars] + f"\n... (truncated {len(self.body) - max_chars} chars)"
            parts.append(body)
        if self.error:
            parts.append(f"ERROR: {self.error}")
        return "\n".join(parts) if parts else ("ok" if self.ok else "fail")


@dataclass
class Tool:
    """A single executable tool exposed to the agent loop."""

    name: str
    description: str
    parameters: List[ToolParam]
    run: Callable[..., ToolResult]
    # The visual verb shown in tool blocks (`Read`, `Write`, `Bash`, …).
    # Defaults to a Title-cased version of the tool name.
    display_verb: Optional[str] = None
    # When True, the agent loop should require user confirmation before
    # executing this tool in Manual mode (Phase 4 will plumb this).
    requires_confirmation: bool = False

    def __post_init__(self) -> None:
        if self.display_verb is None:
            self.display_verb = self.name.replace("_", " ").title().replace(" ", "")

    # ── Schema / prompt rendering ────────────────────────────────

    def schema_dict(self) -> Dict[str, Any]:
        """Return a JSON-shaped schema description (subset of OpenAPI)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    p.name: {
                        "type": p.type,
                        "description": p.description,
                        **({"default": p.default} if not p.required else {}),
                    }
                    for p in self.parameters
                },
                "required": [p.name for p in self.parameters if p.required],
            },
        }

    def schema_for_prompt(self) -> str:
        """Render this tool as a compact line for the system prompt."""
        sig_parts: List[str] = []
        for p in self.parameters:
            tag = p.name + (":" + p.type)
            if not p.required:
                tag += "?"
            sig_parts.append(tag)
        sig = ", ".join(sig_parts)
        return f"- {self.name}({sig}) — {self.description}"

    # ── Execution ────────────────────────────────────────────────

    def validate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate raw args against the schema. Raise on missing requireds."""
        cleaned: Dict[str, Any] = {}
        for p in self.parameters:
            if p.name in args:
                cleaned[p.name] = args[p.name]
            elif p.required:
                raise ToolError(
                    f"tool {self.name!r}: missing required parameter {p.name!r}"
                )
            elif p.default is not None:
                cleaned[p.name] = p.default
        # Forward unknown args; tools may accept extras for forward compat.
        for k, v in args.items():
            cleaned.setdefault(k, v)
        return cleaned


@dataclass
class ToolRegistry:
    """Holds named tools and dispatches calls."""

    tools: Dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ToolError(f"duplicate tool: {tool.name}")
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        tool = self.tools.get(name)
        if tool is None:
            raise ToolError(f"unknown tool: {name}")
        return tool

    def names(self) -> List[str]:
        return sorted(self.tools)

    def call(self, name: str, args: Dict[str, Any]) -> ToolResult:
        try:
            tool = self.get(name)
            clean = tool.validate(args)
            result = tool.run(**clean)
            if not isinstance(result, ToolResult):
                # Tolerate tools that return strings — wrap them.
                result = ToolResult(ok=True, body=str(result))
            return result
        except ToolError as e:
            return ToolResult(ok=False, error=str(e))
        except Exception as e:  # pragma: no cover — defensive
            return ToolResult(ok=False, error=f"{type(e).__name__}: {e}")

    def schema_block(self) -> str:
        """Return all tool schemas as a markdown block for the system prompt."""
        lines = ["Available tools:"]
        for name in sorted(self.tools):
            lines.append(self.tools[name].schema_for_prompt())
        return "\n".join(lines)
