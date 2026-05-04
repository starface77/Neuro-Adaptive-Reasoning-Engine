"""
Planning Agent — Task decomposition with proper LLM understanding.

System prompt ensures the LLM:
  1. Understands the user's intent (even in Russian, broken English, etc.)
  2. Produces a concrete, actionable plan
  3. Identifies exact files to modify
"""

import re
import logging
from typing import Dict, Any, Optional

from nare.reasoning import llm

log = logging.getLogger("nare.agents.planning")

SYSTEM_PROMPT = """
You are a senior software engineer planning a code change.

Your job:
1. Understand what the user ACTUALLY wants (even if their message is vague, in another language, or has typos).
2. Produce a detailed, concrete plan with clear steps.
3. List the exact files that need to be read or modified.
4. Explain WHY each step is needed.

Complexity guidelines:
- trivial: Simple one-file edits (add line, fix typo, rename variable)
- moderate: Multi-step changes, new features, refactoring (DEFAULT)
- complex: Architecture changes, multiple files, requires deep understanding
- unclear: ONLY when the task itself is fundamentally unclear (e.g., "make it better", "fix everything")

IMPORTANT: If the user's request is clear but missing details (like which file), make a reasonable assumption based on context (loaded files, repo structure) and set complexity to "trivial" or "moderate". Do NOT use "unclear" for missing details - infer from context.

Rules:
- Use REAL file paths from the repository structure provided.
- If files are already loaded, prefer those as targets.
- Order steps logically: read before write, dependencies first.
- Be detailed and specific - explain what will be done in each step.
- Include reasoning for each step.

Always respond in this exact XML format:
<complexity>trivial|moderate|complex|unclear</complexity>
<reasoning>
[Explain your understanding of the task and approach]
</reasoning>
<plan>
1. [Detailed step with explanation]
2. [Detailed step with explanation]
...
</plan>
<target_files>
path/to/file.py
another/file.js
</target_files>

Example:
<complexity>moderate</complexity>
<reasoning>
User wants to create a Snake game. This requires:
- Main game loop with pygame
- Snake class with movement logic
- Food spawning system
- Collision detection
- Score tracking
</reasoning>
<plan>
1. Create main.py with pygame initialization and game window setup
2. Implement Snake class with position tracking, movement, and growth logic
3. Implement Food class with random positioning
4. Add collision detection for snake-food and snake-self
5. Add score display and game over handling
6. Create requirements.txt with pygame dependency
7. Write README.md with installation and usage instructions
</plan>
<target_files>
main.py
requirements.txt
README.md
</target_files>
"""

class PlanningAgent:
    """Agent that creates execution plans before coding begins."""

    def generate_plan(
        self,
        task: str,
        repo_map: Optional[str] = None,
        existing_context: Optional[str] = None,
        thinking_display=None,
    ) -> Dict[str, Any]:
        """Generate a step-by-step plan for a given task."""

        import re
        target_dir = None
        dir_patterns = [
            r'в папке\s+([^\s]+)',
            r'в директории\s+([^\s]+)',
            r'в\s+([A-Z]:[/\\][^\s]+)',
            r'in folder\s+([^\s]+)',
            r'in directory\s+([^\s]+)',
            r'to\s+([A-Z]:[/\\][^\s]+)',
        ]

        for pattern in dir_patterns:
            match = re.search(pattern, task, re.IGNORECASE)
            if match:
                target_dir = match.group(1)
                break

        prompt = f"USER REQUEST:\n{task}\n"

        if target_dir:
            prompt += f"\nTARGET DIRECTORY: {target_dir}\n"
            prompt += f"IMPORTANT: All file paths must be relative to or inside {target_dir}\n"

        if repo_map:
            prompt += f"\nREPOSITORY STRUCTURE:\n{repo_map}\n"

        if existing_context:
            prompt += f"\nFILES ALREADY LOADED:\n{existing_context}\n"

        log.info("[Planning] Generating execution plan...")

        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

        samples, _ = llm.generate_samples(
            full_prompt, n=1, temperature=0.2, mode="DIRECT", thinking_display=None
        )

        if not samples:
            log.warning("[Planning] LLM returned no samples")
            return {
                "plan_steps": [],
                "target_files": [],
                "complexity": "moderate",
                "raw_output": "",
                "target_dir": target_dir,
            }

        content = samples[0]["solution"]
        plan = self._parse_output(content)
        plan["target_dir"] = target_dir
        return plan

    def _parse_output(self, content: str) -> Dict[str, Any]:
        """Parse structured LLM output - handles both XML and plain text."""
        result = {
            "plan_steps": [],
            "target_files": [],
            "complexity": "moderate",
            "reasoning": "",
            "raw_output": content,
        }

        c_match = re.search(
            r"<complexity>\s*(trivial|moderate|complex|unclear)\s*</complexity>",
            content, re.IGNORECASE,
        )
        if c_match:
            result["complexity"] = c_match.group(1).lower()

        r_match = re.search(
            r"<reasoning>(.*?)</reasoning>",
            content, re.DOTALL | re.IGNORECASE,
        )
        if r_match:
            result["reasoning"] = r_match.group(1).strip()

        plan_match = re.search(r"<plan>(.*?)</plan>", content, re.DOTALL)
        if plan_match:
            lines = plan_match.group(1).strip().split("\n")
            result["plan_steps"] = [
                re.sub(r"^\d+[\.\)]\s*", "", line.strip())
                for line in lines
                if line.strip() and not line.strip().startswith("#")
            ]

        files_match = re.search(
            r"<target_files>(.*?)</target_files>", content, re.DOTALL
        )
        if files_match:
            lines = files_match.group(1).strip().split("\n")
            result["target_files"] = [
                line.strip().lstrip("- ")
                for line in lines
                if line.strip() and not line.strip().startswith("#")
            ]

        if not result["plan_steps"]:

            numbered_lines = re.findall(r'^\s*\d+[\.\)]\s+(.+)$', content, re.MULTILINE)
            if numbered_lines:
                result["plan_steps"] = [line.strip() for line in numbered_lines if line.strip()]

        if not result["target_files"]:

            file_paths = re.findall(r'[\w/\-]+\.(?:py|js|ts|jsx|tsx|html|css|json|md|txt|yaml|yml)\b', content)
            result["target_files"] = list(dict.fromkeys(file_paths))[:5]

        if not result["reasoning"]:

            lines = content.split('\n')
            reasoning_lines = []
            for line in lines[:10]:
                if line.strip() and not line.strip().startswith(('1.', '2.', '3.', '<', '#')):
                    reasoning_lines.append(line.strip())
            if reasoning_lines:
                result["reasoning"] = ' '.join(reasoning_lines)[:500]

        return result
