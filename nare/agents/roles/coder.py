"""
Coder Agent — Code generation based on plan.

Takes a plan and writes clean, working code.
Simple and focused - implements the plan step by step.
"""

from nare.utils.logger import get_logger
from typing import Dict, Any, List

from nare.reasoning import llm

log = get_logger("nare.agents.roles.coder")

SYSTEM_PROMPT = """
You are a coding agent. Your job is to write clean, working code based on a plan.

Rules:
1. Follow the plan exactly
2. Write clean, readable code
3. Add brief comments where needed
4. Handle errors properly
5. Keep it simple - no over-engineering

Always respond with:
<code>
[your code here]
</code>
<explanation>
Brief explanation of what the code does
</explanation>
"""

class CoderAgent:
    """Generates code based on plan."""

    def code(
        self,
        plan_steps: List[str],
        context: str = "",
        thinking_display=None,
    ) -> Dict[str, Any]:
        """Generate code from plan.

        Returns:
            {
                'code': str,
                'explanation': str
            }
        """

        if thinking_display:
            thinking_display.update_waiting("Coder writing code...")

        plan_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan_steps))

        user_prompt = f"""
PLAN:
{plan_text}
"""

        if context:
            user_prompt += f"\n\nCONTEXT:\n{context}\n"

        user_prompt += "\nProvide clean, working code."

        try:

            samples, _ = llm.generate_samples(
                SYSTEM_PROMPT + "\n\n" + user_prompt,
                n=1,
                temperature=0.5,
                mode="DIRECT",
                thinking_display=thinking_display
            )

            if not samples or not isinstance(samples[0], dict) or 'solution' not in samples[0]:
                return self._default_code()

            response = samples[0]['solution']

            import re

            code_match = re.search(r'<code>(.*?)</code>', response, re.DOTALL)
            code = code_match.group(1).strip() if code_match else response

            explanation_match = re.search(r'<explanation>(.*?)</explanation>', response, re.DOTALL)
            explanation = explanation_match.group(1).strip() if explanation_match else "Code generated"

            log.info(f"[Coder] Generated {len(code)} chars of code")

            return {
                'code': code,
                'explanation': explanation
            }

        except Exception as e:
            log.error(f"[Coder] Failed: {e}")
            return self._default_code()

    def _default_code(self) -> Dict[str, Any]:
        """Fallback when LLM fails."""
        return {
            'code': '# Unable to generate code',
            'explanation': 'Code generation failed'
        }
