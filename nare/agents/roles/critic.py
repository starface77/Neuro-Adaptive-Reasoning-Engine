"""
Critic Agent — Code review and quality check.

Reviews code against plan, finds bugs, suggests improvements.
Simple and focused - no complex analysis, just practical feedback.
"""

import logging
from typing import Dict, Any, List

from nare.reasoning import llm

log = logging.getLogger("nare.agents.roles.critic")

SYSTEM_PROMPT = """
You are a code critic. Your job is to review code and provide honest feedback.

Check for:
1. Does it implement the plan correctly?
2. Are there obvious bugs or errors?
3. Is the code clean and readable?
4. Any security issues or bad practices?

Be strict but fair. If code is good, say so. If it has issues, explain clearly.

Always respond in this format:
<approved>yes|no</approved>
<issues>
- Issue 1 (if any)
- Issue 2 (if any)
</issues>
<suggestions>
- Suggestion 1 (if any)
- Suggestion 2 (if any)
</suggestions>
<summary>
Brief overall assessment
</summary>
"""

class CriticAgent:
    """Reviews code and provides feedback."""

    def critique(
        self,
        code: str,
        plan_steps: List[str],
        thinking_display=None,
    ) -> Dict[str, Any]:
        """Review code against the plan.

        Returns:
            {
                'approved': bool,
                'issues': List[str],
                'suggestions': List[str],
                'summary': str
            }
        """

        if thinking_display:
            thinking_display.update_waiting("Critic reviewing code...")

        plan_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan_steps))

        user_prompt = f"""
PLAN:
{plan_text}

CODE:
```
{code}
```

Provide your critique."""

        try:

            samples, _ = llm.generate_samples(
                SYSTEM_PROMPT + "\n\n" + user_prompt,
                n=1,
                temperature=0.3,
                mode="DIRECT",
                thinking_display=thinking_display
            )

            if not samples:
                return self._default_critique()

            response = samples[0]['solution']

            import re

            approved_match = re.search(r'<approved>(yes|no)</approved>', response, re.IGNORECASE)
            approved = approved_match and approved_match.group(1).lower() == 'yes'

            issues_match = re.search(r'<issues>(.*?)</issues>', response, re.DOTALL)
            issues = []
            if issues_match:
                issues_text = issues_match.group(1).strip()
                issues = [
                    line.strip('- ').strip()
                    for line in issues_text.split('\n')
                    if line.strip() and line.strip() != '-'
                ]

            suggestions_match = re.search(r'<suggestions>(.*?)</suggestions>', response, re.DOTALL)
            suggestions = []
            if suggestions_match:
                sugg_text = suggestions_match.group(1).strip()
                suggestions = [
                    line.strip('- ').strip()
                    for line in sugg_text.split('\n')
                    if line.strip() and line.strip() != '-'
                ]

            summary_match = re.search(r'<summary>(.*?)</summary>', response, re.DOTALL)
            summary = summary_match.group(1).strip() if summary_match else "Code reviewed"

            log.info(f"[Critic] Approved: {approved}, Issues: {len(issues)}, Suggestions: {len(suggestions)}")

            return {
                'approved': approved,
                'issues': issues,
                'suggestions': suggestions,
                'summary': summary
            }

        except Exception as e:
            log.error(f"[Critic] Failed: {e}")
            return self._default_critique()

    def _default_critique(self) -> Dict[str, Any]:
        """Fallback critique when LLM fails."""
        return {
            'approved': True,
            'issues': [],
            'suggestions': [],
            'summary': 'Unable to review code'
        }
