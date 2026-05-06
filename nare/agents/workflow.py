"""
Multi-Agent Workflow — Planner → Coder → Critic

Orchestrates the 3-agent system:
1. Planning agent breaks down task
2. Coder agent writes code
3. Critic agent reviews and provides feedback
4. Retry if needed (max 2 iterations)

Simple workflow without over-engineering.
"""

from nare.utils.logger import get_logger
from typing import Dict, Any, Optional

from .planning import PlanningAgent
from .roles.coder import CoderAgent
from .roles.critic import CriticAgent

log = get_logger("nare.agents.workflow")

class MultiAgentWorkflow:
    """Orchestrates Planner → Coder → Critic workflow."""

    def __init__(self):
        self.planner = PlanningAgent()
        self.coder = CoderAgent()
        self.critic = CriticAgent()

    def execute(
        self,
        task: str,
        repo_map: Optional[str] = None,
        context: Optional[str] = None,
        thinking_display=None,
        max_iterations: int = 2,
    ) -> Dict[str, Any]:
        """Execute full workflow: Plan → Code → Critique (with retry).

        Returns:
            {
                'plan': dict,
                'code': str,
                'explanation': str,
                'critique': dict,
                'iterations': int,
                'success': bool
            }
        """

        log.info(f"[Workflow] Starting: {task[:50]}...")

        if thinking_display:
            thinking_display.stream_token("🧠 Agent 1: Planning...\n")

        plan = self.planner.generate_plan(
            task=task,
            repo_map=repo_map,
            existing_context=context,
            thinking_display=thinking_display
        )

        plan_steps = plan.get('plan_steps', [])
        complexity = plan.get('complexity', 'moderate')

        if thinking_display:
            thinking_display.stream_token(f"   Complexity: {complexity}\n")
            thinking_display.stream_token(f"   Steps: {len(plan_steps)}\n\n")

        if not plan_steps:
            log.warning("[Workflow] No plan generated")
            return self._failed_result(plan, "No plan generated")

        code_result = None
        critique_result = None
        iteration = 0
        previous_code = None
        previous_critique = None

        while iteration < max_iterations:
            iteration += 1

            if thinking_display:
                thinking_display.stream_token(f"💻 Agent 2: Coding (attempt {iteration})...\n")

            iteration_context = context or ""
            if previous_code and previous_critique:
                iteration_context += f"\n\n--- PREVIOUS ATTEMPT ---\n"
                iteration_context += f"Code:\n```python\n{previous_code}\n```\n\n"
                iteration_context += f"Issues found:\n"
                for issue in previous_critique['issues']:
                    iteration_context += f"- {issue}\n"
                iteration_context += "\nPlease fix these specific issues in the code above.\n"

            code_result = self.coder.code(
                plan_steps=plan_steps,
                context=iteration_context,
                thinking_display=thinking_display
            )

            code = code_result['code']
            explanation = code_result['explanation']

            if thinking_display:
                thinking_display.stream_token(f"   Generated {len(code)} chars\n\n")

            if thinking_display:
                thinking_display.stream_token("🔍 Agent 3: Reviewing...\n")

            critique_result = self.critic.critique(
                code=code,
                plan_steps=plan_steps,
                thinking_display=thinking_display
            )

            approved = critique_result['approved']
            issues = critique_result['issues']

            if approved:
                if thinking_display:
                    thinking_display.stream_token("   ✅ Code approved!\n\n")
                log.info(f"[Workflow] Success after {iteration} iteration(s)")
                break

            # Save for next iteration
            previous_code = code
            previous_critique = critique_result

            if iteration < max_iterations:
                if thinking_display:
                    thinking_display.stream_token(f"   ❌ Issues found: {len(issues)}\n")
                    for issue in issues[:3]:
                        thinking_display.stream_token(f"      - {issue}\n")
                    thinking_display.stream_token("   Retrying...\n\n")

                plan_steps.append(f"Fix issues: {', '.join(issues[:3])}")
            else:
                if thinking_display:
                    thinking_display.stream_token(f"   ⚠️  Max iterations reached\n\n")
                log.warning(f"[Workflow] Max iterations reached, issues remain: {issues}")

        return {
            'plan': plan,
            'code': code_result['code'] if code_result else '',
            'explanation': code_result['explanation'] if code_result else '',
            'critique': critique_result or {},
            'iterations': iteration,
            'success': critique_result.get('approved', False) if critique_result else False
        }

    def _failed_result(self, plan: dict, reason: str) -> Dict[str, Any]:
        """Return failed result."""
        return {
            'plan': plan,
            'code': '',
            'explanation': reason,
            'critique': {'approved': False, 'issues': [reason], 'suggestions': [], 'summary': reason},
            'iterations': 0,
            'success': False
        }
