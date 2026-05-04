"""
Analyzer Agent — Summarizes work and provides conclusions.

After completing a task, analyzes what was done and provides:
- Summary of findings
- Key insights
- Recommendations
- Clear conclusion

Prevents the "no conclusion" problem where agent just stops.
"""

import logging
from typing import Dict, Any, List

from nare.reasoning import llm

log = logging.getLogger("nare.agents.roles.analyzer")

SYSTEM_PROMPT = """
You are an analyzer agent. Your job is to summarize what was done and provide clear conclusions.

Given a task and the actions taken, provide:
1. What was found/discovered
2. Key insights
3. Recommendations (if any)
4. Clear conclusion

Be concise but complete. Always end with a clear conclusion.

Format:
<summary>
Brief summary of what was done
</summary>
<findings>
- Finding 1
- Finding 2
</findings>
<insights>
Key insights or patterns discovered
</insights>
<recommendations>
- Recommendation 1 (if any)
- Recommendation 2 (if any)
</recommendations>
<conclusion>
Clear final conclusion
</conclusion>
"""

class AnalyzerAgent:
    """Analyzes completed work and provides summary."""

    def analyze(
        self,
        task: str,
        actions: List[Dict[str, Any]],
        results: Dict[str, Any],
        thinking_display=None,
    ) -> Dict[str, Any]:
        """Analyze completed task and provide summary.

        Args:
            task: Original task
            actions: List of actions taken (tool calls, etc.)
            results: Results from actions
            thinking_display: Optional display for streaming

        Returns:
            {
                'summary': str,
                'findings': List[str],
                'insights': str,
                'recommendations': List[str],
                'conclusion': str
            }
        """

        if thinking_display:
            thinking_display.update_waiting("Analyzer summarizing...")

        actions_text = self._format_actions(actions)
        results_text = self._format_results(results)

        user_prompt = f"""
TASK:
{task}

ACTIONS TAKEN:
{actions_text}

RESULTS:
{results_text}

Provide your analysis."""

        try:

            samples, _ = llm.generate_samples(
                SYSTEM_PROMPT + "\n\n" + user_prompt,
                n=1,
                temperature=0.3,
                mode="DIRECT",
                thinking_display=thinking_display
            )

            if not samples:
                return self._default_analysis(task)

            response = samples[0]['solution']

            import re

            summary_match = re.search(r'<summary>(.*?)</summary>', response, re.DOTALL)
            summary = summary_match.group(1).strip() if summary_match else "Task completed"

            findings_match = re.search(r'<findings>(.*?)</findings>', response, re.DOTALL)
            findings = []
            if findings_match:
                findings_text = findings_match.group(1).strip()
                findings = [
                    line.strip('- ').strip()
                    for line in findings_text.split('\n')
                    if line.strip() and line.strip() != '-'
                ]

            insights_match = re.search(r'<insights>(.*?)</insights>', response, re.DOTALL)
            insights = insights_match.group(1).strip() if insights_match else ""

            recommendations_match = re.search(r'<recommendations>(.*?)</recommendations>', response, re.DOTALL)
            recommendations = []
            if recommendations_match:
                rec_text = recommendations_match.group(1).strip()
                recommendations = [
                    line.strip('- ').strip()
                    for line in rec_text.split('\n')
                    if line.strip() and line.strip() != '-'
                ]

            conclusion_match = re.search(r'<conclusion>(.*?)</conclusion>', response, re.DOTALL)
            conclusion = conclusion_match.group(1).strip() if conclusion_match else "Analysis complete"

            log.info(f"[Analyzer] Summary: {summary[:50]}...")

            return {
                'summary': summary,
                'findings': findings,
                'insights': insights,
                'recommendations': recommendations,
                'conclusion': conclusion
            }

        except Exception as e:
            log.error(f"[Analyzer] Failed: {e}")
            return self._default_analysis(task)

    def _format_actions(self, actions: List[Dict[str, Any]]) -> str:
        """Format actions into readable text."""
        if not actions:
            return "No actions recorded"

        lines = []
        for i, action in enumerate(actions, 1):
            tool = action.get('tool', 'unknown')
            args = action.get('args', {})
            result = action.get('result', 'ok')
            lines.append(f"{i}. {tool}({args}) → {result}")

        return "\n".join(lines)

    def _format_results(self, results: Dict[str, Any]) -> str:
        """Format results into readable text."""
        if not results:
            return "No results"

        lines = []
        for key, value in results.items():
            if isinstance(value, (list, dict)):
                lines.append(f"{key}: {len(value)} items")
            else:
                lines.append(f"{key}: {value}")

        return "\n".join(lines)

    def _default_analysis(self, task: str) -> Dict[str, Any]:
        """Fallback analysis when LLM fails."""
        return {
            'summary': f"Completed task: {task}",
            'findings': [],
            'insights': '',
            'recommendations': [],
            'conclusion': 'Task completed'
        }
