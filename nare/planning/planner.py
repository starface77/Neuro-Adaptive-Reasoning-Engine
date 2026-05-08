"""
Task Planner
Analyzes tasks and creates execution plans.
"""

from typing import Dict, Any, List, Optional


class TaskPlanner:
    """Plans task execution strategies."""
    
    def __init__(self, llm_client: Any, memory: Any):
        """
        Initialize planner.
        
        Args:
            llm_client: LLM client for reasoning
            memory: Memory system for context
        """
        self.llm = llm_client
        self.memory = memory
    
    def analyze_task(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze task requirements and complexity.
        
        Args:
            task: Task description
            context: Current context
            
        Returns:
            Analysis with complexity, requirements, and approach
        """
        # Retrieve relevant memories
        relevant_memories = self.memory.search(task, limit=5)
        
        # Build analysis prompt
        prompt = self._build_analysis_prompt(task, context, relevant_memories)
        
        # Get LLM analysis
        response = self.llm.complete(prompt)
        
        return self._parse_analysis(response)
    
    def create_plan(self, task: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create execution plan for the task.
        
        Args:
            task: Task description
            analysis: Task analysis result
            
        Returns:
            List of execution steps
        """
        prompt = self._build_planning_prompt(task, analysis)
        response = self.llm.complete(prompt)
        
        return self._parse_plan(response)
    
    def _build_analysis_prompt(
        self,
        task: str,
        context: Dict[str, Any],
        memories: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for task analysis."""
        memory_context = "\n".join([
            f"- {m['description']}: {m['solution']}"
            for m in memories
        ])
        
        return f"""Analyze this task:
Task: {task}

Context:
- Working directory: {context.get('cwd', 'unknown')}
- Available tools: {', '.join(context.get('tools', []))}

Relevant past solutions:
{memory_context}

Provide:
1. Complexity (low/medium/high)
2. Required tools
3. Potential challenges
4. Recommended approach
"""
    
    def _build_planning_prompt(self, task: str, analysis: Dict[str, Any]) -> str:
        """Build prompt for plan creation."""
        return f"""Create execution plan for:
Task: {task}

Analysis:
- Complexity: {analysis.get('complexity', 'unknown')}
- Approach: {analysis.get('approach', 'unknown')}

Provide step-by-step plan with:
1. Action to take
2. Parameters needed
3. Expected outcome
"""
    
    def _parse_analysis(self, response: str) -> Dict[str, Any]:
        """Parse LLM analysis response."""
        # Simple parsing - in production use structured output
        lines = response.strip().split('\n')
        
        analysis = {
            'complexity': 'medium',
            'tools': [],
            'challenges': [],
            'approach': ''
        }
        
        for line in lines:
            if 'complexity:' in line.lower():
                analysis['complexity'] = line.split(':')[1].strip().lower()
            elif 'approach:' in line.lower():
                analysis['approach'] = line.split(':')[1].strip()
        
        return analysis
    
    def _parse_plan(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM plan response."""
        # Simple parsing - in production use structured output
        steps = []
        
        lines = response.strip().split('\n')
        current_step = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                if current_step:
                    steps.append(current_step)
                current_step = {
                    'action': line.split('.', 1)[1].strip(),
                    'parameters': {}
                }
        
        if current_step:
            steps.append(current_step)
        
        return steps