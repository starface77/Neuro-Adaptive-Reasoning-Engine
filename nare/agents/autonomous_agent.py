"""
Autonomous Agent
Main agent orchestrating task execution with planning and memory.
"""

from typing import Dict, Any, List, Optional
import logging

from ..execution.executor import TaskExecutor
from ..planning.planner import TaskPlanner


logger = logging.getLogger(__name__)


class AutonomousAgent:
    """
    Autonomous agent that plans and executes tasks.
    
    This agent:
    - Analyzes tasks to understand requirements
    - Creates execution plans
    - Executes steps using available tools
    - Learns from successful completions
    """
    
    def __init__(
        self,
        llm_client: Any,
        memory_system: Any,
        tools_registry: Dict[str, Any],
        file_operations: Any,
        terminal: Any,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize autonomous agent.
        
        Args:
            llm_client: LLM client for reasoning
            memory_system: Memory system for learning
            tools_registry: Available tools
            file_operations: File operations handler
            terminal: Terminal operations handler
            config: Optional configuration
        """
        self.llm = llm_client
        self.memory = memory_system
        self.config = config or {}
        
        # Initialize components
        self.planner = TaskPlanner(llm_client, memory_system)
        self.executor = TaskExecutor(tools_registry, file_operations, terminal)
        
        # State
        self.current_task = None
        self.execution_history = []
    
    def execute_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a task autonomously.
        
        Args:
            task: Task description
            context: Optional execution context
            
        Returns:
            Execution result with status and output
        """
        self.current_task = task
        context = context or {}
        
        try:
            # Step 1: Analyze task
            logger.info(f"Analyzing task: {task}")
            analysis = self.planner.analyze_task(task, context)
            
            # Step 2: Create plan
            logger.info("Creating execution plan")
            plan = self.planner.create_plan(task, analysis)
            
            # Step 3: Execute plan
            logger.info(f"Executing {len(plan)} steps")
            results = []
            
            for i, step in enumerate(plan, 1):
                logger.info(f"Step {i}/{len(plan)}: {step.get('action')}")
                result = self.executor.execute_step(step, context)
                results.append(result)
                
                # Stop on failure if configured
                if result['status'] == 'failed' and self.config.get('stop_on_failure', True):
                    logger.error(f"Step failed: {result.get('error')}")
                    break
            
            # Step 4: Store successful execution in memory
            if all(r['status'] == 'success' for r in results):
                self._store_success(task, plan, results)
            
            return {
                'status': 'success' if all(r['status'] == 'success' for r in results) else 'partial',
                'task': task,
                'analysis': analysis,
                'plan': plan,
                'results': results
            }
            
        except Exception as e:
            logger.exception(f"Task execution failed: {e}")
            return {
                'status': 'failed',
                'task': task,
                'error': str(e)
            }
    
    def execute_with_reasoning(self, task: str) -> Dict[str, Any]:
        """
        Execute task with step-by-step reasoning.
        
        Args:
            task: Task description
            
        Returns:
            Execution result with reasoning trace
        """
        reasoning_steps = []
        
        # Build reasoning prompt
        prompt = self._build_reasoning_prompt(task)
        
        # Get LLM response with tool calls
        response = self.llm.complete(prompt)
        
        # Parse and execute tool calls
        tool_calls = self.executor.parse_tool_calls(response)
        
        for tool_call in tool_calls:
            result = self.executor.execute_tool_call(
                tool_call['tool'],
                tool_call['arguments']
            )
            reasoning_steps.append({
                'tool': tool_call['tool'],
                'arguments': tool_call['arguments'],
                'result': result
            })
        
        return {
            'status': 'success',
            'task': task,
            'reasoning': response,
            'steps': reasoning_steps
        }
    
    def _build_reasoning_prompt(self, task: str) -> str:
        """Build prompt for reasoning-based execution."""
        # Retrieve relevant memories
        memories = self.memory.search(task, limit=3)
        memory_context = "\n".join([
            f"Past solution: {m['description']}"
            for m in memories
        ])
        
        return f"""Task: {task}

Relevant past solutions:
{memory_context}

Think step-by-step and use available tools to complete the task.
Available tools: read_file, edit_file, create_file, run_command

Respond with tool calls in XML format:
<tool_name>{{\"arg\": \"value\"}}</tool_name>
"""
    
    def _store_success(
        self,
        task: str,
        plan: List[Dict[str, Any]],
        results: List[Dict[str, Any]]
    ) -> None:
        """Store successful execution in memory."""
        try:
            self.memory.store({
                'task': task,
                'plan': plan,
                'results': results,
                'timestamp': self._get_timestamp()
            })
            logger.info("Stored successful execution in memory")
        except Exception as e:
            logger.warning(f"Failed to store in memory: {e}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            'current_task': self.current_task,
            'execution_history_size': len(self.execution_history),
            'memory_size': self.memory.size() if hasattr(self.memory, 'size') else 'unknown'
        }