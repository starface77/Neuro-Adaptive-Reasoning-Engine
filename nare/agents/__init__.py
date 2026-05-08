"""
Agents module for NARE.

Contains domain-specific agents that form the autonomous workflow:
  - TriageAgent:       Fast intent classification (QUESTION/EXPLORE/EDIT)
  - PlanningAgent:     Step-by-step execution plan generation
  - CoderAgent:        Code generation based on plan
  - CriticAgent:       Code review and quality check
  - AnalyzerAgent:     Summarizes work and provides conclusions
  - MultiAgentWorkflow: Orchestrates Planner → Coder → Critic
  - AutonomousAgent:   Tool-calling autonomous agent
  - repo_map:          Repository structure mapper for LLM context
"""
from .roles.triage import TriageAgent
from .planning import PlanningAgent
from .roles.coder import CoderAgent
from .roles.critic import CriticAgent
from .roles.analyzer import AnalyzerAgent
from .workflow import MultiAgentWorkflow
from .repo_map import generate_repo_map
from .state import ToolCall, ToolResult, AgentState
from .autonomous import AutonomousAgent

__all__ = [
    "TriageAgent",
    "PlanningAgent",
    "CoderAgent",
    "CriticAgent",
    "AnalyzerAgent",
    "MultiAgentWorkflow",
    "AutonomousAgent",
    "generate_repo_map",
    "ToolCall",
    "ToolResult",
    "AgentState"
]
