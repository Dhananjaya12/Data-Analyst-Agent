"""
Base classes for all agents
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ExecutionState:
    """Shared state across all agents"""
    user_query: str
    plan: str = ""
    query_generated: str = ""
    data_retrieved: List = field(default_factory=list)
    data_summary: str = ""
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    execution_steps: List[str] = field(default_factory=list)

    insights: List = field(default_factory=list)
    is_valid: bool = True
    validation_feedback: str = ""
    error: str = ""

    final_answer: str = ""
    confidence: float = 0.0

    selected_file_ids: list = field(default_factory=list)
    selected_files_context: str = ""
    critic_report: dict = field(default_factory=dict)
    refinement_count: int = 0 

    logic_score: float = 0.5
    data_score: float = 0.5
    insights_score: float = 0.5
    logic_issue: str = ""
    data_issue: str = ""
    insights_issue: str = ""
    conversation_context: str = ""
    answerability: str = "ANSWERABLE"
    answerability_reason: str = ""

class StopExecution(Exception):
    def __init__(self, state):
        self.state = state

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, name, llm):
        self.name = name
        self.llm = llm
    
    @abstractmethod
    async def execute(self, state: ExecutionState) -> ExecutionState:
        """Execute agent logic"""
        pass
   
    @staticmethod
    def stop_execution(state: ExecutionState, message: str) -> ExecutionState:
        state.error = message
        raise StopExecution(state)