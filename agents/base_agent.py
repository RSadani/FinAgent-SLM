from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
from loguru import logger

class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class AgentAction:
    agent_id: str
    action_type: str
    tool_name: str
    tool_input: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    result: Optional[Any] = None

@dataclass
class AgentThought:
    agent_id: str
    thought: str
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass 
class AuditEntry:
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    agent_id: str = ""
    action: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    risk_level: str = "low"
    evidence: List[str] = field(default_factory=list)

class BaseAgent(ABC):
    def __init__(self, agent_id: str, name: str, description: str):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.state = AgentState.IDLE
        self.memory: List[Dict[str, Any]] = []
        self.audit_trail: List[AuditEntry] = []
        self.tools: Dict[str, callable] = {}
        self.action_history: List[AgentAction] = []
        logger.info(f"Agent '{name}' ({agent_id}) initialized")

    def register_tool(self, name: str, func: callable, description: str = ""):
        self.tools[name] = {"func": func, "description": description}
        logger.debug(f"Tool '{name}' registered for agent '{self.name}'")

    def add_to_memory(self, item: Dict[str, Any]):
        self.memory.append({**item, "timestamp": datetime.now().isoformat()})

    def log_audit(self, action: str, details: Dict[str, Any], risk_level: str = "low", evidence: List[str] = None):
        entry = AuditEntry(agent_id=self.agent_id, action=action, details=details, risk_level=risk_level, evidence=evidence or [])
        self.audit_trail.append(entry)
        return entry

    @abstractmethod
    def think(self, observation: Dict[str, Any]) -> AgentThought:
        pass

    @abstractmethod
    def act(self, thought: AgentThought) -> AgentAction:
        pass

    @abstractmethod
    def observe(self, action_result: Any) -> Dict[str, Any]:
        pass

    def react_loop(self, initial_input: Dict[str, Any], max_iterations: int = 10) -> Dict[str, Any]:
        """Execute the ReAct (Reason + Act) loop."""
        observation = initial_input
        results = []
        for i in range(max_iterations):
            logger.info(f"[{self.name}] ReAct iteration {i+1}/{max_iterations}")
            self.state = AgentState.THINKING
            thought = self.think(observation)
            if thought.reasoning == "DONE":
                self.state = AgentState.COMPLETED
                break
            self.state = AgentState.ACTING
            action = self.act(thought)
            self.action_history.append(action)
            self.state = AgentState.OBSERVING
            observation = self.observe(action.result)
            results.append({"thought": thought, "action": action, "observation": observation})
        self.state = AgentState.COMPLETED
        return {"results": results, "audit_trail": self.audit_trail}
