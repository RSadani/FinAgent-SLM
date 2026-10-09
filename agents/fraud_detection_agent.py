from typing import Dict, Any, List
from loguru import logger
import random
from .base_agent import BaseAgent, AgentThought, AgentAction

class FraudDetectionAgent(BaseAgent):
    def __init__(self, agent_id: str, name: str = "FraudDetectionAgent", description: str = "Analyzes transactions for fraudulent patterns."):
        super().__init__(agent_id, name, description)
        self.register_tool("analyze_transaction", self.analyze_transaction, "Analyze a transaction's basic parameters.")
        self.register_tool("calculate_risk_score", self.calculate_risk_score, "Calculate risk score using PyTorch model (mock).")
        self.register_tool("check_patterns", self.check_patterns, "Check transaction against known fraud patterns.")
        self.register_tool("flag_suspicious", self.flag_suspicious, "Flag a transaction as suspicious and generate report.")
        self.current_tx = {}
        self.analysis_state = {}

    def analyze_transaction(self, tx_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Analyzing transaction {tx_data.get('id')}")
        amount = tx_data.get("amount", 0)
        return {"amount_high": amount > 10000, "is_international": tx_data.get("country") != "US"}

    def calculate_risk_score(self, tx_data: Dict[str, Any]) -> float:
        logger.info(f"Calculating risk score for transaction {tx_data.get('id')} using mock PyTorch model")
        base_risk = 0.1
        if tx_data.get("amount", 0) > 10000:
            base_risk += 0.4
        if tx_data.get("country") != "US":
            base_risk += 0.3
        return min(base_risk + random.uniform(0.0, 0.2), 1.0)

    def check_patterns(self, tx_data: Dict[str, Any]) -> List[str]:
        logger.info("Checking transaction for known fraud patterns")
        patterns = []
        if tx_data.get("time", "") == "night":
            patterns.append("late_night_transaction")
        if tx_data.get("merchant_category", "") == "crypto":
            patterns.append("high_risk_merchant")
        return patterns

    def flag_suspicious(self, tx_data: Dict[str, Any], risk_score: float, patterns: List[str]) -> Dict[str, Any]:
        logger.warning(f"Flagging transaction {tx_data.get('id')} as suspicious")
        report = {
            "transaction_id": tx_data.get("id"),
            "risk_score": risk_score,
            "patterns": patterns,
            "status": "SUSPICIOUS"
        }
        self.log_audit("flag_suspicious", details=report, risk_level="high", evidence=patterns)
        return report

    def think(self, observation: Dict[str, Any]) -> AgentThought:
        if "transaction" in observation and not self.analysis_state:
            self.current_tx = observation["transaction"]
            self.analysis_state = {"step": "analyze"}
            return AgentThought(self.agent_id, "New transaction received. Starting analysis.", "analyze")
        
        step = self.analysis_state.get("step")
        
        if step == "analyze":
            return AgentThought(self.agent_id, "Basic analysis complete. Calculating risk score.", "score")
        elif step == "score":
            return AgentThought(self.agent_id, "Risk score calculated. Checking patterns.", "patterns")
        elif step == "patterns":
            if self.analysis_state.get("risk_score", 0) > 0.6 or len(self.analysis_state.get("patterns", [])) > 0:
                return AgentThought(self.agent_id, "High risk detected. Flagging transaction.", "flag")
            else:
                return AgentThought(self.agent_id, "Transaction appears normal. Analysis complete.", "DONE")
        elif step == "flag":
            return AgentThought(self.agent_id, "Transaction flagged. Analysis complete.", "DONE")
            
        return AgentThought(self.agent_id, "Nothing to do.", "DONE")

    def act(self, thought: AgentThought) -> AgentAction:
        action_type = "tool_use"
        if thought.reasoning == "analyze":
            result = self.analyze_transaction(self.current_tx)
            self.analysis_state["step"] = "score"
            return AgentAction(self.agent_id, action_type, "analyze_transaction", {"tx_data": self.current_tx}, result=result)
        elif thought.reasoning == "score":
            result = self.calculate_risk_score(self.current_tx)
            self.analysis_state["risk_score"] = result
            self.analysis_state["step"] = "patterns"
            return AgentAction(self.agent_id, action_type, "calculate_risk_score", {"tx_data": self.current_tx}, result=result)
        elif thought.reasoning == "patterns":
            result = self.check_patterns(self.current_tx)
            self.analysis_state["patterns"] = result
            self.analysis_state["step"] = "flag"
            return AgentAction(self.agent_id, action_type, "check_patterns", {"tx_data": self.current_tx}, result=result)
        elif thought.reasoning == "flag":
            result = self.flag_suspicious(self.current_tx, self.analysis_state.get("risk_score", 0.0), self.analysis_state.get("patterns", []))
            self.analysis_state["step"] = "done"
            self.analysis_state["report"] = result
            return AgentAction(self.agent_id, action_type, "flag_suspicious", {"tx_data": self.current_tx}, result=result)
            
        return AgentAction(self.agent_id, "none", "none", {})

    def observe(self, action_result: Any) -> Dict[str, Any]:
        return {"last_result": action_result, "state": self.analysis_state}
