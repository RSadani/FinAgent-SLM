from typing import Dict, Any, List
from loguru import logger
from datetime import datetime
from .base_agent import BaseAgent, AgentThought, AgentAction

class AuditTrailAgent(BaseAgent):
    def __init__(self, agent_id: str, name: str = "AuditTrailAgent", description: str = "Generates compliance-ready audit documentation."):
        super().__init__(agent_id, name, description)
        self.register_tool("collect_evidence", self.collect_evidence, "Collect all relevant evidence for the case.")
        self.register_tool("build_timeline", self.build_timeline, "Build a chronological timeline of events.")
        self.register_tool("generate_report", self.generate_report, "Generate an internal audit report.")
        self.register_tool("create_sar", self.create_sar, "Create a Suspicious Activity Report for regulators.")
        self.current_case = {}
        self.audit_state = {}

    def collect_evidence(self, case_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        logger.info(f"Collecting evidence for case {case_data.get('case_id')}")
        evidence = [{"type": "transaction_record", "data": case_data.get("transaction", {})}]
        if "patterns" in case_data:
            evidence.append({"type": "detected_patterns", "data": case_data["patterns"]})
        if "tax_analysis" in case_data:
            evidence.append({"type": "tax_discrepancies", "data": case_data["tax_analysis"]})
        return evidence

    def build_timeline(self, case_data: Dict[str, Any], evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info("Building case timeline")
        timeline = []
        tx_time = case_data.get("transaction", {}).get("timestamp", datetime.now().isoformat())
        timeline.append({"timestamp": tx_time, "event": "Transaction occurred"})
        timeline.append({"timestamp": datetime.now().isoformat(), "event": "Investigation initiated"})
        return timeline

    def generate_report(self, case_data: Dict[str, Any], evidence: List[Dict[str, Any]], timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info("Generating internal audit report")
        report = {
            "case_id": case_data.get("case_id"),
            "summary": "Internal audit report generated.",
            "evidence_count": len(evidence),
            "timeline_events": len(timeline),
            "conclusion": "Requires further review." if case_data.get("risk_score", 0) > 0.8 else "Logged for record."
        }
        return report

    def create_sar(self, case_data: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Creating Suspicious Activity Report (SAR)")
        sar = {
            "sar_id": f"SAR-{case_data.get('case_id', 'unknown')}",
            "filing_date": datetime.now().isoformat(),
            "subject_info": case_data.get("transaction", {}).get("user_id", "UNKNOWN"),
            "suspicious_activity_info": report,
            "status": "READY_FOR_FILING"
        }
        self.log_audit("create_sar", details=sar, risk_level="critical", evidence=["Transaction involves significant risk"])
        return sar

    def think(self, observation: Dict[str, Any]) -> AgentThought:
        if "case_data" in observation and not self.audit_state:
            self.current_case = observation["case_data"]
            self.audit_state = {"step": "collect_evidence"}
            return AgentThought(self.agent_id, "New case received. Collecting evidence.", "collect_evidence")
        
        step = self.audit_state.get("step")
        if step == "collect_evidence":
            return AgentThought(self.agent_id, "Evidence collected. Building timeline.", "build_timeline")
        elif step == "build_timeline":
            return AgentThought(self.agent_id, "Timeline built. Generating report.", "generate_report")
        elif step == "generate_report":
            if self.current_case.get("risk_score", 0) > 0.8:
                return AgentThought(self.agent_id, "High risk case. Creating SAR.", "create_sar")
            return AgentThought(self.agent_id, "Report generated. Audit complete.", "DONE")
        elif step == "create_sar":
            return AgentThought(self.agent_id, "SAR created. Audit complete.", "DONE")
            
        return AgentThought(self.agent_id, "Nothing to do.", "DONE")

    def act(self, thought: AgentThought) -> AgentAction:
        if thought.reasoning == "collect_evidence":
            res = self.collect_evidence(self.current_case)
            self.audit_state["evidence"] = res
            self.audit_state["step"] = "build_timeline"
            return AgentAction(self.agent_id, "tool_use", "collect_evidence", {"case_data": self.current_case}, result=res)
        elif thought.reasoning == "build_timeline":
            evidence = self.audit_state.get("evidence", [])
            res = self.build_timeline(self.current_case, evidence)
            self.audit_state["timeline"] = res
            self.audit_state["step"] = "generate_report"
            return AgentAction(self.agent_id, "tool_use", "build_timeline", {"case_data": self.current_case, "evidence": evidence}, result=res)
        elif thought.reasoning == "generate_report":
            evidence = self.audit_state.get("evidence", [])
            timeline = self.audit_state.get("timeline", [])
            res = self.generate_report(self.current_case, evidence, timeline)
            self.audit_state["report"] = res
            self.audit_state["step"] = "create_sar"
            return AgentAction(self.agent_id, "tool_use", "generate_report", {"case_data": self.current_case, "evidence": evidence, "timeline": timeline}, result=res)
        elif thought.reasoning == "create_sar":
            report = self.audit_state.get("report", {})
            res = self.create_sar(self.current_case, report)
            self.audit_state["sar"] = res
            self.audit_state["step"] = "done"
            return AgentAction(self.agent_id, "tool_use", "create_sar", {"case_data": self.current_case, "report": report}, result=res)
            
        return AgentAction(self.agent_id, "none", "none", {})

    def observe(self, action_result: Any) -> Dict[str, Any]:
        return {"last_result": action_result, "state": self.audit_state}
