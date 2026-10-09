from typing import Dict, Any, List
from loguru import logger
import time

from .fraud_detection_agent import FraudDetectionAgent
from .tax_analysis_agent import TaxAnalysisAgent
from .audit_trail_agent import AuditTrailAgent

class FinAgentOrchestrator:
    def __init__(self, model=None, device: str = "cpu"):
        logger.info("Initializing FinAgent Orchestrator")
        self.model = model
        self.device = device
        self.fraud_agent = FraudDetectionAgent(agent_id="fraud_01")
        self.tax_agent = TaxAnalysisAgent(agent_id="tax_01")
        self.audit_agent = AuditTrailAgent(agent_id="audit_01")
        self.metrics = {"processed_count": 0, "suspicious_count": 0, "total_time_ms": 0}
        self._all_audit_entries: list = []

    def process_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        tx_id = transaction.get("id", "unknown")
        logger.info(f"Orchestrator processing transaction {tx_id}")
        
        # 1. Fraud Detection
        self.fraud_agent.analysis_state = {} # reset state for new run
        fraud_result = self.fraud_agent.react_loop({"transaction": transaction})
        fraud_state = fraud_result["results"][-1]["observation"]["state"] if fraud_result["results"] else {}
        is_suspicious = fraud_state.get("report", {}).get("status") == "SUSPICIOUS"
        
        case_data = {
            "case_id": f"CASE-{tx_id}",
            "transaction": transaction,
            "risk_score": fraud_state.get("risk_score", 0.0),
            "patterns": fraud_state.get("patterns", [])
        }
        
        # 2. Tax Analysis (if suspicious)
        tax_result = None
        if is_suspicious:
            logger.info(f"Transaction {tx_id} marked suspicious, routing to TaxAnalysisAgent")
            self.tax_agent.tax_state = {} # reset state
            tax_result = self.tax_agent.react_loop({"case_data": case_data})
            tax_state = tax_result["results"][-1]["observation"]["state"] if tax_result["results"] else {}
            case_data["tax_analysis"] = tax_state
        
        # 3. Audit Trail Generation
        logger.info(f"Generating audit trail for transaction {tx_id}")
        self.audit_agent.audit_state = {} # reset state
        audit_result = self.audit_agent.react_loop({"case_data": case_data})
        audit_state = audit_result["results"][-1]["observation"]["state"] if audit_result["results"] else {}
        
        # Aggregate audit trails
        combined_audit = []
        combined_audit.extend([a.__dict__ for a in fraud_result["audit_trail"]])
        if tax_result:
            combined_audit.extend([a.__dict__ for a in tax_result["audit_trail"]])
        combined_audit.extend([a.__dict__ for a in audit_result["audit_trail"]])
        self._all_audit_entries.extend(combined_audit)
        
        end_time = time.time()
        process_time_ms = (end_time - start_time) * 1000
        
        self.metrics["processed_count"] += 1
        self.metrics["total_time_ms"] += process_time_ms
        if is_suspicious:
            self.metrics["suspicious_count"] += 1
            
        return {
            "transaction_id": tx_id,
            "is_suspicious": is_suspicious,
            "fraud_analysis": fraud_state,
            "tax_analysis": case_data.get("tax_analysis"),
            "audit_summary": audit_state.get("report"),
            "sar": audit_state.get("sar"),
            "combined_audit_trail": combined_audit,
            "process_time_ms": process_time_ms
        }

    def process_batch(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"Processing batch of {len(transactions)} transactions")
        results = []
        for tx in transactions:
            results.append(self.process_transaction(tx))
        return results

    def get_metrics(self) -> Dict[str, Any]:
        avg_time = 0
        if self.metrics["processed_count"] > 0:
            avg_time = self.metrics["total_time_ms"] / self.metrics["processed_count"]
        return {
            "processed_count": self.metrics["processed_count"],
            "suspicious_count": self.metrics["suspicious_count"],
            "avg_process_time_ms": avg_time
        }

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Return the complete accumulated audit trail from all processed transactions."""
        return self._all_audit_entries
