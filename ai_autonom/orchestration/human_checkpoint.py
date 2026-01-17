"""
Human-in-the-Loop (HITL) Checkpoint System

Provides approval workflow for critical security actions, synthesis steps,
and error recovery decisions.
"""

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class CheckpointType(Enum):
    """Types of checkpoints requiring human approval"""
    CRITICAL = "critical"  # High-risk security actions
    SYNTHESIS = "synthesis"  # Before final output combination
    ERROR = "error"  # After retry exhaustion
    VALIDATION = "validation"  # Before executing validated exploits
    CONFIGURATION = "configuration"  # System configuration changes


class CheckpointDecision(Enum):
    """Possible human decisions at checkpoint"""
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    SKIP = "skip"


class RiskLevel(Enum):
    """Risk assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CheckpointContext:
    """Context information for a checkpoint"""
    checkpoint_id: str
    checkpoint_type: CheckpointType
    task_id: str
    agent_id: str
    description: str
    agent_output: str
    risk_level: RiskLevel
    risk_assessment: str
    proposed_action: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CheckpointResult:
    """Result of checkpoint approval process"""
    checkpoint_id: str
    decision: CheckpointDecision
    approved: bool
    rejection_reason: Optional[str] = None
    modifications: Optional[Dict[str, Any]] = None
    response_time_sec: float = 0.0
    reviewer: str = "human"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class HumanCheckpointManager:
    """
    Manages human-in-the-loop checkpoints for critical decisions.
    
    Integrates with LangGraph workflows to pause execution and request
    human approval before proceeding with high-risk actions.
    """
    
    def __init__(
        self,
        db_path: str = ".runtime/data/checkpoints.db",
        auto_approve_low_risk: bool = False,
        timeout_sec: Optional[int] = None
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            db_path: Path to SQLite database for checkpoint logging
            auto_approve_low_risk: Automatically approve LOW risk checkpoints
            timeout_sec: Maximum wait time for human response (None = no limit)
        """
        self.db_path = Path(db_path)
        self.auto_approve_low_risk = auto_approve_low_risk
        self.timeout_sec = timeout_sec
        self.lock = threading.RLock()
        
        # Checkpoint callbacks for custom handling
        self.callbacks: Dict[CheckpointType, List[Callable]] = {
            ctype: [] for ctype in CheckpointType
        }
        
        # Active checkpoints waiting for approval
        self.pending_checkpoints: Dict[str, CheckpointContext] = {}
        
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database for checkpoint logging"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Checkpoints table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                checkpoint_type TEXT NOT NULL,
                task_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                description TEXT,
                risk_level TEXT NOT NULL,
                risk_assessment TEXT,
                proposed_action TEXT,
                decision TEXT,
                approved INTEGER,
                rejection_reason TEXT,
                response_time_sec REAL,
                reviewer TEXT,
                timestamp TEXT NOT NULL,
                metadata TEXT
            )
        """)
        
        # Checkpoint history for analytics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkpoint_stats (
                checkpoint_type TEXT,
                risk_level TEXT,
                total_count INTEGER DEFAULT 0,
                approved_count INTEGER DEFAULT 0,
                rejected_count INTEGER DEFAULT 0,
                modified_count INTEGER DEFAULT 0,
                skipped_count INTEGER DEFAULT 0,
                avg_response_time_sec REAL,
                PRIMARY KEY (checkpoint_type, risk_level)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def register_callback(
        self,
        checkpoint_type: CheckpointType,
        callback: Callable[[CheckpointContext], None]
    ) -> None:
        """
        Register callback to execute when checkpoint is triggered.
        
        Args:
            checkpoint_type: Type of checkpoint to listen for
            callback: Function to call with checkpoint context
        """
        with self.lock:
            self.callbacks[checkpoint_type].append(callback)
            logger.info(f"Registered callback for {checkpoint_type.value} checkpoints")
    
    def should_checkpoint(self, task: Dict, risk_level: RiskLevel = RiskLevel.LOW) -> bool:
        """
        Determine if a task requires human checkpoint.
        
        Args:
            task: Task dictionary
            risk_level: Assessed risk level
            
        Returns:
            True if checkpoint required
        """
        # Auto-approve if configured for low risk
        if self.auto_approve_low_risk and risk_level == RiskLevel.LOW:
            return False
        
        # Always checkpoint for high/critical risk
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            return True
        
        # Medium risk - check task type
        if risk_level == RiskLevel.MEDIUM:
            # Checkpoint for destructive operations
            destructive_keywords = ['delete', 'remove', 'drop', 'rm ', 'destroy']
            task_desc = task.get('description', '').lower()
            if any(kw in task_desc for kw in destructive_keywords):
                return True
        
        return False
    
    def request_approval(
        self,
        context: CheckpointContext,
        interactive: bool = True
    ) -> CheckpointResult:
        """
        Request human approval for a checkpoint.
        
        Args:
            context: Checkpoint context with task details
            interactive: If True, prompt user for input; if False, use auto-approval rules
        
        Returns:
            CheckpointResult with human decision
        """
        start_time = time.time()
        
        # Auto-approve low-risk if configured
        if self.auto_approve_low_risk and context.risk_level == RiskLevel.LOW:
            logger.info(f"Auto-approving LOW risk checkpoint: {context.checkpoint_id}")
            result = CheckpointResult(
                checkpoint_id=context.checkpoint_id,
                decision=CheckpointDecision.APPROVE,
                approved=True,
                response_time_sec=time.time() - start_time,
                reviewer="auto"
            )
            self._log_checkpoint(context, result)
            return result
        
        # Execute callbacks
        with self.lock:
            for callback in self.callbacks.get(context.checkpoint_type, []):
                try:
                    callback(context)
                except Exception as e:
                    logger.error(f"Checkpoint callback failed: {e}")
        
        # Store as pending
        with self.lock:
            self.pending_checkpoints[context.checkpoint_id] = context
        
        # Interactive approval
        if interactive:
            result = self._prompt_user(context, start_time)
        else:
            # Non-interactive mode: return pending status
            result = CheckpointResult(
                checkpoint_id=context.checkpoint_id,
                decision=CheckpointDecision.APPROVE,
                approved=False,
                response_time_sec=time.time() - start_time,
                reviewer="pending"
            )
        
        # Log checkpoint
        self._log_checkpoint(context, result)
        
        # Remove from pending
        with self.lock:
            self.pending_checkpoints.pop(context.checkpoint_id, None)
        
        return result
    
    def _prompt_user(
        self,
        context: CheckpointContext,
        start_time: float
    ) -> CheckpointResult:
        """
        Display checkpoint and prompt user for decision.
        
        Args:
            context: Checkpoint context
            start_time: Timestamp when approval was requested
        
        Returns:
            CheckpointResult with user decision
        """
        # Display checkpoint information
        print("\n" + "=" * 80)
        print(f"CHECKPOINT REQUIRED: {context.checkpoint_type.value.upper()}")
        print("=" * 80)
        print(f"Task ID: {context.task_id}")
        print(f"Agent: {context.agent_id}")
        print(f"Risk Level: {context.risk_level.value.upper()}")
        print(f"\nDescription:")
        print(f"  {context.description}")
        print(f"\nRisk Assessment:")
        print(f"  {context.risk_assessment}")
        print(f"\nProposed Action:")
        print(f"  {context.proposed_action}")
        print(f"\nAgent Output:")
        print(f"  {context.agent_output[:500]}..." if len(context.agent_output) > 500 else f"  {context.agent_output}")
        print("\n" + "-" * 80)
        print("Options:")
        print("  [A] Approve - Proceed with proposed action")
        print("  [R] Reject - Halt workflow and log reason")
        print("  [M] Modify - Provide modifications to action")
        print("  [S] Skip - Skip this step and continue")
        print("=" * 80)
        
        # Prompt for decision
        while True:
            try:
                choice = input("\nYour decision [A/R/M/S]: ").strip().upper()
                
                if choice == 'A':
                    return CheckpointResult(
                        checkpoint_id=context.checkpoint_id,
                        decision=CheckpointDecision.APPROVE,
                        approved=True,
                        response_time_sec=time.time() - start_time
                    )
                
                elif choice == 'R':
                    reason = input("Rejection reason: ").strip()
                    return CheckpointResult(
                        checkpoint_id=context.checkpoint_id,
                        decision=CheckpointDecision.REJECT,
                        approved=False,
                        rejection_reason=reason,
                        response_time_sec=time.time() - start_time
                    )
                
                elif choice == 'M':
                    print("Enter modifications (JSON format or key=value pairs):")
                    modifications_input = input().strip()
                    
                    # Parse modifications
                    modifications = {}
                    try:
                        import json
                        modifications = json.loads(modifications_input)
                    except json.JSONDecodeError:
                        # Try key=value format
                        for pair in modifications_input.split(','):
                            if '=' in pair:
                                key, value = pair.split('=', 1)
                                modifications[key.strip()] = value.strip()
                    
                    return CheckpointResult(
                        checkpoint_id=context.checkpoint_id,
                        decision=CheckpointDecision.MODIFY,
                        approved=True,
                        modifications=modifications,
                        response_time_sec=time.time() - start_time
                    )
                
                elif choice == 'S':
                    return CheckpointResult(
                        checkpoint_id=context.checkpoint_id,
                        decision=CheckpointDecision.SKIP,
                        approved=True,
                        response_time_sec=time.time() - start_time
                    )
                
                else:
                    print("Invalid choice. Please enter A, R, M, or S.")
            
            except KeyboardInterrupt:
                print("\nCheckpoint interrupted. Rejecting by default.")
                return CheckpointResult(
                    checkpoint_id=context.checkpoint_id,
                    decision=CheckpointDecision.REJECT,
                    approved=False,
                    rejection_reason="User interrupted",
                    response_time_sec=time.time() - start_time
                )
    
    def _log_checkpoint(
        self,
        context: CheckpointContext,
        result: CheckpointResult
    ) -> None:
        """Log checkpoint to database"""
        import json
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO checkpoints (
                checkpoint_id, checkpoint_type, task_id, agent_id,
                description, risk_level, risk_assessment, proposed_action,
                decision, approved, rejection_reason, response_time_sec,
                reviewer, timestamp, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            context.checkpoint_id,
            context.checkpoint_type.value,
            context.task_id,
            context.agent_id,
            context.description,
            context.risk_level.value,
            context.risk_assessment,
            context.proposed_action,
            result.decision.value,
            1 if result.approved else 0,
            result.rejection_reason,
            result.response_time_sec,
            result.reviewer,
            result.timestamp,
            json.dumps(context.metadata)
        ))
        
        # Update statistics
        cursor.execute("""
            INSERT INTO checkpoint_stats (
                checkpoint_type, risk_level, total_count,
                approved_count, rejected_count, modified_count, skipped_count
            ) VALUES (?, ?, 1, ?, ?, ?, ?)
            ON CONFLICT(checkpoint_type, risk_level) DO UPDATE SET
                total_count = total_count + 1,
                approved_count = approved_count + excluded.approved_count,
                rejected_count = rejected_count + excluded.rejected_count,
                modified_count = modified_count + excluded.modified_count,
                skipped_count = skipped_count + excluded.skipped_count
        """, (
            context.checkpoint_type.value,
            context.risk_level.value,
            1 if result.decision == CheckpointDecision.APPROVE else 0,
            1 if result.decision == CheckpointDecision.REJECT else 0,
            1 if result.decision == CheckpointDecision.MODIFY else 0,
            1 if result.decision == CheckpointDecision.SKIP else 0
        ))
        
        conn.commit()
        conn.close()
    
    def get_checkpoint_stats(
        self,
        checkpoint_type: Optional[CheckpointType] = None
    ) -> Dict[str, Any]:
        """
        Get checkpoint statistics.
        
        Args:
            checkpoint_type: Filter by specific type (None = all types)
        
        Returns:
            Statistics dictionary
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        if checkpoint_type:
            cursor.execute("""
                SELECT checkpoint_type, risk_level, total_count,
                       approved_count, rejected_count, modified_count, skipped_count
                FROM checkpoint_stats
                WHERE checkpoint_type = ?
            """, (checkpoint_type.value,))
        else:
            cursor.execute("""
                SELECT checkpoint_type, risk_level, total_count,
                       approved_count, rejected_count, modified_count, skipped_count
                FROM checkpoint_stats
            """)
        
        rows = cursor.fetchall()
        conn.close()
        
        stats = {
            "total_checkpoints": 0,
            "by_type": {},
            "by_risk": {},
            "approval_rate": 0.0
        }
        
        total_approved = 0
        for row in rows:
            ctype, risk, total, approved, rejected, modified, skipped = row
            
            stats["total_checkpoints"] += total
            total_approved += approved
            
            if ctype not in stats["by_type"]:
                stats["by_type"][ctype] = {
                    "total": 0,
                    "approved": 0,
                    "rejected": 0,
                    "modified": 0,
                    "skipped": 0
                }
            
            stats["by_type"][ctype]["total"] += total
            stats["by_type"][ctype]["approved"] += approved
            stats["by_type"][ctype]["rejected"] += rejected
            stats["by_type"][ctype]["modified"] += modified
            stats["by_type"][ctype]["skipped"] += skipped
            
            if risk not in stats["by_risk"]:
                stats["by_risk"][risk] = total
            else:
                stats["by_risk"][risk] += total
        
        if stats["total_checkpoints"] > 0:
            stats["approval_rate"] = total_approved / stats["total_checkpoints"]
        
        return stats
    
    def get_pending_checkpoints(self) -> List[CheckpointContext]:
        """Get list of checkpoints awaiting approval"""
        with self.lock:
            return list(self.pending_checkpoints.values())


# Helper functions for common checkpoint scenarios

def create_critical_action_checkpoint(
    task_id: str,
    agent_id: str,
    action_description: str,
    risk_assessment: str,
    agent_output: str = ""
) -> CheckpointContext:
    """Create checkpoint for critical security action"""
    import uuid
    
    return CheckpointContext(
        checkpoint_id=f"cp_{uuid.uuid4().hex[:8]}",
        checkpoint_type=CheckpointType.CRITICAL,
        task_id=task_id,
        agent_id=agent_id,
        description=f"Critical action requires approval: {action_description}",
        agent_output=agent_output,
        risk_level=RiskLevel.CRITICAL,
        risk_assessment=risk_assessment,
        proposed_action=action_description
    )


def create_synthesis_checkpoint(
    task_id: str,
    agent_id: str,
    outputs_to_combine: List[str],
    synthesis_plan: str
) -> CheckpointContext:
    """Create checkpoint before final output synthesis"""
    import uuid
    
    return CheckpointContext(
        checkpoint_id=f"cp_{uuid.uuid4().hex[:8]}",
        checkpoint_type=CheckpointType.SYNTHESIS,
        task_id=task_id,
        agent_id=agent_id,
        description="Ready to synthesize final output from multiple agent results",
        agent_output="\n\n".join(outputs_to_combine),
        risk_level=RiskLevel.MEDIUM,
        risk_assessment="Combining outputs may introduce inconsistencies",
        proposed_action=synthesis_plan
    )


def create_error_recovery_checkpoint(
    task_id: str,
    agent_id: str,
    error_message: str,
    recovery_strategy: str,
    attempt_number: int
) -> CheckpointContext:
    """Create checkpoint after retry exhaustion"""
    import uuid
    
    return CheckpointContext(
        checkpoint_id=f"cp_{uuid.uuid4().hex[:8]}",
        checkpoint_type=CheckpointType.ERROR,
        task_id=task_id,
        agent_id=agent_id,
        description=f"Error recovery failed after {attempt_number} attempts",
        agent_output=error_message,
        risk_level=RiskLevel.HIGH,
        risk_assessment="Task execution has failed multiple times",
        proposed_action=f"Retry with strategy: {recovery_strategy}",
        metadata={"attempt_number": attempt_number}
    )
