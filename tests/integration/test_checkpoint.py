"""
Tests for Human Checkpoint Manager
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestration.human_checkpoint import (
    HumanCheckpointManager,
    CheckpointType,
    CheckpointDecision,
    RiskLevel,
    CheckpointContext,
    create_critical_action_checkpoint,
    create_synthesis_checkpoint,
    create_error_recovery_checkpoint
)


def test_checkpoint_creation():
    """Test checkpoint context creation"""
    cp = create_critical_action_checkpoint(
        task_id="test_task_1",
        agent_id="web_pentester",
        action_description="Execute SQL injection attack on target database",
        risk_assessment="High risk: May cause data corruption or unauthorized access"
    )
    
    assert cp.checkpoint_type == CheckpointType.CRITICAL
    assert cp.task_id == "test_task_1"
    assert cp.agent_id == "web_pentester"
    assert cp.risk_level == RiskLevel.CRITICAL
    assert "SQL injection" in cp.proposed_action


def test_auto_approve_low_risk():
    """Test automatic approval of low-risk checkpoints"""
    import uuid
    
    manager = HumanCheckpointManager(auto_approve_low_risk=True)
    
    context = CheckpointContext(
        checkpoint_id=f"cp_{uuid.uuid4().hex[:8]}",
        checkpoint_type=CheckpointType.VALIDATION,
        task_id="test_task_2",
        agent_id="retester",
        description="Validate previous scan results",
        agent_output="All scans completed successfully",
        risk_level=RiskLevel.LOW,
        risk_assessment="No risk: Read-only validation",
        proposed_action="Review and confirm findings"
    )
    
    result = manager.request_approval(context, interactive=False)
    
    assert result.approved == True
    assert result.decision == CheckpointDecision.APPROVE
    assert result.reviewer == "auto"


def test_checkpoint_callback():
    """Test callback execution on checkpoint"""
    manager = HumanCheckpointManager()
    
    callback_executed = []
    
    def test_callback(ctx: CheckpointContext):
        callback_executed.append(ctx.checkpoint_id)
    
    manager.register_callback(CheckpointType.CRITICAL, test_callback)
    
    cp = create_critical_action_checkpoint(
        task_id="test_task_3",
        agent_id="web_pentester",
        action_description="Test action",
        risk_assessment="Test risk"
    )
    
    # Non-interactive mode to avoid prompting
    manager.request_approval(cp, interactive=False)
    
    assert cp.checkpoint_id in callback_executed


def test_checkpoint_statistics():
    """Test checkpoint statistics tracking"""
    import uuid
    
    manager = HumanCheckpointManager()
    
    # Create and approve several checkpoints
    for i in range(3):
        context = CheckpointContext(
            checkpoint_id=f"cp_{uuid.uuid4().hex[:8]}",
            checkpoint_type=CheckpointType.CRITICAL,
            task_id=f"test_task_{i}",
            agent_id="web_pentester",
            description="Test checkpoint",
            agent_output="Test output",
            risk_level=RiskLevel.HIGH,
            risk_assessment="Test assessment",
            proposed_action="Test action"
        )
        
        manager.request_approval(context, interactive=False)
    
    stats = manager.get_checkpoint_stats()
    
    assert stats["total_checkpoints"] >= 3
    assert CheckpointType.CRITICAL.value in stats["by_type"]


def test_synthesis_checkpoint():
    """Test synthesis checkpoint creation"""
    outputs = [
        "Agent 1: Found 5 vulnerabilities",
        "Agent 2: Validated 3 exploits",
        "Agent 3: Generated remediation report"
    ]
    
    cp = create_synthesis_checkpoint(
        task_id="test_task_4",
        agent_id="report_agent",
        outputs_to_combine=outputs,
        synthesis_plan="Combine all findings into comprehensive report"
    )
    
    assert cp.checkpoint_type == CheckpointType.SYNTHESIS
    assert cp.risk_level == RiskLevel.MEDIUM
    assert "5 vulnerabilities" in cp.agent_output


def test_error_recovery_checkpoint():
    """Test error recovery checkpoint creation"""
    cp = create_error_recovery_checkpoint(
        task_id="test_task_5",
        agent_id="web_pentester",
        error_message="Connection timeout after 30 seconds",
        recovery_strategy="exponential_backoff",
        attempt_number=3
    )
    
    assert cp.checkpoint_type == CheckpointType.ERROR
    assert cp.risk_level == RiskLevel.HIGH
    assert cp.metadata["attempt_number"] == 3
    assert "exponential_backoff" in cp.proposed_action


if __name__ == "__main__":
    print("Running checkpoint manager tests...")
    test_checkpoint_creation()
    print("PASS: Checkpoint creation")
    
    test_auto_approve_low_risk()
    print("PASS: Auto-approve low risk")
    
    test_checkpoint_callback()
    print("PASS: Checkpoint callback")
    
    test_checkpoint_statistics()
    print("PASS: Checkpoint statistics")
    
    test_synthesis_checkpoint()
    print("PASS: Synthesis checkpoint")
    
    test_error_recovery_checkpoint()
    print("PASS: Error recovery checkpoint")
    
    print("\nAll tests passed!")
