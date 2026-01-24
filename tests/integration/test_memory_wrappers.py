from multi_agent_framework.memory import WorkingMemory


def test_working_memory_plan_roundtrip(tmp_path):
    db_path = tmp_path / "task_memory.db"
    memory = WorkingMemory(db_path=str(db_path))
    workflow_id = memory.start_workflow()

    plan = [{"id": "task_1", "description": "Do thing"}]
    memory.save_plan(workflow_id, plan)
    stored = memory.get_plan(workflow_id)

    assert stored == plan
