import pytest
from env.grader import EasyTaskGrader, MediumTaskGrader, HardTaskGrader
from env.tasks.task_easy import generate_easy_task
from env.tasks.task_medium import generate_medium_task
from env.tasks.task_hard import generate_hard_task

def test_easy_grader_perfect():
    task_data = generate_easy_task()
    grader = EasyTaskGrader(task_data)
    
    # Simulate perfect actions
    actions = []
    for email in task_data["emails"]:
        if email["priority"] == "urgent":
            actions.append({"action_type": "flag_email", "email_id": email["id"]})
        if email["category"] == "spam":
            actions.append({"action_type": "archive_email", "email_id": email["id"]})
        if email["requires_reply"]:
            actions.append({"action_type": "reply_email", "email_id": email["id"]})
            
    score = grader.score(actions)
    assert score == 1.0

def test_easy_grader_empty():
    task_data = generate_easy_task()
    grader = EasyTaskGrader(task_data)
    score = grader.score([])
    assert score == 0.0

def test_medium_grader_perfect():
    task_data = generate_medium_task()
    grader = MediumTaskGrader(task_data)
    actions = []
    for email in task_data["emails"]:
        if email["requires_reply"]:
            actions.append({"action_type": "reply_email", "email_id": email["id"]})
    for _ in range(task_data.get("required_meetings", 5)):
        actions.append({"action_type": "schedule_meeting"})
    score = grader.score(actions)
    assert score == 1.0

def test_medium_grader_empty():
    task_data = generate_medium_task()
    grader = MediumTaskGrader(task_data)
    score = grader.score([])
    assert score == 0.0

def test_hard_grader_perfect():
    task_data = generate_hard_task()
    grader = HardTaskGrader(task_data)
    actions = []
    for email in task_data["emails"]:
        if email["requires_reply"]:
            actions.append({"action_type": "reply_email", "email_id": email["id"]})
    for _ in range(task_data.get("required_meetings", 10)):
        actions.append({"action_type": "schedule_meeting"})
    score = grader.score(actions)
    assert score == 1.0

def test_hard_grader_empty():
    task_data = generate_hard_task()
    grader = HardTaskGrader(task_data)
    score = grader.score([])
    assert score == 0.0
