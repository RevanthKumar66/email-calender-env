import pytest
from env.grader import EasyTaskGrader
from env.tasks.task_easy import generate_easy_task

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
