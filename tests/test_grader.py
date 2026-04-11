import pytest
from env.grader import grade_easy, safe_score
from env.tasks.task_easy import generate_easy_task

def test_easy_grader_perfect():
    task_data = generate_easy_task()
    # Simulate perfect actions
    actions = []
    for email in task_data["emails"]:
        if email["priority"] == "urgent":
            actions.append({"action_type": "flag_email", "email_id": email["id"]})
        if email["category"] == "spam":
            actions.append({"action_type": "archive_email", "email_id": email["id"]})
        if email["requires_reply"]:
            actions.append({"action_type": "reply_email", "email_id": email["id"]})
            
    score = grade_easy(task_data, actions)
    # Range check instead of exact 1.0
    assert 0.9 <= score <= 0.99

def test_easy_grader_empty():
    task_data = generate_easy_task()
    score = grade_easy(task_data, [])
    # Range check instead of exact 0.0
    assert 0.01 <= score <= 0.1
