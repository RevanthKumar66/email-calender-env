import pytest
from env.email_calendar_env import EmailCalendarEnv
from env.models import Action

def test_reset():
    env = EmailCalendarEnv("easy")
    obs = env.reset()
    assert obs.task_id == "easy"
    assert len(obs.inbox_emails) == 10

def test_step():
    env = EmailCalendarEnv("easy")
    env.reset()
    action = Action(action_type="no_op")
    result = env.step(action)
    assert result.reward == -0.01
    assert result.done == False
