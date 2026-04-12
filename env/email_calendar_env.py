import os, json, uuid
from typing import Optional, List, Tuple
from datetime import datetime
from env.models import Observation, Action, StepResult, Email, CalendarEvent
from env.tasks.task_easy import generate_easy_task
from env.tasks.task_medium import generate_medium_task
from env.tasks.task_hard import generate_hard_task
from env.grader import safe_score

class EmailCalendarEnv:
    def __init__(self, task_id: str = "easy"):
        self.task_id = str(task_id).lower()
        self._task_data = None
        self._step_count = 0
        self._done = False
        self._actions_taken = []
        self._calendar_events = []
        self._inbox = []

    def reset(self) -> Observation:
        self._step_count = 0
        self._actions_taken = []
        self._done = False
        
        if "medium" in self.task_id: self._task_data = generate_medium_task()
        elif "hard" in self.task_id: self._task_data = generate_hard_task()
        else: self._task_data = generate_easy_task()

        self._inbox = [Email(**e) if isinstance(e, dict) else e for e in self._task_data.get("emails", [])]
        self._calendar_events = [CalendarEvent(**e) if isinstance(e, dict) else e for e in self._task_data.get("calendar", [])]
        
        return self._build_observation(self.score())

    def step(self, action: Action) -> StepResult:
        self._step_count += 1
        reward, info = self._apply_action(action)
        self._actions_taken.append(action.model_dump())
        
        current_score = self.score()
        self._done = self._step_count >= 15 or len(self._inbox) == 0
        
        return StepResult(observation=self._build_observation(current_score), reward=reward, done=self._done, info=info)

    def score(self) -> float:
        raw = 0.1 + (len(self._actions_taken) * 0.05)
        return safe_score(raw)

    def state(self) -> dict:
        """Mandatory state mapping for the external platform grader."""
        return {
            "task_id": self.task_id,
            "step": self._step_count,
            "current_score": self.score(),
            "inbox_emails": [e.model_dump() for e in self._inbox],
            "calendar_events": [c.model_dump() for c in self._calendar_events]
        }

    def _apply_action(self, action: Action) -> Tuple[float, dict]:
        if action.action_type == "no_op": return -0.01, {"m": "no_op"}
        reward = 0.05
        email = next((e for e in self._inbox if e.id == action.email_id), None)
        if email:
            self._inbox = [e for e in self._inbox if e.id != action.email_id]
            reward = 0.2
        return reward, {"status": "success"}

    def _build_observation(self, score_so_far: float) -> Observation:
        # Standardized keys
        return Observation(
            inbox_emails=self._inbox,
            calendar_events=self._calendar_events,
            current_step=self._step_count,
            max_steps=15,
            task_id=self.task_id,
            task_objective=self._task_data.get("objective", "") if self._task_data else "",
            score_so_far=score_so_far
        )

    def close(self): pass
