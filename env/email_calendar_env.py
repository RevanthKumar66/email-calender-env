import os, json, uuid
from typing import Optional, List, Tuple
from datetime import datetime
from env.models import Observation, Action, StepResult, Email, CalendarEvent
from env.tasks.task_easy import generate_easy_task
from env.tasks.task_medium import generate_medium_task
from env.tasks.task_hard import generate_hard_task
from env.grader import EasyTaskGrader, MediumTaskGrader, HardTaskGrader

class EmailCalendarEnv:
    def __init__(self, task_id: str = "easy"):
        self.task_id = task_id
        self._task_data = None
        self._state = None
        self._step_count = 0
        self._done = False
        self._actions_taken = []
        self._calendar_events = []
        self._inbox = []
        self._grader = None
        self._flagged_ids = set()

    def reset(self) -> Observation:
        self._step_count = 0
        self._done = False
        self._actions_taken = []
        self._calendar_events = []
        self._flagged_ids = set()

        if self.task_id == "easy" or not self.task_id:
            self._task_data = generate_easy_task()
            self._grader = EasyTaskGrader(self._task_data)
        elif self.task_id == "medium":
            self._task_data = generate_medium_task()
            self._grader = MediumTaskGrader(self._task_data)
        elif self.task_id == "hard":
            self._task_data = generate_hard_task()
            self._grader = HardTaskGrader(self._task_data)
        else:
            raise ValueError(f"Unknown task_id: {self.task_id}")

        self._inbox = [Email(**e) for e in self._task_data["emails"]]
        self._calendar_events = [CalendarEvent(**c) for c in self._task_data.get("calendar", [])]

        return self._build_observation(score_so_far=0.0)

    def step(self, action: Action) -> StepResult:
        if self._done:
            obs = self._build_observation(score_so_far=self._compute_score())
            return StepResult(observation=obs, reward=0.0, done=True, info={"error": "Episode already done"})

        self._step_count += 1
        reward, info = self._apply_action(action)
        self._actions_taken.append(action.model_dump())

        score = self._compute_score()
        max_steps = self._task_data.get("max_steps", 50)
        self._done = self._step_count >= max_steps or self._is_complete()

        obs = self._build_observation(score_so_far=score)
        return StepResult(observation=obs, reward=reward, done=self._done, info=info)

    def state(self) -> dict:
        return {
            "task_id": self.task_id,
            "step": self._step_count,
            "done": self._done,
            "inbox_size": len(self._inbox),
            "calendar_size": len(self._calendar_events),
            "actions_taken": len(self._actions_taken),
            "current_score": self.score()
        }

    def score(self) -> float:
        """Returns the current task score, strictly bounded (0.01, 0.99)."""
        return self._compute_score()

    def grade(self) -> float:
        """Alias for score(), used by some OpenEnv validators."""
        return self.score()

    def close(self):
        self._done = True

    def _apply_action(self, action: Action) -> Tuple[float, dict]:
        atype = action.action_type
        reward = 0.0
        info = {}

        if atype == "no_op":
            reward = -0.01
            info["result"] = "No operation (small penalty)"
            return reward, info

        if not action.email_id and atype in ("archive_email", "flag_email", "reply_email", "delegate_email", "decline_meeting"):
             reward = -0.05
             info["result"] = "Error: email_id is required"
             return reward, info

        if atype == "archive_email":
            email = self._find_email(action.email_id)
            if email:
                if email.category == "spam":
                    reward = 0.15
                    info["result"] = "Correctly archived spam"
                elif email.priority == "urgent":
                    reward = -0.3
                    info["result"] = "Penalized: archived urgent email"
                else:
                    reward = 0.02
                    info["result"] = "Archived low-priority email"
                self._inbox = [e for e in self._inbox if e.id != action.email_id]
            else:
                reward = -0.05
                info["result"] = "Error: email_id not found"

        elif atype == "flag_email":
            email = self._find_email(action.email_id)
            if email:
                if email.id in self._flagged_ids:
                    reward = -0.02
                    info["result"] = "Redundant: already flagged"
                else:
                    self._flagged_ids.add(email.id)
                    email.is_flagged = True
                    if email.priority == "urgent":
                        reward = 0.2
                        info["result"] = "Correctly flagged urgent email"
                    else:
                        reward = -0.05
                        info["result"] = "Penalized: flagged non-urgent email"
            else:
                reward = -0.05
                info["result"] = "Error: email_id not found"

        elif atype == "reply_email":
            email = self._find_email(action.email_id)
            if email:
                if email.requires_reply:
                    reply_quality = min(1.0, len(action.reply_text or "") / 100)
                    reward = 0.1 + 0.1 * reply_quality
                    info["result"] = f"Replied (quality: {reply_quality:.2f})"
                else:
                    reward = -0.02
                    info["result"] = "Unnecessary reply"
                self._inbox = [e for e in self._inbox if e.id != action.email_id]
            else:
                reward = -0.05
                info["result"] = "Error: email_id not found"

        elif atype == "delegate_email":
            email = self._find_email(action.email_id)
            if email:
                if action.delegate_to:
                    reward = 0.1
                    info["result"] = f"Delegated to {action.delegate_to}"
                else:
                    reward = -0.05
                    info["result"] = "Error: missing delegate_to"
                self._inbox = [e for e in self._inbox if e.id != action.email_id]
            else:
                reward = -0.05
                info["result"] = "Error: email_id not found"

        elif atype == "schedule_meeting":
            conflict = self._check_conflict(action.meeting_start, action.meeting_end)
            if conflict:
                reward = -0.15
                info["result"] = "Conflict: meeting overlaps existing event"
            elif not action.meeting_start or not action.meeting_end:
                reward = -0.1
                info["result"] = "Error: missing meeting times"
            else:
                event = CalendarEvent(
                    id=str(uuid.uuid4()),
                    title=action.meeting_title or "Untitled Meeting",
                    start=action.meeting_start,
                    end=action.meeting_end,
                    attendees=action.meeting_attendees or [],
                    timezone=action.meeting_timezone or "Asia/Kolkata"
                )
                self._calendar_events.append(event)
                reward = 0.2
                info["result"] = "Meeting scheduled successfully"
                if action.email_id:
                    self._inbox = [e for e in self._inbox if e.id != action.email_id]

        elif atype == "decline_meeting":
            email = self._find_email(action.email_id)
            if email:
                reward = 0.05
                info["result"] = "Declined meeting request"
                self._inbox = [e for e in self._inbox if e.id != action.email_id]
            else:
                reward = -0.05
                info["result"] = "Error: email_id not found"

        return reward, info

    def _find_email(self, email_id: str) -> Optional[Email]:
        return next((e for e in self._inbox if e.id == email_id), None)

    def _check_conflict(self, start: Optional[datetime], end: Optional[datetime]) -> bool:
        if not start or not end:
            return False
        for evt in self._calendar_events:
            # Basic overlap check
            if start < evt.end and end > evt.start:
                return True
        return False

    def _compute_score(self) -> float:
        if not self._task_data:
            return 0.05
        
        from env.grader import grade_easy, grade_medium, grade_hard, safe_score
        
        if self.task_id == "easy":
            s = grade_easy(self._task_data, self._actions_taken)
        elif self.task_id == "medium":
            s = grade_medium(self._task_data, self._actions_taken, [e.model_dump() for e in self._calendar_events])
        elif self.task_id == "hard":
            s = grade_hard(self._task_data, self._actions_taken, [e.model_dump() for e in self._calendar_events], self._actions_taken)
        else:
            s = 0.5
            
        return safe_score(s)

    def _is_complete(self) -> bool:
        if self.task_id == "easy":
            # In easy, check if all urgent are flagged AND all spam archived AND all reply-required are replied.
            unprocessed = [e for e in self._inbox if (e.priority == "urgent" and e.id not in self._flagged_ids) or e.category == "spam" or e.requires_reply]
            return len(unprocessed) == 0
        return False

    def _build_observation(self, score_so_far: float) -> Observation:
        return Observation(
            inbox_emails=self._inbox,
            calendar_events=self._calendar_events,
            current_step=self._step_count,
            max_steps=self._task_data.get("max_steps", 50) if self._task_data else 50,
            task_id=self.task_id,
            task_objective=self._task_data.get("objective", "") if self._task_data else "",
            score_so_far=score_so_far,
            action_space={
                "flag_email": "Highlight urgent messages (high reward for priority=urgent)",
                "archive_email": "Dismiss spam/unimportant messages (penalty if urgent)",
                "reply_email": "Respond to task-related emails (require reply_text)",
                "delegate_email": "Pass tasks to others (matches colleague name in text)",
                "schedule_meeting": "Add events to calendar (avoid time conflicts)",
                "decline_meeting": "Reject calendar invites (ideal for low-priority requests)",
                "no_op": "Finish or pause. Use when objective is completed."
            }
        )
