from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime

class Email(BaseModel):
    id: str
    sender: str
    subject: str
    body: str
    timestamp: datetime
    priority: Literal["urgent", "normal", "low"]
    category: Literal["action_required", "fyi", "spam", "meeting_request"]
    requires_reply: bool
    is_flagged: bool = False
    deadline: Optional[datetime] = None

class CalendarEvent(BaseModel):
    id: str
    title: str
    start: datetime
    end: datetime
    attendees: List[str]
    timezone: str

class Observation(BaseModel):
    inbox_emails: List[Email]
    calendar_events: List[CalendarEvent]
    current_step: int
    max_steps: int
    task_id: str
    task_objective: str
    score_so_far: float
    action_space: Optional[dict] = None

class Action(BaseModel):
    action_type: Literal[
        "archive_email",
        "flag_email",
        "reply_email",
        "delegate_email",
        "schedule_meeting",
        "decline_meeting",
        "no_op"
    ]
    email_id: Optional[str] = None
    reply_text: Optional[str] = None
    delegate_to: Optional[str] = None
    meeting_title: Optional[str] = None
    meeting_start: Optional[datetime] = None
    meeting_end: Optional[datetime] = None
    meeting_attendees: Optional[List[str]] = None
    meeting_timezone: Optional[str] = "Asia/Kolkata"

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict
