# 🤖 AGENT.md — Full Build Guide
## Meta OpenEnv Hackathon | Round 1 Submission

---

## 🎯 PROJECT: Email Triage + Calendar Scheduling RL Environment

**Name:** `email-calendar-env`  
**Concept:** A realistic RL environment where an AI agent must triage a simulated email inbox and schedule meetings on a calendar. The agent must read emails, decide actions (reply / archive / flag / delegate / schedule), avoid calendar conflicts, respect deadlines and timezones, and maximize task completion within a limited number of steps.

**Why this project?**
- Represents real human work (email + scheduling = most common office tasks)
- Rich reward signal at every step (incremental progress, penalties for mistakes)
- Natural 3-task difficulty ladder: Easy → Medium → Hard
- Programmatic grading is clean, deterministic, and reproducible
- Judges from Meta/Hugging Face will immediately understand the real-world value

---

## 📁 EXACT FOLDER & FILE STRUCTURE TO CREATE

Create a folder called `email-calendar-env` and open it in your IDE. The agent must create ALL of the following files:

```
email-calendar-env/
├── inference.py              ← REQUIRED: Main agent inference script
├── openenv.yaml              ← REQUIRED: OpenEnv metadata file
├── Dockerfile                ← REQUIRED: Container definition
├── requirements.txt          ← Python dependencies
├── README.md                 ← Documentation (required by judges)
├── env/
│   ├── __init__.py
│   ├── email_calendar_env.py ← Core OpenEnv environment class
│   ├── models.py             ← Pydantic models for Observation/Action/Reward
│   ├── grader.py             ← Task graders (returns 0.0 to 1.0)
│   └── tasks/
│       ├── __init__.py
│       ├── task_easy.py      ← Easy task definition + data
│       ├── task_medium.py    ← Medium task definition + data
│       └── task_hard.py      ← Hard task definition + data
├── server/
│   ├── __init__.py
│   └── app.py               ← FastAPI server exposing env as HTTP API
└── tests/
    ├── test_env.py
    └── test_grader.py
```

---

## 📋 INSTRUCTIONS FOR THE AGENT (Paste these into your IDE's AI agent)

> Copy everything below this line and give it to your coding agent:

---

### AGENT INSTRUCTIONS START

You are building a complete OpenEnv-compliant Reinforcement Learning environment for the Meta OpenEnv Hackathon. Build every file listed below completely and correctly. Do not leave placeholders or TODOs. The submission deadline is April 8, 2026.

---

#### FILE 1: `requirements.txt`

```
fastapi>=0.110.0
uvicorn>=0.29.0
pydantic>=2.0.0
openai>=1.0.0
python-dotenv>=1.0.0
httpx>=0.27.0
pytest>=8.0.0
python-dateutil>=2.9.0
pytz>=2024.1
faker>=24.0.0
```

---

#### FILE 2: `openenv.yaml`

```yaml
name: email-calendar-env
version: "1.0.0"
description: >
  An RL environment simulating real-world email triage and calendar scheduling.
  The agent must process a simulated inbox, prioritize emails, draft replies,
  delegate tasks, and schedule meetings without conflicts.
author: your-hf-username
tags:
  - openenv
  - email
  - calendar
  - scheduling
  - productivity
  - real-world
tasks:
  - id: easy
    name: "Simple Inbox Triage"
    difficulty: easy
    description: "Triage 10 emails into correct folders and flag urgent ones."
  - id: medium
    name: "Inbox + Meeting Scheduling"
    difficulty: medium
    description: "Triage 25 emails and schedule 5 meetings respecting timezone constraints."
  - id: hard
    name: "Full Workday Simulation"
    difficulty: hard
    description: "Handle 50 emails, schedule 10 meetings, manage conflicts, respond to VIP threads, and meet 3 deadlines."
observation_space:
  type: dict
  fields:
    - inbox_emails
    - calendar_events
    - current_step
    - task_objective
action_space:
  type: discrete_parametric
  actions:
    - archive_email
    - flag_email
    - reply_email
    - delegate_email
    - schedule_meeting
    - decline_meeting
reward_range: [-1.0, 1.0]
```

---

#### FILE 3: `env/models.py`

Create Pydantic v2 models:

```python
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
    deadline: Optional[datetime] = None

class CalendarEvent(BaseModel):
    id: str
    title: str
    start: datetime
    end: datetime
    attendees: List[str]
    timezone: str

class Observation(BaseModel):
    inbox: List[Email]
    calendar: List[CalendarEvent]
    current_step: int
    max_steps: int
    task_id: str
    task_objective: str
    score_so_far: float

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
```

---

#### FILE 4: `env/tasks/task_easy.py`

Generate 10 synthetic emails using Faker. Include 3 urgent ones, 2 spam, 3 FYI, 2 action-required. The grader checks:
- Urgent emails flagged: +0.15 each (max 0.45)
- Spam archived: +0.10 each (max 0.20)
- Action-required emails replied: +0.175 each (max 0.35)
- Total max score: 1.0

```python
from faker import Faker
from datetime import datetime, timedelta
import random, uuid, pytz

fake = Faker()
IST = pytz.timezone("Asia/Kolkata")

def generate_easy_task():
    emails = []
    # Generate emails deterministically with seed
    Faker.seed(42)
    random.seed(42)
    
    specs = [
        ("urgent", "action_required", True, "Server outage alert"),
        ("urgent", "action_required", True, "Client escalation URGENT"),
        ("urgent", "meeting_request", False, "Emergency board call"),
        ("low", "spam", False, "You won a prize!"),
        ("low", "spam", False, "Limited offer expires today"),
        ("normal", "fyi", False, "Weekly newsletter"),
        ("normal", "fyi", False, "Team lunch Thursday"),
        ("normal", "fyi", False, "Policy update Q2"),
        ("normal", "action_required", True, "Please review attached contract"),
        ("normal", "action_required", True, "Approval needed for budget"),
    ]
    
    base_time = datetime(2026, 4, 7, 9, 0, 0, tzinfo=IST)
    for i, (priority, category, requires_reply, subject) in enumerate(specs):
        emails.append({
            "id": f"email_{i+1:03d}",
            "sender": fake.email(),
            "subject": subject,
            "body": fake.paragraph(nb_sentences=4),
            "timestamp": (base_time + timedelta(minutes=i*15)).isoformat(),
            "priority": priority,
            "category": category,
            "requires_reply": requires_reply,
            "deadline": (base_time + timedelta(hours=4)).isoformat() if priority == "urgent" else None
        })
    
    return {
        "task_id": "easy",
        "objective": "Triage the inbox: flag urgent emails, archive spam, and reply to action-required emails.",
        "emails": emails,
        "calendar": [],
        "max_steps": 20
    }
```

---

#### FILE 5: `env/tasks/task_medium.py`

25 emails + 5 meeting scheduling requests. Grader checks:
- Correct email triage: 0.5 points total
- Meetings scheduled without conflicts: 0.1 per meeting (0.5 total)
- Timezone correctly handled: bonus 0.1 if all timezone correct (capped at 1.0)

Use Faker with seed=123. Mix of priorities. Include 5 emails that are meeting requests that the agent must convert into calendar events.

---

#### FILE 6: `env/tasks/task_hard.py`

50 emails + 10 meeting requests. Add:
- 3 VIP senders (must reply within 5 steps or -0.2 penalty each)
- 3 deadline-bound emails
- Calendar already has 5 existing conflicting events the agent must navigate around
- Grader is a weighted composite scoring all of the above

---

#### FILE 7: `env/grader.py`

```python
from typing import List, Dict, Any

class EasyTaskGrader:
    def __init__(self, task_data: dict):
        self.task_data = task_data
        self.expected = self._build_expected()
    
    def _build_expected(self):
        emails = self.task_data["emails"]
        return {
            "flag": [e["id"] for e in emails if e["priority"] == "urgent"],
            "archive": [e["id"] for e in emails if e["category"] == "spam"],
            "reply": [e["id"] for e in emails if e["requires_reply"]],
        }
    
    def score(self, agent_actions: List[Dict]) -> float:
        flagged = {a["email_id"] for a in agent_actions if a["action_type"] == "flag_email"}
        archived = {a["email_id"] for a in agent_actions if a["action_type"] == "archive_email"}
        replied = {a["email_id"] for a in agent_actions if a["action_type"] == "reply_email"}
        
        flag_score = len(flagged & set(self.expected["flag"])) / max(len(self.expected["flag"]), 1)
        archive_score = len(archived & set(self.expected["archive"])) / max(len(self.expected["archive"]), 1)
        reply_score = len(replied & set(self.expected["reply"])) / max(len(self.expected["reply"]), 1)
        
        # Penalties for wrong actions
        wrong_flags = len(flagged - set(self.expected["flag"])) * 0.05
        wrong_archives = len(archived - set(self.expected["archive"])) * 0.05
        
        total = (flag_score * 0.35 + archive_score * 0.35 + reply_score * 0.30) - wrong_flags - wrong_archives
        return max(0.0, min(1.0, total))


class MediumTaskGrader:
    def __init__(self, task_data: dict):
        self.task_data = task_data
    
    def score(self, agent_actions: List[Dict], final_calendar: List[Dict]) -> float:
        email_score = self._score_emails(agent_actions)
        calendar_score = self._score_calendar(final_calendar)
        return min(1.0, email_score * 0.5 + calendar_score * 0.5)
    
    def _score_emails(self, actions):
        # Same logic as easy but with 25 emails
        emails = self.task_data["emails"]
        expected_flag = [e["id"] for e in emails if e["priority"] == "urgent"]
        expected_archive = [e["id"] for e in emails if e["category"] == "spam"]
        flagged = {a["email_id"] for a in actions if a["action_type"] == "flag_email"}
        archived = {a["email_id"] for a in actions if a["action_type"] == "archive_email"}
        fs = len(flagged & set(expected_flag)) / max(len(expected_flag), 1)
        as_ = len(archived & set(expected_archive)) / max(len(expected_archive), 1)
        return (fs + as_) / 2
    
    def _score_calendar(self, events):
        if not events:
            return 0.0
        conflicts = self._count_conflicts(events)
        required = self.task_data.get("required_meetings", 5)
        scheduled = min(len(events), required)
        conflict_penalty = conflicts * 0.1
        return max(0.0, (scheduled / required) - conflict_penalty)
    
    def _count_conflicts(self, events):
        conflicts = 0
        for i, e1 in enumerate(events):
            for e2 in events[i+1:]:
                if e1["start"] < e2["end"] and e1["end"] > e2["start"]:
                    conflicts += 1
        return conflicts


class HardTaskGrader:
    def __init__(self, task_data: dict):
        self.task_data = task_data
    
    def score(self, agent_actions: List[Dict], final_calendar: List[Dict], step_history: List[Dict]) -> float:
        email_score = self._score_emails(agent_actions) * 0.4
        calendar_score = self._score_calendar(final_calendar) * 0.3
        vip_score = self._score_vip_response(agent_actions, step_history) * 0.2
        deadline_score = self._score_deadlines(agent_actions) * 0.1
        return min(1.0, email_score + calendar_score + vip_score + deadline_score)
    
    def _score_emails(self, actions):
        emails = self.task_data["emails"]
        urgent = [e["id"] for e in emails if e["priority"] == "urgent"]
        spam = [e["id"] for e in emails if e["category"] == "spam"]
        flagged = {a["email_id"] for a in actions if a["action_type"] == "flag_email"}
        archived = {a["email_id"] for a in actions if a["action_type"] == "archive_email"}
        return (
            len(flagged & set(urgent)) / max(len(urgent), 1) * 0.5 +
            len(archived & set(spam)) / max(len(spam), 1) * 0.5
        )
    
    def _score_calendar(self, events):
        if not events:
            return 0.0
        required = self.task_data.get("required_meetings", 10)
        conflicts = sum(
            1 for i, e1 in enumerate(events)
            for e2 in events[i+1:]
            if e1["start"] < e2["end"] and e1["end"] > e2["start"]
        )
        return max(0.0, min(len(events), required) / required - conflicts * 0.05)
    
    def _score_vip_response(self, actions, step_history):
        vip_ids = self.task_data.get("vip_email_ids", [])
        if not vip_ids:
            return 1.0
        replied_vips = {a["email_id"] for a in actions if a["action_type"] == "reply_email" and a["email_id"] in vip_ids}
        return len(replied_vips) / len(vip_ids)
    
    def _score_deadlines(self, actions):
        deadline_ids = self.task_data.get("deadline_email_ids", [])
        if not deadline_ids:
            return 1.0
        handled = {a["email_id"] for a in actions if a["action_type"] in ("reply_email", "delegate_email") and a["email_id"] in deadline_ids}
        return len(handled) / len(deadline_ids)
```

---

#### FILE 8: `env/email_calendar_env.py`

Build the full OpenEnv-compliant environment class:

```python
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

    def reset(self) -> Observation:
        self._step_count = 0
        self._done = False
        self._actions_taken = []
        self._calendar_events = []

        if self.task_id == "easy":
            self._task_data = generate_easy_task()
            self._grader = EasyTaskGrader(self._task_data)
        elif self.task_id == "medium":
            self._task_data = generate_medium_task()
            self._grader = MediumTaskGrader(self._task_data)
        else:
            self._task_data = generate_hard_task()
            self._grader = HardTaskGrader(self._task_data)

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
            "current_score": self._compute_score()
        }

    def close(self):
        self._done = True

    def _apply_action(self, action: Action) -> Tuple[float, dict]:
        atype = action.action_type
        reward = 0.0
        info = {}

        if atype == "archive_email" and action.email_id:
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

        elif atype == "flag_email" and action.email_id:
            email = self._find_email(action.email_id)
            if email:
                if email.priority == "urgent":
                    reward = 0.2
                    info["result"] = "Correctly flagged urgent email"
                else:
                    reward = -0.05
                    info["result"] = "Penalized: flagged non-urgent email"

        elif atype == "reply_email" and action.email_id:
            email = self._find_email(action.email_id)
            if email:
                if email.requires_reply:
                    reply_quality = min(1.0, len(action.reply_text or "") / 100)
                    reward = 0.1 + 0.1 * reply_quality
                    info["result"] = f"Replied (quality: {reply_quality:.2f})"
                else:
                    reward = -0.02
                    info["result"] = "Unnecessary reply"

        elif atype == "schedule_meeting":
            conflict = self._check_conflict(action.meeting_start, action.meeting_end)
            if conflict:
                reward = -0.15
                info["result"] = "Conflict: meeting overlaps existing event"
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

        elif atype == "no_op":
            reward = -0.01
            info["result"] = "No operation (small penalty)"

        return reward, info

    def _find_email(self, email_id: str) -> Optional[Email]:
        return next((e for e in self._inbox if e.id == email_id), None)

    def _check_conflict(self, start: Optional[datetime], end: Optional[datetime]) -> bool:
        if not start or not end:
            return False
        for evt in self._calendar_events:
            if start < evt.end and end > evt.start:
                return True
        return False

    def _compute_score(self) -> float:
        if not self._grader:
            return 0.0
        if self.task_id == "easy":
            return self._grader.score(self._actions_taken)
        elif self.task_id == "medium":
            return self._grader.score(self._actions_taken, [e.model_dump() for e in self._calendar_events])
        else:
            return self._grader.score(self._actions_taken, [e.model_dump() for e in self._calendar_events], [])

    def _is_complete(self) -> bool:
        if self.task_id == "easy":
            unprocessed = [e for e in self._inbox if e.priority == "urgent" or e.category == "spam" or e.requires_reply]
            return len(unprocessed) == 0
        return False

    def _build_observation(self, score_so_far: float) -> Observation:
        return Observation(
            inbox=self._inbox,
            calendar=self._calendar_events,
            current_step=self._step_count,
            max_steps=self._task_data.get("max_steps", 50) if self._task_data else 50,
            task_id=self.task_id,
            task_objective=self._task_data.get("objective", "") if self._task_data else "",
            score_so_far=score_so_far
        )
```

---

#### FILE 9: `server/app.py`

Build a FastAPI server exposing the env as REST API:

```python
from fastapi import FastAPI, HTTPException
from env.email_calendar_env import EmailCalendarEnv
from env.models import Action, Observation, StepResult
import os

app = FastAPI(title="Email-Calendar OpenEnv", version="1.0.0")

_envs: dict = {}

@app.get("/health")
def health():
    return {"status": "ok", "env": "email-calendar-env"}

@app.post("/env/reset")
def reset(task_id: str = "easy"):
    env = EmailCalendarEnv(task_id=task_id)
    session_id = f"session_{task_id}"
    _envs[session_id] = env
    obs = env.reset()
    return {"session_id": session_id, "observation": obs.model_dump()}

@app.post("/env/step/{session_id}")
def step(session_id: str, action: Action):
    env = _envs.get(session_id)
    if not env:
        raise HTTPException(status_code=404, detail="Session not found. Call /env/reset first.")
    result = env.step(action)
    return result.model_dump()

@app.get("/env/state/{session_id}")
def state(session_id: str):
    env = _envs.get(session_id)
    if not env:
        raise HTTPException(status_code=404, detail="Session not found.")
    return env.state()

@app.post("/env/close/{session_id}")
def close(session_id: str):
    env = _envs.get(session_id)
    if env:
        env.close()
        del _envs[session_id]
    return {"status": "closed"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

---

#### FILE 10: `inference.py` (CRITICAL — must match spec exactly)

```python
import os
import json
from openai import OpenAI
from env.email_calendar_env import EmailCalendarEnv
from env.models import Action
from datetime import datetime, timedelta
import pytz

# ── Environment Variables ──────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ── OpenAI Client ──────────────────────────────────────────────────────────────
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── LLM Action Planner ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert email and calendar management AI agent.

Given the current inbox state and calendar, decide the SINGLE BEST next action.
Respond ONLY with a valid JSON object matching this schema:
{
  "action_type": one of ["archive_email","flag_email","reply_email","delegate_email","schedule_meeting","decline_meeting","no_op"],
  "email_id": "email_XXX" or null,
  "reply_text": "reply content" or null,
  "delegate_to": "person@example.com" or null,
  "meeting_title": "title" or null,
  "meeting_start": "ISO datetime" or null,
  "meeting_end": "ISO datetime" or null,
  "meeting_attendees": ["email1", "email2"] or null,
  "meeting_timezone": "timezone string" or null
}

Rules:
- Flag emails with priority=urgent
- Archive emails with category=spam
- Reply to emails where requires_reply=true
- Schedule meetings for category=meeting_request emails (avoid conflicts!)
- Penalized for wrong actions, so be precise
- Respond ONLY with the JSON object, nothing else.
"""

def get_llm_action(observation: dict) -> Action:
    inbox_summary = []
    for e in observation.get("inbox", [])[:10]:  # Limit to avoid token overflow
        inbox_summary.append({
            "id": e["id"],
            "subject": e["subject"],
            "priority": e["priority"],
            "category": e["category"],
            "requires_reply": e["requires_reply"],
            "sender": e["sender"]
        })
    
    calendar_summary = []
    for c in observation.get("calendar", [])[:5]:
        calendar_summary.append({
            "title": c["title"],
            "start": c["start"],
            "end": c["end"]
        })

    user_msg = f"""
OBJECTIVE: {observation.get('task_objective', '')}
STEP: {observation.get('current_step', 0)} / {observation.get('max_steps', 50)}
SCORE SO FAR: {observation.get('score_so_far', 0.0):.2f}

INBOX ({len(observation.get('inbox', []))} emails shown, first 10):
{json.dumps(inbox_summary, indent=2, default=str)}

CALENDAR ({len(observation.get('calendar', []))} events):
{json.dumps(calendar_summary, indent=2, default=str)}

Decide your next single best action:
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg}
        ],
        max_tokens=300,
        temperature=0.1,
    )
    
    raw = response.choices[0].message.content.strip()
    
    try:
        action_dict = json.loads(raw)
        # Parse datetime strings if present
        for field in ("meeting_start", "meeting_end"):
            if action_dict.get(field):
                action_dict[field] = datetime.fromisoformat(action_dict[field])
        return Action(**action_dict)
    except Exception as e:
        return Action(action_type="no_op")


def run_task(task_id: str = "easy"):
    env = EmailCalendarEnv(task_id=task_id)
    obs = env.reset()
    task_name = task_id
    
    print(f"[START] task={task_name} env=email-calendar-env model={MODEL_NAME}")
    
    step_num = 0
    rewards = []
    last_error = None
    
    try:
        while True:
            action = get_llm_action(obs.model_dump())
            result = env.step(action)
            step_num += 1
            reward = round(result.reward, 2)
            rewards.append(reward)
            done = result.done
            error = result.info.get("error", None)
            
            print(
                f"[STEP] step={step_num} "
                f"action={action.action_type} "
                f"reward={reward:.2f} "
                f"done={'true' if done else 'false'} "
                f"error={error if error else 'null'}"
            )
            
            obs = result.observation
            if done:
                break

    except Exception as ex:
        last_error = str(ex)
        done = True
    finally:
        env.close()
        success = env.state().get("current_score", 0.0) >= 0.5
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={'true' if success else 'false'} "
            f"steps={step_num} "
            f"rewards={rewards_str}"
        )


if __name__ == "__main__":
    import sys
    task = sys.argv[1] if len(sys.argv) > 1 else "easy"
    run_task(task_id=task)
```

---

#### FILE 11: `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV API_BASE_URL="https://api-inference.huggingface.co/v1"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
ENV PORT=7860

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

#### FILE 12: `README.md`

Write a complete README with these sections:
1. **Overview & Motivation** — Why email triage + calendar scheduling matters as an RL benchmark
2. **Environment Architecture** — How the env is structured, what OpenEnv interfaces are implemented
3. **Action Space** — List and explain all 7 actions
4. **Observation Space** — Describe inbox, calendar, step, score fields
5. **Task Descriptions** — Easy / Medium / Hard with expected difficulty and grading criteria
6. **Reward Function** — Explain incremental rewards, penalties, and final scoring
7. **Setup & Usage** — `pip install -r requirements.txt`, `python inference.py easy`, docker commands
8. **Baseline Performance** — Placeholder table (fill with actual scores after running)
9. **Hugging Face Space** — Link and deployment notes

---

#### FILE 13: All `__init__.py` files

Create empty `__init__.py` files in:
- `env/__init__.py`
- `env/tasks/__init__.py`
- `server/__init__.py`
- `tests/__init__.py`

---

#### VALIDATION CHECKLIST

After creating all files, run these commands to validate:

```bash
# Install dependencies
pip install -r requirements.txt

# Test env directly
python -c "from env.email_calendar_env import EmailCalendarEnv; env = EmailCalendarEnv('easy'); obs = env.reset(); print('ENV OK:', obs.task_id)"

# Test inference script (requires HF_TOKEN)
export HF_TOKEN=your_token_here
python inference.py easy

# Test server
uvicorn server.app:app --port 7860 &
curl http://localhost:7860/health

# Docker build test
docker build -t email-calendar-env .
docker run -p 7860:7860 -e HF_TOKEN=your_token email-calendar-env
```

### AGENT INSTRUCTIONS END

---

## 🔑 CREDENTIALS & API KEYS REQUIRED

Before submission, you need these:

| Credential | Where to Get | Required? |
|---|---|---|
| `HF_TOKEN` | huggingface.co → Settings → Access Tokens → New Token (write scope) | ✅ MANDATORY |
| `API_BASE_URL` | Default: `https://api-inference.huggingface.co/v1` (no signup needed) | Has default |
| `MODEL_NAME` | Default: `Qwen/Qwen2.5-72B-Instruct` (free on HF Inference API) | Has default |

**Get your HF Token:** https://huggingface.co/settings/tokens
- Select **"write"** scope (needed to push your Space)

---

## 🛠️ LOCAL INSTALLATIONS REQUIRED

```bash
# Python 3.11+ required
python --version

# Install Docker Desktop (for containerization)
# https://docs.docker.com/get-docker/

# Install Hugging Face CLI (for pushing your Space)
pip install huggingface_hub
huggingface-cli login   # Enter your HF_TOKEN when prompted

# Install openenv validator
pip install openenv
openenv validate .      # Run this in your project folder
```

---

## 🚀 SUBMISSION STEPS (Do in order)

1. Build and test locally with `python inference.py easy`
2. Verify output format matches exactly:
   ```
   [START] task=easy env=email-calendar-env model=Qwen/Qwen2.5-72B-Instruct
   [STEP] step=1 action=flag_email reward=0.20 done=false error=null
   ...
   [END] success=true steps=12 rewards=0.20,0.15,...
   ```
3. Run `openenv validate .` — must pass with no errors
4. Create Hugging Face Space:
   ```bash
   huggingface-cli repo create email-calendar-env --type space --space_sdk docker
   git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/email-calendar-env
   git push hf main
   ```
5. Wait for Space to show **"Running"** status (not "Building")
6. Submit at: https://sclr.ac/w8rE3c

---

## ⚠️ COMMON FAILURE CASES TO AVOID

- `inference.py` not in root directory → keep it at root
- Missing default for `API_BASE_URL` or `MODEL_NAME` → defaults are set in the file
- HF Space still "Building" when you submit → wait for "Running"
- Multiple HF Spaces active → pause all others before submitting
- Docker container using more than 2 vCPU / 8 GB RAM → keep model calls remote (via HF Inference API), not local

---

## 📊 SCORING RUBRIC (What judges look at)

| Criterion | Weight |
|---|---|
| OpenEnv spec compliance (validate passes) | High |
| Real-world task quality | High |
| 3+ tasks with difficulty ladder | Required |
| Programmatic grader (0.0–1.0, deterministic) | Required |
| Meaningful incremental reward function | High |
| Working Dockerfile | Required |
| HF Space is live and Running | Required |
| README quality | Medium |
| Baseline inference script runs | Medium |

---

*Deadline: April 8, 2026 at midnight. Good luck! 🚀*
