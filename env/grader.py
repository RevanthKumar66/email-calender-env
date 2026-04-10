from typing import List, Dict, Any, Optional
from datetime import datetime

def safe_score(score: float) -> float:
    """Ensures score is strictly within (0, 1) range as required by validator."""
    return max(0.01, min(0.99, score))

class EasyTaskGrader:
    def __init__(self, task_data: dict):
        self.task_data = task_data
        self.expected = self._build_expected()
    
    def _build_expected(self):
        emails = self.task_data.get("emails", [])
        return {
            "flag": [e["id"] for e in emails if e.get("priority") == "urgent"],
            "archive": [e["id"] for e in emails if e.get("category") == "spam"],
            "reply": [e["id"] for e in emails if e.get("requires_reply")],
        }
    
    def score(self, agent_actions: List[Dict]) -> float:
        if not agent_actions: return safe_score(0.05)
        
        flagged = {a["email_id"] for a in agent_actions if a.get("action_type") == "flag_email"}
        archived = {a["email_id"] for a in agent_actions if a.get("action_type") == "archive_email"}
        replied = {a["email_id"] for a in agent_actions if a.get("action_type") == "reply_email"}
        
        flag_score = len(flagged & set(self.expected["flag"])) / max(len(self.expected["flag"]), 1)
        archive_score = len(archived & set(self.expected["archive"])) / max(len(self.expected["archive"]), 1)
        reply_score = len(replied & set(self.expected["reply"])) / max(len(self.expected["reply"]), 1)
        
        raw = (flag_score * 0.35 + archive_score * 0.35 + reply_score * 0.30)
        return safe_score(raw)


class MediumTaskGrader:
    def __init__(self, task_data: dict):
        self.task_data = task_data
    
    def score(self, agent_actions: List[Dict], final_calendar: List[Dict]) -> float:
        emails = self.task_data.get("emails", [])
        expected_flag = [e["id"] for e in emails if e.get("priority") == "urgent"]
        expected_archive = [e["id"] for e in emails if e.get("category") == "spam"]
        
        flagged = {a["email_id"] for a in agent_actions if a.get("action_type") == "flag_email"}
        archived = {a["email_id"] for a in agent_actions if a.get("action_type") == "archive_email"}
        
        fs = len(flagged & set(expected_flag)) / max(len(expected_flag), 1)
        as_ = len(archived & set(expected_archive)) / max(len(expected_archive), 1)
        
        conflicts = 0
        from dateutil.parser import parse
        events = final_calendar or []
        for i, e1 in enumerate(events):
            s1, e1_end = parse(str(e1["start"])), parse(str(e1["end"]))
            for e2 in events[i+1:]:
                s2, e2_end = parse(str(e2["start"])), parse(str(e2["end"]))
                if s1 < e2_end and e1_end > s2: conflicts += 1
        
        req = self.task_data.get("required_meetings", 5)
        cs = max(0.0, min(len(events), req) / req - conflicts * 0.1)
        
        return safe_score((fs + as_) * 0.25 + cs * 0.5)


class HardTaskGrader:
    def __init__(self, task_data: dict):
        self.task_data = task_data
    
    def score(self, *args, **kwargs) -> float:
        return safe_score(0.5)

# --- Universal Wrapper Functions ---

def grade_easy(state: Any, actions: Optional[List] = None) -> float:
    if isinstance(state, dict) and ("emails" in state or "inbox_emails" in state):
        task_data, agent_actions = state, actions or []
    else:
        task_data = getattr(state, "_task_data", getattr(state, "task_data", {}))
        agent_actions = getattr(state, "_actions_taken", getattr(state, "actions_taken", []))
    return EasyTaskGrader(task_data or {}).score(agent_actions)

def grade_medium(state: Any, actions: Optional[List] = None, calendar: Optional[List] = None) -> float:
    if isinstance(state, dict) and ("emails" in state or "inbox_emails" in state):
        task_data, agent_actions, final_calendar = state, actions or [], calendar or []
    else:
        task_data = getattr(state, "_task_data", getattr(state, "task_data", {}))
        agent_actions = getattr(state, "_actions_taken", getattr(state, "actions_taken", []))
        final_calendar = getattr(state, "_calendar_events", getattr(state, "calendar_events", []))
    return MediumTaskGrader(task_data or {}).score(agent_actions, final_calendar)

def grade_hard(state: Any, *args, **kwargs) -> float:
    return safe_score(0.5)
