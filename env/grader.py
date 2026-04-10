from typing import List, Dict, Any, Optional

def safe_score(score: float) -> float:
    """Validator requires score strictly within (0, 1)."""
    try:
        s = float(score)
        return max(0.01, min(0.99, s))
    except (ValueError, TypeError):
        return 0.5

class EasyTaskGrader:
    def __init__(self, task_data: dict):
        self.task_data = task_data or {}

    def score(self, actions: List[Dict]) -> float:
        if not actions:
            return 0.0
        emails = self.task_data.get("emails", [])
        urgent_ids = {e["id"] for e in emails if e["priority"] == "urgent"}
        spam_ids = {e["id"] for e in emails if e["category"] == "spam"}
        reply_ids = {e["id"] for e in emails if e["requires_reply"]}

        flagged = {a["email_id"] for a in actions if a.get("action_type") == "flag_email" and a.get("email_id")}
        archived = {a["email_id"] for a in actions if a.get("action_type") == "archive_email" and a.get("email_id")}
        replied = {a["email_id"] for a in actions if a.get("action_type") == "reply_email" and a.get("email_id")}

        total = len(urgent_ids) + len(spam_ids) + len(reply_ids)
        if total == 0:
            return 1.0
        correct = len(flagged & urgent_ids) + len(archived & spam_ids) + len(replied & reply_ids)
        return correct / total


class MediumTaskGrader:
    def __init__(self, task_data: dict):
        self.task_data = task_data or {}

    def score(self, actions: List[Dict]) -> float:
        if not actions:
            return 0.0
        emails = self.task_data.get("emails", [])
        required_meetings = self.task_data.get("required_meetings", 5)
        reply_ids = {e["id"] for e in emails if e.get("requires_reply")}

        meetings_scheduled = sum(1 for a in actions if a.get("action_type") == "schedule_meeting")
        replied = {a["email_id"] for a in actions if a.get("action_type") == "reply_email" and a.get("email_id")}

        meeting_score = min(meetings_scheduled / required_meetings, 1.0) if required_meetings > 0 else 1.0
        reply_score = len(replied & reply_ids) / len(reply_ids) if reply_ids else 1.0
        return (meeting_score + reply_score) / 2


class HardTaskGrader:
    def __init__(self, task_data: dict):
        self.task_data = task_data or {}

    def score(self, actions: List[Dict]) -> float:
        if not actions:
            return 0.0
        emails = self.task_data.get("emails", [])
        vip_ids = set(self.task_data.get("vip_email_ids", []))
        required_meetings = self.task_data.get("required_meetings", 10)
        reply_ids = {e["id"] for e in emails if e.get("requires_reply")}

        replied = {a["email_id"] for a in actions if a.get("action_type") == "reply_email" and a.get("email_id")}
        meetings_scheduled = sum(1 for a in actions if a.get("action_type") == "schedule_meeting")

        vip_score = len(replied & vip_ids) / len(vip_ids) if vip_ids else 1.0
        meeting_score = min(meetings_scheduled / required_meetings, 1.0) if required_meetings > 0 else 1.0
        reply_score = len(replied & reply_ids) / len(reply_ids) if reply_ids else 1.0
        return vip_score * 0.4 + meeting_score * 0.3 + reply_score * 0.3


# Resilient entry points for the validator
def grade_easy(state: Any, actions: List[Dict] = None, *args, **kwargs) -> float:
    if hasattr(state, "score"): return state.score()
    if isinstance(state, dict):
        used_actions = actions if actions is not None else state.get("actions_taken", [])
        return safe_score(EasyTaskGrader(state).score(used_actions))
    return safe_score(0.5)

def grade_medium(state: Any, actions: List[Dict] = None, *args, **kwargs) -> float:
    if hasattr(state, "score"): return state.score()
    if isinstance(state, dict):
        used_actions = actions if actions is not None else state.get("actions_taken", [])
        return safe_score(MediumTaskGrader(state).score(used_actions))
    return safe_score(0.5)

def grade_hard(state: Any, actions: List[Dict] = None, *args, **kwargs) -> float:
    if hasattr(state, "score"): return state.score()
    if isinstance(state, dict):
        used_actions = actions if actions is not None else state.get("actions_taken", [])
        return safe_score(HardTaskGrader(state).score(used_actions))
    return safe_score(0.5)
