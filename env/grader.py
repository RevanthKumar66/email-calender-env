from typing import List, Dict, Any

class EasyTaskGrader:
    """Grader for simple triage tasks involving flagging and archiving."""
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
        
        # Deduction for incorrect operations
        wrong_flags = len(flagged - set(self.expected["flag"])) * 0.05
        wrong_archives = len(archived - set(self.expected["archive"])) * 0.05
        
        total = (flag_score * 0.35 + archive_score * 0.35 + reply_score * 0.30) - wrong_flags - wrong_archives
        return max(0.0, min(1.0, total))


class MediumTaskGrader:
    """Grader for tasks involving email management and calendar scheduling."""
    def __init__(self, task_data: dict):
        self.task_data = task_data
    
    def score(self, agent_actions: List[Dict], final_calendar: List[Dict]) -> float:
        email_score = self._score_emails(agent_actions)
        calendar_score = self._score_calendar(final_calendar)
        return min(1.0, email_score * 0.5 + calendar_score * 0.5)
    
    def _score_emails(self, actions):
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
        import datetime
        from dateutil.parser import parse
        conflicts = 0
        for i, e1 in enumerate(events):
            start1 = parse(str(e1["start"]))
            end1 = parse(str(e1["end"]))
            for e2 in events[i+1:]:
                start2 = parse(str(e2["start"]))
                end2 = parse(str(e2["end"]))
                if start1 < end2 and end1 > start2:
                    conflicts += 1
        return conflicts


class HardTaskGrader:
    """Complex grader for high-volume tasks with VIP priorities and deadlines."""
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
        from dateutil.parser import parse
        conflicts = 0
        for i, e1 in enumerate(events):
            start1 = parse(str(e1["start"]))
            end1 = parse(str(e1["end"]))
            for e2 in events[i+1:]:
                start2 = parse(str(e2["start"]))
                end2 = parse(str(e2["end"]))
                if start1 < end2 and end1 > start2:
                    conflicts += 1
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

def grade_easy(task_data: dict, agent_actions: List[Dict]) -> float:
    """Calculates score for the easy task category."""
    grader = EasyTaskGrader(task_data)
    return grader.score(agent_actions)

def grade_medium(task_data: dict, agent_actions: List[Dict], final_calendar: List[Dict]) -> float:
    """Calculates score for the medium task category."""
    grader = MediumTaskGrader(task_data)
    return grader.score(agent_actions, final_calendar)

def grade_hard(task_data: dict, agent_actions: List[Dict], final_calendar: List[Dict], step_history: List[Dict]) -> float:
    """Calculates score for the hard task category."""
    grader = HardTaskGrader(task_data)
    return grader.score(agent_actions, final_calendar, step_history)
