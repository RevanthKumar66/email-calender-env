import os
import json
import random
from typing import Optional, Set
from openai import OpenAI
from env.email_calendar_env import EmailCalendarEnv
from env.models import Action
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file (do NOT override system environment)
load_dotenv()

# ── Environment Variables & LLM Client ─────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY      = os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# Fallback for local development or HuggingFace Spaces
if not API_BASE_URL or not API_KEY:
    API_BASE_URL = "https://router.huggingface.co/v1"
    API_KEY      = os.getenv("HF_TOKEN")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

# ── Simplified SYSTEM_PROMPT ───────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an intelligent executive assistant AI. 

Goal: Maximize reward by triaging emails and managing the calendar.

Action Priorities:
1. flag_email: For priority='urgent' emails.
2. archive_email: For category='spam'.
3. reply_email: For 'action_required' + requires_reply=true.
4. schedule_meeting: For 'meeting_request'.

Rules:
- DO NOT repeat actions on the same email_id.
- Be professional and detailed in your replies.
- Return ONLY valid JSON.
"""

# ── Rule-Based Decision Layer (Adaptive & Refined) ──────────────────────────
def get_rule_based_action(observation: dict, processed_ids: Set[str]) -> Optional[Action]:
    """
    Refined logic: Urgent -> Spam -> Reply -> Meeting.
    Includes memory persistence and inbox-empty check.
    """
    inbox = observation.get("inbox", [])
    score = observation.get("score_so_far", 0.0)
    
    # Termination Intelligence: If inbox is empty, don't waste steps
    if not inbox:
        return Action(action_type="no_op")
        
    # Filter available (not in processed_ids)
    available = [e for e in inbox if e["id"] not in processed_ids]
    
    # ADAPTIVE INTELLIGENCE: Strictly prioritize urgent if score is low
    if score < 0.3:
        for e in available:
            if e.get("priority") == "urgent" and not e.get("is_flagged", False):
                return Action(action_type="flag_email", email_id=e["id"])

    # 1. Urgent (General)
    for e in available:
        if e.get("priority") == "urgent" and not e.get("is_flagged", False):
            return Action(action_type="flag_email", email_id=e["id"])
            
    # 2. Spam
    for e in available:
        if e.get("category") == "spam":
            return Action(action_type="archive_email", email_id=e["id"])
            
    # 3. High Quality Reply (Improved Text)
    for e in available:
        if e.get("requires_reply"):
            return Action(
                action_type="reply_email", 
                email_id=e["id"], 
                reply_text="Thank you for your message. We have received your request and will take appropriate action shortly. Our team is currently reviewing the details, and we will get back to you with a comprehensive update. Please let us know if you need anything else in the meantime."
            )
            
    # 4. Meeting Request
    for e in available:
        if e.get("category") == "meeting_request":
            start = datetime.now() + timedelta(days=5)
            end = start + timedelta(hours=1)
            return Action(
                action_type="schedule_meeting",
                email_id=e["id"],
                meeting_title=e.get("subject", "Sync Meeting"),
                meeting_start=start,
                meeting_end=end
            )

    return None

def get_llm_action(observation: dict) -> Action:
    """Fallback LLM decision layer."""
    try:
        obs_trimmed = {
            "inbox": observation.get("inbox", [])[:5],
            "calendar": observation.get("calendar", [])[:3],
            "score": observation.get("score_so_far")
        }
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": json.dumps(obs_trimmed, default=str)}],
            temperature=0.1
        )
        data = json.loads(response.choices[0].message.content.strip().replace("```json", "").replace("```", ""))
        return Action(**data)
    except:
        return Action(action_type="no_op")


def run_task(task_id: str = "easy"):
    env = EmailCalendarEnv(task_id=task_id)
    obs = env.reset()
    processed_ids = set()
    
    print(f"[START] task={task_id} env=email-calendar-env model={MODEL_NAME}")
    
    step_num = 0
    rewards = []
    
    try:
        while True:
            obs_dict = obs.model_dump()
            
            # Realism Variation (Reduced to 2% for precision)
            if random.random() < 0.02:
                action = Action(action_type="no_op")
            else:
                # Hybrid Decision Logic
                action = get_rule_based_action(obs_dict, processed_ids)
                if not action:
                    action = get_llm_action(obs_dict)
            
            # Step
            result = env.step(action)
            step_num += 1
            reward = round(result.reward, 2)
            rewards.append(reward)
            
            # Tracking Memory
            if action.email_id:
                # Urgent emails shouldn't be 'processed' immediately if they still require reply
                if action.action_type != "flag_email":
                    processed_ids.add(action.email_id)
                elif action.action_type == "flag_email" and not obs_dict.get("inbox", [])[0].get("requires_reply", False):
                     # If it doesn't need reply, we can archive it next or just mark as handled in memory
                     pass

            print(
                f"[STEP] step={step_num} "
                f"action={action.action_type} "
                f"reward={reward:.2f} "
                f"done={'true' if result.done else 'false'} "
                f"error={result.info.get('error', 'null')}"
            )
            
            obs = result.observation
            if result.done:
                break

    except Exception:
        pass
    finally:
        env.close()
        final_score = env.state().get("current_score", 0.0)
        success = final_score >= 0.5
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
