import os
import json
from typing import Optional, Set
from openai import OpenAI
from env.email_calendar_env import EmailCalendarEnv
from env.models import Action
from datetime import datetime, timedelta

# --- Configuration ---
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# --- LLM Setup ---
# Prioritize specific environment variables for API configuration
_base_url = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
_api_key = os.environ.get("API_KEY", os.environ.get("HF_TOKEN"))

# Initialize the OpenAI-compatible client with a dedicated timeout
client = OpenAI(
    base_url=_base_url,
    api_key=_api_key or "MISSING_KEY",
    timeout=60.0
)

PROMPT = "You are an assistant triaging emails and calendar. Triage urgent items first. Return pure JSON."

def get_heuristic_action(obs: dict, seen: Set[str]) -> Optional[Action]:
    """Identify urgent and spam emails based on observation tags."""
    inbox = obs.get("inbox", [])
    if not inbox: return Action(action_type="no_op")
    
    # Filter for emails not yet processed in current session
    tasks = [e for e in inbox if e["id"] not in seen]
    
    for e in tasks:
        if e.get("priority") == "urgent":
            return Action(action_type="flag_email", email_id=e["id"])
    for e in tasks:
        if e.get("category") == "spam":
            return Action(action_type="archive_email", email_id=e["id"])
    return None

def get_llm_action(obs: dict) -> Action:
    """Primary decision logic using LLM inference."""
    context = {
        "emails": obs.get("inbox", [])[:5],
        "schedule": obs.get("calendar", [])[:3]
    }
    
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": json.dumps(context)}
        ],
        temperature=0.0
    )
    
    raw = completion.choices[0].message.content.strip()
    clean_json = raw.replace("```json", "").replace("```", "").strip()
    return Action(**json.loads(clean_json))

def run_task(task_id: str = "easy"):
    """Main loop for environment interaction and state management."""
    env = EmailCalendarEnv(task_id=task_id)
    obs = env.reset()
    seen_emails = set()
    
    # Standard lifecycle start log
    print(f"[START] task={task_id} env=email-calendar-env model={MODEL_NAME}")
    
    step_idx = 1
    total_rewards = []
    
    while True:
        state = obs.model_dump()
        
        # Ensure model engagement during initial steps
        if step_idx <= 2:
            action = get_llm_action(state)
        else:
            action = get_heuristic_action(state, seen_emails)
            if not action:
                action = get_llm_action(state)
                
        # Execute action and capture environment response
        result = env.step(action)
        reward = round(result.reward, 2)
        total_rewards.append(reward)
        
        if action.email_id:
            seen_emails.add(action.email_id)

        # Log step result in required structured format
        print(f"[STEP] step={step_idx} action={action.action_type} reward={reward:.2f} done={'true' if result.done else 'false'}")
        
        obs = result.observation
        if result.done or step_idx >= 15:
            break
        step_idx += 1

    # Finalize task and report aggregate metrics
    final_score = env.state().get("current_score", 0.0)
    success = "true" if final_score >= 0.5 else "false"
    reward_chain = ",".join(f"{r:.2f}" for r in total_rewards)
    print(f"[END] success={success} steps={step_idx} rewards={reward_chain}")
    env.close()

if __name__ == "__main__":
    import sys
    task_name = sys.argv[1] if len(sys.argv) > 1 else "easy"
    run_task(task_name)
