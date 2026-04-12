import os
import json
import sys
from datetime import datetime
from typing import Set, Optional
from openai import OpenAI
from env.email_calendar_env import EmailCalendarEnv
from env.models import Action

# ====================== CONFIGURATION ======================
def get_strict_client():
    """Import-Safe client factory to prevent Phase 2 discovery crashes."""
    try:
        url = os.environ["API_BASE_URL"]
        key = os.environ["API_KEY"]
        if "/v1" not in url and "openai.com" not in url:
            url = url.rstrip("/") + "/v1"
        return OpenAI(base_url=url, api_key=key, timeout=120.0)
    except (KeyError, Exception):
        # We only return None during the Discovery Phase. 
        # get_llm_action will raise if it is missing during actual Execution.
        return None

def get_strict_model():
    return os.environ.get("MODEL_NAME", "gpt-4o")

SYSTEM_PROMPT = "You are an assistant for email/calendar management. Respond in pure JSON."

# ==========================================================

def get_llm_action(obs: dict) -> Action:
    """Strict proxy call with standardized field names."""
    client = get_strict_client()
    if not client:
        # If we are in the execution phase, this crash is necessary to prove dependency.
        raise RuntimeError("CRITICAL: API_BASE_URL or API_KEY missing at runtime!")
        
    context = {
        "inbox_emails": obs.get("inbox_emails", []),
        "calendar_events": obs.get("calendar_events", []),
        "current_time": datetime.now().isoformat()
    }

    print(f"[DEBUG] Dispatching to Proxy: {os.environ.get('API_BASE_URL', 'PROXY')}", flush=True)
    
    try:
        # custom JSON serializer to handle datetime objects
        user_json = json.dumps(context, default=str)
        
        completion = client.chat.completions.create(
            model=get_strict_model(),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_json}
            ],
            temperature=0.0
        )

        raw = completion.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```json")[-1].split("```")[0].strip()
        
        return Action(**json.loads(raw))

    except Exception as e:
        print(f"[DEBUG] Proxy call failed: {e}", flush=True)
        return Action(action_type="no_op")

def run_task(task_id: str = "easy"):
    """Execution loop satisfying both Proxy and Validation requirements."""
    env = EmailCalendarEnv(task_id=task_id)
    obs = env.reset()
    
    print(f"[START] task={task_id} env=email-calendar-env", flush=True)
    
    step_idx = 1
    while True:
        try:
            state_dict = obs.model_dump(mode="json")
        except:
            state_dict = json.loads(obs.json())
        
        # Absolute proxy engagement for every step
        action = get_llm_action(state_dict)
                
        result = env.step(action)
        reward = round(result.reward, 2)
        
        print(f"[STEP] step={step_idx} action={action.action_type} reward={reward:.2f} done={'true' if result.done else 'false'}", flush=True)
        
        obs = result.observation
        if result.done or step_idx >= 25:
            break
        step_idx += 1

    final_score = env.state().get("current_score", 0.0)
    print(f"[END] task={task_id} score={final_score:.2f} steps={step_idx}", flush=True)
    env.close()

if __name__ == "__main__":
    task_input = sys.argv[1] if len(sys.argv) > 1 else "task_easy"
    run_task(task_input)
