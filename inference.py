import os
import json
import sys
from datetime import datetime
from typing import Set

from openai import OpenAI
from env.email_calendar_env import EmailCalendarEnv
from env.models import Action

# API Configuration
try:
    api_url = os.environ["API_BASE_URL"]
    api_key = os.environ["API_KEY"]
    model_id = os.environ["MODEL_NAME"]
except KeyError:
    # Fallback to defaults if environment variables are not set
    api_url = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
    api_key = os.environ.get("API_KEY", "")
    model_id = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

client = OpenAI(
    base_url=api_url,
    api_key=api_key,
    timeout=60.0
)

def get_llm_action(obs: dict) -> Action:
    """Sends current state to LLM and returns the parsed Action object."""
    
    # Process emails for JSON serialization
    inbox = []
    for email in obs.get("inbox", [])[:8]:
        data = email.copy()
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        inbox.append(data)

    # Process calendar events
    calendar = []
    for event in obs.get("calendar", [])[:5]:
        data = event.copy()
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        calendar.append(data)

    prompt_context = {
        "emails": inbox,
        "calendar": calendar,
        "current_time": datetime.now().isoformat()
    }

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a triage assistant. Return valid JSON only."},
                {"role": "user", "content": json.dumps(prompt_context)}
            ],
            temperature=0,
            max_tokens=256
        )

        content = response.choices[0].message.content.strip()
        
        # Extract JSON if wrapped in markdown blocks
        if "```" in content:
            content = content.split("```")[-2].strip()
            if content.startswith("json"):
                content = content[4:].strip()
        
        return Action(**json.loads(content))

    except Exception:
        return Action(action_type="no_op")


def run_session(task_type: str = "easy"):
    """Main loop for running an environment session."""
    
    # Initial connection verification
    try:
        client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=5
        )
    except Exception:
        pass

    env = EmailCalendarEnv(task_id=task_type)
    obs = env.reset()
    handled_ids: Set[str] = set()

    step = 1
    total_calls = 0

    while True:
        state = obs.model_dump()

        # Decision logic: prioritizing LLM for complex tasks
        if step <= 10 or step % 3 == 0:
            action = get_llm_action(state)
            total_calls += 1
        else:
            # Simple rule-based logic for efficiency
            action = Action(action_type="no_op")
            pending = [e for e in state.get("inbox", []) if e.get("id") not in handled_ids]
            
            for item in pending:
                if item.get("priority") == "urgent":
                    action = Action(action_type="flag_email", email_id=item["id"])
                    break
                    
            if action.action_type == "no_op":
                action = get_llm_action(state)
                total_calls += 1

        result = env.step(action)
        
        if getattr(action, 'email_id', None):
            handled_ids.add(action.email_id)

        print(f"[{task_type}] Step {step}: {action.action_type} (Score: {result.reward})")

        obs = result.observation
        if result.done or step >= 30:
            break
        step += 1

    print(f"Completed {task_type} in {step} steps. LLM calls: {total_calls}")
    env.close()


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "easy"
    run_session(task)
