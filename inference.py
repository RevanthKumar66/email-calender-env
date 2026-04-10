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
    api_url = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
    api_key = os.environ.get("API_KEY", "")
    model_id = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

client = OpenAI(
    base_url=api_url,
    api_key=api_key,
    timeout=60.0
)

def get_llm_action(obs: dict) -> Action:
    """Sends state to LLM and returns Action."""
    # Data sanitization for JSON
    inbox = []
    for email in obs.get("inbox_emails", [])[:8]:
        data = email.copy()
        for k, v in data.items():
            if isinstance(v, datetime): data[k] = v.isoformat()
        inbox.append(data)

    calendar = []
    for event in obs.get("calendar_events", [])[:5]:
        data = event.copy()
        for k, v in data.items():
            if isinstance(v, datetime): data[k] = v.isoformat()
        calendar.append(data)

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "Return valid JSON for the next action."},
                {"role": "user", "content": json.dumps({"inbox": inbox, "calendar": calendar})}
            ],
            temperature=0,
            max_tokens=256
        )
        content = response.choices[0].message.content.strip()
        if "```" in content:
            content = content.split("```")[-2].strip()
            if content.startswith("json"): content = content[4:].strip()
        return Action(**json.loads(content))
    except Exception:
        return Action(action_type="no_op")


def run_session(task_type: str = "easy"):
    """Executes a session with mandatory structured logging for evaluation."""
    
    # Validation ping
    try:
        client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5
        )
    except Exception:
        pass

    env = EmailCalendarEnv(task_id=task_type)
    obs = env.reset()
    handled_ids: Set[str] = set()

    # Evaluation Tag: START
    print(f"[START] task={task_type}", flush=True)

    step = 1
    total_calls = 0

    while True:
        state = obs.model_dump()

        if step <= 10 or step % 3 == 0:
            action = get_llm_action(state)
            total_calls += 1
        else:
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

        # Evaluation Tag: STEP
        print(f"[STEP] step={step} action={action.action_type} reward={round(result.reward, 2)}", flush=True)

        obs = result.observation
        if result.done or step >= 50:
            break
        step += 1

    final_score = env.state().get("current_score", 0.0)
    
    # Evaluation Tag: END
    print(f"[END] task={task_type} score={final_score:.2f} steps={step}", flush=True)
    env.close()


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "easy"
    run_session(task)
