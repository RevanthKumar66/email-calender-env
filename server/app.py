from fastapi import FastAPI, HTTPException
from env.email_calendar_env import EmailCalendarEnv
from env.models import Action, Observation, StepResult
import os
# from dotenv import load_dotenv

# Load environment variables from .env file
# load_dotenv()

app = FastAPI(title="Email-Calendar OpenEnv", version="1.0.0")

_envs: dict = {}

@app.get("/")
def root():
    return {
        "message": "Email + Calendar RL Environment is ONLINE",
        "docs": "/docs",
        "status": "ready"
    }

@app.get("/health")
def health():
    return {"status": "ok", "env": "email-calendar-env"}

@app.post("/reset")
@app.post("/env/reset")
def reset(task_id: str = "easy"):
    env = EmailCalendarEnv(task_id=task_id)
    session_id = f"session_{task_id}"
    _envs[session_id] = env
    obs = env.reset()
    return {"session_id": session_id, "observation": obs.model_dump()}

@app.post("/step/{session_id}")
@app.post("/env/step/{session_id}")
def step(session_id: str, action: Action):
    env = _envs.get(session_id)
    if not env:
        raise HTTPException(status_code=404, detail="Session not found. Call /env/reset first.")
    result = env.step(action)
    return result.model_dump()

@app.get("/state/{session_id}")
@app.get("/env/state/{session_id}")
def state(session_id: str):
    env = _envs.get(session_id)
    if not env:
        raise HTTPException(status_code=404, detail="Session not found.")
    return env.state()

@app.post("/close/{session_id}")
@app.post("/env/close/{session_id}")
def close(session_id: str):
    env = _envs.get(session_id)
    if env:
        env.close()
        del _envs[session_id]
    return {"status": "closed"}

def main():
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
