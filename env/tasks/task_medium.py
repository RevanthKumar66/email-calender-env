from faker import Faker
from datetime import datetime, timedelta
import random, uuid, pytz

fake = Faker()
IST = pytz.timezone("Asia/Kolkata")

def generate_medium_task():
    Faker.seed(123)
    random.seed(123)
    
    emails = []
    base_time = datetime(2026, 4, 7, 9, 0, 0, tzinfo=IST)
    
    # 5 specific meeting request emails
    for i in range(5):
        emails.append({
            "id": f"email_{i+1:03d}",
            "sender": fake.email(),
            "subject": f"Meeting Request: {fake.catch_phrase()}",
            "body": f"Hi, can we schedule a meeting at {(base_time + timedelta(hours=i+2)).strftime('%H:%M')} today? It should take 1 hour.",
            "timestamp": (base_time + timedelta(minutes=i*10)).isoformat(),
            "priority": "normal",
            "category": "meeting_request",
            "requires_reply": True,
            "deadline": None
        })
    
    # 20 more mixed emails
    for i in range(5, 25):
        priority = random.choice(["urgent", "normal", "low"])
        category = random.choice(["action_required", "fyi", "spam"])
        requires_reply = (category == "action_required")
        
        emails.append({
            "id": f"email_{i+1:03d}",
            "sender": fake.email(),
            "subject": fake.sentence(nb_words=4),
            "body": fake.paragraph(nb_sentences=3),
            "timestamp": (base_time + timedelta(minutes=i*12)).isoformat(),
            "priority": priority,
            "category": category,
            "requires_reply": requires_reply,
            "deadline": (base_time + timedelta(hours=6)).isoformat() if priority == "urgent" else None
        })
    
    return {
        "task_id": "medium",
        "objective": "Triage 25 emails and schedule 5 meetings without conflicts.",
        "emails": emails,
        "calendar": [],
        "max_steps": 40,
        "required_meetings": 5
    }
