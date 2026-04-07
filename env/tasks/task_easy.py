from faker import Faker
from datetime import datetime, timedelta
import random, uuid, pytz

fake = Faker()
IST = pytz.timezone("Asia/Kolkata")

def generate_easy_task():
    emails = []
    # Generate emails deterministically with seed
    Faker.seed(42)
    random.seed(42)
    
    specs = [
        ("urgent", "action_required", True, "Server outage alert"),
        ("urgent", "action_required", True, "Client escalation URGENT"),
        ("urgent", "meeting_request", False, "Emergency board call"),
        ("low", "spam", False, "You won a prize!"),
        ("low", "spam", False, "Limited offer expires today"),
        ("normal", "fyi", False, "Weekly newsletter"),
        ("normal", "fyi", False, "Team lunch Thursday"),
        ("normal", "fyi", False, "Policy update Q2"),
        ("normal", "action_required", True, "Please review attached contract"),
        ("normal", "action_required", True, "Approval needed for budget"),
    ]
    
    base_time = datetime(2026, 4, 7, 9, 0, 0, tzinfo=IST)
    for i, (priority, category, requires_reply, subject) in enumerate(specs):
        emails.append({
            "id": f"email_{i+1:03d}",
            "sender": fake.email(),
            "subject": subject,
            "body": fake.paragraph(nb_sentences=4),
            "timestamp": (base_time + timedelta(minutes=i*15)).isoformat(),
            "priority": priority,
            "category": category,
            "requires_reply": requires_reply,
            "deadline": (base_time + timedelta(hours=4)).isoformat() if priority == "urgent" else None
        })
    
    return {
        "task_id": "easy",
        "objective": "Triage the inbox: flag urgent emails, archive spam, and reply to action-required emails.",
        "emails": emails,
        "calendar": [],
        "max_steps": 20
    }
