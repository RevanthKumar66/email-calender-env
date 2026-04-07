from faker import Faker
from datetime import datetime, timedelta
import random, uuid, pytz

fake = Faker()
IST = pytz.timezone("Asia/Kolkata")

def generate_hard_task():
    Faker.seed(789)
    random.seed(789)
    
    emails = []
    base_time = datetime(2026, 4, 7, 9, 0, 0, tzinfo=IST)
    
    # 3 VIP emails
    vip_ids = []
    for i in range(3):
        eid = f"email_{i+1:03d}"
        vip_ids.append(eid)
        emails.append({
            "id": eid,
            "sender": f"vip_{i+1}@boss.com",
            "subject": f"URGENT: VIP Request {i+1}",
            "body": "This needs your immediate attention.",
            "timestamp": (base_time + timedelta(minutes=i*5)).isoformat(),
            "priority": "urgent",
            "category": "action_required",
            "requires_reply": True,
            "deadline": (base_time + timedelta(hours=1)).isoformat()
        })
        
    # 10 meeting requests
    for i in range(3, 13):
        emails.append({
            "id": f"email_{i+1:03d}",
            "sender": fake.email(),
            "subject": f"Meeting Request: {fake.catch_phrase()}",
            "body": f"Hi, let's meet to discuss {fake.word()}.",
            "timestamp": (base_time + timedelta(minutes=i*10)).isoformat(),
            "priority": "normal",
            "category": "meeting_request",
            "requires_reply": True,
            "deadline": None
        })
        
    # Others to make 50
    deadline_ids = []
    for i in range(13, 50):
        priority = random.choice(["urgent", "normal", "low"])
        category = random.choice(["action_required", "fyi", "spam"])
        requires_reply = (category == "action_required")
        eid = f"email_{i+1:03d}"
        deadline = None
        if i < 16: # 3 deadline-bound
            deadline = (base_time + timedelta(hours=random.randint(2, 4)))
            deadline_ids.append(eid)
            
        emails.append({
            "id": eid,
            "sender": fake.email(),
            "subject": fake.sentence(nb_words=4),
            "body": fake.paragraph(nb_sentences=3),
            "timestamp": (base_time + timedelta(minutes=i*12)).isoformat(),
            "priority": priority,
            "category": category,
            "requires_reply": requires_reply,
            "deadline": deadline.isoformat() if deadline else (base_time + timedelta(hours=6)).isoformat() if priority == "urgent" else None
        })
        
    # 5 conflicting calendar events
    calendar = []
    for i in range(5):
        calendar.append({
            "id": f"evt_{i+1:03d}",
            "title": f"Busy block {i+1}",
            "start": (base_time + timedelta(hours=i*2 + 1)).isoformat(),
            "end": (base_time + timedelta(hours=i*2 + 2)).isoformat(),
            "attendees": ["me@work.com"],
            "timezone": "Asia/Kolkata"
        })
        
    return {
        "task_id": "hard",
        "objective": "Handle 50 emails and 10 meetings while managing VIPs and deadlines.",
        "emails": emails,
        "calendar": calendar,
        "max_steps": 100,
        "vip_email_ids": vip_ids,
        "deadline_email_ids": deadline_ids,
        "required_meetings": 10
    }
