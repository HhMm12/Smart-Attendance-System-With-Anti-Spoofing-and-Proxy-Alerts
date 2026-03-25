import numpy as np
from database.db_handler import DatabaseHandler

db = DatabaseHandler()

# Test enrollment
fake_embedding = np.random.rand(512).astype(np.float32)
db.enroll_student("RA2011001", "Hrithik", fake_embedding)

# Test retrieval
embeddings = db.get_all_embeddings()
print(f"✅ Students enrolled: {len(embeddings)}")
for sid, data in embeddings.items():
    print(f"   - {sid}: {data['name']} | embedding shape: {data['embedding'].shape}")

# Test attendance recording
db.record_attendance(
    student_id="RA2011001",
    student_name="Hrithik",
    session_id="CS101_2026_03_10",
    confidence=0.91,
    spoof_probability=0.12,
    status="PRESENT",
)

# Test spoof alert
db.record_spoof_alert(
    session_id="CS101_2026_03_10",
    spoof_probability=0.87,
    blink_count=0,
    head_movement=0.5,
    ear_value=0.18,
    reason="Blink rate below threshold. Head movement insufficient.",
)

# Fetch records
records = db.get_all_attendance()
print(f"✅ Attendance records: {len(records)}")

alerts = db.get_spoof_alerts()
print(f"✅ Spoof alerts: {len(alerts)}")
