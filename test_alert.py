from modules.alert_engine import AlertEngine

alert_engine = AlertEngine()

# Simulate a SPOOF case
spoof_result = {"spoof_probability": 0.87, "is_spoof": True}
liveness_details = {"blink_count": 0, "head_movement": 0.3, "ear_value": 0.12}

alert = alert_engine.generate_explanation(
    spoof_result=spoof_result,
    liveness_details=liveness_details,
    student_id="RA2011001",
    session_id="CS101_2026_03_10",
)

print(alert_engine.format_for_dashboard(alert))
print()

# Simulate a GENUINE case
spoof_result2 = {"spoof_probability": 0.1, "is_spoof": False}
liveness_details2 = {"blink_count": 3, "head_movement": 5.2, "ear_value": 0.31}

alert2 = alert_engine.generate_explanation(
    spoof_result=spoof_result2,
    liveness_details=liveness_details2,
    student_id="RA2011001",
    session_id="CS101_2026_03_10",
)

print(alert_engine.format_for_dashboard(alert2))
