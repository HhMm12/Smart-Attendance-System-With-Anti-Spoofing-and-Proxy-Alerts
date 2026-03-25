import cv2
from modules.session_controller import AttendanceSessionController

controller = AttendanceSessionController()

print(
    """
Commands:
  E — Enroll your face
  S — Start attendance session
  Q — Quit and show summary
"""
)

cap = cv2.VideoCapture(0)
session_started = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if session_started:
        # Process frame through full pipeline
        result, frame = controller.process_frame(frame)

        # Print result to terminal
        if result["status"] not in ["NO_FACE", "PROCESSING"]:
            print(
                f"Status: {result['status']} | "
                f"Spoof: {result['spoof_probability']} | "
                f"Student: {result['student_name']} | "
                f"Confidence: {result['confidence']}"
            )

    else:
        cv2.putText(
            frame,
            "Press E to Enroll | S to Start Session",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

    cv2.imshow("Smart Attendance System", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("e"):
        cap.release()
        cv2.destroyAllWindows()
        controller.enroll_student("RA2011001", "Hrithik")
        cap = cv2.VideoCapture(0)

    elif key == ord("s"):
        controller.start_session("CS101")
        session_started = True
        print("✅ Session started - stand in front of camera")

    elif key == ord("q"):
        summary = controller.end_session()
        print("\n===== SESSION SUMMARY =====")
        print(f"Session ID    : {summary['session_id']}")
        print(f"Total Present : {summary['total_present']}")
        print(f"Total Alerts  : {summary['total_alerts']}")
        for r in summary["records"]:
            print(f"  ✅ {r['name']} | Confidence: {r['confidence']}")
        for a in summary["alerts"]:
            print(f"  ⚠️  Alert: {a['reason']}")
        break

cap.release()
cv2.destroyAllWindows()
