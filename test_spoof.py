import cv2
from modules.face_detection import FaceDetector
from modules.liveness import LivenessDetector
from modules.spoof_engine import SpoofEngine

detector = FaceDetector()
liveness = LivenessDetector()
spoof = SpoofEngine()

cap = cv2.VideoCapture(0)
print("📷 Webcam opened - Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face, coords = detector.detect_face(frame)

    if face is not None:
        # Get liveness details
        is_live, details = liveness.check_liveness(face)

        # Compute spoof probability
        result = spoof.compute_spoof_probability(
            blink_count=details["blink_count"],
            movement=details["head_movement"],
            ear=details["ear_value"],
        )

        # Display on screen
        color = (0, 255, 0) if not result["is_spoof"] else (0, 0, 255)

        cv2.putText(
            frame,
            f"Verdict: {result['verdict']}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )
        cv2.putText(
            frame,
            f"Spoof Prob: {result['spoof_probability']}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Blinks: {details['blink_count']}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Movement: {details['head_movement']}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"EAR: {details['ear_value']}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        print(
            f"Verdict: {result['verdict']} | "
            f"Spoof Prob: {result['spoof_probability']} | "
            f"Blinks: {details['blink_count']} | "
            f"Movement: {details['head_movement']}"
        )

    cv2.imshow("Spoof Detection Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
