import cv2
from modules.face_detection import FaceDetector
from modules.liveness import LivenessDetector

detector = FaceDetector()
liveness = LivenessDetector()

cap = cv2.VideoCapture(0)
print("📷 Webcam opened - Blink twice and move your head - Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect face
    face, coords = detector.detect_face(frame)

    if face is not None:
        # Check liveness
        is_live, details = liveness.check_liveness(face)

        # Display info on screen
        color = (0, 255, 0) if is_live else (0, 0, 255)
        status = "LIVE ✅" if is_live else "SPOOF ❌"

        cv2.putText(
            frame,
            f"Status: {status}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )
        cv2.putText(
            frame,
            f"Blinks: {details['blink_count']}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"EAR: {details['ear_value']}",
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

        print(
            f"Blinks: {details['blink_count']} | "
            f"EAR: {details['ear_value']} | "
            f"Movement: {details['head_movement']} | "
            f"Live: {is_live}"
        )

    cv2.imshow("Liveness Detection Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
