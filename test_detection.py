import cv2
from modules.face_detection import FaceDetector

# Initialize detector
detector = FaceDetector()

# Open webcam
cap = cv2.VideoCapture(0)
print("📷 Webcam opened - Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read from webcam")
        break

    # Detect face
    face, coords = detector.detect_face(frame)

    # Draw box around face
    if coords is not None:
        frame = detector.draw_box(frame, coords, label="Face Detected")
        print("✅ Face detected")
    else:
        print("❌ No face found")

    # Show the frame
    cv2.imshow("Face Detection Test", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
