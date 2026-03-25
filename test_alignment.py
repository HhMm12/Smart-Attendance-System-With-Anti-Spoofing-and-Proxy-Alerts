import cv2
from modules.face_detection import FaceDetector
from modules.face_alignment import FaceAligner

detector = FaceDetector()
aligner = FaceAligner()

cap = cv2.VideoCapture(0)
print("📷 Webcam opened - Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Step 1 - Detect face
    face, coords = detector.detect_face(frame)

    if face is not None:
        # Step 2 - Align face
        aligned = aligner.align_face(face)

        # Show both original and aligned face
        cv2.imshow("Original Face", face)
        cv2.imshow("Aligned Face (112x112)", aligned)
        print("✅ Face aligned successfully")
    else:
        print("❌ No face detected")

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
