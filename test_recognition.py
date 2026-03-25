import cv2
import numpy as np
from modules.face_detection import FaceDetector
from modules.face_alignment import FaceAligner
from modules.recognition import FaceRecognizer

detector = FaceDetector()
aligner = FaceAligner()
recognizer = FaceRecognizer()

# Simulate stored embeddings database with a test entry
stored_embeddings = {}

cap = cv2.VideoCapture(0)
print("📷 Press E to enroll your face, R to recognize, Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face, coords = detector.detect_face(frame)

    if face is not None:
        aligned = aligner.align_face(face)

        cv2.putText(
            frame,
            "Press E to Enroll | R to Recognize",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        if coords:
            frame = detector.draw_box(frame, coords)

    cv2.imshow("Recognition Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("e") and face is not None:
        # Enroll current face
        aligned = aligner.align_face(face)
        embedding = recognizer.generate_embedding(aligned)
        stored_embeddings["Student_001"] = embedding
        print("✅ Face enrolled as Student_001")

    elif key == ord("r") and face is not None:
        if len(stored_embeddings) == 0:
            print("❌ No enrolled faces. Press E first.")
        else:
            # Recognize current face
            aligned = aligner.align_face(face)
            embedding = recognizer.generate_embedding(aligned)
            result = recognizer.match_face(embedding, stored_embeddings)
            print(f"Result: {result['verdict']}")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
