import cv2
import numpy as np
import mediapipe as mp


class FaceAligner:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        self.target_size = (112, 112)
        print("✅ Face Aligner initialized")

    def align_face(self, frame):
        """
        Aligns face using eye positions from MediaPipe
        Returns 112x112 aligned face
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return cv2.resize(frame, self.target_size)

        h, w = frame.shape[:2]
        landmarks = results.multi_face_landmarks[0].landmark

        # Left eye center (index 33), Right eye center (index 263)
        left_eye = (int(landmarks[33].x * w), int(landmarks[33].y * h))
        right_eye = (int(landmarks[263].x * w), int(landmarks[263].y * h))

        # Calculate angle between eyes
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # Eye center point
        eye_center = (
            (left_eye[0] + right_eye[0]) // 2,
            (left_eye[1] + right_eye[1]) // 2,
        )

        # Rotate frame to straighten face
        M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        rotated = cv2.warpAffine(frame, M, (w, h))

        # Resize to standard size
        aligned = cv2.resize(rotated, self.target_size)
        return aligned

    def preprocess_for_arcface(self, aligned_face):
        """
        Converts aligned face to format ready for ArcFace
        """
        rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        norm = (rgb.astype(np.float32) - 127.5) / 127.5
        return norm
