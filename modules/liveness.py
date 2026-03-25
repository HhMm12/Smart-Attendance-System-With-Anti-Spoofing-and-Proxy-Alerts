import cv2
import numpy as np
import mediapipe as mp


class LivenessDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # EAR threshold
        self.EAR_THRESHOLD = 0.25
        self.MIN_BLINKS = 2
        self.MIN_HEAD_MOVEMENT = 3.0

        # Tracking variables
        self.blink_counter = 0
        self.eye_closed = False
        self.head_pose_history = []

        print("✅ Liveness Detector initialized")

    def get_landmarks(self, frame):
        """
        Gets facial landmarks using MediaPipe
        Returns list of (x,y) tuples or None
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        h, w = frame.shape[:2]
        landmarks = []
        for lm in results.multi_face_landmarks[0].landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            landmarks.append((x, y))

        return landmarks

    def calculate_EAR(self, eye_points):
        """
        Eye Aspect Ratio:
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        High EAR = eye open, Low EAR = eye closed
        """
        p1 = np.array(eye_points[0])
        p2 = np.array(eye_points[1])
        p3 = np.array(eye_points[2])
        p4 = np.array(eye_points[3])
        p5 = np.array(eye_points[4])
        p6 = np.array(eye_points[5])

        A = np.linalg.norm(p2 - p6)
        B = np.linalg.norm(p3 - p5)
        C = np.linalg.norm(p1 - p4)

        if C == 0:
            return 0.0

        ear = (A + B) / (2.0 * C)
        return ear

    def detect_blink(self, frame):
        """
        Detects blinks using MediaPipe landmark indices
        Left eye: 33,160,158,133,153,144
        Right eye: 362,385,387,263,373,380
        """
        landmarks = self.get_landmarks(frame)
        if landmarks is None:
            return self.blink_counter, 0.0

        # Extract eye landmarks using MediaPipe indices
        left_eye_idx = [33, 160, 158, 133, 153, 144]
        right_eye_idx = [362, 385, 387, 263, 373, 380]

        left_eye = [landmarks[i] for i in left_eye_idx]
        right_eye = [landmarks[i] for i in right_eye_idx]

        left_EAR = self.calculate_EAR(left_eye)
        right_EAR = self.calculate_EAR(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2.0

        # Blink detection — eye closes then reopens
        if avg_EAR < self.EAR_THRESHOLD:
            self.eye_closed = True
        else:
            if self.eye_closed:
                self.blink_counter += 1
                self.eye_closed = False

        return self.blink_counter, round(avg_EAR, 3)

    def track_head_movement(self, frame):
        """
        Tracks head movement using nose tip landmark (index 1)
        Measures variance of nose position over last 30 frames
        """
        landmarks = self.get_landmarks(frame)
        if landmarks is None:
            return 0.0

        # Nose tip is landmark index 1 in MediaPipe
        nose_tip = landmarks[1]
        self.head_pose_history.append(nose_tip)

        # Keep only last 30 frames
        if len(self.head_pose_history) > 30:
            self.head_pose_history.pop(0)

        if len(self.head_pose_history) < 5:
            return 0.0

        positions = np.array(self.head_pose_history)
        variance = np.var(positions, axis=0)
        movement_score = float(np.mean(variance))

        return round(movement_score, 3)

    def check_liveness(self, frame):
        """
        Main liveness check — combines blink + head movement
        Returns: is_live (bool), details (dict)
        """
        blinks, ear = self.detect_blink(frame)
        movement = self.track_head_movement(frame)

        is_live = blinks >= self.MIN_BLINKS and movement >= self.MIN_HEAD_MOVEMENT

        details = {
            "blink_count": blinks,
            "ear_value": ear,
            "head_movement": movement,
            "is_live": is_live,
        }

        return is_live, details

    def reset(self):
        """
        Resets all counters for a new session
        """
        self.blink_counter = 0
        self.eye_closed = False
        self.head_pose_history = []
