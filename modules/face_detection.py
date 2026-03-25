import cv2
import numpy as np


class FaceDetector:
    def __init__(self):
        # Load OpenCV's built-in face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        print("✅ Face Detector initialized")

    def detect_face(self, frame):
        """
        Takes a webcam frame and returns the cropped face region
        Returns: cropped face image or None if no face found
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),  # minimum face size in pixels
        )

        # If no face found
        if len(faces) == 0:
            return None, None

        # Take the largest face found
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face

        # Add padding around the face
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)

        # Crop the face from the frame
        cropped_face = frame[y : y + h, x : x + w]

        # Return cropped face and its coordinates
        return cropped_face, (x, y, w, h)

    def draw_box(self, frame, coordinates, label="Face", color=(0, 255, 0)):
        """
        Draws a rectangle around detected face on the frame
        """
        if coordinates is None:
            return frame

        x, y, w, h = coordinates
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

    def is_valid_frame(self, frame):
        """
        Checks if the frame is valid and not empty
        """
        if frame is None:
            return False
        if frame.size == 0:
            return False
        return True
