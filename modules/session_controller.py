import cv2
import time
import numpy as np
from datetime import datetime
from modules.face_detection import FaceDetector
from modules.face_alignment import FaceAligner
from modules.liveness import LivenessDetector
from modules.spoof_engine import SpoofEngine
from modules.recognition import FaceRecognizer
from modules.alert_engine import AlertEngine
from database.db_handler import DatabaseHandler
from database.models import AttendanceRecord


class AttendanceSessionController:
    def __init__(self):
        self.detector = FaceDetector()
        self.aligner = FaceAligner()
        self.liveness = LivenessDetector()
        self.spoof = SpoofEngine()
        self.recognizer = FaceRecognizer()
        self.alert_engine = AlertEngine()
        self.db = DatabaseHandler()

        self.session_id = None
        self.is_active = False
        self.results = []
        self._last_alert_time = 0

        print("✅ AttendanceSession Controller initialized")

    def start_session(self, course_id="CS101"):
        self.session_id = f"{course_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.is_active = True
        self.results = []
        self._last_alert_time = 0
        self.liveness.reset()
        print(f"✅ Session started: {self.session_id}")
        return self.session_id

    def end_session(self):
        self.is_active = False
        print(f"✅ Session ended: {self.session_id}")
        return self.get_session_summary()

    def enroll_student(self, student_id: str, name: str):
        """Enroll a student by capturing frames from webcam silently (no GUI)"""
        print(f"📷 Enrolling {name}. Look at the camera...")

        cap = cv2.VideoCapture(0)
        embeddings = []
        frame_count = 0
        target_frames = 15
        attempts = 0
        max_attempts = 100

        print(f"   Capturing {target_frames} frames...")

        while frame_count < target_frames and attempts < max_attempts:
            ret, frame = cap.read()
            if not ret:
                attempts += 1
                continue

            attempts += 1

            face, coords = self.detector.detect_face(frame)
            if face is None:
                continue

            aligned = self.aligner.align_face(face)
            if aligned is None:
                continue

            embedding = self.recognizer.generate_embedding(aligned)
            if embedding is not None:
                embeddings.append(embedding)
                frame_count += 1
                print(f"   Frame {frame_count}/{target_frames} captured")

        cap.release()

        if len(embeddings) == 0:
            print(f"❌ Enrollment failed — no face detected for {name}")
            return False

        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        self.db.enroll_student(student_id, name, avg_embedding)
        print(f"✅ {name} enrolled successfully with {len(embeddings)} frames")
        return True

    def _record_spoof(self, spoof_result, liveness_details):
        """Record spoof alert with 5 second cooldown"""
        current_time = time.time()
        if current_time - self._last_alert_time > 5:
            self._last_alert_time = current_time
            alert = self.alert_engine.generate_explanation(
                spoof_result=spoof_result,
                liveness_details=liveness_details,
                session_id=self.session_id,
            )
            self.db.record_spoof_alert(
                session_id=self.session_id,
                spoof_probability=spoof_result["spoof_probability"],
                blink_count=liveness_details["blink_count"],
                head_movement=liveness_details["head_movement"],
                ear_value=liveness_details["ear_value"],
                reason=alert["summary"],
            )
            return alert
        return None

    def _auto_mark_absent(self, face, spoof_result, match_result=None):
        """Auto mark student absent when high confidence spoof detected"""
        try:
            stored = self.db.get_all_embeddings()
            if not stored:
                return

            # Use provided match or try to identify who it is
            if match_result and match_result.get("matched"):
                student_id = match_result["student_id"]
                student_name = stored[student_id]["name"]
                confidence = match_result["confidence"]
            else:
                # Try to identify
                embeddings_dict = {
                    sid: data["embedding"] for sid, data in stored.items()
                }
                aligned = self.aligner.align_face(face)
                live_embedding = self.recognizer.generate_embedding(aligned)
                match = self.recognizer.match_face(live_embedding, embeddings_dict)
                if not match["matched"]:
                    return
                student_id = match["student_id"]
                student_name = stored[student_id]["name"]
                confidence = match["confidence"]

            existing = (
                self.db.session.query(AttendanceRecord)
                .filter_by(student_id=student_id, session_id=self.session_id)
                .first()
            )

            if existing:
                existing.status = "ABSENT"
                self.db.session.commit()
                print(f"🚨 {student_name} marked ABSENT — high confidence spoof")
            else:
                self.db.record_attendance(
                    student_id=student_id,
                    student_name=student_name,
                    session_id=self.session_id,
                    confidence=confidence,
                    spoof_probability=spoof_result["spoof_probability"],
                    status="ABSENT",
                )
                print(f"🚨 {student_name} recorded as ABSENT — high confidence spoof")

        except Exception as e:
            print(f"Auto-absent error: {e}")
            try:
                self.db.session.rollback()
            except:
                pass

    def process_frame(self, frame):
        """Main pipeline — processes one webcam frame through all modules"""
        result = {
            "face_detected": False,
            "is_live": False,
            "is_spoof": False,
            "spoof_probability": 0.0,
            "matched": False,
            "student_id": None,
            "student_name": None,
            "confidence": 0.0,
            "alert": None,
            "status": "PROCESSING",
        }

        # Step 1 — Face Detection
        face, coords = self.detector.detect_face(frame)
        if face is None:
            result["status"] = "NO_FACE"
            return result, frame

        result["face_detected"] = True
        frame = self.detector.draw_box(frame, coords)

        # Step 2 — Face Alignment
        aligned = self.aligner.align_face(face)

        # Step 3 — Liveness Detection
        is_live, liveness_details = self.liveness.check_liveness(aligned)
        result["is_live"] = is_live

        # Step 4 — Spoof Probability
        spoof_result = self.spoof.compute_spoof_probability(
            face_img=face,
            blink_count=liveness_details["blink_count"],
            movement=liveness_details["head_movement"],
            ear=liveness_details["ear_value"],
        )
        result["is_spoof"] = spoof_result["is_spoof"]
        result["spoof_probability"] = spoof_result["spoof_probability"]

        # Get ML score for hard decisions
        ml_score = spoof_result.get("ml_score", 0) or 0

        color = (0, 255, 0) if not spoof_result["is_spoof"] else (0, 0, 255)
        cv2.putText(
            frame,
            f"Spoof: {spoof_result['spoof_probability']} | Blinks: {liveness_details['blink_count']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        # Step 5 — Reject if spoof
        # ML score above 0.9 = high confidence photo/screen attack → auto mark ABSENT
        # ML score above 0.5 = spoof detected → reject but don't record
        if ml_score > 0.9 or spoof_result["is_spoof"]:
            alert = self._record_spoof(spoof_result, liveness_details)
            result["alert"] = alert or self.alert_engine.generate_explanation(
                spoof_result=spoof_result,
                liveness_details=liveness_details,
                session_id=self.session_id,
            )
            result["status"] = "SPOOF_REJECTED"

            # High confidence spoof — auto mark absent
            if ml_score > 0.9:
                self._auto_mark_absent(face, spoof_result)

            cv2.putText(
                frame,
                "SPOOF DETECTED - REJECTED",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            return result, frame

        # Step 6 — Face Recognition
        stored = self.db.get_all_embeddings()

        if len(stored) == 0:
            result["status"] = "NO_STUDENTS_ENROLLED"
            return result, frame

        embeddings_dict = {sid: data["embedding"] for sid, data in stored.items()}
        live_embedding = self.recognizer.generate_embedding(aligned)
        match = self.recognizer.match_face(live_embedding, embeddings_dict)

        result["matched"] = match["matched"]
        result["confidence"] = match["confidence"]

        if match["matched"]:
            student_id = match["student_id"]
            student_name = stored[student_id]["name"]

            # ── HARD SPOOF GATE ──
            # Second check after recognition — if ML score still high, reject and mark absent
            if ml_score > 0.9:
                alert = self._record_spoof(spoof_result, liveness_details)
                result["alert"] = alert or self.alert_engine.generate_explanation(
                    spoof_result=spoof_result,
                    liveness_details=liveness_details,
                    session_id=self.session_id,
                )
                result["status"] = "SPOOF_REJECTED"

                self._auto_mark_absent(
                    face,
                    spoof_result,
                    match_result={
                        "matched": True,
                        "student_id": student_id,
                        "confidence": match["confidence"],
                    },
                )

                cv2.putText(
                    frame,
                    "SPOOF DETECTED - REJECTED",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                return result, frame

            # ── GENUINE — safe to record attendance ──
            result["student_id"] = student_id
            result["student_name"] = student_name
            result["status"] = "PRESENT"

            alert = self.alert_engine.generate_explanation(
                spoof_result=spoof_result,
                liveness_details=liveness_details,
                student_id=student_id,
                session_id=self.session_id,
            )
            result["alert"] = alert

            already_recorded = (
                self.db.session.query(AttendanceRecord)
                .filter_by(student_id=student_id, session_id=self.session_id)
                .first()
            )

            if not already_recorded:
                self.db.record_attendance(
                    student_id=student_id,
                    student_name=student_name,
                    session_id=self.session_id,
                    confidence=match["confidence"],
                    spoof_probability=spoof_result["spoof_probability"],
                    status="PRESENT",
                )
                print(f"✅ Attendance recorded: {student_name}")

            cv2.putText(
                frame,
                f"✅ {student_name} | {match['confidence']*100:.1f}%",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        else:
            result["status"] = "NO_MATCH"
            cv2.putText(
                frame,
                "❌ Face not recognized",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
            )

        return result, frame

    def get_session_summary(self):
        records = self.db.get_attendance_by_session(self.session_id)
        alerts = self.db.get_spoof_alerts(self.session_id)

        return {
            "session_id": self.session_id,
            "total_present": len(records),
            "total_alerts": len(alerts),
            "records": [
                {
                    "student_id": r.student_id,
                    "name": r.student_name,
                    "confidence": r.confidence,
                    "spoof_prob": r.spoof_probability,
                    "status": r.status,
                    "time": str(r.timestamp),
                }
                for r in records
            ],
            "alerts": [
                {
                    "spoof_prob": a.spoof_probability,
                    "reason": a.reason,
                    "time": str(a.timestamp),
                }
                for a in alerts
            ],
        }
