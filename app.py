from flask import Flask, render_template, jsonify, request, Response
import cv2
import threading
from database.models import Student, AttendanceRecord
from modules.session_controller import AttendanceSessionController

app = Flask(__name__)

# ── GLOBALS ──
controller = AttendanceSessionController()
camera = None
session_active = False
latest_result = {}
frame_lock = threading.Lock()
current_frame = None


# ── CAMERA ──
def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera


def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None


def capture_frames():
    global current_frame, latest_result, session_active
    cam = get_camera()
    while session_active:
        ret, frame = cam.read()
        if not ret:
            break
        result, processed_frame = controller.process_frame(frame)
        latest_result = result
        _, buffer = cv2.imencode(".jpg", processed_frame)
        with frame_lock:
            current_frame = buffer.tobytes()
    release_camera()


def generate_frames():
    global current_frame
    while True:
        with frame_lock:
            if current_frame is None:
                continue
            frame = current_frame
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


# ── PAGES ──
@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ── SESSION ──
@app.route("/start_session", methods=["POST"])
def start_session():
    global session_active
    data = request.json
    course_id = data.get("course_id", "CS101")

    controller.start_session(course_id)
    session_active = True

    thread = threading.Thread(target=capture_frames, daemon=True)
    thread.start()

    return jsonify(
        {
            "success": True,
            "session_id": controller.session_id,
            "message": f"Session {controller.session_id} started",
        }
    )


@app.route("/end_session", methods=["POST"])
def end_session():
    global session_active
    session_active = False
    summary = controller.end_session()

    try:
        controller.db.session.commit()
    except Exception as e:
        controller.db.session.rollback()
        print(f"DB commit error: {e}")

    return jsonify({"success": True, "summary": summary})


# ── ENROLL ──
@app.route("/enroll", methods=["POST"])
def enroll():
    data = request.json
    student_id = data.get("student_id")
    name = data.get("name")

    if not student_id or not name:
        return jsonify(
            {"success": False, "message": "student_id and name are required"}
        )

    def do_enroll():
        controller.enroll_student(student_id, name)

    thread = threading.Thread(target=do_enroll, daemon=True)
    thread.start()

    return jsonify(
        {"success": True, "message": f"Enrolling {name} — please look at the camera."}
    )


# ── LATEST RESULT ──
@app.route("/latest_result")
def get_latest_result():
    return jsonify(latest_result)


# ── ATTENDANCE ──
@app.route("/attendance")
def get_attendance():
    try:
        controller.db.session.expire_all()
        records = controller.db.get_all_attendance()
        data = [
            {
                "student_id": r.student_id,
                "name": r.student_name,
                "session_id": r.session_id,
                "confidence": r.confidence,
                "spoof_prob": r.spoof_probability,
                "status": r.status,
                "timestamp": str(r.timestamp),
            }
            for r in records
        ]
        return jsonify(data)
    except Exception as e:
        print(f"Attendance fetch error: {e}")
        return jsonify([])


# ── ALERTS ──
@app.route("/alerts")
def get_alerts():
    try:
        controller.db.session.expire_all()
        alerts = controller.db.get_spoof_alerts()
        data = [
            {
                "session_id": a.session_id,
                "spoof_prob": a.spoof_probability,
                "blink_count": a.blink_count,
                "head_movement": a.head_movement,
                "ear_value": a.ear_value,
                "reason": a.reason,
                "timestamp": str(a.timestamp),
            }
            for a in alerts
        ]
        return jsonify(data)
    except Exception as e:
        print(f"Alerts fetch error: {e}")
        return jsonify([])


# ── STUDENTS ──
@app.route("/students_page")
def students_page():
    try:
        controller.db.session.expire_all()
        students = controller.db.session.query(Student).all()
        data = [
            {
                "student_id": s.student_id,
                "name": s.name,
                "enrolled_at": str(s.enrolled_at),
            }
            for s in students
        ]
        return jsonify(data)
    except Exception as e:
        print(f"Students fetch error: {e}")
        return jsonify([])


@app.route("/delete_student", methods=["POST"])
def delete_student():
    try:
        data = request.json
        student_id = data.get("student_id")
        student = (
            controller.db.session.query(Student)
            .filter_by(student_id=student_id)
            .first()
        )
        if student:
            controller.db.session.delete(student)
            controller.db.session.commit()
            return jsonify(
                {"success": True, "message": f"Student {student_id} deleted"}
            )
        return jsonify({"success": False, "message": "Student not found"})
    except Exception as e:
        controller.db.session.rollback()
        return jsonify({"success": False, "message": str(e)})


# ── MANUAL MARK ──
@app.route("/manual_mark", methods=["POST"])
def manual_mark():
    try:
        data = request.json
        student_id = data.get("student_id")
        session_id = data.get("session_id")
        student_name = data.get("student_name", student_id)
        status = data.get("status")  # 'PRESENT' or 'ABSENT'

        record = (
            controller.db.session.query(AttendanceRecord)
            .filter_by(student_id=student_id, session_id=session_id)
            .first()
        )

        if record:
            record.status = status
            controller.db.session.commit()
            return jsonify(
                {"success": True, "message": f"{student_name} marked {status}"}
            )
        else:
            # Create new record if it doesn't exist
            controller.db.record_attendance(
                student_id=student_id,
                student_name=student_name,
                session_id=session_id,
                confidence=0.0,
                spoof_probability=0.0,
                status=status,
            )
            return jsonify(
                {"success": True, "message": f"{student_name} manually marked {status}"}
            )
    except Exception as e:
        controller.db.session.rollback()
        return jsonify({"success": False, "message": str(e)})


# ── OVERRIDE ALERT ──
@app.route("/override_alert", methods=["POST"])
def override_alert():
    try:
        data = request.json
        student_id = data.get("student_id")
        student_name = data.get("student_name", student_id)
        session_id = data.get("session_id")
        status = data.get("status")

        existing = (
            controller.db.session.query(AttendanceRecord)
            .filter_by(student_id=student_id, session_id=session_id)
            .first()
        )

        if existing:
            existing.status = status
        else:
            controller.db.record_attendance(
                student_id=student_id,
                student_name=student_name,
                session_id=session_id,
                confidence=data.get("confidence", 0.0),
                spoof_probability=data.get("spoof_prob", 0.0),
                status=status,
            )

        controller.db.session.commit()
        return jsonify(
            {
                "success": True,
                "message": f"Override applied — {student_name} marked {status}",
            }
        )
    except Exception as e:
        controller.db.session.rollback()
        return jsonify({"success": False, "message": str(e)})


if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5001)
