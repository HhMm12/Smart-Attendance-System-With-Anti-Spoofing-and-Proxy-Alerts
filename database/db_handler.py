import numpy as np
import pickle
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base, Student, AttendanceRecord, SpoofAlert
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import os


class DatabaseHandler:
    def __init__(self, db_path="attendance.db"):
        # Create SQLite database
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        # AES encryption key (32 bytes = AES-256)
        self.key = b"SmartAttendance!SmartAttendance!"

        print("✅ Database initialized")

    # ---------- ENCRYPTION ----------

    def encrypt_embedding(self, embedding):
        """
        Encrypts numpy embedding array using AES-256
        """
        # Convert numpy array to bytes
        embedding_bytes = pickle.dumps(embedding)

        # AES encryption
        cipher = AES.new(self.key, AES.MODE_CBC)
        encrypted = cipher.encrypt(pad(embedding_bytes, AES.block_size))

        # Store IV + encrypted data together
        return cipher.iv + encrypted

    def decrypt_embedding(self, encrypted_data):
        """
        Decrypts AES-256 encrypted embedding back to numpy array
        """
        # Extract IV (first 16 bytes) and encrypted data
        iv = encrypted_data[:16]
        encrypted = encrypted_data[16:]

        # AES decryption
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        decrypted = unpad(cipher.decrypt(encrypted), AES.block_size)

        # Convert bytes back to numpy array
        return pickle.loads(decrypted)

    # ---------- STUDENT ENROLLMENT ----------

    def enroll_student(self, student_id, name, embedding):
        """
        Enrolls a new student with encrypted embedding
        """
        # Check if already enrolled
        existing = self.session.query(Student).filter_by(student_id=student_id).first()

        if existing:
            # Update existing embedding
            existing.embedding = self.encrypt_embedding(embedding)
            self.session.commit()
            print(f"✅ Updated enrollment for {name}")
            return True

        # Create new student record
        student = Student(
            student_id=student_id,
            name=name,
            embedding=self.encrypt_embedding(embedding),
        )
        self.session.add(student)
        self.session.commit()
        print(f"✅ Enrolled new student: {name} ({student_id})")
        return True

    def get_all_embeddings(self):
        """
        Returns all stored embeddings as dict
        Format: {student_id: (name, embedding)}
        """
        students = self.session.query(Student).all()
        embeddings = {}

        for student in students:
            embedding = self.decrypt_embedding(student.embedding)
            embeddings[student.student_id] = {
                "name": student.name,
                "embedding": embedding,
            }

        return embeddings

    def get_student(self, student_id):
        """
        Returns a single student record
        """
        return self.session.query(Student).filter_by(student_id=student_id).first()

    # ---------- ATTENDANCE RECORDING ----------

    def record_attendance(
        self,
        student_id,
        student_name,
        session_id,
        confidence,
        spoof_probability,
        status,
    ):
        """
        Records attendance entry in database
        """
        record = AttendanceRecord(
            student_id=student_id,
            student_name=student_name,
            session_id=session_id,
            confidence=confidence,
            spoof_probability=spoof_probability,
            status=status,
        )
        self.session.add(record)
        self.session.commit()
        print(f"✅ Attendance recorded: {student_name} - {status}")
        return True

    def get_attendance_by_session(self, session_id):
        """
        Returns all attendance records for a session
        """
        return (
            self.session.query(AttendanceRecord).filter_by(session_id=session_id).all()
        )

    def get_all_attendance(self):
        """
        Returns all attendance records
        """
        return self.session.query(AttendanceRecord).all()

    # ---------- SPOOF ALERTS ----------

    def record_spoof_alert(
        self,
        session_id,
        spoof_probability,
        blink_count,
        head_movement,
        ear_value,
        reason,
    ):
        """
        Records a spoof alert in database
        """
        alert = SpoofAlert(
            session_id=session_id,
            spoof_probability=spoof_probability,
            blink_count=blink_count,
            head_movement=head_movement,
            ear_value=ear_value,
            reason=reason,
        )
        self.session.add(alert)
        self.session.commit()
        print(f"⚠️  Spoof alert recorded: probability {spoof_probability}")
        return True

    def get_spoof_alerts(self, session_id=None):
        """
        Returns spoof alerts, optionally filtered by session
        """
        if session_id:
            return self.session.query(SpoofAlert).filter_by(session_id=session_id).all()
        return self.session.query(SpoofAlert).all()
