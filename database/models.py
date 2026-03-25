from sqlalchemy import Column, Integer, String, Float, DateTime, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True)
    student_id = Column(String(50), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    embedding = Column(LargeBinary, nullable=False)  # encrypted embedding
    enrolled_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Student {self.student_id} - {self.name}>"


class AttendanceRecord(Base):
    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True)
    student_id = Column(String(50), nullable=False)
    student_name = Column(String(100), nullable=False)
    session_id = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    spoof_probability = Column(Float, nullable=False)
    status = Column(String(20), nullable=False)  # PRESENT / REJECTED
    timestamp = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Attendance {self.student_id} - {self.status}>"


class SpoofAlert(Base):
    __tablename__ = "spoof_alerts"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), nullable=False)
    spoof_probability = Column(Float, nullable=False)
    blink_count = Column(Integer, nullable=False)
    head_movement = Column(Float, nullable=False)
    ear_value = Column(Float, nullable=False)
    reason = Column(String(500), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<SpoofAlert {self.session_id} - {self.spoof_probability}>"
