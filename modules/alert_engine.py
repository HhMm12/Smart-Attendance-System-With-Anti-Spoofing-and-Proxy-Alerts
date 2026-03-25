from datetime import datetime


class AlertEngine:
    def __init__(self):
        # Thresholds for normal behavior
        self.NORMAL_BLINK_MIN = 2
        self.NORMAL_MOVEMENT_MIN = 3.0
        self.NORMAL_EAR_MIN = 0.25
        self.SPOOF_THRESHOLD = 0.6

        print("✅ Alert Engine initialized")

    def generate_explanation(
        self, spoof_result, liveness_details, student_id=None, session_id=None
    ):
        """
        Generates human readable explanation of why
        a spoof alert was raised or why attendance was approved
        """
        reasons = []
        warnings = []

        blink_count = liveness_details["blink_count"]
        head_movement = liveness_details["head_movement"]
        ear_value = liveness_details["ear_value"]
        spoof_prob = spoof_result["spoof_probability"]
        is_spoof = spoof_result["is_spoof"]

        # --- Analyze blink behavior ---
        if blink_count == 0:
            reasons.append(
                f"❌ No blinks detected. "
                f"Normal humans blink 2+ times. "
                f"Possible printed photo or screen display."
            )
        elif blink_count == 1:
            warnings.append(
                f"⚠️  Only 1 blink detected. " f"Below normal range of 2+ blinks."
            )
        else:
            reasons.append(f"✅ Normal blink pattern detected ({blink_count} blinks).")

        # --- Analyze head movement ---
        if head_movement < 1.0:
            reasons.append(
                f"❌ Head movement extremely low ({head_movement}). "
                f"Normal variance is above {self.NORMAL_MOVEMENT_MIN}. "
                f"Face appears completely static — possible photo attack."
            )
        elif head_movement < self.NORMAL_MOVEMENT_MIN:
            warnings.append(
                f"⚠️  Head movement below normal ({head_movement}). "
                f"Expected above {self.NORMAL_MOVEMENT_MIN}."
            )
        else:
            reasons.append(f"✅ Normal head movement detected ({head_movement}).")

        # --- Analyze EAR value ---
        if ear_value < 0.15:
            reasons.append(
                f"❌ Eye Aspect Ratio critically low ({ear_value}). "
                f"Eyes appear closed or absent — possible mask or photo."
            )
        elif ear_value < self.NORMAL_EAR_MIN:
            warnings.append(
                f"⚠️  EAR below normal ({ear_value}). "
                f"Expected above {self.NORMAL_EAR_MIN}."
            )
        else:
            reasons.append(f"✅ Normal EAR value ({ear_value}).")

        # --- Build final alert ---
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if is_spoof:
            alert = {
                "type": "SPOOF_DETECTED",
                "status": "❌ ATTENDANCE REJECTED",
                "student_id": student_id or "Unknown",
                "session_id": session_id or "Unknown",
                "spoof_probability": spoof_prob,
                "blink_count": blink_count,
                "head_movement": head_movement,
                "ear_value": ear_value,
                "reasons": reasons,
                "warnings": warnings,
                "summary": self._build_summary(
                    spoof_prob, blink_count, head_movement, ear_value
                ),
                "timestamp": timestamp,
            }
        else:
            alert = {
                "type": "GENUINE_DETECTED",
                "status": "✅ ATTENDANCE APPROVED",
                "student_id": student_id or "Unknown",
                "session_id": session_id or "Unknown",
                "spoof_probability": spoof_prob,
                "blink_count": blink_count,
                "head_movement": head_movement,
                "ear_value": ear_value,
                "reasons": reasons,
                "warnings": warnings,
                "summary": "Liveness confirmed. Attendance approved.",
                "timestamp": timestamp,
            }

        return alert

    def _build_summary(self, spoof_prob, blink_count, head_movement, ear_value):
        """
        Builds a short one-line summary for the dashboard
        """
        issues = []

        if blink_count < self.NORMAL_BLINK_MIN:
            issues.append(f"abnormal blink rate ({blink_count} blinks)")

        if head_movement < self.NORMAL_MOVEMENT_MIN:
            issues.append(f"insufficient head movement ({head_movement})")

        if ear_value < self.NORMAL_EAR_MIN:
            issues.append(f"low EAR value ({ear_value})")

        if issues:
            return (
                f"Proxy suspected. Spoof probability: {spoof_prob}. "
                f"Issues: {', '.join(issues)}."
            )
        else:
            return f"Spoof probability {spoof_prob} exceeded threshold."

    def format_for_dashboard(self, alert):
        """
        Formats alert as clean text for Teacher Dashboard display
        """
        lines = [
            f"{'='*50}",
            f"STATUS   : {alert['status']}",
            f"Student  : {alert['student_id']}",
            f"Session  : {alert['session_id']}",
            f"Time     : {alert['timestamp']}",
            f"{'─'*50}",
            f"Spoof Probability : {alert['spoof_probability']}",
            f"Blink Count       : {alert['blink_count']}",
            f"Head Movement     : {alert['head_movement']}",
            f"EAR Value         : {alert['ear_value']}",
            f"{'─'*50}",
            f"ANALYSIS :",
        ]

        for reason in alert["reasons"]:
            lines.append(f"  {reason}")

        if alert["warnings"]:
            lines.append("WARNINGS :")
            for warning in alert["warnings"]:
                lines.append(f"  {warning}")

        lines.append(f"{'─'*50}")
        lines.append(f"SUMMARY  : {alert['summary']}")
        lines.append(f"{'='*50}")

        return "\n".join(lines)
