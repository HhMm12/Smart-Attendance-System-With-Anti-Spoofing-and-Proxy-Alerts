import cv2
import numpy as np
import onnxruntime as ort


class SpoofEngine:
    def __init__(self):
        self.model_path = "models/spoof_classifier.onnx"
        self.img_size = 64
        self.threshold = 0.4
        self.session = None
        self._load_model()
        print("✅ Spoof Engine initialized")

    def _load_model(self):
        try:
            self.session = ort.InferenceSession(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
            print("   Using ML-based spoof classifier (LCC FASD trained)")
        except Exception as e:
            print(f"   ⚠️ ML model not found, using rule-based fallback: {e}")
            self.session = None

    def _preprocess(self, face_img):
        img = cv2.resize(face_img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # (1, 64, 64, 3)
        return img

    def _ml_spoof_score(self, face_img):
        try:
            inp = self._preprocess(face_img)
            output = self.session.run(None, {self.input_name: inp})
            probs = output[0][0]  # [real_prob, spoof_prob]
            return float(probs[1])  # spoof probability
        except Exception as e:
            print(f"   ML inference error: {e}")
            return None

    def _rule_based_score(self, blink_count, movement, ear):
        blink_score = 1.0 if blink_count < 2 else 0.0
        movement_score = 1.0 if movement < 3.0 else 0.0
        ear_score = 1.0 if ear < 0.25 else 0.0
        return (0.4 * blink_score) + (0.4 * movement_score) + (0.2 * ear_score)

    def compute_spoof_probability(
        self, face_img=None, blink_count=0, movement=0.0, ear=0.3
    ):
        if self.session is not None and face_img is not None:
            ml_score = self._ml_spoof_score(face_img)
            if ml_score is not None:
                rule_score = self._rule_based_score(blink_count, movement, ear)
                final_score = (0.7 * ml_score) + (0.3 * rule_score)
                return {
                    "spoof_probability": round(final_score, 4),
                    "ml_score": round(ml_score, 4),
                    "rule_score": round(rule_score, 4),
                    # ML score alone decides is_spoof — behavioral signals cannot override
                    "is_spoof": ml_score > 0.5,
                    "method": "ml+behavioral",
                }

        # Fallback to rule-based only
        rule_score = self._rule_based_score(blink_count, movement, ear)
        return {
            "spoof_probability": round(rule_score, 4),
            "ml_score": None,
            "rule_score": round(rule_score, 4),
            "is_spoof": rule_score > self.threshold,
            "method": "rule-based",
        }
