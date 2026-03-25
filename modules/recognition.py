import cv2
import numpy as np
import onnxruntime as ort


class FaceRecognizer:
    def __init__(self, model_path="models/arcface.onnx"):
        # Load ArcFace ONNX model
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.target_size = (112, 112)
        print("✅ ArcFace Recognition Module initialized")

    def preprocess(self, face_img):
        """
        Prepares face image for ArcFace
        Resizes to 112x112, normalizes to [-1, 1]
        """
        # Resize to 112x112
        face = cv2.resize(face_img, self.target_size)

        # Convert BGR to RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Normalize to [-1, 1]
        face = (face.astype(np.float32) - 127.5) / 127.5

        # Reshape to (1, 3, 112, 112) — model expects NCHW format
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)

        return face

    def generate_embedding(self, face_img):
        """
        Converts face image to 512-dimensional embedding vector
        This is the unique numerical fingerprint of the face
        """
        preprocessed = self.preprocess(face_img)

        # Run ArcFace model
        embedding = self.session.run(None, {self.input_name: preprocessed})[0]

        # Flatten to 1D array
        embedding = embedding.flatten()

        # L2 normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def cosine_similarity(self, embedding1, embedding2):
        """
        Measures similarity between two embeddings
        Returns value between 0 and 1
        1.0 = identical person
        0.0 = completely different person
        """
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        return float(similarity)

    def match_face(self, live_embedding, stored_embeddings, threshold=0.5):
        """
        Compares live embedding against all stored embeddings
        Returns: matched student ID and confidence score
        stored_embeddings format: {student_id: embedding_array}
        """
        best_match = None
        best_score = 0.0

        for student_id, stored_emb in stored_embeddings.items():
            score = self.cosine_similarity(live_embedding, stored_emb)
            if score > best_score:
                best_score = score
                best_match = student_id

        # Check if best score exceeds threshold
        if best_score >= threshold:
            return {
                "matched": True,
                "student_id": best_match,
                "confidence": round(best_score, 3),
                "verdict": f"✅ Matched: {best_match} ({round(best_score*100, 1)}%)",
            }
        else:
            return {
                "matched": False,
                "student_id": None,
                "confidence": round(best_score, 3),
                "verdict": f"❌ No match found (best: {round(best_score*100, 1)}%)",
            }
