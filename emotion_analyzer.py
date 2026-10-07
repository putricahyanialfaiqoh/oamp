import math

from deepface import DeepFace

class EmotionAnalyzer:
    @staticmethod
    def _fuzzy_percentage(score):
        try:
            value = float(score)
        except (TypeError, ValueError):
            return 0.0

        if not math.isfinite(value):
            return 0.0

        return round(max(0.0, min(100.0, value)), 2)

    def analyze_emotion(self, face_crop):
        try:
            analysis = DeepFace.analyze(
                img_path=face_crop, 
                actions=['emotion'], 
                detector_backend='skip',
                enforce_detection=False
            )

            # DeepFace can return either a list or a single result dict.
            result = analysis[0] if isinstance(analysis, list) else analysis
            if not isinstance(result, dict):
                raise ValueError("Format hasil analisis DeepFace tidak valid")

            emotions = result.get("emotion", {})
            if not isinstance(emotions, dict) or not emotions:
                raise ValueError("Skor emosi tidak ditemukan")

            normalized_emotions = {
                str(emotion).lower(): self._fuzzy_percentage(score)
                for emotion, score in emotions.items()
            }
            result["emotion"] = normalized_emotions
            result["fuzzy_emotions"] = normalized_emotions.copy()
            result["dominant_emotion"] = max(
                result["fuzzy_emotions"],
                key=result["fuzzy_emotions"].get
            )
            return result
        except Exception as error:
            return {
                "dominant_emotion": "unknown",
                "analysis_error": str(error),
                "emotion": {
                    "angry": 0.0,
                    "disgust": 0.0,
                    "fear": 0.0,
                    "happy": 0.0,
                    "sad": 0.0,
                    "surprise": 0.0,
                    "neutral": 0.0
                },
                "fuzzy_emotions": {
                    "angry": 0.0,
                    "disgust": 0.0,
                    "fear": 0.0,
                    "happy": 0.0,
                    "sad": 0.0,
                    "surprise": 0.0,
                    "neutral": 0.0
                }
            }